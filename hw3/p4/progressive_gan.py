"""Train a progressively growing WGAN-GP on the penguin dataset."""

from __future__ import annotations

import argparse
import csv
import math
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image

from my_dcgan import make_dataloader, nonnegative_int, positive_int


LATENT_SIZE = 100
RESOLUTIONS = (4, 8, 16, 32, 64, 128, 256)
IMAGE_CHANNELS = 3


def channels_at(resolution: int) -> int:
    return max(512 // (2 ** max(resolution.bit_length() - 4, 0)), 16)


class EqualizedConv2d(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1,
        padding: int = 0,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(
            torch.randn(out_channels, in_channels, kernel_size, kernel_size)
        )
        self.bias = nn.Parameter(torch.zeros(out_channels))
        self.scale = math.sqrt(2 / (in_channels * kernel_size * kernel_size))
        self.stride = stride
        self.padding = padding

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return F.conv2d(
            inputs,
            self.weight * self.scale,
            self.bias,
            stride=self.stride,
            padding=self.padding,
        )


def activate(inputs: torch.Tensor) -> torch.Tensor:
    return F.leaky_relu(inputs, 0.2) * math.sqrt(2)


def pixel_normalize(inputs: torch.Tensor) -> torch.Tensor:
    return inputs * torch.rsqrt(inputs.square().mean(dim=1, keepdim=True) + 1e-8)


class Generator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.initial = nn.ModuleList(
            [
                EqualizedConv2d(LATENT_SIZE, channels_at(4), 4, padding=3),
                EqualizedConv2d(channels_at(4), channels_at(4), 3, padding=1),
            ]
        )
        self.blocks = nn.ModuleDict()
        self.to_rgb = nn.ModuleDict()
        for resolution in RESOLUTIONS:
            self.to_rgb[str(resolution)] = EqualizedConv2d(
                channels_at(resolution), IMAGE_CHANNELS, 1
            )
            if resolution > 4:
                previous_channels = channels_at(resolution // 2)
                current_channels = channels_at(resolution)
                self.blocks[str(resolution)] = nn.ModuleList(
                    [
                        EqualizedConv2d(previous_channels, current_channels, 3, padding=1),
                        EqualizedConv2d(current_channels, current_channels, 3, padding=1),
                    ]
                )

    def forward(
        self, noise: torch.Tensor, resolution: int, alpha: float = 1.0
    ) -> torch.Tensor:
        features = pixel_normalize(activate(self.initial[0](noise)))
        features = pixel_normalize(activate(self.initial[1](features)))
        if resolution == 4:
            return torch.tanh(self.to_rgb["4"](features))
        previous_rgb = self.to_rgb["4"](features)
        for current_resolution in RESOLUTIONS[1:]:
            features = F.interpolate(features, scale_factor=2, mode="nearest")
            first, second = self.blocks[str(current_resolution)]
            features = pixel_normalize(activate(first(features)))
            features = pixel_normalize(activate(second(features)))
            if current_resolution == resolution:
                current_rgb = self.to_rgb[str(current_resolution)](features)
                if previous_rgb is not None:
                    old_rgb = F.interpolate(
                        previous_rgb, scale_factor=2, mode="nearest"
                    )
                    current_rgb = alpha * current_rgb + (1 - alpha) * old_rgb
                return torch.tanh(current_rgb)
            previous_rgb = self.to_rgb[str(current_resolution)](features)
        return torch.tanh(self.to_rgb["4"](features))


class Discriminator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.from_rgb = nn.ModuleDict(
            {
                str(resolution): EqualizedConv2d(IMAGE_CHANNELS, channels_at(resolution), 1)
                for resolution in RESOLUTIONS
            }
        )
        self.blocks = nn.ModuleDict()
        for resolution in RESOLUTIONS[1:]:
            self.blocks[str(resolution)] = nn.ModuleList(
                [
                    EqualizedConv2d(
                        channels_at(resolution), channels_at(resolution), 3, padding=1
                    ),
                    EqualizedConv2d(
                        channels_at(resolution),
                        channels_at(resolution // 2),
                        3,
                        stride=2,
                        padding=1,
                    ),
                ]
            )
        self.final_conv = EqualizedConv2d(channels_at(4) + 1, channels_at(4), 3, padding=1)
        self.final_score = EqualizedConv2d(channels_at(4), 1, 4)

    def forward(
        self, images: torch.Tensor, resolution: int, alpha: float = 1.0
    ) -> torch.Tensor:
        features = activate(self.from_rgb[str(resolution)](images))
        if resolution > 4:
            first, second = self.blocks[str(resolution)]
            features = activate(first(features))
            downsampled = activate(second(features))
            old_images = F.avg_pool2d(images, 2)
            old_features = activate(self.from_rgb[str(resolution // 2)](old_images))
            features = alpha * downsampled + (1 - alpha) * old_features

        current_resolution = resolution // 2
        while current_resolution > 4:
            first, second = self.blocks[str(current_resolution)]
            features = activate(first(features))
            features = activate(second(features))
            current_resolution //= 2

        batch_std = torch.sqrt(
            features.var(dim=0, unbiased=False, keepdim=True) + 1e-8
        ).mean()
        batch_feature = batch_std.expand(
            features.shape[0], 1, features.shape[2], features.shape[3]
        )
        features = torch.cat((features, batch_feature), dim=1)
        features = activate(self.final_conv(features))
        return self.final_score(features).view(-1)


def gradient_penalty(
    discriminator: Discriminator,
    real_images: torch.Tensor,
    fake_images: torch.Tensor,
    resolution: int,
    alpha: float,
) -> torch.Tensor:
    interpolation = torch.rand(
        real_images.shape[0], 1, 1, 1, device=real_images.device
    )
    mixed = (
        interpolation * real_images + (1 - interpolation) * fake_images
    ).requires_grad_(True)
    mixed_scores = discriminator(mixed, resolution, alpha)
    gradients = torch.autograd.grad(
        outputs=mixed_scores,
        inputs=mixed,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    return (gradients.flatten(1).norm(2, dim=1) - 1).square().mean()


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=script_dir / "penguins" / "data",
        help="dataset root (default: P4/penguins/data; includes all splits)",
    )
    parser.add_argument("--batch-size", type=positive_int, default=4)
    parser.add_argument("--steps-per-stage", type=positive_int, default=1000)
    parser.add_argument("--workers", type=nonnegative_int, default=0)
    parser.add_argument(
        "--output-dir", type=Path, default=script_dir / "outputs" / "progressive_gan"
    )
    parser.add_argument("--seed", type=nonnegative_int, default=999)
    parser.add_argument("--horizontal-flip", action="store_true")
    return parser.parse_args()


def train(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)
    optimizer_g = torch.optim.Adam(
        generator.parameters(), lr=0.001, betas=(0.0, 0.99), eps=1e-8
    )
    optimizer_d = torch.optim.Adam(
        discriminator.parameters(), lr=0.001, betas=(0.0, 0.99), eps=1e-8
    )

    total_steps = args.steps_per_stage * len(RESOLUTIONS)
    run_dir = args.output_dir / f"256_steps_{total_steps}"
    run_dir.mkdir(parents=True, exist_ok=True)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    fixed_noise_generator = torch.Generator(device=device).manual_seed(args.seed + 1)
    fixed_noise = torch.randn(
        64, LATENT_SIZE, 1, 1, device=device, generator=fixed_noise_generator
    )
    training_start = time.perf_counter()
    global_step = 0
    images_seen = 0

    with (run_dir / "training_metrics.csv").open("w", newline="") as metrics_file:
        writer = csv.DictWriter(
            metrics_file,
            fieldnames=(
                "stage_resolution",
                "stage_step",
                "global_step",
                "images_seen",
                "discriminator_loss",
                "generator_loss",
                "elapsed_seconds",
                "peak_cuda_mb",
            ),
        )
        writer.writeheader()
        for resolution in RESOLUTIONS:
            loader = make_dataloader(
                args.data_dir,
                args.batch_size,
                args.workers,
                resolution,
                args.horizontal_flip,
            )
            data_iterator = iter(loader)
            stage_g_loss = 0.0
            stage_d_loss = 0.0
            stage_images = 0
            for stage_step in range(args.steps_per_stage):
                try:
                    real_images, _ = next(data_iterator)
                except StopIteration:
                    data_iterator = iter(loader)
                    real_images, _ = next(data_iterator)
                real_images = real_images.to(device, non_blocking=True)
                current_batch_size = real_images.shape[0]
                images_seen += current_batch_size
                alpha = (
                    1.0
                    if resolution == 4
                    else min(1.0, stage_step / max(args.steps_per_stage // 2, 1))
                )
                noise = torch.randn(
                    current_batch_size, LATENT_SIZE, 1, 1, device=device
                )
                fake_images = generator(noise, resolution, alpha)

                optimizer_d.zero_grad(set_to_none=True)
                real_scores = discriminator(real_images, resolution, alpha)
                fake_scores = discriminator(
                    fake_images.detach(), resolution, alpha
                )
                penalty = gradient_penalty(
                    discriminator,
                    real_images,
                    fake_images.detach(),
                    resolution,
                    alpha,
                )
                discriminator_loss = (
                    fake_scores.mean() - real_scores.mean() + 10 * penalty
                )
                discriminator_loss.backward()
                optimizer_d.step()

                optimizer_g.zero_grad(set_to_none=True)
                generated_scores = discriminator(fake_images, resolution, alpha)
                generator_loss = -generated_scores.mean()
                generator_loss.backward()
                optimizer_g.step()

                stage_g_loss += generator_loss.item()
                stage_d_loss += discriminator_loss.item()
                global_step += 1

            elapsed = time.perf_counter() - training_start
            peak_cuda_mb = (
                f"{torch.cuda.max_memory_allocated(device) / (1024**2):.2f}"
                if device.type == "cuda"
                else ""
            )
            row = {
                "stage_resolution": resolution,
                "stage_step": args.steps_per_stage,
                "global_step": global_step,
                "images_seen": images_seen,
                "discriminator_loss": f"{stage_d_loss / args.steps_per_stage:.6f}",
                "generator_loss": f"{stage_g_loss / args.steps_per_stage:.6f}",
                "elapsed_seconds": f"{elapsed:.2f}",
                "peak_cuda_mb": peak_cuda_mb,
            }
            writer.writerow(row)
            metrics_file.flush()
            print(
                f"Resolution {resolution}: step {global_step}/{total_steps} "
                f"D={stage_d_loss / args.steps_per_stage:.4f} "
                f"G={stage_g_loss / args.steps_per_stage:.4f} "
                f"elapsed={elapsed:.1f}s"
            )

    generator.eval()
    with torch.inference_mode():
        samples = generator(fixed_noise, 256).cpu()
    save_image(
        samples,
        run_dir / "final_samples.png",
        normalize=True,
        value_range=(-1, 1),
        nrow=8,
    )
    torch.save(
        {
            "image_size": 256,
            "latent_size": LATENT_SIZE,
            "steps": global_step,
            "batch_size": args.batch_size,
            "generator": generator.state_dict(),
            "discriminator": discriminator.state_dict(),
            "optimizer_g": optimizer_g.state_dict(),
            "optimizer_d": optimizer_d.state_dict(),
        },
        run_dir / "progressive_gan_checkpoint.pt",
    )
    print(f"Samples, metrics, and checkpoint saved in: {run_dir}")


if __name__ == "__main__":
    train(parse_args())
