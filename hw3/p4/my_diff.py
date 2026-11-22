"""Train a small DDPM U-Net on the penguin images and visualize its samples.

Example:
    python P4/my_diff.py --steps 5000 --batch-size 4
"""

from __future__ import annotations

import argparse
import csv
import math
import time
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image

from my_dcgan import make_dataloader, nonnegative_int, positive_int


IMAGE_SIZE = 256
IMAGE_CHANNELS = 3


def sinusoidal_time_embedding(timesteps: torch.Tensor, dimensions: int) -> torch.Tensor:
    half_dimensions = dimensions // 2
    if half_dimensions == 0:
        return timesteps.float().unsqueeze(1)
    frequencies = torch.exp(
        -math.log(10000)
        * torch.arange(half_dimensions, device=timesteps.device).float()
        / max(half_dimensions - 1, 1)
    )
    angles = timesteps.float().unsqueeze(1) * frequencies.unsqueeze(0)
    embedding = torch.cat((angles.sin(), angles.cos()), dim=1)
    if dimensions % 2:
        embedding = F.pad(embedding, (0, 1))
    return embedding


class ResidualBlock(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, time_channels: int
    ) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(math.gcd(in_channels, 8), in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.time_projection = nn.Linear(time_channels, out_channels)
        self.norm2 = nn.GroupNorm(math.gcd(out_channels, 8), out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, 1)
        )

    def forward(
        self, inputs: torch.Tensor, time_embedding: torch.Tensor
    ) -> torch.Tensor:
        hidden = self.conv1(F.silu(self.norm1(inputs)))
        hidden = hidden + self.time_projection(F.silu(time_embedding))[:, :, None, None]
        hidden = self.conv2(F.silu(self.norm2(hidden)))
        return hidden + self.skip(inputs)


class UNet(nn.Module):
    """A compact three-level U-Net that predicts noise at 256x256 resolution."""

    def __init__(self, base_channels: int = 32) -> None:
        super().__init__()
        time_channels = base_channels * 4
        self.base_channels = base_channels
        self.time_mlp = nn.Sequential(
            nn.Linear(base_channels, time_channels),
            nn.SiLU(),
            nn.Linear(time_channels, time_channels),
        )

        self.input_conv = nn.Conv2d(IMAGE_CHANNELS, base_channels, 3, padding=1)
        self.down1 = ResidualBlock(base_channels, base_channels, time_channels)
        self.downsample1 = nn.Conv2d(base_channels, base_channels * 2, 4, 2, 1)
        self.down2 = ResidualBlock(base_channels * 2, base_channels * 2, time_channels)
        self.downsample2 = nn.Conv2d(base_channels * 2, base_channels * 4, 4, 2, 1)
        self.down3 = ResidualBlock(base_channels * 4, base_channels * 4, time_channels)
        self.downsample3 = nn.Conv2d(base_channels * 4, base_channels * 8, 4, 2, 1)

        self.middle1 = ResidualBlock(base_channels * 8, base_channels * 8, time_channels)
        self.middle2 = ResidualBlock(base_channels * 8, base_channels * 8, time_channels)

        self.up3_conv = nn.Conv2d(base_channels * 8, base_channels * 4, 3, padding=1)
        self.up3 = ResidualBlock(base_channels * 8, base_channels * 4, time_channels)
        self.up2_conv = nn.Conv2d(base_channels * 4, base_channels * 2, 3, padding=1)
        self.up2 = ResidualBlock(base_channels * 4, base_channels * 2, time_channels)
        self.up1_conv = nn.Conv2d(base_channels * 2, base_channels, 3, padding=1)
        self.up1 = ResidualBlock(base_channels * 2, base_channels, time_channels)
        self.output_norm = nn.GroupNorm(math.gcd(base_channels, 8), base_channels)
        self.output_conv = nn.Conv2d(base_channels, IMAGE_CHANNELS, 3, padding=1)

    def forward(self, images: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        time_embedding = self.time_mlp(
            sinusoidal_time_embedding(timesteps, self.base_channels)
        )

        skip1 = self.down1(self.input_conv(images), time_embedding)
        skip2 = self.down2(self.downsample1(skip1), time_embedding)
        skip3 = self.down3(self.downsample2(skip2), time_embedding)
        hidden = self.downsample3(skip3)
        hidden = self.middle1(hidden, time_embedding)
        hidden = self.middle2(hidden, time_embedding)

        hidden = F.interpolate(hidden, size=skip3.shape[-2:], mode="nearest")
        hidden = self.up3_conv(hidden)
        hidden = self.up3(torch.cat((hidden, skip3), dim=1), time_embedding)
        hidden = F.interpolate(hidden, size=skip2.shape[-2:], mode="nearest")
        hidden = self.up2_conv(hidden)
        hidden = self.up2(torch.cat((hidden, skip2), dim=1), time_embedding)
        hidden = F.interpolate(hidden, size=skip1.shape[-2:], mode="nearest")
        hidden = self.up1_conv(hidden)
        hidden = self.up1(torch.cat((hidden, skip1), dim=1), time_embedding)
        return self.output_conv(F.silu(self.output_norm(hidden)))


def make_noise_schedule(
    timesteps: int, device: torch.device
) -> dict[str, torch.Tensor]:
    betas = torch.linspace(1e-4, 0.02, timesteps, device=device)
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    previous_alpha_bars = F.pad(alpha_bars[:-1], (1, 0), value=1.0)
    posterior_variance = betas * (1.0 - previous_alpha_bars) / (1.0 - alpha_bars)
    return {
        "betas": betas,
        "alphas": alphas,
        "alpha_bars": alpha_bars,
        "sqrt_alpha_bars": alpha_bars.sqrt(),
        "sqrt_one_minus_alpha_bars": (1.0 - alpha_bars).sqrt(),
        "posterior_variance": posterior_variance,
    }


def extract(values: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
    return values[timesteps].view(-1, 1, 1, 1)


def add_noise(
    images: torch.Tensor,
    timesteps: torch.Tensor,
    noise: torch.Tensor,
    schedule: dict[str, torch.Tensor],
) -> torch.Tensor:
    return (
        extract(schedule["sqrt_alpha_bars"], timesteps) * images
        + extract(schedule["sqrt_one_minus_alpha_bars"], timesteps) * noise
    )


def sample_ddpm(
    model: UNet,
    schedule: dict[str, torch.Tensor],
    sample_count: int,
    timesteps: int,
    device: torch.device,
) -> tuple[torch.Tensor, list[tuple[int, torch.Tensor]]]:
    samples = torch.randn(sample_count, IMAGE_CHANNELS, IMAGE_SIZE, IMAGE_SIZE, device=device)
    trajectory = [(timesteps, samples[0].detach().cpu())]
    snapshot_timesteps = {
        round((timesteps - 1) * fraction)
        for fraction in (0.75, 0.5, 0.25, 0.0)
    }

    for step in range(timesteps - 1, -1, -1):
        step_tensor = torch.full((sample_count,), step, device=device, dtype=torch.long)
        predicted_noise = model(samples, step_tensor)
        beta = schedule["betas"][step]
        alpha = schedule["alphas"][step]
        alpha_bar = schedule["alpha_bars"][step]
        mean = (samples - beta / (1.0 - alpha_bar).sqrt() * predicted_noise) / alpha.sqrt()
        if step > 0:
            samples = mean + schedule["posterior_variance"][step].sqrt() * torch.randn_like(
                samples
            )
        else:
            samples = mean
        if step in snapshot_timesteps:
            trajectory.append((step, samples[0].detach().cpu()))

    return samples.clamp(-1, 1), trajectory


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=script_dir / "penguins" / "data",
        help="dataset root (default: P4/penguins/data; uses the same loader as the GANs)",
    )
    parser.add_argument("--batch-size", type=positive_int, default=4)
    parser.add_argument("--steps", type=positive_int, default=5000)
    parser.add_argument("--timesteps", type=positive_int, default=1000)
    parser.add_argument("--base-channels", type=positive_int, default=32)
    parser.add_argument("--sample-count", type=positive_int, default=16)
    parser.add_argument("--workers", type=nonnegative_int, default=0)
    parser.add_argument("--log-interval", type=positive_int, default=100)
    parser.add_argument(
        "--output-dir", type=Path, default=script_dir / "outputs" / "ddpm"
    )
    parser.add_argument("--seed", type=nonnegative_int, default=999)
    parser.add_argument("--horizontal-flip", action="store_true")
    return parser.parse_args()


def tensor_to_image(tensor: torch.Tensor):
    return tensor.detach().cpu().clamp(-1, 1).add(1).div(2).permute(1, 2, 0).numpy()


def save_sequence(
    images: list[torch.Tensor],
    labels: list[str],
    output_path: Path,
    title: str,
) -> None:
    figure, axes = plt.subplots(1, len(images), figsize=(2.4 * len(images), 2.8))
    axes = [axes] if len(images) == 1 else list(axes)
    for axis, image, label in zip(axes, images, labels):
        axis.imshow(tensor_to_image(image))
        axis.set_title(label)
        axis.axis("off")
    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(output_path, bbox_inches="tight")


def save_sample_grid(samples: torch.Tensor, output_path: Path) -> None:
    save_image(
        samples,
        output_path,
        nrow=4,
        normalize=True,
        value_range=(-1, 1),
        padding=2,
    )
    columns = min(4, samples.shape[0])
    rows = math.ceil(samples.shape[0] / columns)
    figure, axes = plt.subplots(rows, columns, squeeze=False, figsize=(columns * 2.5, rows * 2.5))
    for index, axis in enumerate(axes.flat):
        if index < samples.shape[0]:
            axis.imshow(tensor_to_image(samples[index]))
        axis.axis("off")
    figure.suptitle("DDPM generated samples")
    figure.tight_layout()


def train(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data_loader = make_dataloader(
        args.data_dir,
        args.batch_size,
        args.workers,
        IMAGE_SIZE,
        args.horizontal_flip,
    )
    real_example, _ = next(iter(data_loader))
    real_example = real_example[:1].to(device)
    print(f"Training images: {len(data_loader.dataset)} at {IMAGE_SIZE}x{IMAGE_SIZE}")

    model = UNet(args.base_channels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
    schedule = make_noise_schedule(args.timesteps, device)
    run_dir = args.output_dir / f"256_steps_{args.steps}"
    run_dir.mkdir(parents=True, exist_ok=True)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    training_start = time.perf_counter()
    images_seen = 0
    running_loss = 0.0
    logged_steps = 0
    data_iterator = iter(data_loader)
    metrics_path = run_dir / "training_metrics.csv"
    with metrics_path.open("w", newline="") as metrics_file:
        writer = csv.DictWriter(
            metrics_file,
            fieldnames=(
                "step",
                "images_seen",
                "noise_loss",
                "elapsed_seconds",
                "peak_cuda_mb",
            ),
        )
        writer.writeheader()
        for step in range(1, args.steps + 1):
            try:
                images, _ = next(data_iterator)
            except StopIteration:
                data_iterator = iter(data_loader)
                images, _ = next(data_iterator)
            images = images.to(device, non_blocking=True)
            batch_size = images.shape[0]
            time_indices = torch.randint(args.timesteps, (batch_size,), device=device)
            noise = torch.randn_like(images)
            noisy_images = add_noise(images, time_indices, noise, schedule)

            model.train()
            optimizer.zero_grad(set_to_none=True)
            predicted_noise = model(noisy_images, time_indices)
            loss = F.mse_loss(predicted_noise, noise)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            logged_steps += 1
            images_seen += batch_size
            if step % args.log_interval == 0 or step == args.steps:
                peak_cuda_mb = (
                    f"{torch.cuda.max_memory_allocated(device) / (1024**2):.2f}"
                    if device.type == "cuda"
                    else ""
                )
                elapsed = time.perf_counter() - training_start
                writer.writerow(
                    {
                        "step": step,
                        "images_seen": images_seen,
                        "noise_loss": f"{running_loss / logged_steps:.6f}",
                        "elapsed_seconds": f"{elapsed:.2f}",
                        "peak_cuda_mb": peak_cuda_mb,
                    }
                )
                metrics_file.flush()
                print(
                    f"Step [{step}/{args.steps}] noise loss: "
                    f"{running_loss / logged_steps:.5f} elapsed={elapsed:.1f}s"
                )
                running_loss = 0.0
                logged_steps = 0

    training_seconds = time.perf_counter() - training_start
    final_noise_loss = loss.item()
    training_peak_cuda_mb = (
        f"{torch.cuda.max_memory_allocated(device) / (1024**2):.2f}"
        if device.type == "cuda"
        else ""
    )
    model.eval()
    torch.save(
        {
            "image_size": IMAGE_SIZE,
            "timesteps": args.timesteps,
            "base_channels": args.base_channels,
            "steps": args.steps,
            "model": model.state_dict(),
        },
        run_dir / "ddpm_checkpoint.pt",
    )

    forward_steps = torch.linspace(
        0, args.timesteps - 1, min(6, args.timesteps), device=device
    ).long()
    forward_noise = torch.randn_like(real_example)
    forward_images = [
        add_noise(
            real_example,
            forward_steps[index : index + 1],
            forward_noise,
            schedule,
        )[0].cpu()
        for index in range(len(forward_steps))
    ]
    forward_labels = [f"t={step.item()}" for step in forward_steps]
    save_sequence(
        forward_images,
        forward_labels,
        run_dir / "forward_diffusion.png",
        "Forward diffusion: real penguin to noise",
    )

    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    inference_start = time.perf_counter()
    with torch.inference_mode():
        samples, reverse_trajectory = sample_ddpm(
            model, schedule, args.sample_count, args.timesteps, device
        )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        peak_inference_cuda_mb = (
            f"{torch.cuda.max_memory_allocated(device) / (1024**2):.2f}"
        )
    else:
        peak_inference_cuda_mb = ""
    inference_seconds = time.perf_counter() - inference_start

    reverse_images = [image for _, image in reverse_trajectory]
    reverse_labels = ["Start"] + [
        f"t={step}" for step, _ in reverse_trajectory[1:]
    ]
    save_sequence(
        reverse_images,
        reverse_labels,
        run_dir / "reverse_diffusion.png",
        "Reverse diffusion: noise to generated penguin",
    )
    final_samples_path = run_dir / "final_samples.png"
    save_sample_grid(samples.cpu(), final_samples_path)

    with (run_dir / "run_summary.csv").open("w", newline="") as summary_file:
        writer = csv.DictWriter(
            summary_file,
            fieldnames=(
                "steps",
                "images_seen",
                "dataset_images",
                "timesteps",
                "final_noise_loss",
                "training_seconds",
                "inference_seconds",
                "inference_ms_per_image",
                "peak_cuda_mb",
                "peak_inference_cuda_mb",
            ),
        )
        writer.writeheader()
        writer.writerow(
            {
                "steps": args.steps,
                "images_seen": images_seen,
                "dataset_images": len(data_loader.dataset),
                "timesteps": args.timesteps,
                "final_noise_loss": f"{final_noise_loss:.6f}",
                "training_seconds": f"{training_seconds:.2f}",
                "inference_seconds": f"{inference_seconds:.2f}",
                "inference_ms_per_image": (
                    f"{inference_seconds * 1000 / args.sample_count:.2f}"
                ),
                "peak_cuda_mb": training_peak_cuda_mb,
                "peak_inference_cuda_mb": peak_inference_cuda_mb,
            }
        )

    print(
        f"Training: {training_seconds:.1f}s; sampling {args.sample_count} images: "
        f"{inference_seconds:.1f}s "
        f"({inference_seconds * 1000 / args.sample_count:.1f} ms/image); "
        f"peak training CUDA memory: {training_peak_cuda_mb or 'not measured'} MB; "
        f"peak inference CUDA memory: {peak_inference_cuda_mb or 'not measured'} MB"
    )
    print(f"Metrics, checkpoint, and visualizations saved in: {run_dir}")
    plt.show()


if __name__ == "__main__":
    train(parse_args())
