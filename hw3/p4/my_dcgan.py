"""Train a DCGAN on an image dataset.

Based on the PyTorch DCGAN tutorial:
https://docs.pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html

Flat image-directory layout (also supported):
    P4/squares/0.png
    P4/squares/1.png

Example:
    python my_dcgan.py --data-dir squares --image-size 64 --batch-size 64 --epochs 300 --workers 0 --output-dir outputs/square_outputs 2>&1 | tee my_dcgan_square.log
    python P4/my_dcgan.py --data-dir P4/penguins/data --image-size 256 --batch-size 4 --steps 6000
"""

from __future__ import annotations

import argparse
import csv
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms, utils


CHANNELS = 3
LATENT_SIZE = 100
GENERATOR_FEATURES = 64
DISCRIMINATOR_FEATURES = 64
LEARNING_RATE = 0.0002
BETA1 = 0.5
IMAGE_EXTENSIONS = {
    ".bmp",
    ".jpeg",
    ".jpg",
    ".png",
    ".ppm",
    ".tif",
    ".tiff",
    ".webp",
}


class Generator(nn.Module):
    """Map latent noise vectors to power-of-two RGB images from 64 to 256."""

    def __init__(
        self,
        latent_size: int = LATENT_SIZE,
        channels: int = CHANNELS,
        feature_maps: int = GENERATOR_FEATURES,
        image_size: int = 64,
    ) -> None:
        super().__init__()
        if image_size not in (64, 128, 256):
            raise ValueError("image_size must be one of 64, 128, or 256")

        blocks: list[nn.Module] = [
            nn.ConvTranspose2d(latent_size, feature_maps * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(feature_maps * 8),
            nn.ReLU(True),
        ]
        current_channels = feature_maps * 8
        resolution = 4
        while resolution < image_size:
            resolution *= 2
            channel_exponent = max(resolution.bit_length() - 4, 0)
            next_channels = max(feature_maps * 8 // (2**channel_exponent), 8)
            blocks.extend(
                [
                    nn.ConvTranspose2d(
                        current_channels, next_channels, 4, 2, 1, bias=False
                    ),
                    nn.BatchNorm2d(next_channels),
                    nn.ReLU(True),
                ]
            )
            current_channels = next_channels
        blocks.extend([nn.Conv2d(current_channels, channels, 3, 1, 1), nn.Tanh()])
        self.network = nn.Sequential(*blocks)

    def forward(self, noise: torch.Tensor) -> torch.Tensor:
        return self.network(noise)


class Discriminator(nn.Module):
    """Classify power-of-two RGB images from 64 to 256."""

    def __init__(
        self,
        channels: int = CHANNELS,
        feature_maps: int = DISCRIMINATOR_FEATURES,
        image_size: int = 64,
    ) -> None:
        super().__init__()
        if image_size not in (64, 128, 256):
            raise ValueError("image_size must be one of 64, 128, or 256")

        channels_by_resolution = {
            resolution: max(
                feature_maps * 8 // max(2 ** max(resolution.bit_length() - 4, 0), 1),
                8,
            )
            for resolution in (4, 8, 16, 32, 64, 128, 256)
        }
        blocks: list[nn.Module] = []
        current_resolution = image_size
        current_channels = channels
        first = True
        while current_resolution > 4:
            next_resolution = current_resolution // 2
            next_channels = channels_by_resolution[next_resolution]
            blocks.extend(
                [
                    nn.Conv2d(
                        current_channels,
                        next_channels,
                        4,
                        2,
                        1,
                        bias=False,
                    ),
                    nn.LeakyReLU(0.2, inplace=True),
                ]
            )
            if not first:
                blocks.insert(-1, nn.BatchNorm2d(next_channels))
            current_channels = next_channels
            current_resolution = next_resolution
            first = False
        blocks.append(nn.Conv2d(current_channels, 1, 4, 1, 0, bias=False))
        self.network = nn.Sequential(*blocks)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.network(images).view(-1)


class FlatImageDataset(Dataset):
    """Load image files directly inside a directory without class subfolders."""

    def __init__(self, image_paths: list[Path], transform: transforms.Compose) -> None:
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        with Image.open(self.image_paths[index]) as image:
            image_tensor = self.transform(image.convert("RGB"))
        return image_tensor, 0


def initialize_weights(module: nn.Module) -> None:
    """Apply the DCGAN paper's initialization to convolution and batch norm."""
    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.normal_(module.weight, mean=1.0, std=0.02)
        nn.init.constant_(module.bias, 0)


def positive_int(value: str) -> int:
    """Parse a positive integer argument."""
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return number


def nonnegative_int(value: str) -> int:
    """Parse a nonnegative integer argument."""
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("must be at least 0")
    return number


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Train a DCGAN to generate images from a dataset."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=script_dir / "penguins" / "data",
        help="image folder (default: P4/penguins/data; includes all splits)",
    )
    parser.add_argument(
        "--batch-size",
        type=positive_int,
        default=256,
        help="number of training images per batch (default: 256)",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        choices=(64, 128, 256),
        default=64,
        help="generated and training image resolution (default: 64)",
    )
    parser.add_argument(
        "--epochs",
        type=positive_int,
        default=5,
        help="number of complete passes through the dataset (default: 5)",
    )
    parser.add_argument(
        "--steps",
        type=positive_int,
        default=None,
        help="optimizer updates (overrides the epoch budget when provided)",
    )
    parser.add_argument(
        "--workers",
        type=nonnegative_int,
        default=2,
        help="number of data-loading worker processes (default: 2)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=script_dir / "outputs",
        help="directory in which to save generated images and model checkpoints",
    )
    parser.add_argument(
        "--seed",
        type=nonnegative_int,
        default=999,
        help="random seed (default: 999)",
    )
    parser.add_argument(
        "--horizontal-flip",
        action="store_true",
        help="randomly flip training images horizontally",
    )
    return parser.parse_args()


def make_dataloader(
    data_dir: Path,
    batch_size: int,
    workers: int,
    image_size: int = 64,
    horizontal_flip: bool = False,
) -> DataLoader:
    if not data_dir.is_dir():
        raise FileNotFoundError(
            f"Dataset directory not found: {data_dir}\n"
            "Pass a directory containing images or an ImageFolder root with --data-dir."
        )

    transform_steps = [transforms.Resize(image_size), transforms.CenterCrop(image_size)]
    if horizontal_flip:
        transform_steps.append(transforms.RandomHorizontalFlip())
    transform_steps.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5,) * CHANNELS, (0.5,) * CHANNELS),
        ]
    )
    image_transform = transforms.Compose(transform_steps)
    flat_images = sorted(
        path
        for path in data_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if flat_images:
        dataset = FlatImageDataset(flat_images, image_transform)
    else:
        dataset = datasets.ImageFolder(root=str(data_dir), transform=image_transform)
    if len(dataset) == 0:
        raise ValueError(f"No images were found under dataset directory: {data_dir}")

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


def train(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    image_size = getattr(args, "image_size", 64)
    data_loader = make_dataloader(
        args.data_dir,
        args.batch_size,
        args.workers,
        image_size,
        args.horizontal_flip,
    )
    print(f"Training images: {len(data_loader.dataset)} at {image_size}x{image_size}")

    max_steps = args.steps or (len(data_loader) * args.epochs)
    budget = f"steps_{max_steps}"
    run_dir = args.output_dir / f"dcgan_{image_size}_{budget}"
    run_dir.mkdir(parents=True, exist_ok=True)

    generator = Generator(image_size=image_size).to(device)
    discriminator = Discriminator(image_size=image_size).to(device)
    generator.apply(initialize_weights)
    discriminator.apply(initialize_weights)

    criterion = nn.BCEWithLogitsLoss()
    optimizer_g = torch.optim.Adam(
        generator.parameters(), lr=LEARNING_RATE, betas=(BETA1, 0.999)
    )
    optimizer_d = torch.optim.Adam(
        discriminator.parameters(), lr=LEARNING_RATE, betas=(BETA1, 0.999)
    )
    fixed_noise_generator = torch.Generator(device=device).manual_seed(args.seed + 1)
    fixed_noise = torch.randn(
        64,
        LATENT_SIZE,
        1,
        1,
        device=device,
        generator=fixed_noise_generator,
    )

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    training_start = time.perf_counter()
    step = 0
    epoch = 0
    images_seen = 0
    with (run_dir / "training_metrics.csv").open("w", newline="") as metrics_file:
        metrics_writer = csv.DictWriter(
            metrics_file,
            fieldnames=(
                "epoch",
                "step",
                "images_seen",
                "discriminator_loss",
                "generator_loss",
                "elapsed_seconds",
                "peak_cuda_mb",
            ),
        )
        metrics_writer.writeheader()
        while step < max_steps:
            epoch += 1
            generator.train()
            discriminator.train()
            total_g_loss = 0.0
            total_d_loss = 0.0
            total_images = 0

            for real_images, _ in data_loader:
                if step >= max_steps:
                    break
                real_images = real_images.to(device, non_blocking=True)
                current_batch_size = real_images.size(0)
                real_labels = torch.ones(current_batch_size, device=device)
                fake_labels = torch.zeros(current_batch_size, device=device)

                optimizer_d.zero_grad(set_to_none=True)
                real_logits = discriminator(real_images)
                real_loss = criterion(real_logits, real_labels)
                noise = torch.randn(
                    current_batch_size, LATENT_SIZE, 1, 1, device=device
                )
                fake_images = generator(noise)
                fake_logits = discriminator(fake_images.detach())
                fake_loss = criterion(fake_logits, fake_labels)
                discriminator_loss = real_loss + fake_loss
                discriminator_loss.backward()
                optimizer_d.step()

                optimizer_g.zero_grad(set_to_none=True)
                generated_logits = discriminator(fake_images)
                generator_loss = criterion(generated_logits, real_labels)
                generator_loss.backward()
                optimizer_g.step()

                total_g_loss += generator_loss.item() * current_batch_size
                total_d_loss += discriminator_loss.item() * current_batch_size
                total_images += current_batch_size
                images_seen += current_batch_size
                step += 1

            mean_g_loss = total_g_loss / total_images
            mean_d_loss = total_d_loss / total_images
            elapsed = time.perf_counter() - training_start
            peak_cuda_mb = (
                torch.cuda.max_memory_allocated(device) / (1024**2)
                if device.type == "cuda"
                else ""
            )
            metrics_writer.writerow(
                {
                    "epoch": epoch,
                    "step": step,
                    "images_seen": images_seen,
                    "discriminator_loss": f"{mean_d_loss:.6f}",
                    "generator_loss": f"{mean_g_loss:.6f}",
                    "elapsed_seconds": f"{elapsed:.2f}",
                    "peak_cuda_mb": (
                        f"{peak_cuda_mb:.2f}" if peak_cuda_mb != "" else ""
                    ),
                }
            )
            metrics_file.flush()
            print(
                f"Epoch [{epoch}] step={step}/{max_steps} "
                f"Loss_D: {mean_d_loss:.4f} Loss_G: {mean_g_loss:.4f} "
                f"elapsed={elapsed:.1f}s"
            )

            generator.eval()
            with torch.inference_mode():
                samples = generator(fixed_noise).cpu()
            utils.save_image(
                samples,
                run_dir / f"epoch_{epoch:03d}.png",
                normalize=True,
                value_range=(-1, 1),
                nrow=8,
            )

    generator.eval()
    with torch.inference_mode():
        final_samples = generator(fixed_noise).cpu()
    utils.save_image(
        final_samples,
        run_dir / "final_samples.png",
        normalize=True,
        value_range=(-1, 1),
        nrow=8,
    )

    checkpoint_path = run_dir / "dcgan_checkpoint.pt"
    torch.save(
        {
            "epochs": epoch,
            "steps": step,
            "batch_size": args.batch_size,
            "image_size": image_size,
            "generator": generator.state_dict(),
            "discriminator": discriminator.state_dict(),
            "optimizer_g": optimizer_g.state_dict(),
            "optimizer_d": optimizer_d.state_dict(),
        },
        checkpoint_path,
    )
    print(f"Samples, metrics, and checkpoint saved in: {run_dir}")


if __name__ == "__main__":
    train(parse_args())
