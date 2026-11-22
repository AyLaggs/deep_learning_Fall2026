"""Visualize multiple latent-space interpolations from a trained DCGAN.

Example:
    python P4/latent_space.py P4/outputs/dcgan_64_steps_6000/dcgan_checkpoint.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from my_dcgan import LATENT_SIZE, Generator, positive_int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Display and save sequences of DCGAN latent interpolations."
    )
    parser.add_argument(
        "checkpoint",
        type=Path,
        help="checkpoint produced by my_dcgan.py",
    )
    parser.add_argument(
        "--examples",
        type=positive_int,
        default=4,
        help="number of independent latent pairs to interpolate (default: 4)",
    )
    parser.add_argument(
        "--steps",
        type=positive_int,
        default=10,
        help="number of frames per interpolation, including both endpoints (default: 10)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="seed for choosing latent endpoints (default: 42)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="output directory (default: <checkpoint directory>/latent_interpolations)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    if not isinstance(checkpoint, dict) or "generator" not in checkpoint:
        raise ValueError(
            "Checkpoint must contain the 'generator' state saved by my_dcgan.py."
        )

    image_size = int(checkpoint.get("image_size", 64))
    generator = Generator(image_size=image_size).to(device)
    generator.load_state_dict(checkpoint["generator"])
    generator.eval()

    random_generator = torch.Generator(device=device).manual_seed(args.seed)
    start = torch.randn(
        args.examples,
        LATENT_SIZE,
        1,
        1,
        device=device,
        generator=random_generator,
    )
    end = torch.randn(
        args.examples,
        LATENT_SIZE,
        1,
        1,
        device=device,
        generator=random_generator,
    )
    fractions = torch.linspace(0, 1, args.steps, device=device).view(
        1, args.steps, 1, 1, 1
    )
    latent_sequences = torch.lerp(start[:, None], end[:, None], fractions)

    with torch.inference_mode():
        images = generator(
            latent_sequences.reshape(
                args.examples * args.steps, LATENT_SIZE, 1, 1
            )
        ).cpu()

    output_dir = args.output_dir or args.checkpoint.parent / "latent_interpolations"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "latent_interpolations.png"

    figure, axes = plt.subplots(
        args.examples,
        args.steps,
        figsize=(args.steps * 1.6, args.examples * 1.8),
        squeeze=False,
    )
    for example in range(args.examples):
        for step in range(args.steps):
            image = images[example * args.steps + step]
            image = image.permute(1, 2, 0).clamp(-1, 1).add(1).div(2)
            axis = axes[example, step]
            axis.imshow(image)
            axis.set_axis_off()
            if example == 0:
                axis.set_title(f"t={step / max(args.steps - 1, 1):.2f}")
        axes[example, 0].set_ylabel(
            f"Pair {example + 1}", rotation=0, labelpad=35, va="center"
        )

    figure.suptitle("DCGAN latent-space interpolations")
    figure.tight_layout()
    figure.savefig(output_path, bbox_inches="tight")
    print(f"Interpolation sequences saved to: {output_path}")
    plt.show()


if __name__ == "__main__":
    main()
