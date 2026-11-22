"""Build a side-by-side sample sheet and print each run's final measurements."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from PIL import Image, ImageDraw


def last_metrics_row(path: Path) -> dict[str, str]:
    with path.open(newline="") as metrics_file:
        rows = list(csv.DictReader(metrics_file))
    if not rows:
        raise ValueError(f"No training metrics found in {path}")
    return rows[-1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dcgan-dir", type=Path, required=True)
    parser.add_argument("--progressive-dir", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("gan_comparison.png"),
        help="destination for the side-by-side sample sheet",
    )
    args = parser.parse_args()

    run_details = (
        ("DCGAN", args.dcgan_dir),
        ("Progressive GAN", args.progressive_dir),
    )
    sample_images = []
    for name, run_dir in run_details:
        sample_path = run_dir / "final_samples.png"
        if not sample_path.is_file():
            raise FileNotFoundError(f"Generated sample grid not found: {sample_path}")
        sample_images.append((name, Image.open(sample_path).convert("RGB")))

    grid_width = sample_images[0][1].width
    grid_height = sample_images[0][1].height
    if any(image.size != (grid_width, grid_height) for _, image in sample_images):
        raise ValueError("Both runs must use the same sample-grid dimensions.")
    heading_height = 36
    comparison = Image.new("RGB", (grid_width * 2, grid_height + heading_height), "white")
    draw = ImageDraw.Draw(comparison)
    for index, (name, image) in enumerate(sample_images):
        x = index * grid_width
        draw.text((x + 8, 8), name, fill="black")
        comparison.paste(image, (x, heading_height))
        image.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    comparison.save(args.output)
    print(f"Side-by-side samples saved to: {args.output}")

    for name, run_dir in run_details:
        metrics_path = run_dir / "training_metrics.csv"
        row = last_metrics_row(metrics_path)
        steps = row.get("global_step") or row.get("step", "")
        print(
            f"{name}: steps={steps}, "
            f"elapsed={row['elapsed_seconds']}s, "
            f"peak CUDA memory={row['peak_cuda_mb'] or 'not measured'} MB, "
            f"D loss={row['discriminator_loss']}, G loss={row['generator_loss']}"
        )


if __name__ == "__main__":
    main()
