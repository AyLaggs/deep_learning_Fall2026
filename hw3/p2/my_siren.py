"""Compare SIREN with exp(sin(w*x)) variants on a high-resolution RGB image."""

import argparse
import csv
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from scipy.ndimage import gaussian_filter
from torch import nn


class ActivationLayer(nn.Module):
    """Linear layer with either the SIREN sine or custom exp(sin) activation."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        activation: str,
        omega_0: float,
        is_first: bool = False,
        init_omega_0: float = 30.0,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.activation = activation
        self.omega_0 = omega_0
        self.linear = nn.Linear(in_features, out_features)
        self.init_weights(is_first, init_omega_0)

    def init_weights(self, is_first: bool, init_omega_0: float) -> None:
        """Use SIREN initialization, held fixed across activation comparisons."""
        with torch.no_grad():
            if is_first:
                # This is the original SIREN first-layer initialization.
                self.linear.weight.uniform_(-1 / self.in_features, 1 / self.in_features)
            else:
                # Keep the original SIREN weight scale even when custom w is changed.
                bound = np.sqrt(6 / self.in_features) / init_omega_0
                self.linear.weight.uniform_(-bound, bound)

    def forward(self, coordinates: torch.Tensor) -> torch.Tensor:
        scaled_input = self.omega_0 * self.linear(coordinates)
        if self.activation == "sine":
            return torch.sin(scaled_input)
        if self.activation == "exp_sine":
            return torch.exp(torch.sin(scaled_input))
        raise ValueError(f"Unsupported activation: {self.activation}")


class Siren(nn.Module):
    """Coordinate network whose layers can use either experiment activation."""

    def __init__(
        self,
        activation: str,
        omega_0: float,
        hidden_features: int = 256,
        hidden_layers: int = 3,
        init_omega_0: float = 30.0,
    ) -> None:
        super().__init__()
        # All variants have the same topology; only activation and its frequency vary.
        layers = [
            ActivationLayer(
                2,
                hidden_features,
                activation=activation,
                omega_0=omega_0,
                is_first=True,
                init_omega_0=init_omega_0,
            )
        ]
        layers.extend(
            ActivationLayer(
                hidden_features,
                hidden_features,
                activation=activation,
                omega_0=omega_0,
                init_omega_0=init_omega_0,
            )
            for _ in range(hidden_layers)
        )

        # Match the original SIREN: a linear three-channel RGB output layer.
        final_linear = nn.Linear(hidden_features, 3)
        with torch.no_grad():
            bound = np.sqrt(6 / hidden_features) / init_omega_0
            final_linear.weight.uniform_(-bound, bound)
        layers.append(final_linear)
        self.net = nn.Sequential(*layers)

    def forward(self, coordinates: torch.Tensor) -> torch.Tensor:
        return self.net(coordinates)


def load_image(image_path: Path) -> tuple[np.ndarray, torch.Tensor, torch.Tensor]:
    """Load RGB data and produce aligned [-1, 1] coordinates and [0, 1] targets."""
    with Image.open(image_path) as source:
        image = source.convert("RGB")

    width, height = image.size
    scale = max(1.0, 1024 / min(width, height))
    if scale > 1:
        image = image.resize(
            (round(width * scale), round(height * scale)),
            Image.Resampling.LANCZOS,
        )

    pixels = np.asarray(image, dtype=np.float32) / 255.0
    height, width, _ = pixels.shape
    x = torch.linspace(-1.0, 1.0, width)
    y = torch.linspace(-1.0, 1.0, height)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    coordinates = torch.stack((xx, yy), dim=-1).reshape(-1, 2)
    targets = torch.from_numpy(pixels).reshape(-1, 3)
    return pixels, coordinates, targets


def evaluate_mse(
    model: nn.Module,
    coordinates: torch.Tensor,
    targets: torch.Tensor,
    device: torch.device,
    chunk_size: int = 16384,
) -> float:
    """Compute un-clamped RGB MSE in chunks over a fixed coordinate set."""
    model.eval()
    squared_error_sum = 0.0
    with torch.no_grad():
        for start in range(0, len(coordinates), chunk_size):
            end = start + chunk_size
            predictions = model(coordinates[start:end].to(device))
            expected = targets[start:end].to(device)
            squared_error_sum += torch.sum((predictions - expected) ** 2).item()
    return squared_error_sum / targets.numel()


def predict_image(
    model: nn.Module,
    coordinates: torch.Tensor,
    image_shape: tuple[int, int, int],
    device: torch.device,
    chunk_size: int = 16384,
) -> np.ndarray:
    """Evaluate every pixel coordinate and return a clamped HxWx3 RGB image."""
    model.eval()
    predictions = []
    with torch.no_grad():
        for coordinate_chunk in coordinates.split(chunk_size):
            output = model(coordinate_chunk.to(device))
            predictions.append(output.clamp(0.0, 1.0).cpu())
    return torch.cat(predictions).reshape(image_shape).numpy()


def calculate_psnr(reference: np.ndarray, reconstruction: np.ndarray) -> float:
    """Compute PSNR in decibels for images with values in [0, 1]."""
    mse = np.mean((reference.astype(np.float64) - reconstruction) ** 2)
    if mse == 0:
        return float("inf")
    return float(10 * np.log10(1.0 / mse))


def calculate_ssim(reference: np.ndarray, reconstruction: np.ndarray) -> float:
    """Compute Gaussian-window SSIM averaged over the three color channels."""
    c1 = 0.01**2
    c2 = 0.03**2
    channel_scores = []
    for channel in range(reference.shape[2]):
        original = reference[..., channel].astype(np.float64)
        predicted = reconstruction[..., channel].astype(np.float64)
        mu_original = gaussian_filter(original, sigma=1.5)
        mu_predicted = gaussian_filter(predicted, sigma=1.5)
        variance_original = gaussian_filter(original**2, sigma=1.5) - mu_original**2
        variance_predicted = gaussian_filter(predicted**2, sigma=1.5) - mu_predicted**2
        covariance = (
            gaussian_filter(original * predicted, sigma=1.5)
            - mu_original * mu_predicted
        )
        numerator = (2 * mu_original * mu_predicted + c1) * (2 * covariance + c2)
        denominator = (
            (mu_original**2 + mu_predicted**2 + c1)
            * (variance_original + variance_predicted + c2)
        )
        channel_scores.append(np.mean(numerator / denominator))
    return float(np.mean(channel_scores))


def parse_args() -> argparse.Namespace:
    """Parse shared model, training, validation, and output settings."""
    default_image = Path(__file__).with_name("p2_1024.jpg")
    default_output_dir = Path(__file__).with_name("siren_experiment")
    parser = argparse.ArgumentParser(
        description="Compare SIREN and exp(sin(w*x)) image representations."
    )
    parser.add_argument("--image", type=Path, default=default_image)
    parser.add_argument("--output-dir", type=Path, default=default_output_dir)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--loss-samples", type=int, default=8192)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--hidden-features", type=int, default=256)
    parser.add_argument("--hidden-layers", type=int, default=3)
    parser.add_argument(
        "--custom-w",
        type=float,
        default=5.0,
        help="Frequency for the changed-w custom variant; the activation-only "
        "comparison uses the original SIREN frequency w=30.",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.steps < 1 or args.batch_size < 1 or args.eval_every < 1:
        parser.error("--steps, --batch-size, and --eval-every must be positive")
    if args.hidden_features < 1 or args.hidden_layers < 0:
        parser.error("--hidden-features must be positive and --hidden-layers nonnegative")
    if args.loss_samples < 1:
        parser.error("--loss-samples must be positive")
    if not 0 < args.validation_fraction < 1:
        parser.error("--validation-fraction must be between 0 and 1")
    if args.learning_rate <= 0 or args.custom_w <= 0:
        parser.error("--learning-rate and --custom-w must be positive")
    if args.custom_w == 30.0:
        parser.error("--custom-w must differ from 30 to test a changed frequency")
    return args


def train_experiment(
    name: str,
    slug: str,
    activation: str,
    omega_0: float,
    args: argparse.Namespace,
    coordinates: torch.Tensor,
    targets: torch.Tensor,
    image_shape: tuple[int, int, int],
    train_indices: torch.Tensor,
    validation_indices: torch.Tensor,
    device: torch.device,
) -> dict:
    """Train one variant and return losses, timing, and its final reconstruction."""
    # Resetting both seeds gives each model matching initial weights and batch order.
    torch.manual_seed(args.seed)
    batch_generator = torch.Generator(device="cpu").manual_seed(args.seed)
    start_time = time.perf_counter()
    model = Siren(
        activation=activation,
        omega_0=omega_0,
        hidden_features=args.hidden_features,
        hidden_layers=args.hidden_layers,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.MSELoss()

    # Use the same fixed training monitor sample and held-out pixels at every checkpoint.
    monitor_indices = train_indices[: min(args.loss_samples, len(train_indices))]
    monitor_coordinates = coordinates[monitor_indices]
    monitor_targets = targets[monitor_indices]
    validation_coordinates = coordinates[validation_indices]
    validation_targets = targets[validation_indices]
    training_history = []
    validation_history = []
    step_history = []

    print(f"\nTraining {name}...")
    for step in range(1, args.steps + 1):
        model.train()
        offsets = torch.randint(
            len(train_indices),
            (args.batch_size,),
            generator=batch_generator,
        )
        batch_indices = train_indices[offsets]
        batch_coordinates = coordinates[batch_indices].to(device)
        batch_targets = targets[batch_indices].to(device)

        predictions = model(batch_coordinates)
        loss = criterion(predictions, batch_targets)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"{name} produced a non-finite loss at step {step}")

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step % args.eval_every == 0 or step == args.steps:
            train_mse = evaluate_mse(
                model, monitor_coordinates, monitor_targets, device
            )
            validation_mse = evaluate_mse(
                model, validation_coordinates, validation_targets, device
            )
            step_history.append(step)
            training_history.append(train_mse)
            validation_history.append(validation_mse)
            print(
                f"  step {step}/{args.steps} | "
                f"train MSE: {train_mse:.6f} | held-out MSE: {validation_mse:.6f}"
            )

    training_seconds = time.perf_counter() - start_time
    reconstruction = predict_image(model, coordinates, image_shape, device)
    return {
        "name": name,
        "slug": slug,
        "activation": activation,
        "omega_0": omega_0,
        "train_mse": training_history[-1],
        "validation_mse": validation_history[-1],
        "training_seconds": training_seconds,
        "steps": step_history,
        "training_history": training_history,
        "validation_history": validation_history,
        "reconstruction": reconstruction,
    }


def save_results(
    original: np.ndarray,
    results: list[dict],
    output_dir: Path,
) -> None:
    """Write per-model reconstructions, comparison plots, and a metrics CSV."""
    output_dir.mkdir(parents=True, exist_ok=True)

    for result in results:
        image_data = np.uint8(np.rint(result["reconstruction"] * 255.0))
        Image.fromarray(image_data).save(output_dir / f"{result['slug']}.png")

    # Compare visual quality in one figure, with PSNR and SSIM in each title.
    figure, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.ravel()
    axes[0].imshow(original)
    axes[0].set_title("Original")
    for axis, result in zip(axes[1:], results):
        axis.imshow(result["reconstruction"])
        axis.set_title(
            f"{result['name']}\n"
            f"PSNR {result['psnr']:.2f} dB | SSIM {result['ssim']:.4f}"
        )
    for axis in axes:
        axis.axis("off")
    figure.tight_layout()
    figure.savefig(output_dir / "reconstructions.png", dpi=200, bbox_inches="tight")
    plt.close(figure)

    # Plot fixed-subset training MSE and full held-out-coordinate MSE over time.
    loss_figure, loss_axes = plt.subplots(1, 2, figsize=(13, 5))
    for result in results:
        loss_axes[0].plot(
            result["steps"], result["training_history"], label=result["name"]
        )
        loss_axes[1].plot(
            result["steps"], result["validation_history"], label=result["name"]
        )
    loss_axes[0].set_title("Training MSE (fixed training subset)")
    loss_axes[1].set_title("Held-out-coordinate MSE")
    for axis in loss_axes:
        axis.set_xlabel("Training step")
        axis.set_ylabel("MSE")
        axis.set_yscale("log")
        axis.grid(True, alpha=0.3)
        axis.legend()
    loss_figure.tight_layout()
    loss_figure.savefig(output_dir / "loss_curves.png", dpi=200, bbox_inches="tight")
    plt.close(loss_figure)

    columns = (
        "model",
        "activation",
        "omega_0",
        "training_mse",
        "held_out_mse",
        "psnr_db",
        "ssim",
        "training_seconds",
    )
    with (output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "model": result["name"],
                    "activation": result["activation"],
                    "omega_0": result["omega_0"],
                    "training_mse": result["train_mse"],
                    "held_out_mse": result["validation_mse"],
                    "psnr_db": result["psnr"],
                    "ssim": result["ssim"],
                    "training_seconds": result["training_seconds"],
                }
            )


def main() -> None:
    """Run the controlled three-model comparison and save its measurements."""
    args = parse_args()
    if not args.image.is_file():
        raise FileNotFoundError(f"Image file not found: {args.image}")

    # A single random split is reused by every model to keep validation comparable.
    split_generator = torch.Generator(device="cpu").manual_seed(args.seed)
    original, coordinates, targets = load_image(args.image)
    permutation = torch.randperm(len(coordinates), generator=split_generator)
    validation_count = max(
        1, int(round(len(coordinates) * args.validation_fraction))
    )
    validation_indices = permutation[:validation_count]
    train_indices = permutation[validation_count:]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    height, width, _ = original.shape
    print(
        f"Image: {width}x{height} | device: {device} | "
        f"train coordinates: {len(train_indices):,} | "
        f"held-out coordinates: {len(validation_indices):,}"
    )

    # First isolate the activation change at w=30, then test the plot's changed w=5.
    experiments = [
        {
            "name": "SIREN sin(w*x), w=30",
            "slug": "siren_baseline",
            "activation": "sine",
            "omega_0": 30.0,
        },
        {
            "name": "Custom exp(sin(w*x)), w=30 (activation-only change)",
            "slug": "custom_exp_sin_same_w",
            "activation": "exp_sine",
            "omega_0": 30.0,
        },
        {
            "name": f"Custom exp(sin(w*x)), w={args.custom_w:g}",
            "slug": "custom_exp_sin_changed_w",
            "activation": "exp_sine",
            "omega_0": args.custom_w,
        },
    ]

    results = []
    for experiment in experiments:
        result = train_experiment(
            **experiment,
            args=args,
            coordinates=coordinates,
            targets=targets,
            image_shape=original.shape,
            train_indices=train_indices,
            validation_indices=validation_indices,
            device=device,
        )
        result["psnr"] = calculate_psnr(original, result["reconstruction"])
        result["ssim"] = calculate_ssim(original, result["reconstruction"])
        results.append(result)
        print(
            f"{result['name']}: train MSE={result['train_mse']:.6f}, "
            f"held-out MSE={result['validation_mse']:.6f}, "
            f"PSNR={result['psnr']:.2f} dB, SSIM={result['ssim']:.4f}, "
            f"training time={result['training_seconds']:.1f}s"
        )

    save_results(original, results, args.output_dir)
    print(f"\nSaved experiment outputs to: {args.output_dir.resolve()}")
    print("  reconstructions.png - original and all three model outputs")
    print("  loss_curves.png - training and held-out loss curves")
    print("  metrics.csv - losses, image metrics, and training times")


if __name__ == "__main__":
    main()
