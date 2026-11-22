"""Plot the custom exp(sin(w*x)) activation beside the SIREN sine activation."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    # Use the same frequency for both functions so their oscillations are comparable.
    # SIREN commonly uses larger values (such as 30) for its first layer.
    w = 5.0
    x = np.linspace(-5.0, 5.0, 2000)

    custom_activation = np.exp(np.sin(w * x))
    siren_activation = np.sin(w * x)

    # Share the input and output scales to make the different ranges easy to compare.
    figure, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)
    axes[0].plot(x, custom_activation, color="tab:orange")
    axes[0].set_title(r"Custom: $\exp(\sin(wx))$")
    axes[1].plot(x, siren_activation, color="tab:blue")
    axes[1].set_title(r"SIREN: $\sin(wx)$")

    for axis in axes:
        axis.set_xlabel("Input x")
        axis.grid(True, alpha=0.3)
        axis.axhline(0, color="black", linewidth=0.7, alpha=0.6)

    axes[0].set_ylabel("Activation output")
    figure.suptitle(f"Activation comparison (frequency scale w = {w:g})")
    figure.tight_layout()

    output_path = Path(__file__).with_name("activation_comparison.png")
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved activation comparison to: {output_path.resolve()}")


if __name__ == "__main__":
    main()
