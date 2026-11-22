"""Train a GRU to predict MIDI pitches and generate notes autoregressively."""

from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class MusicGRU(nn.Module):
    """Map batches of MIDI-pitch tokens to next-pitch vocabulary logits.

    The embedding turns each integer pitch into a learned dense vector. The
    GRU reads those vectors in order, and the final linear layer converts each
    GRU output into one score (logit) per possible MIDI pitch.
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 2,
    ) -> None:
        super().__init__()
        # MIDI pitches are integer tokens, so embedding replaces each token
        # with a trainable vector before it is read by the recurrent layer.
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # batch_first means inputs and outputs use (batch, time, features).
        # Dropout is only supported between stacked GRU layers, not a single
        # layer, so disable it when num_layers is one.
        self.gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0.0,
        )
        # Produce one unnormalized score per pitch at every sequence position.
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(
        self, x: torch.Tensor, hidden: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return pitch logits for each time step and the final GRU state.

        ``x`` has shape (batch, time) and contains integer pitch IDs.
        Returned logits have shape (batch, time, vocab_size); the hidden state
        can be passed back in when continuing a sequence in chunks.
        """
        embedded = self.embedding(x)
        output, hidden = self.gru(embedded, hidden)
        return self.fc(output), hidden


def load_midi_sequences(
    midi_files: list[Path], vocab_size: int
) -> list[tuple[list[int], list[float], list[float]]]:
    """Read each MIDI file's first nonempty track.

    Each returned tuple contains parallel lists of pitch IDs, time between
    consecutive note onsets (steps), and note durations. Only pitches are
    model inputs; steps and durations are retained to give generated pitches
    timing when the output MIDI file is written.
    """
    try:
        import pretty_midi
    except ImportError as error:
        raise ImportError(
            "Reading and writing MIDI requires pretty_midi. "
            "Install it with: pip install pretty_midi"
        ) from error

    sequences = []
    for midi_file in midi_files:
        midi = pretty_midi.PrettyMIDI(str(midi_file))
        # Choose the first track with notes; metadata-only or empty tracks
        # cannot provide a training sequence.
        instrument = next(
            (item for item in midi.instruments if item.notes), None
        )
        if instrument is None:
            continue

        notes = sorted(instrument.notes, key=lambda note: note.start)
        pitches = [int(note.pitch) for note in notes]
        # The model's vocabulary is the standard MIDI range [0, 127].
        if any(pitch < 0 or pitch >= vocab_size for pitch in pitches):
            raise ValueError(
                f"{midi_file} contains a pitch outside [0, {vocab_size - 1}]"
            )

        # The first note starts at time zero for this track; subsequent values
        # store onset-to-onset intervals so the rhythm can later be replayed.
        steps = [0.0]
        steps.extend(
            max(0.0, notes[index].start - notes[index - 1].start)
            for index in range(1, len(notes))
        )
        durations = [max(0.0, note.end - note.start) for note in notes]
        sequences.append((pitches, steps, durations))
    return sequences


def make_dataloader(
    sequences: list[tuple[list[int], list[float], list[float]]],
    sequence_length: int,
    batch_size: int,
) -> DataLoader:
    """Create shuffled next-pitch examples from each pitch sequence.

    A window contains sequence_length input pitches followed by one target
    pitch. Windows never cross MIDI-file boundaries, avoiding artificial
    transitions between unrelated performances.
    """
    if sequence_length < 1:
        raise ValueError("sequence_length must be at least 1")
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")

    windows = []
    for pitches, _, _ in sequences:
        if len(pitches) > sequence_length:
            tokens = torch.tensor(pitches, dtype=torch.long)
            # unfold yields every overlapping window of length L+1:
            # [p0..pL] becomes input [p0..p(L-1)] and target pL.
            windows.append(tokens.unfold(0, sequence_length + 1, 1))
    if not windows:
        raise ValueError(
            "No MIDI track is longer than sequence_length; "
            "use longer tracks or a shorter sequence_length."
        )

    examples = torch.cat(windows)
    # Each example uses all but its last token as context; the held-out final
    # token is the single next pitch the model learns to predict.
    inputs = examples[:, :-1].contiguous()
    targets = examples[:, -1].contiguous()
    return DataLoader(
        TensorDataset(inputs, targets),
        batch_size=batch_size,
        # Shuffle examples each epoch; keep a smaller final batch rather than
        # discarding training examples that do not fill a whole batch.
        shuffle=True,
        drop_last=False,
    )


def train_model(
    model: MusicGRU,
    data_loader: DataLoader,
    epochs: int,
    learning_rate: float,
    device: torch.device,
) -> list[float]:
    """Train the model and return the example-weighted loss per epoch."""
    if epochs < 1:
        raise ValueError("epochs must be at least 1")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")

    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    epoch_losses = []

    for epoch in range(epochs):
        model.train()
        # Weight batch losses by their size so the reported average is correct
        # even when the last batch is smaller than the requested batch size.
        total_loss = 0.0
        total_examples = 0
        for inputs, targets in data_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits, _ = model(inputs)
            # The task labels one next pitch per input window, so use the
            # logits at the final context position, not every intermediate
            # position in the window.
            loss = criterion(logits[:, -1, :], targets)
            loss.backward()
            # Clip unusually large gradients to make recurrent training more
            # stable without changing ordinary-sized updates.
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            batch_size = inputs.size(0)
            total_loss += loss.item() * batch_size
            total_examples += batch_size

        mean_loss = total_loss / total_examples
        epoch_losses.append(mean_loss)
        print(f"Epoch {epoch + 1}/{epochs} - loss: {mean_loss:.4f}")
    return epoch_losses


def plot_learning_curve(epoch_losses: list[float], output_file: Path) -> None:
    """Save a plot of average training loss by epoch."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise ImportError(
            "Saving learning graphs requires matplotlib. "
            "Install it with: pip install matplotlib"
        ) from error

    if not epoch_losses:
        raise ValueError("epoch_losses must contain at least one value")

    figure, axis = plt.subplots(figsize=(8, 5))
    epochs = range(1, len(epoch_losses) + 1)
    axis.plot(epochs, epoch_losses, marker="o", linewidth=2)
    axis.set(
        title="GRU Training Learning Curve",
        xlabel="Epoch",
        ylabel="Cross-entropy loss",
    )
    axis.grid(True, linestyle="--", alpha=0.5)
    figure.tight_layout()
    figure.savefig(output_file, dpi=160)
    plt.close(figure)


def plot_model_architecture(model: MusicGRU, output_file: Path) -> None:
    """Save a conceptual diagram of the model and its GRU update/reset gates."""
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
    except ImportError as error:
        raise ImportError(
            "Saving the model diagram requires matplotlib. "
            "Install it with: pip install matplotlib"
        ) from error

    figure, axis = plt.subplots(figsize=(14, 9))
    axis.set_xlim(0, 14)
    axis.set_ylim(0, 9)
    axis.axis("off")

    def add_box(
        x: float,
        y: float,
        width: float,
        height: float,
        label: str,
        *,
        fontsize: int = 10,
        facecolor: str = "#eaf2f8",
    ) -> None:
        box = FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.08",
            linewidth=1.4,
            edgecolor="#34495e",
            facecolor=facecolor,
        )
        axis.add_patch(box)
        axis.text(
            x + width / 2,
            y + height / 2,
            label,
            ha="center",
            va="center",
            fontsize=fontsize,
            wrap=True,
        )

    def add_arrow(
        start: tuple[float, float], end: tuple[float, float]
    ) -> None:
        axis.add_patch(
            FancyArrowPatch(
                start,
                end,
                arrowstyle="-|>",
                mutation_scale=13,
                linewidth=1.3,
                color="#34495e",
            )
        )

    axis.text(
        7,
        8.55,
        "MusicGRU: next-pitch prediction and GRU cell",
        ha="center",
        fontsize=17,
        fontweight="bold",
    )
    axis.text(
        7,
        8.08,
        f"Embedding: {model.embedding.embedding_dim}D   |   "
        f"GRU: {model.gru.num_layers} layer(s), "
        f"{model.gru.hidden_size} hidden units   |   "
        f"Pitch vocabulary: {model.fc.out_features}",
        ha="center",
        fontsize=10,
        color="#34495e",
    )

    # Show the full sequence-to-distribution path at the top of the figure.
    pipeline = [
        (0.35, "Input pitch IDs\n(batch, time)"),
        (2.55, f"Embedding\n({model.embedding.embedding_dim}D)"),
        (
            4.75,
            f"GRU layer(s)\n{model.gru.hidden_size} hidden units\n"
            f"dropout={model.gru.dropout:g}",
        ),
        (7.2, "Fully connected\nhidden → vocabulary"),
        (9.65, "Softmax\npitch probabilities"),
        (11.8, "Sample next\npitch token"),
    ]
    widths = [1.65, 1.65, 1.9, 1.9, 1.65, 1.75]
    for (x, label), width in zip(pipeline, widths):
        add_box(x, 6.45, width, 0.95, label, fontsize=9)
    for index in range(len(pipeline) - 1):
        start_x = pipeline[index][0] + widths[index]
        end_x = pipeline[index + 1][0]
        add_arrow((start_x + 0.05, 6.93), (end_x - 0.05, 6.93))
    axis.text(
        7,
        6.05,
        "The sampled pitch is appended to the context and fed through the "
        "same path to generate the next note.",
        ha="center",
        fontsize=9,
        color="#34495e",
    )

    # Expand one recurrent time step to show the gates that maintain and
    # update the GRU hidden state.
    axis.text(
        7,
        5.35,
        "Inside one GRU time step",
        ha="center",
        fontsize=14,
        fontweight="bold",
    )
    add_box(
        0.45,
        3.35,
        2.1,
        1.15,
        "Current embedded input\nxₜ",
        facecolor="#fef9e7",
    )
    add_box(
        0.45,
        1.65,
        2.1,
        1.15,
        "Previous hidden state\nhₜ₋₁",
        facecolor="#fef9e7",
    )
    add_box(
        3.45,
        4.05,
        2.15,
        0.9,
        "Update gate\nzₜ = σ(Wz·xₜ + Uz·hₜ₋₁)",
        fontsize=9,
    )
    add_box(
        3.45,
        2.7,
        2.15,
        0.9,
        "Reset gate\nrₜ = σ(Wr·xₜ + Ur·hₜ₋₁)",
        fontsize=9,
    )
    add_box(
        3.45,
        1.05,
        2.15,
        1.1,
        "Candidate state\nnₜ = tanh(Wn·xₜ + rₜ ⊙ (Un·hₜ₋₁))",
        fontsize=8,
    )
    add_box(
        7,
        2.55,
        3.0,
        1.25,
        "State interpolation\nhₜ = zₜ ⊙ hₜ₋₁ + (1 − zₜ) ⊙ nₜ",
        facecolor="#e8f8f5",
        fontsize=9,
    )
    add_box(
        11.25,
        2.55,
        2.1,
        1.25,
        "New hidden state\nhₜ",
        facecolor="#e8f8f5",
    )
    add_arrow((2.55, 3.95), (3.45, 4.5))
    add_arrow((2.55, 3.8), (3.45, 3.15))
    add_arrow((2.55, 3.55), (3.45, 1.6))
    add_arrow((2.55, 2.2), (3.45, 4.3))
    add_arrow((2.55, 2.2), (3.45, 3.1))
    add_arrow((2.55, 2.15), (3.45, 1.6))
    add_arrow((5.6, 4.5), (7, 3.55))
    add_arrow((5.6, 3.15), (6.2, 1.95))
    add_arrow((6.2, 1.95), (7, 2.8))
    add_arrow((5.6, 1.6), (7, 2.75))
    add_arrow((10, 3.18), (11.25, 3.18))
    axis.text(
        12.3,
        2.15,
        "State is carried\ninto the next step",
        ha="center",
        fontsize=8,
        color="#34495e",
    )
    axis.text(
        7,
        0.45,
        "σ: sigmoid   |   tanh: candidate activation   |   ⊙: element-wise product",
        ha="center",
        fontsize=9,
        color="#34495e",
    )
    figure.tight_layout()
    figure.savefig(output_file, dpi=160, bbox_inches="tight")
    plt.close(figure)


def sample_next_note(
    logits: torch.Tensor,
    recent_notes: list[int],
    top_k: int | None = None,
    repetition_penalty: float = 1.0,
) -> int:
    """Draw one pitch after applying the selected sampling adjustments.

    ``repetition_penalty`` is a multiplier greater than or equal to one. For
    pitches in ``recent_notes``, subtracting its logarithm lowers their
    softmax odds relative to other pitches. ``top_k`` then limits sampling to
    the k highest adjusted logits; ``None`` leaves the full vocabulary.
    """
    if logits.ndim != 1:
        raise ValueError("logits must be a one-dimensional vocabulary vector")
    if top_k is not None and top_k < 1:
        raise ValueError("top_k must be at least 1 or None")
    if repetition_penalty < 1.0:
        raise ValueError("repetition_penalty must be at least 1.0")

    adjusted_logits = logits.clone()
    if repetition_penalty > 1.0 and recent_notes:
        # Penalize each distinct recently used pitch once, even if it appeared
        # multiple times in the selected history window.
        recent = torch.tensor(
            list(set(recent_notes)), dtype=torch.long, device=logits.device
        )
        adjusted_logits[recent] -= math.log(repetition_penalty)

    if top_k is not None:
        k = min(top_k, adjusted_logits.numel())
        top_indices = torch.topk(adjusted_logits, k).indices
        # Set all other logits to negative infinity so their softmax
        # probabilities are exactly zero (including when there are ties).
        filtered_logits = torch.full_like(adjusted_logits, -torch.inf)
        filtered_logits[top_indices] = adjusted_logits[top_indices]
        adjusted_logits = filtered_logits

    # Convert the adjusted scores to a categorical distribution and sample
    # from it; this preserves variety instead of always selecting argmax.
    probabilities = torch.softmax(adjusted_logits, dim=-1)
    return int(torch.multinomial(probabilities, num_samples=1).item())


@torch.no_grad()
def generate(
    model: MusicGRU,
    starting_sequence: list[int],
    num_notes: int,
    sequence_length: int,
    top_k: int | None = 5,
    repetition_penalty: float = 1.2,
    repetition_window: int = 20,
) -> list[int]:
    """Extend a seed pitch sequence by repeatedly predicting one pitch.

    At each step, the model receives the most recent ``sequence_length``
    pitches. Repetition control uses up to ``repetition_window`` previous
    pitches, and the new sample is appended before the next prediction.
    """
    if not starting_sequence:
        raise ValueError("starting_sequence must contain at least one pitch")
    if num_notes < 0:
        raise ValueError("num_notes cannot be negative")
    if sequence_length < 1:
        raise ValueError("sequence_length must be at least 1")
    if repetition_window < 0:
        raise ValueError("repetition_window cannot be negative")

    generated = list(starting_sequence)
    device = next(model.parameters()).device
    # Generation does not need dropout or gradient tracking.
    model.eval()
    for _ in range(num_notes):
        # Keep the model context bounded to the same window length used in
        # training, even as the generated sequence grows.
        context = torch.tensor(
            generated[-sequence_length:], dtype=torch.long, device=device
        ).unsqueeze(0)
        logits, _ = model(context)
        # A zero-length window explicitly disables repetition adjustment.
        recent = generated[-repetition_window:] if repetition_window else []
        next_note = sample_next_note(
            logits[0, -1],
            recent,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
        )
        generated.append(next_note)
    return generated


def write_midi(
    pitches: list[int],
    rhythm: tuple[list[float], list[float]],
    output_file: Path,
) -> None:
    """Write pitch events to MIDI using the supplied source-track rhythm.

    Since this model predicts pitch only, step and duration values are cycled
    from the source track to provide timing for every generated pitch.
    """
    try:
        import pretty_midi
    except ImportError as error:
        raise ImportError(
            "Reading and writing MIDI requires pretty_midi. "
            "Install it with: pip install pretty_midi"
        ) from error

    steps, durations = rhythm
    if not steps or not durations:
        raise ValueError("The source MIDI track has no rhythm information")

    midi = pretty_midi.PrettyMIDI()
    # Use the default piano program for the generated monophonic instrument.
    instrument = pretty_midi.Instrument(program=0)
    start = 0.0
    for index, pitch in enumerate(pitches):
        if index:
            # Index zero starts at time zero; subsequent notes advance by the
            # corresponding inter-onset interval from the reference track.
            start += steps[index % len(steps)]
        duration = durations[index % len(durations)]
        # MIDI notes require a positive interval; use a short minimum for
        # zero-duration source events.
        instrument.notes.append(
            pretty_midi.Note(
                velocity=100,
                pitch=pitch,
                start=start,
                end=start + max(duration, 0.05),
            )
        )
    midi.instruments.append(instrument)
    midi.write(str(output_file))


def parse_args() -> argparse.Namespace:
    """Define the command-line options for training and generation."""
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Train a GRU on MIDI pitches and generate a MIDI file."
    )
    parser.add_argument(
        "--midi-dir",
        type=Path,
        default=script_dir / "data" / "maestro-v2.0.0",
        help="Folder containing MIDI files (searched recursively).",
    )
    parser.add_argument("--num-files", type=int, default=5)
    parser.add_argument("--sequence-length", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--num-notes", type=int, default=120)
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Sample only from the k most likely pitches; use 128 for all.",
    )
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.2,
        help="Multiplier (>1 discourages recent pitches; 1 disables it).",
    )
    parser.add_argument("--repetition-window", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("generated.mid"))
    parser.add_argument(
        "--plots-dir",
        type=Path,
        default=None,
        help="Directory for training_loss.png and gru_architecture.png "
        "(defaults to the output MIDI's directory).",
    )
    return parser.parse_args()


def main() -> None:
    """Load MIDI, train the GRU, then generate and save an output performance."""
    args = parse_args()
    if args.num_files < 1:
        raise ValueError("num_files must be at least 1")
    if args.sequence_length < 1:
        raise ValueError("sequence_length must be at least 1")

    # Seed Python and PyTorch's random generators for repeatable shuffling and
    # weight initialization. CUDA is seeded too when it is available.
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Sort file paths to make the selected subset predictable across runs.
    midi_files = sorted(
        path
        for path in args.midi_dir.rglob("*")
        if path.suffix.lower() in {".mid", ".midi"}
    )[: args.num_files]
    if not midi_files:
        raise FileNotFoundError(
            f"No MIDI files found under {args.midi_dir}. "
            "Download or extract the MAESTRO MIDI dataset there, or pass "
            "a different folder with --midi-dir."
        )

    vocab_size = 128
    sequences = load_midi_sequences(midi_files, vocab_size)
    if not sequences:
        raise ValueError("The selected MIDI files contain no note events")
    data_loader = make_dataloader(
        sequences, args.sequence_length, args.batch_size
    )

    # Use a GPU when available; otherwise the same code runs on the CPU.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MusicGRU(vocab_size=vocab_size)
    epoch_losses = train_model(
        model, data_loader, args.epochs, args.learning_rate, device
    )
    plots_dir = args.plots_dir if args.plots_dir is not None else args.output.parent
    plots_dir.mkdir(parents=True, exist_ok=True)
    loss_plot = plots_dir / "training_loss.png"
    architecture_plot = plots_dir / "gru_architecture.png"
    plot_learning_curve(epoch_losses, loss_plot)
    plot_model_architecture(model, architecture_plot)
    print(f"Saved learning curve: {loss_plot}")
    print(f"Saved GRU architecture diagram: {architecture_plot}")

    # Reuse an initial passage as the prompt, then ask the model for the
    # requested number of additional pitches.
    seed_pitches = sequences[0][0][: args.sequence_length]
    generated = generate(
        model,
        seed_pitches,
        num_notes=args.num_notes,
        sequence_length=args.sequence_length,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
        repetition_window=args.repetition_window,
    )
    # This pitch-only model does not predict timing, so reuse the first
    # training track's steps and durations for MIDI rendering.
    _, steps, durations = sequences[0]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_midi(generated, (steps, durations), args.output)
    print(
        f"Generated {args.num_notes} new notes "
        f"({len(generated)} total including the seed) with "
        f"top_k={args.top_k}, "
        f"repetition_penalty={args.repetition_penalty}; saved {args.output}"
    )


if __name__ == "__main__":
    main()
