import argparse
import gc
import json
import logging
import re
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from datasets import load_dataset
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
import transformers
from transformers import AutoModelForVision2Seq, AutoProcessor

# Use the Habana Trainer so Transformers places and trains the model on the HPU.
# Fail early rather than silently falling back to CPU training.
try:
    from optimum.habana import GaudiConfig, GaudiTrainer, GaudiTrainingArguments
except ImportError as exc:
    raise ImportError(
        "optimum-habana is required. Run this script in the configured Gaudi environment."
    ) from exc


# Defaults match the Hugging Face model and dataset used by this experiment.
MODEL_ID = "llava-hf/llava-v1.6-mistral-7b-hf"
DATASET_ID = "HuggingFaceM4/ChartQA"
LOG = logging.getLogger("chartqa_lora")


def parse_args():
    """Parse settings that are useful to vary between Gaudi runs."""
    parser = argparse.ArgumentParser(description="LoRA fine-tuning of LLaVA-1.6 on ChartQA")
    parser.add_argument("--model_name_or_path", default=MODEL_ID)
    parser.add_argument("--dataset_name", default=DATASET_ID)
    parser.add_argument("--output_dir", default="./output_llava_chartqa")
    # Optimum Habana's example Gaudi config; it can be overridden on the command line.
    parser.add_argument("--gaudi_config_name", default="Habana/gpt2")
    parser.add_argument("--epochs", type=float, default=3)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    return parser.parse_args()


def normalize_answer(answer):
    """Choose one reference answer and normalize it to a displayable string."""
    if isinstance(answer, (list, tuple)):
        answer = answer[0] if answer else ""
    return str(answer or "").strip()


def preprocess_example(example):
    """Map ChartQA's query/label fields to the common question/answer names."""
    return {
        "question": str(example.get("query", example.get("question", "")) or "").strip(),
        "answer": normalize_answer(example.get("label", example.get("answer", ""))),
    }


def preprocess_split(split):
    """Keep the image and normalized fields, dropping unused source metadata."""
    remove_columns = [column for column in split.column_names if column != "image"]
    return split.map(preprocess_example, remove_columns=remove_columns)


def build_prompt(question):
    """Format a LLaVA-1.6 user turn with the image placeholder."""
    return f"USER: <image>\n{question}\nASSISTANT:"


def make_collator(processor):
    """Create batches without truncating expanded image tokens; train on answers only."""
    tokenizer = processor.tokenizer
    collated_batches = 0
    if tokenizer.pad_token_id is None:
        # Some decoder tokenizers have no dedicated padding token.
        tokenizer.pad_token = tokenizer.eos_token

    def collate(examples):
        nonlocal collated_batches
        prompts = [build_prompt(example["question"]) for example in examples]
        texts = [
            f"{prompt} {example['answer']}{tokenizer.eos_token}"
            for prompt, example in zip(prompts, examples)
        ]
        images = [example["image"].convert("RGB") for example in examples]
        batch = processor(
            text=texts,
            images=images,
            padding=True,
            return_tensors="pt",
        )
        # LLaVA-NeXT expands each image placeholder into many image tokens.
        # Process the prompt with its image too, so masking uses its expanded
        # token length rather than the tokenizer's single-placeholder length.
        prompt_batch = processor(
            text=prompts,
            images=images,
            padding=True,
            return_tensors="pt",
        )
        # The model sees the question and image as context, but their tokens
        # should not contribute to the answer-generation loss.
        labels = batch["input_ids"].clone()
        full_lengths = batch["attention_mask"].sum(dim=1)
        prompt_lengths = prompt_batch["attention_mask"].sum(dim=1)
        for index, (full_length, prompt_length) in enumerate(zip(full_lengths, prompt_lengths)):
            full_start = labels.shape[1] - int(full_length) if tokenizer.padding_side == "left" else 0
            labels[index, full_start : full_start + int(prompt_length)] = -100
        # Ignore padded positions as well; -100 is the standard HF loss mask.
        labels[batch["attention_mask"] == 0] = -100
        batch["labels"] = labels
        if collated_batches < 3:
            shapes = {
                name: tuple(value.shape)
                for name, value in batch.items()
                if isinstance(value, torch.Tensor)
            }
            LOG.info("Collated batch %d tensor shapes: %s", collated_batches + 1, shapes)
        collated_batches += 1
        return batch

    return collate


def save_chartqa_examples(dataset, output_dir):
    """Save a two-image preview; an image file works on headless Slurm nodes."""
    examples = [dataset[index] for index in range(min(2, len(dataset)))]
    if len(examples) < 2:
        raise ValueError("ChartQA training split must contain at least two examples.")

    figure, axes = plt.subplots(1, 2, figsize=(14, 6))
    for axis, example in zip(axes, examples):
        axis.imshow(example["image"].convert("RGB"))
        axis.set_title(
            f"Q: {example['question']}\nA: {example['answer']}",
            fontsize=9,
            wrap=True,
        )
        axis.axis("off")
    figure.tight_layout()
    path = output_dir / "chartqa_examples.png"
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)
    LOG.info("Saved two ChartQA image/question/answer examples to %s", path)


def load_base_model(model_id, device):
    """Load the original BF16 checkpoint for baseline or adapter evaluation."""
    model = AutoModelForVision2Seq.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    model.to(device)
    model.eval()
    return model


def predict_test_set(model, processor, dataset, device, max_new_tokens):
    """Greedily generate one answer per row, retaining references for scoring."""
    results = []
    for index, example in enumerate(dataset):
        image = example["image"].convert("RGB")
        encoded = processor(
            text=[build_prompt(example["question"])],
            images=[image],
            return_tensors="pt",
        )
        encoded = {
            key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in encoded.items()
        }
        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        input_length = encoded["input_ids"].shape[1]
        # Exclude the input prompt from the decoded output, leaving the answer.
        prediction = processor.tokenizer.decode(
            generated[0, input_length:],
            skip_special_tokens=True,
        ).strip()
        results.append(
            {
                "question": example["question"],
                "answer": example["answer"],
                "prediction": prediction,
            }
        )
        if (index + 1) % 100 == 0:
            LOG.info("Generated %d/%d test predictions", index + 1, len(dataset))
    return results


def relaxed_correct(prediction, answer):
    """Compare normalized text exactly, or numeric answers within 5% tolerance."""
    prediction = re.sub(r"\s+", " ", prediction.strip().lower())
    answer = re.sub(r"\s+", " ", answer.strip().lower())
    if prediction == answer:
        return True

    def numeric_value(text):
        value = text.replace(",", "").replace("%", "").replace("$", "").strip()
        try:
            return float(value)
        except ValueError:
            return None

    predicted_number = numeric_value(prediction)
    answer_number = numeric_value(answer)
    if predicted_number is None or answer_number is None:
        return False
    return abs(predicted_number - answer_number) <= 0.05 * max(abs(answer_number), 1.0)


def score_predictions(results):
    """Attach per-example correctness and return mean test accuracy."""
    for result in results:
        result["correct"] = relaxed_correct(result["prediction"], result["answer"])
    accuracy = sum(result["correct"] for result in results) / max(len(results), 1)
    return accuracy


def save_loss_plot(log_history, output_dir):
    """Plot Trainer's step-level training losses and epoch validation losses."""
    train_steps = [
        item["step"] for item in log_history if "loss" in item and "eval_loss" not in item
    ]
    train_losses = [
        item["loss"] for item in log_history if "loss" in item and "eval_loss" not in item
    ]
    eval_steps = [item["step"] for item in log_history if "eval_loss" in item]
    eval_losses = [item["eval_loss"] for item in log_history if "eval_loss" in item]
    if not train_losses and not eval_losses:
        LOG.warning("No loss values were recorded; skipping the loss plot.")
        return

    figure, axis = plt.subplots(figsize=(8, 5))
    if train_losses:
        axis.plot(train_steps, train_losses, label="Training loss")
    if eval_losses:
        axis.plot(eval_steps, eval_losses, marker="o", label="Validation loss")
    axis.set_xlabel("Training step")
    axis.set_ylabel("Loss")
    axis.set_title("LLaVA-1.6 LoRA fine-tuning on ChartQA")
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "loss_plot.png", dpi=160)
    plt.close(figure)


def print_comparison_examples(results):
    """Print up to three gains and three remaining errors from the test set."""
    improved = [
        result
        for result in results
        if result["finetuned_correct"] and not result["baseline_correct"]
    ][:3]
    still_wrong = [result for result in results if not result["finetuned_correct"]][:3]

    print("\nQuestions newly answered correctly after fine-tuning:")
    for result in improved:
        print(
            f"- Q: {result['question']}\n  Expected: {result['answer']}"
            f"\n  Baseline: {result['baseline_prediction']}"
            f"\n  Fine-tuned: {result['finetuned_prediction']}"
        )
    if not improved:
        print("- No newly correct examples were found in this test set.")

    print("\nQuestions still answered incorrectly by the fine-tuned model:")
    for result in still_wrong:
        print(
            f"- Q: {result['question']}\n  Expected: {result['answer']}"
            f"\n  Fine-tuned: {result['finetuned_prediction']}"
        )
    if not still_wrong:
        print("- None; all evaluated test questions were correct.")


def main():
    """Run data preparation, baseline evaluation, LoRA training, and comparison."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # This script is intentionally Gaudi-only: CPU fallback would not represent
    # the requested training run or its HPU memory/time metrics.
    if not torch.hpu.is_available():
        raise RuntimeError("No Gaudi HPU is available. Run this script on a Gaudi node.")
    device = torch.device("hpu")

    # ChartQA's official training split is re-split into train and validation;
    # the official test split remains untouched for the final comparison.
    LOG.info("Loading processor and ChartQA dataset")
    processor = AutoProcessor.from_pretrained(args.model_name_or_path)
    raw_dataset = load_dataset(args.dataset_name)
    train_validation = raw_dataset["train"].train_test_split(test_size=0.2, seed=42)
    train_dataset = preprocess_split(train_validation["train"])
    validation_dataset = preprocess_split(train_validation["test"])
    test_dataset = preprocess_split(raw_dataset["test"])
    LOG.info(
        "80/20 split: %d training samples, %d validation samples",
        len(train_dataset),
        len(validation_dataset),
    )
    print(
        f"Training samples after 80/20 split: {len(train_dataset):,}; "
        f"validation samples: {len(validation_dataset):,}"
    )
    save_chartqa_examples(train_dataset, output_dir)

    # Keep baseline predictions between Slurm retries so a training-side
    # failure does not force another full pass over the test set.
    baseline_cache_path = output_dir / "baseline_test_predictions.json"
    baseline_results = None
    if baseline_cache_path.is_file():
        with baseline_cache_path.open(encoding="utf-8") as file:
            cached = json.load(file)
        cached_results = cached.get("results", [])
        cache_matches = (
            cached.get("model_name_or_path") == args.model_name_or_path
            and cached.get("dataset_name") == args.dataset_name
            and cached.get("max_new_tokens") == args.max_new_tokens
            and cached.get("transformers_version") == transformers.__version__
            and len(cached_results) == len(test_dataset)
            and all(
                row.get("question") == test_dataset[index]["question"]
                and row.get("answer") == test_dataset[index]["answer"]
                for index, row in enumerate(cached_results)
            )
        )
        if cache_matches:
            baseline_results = cached_results
            LOG.info("Loaded %d baseline predictions from %s", len(baseline_results), baseline_cache_path)

    if baseline_results is None:
        LOG.info("Evaluating the unmodified checkpoint on %d test questions", len(test_dataset))
        baseline_model = load_base_model(args.model_name_or_path, device)
        baseline_results = predict_test_set(
            baseline_model, processor, test_dataset, device, args.max_new_tokens
        )
        del baseline_model
        gc.collect()
        score_predictions(baseline_results)
        with baseline_cache_path.open("w", encoding="utf-8") as file:
            json.dump(
                {
                    "model_name_or_path": args.model_name_or_path,
                    "dataset_name": args.dataset_name,
                    "max_new_tokens": args.max_new_tokens,
                    "transformers_version": transformers.__version__,
                    "results": baseline_results,
                },
                file,
                indent=2,
            )

    baseline_accuracy = score_predictions(baseline_results)
    gc.collect()

    # Load the trainable model in BF16. LoRA is restricted to language-model
    # attention projections so the vision tower is not adapted.
    LOG.info("Loading LLaVA model for LoRA training")
    model = AutoModelForVision2Seq.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    # Resolve exact linear-layer paths to avoid matching similarly named
    # projections in the image encoder.
    target_suffixes = (".q_proj", ".k_proj", ".v_proj", ".o_proj")
    lora_targets = [
        name
        for name, module in model.named_modules()
        if isinstance(module, torch.nn.Linear)
        and "language_model" in name
        and name.endswith(target_suffixes)
    ]
    if not lora_targets:
        raise RuntimeError("Could not find LLaVA language-model attention layers for LoRA.")
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        target_modules=lora_targets,
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Batch size 1 plus gradient accumulation is a memory-conscious single-HPU
    # setup; epoch evaluation and saving preserve validation history/checkpoints.
    training_args = GaudiTrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        do_train=True,
        do_eval=True,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,
        warmup_ratio=0.03,
        weight_decay=0.01,
        logging_steps=25,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=True,
        use_habana=True,
        use_lazy_mode=False,
        gaudi_config_name=args.gaudi_config_name,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        remove_unused_columns=False,
        dataloader_pin_memory=False,
        report_to="none",
    )
    # LLaVA inputs have variable expanded sequence lengths. Enable eager-mode
    # dynamic shapes and use Habana's fused optimizer; standard Torch AdamW
    # fails in its HPU sqrt update on this node's SynapseAI 1.22 stack.
    gaudi_config = GaudiConfig.from_pretrained(args.gaudi_config_name)
    gaudi_config.use_dynamic_shapes = True
    gaudi_config.use_fused_adam = True
    trainer = GaudiTrainer(
        model=model,
        args=training_args,
        gaudi_config=gaudi_config,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        data_collator=make_collator(processor),
    )

    # Reset peak stats after model setup, synchronize around the timed section,
    # and capture the HPU peak allocated memory for the training run.
    torch.hpu.reset_peak_memory_stats()
    torch.hpu.synchronize()
    start_time = time.perf_counter()
    train_output = trainer.train()
    torch.hpu.synchronize()
    training_seconds = time.perf_counter() - start_time
    peak_memory_bytes = torch.hpu.max_memory_allocated()

    # Save the best adapter and processor separately from temporary checkpoints.
    fine_tuned_dir = output_dir / "fine_tuned_model"
    trainer.save_model(str(fine_tuned_dir))
    processor.save_pretrained(fine_tuned_dir)
    log_history = trainer.state.log_history
    save_loss_plot(log_history, output_dir)

    metrics = {
        "train_samples": len(train_dataset),
        "validation_samples": len(validation_dataset),
        "test_samples": len(test_dataset),
        "training_seconds": training_seconds,
        "training_minutes": training_seconds / 60,
        "peak_hpu_memory_bytes": peak_memory_bytes,
        "peak_hpu_memory_gib": peak_memory_bytes / (1024**3),
        "baseline_test_accuracy": baseline_accuracy,
        "train_metrics": train_output.metrics,
        "log_history": log_history,
    }
    with (output_dir / "training_metrics.json").open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2, default=str)
    LOG.info(
        "Training time %.2f minutes; peak HPU allocated memory %.2f GiB",
        metrics["training_minutes"],
        metrics["peak_hpu_memory_gib"],
    )
    del trainer, model
    gc.collect()

    # Reload the base weights with the saved adapter and run the exact same test
    # rows in order, making per-question baseline/fine-tuned comparisons possible.
    LOG.info("Evaluating the fine-tuned model on the same test questions")
    base_model = load_base_model(args.model_name_or_path, device)
    fine_tuned_model = PeftModel.from_pretrained(base_model, fine_tuned_dir)
    fine_tuned_model.eval()
    finetuned_results = predict_test_set(
        fine_tuned_model, processor, test_dataset, device, args.max_new_tokens
    )
    finetuned_accuracy = score_predictions(finetuned_results)

    # Keep both outputs, references, and per-model correctness for later review.
    comparison = []
    for baseline, finetuned in zip(baseline_results, finetuned_results):
        comparison.append(
            {
                **finetuned,
                "baseline_prediction": baseline["prediction"],
                "baseline_correct": baseline["correct"],
                "finetuned_prediction": finetuned["prediction"],
                "finetuned_correct": finetuned["correct"],
            }
        )
    metrics["finetuned_test_accuracy"] = finetuned_accuracy
    metrics["accuracy_change"] = finetuned_accuracy - baseline_accuracy
    with (output_dir / "training_metrics.json").open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2, default=str)
    with (output_dir / "test_comparison.json").open("w", encoding="utf-8") as file:
        json.dump(comparison, file, indent=2)

    print(
        f"\nTest accuracy ({len(test_dataset):,} questions): "
        f"baseline={baseline_accuracy:.4f}, fine-tuned={finetuned_accuracy:.4f}, "
        f"change={finetuned_accuracy - baseline_accuracy:+.4f}"
    )
    print_comparison_examples(comparison)
    LOG.info("Saved model, plots, metrics, and test comparison under %s", output_dir)


# Permit importing helpers without starting a multi-hour training run.
if __name__ == "__main__":
    main()
