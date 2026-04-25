from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, classification_report, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "artifacts" / "phase1_dataset"
OUTPUT_ROOT = ROOT / "artifacts" / "banglabert_classifier"
MODEL_NAME = "csebuetnlp/banglabert"
LABEL2ID = {"fake": 0, "real": 1}
ID2LABEL = {value: key for key, value in LABEL2ID.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune BanglaBERT on the phase-1 fake news split.")
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--train-batch-size", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-valid-samples", type=int, default=None)
    parser.add_argument("--max-test-samples", type=int, default=None)
    parser.add_argument("--skip-model-save", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_text(frame: pd.DataFrame) -> list[str]:
    category = frame["category"].fillna("").astype(str).str.strip()
    headline = frame["headline"].fillna("").astype(str).str.strip()
    content = frame["content"].fillna("").astype(str).str.strip()
    return (
        "[CATEGORY] "
        + category
        + " [HEADLINE] "
        + headline
        + " [CONTENT] "
        + content
    ).tolist()


def load_frame(name: str, max_samples: int | None, seed: int) -> pd.DataFrame:
    frame = pd.read_csv(DATA_DIR / f"{name}.csv")
    if max_samples is not None and len(frame) > max_samples:
        frame = frame.sample(n=max_samples, random_state=seed).reset_index(drop=True)
    return frame.reset_index(drop=True)


def frame_to_dataset(frame: pd.DataFrame) -> Dataset:
    return Dataset.from_dict(
        {
            "text": build_text(frame),
            "label": [LABEL2ID[label] for label in frame["label"].astype(str).tolist()],
        }
    )


def tokenize_dataset(dataset: Dataset, tokenizer: AutoTokenizer, max_length: int) -> Dataset:
    def tokenize_batch(batch: dict[str, list[str]]) -> dict[str, list[list[int]]]:
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_length,
        )

    return dataset.map(tokenize_batch, batched=True, remove_columns=["text"])


def compute_metrics(eval_pred: tuple[np.ndarray, np.ndarray]) -> dict[str, float]:
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro"),
        "weighted_f1": f1_score(labels, preds, average="weighted"),
    }


def detailed_report(labels: list[int], preds: list[int]) -> dict[str, object]:
    label_names = [ID2LABEL[idx] for idx in sorted(ID2LABEL)]
    return classification_report(
        labels,
        preds,
        target_names=label_names,
        digits=4,
        output_dict=True,
    )


def save_summary(
    output_dir: Path,
    args: argparse.Namespace,
    validation_metrics: dict[str, object],
    test_metrics: dict[str, object],
) -> None:
    summary_lines = [
        "# BanglaBERT Baseline",
        "",
        f"- Model: `{args.model_name}`",
        f"- Device: `{'cuda' if torch.cuda.is_available() else 'cpu'}`",
        f"- Epochs: `{args.epochs}`",
        f"- Max length: `{args.max_length}`",
        f"- Train batch size: `{args.train_batch_size}`",
        f"- Eval batch size: `{args.eval_batch_size}`",
        f"- Model saved: `{'no' if args.skip_model_save else 'yes'}`",
        f"- Validation accuracy: `{validation_metrics['accuracy']:.4f}`",
        f"- Validation macro F1: `{validation_metrics['macro_f1']:.4f}`",
        f"- Test accuracy: `{test_metrics['accuracy']:.4f}`",
        f"- Test macro F1: `{test_metrics['macro_f1']:.4f}`",
        "",
        "## Notes",
        "",
        "- Input text is built from `category + headline + content`.",
        "- This run is directly comparable with the TF-IDF baseline on the same split.",
        "- Detailed metrics are stored in `metrics.json`.",
        "- Use `--skip-model-save` on low-disk machines and rerun on Colab or Kaggle for the full checkpoint.",
        "",
    ]
    (output_dir / "README.md").write_text("\n".join(summary_lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    output_dir = OUTPUT_ROOT
    output_dir.mkdir(parents=True, exist_ok=True)

    train_frame = load_frame("train", args.max_train_samples, args.seed)
    valid_frame = load_frame("valid", args.max_valid_samples, args.seed)
    test_frame = load_frame("test", args.max_test_samples, args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=2,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )
    os.environ["HF_HUB_OFFLINE"] = "1"

    train_dataset = tokenize_dataset(frame_to_dataset(train_frame), tokenizer, args.max_length)
    valid_dataset = tokenize_dataset(frame_to_dataset(valid_frame), tokenizer, args.max_length)
    test_dataset = tokenize_dataset(frame_to_dataset(test_frame), tokenizer, args.max_length)

    training_args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        do_train=True,
        do_eval=True,
        eval_strategy="epoch",
        save_strategy="no",
        logging_strategy="steps",
        logging_steps=50,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        num_train_epochs=args.epochs,
        weight_decay=args.weight_decay,
        load_best_model_at_end=False,
        report_to=[],
        seed=args.seed,
        use_cpu=not torch.cuda.is_available(),
        dataloader_pin_memory=torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )

    trainer.train()

    validation_metrics = trainer.evaluate(valid_dataset)
    test_output = trainer.predict(test_dataset)
    test_preds = np.argmax(test_output.predictions, axis=-1)

    clean_validation_metrics = {
        "loss": float(validation_metrics["eval_loss"]),
        "accuracy": float(validation_metrics["eval_accuracy"]),
        "macro_f1": float(validation_metrics["eval_macro_f1"]),
        "weighted_f1": float(validation_metrics["eval_weighted_f1"]),
    }
    clean_test_metrics = {
        "loss": float(test_output.metrics["test_loss"]),
        "accuracy": float(test_output.metrics["test_accuracy"]),
        "macro_f1": float(test_output.metrics["test_macro_f1"]),
        "weighted_f1": float(test_output.metrics["test_weighted_f1"]),
        "classification_report": detailed_report(test_output.label_ids.tolist(), test_preds.tolist()),
    }

    metrics = {
        "model_name": args.model_name,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "train_rows": len(train_frame),
        "valid_rows": len(valid_frame),
        "test_rows": len(test_frame),
        "max_length": args.max_length,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "train_batch_size": args.train_batch_size,
        "eval_batch_size": args.eval_batch_size,
        "model_saved": not args.skip_model_save,
        "validation": clean_validation_metrics,
        "test": clean_test_metrics,
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    if not args.skip_model_save:
        model_dir = output_dir / "model"
        trainer.save_model(str(model_dir))
        tokenizer.save_pretrained(str(model_dir))

    save_summary(output_dir, args, clean_validation_metrics, clean_test_metrics)

    print("Saved BanglaBERT artifacts to:", output_dir)
    print("Validation accuracy:", f"{clean_validation_metrics['accuracy']:.4f}")
    print("Validation macro F1:", f"{clean_validation_metrics['macro_f1']:.4f}")
    print("Test accuracy:", f"{clean_test_metrics['accuracy']:.4f}")
    print("Test macro F1:", f"{clean_test_metrics['macro_f1']:.4f}")


if __name__ == "__main__":
    main()
