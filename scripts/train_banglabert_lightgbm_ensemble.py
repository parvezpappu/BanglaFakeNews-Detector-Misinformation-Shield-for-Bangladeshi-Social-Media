from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import lightgbm as lgb
from datasets import Dataset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from transformers import (
    AutoModel,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = ROOT / "artifacts" / "phase1_dataset"
DEFAULT_OUTPUT_DIR = ROOT / "artifacts" / "banglabert_lightgbm_ensemble"

MODEL_NAME = "csebuetnlp/banglabert"
LABEL2ID = {"fake": 0, "real": 1}
ID2LABEL = {value: key for key, value in LABEL2ID.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train BanglaBERT + LightGBM ensemble.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument("--epochs", type=float, default=5.0)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--train-batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-valid-samples", type=int, default=None)
    parser.add_argument("--max-test-samples", type=int, default=None)
    parser.add_argument("--lgb-estimators", type=int, default=500)
    parser.add_argument("--lgb-max-depth", type=int, default=7)
    parser.add_argument("--lgb-learning-rate", type=float, default=0.03)
    parser.add_argument("--lgb-subsample", type=float, default=0.8)
    parser.add_argument("--lgb-colsample-bytree", type=float, default=0.8)
    parser.add_argument("--lgb-num-leaves", type=int, default=63)
    parser.add_argument("--lgb-min-child-samples", type=int, default=20)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_text(frame: pd.DataFrame) -> list[str]:
    headline = frame["headline"].fillna("").astype(str).str.strip()
    content = frame["content"].fillna("").astype(str).str.strip()
    return (
        "[HEADLINE] "
        + headline
        + " [CONTENT] "
        + content
    ).tolist()


def load_frame(data_dir: Path, name: str, max_samples: int | None, seed: int) -> pd.DataFrame:
    frame = pd.read_csv(data_dir / f"{name}.csv")
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
        return tokenizer(batch["text"], truncation=True, max_length=max_length)

    return dataset.map(tokenize_batch, batched=True, remove_columns=["text"])


def compute_metrics(eval_pred: tuple[np.ndarray, np.ndarray]) -> dict[str, float]:
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro"),
        "weighted_f1": f1_score(labels, preds, average="weighted"),
    }


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp_values = np.exp(shifted)
    return exp_values / exp_values.sum(axis=1, keepdims=True)


def zero_safe_report(labels: np.ndarray, preds: np.ndarray) -> dict[str, object]:
    label_names = [ID2LABEL[idx] for idx in sorted(ID2LABEL)]
    return classification_report(
        labels, preds, target_names=label_names, digits=4, output_dict=True, zero_division=0,
    )


def basic_features(frame: pd.DataFrame) -> np.ndarray:
    headline = frame["headline"].fillna("").astype(str)
    content = frame["content"].fillna("").astype(str)
    return np.column_stack(
        [
            headline.str.len().to_numpy(),
            content.str.len().to_numpy(),
            headline.str.split().str.len().fillna(0).to_numpy(),
            content.str.split().str.len().fillna(0).to_numpy(),
            headline.str.count(r"\d").to_numpy(),
            content.str.count(r"\d").to_numpy(),
            headline.str.count(r"[!?।,:;]").to_numpy(),
            content.str.count(r"[!?।,:;]").to_numpy(),
            headline.str.count(r"[A-Za-z]").to_numpy(),
            content.str.count(r"[A-Za-z]").to_numpy(),
        ]
    ).astype(np.float32)


def confidence_features(probs: np.ndarray) -> np.ndarray:
    max_prob = probs.max(axis=1, keepdims=True)
    min_prob = probs.min(axis=1, keepdims=True)
    margin = max_prob - min_prob
    entropy = -(probs * np.log(np.clip(probs, 1e-9, 1.0))).sum(axis=1, keepdims=True)
    return np.concatenate([max_prob, margin, entropy], axis=1).astype(np.float32)


def extract_cls_embeddings(
    model_path: str, tokenizer: AutoTokenizer, frame: pd.DataFrame,
    max_length: int, batch_size: int,
) -> np.ndarray:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModel.from_pretrained(model_path)
    model.to(device)
    model.eval()

    texts = build_text(frame)
    outputs: list[np.ndarray] = []

    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        encoded = tokenizer(batch, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        encoded = {k: v.to(device) for k, v in encoded.items()}
        with torch.no_grad():
            hidden = model(**encoded).last_hidden_state[:, 0, :]
        outputs.append(hidden.cpu().numpy())

    return np.concatenate(outputs, axis=0)


def eval_from_probs(labels: np.ndarray, probs: np.ndarray) -> dict[str, object]:
    preds = np.argmax(probs, axis=1)
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "macro_f1": float(f1_score(labels, preds, average="macro")),
        "weighted_f1": float(f1_score(labels, preds, average="weighted")),
        "classification_report": zero_safe_report(labels, preds),
    }


def tune_alpha(valid_labels: np.ndarray, bert_probs: np.ndarray, lgb_probs: np.ndarray) -> tuple[float, dict[str, object]]:
    best_alpha = 0.5
    best_metrics = None
    best_score = -1.0

    for step in range(21):
        alpha = step / 20.0
        ensemble_probs = alpha * bert_probs + (1.0 - alpha) * lgb_probs
        metrics = eval_from_probs(valid_labels, ensemble_probs)
        if metrics["macro_f1"] > best_score:
            best_alpha = alpha
            best_score = float(metrics["macro_f1"])
            best_metrics = metrics

    assert best_metrics is not None
    return best_alpha, best_metrics


def build_meta_features(bert_probs: np.ndarray, lgb_probs: np.ndarray) -> np.ndarray:
    bert_conf = confidence_features(bert_probs)
    lgb_conf = confidence_features(lgb_probs)
    agreement = (np.argmax(bert_probs, axis=1) == np.argmax(lgb_probs, axis=1)).astype(np.float32).reshape(-1, 1)
    diff = np.abs(bert_probs - lgb_probs).astype(np.float32)
    return np.concatenate([bert_probs, lgb_probs, bert_conf, lgb_conf, diff, agreement], axis=1).astype(np.float32)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}")
    print(f"Data directory: {args.data_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Epochs: {args.epochs} | Batch size: {args.train_batch_size} | LR: {args.learning_rate}")

    train_frame = load_frame(args.data_dir, "train", args.max_train_samples, args.seed)
    valid_frame = load_frame(args.data_dir, "valid", args.max_valid_samples, args.seed)
    test_frame = load_frame(args.data_dir, "test", args.max_test_samples, args.seed)

    print(f"Train: {len(train_frame)} | Valid: {len(valid_frame)} | Test: {len(test_frame)}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    bert_classifier = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, num_labels=2, id2label=ID2LABEL, label2id=LABEL2ID,
    )

    train_dataset = tokenize_dataset(frame_to_dataset(train_frame), tokenizer, args.max_length)
    valid_dataset = tokenize_dataset(frame_to_dataset(valid_frame), tokenizer, args.max_length)
    test_dataset = tokenize_dataset(frame_to_dataset(test_frame), tokenizer, args.max_length)

    training_args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        do_train=True,
        do_eval=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=50,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.epochs,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        save_total_limit=args.save_total_limit,
        fp16=torch.cuda.is_available(),
        report_to=[],
        seed=args.seed,
        use_cpu=not torch.cuda.is_available(),
        dataloader_pin_memory=torch.cuda.is_available(),
        dataloader_num_workers=args.num_workers,
    )

    trainer = Trainer(
        model=bert_classifier,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )

    print("\n=== Training BanglaBERT ===")
    trainer.train()
    trainer.save_model(str(output_dir / "banglabert_model"))
    tokenizer.save_pretrained(str(output_dir / "banglabert_model"))

    train_pred = trainer.predict(train_dataset)
    valid_pred = trainer.predict(valid_dataset)
    test_pred = trainer.predict(test_dataset)

    bert_train_probs = softmax(train_pred.predictions)
    bert_valid_probs = softmax(valid_pred.predictions)
    bert_test_probs = softmax(test_pred.predictions)

    print("\n=== Extracting CLS Embeddings ===")
    train_embeddings = extract_cls_embeddings(
        str(output_dir / "banglabert_model"), tokenizer, train_frame, args.max_length, args.eval_batch_size,
    )
    valid_embeddings = extract_cls_embeddings(
        str(output_dir / "banglabert_model"), tokenizer, valid_frame, args.max_length, args.eval_batch_size,
    )
    test_embeddings = extract_cls_embeddings(
        str(output_dir / "banglabert_model"), tokenizer, test_frame, args.max_length, args.eval_batch_size,
    )

    y_train = train_frame["label"].map(LABEL2ID).to_numpy()
    y_valid = valid_frame["label"].map(LABEL2ID).to_numpy()
    y_test = test_frame["label"].map(LABEL2ID).to_numpy()

    x_train = np.concatenate(
        [train_embeddings, bert_train_probs, confidence_features(bert_train_probs), basic_features(train_frame)], axis=1,
    )
    x_valid = np.concatenate(
        [valid_embeddings, bert_valid_probs, confidence_features(bert_valid_probs), basic_features(valid_frame)], axis=1,
    )
    x_test = np.concatenate(
        [test_embeddings, bert_test_probs, confidence_features(bert_test_probs), basic_features(test_frame)], axis=1,
    )

    print(f"\nLightGBM feature dimensions: {x_train.shape[1]}")

    print("\n=== Training LightGBM ===")
    lgb_model = lgb.LGBMClassifier(
        n_estimators=args.lgb_estimators,
        max_depth=args.lgb_max_depth,
        learning_rate=args.lgb_learning_rate,
        subsample=args.lgb_subsample,
        colsample_bytree=args.lgb_colsample_bytree,
        num_leaves=args.lgb_num_leaves,
        min_child_samples=args.lgb_min_child_samples,
        objective="multiclass",
        num_class=2,
        random_state=args.seed,
        verbose=-1,
        force_col_wise=True,
    )
    lgb_model.fit(
        x_train, y_train,
        eval_set=[(x_valid, y_valid)],
        eval_metric="multi_logloss",
    )
    joblib.dump(lgb_model, output_dir / "lightgbm_model.joblib")

    lgb_valid_probs = lgb_model.predict_proba(x_valid)
    lgb_test_probs = lgb_model.predict_proba(x_test)

    print("\n=== Tuning Ensemble Weights ===")
    alpha, ensemble_valid_metrics = tune_alpha(y_valid, bert_valid_probs, lgb_valid_probs)
    ensemble_test_probs = alpha * bert_test_probs + (1.0 - alpha) * lgb_test_probs

    meta_train_x = build_meta_features(bert_valid_probs, lgb_valid_probs)
    meta_test_x = build_meta_features(bert_test_probs, lgb_test_probs)
    stacking_model = LogisticRegression(max_iter=1000, random_state=args.seed)
    stacking_model.fit(meta_train_x, y_valid)
    joblib.dump(stacking_model, output_dir / "stacking_model.joblib")
    stacking_test_probs = stacking_model.predict_proba(meta_test_x)

    metrics = {
        "device": device,
        "train_rows": len(train_frame),
        "valid_rows": len(valid_frame),
        "test_rows": len(test_frame),
        "feature_count": x_train.shape[1],
        "banglabert": {
            "validation": eval_from_probs(y_valid, bert_valid_probs),
            "test": eval_from_probs(y_test, bert_test_probs),
        },
        "lightgbm": {
            "validation": eval_from_probs(y_valid, lgb_valid_probs),
            "test": eval_from_probs(y_test, lgb_test_probs),
        },
        "ensemble": {
            "alpha_for_banglabert": alpha,
            "alpha_for_lightgbm": 1.0 - alpha,
            "validation": ensemble_valid_metrics,
            "test": eval_from_probs(y_test, ensemble_test_probs),
        },
        "stacking_ensemble": {
            "meta_model": "logistic_regression",
            "test": eval_from_probs(y_test, stacking_test_probs),
        },
        "config": {
            "model_name": args.model_name,
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "batch_size": args.train_batch_size,
            "max_length": args.max_length,
            "warmup_ratio": args.warmup_ratio,
            "fp16": torch.cuda.is_available(),
            "lgb_estimators": args.lgb_estimators,
            "lgb_max_depth": args.lgb_max_depth,
            "lgb_num_leaves": args.lgb_num_leaves,
        },
    }

    (output_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = [
        "# BanglaBERT + LightGBM Ensemble",
        "",
        f"- Device: `{device}`",
        f"- Epochs: `{args.epochs}`",
        f"- Batch size: `{args.train_batch_size}`",
        f"- FP16: `{torch.cuda.is_available()}`",
        f"- Feature count: `{x_train.shape[1]}`",
        f"- BanglaBERT test macro F1: `{metrics['banglabert']['test']['macro_f1']:.4f}`",
        f"- LightGBM test macro F1: `{metrics['lightgbm']['test']['macro_f1']:.4f}`",
        f"- Ensemble alpha: `{alpha:.2f}`",
        f"- Ensemble test macro F1: `{metrics['ensemble']['test']['macro_f1']:.4f}`",
        f"- Stacking test macro F1: `{metrics['stacking_ensemble']['test']['macro_f1']:.4f}`",
        "",
        "## Improvements over v1",
        "- LightGBM instead of XGBoost (better with NLP features)",
        "- 5 epochs instead of 2-3",
        "- Larger batch size (16) with fp16 mixed precision",
        "- Warmup ratio 0.1 for better convergence",
        "- More basic features (English char count)",
        "",
    ]
    (output_dir / "README.md").write_text("\n".join(summary), encoding="utf-8")

    print("\n" + "=" * 55)
    print("FINAL RESULTS")
    print("=" * 55)
    print(f"BanglaBERT  test F1: {metrics['banglabert']['test']['macro_f1']:.4f}")
    print(f"LightGBM    test F1: {metrics['lightgbm']['test']['macro_f1']:.4f}")
    print(f"Ensemble    test F1: {metrics['ensemble']['test']['macro_f1']:.4f}")
    print(f"Stacking    test F1: {metrics['stacking_ensemble']['test']['macro_f1']:.4f}")
    print(f"\nSaved to: {output_dir}")


if __name__ == "__main__":
    main()
