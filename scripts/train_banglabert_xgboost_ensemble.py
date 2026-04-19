from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import xgboost as xgb
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
DATA_DIR = ROOT / "artifacts" / "phase1_dataset"
OUTPUT_DIR = ROOT / "artifacts" / "banglabert_xgboost_ensemble"
MODEL_NAME = "csebuetnlp/banglabert"
LABEL2ID = {"fake": 0, "real": 1}
ID2LABEL = {value: key for key, value in LABEL2ID.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train BanglaBERT + XGBoost ensemble.")
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--train-batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-valid-samples", type=int, default=None)
    parser.add_argument("--max-test-samples", type=int, default=None)
    parser.add_argument("--xgb-estimators", type=int, default=400)
    parser.add_argument("--xgb-max-depth", type=int, default=6)
    parser.add_argument("--xgb-learning-rate", type=float, default=0.05)
    parser.add_argument("--xgb-subsample", type=float, default=0.9)
    parser.add_argument("--xgb-colsample-bytree", type=float, default=0.9)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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
        labels,
        preds,
        target_names=label_names,
        digits=4,
        output_dict=True,
        zero_division=0,
    )


def basic_features(frame: pd.DataFrame) -> np.ndarray:
    headline = frame["headline"].fillna("").astype(str)
    content = frame["content"].fillna("").astype(str)
    category = frame["category"].fillna("").astype(str)
    return np.column_stack(
        [
            headline.str.len().to_numpy(),
            content.str.len().to_numpy(),
            headline.str.split().str.len().fillna(0).to_numpy(),
            content.str.split().str.len().fillna(0).to_numpy(),
            category.str.len().to_numpy(),
            headline.str.count(r"\d").to_numpy(),
            content.str.count(r"\d").to_numpy(),
            headline.str.count(r"[!?।,:;]").to_numpy(),
            content.str.count(r"[!?।,:;]").to_numpy(),
        ]
    ).astype(np.float32)


def category_features(
    train_frame: pd.DataFrame,
    valid_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    combined = pd.concat(
        [
            train_frame[["category"]].assign(split="train"),
            valid_frame[["category"]].assign(split="valid"),
            test_frame[["category"]].assign(split="test"),
        ],
        ignore_index=True,
    )
    encoded = pd.get_dummies(combined["category"].fillna(""), prefix="cat", dtype=np.float32)
    train_mask = combined["split"] == "train"
    valid_mask = combined["split"] == "valid"
    test_mask = combined["split"] == "test"
    return (
        encoded.loc[train_mask].to_numpy(),
        encoded.loc[valid_mask].to_numpy(),
        encoded.loc[test_mask].to_numpy(),
    )


def confidence_features(probs: np.ndarray) -> np.ndarray:
    max_prob = probs.max(axis=1, keepdims=True)
    min_prob = probs.min(axis=1, keepdims=True)
    margin = max_prob - min_prob
    entropy = -(probs * np.log(np.clip(probs, 1e-9, 1.0))).sum(axis=1, keepdims=True)
    return np.concatenate([max_prob, margin, entropy], axis=1).astype(np.float32)


def extract_cls_embeddings(
    model_path: str,
    tokenizer: AutoTokenizer,
    frame: pd.DataFrame,
    max_length: int,
    batch_size: int,
) -> np.ndarray:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModel.from_pretrained(model_path)
    model.to(device)
    model.eval()

    texts = build_text(frame)
    outputs: list[np.ndarray] = []

    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
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


def tune_alpha(valid_labels: np.ndarray, bert_probs: np.ndarray, xgb_probs: np.ndarray) -> tuple[float, dict[str, object]]:
    best_alpha = 0.5
    best_metrics = None
    best_score = -1.0

    for step in range(21):
        alpha = step / 20.0
        ensemble_probs = alpha * bert_probs + (1.0 - alpha) * xgb_probs
        metrics = eval_from_probs(valid_labels, ensemble_probs)
        if metrics["macro_f1"] > best_score:
            best_alpha = alpha
            best_score = float(metrics["macro_f1"])
            best_metrics = metrics

    assert best_metrics is not None
    return best_alpha, best_metrics


def build_meta_features(bert_probs: np.ndarray, xgb_probs: np.ndarray) -> np.ndarray:
    bert_conf = confidence_features(bert_probs)
    xgb_conf = confidence_features(xgb_probs)
    agreement = (
        np.argmax(bert_probs, axis=1) == np.argmax(xgb_probs, axis=1)
    ).astype(np.float32).reshape(-1, 1)
    diff = np.abs(bert_probs - xgb_probs).astype(np.float32)
    return np.concatenate(
        [bert_probs, xgb_probs, bert_conf, xgb_conf, diff, agreement],
        axis=1,
    ).astype(np.float32)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    train_frame = load_frame("train", args.max_train_samples, args.seed)
    valid_frame = load_frame("valid", args.max_valid_samples, args.seed)
    test_frame = load_frame("test", args.max_test_samples, args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    bert_classifier = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=2,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    train_dataset = tokenize_dataset(frame_to_dataset(train_frame), tokenizer, args.max_length)
    valid_dataset = tokenize_dataset(frame_to_dataset(valid_frame), tokenizer, args.max_length)
    test_dataset = tokenize_dataset(frame_to_dataset(test_frame), tokenizer, args.max_length)

    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR / "checkpoints"),
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
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        save_total_limit=args.save_total_limit,
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

    trainer.train()
    trainer.save_model(str(OUTPUT_DIR / "banglabert_model"))
    tokenizer.save_pretrained(str(OUTPUT_DIR / "banglabert_model"))

    train_pred = trainer.predict(train_dataset)
    valid_pred = trainer.predict(valid_dataset)
    test_pred = trainer.predict(test_dataset)

    bert_train_probs = softmax(train_pred.predictions)
    bert_valid_probs = softmax(valid_pred.predictions)
    bert_test_probs = softmax(test_pred.predictions)

    train_embeddings = extract_cls_embeddings(
        str(OUTPUT_DIR / "banglabert_model"),
        tokenizer,
        train_frame,
        args.max_length,
        args.eval_batch_size,
    )
    valid_embeddings = extract_cls_embeddings(
        str(OUTPUT_DIR / "banglabert_model"),
        tokenizer,
        valid_frame,
        args.max_length,
        args.eval_batch_size,
    )
    test_embeddings = extract_cls_embeddings(
        str(OUTPUT_DIR / "banglabert_model"),
        tokenizer,
        test_frame,
        args.max_length,
        args.eval_batch_size,
    )

    x_train = np.concatenate([train_embeddings, bert_train_probs, basic_features(train_frame)], axis=1)
    x_valid = np.concatenate([valid_embeddings, bert_valid_probs, basic_features(valid_frame)], axis=1)
    x_test = np.concatenate([test_embeddings, bert_test_probs, basic_features(test_frame)], axis=1)

    y_train = train_frame["label"].map(LABEL2ID).to_numpy()
    y_valid = valid_frame["label"].map(LABEL2ID).to_numpy()
    y_test = test_frame["label"].map(LABEL2ID).to_numpy()

    train_category_features, valid_category_features, test_category_features = category_features(
        train_frame,
        valid_frame,
        test_frame,
    )

    xgb_model = xgb.XGBClassifier(
        n_estimators=args.xgb_estimators,
        max_depth=args.xgb_max_depth,
        learning_rate=args.xgb_learning_rate,
        subsample=args.xgb_subsample,
        colsample_bytree=args.xgb_colsample_bytree,
        objective="multi:softprob",
        num_class=2,
        eval_metric="mlogloss",
        tree_method="hist",
        random_state=args.seed,
    )
    x_train = np.concatenate(
        [
            train_embeddings,
            bert_train_probs,
            confidence_features(bert_train_probs),
            basic_features(train_frame),
            train_category_features,
        ],
        axis=1,
    )
    x_valid = np.concatenate(
        [
            valid_embeddings,
            bert_valid_probs,
            confidence_features(bert_valid_probs),
            basic_features(valid_frame),
            valid_category_features,
        ],
        axis=1,
    )
    x_test = np.concatenate(
        [
            test_embeddings,
            bert_test_probs,
            confidence_features(bert_test_probs),
            basic_features(test_frame),
            test_category_features,
        ],
        axis=1,
    )

    xgb_model.fit(
        x_train,
        y_train,
        eval_set=[(x_valid, y_valid)],
        verbose=False,
    )
    joblib.dump(xgb_model, OUTPUT_DIR / "xgboost_model.joblib")

    xgb_valid_probs = xgb_model.predict_proba(x_valid)
    xgb_test_probs = xgb_model.predict_proba(x_test)

    alpha, ensemble_valid_metrics = tune_alpha(y_valid, bert_valid_probs, xgb_valid_probs)
    ensemble_test_probs = alpha * bert_test_probs + (1.0 - alpha) * xgb_test_probs
    meta_train_x = build_meta_features(bert_valid_probs, xgb_valid_probs)
    meta_test_x = build_meta_features(bert_test_probs, xgb_test_probs)
    stacking_model = LogisticRegression(
        max_iter=1000,
        random_state=args.seed,
    )
    stacking_model.fit(meta_train_x, y_valid)
    joblib.dump(stacking_model, OUTPUT_DIR / "stacking_model.joblib")
    stacking_test_probs = stacking_model.predict_proba(meta_test_x)

    metrics = {
        "device": device,
        "train_rows": len(train_frame),
        "valid_rows": len(valid_frame),
        "test_rows": len(test_frame),
        "banglabert": {
            "validation": eval_from_probs(y_valid, bert_valid_probs),
            "test": eval_from_probs(y_test, bert_test_probs),
        },
        "xgboost": {
            "validation": eval_from_probs(y_valid, xgb_valid_probs),
            "test": eval_from_probs(y_test, xgb_test_probs),
        },
        "ensemble": {
            "alpha_for_banglabert": alpha,
            "alpha_for_xgboost": 1.0 - alpha,
            "validation": ensemble_valid_metrics,
            "test": eval_from_probs(y_test, ensemble_test_probs),
        },
        "stacking_ensemble": {
            "meta_model": "logistic_regression",
            "train_source": "validation_predictions",
            "test": eval_from_probs(y_test, stacking_test_probs),
        },
        "config": {
            "model_name": args.model_name,
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "train_batch_size": args.train_batch_size,
            "eval_batch_size": args.eval_batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "max_length": args.max_length,
            "xgb_estimators": args.xgb_estimators,
            "xgb_max_depth": args.xgb_max_depth,
            "xgb_learning_rate": args.xgb_learning_rate,
        },
    }

    (OUTPUT_DIR / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    summary = [
        "# BanglaBERT + XGBoost Ensemble",
        "",
        f"- Device: `{device}`",
        f"- BanglaBERT test macro F1: `{metrics['banglabert']['test']['macro_f1']:.4f}`",
        f"- XGBoost test macro F1: `{metrics['xgboost']['test']['macro_f1']:.4f}`",
        f"- Ensemble alpha for BanglaBERT: `{alpha:.2f}`",
        f"- Ensemble test macro F1: `{metrics['ensemble']['test']['macro_f1']:.4f}`",
        f"- Stacking ensemble test macro F1: `{metrics['stacking_ensemble']['test']['macro_f1']:.4f}`",
        "",
        "## Notes",
        "",
        "- XGBoost uses BanglaBERT CLS embeddings, BanglaBERT probabilities, confidence features, simple text features, and category one-hot features.",
        "- Weighted ensemble uses validation-tuned alpha.",
        "- Stacking ensemble learns a logistic-regression combiner from validation-set branch outputs.",
        "",
    ]
    (OUTPUT_DIR / "README.md").write_text("\n".join(summary), encoding="utf-8")

    print("Saved ensemble artifacts to:", OUTPUT_DIR)
    print("BanglaBERT test macro F1:", f"{metrics['banglabert']['test']['macro_f1']:.4f}")
    print("XGBoost test macro F1:", f"{metrics['xgboost']['test']['macro_f1']:.4f}")
    print("Ensemble test macro F1:", f"{metrics['ensemble']['test']['macro_f1']:.4f}")
    print("Stacking ensemble test macro F1:", f"{metrics['stacking_ensemble']['test']['macro_f1']:.4f}")


if __name__ == "__main__":
    main()
