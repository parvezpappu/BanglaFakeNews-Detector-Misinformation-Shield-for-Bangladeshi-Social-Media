from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)
from datasets import Dataset
import pandas as pd
import torch
import json
import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score

MODEL_NAME = "csebuetnlp/banglabert"
OUTPUT_DIR = Path("artifacts/category_model")

# Full path to your Drive folder
DATA_PATH = "/content/drive/MyDrive/BanglaFakeNews/category_training"


def build_text(df):
    return (df["headline"].fillna("") + " " + df["content"].fillna("")).tolist()


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "macro_f1": float(f1_score(labels, preds, average="macro")),
    }


def main():
    train = pd.read_csv(f"{DATA_PATH}/train.csv")
    valid = pd.read_csv(f"{DATA_PATH}/valid.csv")

    categories = sorted(train["category"].dropna().unique())
    label2id = {c: i for i, c in enumerate(categories)}
    id2label = {i: c for c, i in label2id.items()}

    print(f"Found {len(categories)} categories: {categories}")

    def convert(df):
        return Dataset.from_dict({
            "text": build_text(df),
            "label": [label2id.get(c, 0) for c in df["category"]]
        })

    train_ds = convert(train)
    valid_ds = convert(valid)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize(x):
        return tokenizer(x["text"], truncation=True, padding=True, max_length=256)

    train_ds = train_ds.map(tokenize, batched=True)
    valid_ds = valid_ds.map(tokenize, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(categories),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )

    model.config.use_cache = False
    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        num_train_epochs=2,
        learning_rate=2e-5,
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        report_to=[],
        use_cpu=not torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
    )

    trainer.train()

    trainer.save_model(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))

    with open(OUTPUT_DIR / "label_map.json", "w", encoding="utf-8") as f:
        json.dump({
            "label2id": label2id,
            "id2label": id2label
        }, f, ensure_ascii=False, indent=2)

    print("✅ Category model saved at:", OUTPUT_DIR)


if __name__ == "__main__":
    main()