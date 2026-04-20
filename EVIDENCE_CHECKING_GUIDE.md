# Evidence-Based Fact Checking Upgrade

The current classifier predicts whether text looks fake or real. The evidence layer adds a second step:

```text
user news -> model prediction -> trusted-source search -> evidence hint
```

This does not require local training and does not save large files locally.

## Backend Environment Variables

Recommended easy setup: use Tavily. Create a Tavily account, copy your API key, and set:

```text
TAVILY_API_KEY=your_tavily_api_key
```

The backend will use Tavily automatically when this key exists.

Google Programmable Search is still supported as a fallback, but Google says Custom Search JSON API is closed to new customers. Use Google only if your project already has JSON API access:

```text
GOOGLE_SEARCH_API_KEY=your_google_api_key
GOOGLE_SEARCH_CX=your_search_engine_id
```

Optional:

```text
EVIDENCE_SEARCH_RESULTS=5
EVIDENCE_SEARCH_TIMEOUT=8
```

Without these keys, the app still works. It returns the BanglaBERT + XGBoost model output and an evidence search link, but marks the evidence result as `model_only`.

## Current Evidence Hints

The backend returns:

- `likely_real`: model predicts real and multiple trusted-source matches were found
- `conflicting_evidence`: model predicts fake, but trusted-source matches exist
- `limited_evidence`: only one trusted-source match was found
- `likely_fake_low_evidence`: model predicts fake and no trusted-source match was found
- `uncertain`: model predicts real but evidence was not found
- `model_only`: evidence search is not configured or failed

These are evidence hints, not final human fact-check verdicts.

## Trusted Sources

The first version searches across selected Bangladeshi news, fact-checking, and official domains. Edit `TRUSTED_EVIDENCE_DOMAINS` in `app/backend/config.py` to add or remove sources.

## Optional Colab Training Direction

If you want a stronger evidence model later, train it in Google Colab, not locally. The target task should be:

```text
claim + evidence snippet -> supports / refutes / not_enough_info
```

Minimal Colab skeleton:

```python
!pip install -q transformers datasets accelerate evaluate scikit-learn

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

MODEL_NAME = "csebuetnlp/banglabert"

# Prepare a CSV with columns:
# claim,evidence,label
# label values: supports, refutes, not_enough_info
dataset = load_dataset("csv", data_files={
    "train": "/content/train_evidence.csv",
    "validation": "/content/valid_evidence.csv",
})

label2id = {"supports": 0, "refutes": 1, "not_enough_info": 2}
id2label = {value: key for key, value in label2id.items()}

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize(batch):
    pairs = [
        f"[CLAIM] {claim} [EVIDENCE] {evidence}"
        for claim, evidence in zip(batch["claim"], batch["evidence"])
    ]
    encoded = tokenizer(pairs, truncation=True, max_length=256)
    encoded["labels"] = [label2id[label] for label in batch["label"]]
    return encoded

tokenized = dataset.map(tokenize, batched=True, remove_columns=dataset["train"].column_names)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=3,
    id2label=id2label,
    label2id=label2id,
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro"),
    }

args = TrainingArguments(
    output_dir="/content/evidence_model",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="macro_f1",
    report_to=[],
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["validation"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()
trainer.save_model("/content/evidence_model")
tokenizer.save_pretrained("/content/evidence_model")
```

After training, export `/content/evidence_model` from Colab and integrate it as a separate evidence-comparison branch.
