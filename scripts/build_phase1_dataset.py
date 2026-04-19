from __future__ import annotations

import argparse
import csv
import hashlib
import random
from collections import Counter
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
REAL_PATH = ROOT / "Dataset" / "Real" / "LabeledAuthentic-7K.csv"
MIXED_PATH = ROOT / "Dataset" / "Fake" / "bengali_fake_news.csv"
OUTPUT_DIR = ROOT / "artifacts" / "phase1_dataset"

REAL_TARGET = 5000
FAKE_TARGET = 5000
RANDOM_SEED = 42
DEFAULT_MIXED_FAKE_LABEL = "1"


def normalize_text(value: str | None) -> str:
    if not value:
        return ""
    return " ".join(value.replace("\ufeff", " ").split())


def stable_key(headline: str, content: str) -> str:
    joined = f"{normalize_text(headline).casefold()}||{normalize_text(content).casefold()}"
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def unify_real_rows(rows: Iterable[dict[str, str]]) -> list[dict[str, str]]:
    cleaned: list[dict[str, str]] = []
    for row in rows:
        headline = normalize_text(row.get("headline"))
        content = normalize_text(row.get("content"))
        if not headline or not content:
            continue
        cleaned.append(
            {
                "record_id": normalize_text(row.get("articleID")),
                "label": "real",
                "label_id": "1",
                "category": normalize_text(row.get("category")),
                "headline": headline,
                "content": content,
                "source": normalize_text(row.get("source")),
                "domain": normalize_text(row.get("domain")),
                "date": normalize_text(row.get("date")),
                "relation": normalize_text(row.get("relation")),
                "origin_file": REAL_PATH.name,
                "raw_label": normalize_text(row.get("label")),
            }
        )
    return cleaned


def unify_mixed_rows(rows: Iterable[dict[str, str]], fake_label: str) -> list[dict[str, str]]:
    cleaned: list[dict[str, str]] = []
    for idx, row in enumerate(rows, start=1):
        if normalize_text(row.get("label")) != fake_label:
            continue
        headline = normalize_text(row.get("headline"))
        content = normalize_text(row.get("content"))
        if not headline or not content:
            continue
        cleaned.append(
            {
                "record_id": str(idx),
                "label": "fake",
                "label_id": "0",
                "category": normalize_text(row.get("category")),
                "headline": headline,
                "content": content,
                "source": "",
                "domain": "",
                "date": "",
                "relation": "",
                "origin_file": MIXED_PATH.name,
                "raw_label": normalize_text(row.get("label")),
            }
        )
    return cleaned


def deduplicate(rows: Iterable[dict[str, str]]) -> list[dict[str, str]]:
    seen: set[str] = set()
    unique_rows: list[dict[str, str]] = []
    for row in rows:
        key = stable_key(row["headline"], row["content"])
        if key in seen:
            continue
        seen.add(key)
        row = dict(row)
        row["dedupe_key"] = key
        unique_rows.append(row)
    return unique_rows


def sample_rows(rows: list[dict[str, str]], target_size: int, rng: random.Random) -> list[dict[str, str]]:
    if len(rows) < target_size:
        raise ValueError(f"Need {target_size} rows but found only {len(rows)} after cleaning.")
    sampled = list(rows)
    rng.shuffle(sampled)
    return sampled[:target_size]


def remove_cross_duplicates(
    primary_rows: list[dict[str, str]], blocked_rows: list[dict[str, str]]
) -> list[dict[str, str]]:
    blocked = {row["dedupe_key"] for row in blocked_rows}
    return [row for row in primary_rows if row["dedupe_key"] not in blocked]


def split_rows(rows: list[dict[str, str]]) -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]]:
    by_label: dict[str, list[dict[str, str]]] = {"real": [], "fake": []}
    for row in rows:
        by_label[row["label"]].append(row)

    train: list[dict[str, str]] = []
    valid: list[dict[str, str]] = []
    test: list[dict[str, str]] = []

    for label_rows in by_label.values():
        total = len(label_rows)
        train_end = int(total * 0.8)
        valid_end = train_end + int(total * 0.1)
        train.extend(label_rows[:train_end])
        valid.extend(label_rows[train_end:valid_end])
        test.extend(label_rows[valid_end:])

    return train, valid, test


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = [
        "record_id",
        "label",
        "label_id",
        "category",
        "headline",
        "content",
        "source",
        "domain",
        "date",
        "relation",
        "origin_file",
        "raw_label",
        "dedupe_key",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_summary(
    rows: list[dict[str, str]],
    train: list[dict[str, str]],
    valid: list[dict[str, str]],
    test: list[dict[str, str]],
    mixed_fake_label: str,
    available_real: int,
    available_fake: int,
) -> str:
    categories = Counter(row["category"] for row in rows)
    avg_headline = sum(len(row["headline"]) for row in rows) / len(rows)
    avg_content = sum(len(row["content"]) for row in rows) / len(rows)
    lines = [
        "# Phase 1 Dataset Summary",
        "",
        f"- Random seed: `{RANDOM_SEED}`",
        f"- Mixed-file label treated as fake: `{mixed_fake_label}`",
        f"- Available cleaned real rows before sampling: `{available_real}`",
        f"- Available cleaned fake rows before sampling: `{available_fake}`",
        f"- Total rows: `{len(rows)}`",
        f"- Label counts: `{dict(Counter(row['label'] for row in rows))}`",
        f"- Split sizes: train=`{len(train)}`, valid=`{len(valid)}`, test=`{len(test)}`",
        f"- Average headline length: `{avg_headline:.2f}` characters",
        f"- Average content length: `{avg_content:.2f}` characters",
        "",
        "## Audit Note",
        "",
        "- The mixed CSV contains both labels and overlaps with the authentic CSV.",
        "- This phase-1 build removes exact cross-file duplicates before sampling.",
        "- If later auditing shows the opposite label should be treated as fake, rerun the script with `--mixed-fake-label 0`.",
        "",
        "## Top Categories",
        "",
    ]
    for category, count in categories.most_common(10):
        lines.append(f"- {category}: {count}")
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the phase-1 Bangla fake news dataset.")
    parser.add_argument("--real-target", type=int, default=REAL_TARGET)
    parser.add_argument("--fake-target", type=int, default=FAKE_TARGET)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument(
        "--mixed-fake-label",
        choices=["0", "1"],
        default=DEFAULT_MIXED_FAKE_LABEL,
        help="Which label from the mixed CSV should be treated as fake for phase 1.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    real_rows = deduplicate(unify_real_rows(read_csv(REAL_PATH)))
    mixed_rows = deduplicate(unify_mixed_rows(read_csv(MIXED_PATH), args.mixed_fake_label))
    fake_rows = remove_cross_duplicates(mixed_rows, real_rows)

    sampled_real = sample_rows(real_rows, args.real_target, rng)
    sampled_fake = sample_rows(fake_rows, args.fake_target, rng)

    combined = sampled_real + sampled_fake
    rng.shuffle(combined)

    train, valid, test = split_rows(combined)

    write_csv(OUTPUT_DIR / "all.csv", combined)
    write_csv(OUTPUT_DIR / "train.csv", train)
    write_csv(OUTPUT_DIR / "valid.csv", valid)
    write_csv(OUTPUT_DIR / "test.csv", test)
    (OUTPUT_DIR / "README.md").write_text(
        build_summary(
            combined,
            train,
            valid,
            test,
            args.mixed_fake_label,
            len(real_rows),
            len(fake_rows),
        ),
        encoding="utf-8",
    )

    print(f"Wrote dataset files to: {OUTPUT_DIR}")
    print(f"Combined rows: {len(combined)}")
    print(f"Train/valid/test: {len(train)}/{len(valid)}/{len(test)}")
    print(f"Mixed-file fake label: {args.mixed_fake_label}")


if __name__ == "__main__":
    main()
