from __future__ import annotations

import argparse
import csv
import hashlib
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT / "Dataset"
REAL_DIR = DATASET_DIR / "Real"
FAKE_DIR = DATASET_DIR / "Fake"
OUTPUT_DIR = ROOT / "artifacts" / "phase1_dataset"

RANDOM_SEED = 42
LABEL_FAKE = 0
LABEL_REAL = 1

TEXT_COLUMNS = ("content", "text", "article", "body", "news")
HEADLINE_COLUMNS = ("headline", "title", "heading")
CATEGORY_COLUMNS = ("category", "class", "topic")
LABEL_COLUMNS = ("label", "target", "class_label", "is_fake", "authenticity")


@dataclass
class SourceStats:
    path: str
    rows_read: int = 0
    rows_kept: int = 0
    rows_skipped_empty: int = 0
    labels: Counter[str] | None = None

    def __post_init__(self) -> None:
        if self.labels is None:
            self.labels = Counter()


def normalize_text(value: object | None) -> str:
    if value is None:
        return ""
    text = str(value).replace("\ufeff", " ").replace("\u00a0", " ")
    return " ".join(text.split())


def normalize_header(value: str) -> str:
    return normalize_text(value).strip().lower().replace(" ", "_")


def stable_key(headline: str, content: str) -> str:
    value = f"{normalize_text(headline).casefold()}||{normalize_text(content).casefold()}"
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def first_existing(row: dict[str, str], names: Iterable[str]) -> str:
    for name in names:
        if name in row:
            value = normalize_text(row.get(name))
            if value:
                return value
    return ""


def sniff_has_header(path: Path) -> bool:
    sample = path.read_text(encoding="utf-8-sig", errors="replace")[:4096]
    try:
        return csv.Sniffer().has_header(sample)
    except csv.Error:
        first_line = sample.splitlines()[0].lower() if sample.splitlines() else ""
        return any(column in first_line for column in ("headline", "content", "label", "title"))


def read_csv_rows(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    has_header = sniff_has_header(path)
    with path.open("r", encoding="utf-8-sig", newline="", errors="replace") as handle:
        if has_header:
            reader = csv.DictReader(handle)
            raw_fields = reader.fieldnames or []
            fields = [normalize_header(field) for field in raw_fields]
            rows: list[dict[str, str]] = []
            for raw_row in reader:
                row = {
                    normalize_header(key): normalize_text(value)
                    for key, value in raw_row.items()
                    if key is not None
                }
                rows.append(row)
            return rows, fields

        reader = csv.reader(handle)
        rows = []
        max_width = 0
        for values in reader:
            cleaned = [normalize_text(value) for value in values]
            max_width = max(max_width, len(cleaned))
            rows.append({f"column_{idx}": value for idx, value in enumerate(cleaned)})
        return rows, [f"column_{idx}" for idx in range(max_width)]


def label_from_real_split(raw_label: str) -> tuple[int | None, str]:
    normalized = normalize_text(raw_label).lower()
    if normalized in {"3", "3.0", "real", "true", "authentic", "1_real"}:
        return LABEL_REAL, "real_split_label_3"
    if normalized in {"0", "0.0", "1", "1.0", "2", "2.0", "fake", "false"}:
        return LABEL_FAKE, "real_split_label_0_1_2"
    return None, "unknown_real_split_label"


def label_from_fake_folder(raw_label: str) -> tuple[int | None, str]:
    normalized = normalize_text(raw_label).lower()
    if normalized in {"0", "0.0", "fake", "false"}:
        return LABEL_FAKE, "fake_folder_label_0"
    if normalized in {"1", "1.0", "real", "true", "authentic"}:
        return LABEL_REAL, "fake_folder_label_1"
    if not normalized:
        return LABEL_FAKE, "fake_folder_no_label"
    return None, "unknown_fake_folder_label"


def row_to_record(
    *,
    row: dict[str, str],
    source_path: Path,
    folder_kind: str,
    row_number: int,
) -> dict[str, str] | None:
    headline = first_existing(row, HEADLINE_COLUMNS)
    content = first_existing(row, TEXT_COLUMNS)
    category = first_existing(row, CATEGORY_COLUMNS)
    raw_label = first_existing(row, LABEL_COLUMNS)

    if not content:
        positional_values = [value for key, value in sorted(row.items()) if key.startswith("column_")]
        if len(positional_values) == 1:
            content = positional_values[0]
        elif len(positional_values) >= 2:
            headline = headline or positional_values[0]
            content = positional_values[1]
            raw_label = raw_label or positional_values[-1]

    if not content and headline:
        content = headline
        headline = ""

    if not content:
        return None

    if folder_kind == "real":
        label_id, label_source = label_from_real_split(raw_label)
        if label_id is None:
            return None
    else:
        label_id, label_source = label_from_fake_folder(raw_label)
        if label_id is None:
            return None

    label = "real" if label_id == LABEL_REAL else "fake"
    key = stable_key(headline, content)
    return {
        "record_id": f"{source_path.stem}-{row_number}",
        "label": label,
        "label_id": str(label_id),
        "category": category,
        "headline": headline,
        "content": content,
        "source_file": str(source_path.relative_to(ROOT)),
        "source_folder": folder_kind,
        "raw_label": raw_label,
        "label_source": label_source,
        "dedupe_key": key,
    }


def load_folder(folder: Path, folder_kind: str) -> tuple[list[dict[str, str]], list[SourceStats]]:
    records: list[dict[str, str]] = []
    stats: list[SourceStats] = []
    for path in sorted(folder.glob("*.csv")):
        rows, _fields = read_csv_rows(path)
        source_stats = SourceStats(path=str(path.relative_to(ROOT)))
        for idx, row in enumerate(rows, start=1):
            source_stats.rows_read += 1
            raw_label = first_existing(row, LABEL_COLUMNS)
            if raw_label:
                source_stats.labels[raw_label] += 1
            record = row_to_record(
                row=row,
                source_path=path,
                folder_kind=folder_kind,
                row_number=idx,
            )
            if record is None:
                source_stats.rows_skipped_empty += 1
                continue
            source_stats.rows_kept += 1
            records.append(record)
        stats.append(source_stats)
    return records, stats


def deduplicate(rows: Iterable[dict[str, str]]) -> tuple[list[dict[str, str]], list[dict[str, str]], int]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["dedupe_key"]].append(row)

    unique_rows: list[dict[str, str]] = []
    conflicts: list[dict[str, str]] = []
    duplicate_count = 0
    for key, key_rows in grouped.items():
        labels = {row["label"] for row in key_rows}
        if len(labels) > 1:
            conflict_sources = "; ".join(
                f"{row['source_file']}:{row['raw_label']}->{row['label']}"
                for row in key_rows
            )
            conflict_row = dict(key_rows[0])
            conflict_row["label"] = "conflict"
            conflict_row["label_id"] = ""
            conflict_row["label_source"] = "conflicting_duplicate_labels"
            conflict_row["raw_label"] = conflict_sources
            conflicts.append(conflict_row)
            duplicate_count += len(key_rows) - 1
            continue
        unique_rows.append(key_rows[0])
        duplicate_count += len(key_rows) - 1
    return unique_rows, conflicts, duplicate_count


def balance_rows(rows: list[dict[str, str]], target_per_class: int | None, rng: random.Random) -> list[dict[str, str]]:
    by_label: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_label[row["label"]].append(row)

    available = {label: len(label_rows) for label, label_rows in by_label.items()}
    if set(available) != {"fake", "real"}:
        raise ValueError(f"Need both fake and real rows. Found: {available}")

    size = min(available.values())
    if target_per_class is not None:
        size = min(size, target_per_class)

    balanced: list[dict[str, str]] = []
    for label in ("fake", "real"):
        label_rows = list(by_label[label])
        rng.shuffle(label_rows)
        balanced.extend(label_rows[:size])

    rng.shuffle(balanced)
    return balanced


def split_rows(rows: list[dict[str, str]]) -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]]:
    rng = random.Random(RANDOM_SEED)
    by_label: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_label[row["label"]].append(row)

    train: list[dict[str, str]] = []
    valid: list[dict[str, str]] = []
    test: list[dict[str, str]] = []

    for label in ("fake", "real"):
        label_rows = list(by_label[label])
        rng.shuffle(label_rows)
        total = len(label_rows)
        train_end = int(total * 0.8)
        valid_end = train_end + int(total * 0.1)
        train.extend(label_rows[:train_end])
        valid.extend(label_rows[train_end:valid_end])
        test.extend(label_rows[valid_end:])

    rng.shuffle(train)
    rng.shuffle(valid)
    rng.shuffle(test)
    return train, valid, test


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = [
        "record_id",
        "label",
        "label_id",
        "category",
        "headline",
        "content",
        "source_file",
        "source_folder",
        "raw_label",
        "label_source",
        "dedupe_key",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_summary(
    *,
    all_rows: list[dict[str, str]],
    balanced_rows: list[dict[str, str]],
    train: list[dict[str, str]],
    valid: list[dict[str, str]],
    test: list[dict[str, str]],
    duplicate_count: int,
    conflict_count: int,
    source_stats: list[SourceStats],
    target_per_class: int | None,
) -> str:
    all_counts = Counter(row["label"] for row in all_rows)
    balanced_counts = Counter(row["label"] for row in balanced_rows)
    source_counts = Counter(row["source_file"] for row in balanced_rows)
    label_sources = Counter(row["label_source"] for row in balanced_rows)

    lines = [
        "# Phase 1 Balanced Dataset",
        "",
        f"- Label convention: `0 = fake`, `1 = real`",
        f"- Random seed: `{RANDOM_SEED}`",
        f"- Requested target per class: `{target_per_class if target_per_class is not None else 'max balanced'}`",
        f"- Unique rows before balancing: `{len(all_rows)}`",
        f"- Conflicting duplicate texts excluded: `{conflict_count}`",
        f"- Duplicate rows removed: `{duplicate_count}`",
        f"- Available label counts before balancing: `{dict(all_counts)}`",
        f"- Final balanced label counts: `{dict(balanced_counts)}`",
        f"- Final total rows: `{len(balanced_rows)}`",
        f"- Split sizes: train=`{len(train)}`, valid=`{len(valid)}`, test=`{len(test)}`",
        "",
        "## Label Rules",
        "",
        "- Files in `Dataset/Fake/` are also mixed: raw label `0`/`0.0` is treated as fake; raw label `1`/`1.0` is treated as real.",
        "- Files in `Dataset/Real/` are mixed already-split files: raw label `3` is treated as real; raw labels `0`, `1`, and `2` are treated as fake.",
        "- Empty content rows are skipped.",
        "- Exact duplicate headline+content pairs are removed before balancing.",
        "- If the exact same text appears with both fake and real labels, it is excluded and written to `conflicting_duplicates.csv`.",
        "",
        "## Source Files",
        "",
    ]

    for stat in source_stats:
        lines.extend(
            [
                f"### `{stat.path}`",
                f"- Rows read: `{stat.rows_read}`",
                f"- Rows kept: `{stat.rows_kept}`",
                f"- Rows skipped: `{stat.rows_skipped_empty}`",
                f"- Raw labels: `{dict(stat.labels or {})}`",
                "",
            ]
        )

    lines.extend(["## Final Source Contribution", ""])
    for source, count in source_counts.most_common():
        lines.append(f"- `{source}`: `{count}`")

    lines.extend(["", "## Final Label Sources", ""])
    for source, count in label_sources.most_common():
        lines.append(f"- `{source}`: `{count}`")

    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a balanced Bangla fake-news dataset from messy source CSVs.")
    parser.add_argument(
        "--target-per-class",
        type=int,
        default=None,
        help="Optional maximum rows per class. By default, uses the largest balanced size available.",
    )
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    real_folder_rows, real_stats = load_folder(REAL_DIR, "real")
    fake_folder_rows, fake_stats = load_folder(FAKE_DIR, "fake")
    unique_rows, conflicts, duplicate_count = deduplicate(real_folder_rows + fake_folder_rows)
    balanced_rows = balance_rows(unique_rows, args.target_per_class, rng)
    train, valid, test = split_rows(balanced_rows)

    write_csv(OUTPUT_DIR / "all_unbalanced_unique.csv", unique_rows)
    write_csv(OUTPUT_DIR / "conflicting_duplicates.csv", conflicts)
    write_csv(OUTPUT_DIR / "all.csv", balanced_rows)
    write_csv(OUTPUT_DIR / "train.csv", train)
    write_csv(OUTPUT_DIR / "valid.csv", valid)
    write_csv(OUTPUT_DIR / "test.csv", test)
    (OUTPUT_DIR / "README.md").write_text(
        build_summary(
            all_rows=unique_rows,
            balanced_rows=balanced_rows,
            train=train,
            valid=valid,
            test=test,
            duplicate_count=duplicate_count,
            conflict_count=len(conflicts),
            source_stats=real_stats + fake_stats,
            target_per_class=args.target_per_class,
        ),
        encoding="utf-8",
    )

    print(f"Wrote dataset files to: {OUTPUT_DIR}")
    print(f"Unique rows before balancing: {len(unique_rows)}")
    print(f"Final balanced counts: {dict(Counter(row['label'] for row in balanced_rows))}")
    print(f"Train/valid/test: {len(train)}/{len(valid)}/{len(test)}")


if __name__ == "__main__":
    main()
