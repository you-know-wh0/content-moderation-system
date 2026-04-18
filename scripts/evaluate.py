from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from time import perf_counter

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.main import ModerationService


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the moderation service against the training CSV."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("app/train.csv"),
        help="CSV file with a 'comment_text' column and binary 'toxic' label.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=200,
        help="Maximum number of rows to evaluate.",
    )
    return parser.parse_args()


def safe_divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def compute_binary_metrics(tp: int, fp: int, tn: int, fn: int) -> dict[str, float]:
    precision = safe_divide(tp, tp + fp)
    recall = safe_divide(tp, tp + fn)
    f1 = safe_divide(2 * precision * recall, precision + recall)
    accuracy = safe_divide(tp + tn, tp + tn + fp + fn)
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "accuracy": round(accuracy, 4),
    }


def main() -> None:
    args = parse_args()
    service = ModerationService.from_artifacts()

    tp = fp = tn = fn = review_count = 0
    latencies_ms: list[float] = []

    with args.data.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for index, row in enumerate(reader):
            if index >= args.limit:
                break

            text = (row.get("comment_text") or "").strip()
            if not text:
                continue

            actual = int(row["toxic"])
            start = perf_counter()
            result = service.moderate(text)
            latencies_ms.append((perf_counter() - start) * 1000)

            if result.status == "HUMAN_REVIEW":
                review_count += 1

            predicted = 1 if result.status == "REJECTED" else 0

            if predicted == 1 and actual == 1:
                tp += 1
            elif predicted == 1 and actual == 0:
                fp += 1
            elif predicted == 0 and actual == 0:
                tn += 1
            else:
                fn += 1

    metrics = compute_binary_metrics(tp, fp, tn, fn)
    sample_count = tp + fp + tn + fn
    sorted_latencies = sorted(latencies_ms)

    def percentile(percent: float) -> float:
        if not sorted_latencies:
            return 0.0
        index = min(len(sorted_latencies) - 1, round((percent / 100) * (len(sorted_latencies) - 1)))
        return round(sorted_latencies[index], 3)

    summary = {
        "samples_evaluated": sample_count,
        "human_review_rate": round(safe_divide(review_count, sample_count), 4),
        "latency_ms": {
            "p50": percentile(50),
            "p95": percentile(95),
            "max": round(max(sorted_latencies), 3) if sorted_latencies else 0.0,
        },
        "confusion_matrix": {
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
        },
        "metrics": metrics,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
