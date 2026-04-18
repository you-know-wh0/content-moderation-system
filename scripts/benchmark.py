from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
import sys
from time import perf_counter

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.main import ModerationService


SAMPLE_TEXTS = [
    "I love this project and the documentation is clear.",
    "Your work is terrible and you should feel bad.",
    "I am not sure whether this is acceptable.",
    "I will beat you up if you do that again.",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure inference latency.")
    parser.add_argument("--iterations", type=int, default=25)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    service = ModerationService.from_artifacts()

    for text in SAMPLE_TEXTS:
        service.moderate(text)

    latencies: list[float] = []
    for index in range(args.iterations):
        text = SAMPLE_TEXTS[index % len(SAMPLE_TEXTS)]
        start = perf_counter()
        service.moderate(text)
        latencies.append((perf_counter() - start) * 1000)

    sorted_latencies = sorted(latencies)

    def percentile(percent: float) -> float:
        if not sorted_latencies:
            return 0.0
        index = min(
            len(sorted_latencies) - 1,
            round((percent / 100) * (len(sorted_latencies) - 1)),
        )
        return round(sorted_latencies[index], 3)

    summary = {
        "iterations": args.iterations,
        "latency_ms": {
            "mean": round(statistics.mean(latencies), 3),
            "median": round(statistics.median(latencies), 3),
            "p95": percentile(95),
            "max": round(max(latencies), 3),
        },
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
