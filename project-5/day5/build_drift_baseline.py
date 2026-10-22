"""Create reference text-feature statistics from the original training split."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))
from day3.data_utili import load_airline  # noqa: E402
from fast_api import text_features  # noqa: E402


def main() -> None:
    data = load_airline()
    train, _ = train_test_split(
        data,
        test_size=0.2,
        stratify=data["airline_sentiment"],
        random_state=42,
    )
    feature_frame = pd.DataFrame([text_features(text) for text in train["text"]])
    baseline = {
        name: {"mean": float(feature_frame[name].mean()), "std": float(feature_frame[name].std(ddof=0))}
        for name in ("char_count", "word_count", "punctuation_ratio")
    }
    output = Path(__file__).with_name("drift_baseline.json")
    output.write_text(json.dumps(baseline, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {output} from {len(train)} training reviews")


if __name__ == "__main__":
    main()
