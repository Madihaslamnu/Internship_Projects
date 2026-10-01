
from __future__ import annotations

import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from day3.data_utili import get_split
from day4.llm_classification import classify_zero_shot

ART = Path(__file__).parent / "artifacts"
ART.mkdir(parents=True, exist_ok=True)


def main() -> None:
  _, test = get_split()
  
  eval_df = test.sample(n=100, random_state=42).reset_index(drop=True)

  preds = []
  latencies = []

  print(
      f"Running zero-shot LLM evaluation on {len(eval_df)} test samples..."
  )

  for i, row in eval_df.iterrows():
    start = time.time()
    try:
      res = classify_zero_shot(row["text"])
      pred = res.get("sentiment", "neutral").lower()
    except Exception as e:
      print(f"Error on row {i}: {e}")
      pred = "neutral"
    latencies.append((time.time() - start) * 1000)
    preds.append(pred)

  y_true = eval_df["airline_sentiment"].values
  acc = accuracy_score(y_true, preds)
  macro_f1 = f1_score(y_true, preds, average="macro")

  input_tokens_per_req = 30
  output_tokens_per_req = 15
  cost_per_m_input = 0.59  
  cost_per_m_output = 0.79 

  cost_per_1k = (
      (input_tokens_per_req * 1000 / 1_000_000) * cost_per_m_input
      + (output_tokens_per_req * 1000 / 1_000_000) * cost_per_m_output
  )

  metrics = {
      "model": "LLM API (Llama-3.3-70B-Versatile via Groq)",
      "accuracy": round(acc, 4),
      "macro_f1": round(macro_f1, 4),
      "mean_latency_ms": round(np.mean(latencies), 2),
      "p50_latency_ms": round(np.percentile(latencies, 50), 2),
      "estimated_cost_per_1000_predictions_usd": round(cost_per_1k, 4),
  }

  print(json.dumps(metrics, indent=2))
  (ART / "llm_evaluation_metrics.json").write_text(
      json.dumps(metrics, indent=2), encoding="utf-8"
  )


if __name__ == "__main__":
  main()