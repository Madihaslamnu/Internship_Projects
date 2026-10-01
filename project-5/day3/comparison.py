

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import accuracy_score, f1_score

ART = Path(__file__).parent / "artifacts"


def paired_bootstrap(y, p_a, p_b, n_boot: int = 2000, seed: int = 0) -> dict:
    """B minus A, resampling tweets jointly. y, p_a, p_b: aligned numpy arrays."""
    rng = np.random.default_rng(seed)
    n = len(y)
    d_f1, d_acc = [], []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        d_f1.append(f1_score(y[i], p_b[i], average="macro") - f1_score(y[i], p_a[i], average="macro"))
        d_acc.append(accuracy_score(y[i], p_b[i]) - accuracy_score(y[i], p_a[i]))
    q = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    return {
        "delta_macro_f1": float(f1_score(y, p_b, average="macro") - f1_score(y, p_a, average="macro")),
        "delta_macro_f1_ci95": q(d_f1),
        "delta_accuracy": float(accuracy_score(y, p_b) - accuracy_score(y, p_a)),
        "delta_accuracy_ci95": q(d_acc),
    }


def mcnemar_exact(y, p_a, p_b) -> dict:
    a_ok, b_ok = p_a == y, p_b == y
    only_a, only_b = int((a_ok & ~b_ok).sum()), int((~a_ok & b_ok).sum())
    n = only_a + only_b
    p = binomtest(only_b, n, 0.5).pvalue if n else 1.0
    return {"only_baseline_correct": only_a, "only_distilbert_correct": only_b, "p_value": float(p)}


def load_aligned():
    a = pd.read_csv(ART / "test_predictions.csv", index_col="row_id")
    b = pd.read_csv(ART / "distilbert_test_predictions.csv", index_col="row_id")
    common = a.index.intersection(b.index)
    if len(common) != len(a) or len(common) != len(b):
        print(f"[warn] prediction files cover different tweets ({len(a)} vs {len(b)}); using {len(common)} shared")
    a, b = a.loc[common], b.loc[common]
    assert (a["true"] == b["true"]).all(), "label mismatch -> the two runs did not use the same split"
    return a["true"].values, a["pred"].values, b["pred"].values


def main() -> None:
    y, p_base, p_bert = load_aligned()
    stats = paired_bootstrap(y, p_base, p_bert)
    mc = mcnemar_exact(y, p_base, p_bert)

    def row(name, p):
        return {"model": name, "accuracy": accuracy_score(y, p), "macro_f1": f1_score(y, p, average="macro"),
                **{f"f1_{c}": v for c, v in zip(["negative", "neutral", "positive"],
                                                f1_score(y, p, average=None, labels=["negative", "neutral", "positive"]))}}

    table = pd.DataFrame([row("TF-IDF baseline", p_base), row("DistilBERT", p_bert)]).set_index("model").round(4)
    lo, hi = stats["delta_macro_f1_ci95"]
    verdict = ("DistilBERT beats the baseline (95% CI for the macro-F1 gain excludes 0)." if lo > 0 else
               "DistilBERT is worse than the baseline (CI entirely below 0)." if hi < 0 else
               "Inconclusive: the macro-F1 difference is within noise (CI includes 0).")

    lines = ["## Quality (same held-out test tweets, n=%d)" % len(y), "", table.to_markdown(), "",
             f"Δ macro-F1 = {stats['delta_macro_f1']:+.4f}  (95% CI {lo:+.4f} .. {hi:+.4f})",
             f"Δ accuracy = {stats['delta_accuracy']:+.4f}  (95% CI {stats['delta_accuracy_ci95'][0]:+.4f} .. {stats['delta_accuracy_ci95'][1]:+.4f})",
             f"McNemar exact: baseline-only-correct={mc['only_baseline_correct']}, "
             f"DistilBERT-only-correct={mc['only_distilbert_correct']}, p={mc['p_value']:.4g}", "",
             f"**{verdict}**", ""]

    lat_path = ART / "latency.json"
    if lat_path.exists():
        lat = json.loads(lat_path.read_text())
        rows = []
        for key, label in [("tfidf_baseline", "TF-IDF baseline (CPU)"), ("distilbert_cpu", "DistilBERT (CPU)"),
                           ("distilbert_gpu", "DistilBERT (GPU)")]:
            if key in lat:
                r = lat[key]
                thr = next(v for k, v in r.items() if k.startswith("throughput"))
                rows.append({"model": label, **r["single"], "throughput/s (bs32)": thr, "size_MB": r["size_mb"],
                             "params_M": r["params_millions"]})
        lt = pd.DataFrame(rows).set_index("model")
        lines += ["## Inference cost (end-to-end from raw text; hardware: %s, %s cores)" %
                  (lat["hardware"].get("processor", "?"), lat["hardware"].get("cpu_count", "?")), "", lt.to_markdown(), ""]
        if "distilbert_cpu" in lat and "tfidf_baseline" in lat:
            b, d = lat["tfidf_baseline"]["single"]["p50_ms"], lat["distilbert_cpu"]["single"]["p50_ms"]
            gain_pts = 100 * stats["delta_macro_f1"]
            lines += [f"On CPU, DistilBERT's median single-request latency is **{d / b:.0f}x** the baseline's "
                      f"({d:.1f} ms vs {b:.1f} ms) for a macro-F1 change of {gain_pts:+.1f} points.", ""]
    else:
        lines += ["(no latency.json yet -- run benchmark_latency.py)", ""]

    md = "\n".join(lines)
    print(md)
    (ART / "comparison.md").write_text(md, encoding="utf-8")
    (ART / "comparison.json").write_text(json.dumps({"stats": stats, "mcnemar": mc, "verdict": verdict}, indent=2))


if __name__ == "__main__":
    main()