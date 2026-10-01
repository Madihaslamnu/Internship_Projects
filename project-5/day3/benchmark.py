
from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from pathlib import Path

import joblib
import numpy as np

from day3.data_utili import get_split
from day3.train import batch_preprocess

ART = Path(__file__).parent / "artifacts"


def _dir_size_mb(path: Path) -> float:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file() and "checkpoints" not in f.parts) / 1e6


def time_calls(fn, items, warmup: int = 20) -> list[float]:
    """Per-call wall time in ms, after a warmup (first calls pay lazy-init costs)."""
    for x in items[:warmup]:
        fn(x)
    out = []
    for x in items:
        t = time.perf_counter()
        fn(x)
        out.append((time.perf_counter() - t) * 1000)
    return out


def summarise(ms: list[float]) -> dict:
    a = np.array(ms)
    return {"mean_ms": round(a.mean(), 2), "p50_ms": round(np.percentile(a, 50), 2),
            "p95_ms": round(np.percentile(a, 95), 2), "p99_ms": round(np.percentile(a, 99), 2)}


def bench_baseline(texts: list[str], batch: int) -> dict:
    path = ART / "baseline.joblib"
    model = joblib.load(path)
    single = summarise(time_calls(lambda t: model.predict([t]), texts))
    chunks = [texts[i:i + batch] for i in range(0, len(texts) - batch + 1, batch)]
    t = time.perf_counter()
    for c in chunks:
        model.predict(c)
    thr = len(chunks) * batch / (time.perf_counter() - t)
    return {"device": "cpu", "single": single, f"throughput_tweets_per_s_bs{batch}": round(thr, 1),
            "size_mb": round(path.stat().st_size / 1e6, 2), "params_millions": None}


def bench_transformer(texts: list[str], batch: int, device: str, model_dir: Path, max_len: int) -> dict:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    from day3.distilbert import clean_for_transformer

    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device).eval()

    def predict(batch_texts: list[str]):
        enc = tok([clean_for_transformer(t) for t in batch_texts], truncation=True, max_length=max_len,
                  padding=True, return_tensors="pt").to(device)
        with torch.inference_mode():
            logits = model(**enc).logits
        if device == "cuda":
            torch.cuda.synchronize()   # otherwise we'd time only the kernel *launch*
        return logits.argmax(-1)

    single = summarise(time_calls(lambda t: predict([t]), texts))
    chunks = [texts[i:i + batch] for i in range(0, len(texts) - batch + 1, batch)]
    predict(chunks[0])  # warmup
    t = time.perf_counter()
    for c in chunks:
        predict(c)
    thr = len(chunks) * batch / (time.perf_counter() - t)
    return {"device": device, "single": single, f"throughput_tweets_per_s_bs{batch}": round(thr, 1),
            "size_mb": round(_dir_size_mb(model_dir), 1),
            "params_millions": round(sum(p.numel() for p in model.parameters()) / 1e6, 1)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["baseline", "distilbert"], default=None)
    ap.add_argument("--n", type=int, default=300, help="number of test tweets to time")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--threads", type=int, default=None, help="torch CPU threads (default: torch's choice)")
    ap.add_argument("--model_dir", default=str(ART / "distilbert"))
    ap.add_argument("--max_len", type=int, default=64)
    ap.add_argument("--out", default=str(ART / "latency.json"))
    args = ap.parse_args()

    _, test = get_split()
    texts = test["text"].sample(args.n, random_state=0).tolist()

    hw = {"machine": platform.machine(), "processor": platform.processor() or platform.platform(),
          "python": platform.python_version()}
    import os
    hw["cpu_count"] = os.cpu_count()
    results: dict = {"hardware": hw, "n_timed": args.n, "batch": args.batch}

    if args.only != "distilbert":
        results["tfidf_baseline"] = bench_baseline(texts, args.batch)
        print("baseline  ", json.dumps(results["tfidf_baseline"]))

    if args.only != "baseline":
        model_dir = Path(args.model_dir)
        if not (model_dir / "config.json").exists():
            print(f"[skip] no fine-tuned model at {model_dir} -- run finetune_distilbert.py first")
        else:
            import torch
            if args.threads:
                torch.set_num_threads(args.threads)
            hw["torch"] = torch.__version__
            hw["torch_threads"] = torch.get_num_threads()
            hw["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
            results["distilbert_cpu"] = bench_transformer(texts, args.batch, "cpu", model_dir, args.max_len)
            print("distil cpu", json.dumps(results["distilbert_cpu"]))
            if torch.cuda.is_available():
                results["distilbert_gpu"] = bench_transformer(texts, args.batch, "cuda", model_dir, args.max_len)
                print("distil gpu", json.dumps(results["distilbert_gpu"]))

    Path(args.out).write_text(json.dumps(results, indent=2))
    print("saved ->", args.out)


if __name__ == "__main__":
    main()