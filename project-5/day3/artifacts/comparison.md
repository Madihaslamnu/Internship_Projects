## Quality (same held-out test tweets, n=2886)

| model           |   accuracy |   macro_f1 |   f1_negative |   f1_neutral |   f1_positive |
|:----------------|-----------:|-----------:|--------------:|-------------:|--------------:|
| TF-IDF baseline |     0.8222 |     0.7673 |        0.8895 |       0.6649 |        0.7475 |
| DistilBERT      |     0.8375 |     0.7868 |        0.9045 |       0.6756 |        0.7802 |

Δ macro-F1 = +0.0194  (95% CI +0.0023 .. +0.0368)
Δ accuracy = +0.0152  (95% CI +0.0024 .. +0.0281)
McNemar exact: baseline-only-correct=162, DistilBERT-only-correct=206, p=0.02486

**DistilBERT beats the baseline (95% CI for the macro-F1 gain excludes 0).**

## Inference cost (end-to-end from raw text; hardware: AMD64 Family 25 Model 80 Stepping 0, AuthenticAMD, 8 cores)

| model                 |   mean_ms |   p50_ms |   p95_ms |   p99_ms |   throughput/s (bs32) |   size_MB |   params_M |
|:----------------------|----------:|---------:|---------:|---------:|----------------------:|----------:|-----------:|
| TF-IDF baseline (CPU) |      9.03 |     8.47 |    13.61 |    20.84 |                 206.4 |      1.13 |        nan |
| DistilBERT (CPU)      |     39.39 |    38.81 |    52.85 |    71.01 |                  33   |    268.6  |         67 |

On CPU, DistilBERT's median single-request latency is **5x** the baseline's (38.8 ms vs 8.5 ms) for a macro-F1 change of +1.9 points.
