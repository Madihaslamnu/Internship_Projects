# Production Architecture Recommendation: Airline Sentiment Classification

## Executive Summary
This report analyzes the performance, latency, economic trade-offs, and scaling profile of three architectures evaluated across the NLP development pipeline:
1. **Classical Baseline**: TF-IDF + Linear SVM 
2. **Fine-Tuned Local Transformer**: DistilBERT-base-uncased 
3. **Cloud LLM API**: OpenAI/GPT-OSS-20B via Groq 

---

## Performance & Economic Comparison Matrix

| Architecture | Accuracy | Macro-F1 | Median Latency (CPU/API) | Cost per 1,000 Predictions | Infra Overhead |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **TF-IDF + Linear SVM** | 82.22% | 0.7673 | **8.47 ms** | **$0.00** (Free local CPU) | Minimal (Joblib pickling) |
| **DistilBERT (Fine-Tuned)** | 83.75% | 0.7868 | **38.81 ms** | **$0.00** (Free local CPU/GPU) | Medium (PyTorch / Safetensors) |
| **LLM API (GPT-OSS-20B)** | **84.00%** | **0.7888** | 2,650.13 ms | **$0.00** | Low (Stateless REST API) |

---

## Production Deployment Guidelines: When to Use What

### 1. Choose TF-IDF + Linear SVM when:
* **Edge / Real-Time constraints are absolute**: Sub-10ms response targets are mandatory with strict CPU-only constraints.
* **Low resource environments**: You want an ultra-lightweight binary model (~1.1 MB) that can be embedded anywhere without deep learning dependencies.

### 2. Choose Fine-Tuned DistilBERT when:
* **Balanced low-latency and high accuracy is required**: DistilBERT hits a sweet spot with a ~38ms response time and top-tier local inference performance without recurring token costs.
* **Data privacy is strict**: Customer data cannot leave your local virtual private cloud (VPC).

### 3. Choose Cloud LLM APIs (Groq / GPT-OSS-20B) when:
* **Cold-start / Zero training data scenarios**: You need peak zero-shot accuracy (~0.7888 macro-F1) immediately without labeling data or managing model weights.
* **Dynamic taxonomy changes**: Business requirements shift frequently and require adjusting categories via system prompts instead of model retraining.

---
*Report automatically compiled and verified for Day 4 Deliverable.*
