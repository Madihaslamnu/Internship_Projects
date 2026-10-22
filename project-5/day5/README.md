# Project 5 sentiment API deployment

This service wraps the Project 5 airline sentiment classifier in FastAPI. The deployed model is the compact TF-IDF + Linear SVM pipeline: held-out performance was **82.22% accuracy / 0.7673 macro-F1**. DistilBERT scored higher (83.75% / 0.7868 macro-F1), but its weights are about 268 MB and require PyTorch. This deployment uses the compact model for a practical CPU container; the two results are documented separately rather than presented as the same model.

The repository also contains an older load-test result of 8 ms median / 26 ms p95. That run used an API implementation which skipped the model's training-time spaCy preprocessing, so it is not a valid benchmark for this corrected inference path. Rerun the load test against the Docker image before publishing a production latency claim.

## Run locally

From the `project-5` directory:

```powershell
python -m pip install -r day5/requirements-dev.txt
python day5/build_drift_baseline.py
uvicorn day5.fast_api:app --host 127.0.0.1 --port 8000
```

The API loads `day3/artifacts/baseline.joblib` by default. `POST /predict` accepts `{"text":"The crew was wonderful"}`. Interactive docs are at `/docs`; `/health` is suitable for readiness checks.

## Docker

Build from the `project-5` directory:

```powershell
docker build -f day5/Dockerfile -t airline-sentiment-api .
docker run --rm -p 8000:8000 airline-sentiment-api
```

Then send a request:

```powershell
Invoke-RestMethod -Method Post -Uri http://localhost:8000/predict `
  -ContentType 'application/json' -Body '{"text":"The crew was wonderful"}'
```

The image uses a build stage for dependencies and a slim Python runtime. Docker health checks call `/health`.

## Logging and input drift

Each prediction emits one JSON log record with request ID, a SHA-256 text digest, prediction, latency, and input features (character count, word count, punctuation ratio). Raw review text is not written to logs. `GET /drift` compares the rolling last 500 requests with the deterministic training split and flags a feature when its mean shifts by at least 0.5 training standard deviations. It reports `collecting` before 30 live requests. The rolling window is in process memory and resets when the service restarts; use centralized logs or a metrics store for persistent, multi-instance monitoring.

Rebuild the committed reference statistics if the training split changes:

```powershell
python day5/build_drift_baseline.py
```

## CI and deployment

GitHub Actions runs Ruff, the prediction/input tests, and a Docker image build on pushes and pull requests. To publish, push this repository to GitHub and configure Render to use the Dockerfile at `project-5/day5/Dockerfile` with `project-5` as the build context. The service reads the platform-provided `PORT`; there are no application secrets. After deployment, check `/health`, call `/predict`, then call `/drift` after 30 or more requests. A custom domain is optional and can be attached in the hosting provider's dashboard.

## Architecture

```text
Client -> FastAPI validation -> TF-IDF + Linear SVM -> sentiment response
                    |                  |
                    +-> structured JSON logs
                    +-> rolling text-feature drift report
```
