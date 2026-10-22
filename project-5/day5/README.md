# Project 5: Airline Sentiment API

FastAPI service that classifies airline review text as **negative**, **neutral**, or **positive**. It packages the Project 5 TF-IDF + Linear SVM model, validates incoming text, writes structured prediction logs, and reports simple input drift against the model's training data.

## Results and deployment model

The held-out set has 2,886 reviews. The compact baseline achieved **82.22% accuracy** and **0.7673 macro-F1**. DistilBERT achieved **83.75% accuracy** and **0.7868 macro-F1**, but its weights are about 268 MB and it requires PyTorch. This service uses the 1.13 MB TF-IDF + SVM pipeline to keep CPU deployment small. These are separate models and their results are not conflated.

| Model | Accuracy | Macro-F1 | Model size | Decision |
|---|---:|---:|---:|---|
| TF-IDF + Linear SVM | 82.22% | 0.7673 | 1.13 MB | Served by this API |
| Fine-tuned DistilBERT | 83.75% | 0.7868 | 268.6 MB | Higher score; larger runtime |

The previous repository load test recorded 8 ms median and 26 ms p95, but it used a serving implementation that skipped training-time spaCy preprocessing. Those figures are **not a valid benchmark for this corrected API**; rerun load testing against the deployed image before making latency claims.

## Data and approach

The project uses the airline review `Tweets.csv` dataset and predicts its three sentiment labels. Reviews are de-duplicated and split into 11,541 training and 2,886 test examples using a stratified 80/20 split. The baseline pipeline applies the same spaCy lemmatization preprocessing used during training, then TF-IDF features and a Linear SVM classifier.

## Architecture

```text
Client -> FastAPI /predict -> Pydantic text validation
                           -> spaCy preprocessing -> TF-IDF + Linear SVM
                           -> sentiment response
                           -> JSON prediction log (stdout)
                           -> rolling feature window -> /drift report
```

`GET /health` is a readiness check. `POST /predict` returns the submitted text and predicted label. `GET /drift` compares recent requests with training-reference means for character count, word count, and punctuation ratio. It starts reporting after 30 requests and retains up to 500 inputs in process memory. A feature is flagged when its mean moves at least 0.5 training standard deviations.

Prediction logs include a request ID, text SHA-256 digest, predicted label, latency, and those three numeric features. Raw review text is not logged. The drift window clears when the process restarts; this is a lightweight demo monitor, not persistent multi-instance monitoring.

## Run locally

From the `project-5` directory, install dependencies and run the API:

```powershell
python -m pip install -r day5/requirements-dev.txt
python day5/build_drift_baseline.py
uvicorn day5.fast_api:app --host 127.0.0.1 --port 8000
```

Try the interactive docs at `http://localhost:8000/docs`, or send a prediction:

```powershell
Invoke-RestMethod -Method Post -Uri http://localhost:8000/predict `
  -ContentType 'application/json' -Body '{"text":"The crew was wonderful"}'
```

Run the tests with `python -m pytest -q day5/tests`.

## Docker

Build from the `project-5` directory:

```powershell
docker build -f day5/Dockerfile -t airline-sentiment-api .
docker run --rm -p 8000:8000 airline-sentiment-api
```

The multi-stage Dockerfile installs runtime dependencies in a builder stage and uses a slim Python runtime. Its health check calls `/health`. The service uses the host-provided `PORT` and does not require application secrets.

## CI and deployment

GitHub Actions runs Ruff, nine API/model tests, and a Docker build on pushes and pull requests. The model artifact uses Git LFS; CI checks out LFS content before testing and building.

The service is deployed on Render at **[https://internship-projects-uyfv.onrender.com](https://internship-projects-uyfv.onrender.com)**. The live `/health` endpoint returns HTTP 200, and a live `/predict` request returned `positive` for “The crew was kind and helpful.” Open `/docs` to explore the API. A free Render instance may sleep when idle, so its first request after inactivity can take longer.

Deployment settings: Docker runtime, root directory `project-5`, Dockerfile path `day5/Dockerfile`, Docker context `.`, and health check path `/health`. The service does not require application secrets.

## Limitations and next steps

- The service predicts the label distribution represented by its training data; it does not explain individual predictions.
- The in-memory drift report resets at restart and is local to one service process.
- Re-run the load test on the corrected Docker API before citing latency.
- Add the final live service URL and deployment date here after deployment.
