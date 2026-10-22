FROM python:3.12-slim AS builder

ENV VIRTUAL_ENV=/opt/venv
RUN python -m venv "$VIRTUAL_ENV"
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
WORKDIR /build
COPY day5/requirements-deploy.txt ./requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

FROM python:3.12-slim AS runtime
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000
WORKDIR /app
COPY --from=builder /opt/venv /opt/venv
COPY text_preprocessing.py ./text_preprocessing.py
COPY day3/artifacts/baseline.joblib ./models/baseline.joblib
COPY day5/fast_api.py ./fast_api.py
COPY day5/drift_baseline.json ./drift_baseline.json
ENV MODEL_PATH=/app/models/baseline.joblib \
    DRIFT_BASELINE_PATH=/app/drift_baseline.json
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2)"
CMD ["sh", "-c", "uvicorn fast_api:app --host 0.0.0.0 --port ${PORT}"]
