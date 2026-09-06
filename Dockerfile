FROM python:3.11-slim

ARG NER_MODEL_REPO=AsphodelRem/uz-ner-rembert
ARG NER_MODEL_REVISION=f11db5a111e8f31f5907786798b01e0eff33f66a

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_DATASETS_OFFLINE=1 \
    TOKENIZERS_PARALLELISM=false

WORKDIR /app

COPY requirements.txt requirements-api.txt ./
RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir -r requirements-api.txt

RUN addgroup --system ner && adduser --system --ingroup ner --home /app ner

RUN HF_HUB_OFFLINE=0 python -c "\
from huggingface_hub import snapshot_download; \
snapshot_download('${NER_MODEL_REPO}', revision='${NER_MODEL_REVISION}', local_dir='/app/artifacts')" \
    && rm -rf /root/.cache/huggingface \
    && test -f /app/artifacts/model/config.json \
    && test -f /app/artifacts/run_config.json \
    && test -f /app/artifacts/transitions.json \
    && chown -R ner:ner /app/artifacts

COPY --chown=ner:ner . .

USER ner

EXPOSE 8000

HEALTHCHECK --interval=10s --timeout=3s --start-period=30s --retries=5 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=2).read()"]

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--no-access-log"]
