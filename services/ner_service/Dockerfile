# ner-service (CPU). Build from the repo root:
#   docker build -f services/ner_service/Dockerfile -t ner-service .
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1

# Only what this service imports; deliberately NOT the repo's full dependency set.
RUN pip install "fastapi>=0.115" "uvicorn[standard]>=0.30" "pydantic>=2" \
        "PyJWT>=2.8" "cryptography>=42" "spacy>=3.7,<4" "prometheus-client>=0.20" \
    && python -m spacy download en_core_web_sm

RUN useradd --system --uid 10001 --no-create-home ner
WORKDIR /app
COPY libs/__init__.py libs/__init__.py
# metrics.py: app.py's create_app installs /metrics (without it the service dies on import).
COPY libs/shared/__init__.py libs/shared/app.py libs/shared/auth.py libs/shared/metrics.py libs/shared/
COPY services/__init__.py services/__init__.py
COPY services/ner_service services/ner_service
USER 10001

EXPOSE 8080
# Required at runtime: CLERK_JWKS_URL, CLERK_ISSUER (optional: CLERK_JWT_INTERNAL_ORG_CLAIM,
# NER_MAX_CHARS, NER_SPACY_MODEL).
HEALTHCHECK --interval=15s --timeout=3s CMD python -c "import urllib.request as u; u.urlopen('http://127.0.0.1:8080/healthz')"
CMD ["uvicorn", "services.ner_service.main:app", "--host", "0.0.0.0", "--port", "8080"]
