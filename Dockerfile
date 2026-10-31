FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN addgroup --system app && adduser --system --ingroup app app

COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

COPY . .

RUN mkdir -p /app/data/chroma_db /app/data/cache /app/data/logs /app/data/kb /app/data/kb_staging \
    && chown -R app:app /app

USER app

EXPOSE 8122

HEALTHCHECK --interval=30s --timeout=5s --start-period=45s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8122/', timeout=3).read()"

# Keep one process so in-memory sessions and runtime Control Center changes stay consistent.
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8122", "--workers", "1"]
