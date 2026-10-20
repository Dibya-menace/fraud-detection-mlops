FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 appuser
WORKDIR /app

COPY requirements-runtime.txt .
RUN pip install --no-cache-dir -r requirements-runtime.txt

COPY src/ src/
COPY models/ models/

ENV PYTHONPATH=/app/src
USER appuser
EXPOSE 7860
CMD ["uvicorn", "fraud.api:app", "--host", "0.0.0.0", "--port", "7860"]