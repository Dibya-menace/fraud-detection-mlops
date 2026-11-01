FROM python:3.11-slim

# LightGBM needs the OpenMP runtime
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
ENV PYTHONUNBUFFERED=1 PYTHONPATH=/app/src

COPY requirements-runtime.txt .
RUN pip install --no-cache-dir -r requirements-runtime.txt

COPY src/ ./src/
COPY models/ ./models/

# Hugging Face Spaces runs as a non-root user and expects port 7860
RUN useradd -m appuser
USER appuser

EXPOSE 7860
CMD ["uvicorn", "fraud.api:app", "--host", "0.0.0.0", "--port", "7860"]