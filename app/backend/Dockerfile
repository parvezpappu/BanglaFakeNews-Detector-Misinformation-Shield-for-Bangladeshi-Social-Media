FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

WORKDIR /app

COPY app/backend/requirements.txt /app/app/backend/requirements.txt
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.5.1+cpu
RUN pip install --no-cache-dir -r /app/app/backend/requirements.txt

COPY app /app/app
COPY artifacts /app/artifacts

EXPOSE 8000

CMD ["sh", "-c", "python -m uvicorn app.backend.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
