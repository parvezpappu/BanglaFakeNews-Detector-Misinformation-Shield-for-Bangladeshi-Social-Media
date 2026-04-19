FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

WORKDIR /app

COPY app/backend/requirements.txt /app/app/backend/requirements.txt
RUN pip install --no-cache-dir -r /app/app/backend/requirements.txt

COPY app /app/app
COPY Improved /app/Improved

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "app.backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
