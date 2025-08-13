FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y build-essential poppler-utils libmagic1 git && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENV PYTHONUNBUFFERED=1