FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app
ENV PORT=7860

# minimum system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ffmpeg\
    && rm -rf /var/lib/apt/lists/*

# installing python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copyng the project
COPY . .

EXPOSE 7860

# Launching the app
CMD ["python", "-m", "src.app.web_app"]