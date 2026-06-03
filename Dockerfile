FROM python:3.11-slim

# Install Tesseract OCR + English language pack
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-eng \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Render sets PORT env variable; default to 5000 locally
ENV PORT=5000

EXPOSE $PORT

CMD ["sh", "-c", "gunicorn app:app --bind 0.0.0.0:$PORT"]
