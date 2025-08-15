FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    tesseract-ocr \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt dev-requirements.txt ./

RUN pip install --no-cache-dir -r requirements.txt \
 && pip install --no-cache-dir -r dev-requirements.txt

COPY . .

EXPOSE 8083

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8083"]
