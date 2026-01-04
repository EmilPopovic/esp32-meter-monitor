FROM python:3.11-slim

# Install Tesseract OCR
RUN apt-get update && \
    apt-get install -y tesseract-ocr && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir \
    paho-mqtt==1.6.1 \
    pytesseract==0.3.10 \
    Pillow==10.1.0

# Copy service script
COPY meter_ocr_service.py /app/meter_ocr_service.py
WORKDIR /app

# Run service
CMD ["python", "-u", "meter_ocr_service.py"]
