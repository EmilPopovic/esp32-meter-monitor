FROM python:3.11-slim

# System libs required by OpenCV (used internally by EasyOCR)
RUN apt-get update && \
    apt-get install -y libgl1 libglib2.0-0 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir \
    paho-mqtt==1.6.1 \
    Pillow==10.1.0 \
    easyocr

# Pre-download EasyOCR English models at build time so the container
# starts instantly and doesn't need internet access at runtime.
RUN python -c "import easyocr; easyocr.Reader(['en'], gpu=False, verbose=False)"

# Copy service script
COPY meter_ocr_service.py /app/meter_ocr_service.py
WORKDIR /app

# Run service
CMD ["python", "-u", "meter_ocr_service.py"]
