#!/usr/bin/env python3
import paho.mqtt.client as mqtt
import easyocr
import numpy as np
from PIL import Image, ImageOps, ImageDraw
import io
import re
import json
import time
import os
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler

# Configuration from environment variables
MQTT_BROKER = os.getenv('MQTT_BROKER', 'localhost')
MQTT_PORT = int(os.getenv('MQTT_PORT', 1883))
MQTT_USER = os.getenv('MQTT_USER', None)
MQTT_PASS = os.getenv('MQTT_PASS', None)

IMAGE_TOPIC = os.getenv('IMAGE_TOPIC', 'home/meter/electric/image')
STATE_TOPIC = os.getenv('STATE_TOPIC', 'homeassistant/sensor/electric_meter/state')
CONFIG_TOPIC = os.getenv('CONFIG_TOPIC', 'homeassistant/sensor/electric_meter/config')
READING_TOPIC = IMAGE_TOPIC.replace('/image', '/reading')

METER_NAME = os.getenv('METER_NAME', 'Electric Meter Reading')
METER_ID = os.getenv('METER_ID', 'electric_meter_reading')
METER_UNIT = os.getenv('METER_UNIT', 'kWh')
DEVICE_CLASS = os.getenv('DEVICE_CLASS', 'energy')

last_reading = None
reading_history = []
last_image_raw = None
last_image_processed = None

HTTP_PORT = int(os.getenv('HTTP_PORT', 8080))

# --- Crop (% of image dimensions) ---
# View /image/raw for the red box; adjust until it covers only the digit faces.
CROP_TOP_PCT    = float(os.getenv('CROP_TOP_PCT',    40))
CROP_BOTTOM_PCT = float(os.getenv('CROP_BOTTOM_PCT', 62))
CROP_LEFT_PCT   = float(os.getenv('CROP_LEFT_PCT',    0))
CROP_RIGHT_PCT  = float(os.getenv('CROP_RIGHT_PCT',  82))

# --- Pre-processing ---
AUTOCONTRAST_CUTOFF = float(os.getenv('AUTOCONTRAST_CUTOFF', 2))

# Initialise EasyOCR once — loading the model takes a few seconds
print("Loading EasyOCR model...")
ocr_reader = easyocr.Reader(['en'], gpu=False, verbose=False)
print("✓ EasyOCR ready")


def annotate_raw_with_crop(raw_bytes):
    """Return the raw image with the active crop region drawn as a red rectangle."""
    img = Image.open(io.BytesIO(raw_bytes))
    h = img.height
    top    = int(h * CROP_TOP_PCT    / 100)
    bottom = int(h * CROP_BOTTOM_PCT / 100)
    left   = int(img.width * CROP_LEFT_PCT   / 100)
    right  = int(img.width * CROP_RIGHT_PCT  / 100)
    draw = ImageDraw.Draw(img)
    draw.rectangle([(left, top), (right, bottom)], outline='red', width=3)
    buf = io.BytesIO()
    img.save(buf, format='JPEG')
    return buf.getvalue()


class ImageHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/image/raw':
            data = annotate_raw_with_crop(last_image_raw) if last_image_raw else None
            self._serve_image(data)
        elif self.path in ('/image', '/image/processed'):
            self._serve_image(last_image_processed)
        elif self.path == '/':
            self._serve_status()
        else:
            self.send_error(404)

    def _serve_image(self, img_bytes):
        if img_bytes is None:
            self.send_response(503)
            self.end_headers()
            self.wfile.write(b'No image received yet')
            return
        self.send_response(200)
        self.send_header('Content-Type', 'image/jpeg')
        self.send_header('Content-Length', str(len(img_bytes)))
        self.end_headers()
        self.wfile.write(img_bytes)

    def _serve_status(self):
        body = f"""<html><body>
<h2>{METER_NAME}</h2>
<p>Last reading: {last_reading} {METER_UNIT if last_reading else 'N/A'}</p>
<p><a href="/image/raw">Raw image (with crop box)</a> &nbsp; <a href="/image/processed">Cropped image sent to OCR</a></p>
</body></html>""".encode()
        self.send_response(200)
        self.send_header('Content-Type', 'text/html')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        pass  # suppress access logs


def start_http_server():
    server = HTTPServer(('0.0.0.0', HTTP_PORT), ImageHandler)
    print(f"✓ Image viewer running on port {HTTP_PORT}")
    server.serve_forever()


def extract_meter_reading(image_data):
    """Extract meter reading from image using EasyOCR."""
    global last_image_raw, last_image_processed
    try:
        last_image_raw = image_data

        # Load and crop to the digit strip (keep colour — EasyOCR handles it natively)
        image = Image.open(io.BytesIO(image_data))
        h, w = image.height, image.width
        cropped = image.crop((int(w * CROP_LEFT_PCT   / 100), int(h * CROP_TOP_PCT    / 100),
                              int(w * CROP_RIGHT_PCT  / 100), int(h * CROP_BOTTOM_PCT / 100)))

        # Light contrast normalisation helps on very dark/bright images
        cropped_gray = ImageOps.autocontrast(cropped.convert('L'), cutoff=AUTOCONTRAST_CUTOFF)
        cropped = cropped_gray.convert('RGB')

        # Save cropped image so /image/processed shows exactly what EasyOCR receives
        buf = io.BytesIO()
        cropped.save(buf, format='JPEG')
        last_image_processed = buf.getvalue()

        # Run EasyOCR — allowlist restricts recognition to digits only
        # detail=1 gives bounding boxes so we can sort left-to-right
        results = ocr_reader.readtext(
            np.array(cropped),
            allowlist='0123456789',
            detail=1,
            paragraph=False,
        )
        # Sort detections left-to-right by the x-coordinate of the top-left corner
        results.sort(key=lambda r: r[0][0][0])

        for bbox, txt, conf in results:
            print(f"  EasyOCR: '{txt}' (conf={conf:.2f}, x={int(bbox[0][0])})")

        text = ''.join(txt for _, txt, _ in results)
        clean_text = re.sub(r'[^0-9]', '', text)
        print(f"  Joined: '{clean_text}'")

        # Accept 4–7 digits: meter shows ~4 integer digits + optional decimal digits
        matches = re.findall(r'\d{4,7}', clean_text)

        if matches:
            # Take the longest match (most complete reading)
            raw_digits = max(matches, key=len)
            reading = int(raw_digits)

            global last_reading
            if last_reading is not None:
                if reading < last_reading - 10:
                    print(f"  ⚠ Reading decreased suspiciously: {last_reading} -> {reading}, skipping")
                    return None
                elif reading > last_reading + 1000:
                    print(f"  ⚠ Reading increased suspiciously: {last_reading} -> {reading}, skipping")
                    return None

            print(f"  ✓ Reading extracted: {reading} {METER_UNIT}")
            last_reading = reading

            reading_history.append({
                'reading': reading,
                'timestamp': time.time(),
                'raw_text': text,
            })
            if len(reading_history) > 100:
                reading_history.pop(0)

            return reading
        else:
            print(f"  ✗ No valid 4-7 digit reading found")
            return None

    except Exception as e:
        print(f"  ✗ OCR error: {e}")
        return None


def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print(f"✓ Connected to MQTT broker at {MQTT_BROKER}:{MQTT_PORT}")
        client.subscribe(IMAGE_TOPIC)
        print(f"✓ Subscribed to {IMAGE_TOPIC}")

        discovery_config = {
            "name": METER_NAME,
            "unique_id": METER_ID,
            "state_topic": STATE_TOPIC,
            "unit_of_measurement": METER_UNIT,
            "device_class": DEVICE_CLASS,
            "state_class": "total_increasing",
            "icon": "mdi:counter"
        }
        client.publish(CONFIG_TOPIC, json.dumps(discovery_config), retain=True)
        print(f"✓ Published HA discovery config to {CONFIG_TOPIC}")
    else:
        print(f"✗ Connection failed with code {rc}")


def on_message(client, userdata, msg):
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    print(f"\n[{timestamp}] Image received ({len(msg.payload)} bytes)")

    reading = extract_meter_reading(msg.payload)

    if reading:
        client.publish(STATE_TOPIC, str(reading), retain=True)
        detail = {
            "reading": reading,
            "timestamp": time.time(),
            "timestamp_human": timestamp
        }
        client.publish(READING_TOPIC, json.dumps(detail))
        print(f"  ✓ Published to HA: {reading} {METER_UNIT}")
    else:
        print(f"  ✗ No valid reading extracted")


def main():
    print("=" * 60)
    print(f"Meter OCR Service Starting - {METER_NAME}")
    print("=" * 60)
    print(f"MQTT Broker: {MQTT_BROKER}:{MQTT_PORT}")
    print(f"Unit: {METER_UNIT} (Class: {DEVICE_CLASS})")
    print(f"Image Topic: {IMAGE_TOPIC}")
    print(f"State Topic: {STATE_TOPIC}")
    print("=" * 60)

    t = threading.Thread(target=start_http_server, daemon=True)
    t.start()

    client = mqtt.Client()
    client.on_connect = on_connect
    client.on_message = on_message

    if MQTT_USER:
        client.username_pw_set(MQTT_USER, MQTT_PASS)

    while True:
        try:
            print(f"\nConnecting to MQTT broker...")
            client.connect(MQTT_BROKER, MQTT_PORT, 60)
            break
        except Exception as e:
            print(f"✗ Connection failed: {e}")
            print("  Retrying in 5 seconds...")
            time.sleep(5)

    print("✓ Service running. Waiting for images...\n")
    client.loop_forever()


if __name__ == "__main__":
    main()
