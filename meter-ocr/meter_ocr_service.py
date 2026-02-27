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
last_image_annotated = None

HTTP_PORT = int(os.getenv('HTTP_PORT', 8080))

# --- Vertical crop (% of image height) — shared by both fields ---
CROP_TOP_PCT    = float(os.getenv('CROP_TOP_PCT',    40))
CROP_BOTTOM_PCT = float(os.getenv('CROP_BOTTOM_PCT', 62))

# --- Two detection fields (% of image width) ---
# Field 1 (red box):  integer digits, left of the decimal separator
# Field 2 (blue box): decimal digits, right of the decimal separator
# Set FIELD2_RIGHT_PCT <= FIELD2_LEFT_PCT to disable field 2 (single-field mode)
FIELD1_LEFT_PCT  = float(os.getenv('FIELD1_LEFT_PCT',   0))
FIELD1_RIGHT_PCT = float(os.getenv('FIELD1_RIGHT_PCT', 50))
FIELD2_LEFT_PCT  = float(os.getenv('FIELD2_LEFT_PCT',  52))
FIELD2_RIGHT_PCT = float(os.getenv('FIELD2_RIGHT_PCT', 82))

# --- Pre-processing ---
AUTOCONTRAST_CUTOFF = float(os.getenv('AUTOCONTRAST_CUTOFF', 2))

print("Loading EasyOCR model...")
ocr_reader = easyocr.Reader(['en'], gpu=False, verbose=False)
print("✓ EasyOCR ready")


def _preprocess(region):
    gray = ImageOps.autocontrast(region.convert('L'), cutoff=AUTOCONTRAST_CUTOFF)
    return gray.convert('RGB')


def annotate_raw_with_crop(raw_bytes):
    """Return raw image with field regions as coloured rectangles (red=field1, blue=field2)."""
    img = Image.open(io.BytesIO(raw_bytes))
    h, w = img.height, img.width
    top    = int(h * CROP_TOP_PCT    / 100)
    bottom = int(h * CROP_BOTTOM_PCT / 100)
    f1l = int(w * FIELD1_LEFT_PCT  / 100)
    f1r = int(w * FIELD1_RIGHT_PCT / 100)
    f2l = int(w * FIELD2_LEFT_PCT  / 100)
    f2r = int(w * FIELD2_RIGHT_PCT / 100)
    draw = ImageDraw.Draw(img)
    draw.rectangle([(f1l, top), (f1r, bottom)], outline='red',  width=3)
    if f2r > f2l:
        draw.rectangle([(f2l, top), (f2r, bottom)], outline='blue', width=3)
    buf = io.BytesIO()
    img.save(buf, format='JPEG')
    return buf.getvalue()


def _ocr_field(img_rgb, label):
    """Run EasyOCR on a preprocessed PIL image. Returns (digit_str, raw_results)."""
    results = ocr_reader.readtext(
        np.array(img_rgb),
        allowlist='0123456789',
        detail=1,
        paragraph=False,
    )
    results.sort(key=lambda r: r[0][0][0])
    for bbox, txt, conf in results:
        print(f"  {label}: '{txt}' (conf={conf:.2f}, x={int(bbox[0][0])})")
    digits = re.sub(r'[^0-9]', '', ''.join(txt for _, txt, _ in results))
    return digits, results


class ImageHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/image/raw':
            data = annotate_raw_with_crop(last_image_raw) if last_image_raw else None
            self._serve_image(data)
        elif self.path in ('/image', '/image/processed'):
            self._serve_image(last_image_processed)
        elif self.path == '/image/annotated':
            self._serve_image(last_image_annotated)
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
<p>
  <a href="/image/raw">Raw image (field boxes: red=integers, blue=decimals)</a><br>
  <a href="/image/processed">Processed fields (what EasyOCR receives)</a><br>
  <a href="/image/annotated">Annotated detections (red=field1, blue=field2)</a>
</p>
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
    global last_image_raw, last_image_processed, last_image_annotated, last_reading
    try:
        last_image_raw = image_data
        image = Image.open(io.BytesIO(image_data))
        h, w = image.height, image.width

        top    = int(h * CROP_TOP_PCT    / 100)
        bottom = int(h * CROP_BOTTOM_PCT / 100)
        f1l = int(w * FIELD1_LEFT_PCT  / 100)
        f1r = int(w * FIELD1_RIGHT_PCT / 100)
        f2l = int(w * FIELD2_LEFT_PCT  / 100)
        f2r = int(w * FIELD2_RIGHT_PCT / 100)

        use_field2 = f2r > f2l

        field1 = _preprocess(image.crop((f1l, top, f1r, bottom)))
        field2 = _preprocess(image.crop((f2l, top, f2r, bottom))) if use_field2 else None

        # --- Combined processed image (field1 | grey gap | field2) ---
        GAP = 8
        f1w = f1r - f1l
        f2w = (f2r - f2l) if use_field2 else 0
        combined_w = f1w + (GAP + f2w if use_field2 else 0)
        combined_h = bottom - top
        combined = Image.new('RGB', (combined_w, combined_h), color=(160, 160, 160))
        combined.paste(field1, (0, 0))
        if use_field2:
            combined.paste(field2, (f1w + GAP, 0))
        buf = io.BytesIO()
        combined.save(buf, format='JPEG')
        last_image_processed = buf.getvalue()

        # --- OCR each field independently ---
        digits1, det1 = _ocr_field(field1, 'Field1')
        digits2, det2 = _ocr_field(field2, 'Field2') if use_field2 else ('', [])
        clean_text = digits1 + digits2
        print(f"  Field1='{digits1}' Field2='{digits2}' Combined='{clean_text}'")

        # --- Annotated image with bounding boxes ---
        annotated = combined.copy()
        draw = ImageDraw.Draw(annotated)
        for bbox, txt, conf in det1:
            x0, y0 = int(bbox[0][0]), int(bbox[0][1])
            x1, y1 = int(bbox[2][0]), int(bbox[2][1])
            draw.rectangle([(x0, y0), (x1, y1)], outline='red', width=2)
            draw.text((x0 + 2, y0 + 2), txt, fill='red')
        if use_field2:
            offset_x = f1w + GAP
            for bbox, txt, conf in det2:
                x0, y0 = int(bbox[0][0]) + offset_x, int(bbox[0][1])
                x1, y1 = int(bbox[2][0]) + offset_x, int(bbox[2][1])
                draw.rectangle([(x0, y0), (x1, y1)], outline='blue', width=2)
                draw.text((x0 + 2, y0 + 2), txt, fill='blue')
        buf = io.BytesIO()
        annotated.save(buf, format='JPEG')
        last_image_annotated = buf.getvalue()

        # --- Validate and sanity-check the reading ---
        matches = re.findall(r'\d{4,7}', clean_text)
        if not matches:
            print(f"  ✗ No valid 4-7 digit reading found in '{clean_text}'")
            return None

        raw_digits = max(matches, key=len)
        reading = int(raw_digits)

        if last_reading is not None:
            if reading < last_reading - 10:
                print(f"  ⚠ Reading decreased suspiciously: {last_reading} -> {reading}, skipping")
                return None
            elif reading > last_reading + 1000:
                print(f"  ⚠ Reading increased suspiciously: {last_reading} -> {reading}, skipping")
                return None

        print(f"  ✓ Reading extracted: {reading} {METER_UNIT}")
        last_reading = reading
        reading_history.append({'reading': reading, 'timestamp': time.time(), 'raw_text': clean_text})
        if len(reading_history) > 100:
            reading_history.pop(0)
        return reading

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
