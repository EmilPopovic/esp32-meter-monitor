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

# Vertical crop shared by all digit fields (% of image height)
CROP_TOP_PCT    = float(os.getenv('CROP_TOP_PCT',    40))
CROP_BOTTOM_PCT = float(os.getenv('CROP_BOTTOM_PCT', 62))

# Pre-processing
AUTOCONTRAST_CUTOFF = float(os.getenv('AUTOCONTRAST_CUTOFF', 2))


def _parse_fields(env_var, default):
    """Parse 'L:R,L:R,...' (% of image width) into [(left_pct, right_pct), ...]."""
    raw = os.getenv(env_var, default)
    fields = []
    for part in raw.split(','):
        part = part.strip()
        if ':' in part:
            try:
                l, r = part.split(':', 1)
                fields.append((float(l), float(r)))
            except ValueError:
                pass
    return fields


# Each entry covers exactly one digit roller.
# View /image/raw to see the numbered boxes and tune boundaries.
# Integer digits (red boxes): left of decimal separator
DIGIT_FIELDS   = _parse_fields('DIGIT_FIELDS',   '5:15,17:27,29:39,41:50')
# Decimal digits (blue boxes): right of decimal separator
DECIMAL_FIELDS = _parse_fields('DECIMAL_FIELDS', '53:62,64:72')


print("Loading EasyOCR model...")
ocr_reader = easyocr.Reader(['en'], gpu=False, verbose=False)
print("✓ EasyOCR ready")


def _preprocess(region):
    gray = ImageOps.autocontrast(region.convert('L'), cutoff=AUTOCONTRAST_CUTOFF)
    return gray.convert('RGB')


def _ocr_digit(img_rgb, label):
    """OCR a single-roller crop. Returns (digit_char, results)."""
    results = ocr_reader.readtext(
        np.array(img_rgb),
        allowlist='0123456789',
        detail=1,
        paragraph=False,
    )
    if not results:
        print(f"  {label}: - (nothing detected)")
        return '', []
    # Take the highest-confidence detection and use its first digit
    best_bbox, best_txt, best_conf = max(results, key=lambda r: r[2])
    digit = re.sub(r'[^0-9]', '', best_txt)
    digit = digit[0] if digit else ''
    print(f"  {label}: '{digit}' (conf={best_conf:.2f})")
    return digit, results


def annotate_raw_with_crop(raw_bytes):
    """Raw image with each digit field drawn as a numbered box (red=integer, blue=decimal)."""
    img = Image.open(io.BytesIO(raw_bytes))
    h, w = img.height, img.width
    top    = int(h * CROP_TOP_PCT    / 100)
    bottom = int(h * CROP_BOTTOM_PCT / 100)
    draw = ImageDraw.Draw(img)

    for i, (l, r) in enumerate(DIGIT_FIELDS):
        x0, x1 = int(w * l / 100), int(w * r / 100)
        draw.rectangle([(x0, top), (x1, bottom)], outline='red', width=2)
        draw.text((x0 + 3, top + 3), str(i + 1), fill='red')

    for i, (l, r) in enumerate(DECIMAL_FIELDS):
        x0, x1 = int(w * l / 100), int(w * r / 100)
        draw.rectangle([(x0, top), (x1, bottom)], outline='blue', width=2)
        draw.text((x0 + 3, top + 3), f"d{i + 1}", fill='blue')

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
  <a href="/image/raw">Raw image — numbered field boxes (red=integer, blue=decimal)</a><br>
  <a href="/image/processed">Processed — individual digit crops</a><br>
  <a href="/image/annotated">Annotated — EasyOCR detections per field</a>
</p>
</body></html>""".encode()
        self.send_response(200)
        self.send_header('Content-Type', 'text/html')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        pass


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
        strip_h = bottom - top

        # --- Crop each field ---
        def crop_fields(field_defs):
            return [_preprocess(image.crop((int(w * l / 100), top, int(w * r / 100), bottom)))
                    for l, r in field_defs]

        int_imgs = crop_fields(DIGIT_FIELDS)
        dec_imgs = crop_fields(DECIMAL_FIELDS)

        # --- Build composite processed image ---
        INNER_GAP = 4
        OUTER_GAP = 14
        # Lay out: [int fields] [outer gap] [dec fields]
        rendered = []   # (img, x_offset, is_decimal, field_index)
        x = 0
        for i, img in enumerate(int_imgs):
            if i > 0:
                x += INNER_GAP
            rendered.append((img, x, False, i))
            x += img.width
        if dec_imgs:
            x += OUTER_GAP
            for i, img in enumerate(dec_imgs):
                if i > 0:
                    x += INNER_GAP
                rendered.append((img, x, True, i))
                x += img.width

        combined = Image.new('RGB', (x, strip_h), color=(140, 140, 140))
        for img, x_off, _, _ in rendered:
            combined.paste(img, (x_off, 0))
        buf = io.BytesIO()
        combined.save(buf, format='JPEG')
        last_image_processed = buf.getvalue()

        # --- OCR each field independently ---
        results_per_field = []
        for i, img in enumerate(int_imgs):
            digit, dets = _ocr_digit(img, f"INT{i+1}")
            results_per_field.append((digit, dets, False, i))
        for i, img in enumerate(dec_imgs):
            digit, dets = _ocr_digit(img, f"DEC{i+1}")
            results_per_field.append((digit, dets, True, i))

        # --- Annotated image ---
        annotated = combined.copy()
        draw = ImageDraw.Draw(annotated)
        for (digit, dets, is_dec, field_idx), (img, x_off, _, _) in zip(results_per_field, rendered):
            color = 'blue' if is_dec else 'red'
            for bbox, txt, conf in dets:
                x0, y0 = int(bbox[0][0]) + x_off, int(bbox[0][1])
                x1, y1 = int(bbox[2][0]) + x_off, int(bbox[2][1])
                draw.rectangle([(x0, y0), (x1, y1)], outline=color, width=2)
            # Show what we extracted from this field
            draw.text((x_off + 2, 2), digit if digit else '?', fill=color)
        buf = io.BytesIO()
        annotated.save(buf, format='JPEG')
        last_image_annotated = buf.getvalue()

        # --- Assemble reading ---
        int_digits  = ''.join(d for d, _, is_dec, _ in results_per_field if not is_dec)
        dec_digits  = ''.join(d for d, _, is_dec, _ in results_per_field if is_dec)
        clean_text  = int_digits + dec_digits
        print(f"  INT='{int_digits}' DEC='{dec_digits}' Combined='{clean_text}'")

        # Require all configured fields to return a digit (any blank = roller mid-transition)
        expected_len = len(DIGIT_FIELDS) + len(DECIMAL_FIELDS)
        if len(clean_text) < expected_len:
            print(f"  ✗ Only {len(clean_text)}/{expected_len} digits read — likely mid-transition")
            return None

        reading = int(clean_text[:7])  # cap at 7 digits for safety

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
        detail = {"reading": reading, "timestamp": time.time(), "timestamp_human": timestamp}
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
