#!/usr/bin/env python3
import paho.mqtt.client as mqtt
import pytesseract
from PIL import Image, ImageEnhance
import io
import re
import json
import time
import os

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

# Tesseract configuration for digits
TESSERACT_CONFIG = '--psm 7 -c tessedit_char_whitelist=0123456789'

last_reading = None
reading_history = []

def extract_meter_reading(image_data):
    """Extract 7-digit meter reading from image"""
    try:
        # Load image
        image = Image.open(io.BytesIO(image_data))
        
        # Convert to grayscale
        image = image.convert('L')
        
        # Enhance contrast for better OCR
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(2.0)
        
        # Perform OCR
        text = pytesseract.image_to_string(image, config=TESSERACT_CONFIG)
        
        # Clean text - remove spaces and non-digits
        clean_text = re.sub(r'[^0-9]', '', text)
        
        print(f"  OCR raw text: '{text.strip()}'")
        print(f"  Cleaned: '{clean_text}'")
        
        # Extract 6 or 7 digit number (some meters have 6 digits)
        matches = re.findall(r'\d{6,7}', clean_text)
        
        if matches:
            reading = int(matches[0])
            
            # Sanity check: reading should only increase or stay same
            global last_reading
            if last_reading is not None:
                if reading < last_reading - 10:  # Allow small decreases for OCR errors
                    print(f"  ⚠ Reading decreased suspiciously: {last_reading} -> {reading}")
                    print(f"  Skipping potentially incorrect reading")
                    return None
                elif reading > last_reading + 1000:  # Unrealistic increase
                    print(f"  ⚠ Reading increased suspiciously: {last_reading} -> {reading}")
                    print(f"  Skipping potentially incorrect reading")
                    return None
            
            print(f"  ✓ Reading extracted: {reading} {METER_UNIT}")
            last_reading = reading
            
            # Keep history for debugging
            reading_history.append({
                'reading': reading,
                'timestamp': time.time(),
                'raw_text': text.strip()
            })
            if len(reading_history) > 100:
                reading_history.pop(0)
            
            return reading
        else:
            print(f"  ✗ No valid 6-7 digit reading found")
            return None
            
    except Exception as e:
        print(f"  ✗ OCR error: {e}")
        return None

def on_connect(client, userdata, flags, rc):
    """Callback when connected to MQTT broker"""
    if rc == 0:
        print(f"✓ Connected to MQTT broker at {MQTT_BROKER}:{MQTT_PORT}")
        
        # Subscribe to image topic
        client.subscribe(IMAGE_TOPIC)
        print(f"✓ Subscribed to {IMAGE_TOPIC}")
        
        # Publish Home Assistant discovery config
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
    """Callback when image received"""
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    print(f"\n[{timestamp}] Image received ({len(msg.payload)} bytes)")
    
    # Extract reading
    reading = extract_meter_reading(msg.payload)
    
    if reading:
        # Publish to Home Assistant
        client.publish(STATE_TOPIC, str(reading), retain=True)
        
        # Also publish detailed info for debugging
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
    """Main loop"""
    print("=" * 60)
    print(f"Meter OCR Service Starting - {METER_NAME}")
    print("=" * 60)
    print(f"MQTT Broker: {MQTT_BROKER}:{MQTT_PORT}")
    print(f"Unit: {METER_UNIT} (Class: {DEVICE_CLASS})")
    print(f"Image Topic: {IMAGE_TOPIC}")
    print(f"State Topic: {STATE_TOPIC}")
    print("=" * 60)
    
    # Create MQTT client
    client = mqtt.Client()
    client.on_connect = on_connect
    client.on_message = on_message
    
    if MQTT_USER:
        client.username_pw_set(MQTT_USER, MQTT_PASS)
    
    # Connect to broker with retry
    while True:
        try:
            print(f"\nConnecting to MQTT broker...")
            client.connect(MQTT_BROKER, MQTT_PORT, 60)
            break
        except Exception as e:
            print(f"✗ Connection failed: {e}")
            print("  Retrying in 5 seconds...")
            time.sleep(5)
    
    # Start loop
    print("✓ Service running. Waiting for images...\n")
    client.loop_forever()

if __name__ == "__main__":
    main()
