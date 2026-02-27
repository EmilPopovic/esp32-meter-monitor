#!/usr/bin/env python3
import paho.mqtt.client as mqtt
import easyocr
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFilter
import io
import re
import json
import time
import os
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler

# ── MQTT ────────────────────────────────────────────────────────────────────
MQTT_BROKER = os.getenv('MQTT_BROKER', 'localhost')
MQTT_PORT   = int(os.getenv('MQTT_PORT', 1883))
MQTT_USER   = os.getenv('MQTT_USER', None)
MQTT_PASS   = os.getenv('MQTT_PASS', None)

IMAGE_TOPIC   = os.getenv('IMAGE_TOPIC',   'home/meter/electric/image')
STATE_TOPIC   = os.getenv('STATE_TOPIC',   'homeassistant/sensor/electric_meter/state')
CONFIG_TOPIC  = os.getenv('CONFIG_TOPIC',  'homeassistant/sensor/electric_meter/config')
READING_TOPIC = IMAGE_TOPIC.replace('/image', '/reading')

METER_NAME   = os.getenv('METER_NAME',   'Electric Meter Reading')
METER_ID     = os.getenv('METER_ID',     'electric_meter_reading')
METER_UNIT   = os.getenv('METER_UNIT',   'kWh')
DEVICE_CLASS = os.getenv('DEVICE_CLASS', 'energy')

# ── State ────────────────────────────────────────────────────────────────────
last_reading        = None
reading_history     = []
last_image_raw      = None
last_image_processed = None
last_image_annotated = None

HTTP_PORT = int(os.getenv('HTTP_PORT', 8080))

# ── Field config (overridden by config file if present) ──────────────────────
CROP_TOP_PCT    = float(os.getenv('CROP_TOP_PCT',    40))
CROP_BOTTOM_PCT = float(os.getenv('CROP_BOTTOM_PCT', 62))

def _parse_fields(env_var, default):
    raw = os.getenv(env_var, default)
    fields = []
    for part in raw.split(','):
        part = part.strip()
        if ':' in part:
            try:
                l, r = part.split(':', 1)
                fields.append([float(l), float(r)])
            except ValueError:
                pass
    return fields

DIGIT_FIELDS   = _parse_fields('DIGIT_FIELDS',   '5:15,17:27,29:39,41:50')
DECIMAL_FIELDS = _parse_fields('DECIMAL_FIELDS', '53:62,64:72')

# ── Pre-processing ───────────────────────────────────────────────────────────
CLAHE_CLIP_LIMIT     = float(os.getenv('CLAHE_CLIP_LIMIT',     3.0))
CLAHE_TILE_SIZE      = int(os.getenv('CLAHE_TILE_SIZE',        2))
BRIGHTNESS_BOOST     = float(os.getenv('BRIGHTNESS_BOOST',     1.0))
DIGIT_SCALE          = int(os.getenv('DIGIT_SCALE',            3))
DIGIT_CONF_THRESHOLD = float(os.getenv('DIGIT_CONF_THRESHOLD', 0.15))

# ── Persistent config file ───────────────────────────────────────────────────
CONFIG_PATH = os.getenv('CONFIG_PATH', '/app/config/fields.json')

def load_config():
    global CROP_TOP_PCT, CROP_BOTTOM_PCT, DIGIT_FIELDS, DECIMAL_FIELDS
    if not os.path.exists(CONFIG_PATH):
        return
    try:
        with open(CONFIG_PATH) as f:
            cfg = json.load(f)
        CROP_TOP_PCT    = float(cfg.get('crop_top_pct',    CROP_TOP_PCT))
        CROP_BOTTOM_PCT = float(cfg.get('crop_bottom_pct', CROP_BOTTOM_PCT))
        DIGIT_FIELDS    = [[float(v) for v in x] for x in cfg.get('digit_fields',   DIGIT_FIELDS)]
        DECIMAL_FIELDS  = [[float(v) for v in x] for x in cfg.get('decimal_fields', DECIMAL_FIELDS)]
        print(f"✓ Config loaded from {CONFIG_PATH} "
              f"({len(DIGIT_FIELDS)} int, {len(DECIMAL_FIELDS)} dec fields)")
    except Exception as e:
        print(f"⚠ Could not load config: {e}")

def save_config(cfg_dict):
    global CROP_TOP_PCT, CROP_BOTTOM_PCT, DIGIT_FIELDS, DECIMAL_FIELDS
    CROP_TOP_PCT    = float(cfg_dict['crop_top_pct'])
    CROP_BOTTOM_PCT = float(cfg_dict['crop_bottom_pct'])
    DIGIT_FIELDS    = [[float(v) for v in x] for x in cfg_dict['digit_fields']]
    DECIMAL_FIELDS  = [[float(v) for v in x] for x in cfg_dict['decimal_fields']]
    os.makedirs(os.path.dirname(CONFIG_PATH) or '.', exist_ok=True)
    with open(CONFIG_PATH, 'w') as f:
        json.dump(cfg_dict, f, indent=2)
    print(f"✓ Config saved ({len(DIGIT_FIELDS)} int, {len(DECIMAL_FIELDS)} dec fields)")

# ── EasyOCR ──────────────────────────────────────────────────────────────────
print("Loading EasyOCR model...")
ocr_reader = easyocr.Reader(['en'], gpu=False, verbose=False)
print("✓ EasyOCR ready")

_clahe = None

def _preprocess(region):
    global _clahe
    gray = np.array(region.convert('L'))
    if _clahe is None:
        _clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT,
                                  tileGridSize=(CLAHE_TILE_SIZE, CLAHE_TILE_SIZE))
    gray = _clahe.apply(gray)
    if BRIGHTNESS_BOOST != 1.0:
        gray = np.clip(gray.astype(np.float32) * BRIGHTNESS_BOOST, 0, 255).astype(np.uint8)
    img = Image.fromarray(gray)
    img = img.filter(ImageFilter.UnsharpMask(radius=1, percent=120, threshold=2))
    return img.convert('RGB')

def _ocr_digit(img_rgb, label):
    img_np   = np.array(img_rgb)
    h, w     = img_np.shape[:2]
    img_gray = np.array(img_rgb.convert('L'))
    results  = None
    try:
        results = ocr_reader.recognize(
            img_gray, horizontal_list=[[0, w, 0, h]], free_list=[],
            allowlist='0123456789', detail=1,
        )
    except Exception as e:
        print(f"  {label}: recognize() unavailable ({e}), using readtext fallback")
    if not results:
        scaled = img_rgb.resize((w * DIGIT_SCALE, h * DIGIT_SCALE), Image.LANCZOS) \
                 if DIGIT_SCALE > 1 else img_rgb
        results = ocr_reader.readtext(
            np.array(scaled), allowlist='0123456789', detail=1, paragraph=False,
            text_threshold=0.4, low_text=0.2, link_threshold=0.3,
        )
    if not results:
        print(f"  {label}: - (nothing detected)")
        return '', []
    best_bbox, best_txt, best_conf = max(results, key=lambda r: r[2])
    if best_conf < DIGIT_CONF_THRESHOLD:
        print(f"  {label}: - (conf={best_conf:.2f} below threshold)")
        return '', []
    digit = re.sub(r'[^0-9]', '', best_txt)
    digit = digit[0] if digit else ''
    print(f"  {label}: '{digit}' (conf={best_conf:.2f})")
    return digit, [(best_bbox, best_txt, best_conf)]

# ── Image annotation ─────────────────────────────────────────────────────────
def annotate_raw_with_crop(raw_bytes):
    img  = Image.open(io.BytesIO(raw_bytes))
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

# ── Configure page HTML ──────────────────────────────────────────────────────
_CONFIGURE_HTML = r"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Configure — METER_NAME_PLACEHOLDER</title>
<style>
*{box-sizing:border-box}
body{font:14px monospace;margin:0;padding:16px;background:#111;color:#ddd}
h2{margin:0 0 8px}
#tb{display:flex;gap:6px;margin-bottom:8px;flex-wrap:wrap;align-items:center}
button{padding:5px 10px;background:#333;color:#ddd;border:1px solid #555;border-radius:3px;cursor:pointer;font:inherit}
button:hover{background:#444}
#sav{background:#175;border-color:#1a7}
#sav:hover{background:#1a6}
#del:disabled{opacity:.4;cursor:default}
#st{font-size:12px;margin-left:6px}
canvas{display:block;cursor:default}
#hint{font-size:11px;color:#666;margin-top:6px}
a{color:#888}
</style></head>
<body>
<h2>&#9881; METER_NAME_PLACEHOLDER &mdash; Field Configuration</h2>
<div id="tb">
  <button onclick="addField('digit')">&#xFF0B; Integer digit (red)</button>
  <button onclick="addField('decimal')">&#xFF0B; Decimal digit (blue)</button>
  <button id="del" disabled onclick="delSel()">&#x2715; Delete selected</button>
  <button id="sav" onclick="saveCfg()">&#x1F4BE; Save</button>
  <a href="/">&#x2190; back</a>
  <span id="st"></span>
</div>
<canvas id="cv"></canvas>
<p id="hint">Drag box body to move &nbsp;|&nbsp; Drag left/right edge to resize &nbsp;|&nbsp;
Drag yellow line to adjust crop &nbsp;|&nbsp; Click to select then delete &nbsp;|&nbsp;
<a href="/configure">Reload image</a></p>
<script>
(function(){
const cv=document.getElementById('cv'),ctx=cv.getContext('2d');
let cfg=null,img=null,drag=null,sel=null;

async function init(){
  cfg=await fetch('/config').then(r=>r.json());
  img=new Image();
  img.onload=()=>{
    const mw=Math.min(900,window.innerWidth-40);
    cv.width=mw; cv.height=mw*(img.naturalHeight/img.naturalWidth);
    draw();
  };
  img.onerror=()=>{cv.width=800;cv.height=500;draw();};
  img.src='/image/reference?'+Date.now();
}

const px=p=>p/100*cv.width, py=p=>p/100*cv.height;
const xp=x=>Math.max(0,Math.min(100,x/cv.width*100));
const yp=y=>Math.max(0,Math.min(100,y/cv.height*100));

function draw(){
  ctx.clearRect(0,0,cv.width,cv.height);
  if(img&&img.complete&&img.naturalWidth>0) ctx.drawImage(img,0,0,cv.width,cv.height);
  else{ctx.fillStyle='#222';ctx.fillRect(0,0,cv.width,cv.height);}
  if(!cfg)return;
  const t=py(cfg.crop_top_pct),b=py(cfg.crop_bottom_pct);
  // Crop lines
  ctx.strokeStyle='rgba(255,220,0,.85)';ctx.lineWidth=1;ctx.setLineDash([5,3]);
  [t,b].forEach(y=>{ctx.beginPath();ctx.moveTo(0,y);ctx.lineTo(cv.width,y);ctx.stroke();});
  ctx.setLineDash([]);
  // Crop handles
  ctx.fillStyle='#fd0';
  [t,b].forEach(y=>ctx.fillRect(cv.width/2-24,y-4,48,8));
  // Fields
  function drawG(fields,group,col){
    fields.forEach((f,i)=>{
      const x0=px(f[0]),x1=px(f[1]),is=sel&&sel.g===group&&sel.i===i;
      ctx.strokeStyle=is?'#fff':col;ctx.lineWidth=is?3:2;
      ctx.strokeRect(x0,t,x1-x0,b-t);
      ctx.fillStyle=is?'#fff':col;ctx.font='bold 12px monospace';
      ctx.fillText(group==='digit'?String(i+1):'d'+(i+1),x0+3,t+14);
      const my=(t+b)/2,eh=20;
      ctx.fillRect(x0-3,my-eh/2,6,eh);ctx.fillRect(x1-3,my-eh/2,6,eh);
    });
  }
  drawG(cfg.digit_fields,'digit','red');
  drawG(cfg.decimal_fields,'decimal','#55f');
}

function hitTest(x,y){
  if(!cfg)return null;
  const t=py(cfg.crop_top_pct),b=py(cfg.crop_bottom_pct),E=7,cx=cv.width/2;
  if(Math.abs(y-t)<8&&Math.abs(x-cx)<24)return{type:'crop',w:'top'};
  if(Math.abs(y-b)<8&&Math.abs(x-cx)<24)return{type:'crop',w:'bottom'};
  function chk(fields,group){
    for(let i=fields.length-1;i>=0;i--){
      const x0=px(fields[i][0]),x1=px(fields[i][1]);
      if(y<t-5||y>b+5)continue;
      if(Math.abs(x-x0)<E)return{type:'f',g:group,i,e:'l'};
      if(Math.abs(x-x1)<E)return{type:'f',g:group,i,e:'r'};
      if(x>x0&&x<x1)return{type:'f',g:group,i,e:'b'};
    }
  }
  return chk(cfg.digit_fields,'digit')||chk(cfg.decimal_fields,'decimal');
}

cv.onmousedown=e=>{
  const r=cv.getBoundingClientRect(),x=e.clientX-r.left,y=e.clientY-r.top,h=hitTest(x,y);
  if(!h){sel=null;document.getElementById('del').disabled=true;draw();return;}
  if(h.type==='f'){
    sel={g:h.g,i:h.i};document.getElementById('del').disabled=false;
    const fs=h.g==='digit'?cfg.digit_fields:cfg.decimal_fields;
    drag={...h,sx:x,ol:fs[h.i][0],or:fs[h.i][1]};
  }else{
    drag={...h,sy:y,op:h.w==='top'?cfg.crop_top_pct:cfg.crop_bottom_pct};
  }
  draw();
};
cv.onmousemove=e=>{
  const r=cv.getBoundingClientRect(),x=e.clientX-r.left,y=e.clientY-r.top;
  if(!drag){
    const h=hitTest(x,y);
    cv.style.cursor=h?(h.type==='crop'?'ns-resize':h.e==='b'?'move':'ew-resize'):'default';
    return;
  }
  if(drag.type==='crop'){
    const np=drag.op+yp(y-drag.sy);
    if(drag.w==='top')cfg.crop_top_pct=Math.min(np,cfg.crop_bottom_pct-2);
    else cfg.crop_bottom_pct=Math.max(np,cfg.crop_top_pct+2);
  }else{
    const fs=drag.g==='digit'?cfg.digit_fields:cfg.decimal_fields;
    const dx=xp(x-drag.sx);
    if(drag.e==='l')fs[drag.i][0]=Math.max(0,Math.min(drag.ol+dx,fs[drag.i][1]-1));
    else if(drag.e==='r')fs[drag.i][1]=Math.max(fs[drag.i][0]+1,Math.min(100,drag.or+dx));
    else{const w=drag.or-drag.ol,nl=Math.max(0,Math.min(100-w,drag.ol+dx));fs[drag.i][0]=nl;fs[drag.i][1]=nl+w;}
  }
  draw();
};
cv.onmouseup=cv.onmouseleave=()=>{drag=null;};

function addField(group){
  const fs=group==='digit'?cfg.digit_fields:cfg.decimal_fields;
  const lr=fs.length?fs[fs.length-1][1]:10;
  fs.push([lr+2,lr+12]);
  sel={g:group,i:fs.length-1};document.getElementById('del').disabled=false;
  draw();
}
function delSel(){
  if(!sel)return;
  (sel.g==='digit'?cfg.digit_fields:cfg.decimal_fields).splice(sel.i,1);
  sel=null;document.getElementById('del').disabled=true;draw();
}
async function saveCfg(){
  const s=document.getElementById('st');
  s.style.color='#ff8';s.textContent='Saving\u2026';
  try{
    const r=await fetch('/config',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(cfg)});
    if(!r.ok)throw new Error(await r.text());
    s.style.color='#4f4';s.textContent='\u2713 Saved \u2014 next image will use new fields';
    setTimeout(()=>s.textContent='',4000);
  }catch(e){s.style.color='#f44';s.textContent='\u2717 '+e.message;}
}
init();
})();
</script>
</body></html>
"""

# ── HTTP server ──────────────────────────────────────────────────────────────
class ImageHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        p = self.path.split('?')[0]  # strip query string
        if p == '/image/raw':
            data = annotate_raw_with_crop(last_image_raw) if last_image_raw else None
            self._serve_image(data)
        elif p in ('/image', '/image/processed'):
            self._serve_image(last_image_processed)
        elif p == '/image/annotated':
            self._serve_image(last_image_annotated)
        elif p == '/image/reference':
            self._serve_image(last_image_raw)   # undecorated, for the configure canvas
        elif p == '/config':
            self._serve_config_json()
        elif p == '/configure':
            self._serve_configure()
        elif p == '/':
            self._serve_status()
        else:
            self.send_error(404)

    def do_POST(self):
        if self.path == '/config':
            length = int(self.headers.get('Content-Length', 0))
            body   = self.rfile.read(length)
            try:
                save_config(json.loads(body))
                self._json({'ok': True})
            except Exception as e:
                self.send_response(400)
                self.end_headers()
                self.wfile.write(str(e).encode())
        else:
            self.send_error(405)

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

    def _serve_config_json(self):
        data = {
            'crop_top_pct':    CROP_TOP_PCT,
            'crop_bottom_pct': CROP_BOTTOM_PCT,
            'digit_fields':    [list(f) for f in DIGIT_FIELDS],
            'decimal_fields':  [list(f) for f in DECIMAL_FIELDS],
        }
        self._json(data)

    def _serve_configure(self):
        body = _CONFIGURE_HTML.replace('METER_NAME_PLACEHOLDER', METER_NAME).encode()
        self.send_response(200)
        self.send_header('Content-Type', 'text/html; charset=utf-8')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _serve_status(self):
        body = f"""<!DOCTYPE html><html><body style="font:14px monospace;background:#111;color:#ddd;padding:16px">
<h2>{METER_NAME}</h2>
<p>Last reading: <b>{last_reading}</b> {METER_UNIT if last_reading is not None else ''}</p>
<p>
  <a href="/configure" style="color:#4af">&#9881; Configure field positions</a><br><br>
  <a href="/image/raw">Raw image (annotated)</a><br>
  <a href="/image/processed">Processed digit crops</a><br>
  <a href="/image/annotated">Annotated detections</a>
</p>
</body></html>""".encode()
        self.send_response(200)
        self.send_header('Content-Type', 'text/html; charset=utf-8')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, data):
        body = json.dumps(data).encode()
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        pass


def start_http_server():
    server = HTTPServer(('0.0.0.0', HTTP_PORT), ImageHandler)
    print(f"✓ HTTP server running on port {HTTP_PORT}")
    server.serve_forever()

# ── Meter reading ─────────────────────────────────────────────────────────────
def extract_meter_reading(image_data):
    global last_image_raw, last_image_processed, last_image_annotated, last_reading
    try:
        last_image_raw = image_data
        image  = Image.open(io.BytesIO(image_data))
        h, w   = image.height, image.width
        top    = int(h * CROP_TOP_PCT    / 100)
        bottom = int(h * CROP_BOTTOM_PCT / 100)
        strip_h = bottom - top

        def crop_fields(field_defs):
            return [_preprocess(image.crop((int(w * l / 100), top, int(w * r / 100), bottom)))
                    for l, r in field_defs]

        int_imgs = crop_fields(DIGIT_FIELDS)
        dec_imgs = crop_fields(DECIMAL_FIELDS)

        # Composite processed image
        INNER_GAP, OUTER_GAP = 4, 14
        rendered = []
        x = 0
        for i, img in enumerate(int_imgs):
            if i > 0: x += INNER_GAP
            rendered.append((img, x, False, i)); x += img.width
        if dec_imgs:
            x += OUTER_GAP
            for i, img in enumerate(dec_imgs):
                if i > 0: x += INNER_GAP
                rendered.append((img, x, True, i)); x += img.width
        combined = Image.new('RGB', (x, strip_h), color=(140, 140, 140))
        for img, x_off, _, _ in rendered:
            combined.paste(img, (x_off, 0))
        buf = io.BytesIO(); combined.save(buf, format='JPEG')
        last_image_processed = buf.getvalue()

        # OCR
        results_per_field = []
        for i, img in enumerate(int_imgs):
            digit, dets = _ocr_digit(img, f"INT{i+1}")
            results_per_field.append((digit, dets, False, i))
        for i, img in enumerate(dec_imgs):
            digit, dets = _ocr_digit(img, f"DEC{i+1}")
            results_per_field.append((digit, dets, True, i))

        # Annotated image
        annotated = combined.copy()
        draw = ImageDraw.Draw(annotated)
        for (digit, dets, is_dec, _), (_, x_off, _, _) in zip(results_per_field, rendered):
            color = 'blue' if is_dec else 'red'
            for bbox, txt, conf in dets:
                x0, y0 = int(bbox[0][0]) + x_off, int(bbox[0][1])
                x1, y1 = int(bbox[2][0]) + x_off, int(bbox[2][1])
                draw.rectangle([(x0, y0), (x1, y1)], outline=color, width=2)
            draw.text((x_off + 2, 2), digit if digit else '?', fill=color)
        buf = io.BytesIO(); annotated.save(buf, format='JPEG')
        last_image_annotated = buf.getvalue()

        # Assemble
        int_digits = ''.join(d for d, _, is_dec, _ in results_per_field if not is_dec)
        dec_digits = ''.join(d for d, _, is_dec, _ in results_per_field if is_dec)
        clean_text = int_digits + dec_digits
        print(f"  INT='{int_digits}' DEC='{dec_digits}' Combined='{clean_text}'")

        expected_len = len(DIGIT_FIELDS) + len(DECIMAL_FIELDS)
        if len(clean_text) < expected_len:
            print(f"  ✗ Only {len(clean_text)}/{expected_len} digits read — likely mid-transition")
            return None

        n_dec   = len(DECIMAL_FIELDS)
        raw_int = int(clean_text[:7])
        reading = round(raw_int / (10 ** n_dec), n_dec)

        if last_reading is not None:
            if reading < last_reading - 1.0:
                print(f"  ⚠ Reading decreased suspiciously: {last_reading} -> {reading}, skipping")
                return None
            elif reading > last_reading + 10.0:
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

# ── MQTT callbacks ───────────────────────────────────────────────────────────
def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print(f"✓ Connected to MQTT broker at {MQTT_BROKER}:{MQTT_PORT}")
        client.subscribe(IMAGE_TOPIC)
        print(f"✓ Subscribed to {IMAGE_TOPIC}")
        discovery_config = {
            "name": METER_NAME, "unique_id": METER_ID,
            "state_topic": STATE_TOPIC, "unit_of_measurement": METER_UNIT,
            "device_class": DEVICE_CLASS, "state_class": "total_increasing",
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
    if reading is not None:
        client.publish(STATE_TOPIC, str(reading), retain=True)
        detail = {"reading": reading, "timestamp": time.time(), "timestamp_human": timestamp}
        client.publish(READING_TOPIC, json.dumps(detail))
        print(f"  ✓ Published to HA: {reading} {METER_UNIT}")
    else:
        print(f"  ✗ No valid reading extracted")

# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    load_config()  # override env-var defaults with saved config if present

    print("=" * 60)
    print(f"Meter OCR Service Starting - {METER_NAME}")
    print("=" * 60)
    print(f"MQTT: {MQTT_BROKER}:{MQTT_PORT}  |  Unit: {METER_UNIT} ({DEVICE_CLASS})")
    print(f"Image topic: {IMAGE_TOPIC}")
    print(f"Int fields:  {DIGIT_FIELDS}")
    print(f"Dec fields:  {DECIMAL_FIELDS}")
    print("=" * 60)

    threading.Thread(target=start_http_server, daemon=True).start()

    client = mqtt.Client()
    client.on_connect = on_connect
    client.on_message = on_message
    if MQTT_USER:
        client.username_pw_set(MQTT_USER, MQTT_PASS)

    while True:
        try:
            client.connect(MQTT_BROKER, MQTT_PORT, 60)
            break
        except Exception as e:
            print(f"✗ MQTT connection failed: {e} — retrying in 5s")
            time.sleep(5)

    print("✓ Service running. Waiting for images…\n")
    client.loop_forever()

if __name__ == "__main__":
    main()
