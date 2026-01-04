#include "esp_camera.h"
#include <WiFi.h>
#include <PubSubClient.h>
#include "credentials.h"

const int mqtt_port = 1883;
const char* mqtt_topic = "home/meter/electric/image";
const char* device_name = "Electric Meter";

// Camera pins for AI-Thinker ESP32-CAM
#define PWDN_GPIO_NUM     32
#define RESET_GPIO_NUM    -1
#define XCLK_GPIO_NUM      0
#define SIOD_GPIO_NUM     26
#define SIOC_GPIO_NUM     27
#define Y9_GPIO_NUM       35
#define Y8_GPIO_NUM       34
#define Y7_GPIO_NUM       39
#define Y6_GPIO_NUM       36
#define Y5_GPIO_NUM       21
#define Y4_GPIO_NUM       19
#define Y3_GPIO_NUM       18
#define Y2_GPIO_NUM        5
#define VSYNC_GPIO_NUM    25
#define HREF_GPIO_NUM     23
#define PCLK_GPIO_NUM     22

// Capture interval (milliseconds)
const unsigned long captureInterval = 300000;  // 5 minutes (change as needed)
// 60000 = 1 min, 300000 = 5 min, 600000 = 10 min, 1800000 = 30 min
unsigned long lastCapture = 0;

// Built-in LED for status
#define LED_BUILTIN 33

WiFiClient espClient;
PubSubClient mqtt(espClient);

void setup() {
  Serial.begin(115200);
  Serial.println("\n\n=================================");
  Serial.print("ESP32-CAM Meter Reader: ");
  Serial.println(device_name);
  Serial.println("=================================");
  
  // Setup LED
  pinMode(LED_BUILTIN, OUTPUT);
  digitalWrite(LED_BUILTIN, LOW);
  
  // Initialize camera
  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;
  config.pin_d0 = Y2_GPIO_NUM;
  config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM;
  config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM;
  config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM;
  config.pin_d7 = Y9_GPIO_NUM;
  config.pin_xclk = XCLK_GPIO_NUM;
  config.pin_pclk = PCLK_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM;
  config.pin_href = HREF_GPIO_NUM;
  config.pin_sscb_sda = SIOD_GPIO_NUM;
  config.pin_sscb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;
  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_JPEG;
  
  // High quality settings for OCR
  config.frame_size = FRAMESIZE_SVGA;  // 800x600
  config.jpeg_quality = 10;
  config.fb_count = 1;
  
  // Initialize camera
  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("Camera init failed with error 0x%x", err);
    ESP.restart();
  }
  
  // Connect to WiFi
  WiFi.begin(ssid, password);
  Serial.print("Connecting to WiFi");
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("\nWiFi connected!");
  Serial.print("IP: ");
  Serial.println(WiFi.localIP());
  
  // Setup MQTT
  mqtt.setServer(mqtt_server, mqtt_port);
  mqtt.setBufferSize(32768);  // Increase buffer for images
}

void reconnectMQTT() {
  while (!mqtt.connected()) {
    Serial.print("Connecting to MQTT...");
    String clientId = "ESP32CAM-Electric-" + String(random(0xffff), HEX);
    
    if (mqtt.connect(clientId.c_str(), mqtt_user, mqtt_password)) {
      Serial.println("connected!");
    } else {
      Serial.print("failed, rc=");
      Serial.print(mqtt.state());
      Serial.println(" retrying in 5 seconds");
      delay(5000);
    }
  }
}

void captureAndSend() {
  Serial.println("\n>>> Capturing image...");
  digitalWrite(LED_BUILTIN, HIGH);  // LED on during capture
  
  camera_fb_t * fb = esp_camera_fb_get();
  if (!fb) {
    Serial.println("✗ Camera capture failed");
    digitalWrite(LED_BUILTIN, LOW);
    return;
  }
  
  Serial.printf("✓ Image captured: %d bytes\n", fb->len);
  
  // Send image via MQTT
  if (!mqtt.connected()) {
    reconnectMQTT();
  }
  
  // Send as binary
  bool success = mqtt.publish(mqtt_topic, fb->buf, fb->len);
  
  if (success) {
    Serial.println("✓ Image sent successfully to MQTT!");
  } else {
    Serial.println("✗ Failed to send image");
  }
  
  esp_camera_fb_return(fb);
  digitalWrite(LED_BUILTIN, LOW);  // LED off after capture
}

void loop() {
  if (!mqtt.connected()) {
    reconnectMQTT();
  }
  mqtt.loop();
  
  // Capture at interval
  unsigned long currentMillis = millis();
  if (currentMillis - lastCapture >= captureInterval) {
    lastCapture = currentMillis;
    captureAndSend();
  }
  
  delay(100);
}
