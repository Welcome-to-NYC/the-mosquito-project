// First test sketch — confirms the ESP32-CAM board boots, runs our code,
// and emits Serial output through the ESP32-CAM-MB USB bridge.
//
// Upload via:
//   arduino-cli compile --fqbn esp32:esp32:esp32cam firmware/hello_world
//   arduino-cli upload  --fqbn esp32:esp32:esp32cam --port /dev/cu.usbserial-1120 firmware/hello_world
//
// Then read with:
//   screen /dev/cu.usbserial-1120 115200

unsigned long boot_ms = 0;
int counter = 0;

void setup() {
  Serial.begin(115200);
  delay(500);
  Serial.println();
  Serial.println("=====================================");
  Serial.println("  ESP32-CAM hello-world boot");
  Serial.println("  built " __DATE__ " " __TIME__);
  Serial.println("=====================================");
  boot_ms = millis();
}

void loop() {
  unsigned long uptime_s = (millis() - boot_ms) / 1000;
  Serial.printf("alive  counter=%d  uptime=%lus  free_heap=%u bytes\n",
                ++counter, uptime_s, ESP.getFreeHeap());
  delay(2000);
}
