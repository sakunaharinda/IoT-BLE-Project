
#include <WiFi.h>
#include <PubSubClient.h>
#include <Arduino_JSON.h>
#include <ArduinoJson.h>
#include <Arduino.h>
#include <BLEDevice.h>
#include <BLEUtils.h>
#include <BLEScan.h>
#include <BLEAdvertisedDevice.h>


// Update these with values suitable for your network.
const char* ssid = "Avatar_Roshan";
const char* password = "roshan@123";
const char* mqtt_server = "mqtt.eclipse.org";
const char ledPin = 2;
bool sent = LOW;
char msg[1000];
#define mqtt_port 1883
#define MQTT_USER "roshan"
#define MQTT_PASSWORD "avatar"
#define MQTT_SERIAL_PUBLISH_CH "esp32/tx"
#define MQTT_SERIAL_RECEIVER_CH "esp32/rx"

JSONVar Beacon;
WiFiClient wifiClient;
PubSubClient client(wifiClient);

int scanTime = 5; //In seconds
BLEScan* pBLEScan;

void setup_wifi() {
    delay(100);
    // We start by connecting to a WiFi network
    Serial.println();
    Serial.print("Connecting to ");
    Serial.println(ssid);
    WiFi.begin(ssid, password);
    while (WiFi.status() != WL_CONNECTED) {
      delay(500);
      Serial.print(".");
    }
    randomSeed(micros());
    Serial.println();
    Serial.println("WiFi connected");
    Serial.print("IP address: ");
    Serial.println(WiFi.localIP());
}

void reconnect() {
  // Loop until we're reconnected
  while (!client.connected()) {
    Serial.print("Attempting MQTT connection...");
    String clientId = "ESP32Client-";
    clientId += String(random(0xffff), HEX);
    // Attempt to connect
    if (client.connect(clientId.c_str(),MQTT_USER,MQTT_PASSWORD)) {
      Serial.println("connected");
      client.subscribe(MQTT_SERIAL_RECEIVER_CH);
    } else {
      Serial.print("failed, rc=");
      Serial.print(client.state());
      Serial.println(" try again in 5 seconds");
      // Wait 5 seconds before retrying
      delay(5000);
    }
  }
}

void callback(char* topic, byte *payload, unsigned int length) {
    Serial.println("-------new message from broker-----");
    Serial.print("channel:");
    Serial.println(topic);
    Serial.print("data:");  
    Serial.write(payload, length);
    Serial.println();
}

void scan_ble(){
  String dump = "{";
  BLEScanResults foundDevices = pBLEScan->start(scanTime, false);
  int count = foundDevices.getCount();
  Serial.print("Devices found: ");
  Serial.println(count);
  for (int i=0;i<count;i++){
    BLEAdvertisedDevice d = foundDevices.getDevice(i);
    if (i!= 0){
        dump+=",";
    }
    dump+="\"";
    dump += d.getAddress().toString().c_str();
    dump+="\"";
    dump+=":";
    dump+=d.getRSSI();
    delay(10);
  }

  dump+="}";
  dump.toCharArray(msg,500000);
  pBLEScan->clearResults();   // delete results fromBLEScan buffer to release memory
  delay(200);  
  
  Serial.println(msg);
  sent = client.publish(MQTT_SERIAL_PUBLISH_CH, msg);
  if (sent) {
    Serial.print("Published: ");
    Serial.println(msg);
  } else {
    Serial.print("Failed: ");
    Serial.println(msg);
  }
}


void setup() {
  pinMode(ledPin, OUTPUT);
  digitalWrite(ledPin, HIGH);
  Serial.begin(115200);
  Serial.setTimeout(500);
  
  setup_wifi();
  client.setServer(mqtt_server, mqtt_port);
  client.setCallback(callback);

  // BLE Initialising
  Serial.println("Scanning...");
  BLEDevice::init("");
  pBLEScan = BLEDevice::getScan(); //create new scan
  //pBLEScan->setAdvertisedDeviceCallbacks(new MyAdvertisedDeviceCallbacks());
  pBLEScan->setActiveScan(true); //active scan uses more power, but get results faster
  pBLEScan->setInterval(100);
  pBLEScan->setWindow(99);  // less or equal setInterval value
}


void loop() {
  digitalWrite(ledPin, LOW);
  delay(100);
  digitalWrite(ledPin, HIGH);
  
  if (!client.connected()) {
    reconnect();
  }
  
  client.loop();
  scan_ble();

  delay(2000);
 }
