#include <Arduino.h>
#include <M5StickCPlus2.h>
#include <WiFi.h>
#include <WiFiUdp.h>

const char *ssid = "Redmi Note 11";
const char *password = "passpass123";
const char *hostIp = "10.119.193.188";
const int udpPort = 1234;
const int audioUdpPort = 1235; // Audio packets on separate port

WiFiUDP udp;
WiFiUDP audioUdp;

// Audio double-buffer to prevent race conditions
// Reduced to 512 samples per packet to avoid UDP fragmentation (6 + 512*2 = 1030 bytes < 1500 MTU)
static constexpr size_t AUDIO_BUFFER_SIZE = 512;
int16_t audioBuffer[2][AUDIO_BUFFER_SIZE] = {{0}, {0}}; // Double buffer, zero-initialized
int currentWriteBuffer = 0;                             // Which buffer mic.record() writes to
uint32_t audioPacketSeq = 0;
unsigned long lastAudioRead = 0;

// Combined packet buffer to ensure atomic UDP transmission
// Packet format: [4B seq][2B count][8B timestamp_us][audio]
static constexpr size_t AUDIO_PACKET_SIZE = 14 + (AUDIO_BUFFER_SIZE * 2); // header + timestamp + audio
uint8_t audioPacketBuffer[AUDIO_PACKET_SIZE];

unsigned long packetSequence = 0; // To track dropped packets

void setup(void)
{
  auto cfg = M5.config();
  StickCP2.begin(cfg);

  // Connect to WiFi
  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED)
  {
    delay(500);
    StickCP2.Display.print(".");
  }
  StickCP2.Display.println("\nWiFi Connected!");

  // Disable WiFi power saving for real-time streaming (reduces latency spikes)
  WiFi.setSleep(false);

  // Initialize UDP
  udp.begin(udpPort);
  audioUdp.begin(audioUdpPort);

  // Initialize microphone: 16 kHz, mono, 16-bit
  auto micCfg = StickCP2.Mic.config();
  micCfg.sample_rate = 16000;
  StickCP2.Mic.config(micCfg);
  StickCP2.Mic.begin();
  StickCP2.Display.println("Mic initialized");
}

void loop()
{
  // Check WiFi connection and attempt reconnect if needed
  if (WiFi.status() != WL_CONNECTED)
  {
    StickCP2.Display.println("Wifi Disconnected!");
    WiFi.reconnect();
    delay(100); // Brief delay before retrying
    if (WiFi.status() == WL_CONNECTED)
    {
      StickCP2.Display.println("Wifi Connected again!");
    }
    return; // Skip sending until reconnected
  }

  // Check if new data is actually ready
  if (StickCP2.Imu.update())
  {
    auto data = StickCP2.Imu.getImuData();

    unsigned long currentMicros = micros(); // Grab the exact timestamp (microsecond precision)

    // Format: timestamp, sequence, ax, ay, az, gx, gy, gz
    char buf[128];
    snprintf(buf, sizeof(buf), "%lu,%lu,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f",
             currentMicros,
             packetSequence++,
             data.accel.x, data.accel.y, data.accel.z,
             data.gyro.x, data.gyro.y, data.gyro.z);

    // Send UDP packet
    udp.beginPacket(hostIp, udpPort);
    udp.write((uint8_t *)buf, strlen(buf));
    udp.endPacket();
  }

  // Capture audio with synchronization to prevent race conditions
  static int lastFilledBuffer = -1;

  unsigned long now = micros();
  if (now - lastAudioRead >= 32000) // 512 samples at 16kHz = 32ms = 32000 microseconds
  {
    lastAudioRead = now;

    // Record into current write buffer
    if (StickCP2.Mic.record(audioBuffer[currentWriteBuffer], AUDIO_BUFFER_SIZE))
    {
      // Send the previously filled buffer (skip first iteration)
      if (lastFilledBuffer != -1)
      {
        unsigned long audioMicros = micros();

        // Construct complete packet in buffer: 4-byte seq + 2-byte count + 8-byte timestamp + audio
        *(uint32_t *)&audioPacketBuffer[0] = audioPacketSeq++;
        *(uint16_t *)&audioPacketBuffer[4] = (uint16_t)AUDIO_BUFFER_SIZE;
        *(uint64_t *)&audioPacketBuffer[6] = audioMicros; // 8-byte timestamp in microseconds
        memcpy(&audioPacketBuffer[14], audioBuffer[lastFilledBuffer], AUDIO_BUFFER_SIZE * 2);

        // Send as single atomic UDP packet
        audioUdp.beginPacket(hostIp, audioUdpPort);
        audioUdp.write(audioPacketBuffer, AUDIO_PACKET_SIZE);
        audioUdp.endPacket();
      }

      // Mark current buffer as "last filled" and swap for next iteration
      lastFilledBuffer = currentWriteBuffer;
      currentWriteBuffer = 1 - currentWriteBuffer;
    }
  }
}