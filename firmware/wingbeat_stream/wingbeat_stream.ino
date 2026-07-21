// Streaming inference: host pumps test windows over USB serial; ESP32 runs
// each one and replies with logits + argmax. Lets us measure full-test-set
// accuracy on real silicon instead of the 6-sample bundled self-check.
//
// Wire protocol (binary, self-synchronizing via 4-byte magic on each frame):
//   host -> chip : "INFR" + 4096 bytes (1024 little-endian float32)
//   chip -> host : "RSLT" + 3x float32 (logits, little-endian) + 1 byte argmax
//
// Chip's input loop scans for "INFR" byte-by-byte, so it recovers
// gracefully from a desync (stale bytes in buffer after a failed run).
//
// On boot the chip prints a one-line ASCII banner ending with "READY\n"
// so the host knows to start. Baud is 460800 — ~6x faster than 115200,
// which matters: at 115200 baud the payload alone takes 285 ms / sample
// and the full test set would take 20+ min.
//
// Upload via:
//   arduino-cli compile --fqbn esp32:esp32:esp32cam firmware/wingbeat_stream
//   arduino-cli upload  --fqbn esp32:esp32:esp32cam --port /dev/cu.wchusbserial1120 \
//                       firmware/wingbeat_stream

#include "../wingbeat_inference/model_weights.h"
#include <math.h>

static const uint32_t SERIAL_BAUD = 460800;
static const uint32_t READ_TIMEOUT_MS = 30000;  // generous for first sample

// Working buffers — same layout as wingbeat_inference.ino.
static float stream_input[MODEL_INPUT_LEN];      // 4 KB
static float buf_a[16 * 512];                    // 32 KB
static float buf_b[24 * 256];                    // 24 KB

// --- math primitives (duplicated from wingbeat_inference.ino — small + self-contained)

static inline float relu(float x) { return x > 0.0f ? x : 0.0f; }

static void conv1d_same(const float* in, int in_ch, int in_len,
                        const float* weight, const float* bias,
                        int out_ch, int kernel, float* out) {
  const int pad = kernel / 2;
  for (int oc = 0; oc < out_ch; ++oc) {
    const float b = bias[oc];
    for (int t = 0; t < in_len; ++t) {
      float acc = b;
      for (int ic = 0; ic < in_ch; ++ic) {
        const float* in_row = in + ic * in_len;
        const float* w_row = weight + ((oc * in_ch) + ic) * kernel;
        for (int k = 0; k < kernel; ++k) {
          int in_t = t - pad + k;
          if (in_t >= 0 && in_t < in_len) acc += in_row[in_t] * w_row[k];
        }
      }
      out[oc * in_len + t] = acc;
    }
  }
}

static void relu_inplace(float* x, int n) {
  for (int i = 0; i < n; ++i) x[i] = relu(x[i]);
}

static void maxpool1d_2(const float* in, int channels, int in_len, float* out) {
  const int out_len = in_len / 2;
  for (int c = 0; c < channels; ++c) {
    const float* in_row = in + c * in_len;
    float* out_row = out + c * out_len;
    for (int t = 0; t < out_len; ++t) {
      float a = in_row[2 * t], b = in_row[2 * t + 1];
      out_row[t] = a > b ? a : b;
    }
  }
}

static void global_avgpool1d(const float* in, int channels, int in_len, float* out) {
  for (int c = 0; c < channels; ++c) {
    const float* row = in + c * in_len;
    double sum = 0.0;
    for (int t = 0; t < in_len; ++t) sum += row[t];
    out[c] = (float)(sum / in_len);
  }
}

static void linear(const float* x, int in_dim, const float* w, const float* b,
                   int out_dim, float* y) {
  for (int o = 0; o < out_dim; ++o) {
    float acc = b[o];
    const float* w_row = w + o * in_dim;
    for (int i = 0; i < in_dim; ++i) acc += w_row[i] * x[i];
    y[o] = acc;
  }
}

static uint8_t argmaxN(const float* x, int n) {
  uint8_t best = 0;
  for (int i = 1; i < n; ++i) if (x[i] > x[best]) best = i;
  return best;
}

static void forward(const float* input, float* out_logits) {
  conv1d_same(input, 1, MODEL_INPUT_LEN,
              BLOCK0_WEIGHT, BLOCK0_BIAS,
              MODEL_BLOCK0_OUT_CH, MODEL_BLOCK0_KERNEL, buf_a);
  relu_inplace(buf_a, MODEL_BLOCK0_OUT_CH * MODEL_INPUT_LEN);
  maxpool1d_2(buf_a, MODEL_BLOCK0_OUT_CH, MODEL_INPUT_LEN, buf_b);

  conv1d_same(buf_b, MODEL_BLOCK1_IN_CH, MODEL_INPUT_LEN / 2,
              BLOCK1_WEIGHT, BLOCK1_BIAS,
              MODEL_BLOCK1_OUT_CH, MODEL_BLOCK1_KERNEL, buf_a);
  relu_inplace(buf_a, MODEL_BLOCK1_OUT_CH * (MODEL_INPUT_LEN / 2));
  maxpool1d_2(buf_a, MODEL_BLOCK1_OUT_CH, MODEL_INPUT_LEN / 2, buf_b);

  conv1d_same(buf_b, MODEL_BLOCK2_IN_CH, MODEL_INPUT_LEN / 4,
              BLOCK2_WEIGHT, BLOCK2_BIAS,
              MODEL_BLOCK2_OUT_CH, MODEL_BLOCK2_KERNEL, buf_a);
  relu_inplace(buf_a, MODEL_BLOCK2_OUT_CH * (MODEL_INPUT_LEN / 4));
  maxpool1d_2(buf_a, MODEL_BLOCK2_OUT_CH, MODEL_INPUT_LEN / 4, buf_b);

  float feat[MODEL_FC1_IN];
  global_avgpool1d(buf_b, MODEL_BLOCK2_OUT_CH, MODEL_INPUT_LEN / 8, feat);

  float h[MODEL_FC1_OUT];
  linear(feat, MODEL_FC1_IN, FC1_WEIGHT, FC1_BIAS, MODEL_FC1_OUT, h);
  relu_inplace(h, MODEL_FC1_OUT);
  linear(h, MODEL_FC2_IN, FC2_WEIGHT, FC2_BIAS, MODEL_FC2_OUT, out_logits);
}

// --- serial helpers

// Read exactly `n` bytes from Serial, blocking up to `timeout_ms` total.
// Returns true on success, false on timeout.
static bool read_exact(uint8_t* dst, size_t n, uint32_t timeout_ms) {
  size_t got = 0;
  uint32_t t0 = millis();
  while (got < n) {
    if (millis() - t0 > timeout_ms) return false;
    int avail = Serial.available();
    if (avail <= 0) { delay(1); continue; }
    int r = Serial.readBytes(dst + got, min((size_t)avail, n - got));
    if (r > 0) got += r;
  }
  return true;
}

void setup() {
  Serial.begin(SERIAL_BAUD);
  delay(500);
  Serial.println();
  Serial.printf("ESP32 wingbeat-stream v1 baud=%u input=%d classes=%d\n",
                SERIAL_BAUD, MODEL_INPUT_LEN, MODEL_N_CLASSES);
  Serial.printf("free_heap=%u\n", ESP.getFreeHeap());
  Serial.println("READY");
  Serial.flush();
}

// Scan serial byte-by-byte until the 4-byte "INFR" magic is seen.
// Returns false on timeout (and the caller just loops).
static bool wait_for_magic() {
  static const char MAGIC[4] = {'I', 'N', 'F', 'R'};
  int matched = 0;
  uint32_t t0 = millis();
  while (matched < 4) {
    if (millis() - t0 > READ_TIMEOUT_MS) return false;
    int b = Serial.read();
    if (b < 0) { delay(1); continue; }
    if ((uint8_t)b == (uint8_t)MAGIC[matched]) matched++;
    else if ((uint8_t)b == (uint8_t)MAGIC[0]) matched = 1;
    else matched = 0;
  }
  return true;
}

void loop() {
  if (!wait_for_magic()) return;
  if (!read_exact((uint8_t*)stream_input,
                  MODEL_INPUT_LEN * sizeof(float),
                  READ_TIMEOUT_MS)) {
    return;
  }

  float logits[MODEL_N_CLASSES];
  forward(stream_input, logits);
  uint8_t pred = argmaxN(logits, MODEL_N_CLASSES);

  // Reply: "RSLT" + 12 bytes logits + 1 byte pred = 17 bytes total.
  uint8_t out[4 + MODEL_N_CLASSES * 4 + 1];
  out[0] = 'R'; out[1] = 'S'; out[2] = 'L'; out[3] = 'T';
  memcpy(out + 4, logits, MODEL_N_CLASSES * sizeof(float));
  out[4 + MODEL_N_CLASSES * 4] = pred;
  Serial.write(out, sizeof(out));
  Serial.flush();
}
