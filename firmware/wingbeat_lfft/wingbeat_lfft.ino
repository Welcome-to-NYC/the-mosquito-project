// LearnableFFT mosquito detector on ESP32 — streaming inference.
//
// Pipeline (matches scripts/improve_audio_detector_1d.py LFFTDetector):
//   input (1024) -> LearnableFFT (fused conv+magnitude) -> (48, 256)
//                -> Conv1d(48->32,k5,same)+ReLU+MaxPool2 -> (32,128)
//                -> Conv1d(32->32,k3,same)+ReLU+MaxPool2 -> (32,64)
//                -> GAP -> (32) -> Linear(32->2) -> logits
//
// Wire protocol (self-syncing, same as wingbeat_stream):
//   host -> chip : "INFR" + 1024 float32 (4096 bytes)
//   chip -> host : "RSLT" + MODEL_N_CLASSES float32 + 1 byte argmax
//
// Upload:
//   arduino-cli compile --fqbn esp32:esp32:esp32cam firmware/wingbeat_lfft
//   arduino-cli upload  --fqbn esp32:esp32:esp32cam --port /dev/cu.wchusbserial1120 firmware/wingbeat_lfft

#include "model_lfft.h"
#include <math.h>

static const uint32_t SERIAL_BAUD = 460800;
static const uint32_t READ_TIMEOUT_MS = 30000;

// Buffers: g_mag holds the FFT magnitude (48x256), then is reused as the
// pooling output once the FFT stage is consumed — saves 32 KB of DRAM.
static float g_input[MODEL_INPUT_LEN];        // 1024 (4 KB)
static float g_mag[FFT_N_FILT * FFT_T];       // 48 x 256 = 12288 (48 KB)
static float g_a[B1_OUT * FFT_T];             // 32 x 256 = 8192 (32 KB)

static inline float relu(float x) { return x > 0.0f ? x : 0.0f; }

// LearnableFFT front-end with fused magnitude.
// For each filter f and time t: cos/sin conv at stride 4, pad 64, then |.|.
// FFT_WEIGHT layout: (2*F, K), rows 2f = cos, 2f+1 = sin.
static void learnable_fft(const float* in, float* mag) {
  for (int f = 0; f < FFT_N_FILT; ++f) {
    const float* w_cos = FFT_WEIGHT + (2 * f) * FFT_KERNEL;
    const float* w_sin = FFT_WEIGHT + (2 * f + 1) * FFT_KERNEL;
    for (int t = 0; t < FFT_T; ++t) {
      const int in_start = t * FFT_STRIDE - FFT_PAD;
      float acc_cos = 0.0f, acc_sin = 0.0f;
      for (int k = 0; k < FFT_KERNEL; ++k) {
        const int idx = in_start + k;
        if (idx >= 0 && idx < MODEL_INPUT_LEN) {
          const float v = in[idx];
          acc_cos += v * w_cos[k];
          acc_sin += v * w_sin[k];
        }
      }
      mag[f * FFT_T + t] = sqrtf(acc_cos * acc_cos + acc_sin * acc_sin + FFT_EPS);
    }
  }
}

// Conv1d, stride 1, symmetric "same" padding (pad = k/2).
// in: (in_ch, len), weight: (out_ch, in_ch, k), bias: (out_ch). out: (out_ch, len).
static void conv1d_same(const float* in, int in_ch, int len,
                        const float* weight, const float* bias,
                        int out_ch, int kernel, float* out) {
  const int pad = kernel / 2;
  for (int oc = 0; oc < out_ch; ++oc) {
    const float b = bias[oc];
    for (int t = 0; t < len; ++t) {
      float acc = b;
      for (int ic = 0; ic < in_ch; ++ic) {
        const float* in_row = in + ic * len;
        const float* w_row = weight + ((oc * in_ch) + ic) * kernel;
        for (int k = 0; k < kernel; ++k) {
          int it = t - pad + k;
          if (it >= 0 && it < len) acc += in_row[it] * w_row[k];
        }
      }
      out[oc * len + t] = acc;
    }
  }
}

static void relu_inplace(float* x, int n) { for (int i = 0; i < n; ++i) x[i] = relu(x[i]); }

static void maxpool2(const float* in, int ch, int len, float* out) {
  const int ol = len / 2;
  for (int c = 0; c < ch; ++c)
    for (int t = 0; t < ol; ++t) {
      float a = in[c * len + 2 * t], b = in[c * len + 2 * t + 1];
      out[c * ol + t] = a > b ? a : b;
    }
}

static void gap(const float* in, int ch, int len, float* out) {
  for (int c = 0; c < ch; ++c) {
    double s = 0; for (int t = 0; t < len; ++t) s += in[c * len + t];
    out[c] = (float)(s / len);
  }
}

static void linear(const float* x, int in_dim, const float* w, const float* b,
                   int out_dim, float* y) {
  for (int o = 0; o < out_dim; ++o) {
    float acc = b[o]; const float* wr = w + o * in_dim;
    for (int i = 0; i < in_dim; ++i) acc += wr[i] * x[i];
    y[o] = acc;
  }
}

static uint8_t argmaxN(const float* x, int n) {
  uint8_t best = 0; for (int i = 1; i < n; ++i) if (x[i] > x[best]) best = i; return best;
}

static void forward(const float* input, float* logits) {
  learnable_fft(input, g_mag);                 // g_mag: (48, 256)
  conv1d_same(g_mag, B1_IN, FFT_T, B1_WEIGHT, B1_BIAS, B1_OUT, B1_K, g_a);  // g_a: (32,256)
  relu_inplace(g_a, B1_OUT * FFT_T);
  maxpool2(g_a, B1_OUT, FFT_T, g_mag);         // reuse g_mag: (32,128)
  conv1d_same(g_mag, B2_IN, FFT_T / 2, B2_WEIGHT, B2_BIAS, B2_OUT, B2_K, g_a); // g_a: (32,128)
  relu_inplace(g_a, B2_OUT * (FFT_T / 2));
  maxpool2(g_a, B2_OUT, FFT_T / 2, g_mag);     // reuse g_mag: (32,64)
  float feat[FC_IN];
  gap(g_mag, B2_OUT, FFT_T / 4, feat);         // (32)
  linear(feat, FC_IN, FC_WEIGHT, FC_BIAS, FC_OUT, logits);
}

static bool read_exact(uint8_t* dst, size_t n, uint32_t timeout_ms) {
  size_t got = 0; uint32_t t0 = millis();
  while (got < n) {
    if (millis() - t0 > timeout_ms) return false;
    int a = Serial.available();
    if (a <= 0) { delay(1); continue; }
    int r = Serial.readBytes(dst + got, min((size_t)a, n - got));
    if (r > 0) got += r;
  }
  return true;
}

static bool wait_magic() {
  static const char M[4] = {'I', 'N', 'F', 'R'};
  int matched = 0; uint32_t t0 = millis();
  while (matched < 4) {
    if (millis() - t0 > READ_TIMEOUT_MS) return false;
    int b = Serial.read();
    if (b < 0) { delay(1); continue; }
    if ((uint8_t)b == (uint8_t)M[matched]) matched++;
    else if ((uint8_t)b == (uint8_t)M[0]) matched = 1;
    else matched = 0;
  }
  return true;
}

void setup() {
  Serial.begin(SERIAL_BAUD);
  delay(500);
  Serial.println();
  Serial.printf("ESP32 LearnableFFT detector baud=%u input=%d classes=%d\n",
                SERIAL_BAUD, MODEL_INPUT_LEN, MODEL_N_CLASSES);
  Serial.printf("free_heap=%u\n", ESP.getFreeHeap());
  Serial.println("READY");
  Serial.flush();
}

void loop() {
  if (!wait_magic()) return;
  if (!read_exact((uint8_t*)g_input, MODEL_INPUT_LEN * sizeof(float), READ_TIMEOUT_MS)) return;

  float logits[MODEL_N_CLASSES];
  forward(g_input, logits);
  uint8_t pred = argmaxN(logits, MODEL_N_CLASSES);

  uint8_t out[4 + MODEL_N_CLASSES * 4 + 1];
  out[0] = 'R'; out[1] = 'S'; out[2] = 'L'; out[3] = 'T';
  memcpy(out + 4, logits, MODEL_N_CLASSES * sizeof(float));
  out[4 + MODEL_N_CLASSES * 4] = pred;
  Serial.write(out, sizeof(out));
  Serial.flush();
}
