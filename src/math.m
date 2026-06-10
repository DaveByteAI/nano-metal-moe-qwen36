#include "nmoe/math.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

float nmoe_bf16_to_f32(uint16_t bf16) {
    uint32_t bits = ((uint32_t)bf16) << 16;
    float value = 0.0f;
    memcpy(&value, &bits, sizeof(value));
    return value;
}

uint16_t nmoe_f32_to_bf16(float value) {
    uint32_t bits = 0;
    memcpy(&bits, &value, sizeof(bits));
    uint32_t rounded = bits + 0x7FFFu + ((bits >> 16) & 1u);
    return (uint16_t)(rounded >> 16);
}

float nmoe_cpu_sigmoid(float x) {
    if (x >= 0.0f) {
        float z = expf(-x);
        return 1.0f / (1.0f + z);
    }
    float z = expf(x);
    return z / (1.0f + z);
}

float nmoe_cpu_silu(float x) {
    return x * nmoe_cpu_sigmoid(x);
}

void nmoe_cpu_rms_norm(const float *input,
                       const uint16_t *weight_bf16,
                       float *output,
                       size_t count,
                       float eps,
                       int add_one) {
    if (input == NULL || output == NULL || count == 0) {
        return;
    }

    long double sumsq = 0.0;
    for (size_t i = 0; i < count; i++) {
        long double v = input[i];
        sumsq += v * v;
    }

    float rms = sqrtf((float)(sumsq / (long double)count) + eps);
    if (rms == 0.0f) {
        rms = 1.0f;
    }

    for (size_t i = 0; i < count; i++) {
        float weight = 1.0f;
        if (weight_bf16 != NULL) {
            weight = nmoe_bf16_to_f32(weight_bf16[i]);
            if (add_one) {
                weight += 1.0f;
            }
        }
        output[i] = (input[i] / rms) * weight;
    }
}

void nmoe_cpu_softmax(float *values, size_t count) {
    if (values == NULL || count == 0) {
        return;
    }
    float maxv = values[0];
    for (size_t i = 1; i < count; i++) {
        if (values[i] > maxv) {
            maxv = values[i];
        }
    }
    long double sum = 0.0;
    for (size_t i = 0; i < count; i++) {
        values[i] = expf(values[i] - maxv);
        sum += values[i];
    }
    if (sum == 0.0) {
        float uniform = 1.0f / (float)count;
        for (size_t i = 0; i < count; i++) {
            values[i] = uniform;
        }
        return;
    }
    float inv_sum = 1.0f / (float)sum;
    for (size_t i = 0; i < count; i++) {
        values[i] *= inv_sum;
    }
}

size_t nmoe_cpu_topk(const float *values, size_t count, size_t k, size_t *indices, float *selected_values) {
    if (values == NULL || count == 0 || indices == NULL || k == 0) {
        return 0;
    }
    if (k > count) {
        k = count;
    }

    uint8_t *taken = calloc(count, sizeof(uint8_t));
    if (taken == NULL) {
        return 0;
    }

    size_t produced = 0;
    while (produced < k) {
        size_t best_index = 0;
        float best_value = -INFINITY;
        for (size_t i = 0; i < count; i++) {
            if (taken[i]) {
                continue;
            }
            float v = values[i];
            if (v > best_value || (v == best_value && i < best_index)) {
                best_value = v;
                best_index = i;
            }
        }
        taken[best_index] = 1;
        indices[produced] = best_index;
        if (selected_values != NULL) {
            selected_values[produced] = best_value;
        }
        produced++;
    }

    free(taken);
    return produced;
}

void nmoe_cpu_renormalize(float *values, size_t count) {
    if (values == NULL || count == 0) {
        return;
    }
    float sum = 0.0f;
    for (size_t i = 0; i < count; i++) {
        sum += values[i];
    }
    if (sum == 0.0f) {
        float uniform = 1.0f / (float)count;
        for (size_t i = 0; i < count; i++) {
            values[i] = uniform;
        }
        return;
    }
    float inv = 1.0f / sum;
    for (size_t i = 0; i < count; i++) {
        values[i] *= inv;
    }
}

void nmoe_cpu_swiglu(const float *gate, const float *up, float *output, size_t count) {
    if (gate == NULL || up == NULL || output == NULL) {
        return;
    }
    for (size_t i = 0; i < count; i++) {
        output[i] = nmoe_cpu_silu(gate[i]) * up[i];
    }
}

void nmoe_cpu_rope_apply(float *q, float *k, size_t head_dim, size_t rotary_dim, size_t pos, float theta) {
    if (q == NULL || k == NULL || head_dim == 0 || rotary_dim == 0) {
        return;
    }
    size_t half = rotary_dim / 2;
    if (half == 0 || half > head_dim) {
        return;
    }
    for (size_t i = 0; i < half; i++) {
        float freq = 1.0f / powf(theta, (2.0f * (float)i) / (float)rotary_dim);
        float angle = (float)pos * freq;
        float cs = cosf(angle);
        float sn = sinf(angle);

        size_t j = i + half;
        float q0 = q[i];
        float q1 = q[j];
        q[i] = q0 * cs - q1 * sn;
        q[j] = q0 * sn + q1 * cs;

        float k0 = k[i];
        float k1 = k[j];
        k[i] = k0 * cs - k1 * sn;
        k[j] = k0 * sn + k1 * cs;
    }
}

void nmoe_cpu_unpack_row(const uint32_t *packed_row, uint16_t *output, size_t in_dim, int quant_bits) {
    if (packed_row == NULL || output == NULL || in_dim == 0) {
        return;
    }
    size_t values_per_word = (size_t)(32 / quant_bits);
    size_t packed_cols = in_dim / values_per_word;
    for (size_t w = 0; w < packed_cols; w++) {
        uint32_t word = packed_row[w];
        for (size_t v = 0; v < values_per_word; v++) {
            size_t idx = w * values_per_word + v;
            output[idx] = (uint16_t)((word >> (v * (size_t)quant_bits)) & (quant_bits == 2 ? 0x3u : 0xFu));
        }
    }
}

void nmoe_cpu_dequant_row(const uint32_t *packed_row,
                          const uint16_t *scales_row,
                          const uint16_t *biases_row,
                          float *output,
                          size_t in_dim,
                          int quant_bits) {
    if (packed_row == NULL || scales_row == NULL || biases_row == NULL || output == NULL || in_dim == 0) {
        return;
    }

    size_t values_per_word = (size_t)(32 / quant_bits);
    size_t groups = in_dim / NMOE_CPU_QUANT_GROUP_SIZE;
    size_t words_per_group = NMOE_CPU_QUANT_GROUP_SIZE / values_per_word;

    for (size_t g = 0; g < groups; g++) {
        float scale = nmoe_bf16_to_f32(scales_row[g]);
        float bias = nmoe_bf16_to_f32(biases_row[g]);
        size_t group_word_offset = g * words_per_group;
        size_t group_col_offset = g * NMOE_CPU_QUANT_GROUP_SIZE;
        for (size_t w = 0; w < words_per_group; w++) {
            uint32_t word = packed_row[group_word_offset + w];
            for (size_t v = 0; v < values_per_word; v++) {
                size_t col = group_col_offset + w * values_per_word + v;
                uint32_t q = (word >> (v * (size_t)quant_bits)) & (quant_bits == 2 ? 0x3u : 0xFu);
                output[col] = (float)q * scale + bias;
            }
        }
    }
}

void nmoe_cpu_dequant_matvec(const uint32_t *packed,
                             const uint16_t *scales,
                             const uint16_t *biases,
                             const float *input,
                             float *output,
                             size_t out_dim,
                             size_t in_dim,
                             int quant_bits) {
    if (packed == NULL || scales == NULL || biases == NULL || input == NULL || output == NULL) {
        return;
    }

    size_t values_per_word = (size_t)(32 / quant_bits);
    size_t packed_cols = in_dim / values_per_word;
    size_t groups = in_dim / NMOE_CPU_QUANT_GROUP_SIZE;
    size_t words_per_group = NMOE_CPU_QUANT_GROUP_SIZE / values_per_word;

    for (size_t row = 0; row < out_dim; row++) {
        const uint32_t *packed_row = packed + row * packed_cols;
        const uint16_t *scales_row = scales + row * groups;
        const uint16_t *biases_row = biases + row * groups;
        float acc = 0.0f;
        for (size_t g = 0; g < groups; g++) {
            float scale = nmoe_bf16_to_f32(scales_row[g]);
            float bias = nmoe_bf16_to_f32(biases_row[g]);
            size_t group_word_offset = g * words_per_group;
            size_t group_col_offset = g * NMOE_CPU_QUANT_GROUP_SIZE;
            for (size_t w = 0; w < words_per_group; w++) {
                uint32_t word = packed_row[group_word_offset + w];
                for (size_t v = 0; v < values_per_word; v++) {
                    size_t col = group_col_offset + w * values_per_word + v;
                    uint32_t q = (word >> (v * (size_t)quant_bits)) & (quant_bits == 2 ? 0x3u : 0xFu);
                    acc += ((float)q * scale + bias) * input[col];
                }
            }
        }
        output[row] = acc;
    }
}
