#ifndef NMOE_MATH_H
#define NMOE_MATH_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define NMOE_CPU_QUANT_GROUP_SIZE 64u

typedef struct nmoe_dequant_linear {
    const uint32_t *weight;
    const uint16_t *scales;
    const uint16_t *biases;
    size_t out_dim;
    size_t in_dim;
    int quant_bits;
} nmoe_dequant_linear;

float nmoe_bf16_to_f32(uint16_t bf16);
uint16_t nmoe_f32_to_bf16(float value);

float nmoe_cpu_sigmoid(float x);
float nmoe_cpu_silu(float x);

void nmoe_cpu_rms_norm(const float *input,
                       const uint16_t *weight_bf16,
                       float *output,
                       size_t count,
                       float eps,
                       int add_one);

void nmoe_cpu_softmax(float *values, size_t count);
size_t nmoe_cpu_topk(const float *values, size_t count, size_t k, size_t *indices, float *selected_values);
void nmoe_cpu_renormalize(float *values, size_t count);

void nmoe_cpu_swiglu(const float *gate, const float *up, float *output, size_t count);
void nmoe_cpu_rope_apply(float *q, float *k, size_t head_dim, size_t rotary_dim, size_t pos, float theta);
void nmoe_cpu_unpack_row(const uint32_t *packed_row, uint16_t *output, size_t in_dim, int quant_bits);

void nmoe_cpu_dequant_row(const uint32_t *packed_row,
                          const uint16_t *scales_row,
                          const uint16_t *biases_row,
                          float *output,
                          size_t in_dim,
                          int quant_bits);

void nmoe_cpu_dequant_matvec(const uint32_t *packed,
                             const uint16_t *scales,
                             const uint16_t *biases,
                             const float *input,
                             float *output,
                             size_t out_dim,
                             size_t in_dim,
                             int quant_bits);

#ifdef __cplusplus
}
#endif

#endif
