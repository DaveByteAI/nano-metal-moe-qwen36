#ifndef NMOE_BACKEND_H
#define NMOE_BACKEND_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct nmoe_backend nmoe_backend;

typedef enum nmoe_backend_kernel_kind {
    NMOE_BACKEND_KERNEL_DEQUANT_MATVEC_Q4 = 0,
    NMOE_BACKEND_KERNEL_DEQUANT_MATVEC_Q2 = 1,
    NMOE_BACKEND_KERNEL_RMS_NORM = 2,
    NMOE_BACKEND_KERNEL_ROPE = 3,
    NMOE_BACKEND_KERNEL_ATTN_SCORES_BATCHED = 4,
    NMOE_BACKEND_KERNEL_ATTN_SOFTMAX_BATCHED = 5,
    NMOE_BACKEND_KERNEL_ATTN_VALUES_BATCHED = 6,
    NMOE_BACKEND_KERNEL_SIGMOID_GATE = 7,
    NMOE_BACKEND_KERNEL_LINEAR_CONV1D = 8,
    NMOE_BACKEND_KERNEL_GATED_DELTA_NET = 9,
    NMOE_BACKEND_KERNEL_MOE_EXPERT_GATE_UP = 10,
    NMOE_BACKEND_KERNEL_MOE_EXPERT_DOWN = 11,
    NMOE_BACKEND_KERNEL_MOE_COMBINE = 12,
    NMOE_BACKEND_KERNEL_DEQUANT_ROW_Q4 = 13,
    NMOE_BACKEND_KERNEL_DEQUANT_ROW_Q2 = 14,
    NMOE_BACKEND_KERNEL_ARGMAX_TOP1 = 15,
    NMOE_BACKEND_KERNEL_RMS_NORM_APPLY_BF16 = 16,
    NMOE_BACKEND_KERNEL_RMS_NORM_APPLY_F32 = 17,
    NMOE_BACKEND_KERNEL_RMS_NORM_QK = 18,
    NMOE_BACKEND_KERNEL_COMPUTE_DECAY_BETA = 19,
    NMOE_BACKEND_KERNEL_GATED_RMS_NORM = 20,
    NMOE_BACKEND_KERNEL_RESIDUAL_ADD = 21,
    NMOE_BACKEND_KERNEL_WEIGHTED_EXPERT_SUM = 22,
    NMOE_BACKEND_KERNEL_COPY_F32 = 23,
    NMOE_BACKEND_KERNEL_ROUTE_TOPK = 24,
    NMOE_BACKEND_KERNEL_EXPERT_GATE_UP_Q4 = 25,
    NMOE_BACKEND_KERNEL_EXPERT_GATE_UP_Q4_BATCHED = 26,
    NMOE_BACKEND_KERNEL_EXPERT_DOWN_Q4_BATCHED = 27,
    NMOE_BACKEND_KERNEL_EXPERT_GATE_UP_Q4_ROUTED = 28,
    NMOE_BACKEND_KERNEL_EXPERT_DOWN_Q4_ROUTED = 29,
    NMOE_BACKEND_KERNEL_WEIGHTED_EXPERT_SUM_ROUTED = 30,
    NMOE_BACKEND_KERNEL_EXPERT_DOWN_COMBINE_Q4 = 31,
    NMOE_BACKEND_KERNEL_ROUTE_SHARED_Q4 = 32,
    NMOE_BACKEND_KERNEL_LM_HEAD_ARGMAX_Q4 = 33,
    NMOE_BACKEND_KERNEL_LM_HEAD_ARGMAX_REDUCE = 34,
    NMOE_BACKEND_KERNEL_FULL_QK_PREP = 35,
    NMOE_BACKEND_KERNEL_LINEAR_PROJ_Q4 = 36,
    NMOE_BACKEND_KERNEL_ATTN_DECODE_FUSED = 37,
    NMOE_BACKEND_KERNEL_COUNT = 38,
} nmoe_backend_kernel_kind;

nmoe_backend *nmoe_backend_create(int quiet);
void nmoe_backend_destroy(nmoe_backend *backend);
void nmoe_backend_reset_linear_state(nmoe_backend *backend);
void nmoe_backend_set_weight_buffer(nmoe_backend *backend, void *data, size_t size);
void *nmoe_backend_device(nmoe_backend *backend);
void *nmoe_backend_command_queue(nmoe_backend *backend);
void *nmoe_backend_library(nmoe_backend *backend);
const char *nmoe_backend_kernel_name(nmoe_backend_kernel_kind kind);
void *nmoe_backend_pipeline_state(nmoe_backend *backend, nmoe_backend_kernel_kind kind);
void *nmoe_backend_weight_buffer(nmoe_backend *backend);
void *nmoe_backend_lookup_shared_buffer(nmoe_backend *backend, const char *label);
void *nmoe_backend_register_shared_buffer(nmoe_backend *backend, const char *label, const void *data, size_t size);
void *nmoe_backend_register_state_buffer(nmoe_backend *backend, const char *label, size_t size);

#ifdef __cplusplus
}
#endif

#endif
