#ifndef NMOE_EXPERT_IO_H
#define NMOE_EXPERT_IO_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define NMOE_MAX_K 8

typedef struct nmoe_expert_layout {
    size_t expert_size;
    int num_layers;
    int num_experts;
} nmoe_expert_layout;

typedef struct nmoe_expert_store nmoe_expert_store;

nmoe_expert_store *nmoe_expert_store_open(const char *model_path, int quant_bits, int quiet);
void nmoe_expert_store_close(nmoe_expert_store *store);
size_t nmoe_expert_active_size(int quant_bits);
const nmoe_expert_layout *nmoe_expert_layout_for_bits(int quant_bits);

#ifdef __cplusplus
}
#endif

#endif

