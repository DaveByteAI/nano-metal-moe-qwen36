#ifndef NMOE_MANIFEST_H
#define NMOE_MANIFEST_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct nmoe_tensor_info {
    const char *name;
    size_t offset;
    size_t size;
    int ndim;
    int shape[4];
    char dtype[8];
} nmoe_tensor_info;

typedef struct nmoe_manifest {
    nmoe_tensor_info *tensors;
    int num_tensors;
    int capacity;
} nmoe_manifest;

typedef struct nmoe_weight_file {
    void *data;
    size_t size;
    nmoe_manifest *manifest;
} nmoe_weight_file;

nmoe_manifest *nmoe_manifest_load(const char *json_path, int quiet);
void nmoe_manifest_free(nmoe_manifest *m);
nmoe_weight_file *nmoe_weight_open(const char *bin_path, const char *json_path, int quiet);
void nmoe_weight_close(nmoe_weight_file *wf);
void *nmoe_weight_tensor_ptr(nmoe_weight_file *wf, const char *name);
nmoe_tensor_info *nmoe_weight_tensor_info(nmoe_weight_file *wf, const char *name);

#ifdef __cplusplus
}
#endif

#endif

