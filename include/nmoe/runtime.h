#ifndef NMOE_RUNTIME_H
#define NMOE_RUNTIME_H

#include "nmoe/app_config.h"
#include "nmoe/backend.h"
#include "nmoe/expert_io.h"
#include "nmoe/manifest.h"
#include "nmoe/tokenizer.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct nmoe_runtime nmoe_runtime;

nmoe_runtime *nmoe_runtime_create(const nmoe_app_config *cfg);
void nmoe_runtime_destroy(nmoe_runtime *rt);
int nmoe_runtime_run(nmoe_runtime *rt);

#ifdef __cplusplus
}
#endif

#endif

