#ifndef NMOE_APP_CONFIG_H
#define NMOE_APP_CONFIG_H

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    NMOE_RUN_ASK = 0,
    NMOE_RUN_CHAT = 1,
    NMOE_RUN_BENCH = 2,
} nmoe_run_mode;

typedef struct {
    nmoe_run_mode mode;
    const char *model_path;
    const char *prompt;
    char *owned_prompt;
    int max_tokens;
    int tokens_set;
    int experts;
    int quant_bits; /* 0 auto, 2 q2, 4 q4 */
    int think_budget;
    int timing;
    int quiet;
    int cpu_linear;
    int trace_tokens;
} nmoe_app_config;

void nmoe_app_config_init(nmoe_app_config *cfg);
int nmoe_app_parse(int argc, char **argv, nmoe_app_config *cfg);
void nmoe_app_apply(const nmoe_app_config *cfg);
const char *nmoe_run_mode_name(nmoe_run_mode mode);
void nmoe_app_print_usage(const char *prog);

#ifdef __cplusplus
}
#endif

#endif
