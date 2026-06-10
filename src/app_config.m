#import "nmoe/app_config.h"
#import "nmoe/expert_io.h"

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static nmoe_app_config g_applied_config;
static int g_applied_config_valid = 0;

static char *nmoe_dup_cstr(const char *s) {
    if (!s) {
        return NULL;
    }
    size_t len = strlen(s);
    char *copy = (char *)malloc(len + 1);
    if (!copy) {
        return NULL;
    }
    memcpy(copy, s, len + 1);
    return copy;
}

static int nmoe_parse_int(const char *s, int *out_value) {
    if (!s || !*s) {
        return -1;
    }
    char *end = NULL;
    long v = strtol(s, &end, 10);
    if (!end || *end != '\0') {
        return -1;
    }
    if (v < INT32_MIN || v > INT32_MAX) {
        return -1;
    }
    *out_value = (int)v;
    return 0;
}

static const char *nmoe_arg_value(const char *arg, int argc, char **argv, int *index) {
    const char *eq = strchr(arg, '=');
    if (eq && eq[1] != '\0') {
        return eq + 1;
    }
    if (*index + 1 >= argc) {
        return NULL;
    }
    (*index)++;
    return argv[*index];
}

void nmoe_app_config_init(nmoe_app_config *cfg) {
    if (!cfg) {
        return;
    }
    memset(cfg, 0, sizeof(*cfg));
    cfg->mode = NMOE_RUN_ASK;
    cfg->model_path = "qwen36_35b";
    cfg->max_tokens = 256;
    cfg->tokens_set = 0;
    cfg->experts = 8;
    cfg->quant_bits = 0;
    cfg->think_budget = 1;
    cfg->timing = 0;
    cfg->quiet = 0;
    cfg->cpu_linear = 0;
    cfg->trace_tokens = 0;
}

const char *nmoe_run_mode_name(nmoe_run_mode mode) {
    switch (mode) {
        case NMOE_RUN_ASK:
            return "ask";
        case NMOE_RUN_CHAT:
            return "chat";
        case NMOE_RUN_BENCH:
            return "bench";
        default:
            return "ask";
    }
}

void nmoe_app_print_usage(const char *prog) {
    fprintf(stderr,
            "Usage: %s [ask|chat|bench] [prompt] [options]\n"
            "\n"
            "Options:\n"
            "  --model PATH        model package directory (default: qwen36_35b)\n"
            "  --prompt TEXT       prompt text for ask/bench\n"
            "  --tokens N          generation limit (ask/bench default: 256, chat: 512)\n"
            "  --experts N         active experts per layer, 1..8 (default: 8)\n"
            "  --think N           force </think> after N thinking tokens; 0 disables (default: 1)\n"
            "  --quant auto|2|4    expert quantization selection\n"
            "  --q2 / --q4         shortcut for --quant 2|4\n"
            "  --timing            print timing summary\n"
            "  --quiet             suppress token streaming\n"
            "  --cpu-linear        force CPU linear-attention path\n"
            "  --trace-tokens      print generated token ids\n"
            "  -h, --help          show this help\n",
            prog ? prog : "nmoe");
}

static void nmoe_set_prompt(nmoe_app_config *cfg, const char *value) {
    if (!cfg) {
        return;
    }
    if (cfg->owned_prompt) {
        free(cfg->owned_prompt);
        cfg->owned_prompt = NULL;
    }
    if (value) {
        cfg->owned_prompt = nmoe_dup_cstr(value);
        cfg->prompt = cfg->owned_prompt ? cfg->owned_prompt : value;
    } else {
        cfg->prompt = NULL;
    }
}

static int nmoe_parse_mode(const char *arg, nmoe_app_config *cfg) {
    if (strcmp(arg, "ask") == 0) {
        cfg->mode = NMOE_RUN_ASK;
        return 1;
    }
    if (strcmp(arg, "chat") == 0) {
        cfg->mode = NMOE_RUN_CHAT;
        return 1;
    }
    if (strcmp(arg, "bench") == 0) {
        cfg->mode = NMOE_RUN_BENCH;
        return 1;
    }
    return 0;
}

int nmoe_app_parse(int argc, char **argv, nmoe_app_config *cfg) {
    if (!cfg) {
        return -1;
    }
    nmoe_app_config_init(cfg);
    if (argc <= 0 || !argv) {
        return -1;
    }

    int i = 1;
    if (i < argc && argv[i] && argv[i][0] != '-') {
        if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            nmoe_app_print_usage(argv[0]);
            return 2;
        }
        if (!nmoe_parse_mode(argv[i], cfg)) {
            cfg->mode = NMOE_RUN_ASK;
            nmoe_set_prompt(cfg, argv[i]);
        }
        i++;
    }

    for (; i < argc; i++) {
        const char *arg = argv[i];
        if (!arg) {
            continue;
        }
        if (strcmp(arg, "-h") == 0 || strcmp(arg, "--help") == 0) {
            nmoe_app_print_usage(argv[0]);
            return 2;
        }
        if (strcmp(arg, "--quiet") == 0) {
            cfg->quiet = 1;
            continue;
        }
        if (strcmp(arg, "--timing") == 0) {
            cfg->timing = 1;
            continue;
        }
        if (strcmp(arg, "--cpu-linear") == 0) {
            cfg->cpu_linear = 1;
            continue;
        }
        if (strcmp(arg, "--trace-tokens") == 0) {
            cfg->trace_tokens = 1;
            continue;
        }
        if (strcmp(arg, "--q2") == 0) {
            cfg->quant_bits = 2;
            continue;
        }
        if (strcmp(arg, "--q4") == 0) {
            cfg->quant_bits = 4;
            continue;
        }
        if (strncmp(arg, "--model", 7) == 0) {
            const char *value = nmoe_arg_value(arg, argc, argv, &i);
            if (!value) {
                return -1;
            }
            cfg->model_path = value;
            continue;
        }
        if (strncmp(arg, "--prompt", 8) == 0) {
            const char *value = nmoe_arg_value(arg, argc, argv, &i);
            if (!value) {
                return -1;
            }
            nmoe_set_prompt(cfg, value);
            continue;
        }
        if (strncmp(arg, "--tokens", 8) == 0) {
            const char *value = nmoe_arg_value(arg, argc, argv, &i);
            if (!value || nmoe_parse_int(value, &cfg->max_tokens) != 0) {
                return -1;
            }
            cfg->tokens_set = 1;
            continue;
        }
        if (strncmp(arg, "--experts", 9) == 0) {
            const char *value = nmoe_arg_value(arg, argc, argv, &i);
            if (!value || nmoe_parse_int(value, &cfg->experts) != 0) {
                return -1;
            }
            if (cfg->experts < 1) {
                cfg->experts = 1;
            }
            if (cfg->experts > NMOE_MAX_K) {
                cfg->experts = NMOE_MAX_K;
            }
            continue;
        }
        if (strncmp(arg, "--think", 7) == 0) {
            const char *value = nmoe_arg_value(arg, argc, argv, &i);
            if (!value || nmoe_parse_int(value, &cfg->think_budget) != 0) {
                return -1;
            }
            if (cfg->think_budget < 0) {
                return -1;
            }
            continue;
        }
        if (strncmp(arg, "--quant", 7) == 0) {
            const char *value = nmoe_arg_value(arg, argc, argv, &i);
            if (!value) {
                return -1;
            }
            if (strcmp(value, "auto") == 0) {
                cfg->quant_bits = 0;
            } else if (strcmp(value, "2") == 0) {
                cfg->quant_bits = 2;
            } else if (strcmp(value, "4") == 0) {
                cfg->quant_bits = 4;
            } else {
                return -1;
            }
            continue;
        }
        if (arg[0] != '-') {
            if (!cfg->prompt && cfg->mode != NMOE_RUN_CHAT) {
                nmoe_set_prompt(cfg, arg);
                continue;
            }
        }
        return -1;
    }

    if (cfg->mode == NMOE_RUN_ASK || cfg->mode == NMOE_RUN_BENCH) {
        if (!cfg->prompt) {
            return -1;
        }
    }
    if (cfg->mode == NMOE_RUN_CHAT && !cfg->tokens_set) {
        cfg->max_tokens = 512;
    }

    return 1;
}

void nmoe_app_apply(const nmoe_app_config *cfg) {
    if (!cfg) {
        g_applied_config_valid = 0;
        memset(&g_applied_config, 0, sizeof(g_applied_config));
        return;
    }
    g_applied_config = *cfg;
    g_applied_config_valid = 1;
}
