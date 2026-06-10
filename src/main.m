#import <Foundation/Foundation.h>

#include "nmoe/nmoe.h"

int main(int argc, char **argv) {
    @autoreleasepool {
        setvbuf(stdout, NULL, _IOLBF, 0);

        nmoe_app_config cfg;
        nmoe_app_config_init(&cfg);

        int parse_ok = nmoe_app_parse(argc, argv, &cfg);
        if (parse_ok == 2) {
            return 0;
        }
        if (parse_ok != 1) {
            nmoe_app_print_usage(argc > 0 ? argv[0] : "nmoe");
            return 1;
        }

        nmoe_app_apply(&cfg);

        nmoe_runtime *rt = nmoe_runtime_create(&cfg);
        if (rt == NULL) {
            return 1;
        }

        int rc = nmoe_runtime_run(rt);
        nmoe_runtime_destroy(rt);
        return rc == 0 ? 0 : 1;
    }
}
