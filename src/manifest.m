#import "nmoe/manifest.h"

#include <ctype.h>
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#import <Foundation/Foundation.h>

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

static void nmoe_manifest_reserve(nmoe_manifest *m, int needed) {
    if (!m) {
        return;
    }
    if (needed <= m->capacity) {
        return;
    }
    int cap = m->capacity > 0 ? m->capacity : 32;
    while (cap < needed) {
        cap *= 2;
    }
    nmoe_tensor_info *next = (nmoe_tensor_info *)realloc(m->tensors, (size_t)cap * sizeof(nmoe_tensor_info));
    if (!next) {
        return;
    }
    memset(next + m->capacity, 0, (size_t)(cap - m->capacity) * sizeof(nmoe_tensor_info));
    m->tensors = next;
    m->capacity = cap;
}

static void nmoe_tensor_free(nmoe_tensor_info *ti) {
    if (!ti) {
        return;
    }
    free((void *)ti->name);
    memset(ti, 0, sizeof(*ti));
}

static NSString *nmoe_string_from_id(id obj) {
    if ([obj isKindOfClass:[NSString class]]) {
        return (NSString *)obj;
    }
    if ([obj respondsToSelector:@selector(stringValue)]) {
        return [obj stringValue];
    }
    return nil;
}

static void nmoe_parse_shape(NSArray *shape_array, nmoe_tensor_info *ti) {
    if (!shape_array || !ti) {
        return;
    }
    ti->ndim = (int)[shape_array count];
    if (ti->ndim > 4) {
        ti->ndim = 4;
    }
    for (int i = 0; i < ti->ndim; i++) {
        id item = shape_array[(NSUInteger)i];
        ti->shape[i] = (int)[item integerValue];
    }
}

static void nmoe_parse_tensor_dict(nmoe_manifest *m, NSString *name, NSDictionary *dict) {
    if (!m || !name || !dict) {
        return;
    }
    id offset_obj = dict[@"offset"];
    id size_obj = dict[@"size"];
    id dtype_obj = dict[@"dtype"];
    if (!offset_obj || !size_obj || !dtype_obj) {
        return;
    }
    nmoe_manifest_reserve(m, m->num_tensors + 1);
    if (m->num_tensors >= m->capacity) {
        return;
    }
    nmoe_tensor_info *ti = &m->tensors[m->num_tensors++];
    memset(ti, 0, sizeof(*ti));
    ti->name = nmoe_dup_cstr(name.UTF8String);
    ti->offset = (size_t)[offset_obj longLongValue];
    ti->size = (size_t)[size_obj longLongValue];
    NSString *dtype = [[nmoe_string_from_id(dtype_obj) uppercaseString] copy];
    const char *dtype_utf8 = dtype.UTF8String;
    if (dtype_utf8) {
        strncpy(ti->dtype, dtype_utf8, sizeof(ti->dtype) - 1);
        ti->dtype[sizeof(ti->dtype) - 1] = '\0';
    }
    id shape_obj = dict[@"shape"];
    if ([shape_obj isKindOfClass:[NSArray class]]) {
        nmoe_parse_shape((NSArray *)shape_obj, ti);
    }
}

static BOOL nmoe_dict_looks_like_tensor(NSDictionary *dict) {
    return dict[@"offset"] != nil && dict[@"size"] != nil && dict[@"dtype"] != nil;
}

static void nmoe_walk_manifest_node(nmoe_manifest *m, NSString *name_hint, id node) {
    if (!node || !m) {
        return;
    }
    if ([node isKindOfClass:[NSDictionary class]]) {
        NSDictionary *dict = (NSDictionary *)node;
        if (nmoe_dict_looks_like_tensor(dict) && name_hint) {
            nmoe_parse_tensor_dict(m, name_hint, dict);
            return;
        }
        if (dict[@"tensors"]) {
            nmoe_walk_manifest_node(m, nil, dict[@"tensors"]);
            return;
        }
        if (dict[@"entries"]) {
            nmoe_walk_manifest_node(m, nil, dict[@"entries"]);
            return;
        }
        if (dict[@"weights"]) {
            nmoe_walk_manifest_node(m, nil, dict[@"weights"]);
            return;
        }
        if (dict[@"weight_map"] && [dict[@"weight_map"] isKindOfClass:[NSDictionary class]]) {
            NSDictionary *map = dict[@"weight_map"];
            for (NSString *key in map) {
                nmoe_walk_manifest_node(m, key, map[key]);
            }
            return;
        }
        for (NSString *key in dict) {
            id value = dict[key];
            if ([value isKindOfClass:[NSDictionary class]] || [value isKindOfClass:[NSArray class]]) {
                if (nmoe_dict_looks_like_tensor((NSDictionary *)value)) {
                    nmoe_walk_manifest_node(m, key, value);
                } else if (![key isEqualToString:@"metadata"] &&
                           ![key isEqualToString:@"config"] &&
                           ![key isEqualToString:@"model"] &&
                           ![key isEqualToString:@"version"]) {
                    nmoe_walk_manifest_node(m, key, value);
                }
            }
        }
        return;
    }
    if ([node isKindOfClass:[NSArray class]]) {
        NSArray *array = (NSArray *)node;
        for (id item in array) {
            if ([item isKindOfClass:[NSDictionary class]]) {
                NSDictionary *dict = (NSDictionary *)item;
                NSString *name = name_hint;
                id explicit_name = dict[@"name"];
                if (explicit_name) {
                    name = nmoe_string_from_id(explicit_name);
                }
                if (nmoe_dict_looks_like_tensor(dict) && name) {
                    nmoe_parse_tensor_dict(m, name, dict);
                } else {
                    nmoe_walk_manifest_node(m, name, item);
                }
            } else {
                nmoe_walk_manifest_node(m, name_hint, item);
            }
        }
    }
}

nmoe_manifest *nmoe_manifest_load(const char *json_path, int quiet) {
    @autoreleasepool {
        if (!json_path) {
            return NULL;
        }
        NSString *path = [NSString stringWithUTF8String:json_path];
        NSData *data = [NSData dataWithContentsOfFile:path options:0 error:nil];
        if (!data) {
            if (!quiet) {
                fprintf(stderr, "nmoe: unable to read manifest %s\n", json_path);
            }
            return NULL;
        }
        NSError *error = nil;
        id root = [NSJSONSerialization JSONObjectWithData:data options:NSJSONReadingAllowFragments error:&error];
        if (!root || error) {
            if (!quiet) {
                fprintf(stderr, "nmoe: unable to parse manifest %s: %s\n", json_path, error.localizedDescription.UTF8String ?: "unknown error");
            }
            return NULL;
        }
        nmoe_manifest *m = (nmoe_manifest *)calloc(1, sizeof(nmoe_manifest));
        if (!m) {
            return NULL;
        }
        nmoe_walk_manifest_node(m, nil, root);
        if (m->num_tensors == 0) {
            if (!quiet) {
                fprintf(stderr, "nmoe: manifest %s did not contain any tensors\n", json_path);
            }
            nmoe_manifest_free(m);
            return NULL;
        }
        return m;
    }
}

void nmoe_manifest_free(nmoe_manifest *m) {
    if (!m) {
        return;
    }
    for (int i = 0; i < m->num_tensors; i++) {
        nmoe_tensor_free(&m->tensors[i]);
    }
    free(m->tensors);
    free(m);
}

nmoe_weight_file *nmoe_weight_open(const char *bin_path, const char *json_path, int quiet) {
    if (!bin_path || !json_path) {
        return NULL;
    }
    int fd = open(bin_path, O_RDONLY);
    if (fd < 0) {
        if (!quiet) {
            fprintf(stderr, "nmoe: unable to open weights %s: %s\n", bin_path, strerror(errno));
        }
        return NULL;
    }
    struct stat st;
    if (fstat(fd, &st) != 0) {
        if (!quiet) {
            fprintf(stderr, "nmoe: unable to stat weights %s: %s\n", bin_path, strerror(errno));
        }
        close(fd);
        return NULL;
    }
    if (st.st_size <= 0) {
        if (!quiet) {
            fprintf(stderr, "nmoe: weights file %s is empty\n", bin_path);
        }
        close(fd);
        return NULL;
    }
    void *data = mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (data == MAP_FAILED) {
        if (!quiet) {
            fprintf(stderr, "nmoe: mmap failed for %s: %s\n", bin_path, strerror(errno));
        }
        return NULL;
    }

    nmoe_manifest *manifest = nmoe_manifest_load(json_path, quiet);
    if (!manifest) {
        munmap(data, (size_t)st.st_size);
        return NULL;
    }

    nmoe_weight_file *wf = (nmoe_weight_file *)calloc(1, sizeof(nmoe_weight_file));
    if (!wf) {
        munmap(data, (size_t)st.st_size);
        nmoe_manifest_free(manifest);
        return NULL;
    }
    wf->data = data;
    wf->size = (size_t)st.st_size;
    wf->manifest = manifest;
    return wf;
}

void nmoe_weight_close(nmoe_weight_file *wf) {
    if (!wf) {
        return;
    }
    if (wf->data && wf->size) {
        munmap(wf->data, wf->size);
    }
    nmoe_manifest_free(wf->manifest);
    free(wf);
}

nmoe_tensor_info *nmoe_weight_tensor_info(nmoe_weight_file *wf, const char *name) {
    if (!wf || !wf->manifest || !name) {
        return NULL;
    }
    for (int i = 0; i < wf->manifest->num_tensors; i++) {
        nmoe_tensor_info *ti = &wf->manifest->tensors[i];
        if (ti->name && strcmp(ti->name, name) == 0) {
            return ti;
        }
    }
    return NULL;
}

void *nmoe_weight_tensor_ptr(nmoe_weight_file *wf, const char *name) {
    nmoe_tensor_info *ti = nmoe_weight_tensor_info(wf, name);
    if (!wf || !ti || !wf->data) {
        return NULL;
    }
    if (ti->offset + ti->size > wf->size) {
        return NULL;
    }
    return (void *)((unsigned char *)wf->data + ti->offset);
}

