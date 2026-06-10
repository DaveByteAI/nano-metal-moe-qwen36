#import <Foundation/Foundation.h>

#include "nmoe/expert_io.h"

#import <fcntl.h>
#import <stdio.h>
#import <stdlib.h>
#import <stdint.h>
#import <sys/mman.h>
#import <sys/stat.h>
#import <unistd.h>
#import <string.h>

typedef NS_ENUM(NSUInteger, NMMExpertQuantBits) {
    NMMExpertQuantBitsQ2 = 2,
    NMMExpertQuantBitsQ4 = 4,
};

typedef struct {
    const char *name;
    size_t offset;
    size_t size;
    size_t rows;
    size_t logicalCols;
    size_t packedCols;
    const char *dtype;
} NMMExpertComponentLayout;

typedef struct {
    NMMExpertQuantBits quantBits;
    size_t numExperts;
    size_t expertSize;
    size_t componentCount;
    NMMExpertComponentLayout components[9];
} NMMExpertLayout;

typedef struct {
    int fd;
    void *mmapBase;
    size_t fileSize;
    BOOL mapped;
    NMMExpertLayout layout;
} NMMExpertLayerFile;

struct nmoe_expert_store {
    char *model_path;
    int quant_bits;
    int quiet;
    nmoe_expert_layout layout;
};

static NSString *const NMMExpertIOErrorDomain = @"nmoe.expert_io";

static NSError *NMMMakeError(NSInteger code, NSString *message) {
    return [NSError errorWithDomain:NMMExpertIOErrorDomain
                               code:code
                           userInfo:@{NSLocalizedDescriptionKey : message ?: @"expert I/O error"}];
}

static BOOL NMMSetError(NSError **error, NSInteger code, NSString *message) {
    if (error) {
        *error = NMMMakeError(code, message);
    }
    return NO;
}

static BOOL NMMReadFullyAtOffset(int fd, void *destination, size_t size, off_t offset, NSError **error) {
    uint8_t *cursor = (uint8_t *)destination;
    size_t remaining = size;
    off_t currentOffset = offset;
    while (remaining > 0) {
        ssize_t readBytes = pread(fd, cursor, remaining, currentOffset);
        if (readBytes < 0) {
            return NMMSetError(error, 30, @"pread failed while reading expert data");
        }
        if (readBytes == 0) {
            return NMMSetError(error, 31, @"unexpected EOF while reading expert data");
        }
        cursor += (size_t)readBytes;
        currentOffset += (off_t)readBytes;
        remaining -= (size_t)readBytes;
    }
    return YES;
}

static size_t NMMValuesPerWordForBits(NMMExpertQuantBits bits) {
    return 32u / (size_t)bits;
}

static size_t NMMActiveExpertSizeForBits(NMMExpertQuantBits bits) {
    switch (bits) {
        case NMMExpertQuantBitsQ4:
            return 1769472;
        case NMMExpertQuantBitsQ2:
            return 983040;
        default:
            return 0;
    }
}

static size_t NMMBytesPerDType(const char *dtype) {
    if (strcmp(dtype, "u32") == 0) {
        return 4;
    }
    if (strcmp(dtype, "u16") == 0) {
        return 2;
    }
    if (strcmp(dtype, "f32") == 0) {
        return 4;
    }
    return 0;
}

static NMMExpertComponentLayout NMMComponent(const char *name,
                                             size_t offset,
                                             size_t size,
                                             size_t rows,
                                             size_t logicalCols,
                                             size_t packedCols,
                                             const char *dtype) {
    NMMExpertComponentLayout component;
    component.name = name;
    component.offset = offset;
    component.size = size;
    component.rows = rows;
    component.logicalCols = logicalCols;
    component.packedCols = packedCols;
    component.dtype = dtype;
    return component;
}

static BOOL NMMBuildExpertLayout(NMMExpertQuantBits bits, size_t numExperts, NMMExpertLayout *outLayout) {
    if (!outLayout) {
        return NO;
    }
    size_t valuesPerWord = NMMValuesPerWordForBits(bits);
    if (bits != NMMExpertQuantBitsQ2 && bits != NMMExpertQuantBitsQ4) {
        return NO;
    }

    NMMExpertLayout layout;
    memset(&layout, 0, sizeof(layout));
    layout.quantBits = bits;
    layout.numExperts = numExperts;
    layout.componentCount = 9;

    const size_t gateRows = 512;
    const size_t upRows = 512;
    const size_t downRows = 2048;
    const size_t gateCols = 2048;
    const size_t upCols = 2048;
    const size_t downCols = 512;
    const size_t groupCols = gateCols / 64;
    const size_t downGroupCols = downCols / 64;
    const size_t gatePackedCols = gateCols / valuesPerWord;
    const size_t downPackedCols = downCols / valuesPerWord;

    size_t offset = 0;
    layout.components[0] = NMMComponent("gate_proj.weight", offset, gateRows * gatePackedCols * 4, gateRows, gateCols, gatePackedCols, "u32");
    offset += layout.components[0].size;
    layout.components[1] = NMMComponent("gate_proj.scales", offset, gateRows * groupCols * 2, gateRows, gateCols, groupCols, "u16");
    offset += layout.components[1].size;
    layout.components[2] = NMMComponent("gate_proj.biases", offset, gateRows * groupCols * 2, gateRows, gateCols, groupCols, "u16");
    offset += layout.components[2].size;
    layout.components[3] = NMMComponent("up_proj.weight", offset, upRows * gatePackedCols * 4, upRows, upCols, gatePackedCols, "u32");
    offset += layout.components[3].size;
    layout.components[4] = NMMComponent("up_proj.scales", offset, upRows * groupCols * 2, upRows, upCols, groupCols, "u16");
    offset += layout.components[4].size;
    layout.components[5] = NMMComponent("up_proj.biases", offset, upRows * groupCols * 2, upRows, upCols, groupCols, "u16");
    offset += layout.components[5].size;
    layout.components[6] = NMMComponent("down_proj.weight", offset, downRows * downPackedCols * 4, downRows, downCols, downPackedCols, "u32");
    offset += layout.components[6].size;
    layout.components[7] = NMMComponent("down_proj.scales", offset, downRows * downGroupCols * 2, downRows, downCols, downGroupCols, "u16");
    offset += layout.components[7].size;
    layout.components[8] = NMMComponent("down_proj.biases", offset, downRows * downGroupCols * 2, downRows, downCols, downGroupCols, "u16");
    offset += layout.components[8].size;

    layout.expertSize = offset;
    *outLayout = layout;
    return YES;
}

static BOOL NMMParseQuantBits(id quantValue, NMMExpertQuantBits *outBits) {
    if ([quantValue isKindOfClass:[NSNumber class]]) {
        NSUInteger value = [quantValue unsignedIntegerValue];
        if (value == 2 || value == 4) {
            if (outBits) {
                *outBits = (NMMExpertQuantBits)value;
            }
            return YES;
        }
    }
    if ([quantValue isKindOfClass:[NSString class]]) {
        NSString *text = [(NSString *)quantValue lowercaseString];
        if ([text isEqualToString:@"q2"] || [text isEqualToString:@"2"]) {
            if (outBits) {
                *outBits = NMMExpertQuantBitsQ2;
            }
            return YES;
        }
        if ([text isEqualToString:@"q4"] || [text isEqualToString:@"4"]) {
            if (outBits) {
                *outBits = NMMExpertQuantBitsQ4;
            }
            return YES;
        }
    }
    return NO;
}

static NSDictionary *NMMDictionaryFromJSONFile(NSString *path, NSError **error) {
    if (!path) {
        if (error) {
            *error = NMMMakeError(1, @"layout path is null");
        }
        return nil;
    }
    NSData *data = [NSData dataWithContentsOfFile:path options:0 error:error];
    if (!data) {
        return nil;
    }
    id json = [NSJSONSerialization JSONObjectWithData:data options:NSJSONReadingMutableContainers error:error];
    if (![json isKindOfClass:[NSDictionary class]]) {
        if (error && !*error) {
            *error = NMMMakeError(1, [NSString stringWithFormat:@"layout file %@ did not contain a JSON object", path]);
        }
        return nil;
    }
    return (NSDictionary *)json;
}

static BOOL NMMValidateLayoutComponent(const NMMExpertComponentLayout *expected, NSDictionary *component, NSError **error) {
    NSString *name = component[@"name"];
    NSNumber *offset = component[@"offset"];
    NSNumber *size = component[@"size"];
    NSString *dtype = component[@"dtype"];
    NSArray *shape = component[@"shape"];
    NSArray *logicalShape = component[@"logical_shape"];

    if (!name || !offset || !size || !dtype || !shape) {
        return NMMSetError(error, 2, @"layout component is missing required keys");
    }
    dtype = [dtype lowercaseString];
    if (strcmp(expected->name, name.UTF8String) != 0) {
        return NMMSetError(error, 3, [NSString stringWithFormat:@"unexpected layout component %@, expected %s", name, expected->name]);
    }
    if (expected->offset != offset.unsignedLongLongValue || expected->size != size.unsignedLongLongValue) {
        return NMMSetError(error, 4, [NSString stringWithFormat:@"layout component %@ has offset/size mismatch", name]);
    }
    if (strcmp(expected->dtype, dtype.UTF8String) != 0) {
        return NMMSetError(error, 5, [NSString stringWithFormat:@"layout component %@ has dtype %@, expected %s", name, dtype, expected->dtype]);
    }
    if (shape.count != 2) {
        return NMMSetError(error, 6, [NSString stringWithFormat:@"layout component %@ shape must be rank 2", name]);
    }
    if (logicalShape && logicalShape.count != 2) {
        return NMMSetError(error, 7, [NSString stringWithFormat:@"layout component %@ logical_shape must be rank 2", name]);
    }
    if ([shape[0] unsignedLongLongValue] != expected->rows || [shape[1] unsignedLongLongValue] != expected->packedCols) {
        return NMMSetError(error, 8, [NSString stringWithFormat:@"layout component %@ shape mismatch", name]);
    }
    if (logicalShape) {
        if ([logicalShape[0] unsignedLongLongValue] != expected->rows ||
            [logicalShape[1] unsignedLongLongValue] != expected->logicalCols) {
            return NMMSetError(error, 9, [NSString stringWithFormat:@"layout component %@ logical_shape mismatch", name]);
        }
    }
    if (NMMBytesPerDType(expected->dtype) * expected->rows * expected->packedCols != expected->size) {
        return NMMSetError(error, 10, [NSString stringWithFormat:@"layout component %@ has inconsistent byte size", name]);
    }
    return YES;
}

BOOL NMMExpertLayoutLoadFromJSONPath(NSString *layoutPath, NMMExpertLayout *outLayout, NSError **error) {
    NSDictionary *root = NMMDictionaryFromJSONFile(layoutPath, error);
    if (!root) {
        return NO;
    }

    NMMExpertQuantBits bits = NMMExpertQuantBitsQ4;
    id quantValue = root[@"bits"] ?: root[@"quant"];
    if (!NMMParseQuantBits(quantValue, &bits)) {
        return NMMSetError(error, 11, [NSString stringWithFormat:@"could not parse quant value in %@", layoutPath]);
    }

    NSNumber *numExpertsValue = root[@"num_experts"];
    size_t numExperts = numExpertsValue ? numExpertsValue.unsignedIntegerValue : 256;
    NMMExpertLayout expected;
    if (!NMMBuildExpertLayout(bits, numExperts, &expected)) {
        return NMMSetError(error, 12, @"failed to build built-in expert layout");
    }

    NSNumber *expertSize = root[@"expert_size"];
    if (expertSize && expertSize.unsignedLongLongValue != expected.expertSize) {
        return NMMSetError(error, 13, [NSString stringWithFormat:@"layout expert_size mismatch in %@", layoutPath]);
    }

    NSArray *components = root[@"components"];
    if (![components isKindOfClass:[NSArray class]] || components.count != expected.componentCount) {
        return NMMSetError(error, 14, [NSString stringWithFormat:@"layout components must contain %zu entries", expected.componentCount]);
    }

    for (NSUInteger index = 0; index < components.count; index++) {
        if (![components[index] isKindOfClass:[NSDictionary class]]) {
            return NMMSetError(error, 15, @"layout components must be objects");
        }
        if (!NMMValidateLayoutComponent(&expected.components[index], components[index], error)) {
            return NO;
        }
    }

    if (outLayout) {
        *outLayout = expected;
    }
    return YES;
}

size_t NMMExpertActiveSizeForBits(NSUInteger bits) {
    return NMMActiveExpertSizeForBits((NMMExpertQuantBits)bits);
}

size_t NMMExpertPackedWeightCols(size_t logicalCols, NSUInteger bits) {
    size_t valuesPerWord = NMMValuesPerWordForBits((NMMExpertQuantBits)bits);
    return logicalCols / valuesPerWord;
}

BOOL NMMExpertLayoutForBits(NSUInteger bits, size_t numExperts, NMMExpertLayout *outLayout) {
    return NMMBuildExpertLayout((NMMExpertQuantBits)bits, numExperts, outLayout);
}

BOOL NMMExpertLayoutFindComponent(const NMMExpertLayout *layout, const char *name, NMMExpertComponentLayout *outComponent) {
    if (!layout || !name) {
        return NO;
    }
    for (NSUInteger index = 0; index < layout->componentCount; index++) {
        const NMMExpertComponentLayout component = layout->components[index];
        if (strcmp(component.name, name) == 0) {
            if (outComponent) {
                *outComponent = component;
            }
            return YES;
        }
    }
    return NO;
}

static BOOL NMMOpenLayerBackingFile(NSString *path,
                                    const NMMExpertLayout *layout,
                                    NMMExpertLayerFile *outFile,
                                    NSError **error) {
    if (!path) {
        return NMMSetError(error, 16, @"layer path is null");
    }
    if (!outFile) {
        return NMMSetError(error, 16, @"outFile is null");
    }

    int fd = open(path.fileSystemRepresentation, O_RDONLY);
    if (fd < 0) {
        return NMMSetError(error, 17, [NSString stringWithFormat:@"failed to open %@", path]);
    }

#ifdef F_RDAHEAD
    (void)fcntl(fd, F_RDAHEAD, 0);
#endif

    struct stat st;
    if (fstat(fd, &st) != 0) {
        close(fd);
        return NMMSetError(error, 18, [NSString stringWithFormat:@"failed to stat %@", path]);
    }

    size_t fileSize = (size_t)st.st_size;
    if (layout && layout->expertSize > 0) {
        size_t expected = layout->expertSize * layout->numExperts;
        if (fileSize != expected) {
            close(fd);
            return NMMSetError(error, 19, [NSString stringWithFormat:@"expert file %@ has size %zu, expected %zu", path, fileSize, expected]);
        }
    }

    uint8_t warmup[4096];
    size_t warmupSize = fileSize < sizeof(warmup) ? fileSize : sizeof(warmup);
    if (warmupSize > 0) {
        (void)pread(fd, warmup, warmupSize, 0);
    }

    void *base = NULL;
    BOOL mapped = NO;
    if (fileSize > 0) {
        base = mmap(NULL, fileSize, PROT_READ, MAP_PRIVATE, fd, 0);
        if (base != MAP_FAILED) {
            mapped = YES;
        } else {
            base = NULL;
        }
    }

    outFile->fd = fd;
    outFile->mmapBase = base;
    outFile->fileSize = fileSize;
    outFile->mapped = mapped;
    if (layout) {
        outFile->layout = *layout;
    } else {
        memset(&outFile->layout, 0, sizeof(outFile->layout));
    }
    return YES;
}

BOOL NMMExpertLayerOpenAtPath(NSString *path, const NMMExpertLayout *layout, NMMExpertLayerFile *outFile, NSError **error) {
    return NMMOpenLayerBackingFile(path, layout, outFile, error);
}

BOOL NMMExpertLayerOpenInDirectory(NSString *directoryPath, NSUInteger layerIndex, NMMExpertLayerFile *outFile, NSError **error) {
    NSString *layoutPath = [directoryPath stringByAppendingPathComponent:@"layout.json"];
    NMMExpertLayout layout;
    if (!NMMExpertLayoutLoadFromJSONPath(layoutPath, &layout, error)) {
        return NO;
    }
    NSString *layerName = [NSString stringWithFormat:@"layer_%02lu.bin", (unsigned long)layerIndex];
    NSString *layerPath = [directoryPath stringByAppendingPathComponent:layerName];
    return NMMExpertLayerOpenAtPath(layerPath, &layout, outFile, error);
}

void NMMExpertLayerClose(NMMExpertLayerFile *file) {
    if (!file) {
        return;
    }
    if (file->mapped && file->mmapBase && file->fileSize > 0) {
        munmap(file->mmapBase, file->fileSize);
    }
    if (file->fd >= 0) {
        close(file->fd);
    }
    memset(file, 0, sizeof(*file));
    file->fd = -1;
}

BOOL NMMExpertLayerCopyExpert(const NMMExpertLayerFile *file, NSUInteger expertIndex, void *destination, NSError **error) {
    if (!file || !destination) {
        return NMMSetError(error, 20, @"invalid expert copy arguments");
    }
    if (expertIndex >= file->layout.numExperts) {
        return NMMSetError(error, 21, @"expert index out of range");
    }

    size_t offset = (size_t)expertIndex * file->layout.expertSize;
    if (offset + file->layout.expertSize > file->fileSize) {
        return NMMSetError(error, 22, @"expert offset exceeds file size");
    }

    if (file->mapped && file->mmapBase) {
        memcpy(destination, (uint8_t *)file->mmapBase + offset, file->layout.expertSize);
        return YES;
    }

    return NMMReadFullyAtOffset(file->fd, destination, file->layout.expertSize, (off_t)offset, error);
}

BOOL NMMExpertLayerCopyExperts(const NMMExpertLayerFile *file,
                               const NSUInteger *indices,
                               NSUInteger count,
                               void *destination,
                               NSError **error) {
    if (!file || !indices || !destination) {
        return NMMSetError(error, 24, @"invalid expert batch copy arguments");
    }
    uint8_t *cursor = (uint8_t *)destination;
    for (NSUInteger i = 0; i < count; i++) {
        if (!NMMExpertLayerCopyExpert(file, indices[i], cursor, error)) {
            return NO;
        }
        cursor += file->layout.expertSize;
    }
    return YES;
}

BOOL NMMExpertLayerCopyComponent(const NMMExpertLayerFile *file,
                                 NSUInteger expertIndex,
                                 const char *componentName,
                                 void *destination,
                                 NSError **error) {
    if (!file || !componentName || !destination) {
        return NMMSetError(error, 25, @"invalid component copy arguments");
    }

    NMMExpertComponentLayout component;
    if (!NMMExpertLayoutFindComponent(&file->layout, componentName, &component)) {
        return NMMSetError(error, 26, [NSString stringWithFormat:@"unknown expert component %s", componentName]);
    }
    if (expertIndex >= file->layout.numExperts) {
        return NMMSetError(error, 27, @"expert index out of range");
    }

    size_t offset = (size_t)expertIndex * file->layout.expertSize + component.offset;
    if (offset + component.size > file->fileSize) {
        return NMMSetError(error, 28, @"component offset exceeds file size");
    }

    if (file->mapped && file->mmapBase) {
        memcpy(destination, (uint8_t *)file->mmapBase + offset, component.size);
        return YES;
    }

    return NMMReadFullyAtOffset(file->fd, destination, component.size, (off_t)offset, error);
}

nmoe_expert_store *nmoe_expert_store_open(const char *model_path, int quant_bits, int quiet) {
    if (model_path == NULL || model_path[0] == '\0') {
        return NULL;
    }

    int effective_bits = (quant_bits == 2) ? 2 : 4;

    nmoe_expert_store *store = calloc(1, sizeof(*store));
    if (store == NULL) {
        return NULL;
    }

    store->model_path = strdup(model_path);
    if (store->model_path == NULL) {
        free(store);
        return NULL;
    }

    store->quant_bits = effective_bits;
    store->quiet = quiet;
    store->layout.expert_size = nmoe_expert_active_size(effective_bits);
    store->layout.num_layers = 40;
    store->layout.num_experts = 256;

    if (!quiet) {
        fprintf(stderr, "experts: model=%s quant=%d active_size=%zu\n",
                store->model_path,
                effective_bits,
                store->layout.expert_size);
    }
    return store;
}

void nmoe_expert_store_close(nmoe_expert_store *store) {
    if (store == NULL) {
        return;
    }
    free(store->model_path);
    free(store);
}

size_t nmoe_expert_active_size(int quant_bits) {
    int effective_bits = (quant_bits == 2) ? 2 : 4;
    return NMMExpertActiveSizeForBits((NSUInteger)effective_bits);
}

const nmoe_expert_layout *nmoe_expert_layout_for_bits(int quant_bits) {
    static nmoe_expert_layout q4_layout = {0};
    static nmoe_expert_layout q2_layout = {0};
    static BOOL initialized = NO;

    if (!initialized) {
        q4_layout.expert_size = NMMExpertActiveSizeForBits(NMMExpertQuantBitsQ4);
        q4_layout.num_layers = 40;
        q4_layout.num_experts = 256;

        q2_layout.expert_size = NMMExpertActiveSizeForBits(NMMExpertQuantBitsQ2);
        q2_layout.num_layers = 40;
        q2_layout.num_experts = 256;
        initialized = YES;
    }

    return quant_bits == 2 ? &q2_layout : &q4_layout;
}
