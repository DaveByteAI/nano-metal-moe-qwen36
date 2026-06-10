#include "nmoe/backend.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static NSString *const kNMOEKernelNames[NMOE_BACKEND_KERNEL_COUNT] = {
    @"nmoe_dequant_matvec_q4",
    @"nmoe_dequant_matvec_q2",
    @"nmoe_rms_norm",
    @"nmoe_rope",
    @"nmoe_attn_scores_batched",
    @"nmoe_attn_softmax_batched",
    @"nmoe_attn_values_batched",
    @"nmoe_sigmoid_gate",
    @"nmoe_conv1d_step",
    @"nmoe_gated_delta_net_step",
    @"nmoe_moe_expert_gate_up",
    @"nmoe_moe_expert_down",
    @"nmoe_moe_combine",
    @"nmoe_dequant_row_q4",
    @"nmoe_dequant_row_q2",
    @"nmoe_argmax_top1",
    @"nmoe_rms_norm_apply_bf16",
    @"nmoe_rms_norm_apply_f32",
    @"nmoe_rms_norm_qk",
    @"nmoe_compute_decay_beta",
    @"nmoe_gated_rms_norm",
    @"nmoe_residual_add",
    @"nmoe_weighted_expert_sum",
    @"nmoe_copy_f32",
    @"nmoe_route_topk",
    @"nmoe_expert_gate_up_q4",
    @"nmoe_expert_gate_up_q4_batched",
    @"nmoe_expert_down_q4_batched",
    @"nmoe_expert_gate_up_q4_routed",
    @"nmoe_expert_down_q4_routed",
    @"nmoe_weighted_expert_sum_routed",
    @"nmoe_expert_down_combine_q4",
    @"nmoe_route_shared_q4",
    @"nmoe_lm_head_argmax_q4",
    @"nmoe_lm_head_argmax_reduce",
    @"nmoe_full_qk_prep",
};

static NSString *const kNMOEWeightBufferLabel = @"model_weights";
static NSString *const kNMOEBackendErrorDomain = @"nmoe.backend.metal";

@interface NMOEMetalBackendImpl : NSObject

@property(nonatomic, assign) BOOL quiet;
@property(nonatomic, strong, nullable) id<MTLDevice> device;
@property(nonatomic, strong, nullable) id<MTLCommandQueue> queue;
@property(nonatomic, strong, nullable) id<MTLLibrary> library;
@property(nonatomic, strong) NSMutableDictionary<NSString *, id<MTLComputePipelineState>> *pipelines;
@property(nonatomic, strong) NSMutableDictionary<NSString *, id<MTLBuffer>> *sharedBuffers;
@property(nonatomic, strong) NSMutableArray<id<MTLBuffer>> *linearStateBuffers;
@property(nonatomic, strong, nullable) id<MTLBuffer> weightBuffer;
@property(nonatomic, copy, nullable) NSString *kernelSourcePath;

@end

@implementation NMOEMetalBackendImpl

- (instancetype)init {
    self = [super init];
    if (self != nil) {
        _pipelines = [[NSMutableDictionary alloc] init];
        _sharedBuffers = [[NSMutableDictionary alloc] init];
        _linearStateBuffers = [[NSMutableArray alloc] init];
    }
    return self;
}

@end

struct nmoe_backend {
    void *impl;
};

static NMOEMetalBackendImpl *NMOEBackendImpl(nmoe_backend *backend) {
    if (backend == NULL || backend->impl == NULL) {
        return nil;
    }
    return (__bridge NMOEMetalBackendImpl *)backend->impl;
}

static NSString *NMOEStringFromCString(const char *label) {
    if (label == NULL || label[0] == '\0') {
        return nil;
    }
    NSString *value = [NSString stringWithUTF8String:label];
    if (value == nil) {
        value = [NSString stringWithFormat:@"buffer-%p", label];
    }
    return value;
}

static NSString *NMOEKernelNameForKind(nmoe_backend_kernel_kind kind) {
    if (kind < 0 || kind >= NMOE_BACKEND_KERNEL_COUNT) {
        return nil;
    }
    return kNMOEKernelNames[kind];
}

const char *nmoe_backend_kernel_name(nmoe_backend_kernel_kind kind) {
    NSString *name = NMOEKernelNameForKind(kind);
    return name.UTF8String;
}

static NSError *NMOEMakeError(NSInteger code, NSString *message) {
    return [NSError errorWithDomain:kNMOEBackendErrorDomain
                               code:code
                           userInfo:@{NSLocalizedDescriptionKey : message ?: @"Metal backend error."}];
}

static NSArray<NSString *> *NMOEKernelSourceSearchPaths(void) {
    NSFileManager *fileManager = [NSFileManager defaultManager];
    NSMutableArray<NSString *> *paths = [[NSMutableArray alloc] init];

    void (^appendPath)(NSString *) = ^(NSString *candidate) {
        if (candidate == nil || candidate.length == 0) {
            return;
        }
        if (![paths containsObject:candidate] && [fileManager fileExistsAtPath:candidate]) {
            [paths addObject:candidate];
        }
    };

    NSString *cwd = fileManager.currentDirectoryPath;
    if (cwd.length > 0) {
        appendPath([cwd stringByAppendingPathComponent:@"metal/kernels.metal"]);
        appendPath([[cwd stringByAppendingPathComponent:@".."] stringByAppendingPathComponent:@"metal/kernels.metal"]);
    }

    NSString *executablePath = [[NSBundle mainBundle] executablePath];
    if (executablePath.length > 0) {
        NSString *exeDir = [executablePath stringByDeletingLastPathComponent];
        appendPath([exeDir stringByAppendingPathComponent:@"metal/kernels.metal"]);
        appendPath([[exeDir stringByAppendingPathComponent:@".."] stringByAppendingPathComponent:@"metal/kernels.metal"]);
    }

    return paths;
}

static NSString *NMOEResolveKernelSourcePath(void) {
    NSArray<NSString *> *paths = NMOEKernelSourceSearchPaths();
    return paths.count > 0 ? paths.firstObject : nil;
}

static id<MTLBuffer> NMOEMakeSharedBuffer(id<MTLDevice> device,
                                          const void *data,
                                          size_t size,
                                          NSString *label,
                                          BOOL zeroFill) {
    if (device == nil || size == 0) {
        return nil;
    }

    id<MTLBuffer> buffer = nil;
    MTLResourceOptions options = MTLResourceStorageModeShared;
    if (data != NULL) {
        buffer = [device newBufferWithBytesNoCopy:(void *)data
                                           length:size
                                          options:options
                                      deallocator:nil];
    } else {
        buffer = [device newBufferWithLength:size options:options];
        if (buffer != nil && zeroFill) {
            memset((void *)buffer.contents, 0, size);
        }
    }

    if (buffer != nil) {
        buffer.label = label;
    }
    return buffer;
}

static void NMOERegisterBuffer(NMOEMetalBackendImpl *impl,
                               NSString *label,
                               id<MTLBuffer> buffer,
                               BOOL resettable) {
    if (impl == nil || label == nil || buffer == nil) {
        return;
    }

    id<MTLBuffer> previous = impl.sharedBuffers[label];
    if (previous != nil) {
        NSUInteger index = [impl.linearStateBuffers indexOfObjectIdenticalTo:previous];
        if (index != NSNotFound) {
            [impl.linearStateBuffers removeObjectAtIndex:index];
        }
    }

    impl.sharedBuffers[label] = buffer;
    if (resettable) {
        [impl.linearStateBuffers addObject:buffer];
    }
    if ([label isEqualToString:kNMOEWeightBufferLabel]) {
        impl.weightBuffer = buffer;
    }
}

static BOOL NMOECompilePipelines(NMOEMetalBackendImpl *impl, NSError **errorOut) {
    if (impl.device == nil || impl.library == nil) {
        if (errorOut != NULL) {
            *errorOut = NMOEMakeError(1, @"Metal device or library is unavailable.");
        }
        return NO;
    }

    NSMutableDictionary<NSString *, id<MTLComputePipelineState>> *pipelines =
        [[NSMutableDictionary alloc] initWithCapacity:NMOE_BACKEND_KERNEL_COUNT];

    for (NSInteger kind = 0; kind < NMOE_BACKEND_KERNEL_COUNT; ++kind) {
        NSString *name = NMOEKernelNameForKind((nmoe_backend_kernel_kind)kind);
        if (name == nil) {
            if (errorOut != NULL) {
                *errorOut = NMOEMakeError(2, [NSString stringWithFormat:@"Unknown kernel kind %ld.", (long)kind]);
            }
            return NO;
        }

        id<MTLFunction> function = [impl.library newFunctionWithName:name];
        if (function == nil) {
            if (errorOut != NULL) {
                *errorOut = NMOEMakeError(3, [NSString stringWithFormat:@"Missing Metal entrypoint '%@'.", name]);
            }
            return NO;
        }

        NSError *pipelineError = nil;
        id<MTLComputePipelineState> pipeline = [impl.device newComputePipelineStateWithFunction:function
                                                                                         error:&pipelineError];
        if (pipeline == nil) {
            if (errorOut != NULL) {
                *errorOut = pipelineError ?: NMOEMakeError(4, [NSString stringWithFormat:@"Failed to compile pipeline '%@'.", name]);
            }
            return NO;
        }

        pipelines[name] = pipeline;
    }

    impl.pipelines = pipelines;
    return YES;
}

static BOOL NMOEInitializeBackend(NMOEMetalBackendImpl *impl, NSError **errorOut) {
    if (impl == nil) {
        if (errorOut != NULL) {
            *errorOut = NMOEMakeError(5, @"Backend implementation object is missing.");
        }
        return NO;
    }

    impl.device = MTLCreateSystemDefaultDevice();
    if (impl.device == nil) {
        if (errorOut != NULL) {
            *errorOut = NMOEMakeError(6, @"Metal system device is unavailable.");
        }
        return NO;
    }

    impl.queue = [impl.device newCommandQueue];
    if (impl.queue == nil) {
        if (errorOut != NULL) {
            *errorOut = NMOEMakeError(7, @"Failed to create Metal command queue.");
        }
        return NO;
    }

    NSString *kernelSourcePath = NMOEResolveKernelSourcePath();
    if (kernelSourcePath == nil) {
        if (errorOut != NULL) {
            *errorOut = NMOEMakeError(8, @"Unable to locate metal/kernels.metal.");
        }
        return NO;
    }
    impl.kernelSourcePath = kernelSourcePath;

    NSError *sourceError = nil;
    NSString *source = [NSString stringWithContentsOfFile:kernelSourcePath
                                                 encoding:NSUTF8StringEncoding
                                                    error:&sourceError];
    if (source == nil) {
        if (errorOut != NULL) {
            *errorOut = sourceError ?: NMOEMakeError(9, [NSString stringWithFormat:@"Failed to read '%@'.", kernelSourcePath]);
        }
        return NO;
    }

    MTLCompileOptions *compileOptions = [[MTLCompileOptions alloc] init];
    compileOptions.mathMode = MTLMathModeFast;

    NSError *libraryError = nil;
    impl.library = [impl.device newLibraryWithSource:source
                                             options:compileOptions
                                               error:&libraryError];
    if (impl.library == nil) {
        if (errorOut != NULL) {
            *errorOut = libraryError ?: NMOEMakeError(10, [NSString stringWithFormat:@"Failed to compile Metal library from '%@'.", kernelSourcePath]);
        }
        return NO;
    }

    if (!NMOECompilePipelines(impl, errorOut)) {
        return NO;
    }

    return YES;
}

nmoe_backend *nmoe_backend_create(int quiet) {
    nmoe_backend *backend = calloc(1, sizeof(*backend));
    if (backend == NULL) {
        return NULL;
    }

    NMOEMetalBackendImpl *impl = [[NMOEMetalBackendImpl alloc] init];
    impl.quiet = quiet != 0;

    NSError *error = nil;
    if (!NMOEInitializeBackend(impl, &error)) {
        if (!impl.quiet) {
            fprintf(stderr, "nmoe metal backend init failed: %s\n", error.localizedDescription.UTF8String ?: "unknown error");
        }
        free(backend);
        return NULL;
    }

    backend->impl = (__bridge_retained void *)impl;

    if (!impl.quiet) {
        fprintf(stderr,
                "nmoe metal backend ready: device=%s, kernels=%lu, shared-memory buffers enabled\n",
                impl.device.name.UTF8String ?: "unknown",
                (unsigned long)impl.pipelines.count);
    }

    return backend;
}

void nmoe_backend_destroy(nmoe_backend *backend) {
    if (backend == NULL) {
        return;
    }
    if (backend->impl != NULL) {
        CFBridgingRelease(backend->impl);
        backend->impl = NULL;
    }
    free(backend);
}

void nmoe_backend_reset_linear_state(nmoe_backend *backend) {
    NMOEMetalBackendImpl *impl = NMOEBackendImpl(backend);
    if (impl == nil) {
        return;
    }

    for (id<MTLBuffer> buffer in impl.linearStateBuffers) {
        if (buffer == nil || buffer.length == 0) {
            continue;
        }
        memset((void *)buffer.contents, 0, buffer.length);
    }
}

void nmoe_backend_set_weight_buffer(nmoe_backend *backend, void *data, size_t size) {
    (void)nmoe_backend_register_shared_buffer(backend, kNMOEWeightBufferLabel.UTF8String, data, size);
}

void *nmoe_backend_device(nmoe_backend *backend) {
    NMOEMetalBackendImpl *impl = NMOEBackendImpl(backend);
    return impl.device != nil ? (__bridge void *)impl.device : NULL;
}

void *nmoe_backend_command_queue(nmoe_backend *backend) {
    NMOEMetalBackendImpl *impl = NMOEBackendImpl(backend);
    return impl.queue != nil ? (__bridge void *)impl.queue : NULL;
}

void *nmoe_backend_library(nmoe_backend *backend) {
    NMOEMetalBackendImpl *impl = NMOEBackendImpl(backend);
    return impl.library != nil ? (__bridge void *)impl.library : NULL;
}

void *nmoe_backend_pipeline_state(nmoe_backend *backend, nmoe_backend_kernel_kind kind) {
    NMOEMetalBackendImpl *impl = NMOEBackendImpl(backend);
    if (impl == nil) {
        return NULL;
    }

    NSString *name = NMOEKernelNameForKind(kind);
    if (name == nil) {
        return NULL;
    }

    id<MTLComputePipelineState> pipeline = impl.pipelines[name];
    if (pipeline == nil && impl.device != nil && impl.library != nil) {
        id<MTLFunction> function = [impl.library newFunctionWithName:name];
        if (function != nil) {
            NSError *error = nil;
            pipeline = [impl.device newComputePipelineStateWithFunction:function error:&error];
            if (pipeline != nil) {
                impl.pipelines[name] = pipeline;
            } else if (!impl.quiet) {
                fprintf(stderr,
                        "nmoe metal backend pipeline compile failed for %s: %s\n",
                        name.UTF8String,
                        error.localizedDescription.UTF8String ?: "unknown error");
            }
        }
    }

    return pipeline != nil ? (__bridge void *)pipeline : NULL;
}

void *nmoe_backend_weight_buffer(nmoe_backend *backend) {
    NMOEMetalBackendImpl *impl = NMOEBackendImpl(backend);
    return impl.weightBuffer != nil ? (__bridge void *)impl.weightBuffer : NULL;
}

void *nmoe_backend_lookup_shared_buffer(nmoe_backend *backend, const char *label) {
    NMOEMetalBackendImpl *impl = NMOEBackendImpl(backend);
    NSString *key = NMOEStringFromCString(label);
    if (impl == nil || key == nil) {
        return NULL;
    }
    id<MTLBuffer> buffer = impl.sharedBuffers[key];
    return buffer != nil ? (__bridge void *)buffer : NULL;
}

void *nmoe_backend_register_shared_buffer(nmoe_backend *backend, const char *label, const void *data, size_t size) {
    NMOEMetalBackendImpl *impl = NMOEBackendImpl(backend);
    NSString *key = NMOEStringFromCString(label);
    if (impl == nil || key == nil || size == 0) {
        return NULL;
    }

    id<MTLBuffer> buffer = NMOEMakeSharedBuffer(impl.device, data, size, key, NO);
    if (buffer == nil) {
        if (!impl.quiet) {
            fprintf(stderr, "nmoe metal backend could not create shared buffer '%s'\n", label ?: "(null)");
        }
        return NULL;
    }

    NMOERegisterBuffer(impl, key, buffer, NO);
    return (__bridge void *)buffer;
}

void *nmoe_backend_register_state_buffer(nmoe_backend *backend, const char *label, size_t size) {
    NMOEMetalBackendImpl *impl = NMOEBackendImpl(backend);
    NSString *key = NMOEStringFromCString(label);
    if (impl == nil || key == nil || size == 0) {
        return NULL;
    }

    id<MTLBuffer> buffer = NMOEMakeSharedBuffer(impl.device, NULL, size, key, YES);
    if (buffer == nil) {
        if (!impl.quiet) {
            fprintf(stderr, "nmoe metal backend could not create state buffer '%s'\n", label ?: "(null)");
        }
        return NULL;
    }

    NMOERegisterBuffer(impl, key, buffer, YES);
    return (__bridge void *)buffer;
}
