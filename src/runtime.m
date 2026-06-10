#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "nmoe/runtime.h"
#include "nmoe/math.h"

#include <fcntl.h>
#include <math.h>
#include <stdarg.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/utsname.h>
#include <time.h>
#include <unistd.h>

extern NSString *NMOEChatSystemPrompt(void);
extern NSString *NMOEChatUserPrompt(NSString *userText);
static void NMOEWriteDecodedToken(nmoe_runtime *rt, uint32_t token);
static int NMOERunChat(nmoe_runtime *rt);
static NSString *NMOEModelTensorName(NSString *suffix);
static const void *NMOETensorPointer(nmoe_weight_file *weights, NSString *name, NSString *dtype, NSError **error);
static void NMOETraceEscapedString(FILE *stream, const char *text);
static void NMOETraceToken(nmoe_runtime *rt, FILE *stream, uint32_t token);
static void NMOETracePromptTokens(nmoe_runtime *rt, const uint32_t *tokens, int tokenCount);
static void NMOETraceGenerationToken(nmoe_runtime *rt, uint32_t token);

static NSString *const NMOERuntimeErrorDomain = @"nmoe.runtime";

enum {
    kNMOELayers = 40,
    kNMOEHiddenDim = 2048,
    kNMOEFullHeads = 16,
    kNMOEFullKVHeads = 2,
    kNMOEHeadDim = 256,
    kNMOERotaryDim = 64,
    kNMOELinearKVHeads = 16,
    kNMOELinearVHeads = 32,
    kNMOELinearKeyDim = 128,
    kNMOELinearValueDim = 128,
    kNMOELinearConvDim = 8192,
    kNMOEMaxExperts = 8,
};
static const uint32_t kNMOEEOS1 = 248046;
static const uint32_t kNMOEEOS2 = 248044;
static const uint32_t kNMOEThinkStart = 248068;
static const uint32_t kNMOEThinkEnd = 248069;

static double NMOENowSeconds(void);
static BOOL NMOEUseRoutedExperts(void);
static uint32_t NMOEExpertRowsPerThreadgroup(id<MTLComputePipelineState> pipeline);
static uint32_t NMOEMatvecRowsPerThreadgroup(id<MTLComputePipelineState> pipeline);

typedef struct {
    double context;
    double route;
    double expertFetch;
    double expert;
    double contextSync;
    double deferredWait;
    double total;
    size_t layerCount;
    double fullContext;
    double fullSync;
    double fullTotal;
    size_t fullLayerCount;
    double linearContext;
    double linearSync;
    double linearTotal;
    size_t linearLayerCount;
} NMOEPerfStats;

static BOOL NMOEFullAttentionStepMetal(nmoe_runtime *rt,
                                       int layerIndex,
                                       const float *residual,
                                       float *layerOutput,
                                       size_t position,
                                       NMOEPerfStats *stats);
static BOOL NMOELinearAttentionStepMetal(nmoe_runtime *rt,
                                         int layerIndex,
                                         const float *residual,
                                         float *layerOutput,
                                         size_t position,
                                         NMOEPerfStats *stats);

// Deferred expert stage: expert commit is submitted without wait,
// so the GPU can process experts while CPU starts the next layer.
typedef struct {
    BOOL active;
    BOOL gpuCombined;
    BOOL nextInputNormReady;
    id<MTLCommandBuffer> cmdExperts;
    __strong id<MTLBuffer> expertBuffers[8];
    float *hiddenPtr;
    float expertWeights[8];
    int valid[8];
    int actualK;
    float sharedGateScore;
    int layerIndex;
    int nextInputNormLayer;
} NMOEDeferredExpertState;

static NMOEDeferredExpertState g_deferredExperts = {
    .active = NO,
    .gpuCombined = NO,
    .cmdExperts = nil,
};

static void NMOEFinalizeDeferredExperts(nmoe_runtime *rt) {
    (void)rt;
    if (!g_deferredExperts.active) return;
    if (g_deferredExperts.cmdExperts != nil) {
        [g_deferredExperts.cmdExperts waitUntilCompleted];
    }
    g_deferredExperts.active = NO;
    g_deferredExperts.gpuCombined = NO;
    g_deferredExperts.nextInputNormReady = NO;
    g_deferredExperts.cmdExperts = nil;
    g_deferredExperts.nextInputNormLayer = -1;
    for (int i = 0; i < 8; ++i) {
        g_deferredExperts.expertBuffers[i] = nil;
    }
}

static void NMOECancelDeferredExperts(void) {
    g_deferredExperts.active = NO;
    g_deferredExperts.gpuCombined = NO;
    g_deferredExperts.nextInputNormReady = NO;
    g_deferredExperts.cmdExperts = nil;
    g_deferredExperts.nextInputNormLayer = -1;
    for (int i = 0; i < 8; ++i) {
        g_deferredExperts.expertBuffers[i] = nil;
    }
}

static BOOL NMOEDeferredInputNormReadyForLayer(int layerIndex) {
    return g_deferredExperts.active &&
           g_deferredExperts.gpuCombined &&
           g_deferredExperts.nextInputNormReady &&
           g_deferredExperts.nextInputNormLayer == layerIndex;
}

typedef struct {
    const uint32_t *weight;
    const uint16_t *scales;
    const uint16_t *biases;
} NMOEQuantTensor;

typedef struct {
    const uint16_t *weight;
} NMOEBF16Tensor;

typedef struct {
    const float *value;
} NMOEF32Tensor;

typedef struct {
    size_t gate_weight;
    size_t gate_scales;
    size_t gate_biases;
    size_t up_weight;
    size_t up_scales;
    size_t up_biases;
    size_t down_weight;
    size_t down_scales;
    size_t down_biases;
} NMOEExpertOffsets;

typedef struct {
    int fd;
    void *base;
    size_t size;
    BOOL mapped;
    size_t expertSize;
    void *metalBuffer;
    void *expertMetalBuffers[256];
} NMOEExpertLayerFile;

static const NMOEExpertOffsets *NMOEExpertOffsetsForBits(int bits);

typedef struct {
    BOOL isFull;
    NMOEBF16Tensor inputNorm;
    NMOEBF16Tensor postNorm;
    NMOEQuantTensor routerGate;
    NMOEQuantTensor sharedGateProj;
    NMOEQuantTensor sharedUpProj;
    NMOEQuantTensor sharedDownProj;
    NMOEQuantTensor sharedGateScore;
    union {
        struct {
            NMOEQuantTensor qProj;
            NMOEQuantTensor kProj;
            NMOEQuantTensor vProj;
            NMOEQuantTensor oProj;
            NMOEBF16Tensor qNorm;
            NMOEBF16Tensor kNorm;
        } full;
        struct {
            NMOEQuantTensor qkvProj;
            NMOEQuantTensor zProj;
            NMOEQuantTensor betaProj;
            NMOEQuantTensor alphaProj;
            NMOEBF16Tensor convWeight;
            NMOEF32Tensor ALog;
            NMOEBF16Tensor dtBias;
            NMOEBF16Tensor normWeight;
            NMOEQuantTensor outProj;
        } linear;
    } u;
} NMOELayerWeights;

typedef struct {
    void *fullKCache;
    void *fullVCache;
    void *linearConvState;
    void *linearDeltaState;
    NMOEExpertLayerFile expertFile;
} NMOELayerState;

struct nmoe_runtime {
    nmoe_app_config cfg;
    nmoe_weight_file *weights;
    nmoe_tokenizer *tokenizer;
    nmoe_vocab *vocab;
    nmoe_backend *backend;
    nmoe_expert_store *expertStore;
    char *modelPath;
    char *expertDirectory;
    int quantBits;
    size_t sequenceCapacity;
    size_t sequencePosition;
    BOOL quiet;
    BOOL cpuLinear;
    BOOL traceTokens;
    BOOL inThink;
    size_t thinkCount;
    NMOELayerWeights layers[kNMOELayers];
    NMOELayerState layerState[kNMOELayers];
    void *pipelineStates[NMOE_BACKEND_KERNEL_COUNT];
    void *hiddenBuffers[2];
    void *normBuffer;
    void *fullQProjBuffer;
    void *fullKProjBuffer;
    void *fullVProjBuffer;
    void *fullAttnBuffer;
    void *linearQkvBuffer;
    void *linearZBuffer;
    void *linearBetaBuffer;
    void *linearAlphaBuffer;
    void *linearConvBuffer;
    void *linearOutBuffer;
    void *routerScoresBuffer;
    void *routerProbsBuffer;
    void *routeIndicesBuffer;
    void *routeWeightsBuffer;
    void *sharedGateBuffer;
    void *sharedUpBuffer;
    void *sharedOutBuffer;
    void *expertGateBuffer;
    void *expertUpBuffer;
    void *expertActBuffer;
    void *expertOutBuffer;
    void *attnScoresBuffer;
    void *attnProbsBuffer;
    void *logitsBuffer;
    void *lmHeadPartialIndicesBuffer;
    void *nextTokenBuffer;
    void *promptTokensBuffer;
    void *embeddingBuffer;
};

static inline id<MTLBuffer> NMOEBridgeBuffer(void *buffer) {
    return (__bridge id<MTLBuffer>)buffer;
}

static inline float *NMOEFloatBuffer(void *buffer) {
    id<MTLBuffer> mtl = NMOEBridgeBuffer(buffer);
    return mtl != nil ? (float *)mtl.contents : NULL;
}

static inline const uint32_t *NMOEU32Buffer(void *buffer) {
    id<MTLBuffer> mtl = NMOEBridgeBuffer(buffer);
    return mtl != nil ? (const uint32_t *)mtl.contents : NULL;
}

static BOOL NMOEWeightPointerOffset(nmoe_runtime *rt, const void *ptr, NSUInteger *outOffset) {
    if (rt == NULL || rt->weights == NULL || rt->weights->data == NULL || ptr == NULL || outOffset == NULL) return NO;
    const char *base = (const char *)rt->weights->data;
    const char *p = (const char *)ptr;
    if (p < base || p >= base + rt->weights->size) return NO;
    *outOffset = (NSUInteger)(p - base);
    return YES;
}

typedef struct {
    uint32_t out_rows;
    uint32_t in_dim;
    uint32_t packed_cols;
    uint32_t group_size;
    uint32_t rows_per_tg;
    uint32_t reserved0;
    uint32_t reserved1;
    uint32_t reserved2;
} NMOEDequantMatvecArgs;

typedef struct {
    uint32_t dim;
    uint32_t add_one;
    float epsilon;
    float reserved;
} NMOERMSNormArgs;

typedef struct {
    uint32_t rotary_dim;
    uint32_t position;
    float theta;
    float reserved;
} NMOERopeArgs;

typedef struct {
    uint32_t count;
    uint32_t reserved0;
    uint32_t reserved1;
    uint32_t reserved2;
} NMOESigmoidGateArgs;

typedef struct {
    uint32_t head_dim;
    uint32_t q_stride;
    uint32_t full_heads;
    uint32_t kv_heads;
    uint32_t rotary_dim;
    uint32_t position;
    uint32_t add_one;
    float epsilon;
    float theta;
    uint32_t reserved0;
    uint32_t reserved1;
    uint32_t reserved2;
} NMOEFullQKPrepArgs;

typedef struct {
    uint32_t inDim;
    uint32_t packedCols;
    uint32_t groupSize;
    uint32_t reserved0;
} NMOEDequantRowArgs;

typedef struct {
    uint32_t seqLen;
    uint32_t seqStride;
    uint32_t headDim;
    uint32_t qStride;
    uint32_t kvStride;
    uint32_t cacheStride;
    uint32_t fullHeads;
    uint32_t fullKVHeads;
    uint32_t position;
    uint32_t reserved0;
    float invScale;
    float reserved1;
} NMOEAttentionArgs;

typedef struct {
    uint32_t dim;
    uint32_t stateStride;
    uint32_t reserved0;
    uint32_t reserved1;
} NMOELinearConv1DArgs;

typedef struct {
    uint32_t vHeads;
    uint32_t kvHeads;
    uint32_t valueDim;
    uint32_t keyDim;
    float qScale;
    float kScale;
    float epsilon;
    float reserved0;
} NMOEGatedDeltaNetArgs;

typedef struct {
    uint32_t count;
    float weight;
    uint32_t reserved0;
    uint32_t reserved1;
} NMOEMoeCombineArgs;

typedef struct {
    uint32_t count;
    uint32_t reserved0;
    uint32_t reserved1;
    uint32_t reserved2;
} NMOEArgmaxArgs;

typedef struct {
    uint32_t vocab_size;
    uint32_t in_dim;
    uint32_t packed_cols;
    uint32_t group_size;
    uint32_t rows_per_tg;
    uint32_t partial_count;
    uint32_t reserved0;
    uint32_t reserved1;
} NMOELmHeadArgmaxArgs;

typedef struct {
    uint32_t count;
    uint32_t k;
    uint32_t reserved0;
    uint32_t reserved1;
} NMOERouteTopKArgs;

typedef struct {
    uint32_t expert_count;
    uint32_t out_rows;
    uint32_t in_dim;
    uint32_t packed_cols;
    uint32_t group_size;
    uint32_t gate_weight;
    uint32_t gate_scales;
    uint32_t gate_biases;
    uint32_t up_weight;
    uint32_t up_scales;
    uint32_t up_biases;
    uint32_t down_weight;
    uint32_t down_scales;
    uint32_t down_biases;
    uint32_t act_stride;
    uint32_t expert_size;
    uint32_t rows_per_tg;
    uint32_t reserved0;
    uint32_t reserved1;
    uint32_t reserved2;
} NMOEExpertBatchedArgs;

typedef struct {
    uint32_t hidden_dim;
    uint32_t packed_cols;
    uint32_t group_size;
    uint32_t rows_per_tg;
} NMOERouteSharedQ4Args;

static inline float *NMOEFloatBufferAtOffset(void *buffer, size_t offsetBytes) {
    id<MTLBuffer> mtl = NMOEBridgeBuffer(buffer);
    if (mtl == nil || offsetBytes >= mtl.length) {
        return NULL;
    }
    return (float *)((uint8_t *)mtl.contents + offsetBytes);
}

static inline void *NMOEHiddenBufferForPointer(nmoe_runtime *rt, const float *ptr) {
    if (rt == NULL || ptr == NULL) {
        return NULL;
    }
    float *hiddenA = NMOEFloatBuffer(rt->hiddenBuffers[0]);
    if (ptr == hiddenA) {
        return rt->hiddenBuffers[0];
    }
    return rt->hiddenBuffers[1];
}

static id<MTLBuffer> NMOEMakeTemporaryBuffer(nmoe_runtime *rt, const void *bytes, size_t length, NSString *label) {
    if (rt == NULL || rt->backend == NULL || bytes == NULL || length == 0) {
        return nil;
    }

    id<MTLDevice> device = (__bridge id<MTLDevice>)nmoe_backend_device(rt->backend);
    if (device == nil) {
        return nil;
    }

    id<MTLBuffer> buffer = [device newBufferWithBytes:bytes length:length options:MTLResourceStorageModeShared];
    if (buffer != nil) {
        buffer.label = label;
    }
    return buffer;
}

// Encode a kernel into an existing command buffer — no commit, no wait.
static BOOL NMOEEncodeKernel(id<MTLCommandBuffer> cmd,
                              id<MTLComputePipelineState> pipeline,
                              NSUInteger threadCount,
                              void (^configure)(id<MTLComputeCommandEncoder> encoder)) {
    if (cmd == nil || pipeline == nil || threadCount == 0) return NO;

    id<MTLComputeCommandEncoder> encoder = [cmd computeCommandEncoder];
    if (encoder == nil) return NO;

    [encoder setComputePipelineState:pipeline];
    if (configure != nil) configure(encoder);

    NSUInteger tgWidth = pipeline.threadExecutionWidth;
    if (tgWidth == 0) tgWidth = 1;
    if (tgWidth > pipeline.maxTotalThreadsPerThreadgroup)
        tgWidth = pipeline.maxTotalThreadsPerThreadgroup;
    if (tgWidth == 0) tgWidth = 1;

    [encoder dispatchThreads:MTLSizeMake(threadCount, 1, 1)
       threadsPerThreadgroup:MTLSizeMake(MIN(tgWidth, threadCount), 1, 1)];
    [encoder endEncoding];
    return YES;
}

// Encode a kernel with explicit threadgroup count (for kernels that use
// threadgroup_position_in_grid).
static BOOL NMOEEncodeKernelTG(id<MTLCommandBuffer> cmd,
                                id<MTLComputePipelineState> pipeline,
                                NSUInteger threadgroupCount,
                                NSUInteger threadsPerTG,
                                void (^configure)(id<MTLComputeCommandEncoder> encoder)) {
    if (cmd == nil || pipeline == nil || threadgroupCount == 0 || threadsPerTG == 0) return NO;

    id<MTLComputeCommandEncoder> encoder = [cmd computeCommandEncoder];
    if (encoder == nil) return NO;

    [encoder setComputePipelineState:pipeline];
    if (configure != nil) configure(encoder);

    [encoder dispatchThreadgroups:MTLSizeMake(threadgroupCount, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(threadsPerTG, 1, 1)];
    [encoder endEncoding];
    return YES;
}

// Legacy dispatch kernel with commit+wait (for single-kernel fire-and-forget cases).
static BOOL NMOEDispatchKernel1D(id<MTLCommandQueue> queue,
                                 id<MTLComputePipelineState> pipeline,
                                 NSUInteger threadCount,
                                 void (^configure)(id<MTLComputeCommandEncoder> encoder)) {
    if (queue == nil || pipeline == nil || threadCount == 0) return NO;

    id<MTLCommandBuffer> cmd = [queue commandBuffer];
    if (cmd == nil) return NO;

    if (!NMOEEncodeKernel(cmd, pipeline, threadCount, configure)) return NO;

    [cmd commit];
    [cmd waitUntilCompleted];
    return cmd.error == nil;
}

static BOOL NMOEDispatchKernelTG(id<MTLCommandQueue> queue,
                                 id<MTLComputePipelineState> pipeline,
                                 NSUInteger threadgroupCount,
                                 NSUInteger threadsPerTG,
                                 void (^configure)(id<MTLComputeCommandEncoder> encoder)) {
    if (queue == nil || pipeline == nil || threadgroupCount == 0 || threadsPerTG == 0) return NO;

    id<MTLCommandBuffer> cmd = [queue commandBuffer];
    if (cmd == nil) return NO;

    if (!NMOEEncodeKernelTG(cmd, pipeline, threadgroupCount, threadsPerTG, configure)) return NO;

    [cmd commit];
    [cmd waitUntilCompleted];
    return cmd.error == nil;
}

// --- Encode helpers (encode into an existing command buffer, no commit/wait) ---

static BOOL NMOEEncodeDequantMatVec(id<MTLCommandBuffer> cmd,
                                     nmoe_runtime *rt,
                                     NSString *baseName,
                                     void *inputBuffer, size_t inputOffset,
                                     void *outputBuffer, size_t outputOffset,
                                     size_t outRows, size_t inDim) {
    int bits = 4;
    NSString *weightName = NMOEModelTensorName([baseName stringByAppendingString:@".weight"]);
    id<MTLBuffer> wfBuf = (__bridge id<MTLBuffer>)nmoe_backend_weight_buffer(rt->backend);
    nmoe_tensor_info *wi = nmoe_weight_tensor_info(rt->weights, weightName.UTF8String);
    nmoe_tensor_info *si = nmoe_weight_tensor_info(rt->weights, NMOEModelTensorName([baseName stringByAppendingString:@".scales"]).UTF8String);
    nmoe_tensor_info *bi = nmoe_weight_tensor_info(rt->weights, NMOEModelTensorName([baseName stringByAppendingString:@".biases"]).UTF8String);
    if (wfBuf == nil || wi == NULL || si == NULL || bi == NULL) return NO;

    id<MTLBuffer> inputMTL = NMOEBridgeBuffer(inputBuffer);
    id<MTLBuffer> outputMTL = NMOEBridgeBuffer(outputBuffer);
    if (inputMTL == nil || outputMTL == nil) return NO;

    id<MTLComputePipelineState> pipeline = (__bridge id<MTLComputePipelineState>)nmoe_backend_pipeline_state(rt->backend,
        bits == 2 ? NMOE_BACKEND_KERNEL_DEQUANT_MATVEC_Q2 : NMOE_BACKEND_KERNEL_DEQUANT_MATVEC_Q4);
    if (pipeline == nil) return NO;

    uint32_t rowsPerTG = NMOEMatvecRowsPerThreadgroup(pipeline);
    uint32_t valuesPerWord = (uint32_t)(bits == 2 ? 16 : 8);
    uint32_t packedCols = (uint32_t)(inDim / valuesPerWord);
    NMOEDequantMatvecArgs args = {
        .out_rows = (uint32_t)outRows,
        .in_dim = (uint32_t)inDim,
        .packed_cols = packedCols,
        .group_size = 64u,
        .rows_per_tg = rowsPerTG,
    };

    NSUInteger groups = (NSUInteger)((outRows + rowsPerTG - 1u) / rowsPerTG);
    return NMOEEncodeKernelTG(cmd, pipeline, groups, (NSUInteger)rowsPerTG * 32u, ^(id<MTLComputeCommandEncoder> encoder) {
        [encoder setBuffer:wfBuf offset:wi->offset atIndex:0];
        [encoder setBuffer:wfBuf offset:si->offset atIndex:1];
        [encoder setBuffer:wfBuf offset:bi->offset atIndex:2];
        [encoder setBuffer:inputMTL offset:inputOffset atIndex:3];
        [encoder setBuffer:outputMTL offset:outputOffset atIndex:4];
        [encoder setBytes:&args length:sizeof(args) atIndex:5];
    });
}

static BOOL NMOEEncodeRMSNormTensor(id<MTLCommandBuffer> cmd,
                                    nmoe_runtime *rt,
                                    const NMOEBF16Tensor *weight,
                                    void *inputBuffer, size_t inputOffset,
                                    void *outputBuffer, size_t outputOffset,
                                    size_t dim, BOOL addOne, float eps) {
    id<MTLBuffer> inputMTL = NMOEBridgeBuffer(inputBuffer);
    id<MTLBuffer> outputMTL = NMOEBridgeBuffer(outputBuffer);
    id<MTLBuffer> weightBuf = (__bridge id<MTLBuffer>)nmoe_backend_weight_buffer(rt->backend);
    id<MTLComputePipelineState> pipe = (__bridge id<MTLComputePipelineState>)nmoe_backend_pipeline_state(rt->backend, NMOE_BACKEND_KERNEL_RMS_NORM);
    NSUInteger weightOff = 0;
    if (inputMTL == nil || outputMTL == nil || weightBuf == nil || pipe == nil ||
        !NMOEWeightPointerOffset(rt, weight != NULL ? weight->weight : NULL, &weightOff)) return NO;

    NMOERMSNormArgs args = {
        .dim = (uint32_t)dim, .add_one = addOne ? 1u : 0u,
        .epsilon = eps, .reserved = 0.0f,
    };
    return NMOEEncodeKernelTG(cmd, pipe, 1, 256, ^(id<MTLComputeCommandEncoder> encoder) {
        [encoder setBuffer:inputMTL offset:inputOffset atIndex:0];
        [encoder setBuffer:weightBuf offset:weightOff atIndex:1];
        [encoder setBuffer:outputMTL offset:outputOffset atIndex:2];
        [encoder setBytes:&args length:sizeof(args) atIndex:3];
    });
}

static BOOL NMOEEncodeDequantMatVecTensor(id<MTLCommandBuffer> cmd,
                                          nmoe_runtime *rt,
                                          const NMOEQuantTensor *tensor,
                                          void *inputBuffer, size_t inputOffset,
                                          void *outputBuffer, size_t outputOffset,
                                          size_t outRows, size_t inDim) {
    if (cmd == nil || rt == NULL || tensor == NULL) return NO;
    id<MTLBuffer> wfBuf = (__bridge id<MTLBuffer>)nmoe_backend_weight_buffer(rt->backend);
    id<MTLBuffer> inputMTL = NMOEBridgeBuffer(inputBuffer);
    id<MTLBuffer> outputMTL = NMOEBridgeBuffer(outputBuffer);
    id<MTLComputePipelineState> pipeline = (__bridge id<MTLComputePipelineState>)nmoe_backend_pipeline_state(rt->backend,
        NMOE_BACKEND_KERNEL_DEQUANT_MATVEC_Q4);
    NSUInteger weightOff = 0, scalesOff = 0, biasesOff = 0;
    if (wfBuf == nil || inputMTL == nil || outputMTL == nil || pipeline == nil ||
        !NMOEWeightPointerOffset(rt, tensor->weight, &weightOff) ||
        !NMOEWeightPointerOffset(rt, tensor->scales, &scalesOff) ||
        !NMOEWeightPointerOffset(rt, tensor->biases, &biasesOff)) return NO;

    uint32_t rowsPerTG = NMOEMatvecRowsPerThreadgroup(pipeline);
    NMOEDequantMatvecArgs args = {
        .out_rows = (uint32_t)outRows,
        .in_dim = (uint32_t)inDim,
        .packed_cols = (uint32_t)(inDim / 8u),
        .group_size = 64u,
        .rows_per_tg = rowsPerTG,
    };

    NSUInteger groups = (NSUInteger)((outRows + rowsPerTG - 1u) / rowsPerTG);
    return NMOEEncodeKernelTG(cmd, pipeline, groups, (NSUInteger)rowsPerTG * 32u, ^(id<MTLComputeCommandEncoder> encoder) {
        [encoder setBuffer:wfBuf offset:weightOff atIndex:0];
        [encoder setBuffer:wfBuf offset:scalesOff atIndex:1];
        [encoder setBuffer:wfBuf offset:biasesOff atIndex:2];
        [encoder setBuffer:inputMTL offset:inputOffset atIndex:3];
        [encoder setBuffer:outputMTL offset:outputOffset atIndex:4];
        [encoder setBytes:&args length:sizeof(args) atIndex:5];
    });
}

static BOOL NMOEEncodeDequantMatVecFromBufferOnEncoder(id<MTLComputeCommandEncoder> encoder,
                                                       id<MTLComputePipelineState> pipeline,
                                                       id<MTLBuffer> expertBuf,
                                                       size_t weightOff, size_t scalesOff, size_t biasesOff,
                                                       id<MTLBuffer> inputMTL, size_t inputOffset,
                                                       id<MTLBuffer> outputMTL, size_t outputOffset,
                                                       size_t outRows, size_t inDim, int bits) {
    if (encoder == nil || pipeline == nil || expertBuf == nil || inputMTL == nil || outputMTL == nil) return NO;

    uint32_t rowsPerTG = NMOEMatvecRowsPerThreadgroup(pipeline);
    uint32_t valuesPerWord = (uint32_t)(bits == 2 ? 16 : 8);
    NMOEDequantMatvecArgs args = {
        .out_rows = (uint32_t)outRows,
        .in_dim = (uint32_t)inDim,
        .packed_cols = (uint32_t)(inDim / valuesPerWord),
        .group_size = 64u,
        .rows_per_tg = rowsPerTG,
    };

    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:expertBuf offset:weightOff atIndex:0];
    [encoder setBuffer:expertBuf offset:scalesOff atIndex:1];
    [encoder setBuffer:expertBuf offset:biasesOff atIndex:2];
    [encoder setBuffer:inputMTL offset:inputOffset atIndex:3];
    [encoder setBuffer:outputMTL offset:outputOffset atIndex:4];
    [encoder setBytes:&args length:sizeof(args) atIndex:5];
    [encoder dispatchThreadgroups:MTLSizeMake((NSUInteger)((outRows + rowsPerTG - 1u) / rowsPerTG), 1, 1)
            threadsPerThreadgroup:MTLSizeMake((NSUInteger)rowsPerTG * 32u, 1, 1)];
    return YES;
}

static BOOL NMOEEncodeFullQKPrep(id<MTLCommandBuffer> cmd,
                                 nmoe_runtime *rt,
                                 const NMOELayerWeights *layer,
                                 BOOL includeQ,
                                 size_t position) {
    if (cmd == nil || rt == NULL || layer == NULL) return NO;
    id<MTLBuffer> qMTL = NMOEBridgeBuffer(includeQ ? rt->fullQProjBuffer : rt->fullKProjBuffer);
    id<MTLBuffer> kMTL = NMOEBridgeBuffer(rt->fullKProjBuffer);
    id<MTLBuffer> wfBuf = (__bridge id<MTLBuffer>)nmoe_backend_weight_buffer(rt->backend);
    id<MTLComputePipelineState> pipeline = (__bridge id<MTLComputePipelineState>)nmoe_backend_pipeline_state(rt->backend, NMOE_BACKEND_KERNEL_FULL_QK_PREP);
    NSUInteger qWeightOff = 0, kWeightOff = 0;
    const NMOEBF16Tensor *qWeight = includeQ ? &layer->u.full.qNorm : &layer->u.full.kNorm;
    if (qMTL == nil || kMTL == nil || wfBuf == nil || pipeline == nil ||
        !NMOEWeightPointerOffset(rt, qWeight->weight, &qWeightOff) ||
        !NMOEWeightPointerOffset(rt, layer->u.full.kNorm.weight, &kWeightOff)) return NO;

    NMOEFullQKPrepArgs args = {
        .head_dim = (uint32_t)kNMOEHeadDim,
        .q_stride = 512u,
        .full_heads = includeQ ? (uint32_t)kNMOEFullHeads : 0u,
        .kv_heads = (uint32_t)kNMOEFullKVHeads,
        .rotary_dim = (uint32_t)kNMOERotaryDim,
        .position = (uint32_t)position,
        .add_one = 1u,
        .epsilon = 1e-6f,
        .theta = 10000000.0f,
    };
    NSUInteger groups = (NSUInteger)args.full_heads + (NSUInteger)args.kv_heads;
    return NMOEEncodeKernelTG(cmd, pipeline, groups, kNMOEHeadDim, ^(id<MTLComputeCommandEncoder> encoder) {
        [encoder setBuffer:qMTL offset:0 atIndex:0];
        [encoder setBuffer:kMTL offset:0 atIndex:1];
        [encoder setBuffer:wfBuf offset:qWeightOff atIndex:2];
        [encoder setBuffer:wfBuf offset:kWeightOff atIndex:3];
        [encoder setBytes:&args length:sizeof(args) atIndex:4];
    });
}

static BOOL NMOEEncodeMoeExpertGateUpOnEncoder(id<MTLComputeCommandEncoder> encoder,
                                               id<MTLComputePipelineState> pipeline,
                                               id<MTLBuffer> gateMTL, size_t gateOffset,
                                               id<MTLBuffer> upMTL, size_t upOffset,
                                               id<MTLBuffer> outMTL, size_t outputOffset,
                                               size_t count) {
    if (encoder == nil || pipeline == nil || gateMTL == nil || upMTL == nil || outMTL == nil) return NO;

    NMOESigmoidGateArgs args = { .count = (uint32_t)count };
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:gateMTL offset:gateOffset atIndex:0];
    [encoder setBuffer:upMTL offset:upOffset atIndex:1];
    [encoder setBuffer:outMTL offset:outputOffset atIndex:2];
    [encoder setBytes:&args length:sizeof(args) atIndex:3];

    NSUInteger tgWidth = pipeline.threadExecutionWidth;
    if (tgWidth == 0) tgWidth = 1;
    if (tgWidth > pipeline.maxTotalThreadsPerThreadgroup) tgWidth = pipeline.maxTotalThreadsPerThreadgroup;
    [encoder dispatchThreads:MTLSizeMake((NSUInteger)count, 1, 1)
       threadsPerThreadgroup:MTLSizeMake(MIN(tgWidth, (NSUInteger)count), 1, 1)];
    return YES;
}

static BOOL NMOEEncodeExpertGateUpQ4OnEncoder(id<MTLComputeCommandEncoder> encoder,
                                               id<MTLComputePipelineState> pipeline,
                                               id<MTLBuffer> expertBuf,
                                               const NMOEExpertOffsets *offsets,
                                               id<MTLBuffer> inputMTL,
                                               id<MTLBuffer> outputMTL) {
    if (encoder == nil || pipeline == nil || expertBuf == nil || offsets == NULL ||
        inputMTL == nil || outputMTL == nil) return NO;

    uint32_t rowsPerTG = NMOEExpertRowsPerThreadgroup(pipeline);
    NMOEDequantMatvecArgs args = {
        .out_rows = 512u,
        .in_dim = (uint32_t)kNMOEHiddenDim,
        .packed_cols = (uint32_t)(kNMOEHiddenDim / 8u),
        .group_size = 64u,
        .rows_per_tg = rowsPerTG,
    };

    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:expertBuf offset:offsets->gate_weight atIndex:0];
    [encoder setBuffer:expertBuf offset:offsets->gate_scales atIndex:1];
    [encoder setBuffer:expertBuf offset:offsets->gate_biases atIndex:2];
    [encoder setBuffer:expertBuf offset:offsets->up_weight atIndex:3];
    [encoder setBuffer:expertBuf offset:offsets->up_scales atIndex:4];
    [encoder setBuffer:expertBuf offset:offsets->up_biases atIndex:5];
    [encoder setBuffer:inputMTL offset:0 atIndex:6];
    [encoder setBuffer:outputMTL offset:0 atIndex:7];
    [encoder setBytes:&args length:sizeof(args) atIndex:8];
    [encoder dispatchThreadgroups:MTLSizeMake((NSUInteger)((512u + rowsPerTG - 1u) / rowsPerTG), 1, 1)
            threadsPerThreadgroup:MTLSizeMake((NSUInteger)rowsPerTG * 32u, 1, 1)];
    return YES;
}

static BOOL NMOEEncodeGateUpQ4(id<MTLCommandBuffer> cmd,
                               nmoe_runtime *rt,
                               NSString *gateBaseName,
                               NSString *upBaseName,
                               void *inputBuffer,
                               size_t inputOffset,
                               void *outputBuffer,
                               size_t outputOffset) {
    if (cmd == nil || rt == NULL || gateBaseName.length == 0 || upBaseName.length == 0) return NO;

    id<MTLBuffer> weightBuffer = (__bridge id<MTLBuffer>)nmoe_backend_weight_buffer(rt->backend);
    id<MTLBuffer> inputMTL = NMOEBridgeBuffer(inputBuffer);
    id<MTLBuffer> outputMTL = NMOEBridgeBuffer(outputBuffer);
    id<MTLComputePipelineState> pipeline = (__bridge id<MTLComputePipelineState>)nmoe_backend_pipeline_state(rt->backend, NMOE_BACKEND_KERNEL_EXPERT_GATE_UP_Q4);
    if (weightBuffer == nil || inputMTL == nil || outputMTL == nil || pipeline == nil) return NO;

    nmoe_tensor_info *gw = nmoe_weight_tensor_info(rt->weights, NMOEModelTensorName([gateBaseName stringByAppendingString:@".weight"]).UTF8String);
    nmoe_tensor_info *gs = nmoe_weight_tensor_info(rt->weights, NMOEModelTensorName([gateBaseName stringByAppendingString:@".scales"]).UTF8String);
    nmoe_tensor_info *gb = nmoe_weight_tensor_info(rt->weights, NMOEModelTensorName([gateBaseName stringByAppendingString:@".biases"]).UTF8String);
    nmoe_tensor_info *uw = nmoe_weight_tensor_info(rt->weights, NMOEModelTensorName([upBaseName stringByAppendingString:@".weight"]).UTF8String);
    nmoe_tensor_info *us = nmoe_weight_tensor_info(rt->weights, NMOEModelTensorName([upBaseName stringByAppendingString:@".scales"]).UTF8String);
    nmoe_tensor_info *ub = nmoe_weight_tensor_info(rt->weights, NMOEModelTensorName([upBaseName stringByAppendingString:@".biases"]).UTF8String);
    if (gw == NULL || gs == NULL || gb == NULL || uw == NULL || us == NULL || ub == NULL) return NO;

    uint32_t rowsPerTG = NMOEMatvecRowsPerThreadgroup(pipeline);
    NMOEDequantMatvecArgs args = {
        .out_rows = 512u,
        .in_dim = (uint32_t)kNMOEHiddenDim,
        .packed_cols = (uint32_t)(kNMOEHiddenDim / 8u),
        .group_size = 64u,
        .rows_per_tg = rowsPerTG,
    };

    return NMOEEncodeKernelTG(cmd, pipeline, (NSUInteger)((512u + rowsPerTG - 1u) / rowsPerTG), (NSUInteger)rowsPerTG * 32u, ^(id<MTLComputeCommandEncoder> encoder) {
        [encoder setBuffer:weightBuffer offset:gw->offset atIndex:0];
        [encoder setBuffer:weightBuffer offset:gs->offset atIndex:1];
        [encoder setBuffer:weightBuffer offset:gb->offset atIndex:2];
        [encoder setBuffer:weightBuffer offset:uw->offset atIndex:3];
        [encoder setBuffer:weightBuffer offset:us->offset atIndex:4];
        [encoder setBuffer:weightBuffer offset:ub->offset atIndex:5];
        [encoder setBuffer:inputMTL offset:inputOffset atIndex:6];
        [encoder setBuffer:outputMTL offset:outputOffset atIndex:7];
        [encoder setBytes:&args length:sizeof(args) atIndex:8];
    });
}

static BOOL NMOEEncodeRouteSharedQ4Tensors(id<MTLCommandBuffer> cmd,
                                           nmoe_runtime *rt,
                                           const NMOEQuantTensor *router,
                                           const NMOEQuantTensor *sharedGate,
                                           const NMOEQuantTensor *sharedUp,
                                           const NMOEQuantTensor *sharedScore,
                                           void *inputBuffer,
                                           size_t inputOffset,
                                           void *routerOutputBuffer,
                                           size_t routerOutputOffset,
                                           void *sharedActBuffer,
                                           size_t sharedActOffset,
                                           void *sharedScoreOutputBuffer,
                                           size_t sharedScoreOutputOffset) {
    if (cmd == nil || rt == NULL || router == NULL || sharedGate == NULL ||
        sharedUp == NULL || sharedScore == NULL) return NO;

    id<MTLBuffer> weightBuffer = (__bridge id<MTLBuffer>)nmoe_backend_weight_buffer(rt->backend);
    id<MTLBuffer> inputMTL = NMOEBridgeBuffer(inputBuffer);
    id<MTLBuffer> routerMTL = NMOEBridgeBuffer(routerOutputBuffer);
    id<MTLBuffer> sharedActMTL = NMOEBridgeBuffer(sharedActBuffer);
    id<MTLBuffer> sharedScoreMTL = NMOEBridgeBuffer(sharedScoreOutputBuffer);
    id<MTLComputePipelineState> pipeline = (__bridge id<MTLComputePipelineState>)nmoe_backend_pipeline_state(rt->backend, NMOE_BACKEND_KERNEL_ROUTE_SHARED_Q4);
    if (weightBuffer == nil || inputMTL == nil || routerMTL == nil ||
        sharedActMTL == nil || sharedScoreMTL == nil || pipeline == nil) return NO;

    NSUInteger rw = 0, rs = 0, rb = 0, gw = 0, gs = 0, gb = 0;
    NSUInteger uw = 0, us = 0, ub = 0, sw = 0, ss = 0, sb = 0;
    if (!NMOEWeightPointerOffset(rt, router->weight, &rw) ||
        !NMOEWeightPointerOffset(rt, router->scales, &rs) ||
        !NMOEWeightPointerOffset(rt, router->biases, &rb) ||
        !NMOEWeightPointerOffset(rt, sharedGate->weight, &gw) ||
        !NMOEWeightPointerOffset(rt, sharedGate->scales, &gs) ||
        !NMOEWeightPointerOffset(rt, sharedGate->biases, &gb) ||
        !NMOEWeightPointerOffset(rt, sharedUp->weight, &uw) ||
        !NMOEWeightPointerOffset(rt, sharedUp->scales, &us) ||
        !NMOEWeightPointerOffset(rt, sharedUp->biases, &ub) ||
        !NMOEWeightPointerOffset(rt, sharedScore->weight, &sw) ||
        !NMOEWeightPointerOffset(rt, sharedScore->scales, &ss) ||
        !NMOEWeightPointerOffset(rt, sharedScore->biases, &sb)) return NO;

    uint32_t rowsPerTG = NMOEMatvecRowsPerThreadgroup(pipeline);
    NMOERouteSharedQ4Args args = {
        .hidden_dim = (uint32_t)kNMOEHiddenDim,
        .packed_cols = (uint32_t)(kNMOEHiddenDim / 8u),
        .group_size = 64u,
        .rows_per_tg = rowsPerTG,
    };

    return NMOEEncodeKernelTG(cmd, pipeline, (NSUInteger)((769u + rowsPerTG - 1u) / rowsPerTG), (NSUInteger)rowsPerTG * 32u, ^(id<MTLComputeCommandEncoder> encoder) {
        [encoder setBuffer:weightBuffer offset:rw atIndex:0];
        [encoder setBuffer:weightBuffer offset:rs atIndex:1];
        [encoder setBuffer:weightBuffer offset:rb atIndex:2];
        [encoder setBuffer:weightBuffer offset:gw atIndex:3];
        [encoder setBuffer:weightBuffer offset:gs atIndex:4];
        [encoder setBuffer:weightBuffer offset:gb atIndex:5];
        [encoder setBuffer:weightBuffer offset:uw atIndex:6];
        [encoder setBuffer:weightBuffer offset:us atIndex:7];
        [encoder setBuffer:weightBuffer offset:ub atIndex:8];
        [encoder setBuffer:weightBuffer offset:sw atIndex:9];
        [encoder setBuffer:weightBuffer offset:ss atIndex:10];
        [encoder setBuffer:weightBuffer offset:sb atIndex:11];
        [encoder setBuffer:inputMTL offset:inputOffset atIndex:12];
        [encoder setBuffer:routerMTL offset:routerOutputOffset atIndex:13];
        [encoder setBuffer:sharedActMTL offset:sharedActOffset atIndex:14];
        [encoder setBuffer:sharedScoreMTL offset:sharedScoreOutputOffset atIndex:15];
        [encoder setBytes:&args length:sizeof(args) atIndex:16];
    });
}

static void NMOESetExpertBaseBuffers(id<MTLComputeCommandEncoder> encoder,
                                      __strong id<MTLBuffer> *expertBuffers) {
    id<MTLBuffer> fallback = expertBuffers[0];
    for (int i = 0; i < 8; ++i) {
        id<MTLBuffer> buf = expertBuffers[i] != nil ? expertBuffers[i] : fallback;
        [encoder setBuffer:buf offset:0 atIndex:(NSUInteger)i];
    }
}

static BOOL NMOEEncodeExpertGateUpQ4Batched(id<MTLCommandBuffer> cmd,
                                             nmoe_runtime *rt,
                                             __strong id<MTLBuffer> *expertBuffers,
                                             size_t count,
                                             id<MTLBuffer> inputMTL,
                                             id<MTLBuffer> actMTL,
                                             const NMOEExpertOffsets *offsets) {
    if (cmd == nil || rt == NULL || expertBuffers == NULL || count == 0 ||
        inputMTL == nil || actMTL == nil || offsets == NULL || expertBuffers[0] == nil) return NO;

    id<MTLComputePipelineState> pipeline = (__bridge id<MTLComputePipelineState>)nmoe_backend_pipeline_state(rt->backend, NMOE_BACKEND_KERNEL_EXPERT_GATE_UP_Q4_BATCHED);
    if (pipeline == nil) return NO;

    uint32_t rowsPerTG = NMOEExpertRowsPerThreadgroup(pipeline);
    NMOEExpertBatchedArgs args = {
        .expert_count = (uint32_t)MIN(count, kNMOEMaxExperts),
        .out_rows = 512u,
        .in_dim = (uint32_t)kNMOEHiddenDim,
        .packed_cols = (uint32_t)(kNMOEHiddenDim / 8u),
        .group_size = 64u,
        .gate_weight = (uint32_t)offsets->gate_weight,
        .gate_scales = (uint32_t)offsets->gate_scales,
        .gate_biases = (uint32_t)offsets->gate_biases,
        .up_weight = (uint32_t)offsets->up_weight,
        .up_scales = (uint32_t)offsets->up_scales,
        .up_biases = (uint32_t)offsets->up_biases,
        .down_weight = (uint32_t)offsets->down_weight,
        .down_scales = (uint32_t)offsets->down_scales,
        .down_biases = (uint32_t)offsets->down_biases,
        .act_stride = 512u,
        .expert_size = 0u,
        .rows_per_tg = rowsPerTG,
    };

    NSUInteger rowGroups = (NSUInteger)((args.out_rows + rowsPerTG - 1u) / rowsPerTG);
    return NMOEEncodeKernelTG(cmd, pipeline, (NSUInteger)args.expert_count * rowGroups, (NSUInteger)rowsPerTG * 32u, ^(id<MTLComputeCommandEncoder> encoder) {
        NMOESetExpertBaseBuffers(encoder, expertBuffers);
        [encoder setBuffer:inputMTL offset:0 atIndex:8];
        [encoder setBuffer:actMTL offset:0 atIndex:9];
        [encoder setBytes:&args length:sizeof(args) atIndex:10];
    });
}

static BOOL NMOEEncodeExpertDownQ4Batched(id<MTLCommandBuffer> cmd,
                                           nmoe_runtime *rt,
                                           __strong id<MTLBuffer> *expertBuffers,
                                           void **expertOutBuffers,
                                           size_t count,
                                           id<MTLBuffer> actMTL,
                                           const NMOEExpertOffsets *offsets) {
    if (cmd == nil || rt == NULL || expertBuffers == NULL || expertOutBuffers == NULL ||
        count == 0 || actMTL == nil || offsets == NULL || expertBuffers[0] == nil) return NO;

    id<MTLComputePipelineState> pipeline = (__bridge id<MTLComputePipelineState>)nmoe_backend_pipeline_state(rt->backend, NMOE_BACKEND_KERNEL_EXPERT_DOWN_Q4_BATCHED);
    if (pipeline == nil) return NO;

    uint32_t rowsPerTG = NMOEExpertRowsPerThreadgroup(pipeline);
    NMOEExpertBatchedArgs args = {
        .expert_count = (uint32_t)MIN(count, kNMOEMaxExperts),
        .out_rows = (uint32_t)kNMOEHiddenDim,
        .in_dim = 512u,
        .packed_cols = 64u,
        .group_size = 64u,
        .gate_weight = (uint32_t)offsets->gate_weight,
        .gate_scales = (uint32_t)offsets->gate_scales,
        .gate_biases = (uint32_t)offsets->gate_biases,
        .up_weight = (uint32_t)offsets->up_weight,
        .up_scales = (uint32_t)offsets->up_scales,
        .up_biases = (uint32_t)offsets->up_biases,
        .down_weight = (uint32_t)offsets->down_weight,
        .down_scales = (uint32_t)offsets->down_scales,
        .down_biases = (uint32_t)offsets->down_biases,
        .act_stride = 512u,
        .expert_size = 0u,
        .rows_per_tg = rowsPerTG,
    };

    NSUInteger rowGroups = (NSUInteger)((args.out_rows + rowsPerTG - 1u) / rowsPerTG);
    return NMOEEncodeKernelTG(cmd, pipeline, (NSUInteger)args.expert_count * rowGroups, (NSUInteger)rowsPerTG * 32u, ^(id<MTLComputeCommandEncoder> encoder) {
        NMOESetExpertBaseBuffers(encoder, expertBuffers);
        [encoder setBuffer:actMTL offset:0 atIndex:8];
        for (int i = 0; i < 8; ++i) {
            id<MTLBuffer> out = (__bridge id<MTLBuffer>)expertOutBuffers[i];
            if (out == nil) out = (__bridge id<MTLBuffer>)expertOutBuffers[0];
            [encoder setBuffer:out offset:0 atIndex:(NSUInteger)(9 + i)];
        }
        [encoder setBytes:&args length:sizeof(args) atIndex:17];
    });
}

static BOOL NMOEEncodeExpertDownCombineQ4Tensor(id<MTLCommandBuffer> cmd,
                                                nmoe_runtime *rt,
                                                __strong id<MTLBuffer> *expertBuffers,
                                                size_t count,
                                                id<MTLBuffer> actMTL,
                                                void *hMidBuffer,
                                                void *sharedActBuffer,
                                                void *sharedOutBuffer,
                                                void *hiddenOutBuffer,
                                                const float *expertWeights,
                                                float sharedGateScore,
                                                const NMOEQuantTensor *sharedDown,
                                                const NMOEExpertOffsets *offsets) {
    if (cmd == nil || rt == NULL || expertBuffers == NULL || count == 0 ||
        actMTL == nil || hMidBuffer == NULL || sharedActBuffer == NULL || sharedOutBuffer == NULL ||
        hiddenOutBuffer == NULL || expertWeights == NULL || sharedDown == NULL || offsets == NULL ||
        expertBuffers[0] == nil) return NO;

    id<MTLComputePipelineState> pipeline = (__bridge id<MTLComputePipelineState>)nmoe_backend_pipeline_state(rt->backend, NMOE_BACKEND_KERNEL_EXPERT_DOWN_COMBINE_Q4);
    id<MTLBuffer> weightBuffer = (__bridge id<MTLBuffer>)nmoe_backend_weight_buffer(rt->backend);
    id<MTLBuffer> hMidMTL = NMOEBridgeBuffer(hMidBuffer);
    id<MTLBuffer> sharedMTL = NMOEBridgeBuffer(sharedOutBuffer);
    id<MTLBuffer> hiddenMTL = NMOEBridgeBuffer(hiddenOutBuffer);
    id<MTLBuffer> sharedActMTL = NMOEBridgeBuffer(sharedActBuffer);
    id<MTLBuffer> paramsBuf = NMOEBridgeBuffer(rt->normBuffer);
    NSUInteger sw = 0, ss = 0, sb = 0;
    if (pipeline == nil || weightBuffer == nil || hMidMTL == nil || sharedMTL == nil ||
        hiddenMTL == nil || sharedActMTL == nil || paramsBuf == nil ||
        !NMOEWeightPointerOffset(rt, sharedDown->weight, &sw) ||
        !NMOEWeightPointerOffset(rt, sharedDown->scales, &ss) ||
        !NMOEWeightPointerOffset(rt, sharedDown->biases, &sb)) return NO;

    float params[10];
    memset(params, 0, sizeof(params));
    for (size_t i = 0; i < count && i < 8; ++i) params[i] = expertWeights[i];
    params[8] = sharedGateScore;
    memcpy(paramsBuf.contents, params, sizeof(params));

    NMOEExpertBatchedArgs args = {
        .expert_count = (uint32_t)MIN(count, kNMOEMaxExperts),
        .out_rows = (uint32_t)kNMOEHiddenDim,
        .in_dim = 512u,
        .packed_cols = 64u,
        .group_size = 64u,
        .gate_weight = (uint32_t)offsets->gate_weight,
        .gate_scales = (uint32_t)offsets->gate_scales,
        .gate_biases = (uint32_t)offsets->gate_biases,
        .up_weight = (uint32_t)offsets->up_weight,
        .up_scales = (uint32_t)offsets->up_scales,
        .up_biases = (uint32_t)offsets->up_biases,
        .down_weight = (uint32_t)offsets->down_weight,
        .down_scales = (uint32_t)offsets->down_scales,
        .down_biases = (uint32_t)offsets->down_biases,
        .act_stride = 512u,
        .expert_size = 0u,
    };

    return NMOEEncodeKernelTG(cmd, pipeline, (NSUInteger)((kNMOEHiddenDim + 7u) / 8u), 256, ^(id<MTLComputeCommandEncoder> encoder) {
        NMOESetExpertBaseBuffers(encoder, expertBuffers);
        [encoder setBuffer:actMTL offset:0 atIndex:8];
        [encoder setBuffer:hMidMTL offset:0 atIndex:9];
        [encoder setBuffer:sharedMTL offset:0 atIndex:10];
        [encoder setBuffer:hiddenMTL offset:0 atIndex:11];
        [encoder setBuffer:paramsBuf offset:0 atIndex:12];
        [encoder setBytes:&args length:sizeof(args) atIndex:13];
        [encoder setBuffer:weightBuffer offset:sw atIndex:14];
        [encoder setBuffer:weightBuffer offset:ss atIndex:15];
        [encoder setBuffer:weightBuffer offset:sb atIndex:16];
        [encoder setBuffer:sharedActMTL offset:0 atIndex:17];
    });
}

static BOOL NMOEEncodeRoutedExpertQ4(id<MTLCommandBuffer> cmd,
                                      nmoe_runtime *rt,
                                      NMOEExpertLayerFile *expertFile,
                                      size_t count,
                                      void *expertActBuffer,
                                      void **expertOutBuffers) {
    if (cmd == nil || rt == NULL || expertFile == NULL || expertFile->metalBuffer == NULL ||
        count == 0 || expertActBuffer == NULL || expertOutBuffers == NULL) return NO;

    id<MTLBuffer> layerMTL = (__bridge id<MTLBuffer>)expertFile->metalBuffer;
    id<MTLBuffer> routeIndicesMTL = NMOEBridgeBuffer(rt->routeIndicesBuffer);
    id<MTLBuffer> inputMTL = NMOEBridgeBuffer(rt->normBuffer);
    id<MTLBuffer> actMTL = NMOEBridgeBuffer(expertActBuffer);
    id<MTLComputePipelineState> gateUpPipe = (__bridge id<MTLComputePipelineState>)nmoe_backend_pipeline_state(rt->backend, NMOE_BACKEND_KERNEL_EXPERT_GATE_UP_Q4_ROUTED);
    id<MTLComputePipelineState> downPipe = (__bridge id<MTLComputePipelineState>)nmoe_backend_pipeline_state(rt->backend, NMOE_BACKEND_KERNEL_EXPERT_DOWN_Q4_ROUTED);
    if (layerMTL == nil || routeIndicesMTL == nil || inputMTL == nil || actMTL == nil ||
        gateUpPipe == nil || downPipe == nil) return NO;

    NMOEExpertOffsets offsets = *NMOEExpertOffsetsForBits(4);
    uint32_t gateRowsPerTG = NMOEExpertRowsPerThreadgroup(gateUpPipe);
    NMOEExpertBatchedArgs gateArgs = {
        .expert_count = (uint32_t)MIN(count, kNMOEMaxExperts),
        .out_rows = 512u,
        .in_dim = (uint32_t)kNMOEHiddenDim,
        .packed_cols = (uint32_t)(kNMOEHiddenDim / 8u),
        .group_size = 64u,
        .gate_weight = (uint32_t)offsets.gate_weight,
        .gate_scales = (uint32_t)offsets.gate_scales,
        .gate_biases = (uint32_t)offsets.gate_biases,
        .up_weight = (uint32_t)offsets.up_weight,
        .up_scales = (uint32_t)offsets.up_scales,
        .up_biases = (uint32_t)offsets.up_biases,
        .down_weight = (uint32_t)offsets.down_weight,
        .down_scales = (uint32_t)offsets.down_scales,
        .down_biases = (uint32_t)offsets.down_biases,
        .act_stride = 512u,
        .expert_size = (uint32_t)expertFile->expertSize,
        .rows_per_tg = gateRowsPerTG,
    };
    NSUInteger gateRowGroups = (NSUInteger)((gateArgs.out_rows + gateRowsPerTG - 1u) / gateRowsPerTG);
    if (!NMOEEncodeKernelTG(cmd, gateUpPipe, (NSUInteger)gateArgs.expert_count * gateRowGroups, (NSUInteger)gateRowsPerTG * 32u, ^(id<MTLComputeCommandEncoder> encoder) {
        [encoder setBuffer:layerMTL offset:0 atIndex:0];
        [encoder setBuffer:routeIndicesMTL offset:0 atIndex:1];
        [encoder setBuffer:inputMTL offset:0 atIndex:2];
        [encoder setBuffer:actMTL offset:0 atIndex:3];
        [encoder setBytes:&gateArgs length:sizeof(gateArgs) atIndex:4];
    })) return NO;

    uint32_t downRowsPerTG = NMOEExpertRowsPerThreadgroup(downPipe);
    NMOEExpertBatchedArgs downArgs = gateArgs;
    downArgs.out_rows = (uint32_t)kNMOEHiddenDim;
    downArgs.in_dim = 512u;
    downArgs.packed_cols = 64u;
    downArgs.rows_per_tg = downRowsPerTG;
    NSUInteger downRowGroups = (NSUInteger)((downArgs.out_rows + downRowsPerTG - 1u) / downRowsPerTG);
    if (!NMOEEncodeKernelTG(cmd, downPipe, (NSUInteger)downArgs.expert_count * downRowGroups, (NSUInteger)downRowsPerTG * 32u, ^(id<MTLComputeCommandEncoder> encoder) {
        [encoder setBuffer:layerMTL offset:0 atIndex:0];
        [encoder setBuffer:routeIndicesMTL offset:0 atIndex:1];
        [encoder setBuffer:actMTL offset:0 atIndex:2];
        for (int i = 0; i < 8; ++i) {
            id<MTLBuffer> out = (__bridge id<MTLBuffer>)expertOutBuffers[i];
            if (out == nil) out = (__bridge id<MTLBuffer>)expertOutBuffers[0];
            [encoder setBuffer:out offset:0 atIndex:(NSUInteger)(3 + i)];
        }
        [encoder setBytes:&downArgs length:sizeof(downArgs) atIndex:11];
    })) return NO;

    return YES;
}

static BOOL NMOEEncodeWeightedExpertSumRouted(id<MTLCommandBuffer> cmd,
                                               nmoe_runtime *rt,
                                               void *hMidBuffer,
                                               void *sharedOutBuffer,
                                               void *hiddenOutBuffer,
                                               void **expertOutBuffers,
                                               size_t dim,
                                               size_t K) {
    if (cmd == nil || rt == NULL || hMidBuffer == NULL || sharedOutBuffer == NULL ||
        hiddenOutBuffer == NULL || expertOutBuffers == NULL || K == 0) return NO;

    id<MTLBuffer> hMidMTL = NMOEBridgeBuffer(hMidBuffer);
    id<MTLBuffer> sharedMTL = NMOEBridgeBuffer(sharedOutBuffer);
    id<MTLBuffer> hiddenMTL = NMOEBridgeBuffer(hiddenOutBuffer);
    id<MTLBuffer> weightsMTL = NMOEBridgeBuffer(rt->routeWeightsBuffer);
    id<MTLBuffer> sharedGateMTL = NMOEBridgeBuffer(rt->sharedOutBuffer);
    id<MTLComputePipelineState> pipeline = (__bridge id<MTLComputePipelineState>)nmoe_backend_pipeline_state(rt->backend, NMOE_BACKEND_KERNEL_WEIGHTED_EXPERT_SUM_ROUTED);
    if (hMidMTL == nil || sharedMTL == nil || hiddenMTL == nil || weightsMTL == nil ||
        sharedGateMTL == nil || pipeline == nil) return NO;

    uint32_t dim32 = (uint32_t)dim;
    uint32_t k32 = (uint32_t)MIN(K, kNMOEMaxExperts);
    return NMOEEncodeKernel(cmd, pipeline, dim, ^(id<MTLComputeCommandEncoder> encoder) {
        [encoder setBuffer:hMidMTL offset:0 atIndex:0];
        [encoder setBuffer:sharedMTL offset:0 atIndex:1];
        [encoder setBuffer:hiddenMTL offset:0 atIndex:2];
        for (int i = 0; i < 8; ++i) {
            id<MTLBuffer> out = (__bridge id<MTLBuffer>)expertOutBuffers[i];
            if (out == nil) out = (__bridge id<MTLBuffer>)expertOutBuffers[0];
            [encoder setBuffer:out offset:0 atIndex:(NSUInteger)(3 + i)];
        }
        [encoder setBuffer:weightsMTL offset:0 atIndex:11];
        [encoder setBuffer:sharedGateMTL offset:0 atIndex:12];
        [encoder setBytes:&dim32 length:sizeof(dim32) atIndex:13];
        [encoder setBytes:&k32 length:sizeof(k32) atIndex:14];
    });
}

static BOOL NMOEEncodeResidualAdd(id<MTLCommandBuffer> cmd,
                                   nmoe_runtime *rt,
                                   void *aBuffer, size_t aOffset,
                                   void *bBuffer, size_t bOffset,
                                   void *outBuffer, size_t outOffset,
                                   size_t dim) {
    id<MTLBuffer> aMTL = NMOEBridgeBuffer(aBuffer);
    id<MTLBuffer> bMTL = NMOEBridgeBuffer(bBuffer);
    id<MTLBuffer> outMTL = NMOEBridgeBuffer(outBuffer);
    if (aMTL == nil || bMTL == nil || outMTL == nil) return NO;

    id<MTLComputePipelineState> pipeline = (__bridge id<MTLComputePipelineState>)nmoe_backend_pipeline_state(rt->backend, NMOE_BACKEND_KERNEL_RESIDUAL_ADD);
    if (pipeline == nil) return NO;

    uint32_t dim32 = (uint32_t)dim;
    return NMOEEncodeKernel(cmd, pipeline, dim, ^(id<MTLComputeCommandEncoder> encoder) {
        [encoder setBuffer:aMTL offset:aOffset atIndex:0];
        [encoder setBuffer:bMTL offset:bOffset atIndex:1];
        [encoder setBuffer:outMTL offset:outOffset atIndex:2];
        [encoder setBytes:&dim32 length:sizeof(dim32) atIndex:3];
    });
}

static BOOL NMOEEncodeWeightedExpertSum(id<MTLCommandBuffer> cmd,
                                         nmoe_runtime *rt,
                                         void *hMidBuffer,
                                         void *sharedOutBuffer,
                                         void *hiddenOutBuffer,
                                         void **expertOutBuffers,
                                         size_t dim, size_t K,
                                         const float *expertWeights,
                                         float sharedGateScore) {
    id<MTLBuffer> hMidMTL = NMOEBridgeBuffer(hMidBuffer);
    id<MTLBuffer> sharedMTL = NMOEBridgeBuffer(sharedOutBuffer);
    id<MTLBuffer> hiddenMTL = NMOEBridgeBuffer(hiddenOutBuffer);
    if (hMidMTL == nil || sharedMTL == nil || hiddenMTL == nil) return NO;

    id<MTLComputePipelineState> pipeline = (__bridge id<MTLComputePipelineState>)nmoe_backend_pipeline_state(rt->backend, NMOE_BACKEND_KERNEL_WEIGHTED_EXPERT_SUM);
    if (pipeline == nil) return NO;

    // Pack params: 8 expert weights + shared gate score + padding
    float params[10];
    memset(params, 0, sizeof(params));
    for (size_t i = 0; i < K && i < 8; i++) params[i] = expertWeights[i];
    params[8] = sharedGateScore;

    uint32_t dim32 = (uint32_t)dim;
    uint32_t k32 = (uint32_t)K;

    id<MTLBuffer> e0 = (__bridge id<MTLBuffer>)(K > 0 ? expertOutBuffers[0] : nil);
    id<MTLBuffer> e1 = (__bridge id<MTLBuffer>)(K > 1 ? expertOutBuffers[1] : nil);
    id<MTLBuffer> e2 = (__bridge id<MTLBuffer>)(K > 2 ? expertOutBuffers[2] : nil);
    id<MTLBuffer> e3 = (__bridge id<MTLBuffer>)(K > 3 ? expertOutBuffers[3] : nil);
    id<MTLBuffer> e4 = (__bridge id<MTLBuffer>)(K > 4 ? expertOutBuffers[4] : nil);
    id<MTLBuffer> e5 = (__bridge id<MTLBuffer>)(K > 5 ? expertOutBuffers[5] : nil);
    id<MTLBuffer> e6 = (__bridge id<MTLBuffer>)(K > 6 ? expertOutBuffers[6] : nil);
    id<MTLBuffer> e7 = (__bridge id<MTLBuffer>)(K > 7 ? expertOutBuffers[7] : nil);

    // Use the runtime's normBuffer as a temporary params buffer
    id<MTLBuffer> paramsBuf = NMOEBridgeBuffer(rt->normBuffer);
    if (paramsBuf == nil) return NO;
    memcpy(paramsBuf.contents, params, sizeof(params));

    return NMOEEncodeKernel(cmd, pipeline, dim, ^(id<MTLComputeCommandEncoder> encoder) {
        [encoder setBuffer:hMidMTL offset:0 atIndex:0];
        [encoder setBuffer:sharedMTL offset:0 atIndex:1];
        [encoder setBuffer:hiddenMTL offset:0 atIndex:2];
        for (int i = 0; i < 8; i++) {
            [encoder setBuffer:(i == 0 ? e0 : i == 1 ? e1 : i == 2 ? e2 : i == 3 ? e3 :
                                  i == 4 ? e4 : i == 5 ? e5 : i == 6 ? e6 : e7)
                        offset:0 atIndex:(3u + i)];
        }
        [encoder setBuffer:paramsBuf offset:0 atIndex:11];
        [encoder setBytes:&dim32 length:sizeof(dim32) atIndex:12];
        [encoder setBytes:&k32 length:sizeof(k32) atIndex:13];
    });
}

static BOOL NMOEEncodeRouteTopK(id<MTLCommandBuffer> cmd, nmoe_runtime *rt, size_t k) {
    if (cmd == nil || rt == NULL) return NO;

    id<MTLBuffer> scoresMTL = NMOEBridgeBuffer(rt->routerScoresBuffer);
    id<MTLBuffer> indicesMTL = NMOEBridgeBuffer(rt->routeIndicesBuffer);
    id<MTLBuffer> weightsMTL = NMOEBridgeBuffer(rt->routeWeightsBuffer);
    if (scoresMTL == nil || indicesMTL == nil || weightsMTL == nil) return NO;

    id<MTLComputePipelineState> pipeline = (__bridge id<MTLComputePipelineState>)nmoe_backend_pipeline_state(rt->backend, NMOE_BACKEND_KERNEL_ROUTE_TOPK);
    if (pipeline == nil) return NO;

    NMOERouteTopKArgs args = {
        .count = 256u,
        .k = (uint32_t)MAX((size_t)1, MIN(k, kNMOEMaxExperts)),
    };
    return NMOEEncodeKernel(cmd, pipeline, 1, ^(id<MTLComputeCommandEncoder> encoder) {
        [encoder setBuffer:scoresMTL offset:0 atIndex:0];
        [encoder setBuffer:indicesMTL offset:0 atIndex:1];
        [encoder setBuffer:weightsMTL offset:0 atIndex:2];
        [encoder setBytes:&args length:sizeof(args) atIndex:3];
    });
}

static BOOL NMOEUseCPURouter(void) {
    const char *value = getenv("NMOE_CPU_ROUTER");
    if (value == NULL || value[0] == '\0') return YES;
    if (value[0] == '0' || value[0] == 'f' || value[0] == 'F' ||
        value[0] == 'n' || value[0] == 'N') return NO;
    return value[0] == '1' || value[0] == 't' || value[0] == 'T' ||
           value[0] == 'y' || value[0] == 'Y';
}

static BOOL NMOEUseRoutedExperts(void) {
    const char *value = getenv("NMOE_GPU_ROUTED_EXPERTS");
    if (value == NULL || value[0] == '\0') return NO;
    return value[0] == '1' || value[0] == 't' || value[0] == 'T' ||
           value[0] == 'y' || value[0] == 'Y';
}

static BOOL NMOEUseFusedDownCombineQ4(void) {
    const char *value = getenv("NMOE_FUSED_DOWN_COMBINE_Q4");
    if (value == NULL || value[0] == '\0') return YES;
    if (value[0] == '0' || value[0] == 'f' || value[0] == 'F' ||
        value[0] == 'n' || value[0] == 'N') return NO;
    return value[0] == '1' || value[0] == 't' || value[0] == 'T' ||
           value[0] == 'y' || value[0] == 'Y';
}

static BOOL NMOECopyExpertsToMetalBuffers(void) {
    const char *value = getenv("NMOE_COPY_EXPERTS");
    if (value == NULL || value[0] == '\0') return NO;
    return value[0] == '1' || value[0] == 't' || value[0] == 'T' ||
           value[0] == 'y' || value[0] == 'Y';
}

static uint32_t NMOEExpertRowsPerThreadgroup(id<MTLComputePipelineState> pipeline) {
    const char *value = getenv("NMOE_EXPERT_ROWS_PER_TG");
    uint32_t rows = 32u;
    if (value != NULL && value[0] != '\0') {
        long parsed = strtol(value, NULL, 10);
        if (parsed == 8 || parsed == 16 || parsed == 24 || parsed == 32) rows = (uint32_t)parsed;
    }
    if (pipeline != nil && pipeline.maxTotalThreadsPerThreadgroup < (NSUInteger)rows * 32u) {
        rows = 8u;
    }
    return rows;
}

static uint32_t NMOEMatvecRowsPerThreadgroup(id<MTLComputePipelineState> pipeline) {
    const char *value = getenv("NMOE_MATVEC_ROWS_PER_TG");
    uint32_t rows = 24u;
    if (value != NULL && value[0] != '\0') {
        long parsed = strtol(value, NULL, 10);
        if (parsed == 8 || parsed == 16 || parsed == 24 || parsed == 32) rows = (uint32_t)parsed;
    }
    if (pipeline != nil && pipeline.maxTotalThreadsPerThreadgroup < (NSUInteger)rows * 32u) {
        rows = 8u;
    }
    return rows;
}

static size_t NMOESelectRouteTopKCPU(nmoe_runtime *rt,
                                     size_t k,
                                     size_t *selectedIndices,
                                     float *selectedValues) {
    if (rt == NULL || selectedIndices == NULL || selectedValues == NULL || k == 0) return 0;
    float *routerScores = NMOEFloatBuffer(rt->routerScoresBuffer);
    if (routerScores == NULL) return 0;

    size_t kVal = (size_t)MAX(1, MIN((int)kNMOEMaxExperts, (int)k));
    nmoe_cpu_softmax(routerScores, 256u);
    size_t selectedCount = nmoe_cpu_topk(routerScores, 256u, kVal, selectedIndices, selectedValues);
    nmoe_cpu_renormalize(selectedValues, selectedCount);
    return selectedCount;
}

static size_t NMOEReadRouteTopKMetal(nmoe_runtime *rt,
                                     size_t k,
                                     size_t *selectedIndices,
                                     float *selectedValues) {
    if (rt == NULL || selectedIndices == NULL || selectedValues == NULL || k == 0) return 0;
    const uint32_t *routeIndices = NMOEU32Buffer(rt->routeIndicesBuffer);
    const float *routeWeights = NMOEFloatBuffer(rt->routeWeightsBuffer);
    if (routeIndices == NULL || routeWeights == NULL) return 0;

    size_t selectedCount = (size_t)MAX(1, MIN((int)kNMOEMaxExperts, (int)k));
    for (size_t i = 0; i < selectedCount; ++i) {
        selectedIndices[i] = (size_t)routeIndices[i];
        selectedValues[i] = routeWeights[i];
    }
    return selectedCount;
}

// Pre-allocated expert I/O buffers (allocated once, reused across all layers)
static void *g_expertIOBuffers[8] = {nil};
static void *g_expertOutBuffers[8] = {nil};
static void *g_expertIOInputBuf = nil;
static void *g_expertIOSumSqBuf = nil;
static void *g_expertIOHMidBuf = nil;
static void *g_expertIOSharedActBuf = nil;
static void *g_expertIOSharedDownBuf = nil;
static void *g_expertIOExpertActBuf = nil;
static int g_expertIOBits = 0;
static size_t g_expertIOSize = 0;

static BOOL NMOEInitExpertIOBuffers(nmoe_runtime *rt) {
    if (g_expertIOBuffers[0] != NULL && g_expertIOBits == rt->quantBits) return YES;


    id<MTLDevice> device = (__bridge id<MTLDevice>)nmoe_backend_device(rt->backend);
    if (device == nil) return NO;


    g_expertIOBits = rt->quantBits;
    g_expertIOSize = nmoe_expert_active_size(rt->quantBits);

    const size_t expertAlign = 2u * 1024u * 1024u;
    size_t expertAllocSize = (g_expertIOSize + expertAlign - 1u) & ~(expertAlign - 1u);
    for (int k = 0; k < 8; k++) {
        void *alignedData = NULL;
        if (posix_memalign(&alignedData, expertAlign, expertAllocSize) == 0 && alignedData != NULL) {
            memset(alignedData, 0, expertAllocSize);
            g_expertIOBuffers[k] = (__bridge_retained void *)[device newBufferWithBytesNoCopy:alignedData
                                                                                       length:expertAllocSize
                                                                                      options:MTLResourceStorageModeShared
                                                                                  deallocator:nil];
        }
        if (g_expertIOBuffers[k] == NULL) {
            free(alignedData);
            g_expertIOBuffers[k] = (__bridge_retained void *)[device newBufferWithLength:g_expertIOSize options:MTLResourceStorageModeShared];
        }
    }
    for (int k = 0; k < 8; k++) {
        g_expertOutBuffers[k] = (__bridge_retained void *)[device newBufferWithLength:kNMOEHiddenDim * sizeof(float) options:MTLResourceStorageModeShared];
    }
    g_expertIOInputBuf = (__bridge_retained void *)[device newBufferWithLength:kNMOEHiddenDim * sizeof(float) options:MTLResourceStorageModeShared];
    g_expertIOSumSqBuf = (__bridge_retained void *)[device newBufferWithLength:sizeof(float) options:MTLResourceStorageModeShared];
    g_expertIOHMidBuf = (__bridge_retained void *)[device newBufferWithLength:kNMOEHiddenDim * sizeof(float) options:MTLResourceStorageModeShared];
    g_expertIOSharedActBuf = (__bridge_retained void *)[device newBufferWithLength:512u * sizeof(float) options:MTLResourceStorageModeShared];
    g_expertIOSharedDownBuf = (__bridge_retained void *)[device newBufferWithLength:kNMOEHiddenDim * sizeof(float) options:MTLResourceStorageModeShared];
    g_expertIOExpertActBuf = (__bridge_retained void *)[device newBufferWithLength:kNMOEMaxExperts * 512u * sizeof(float) options:MTLResourceStorageModeShared];
    return YES;
}

// Async parallel pread experts directly into pre-allocated Metal buffers
static BOOL NMOEAsyncReadExperts(int fd, const uint8_t *mmapBase,
                                  const size_t *indices, size_t count,
                                  void **buffers, int *outValid) {
    size_t esz = g_expertIOSize;
    dispatch_group_t group = dispatch_group_create();
    dispatch_queue_t ioQueue = dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0);

    for (size_t k = 0; k < count; k++) {
        outValid[k] = 1;
        size_t expertIdx = indices[k];
        size_t offset = expertIdx * esz;
        id<MTLBuffer> buf = (__bridge id<MTLBuffer>)buffers[k];
        if (buf == nil || buf.length < esz) {
            outValid[k] = 0;
            continue;
        }

        if (mmapBase != NULL) {
            dispatch_group_async(group, ioQueue, ^{
                memcpy(buf.contents, mmapBase + offset, esz);
            });
        } else {
            dispatch_group_async(group, ioQueue, ^{
                uint8_t *dst = (uint8_t *)buf.contents;
                size_t remaining = esz;
                off_t pos = (off_t)offset;
                while (remaining > 0) {
                    ssize_t rc = pread(fd, dst, remaining, pos);
                    if (rc <= 0) {
                        outValid[k] = 0;
                        break;
                    }
                    dst += rc;
                    pos += rc;
                    remaining -= (size_t)rc;
                }
            });
        }
    }

    dispatch_group_wait(group, DISPATCH_TIME_FOREVER);
    return YES;
}

static BOOL NMOEWrapMappedExperts(nmoe_runtime *rt,
                                  NMOEExpertLayerFile *expertFile,
                                  const uint8_t *mmapBase,
                                  const size_t *indices,
                                  size_t count,
                                  __strong id<MTLBuffer> *outBuffers,
                                  int *outValid) {
    if (rt == NULL || expertFile == NULL || mmapBase == NULL || indices == NULL ||
        outBuffers == NULL || outValid == NULL) return NO;

    id<MTLDevice> device = (__bridge id<MTLDevice>)nmoe_backend_device(rt->backend);
    if (device == nil) return NO;

    size_t esz = g_expertIOSize;
    for (size_t k = 0; k < count; ++k) {
        outValid[k] = 1;
        size_t expertIdx = indices[k];
        if (expertIdx >= 256u) {
            outValid[k] = 0;
            continue;
        }
        id<MTLBuffer> cached = (__bridge id<MTLBuffer>)expertFile->expertMetalBuffers[expertIdx];
        if (cached == nil) {
            void *ptr = (void *)(mmapBase + expertIdx * esz);
            cached = [device newBufferWithBytesNoCopy:ptr
                                               length:esz
                                              options:MTLResourceStorageModeShared
                                          deallocator:nil];
            if (cached != nil) {
                expertFile->expertMetalBuffers[expertIdx] = (__bridge_retained void *)cached;
            }
        }
        outBuffers[k] = cached;
        if (outBuffers[k] == nil) outValid[k] = 0;
    }
    return YES;
}

static BOOL NMOEPrepareExpertBuffers(nmoe_runtime *rt,
                                     NMOEExpertLayerFile *expertFile,
                                     const size_t *indices,
                                     size_t count,
                                     __strong id<MTLBuffer> *expertBuffers,
                                     int *valid,
                                     NMOEPerfStats *stats) {
    if (rt == NULL || expertFile == NULL || indices == NULL || expertBuffers == NULL || valid == NULL) return NO;

    for (size_t i = 0; i < count; ++i) {
        expertBuffers[i] = nil;
        valid[i] = 1;
    }

    double tFetch = NMOENowSeconds();
    const uint8_t *expertBase = expertFile->mapped ? (const uint8_t *)expertFile->base : NULL;
    BOOL wrapped = NO;
    if (expertBase != NULL && !NMOECopyExpertsToMetalBuffers()) {
        wrapped = NMOEWrapMappedExperts(rt, expertFile, expertBase, indices, count, expertBuffers, valid);
    }
    if (!wrapped) {
        NMOEAsyncReadExperts(expertFile->fd, expertBase, indices, count, g_expertIOBuffers, valid);
        for (size_t i = 0; i < count; ++i) {
            expertBuffers[i] = (__bridge id<MTLBuffer>)g_expertIOBuffers[i];
        }
    }
    if (stats != NULL) stats->expertFetch += NMOENowSeconds() - tFetch;
    return YES;
}

static BOOL NMOERunDequantMatVecMetalWithBuffer(nmoe_runtime *rt,
                                                id<MTLBuffer> weightBuffer,
                                                size_t weightOffsetBytes,
                                                size_t scalesOffsetBytes,
                                                size_t biasesOffsetBytes,
                                                void *inputBuffer,
                                                size_t inputOffsetBytes,
                                                void *outputBuffer,
                                                size_t outputOffsetBytes,
                                                size_t outRows,
                                                size_t inDim,
                                                int bits) {
    if (rt == NULL || weightBuffer == nil) {
        return NO;
    }

    id<MTLBuffer> inputMTL = NMOEBridgeBuffer(inputBuffer);
    id<MTLBuffer> outputMTL = NMOEBridgeBuffer(outputBuffer);
    id<MTLComputePipelineState> pipeline = (__bridge id<MTLComputePipelineState>)nmoe_backend_pipeline_state(rt->backend, bits == 2 ? NMOE_BACKEND_KERNEL_DEQUANT_MATVEC_Q2 : NMOE_BACKEND_KERNEL_DEQUANT_MATVEC_Q4);
    id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)nmoe_backend_command_queue(rt->backend);
    if (rt->backend == NULL || inputMTL == nil || outputMTL == nil || pipeline == nil || queue == nil) {
        const float *input = NMOEFloatBufferAtOffset(inputBuffer, inputOffsetBytes);
        float *output = NMOEFloatBufferAtOffset(outputBuffer, outputOffsetBytes);
        if (input != NULL && output != NULL) {
            const uint32_t *cpuWeight = (const uint32_t *)((uint8_t *)weightBuffer.contents + weightOffsetBytes);
            const uint16_t *cpuScales = (const uint16_t *)((uint8_t *)weightBuffer.contents + scalesOffsetBytes);
            const uint16_t *cpuBiases = (const uint16_t *)((uint8_t *)weightBuffer.contents + biasesOffsetBytes);
            nmoe_cpu_dequant_matvec(cpuWeight, cpuScales, cpuBiases, input, output, outRows, inDim, bits);
            return YES;
        }
        return NO;
    }

    uint32_t valuesPerWord = bits == 2 ? 16u : 8u;
    uint32_t packedCols = (uint32_t)(inDim / valuesPerWord);
    uint32_t rowsPerTG = NMOEMatvecRowsPerThreadgroup(pipeline);
    NMOEDequantMatvecArgs args = {
        .out_rows = (uint32_t)outRows,
        .in_dim = (uint32_t)inDim,
        .packed_cols = packedCols,
        .group_size = 64u,
        .rows_per_tg = rowsPerTG,
    };

    NSUInteger groups = (NSUInteger)((outRows + rowsPerTG - 1u) / rowsPerTG);
    if (NMOEDispatchKernelTG(queue, pipeline, groups, (NSUInteger)rowsPerTG * 32u, ^(id<MTLComputeCommandEncoder> encoder) {
        [encoder setBuffer:weightBuffer offset:weightOffsetBytes atIndex:0];
        [encoder setBuffer:weightBuffer offset:scalesOffsetBytes atIndex:1];
        [encoder setBuffer:weightBuffer offset:biasesOffsetBytes atIndex:2];
        [encoder setBuffer:inputMTL offset:inputOffsetBytes atIndex:3];
        [encoder setBuffer:outputMTL offset:outputOffsetBytes atIndex:4];
        [encoder setBytes:&args length:sizeof(args) atIndex:5];
    })) {
        return YES;
    }

    const float *input = NMOEFloatBufferAtOffset(inputBuffer, inputOffsetBytes);
    float *output = NMOEFloatBufferAtOffset(outputBuffer, outputOffsetBytes);
    if (input != NULL && output != NULL) {
        const uint32_t *cpuWeight = (const uint32_t *)((uint8_t *)weightBuffer.contents + weightOffsetBytes);
        const uint16_t *cpuScales = (const uint16_t *)((uint8_t *)weightBuffer.contents + scalesOffsetBytes);
        const uint16_t *cpuBiases = (const uint16_t *)((uint8_t *)weightBuffer.contents + biasesOffsetBytes);
        nmoe_cpu_dequant_matvec(cpuWeight, cpuScales, cpuBiases, input, output, outRows, inDim, bits);
        return YES;
    }
    return NO;
}

static BOOL NMOERunDequantMatVecMetal(nmoe_runtime *rt,
                                      NSString *baseName,
                                      void *inputBuffer,
                                      size_t inputOffsetBytes,
                                      void *outputBuffer,
                                      size_t outputOffsetBytes,
                                      size_t outRows,
                                      size_t inDim,
                                      int bits) {
    if (rt == NULL || baseName.length == 0) {
        return NO;
    }

    // model_weights.bin stores all non-expert quantized tensors as q4.
    // The q2/q4 runtime switch applies only to SSD-streamed MoE expert blobs.
    bits = 4;

    NSString *weightName = NMOEModelTensorName([baseName stringByAppendingString:@".weight"]);
    NSString *scalesName = NMOEModelTensorName([baseName stringByAppendingString:@".scales"]);
    NSString *biasesName = NMOEModelTensorName([baseName stringByAppendingString:@".biases"]);
    const uint32_t *cpuWeight = (const uint32_t *)NMOETensorPointer(rt->weights, weightName, @"U32", NULL);
    const uint16_t *cpuScales = (const uint16_t *)NMOETensorPointer(rt->weights, scalesName, @"BF16", NULL);
    const uint16_t *cpuBiases = (const uint16_t *)NMOETensorPointer(rt->weights, biasesName, @"BF16", NULL);
    id<MTLBuffer> weightBuffer = (__bridge id<MTLBuffer>)nmoe_backend_weight_buffer(rt->backend);
    nmoe_tensor_info *weightInfo = nmoe_weight_tensor_info(rt->weights, weightName.UTF8String);
    nmoe_tensor_info *scalesInfo = nmoe_weight_tensor_info(rt->weights, scalesName.UTF8String);
    nmoe_tensor_info *biasesInfo = nmoe_weight_tensor_info(rt->weights, biasesName.UTF8String);

    if (weightBuffer == nil || weightInfo == NULL || scalesInfo == NULL || biasesInfo == NULL) {
        const float *input = NMOEFloatBufferAtOffset(inputBuffer, inputOffsetBytes);
        float *output = NMOEFloatBufferAtOffset(outputBuffer, outputOffsetBytes);
        if (cpuWeight != NULL && cpuScales != NULL && cpuBiases != NULL && input != NULL && output != NULL) {
            nmoe_cpu_dequant_matvec(cpuWeight, cpuScales, cpuBiases, input, output, outRows, inDim, bits);
            return YES;
        }
        return NO;
    }

    return NMOERunDequantMatVecMetalWithBuffer(rt,
                                               weightBuffer,
                                               weightInfo->offset,
                                               scalesInfo->offset,
                                               biasesInfo->offset,
                                               inputBuffer,
                                               inputOffsetBytes,
                                               outputBuffer,
                                               outputOffsetBytes,
                                               outRows,
                                               inDim,
                                               bits);
}

static BOOL NMOERunArgmaxTop1Metal(nmoe_runtime *rt,
                                   void *valuesBuffer,
                                   size_t valuesOffsetBytes,
                                   size_t count,
                                   uint32_t *outToken) {
    if (rt == NULL || outToken == NULL || count == 0) {
        return NO;
    }

    id<MTLBuffer> valuesMTL = NMOEBridgeBuffer(valuesBuffer);
    id<MTLBuffer> resultMTL = NMOEBridgeBuffer(rt->nextTokenBuffer);
    id<MTLComputePipelineState> pipeline = (__bridge id<MTLComputePipelineState>)nmoe_backend_pipeline_state(rt->backend, NMOE_BACKEND_KERNEL_ARGMAX_TOP1);
    id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)nmoe_backend_command_queue(rt->backend);
    uint32_t *result = (uint32_t *)NMOEU32Buffer(rt->nextTokenBuffer);
    if (valuesMTL == nil || resultMTL == nil || pipeline == nil || queue == nil || result == NULL) {
        return NO;
    }

    result[0] = 0;
    NMOEArgmaxArgs args = {
        .count = (uint32_t)count,
        .reserved0 = 0,
        .reserved1 = 0,
        .reserved2 = 0,
    };

    id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
    if (commandBuffer == nil) {
        return NO;
    }
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    if (encoder == nil) {
        return NO;
    }

    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:valuesMTL offset:valuesOffsetBytes atIndex:0];
    [encoder setBuffer:resultMTL offset:0 atIndex:1];
    [encoder setBytes:&args length:sizeof(args) atIndex:2];
    [encoder dispatchThreadgroups:MTLSizeMake(1, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [encoder endEncoding];
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
    if (commandBuffer.error != nil) {
        return NO;
    }

    *outToken = result[0];
    return YES;
}

static BOOL NMOERunLmHeadArgmaxQ4Metal(nmoe_runtime *rt,
                                       void *inputBuffer,
                                       size_t inputOffsetBytes,
                                       uint32_t *outToken) {
    if (rt == NULL || inputBuffer == NULL || outToken == NULL) return NO;

    NSString *baseName = @"lm_head";
    NSString *weightName = NMOEModelTensorName([baseName stringByAppendingString:@".weight"]);
    NSString *scalesName = NMOEModelTensorName([baseName stringByAppendingString:@".scales"]);
    NSString *biasesName = NMOEModelTensorName([baseName stringByAppendingString:@".biases"]);
    id<MTLBuffer> weightBuffer = (__bridge id<MTLBuffer>)nmoe_backend_weight_buffer(rt->backend);
    nmoe_tensor_info *weightInfo = nmoe_weight_tensor_info(rt->weights, weightName.UTF8String);
    nmoe_tensor_info *scalesInfo = nmoe_weight_tensor_info(rt->weights, scalesName.UTF8String);
    nmoe_tensor_info *biasesInfo = nmoe_weight_tensor_info(rt->weights, biasesName.UTF8String);

    id<MTLBuffer> inputMTL = NMOEBridgeBuffer(inputBuffer);
    id<MTLBuffer> partialValuesMTL = NMOEBridgeBuffer(rt->logitsBuffer);
    id<MTLBuffer> partialIndicesMTL = NMOEBridgeBuffer(rt->lmHeadPartialIndicesBuffer);
    id<MTLBuffer> resultMTL = NMOEBridgeBuffer(rt->nextTokenBuffer);
    id<MTLComputePipelineState> stagePipe = (__bridge id<MTLComputePipelineState>)nmoe_backend_pipeline_state(rt->backend, NMOE_BACKEND_KERNEL_LM_HEAD_ARGMAX_Q4);
    id<MTLComputePipelineState> reducePipe = (__bridge id<MTLComputePipelineState>)nmoe_backend_pipeline_state(rt->backend, NMOE_BACKEND_KERNEL_LM_HEAD_ARGMAX_REDUCE);
    id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)nmoe_backend_command_queue(rt->backend);
    uint32_t *result = (uint32_t *)NMOEU32Buffer(rt->nextTokenBuffer);
    if (weightBuffer == nil || weightInfo == NULL || scalesInfo == NULL || biasesInfo == NULL ||
        inputMTL == nil || partialValuesMTL == nil || partialIndicesMTL == nil || resultMTL == nil ||
        stagePipe == nil || reducePipe == nil || queue == nil || result == NULL) return NO;

    uint32_t rowsPerTG = MIN(NMOEMatvecRowsPerThreadgroup(stagePipe), 16u);
    uint32_t vocabSize = 248320u;
    uint32_t partialCount = (vocabSize + rowsPerTG - 1u) / rowsPerTG;
    NMOELmHeadArgmaxArgs args = {
        .vocab_size = vocabSize,
        .in_dim = (uint32_t)kNMOEHiddenDim,
        .packed_cols = (uint32_t)(kNMOEHiddenDim / 8u),
        .group_size = 64u,
        .rows_per_tg = rowsPerTG,
        .partial_count = partialCount,
    };

    result[0] = 0;
    id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
    if (commandBuffer == nil) return NO;

    if (!NMOEEncodeKernelTG(commandBuffer, stagePipe, partialCount, (NSUInteger)rowsPerTG * 32u, ^(id<MTLComputeCommandEncoder> encoder) {
        [encoder setBuffer:weightBuffer offset:weightInfo->offset atIndex:0];
        [encoder setBuffer:weightBuffer offset:scalesInfo->offset atIndex:1];
        [encoder setBuffer:weightBuffer offset:biasesInfo->offset atIndex:2];
        [encoder setBuffer:inputMTL offset:inputOffsetBytes atIndex:3];
        [encoder setBuffer:partialValuesMTL offset:0 atIndex:4];
        [encoder setBuffer:partialIndicesMTL offset:0 atIndex:5];
        [encoder setBytes:&args length:sizeof(args) atIndex:6];
    })) return NO;

    if (!NMOEEncodeKernelTG(commandBuffer, reducePipe, 1, 256, ^(id<MTLComputeCommandEncoder> encoder) {
        [encoder setBuffer:partialValuesMTL offset:0 atIndex:0];
        [encoder setBuffer:partialIndicesMTL offset:0 atIndex:1];
        [encoder setBuffer:resultMTL offset:0 atIndex:2];
        [encoder setBytes:&args length:sizeof(args) atIndex:3];
    })) return NO;

    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
    if (commandBuffer.error != nil) return NO;

    *outToken = result[0];
    return YES;
}

static BOOL NMOERunRMSNormMetal(nmoe_runtime *rt,
                                NSString *weightName,
                                void *inputBuffer,
                                size_t inputOffsetBytes,
                                void *outputBuffer,
                                size_t outputOffsetBytes,
                                size_t dim,
                                BOOL addOne) {
    if (rt == NULL) {
        return NO;
    }

    BOOL useUnitWeight = (weightName.length == 0);
    const uint16_t *cpuWeight = useUnitWeight ? NULL : (const uint16_t *)NMOETensorPointer(rt->weights, weightName, @"BF16", NULL);
    if (rt->backend == NULL) {
        float *input = NMOEFloatBufferAtOffset(inputBuffer, inputOffsetBytes);
        float *output = NMOEFloatBufferAtOffset(outputBuffer, outputOffsetBytes);
        if ((cpuWeight != NULL || useUnitWeight) && input != NULL && output != NULL) {
            nmoe_cpu_rms_norm(input, cpuWeight, output, dim, 1e-6f, addOne ? 1 : 0);
            return YES;
        }
        return NO;
    }

    id<MTLBuffer> weightBuffer = useUnitWeight ? nil : (__bridge id<MTLBuffer>)nmoe_backend_weight_buffer(rt->backend);
    id<MTLBuffer> inputMTL = (__bridge id<MTLBuffer>)inputBuffer;
    id<MTLBuffer> outputMTL = (__bridge id<MTLBuffer>)outputBuffer;
    id<MTLComputePipelineState> pipeline = (__bridge id<MTLComputePipelineState>)nmoe_backend_pipeline_state(rt->backend, NMOE_BACKEND_KERNEL_RMS_NORM);
    id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)nmoe_backend_command_queue(rt->backend);
    id<MTLBuffer> unitWeightBuffer = nil;
    if (useUnitWeight && weightBuffer == nil && rt->backend != nil) {
        uint16_t *ones = calloc(dim, sizeof(uint16_t));
        if (ones != NULL) {
            for (size_t i = 0; i < dim; ++i) {
                ones[i] = nmoe_f32_to_bf16(1.0f);
            }
            unitWeightBuffer = NMOEMakeTemporaryBuffer(rt, ones, dim * sizeof(uint16_t), @"rms_unit_weight");
            free(ones);
            weightBuffer = unitWeightBuffer;
        }
    }

    if (weightBuffer == nil || inputMTL == nil || outputMTL == nil || pipeline == nil || queue == nil) {
        float *input = NMOEFloatBufferAtOffset(inputBuffer, inputOffsetBytes);
        float *output = NMOEFloatBufferAtOffset(outputBuffer, outputOffsetBytes);
        if ((cpuWeight != NULL || useUnitWeight) && input != NULL && output != NULL) {
            nmoe_cpu_rms_norm(input, cpuWeight, output, dim, 1e-6f, addOne ? 1 : 0);
            return YES;
        }
        return NO;
    }

    nmoe_tensor_info *weightInfo = useUnitWeight ? NULL : nmoe_weight_tensor_info(rt->weights, weightName.UTF8String);
    if (!useUnitWeight && weightInfo == NULL) {
        float *input = NMOEFloatBufferAtOffset(inputBuffer, inputOffsetBytes);
        float *output = NMOEFloatBufferAtOffset(outputBuffer, outputOffsetBytes);
        if ((cpuWeight != NULL || useUnitWeight) && input != NULL && output != NULL) {
            nmoe_cpu_rms_norm(input, cpuWeight, output, dim, 1e-6f, addOne ? 1 : 0);
            return YES;
        }
        return NO;
    }

    // Use old single-threaded kernel (one-pass, correct)
    NMOERMSNormArgs args = {
        .dim = (uint32_t)dim,
        .add_one = addOne ? 1u : 0u,
        .epsilon = 1e-6f,
        .reserved = 0.0f,
    };

    if (NMOEDispatchKernelTG(queue, pipeline, 1, 256, ^(id<MTLComputeCommandEncoder> encoder) {
        [encoder setBuffer:inputMTL offset:inputOffsetBytes atIndex:0];
        [encoder setBuffer:weightBuffer offset:useUnitWeight ? 0 : weightInfo->offset atIndex:1];
        [encoder setBuffer:outputMTL offset:outputOffsetBytes atIndex:2];
        [encoder setBytes:&args length:sizeof(args) atIndex:3];
    })) {
        return YES;
    }

    float *input = NMOEFloatBufferAtOffset(inputBuffer, inputOffsetBytes);
    float *output = NMOEFloatBufferAtOffset(outputBuffer, outputOffsetBytes);
    if ((cpuWeight != NULL || useUnitWeight) && input != NULL && output != NULL) {
        nmoe_cpu_rms_norm(input, cpuWeight, output, dim, 1e-6f, addOne ? 1 : 0);
        return YES;
    }
    return NO;
}

static BOOL NMOERunDequantRowMetal(nmoe_runtime *rt,
                                   NSString *baseName,
                                   size_t rowIndex,
                                   void *outputBuffer,
                                   size_t outputOffsetBytes,
                                   size_t inDim,
                                   int bits) {
    if (rt == NULL || baseName.length == 0) {
        return NO;
    }

    NSString *weightName = NMOEModelTensorName([baseName stringByAppendingString:@".weight"]);
    NSString *scalesName = NMOEModelTensorName([baseName stringByAppendingString:@".scales"]);
    NSString *biasesName = NMOEModelTensorName([baseName stringByAppendingString:@".biases"]);
    const uint32_t *cpuWeight = (const uint32_t *)NMOETensorPointer(rt->weights, weightName, @"U32", NULL);
    const uint16_t *cpuScales = (const uint16_t *)NMOETensorPointer(rt->weights, scalesName, @"BF16", NULL);
    const uint16_t *cpuBiases = (const uint16_t *)NMOETensorPointer(rt->weights, biasesName, @"BF16", NULL);
    id<MTLBuffer> weightBuffer = (__bridge id<MTLBuffer>)nmoe_backend_weight_buffer(rt->backend);
    nmoe_tensor_info *weightInfo = nmoe_weight_tensor_info(rt->weights, weightName.UTF8String);
    nmoe_tensor_info *scalesInfo = nmoe_weight_tensor_info(rt->weights, scalesName.UTF8String);
    nmoe_tensor_info *biasesInfo = nmoe_weight_tensor_info(rt->weights, biasesName.UTF8String);
    id<MTLBuffer> outputMTL = NMOEBridgeBuffer(outputBuffer);

    uint32_t valuesPerWord = bits == 2 ? 16u : 8u;
    size_t packedCols = inDim / valuesPerWord;
    size_t groups = inDim / 64u;
    if (packedCols == 0 || groups == 0) {
        return NO;
    }
    if (weightInfo != NULL && weightInfo->ndim >= 2 && rowIndex >= (size_t)weightInfo->shape[0]) {
        return NO;
    }

    if (weightBuffer == nil || outputMTL == nil || weightInfo == NULL || scalesInfo == NULL || biasesInfo == NULL) {
        const uint32_t *cpuPacked = cpuWeight != NULL ? cpuWeight + rowIndex * packedCols : NULL;
        const uint16_t *cpuScale = cpuScales != NULL ? cpuScales + rowIndex * groups : NULL;
        const uint16_t *cpuBias = cpuBiases != NULL ? cpuBiases + rowIndex * groups : NULL;
        float *output = NMOEFloatBufferAtOffset(outputBuffer, outputOffsetBytes);
        if (cpuPacked != NULL && cpuScale != NULL && cpuBias != NULL && output != NULL) {
            nmoe_cpu_dequant_row(cpuPacked, cpuScale, cpuBias, output, inDim, bits);
            return YES;
        }
        return NO;
    }

    id<MTLComputePipelineState> pipeline = (__bridge id<MTLComputePipelineState>)nmoe_backend_pipeline_state(rt->backend, bits == 2 ? NMOE_BACKEND_KERNEL_DEQUANT_ROW_Q2 : NMOE_BACKEND_KERNEL_DEQUANT_ROW_Q4);
    id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)nmoe_backend_command_queue(rt->backend);
    if (pipeline == nil || queue == nil) {
        float *output = NMOEFloatBufferAtOffset(outputBuffer, outputOffsetBytes);
        if (cpuWeight != NULL && cpuScales != NULL && cpuBiases != NULL && output != NULL) {
            nmoe_cpu_dequant_row(cpuWeight + rowIndex * packedCols,
                                 cpuScales + rowIndex * groups,
                                 cpuBiases + rowIndex * groups,
                                 output,
                                 inDim,
                                 bits);
            return YES;
        }
        return NO;
    }

    NMOEDequantRowArgs args = {
        .inDim = (uint32_t)inDim,
        .packedCols = (uint32_t)packedCols,
        .groupSize = 64u,
        .reserved0 = 0u,
    };

    NSUInteger weightOffset = weightInfo->offset + rowIndex * packedCols * sizeof(uint32_t);
    NSUInteger scalesOffset = scalesInfo->offset + rowIndex * groups * sizeof(uint16_t);
    NSUInteger biasesOffset = biasesInfo->offset + rowIndex * groups * sizeof(uint16_t);
    if (NMOEDispatchKernel1D(queue, pipeline, (NSUInteger)inDim, ^(id<MTLComputeCommandEncoder> encoder) {
        [encoder setBuffer:weightBuffer offset:weightOffset atIndex:0];
        [encoder setBuffer:weightBuffer offset:scalesOffset atIndex:1];
        [encoder setBuffer:weightBuffer offset:biasesOffset atIndex:2];
        [encoder setBuffer:outputMTL offset:outputOffsetBytes atIndex:3];
        [encoder setBytes:&args length:sizeof(args) atIndex:4];
    })) {
        return YES;
    }

    float *output = NMOEFloatBufferAtOffset(outputBuffer, outputOffsetBytes);
    if (cpuWeight != NULL && cpuScales != NULL && cpuBiases != NULL && output != NULL) {
        nmoe_cpu_dequant_row(cpuWeight + rowIndex * packedCols,
                             cpuScales + rowIndex * groups,
                             cpuBiases + rowIndex * groups,
                             output,
                             inDim,
                             bits);
        return YES;
    }
    return NO;
}

static NSError *NMOEMakeError(NSInteger code, NSString *message) {
    return [NSError errorWithDomain:NMOERuntimeErrorDomain
                               code:code
                           userInfo:@{NSLocalizedDescriptionKey : message ?: @"nmoe runtime error"}];
}

static double NMOENowSeconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}

static NSString *NMOEJoinPath(NSString *base, NSString *component) {
    if (base.length == 0) {
        return component ?: @"";
    }
    if (component.length == 0) {
        return base;
    }
    return [base stringByAppendingPathComponent:component];
}

static NSString *NMOEModelTensorName(NSString *suffix) {
    return suffix ?: @"";
}

static NSString *NMOEStringFromC(const char *value) {
    if (value == NULL) {
        return @"";
    }
    return [NSString stringWithUTF8String:value] ?: @"";
}

static BOOL NMOEPathExists(NSString *path, BOOL *isDirectory) {
    return [[NSFileManager defaultManager] fileExistsAtPath:path isDirectory:isDirectory];
}

static BOOL NMOEOpenMappedFile(NSString *path, NMOEExpertLayerFile *file, NSError **error) {
    if (path.length == 0 || file == NULL) {
        if (error != NULL) {
            *error = NMOEMakeError(1, @"invalid expert file open arguments");
        }
        return NO;
    }

    memset(file, 0, sizeof(*file));
    int fd = open(path.fileSystemRepresentation, O_RDONLY);
    if (fd < 0) {
        if (error != NULL) {
            *error = NMOEMakeError(2, [NSString stringWithFormat:@"failed to open %@", path]);
        }
        return NO;
    }

    struct stat st;
    if (fstat(fd, &st) != 0) {
        if (error != NULL) {
            *error = NMOEMakeError(3, [NSString stringWithFormat:@"failed to stat %@", path]);
        }
        close(fd);
        return NO;
    }

    size_t size = (size_t)st.st_size;
    void *base = NULL;
    BOOL mapped = NO;
    if (size > 0) {
        base = mmap(NULL, size, PROT_READ, MAP_PRIVATE, fd, 0);
        if (base != MAP_FAILED) {
            mapped = YES;
        } else {
            base = NULL;
        }
    }

    file->fd = fd;
    file->base = base;
    file->size = size;
    file->mapped = mapped;
    return YES;
}

static void NMOECloseMappedFile(NMOEExpertLayerFile *file) {
    if (file == NULL) {
        return;
    }
    if (file->metalBuffer != NULL) {
        (void)CFBridgingRelease(file->metalBuffer);
        file->metalBuffer = NULL;
    }
    if (file->mapped && file->base != NULL && file->size > 0) {
        munmap(file->base, file->size);
    }
    if (file->fd >= 0) {
        close(file->fd);
    }
    memset(file, 0, sizeof(*file));
    file->fd = -1;
}

static const NMOEExpertOffsets *NMOEExpertOffsetsForBits(int bits) {
    static const NMOEExpertOffsets q4 = {
        0, 524288, 557056,
        589824, 1114112, 1146880,
        1179648, 1703936, 1736704,
    };
    static const NMOEExpertOffsets q2 = {
        0, 262144, 294912,
        327680, 589824, 622592,
        655360, 917504, 950272,
    };
    return bits == 2 ? &q2 : &q4;
}


static BOOL NMOERegisterBuffer(nmoe_runtime *rt, const char *label, size_t size, BOOL zeroFill, void **outBuffer) {
    if (rt == NULL || rt->backend == NULL || label == NULL || size == 0 || outBuffer == NULL) {
        return NO;
    }
    void *buffer = nmoe_backend_register_state_buffer(rt->backend, label, size);
    if (buffer == NULL) {
        return NO;
    }
    if (zeroFill) {
        id<MTLBuffer> mtl = NMOEBridgeBuffer(buffer);
        memset((void *)mtl.contents, 0, size);
    }
    *outBuffer = buffer;
    return YES;
}

static const void *NMOETensorPointer(nmoe_weight_file *weights, NSString *name, NSString *dtype, NSError **error) {
    if (weights == NULL || name.length == 0) {
        if (error != NULL) {
            *error = NMOEMakeError(10, @"invalid tensor lookup arguments");
        }
        return NULL;
    }
    nmoe_tensor_info *ti = nmoe_weight_tensor_info(weights, name.UTF8String);
    if (ti == NULL || ti->name == NULL) {
        if (error != NULL) {
            *error = NMOEMakeError(11, [NSString stringWithFormat:@"missing tensor %@", name]);
        }
        return NULL;
    }
    if (dtype.length > 0 && strcmp(ti->dtype, dtype.UTF8String) != 0) {
        if (error != NULL) {
            *error = NMOEMakeError(12, [NSString stringWithFormat:@"tensor %@ has dtype %s, expected %@", name, ti->dtype, dtype]);
        }
        return NULL;
    }
    void *ptr = nmoe_weight_tensor_ptr(weights, name.UTF8String);
    if (ptr == NULL) {
        if (error != NULL) {
            *error = NMOEMakeError(13, [NSString stringWithFormat:@"tensor %@ is out of range", name]);
        }
        return NULL;
    }
    return ptr;
}

static BOOL NMOELoadQuantTensor(nmoe_weight_file *weights,
                                NSString *base,
                                NMOEQuantTensor *tensor,
                                NSError **error) {
    if (tensor == NULL) {
        return NO;
    }
    NSString *weightName = [base stringByAppendingString:@".weight"];
    NSString *scalesName = [base stringByAppendingString:@".scales"];
    NSString *biasesName = [base stringByAppendingString:@".biases"];
    tensor->weight = (const uint32_t *)NMOETensorPointer(weights, NMOEModelTensorName(weightName), @"U32", error);
    tensor->scales = (const uint16_t *)NMOETensorPointer(weights, NMOEModelTensorName(scalesName), @"BF16", error);
    tensor->biases = (const uint16_t *)NMOETensorPointer(weights, NMOEModelTensorName(biasesName), @"BF16", error);
    return tensor->weight != NULL && tensor->scales != NULL && tensor->biases != NULL;
}

static BOOL NMOELoadBF16Tensor(nmoe_weight_file *weights,
                               NSString *name,
                               NMOEBF16Tensor *tensor,
                               NSError **error) {
    if (tensor == NULL) {
        return NO;
    }
    tensor->weight = (const uint16_t *)NMOETensorPointer(weights, NMOEModelTensorName(name), @"BF16", error);
    return tensor->weight != NULL;
}

static BOOL NMOELoadF32Tensor(nmoe_weight_file *weights,
                              NSString *name,
                              NMOEF32Tensor *tensor,
                              NSError **error) {
    if (tensor == NULL) {
        return NO;
    }
    tensor->value = (const float *)NMOETensorPointer(weights, NMOEModelTensorName(name), @"F32", error);
    return tensor->value != NULL;
}

static BOOL NMOELoadLayerWeights(nmoe_runtime *rt, int layerIndex, NSError **error) {
    if (rt == NULL || rt->weights == NULL || layerIndex < 0 || layerIndex >= (int)kNMOELayers) {
        if (error != NULL) {
            *error = NMOEMakeError(20, @"invalid layer index");
        }
        return NO;
    }

    NMOELayerWeights *layer = &rt->layers[layerIndex];
    memset(layer, 0, sizeof(*layer));

    NSString *prefix = [NSString stringWithFormat:@"model.layers.%d", layerIndex];
    if (!NMOELoadBF16Tensor(rt->weights, [prefix stringByAppendingString:@".input_layernorm.weight"], &layer->inputNorm, error)) {
        return NO;
    }
    if (!NMOELoadBF16Tensor(rt->weights, [prefix stringByAppendingString:@".post_attention_layernorm.weight"], &layer->postNorm, error)) {
        return NO;
    }
    if (!NMOELoadQuantTensor(rt->weights, [prefix stringByAppendingString:@".mlp.gate"], &layer->routerGate, error)) {
        return NO;
    }
    if (!NMOELoadQuantTensor(rt->weights, [prefix stringByAppendingString:@".mlp.shared_expert.gate_proj"], &layer->sharedGateProj, error)) {
        return NO;
    }
    if (!NMOELoadQuantTensor(rt->weights, [prefix stringByAppendingString:@".mlp.shared_expert.up_proj"], &layer->sharedUpProj, error)) {
        return NO;
    }
    if (!NMOELoadQuantTensor(rt->weights, [prefix stringByAppendingString:@".mlp.shared_expert.down_proj"], &layer->sharedDownProj, error)) {
        return NO;
    }
    if (!NMOELoadQuantTensor(rt->weights, [prefix stringByAppendingString:@".mlp.shared_expert_gate"], &layer->sharedGateScore, error)) {
        return NO;
    }

    layer->isFull = (((layerIndex + 1) % 4) == 0);
    if (layer->isFull) {
        if (!NMOELoadQuantTensor(rt->weights, [prefix stringByAppendingString:@".self_attn.q_proj"], &layer->u.full.qProj, error)) {
            return NO;
        }
        if (!NMOELoadQuantTensor(rt->weights, [prefix stringByAppendingString:@".self_attn.k_proj"], &layer->u.full.kProj, error)) {
            return NO;
        }
        if (!NMOELoadQuantTensor(rt->weights, [prefix stringByAppendingString:@".self_attn.v_proj"], &layer->u.full.vProj, error)) {
            return NO;
        }
        if (!NMOELoadQuantTensor(rt->weights, [prefix stringByAppendingString:@".self_attn.o_proj"], &layer->u.full.oProj, error)) {
            return NO;
        }
        if (!NMOELoadBF16Tensor(rt->weights, [prefix stringByAppendingString:@".self_attn.q_norm.weight"], &layer->u.full.qNorm, error)) {
            return NO;
        }
        if (!NMOELoadBF16Tensor(rt->weights, [prefix stringByAppendingString:@".self_attn.k_norm.weight"], &layer->u.full.kNorm, error)) {
            return NO;
        }
    } else {
        if (!NMOELoadQuantTensor(rt->weights, [prefix stringByAppendingString:@".linear_attn.in_proj_qkv"], &layer->u.linear.qkvProj, error)) {
            return NO;
        }
        if (!NMOELoadQuantTensor(rt->weights, [prefix stringByAppendingString:@".linear_attn.in_proj_z"], &layer->u.linear.zProj, error)) {
            return NO;
        }
        if (!NMOELoadQuantTensor(rt->weights, [prefix stringByAppendingString:@".linear_attn.in_proj_b"], &layer->u.linear.betaProj, error)) {
            return NO;
        }
        if (!NMOELoadQuantTensor(rt->weights, [prefix stringByAppendingString:@".linear_attn.in_proj_a"], &layer->u.linear.alphaProj, error)) {
            return NO;
        }
        if (!NMOELoadBF16Tensor(rt->weights, [prefix stringByAppendingString:@".linear_attn.conv1d.weight"], &layer->u.linear.convWeight, error)) {
            return NO;
        }
        if (!NMOELoadF32Tensor(rt->weights, [prefix stringByAppendingString:@".linear_attn.A_log"], &layer->u.linear.ALog, error)) {
            return NO;
        }
        if (!NMOELoadBF16Tensor(rt->weights, [prefix stringByAppendingString:@".linear_attn.dt_bias"], &layer->u.linear.dtBias, error)) {
            return NO;
        }
        if (!NMOELoadBF16Tensor(rt->weights, [prefix stringByAppendingString:@".linear_attn.norm.weight"], &layer->u.linear.normWeight, error)) {
            return NO;
        }
        if (!NMOELoadQuantTensor(rt->weights, [prefix stringByAppendingString:@".linear_attn.out_proj"], &layer->u.linear.outProj, error)) {
            return NO;
        }
    }

    return YES;
}

static BOOL NMOELoadExpertLayer(nmoe_runtime *rt, int layerIndex, NSError **error) {
    if (rt == NULL || rt->expertDirectory == NULL || layerIndex < 0 || layerIndex >= (int)kNMOELayers) {
        if (error != NULL) {
            *error = NMOEMakeError(30, @"invalid expert layer load arguments");
        }
        return NO;
    }

    NMOELayerState *state = &rt->layerState[layerIndex];
    memset(&state->expertFile, 0, sizeof(state->expertFile));
    state->expertFile.fd = -1;

    NSString *layerName = [NSString stringWithFormat:@"layer_%02d.bin", layerIndex];
    NSString *expertDir = [NSString stringWithUTF8String:rt->expertDirectory] ?: @"";
    NSString *layerPath = [expertDir stringByAppendingPathComponent:layerName];
    if (!NMOEOpenMappedFile(layerPath, &state->expertFile, error)) {
        return NO;
    }
    state->expertFile.expertSize = nmoe_expert_active_size(rt->quantBits);
    if (state->expertFile.size < state->expertFile.expertSize * 256u) {
        if (error != NULL) {
            *error = NMOEMakeError(31, [NSString stringWithFormat:@"expert file %@ has wrong size", layerPath]);
        }
        NMOECloseMappedFile(&state->expertFile);
        return NO;
    }
    if (NMOEUseRoutedExperts() && state->expertFile.mapped && state->expertFile.base != NULL && state->expertFile.size > 0) {
        id<MTLDevice> device = (__bridge id<MTLDevice>)nmoe_backend_device(rt->backend);
        id<MTLBuffer> layerBuffer = device != nil
            ? [device newBufferWithBytesNoCopy:state->expertFile.base
                                        length:state->expertFile.size
                                       options:MTLResourceStorageModeShared
                                   deallocator:nil]
            : nil;
        if (layerBuffer != nil) {
            state->expertFile.metalBuffer = (__bridge_retained void *)layerBuffer;
        }
    }
    return YES;
}

static BOOL NMOEAllocateStateBuffers(nmoe_runtime *rt, NSError **error) {
    if (rt == NULL || rt->backend == NULL) {
        if (error != NULL) {
            *error = NMOEMakeError(40, @"backend is missing");
        }
        return NO;
    }

    size_t hiddenBytes = kNMOEHiddenDim * sizeof(float);
    size_t routerBytes = 256u * sizeof(float);
    size_t routeIndexBytes = kNMOEMaxExperts * sizeof(uint32_t);
    size_t routeWeightBytes = kNMOEMaxExperts * sizeof(float);
    size_t sharedBytes = 512u * sizeof(float);
    size_t qProjBytes = 8192u * sizeof(float);
    size_t kBytes = 512u * sizeof(float);
    size_t vBytes = 512u * sizeof(float);
    size_t attnBytes = 4096u * sizeof(float);
    size_t linearQkvBytes = 8192u * sizeof(float);
    size_t linearZBytes = 4096u * sizeof(float);
    size_t linearScalarBytes = 32u * sizeof(float);
    size_t linearConvBytes = 8192u * sizeof(float);
    size_t linearOutBytes = 4096u * sizeof(float);
    size_t scoresBytes = kNMOEFullHeads * rt->sequenceCapacity * sizeof(float);
    size_t probsBytes = kNMOEFullHeads * rt->sequenceCapacity * sizeof(float);
    size_t logitsBytes = 248320u * sizeof(float);
    size_t lmHeadPartialCount = (248320u + 15u) / 16u;
    size_t lmHeadPartialIndexBytes = lmHeadPartialCount * sizeof(uint32_t);
    size_t nextTokenBytes = sizeof(uint32_t);
    size_t promptTokensBytes = MAX((size_t)4096, rt->sequenceCapacity) * sizeof(uint32_t);
    size_t embedBytes = kNMOEHiddenDim * sizeof(float);

    if (!NMOERegisterBuffer(rt, "runtime.hidden_a", hiddenBytes, YES, &rt->hiddenBuffers[0])) return NO;
    if (!NMOERegisterBuffer(rt, "runtime.hidden_b", hiddenBytes, YES, &rt->hiddenBuffers[1])) return NO;
    if (!NMOERegisterBuffer(rt, "runtime.norm", hiddenBytes, YES, &rt->normBuffer)) return NO;
    if (!NMOERegisterBuffer(rt, "runtime.full_qproj", qProjBytes, YES, &rt->fullQProjBuffer)) return NO;
    if (!NMOERegisterBuffer(rt, "runtime.full_kproj", kBytes, YES, &rt->fullKProjBuffer)) return NO;
    if (!NMOERegisterBuffer(rt, "runtime.full_vproj", vBytes, YES, &rt->fullVProjBuffer)) return NO;
    if (!NMOERegisterBuffer(rt, "runtime.full_attn", attnBytes, YES, &rt->fullAttnBuffer)) return NO;
    if (!NMOERegisterBuffer(rt, "runtime.linear_qkv", linearQkvBytes, YES, &rt->linearQkvBuffer)) return NO;
    if (!NMOERegisterBuffer(rt, "runtime.linear_z", linearZBytes, YES, &rt->linearZBuffer)) return NO;
    if (!NMOERegisterBuffer(rt, "runtime.linear_beta", linearScalarBytes, YES, &rt->linearBetaBuffer)) return NO;
    if (!NMOERegisterBuffer(rt, "runtime.linear_alpha", linearScalarBytes, YES, &rt->linearAlphaBuffer)) return NO;
    if (!NMOERegisterBuffer(rt, "runtime.linear_conv", linearConvBytes, YES, &rt->linearConvBuffer)) return NO;
    if (!NMOERegisterBuffer(rt, "runtime.linear_out", linearOutBytes, YES, &rt->linearOutBuffer)) return NO;
    if (!NMOERegisterBuffer(rt, "runtime.router_scores", routerBytes, YES, &rt->routerScoresBuffer)) return NO;
    if (!NMOERegisterBuffer(rt, "runtime.router_probs", routerBytes, YES, &rt->routerProbsBuffer)) return NO;
    if (!NMOERegisterBuffer(rt, "runtime.route_indices", routeIndexBytes, YES, &rt->routeIndicesBuffer)) return NO;
    if (!NMOERegisterBuffer(rt, "runtime.route_weights", routeWeightBytes, YES, &rt->routeWeightsBuffer)) return NO;
    if (!NMOERegisterBuffer(rt, "runtime.shared_gate", sharedBytes, YES, &rt->sharedGateBuffer)) return NO;
    if (!NMOERegisterBuffer(rt, "runtime.shared_up", sharedBytes, YES, &rt->sharedUpBuffer)) return NO;
    if (!NMOERegisterBuffer(rt, "runtime.shared_out", hiddenBytes, YES, &rt->sharedOutBuffer)) return NO;
    if (!NMOERegisterBuffer(rt, "runtime.expert_gate", sharedBytes, YES, &rt->expertGateBuffer)) return NO;
    if (!NMOERegisterBuffer(rt, "runtime.expert_up", sharedBytes, YES, &rt->expertUpBuffer)) return NO;
    if (!NMOERegisterBuffer(rt, "runtime.expert_act", sharedBytes, YES, &rt->expertActBuffer)) return NO;
    if (!NMOERegisterBuffer(rt, "runtime.expert_out", hiddenBytes, YES, &rt->expertOutBuffer)) return NO;
    if (!NMOERegisterBuffer(rt, "runtime.attn_scores", scoresBytes, YES, &rt->attnScoresBuffer)) return NO;
    if (!NMOERegisterBuffer(rt, "runtime.attn_probs", probsBytes, YES, &rt->attnProbsBuffer)) return NO;
    if (!NMOERegisterBuffer(rt, "runtime.logits", logitsBytes, YES, &rt->logitsBuffer)) return NO;
    if (!NMOERegisterBuffer(rt, "runtime.lm_head_partial_indices", lmHeadPartialIndexBytes, YES, &rt->lmHeadPartialIndicesBuffer)) return NO;
    if (!NMOERegisterBuffer(rt, "runtime.next_token", nextTokenBytes, YES, &rt->nextTokenBuffer)) return NO;
    if (!NMOERegisterBuffer(rt, "runtime.prompt_tokens", promptTokensBytes, YES, &rt->promptTokensBuffer)) return NO;
    if (!NMOERegisterBuffer(rt, "runtime.embedding", embedBytes, YES, &rt->embeddingBuffer)) return NO;

    for (size_t i = 0; i < kNMOELayers; ++i) {
        BOOL isFull = (((i + 1u) % 4u) == 0u);
        if (isFull) {
            size_t kvBytes = rt->sequenceCapacity * kNMOEFullKVHeads * kNMOEHeadDim * sizeof(float);
            if (!NMOERegisterBuffer(rt,
                                    [[NSString stringWithFormat:@"layer.%02zu.full.k_cache", i] UTF8String],
                                    kvBytes,
                                    YES,
                                    &rt->layerState[i].fullKCache)) {
                return NO;
            }
            if (!NMOERegisterBuffer(rt,
                                    [[NSString stringWithFormat:@"layer.%02zu.full.v_cache", i] UTF8String],
                                    kvBytes,
                                    YES,
                                    &rt->layerState[i].fullVCache)) {
                return NO;
            }
        } else {
            size_t convBytes = 3u * kNMOELinearConvDim * sizeof(float);
            size_t deltaBytes = kNMOELinearVHeads * kNMOELinearValueDim * kNMOELinearKeyDim * sizeof(float);
            if (!NMOERegisterBuffer(rt,
                                    [[NSString stringWithFormat:@"layer.%02zu.linear.conv_state", i] UTF8String],
                                    convBytes,
                                    YES,
                                    &rt->layerState[i].linearConvState)) {
                return NO;
            }
            if (!NMOERegisterBuffer(rt,
                                    [[NSString stringWithFormat:@"layer.%02zu.linear.delta_state", i] UTF8String],
                                    deltaBytes,
                                    YES,
                                    &rt->layerState[i].linearDeltaState)) {
                return NO;
            }
        }
    }

    return YES;
}

static BOOL NMOELoadPipelineStates(nmoe_runtime *rt, NSError **error) {
    if (rt == NULL || rt->backend == NULL) {
        if (error != NULL) {
            *error = NMOEMakeError(50, @"backend is missing");
        }
        return NO;
    }

    for (NSInteger kind = 0; kind < NMOE_BACKEND_KERNEL_COUNT; ++kind) {
        void *pipeline = nmoe_backend_pipeline_state(rt->backend, (nmoe_backend_kernel_kind)kind);
        if (pipeline == NULL) {
            if (error != NULL) {
                *error = NMOEMakeError(51, [NSString stringWithFormat:@"missing pipeline state for kernel kind %ld", (long)kind]);
            }
            return NO;
        }
        rt->pipelineStates[kind] = pipeline;
    }
    return YES;
}

static BOOL NMOERuntimeInitialise(nmoe_runtime *rt, NSError **error) {
    if (rt == NULL) {
        if (error != NULL) {
            *error = NMOEMakeError(60, @"runtime is missing");
        }
        return NO;
    }

    NSString *modelDir = NMOEStringFromC(rt->modelPath);
    if (modelDir.length == 0) {
        modelDir = @"qwen36_35b";
    }

    NSString *weightsBin = NMOEJoinPath(modelDir, @"model_weights.bin");
    NSString *weightsJson = NMOEJoinPath(modelDir, @"model_weights.json");
    rt->weights = nmoe_weight_open(weightsBin.fileSystemRepresentation, weightsJson.fileSystemRepresentation, rt->quiet ? 1 : 0);
    if (rt->weights == NULL) {
        if (error != NULL) {
            *error = NMOEMakeError(61, [NSString stringWithFormat:@"failed to load weights from %@", modelDir]);
        }
        return NO;
    }

    NSString *tokenizerPath = NMOEJoinPath(modelDir, @"tokenizer.bin");
    NSString *vocabPath = NMOEJoinPath(modelDir, @"vocab.bin");
    rt->tokenizer = nmoe_tokenizer_load(tokenizerPath.fileSystemRepresentation, rt->quiet ? 1 : 0);
    rt->vocab = nmoe_vocab_load(vocabPath.fileSystemRepresentation, rt->quiet ? 1 : 0);
    if (rt->tokenizer == NULL || rt->vocab == NULL) {
        if (error != NULL) {
            *error = NMOEMakeError(62, @"failed to load tokenizer or vocab");
        }
        return NO;
    }

    int q4Exists = 0;
    int q2Exists = 0;
    NSString *q4Layer = NMOEJoinPath(NMOEJoinPath(modelDir, @"packed_experts"), @"layer_00.bin");
    NSString *q2Layer = NMOEJoinPath(NMOEJoinPath(modelDir, @"packed_experts_2bit"), @"layer_00.bin");
    q4Exists = access(q4Layer.fileSystemRepresentation, R_OK) == 0;
    q2Exists = access(q2Layer.fileSystemRepresentation, R_OK) == 0;

    rt->quantBits = rt->cfg.quant_bits;
    if (rt->quantBits == 0) {
        rt->quantBits = (q2Exists && !q4Exists) ? 2 : 4;
    }
    if (rt->quantBits != 2 && rt->quantBits != 4) {
        rt->quantBits = 4;
    }

    NSString *expertDir = rt->quantBits == 2 ? NMOEJoinPath(modelDir, @"packed_experts_2bit") : NMOEJoinPath(modelDir, @"packed_experts");
    if (!NMOEPathExists(expertDir, NULL)) {
        if (error != NULL) {
            *error = NMOEMakeError(63, [NSString stringWithFormat:@"missing expert directory %@", expertDir]);
        }
        return NO;
    }
    rt->expertDirectory = strdup(expertDir.UTF8String);
    if (rt->expertDirectory == NULL) {
        if (error != NULL) {
            *error = NMOEMakeError(64, @"failed to allocate expert directory string");
        }
        return NO;
    }

    rt->expertStore = nmoe_expert_store_open(modelDir.fileSystemRepresentation, rt->quantBits, rt->quiet ? 1 : 0);
    if (rt->expertStore == NULL) {
        if (error != NULL) {
            *error = NMOEMakeError(65, @"failed to open expert store");
        }
        return NO;
    }

    rt->backend = nmoe_backend_create(rt->quiet ? 1 : 0);
    if (rt->backend == NULL) {
        if (error != NULL) {
            *error = NMOEMakeError(66, @"failed to create Metal backend");
        }
        return NO;
    }

    nmoe_backend_set_weight_buffer(rt->backend, rt->weights->data, rt->weights->size);

    rt->sequenceCapacity = MAX((size_t)4096, (size_t)rt->cfg.max_tokens + 512u);
    if (rt->sequenceCapacity < 1024u) {
        rt->sequenceCapacity = 1024u;
    }

    if (!NMOEAllocateStateBuffers(rt, error)) {
        return NO;
    }

    for (size_t i = 0; i < kNMOELayers; ++i) {
        if (!NMOELoadLayerWeights(rt, (int)i, error)) {
            return NO;
        }
        if (!NMOELoadExpertLayer(rt, (int)i, error)) {
            return NO;
        }
    }

    if (!NMOELoadPipelineStates(rt, error)) {
        return NO;
    }

    nmoe_backend_reset_linear_state(rt->backend);
    rt->sequencePosition = 0;
    rt->inThink = NO;
    rt->thinkCount = 0;

    if (!rt->quiet) {
        fprintf(stderr,
                "nmoe runtime: model=%s quant=%d experts=%d seq_capacity=%zu backend=metal\n",
                modelDir.UTF8String,
                rt->quantBits,
                rt->cfg.experts,
                rt->sequenceCapacity);
    }

    return YES;
}

static void NMOERuntimeRelease(nmoe_runtime *rt) {
    if (rt == NULL) {
        return;
    }

    for (size_t i = 0; i < kNMOELayers; ++i) {
        NMOECloseMappedFile(&rt->layerState[i].expertFile);
    }

    if (rt->backend != NULL) {
        nmoe_backend_destroy(rt->backend);
        rt->backend = NULL;
    }
    if (rt->weights != NULL) {
        nmoe_weight_close(rt->weights);
        rt->weights = NULL;
    }
    if (rt->tokenizer != NULL) {
        nmoe_tokenizer_free(rt->tokenizer);
        rt->tokenizer = NULL;
    }
    if (rt->vocab != NULL) {
        nmoe_vocab_free(rt->vocab);
        rt->vocab = NULL;
    }
    if (rt->expertStore != NULL) {
        nmoe_expert_store_close(rt->expertStore);
        rt->expertStore = NULL;
    }
    free(rt->modelPath);
    rt->modelPath = NULL;
    free(rt->expertDirectory);
    rt->expertDirectory = NULL;
}

static void NMOEFullAttentionStep(nmoe_runtime *rt,
                                  int layerIndex,
                                  const float *residual,
                                  float *layerOutput,
                                  size_t position,
                                  NMOEPerfStats *stats) {
    if (!NMOEFullAttentionStepMetal(rt, layerIndex, residual, layerOutput, position, stats)) {
        fprintf(stderr, "nmoe runtime error: Metal full attention step failed at layer %d\n", layerIndex);
        abort();
    }
}

static void NMOELinearAttentionStep(nmoe_runtime *rt,
                                    int layerIndex,
                                    const float *residual,
                                    float *layerOutput,
                                    size_t position,
                                    NMOEPerfStats *stats) {
    if (!NMOELinearAttentionStepMetal(rt, layerIndex, residual, layerOutput, position, stats)) {
        fprintf(stderr, "nmoe runtime error: Metal linear attention step failed at layer %d\n", layerIndex);
        abort();
    }
}

static BOOL NMOEFullAttentionStepMetal(nmoe_runtime *rt,
                                       int layerIndex,
                                       const float *residual,
                                       float *layerOutput,
                                       size_t position,
                                       NMOEPerfStats *stats) {
    if (rt == NULL || rt->backend == NULL || residual == NULL || layerOutput == NULL) return NO;

    NMOELayerState *state = &rt->layerState[layerIndex];
    const NMOELayerWeights *layer = &rt->layers[layerIndex];
    NSString *lp = [NSString stringWithFormat:@"model.layers.%d", layerIndex];
    void *residualBuf = NMOEHiddenBufferForPointer(rt, residual);
    void *outputBuf = NMOEHiddenBufferForPointer(rt, layerOutput);
    if (residualBuf == NULL || outputBuf == NULL) return NO;

    id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)nmoe_backend_command_queue(rt->backend);
    if (!NMOEInitExpertIOBuffers(rt)) return NO;
    BOOL cpuRouter = NMOEUseCPURouter();
    NMOEExpertLayerFile *expertFile = &state->expertFile;
    BOOL gpuRoutedExperts = (NMOEUseRoutedExperts() && rt->quantBits == 4 && !cpuRouter &&
                             expertFile->metalBuffer != NULL && g_expertIOExpertActBuf != NULL);

    size_t seqLen = position + 1u;
    float invScale = 1.0f / sqrtf((float)kNMOEHeadDim);

    // ================================================================
    // Single unified command buffer: ALL sync GPU ops
    // ================================================================
    double t0 = NMOENowSeconds();
    {
        id<MTLCommandBuffer> cmd = [queue commandBuffer];
        if (cmd == nil) return NO;

        // --- input RMS norm ---
        BOOL deferredInputNorm = NMOEDeferredInputNormReadyForLayer(layerIndex);
        if (!deferredInputNorm &&
            !NMOEEncodeRMSNormTensor(cmd, rt, &layer->inputNorm,
                                     residualBuf, 0, rt->normBuffer, 0,
                                     kNMOEHiddenDim, YES, 1e-6f)) return NO;

        // --- Q/K/V projections ---
        if (!NMOEEncodeDequantMatVecTensor(cmd, rt, &layer->u.full.qProj,
                                           rt->normBuffer, 0, rt->fullQProjBuffer, 0, 8192u, kNMOEHiddenDim)) return NO;
        if (!NMOEEncodeDequantMatVecTensor(cmd, rt, &layer->u.full.kProj,
                                           rt->normBuffer, 0, rt->fullKProjBuffer, 0, 512u, kNMOEHiddenDim)) return NO;
        if (!NMOEEncodeDequantMatVecTensor(cmd, rt, &layer->u.full.vProj,
                                           rt->normBuffer, 0, rt->fullVProjBuffer, 0, 512u, kNMOEHiddenDim)) return NO;

        // --- Q/K norm + RoPE + sigmoid gate per head (fused) ---
        if (!NMOEEncodeFullQKPrep(cmd, rt, layer, YES, position)) return NO;

        // --- GPU copy K/V proj into KV cache (so attention sees current token) ---
        {
            id<MTLComputePipelineState> copyPipe = (__bridge id<MTLComputePipelineState>)nmoe_backend_pipeline_state(rt->backend, NMOE_BACKEND_KERNEL_COPY_F32);
            if (copyPipe == nil) return NO;
            uint32_t kvCount = (uint32_t)kNMOEHeadDim;
            for (size_t kh = 0; kh < kNMOEFullKVHeads; ++kh) {
                size_t offset = kh * kNMOEHeadDim * sizeof(float);
                size_t cacheOff = (position * (kNMOEFullKVHeads * kNMOEHeadDim) + kh * kNMOEHeadDim) * sizeof(float);
                if (!NMOEEncodeKernel(cmd, copyPipe, kNMOEHeadDim, ^(id<MTLComputeCommandEncoder> enc) {
                    [enc setBuffer:NMOEBridgeBuffer(rt->fullKProjBuffer) offset:offset atIndex:0];
                    [enc setBuffer:NMOEBridgeBuffer(state->fullKCache) offset:cacheOff atIndex:1];
                    [enc setBytes:&kvCount length:sizeof(kvCount) atIndex:2];
                })) return NO;
                if (!NMOEEncodeKernel(cmd, copyPipe, kNMOEHeadDim, ^(id<MTLComputeCommandEncoder> enc) {
                    [enc setBuffer:NMOEBridgeBuffer(rt->fullVProjBuffer) offset:offset atIndex:0];
                    [enc setBuffer:NMOEBridgeBuffer(state->fullVCache) offset:cacheOff atIndex:1];
                    [enc setBytes:&kvCount length:sizeof(kvCount) atIndex:2];
                })) return NO;
            }
        }

        // --- attention scores (batched per head*pos) ---
        {
            id<MTLComputePipelineState> p = (__bridge id<MTLComputePipelineState>)nmoe_backend_pipeline_state(rt->backend, NMOE_BACKEND_KERNEL_ATTN_SCORES_BATCHED);
            if (p == nil) return NO;
            NMOEAttentionArgs args = {
                .seqLen = (uint32_t)seqLen, .seqStride = (uint32_t)rt->sequenceCapacity,
                .headDim = (uint32_t)kNMOEHeadDim, .qStride = 512u, .kvStride = (uint32_t)kNMOEHeadDim,
                .cacheStride = (uint32_t)(kNMOEFullKVHeads * kNMOEHeadDim),
                .fullHeads = (uint32_t)kNMOEFullHeads, .fullKVHeads = (uint32_t)kNMOEFullKVHeads,
                .invScale = invScale,
            };
            if (!NMOEEncodeKernelTG(cmd, p, (NSUInteger)(kNMOEFullHeads * seqLen), 256, ^(id<MTLComputeCommandEncoder> enc) {
                [enc setBuffer:NMOEBridgeBuffer(rt->fullQProjBuffer) offset:0 atIndex:0];
                [enc setBuffer:NMOEBridgeBuffer(state->fullKCache) offset:0 atIndex:1];
                [enc setBuffer:NMOEBridgeBuffer(rt->attnScoresBuffer) offset:0 atIndex:2];
                [enc setBytes:&args length:sizeof(args) atIndex:3];
            })) return NO;
        }
        // --- attention softmax ---
        {
            id<MTLComputePipelineState> sp = (__bridge id<MTLComputePipelineState>)nmoe_backend_pipeline_state(rt->backend, NMOE_BACKEND_KERNEL_ATTN_SOFTMAX_BATCHED);
            if (sp == nil) return NO;
            NMOEAttentionArgs sargs = {
                .seqLen = (uint32_t)seqLen, .seqStride = (uint32_t)rt->sequenceCapacity,
                .headDim = (uint32_t)kNMOEHeadDim, .fullHeads = (uint32_t)kNMOEFullHeads,
                .fullKVHeads = (uint32_t)kNMOEFullKVHeads,
            };
            if (!NMOEEncodeKernelTG(cmd, sp, (NSUInteger)kNMOEFullHeads, 256, ^(id<MTLComputeCommandEncoder> enc) {
                [enc setBuffer:NMOEBridgeBuffer(rt->attnScoresBuffer) offset:0 atIndex:0];
                [enc setBuffer:NMOEBridgeBuffer(rt->attnProbsBuffer) offset:0 atIndex:1];
                [enc setBytes:&sargs length:sizeof(sargs) atIndex:2];
            })) return NO;
        }
        // --- attention values ---
        {
            id<MTLComputePipelineState> vp = (__bridge id<MTLComputePipelineState>)nmoe_backend_pipeline_state(rt->backend, NMOE_BACKEND_KERNEL_ATTN_VALUES_BATCHED);
            if (vp == nil) return NO;
            NMOEAttentionArgs vargs = {
                .seqLen = (uint32_t)seqLen, .seqStride = (uint32_t)rt->sequenceCapacity,
                .headDim = (uint32_t)kNMOEHeadDim, .qStride = 512u, .kvStride = (uint32_t)kNMOEHeadDim,
                .cacheStride = (uint32_t)(kNMOEFullKVHeads * kNMOEHeadDim),
                .fullHeads = (uint32_t)kNMOEFullHeads, .fullKVHeads = (uint32_t)kNMOEFullKVHeads,
            };
            if (!NMOEEncodeKernel(cmd, vp, (NSUInteger)(kNMOEFullHeads * kNMOEHeadDim), ^(id<MTLComputeCommandEncoder> enc) {
                [enc setBuffer:NMOEBridgeBuffer(rt->attnProbsBuffer) offset:0 atIndex:0];
                [enc setBuffer:NMOEBridgeBuffer(state->fullVCache) offset:0 atIndex:1];
                [enc setBuffer:NMOEBridgeBuffer(rt->fullQProjBuffer) offset:0 atIndex:2];
                [enc setBuffer:NMOEBridgeBuffer(rt->fullAttnBuffer) offset:0 atIndex:3];
                [enc setBytes:&vargs length:sizeof(vargs) atIndex:4];
            })) return NO;
        }

        // --- o_proj + residual + post_norm + router + shared gate/up/score ---
        if (!NMOEEncodeDequantMatVecTensor(cmd, rt, &layer->u.full.oProj,
                                           rt->fullAttnBuffer, 0, outputBuf, 0, kNMOEHiddenDim, 4096u)) return NO;
        if (!NMOEEncodeResidualAdd(cmd, rt, outputBuf, 0, residualBuf, 0, outputBuf, 0, kNMOEHiddenDim)) return NO;
        if (!NMOEEncodeRMSNormTensor(cmd, rt, &layer->postNorm,
                                     outputBuf, 0, rt->normBuffer, 0,
                                     kNMOEHiddenDim, YES, 1e-6f)) return NO;
        if (rt->quantBits == 4) {
            if (!NMOEEncodeRouteSharedQ4Tensors(cmd, rt,
                                                &layer->routerGate, &layer->sharedGateProj,
                                                &layer->sharedUpProj, &layer->sharedGateScore,
                                                rt->normBuffer, 0, rt->routerScoresBuffer, 0,
                                                g_expertIOSharedActBuf, 0, rt->sharedOutBuffer, 0)) return NO;
        } else {
            if (!NMOEEncodeDequantMatVec(cmd, rt, [lp stringByAppendingString:@".mlp.gate"],
                                          rt->normBuffer, 0, rt->routerScoresBuffer, 0, 256u, kNMOEHiddenDim)) return NO;
            if (!NMOEEncodeGateUpQ4(cmd, rt,
                                    [lp stringByAppendingString:@".mlp.shared_expert.gate_proj"],
                                    [lp stringByAppendingString:@".mlp.shared_expert.up_proj"],
                                    rt->normBuffer, 0, g_expertIOSharedActBuf, 0)) return NO;
            if (!NMOEEncodeDequantMatVec(cmd, rt, [lp stringByAppendingString:@".mlp.shared_expert_gate"],
                                          rt->normBuffer, 0, rt->sharedOutBuffer, 0, 1u, kNMOEHiddenDim)) return NO;
        }
        if (!cpuRouter && !NMOEEncodeRouteTopK(cmd, rt, (size_t)rt->cfg.experts)) return NO;
        if (gpuRoutedExperts) {
            if (!NMOEEncodeDequantMatVecTensor(cmd, rt, &layer->sharedDownProj,
                                               g_expertIOSharedActBuf, 0, g_expertIOSharedDownBuf, 0,
                                               kNMOEHiddenDim, 512u)) return NO;
            void *expertOutPtrs[8] = {0};
            for (int i = 0; i < 8; i++) expertOutPtrs[i] = g_expertOutBuffers[i];
            if (!NMOEEncodeRoutedExpertQ4(cmd, rt, expertFile, (size_t)rt->cfg.experts,
                                          g_expertIOExpertActBuf, expertOutPtrs)) return NO;
            if (!NMOEEncodeWeightedExpertSumRouted(cmd, rt, outputBuf, g_expertIOSharedDownBuf,
                                                   outputBuf, expertOutPtrs, kNMOEHiddenDim,
                                                   (size_t)rt->cfg.experts)) return NO;
            BOOL nextInputNormReady = NO;
            int nextInputNormLayer = layerIndex + 1;
            if (nextInputNormLayer < (int)kNMOELayers) {
                if (!NMOEEncodeRMSNormTensor(cmd, rt, &rt->layers[nextInputNormLayer].inputNorm,
                                             outputBuf, 0, rt->normBuffer, 0,
                                             kNMOEHiddenDim, YES, 1e-6f)) return NO;
                nextInputNormReady = YES;
            }
            [cmd commit];
            g_deferredExperts.active = YES;
            g_deferredExperts.gpuCombined = YES;
            g_deferredExperts.nextInputNormReady = nextInputNormReady;
            g_deferredExperts.nextInputNormLayer = nextInputNormReady ? nextInputNormLayer : -1;
            g_deferredExperts.cmdExperts = cmd;
            for (int i = 0; i < 8; ++i) g_deferredExperts.expertBuffers[i] = nil;
            if (stats != NULL) stats->context += NMOENowSeconds() - t0;
            return YES;
        }

        [cmd commit];
        double syncStart = NMOENowSeconds();
        [cmd waitUntilCompleted];
        if (stats != NULL) stats->contextSync += NMOENowSeconds() - syncStart;
    }
    double deferredStart = NMOENowSeconds();
    NMOEFinalizeDeferredExperts(rt);
    if (stats != NULL) stats->deferredWait += NMOENowSeconds() - deferredStart;

    // KV cache updated on GPU inside command buffer — no CPU copy needed

    double ctx_end = NMOENowSeconds();
    if (stats != NULL) stats->context += ctx_end - t0;
    t0 = NMOENowSeconds();

    float selectedValues[kNMOEMaxExperts] = {0};
    size_t selectedIndices[kNMOEMaxExperts] = {0};
    size_t selectedCount = cpuRouter
        ? NMOESelectRouteTopKCPU(rt, (size_t)rt->cfg.experts, selectedIndices, selectedValues)
        : NMOEReadRouteTopKMetal(rt, (size_t)rt->cfg.experts, selectedIndices, selectedValues);
    if (selectedCount == 0) return NO;
    if (stats != NULL) stats->route += NMOENowSeconds() - t0;

    float sharedGateScore = NMOEFloatBuffer(rt->sharedOutBuffer)[0];

    // ---- Expert I/O: async parallel pread ----
    NMOEExpertOffsets offsets = *NMOEExpertOffsetsForBits(rt->quantBits);
    int valid[8] = {0};
    __strong id<MTLBuffer> expertBuffers[8] = {nil};
    if (!NMOEPrepareExpertBuffers(rt, expertFile, selectedIndices, selectedCount, expertBuffers, valid, stats)) return NO;

    // Copy h_mid (current layer output after o_proj+residual) and normed input
    memcpy(((__bridge id<MTLBuffer>)g_expertIOHMidBuf).contents, NMOEFloatBuffer(outputBuf), kNMOEHiddenDim * sizeof(float));
    memcpy(((__bridge id<MTLBuffer>)g_expertIOInputBuf).contents, NMOEFloatBuffer(rt->normBuffer), kNMOEHiddenDim * sizeof(float));

    // ---- Expert GPU stage (DEFERRED) ----
    {
        id<MTLCommandBuffer> cmd = [queue commandBuffer];
        if (cmd == nil) return NO;

        BOOL useFusedDownCombineQ4 = (rt->quantBits == 4 && g_expertIOExpertActBuf != NULL &&
                                      NMOEUseFusedDownCombineQ4());
        if (!useFusedDownCombineQ4) {
            if (!NMOEEncodeDequantMatVecTensor(cmd, rt, &layer->sharedDownProj,
                                               g_expertIOSharedActBuf, 0, g_expertIOSharedDownBuf, 0,
                                               kNMOEHiddenDim, 512u)) return NO;
        }

	        // Per expert: gate + up + swiglu + down
	        id<MTLComputePipelineState> expertPipe = (__bridge id<MTLComputePipelineState>)nmoe_backend_pipeline_state(rt->backend,
	            rt->quantBits == 2 ? NMOE_BACKEND_KERNEL_DEQUANT_MATVEC_Q2 : NMOE_BACKEND_KERNEL_DEQUANT_MATVEC_Q4);
	        id<MTLComputePipelineState> swigluPipe = (__bridge id<MTLComputePipelineState>)nmoe_backend_pipeline_state(rt->backend, NMOE_BACKEND_KERNEL_MOE_EXPERT_GATE_UP);
	        id<MTLComputePipelineState> gateUpQ4Pipe = (__bridge id<MTLComputePipelineState>)nmoe_backend_pipeline_state(rt->backend, NMOE_BACKEND_KERNEL_EXPERT_GATE_UP_Q4);
	        id<MTLComputePipelineState> gateUpQ4BatchedPipe = (__bridge id<MTLComputePipelineState>)nmoe_backend_pipeline_state(rt->backend, NMOE_BACKEND_KERNEL_EXPERT_GATE_UP_Q4_BATCHED);
	        id<MTLComputePipelineState> downQ4BatchedPipe = (__bridge id<MTLComputePipelineState>)nmoe_backend_pipeline_state(rt->backend, NMOE_BACKEND_KERNEL_EXPERT_DOWN_Q4_BATCHED);
	        id<MTLBuffer> expertInputMTL = (__bridge id<MTLBuffer>)g_expertIOInputBuf;
	        id<MTLBuffer> expertGateMTL = NMOEBridgeBuffer(rt->expertGateBuffer);
	        id<MTLBuffer> expertUpMTL = NMOEBridgeBuffer(rt->expertUpBuffer);
	        id<MTLBuffer> expertActMTL = NMOEBridgeBuffer(rt->expertActBuffer);
	        if (expertPipe == nil || swigluPipe == nil || expertInputMTL == nil ||
	            expertGateMTL == nil || expertUpMTL == nil || expertActMTL == nil) return NO;
	        BOOL usedBatchedQ4 = NO;
	        BOOL usedFusedDownCombineQ4 = NO;
	        if (rt->quantBits == 4 && g_expertIOExpertActBuf != NULL &&
	            gateUpQ4BatchedPipe != nil && downQ4BatchedPipe != nil) {
	            id<MTLBuffer> expertActBatchMTL = (__bridge id<MTLBuffer>)g_expertIOExpertActBuf;
	            if (!NMOEEncodeExpertGateUpQ4Batched(cmd, rt, expertBuffers, selectedCount,
	                                                  expertInputMTL, expertActBatchMTL, &offsets)) return NO;
	            if (useFusedDownCombineQ4) {
		                if (!NMOEEncodeExpertDownCombineQ4Tensor(cmd, rt, expertBuffers, selectedCount,
		                                                         expertActBatchMTL, g_expertIOHMidBuf,
		                                                         g_expertIOSharedActBuf, g_expertIOSharedDownBuf, outputBuf,
		                                                         selectedValues, sharedGateScore,
		                                                         &layer->sharedDownProj, &offsets)) return NO;
	                usedFusedDownCombineQ4 = YES;
	            } else {
	                void *expertOutPtrs[8] = {0};
	                for (int i = 0; i < 8; i++) expertOutPtrs[i] = g_expertOutBuffers[i];
	                if (!NMOEEncodeExpertDownQ4Batched(cmd, rt, expertBuffers, expertOutPtrs,
	                                                   selectedCount, expertActBatchMTL, &offsets)) return NO;
	            }
	            usedBatchedQ4 = YES;
	        }
	        for (size_t idx = 0; !usedBatchedQ4 && idx < selectedCount; ++idx) {
	            if (!valid[idx]) continue;
	            id<MTLBuffer> ebuf = expertBuffers[idx];
	            if (ebuf == nil) return NO;
	            id<MTLBuffer> expertOutMTL = (__bridge id<MTLBuffer>)g_expertOutBuffers[idx];
	            if (expertOutMTL == nil) return NO;
	            if (rt->quantBits == 4 && gateUpQ4Pipe != nil) {
	                id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
	                if (enc == nil) return NO;
	                if (!NMOEEncodeExpertGateUpQ4OnEncoder(enc, gateUpQ4Pipe, ebuf,
	                                                       &offsets, expertInputMTL, expertActMTL)) return NO;
	                if (!NMOEEncodeDequantMatVecFromBufferOnEncoder(enc, expertPipe, ebuf,
	                                                                 offsets.down_weight, offsets.down_scales, offsets.down_biases,
	                                                                 expertActMTL, 0, expertOutMTL, 0,
	                                                                 kNMOEHiddenDim, 512u, rt->quantBits)) return NO;
	                [enc endEncoding];
	                continue;
	            }
	            {
	                id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
	                if (enc == nil) return NO;
	                if (!NMOEEncodeDequantMatVecFromBufferOnEncoder(enc, expertPipe, ebuf,
	                                                                 offsets.gate_weight, offsets.gate_scales, offsets.gate_biases,
	                                                                 expertInputMTL, 0, expertGateMTL, 0,
	                                                                 512u, kNMOEHiddenDim, rt->quantBits)) return NO;
	                if (!NMOEEncodeDequantMatVecFromBufferOnEncoder(enc, expertPipe, ebuf,
	                                                                 offsets.up_weight, offsets.up_scales, offsets.up_biases,
	                                                                 expertInputMTL, 0, expertUpMTL, 0,
	                                                                 512u, kNMOEHiddenDim, rt->quantBits)) return NO;
	                [enc endEncoding];
	            }
	            {
	                id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
	                if (enc == nil) return NO;
	                if (!NMOEEncodeMoeExpertGateUpOnEncoder(enc, swigluPipe,
	                                                        expertGateMTL, 0, expertUpMTL, 0, expertActMTL, 0, 512u)) return NO;
	                if (!NMOEEncodeDequantMatVecFromBufferOnEncoder(enc, expertPipe, ebuf,
	                                                                 offsets.down_weight, offsets.down_scales, offsets.down_biases,
	                                                                 expertActMTL, 0, expertOutMTL, 0,
	                                                                 kNMOEHiddenDim, 512u, rt->quantBits)) return NO;
	                [enc endEncoding];
	            }
	        }

        // Fused weighted combine
        if (!usedFusedDownCombineQ4) {
            void *expertOutPtrs[8] = {0};
            for (int i = 0; i < 8; i++) expertOutPtrs[i] = g_expertOutBuffers[i];
            if (!NMOEEncodeWeightedExpertSum(cmd, rt, g_expertIOHMidBuf, g_expertIOSharedDownBuf,
                                              outputBuf, expertOutPtrs, kNMOEHiddenDim, selectedCount,
                                              selectedValues, sharedGateScore)) return NO;
        }
        BOOL nextInputNormReady = NO;
        int nextInputNormLayer = layerIndex + 1;
        if (nextInputNormLayer < (int)kNMOELayers) {
            if (!NMOEEncodeRMSNormTensor(cmd, rt, &rt->layers[nextInputNormLayer].inputNorm,
                                         outputBuf, 0, rt->normBuffer, 0,
                                         kNMOEHiddenDim, YES, 1e-6f)) return NO;
            nextInputNormReady = YES;
        }

        [cmd commit];
        g_deferredExperts.active = YES;
        g_deferredExperts.gpuCombined = YES;
        g_deferredExperts.nextInputNormReady = nextInputNormReady;
        g_deferredExperts.nextInputNormLayer = nextInputNormReady ? nextInputNormLayer : -1;
        g_deferredExperts.cmdExperts = cmd;
        for (int i = 0; i < 8; ++i) {
            g_deferredExperts.expertBuffers[i] = i < (int)selectedCount ? expertBuffers[i] : nil;
        }
        for (int i = 0; i < (int)selectedCount; i++) {
            g_deferredExperts.expertWeights[i] = selectedValues[i];
            g_deferredExperts.valid[i] = valid[i];
        }
    }
    if (stats != NULL) stats->expert += NMOENowSeconds() - t0;
    return YES;
}

static BOOL NMOEFullAttentionStateOnlyMetal(nmoe_runtime *rt,
                                            int layerIndex,
                                            const float *residual,
                                            size_t position,
                                            NMOEPerfStats *stats) {
    if (rt == NULL || rt->backend == NULL || residual == NULL) return NO;

    NMOELayerState *state = &rt->layerState[layerIndex];
    const NMOELayerWeights *layer = &rt->layers[layerIndex];
    void *residualBuf = NMOEHiddenBufferForPointer(rt, residual);
    if (residualBuf == NULL) return NO;

    id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)nmoe_backend_command_queue(rt->backend);
    if (queue == nil) return NO;

    double t0 = NMOENowSeconds();
    id<MTLCommandBuffer> cmd = [queue commandBuffer];
    if (cmd == nil) return NO;

    if (!NMOEEncodeRMSNormTensor(cmd, rt, &layer->inputNorm,
                                 residualBuf, 0, rt->normBuffer, 0,
                                 kNMOEHiddenDim, YES, 1e-6f)) return NO;
    if (!NMOEEncodeDequantMatVecTensor(cmd, rt, &layer->u.full.kProj,
                                       rt->normBuffer, 0, rt->fullKProjBuffer, 0,
                                       512u, kNMOEHiddenDim)) return NO;
    if (!NMOEEncodeDequantMatVecTensor(cmd, rt, &layer->u.full.vProj,
                                       rt->normBuffer, 0, rt->fullVProjBuffer, 0,
                                       512u, kNMOEHiddenDim)) return NO;

    if (!NMOEEncodeFullQKPrep(cmd, rt, layer, NO, position)) return NO;

    id<MTLComputePipelineState> copyPipe = (__bridge id<MTLComputePipelineState>)nmoe_backend_pipeline_state(rt->backend, NMOE_BACKEND_KERNEL_COPY_F32);
    if (copyPipe == nil) return NO;
    uint32_t kvCount = (uint32_t)kNMOEHeadDim;
    for (size_t kh = 0; kh < kNMOEFullKVHeads; ++kh) {
        size_t offset = kh * kNMOEHeadDim * sizeof(float);
        size_t cacheOff = (position * (kNMOEFullKVHeads * kNMOEHeadDim) + kh * kNMOEHeadDim) * sizeof(float);
        if (!NMOEEncodeKernel(cmd, copyPipe, kNMOEHeadDim, ^(id<MTLComputeCommandEncoder> enc) {
            [enc setBuffer:NMOEBridgeBuffer(rt->fullKProjBuffer) offset:offset atIndex:0];
            [enc setBuffer:NMOEBridgeBuffer(state->fullKCache) offset:cacheOff atIndex:1];
            [enc setBytes:&kvCount length:sizeof(kvCount) atIndex:2];
        })) return NO;
        if (!NMOEEncodeKernel(cmd, copyPipe, kNMOEHeadDim, ^(id<MTLComputeCommandEncoder> enc) {
            [enc setBuffer:NMOEBridgeBuffer(rt->fullVProjBuffer) offset:offset atIndex:0];
            [enc setBuffer:NMOEBridgeBuffer(state->fullVCache) offset:cacheOff atIndex:1];
            [enc setBytes:&kvCount length:sizeof(kvCount) atIndex:2];
        })) return NO;
    }

    [cmd commit];
    double syncStart = NMOENowSeconds();
    [cmd waitUntilCompleted];
    if (stats != NULL) stats->contextSync += NMOENowSeconds() - syncStart;
    double deferredStart = NMOENowSeconds();
    NMOEFinalizeDeferredExperts(rt);
    if (stats != NULL) stats->deferredWait += NMOENowSeconds() - deferredStart;

    if (stats != NULL) {
        stats->context += NMOENowSeconds() - t0;
    }
    return YES;
}

static BOOL NMOELinearAttentionStepMetal(nmoe_runtime *rt,
                                         int layerIndex,
                                         const float *residual,
                                         float *layerOutput,
                                         size_t position,
                                         NMOEPerfStats *stats) {
    (void)position;
    if (rt == NULL || rt->backend == NULL || residual == NULL || layerOutput == NULL) return NO;

    NMOELayerState *state = &rt->layerState[layerIndex];
    const NMOELayerWeights *layer = &rt->layers[layerIndex];
    NSString *lp = [NSString stringWithFormat:@"model.layers.%d", layerIndex];
    NSUInteger convWeightOff = 0;
    if (!NMOEWeightPointerOffset(rt, layer->u.linear.convWeight.weight, &convWeightOff)) return NO;
    void *residualBuf = NMOEHiddenBufferForPointer(rt, residual);
    void *outputBuf = NMOEHiddenBufferForPointer(rt, layerOutput);
    if (residualBuf == NULL || outputBuf == NULL) return NO;

    id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)nmoe_backend_command_queue(rt->backend);
    if (!NMOEInitExpertIOBuffers(rt)) return NO;
    BOOL cpuRouter = NMOEUseCPURouter();
    NMOEExpertLayerFile *expertFile = &state->expertFile;
    BOOL gpuRoutedExperts = (NMOEUseRoutedExperts() && rt->quantBits == 4 && !cpuRouter &&
                             expertFile->metalBuffer != NULL && g_expertIOExpertActBuf != NULL);

    float invScale = 1.0f / sqrtf((float)kNMOELinearKeyDim);

    // ================================================================
    // Single unified command buffer: ALL sync GPU ops
    // input_norm -> QKV/Z/B/A proj -> conv1d -> Q/K norm ->
    // compute_decay_beta -> delta_net -> gated_rms_norm ->
    // out_proj -> residual -> post_norm -> router -> shared gate/up/score
    // ================================================================
    double t0 = NMOENowSeconds();
    id<MTLCommandBuffer> contextCmd = [queue commandBuffer];
    if (contextCmd == nil) return NO;
    {
        // --- input RMS norm ---
        BOOL deferredInputNorm = NMOEDeferredInputNormReadyForLayer(layerIndex);
        if (!deferredInputNorm &&
            !NMOEEncodeRMSNormTensor(contextCmd, rt, &layer->inputNorm,
                                     residualBuf, 0, rt->normBuffer, 0,
                                     kNMOEHiddenDim, YES, 1e-6f)) return NO;

        // --- QKV/Z/Beta/Alpha projections ---
        if (!NMOEEncodeDequantMatVecTensor(contextCmd, rt, &layer->u.linear.qkvProj,
                                           rt->normBuffer, 0, rt->linearQkvBuffer, 0, 8192u, kNMOEHiddenDim)) return NO;
        if (!NMOEEncodeDequantMatVecTensor(contextCmd, rt, &layer->u.linear.zProj,
                                           rt->normBuffer, 0, rt->linearZBuffer, 0, 4096u, kNMOEHiddenDim)) return NO;
        if (!NMOEEncodeDequantMatVecTensor(contextCmd, rt, &layer->u.linear.betaProj,
                                           rt->normBuffer, 0, rt->linearBetaBuffer, 0, 32u, kNMOEHiddenDim)) return NO;
        if (!NMOEEncodeDequantMatVecTensor(contextCmd, rt, &layer->u.linear.alphaProj,
                                           rt->normBuffer, 0, rt->linearAlphaBuffer, 0, 32u, kNMOEHiddenDim)) return NO;

        // --- Conv1d ---
        {
            id<MTLComputePipelineState> cp = (__bridge id<MTLComputePipelineState>)nmoe_backend_pipeline_state(rt->backend, NMOE_BACKEND_KERNEL_LINEAR_CONV1D);
            if (cp == nil) return NO;
            id<MTLBuffer> wfBuf = (__bridge id<MTLBuffer>)nmoe_backend_weight_buffer(rt->backend);
            NMOELinearConv1DArgs cargs = { .dim = (uint32_t)kNMOELinearConvDim, .stateStride = (uint32_t)kNMOELinearConvDim };
            if (!NMOEEncodeKernel(contextCmd, cp, kNMOELinearConvDim, ^(id<MTLComputeCommandEncoder> enc) {
                [enc setBuffer:NMOEBridgeBuffer(state->linearConvState) offset:0 atIndex:0];
                [enc setBuffer:NMOEBridgeBuffer(rt->linearQkvBuffer) offset:0 atIndex:1];
                [enc setBuffer:wfBuf offset:convWeightOff atIndex:2];
                [enc setBuffer:NMOEBridgeBuffer(rt->linearConvBuffer) offset:0 atIndex:3];
                [enc setBytes:&cargs length:sizeof(cargs) atIndex:4];
            })) return NO;
        }

        // --- Q/K RMS norm per head ---
        {
            id<MTLComputePipelineState> qkPipe = (__bridge id<MTLComputePipelineState>)nmoe_backend_pipeline_state(rt->backend, NMOE_BACKEND_KERNEL_RMS_NORM_QK);
            if (qkPipe == nil) return NO;
            uint32_t keyDim = (uint32_t)kNMOELinearKeyDim;
            float invS = invScale;
            float qkEps = 1e-12f;
            if (!NMOEEncodeKernelTG(contextCmd, qkPipe, kNMOELinearKVHeads, kNMOELinearKeyDim, ^(id<MTLComputeCommandEncoder> enc) {
                [enc setBuffer:NMOEBridgeBuffer(rt->linearConvBuffer) offset:0 atIndex:0];
                [enc setBuffer:NMOEBridgeBuffer(rt->linearConvBuffer) offset:(kNMOELinearKVHeads * kNMOELinearKeyDim * sizeof(float)) atIndex:1];
                [enc setBytes:&keyDim length:sizeof(keyDim) atIndex:2];
                [enc setBytes:&invS length:sizeof(invS) atIndex:3];
                [enc setBytes:&qkEps length:sizeof(qkEps) atIndex:4];
            })) return NO;
        }

        // --- Compute decay/beta ---
        {
            id<MTLComputePipelineState> dbPipe = (__bridge id<MTLComputePipelineState>)nmoe_backend_pipeline_state(rt->backend, NMOE_BACKEND_KERNEL_COMPUTE_DECAY_BETA);
            if (dbPipe == nil) return NO;
            id<MTLBuffer> wfBuf = (__bridge id<MTLBuffer>)nmoe_backend_weight_buffer(rt->backend);
            NSUInteger aLogOff = 0, dtBiasOff = 0;
            if (!NMOEWeightPointerOffset(rt, layer->u.linear.ALog.value, &aLogOff) ||
                !NMOEWeightPointerOffset(rt, layer->u.linear.dtBias.weight, &dtBiasOff)) return NO;
            if (!NMOEEncodeKernel(contextCmd, dbPipe, kNMOELinearVHeads, ^(id<MTLComputeCommandEncoder> enc) {
                [enc setBuffer:NMOEBridgeBuffer(rt->linearAlphaBuffer) offset:0 atIndex:0];
                [enc setBuffer:NMOEBridgeBuffer(rt->linearBetaBuffer) offset:0 atIndex:1];
                [enc setBuffer:wfBuf offset:aLogOff atIndex:2];
                [enc setBuffer:wfBuf offset:dtBiasOff atIndex:3];
                [enc setBuffer:NMOEBridgeBuffer(rt->linearAlphaBuffer) offset:0 atIndex:4];  // reuse alpha buf for g_decay
                [enc setBuffer:NMOEBridgeBuffer(rt->linearBetaBuffer) offset:0 atIndex:5];   // reuse beta buf for beta_gate
            })) return NO;
        }

    }
    // At this point: Q/K norm has been applied to linearConvBuffer.
    // Now run GatedDeltaNet (which reads the normalized Q/K).

    // --- GatedDeltaNet step + gated RMS norm (batched) ---
    {
        // Delta net step (parallel, TUI-proven)
        {
            id<MTLComputePipelineState> dp = (__bridge id<MTLComputePipelineState>)nmoe_backend_pipeline_state(rt->backend, NMOE_BACKEND_KERNEL_GATED_DELTA_NET);
            if (dp == nil) return NO;
            NMOEGatedDeltaNetArgs dargs = {
                .vHeads = (uint32_t)kNMOELinearVHeads, .kvHeads = (uint32_t)kNMOELinearKVHeads,
                .valueDim = (uint32_t)kNMOELinearValueDim, .keyDim = (uint32_t)kNMOELinearKeyDim,
                .qScale = 1.0f, .kScale = 1.0f, .epsilon = 1e-6f,
            };
            if (!NMOEEncodeKernelTG(contextCmd, dp, kNMOELinearVHeads, kNMOELinearValueDim, ^(id<MTLComputeCommandEncoder> enc) {
                [enc setBuffer:NMOEBridgeBuffer(state->linearDeltaState) offset:0 atIndex:0];
                [enc setBuffer:NMOEBridgeBuffer(rt->linearConvBuffer) offset:0 atIndex:1];         // q section
                [enc setBuffer:NMOEBridgeBuffer(rt->linearConvBuffer) offset:(2048u * sizeof(float)) atIndex:2]; // k section
                [enc setBuffer:NMOEBridgeBuffer(rt->linearConvBuffer) offset:(4096u * sizeof(float)) atIndex:3]; // v section
                [enc setBuffer:NMOEBridgeBuffer(rt->linearAlphaBuffer) offset:0 atIndex:4];        // g_decay
                [enc setBuffer:NMOEBridgeBuffer(rt->linearBetaBuffer) offset:0 atIndex:5];         // beta_gate
                [enc setBuffer:NMOEBridgeBuffer(rt->linearOutBuffer) offset:0 atIndex:6];          // raw output
                [enc setBytes:&dargs length:sizeof(dargs) atIndex:7];
            })) return NO;
        }

        // Gated RMS norm
        {
            id<MTLComputePipelineState> gp = (__bridge id<MTLComputePipelineState>)nmoe_backend_pipeline_state(rt->backend, NMOE_BACKEND_KERNEL_GATED_RMS_NORM);
            if (gp == nil) return NO;
            id<MTLBuffer> wfBuf = (__bridge id<MTLBuffer>)nmoe_backend_weight_buffer(rt->backend);
            NSUInteger normWeightOff = 0;
            if (!NMOEWeightPointerOffset(rt, layer->u.linear.normWeight.weight, &normWeightOff)) return NO;
            NMOEGatedDeltaNetArgs gargs = {
                .vHeads = (uint32_t)kNMOELinearVHeads, .kvHeads = (uint32_t)kNMOELinearKVHeads,
                .valueDim = (uint32_t)kNMOELinearValueDim, .keyDim = (uint32_t)kNMOELinearKeyDim,
                .epsilon = 1e-6f,
            };
            if (!NMOEEncodeKernelTG(contextCmd, gp, kNMOELinearVHeads, kNMOELinearValueDim, ^(id<MTLComputeCommandEncoder> enc) {
                [enc setBuffer:NMOEBridgeBuffer(rt->linearOutBuffer) offset:0 atIndex:0];     // raw values
                [enc setBuffer:NMOEBridgeBuffer(rt->linearZBuffer) offset:0 atIndex:1];       // z
                [enc setBuffer:wfBuf offset:normWeightOff atIndex:2];                         // norm weight
                [enc setBuffer:NMOEBridgeBuffer(rt->linearOutBuffer) offset:0 atIndex:3];     // output (in-place)
                [enc setBytes:&gargs length:sizeof(gargs) atIndex:4];
            })) return NO;
        }
    }


    // --- out_proj + residual + post_norm + router + shared gate/up/score ---
    {
        if (!NMOEEncodeDequantMatVecTensor(contextCmd, rt, &layer->u.linear.outProj,
                                           rt->linearOutBuffer, 0, outputBuf, 0, kNMOEHiddenDim, 4096u)) return NO;
        if (!NMOEEncodeResidualAdd(contextCmd, rt, outputBuf, 0, residualBuf, 0, outputBuf, 0, kNMOEHiddenDim)) return NO;
        if (!NMOEEncodeRMSNormTensor(contextCmd, rt, &layer->postNorm,
                                     outputBuf, 0, rt->normBuffer, 0,
                                     kNMOEHiddenDim, YES, 1e-6f)) return NO;
        if (rt->quantBits == 4) {
            if (!NMOEEncodeRouteSharedQ4Tensors(contextCmd, rt,
                                                &layer->routerGate, &layer->sharedGateProj,
                                                &layer->sharedUpProj, &layer->sharedGateScore,
                                                rt->normBuffer, 0, rt->routerScoresBuffer, 0,
                                                g_expertIOSharedActBuf, 0, rt->sharedOutBuffer, 0)) return NO;
        } else {
            if (!NMOEEncodeDequantMatVec(contextCmd, rt, [lp stringByAppendingString:@".mlp.gate"],
                                          rt->normBuffer, 0, rt->routerScoresBuffer, 0, 256u, kNMOEHiddenDim)) return NO;
            if (!NMOEEncodeGateUpQ4(contextCmd, rt,
                                    [lp stringByAppendingString:@".mlp.shared_expert.gate_proj"],
                                    [lp stringByAppendingString:@".mlp.shared_expert.up_proj"],
                                    rt->normBuffer, 0, g_expertIOSharedActBuf, 0)) return NO;
            if (!NMOEEncodeDequantMatVec(contextCmd, rt, [lp stringByAppendingString:@".mlp.shared_expert_gate"],
                                          rt->normBuffer, 0, rt->sharedOutBuffer, 0, 1u, kNMOEHiddenDim)) return NO;
        }
        if (!cpuRouter && !NMOEEncodeRouteTopK(contextCmd, rt, (size_t)rt->cfg.experts)) return NO;
        if (gpuRoutedExperts) {
            if (!NMOEEncodeDequantMatVecTensor(contextCmd, rt, &layer->sharedDownProj,
                                               g_expertIOSharedActBuf, 0, g_expertIOSharedDownBuf, 0,
                                               kNMOEHiddenDim, 512u)) return NO;
            void *expertOutPtrs[8] = {0};
            for (int i = 0; i < 8; i++) expertOutPtrs[i] = g_expertOutBuffers[i];
            if (!NMOEEncodeRoutedExpertQ4(contextCmd, rt, expertFile, (size_t)rt->cfg.experts,
                                          g_expertIOExpertActBuf, expertOutPtrs)) return NO;
            if (!NMOEEncodeWeightedExpertSumRouted(contextCmd, rt, outputBuf, g_expertIOSharedDownBuf,
                                                   outputBuf, expertOutPtrs, kNMOEHiddenDim,
                                                   (size_t)rt->cfg.experts)) return NO;
            BOOL nextInputNormReady = NO;
            int nextInputNormLayer = layerIndex + 1;
            if (nextInputNormLayer < (int)kNMOELayers) {
                if (!NMOEEncodeRMSNormTensor(contextCmd, rt, &rt->layers[nextInputNormLayer].inputNorm,
                                             outputBuf, 0, rt->normBuffer, 0,
                                             kNMOEHiddenDim, YES, 1e-6f)) return NO;
                nextInputNormReady = YES;
            }
            [contextCmd commit];
            g_deferredExperts.active = YES;
            g_deferredExperts.gpuCombined = YES;
            g_deferredExperts.nextInputNormReady = nextInputNormReady;
            g_deferredExperts.nextInputNormLayer = nextInputNormReady ? nextInputNormLayer : -1;
            g_deferredExperts.cmdExperts = contextCmd;
            for (int i = 0; i < 8; ++i) g_deferredExperts.expertBuffers[i] = nil;
            if (stats != NULL) stats->context += NMOENowSeconds() - t0;
            return YES;
        }
        [contextCmd commit];
        double syncStart = NMOENowSeconds();
        [contextCmd waitUntilCompleted];
        if (stats != NULL) stats->contextSync += NMOENowSeconds() - syncStart;
    }
    double deferredStart = NMOENowSeconds();
    NMOEFinalizeDeferredExperts(rt);
    if (stats != NULL) stats->deferredWait += NMOENowSeconds() - deferredStart;
    if (stats != NULL) stats->context += NMOENowSeconds() - t0;
    t0 = NMOENowSeconds();

    float selectedValues[kNMOEMaxExperts] = {0};
    size_t selectedIndices[kNMOEMaxExperts] = {0};
    size_t selectedCount = cpuRouter
        ? NMOESelectRouteTopKCPU(rt, (size_t)rt->cfg.experts, selectedIndices, selectedValues)
        : NMOEReadRouteTopKMetal(rt, (size_t)rt->cfg.experts, selectedIndices, selectedValues);
    if (selectedCount == 0) return NO;
    if (stats != NULL) stats->route += NMOENowSeconds() - t0;

    float sharedGateScore = NMOEFloatBuffer(rt->sharedOutBuffer)[0];

    NMOEExpertOffsets offsets = *NMOEExpertOffsetsForBits(rt->quantBits);
    int valid[8] = {0};
    __strong id<MTLBuffer> expertBuffers[8] = {nil};
    if (!NMOEPrepareExpertBuffers(rt, expertFile, selectedIndices, selectedCount, expertBuffers, valid, stats)) return NO;
    memcpy(((__bridge id<MTLBuffer>)g_expertIOHMidBuf).contents, NMOEFloatBuffer(outputBuf), kNMOEHiddenDim * sizeof(float));
    memcpy(((__bridge id<MTLBuffer>)g_expertIOInputBuf).contents, NMOEFloatBuffer(rt->normBuffer), kNMOEHiddenDim * sizeof(float));

    {
        id<MTLCommandBuffer> cmd = [queue commandBuffer];
        if (cmd == nil) return NO;
        BOOL useFusedDownCombineQ4 = (rt->quantBits == 4 && g_expertIOExpertActBuf != NULL &&
                                      NMOEUseFusedDownCombineQ4());
        if (!useFusedDownCombineQ4) {
            if (!NMOEEncodeDequantMatVecTensor(cmd, rt, &layer->sharedDownProj,
                                               g_expertIOSharedActBuf, 0, g_expertIOSharedDownBuf, 0,
                                               kNMOEHiddenDim, 512u)) return NO;
        }
        id<MTLComputePipelineState> expertPipe = (__bridge id<MTLComputePipelineState>)nmoe_backend_pipeline_state(rt->backend,
            rt->quantBits == 2 ? NMOE_BACKEND_KERNEL_DEQUANT_MATVEC_Q2 : NMOE_BACKEND_KERNEL_DEQUANT_MATVEC_Q4);
        id<MTLComputePipelineState> swigluPipe = (__bridge id<MTLComputePipelineState>)nmoe_backend_pipeline_state(rt->backend, NMOE_BACKEND_KERNEL_MOE_EXPERT_GATE_UP);
        id<MTLComputePipelineState> gateUpQ4Pipe = (__bridge id<MTLComputePipelineState>)nmoe_backend_pipeline_state(rt->backend, NMOE_BACKEND_KERNEL_EXPERT_GATE_UP_Q4);
        id<MTLComputePipelineState> gateUpQ4BatchedPipe = (__bridge id<MTLComputePipelineState>)nmoe_backend_pipeline_state(rt->backend, NMOE_BACKEND_KERNEL_EXPERT_GATE_UP_Q4_BATCHED);
        id<MTLComputePipelineState> downQ4BatchedPipe = (__bridge id<MTLComputePipelineState>)nmoe_backend_pipeline_state(rt->backend, NMOE_BACKEND_KERNEL_EXPERT_DOWN_Q4_BATCHED);
        id<MTLBuffer> expertInputMTL = (__bridge id<MTLBuffer>)g_expertIOInputBuf;
        id<MTLBuffer> expertGateMTL = NMOEBridgeBuffer(rt->expertGateBuffer);
        id<MTLBuffer> expertUpMTL = NMOEBridgeBuffer(rt->expertUpBuffer);
        id<MTLBuffer> expertActMTL = NMOEBridgeBuffer(rt->expertActBuffer);
        if (expertPipe == nil || swigluPipe == nil || expertInputMTL == nil ||
            expertGateMTL == nil || expertUpMTL == nil || expertActMTL == nil) return NO;
        BOOL usedBatchedQ4 = NO;
        BOOL usedFusedDownCombineQ4 = NO;
        if (rt->quantBits == 4 && g_expertIOExpertActBuf != NULL &&
            gateUpQ4BatchedPipe != nil && downQ4BatchedPipe != nil) {
            id<MTLBuffer> expertActBatchMTL = (__bridge id<MTLBuffer>)g_expertIOExpertActBuf;
            if (!NMOEEncodeExpertGateUpQ4Batched(cmd, rt, expertBuffers, selectedCount,
                                                  expertInputMTL, expertActBatchMTL, &offsets)) return NO;
            if (useFusedDownCombineQ4) {
                if (!NMOEEncodeExpertDownCombineQ4Tensor(cmd, rt, expertBuffers, selectedCount,
                                                         expertActBatchMTL, g_expertIOHMidBuf,
                                                         g_expertIOSharedActBuf, g_expertIOSharedDownBuf, outputBuf,
                                                         selectedValues, sharedGateScore,
                                                         &layer->sharedDownProj, &offsets)) return NO;
                usedFusedDownCombineQ4 = YES;
            } else {
                void *expertOutPtrs[8] = {0};
                for (int i = 0; i < 8; i++) expertOutPtrs[i] = g_expertOutBuffers[i];
                if (!NMOEEncodeExpertDownQ4Batched(cmd, rt, expertBuffers, expertOutPtrs,
                                                   selectedCount, expertActBatchMTL, &offsets)) return NO;
            }
            usedBatchedQ4 = YES;
        }
        for (size_t idx = 0; !usedBatchedQ4 && idx < selectedCount; ++idx) {
            if (!valid[idx]) continue;
            id<MTLBuffer> ebuf = expertBuffers[idx];
            if (ebuf == nil) return NO;
            id<MTLBuffer> expertOutMTL = (__bridge id<MTLBuffer>)g_expertOutBuffers[idx];
            if (expertOutMTL == nil) return NO;
            if (rt->quantBits == 4 && gateUpQ4Pipe != nil) {
                id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
                if (enc == nil) return NO;
                if (!NMOEEncodeExpertGateUpQ4OnEncoder(enc, gateUpQ4Pipe, ebuf,
                                                       &offsets, expertInputMTL, expertActMTL)) return NO;
                if (!NMOEEncodeDequantMatVecFromBufferOnEncoder(enc, expertPipe, ebuf,
                                                                 offsets.down_weight, offsets.down_scales, offsets.down_biases,
                                                                 expertActMTL, 0, expertOutMTL, 0,
                                                                 kNMOEHiddenDim, 512u, rt->quantBits)) return NO;
                [enc endEncoding];
                continue;
            }
            {
                id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
                if (enc == nil) return NO;
                if (!NMOEEncodeDequantMatVecFromBufferOnEncoder(enc, expertPipe, ebuf,
                                                                 offsets.gate_weight, offsets.gate_scales, offsets.gate_biases,
                                                                 expertInputMTL, 0, expertGateMTL, 0,
                                                                 512u, kNMOEHiddenDim, rt->quantBits)) return NO;
                if (!NMOEEncodeDequantMatVecFromBufferOnEncoder(enc, expertPipe, ebuf,
                                                                 offsets.up_weight, offsets.up_scales, offsets.up_biases,
                                                                 expertInputMTL, 0, expertUpMTL, 0,
                                                                 512u, kNMOEHiddenDim, rt->quantBits)) return NO;
                [enc endEncoding];
            }
            {
                id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
                if (enc == nil) return NO;
                if (!NMOEEncodeMoeExpertGateUpOnEncoder(enc, swigluPipe,
                                                        expertGateMTL, 0, expertUpMTL, 0, expertActMTL, 0, 512u)) return NO;
                if (!NMOEEncodeDequantMatVecFromBufferOnEncoder(enc, expertPipe, ebuf,
                                                                 offsets.down_weight, offsets.down_scales, offsets.down_biases,
                                                                 expertActMTL, 0, expertOutMTL, 0,
                                                                 kNMOEHiddenDim, 512u, rt->quantBits)) return NO;
                [enc endEncoding];
            }
        }
        if (!usedFusedDownCombineQ4) {
            void *expertOutPtrs[8] = {0};
            for (int i = 0; i < 8; i++) expertOutPtrs[i] = g_expertOutBuffers[i];
            if (!NMOEEncodeWeightedExpertSum(cmd, rt, g_expertIOHMidBuf, g_expertIOSharedDownBuf,
                                              outputBuf, expertOutPtrs, kNMOEHiddenDim, selectedCount,
                                              selectedValues, sharedGateScore)) return NO;
        }
        BOOL nextInputNormReady = NO;
        int nextInputNormLayer = layerIndex + 1;
        if (nextInputNormLayer < (int)kNMOELayers) {
            if (!NMOEEncodeRMSNormTensor(cmd, rt, &rt->layers[nextInputNormLayer].inputNorm,
                                         outputBuf, 0, rt->normBuffer, 0,
                                         kNMOEHiddenDim, YES, 1e-6f)) return NO;
            nextInputNormReady = YES;
        }

        [cmd commit];
        g_deferredExperts.active = YES;
        g_deferredExperts.gpuCombined = YES;
        g_deferredExperts.nextInputNormReady = nextInputNormReady;
        g_deferredExperts.nextInputNormLayer = nextInputNormReady ? nextInputNormLayer : -1;
        g_deferredExperts.cmdExperts = cmd;
        for (int i = 0; i < 8; ++i) {
            g_deferredExperts.expertBuffers[i] = i < (int)selectedCount ? expertBuffers[i] : nil;
        }
        for (int i = 0; i < (int)selectedCount; i++) {
            g_deferredExperts.expertWeights[i] = selectedValues[i];
            g_deferredExperts.valid[i] = valid[i];
        }
    }
    if (stats != NULL) stats->expert += NMOENowSeconds() - t0;
    return YES;
}

static void NMOEResetState(nmoe_runtime *rt) {
    if (rt == NULL || rt->backend == NULL) return;
    NMOECancelDeferredExperts();
    nmoe_backend_reset_linear_state(rt->backend);
    rt->sequencePosition = 0;
    rt->inThink = NO;
    rt->thinkCount = 0;
}

static void NMOETraceEscapedString(FILE *stream, const char *text) {
    if (stream == NULL) return;
    fputc('"', stream);
    if (text != NULL) {
        for (const unsigned char *p = (const unsigned char *)text; *p != '\0'; ++p) {
            switch (*p) {
                case '\n': fputs("\\n", stream); break;
                case '\r': fputs("\\r", stream); break;
                case '\t': fputs("\\t", stream); break;
                case '"': fputs("\\\"", stream); break;
                case '\\': fputs("\\\\", stream); break;
                default:
                    if (*p < 0x20) fprintf(stream, "\\x%02x", *p);
                    else fputc((int)*p, stream);
                    break;
            }
        }
    }
    fputc('"', stream);
}

static void NMOETraceToken(nmoe_runtime *rt, FILE *stream, uint32_t token) {
    if (rt == NULL || stream == NULL) return;
    fprintf(stream, "%u:", token);
    NMOETraceEscapedString(stream, nmoe_vocab_decode_token(rt->vocab, (int)token));
}

static void NMOETracePromptTokens(nmoe_runtime *rt, const uint32_t *tokens, int tokenCount) {
    if (rt == NULL || !rt->traceTokens || tokens == NULL || tokenCount <= 0) return;
    fprintf(stderr, "trace: prompt_tokens=%d [", tokenCount);
    for (int i = 0; i < tokenCount; ++i) {
        if (i > 0) fputs(", ", stderr);
        NMOETraceToken(rt, stderr, tokens[i]);
    }
    fputs("]\n", stderr);
}

static void NMOETraceGenerationToken(nmoe_runtime *rt, uint32_t token) {
    if (rt == NULL || !rt->traceTokens) return;

    const float *logits = NMOEFloatBuffer(rt->logitsBuffer);
    const size_t topN = 5;
    float topValues[5];
    uint32_t topIds[5];
    for (size_t i = 0; i < topN; ++i) {
        topValues[i] = -INFINITY;
        topIds[i] = 0;
    }

    size_t nonFinite = 0;
    if (logits != NULL) {
        for (uint32_t i = 0; i < 248320u; ++i) {
            float value = logits[i];
            if (!isfinite(value)) {
                nonFinite += 1;
                continue;
            }
            for (size_t slot = 0; slot < topN; ++slot) {
                if (value > topValues[slot] ||
                    (value == topValues[slot] && i < topIds[slot])) {
                    for (size_t move = topN - 1; move > slot; --move) {
                        topValues[move] = topValues[move - 1];
                        topIds[move] = topIds[move - 1];
                    }
                    topValues[slot] = value;
                    topIds[slot] = i;
                    break;
                }
            }
        }
    }

    fprintf(stderr, "trace: gen_pos=%zu token=", rt->sequencePosition);
    NMOETraceToken(rt, stderr, token);
    if (logits != NULL) {
        fprintf(stderr, " cpu_top1=%u", topIds[0]);
        if (topIds[0] != token) fprintf(stderr, " argmax_mismatch=1");
        fputs(" top5=[", stderr);
        for (size_t i = 0; i < topN; ++i) {
            if (i > 0) fputs(", ", stderr);
            NMOETraceToken(rt, stderr, topIds[i]);
            fprintf(stderr, ":%.4f", topValues[i]);
        }
        fputc(']', stderr);
        if (nonFinite > 0) fprintf(stderr, " non_finite=%zu", nonFinite);
    }
    fputc('\n', stderr);
}

static uint32_t NMOEProcessToken(nmoe_runtime *rt, uint32_t token, BOOL computeLogits, NMOEPerfStats *stats) {
    if (rt == NULL || rt->tokenizer == NULL || rt->vocab == NULL) return 0;
    float *hiddenA = NMOEFloatBuffer(rt->hiddenBuffers[0]);
    float *hiddenB = NMOEFloatBuffer(rt->hiddenBuffers[1]);
    float *current = hiddenA;
    float *next = hiddenB;
    const uint16_t *embedScales = (const uint16_t *)NMOETensorPointer(rt->weights, NMOEModelTensorName(@"model.embed_tokens.scales"), @"BF16", NULL);
    const uint16_t *embedBiases = (const uint16_t *)NMOETensorPointer(rt->weights, NMOEModelTensorName(@"model.embed_tokens.biases"), @"BF16", NULL);
    const uint32_t *embedWeight = (const uint32_t *)NMOETensorPointer(rt->weights, NMOEModelTensorName(@"model.embed_tokens.weight"), @"U32", NULL);
    if (embedWeight == NULL || embedScales == NULL || embedBiases == NULL) return 0;
    size_t embedPackedCols = 256u;
    size_t embedGroups = kNMOEHiddenDim / 64u;
    size_t rowOffset = (size_t)token * embedPackedCols;
    const uint32_t *rowWeight = embedWeight + rowOffset;
    const uint16_t *rowScales = embedScales + ((size_t)token * embedGroups);
    const uint16_t *rowBiases = embedBiases + ((size_t)token * embedGroups);
    (void)rowWeight; (void)rowScales; (void)rowBiases;
    if (!NMOERunDequantRowMetal(rt, @"model.embed_tokens", (size_t)token, rt->hiddenBuffers[0], 0, kNMOEHiddenDim, 4)) {
    nmoe_cpu_dequant_row(rowWeight, rowScales, rowBiases, current, kNMOEHiddenDim, 4);
    }
    for (size_t layerIndex = 0; layerIndex < kNMOELayers; ++layerIndex) {
        double layerStart = NMOENowSeconds();
        NMOEPerfStats layerStats = {0};
        BOOL stateOnlyFinalPrefill = (!computeLogits &&
                                      layerIndex + 1u == kNMOELayers &&
                                      rt->layers[layerIndex].isFull);
        if (stateOnlyFinalPrefill) {
            if (!NMOEFullAttentionStateOnlyMetal(rt, (int)layerIndex, current, rt->sequencePosition, &layerStats)) {
                fprintf(stderr, "nmoe runtime error: Metal full attention state-only step failed at layer %zu\n", layerIndex);
                abort();
            }
        } else if (rt->layers[layerIndex].isFull) {
            NMOEFullAttentionStep(rt, (int)layerIndex, current, next, rt->sequencePosition, &layerStats);
        } else {
            NMOELinearAttentionStep(rt, (int)layerIndex, current, next, rt->sequencePosition, &layerStats);
        }
        if (stats != NULL) {
            stats->context += layerStats.context;
            stats->route += layerStats.route;
            stats->expertFetch += layerStats.expertFetch;
            stats->expert += layerStats.expert;
            stats->contextSync += layerStats.contextSync;
            stats->deferredWait += layerStats.deferredWait;
            double layerElapsed = NMOENowSeconds() - layerStart;
            stats->total += layerElapsed;
            stats->layerCount += 1;
            if (rt->layers[layerIndex].isFull) {
                stats->fullContext += layerStats.context;
                stats->fullSync += layerStats.contextSync;
                stats->fullTotal += layerElapsed;
                stats->fullLayerCount += 1;
            } else {
                stats->linearContext += layerStats.context;
                stats->linearSync += layerStats.contextSync;
                stats->linearTotal += layerElapsed;
                stats->linearLayerCount += 1;
            }
        }
        if (stateOnlyFinalPrefill) break;
        float *tmp = current; current = next; next = tmp;
    }
    double finalWaitStart = NMOENowSeconds();
    NMOEFinalizeDeferredExperts(rt);
    if (stats != NULL) {
        double finalWait = NMOENowSeconds() - finalWaitStart;
        stats->expert += finalWait;
        stats->total += finalWait;
    }
    if (computeLogits) {
        (void)NMOERunRMSNormMetal(rt, NMOEModelTensorName(@"model.norm.weight"), NMOEHiddenBufferForPointer(rt, current), 0, rt->normBuffer, 0, kNMOEHiddenDim, YES);
        uint32_t nextToken = 0;
        BOOL gotToken = NO;
        if (!rt->traceTokens) {
            gotToken = NMOERunLmHeadArgmaxQ4Metal(rt, rt->normBuffer, 0, &nextToken);
        }
        if (!gotToken) {
            (void)NMOERunDequantMatVecMetal(rt, @"lm_head", rt->normBuffer, 0, rt->logitsBuffer, 0, 248320u, kNMOEHiddenDim, 4);
            if (!NMOERunArgmaxTop1Metal(rt, rt->logitsBuffer, 0, 248320u, &nextToken)) { fprintf(stderr, "argmax failed\n"); abort(); }
        }
        if (rt->inThink) { rt->thinkCount += 1; if (rt->cfg.think_budget > 0 && (size_t)rt->cfg.think_budget <= rt->thinkCount) nextToken = kNMOEThinkEnd; }
        NMOETraceGenerationToken(rt, nextToken);
        if (nextToken == kNMOEThinkStart) { rt->inThink = YES; rt->thinkCount = 0; }
        else if (nextToken == kNMOEThinkEnd) { rt->inThink = NO; }
        rt->sequencePosition += 1;
        return nextToken;
    }
    rt->sequencePosition += 1;
    return UINT32_MAX;
}

static uint32_t NMOEProcessPromptAndDecode(nmoe_runtime *rt, NSString *prompt, FILE *output, BOOL quiet, BOOL timing, int maxTokens, NMOEPerfStats *perfOut) {
    if (rt == NULL || prompt.length == 0) return 0;
    uint32_t tokenBuffer[4096];
    int tokenCount = nmoe_tokenizer_encode(rt->tokenizer, prompt.UTF8String, tokenBuffer, 4096);
    if (tokenCount <= 0) return 0;
    NMOETracePromptTokens(rt, tokenBuffer, tokenCount);
    double startTime = NMOENowSeconds();
    rt->sequencePosition = 0; rt->inThink = NO; rt->thinkCount = 0;
    NMOEPerfStats perf = {0};
    for (int i = 0; i < tokenCount - 1; ++i) (void)NMOEProcessToken(rt, tokenBuffer[i], NO, &perf);
    if (maxTokens <= 0) { (void)NMOEProcessToken(rt, tokenBuffer[tokenCount-1], NO, &perf); if (perfOut) *perfOut = perf; return UINT32_MAX; }
    uint32_t nextToken = NMOEProcessToken(rt, tokenBuffer[tokenCount-1], YES, &perf);

    if (nextToken == UINT32_MAX) nextToken = 0;
    if (nextToken == kNMOEEOS1 || nextToken == kNMOEEOS2) {
        if (perfOut) *perfOut = perf;
        return nextToken;
    }
    if (!quiet && output) { NMOEWriteDecodedToken(rt, nextToken); fflush(output); }
    int generated = 1;
    double decodeStartTime = NMOENowSeconds();
    while (generated < maxTokens) {
        if (nextToken == kNMOEEOS1 || nextToken == kNMOEEOS2) break;
        uint32_t predicted = NMOEProcessToken(rt, nextToken, YES, &perf);
        if (predicted == UINT32_MAX) break;
        if (predicted == kNMOEEOS1 || predicted == kNMOEEOS2) {
            nextToken = predicted;
            break;
        }
        nextToken = predicted;
        generated += 1;
        if (!quiet && output) { NMOEWriteDecodedToken(rt, nextToken); fflush(output); }
    }
    double elapsed = NMOENowSeconds() - startTime;
    double decodeElapsed = NMOENowSeconds() - decodeStartTime;
    double decodeTokS = (generated > 1 && decodeElapsed > 0.0) ? (double)(generated - 1) / decodeElapsed : 0.0;
    if (timing && output) fprintf(output, "timing: mode=ask quant=%s experts=%d tokens=%d tok_s=%.3f decode_tok_s=%.3f\n",
            rt->quantBits == 2 ? "q2" : "q4", rt->cfg.experts, generated,
            generated / (elapsed > 0 ? elapsed : 0.001), decodeTokS);
    if (timing && perf.layerCount > 0) {
        double n = (double)perf.layerCount;
        double totalLayerMs = (perf.total * 1000.0) / n;
        double decodeMsPerToken = totalLayerMs * (double)kNMOELayers;
        double decodeTokS = decodeMsPerToken > 0.0 ? 1000.0 / decodeMsPerToken : 0.0;
        fprintf(stderr,
                "timing: per_layer_avg_ms layers=%zu context=%.3f route=%.3f expert_fetch=%.3f expert=%.3f total_layer=%.3f\n",
                perf.layerCount,
                (perf.context * 1000.0) / n,
                (perf.route * 1000.0) / n,
                (perf.expertFetch * 1000.0) / n,
                (perf.expert * 1000.0) / n,
                totalLayerMs);
        fprintf(stderr,
                "timing: decode_est ms_per_token=%.3f tok_s=%.3f\n",
                decodeMsPerToken,
                decodeTokS);
        fprintf(stderr,
                "timing: context_detail_avg_ms sync=%.3f deferred_wait=%.3f encode_cpu=%.3f\n",
                (perf.contextSync * 1000.0) / n,
                (perf.deferredWait * 1000.0) / n,
                ((perf.context - perf.contextSync - perf.deferredWait) * 1000.0) / n);
        if (perf.fullLayerCount > 0 || perf.linearLayerCount > 0) {
            double fn = perf.fullLayerCount > 0 ? (double)perf.fullLayerCount : 1.0;
            double ln = perf.linearLayerCount > 0 ? (double)perf.linearLayerCount : 1.0;
            fprintf(stderr,
                    "timing: context_by_type_avg_ms full_layers=%zu full_context=%.3f full_sync=%.3f full_total=%.3f linear_layers=%zu linear_context=%.3f linear_sync=%.3f linear_total=%.3f\n",
                    perf.fullLayerCount,
                    (perf.fullContext * 1000.0) / fn,
                    (perf.fullSync * 1000.0) / fn,
                    (perf.fullTotal * 1000.0) / fn,
                    perf.linearLayerCount,
                    (perf.linearContext * 1000.0) / ln,
                    (perf.linearSync * 1000.0) / ln,
                    (perf.linearTotal * 1000.0) / ln);
        }
    }
    if (perfOut) *perfOut = perf;
    return nextToken;
}

static int NMOERunAskLike(nmoe_runtime *rt, BOOL benchMode, FILE *output) {
    NSString *prompt = rt->cfg.prompt ? NMOEStringFromC(rt->cfg.prompt) : @"";
    if (prompt.length == 0) prompt = @"";
    NSString *runtimePrompt = [NMOEChatSystemPrompt() stringByAppendingString:NMOEChatUserPrompt(prompt)];
    NMOEResetState(rt);
    NMOEPerfStats perf = {0};
    uint32_t token = NMOEProcessPromptAndDecode(rt, runtimePrompt, output, rt->quiet, rt->cfg.timing, rt->cfg.max_tokens, &perf);
    if (benchMode && output) fprintf(output, "bench: quant=%s tok_s=%.3f\n", rt->quantBits == 2 ? "q2" : "q4", 0.0);
    if (!rt->quiet && output) {
        fputc('\n', output);
        fflush(output);
    }
    (void)token;
    return 0;
}

nmoe_runtime *nmoe_runtime_create(const nmoe_app_config *cfg) {
    nmoe_runtime *rt = calloc(1, sizeof(*rt));
    if (rt == NULL) return NULL;
    if (cfg != NULL) {
        rt->cfg = *cfg;
    } else {
        nmoe_app_config_init(&rt->cfg);
    }
    rt->modelPath = strdup(rt->cfg.model_path != NULL ? rt->cfg.model_path : "qwen36_35b");
    rt->quiet = rt->cfg.quiet ? YES : NO;
    rt->cpuLinear = rt->cfg.cpu_linear ? YES : NO;
    rt->traceTokens = rt->cfg.trace_tokens ? YES : NO;
    if (rt->modelPath == NULL) {
        free(rt);
        return NULL;
    }
    NSError *error = nil;
    if (!NMOERuntimeInitialise(rt, &error)) {
        fprintf(stderr, "nmoe runtime init failed: %s\n", error.localizedDescription.UTF8String ?: "unknown error");
        NMOERuntimeRelease(rt);
        free(rt);
        return NULL;
    }
    return rt;
}

void nmoe_runtime_destroy(nmoe_runtime *rt) {
    if (rt == NULL) return;
    NMOERuntimeRelease(rt);
    free(rt);
}

int nmoe_runtime_run(nmoe_runtime *rt) {
    if (rt == NULL) return 1;
    if (rt->cfg.mode == NMOE_RUN_ASK || rt->cfg.mode == NMOE_RUN_BENCH) return NMOERunAskLike(rt, rt->cfg.mode == NMOE_RUN_BENCH, stdout);
    if (rt->cfg.mode == NMOE_RUN_CHAT) return NMOERunChat(rt);
    return 0;
}

static int NMOERunChat(nmoe_runtime *rt) {
    NSString *systemPrompt = NMOEChatSystemPrompt();
    NMOEResetState(rt);
    (void)NMOEProcessPromptAndDecode(rt, systemPrompt, stdout, YES, NO, 0, NULL);
    while (1) {
        fprintf(stdout, "\n> "); fflush(stdout);
        char line[4096];
        if (fgets(line, sizeof(line), stdin) == NULL) break;
        NSString *userPrompt = NMOEChatUserPrompt(NMOEStringFromC(line));
        NMOEResetState(rt);
        (void)NMOEProcessPromptAndDecode(rt, userPrompt, stdout, NO, rt->cfg.timing, rt->cfg.max_tokens, NULL);
        fprintf(stdout, "\n"); fflush(stdout);
    }
    return 0;
}

static void NMOEWriteDecodedToken(nmoe_runtime *rt, uint32_t token) {
    if (rt == NULL || rt->vocab == NULL) return;
    const char *decoded = nmoe_vocab_decode_token(rt->vocab, (int)token);
    if (decoded != NULL) fputs(decoded, stdout);
}
