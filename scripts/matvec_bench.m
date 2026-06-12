// Microbenchmark for nmoe_dequant_matvec_q4: measures achievable GPU bandwidth
// in isolation, plus per-command-buffer sync overhead.
//
// Build:  clang -O2 -fobjc-arc -framework Foundation -framework Metal \
//             scripts/matvec_bench.m -o /tmp/matvec_bench
// Run from repo root: /tmp/matvec_bench

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

typedef struct {
    uint32_t out_rows;
    uint32_t in_dim;
    uint32_t packed_cols;
    uint32_t group_size;
    uint32_t rows_per_tg;
    uint32_t reserved0;
    uint32_t reserved1;
    uint32_t reserved2;
} Args;

static double now(void) { return [NSDate timeIntervalSinceReferenceDate]; }

int main(void) {
    @autoreleasepool {
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        id<MTLCommandQueue> queue = [dev newCommandQueue];
        NSError *err = nil;
        NSString *src = [NSString stringWithContentsOfFile:@"metal/kernels.metal"
                                                  encoding:NSUTF8StringEncoding error:&err];
        if (!src) { fprintf(stderr, "no kernels.metal (run from repo root)\n"); return 1; }
        id<MTLLibrary> lib = [dev newLibraryWithSource:src options:nil error:&err];
        if (!lib) { fprintf(stderr, "compile: %s\n", err.description.UTF8String); return 1; }
        id<MTLComputePipelineState> pso =
            [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"nmoe_dequant_matvec_q4"]
                                               error:&err];
        if (!pso) { fprintf(stderr, "pso: %s\n", err.description.UTF8String); return 1; }

        const uint32_t OUT_ROWS = 8192, IN_DIM = 2048, GROUP = 64;
        const uint32_t PACKED_COLS = IN_DIM / 8;            // uint32 words per row
        const uint32_t SCALE_GROUPS = IN_DIM / GROUP;
        size_t wBytes = (size_t)OUT_ROWS * PACKED_COLS * 4;  // 8 MB
        size_t sBytes = (size_t)OUT_ROWS * SCALE_GROUPS * 2; // 512 KB
        size_t totalBytes = wBytes + 2 * sBytes + IN_DIM * 4;
        const int SLOTS = 64;                                // 64 x 8MB = 512MB pool, defeats SLC

        id<MTLBuffer> w  = [dev newBufferWithLength:wBytes * SLOTS options:MTLResourceStorageModeShared];
        id<MTLBuffer> sc = [dev newBufferWithLength:sBytes * SLOTS options:MTLResourceStorageModeShared];
        id<MTLBuffer> bi = [dev newBufferWithLength:sBytes * SLOTS options:MTLResourceStorageModeShared];
        id<MTLBuffer> in = [dev newBufferWithLength:IN_DIM * 4 options:MTLResourceStorageModeShared];
        id<MTLBuffer> out = [dev newBufferWithLength:OUT_ROWS * 4 options:MTLResourceStorageModeShared];
        // Non-trivial contents so nothing is optimized to zero pages.
        memset(w.contents, 0x5A, wBytes * SLOTS);
        uint16_t *scp = sc.contents, *bip = bi.contents;
        for (size_t i = 0; i < sBytes * SLOTS / 2; i++) { scp[i] = 0x3F80 >> 0; bip[i] = 0x3F00; } // bf16-ish
        float *inp = in.contents;
        for (uint32_t i = 0; i < IN_DIM; i++) inp[i] = 0.001f * (float)(i % 97);

        uint32_t rowsPerTGOptions[] = {8, 16, 24, 32};
        for (int r = 0; r < 4; r++) {
            uint32_t rowsPerTG = rowsPerTGOptions[r];
            Args args = {OUT_ROWS, IN_DIM, PACKED_COLS, GROUP, rowsPerTG, 0, 0, 0};
            MTLSize tgs = MTLSizeMake((OUT_ROWS + rowsPerTG - 1) / rowsPerTG, 1, 1);
            MTLSize tpt = MTLSizeMake(rowsPerTG * 32, 1, 1);

            // --- A: N dispatches in ONE command buffer (pure kernel throughput) ---
            const int N = 50;
            id<MTLCommandBuffer> cmd = [queue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            [enc setComputePipelineState:pso];
            [enc setBuffer:w offset:0 atIndex:0];
            [enc setBuffer:sc offset:0 atIndex:1];
            [enc setBuffer:bi offset:0 atIndex:2];
            [enc setBuffer:in offset:0 atIndex:3];
            [enc setBuffer:out offset:0 atIndex:4];
            [enc setBytes:&args length:sizeof(args) atIndex:5];
            for (int i = 0; i < N; i++) {
                int slot = i % SLOTS;
                [enc setBuffer:w offset:wBytes * slot atIndex:0];
                [enc setBuffer:sc offset:sBytes * slot atIndex:1];
                [enc setBuffer:bi offset:sBytes * slot atIndex:2];
                [enc dispatchThreadgroups:tgs threadsPerThreadgroup:tpt];
            }
            [enc endEncoding];
            [cmd commit];
            [cmd waitUntilCompleted];
            double gpuMs = (cmd.GPUEndTime - cmd.GPUStartTime) * 1000.0;
            double perDispatchMs = gpuMs / N;
            double gbps = ((double)totalBytes * N) / (gpuMs / 1000.0) / 1e9;
            printf("rows_per_tg=%2u  batched x%d : GPU %7.3f ms total, %6.3f ms/dispatch, %6.1f GB/s\n",
                   rowsPerTG, N, gpuMs, perDispatchMs, gbps);

            // --- B: one dispatch per command buffer with waitUntilCompleted (the runtime's pattern) ---
            const int M = 30;
            double wallStart = now();
            double gpuAccum = 0;
            for (int i = 0; i < M; i++) {
                id<MTLCommandBuffer> c = [queue commandBuffer];
                id<MTLComputeCommandEncoder> e = [c computeCommandEncoder];
                [e setComputePipelineState:pso];
                int slot = i % SLOTS;
                [e setBuffer:w offset:wBytes * slot atIndex:0];
                [e setBuffer:sc offset:sBytes * slot atIndex:1];
                [e setBuffer:bi offset:sBytes * slot atIndex:2];
                [e setBuffer:in offset:0 atIndex:3];
                [e setBuffer:out offset:0 atIndex:4];
                [e setBytes:&args length:sizeof(args) atIndex:5];
                [e dispatchThreadgroups:tgs threadsPerThreadgroup:tpt];
                [e endEncoding];
                [c commit];
                [c waitUntilCompleted];
                gpuAccum += (c.GPUEndTime - c.GPUStartTime) * 1000.0;
            }
            double wallMs = (now() - wallStart) * 1000.0 / M;
            printf("rows_per_tg=%2u  sync 1-by-1: wall %6.3f ms/dispatch (GPU busy %6.3f ms, overhead %6.3f ms)\n\n",
                   rowsPerTG, wallMs, gpuAccum / M, wallMs - gpuAccum / M);
        }

        // --- C: simulate one linear-attention layer: 15 dependent dispatches, mixed sizes,
        //        single command buffer (like the runtime's contextCmd) ---
        {
            Args bigArgs   = {8192, 2048, PACKED_COLS, GROUP, 24, 0, 0, 0}; // QKV
            Args midArgs   = {4096, 2048, PACKED_COLS, GROUP, 24, 0, 0, 0}; // Z
            Args outArgs   = {2048, 2048, PACKED_COLS, GROUP, 24, 0, 0, 0}; // out proj (approx)
            Args tinyArgs  = {32,   2048, PACKED_COLS, GROUP, 1,  0, 0, 0}; // beta/alpha
            const int LAYERS = 40;
            id<MTLCommandBuffer> cmd = [queue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            [enc setComputePipelineState:pso];
            [enc setBuffer:w offset:0 atIndex:0];
            [enc setBuffer:sc offset:0 atIndex:1];
            [enc setBuffer:bi offset:0 atIndex:2];
            [enc setBuffer:in offset:0 atIndex:3];
            [enc setBuffer:out offset:0 atIndex:4];
            for (int l = 0; l < LAYERS; l++) {
                int slot = l % SLOTS;
                [enc setBuffer:w offset:wBytes * slot atIndex:0];
                [enc setBuffer:sc offset:sBytes * slot atIndex:1];
                [enc setBuffer:bi offset:sBytes * slot atIndex:2];
                Args *seq[6] = {&bigArgs, &midArgs, &tinyArgs, &tinyArgs, &outArgs, &outArgs};
                for (int k = 0; k < 6; k++) {
                    Args a = *seq[k];
                    [enc setBytes:&a length:sizeof(a) atIndex:5];
                    MTLSize g = MTLSizeMake((a.out_rows + a.rows_per_tg - 1) / a.rows_per_tg, 1, 1);
                    MTLSize t = MTLSizeMake(a.rows_per_tg * 32, 1, 1);
                    [enc dispatchThreadgroups:g threadsPerThreadgroup:t];
                }
            }
            [enc endEncoding];
            [cmd commit];
            [cmd waitUntilCompleted];
            double gpuMs = (cmd.GPUEndTime - cmd.GPUStartTime) * 1000.0;
            // bytes actually touched per simulated layer (weights are reused but 8MB+4MB+2*2MB+tiny > cache)
            printf("simulated 40-layer projection chain (6 matvecs/layer, 1 cmdbuf): GPU %7.3f ms total, %6.3f ms/layer\n",
                   gpuMs, gpuMs / LAYERS);
        }
        return 0;
    }
}
