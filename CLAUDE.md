# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

`nmoe` is a single-binary Objective-C/Metal inference runtime for **Qwen3.6-35B-A3B** (40 layers, 256 routed experts per layer, top-k activation) targeting a **16GB Apple Silicon Mac mini (M4)**. Shared (non-expert) weights stay resident via mmap; routed expert weights live in quantized per-layer packs (q4 or q2) and only the selected top-k experts are read per token. No Python or server process in the inference loop.

This is a sibling of the larger `flash-moe` project (parent directory's CLAUDE.md describes that 397B/M3 Max system); the same principles apply (trust the OS page cache, deferred GPU expert compute, FMA dequant kernels) but the model, layer count, and hardware target here are different.

## Build and Run

```bash
make            # builds ./nmoe (no incremental build — one clang invocation over all sources)
make clean
```

There are no tests or linters. Verification is done by running inference and checking output quality + tok/s:

```bash
./nmoe ask "请用中文介绍本地大模型推理" --q4 --experts 8 --tokens 128 --timing
./nmoe chat --q4 --experts 8 --tokens 512
./nmoe bench "请介绍一下量子计算" --q2 --experts 6 --tokens 128 --timing --quiet
```

Key flags: `--model PATH` (default `qwen36_35b/`), `--q2`/`--q4`/`--quant auto|2|4`, `--experts N` (1..8), `--tokens N`, `--think N`, `--timing`, `--quiet`.

**Metal shaders are compiled at runtime, not build time.** `metal_backend.m` loads `metal/kernels.metal` source via `newLibraryWithSource:`, searching relative to the cwd and the executable's directory. Run `./nmoe` from the repo root (or keep the binary next to `metal/`). Editing `kernels.metal` does not require `make` unless host code also changed.

The model package is not in git. `qwen36_35b/` is a symlink in this working copy; a fresh checkout needs `scripts/convert_qwen36.py` run against the HF checkpoint (see README.md for the full download/convert procedure, including `--skip-weights --skip-tokenizer` for expert-only re-packs).

## Architecture

All public C APIs live in `include/nmoe/*.h`; implementation modules in `src/`:

- `main.m` → `app_config.m` (CLI parsing) → `runtime.m`
- `runtime.m` (~3800 lines) — the entire model: runtime struct, per-layer decode loop, all kernel encode helpers (`NMOEEncode*`), deferred expert pipeline, ask/chat/bench loops. Almost all changes land here.
- `backend/metal_backend.m` — Metal device/queue/library setup, pipeline-state cache, named shared/state buffer registry (`nmoe_backend_register_shared_buffer` etc.).
- `manifest.m` — parses `model_weights.json` and mmaps `model_weights.bin`; tensors looked up by name.
- `expert_io.m` — opens `packed_experts/` (or `packed_experts_2bit/`), reads selected expert blobs per layer via `pread`.
- `tokenizer.m` — BPE encode/decode from `tokenizer.bin` + `vocab.bin`.
- `math.m` — small CPU helpers (Accelerate).

### Layer structure

Every 4th layer (`(layerIndex + 1) % 4 == 0`, so 10 of 40) is full attention with KV cache; the rest are GatedDeltaNet linear attention with a recurrent state. See `layer->isFull` in `runtime.m`. Per layer: attention/projections on GPU → router top-k → read k expert packs → expert forward + weighted combine + residual on GPU, with expert compute deferred (submitted without waiting) so the GPU overlaps the CPU's next-layer prep.

### Adding or changing a Metal kernel

Three places must stay in sync:
1. The `kernel void nmoe_...` function in `metal/kernels.metal`
2. The `NMOE_BACKEND_KERNEL_*` enum in `include/nmoe/backend.h` (and bump `NMOE_BACKEND_KERNEL_COUNT`)
3. The `kNMOEKernelNames` table at the top of `src/backend/metal_backend.m` (order must match the enum)

Then encode it from `runtime.m` via `nmoe_backend_pipeline_state()` and the `NMOEEncodeKernel`/`NMOEEncodeKernelTG` helpers.

### Experiment toggles (env vars, read in runtime.m)

`NMOE_PIPELINE` (default on — SharedEvent expert pipeline; set 0 for the legacy per-layer waitUntilCompleted path), `NMOE_CPU_ROUTER`, `NMOE_GPU_ROUTED_EXPERTS`, `NMOE_COPY_EXPERTS`, `NMOE_FUSED_DOWN_COMBINE_Q4`, `NMOE_EXPERT_ROWS_PER_TG`, `NMOE_MATVEC_ROWS_PER_TG`.

### SharedEvent expert pipeline (default path)

Each layer is ONE command buffer: context kernels → `encodeSignalEvent(evRoute)` → `encodeWaitForEvent(evData)` → pre-encoded expert kernels + next-layer input norm. The CPU never calls `waitUntilCompleted` per layer; it waits on `evRoute` (mid-buffer), runs top-k, preads the selected experts into the fixed `g_expertIOBuffers`, writes combine weights into `g_expertParamsBuf` (16 floats: [0..7] expert weights, [8] shared-gate raw score), then signals `evData` to release the GPU. Expert kernels are encoded BEFORE routing is known — anything routing-dependent must flow through a GPU-visible buffer, never `setBytes`. Every committed buffer must have its `evData` value signaled even on error paths, or the GPU deadlocks.

## Performance Context

`优化.md` (Chinese) is the optimization log — read it before attempting performance work. Key findings as of June 2026 (measured with the `--timing` instrumentation: `gpu_busy`/`commit_lag`/`queue_wait` breakdown, and `scripts/matvec_bench.m`):

- The dequant-matvec kernel is NOT bandwidth-broken: in isolation it hits 67–70 GB/s on cold DRAM (~58% of M4 theoretical), 0.14ms per 8MB matvec. An earlier theory that "8192 row streams defeat DRAM bursting" was disproven by `scripts/matvec_bench.m`.
- The real pre-pipeline bottleneck was CPU↔GPU latency: per layer only ~1.9ms was GPU compute; ~3.3ms was commit/scheduling stalls. ~2ms of that was the Metal driver wiring mmap-backed expert buffers (`newBufferWithBytesNoCopy` over `packed_experts`) at command-buffer commit, which also blocks scheduling of the NEXT layer's buffer. Diagnostic: `NMOE_COPY_EXPERTS=1` collapses `commit_lag` to ~0 but pays a pread tax instead.
- The SharedEvent pipeline (see above) removed the per-layer sync: ~4.6 → ~5.8–6.4 tok/s (greedy output bit-identical to legacy). Gain is largest with cold/medium page cache; with a fully warm cache the legacy wiring is also cheap and the gap narrows.
- Remaining known costs: (a) expert pread tax ~1.4–1.8ms/layer in pipelined mode (19MB copied per layer even when page-cache-hot); (b) full-attention layers grow from ~6ms to ~10ms as the sequence lengthens — pre-existing in both modes, kernel-level issue, dominates long generations.
- Already tried and rejected: larger threadgroup rows (slower), smaller shared memory variant (slower), direct device-memory input reads (slower), KV-cache direct write (no change). Do NOT load whole-layer expert files as Metal buffers (`NMOE_GPU_ROUTED_EXPERTS=1`) on 16GB machines — residency churn causes severe memory pressure.
- Still-open directions: q2 shared weights, batched prefill, fusing QKV/Z/beta/alpha projections, fixing the full-attention seq-length growth.
