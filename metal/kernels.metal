#include <metal_stdlib>
using namespace metal;

constant float NMOE_ROPE_THETA = 10000000.0f;

struct NMOEDequantMatvecArgs {
    uint out_rows;
    uint in_dim;
    uint packed_cols;
    uint group_size;
    uint rows_per_tg;
    uint reserved0;
    uint reserved1;
    uint reserved2;
};

struct NMOERMSNormArgs {
    uint dim;
    uint add_one;
    float epsilon;
    float reserved;
};

struct NMOERopeArgs {
    uint rotary_dim;
    uint position;
    float theta;
    float reserved;
};

struct NMOESigmoidGateArgs {
    uint count;
    uint reserved0;
    uint reserved1;
    uint reserved2;
};

struct NMOEFullQKPrepArgs {
    uint head_dim;
    uint q_stride;
    uint full_heads;
    uint kv_heads;
    uint rotary_dim;
    uint position;
    uint add_one;
    float epsilon;
    float theta;
    uint reserved0;
    uint reserved1;
    uint reserved2;
};

struct NMOEDequantRowArgs {
    uint in_dim;
    uint packed_cols;
    uint group_size;
    uint reserved0;
};

struct NMOEAttentionArgs {
    uint seq_len;
    uint seq_stride;
    uint head_dim;
    uint q_stride;
    uint kv_stride;
    uint cache_stride;
    uint full_heads;
    uint full_kv_heads;
    uint position;
    uint reserved0;
    float inv_scale;
    float reserved1;
};

struct NMOELinearConv1DArgs {
    uint dim;
    uint state_stride;
    uint reserved0;
    uint reserved1;
};

struct NMOEGatedDeltaNetArgs {
    uint v_heads;
    uint kv_heads;
    uint value_dim;
    uint key_dim;
    float q_scale;
    float k_scale;
    float epsilon;
    float reserved0;
};

struct NMOEMoeCombineArgs {
    uint count;
    float weight;
    uint reserved0;
    uint reserved1;
};

struct NMOEArgmaxArgs {
    uint count;
    uint reserved0;
    uint reserved1;
    uint reserved2;
};

struct NMOELmHeadArgmaxArgs {
    uint vocab_size;
    uint in_dim;
    uint packed_cols;
    uint group_size;
    uint rows_per_tg;
    uint partial_count;
    uint reserved0;
    uint reserved1;
};

struct NMOERouteTopKArgs {
    uint count;
    uint k;
    uint reserved0;
    uint reserved1;
};

struct NMOEExpertBatchedArgs {
    uint expert_count;
    uint out_rows;
    uint in_dim;
    uint packed_cols;
    uint group_size;
    uint gate_weight;
    uint gate_scales;
    uint gate_biases;
    uint up_weight;
    uint up_scales;
    uint up_biases;
    uint down_weight;
    uint down_scales;
    uint down_biases;
    uint act_stride;
    uint expert_size;
    uint rows_per_tg;
    uint reserved0;
    uint reserved1;
    uint reserved2;
};

struct NMOERouteSharedQ4Args {
    uint hidden_dim;
    uint packed_cols;
    uint group_size;
    uint rows_per_tg;
};

static inline float nmoe_bf16_to_f32(ushort value) {
    return as_type<float>((uint)value << 16);
}

static inline float nmoe_sigmoid(float x) {
    return 1.0f / (1.0f + exp(-x));
}

static inline float nmoe_silu(float x) {
    return x * nmoe_sigmoid(x);
}

// ============================================================================
// 4-bit / 2-bit dequant matvec — one SIMD group per output row.
// ============================================================================

#define NMOE_ROWS_PER_TG 8
#define NMOE_MAX_MATVEC_IN_DIM 4096

static inline void nmoe_dequant_matvec_tg_impl(
    const device uint *packed_weight,
    const device ushort *scales,
    const device ushort *biases,
    const device float *input,
    device float *output,
    constant NMOEDequantMatvecArgs &args,
    threadgroup float *x_shared,
    uint bits_per_value, uint values_per_word, uint mask,
    uint tgid, uint lid, uint simd_lane, uint simd_group)
{
    uint in_dim = min(args.in_dim, (uint)NMOE_MAX_MATVEC_IN_DIM);
    uint rows_per_tg = args.rows_per_tg == 0u ? (uint)NMOE_ROWS_PER_TG : args.rows_per_tg;
    uint threads_per_tg = rows_per_tg * 32u;
    for (uint i = lid; i < in_dim; i += threads_per_tg) {
        x_shared[i] = input[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_group >= rows_per_tg) return;
    uint row = tgid * rows_per_tg + simd_group;
    if (row >= args.out_rows || args.group_size == 0u) return;

    const uint scale_groups = args.in_dim / args.group_size;
    const device uint *weight_row = packed_weight + row * args.packed_cols;
    const device ushort *scale_row = scales + row * scale_groups;
    const device ushort *bias_row = biases + row * scale_groups;

    float acc = 0.0f;
    for (uint pi = simd_lane; pi < args.packed_cols; pi += 32u) {
        uint word = weight_row[pi];
        uint base_col = pi * values_per_word;
        uint group = base_col / args.group_size;
        float scale = nmoe_bf16_to_f32(scale_row[group]);
        float bias = nmoe_bf16_to_f32(bias_row[group]);
        for (uint lane = 0u; lane < values_per_word; ++lane) {
            uint col = base_col + lane;
            if (col >= args.in_dim) break;
            uint q = (word >> (lane * bits_per_value)) & mask;
            float x = x_shared[col];
            acc += fma((float)q, scale, bias) * x;
        }
    }

    float sum = simd_sum(acc);
    if (simd_lane == 0u) output[row] = sum;
}

kernel void nmoe_dequant_matvec_q4(
    const device uint *packed_weight [[buffer(0)]],
    const device ushort *scales [[buffer(1)]],
    const device ushort *biases [[buffer(2)]],
    const device float *input [[buffer(3)]],
    device float *output [[buffer(4)]],
    constant NMOEDequantMatvecArgs &args [[buffer(5)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]])
{
    threadgroup float x_shared[NMOE_MAX_MATVEC_IN_DIM];
    uint in_dim = min(args.in_dim, (uint)NMOE_MAX_MATVEC_IN_DIM);
    uint rows_per_tg = args.rows_per_tg == 0u ? (uint)NMOE_ROWS_PER_TG : args.rows_per_tg;
    uint threads_per_tg = rows_per_tg * 32u;
    for (uint i = lid; i < in_dim; i += threads_per_tg) {
        x_shared[i] = input[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_group >= rows_per_tg) return;
    uint row = tgid * rows_per_tg + simd_group;
    if (row >= args.out_rows || args.group_size == 0u) return;

    uint packed_cols = args.packed_cols;
    uint scale_groups = args.in_dim / args.group_size;
    uint packed_per_group = args.group_size / 8u;
    const device uint *weight_row = packed_weight + row * packed_cols;
    const device ushort *scale_row = scales + row * scale_groups;
    const device ushort *bias_row = biases + row * scale_groups;

    float acc = 0.0f;
    for (uint pi = simd_lane; pi < packed_cols; pi += 32u) {
        uint group = pi / packed_per_group;
        float scale = nmoe_bf16_to_f32(scale_row[group]);
        float bias = nmoe_bf16_to_f32(bias_row[group]);
        uint word = weight_row[pi];
        uint x_base = pi * 8u;

        float x0 = x_shared[x_base + 0u];
        float x1 = x_shared[x_base + 1u];
        float x2 = x_shared[x_base + 2u];
        float x3 = x_shared[x_base + 3u];
        float x4 = x_shared[x_base + 4u];
        float x5 = x_shared[x_base + 5u];
        float x6 = x_shared[x_base + 6u];
        float x7 = x_shared[x_base + 7u];

        acc += fma((float)((word >>  0u) & 0xFu), scale * x0, bias * x0);
        acc += fma((float)((word >>  4u) & 0xFu), scale * x1, bias * x1);
        acc += fma((float)((word >>  8u) & 0xFu), scale * x2, bias * x2);
        acc += fma((float)((word >> 12u) & 0xFu), scale * x3, bias * x3);
        acc += fma((float)((word >> 16u) & 0xFu), scale * x4, bias * x4);
        acc += fma((float)((word >> 20u) & 0xFu), scale * x5, bias * x5);
        acc += fma((float)((word >> 24u) & 0xFu), scale * x6, bias * x6);
        acc += fma((float)((word >> 28u) & 0xFu), scale * x7, bias * x7);
    }

    float sum = simd_sum(acc);
    if (simd_lane == 0u) output[row] = sum;
}

kernel void nmoe_dequant_matvec_q2(
    const device uint *packed_weight [[buffer(0)]],
    const device ushort *scales [[buffer(1)]],
    const device ushort *biases [[buffer(2)]],
    const device float *input [[buffer(3)]],
    device float *output [[buffer(4)]],
    constant NMOEDequantMatvecArgs &args [[buffer(5)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]])
{
    threadgroup float x_shared[NMOE_MAX_MATVEC_IN_DIM];
    nmoe_dequant_matvec_tg_impl(packed_weight, scales, biases, input, output, args,
                                x_shared,
                                2u, 16u, 0x3u, tgid, lid, simd_lane, simd_group);
}

kernel void nmoe_expert_gate_up_q4(
    const device uint *gate_weight [[buffer(0)]],
    const device ushort *gate_scales [[buffer(1)]],
    const device ushort *gate_biases [[buffer(2)]],
    const device uint *up_weight [[buffer(3)]],
    const device ushort *up_scales [[buffer(4)]],
    const device ushort *up_biases [[buffer(5)]],
    const device float *input [[buffer(6)]],
    device float *output [[buffer(7)]],
    constant NMOEDequantMatvecArgs &args [[buffer(8)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]])
{
    threadgroup float x_shared[NMOE_MAX_MATVEC_IN_DIM];
    uint in_dim = min(args.in_dim, (uint)NMOE_MAX_MATVEC_IN_DIM);
    uint rows_per_tg = args.rows_per_tg == 0u ? (uint)NMOE_ROWS_PER_TG : args.rows_per_tg;
    uint threads_per_tg = rows_per_tg * 32u;
    for (uint i = lid; i < in_dim; i += threads_per_tg) {
        x_shared[i] = input[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_group >= rows_per_tg) return;
    uint row = tgid * rows_per_tg + simd_group;
    if (row >= args.out_rows || args.group_size == 0u) return;

    const uint scale_groups = args.in_dim / args.group_size;
    const device uint *gate_row = gate_weight + row * args.packed_cols;
    const device ushort *gate_scale_row = gate_scales + row * scale_groups;
    const device ushort *gate_bias_row = gate_biases + row * scale_groups;
    const device uint *up_row = up_weight + row * args.packed_cols;
    const device ushort *up_scale_row = up_scales + row * scale_groups;
    const device ushort *up_bias_row = up_biases + row * scale_groups;

    float gate_acc = 0.0f;
    float up_acc = 0.0f;
    for (uint pi = simd_lane; pi < args.packed_cols; pi += 32u) {
        uint base_col = pi * 8u;
        uint group = base_col / args.group_size;
        float gate_scale = nmoe_bf16_to_f32(gate_scale_row[group]);
        float gate_bias = nmoe_bf16_to_f32(gate_bias_row[group]);
        float up_scale = nmoe_bf16_to_f32(up_scale_row[group]);
        float up_bias = nmoe_bf16_to_f32(up_bias_row[group]);
        uint gate_word = gate_row[pi];
        uint up_word = up_row[pi];

        for (uint lane = 0u; lane < 8u; ++lane) {
            uint col = base_col + lane;
            if (col >= args.in_dim) break;
            float x = x_shared[col];
            uint shift = lane * 4u;
            gate_acc += fma((float)((gate_word >> shift) & 0xFu), gate_scale, gate_bias) * x;
            up_acc += fma((float)((up_word >> shift) & 0xFu), up_scale, up_bias) * x;
        }
    }

    float gate_sum = simd_sum(gate_acc);
    float up_sum = simd_sum(up_acc);
    if (simd_lane == 0u) output[row] = nmoe_silu(gate_sum) * up_sum;
}

kernel void nmoe_route_shared_q4(
    const device uint *router_weight [[buffer(0)]],
    const device ushort *router_scales [[buffer(1)]],
    const device ushort *router_biases [[buffer(2)]],
    const device uint *shared_gate_weight [[buffer(3)]],
    const device ushort *shared_gate_scales [[buffer(4)]],
    const device ushort *shared_gate_biases [[buffer(5)]],
    const device uint *shared_up_weight [[buffer(6)]],
    const device ushort *shared_up_scales [[buffer(7)]],
    const device ushort *shared_up_biases [[buffer(8)]],
    const device uint *shared_score_weight [[buffer(9)]],
    const device ushort *shared_score_scales [[buffer(10)]],
    const device ushort *shared_score_biases [[buffer(11)]],
    const device float *input [[buffer(12)]],
    device float *router_out [[buffer(13)]],
    device float *shared_act [[buffer(14)]],
    device float *shared_score_out [[buffer(15)]],
    constant NMOERouteSharedQ4Args &args [[buffer(16)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]])
{
    threadgroup float x_shared[NMOE_MAX_MATVEC_IN_DIM];
    uint rows_per_tg = args.rows_per_tg == 0u ? (uint)NMOE_ROWS_PER_TG : args.rows_per_tg;
    uint threads_per_tg = rows_per_tg * 32u;
    uint in_dim = min(args.hidden_dim, (uint)NMOE_MAX_MATVEC_IN_DIM);
    for (uint i = lid; i < in_dim; i += threads_per_tg) {
        x_shared[i] = input[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_group >= rows_per_tg || args.group_size == 0u) return;
    uint row = tgid * rows_per_tg + simd_group;
    if (row >= 769u) return;

    uint scale_groups = args.hidden_dim / args.group_size;
    if (row < 512u) {
        const device uint *gate_row = shared_gate_weight + row * args.packed_cols;
        const device ushort *gate_scale_row = shared_gate_scales + row * scale_groups;
        const device ushort *gate_bias_row = shared_gate_biases + row * scale_groups;
        const device uint *up_row = shared_up_weight + row * args.packed_cols;
        const device ushort *up_scale_row = shared_up_scales + row * scale_groups;
        const device ushort *up_bias_row = shared_up_biases + row * scale_groups;
        float gate_acc = 0.0f;
        float up_acc = 0.0f;
        for (uint pi = simd_lane; pi < args.packed_cols; pi += 32u) {
            uint group = pi / (args.group_size / 8u);
            float gate_scale = nmoe_bf16_to_f32(gate_scale_row[group]);
            float gate_bias = nmoe_bf16_to_f32(gate_bias_row[group]);
            float up_scale = nmoe_bf16_to_f32(up_scale_row[group]);
            float up_bias = nmoe_bf16_to_f32(up_bias_row[group]);
            uint gate_word = gate_row[pi];
            uint up_word = up_row[pi];
            uint x_base = pi * 8u;
            float x0 = x_shared[x_base + 0u];
            float x1 = x_shared[x_base + 1u];
            float x2 = x_shared[x_base + 2u];
            float x3 = x_shared[x_base + 3u];
            float x4 = x_shared[x_base + 4u];
            float x5 = x_shared[x_base + 5u];
            float x6 = x_shared[x_base + 6u];
            float x7 = x_shared[x_base + 7u];

            gate_acc += fma((float)((gate_word >>  0u) & 0xFu), gate_scale * x0, gate_bias * x0);
            gate_acc += fma((float)((gate_word >>  4u) & 0xFu), gate_scale * x1, gate_bias * x1);
            gate_acc += fma((float)((gate_word >>  8u) & 0xFu), gate_scale * x2, gate_bias * x2);
            gate_acc += fma((float)((gate_word >> 12u) & 0xFu), gate_scale * x3, gate_bias * x3);
            gate_acc += fma((float)((gate_word >> 16u) & 0xFu), gate_scale * x4, gate_bias * x4);
            gate_acc += fma((float)((gate_word >> 20u) & 0xFu), gate_scale * x5, gate_bias * x5);
            gate_acc += fma((float)((gate_word >> 24u) & 0xFu), gate_scale * x6, gate_bias * x6);
            gate_acc += fma((float)((gate_word >> 28u) & 0xFu), gate_scale * x7, gate_bias * x7);

            up_acc += fma((float)((up_word >>  0u) & 0xFu), up_scale * x0, up_bias * x0);
            up_acc += fma((float)((up_word >>  4u) & 0xFu), up_scale * x1, up_bias * x1);
            up_acc += fma((float)((up_word >>  8u) & 0xFu), up_scale * x2, up_bias * x2);
            up_acc += fma((float)((up_word >> 12u) & 0xFu), up_scale * x3, up_bias * x3);
            up_acc += fma((float)((up_word >> 16u) & 0xFu), up_scale * x4, up_bias * x4);
            up_acc += fma((float)((up_word >> 20u) & 0xFu), up_scale * x5, up_bias * x5);
            up_acc += fma((float)((up_word >> 24u) & 0xFu), up_scale * x6, up_bias * x6);
            up_acc += fma((float)((up_word >> 28u) & 0xFu), up_scale * x7, up_bias * x7);
        }
        float gate_sum = simd_sum(gate_acc);
        float up_sum = simd_sum(up_acc);
        if (simd_lane == 0u) shared_act[row] = nmoe_silu(gate_sum) * up_sum;
        return;
    }

    uint proj_row = row - 512u;
    const device uint *weight = proj_row < 256u
        ? router_weight + proj_row * args.packed_cols
        : shared_score_weight;
    const device ushort *scale_row = proj_row < 256u
        ? router_scales + proj_row * scale_groups
        : shared_score_scales;
    const device ushort *bias_row = proj_row < 256u
        ? router_biases + proj_row * scale_groups
        : shared_score_biases;

    float acc = 0.0f;
    for (uint pi = simd_lane; pi < args.packed_cols; pi += 32u) {
        uint word = weight[pi];
        uint group = pi / (args.group_size / 8u);
        float scale = nmoe_bf16_to_f32(scale_row[group]);
        float bias = nmoe_bf16_to_f32(bias_row[group]);
        uint x_base = pi * 8u;
        float x0 = x_shared[x_base + 0u];
        float x1 = x_shared[x_base + 1u];
        float x2 = x_shared[x_base + 2u];
        float x3 = x_shared[x_base + 3u];
        float x4 = x_shared[x_base + 4u];
        float x5 = x_shared[x_base + 5u];
        float x6 = x_shared[x_base + 6u];
        float x7 = x_shared[x_base + 7u];

        acc += fma((float)((word >>  0u) & 0xFu), scale * x0, bias * x0);
        acc += fma((float)((word >>  4u) & 0xFu), scale * x1, bias * x1);
        acc += fma((float)((word >>  8u) & 0xFu), scale * x2, bias * x2);
        acc += fma((float)((word >> 12u) & 0xFu), scale * x3, bias * x3);
        acc += fma((float)((word >> 16u) & 0xFu), scale * x4, bias * x4);
        acc += fma((float)((word >> 20u) & 0xFu), scale * x5, bias * x5);
        acc += fma((float)((word >> 24u) & 0xFu), scale * x6, bias * x6);
        acc += fma((float)((word >> 28u) & 0xFu), scale * x7, bias * x7);
    }

    float sum = simd_sum(acc);
    if (simd_lane == 0u) {
        if (proj_row < 256u) router_out[proj_row] = sum;
        else shared_score_out[0] = sum;
    }
}

static inline const device uchar *nmoe_select_expert_base(
    uint expert,
    const device uchar *expert0,
    const device uchar *expert1,
    const device uchar *expert2,
    const device uchar *expert3,
    const device uchar *expert4,
    const device uchar *expert5,
    const device uchar *expert6,
    const device uchar *expert7)
{
    switch (expert) {
        case 1u: return expert1;
        case 2u: return expert2;
        case 3u: return expert3;
        case 4u: return expert4;
        case 5u: return expert5;
        case 6u: return expert6;
        case 7u: return expert7;
        default: return expert0;
    }
}

static inline device float *nmoe_select_expert_out(
    uint expert,
    device float *out0,
    device float *out1,
    device float *out2,
    device float *out3,
    device float *out4,
    device float *out5,
    device float *out6,
    device float *out7)
{
    switch (expert) {
        case 1u: return out1;
        case 2u: return out2;
        case 3u: return out3;
        case 4u: return out4;
        case 5u: return out5;
        case 6u: return out6;
        case 7u: return out7;
        default: return out0;
    }
}

kernel void nmoe_expert_gate_up_q4_batched(
    const device uchar *expert0 [[buffer(0)]],
    const device uchar *expert1 [[buffer(1)]],
    const device uchar *expert2 [[buffer(2)]],
    const device uchar *expert3 [[buffer(3)]],
    const device uchar *expert4 [[buffer(4)]],
    const device uchar *expert5 [[buffer(5)]],
    const device uchar *expert6 [[buffer(6)]],
    const device uchar *expert7 [[buffer(7)]],
    const device float *input [[buffer(8)]],
    device float *act [[buffer(9)]],
    constant NMOEExpertBatchedArgs &args [[buffer(10)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]])
{
    uint rows_per_tg = args.rows_per_tg == 0u ? (uint)NMOE_ROWS_PER_TG : args.rows_per_tg;
    uint row_groups = (args.out_rows + rows_per_tg - 1u) / rows_per_tg;
    if (row_groups == 0u) return;
    uint expert = tgid / row_groups;
    if (expert >= args.expert_count) return;

    threadgroup float x_shared[NMOE_MAX_MATVEC_IN_DIM];
    uint in_dim = min(args.in_dim, (uint)NMOE_MAX_MATVEC_IN_DIM);
    uint threads_per_tg = rows_per_tg * 32u;
    for (uint i = lid; i < in_dim; i += threads_per_tg) {
        x_shared[i] = input[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_group >= rows_per_tg) return;
    uint row = (tgid - expert * row_groups) * rows_per_tg + simd_group;
    if (row >= args.out_rows || args.group_size == 0u) return;

    const device uchar *base = nmoe_select_expert_base(expert, expert0, expert1, expert2, expert3,
                                                       expert4, expert5, expert6, expert7);
    const device uint *gate_weight = (const device uint *)(base + args.gate_weight);
    const device ushort *gate_scales = (const device ushort *)(base + args.gate_scales);
    const device ushort *gate_biases = (const device ushort *)(base + args.gate_biases);
    const device uint *up_weight = (const device uint *)(base + args.up_weight);
    const device ushort *up_scales = (const device ushort *)(base + args.up_scales);
    const device ushort *up_biases = (const device ushort *)(base + args.up_biases);

    uint scale_groups = args.in_dim / args.group_size;
    const device uint *gate_row = gate_weight + row * args.packed_cols;
    const device ushort *gate_scale_row = gate_scales + row * scale_groups;
    const device ushort *gate_bias_row = gate_biases + row * scale_groups;
    const device uint *up_row = up_weight + row * args.packed_cols;
    const device ushort *up_scale_row = up_scales + row * scale_groups;
    const device ushort *up_bias_row = up_biases + row * scale_groups;

    float gate_acc = 0.0f;
    float up_acc = 0.0f;
    for (uint pi = simd_lane; pi < args.packed_cols; pi += 32u) {
        uint base_col = pi * 8u;
        uint group = base_col / args.group_size;
        float gate_scale = nmoe_bf16_to_f32(gate_scale_row[group]);
        float gate_bias = nmoe_bf16_to_f32(gate_bias_row[group]);
        float up_scale = nmoe_bf16_to_f32(up_scale_row[group]);
        float up_bias = nmoe_bf16_to_f32(up_bias_row[group]);
        uint gate_word = gate_row[pi];
        uint up_word = up_row[pi];
        for (uint lane = 0u; lane < 8u; ++lane) {
            uint col = base_col + lane;
            if (col >= args.in_dim) break;
            float x = x_shared[col];
            uint shift = lane * 4u;
            gate_acc += fma((float)((gate_word >> shift) & 0xFu), gate_scale, gate_bias) * x;
            up_acc += fma((float)((up_word >> shift) & 0xFu), up_scale, up_bias) * x;
        }
    }

    float gate_sum = simd_sum(gate_acc);
    float up_sum = simd_sum(up_acc);
    if (simd_lane == 0u) {
        act[expert * args.act_stride + row] = nmoe_silu(gate_sum) * up_sum;
    }
}

kernel void nmoe_expert_down_q4_batched(
    const device uchar *expert0 [[buffer(0)]],
    const device uchar *expert1 [[buffer(1)]],
    const device uchar *expert2 [[buffer(2)]],
    const device uchar *expert3 [[buffer(3)]],
    const device uchar *expert4 [[buffer(4)]],
    const device uchar *expert5 [[buffer(5)]],
    const device uchar *expert6 [[buffer(6)]],
    const device uchar *expert7 [[buffer(7)]],
    const device float *act [[buffer(8)]],
    device float *out0 [[buffer(9)]],
    device float *out1 [[buffer(10)]],
    device float *out2 [[buffer(11)]],
    device float *out3 [[buffer(12)]],
    device float *out4 [[buffer(13)]],
    device float *out5 [[buffer(14)]],
    device float *out6 [[buffer(15)]],
    device float *out7 [[buffer(16)]],
    constant NMOEExpertBatchedArgs &args [[buffer(17)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]])
{
    uint rows_per_tg = args.rows_per_tg == 0u ? (uint)NMOE_ROWS_PER_TG : args.rows_per_tg;
    uint row_groups = (args.out_rows + rows_per_tg - 1u) / rows_per_tg;
    if (row_groups == 0u) return;
    uint expert = tgid / row_groups;
    if (expert >= args.expert_count) return;

    threadgroup float x_shared[512];
    uint in_dim = min(args.in_dim, 512u);
    const device float *expert_act = act + expert * args.act_stride;
    uint threads_per_tg = rows_per_tg * 32u;
    for (uint i = lid; i < in_dim; i += threads_per_tg) {
        x_shared[i] = expert_act[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_group >= rows_per_tg) return;
    uint row = (tgid - expert * row_groups) * rows_per_tg + simd_group;
    if (row >= args.out_rows || args.group_size == 0u) return;

    const device uchar *base = nmoe_select_expert_base(expert, expert0, expert1, expert2, expert3,
                                                       expert4, expert5, expert6, expert7);
    const device uint *down_weight = (const device uint *)(base + args.down_weight);
    const device ushort *down_scales = (const device ushort *)(base + args.down_scales);
    const device ushort *down_biases = (const device ushort *)(base + args.down_biases);
    device float *out = nmoe_select_expert_out(expert, out0, out1, out2, out3, out4, out5, out6, out7);

    uint scale_groups = args.in_dim / args.group_size;
    const device uint *weight_row = down_weight + row * args.packed_cols;
    const device ushort *scale_row = down_scales + row * scale_groups;
    const device ushort *bias_row = down_biases + row * scale_groups;

    float acc = 0.0f;
    for (uint pi = simd_lane; pi < args.packed_cols; pi += 32u) {
        uint word = weight_row[pi];
        uint base_col = pi * 8u;
        uint group = base_col / args.group_size;
        float scale = nmoe_bf16_to_f32(scale_row[group]);
        float bias = nmoe_bf16_to_f32(bias_row[group]);
        for (uint lane = 0u; lane < 8u; ++lane) {
            uint col = base_col + lane;
            if (col >= args.in_dim) break;
            float x = x_shared[col];
            acc += fma((float)((word >> (lane * 4u)) & 0xFu), scale, bias) * x;
        }
    }

    float sum = simd_sum(acc);
    if (simd_lane == 0u) out[row] = sum;
}

kernel void nmoe_expert_gate_up_q4_routed(
    const device uchar *layer_experts [[buffer(0)]],
    const device uint *route_indices [[buffer(1)]],
    const device float *input [[buffer(2)]],
    device float *act [[buffer(3)]],
    constant NMOEExpertBatchedArgs &args [[buffer(4)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]])
{
    uint rows_per_tg = args.rows_per_tg == 0u ? (uint)NMOE_ROWS_PER_TG : args.rows_per_tg;
    uint row_groups = (args.out_rows + rows_per_tg - 1u) / rows_per_tg;
    if (row_groups == 0u) return;
    uint slot = tgid / row_groups;
    if (slot >= args.expert_count) return;

    threadgroup float x_shared[NMOE_MAX_MATVEC_IN_DIM];
    uint in_dim = min(args.in_dim, (uint)NMOE_MAX_MATVEC_IN_DIM);
    uint threads_per_tg = rows_per_tg * 32u;
    for (uint i = lid; i < in_dim; i += threads_per_tg) {
        x_shared[i] = input[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_group >= rows_per_tg) return;
    uint row = (tgid - slot * row_groups) * rows_per_tg + simd_group;
    if (row >= args.out_rows || args.group_size == 0u) return;

    uint expert_index = route_indices[slot];
    const device uchar *base = layer_experts + (ulong)expert_index * (ulong)args.expert_size;
    const device uint *gate_weight = (const device uint *)(base + args.gate_weight);
    const device ushort *gate_scales = (const device ushort *)(base + args.gate_scales);
    const device ushort *gate_biases = (const device ushort *)(base + args.gate_biases);
    const device uint *up_weight = (const device uint *)(base + args.up_weight);
    const device ushort *up_scales = (const device ushort *)(base + args.up_scales);
    const device ushort *up_biases = (const device ushort *)(base + args.up_biases);

    uint scale_groups = args.in_dim / args.group_size;
    const device uint *gate_row = gate_weight + row * args.packed_cols;
    const device ushort *gate_scale_row = gate_scales + row * scale_groups;
    const device ushort *gate_bias_row = gate_biases + row * scale_groups;
    const device uint *up_row = up_weight + row * args.packed_cols;
    const device ushort *up_scale_row = up_scales + row * scale_groups;
    const device ushort *up_bias_row = up_biases + row * scale_groups;

    float gate_acc = 0.0f;
    float up_acc = 0.0f;
    for (uint pi = simd_lane; pi < args.packed_cols; pi += 32u) {
        uint base_col = pi * 8u;
        uint group = base_col / args.group_size;
        float gate_scale = nmoe_bf16_to_f32(gate_scale_row[group]);
        float gate_bias = nmoe_bf16_to_f32(gate_bias_row[group]);
        float up_scale = nmoe_bf16_to_f32(up_scale_row[group]);
        float up_bias = nmoe_bf16_to_f32(up_bias_row[group]);
        uint gate_word = gate_row[pi];
        uint up_word = up_row[pi];
        for (uint lane = 0u; lane < 8u; ++lane) {
            uint col = base_col + lane;
            if (col >= args.in_dim) break;
            float x = x_shared[col];
            uint shift = lane * 4u;
            gate_acc += fma((float)((gate_word >> shift) & 0xFu), gate_scale, gate_bias) * x;
            up_acc += fma((float)((up_word >> shift) & 0xFu), up_scale, up_bias) * x;
        }
    }

    float gate_sum = simd_sum(gate_acc);
    float up_sum = simd_sum(up_acc);
    if (simd_lane == 0u) {
        act[slot * args.act_stride + row] = nmoe_silu(gate_sum) * up_sum;
    }
}

kernel void nmoe_expert_down_q4_routed(
    const device uchar *layer_experts [[buffer(0)]],
    const device uint *route_indices [[buffer(1)]],
    const device float *act [[buffer(2)]],
    device float *out0 [[buffer(3)]],
    device float *out1 [[buffer(4)]],
    device float *out2 [[buffer(5)]],
    device float *out3 [[buffer(6)]],
    device float *out4 [[buffer(7)]],
    device float *out5 [[buffer(8)]],
    device float *out6 [[buffer(9)]],
    device float *out7 [[buffer(10)]],
    constant NMOEExpertBatchedArgs &args [[buffer(11)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]])
{
    uint rows_per_tg = args.rows_per_tg == 0u ? (uint)NMOE_ROWS_PER_TG : args.rows_per_tg;
    uint row_groups = (args.out_rows + rows_per_tg - 1u) / rows_per_tg;
    if (row_groups == 0u) return;
    uint slot = tgid / row_groups;
    if (slot >= args.expert_count) return;

    threadgroup float x_shared[512];
    uint in_dim = min(args.in_dim, 512u);
    const device float *expert_act = act + slot * args.act_stride;
    uint threads_per_tg = rows_per_tg * 32u;
    for (uint i = lid; i < in_dim; i += threads_per_tg) {
        x_shared[i] = expert_act[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_group >= rows_per_tg) return;
    uint row = (tgid - slot * row_groups) * rows_per_tg + simd_group;
    if (row >= args.out_rows || args.group_size == 0u) return;

    uint expert_index = route_indices[slot];
    const device uchar *base = layer_experts + (ulong)expert_index * (ulong)args.expert_size;
    const device uint *down_weight = (const device uint *)(base + args.down_weight);
    const device ushort *down_scales = (const device ushort *)(base + args.down_scales);
    const device ushort *down_biases = (const device ushort *)(base + args.down_biases);
    device float *out = nmoe_select_expert_out(slot, out0, out1, out2, out3, out4, out5, out6, out7);

    uint scale_groups = args.in_dim / args.group_size;
    const device uint *weight_row = down_weight + row * args.packed_cols;
    const device ushort *scale_row = down_scales + row * scale_groups;
    const device ushort *bias_row = down_biases + row * scale_groups;

    float acc = 0.0f;
    for (uint pi = simd_lane; pi < args.packed_cols; pi += 32u) {
        uint word = weight_row[pi];
        uint base_col = pi * 8u;
        uint group = base_col / args.group_size;
        float scale = nmoe_bf16_to_f32(scale_row[group]);
        float bias = nmoe_bf16_to_f32(bias_row[group]);
        for (uint lane = 0u; lane < 8u; ++lane) {
            uint col = base_col + lane;
            if (col >= args.in_dim) break;
            float x = x_shared[col];
            acc += fma((float)((word >> (lane * 4u)) & 0xFu), scale, bias) * x;
        }
    }

    float sum = simd_sum(acc);
    if (simd_lane == 0u) out[row] = sum;
}

// ============================================================================
// RMS norm — ORIGINAL single-threaded version (correct behavior)
// ============================================================================

kernel void nmoe_rms_norm(
    const device float *input [[buffer(0)]],
    const device ushort *weight_bf16 [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant NMOERMSNormArgs &args [[buffer(3)]],
    uint lid [[thread_position_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]])
{
    if (args.dim == 0) return;

    threadgroup float partial[32];
    float sum = 0.0f;
    for (uint i = lid; i < args.dim; i += tg_size) {
        float value = input[i];
        sum += value * value;
    }

    float simd_value = simd_sum(sum);
    uint simd_count = (tg_size + 31u) / 32u;
    if (simd_lane == 0u) partial[simd_group] = simd_value;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float total = 0.0f;
    if (simd_group == 0u) {
        total = simd_lane < simd_count ? partial[simd_lane] : 0.0f;
        total = simd_sum(total);
        if (simd_lane == 0u) partial[0] = total;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float inv_rms = rsqrt(partial[0] / (float)args.dim + args.epsilon);
    for (uint i = lid; i < args.dim; i += tg_size) {
        float weight = nmoe_bf16_to_f32(weight_bf16[i]);
        if (args.add_one != 0u) weight += 1.0f;
        output[i] = input[i] * inv_rms * weight;
    }
}

// Parallel RMS norm — sum + apply (NEW, for performance)
kernel void nmoe_rms_norm_sum(
    const device float *input [[buffer(0)]],
    device float *sum_sq_out [[buffer(1)]],
    constant uint &dim [[buffer(2)]],
    uint tid [[thread_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]])
{
    threadgroup float shared[32];
    float acc = 0.0f;
    // CORRECT stride: total grid size, not tg_size
    uint grid_size = tg_size; // will be fixed by dispatch with 1 threadgroup of 256
    for (uint i = tid; i < dim; i += grid_size) {
        float v = input[i];
        acc += v * v;
    }
    float simd_val = simd_sum(acc);
    uint simd_lane = lid % 32;
    uint simd_group = lid / 32;
    uint num_simd_groups = (tg_size + 31) / 32;
    if (simd_lane == 0) shared[simd_group] = simd_val;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_group == 0 && simd_lane < num_simd_groups) {
        float val = simd_sum(shared[simd_lane]);
        if (simd_lane == 0) sum_sq_out[0] = val;
    }
}

kernel void nmoe_rms_norm_apply_bf16(
    const device float *input [[buffer(0)]],
    const device ushort *weight_bf16 [[buffer(1)]],
    const device float *sum_sq [[buffer(2)]],
    device float *output [[buffer(3)]],
    constant uint &dim [[buffer(4)]],
    constant float &eps [[buffer(5)]],
    constant uint &add_one [[buffer(6)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= dim) return;
    float inv_rms = rsqrt(sum_sq[0] / (float)dim + eps);
    float w = add_one ? (1.0f + nmoe_bf16_to_f32(weight_bf16[tid]))
                      : nmoe_bf16_to_f32(weight_bf16[tid]);
    output[tid] = input[tid] * inv_rms * w;
}

kernel void nmoe_rms_norm_apply_f32(
    const device float *input [[buffer(0)]],
    const device float *weight [[buffer(1)]],
    const device float *sum_sq [[buffer(2)]],
    device float *output [[buffer(3)]],
    constant uint &dim [[buffer(4)]],
    constant float &eps [[buffer(5)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= dim) return;
    float inv_rms = rsqrt(sum_sq[0] / (float)dim + eps);
    output[tid] = input[tid] * inv_rms * weight[tid];
}

// ============================================================================
// RoPE
// ============================================================================

kernel void nmoe_rope(
    device float *q [[buffer(0)]],
    device float *k [[buffer(1)]],
    constant NMOERopeArgs &args [[buffer(2)]],
    uint tid [[thread_position_in_grid]])
{
    if (args.rotary_dim == 0) return;
    uint half_dim = args.rotary_dim / 2u;
    if (tid >= half_dim) return;

    float theta = args.theta > 0.0f ? args.theta : NMOE_ROPE_THETA;
    float exponent = -2.0f * (float)tid / (float)args.rotary_dim;
    float freq = pow(theta, exponent);
    float angle = (float)args.position * freq;
    float c = cos(angle);
    float s = sin(angle);

    uint pair = tid + half_dim;
    float q0 = q[tid], q1 = q[pair];
    float k0 = k[tid], k1 = k[pair];

    q[tid] = q0 * c - q1 * s;
    q[pair] = q0 * s + q1 * c;
    k[tid] = k0 * c - k1 * s;
    k[pair] = k0 * s + k1 * c;
}

// ============================================================================
// Sigmoid gate
// ============================================================================

kernel void nmoe_sigmoid_gate(
    device float *values [[buffer(0)]],
    constant NMOESigmoidGateArgs &args [[buffer(1)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= args.count) return;
    values[tid] = nmoe_sigmoid(values[tid]);
}

kernel void nmoe_full_qk_prep(
    device float *q [[buffer(0)]],
    device float *k [[buffer(1)]],
    const device ushort *q_weight_bf16 [[buffer(2)]],
    const device ushort *k_weight_bf16 [[buffer(3)]],
    constant NMOEFullQKPrepArgs &args [[buffer(4)]],
    uint head [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]])
{
    threadgroup float partial[256];
    uint head_dim = args.head_dim;
    if (head_dim == 0u || tid >= head_dim) return;

    bool is_q = head < args.full_heads;
    uint local_head = is_q ? head : (head - args.full_heads);
    device float *vec = is_q ? (q + local_head * args.q_stride)
                             : (k + local_head * head_dim);
    const device ushort *weight_bf16 = is_q ? q_weight_bf16 : k_weight_bf16;

    float value = vec[tid];
    partial[tid] = value * value;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0u) {
        float sum = 0.0f;
        for (uint i = 0u; i < head_dim; ++i) sum += partial[i];
        partial[0] = sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float weight = nmoe_bf16_to_f32(weight_bf16[tid]);
    if (args.add_one != 0u) weight += 1.0f;
    vec[tid] = value * rsqrt(partial[0] / (float)head_dim + args.epsilon) * weight;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint half_dim = args.rotary_dim / 2u;
    if (tid < half_dim) {
        float theta = args.theta > 0.0f ? args.theta : NMOE_ROPE_THETA;
        float exponent = -2.0f * (float)tid / (float)args.rotary_dim;
        float freq = pow(theta, exponent);
        float angle = (float)args.position * freq;
        float c = cos(angle);
        float s = sin(angle);
        uint pair = tid + half_dim;
        float v0 = vec[tid];
        float v1 = vec[pair];
        vec[tid] = v0 * c - v1 * s;
        vec[pair] = v0 * s + v1 * c;
    }

    if (is_q) {
        q[local_head * args.q_stride + head_dim + tid] =
            nmoe_sigmoid(q[local_head * args.q_stride + head_dim + tid]);
    }
}

// ============================================================================
// Dequant row
// ============================================================================

static inline void nmoe_dequant_row_impl(
    const device uint *packed_row,
    const device ushort *scales,
    const device ushort *biases,
    device float *output,
    constant NMOEDequantRowArgs &args,
    uint bits_per_value, uint values_per_word, uint mask, uint tid)
{
    if (tid >= args.in_dim || args.group_size == 0) return;
    uint group = tid / args.group_size;
    uint local = tid - group * args.group_size;
    uint group_words = args.group_size / values_per_word;
    uint word_index = group * group_words + local / values_per_word;
    uint lane = local % values_per_word;
    uint q = (packed_row[word_index] >> (lane * bits_per_value)) & mask;
    float scale = nmoe_bf16_to_f32(scales[group]);
    float bias  = nmoe_bf16_to_f32(biases[group]);
    output[tid] = fma((float)q, scale, bias);
}

kernel void nmoe_dequant_row_q4(
    const device uint *packed_row [[buffer(0)]],
    const device ushort *scales [[buffer(1)]],
    const device ushort *biases [[buffer(2)]],
    device float *output [[buffer(3)]],
    constant NMOEDequantRowArgs &args [[buffer(4)]],
    uint tid [[thread_position_in_grid]])
{
    nmoe_dequant_row_impl(packed_row, scales, biases, output, args, 4u, 8u, 0xFu, tid);
}

kernel void nmoe_dequant_row_q2(
    const device uint *packed_row [[buffer(0)]],
    const device ushort *scales [[buffer(1)]],
    const device ushort *biases [[buffer(2)]],
    device float *output [[buffer(3)]],
    constant NMOEDequantRowArgs &args [[buffer(4)]],
    uint tid [[thread_position_in_grid]])
{
    nmoe_dequant_row_impl(packed_row, scales, biases, output, args, 2u, 16u, 0x3u, tid);
}

// ============================================================================
// Attention — ORIGINAL batched implementations (one thread per head)
// ============================================================================

kernel void nmoe_attn_scores_batched(
    const device float *q [[buffer(0)]],
    const device float *k_cache [[buffer(1)]],
    device float *scores [[buffer(2)]],
    constant NMOEAttentionArgs &args [[buffer(3)]],
    uint head [[thread_position_in_grid]])
{
    if (head >= args.full_heads || args.seq_len == 0 || args.head_dim == 0) return;
    uint kv_ratio = args.full_heads / max(args.full_kv_heads, 1u);
    uint kv_head = head / max(kv_ratio, 1u);
    const device float *q_head = q + head * args.q_stride;
    device float *head_scores = scores + head * args.seq_stride;
    float inv_scale = args.inv_scale > 0.0f ? args.inv_scale : rsqrt((float)args.head_dim);
    for (uint p = 0; p < args.seq_len; ++p) {
        const device float *past_k = k_cache + (p * args.cache_stride) + kv_head * args.kv_stride;
        float acc = 0.0f;
        for (uint d = 0; d < args.head_dim; ++d) acc += q_head[d] * past_k[d];
        head_scores[p] = acc * inv_scale;
    }
}

kernel void nmoe_attn_softmax_batched(
    device float *scores [[buffer(0)]],
    device float *probs [[buffer(1)]],
    constant NMOEAttentionArgs &args [[buffer(2)]],
    uint head [[thread_position_in_grid]])
{
    if (head >= args.full_heads || args.seq_len == 0) return;
    device float *head_scores = scores + head * args.seq_stride;
    device float *head_probs = probs + head * args.seq_stride;

    float max_score = head_scores[0];
    for (uint p = 1; p < args.seq_len; ++p) max_score = max(max_score, head_scores[p]);

    float sum = 0.0f;
    for (uint p = 0; p < args.seq_len; ++p) {
        float value = exp(head_scores[p] - max_score);
        head_probs[p] = value;
        sum += value;
    }

    if (sum == 0.0f) {
        float uniform = 1.0f / (float)args.seq_len;
        for (uint p = 0; p < args.seq_len; ++p) head_probs[p] = uniform;
        return;
    }
    float inv_sum = 1.0f / sum;
    for (uint p = 0; p < args.seq_len; ++p) head_probs[p] *= inv_sum;
}

kernel void nmoe_attn_values_batched(
    const device float *probs [[buffer(0)]],
    const device float *v_cache [[buffer(1)]],
    const device float *q [[buffer(2)]],
    device float *output [[buffer(3)]],
    constant NMOEAttentionArgs &args [[buffer(4)]],
    uint head [[thread_position_in_grid]])
{
    if (head >= args.full_heads || args.seq_len == 0 || args.head_dim == 0) return;
    uint kv_ratio = args.full_heads / max(args.full_kv_heads, 1u);
    uint kv_head = head / max(kv_ratio, 1u);
    const device float *q_head = q + head * args.q_stride;
    const device float *q_gate = q_head + args.head_dim;
    const device float *head_probs = probs + head * args.seq_stride;
    device float *head_out = output + head * args.head_dim;

    for (uint d = 0; d < args.head_dim; ++d) head_out[d] = 0.0f;
    for (uint p = 0; p < args.seq_len; ++p) {
        const device float *past_v = v_cache + (p * args.cache_stride) + kv_head * args.kv_stride;
        float weight = head_probs[p];
        for (uint d = 0; d < args.head_dim; ++d) head_out[d] += weight * past_v[d];
    }
    for (uint d = 0; d < args.head_dim; ++d) head_out[d] *= q_gate[d];
}

// ============================================================================
// Conv1d step
// ============================================================================

kernel void nmoe_conv1d_step(
    device float *conv_state [[buffer(0)]],
    const device float *input [[buffer(1)]],
    const device ushort *weights [[buffer(2)]],
    device float *output [[buffer(3)]],
    constant NMOELinearConv1DArgs &args [[buffer(4)]],
    uint idx [[thread_position_in_grid]])
{
    if (idx >= args.dim) return;
    uint w_base = idx * 4u;
    float acc = 0.0f;
    acc += conv_state[0 * args.dim + idx] * nmoe_bf16_to_f32(weights[w_base + 0]);
    acc += conv_state[1 * args.dim + idx] * nmoe_bf16_to_f32(weights[w_base + 1]);
    acc += conv_state[2 * args.dim + idx] * nmoe_bf16_to_f32(weights[w_base + 2]);
    float inp = input[idx];
    acc += inp * nmoe_bf16_to_f32(weights[w_base + 3]);
    output[idx] = nmoe_silu(acc);
    conv_state[0 * args.dim + idx] = conv_state[1 * args.dim + idx];
    conv_state[1 * args.dim + idx] = conv_state[2 * args.dim + idx];
    conv_state[2 * args.dim + idx] = inp;
}

// ============================================================================
// ORIGINAL GatedDeltaNet — single-threaded per head (correct behavior)
// ============================================================================

// TUI-proven parallel GatedDeltaNet step (no q_scale/k_scale — scaling in Q/K norm)
kernel void nmoe_gated_delta_net_step(
    device float *state [[buffer(0)]],
    const device float *q [[buffer(1)]],
    const device float *k [[buffer(2)]],
    const device float *v [[buffer(3)]],
    const device float *g_decay [[buffer(4)]],
    const device float *beta_gate [[buffer(5)]],
    device float *output [[buffer(6)]],
    constant NMOEGatedDeltaNetArgs &args [[buffer(7)]],
    uint vh [[threadgroup_position_in_grid]],
    uint vi [[thread_position_in_threadgroup]])
{
    if (vh >= args.v_heads || vi >= args.value_dim || args.key_dim == 0) return;

    uint kv_heads = max(args.kv_heads, 1u);
    uint head_ratio = max(args.v_heads / kv_heads, 1u);
    uint kh = vh / head_ratio;
    float g = g_decay[vh];
    float beta_val = beta_gate[vh];

    uint state_base = vh * args.value_dim * args.key_dim + vi * args.key_dim;
    uint k_base = kh * args.key_dim;
    uint v_base = vh * args.value_dim;

    // Step 1+2: Decay state row and compute kv_mem = dot(S[vi][:], k[:])
    float kv_mem = 0.0f;
    for (uint ki = 0; ki < args.key_dim; ki++) {
        float s = state[state_base + ki] * g;
        state[state_base + ki] = s;
        kv_mem += s * k[k_base + ki];
    }

    // Step 3+4: Delta update — S[vi][ki] += k[ki] * delta
    float delta = (v[v_base + vi] - kv_mem) * beta_val;
    for (uint ki = 0; ki < args.key_dim; ki++) {
        state[state_base + ki] += k[k_base + ki] * delta;
    }

    // Step 5: Output — out[vi] = dot(S[vi][:], q[:])
    float out_val = 0.0f;
    for (uint ki = 0; ki < args.key_dim; ki++) {
        out_val += state[state_base + ki] * q[k_base + ki];
    }
    output[v_base + vi] = out_val;
}

// ============================================================================
// MoE expert gate_up + down + combine
// ============================================================================

kernel void nmoe_moe_expert_gate_up(
    const device float *gate [[buffer(0)]],
    const device float *up [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant NMOESigmoidGateArgs &args [[buffer(3)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= args.count) return;
    output[tid] = nmoe_silu(gate[tid]) * up[tid];
}

kernel void nmoe_moe_expert_down(
    device float *dst [[buffer(0)]],
    const device float *src [[buffer(1)]],
    constant NMOEMoeCombineArgs &args [[buffer(2)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= args.count) return;
    dst[tid] += src[tid] * args.weight;
}

kernel void nmoe_moe_combine(
    device float *dst [[buffer(0)]],
    const device float *src [[buffer(1)]],
    constant NMOEMoeCombineArgs &args [[buffer(2)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= args.count) return;
    dst[tid] += src[tid] * args.weight;
}

kernel void nmoe_residual_add(
    const device float *a [[buffer(0)]],
    const device float *b [[buffer(1)]],
    device float *out [[buffer(2)]],
    constant uint &dim [[buffer(3)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= dim) return;
    out[tid] = a[tid] + b[tid];
}

kernel void nmoe_weighted_expert_sum(
    const device float *h_mid [[buffer(0)]],
    const device float *shared_out [[buffer(1)]],
    device float *hidden_out [[buffer(2)]],
    const device float *expert_out0 [[buffer(3)]],
    const device float *expert_out1 [[buffer(4)]],
    const device float *expert_out2 [[buffer(5)]],
    const device float *expert_out3 [[buffer(6)]],
    const device float *expert_out4 [[buffer(7)]],
    const device float *expert_out5 [[buffer(8)]],
    const device float *expert_out6 [[buffer(9)]],
    const device float *expert_out7 [[buffer(10)]],
    const device float *params [[buffer(11)]],
    constant uint &dim [[buffer(12)]],
    constant uint &K [[buffer(13)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= dim) return;
    float shared_gate = nmoe_sigmoid(params[8]);
    float moe = 0.0f;
    if (K > 0) moe += params[0] * expert_out0[tid];
    if (K > 1) moe += params[1] * expert_out1[tid];
    if (K > 2) moe += params[2] * expert_out2[tid];
    if (K > 3) moe += params[3] * expert_out3[tid];
    if (K > 4) moe += params[4] * expert_out4[tid];
    if (K > 5) moe += params[5] * expert_out5[tid];
    if (K > 6) moe += params[6] * expert_out6[tid];
    if (K > 7) moe += params[7] * expert_out7[tid];
    hidden_out[tid] = h_mid[tid] + moe + shared_gate * shared_out[tid];
}

kernel void nmoe_weighted_expert_sum_routed(
    const device float *h_mid [[buffer(0)]],
    const device float *shared_out [[buffer(1)]],
    device float *hidden_out [[buffer(2)]],
    const device float *expert_out0 [[buffer(3)]],
    const device float *expert_out1 [[buffer(4)]],
    const device float *expert_out2 [[buffer(5)]],
    const device float *expert_out3 [[buffer(6)]],
    const device float *expert_out4 [[buffer(7)]],
    const device float *expert_out5 [[buffer(8)]],
    const device float *expert_out6 [[buffer(9)]],
    const device float *expert_out7 [[buffer(10)]],
    const device float *route_weights [[buffer(11)]],
    const device float *shared_gate_raw [[buffer(12)]],
    constant uint &dim [[buffer(13)]],
    constant uint &K [[buffer(14)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= dim) return;
    float moe = 0.0f;
    if (K > 0) moe += route_weights[0] * expert_out0[tid];
    if (K > 1) moe += route_weights[1] * expert_out1[tid];
    if (K > 2) moe += route_weights[2] * expert_out2[tid];
    if (K > 3) moe += route_weights[3] * expert_out3[tid];
    if (K > 4) moe += route_weights[4] * expert_out4[tid];
    if (K > 5) moe += route_weights[5] * expert_out5[tid];
    if (K > 6) moe += route_weights[6] * expert_out6[tid];
    if (K > 7) moe += route_weights[7] * expert_out7[tid];
    hidden_out[tid] = h_mid[tid] + moe + nmoe_sigmoid(shared_gate_raw[0]) * shared_out[tid];
}

kernel void nmoe_expert_down_combine_q4(
    const device uchar *expert0 [[buffer(0)]],
    const device uchar *expert1 [[buffer(1)]],
    const device uchar *expert2 [[buffer(2)]],
    const device uchar *expert3 [[buffer(3)]],
    const device uchar *expert4 [[buffer(4)]],
    const device uchar *expert5 [[buffer(5)]],
    const device uchar *expert6 [[buffer(6)]],
    const device uchar *expert7 [[buffer(7)]],
    const device float *act [[buffer(8)]],
    const device float *h_mid [[buffer(9)]],
    const device float *shared_out [[buffer(10)]],
    device float *hidden_out [[buffer(11)]],
    const device float *params [[buffer(12)]],
    constant NMOEExpertBatchedArgs &args [[buffer(13)]],
    const device uint *shared_weight [[buffer(14)]],
    const device ushort *shared_scales [[buffer(15)]],
    const device ushort *shared_biases [[buffer(16)]],
    const device float *shared_act [[buffer(17)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]])
{
    uint rows_per_tg = args.rows_per_tg == 0u ? (uint)NMOE_ROWS_PER_TG : args.rows_per_tg;
    if (simd_group >= rows_per_tg) return;
    uint row = tgid * rows_per_tg + simd_group;
    if (row >= args.out_rows || args.group_size == 0u) return;

    float moe = 0.0f;
    uint scale_groups = args.in_dim / args.group_size;
    uint k = min(args.expert_count, 8u);

    for (uint slot = 0u; slot < k; ++slot) {
        const device uchar *base = nmoe_select_expert_base(slot, expert0, expert1, expert2, expert3,
                                                           expert4, expert5, expert6, expert7);
        const device uint *down_weight = (const device uint *)(base + args.down_weight);
        const device ushort *down_scales = (const device ushort *)(base + args.down_scales);
        const device ushort *down_biases = (const device ushort *)(base + args.down_biases);
        const device float *expert_act = act + slot * args.act_stride;

        const device uint *weight_row = down_weight + row * args.packed_cols;
        const device ushort *scale_row = down_scales + row * scale_groups;
        const device ushort *bias_row = down_biases + row * scale_groups;

        float acc = 0.0f;
        for (uint pi = simd_lane; pi < args.packed_cols; pi += 32u) {
            uint word = weight_row[pi];
            uint base_col = pi * 8u;
            uint group = base_col / args.group_size;
            float scale = nmoe_bf16_to_f32(scale_row[group]);
            float bias = nmoe_bf16_to_f32(bias_row[group]);
            for (uint lane = 0u; lane < 8u; ++lane) {
                uint col = base_col + lane;
                if (col >= args.in_dim) break;
                float x = expert_act[col];
                acc += fma((float)((word >> (lane * 4u)) & 0xFu), scale, bias) * x;
            }
        }

        float down_sum = simd_sum(acc);
        moe += params[slot] * down_sum;
    }

    const device uint *shared_weight_row = shared_weight + row * args.packed_cols;
    const device ushort *shared_scale_row = shared_scales + row * scale_groups;
    const device ushort *shared_bias_row = shared_biases + row * scale_groups;
    float shared_acc = 0.0f;
    for (uint pi = simd_lane; pi < args.packed_cols; pi += 32u) {
        uint word = shared_weight_row[pi];
        uint base_col = pi * 8u;
        uint group = base_col / args.group_size;
        float scale = nmoe_bf16_to_f32(shared_scale_row[group]);
        float bias = nmoe_bf16_to_f32(shared_bias_row[group]);
        for (uint lane = 0u; lane < 8u; ++lane) {
            uint col = base_col + lane;
            if (col >= args.in_dim) break;
            float x = shared_act[col];
            shared_acc += fma((float)((word >> (lane * 4u)) & 0xFu), scale, bias) * x;
        }
    }
    float shared_sum = simd_sum(shared_acc);

    if (simd_lane == 0u) {
        hidden_out[row] = h_mid[row] + moe + nmoe_sigmoid(params[8]) * shared_sum;
    }
}

// ============================================================================
// GPU copy (for KV cache updates inside command buffers)
// ============================================================================

kernel void nmoe_copy_f32(
    const device float *src [[buffer(0)]],
    device float *dst [[buffer(1)]],
    constant uint &count [[buffer(2)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= count) return;
    dst[tid] = src[tid];
}

// ============================================================================
// Argmax top-1
// ============================================================================

kernel void nmoe_argmax_top1(
    const device float *values [[buffer(0)]],
    device uint *result [[buffer(1)]],
    constant NMOEArgmaxArgs &args [[buffer(2)]],
    uint tid [[thread_index_in_threadgroup]])
{
    threadgroup float best_values[256];
    threadgroup uint best_indices[256];

    float best_value = -3.402823466e38f;
    uint best_index = 0u;
    for (uint i = tid; i < args.count; i += 256u) {
        float value = values[i];
        if (value > best_value || (value == best_value && i < best_index)) {
            best_value = value;
            best_index = i;
        }
    }
    best_values[tid] = best_value;
    best_indices[tid] = best_index;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = 128u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            float other_value = best_values[tid + stride];
            uint other_index = best_indices[tid + stride];
            if (other_value > best_values[tid] ||
                (other_value == best_values[tid] && other_index < best_indices[tid])) {
                best_values[tid] = other_value;
                best_indices[tid] = other_index;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0u) result[0] = best_indices[0];
}

kernel void nmoe_lm_head_argmax_q4(
    const device uint *packed_weight [[buffer(0)]],
    const device ushort *scales [[buffer(1)]],
    const device ushort *biases [[buffer(2)]],
    const device float *input [[buffer(3)]],
    device float *partial_values [[buffer(4)]],
    device uint *partial_indices [[buffer(5)]],
    constant NMOELmHeadArgmaxArgs &args [[buffer(6)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]])
{
    threadgroup float x_shared[NMOE_MAX_MATVEC_IN_DIM];
    threadgroup float row_values[16];
    threadgroup uint row_indices[16];

    uint rows_per_tg = args.rows_per_tg == 0u ? (uint)NMOE_ROWS_PER_TG : args.rows_per_tg;
    rows_per_tg = min(rows_per_tg, 16u);
    uint threads_per_tg = rows_per_tg * 32u;
    uint in_dim = min(args.in_dim, (uint)NMOE_MAX_MATVEC_IN_DIM);
    for (uint i = lid; i < in_dim; i += threads_per_tg) {
        x_shared[i] = input[i];
    }
    if (lid < 16u) {
        row_values[lid] = -3.402823466e38f;
        row_indices[lid] = 0xffffffffu;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_group < rows_per_tg) {
        uint row = tgid * rows_per_tg + simd_group;
        float acc = 0.0f;
        if (row < args.vocab_size && args.group_size != 0u) {
            uint scale_groups = args.in_dim / args.group_size;
            const device uint *weight_row = packed_weight + row * args.packed_cols;
            const device ushort *scale_row = scales + row * scale_groups;
            const device ushort *bias_row = biases + row * scale_groups;
            for (uint pi = simd_lane; pi < args.packed_cols; pi += 32u) {
                uint word = weight_row[pi];
                uint base_col = pi * 8u;
                uint group = base_col / args.group_size;
                float scale = nmoe_bf16_to_f32(scale_row[group]);
                float bias = nmoe_bf16_to_f32(bias_row[group]);
                for (uint lane = 0u; lane < 8u; ++lane) {
                    uint col = base_col + lane;
                    if (col >= args.in_dim) break;
                    float x = x_shared[col];
                    acc += fma((float)((word >> (lane * 4u)) & 0xFu), scale, bias) * x;
                }
            }
            acc = simd_sum(acc);
            if (simd_lane == 0u) {
                row_values[simd_group] = acc;
                row_indices[simd_group] = row;
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (lid == 0u) {
        float best_value = -3.402823466e38f;
        uint best_index = 0u;
        for (uint i = 0u; i < rows_per_tg; ++i) {
            uint idx = row_indices[i];
            float value = row_values[i];
            if (idx != 0xffffffffu &&
                (value > best_value || (value == best_value && idx < best_index))) {
                best_value = value;
                best_index = idx;
            }
        }
        partial_values[tgid] = best_value;
        partial_indices[tgid] = best_index;
    }
}

kernel void nmoe_lm_head_argmax_reduce(
    const device float *partial_values [[buffer(0)]],
    const device uint *partial_indices [[buffer(1)]],
    device uint *result [[buffer(2)]],
    constant NMOELmHeadArgmaxArgs &args [[buffer(3)]],
    uint tid [[thread_index_in_threadgroup]])
{
    threadgroup float best_values[256];
    threadgroup uint best_indices[256];

    float best_value = -3.402823466e38f;
    uint best_index = 0u;
    for (uint i = tid; i < args.partial_count; i += 256u) {
        float value = partial_values[i];
        uint idx = partial_indices[i];
        if (value > best_value || (value == best_value && idx < best_index)) {
            best_value = value;
            best_index = idx;
        }
    }
    best_values[tid] = best_value;
    best_indices[tid] = best_index;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = 128u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            float other_value = best_values[tid + stride];
            uint other_index = best_indices[tid + stride];
            if (other_value > best_values[tid] ||
                (other_value == best_values[tid] && other_index < best_indices[tid])) {
                best_values[tid] = other_value;
                best_indices[tid] = other_index;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0u) result[0] = best_indices[0];
}

kernel void nmoe_route_topk(
    const device float *scores [[buffer(0)]],
    device uint *indices_out [[buffer(1)]],
    device float *weights_out [[buffer(2)]],
    constant NMOERouteTopKArgs &args [[buffer(3)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid != 0u || args.count == 0u) return;

    uint k = min(args.k, 8u);
    k = min(k, args.count);
    if (k == 0u) return;

    float maxv = scores[0];
    for (uint i = 1u; i < args.count; ++i) {
        maxv = max(maxv, scores[i]);
    }

    float selected[8];
    uint selected_indices[8];
    for (uint i = 0u; i < 8u; ++i) {
        selected[i] = -3.402823466e38f;
        selected_indices[i] = 0u;
    }

    float softmax_sum = 0.0f;
    for (uint i = 0u; i < args.count; ++i) {
        float value = exp(scores[i] - maxv);
        softmax_sum += value;

        for (uint slot = 0u; slot < k; ++slot) {
            if (value > selected[slot] ||
                (value == selected[slot] && i < selected_indices[slot])) {
                for (uint move = k - 1u; move > slot; --move) {
                    selected[move] = selected[move - 1u];
                    selected_indices[move] = selected_indices[move - 1u];
                }
                selected[slot] = value;
                selected_indices[slot] = i;
                break;
            }
        }
    }

    float top_sum = 0.0f;
    float inv_softmax_sum = softmax_sum > 0.0f ? 1.0f / softmax_sum : 0.0f;
    for (uint i = 0u; i < k; ++i) {
        selected[i] *= inv_softmax_sum;
        top_sum += selected[i];
    }

    float inv_top_sum = top_sum > 0.0f ? 1.0f / top_sum : 1.0f / (float)k;
    for (uint i = 0u; i < 8u; ++i) {
        if (i < k) {
            indices_out[i] = selected_indices[i];
            weights_out[i] = top_sum > 0.0f ? selected[i] * inv_top_sum : inv_top_sum;
        } else {
            indices_out[i] = 0u;
            weights_out[i] = 0.0f;
        }
    }
}

// Per-head bare RMS normalize for q and k (linear attention).
// Applies inv_scale^2 to q and inv_scale to k (TUI-compatible scaling).
kernel void nmoe_rms_norm_qk(
    device float *q [[buffer(0)]],
    device float *k [[buffer(1)]],
    constant uint &key_dim [[buffer(2)]],
    constant float &inv_scale [[buffer(3)]],
    constant float &eps [[buffer(4)]],
    uint head [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]])
{
    uint base = head * key_dim;

    // RMS norm for q (bare norm, then scale with inv_scale^2)
    threadgroup float q_partial[128];
    float qval = (tid < key_dim) ? q[base + tid] : 0.0f;
    q_partial[tid] = qval * qval;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float s = 0.0f;
        for (uint i = 0; i < key_dim; i++) s += q_partial[i];
        q_partial[0] = s;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float q_inv_rms = rsqrt(q_partial[0] / (float)key_dim + eps);
    if (tid < key_dim) q[base + tid] = qval * q_inv_rms * inv_scale * inv_scale;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // RMS norm for k (bare norm, then scale with inv_scale)
    threadgroup float k_partial[128];
    float kval = (tid < key_dim) ? k[base + tid] : 0.0f;
    k_partial[tid] = kval * kval;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float s = 0.0f;
        for (uint i = 0; i < key_dim; i++) s += k_partial[i];
        k_partial[0] = s;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float k_inv_rms = rsqrt(k_partial[0] / (float)key_dim + eps);
    if (tid < key_dim) k[base + tid] = kval * k_inv_rms * inv_scale;
}

// Compute g_decay and beta_gate for GatedDeltaNet
kernel void nmoe_compute_decay_beta(
    const device float *alpha_out [[buffer(0)]],
    const device float *beta_out [[buffer(1)]],
    const device float *A_log [[buffer(2)]],
    const device ushort *dt_bias [[buffer(3)]],
    device float *g_decay [[buffer(4)]],
    device float *beta_gate [[buffer(5)]],
    uint idx [[thread_position_in_grid]])
{
    float a_val = alpha_out[idx];
    float dt_b = nmoe_bf16_to_f32(dt_bias[idx]);
    float A_val = exp(A_log[idx]);
    float softplus_val = log(1.0f + exp(a_val + dt_b));
    g_decay[idx] = exp(-A_val * softplus_val);
    beta_gate[idx] = nmoe_sigmoid(beta_out[idx]);
}

// Gated RMS norm (z-gated output normalization)
kernel void nmoe_gated_rms_norm(
    const device float *values [[buffer(0)]],
    const device float *z [[buffer(1)]],
    const device ushort *norm_weight [[buffer(2)]],
    device float *output [[buffer(3)]],
    constant NMOEGatedDeltaNetArgs &args [[buffer(4)]],
    uint vh [[threadgroup_position_in_grid]],
    uint vi [[thread_position_in_threadgroup]])
{
    if (vh >= args.v_heads || vi >= args.value_dim) return;
    uint base = vh * args.value_dim;

    // RMS norm reduction within threadgroup
    threadgroup float partial[128];
    float val = values[base + vi];
    partial[vi] = val * val;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (vi == 0) {
        float s = 0.0f;
        for (uint i = 0; i < args.value_dim; i++) s += partial[i];
        partial[0] = s;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv_rms = rsqrt(partial[0] / (float)args.value_dim + args.epsilon);

    float normed = val * inv_rms;
    float zval = z[base + vi];
    float gate = nmoe_silu(zval);
    float w = nmoe_bf16_to_f32(norm_weight[vi]);
    output[base + vi] = normed * gate * w;
}
