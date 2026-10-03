#!/usr/bin/env python3
"""Requantize the q4 expert pack to fewer bits.

The q4 pack is MLX-style affine quantization (group 64, bf16 scale/bias,
w = scale * q + bias, 8 nibbles per u32, low nibble first). This script
dequantizes each group and re-fits a lower-bit affine grid with a clip search
plus a least-squares refit of (scale, bias), which is noticeably more accurate
than plain min/max rounding.

  --container q4    write the requantized values back into the q4 layout
                    ("fake quant"): same file size, runs on the existing q4
                    kernels, used to measure the accuracy cost of fewer bits.
  --container q3    write the native packed q3 layout (see layout.json).

Usage:
  python3 scripts/requant_experts.py --src qwen36_35b/packed_experts \\
      --dst /path/packed_experts_fake3 --bits 3 --container q4
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

HIDDEN = 2048
INTER = 512
GROUP = 64
NUM_EXPERTS = 256
NUM_LAYERS = 40

# q4 per-expert layout: (name, rows, cols)
MATS = [("gate_proj", INTER, HIDDEN), ("up_proj", INTER, HIDDEN), ("down_proj", HIDDEN, INTER)]


def q4_layout():
    off = 0
    comps = []
    for name, rows, cols in MATS:
        for part, size in (("weight", rows * cols // 2), ("scales", rows * cols // GROUP * 2), ("biases", rows * cols // GROUP * 2)):
            comps.append((f"{name}.{part}", off, size))
            off += size
    return comps, off


def q3_layout():
    # Planar 3-bit: 32 values per 3 u32 words (see pack_q3), 12 bytes per 32 values.
    off = 0
    comps = []
    for name, rows, cols in MATS:
        for part, size in (("weight", rows * cols * 3 // 8), ("scales", rows * cols // GROUP * 2), ("biases", rows * cols // GROUP * 2)):
            comps.append((f"{name}.{part}", off, size))
            off += size
    # Round expert size up to 16KB pages so experts start page-aligned.
    page = 16384
    padded = (off + page - 1) // page * page
    return comps, off, padded


def bf16_to_f32(u16):
    return (u16.astype(np.uint32) << 16).view(np.float32)


def f32_to_bf16(f):
    # Round to nearest even.
    u = f.astype(np.float32).view(np.uint32)
    rounding = ((u >> 16) & 1) + 0x7FFF
    return ((u + rounding) >> 16).astype(np.uint16)


def dequant_q4(blob, comps_by_name, name, rows, cols, n):
    """blob: (n, expert_size) uint8 -> torch float32 (n, rows, cols)."""
    _, woff, wsize = comps_by_name[f"{name}.weight"]
    _, soff, ssize = comps_by_name[f"{name}.scales"]
    _, boff, bsize = comps_by_name[f"{name}.biases"]
    words = np.ascontiguousarray(blob[:, woff:woff + wsize]).view(np.uint32).reshape(n, rows, cols // 8)
    shifts = np.arange(8, dtype=np.uint32) * 4
    q = ((words[..., None] >> shifts) & 0xF).reshape(n, rows, cols).astype(np.float32)
    s = bf16_to_f32(np.ascontiguousarray(blob[:, soff:soff + ssize]).view(np.uint16)).reshape(n, rows, cols // GROUP)
    b = bf16_to_f32(np.ascontiguousarray(blob[:, boff:boff + bsize]).view(np.uint16)).reshape(n, rows, cols // GROUP)
    w = q.reshape(n, rows, cols // GROUP, GROUP) * s[..., None] + b[..., None]
    return torch.from_numpy(w.reshape(n, rows, cols))


def fit_affine(w, bits, device):
    """w: (..., GROUP) float32. Returns q (uint8), scale, bias (bf16-rounded f32)."""
    levels = (1 << bits) - 1
    x = w.to(device)
    mn = x.amin(-1, keepdim=True)
    mx = x.amax(-1, keepdim=True)
    best_err = None
    best = None
    for shrink in (1.0, 0.95, 0.9, 0.85, 0.8, 0.75):
        center = (mx + mn) * 0.5
        half = (mx - mn) * 0.5 * shrink
        lo = center - half
        scale = (2 * half / levels).clamp_min(1e-10)
        bias = lo
        for _ in range(2):
            q = ((x - bias) / scale).round().clamp(0, levels)
            # Least-squares refit of scale/bias for the fixed assignment q.
            qm = q.mean(-1, keepdim=True)
            xm = x.mean(-1, keepdim=True)
            var = ((q - qm) ** 2).sum(-1, keepdim=True)
            cov = ((q - qm) * (x - xm)).sum(-1, keepdim=True)
            new_scale = torch.where(var > 0, cov / var.clamp_min(1e-12), scale)
            new_scale = new_scale.clamp_min(1e-10)
            bias = xm - new_scale * qm
            scale = new_scale
        # Round scale/bias to bf16 as stored, then final assignment.
        scale = scale.to(torch.bfloat16).to(torch.float32).clamp_min(1e-10)
        bias = bias.to(torch.bfloat16).to(torch.float32)
        q = ((x - bias) / scale).round().clamp(0, levels)
        err = ((q * scale + bias - x) ** 2).sum(-1, keepdim=True)
        if best_err is None:
            best_err, best = err, (q, scale, bias)
        else:
            better = err < best_err
            best_err = torch.where(better, err, best_err)
            best = tuple(torch.where(better, new, old) for new, old in zip((q, scale, bias), best))
    q, scale, bias = best
    return q.to(torch.uint8).cpu().numpy(), scale.cpu().numpy(), bias.cpu().numpy()


def pack_q4(q):
    """q: (..., cols) uint8 in [0,15] -> uint32 words, low nibble first."""
    q = q.astype(np.uint32).reshape(*q.shape[:-1], -1, 8)
    shifts = np.arange(8, dtype=np.uint32) * 4
    return (q << shifts).sum(-1, dtype=np.uint32)


def pack_q3(q):
    """q: (..., cols) uint8 in [0,7] -> planar q3, 3 uint32 words per 32 values:
    word0 = low 2 bits of values 0..15, word1 = low 2 bits of values 16..31
    (2 bits each, LSB first), word2 = high bit of values 0..31 (bit i = value i).
    A 16-value half-block h (0/1) decodes from word h plus bits [16h, 16h+16) of word2."""
    q = q.astype(np.uint32).reshape(*q.shape[:-1], -1, 32)
    lo = q & 3
    hi = q >> 2
    w0 = (lo[..., :16] << (np.arange(16, dtype=np.uint32) * 2)).sum(-1, dtype=np.uint32)
    w1 = (lo[..., 16:] << (np.arange(16, dtype=np.uint32) * 2)).sum(-1, dtype=np.uint32)
    w2 = (hi << np.arange(32, dtype=np.uint32)).sum(-1, dtype=np.uint32)
    return np.stack([w0, w1, w2], -1).astype(np.uint32).reshape(*q.shape[:-2], -1)


def unpack_q3(words, cols):
    w = words.reshape(*words.shape[:-1], -1, 3)
    i16 = np.arange(16, dtype=np.uint32)
    lo = np.concatenate([(w[..., 0:1] >> (i16 * 2)) & 3, (w[..., 1:2] >> (i16 * 2)) & 3], -1)
    hi = (w[..., 2:3] >> np.arange(32, dtype=np.uint32)) & 1
    return (lo | (hi << 2)).reshape(*words.shape[:-1], cols).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--bits", type=int, choices=[2, 3], required=True)
    ap.add_argument("--container", choices=["q4", "q3"], default="q4")
    ap.add_argument("--layers", default="")
    ap.add_argument("--chunk", type=int, default=16)
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    args = ap.parse_args()
    if args.container == "q3" and args.bits != 3:
        sys.exit("--container q3 requires --bits 3")

    comps, esz = q4_layout()
    assert esz == 1769472, esz
    by_name = {c[0]: c for c in comps}
    if args.container == "q3":
        out_comps, out_raw, out_esz = q3_layout()
    else:
        out_comps, out_esz = comps, esz
    out_by_name = {c[0]: c for c in out_comps}

    os.makedirs(args.dst, exist_ok=True)
    layers = [int(x) for x in args.layers.split(",")] if args.layers else range(NUM_LAYERS)
    for layer in layers:
        src = os.path.join(args.src, f"layer_{layer:02d}.bin")
        dst = os.path.join(args.dst, f"layer_{layer:02d}.bin")
        t0 = time.time()
        sq_err = 0.0
        sq_ref = 0.0
        with open(src, "rb") as fin, open(dst + ".tmp", "wb") as fout:
            for e0 in range(0, NUM_EXPERTS, args.chunk):
                n = min(args.chunk, NUM_EXPERTS - e0)
                blob = np.frombuffer(fin.read(n * esz), dtype=np.uint8).reshape(n, esz)
                out = np.zeros((n, out_esz), dtype=np.uint8)
                for name, rows, cols in MATS:
                    w = dequant_q4(blob, by_name, name, rows, cols, n)
                    g = w.reshape(n, rows, cols // GROUP, GROUP)
                    q, s, b = fit_affine(g, args.bits, args.device)
                    recon = q.astype(np.float32) * s + b
                    sq_err += float(((recon - g.numpy()) ** 2).sum())
                    sq_ref += float((g.numpy() ** 2).sum())
                    q = q.reshape(n, rows, cols)
                    packed = pack_q4(q) if args.container == "q4" else pack_q3(q)
                    for part, arr in (("weight", packed), ("scales", f32_to_bf16(s[..., 0])), ("biases", f32_to_bf16(b[..., 0]))):
                        _, off, size = out_by_name[f"{name}.{part}"]
                        raw = np.ascontiguousarray(arr).reshape(n, -1).view(np.uint8)
                        assert raw.shape[1] == size, (name, part, raw.shape, size)
                        out[:, off:off + size] = raw
                fout.write(out.tobytes())
        os.replace(dst + ".tmp", dst)
        print(f"layer {layer:02d}: rel_err_vs_q4={np.sqrt(sq_err / sq_ref):.4f} {time.time() - t0:.1f}s", flush=True)

    layout = {
        "expert_size": out_esz,
        "num_layers": NUM_LAYERS,
        "num_experts": NUM_EXPERTS,
        "bits": 4 if args.container == "q4" else 3,
        "source_bits": args.bits,
        "components": [{"name": n, "offset": o, "size": s} for n, o, s in out_comps],
    }
    with open(os.path.join(args.dst, "layout.json"), "w") as f:
        json.dump(layout, f, indent=2)


if __name__ == "__main__":
    main()
