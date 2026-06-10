#!/usr/bin/env python3
"""Prepare Qwen3.6-35B-A3B BF16 safetensors for nano-metal-moe.

This converts Hugging Face BF16 weights into the runtime format used by
`nmoe`:

  output/
    model_weights.bin/json       quantized non-expert weights + native vectors
    packed_experts/layer_XX.bin  split gate/up/down experts in 4-bit affine form
    tokenizer.bin                optional, if tokenizer.json exists
    vocab.bin                    optional, if tokenizer.json exists

The script streams large tensors in chunks, so it does not need to hold the full
model in RAM. It does need all 26 safetensors shards to be present.
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np


HIDDEN_DIM = 2048
NUM_LAYERS = 40
NUM_EXPERTS = 256
GROUP_SIZE = 64
MOE_INTERMEDIATE = 512
VOCAB_SIZE = 248320

EXPERT_SIZE_4BIT = 1_769_472

EXPERT_LAYOUT = [
    ("gate_proj.weight", 0, 524_288),
    ("gate_proj.scales", 524_288, 32_768),
    ("gate_proj.biases", 557_056, 32_768),
    ("up_proj.weight", 589_824, 524_288),
    ("up_proj.scales", 1_114_112, 32_768),
    ("up_proj.biases", 1_146_880, 32_768),
    ("down_proj.weight", 1_179_648, 524_288),
    ("down_proj.scales", 1_703_936, 32_768),
    ("down_proj.biases", 1_736_704, 32_768),
]

EXPERT_SIZE_2BIT = 983_040

EXPERT_LAYOUT_2BIT = [
    ("gate_proj.weight", 0, 262_144),
    ("gate_proj.scales", 262_144, 32_768),
    ("gate_proj.biases", 294_912, 32_768),
    ("up_proj.weight", 327_680, 262_144),
    ("up_proj.scales", 589_824, 32_768),
    ("up_proj.biases", 622_592, 32_768),
    ("down_proj.weight", 655_360, 262_144),
    ("down_proj.scales", 917_504, 32_768),
    ("down_proj.biases", 950_272, 32_768),
]


class TensorRef:
    def __init__(self, path: Path, data_start: int, name: str, meta: dict):
        self.path = path
        self.data_start = data_start
        self.name = name
        self.dtype = meta["dtype"]
        self.shape = tuple(meta["shape"])
        self.start, self.end = meta["data_offsets"]

    @property
    def nbytes(self) -> int:
        return self.end - self.start


def parse_safetensors_header(path: Path) -> Tuple[dict, int]:
    with path.open("rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_len))
    return header, 8 + header_len


def scan_model(model_dir: Path) -> Dict[str, TensorRef]:
    tensors: Dict[str, TensorRef] = {}
    shards = sorted(model_dir.glob("model-*.safetensors"))
    if not shards:
        raise SystemExit(f"ERROR: no model-*.safetensors files found in {model_dir}")

    for shard in shards:
        header, data_start = parse_safetensors_header(shard)
        for name, meta in header.items():
            if name == "__metadata__":
                continue
            tensors[name] = TensorRef(shard, data_start, name, meta)
    return tensors


def sanitize_name(name: str) -> str:
    if name.startswith("model.language_model."):
        name = "model." + name[len("model.language_model."):]
    elif name.startswith("language_model."):
        name = name[len("language_model."):]
    if name == "model.lm_head.weight":
        name = "lm_head.weight"
    return name


def is_language_tensor(name: str) -> bool:
    return name.startswith("model.language_model.") or name.startswith("lm_head.")


def is_expert_tensor(name: str) -> bool:
    return ".mlp.experts." in name


def is_visual_tensor(name: str) -> bool:
    return name.startswith("model.visual.") or name.startswith("vision_tower.")


def read_tensor(ref: TensorRef) -> np.ndarray:
    if ref.dtype != "BF16":
        raise ValueError(f"{ref.name}: expected BF16, got {ref.dtype}")
    with ref.path.open("rb") as f:
        f.seek(ref.data_start + ref.start)
        raw = f.read(ref.nbytes)
    return np.frombuffer(raw, dtype="<u2").copy().reshape(ref.shape)


def read_bf16_rows(ref: TensorRef, row_start: int, row_count: int) -> np.ndarray:
    if ref.dtype != "BF16" or len(ref.shape) != 2:
        raise ValueError(f"{ref.name}: expected 2D BF16")
    rows, cols = ref.shape
    if row_start < 0 or row_start + row_count > rows:
        raise ValueError(f"{ref.name}: row slice out of range")
    row_bytes = cols * 2
    with ref.path.open("rb") as f:
        f.seek(ref.data_start + ref.start + row_start * row_bytes)
        raw = f.read(row_count * row_bytes)
    return np.frombuffer(raw, dtype="<u2").copy().reshape(row_count, cols)


def bf16_to_f32(x: np.ndarray) -> np.ndarray:
    return (x.astype(np.uint32) << 16).view(np.float32)


def f32_to_bf16(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    u = x.view(np.uint32)
    rounded = u + 0x7FFF + ((u >> 16) & 1)
    return (rounded >> 16).astype(np.uint16)


def pack_2bit(vals: np.ndarray) -> np.ndarray:
    assert vals.shape[-1] % 16 == 0
    flat = vals.reshape(-1, vals.shape[-1])
    out = np.zeros((flat.shape[0], flat.shape[1] // 16), dtype=np.uint32)
    for i in range(16):
        out |= flat[:, i::16].astype(np.uint32) << (i * 2)
    return out.reshape(vals.shape[:-1] + (vals.shape[-1] // 16,))


def quantize_rows_bf16_2bit(rows_bf16: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows_f32 = bf16_to_f32(rows_bf16)
    rows, in_dim = rows_f32.shape
    if in_dim % GROUP_SIZE != 0:
        raise ValueError(f"in_dim {in_dim} is not divisible by group size {GROUP_SIZE}")
    groups = in_dim // GROUP_SIZE
    grouped = rows_f32.reshape(rows, groups, GROUP_SIZE)
    mn = grouped.min(axis=2, keepdims=True)
    mx = grouped.max(axis=2, keepdims=True)
    scale = (mx - mn) / 3.0
    safe_scale = np.where(scale == 0.0, 1.0, scale)
    q = np.rint((grouped - mn) / safe_scale)
    q = np.clip(q, 0, 3).astype(np.uint8).reshape(rows, in_dim)
    packed = pack_2bit(q)
    scales = f32_to_bf16(scale.squeeze(2))
    biases = f32_to_bf16(mn.squeeze(2))
    return packed, scales, biases


def pack_4bit(vals: np.ndarray) -> np.ndarray:
    assert vals.shape[-1] % 8 == 0
    flat = vals.reshape(-1, vals.shape[-1])
    out = np.zeros((flat.shape[0], flat.shape[1] // 8), dtype=np.uint32)
    for i in range(8):
        out |= flat[:, i::8].astype(np.uint32) << (i * 4)
    return out.reshape(vals.shape[:-1] + (vals.shape[-1] // 8,))


def quantize_rows_bf16(rows_bf16: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows_f32 = bf16_to_f32(rows_bf16)
    rows, in_dim = rows_f32.shape
    if in_dim % GROUP_SIZE != 0:
        raise ValueError(f"in_dim {in_dim} is not divisible by group size {GROUP_SIZE}")
    groups = in_dim // GROUP_SIZE
    grouped = rows_f32.reshape(rows, groups, GROUP_SIZE)
    mn = grouped.min(axis=2, keepdims=True)
    mx = grouped.max(axis=2, keepdims=True)
    scale = (mx - mn) / 15.0
    safe_scale = np.where(scale == 0.0, 1.0, scale)
    q = np.rint((grouped - mn) / safe_scale)
    q = np.clip(q, 0, 15).astype(np.uint8).reshape(rows, in_dim)
    packed = pack_4bit(q)
    scales = f32_to_bf16(scale.squeeze(2))
    biases = f32_to_bf16(mn.squeeze(2))
    return packed, scales, biases


def align_file(f, offset: int, align: int = 64) -> int:
    pad = (-offset) % align
    if pad:
        f.write(b"\0" * pad)
        offset += pad
    return offset


def write_blob(f, manifest: dict, name: str, data: np.ndarray, dtype: str, offset: int) -> int:
    offset = align_file(f, offset)
    raw = data.tobytes(order="C")
    f.write(raw)
    manifest["tensors"][name] = {
        "offset": offset,
        "size": len(raw),
        "shape": list(data.shape),
        "dtype": dtype,
    }
    return offset + len(raw)


def quantize_matrix_to_weights(
    ref: TensorRef,
    out_f,
    manifest: dict,
    base_name: str,
    offset: int,
    row_chunk: int,
) -> int:
    rows, cols = ref.shape
    if cols % GROUP_SIZE != 0:
        raise ValueError(f"{ref.name}: cols={cols} not divisible by {GROUP_SIZE}")

    packed_parts: List[np.ndarray] = []
    scale_parts: List[np.ndarray] = []
    bias_parts: List[np.ndarray] = []
    for r0 in range(0, rows, row_chunk):
        n = min(row_chunk, rows - r0)
        packed, scales, biases = quantize_rows_bf16(read_bf16_rows(ref, r0, n))
        packed_parts.append(packed)
        scale_parts.append(scales)
        bias_parts.append(biases)

    packed_all = np.concatenate(packed_parts, axis=0)
    scales_all = np.concatenate(scale_parts, axis=0)
    biases_all = np.concatenate(bias_parts, axis=0)
    offset = write_blob(out_f, manifest, base_name + ".weight", packed_all, "U32", offset)
    offset = write_blob(out_f, manifest, base_name + ".scales", scales_all, "BF16", offset)
    offset = write_blob(out_f, manifest, base_name + ".biases", biases_all, "BF16", offset)
    return offset


def export_model_weights(tensors: Dict[str, TensorRef], output_dir: Path, row_chunk: int) -> None:
    bin_path = output_dir / "model_weights.bin"
    json_path = output_dir / "model_weights.json"
    manifest = {
        "model": "Qwen3.6-35B-A3B",
        "num_tensors": 0,
        "config": {
            "hidden_size": HIDDEN_DIM,
            "num_hidden_layers": NUM_LAYERS,
            "num_attention_heads": 16,
            "num_key_value_heads": 2,
            "head_dim": 256,
            "vocab_size": VOCAB_SIZE,
            "rms_norm_eps": 1e-6,
            "num_experts": NUM_EXPERTS,
            "num_experts_per_tok": 8,
            "moe_intermediate_size": MOE_INTERMEDIATE,
            "shared_expert_intermediate_size": MOE_INTERMEDIATE,
            "full_attention_interval": 4,
            "linear_num_value_heads": 32,
            "linear_num_key_heads": 16,
            "linear_key_head_dim": 128,
            "linear_value_head_dim": 128,
            "linear_conv_kernel_dim": 4,
            "partial_rotary_factor": 0.25,
            "rope_theta": 10000000.0,
        },
        "tensors": {},
    }

    selected = []
    for name, ref in tensors.items():
        if is_visual_tensor(name) or is_expert_tensor(name) or not is_language_tensor(name):
            continue
        san = sanitize_name(name)
        selected.append((san, ref))
    selected.sort(key=lambda x: x[0])

    print(f"[weights] exporting {len(selected)} non-expert tensors")
    offset = 0
    t0 = time.time()
    with bin_path.open("wb") as out_f:
        for i, (san, ref) in enumerate(selected, 1):
            if ref.dtype != "BF16":
                raise ValueError(f"{ref.name}: unsupported dtype {ref.dtype}")

            if len(ref.shape) == 2 and san.endswith(".weight"):
                base = san[:-len(".weight")]
                offset = quantize_matrix_to_weights(ref, out_f, manifest, base, offset, row_chunk)
            elif san.endswith(".linear_attn.A_log"):
                data = bf16_to_f32(read_tensor(ref))
                offset = write_blob(out_f, manifest, san, data, "F32", offset)
            else:
                data = read_tensor(ref)
                offset = write_blob(out_f, manifest, san, data, "BF16", offset)

            if i % 25 == 0 or i == len(selected):
                print(f"  [{i:4d}/{len(selected)}] {offset/1e9:.2f} GB written")

    manifest["num_tensors"] = len(manifest["tensors"])
    json_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[weights] wrote {bin_path} and {json_path} in {time.time()-t0:.1f}s")


def find_required(tensors: Dict[str, TensorRef], name: str) -> TensorRef:
    if name not in tensors:
        raise KeyError(name)
    return tensors[name]


def write_expert_component(blob: bytearray, offset: int, arr: np.ndarray) -> None:
    raw = arr.tobytes(order="C")
    blob[offset:offset + len(raw)] = raw


def export_packed_experts(
    tensors: Dict[str, TensorRef],
    output_dir: Path,
    layers: Iterable[int],
    bits: int = 4,
) -> None:
    expert_dir = output_dir / ("packed_experts" if bits == 4 else "packed_experts_2bit")
    expert_dir.mkdir(parents=True, exist_ok=True)
    size = EXPERT_SIZE_4BIT if bits == 4 else EXPERT_SIZE_2BIT
    layout_data = EXPERT_LAYOUT if bits == 4 else EXPERT_LAYOUT_2BIT
    layout = {
        "expert_size": size,
        "num_layers": NUM_LAYERS,
        "num_experts": NUM_EXPERTS,
        "components": [
            {"name": name, "offset": off, "size": sz}
            for name, off, sz in layout_data
        ],
    }
    (expert_dir / "layout.json").write_text(json.dumps(layout, indent=2), encoding="utf-8")

    for layer in layers:
        gate_up_name = f"model.language_model.layers.{layer}.mlp.experts.gate_up_proj"
        down_name = f"model.language_model.layers.{layer}.mlp.experts.down_proj"
        gate_up = find_required(tensors, gate_up_name)
        down = find_required(tensors, down_name)
        if gate_up.shape != (NUM_EXPERTS, 2 * MOE_INTERMEDIATE, HIDDEN_DIM):
            raise ValueError(f"{gate_up_name}: unexpected shape {gate_up.shape}")
        if down.shape != (NUM_EXPERTS, HIDDEN_DIM, MOE_INTERMEDIATE):
            raise ValueError(f"{down_name}: unexpected shape {down.shape}")

        out_path = expert_dir / f"layer_{layer:02d}.bin"
        print(f"[experts] layer {layer:02d} -> {out_path}")
        t0 = time.time()
        with out_path.open("wb") as out:
            for expert in range(NUM_EXPERTS):
                gu = read_bf16_rows(
                    TensorRef(gate_up.path, gate_up.data_start, gate_up.name,
                              {"dtype": gate_up.dtype, "shape": (NUM_EXPERTS * 2 * MOE_INTERMEDIATE, HIDDEN_DIM),
                               "data_offsets": [gate_up.start, gate_up.end]}),
                    expert * 2 * MOE_INTERMEDIATE,
                    2 * MOE_INTERMEDIATE,
                )
                gate_bf16 = gu[:MOE_INTERMEDIATE]
                up_bf16 = gu[MOE_INTERMEDIATE:]

                d = read_bf16_rows(
                    TensorRef(down.path, down.data_start, down.name,
                              {"dtype": down.dtype, "shape": (NUM_EXPERTS * HIDDEN_DIM, MOE_INTERMEDIATE),
                               "data_offsets": [down.start, down.end]}),
                    expert * HIDDEN_DIM,
                    HIDDEN_DIM,
                )

                if bits == 4:
                    gate_w, gate_s, gate_b = quantize_rows_bf16(gate_bf16)
                    up_w, up_s, up_b = quantize_rows_bf16(up_bf16)
                    down_w, down_s, down_b = quantize_rows_bf16(d)
                else:
                    gate_w, gate_s, gate_b = quantize_rows_bf16_2bit(gate_bf16)
                    up_w, up_s, up_b = quantize_rows_bf16_2bit(up_bf16)
                    down_w, down_s, down_b = quantize_rows_bf16_2bit(d)

                blob = bytearray(size)
                for i, (w, s, b) in enumerate([(gate_w, gate_s, gate_b), (up_w, up_s, up_b), (down_w, down_s, down_b)]):
                    write_expert_component(blob, layout_data[i*3][1], w)
                    write_expert_component(blob, layout_data[i*3+1][1], s)
                    write_expert_component(blob, layout_data[i*3+2][1], b)
                out.write(blob)

                if (expert + 1) % 32 == 0 or expert == NUM_EXPERTS - 1:
                    print(f"  expert {expert+1:3d}/{NUM_EXPERTS}")
        print(f"  done in {time.time()-t0:.1f}s, size={out_path.stat().st_size/1e9:.2f} GB")


def build_byte_unicode_maps():
    byte_char = {}
    n = 0
    for b in range(256):
        if (0x21 <= b <= 0x7E) or (0xA1 <= b <= 0xAC) or (0xAE <= b <= 0xFF):
            byte_char[b] = b
        else:
            byte_char[b] = 256 + n
            n += 1
    char_byte = {cp: b for b, cp in byte_char.items()}
    return byte_char, char_byte


def decode_vocab_token(token: str, char_byte: dict) -> bytes:
    out = bytearray()
    for ch in token:
        cp = ord(ch)
        if cp in char_byte:
            out.append(char_byte[cp])
        else:
            out.extend(ch.encode("utf-8"))
    return bytes(out)


def split_merge_pair(pair):
    if isinstance(pair, str):
        parts = pair.split(" ", 1)
        if len(parts) != 2:
            raise ValueError(f"invalid merge rule: {pair!r}")
        return parts[0], parts[1]
    if len(pair) != 2:
        raise ValueError(f"invalid merge rule: {pair!r}")
    return pair[0], pair[1]


def export_tokenizer_files(model_dir: Path, output_dir: Path) -> None:
    tok_path = model_dir / "tokenizer.json"
    if not tok_path.exists():
        print(f"[tokenizer] {tok_path} not found; skipping tokenizer.bin/vocab.bin")
        return
    data = json.loads(tok_path.read_text(encoding="utf-8"))
    model = data["model"]
    vocab = model["vocab"]
    merges = model["merges"]
    added = data.get("added_tokens", [])

    tokenizer_bin = output_dir / "tokenizer.bin"
    with tokenizer_bin.open("wb") as f:
        f.write(b"BPET")
        f.write(struct.pack("<I", 1))
        f.write(struct.pack("<I", len(vocab)))
        f.write(struct.pack("<I", len(merges)))
        f.write(struct.pack("<I", len(added)))
        for token, token_id in sorted(vocab.items(), key=lambda kv: kv[1]):
            b = token.encode("utf-8")
            f.write(struct.pack("<I", token_id))
            f.write(struct.pack("<H", len(b)))
            f.write(b)
        for pair in merges:
            a, b = split_merge_pair(pair)
            ab, bb = a.encode("utf-8"), b.encode("utf-8")
            f.write(struct.pack("<H", len(ab))); f.write(ab)
            f.write(struct.pack("<H", len(bb))); f.write(bb)
        for tok in added:
            b = tok["content"].encode("utf-8")
            f.write(struct.pack("<I", tok["id"]))
            f.write(struct.pack("<H", len(b)))
            f.write(b)

    id_to_token = {token_id: token for token, token_id in vocab.items()}
    for tok in added:
        id_to_token[tok["id"]] = tok["content"]
    max_id = max(id_to_token)
    _, char_byte = build_byte_unicode_maps()
    vocab_bin = output_dir / "vocab.bin"
    with vocab_bin.open("wb") as f:
        f.write(struct.pack("<II", max_id + 1, max_id))
        for token_id in range(max_id + 1):
            token = id_to_token.get(token_id, "")
            if token.startswith("<|") and token.endswith("|>"):
                b = token.encode("utf-8")
            else:
                b = decode_vocab_token(token, char_byte)
            f.write(struct.pack("<H", min(len(b), 65535)))
            f.write(b[:65535])
    print(f"[tokenizer] wrote {tokenizer_bin} and {vocab_bin}")


def parse_layers(spec: str | None) -> List[int]:
    if not spec or spec == "all":
        return list(range(NUM_LAYERS))
    out = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return sorted(set(out))


def check_download_complete(model_dir: Path) -> bool:
    shards = sorted(model_dir.glob("model-*.safetensors"))
    complete = True
    if len(shards) != 26:
        print(f"[check] WARNING: found {len(shards)}/26 safetensors shards; download is not complete yet")
        complete = False
    missing = [i for i in range(1, 27) if not (model_dir / f"model-{i:05d}-of-00026.safetensors").exists()]
    if missing:
        print(f"[check] missing shards: {missing}")
        complete = False
    return complete


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", "--model-dir", dest="model", default="../model/Qwen3.6-35B-A3B",
                    help="Path to the downloaded Hugging Face model directory")
    ap.add_argument("--output", "--output-root", dest="output", default="qwen36_35b",
                    help="Output directory for nmoe runtime assets")
    ap.add_argument("--layers", default="all",
                    help='Expert layers to pack, e.g. "0", "0-3", or "all"')
    ap.add_argument("--row-chunk", type=int, default=4096,
                    help="Rows per quantization chunk for non-expert matrices")
    ap.add_argument("--skip-weights", action="store_true")
    ap.add_argument("--skip-experts", action="store_true")
    ap.add_argument("--skip-tokenizer", action="store_true")
    ap.add_argument("--allow-partial", action="store_true",
                    help="Allow export even when not all 26 shards are present")
    ap.add_argument("--bits", type=int, choices=[2, 4], default=4,
                    help="Quantization bits for experts (2 or 4)")
    ap.add_argument("--quant", choices=["2", "4"],
                    help="Compatibility alias for --bits")
    args = ap.parse_args()
    if args.quant is not None:
        args.bits = int(args.quant)

    model_dir = Path(args.model)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    complete = check_download_complete(model_dir)
    if not complete and not args.allow_partial and not (args.skip_weights and args.skip_experts):
        raise SystemExit("ERROR: download is incomplete; wait for all 26 shards or pass --allow-partial for debugging")
    print(f"[scan] scanning {model_dir}")
    tensors = scan_model(model_dir)
    print(f"[scan] {len(tensors)} tensors found")

    if not args.skip_weights:
        export_model_weights(tensors, output_dir, args.row_chunk)
    if not args.skip_experts:
        export_packed_experts(tensors, output_dir, parse_layers(args.layers), bits=args.bits)
    if not args.skip_tokenizer:
        export_tokenizer_files(model_dir, output_dir)


if __name__ == "__main__":
    main()
