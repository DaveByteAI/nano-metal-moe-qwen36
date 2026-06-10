#!/usr/bin/env python3
"""Convert Qwen3.6 routed experts into the packed layout used by nmoe."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np
from safetensors.numpy import safe_open


NUM_LAYERS = 40
NUM_EXPERTS = 256
GROUP_SIZE = 64
GATE_ROWS = 512
UP_ROWS = 512
DOWN_ROWS = 2048
GATE_COLS = 2048
UP_COLS = 2048
DOWN_COLS = 512
Q4_BITS = 4
Q2_BITS = 2

ROLE_SHAPES = {
    "gate_proj": (GATE_ROWS, GATE_COLS),
    "up_proj": (UP_ROWS, UP_COLS),
    "down_proj": (DOWN_ROWS, DOWN_COLS),
}

ROLE_ALIASES = {
    "gate_proj": ("gate_proj", "w1", "gate", "w_gate"),
    "up_proj": ("up_proj", "w3", "up", "w_up"),
    "down_proj": ("down_proj", "w2", "down", "w_down"),
}


@dataclass(frozen=True)
class ComponentLayout:
    name: str
    offset: int
    size: int
    dtype: str
    shape: Tuple[int, ...]
    logical_shape: Tuple[int, ...]


@dataclass(frozen=True)
class ExpertLayout:
    quant_bits: int
    num_experts: int
    expert_size: int
    components: Tuple[ComponentLayout, ...]

    @property
    def output_dir_name(self) -> str:
        return "packed_experts" if self.quant_bits == 4 else "packed_experts_2bit"

    def component(self, name: str) -> ComponentLayout:
        for component in self.components:
            if component.name == name:
                return component
        raise KeyError(name)


def _dtype_name(dtype: np.dtype) -> str:
    if dtype == np.uint32:
        return "u32"
    if dtype == np.uint16:
        return "u16"
    if dtype == np.float32:
        return "f32"
    raise ValueError(f"unsupported dtype {dtype!r}")


def f32_to_bf16(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    bits = arr.view(np.uint32)
    rounded = bits + np.uint32(0x7FFF) + ((bits >> np.uint32(16)) & np.uint32(1))
    return (rounded >> np.uint32(16)).astype(np.uint16)


def bf16_to_f32(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values)
    if arr.dtype == np.float32:
        return arr.astype(np.float32, copy=False)
    if str(arr.dtype) == "bfloat16":
        return arr.astype(np.float32, copy=False)
    if arr.dtype != np.uint16:
        arr = arr.astype(np.uint16, copy=False)
    return (arr.astype(np.uint32) << np.uint32(16)).view(np.float32)


def _pack_words(values: np.ndarray, bits: int) -> np.ndarray:
    values_per_word = 32 // bits
    if values.shape[-1] != values_per_word:
        raise ValueError(
            f"expected {values_per_word} values per word for {bits}-bit packing, "
            f"got {values.shape[-1]}"
        )
    packed = np.zeros(values.shape[:-1], dtype=np.uint32)
    shifts = [np.uint32(i * bits) for i in range(values_per_word)]
    for idx, shift in enumerate(shifts):
        packed |= values[..., idx].astype(np.uint32) << shift
    return packed


def _unpack_words(values: np.ndarray, bits: int) -> np.ndarray:
    values_per_word = 32 // bits
    mask = np.uint32((1 << bits) - 1)
    words = np.asarray(values, dtype=np.uint32)
    unpacked = np.empty(words.shape + (values_per_word,), dtype=np.uint32)
    for idx in range(values_per_word):
        unpacked[..., idx] = (words >> np.uint32(idx * bits)) & mask
    return unpacked


def _quantize_rowwise(matrix: np.ndarray, bits: int, group_size: int = GROUP_SIZE) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if bits not in (Q4_BITS, Q2_BITS):
        raise ValueError(f"unsupported quantization bits: {bits}")

    arr = np.asarray(matrix, dtype=np.float32, order="C")
    if arr.ndim != 2:
        raise ValueError(f"expected 2D matrix, got shape {arr.shape}")
    rows, cols = arr.shape
    if cols % group_size != 0:
        raise ValueError(f"column count {cols} is not divisible by group size {group_size}")

    values_per_word = 32 // bits
    if cols % values_per_word != 0:
        raise ValueError(f"column count {cols} is not divisible by {values_per_word}")

    groups = cols // group_size
    reshaped = arr.reshape(rows, groups, group_size)
    mn = reshaped.min(axis=-1)
    mx = reshaped.max(axis=-1)
    scale = (mx - mn) / float((1 << bits) - 1)
    safe_scale = np.where(scale == 0.0, 1.0, scale)

    q = np.rint((reshaped - mn[..., None]) / safe_scale[..., None]).astype(np.int32)
    np.clip(q, 0, (1 << bits) - 1, out=q)
    q = q.astype(np.uint32, copy=False)
    packed = _pack_words(q.reshape(rows, cols // values_per_word, values_per_word), bits)
    return packed, f32_to_bf16(scale), f32_to_bf16(mn)


def _dequantize_rowwise(
    packed: np.ndarray,
    scales_bf16: np.ndarray,
    biases_bf16: np.ndarray,
    bits: int,
    group_size: int = GROUP_SIZE,
) -> np.ndarray:
    if bits not in (Q4_BITS, Q2_BITS):
        raise ValueError(f"unsupported quantization bits: {bits}")

    packed = np.asarray(packed, dtype=np.uint32, order="C")
    rows, packed_cols = packed.shape
    values_per_word = 32 // bits
    if packed_cols % (group_size // values_per_word) != 0:
        raise ValueError("packed shape does not match group layout")

    unpacked = _unpack_words(packed, bits).reshape(rows, packed_cols * values_per_word)
    scale = bf16_to_f32(scales_bf16).astype(np.float32, copy=False)
    bias = bf16_to_f32(biases_bf16).astype(np.float32, copy=False)
    groups = scale.shape[1]
    dequantized = (unpacked.reshape(rows, groups, group_size).astype(np.float32) * scale[..., None]) + bias[..., None]
    return dequantized.reshape(rows, groups * group_size)


def layout_for_bits(bits: int, num_experts: int = NUM_EXPERTS) -> ExpertLayout:
    if bits not in (Q4_BITS, Q2_BITS):
        raise ValueError(f"unsupported quantization bits: {bits}")

    values_per_word = 32 // bits
    packed_gate_cols = GATE_COLS // values_per_word
    packed_down_cols = DOWN_COLS // values_per_word
    group_cols = GATE_COLS // GROUP_SIZE
    down_group_cols = DOWN_COLS // GROUP_SIZE

    components: List[ComponentLayout] = []
    offset = 0

    def add(name: str, rows: int, logical_cols: int, dtype: np.dtype, shape: Tuple[int, ...]) -> None:
        nonlocal offset
        size = int(np.prod(shape)) * np.dtype(dtype).itemsize
        components.append(
            ComponentLayout(
                name=name,
                offset=offset,
                size=size,
                dtype=_dtype_name(np.dtype(dtype)),
                shape=shape,
                logical_shape=(rows, logical_cols),
            )
        )
        offset += size

    add("gate_proj.weight", GATE_ROWS, GATE_COLS, np.uint32, (GATE_ROWS, packed_gate_cols))
    add("gate_proj.scales", GATE_ROWS, GATE_COLS, np.uint16, (GATE_ROWS, group_cols))
    add("gate_proj.biases", GATE_ROWS, GATE_COLS, np.uint16, (GATE_ROWS, group_cols))
    add("up_proj.weight", UP_ROWS, UP_COLS, np.uint32, (UP_ROWS, packed_gate_cols))
    add("up_proj.scales", UP_ROWS, UP_COLS, np.uint16, (UP_ROWS, group_cols))
    add("up_proj.biases", UP_ROWS, UP_COLS, np.uint16, (UP_ROWS, group_cols))
    add("down_proj.weight", DOWN_ROWS, DOWN_COLS, np.uint32, (DOWN_ROWS, packed_down_cols))
    add("down_proj.scales", DOWN_ROWS, DOWN_COLS, np.uint16, (DOWN_ROWS, down_group_cols))
    add("down_proj.biases", DOWN_ROWS, DOWN_COLS, np.uint16, (DOWN_ROWS, down_group_cols))

    return ExpertLayout(bits, num_experts, offset, tuple(components))


def layout_to_json(layout: ExpertLayout) -> Dict[str, object]:
    return {
        "schema_version": 1,
        "quant": f"q{layout.quant_bits}",
        "bits": layout.quant_bits,
        "group_size": GROUP_SIZE,
        "num_experts": layout.num_experts,
        "expert_size": layout.expert_size,
        "active_expert_size": layout.expert_size,
        "components": [
            {
                "name": c.name,
                "offset": c.offset,
                "size": c.size,
                "dtype": c.dtype,
                "shape": list(c.shape),
                "logical_shape": list(c.logical_shape),
            }
            for c in layout.components
        ],
    }


def write_layout_json(path: Path, layout: ExpertLayout) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(layout_to_json(layout), fh, indent=2, sort_keys=False)
        fh.write("\n")


def load_layout_json(path: Path) -> ExpertLayout:
    with Path(path).open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    bits = int(data.get("bits") or str(data.get("quant", "q4")).lstrip("q"))
    num_experts = int(data.get("num_experts", NUM_EXPERTS))
    components = []
    for item in data.get("components", []):
        shape = tuple(int(x) for x in item["shape"])
        logical_shape = tuple(int(x) for x in item.get("logical_shape", shape))
        components.append(
            ComponentLayout(
                name=str(item["name"]),
                offset=int(item["offset"]),
                size=int(item["size"]),
                dtype=str(item["dtype"]),
                shape=shape,
                logical_shape=logical_shape,
            )
        )
    expert_size = int(data.get("expert_size", sum(component.size for component in components)))
    return ExpertLayout(bits, num_experts, expert_size, tuple(components))


class TensorStore:
    def __init__(self, model_dir: Path):
        self.model_dir = Path(model_dir)
        self._handles: Dict[Path, object] = {}
        self._key_cache: Dict[Path, set] = {}
        self._weight_map: Dict[str, str] = self._load_weight_map()
        self._shards = self._discover_shards()

    def close(self) -> None:
        self._handles.clear()
        self._key_cache.clear()

    def __enter__(self) -> "TensorStore":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def _load_weight_map(self) -> Dict[str, str]:
        index_candidates = sorted(self.model_dir.glob("*.safetensors.index.json"))
        if not index_candidates:
            index_candidates = sorted(self.model_dir.glob("*.index.json"))
        if not index_candidates:
            return {}
        with index_candidates[0].open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        weight_map = data.get("weight_map", {})
        if not isinstance(weight_map, dict):
            raise TypeError(f"weight_map in {index_candidates[0]} is not a dict")
        return {str(k): str(v) for k, v in weight_map.items()}

    def _discover_shards(self) -> List[Path]:
        if self._weight_map:
            return sorted({self.model_dir / shard for shard in self._weight_map.values()})
        return sorted(self.model_dir.glob("*.safetensors"))

    def _open_handle(self, shard: Path):
        handle = self._handles.get(shard)
        if handle is not None:
            return handle
        try:
            handle = safe_open(str(shard), framework="numpy")
        except TypeError:
            handle = safe_open(str(shard), framework="np")
        self._handles[shard] = handle
        try:
            self._key_cache[shard] = set(handle.keys())
        except Exception:
            self._key_cache[shard] = set()
        return handle

    def _resolve_shard(self, tensor_name: str) -> Path:
        mapped = self._weight_map.get(tensor_name)
        if mapped is not None:
            shard = self.model_dir / mapped
            if not shard.exists():
                raise FileNotFoundError(shard)
            return shard
        for shard in self._shards:
            keys = self._key_cache.get(shard)
            if keys is not None and tensor_name in keys:
                return shard
            handle = self._open_handle(shard)
            keys = self._key_cache.get(shard, set())
            if tensor_name in keys:
                return shard
        raise KeyError(tensor_name)

    def get(self, tensor_name: str) -> np.ndarray:
        shard = self._resolve_shard(tensor_name)
        handle = self._open_handle(shard)
        array = handle.get_tensor(tensor_name)
        return np.asarray(array)


def _candidate_tensor_names(layer: int, expert: int, role: str) -> Iterator[str]:
    aliases = ROLE_ALIASES[role]
    prefixes = (
        f"model.layers.{layer}.mlp.experts.{expert}",
        f"model.layers.{layer}.block_sparse_moe.experts.{expert}",
        f"model.layers.{layer}.mlp.expert.{expert}",
    )
    for prefix in prefixes:
        for alias in aliases:
            yield f"{prefix}.{alias}.weight"
            yield f"{prefix}.{alias}"


def _load_expert_matrix(store: TensorStore, layer: int, expert: int, role: str) -> np.ndarray:
    expected = ROLE_SHAPES[role]
    candidates = tuple(_candidate_tensor_names(layer, expert, role))
    last_error: Optional[BaseException] = None
    for name in candidates:
        try:
            tensor = store.get(name)
        except Exception as exc:
            last_error = exc
            continue

        arr = _to_float32(tensor)
        if arr.shape == expected:
            return arr
        if arr.shape == expected[::-1]:
            return np.asarray(arr.T, dtype=np.float32, order="C")
        if arr.ndim == 1 and arr.size == expected[0] * expected[1]:
            return np.asarray(arr.reshape(expected), dtype=np.float32, order="C")
        last_error = ValueError(
            f"tensor {name} has shape {arr.shape}, expected {expected} or {expected[::-1]}"
        )
    raise KeyError(
        f"could not find a tensor for layer={layer} expert={expert} role={role}: "
        f"{', '.join(candidates)}"
    ) from last_error


def _to_float32(array: np.ndarray) -> np.ndarray:
    arr = np.asarray(array)
    if arr.dtype == np.float32:
        return arr.astype(np.float32, copy=False)
    if arr.dtype == np.float16:
        return arr.astype(np.float32)
    if str(arr.dtype) == "bfloat16":
        return arr.astype(np.float32)
    if arr.dtype == np.uint16:
        # Some export pipelines serialize BF16 tensors as raw uint16 payloads.
        return bf16_to_f32(arr)
    return arr.astype(np.float32)


def pack_expert_payload(
    gate: np.ndarray,
    up: np.ndarray,
    down: np.ndarray,
    bits: int,
) -> Tuple[bytes, ExpertLayout]:
    layout = layout_for_bits(bits)
    sections: List[bytes] = []
    for role, matrix in (("gate_proj", gate), ("up_proj", up), ("down_proj", down)):
        packed, scales, biases = _quantize_rowwise(matrix, bits)
        sections.append(np.asarray(packed, dtype=np.uint32, order="C").tobytes())
        sections.append(np.asarray(scales, dtype=np.uint16, order="C").tobytes())
        sections.append(np.asarray(biases, dtype=np.uint16, order="C").tobytes())
    payload = b"".join(sections)
    if len(payload) != layout.expert_size:
        raise AssertionError(
            f"packed expert size {len(payload)} does not match expected {layout.expert_size}"
        )
    return payload, layout


def dequantize_expert_payload(payload: bytes, layout: ExpertLayout) -> Dict[str, np.ndarray]:
    buf = memoryview(payload)
    result: Dict[str, np.ndarray] = {}
    for component in layout.components:
        start = component.offset
        end = start + component.size
        raw = np.frombuffer(buf[start:end], dtype=np.uint8)
        if component.dtype == "u32":
            data = raw.view(np.uint32).reshape(component.shape)
        elif component.dtype == "u16":
            data = raw.view(np.uint16).reshape(component.shape)
        else:
            raise ValueError(component.dtype)
        result[component.name] = np.array(data, copy=True)
    return result


def expert_payload_to_matrices(payload: bytes, bits: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decode a packed expert payload back to float32 matrices."""

    layout = layout_for_bits(bits)
    components = dequantize_expert_payload(payload, layout)
    gate = _dequantize_rowwise(
        components["gate_proj.weight"],
        components["gate_proj.scales"],
        components["gate_proj.biases"],
        bits,
    )
    up = _dequantize_rowwise(
        components["up_proj.weight"],
        components["up_proj.scales"],
        components["up_proj.biases"],
        bits,
    )
    down = _dequantize_rowwise(
        components["down_proj.weight"],
        components["down_proj.scales"],
        components["down_proj.biases"],
        bits,
    )
    return gate, up, down


def _expert_payload_from_store(store: TensorStore, layer: int, expert: int, bits: int) -> bytes:
    gate = _load_expert_matrix(store, layer, expert, "gate_proj")
    up = _load_expert_matrix(store, layer, expert, "up_proj")
    down = _load_expert_matrix(store, layer, expert, "down_proj")
    payload, _ = pack_expert_payload(gate, up, down, bits)
    return payload


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_layer_file(
    store: TensorStore,
    layer: int,
    out_path: Path,
    bits: int,
    num_experts: int = NUM_EXPERTS,
    force: bool = False,
) -> ExpertLayout:
    if out_path.exists() and not force:
        raise FileExistsError(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    layout = layout_for_bits(bits, num_experts=num_experts)
    with out_path.open("wb") as fh:
        for expert in range(num_experts):
            payload = _expert_payload_from_store(store, layer, expert, bits)
            fh.write(payload)
    if out_path.stat().st_size != layout.expert_size * num_experts:
        raise AssertionError(
            f"{out_path} has size {out_path.stat().st_size}, expected {layout.expert_size * num_experts}"
        )
    return layout


def convert_model(
    model_dir: Path,
    output_root: Path,
    bits: int,
    num_layers: int = NUM_LAYERS,
    num_experts: int = NUM_EXPERTS,
    force: bool = False,
) -> Path:
    model_dir = Path(model_dir)
    output_root = Path(output_root)
    layout = layout_for_bits(bits, num_experts=num_experts)
    out_dir = output_root / layout.output_dir_name
    if out_dir.exists() and not force:
        raise FileExistsError(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with TensorStore(model_dir) as store:
        for layer in range(num_layers):
            layer_name = f"layer_{layer:02d}.bin"
            layer_path = out_dir / layer_name
            _write_layer_file(store, layer, layer_path, bits, num_experts=num_experts, force=force)
    write_layout_json(out_dir / "layout.json", layout)
    return out_dir


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Pack Qwen3.6 routed experts")
    parser.add_argument("--model-dir", required=True, help="HF checkpoint directory with safetensors shards")
    parser.add_argument(
        "--output-root",
        required=True,
        help="Directory that will receive packed_experts/ and/or packed_experts_2bit/",
    )
    parser.add_argument(
        "--quant",
        choices=("2", "4", "both"),
        default="4",
        help="Expert quantization to produce",
    )
    parser.add_argument("--layers", type=int, default=NUM_LAYERS, help="Number of layers to pack")
    parser.add_argument("--experts", type=int, default=NUM_EXPERTS, help="Number of experts per layer")
    parser.add_argument("--force", action="store_true", help="Overwrite existing output directories")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    output_root = Path(args.output_root)
    if args.quant in ("4", "both"):
        convert_model(
            Path(args.model_dir),
            output_root,
            bits=4,
            num_layers=args.layers,
            num_experts=args.experts,
            force=args.force,
        )
    if args.quant in ("2", "both"):
        convert_model(
            Path(args.model_dir),
            output_root,
            bits=2,
            num_layers=args.layers,
            num_experts=args.experts,
            force=args.force,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
