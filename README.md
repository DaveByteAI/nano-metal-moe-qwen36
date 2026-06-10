# nano-metal-moe-qwen36

`nmoe` is a compact Apple Silicon Metal runtime for local Qwen3.6-35B-A3B
inference on a base Mac mini with 16GB unified memory. The public tree is kept
small: one command-line binary, the Metal kernels, and a model conversion
script.

The core idea is to keep the shared model weights resident while treating the
routed MoE experts as quantized external packs. Each layer selects only the
top-k experts needed for the current token, then loads and runs just those
expert weights instead of keeping all 256 experts hot in memory. q4 and q2
expert packs reduce bandwidth and storage pressure further, which is what makes
this setup practical on a 16GB machine.

## What Is Included

- `ask`, `chat`, and `bench` commands
- Support for q4 and q2 routed expert packs
- Metal kernels for the runtime hot path
- A Python converter for Hugging Face safetensors expert weights

Model weights are not part of this repository.

## Build

```bash
make
```

The binary is written to `./nmoe`.

## Model Layout

By default the runtime looks for `qwen36_35b/`:

```text
qwen36_35b/
  model_weights.bin
  model_weights.json
  tokenizer.bin
  vocab.bin
  packed_experts/
    layer_00.bin ... layer_39.bin
    layout.json
  packed_experts_2bit/        # optional, used by --q2
    layer_00.bin ... layer_39.bin
    layout.json
```

You can also pass a model directory explicitly:

```bash
./nmoe ask "你好" --model /path/to/qwen36_35b
```

## Run

```bash
./nmoe ask "解释一下 KV cache 和 prefill/decode 的区别" --q4 --experts 8 --tokens 128
./nmoe ask "请用中文介绍本地大模型推理" --q2 --experts 8 --tokens 128 --timing
./nmoe chat --q4 --experts 8 --tokens 512
./nmoe bench "请介绍一下量子计算" --q2 --experts 6 --tokens 128 --timing --quiet
```

Useful options:

- `--model PATH`: model package directory, default `qwen36_35b`
- `--q2` / `--q4` / `--quant auto|2|4`: expert quantization mode
- `--experts N`: active experts per layer, 1..8
- `--tokens N`: generation limit
- `--think N`: force `</think>` after N thinking tokens, `0` disables forcing
- `--timing`: print runtime timing
- `--quiet`: suppress token streaming

## Convert Experts

The converter writes `packed_experts/` for q4 or `packed_experts_2bit/` for q2:

```bash
python3 scripts/convert_qwen36.py \
  --model-dir /path/to/Qwen3.6-35B-A3B \
  --output-root qwen36_35b \
  --quant 4

python3 scripts/convert_qwen36.py \
  --model-dir /path/to/Qwen3.6-35B-A3B \
  --output-root qwen36_35b \
  --quant 2
```

Use `--quant both` to generate both directories in one run. Add `--force` if the
output directory already exists and should be overwritten.

Python dependencies: `numpy` and `safetensors`.
