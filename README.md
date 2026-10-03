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

## Architecture

```text
Hugging Face: Qwen/Qwen3.6-35B-A3B
        |
        v
scripts/convert_qwen36.py
BF16 safetensors -> nmoe runtime package
        |
        +--> model_weights.bin/json
        |    shared non-expert tensors
        |    mmap + Metal buffer
        |
        +--> tokenizer.bin + vocab.bin
        |    prompt encode / token decode
        |
        +--> packed_experts/
             q4 or q2 routed expert packs
             40 layers x 256 experts

Runtime path
------------

ask / chat / bench
        |
        v
Tokenizer
prompt encode using tokenizer.bin + vocab.bin
        |
        v
nmoe runtime
40-layer Qwen3.6 MoE decode loop
        |
        +--> shared weights resident
        |    model_weights.bin/json
        |         |
        |         v
        |    Metal kernels
        |    attention + shared expert
        |
        +--> GPU router
        |    select top-k experts per layer
        |         |
        |         v
        |    expert loader
        |    read only selected expert packs
        |    from packed_experts/
        |         |
        |         v
        |    Metal kernels
        |    routed expert
        |
        v
generated tokens
        |
        v
Tokenizer
decode using tokenizer.bin + vocab.bin
        |
        v
output text
```

What makes this project different:

- It keeps the shared Qwen3.6 tensors resident, but leaves routed experts in
  compact per-layer packs.
- Each token activates only the top-k routed experts, so the runtime reads and
  runs a tiny fraction of the 256 experts per layer.
- The hot path is a single native Objective-C/Metal binary, with no Python or
  server process in the inference loop.
- The same model package can carry q4 experts for quality or q2 experts for
  lower bandwidth experiments.

## What Is Included

- `ask`, `chat`, and `bench` commands
- Support for q4 and q2 routed expert packs
- Metal kernels for the runtime hot path
- A Python converter for Hugging Face safetensors into the runtime package

Model weights are not part of this repository. This repo expects a converted
runtime package made from the Hugging Face checkpoint
`Qwen/Qwen3.6-35B-A3B`.

## Build

```bash
make
```

The binary is written to `./nmoe`.

## Model Layout

By default the runtime looks for `qwen36_35b/`. In this working copy that path
may be a symlink to another converted package; for a fresh checkout you can
either create the directory directly or symlink it yourself.

```text
qwen36_35b/
  model_weights.bin
  model_weights.json
  tokenizer.bin
  vocab.bin
  packed_experts/
    layer_00.bin ... layer_39.bin
    layout.json
  packed_experts_q3/          # recommended on 16GB, used by --q3 (auto prefers it)
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

## Download And Convert The Model

Use the official Hugging Face model:

- Model repo: `Qwen/Qwen3.6-35B-A3B`
- Runtime package name used by this repo: `qwen36_35b`
- The converter expects the original BF16 `model-*.safetensors` shards plus
  `tokenizer.json`.

Install the download/conversion dependencies:

```bash
python3 -m pip install -U huggingface_hub numpy
```

Download the HF checkpoint. The model is large, so keep it outside git:

```bash
mkdir -p ../model
hf download Qwen/Qwen3.6-35B-A3B \
  --local-dir ../model/Qwen3.6-35B-A3B
```

If your `huggingface_hub` install does not provide the `hf` command, use:

```bash
huggingface-cli download Qwen/Qwen3.6-35B-A3B \
  --local-dir ../model/Qwen3.6-35B-A3B
```

Convert the downloaded checkpoint into the runtime layout:

```bash
python3 scripts/convert_qwen36.py \
  --model ../model/Qwen3.6-35B-A3B \
  --output qwen36_35b \
  --bits 4
```

That command writes:

- `model_weights.bin` and `model_weights.json` for non-expert tensors
- `tokenizer.bin` and `vocab.bin`
- `packed_experts/` for q4 routed experts

To additionally create q2 experts, reuse the already generated shared weights
and tokenizer files:

```bash
python3 scripts/convert_qwen36.py \
  --model ../model/Qwen3.6-35B-A3B \
  --output qwen36_35b \
  --bits 2 \
  --skip-weights \
  --skip-tokenizer
```

### q3 experts (recommended on 16GB machines)

The q4 expert pack is 18GB, larger than a 16GB Mac's RAM, so many expert reads
miss the page cache and hit the SSD (~2.8GB/s on a Mac mini). The q3 pack is
13GB (1.38MB per expert instead of 1.77MB): more of it stays cached and every
miss moves fewer bytes. It is derived from the q4 pack (no original checkpoint
needed); the requantizer uses a clip search plus a least-squares refit of each
group's scale/bias:

```bash
python3 scripts/requant_experts.py --src qwen36_35b/packed_experts \
  --dst qwen36_35b/packed_experts_q3 --bits 3 --container q3   # ~15 min on M4
```

Measured on an M4 Mac mini 16GB (128-token decode, `scripts/eval/mixed.txt` perplexity):

| experts | decode tok/s | page-cache hit | ppl | KL vs q4 |
|---|---|---|---|---|
| q4, K=8 | 6.7 | 78% | 5.64 | — |
| **q3, K=8** | **9.0** | 85% | 5.59 | 0.029 |
| q4, K=6 | — | — | 6.13 | 0.039 |

When `packed_experts_q3/` exists, `--quant auto` (the default) selects it.

If you already have a converted package elsewhere, symlink it:

```bash
ln -s /path/to/qwen36_35b qwen36_35b
```

## Run

```bash
./nmoe ask "解释一下 KV cache 和 prefill/decode 的区别" --q4 --experts 8 --tokens 128
./nmoe ask "请用中文介绍本地大模型推理" --q2 --experts 8 --tokens 128 --timing
./nmoe chat --q4 --experts 8 --tokens 512      # multi-turn; type /reset to clear the conversation
./nmoe bench "请介绍一下量子计算" --q2 --experts 6 --tokens 128 --timing --quiet
./nmoe ppl scripts/eval/mixed.txt --q3          # teacher-forced perplexity / accuracy check
```

To compare the accuracy of two configurations, save their predictions and diff
them (KL divergence, top-1 agreement, ΔNLL):

```bash
NMOE_PPL_DUMP=/tmp/q4.bin ./nmoe ppl scripts/eval/mixed.txt --q4
NMOE_PPL_DUMP=/tmp/q3.bin ./nmoe ppl scripts/eval/mixed.txt --q3
python3 scripts/ppl_compare.py /tmp/q4.bin /tmp/q3.bin
```

Useful options:

- `--model PATH`: model package directory, default `qwen36_35b`
- `--q2` / `--q3` / `--q4` / `--quant auto|2|3|4`: expert quantization mode (auto prefers q3)
- `--experts N`: active experts per layer, 1..8
- `--tokens N`: generation limit
- `--think N`: force `</think>` after N thinking tokens, `0` disables forcing
- `--timing`: print runtime timing
- `--quiet`: suppress token streaming

## Convert Experts Only

For expert-only refreshes, skip the shared weights and tokenizer. This is useful
when `model_weights.bin`, `model_weights.json`, `tokenizer.bin`, and `vocab.bin`
already exist:

```bash
python3 scripts/convert_qwen36.py \
  --model ../model/Qwen3.6-35B-A3B \
  --output qwen36_35b \
  --bits 4 \
  --skip-weights \
  --skip-tokenizer

python3 scripts/convert_qwen36.py \
  --model ../model/Qwen3.6-35B-A3B \
  --output qwen36_35b \
  --bits 2 \
  --skip-weights \
  --skip-tokenizer
```

Python dependencies for downloading/conversion: `huggingface_hub` and `numpy`.
