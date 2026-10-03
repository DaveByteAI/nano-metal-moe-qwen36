# nano-metal-moe-qwen36

[English](README.md) | **简体中文**

在一台 **16GB 内存的基础款 Mac mini** 上运行 **Qwen3.6-35B-A3B**（350 亿参数的混合专家模型），
生成速度约 **9 token/秒**。

`nmoe` 是一个用 Objective-C + Metal 写成的原生程序。推理过程中没有 Python、没有服务进程、
没有机器学习框架，Python 只在准备模型时用到。共享权重常驻内存；每层 256 个路由专家放在 SSD 上，
每个 token 只加载路由器选中的 8 个。

```
$ ./nmoe ask "用一段话解释 prefill 和 decode 的区别。"
```

## 亮点

- **16GB 机器跑 35B 模型。** 共享权重约 1.4GB，常驻内存；路由专家（13–18GB）放在 SSD 上按需读取。
  每个 token 只激活约 3B 参数。
- **q3 专家包，16GB 机器推荐。** 把专家重新量化到 3-bit，包从 18GB 降到 13GB，能留在页缓存里的更多，
  decode 比 q4 **快 35%**，精度没有可测的下降。
- **流水线 decode。** 每层只有一个 command buffer。路由结果一出来，GPU 就通过 `MTLSharedEvent`
  通知 CPU；CPU 读取选中的专家，down 权重还在读时，GPU 已经开始算 gate/up。
- **批量 prompt prefill。** prompt 按 32 个 token 一组处理。专家读取与 GPU 计算重叠进行，
  每组里的同一个专家只读一次，而不是每个 token 读一次。结果与逐 token 处理逐位一致。
- **多轮对话。** 对话历史保存在 KV cache 和线性注意力状态里，之前的轮次不会重算。
- **内置精度评测。** `nmoe ppl` 测困惑度，`scripts/ppl_compare.py` 比较两种配置的 KL 散度和
  top-1 一致率。每项提速都用数字衡量精度，不靠肉眼判断。

## 性能

Apple M4 Mac mini，16GB，macOS 26，后台正常开着浏览器、终端等应用：

| 专家包 | Decode | Prompt prefill | 困惑度 ¹ | 与 q4 的 KL ¹ |
|---|---|---|---|---|
| q4（18GB） | 6.8 tok/s | ~10 tok/s | 5.64 | — |
| **q3（13GB，默认）** | **9.0 tok/s** | **15–21 tok/s** | **5.59** | 0.029 |

¹ 在 [`scripts/eval/mixed.txt`](scripts/eval/mixed.txt) 上做 teacher-forced 评测，
共 364 个 token，混合了中文、英文、代码和 JSON。困惑度越低越好；KL 以 q4 的输出分布为基准。

速度取决于专家包有多少能留在页缓存里，其他应用占内存多时会变慢。decode 每层读取约 11MB 专家数据，
缓存未命中时要从 SSD 读，速度约 2.8GB/s。[优化记录](docs/optimization-log.zh-CN.md)
详细记录了每一毫秒花在哪里。

## 环境要求

- Apple Silicon Mac。16GB 内存就够，内存越大越快。
- 支持 Metal 的 macOS，以及 Xcode Command Line Tools（`xcode-select --install`）。
- 磁盘：q4 运行包约 20GB，q3 专家包另需 13GB。原始 BF16 权重（约 70GB）只在转换时需要，
  转换完可以删掉。
- Python 3（只在准备模型时使用）：`numpy`、`huggingface_hub`；生成 q3 包还需要 `torch`。

## 快速开始

### 1. 编译

```bash
make            # 生成 ./nmoe
```

Metal kernel 在运行时从 `metal/kernels.metal` 编译，所以请在仓库根目录下运行 `./nmoe`。

### 2. 下载并转换模型

```bash
python3 -m pip install -U huggingface_hub numpy torch

# 下载官方 BF16 权重（放在仓库外面）
hf download Qwen/Qwen3.6-35B-A3B --local-dir ../model/Qwen3.6-35B-A3B
# （较旧的 huggingface_hub 用：huggingface-cli download ... --local-dir ...）

# 转换成运行包：共享权重、分词器、q4 专家
python3 scripts/convert_qwen36.py \
  --model ../model/Qwen3.6-35B-A3B --output qwen36_35b --bits 4

# 16GB 机器推荐：从 q4 专家生成 q3 专家包（M4 上约 15 分钟）
python3 scripts/requant_experts.py \
  --src qwen36_35b/packed_experts --dst qwen36_35b/packed_experts_q3 \
  --bits 3 --container q3
```

生成的运行包结构：

```text
qwen36_35b/
  model_weights.bin, model_weights.json   # 共享（非专家）张量，mmap 加载
  tokenizer.bin, vocab.bin
  packed_experts/        layer_00..39.bin  # q4 专家（18GB）
  packed_experts_q3/     layer_00..39.bin  # q3 专家（13GB），默认选用
```

如果运行包放在别处，可以用 `ln -s /path/to/qwen36_35b qwen36_35b` 建立软链接，
或者运行时加 `--model PATH`。

### 3. 运行

```bash
./nmoe ask "用两句话解释 KV cache。"
./nmoe chat                                   # 多轮对话；输入 /reset 清空对话
./nmoe bench "请介绍一下量子计算" --tokens 128 --timing --quiet
```

## 用法

| 命令 | 作用 |
|---|---|
| `nmoe ask "问题"` | 一问一答，流式输出 |
| `nmoe chat` | 交互式多轮对话，`/reset` 重新开始 |
| `nmoe bench "问题"` | 与 `ask` 相同，配合 `--timing --quiet` 看速度 |
| `nmoe ppl 文本文件` | 对文本做 teacher-forced 评测，输出困惑度和 top-1 准确率 |

| 参数 | 默认值 | 含义 |
|---|---|---|
| `--model PATH` | `qwen36_35b` | 运行包目录 |
| `--quant auto\|2\|3\|4`、`--q2`/`--q3`/`--q4` | `auto` | 专家包；`auto` 有 q3 就用 q3，否则用 q4 |
| `--experts N` | 8 | 每个 token 激活的路由专家数（1–8），越少越快但越不准 |
| `--tokens N` | 256（chat 为 512） | 最多生成多少 token |
| `--think N` | 1 | 思考 N 个 token 后强制输出 `</think>`；`0` 表示不限制 |
| `--timing` | 关 | 输出每层耗时、decode 和 prefill 速度 |
| `--quiet` | 关 | 不流式打印生成内容 |

### 如何选择专家包

| 专家包 | 大小 | 16GB 机器上的速度 | 精度 |
|---|---|---|---|
| q4 | 18GB | 基准 | 参照 |
| **q3** | 13GB | decode 快 35% | 与 q4 相当，差异在噪声范围内（KL 0.029） |
| q2 | 10GB | 最快 | 本项目未评测，2-bit 预计精度损失较大；实验性质（`convert_qwen36.py --bits 2`） |

减少 `--experts` 不如换 q3 划算。例如 `--experts 6` 会让困惑度上升约 9%（KL 0.039）。

## 精度评测

凡是可能影响数值的改动（量化、路由、kernel），都应该和基准对比：

```bash
NMOE_PPL_DUMP=/tmp/q4.bin ./nmoe ppl scripts/eval/mixed.txt --q4
NMOE_PPL_DUMP=/tmp/q3.bin ./nmoe ppl scripts/eval/mixed.txt --q3
python3 scripts/ppl_compare.py /tmp/q4.bin /tmp/q3.bin   # top-1 一致率、KL、ΔNLL
```

## 工作原理

每个 token 经过 40 层：30 层是 GatedDeltaNet 线性注意力，每第 4 层是带 KV cache 的全注意力。
下图是每层的 decode 流水线；prefill 走同样的步骤，只是一次处理 32 个 token。

```text
GPU  norm → attention → router + shared expert ─┬─▶ [wait] gate/up ─▶ [wait] down + combine ─▶ 下一层
                                                 │         ▲                 ▲
CPU                                    top-k（256 选 8）    │                 │
                                         pread gate+up ────┘                 │
                                         pread down（与 GPU gate/up 重叠）───┘
```

- **共享权重**（`model_weights.bin`，q4）只 mmap 一次，并包装成 Metal buffer。
- **路由专家**每层一个文件，用并行 `pread` 读进固定的 buffer，q3 下每个专家约 1.4MB。
  专家缓存完全交给操作系统的页缓存。
- **专家 kernel 在路由结果出来之前就已编码好。** 路由权重通过 buffer 传给 GPU，
  执行顺序由 `MTLSharedEvent` 控制，所以 CPU 不需要等每一层的 command buffer 执行完。

## 目录结构

```text
src/runtime.m            模型本体、decode/prefill 流水线、ask/chat/bench/ppl
src/backend/             Metal 设备、pipeline、buffer 管理
src/expert_io.m          专家包布局与读取
src/tokenizer.m          BPE 分词器
metal/kernels.metal      全部 GPU kernel（运行时编译）
include/nmoe/            对外 C 头文件
scripts/convert_qwen36.py   HF safetensors → 运行包（q4/q2）
scripts/requant_experts.py  q4 专家 → q3 专家
scripts/ppl_compare.py      比较两次 `nmoe ppl` 的结果
scripts/*_bench.*           微基准（matvec、pread）
docs/optimization-log.zh-CN.md   详细优化记录
```

## 延伸阅读

- [优化记录](docs/optimization-log.zh-CN.md)：各项测量数据，哪些实验有效、哪些无效，以及原因。
- [CLAUDE.md](CLAUDE.md)（英文）：面向贡献者的架构说明，包括如何新增 Metal kernel、
  流水线必须遵守的约束，以及实验用的环境变量。

本仓库不包含模型权重。权重来自
[Qwen/Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B)，使用时须遵守其许可协议。
