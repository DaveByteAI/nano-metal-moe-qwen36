#!/usr/bin/env python3
"""Generate the README figures as SVG (light + dark, English + Chinese).

    python3 docs/assets/make_figures.py

Writes docs/assets/<figure>-<lang>-<theme>.svg. The READMEs embed them with
<picture> so GitHub picks the variant matching the viewer's theme. Edit the
numbers in DATA and re-run; nothing else needs to change.
"""

import os
from xml.sax.saxutils import escape

OUT = os.path.dirname(os.path.abspath(__file__))
FONT = ("system-ui, -apple-system, 'Segoe UI', 'PingFang SC', 'Hiragino Sans GB', "
        "'Microsoft YaHei', 'Noto Sans CJK SC', sans-serif")
MONO = "ui-monospace, 'SF Mono', Menlo, Consolas, monospace"

# Measured on an M4 Mac mini 16GB (see README / optimization log).
DATA = {
    "decode_q4": 6.8, "decode_q3": 9.0,
    "size_q4": 18.1, "size_q3": 13.0,
    "ppl_q4": 5.64, "ppl_q3": 5.59,
}

THEMES = {
    "light": {
        "surface": "#fcfcfb", "plane": "#f4f3ef", "ink": "#0b0b0b", "ink2": "#52514e",
        "muted": "#898781", "grid": "#e1e0d9", "axis": "#c3c2b7", "border": "#e4e3dd",
        "gpu": "#2a78d6", "ssd": "#eb6834", "cpu": "#1baf7a", "mem": "#4a3aa7",
        "base": "#b9b7af", "good": "#006300", "dot": "#dcdad3", "wash": 0.11, "seg": 0.55,
    },
    "dark": {
        "surface": "#1a1a19", "plane": "#222220", "ink": "#ffffff", "ink2": "#c3c2b7",
        "muted": "#898781", "grid": "#2c2c2a", "axis": "#383835", "border": "#2e2e2b",
        "gpu": "#3987e5", "ssd": "#d95926", "cpu": "#199e70", "mem": "#9085e9",
        "base": "#5c5b56", "good": "#0ca30c", "dot": "#34342f", "wash": 0.18, "seg": 0.8,
    },
}

T = {
    "en": {
        "eyebrow": "LOCAL LLM INFERENCE  ·  APPLE SILICON  ·  METAL",
        "title": "nano-metal-moe",
        "subtitle": "Qwen3.6-35B-A3B on a 16 GB Mac mini",
        "tiles": [("9.0", "tok/s", "decode speed"), ("35B", "", "params · 3B active"),
                  ("16 GB", "", "unified memory"), ("1", "binary", "Objective-C + Metal")],
        "grid_caption": "Each layer has 256 experts.",
        "grid_caption2": "A token reads only the 8 the router picks.",

        "arch_title": "How a 35B model fits in 16 GB",
        "arch_sub": "Shared weights stay in memory. Experts stay on the SSD and are read on demand.",
        "mem_title": "16 GB unified memory",
        "mem_rows": [("macOS + apps", "≈ 8 GB"), ("Page cache: hot experts", "≈ 6 GB"),
                     ("Metal buffers", "0.3 GB"), ("Shared weights (resident)", "1.4 GB")],
        "flow_title": "Per token, × 40 layers",
        "flow": [("GPU", "Attention + router", "shared weights, always resident"),
                 ("CPU", "Pick the top 8 of 256 experts", "softmax + top-k"),
                 ("CPU", "Parallel pread, ≈ 11 MB", "85% served by the page cache"),
                 ("GPU", "8 experts + shared expert", "combine, then next layer")],
        "token": "token", "next": "next token", "loop": "× 40 layers",
        "ssd_title": "SSD expert packs",
        "ssd_sub": "40 files × 256 experts",
        "ssd_sub2": "q3: 13 GB, 1.38 MB per expert",
        "hit": "hit", "miss": "miss → SSD 2.8 GB/s",

        "pipe_title": "One layer, three lanes, no idle round-trips",
        "pipe_sub": "One Metal command buffer per layer. MTLSharedEvents hand off between GPU and CPU mid-buffer.",
        "lanes": ["GPU", "CPU", "SSD"],
        "p_attn": "Attention + router", "p_topk": "top-k", "p_w1": "pread gate + up",
        "p_w2": "pread down", "p_gu": "gate/up", "p_down": "down + combine",
        "p_next": "next layer", "p_miss1": "cache misses", "p_miss2": "misses",
        "p_route": "route ready", "p_rel1": "release", "p_rel2": "release",
        "p_wait": "waits on event", "p_axis": "ms (illustrative, q3 on M4 ≈ 2.4 ms per layer)",

        "perf_title": "q3 experts: faster and smaller, same accuracy",
        "perf_sub": "M4 Mac mini 16 GB. Perplexity on scripts/eval/mixed.txt (364 tokens).",
        "panels": [("Decode speed", "tok/s, higher is better"),
                   ("Expert pack size", "GB, smaller is better"),
                   ("Perplexity", "lower is better")],
        "legend": ["q4 experts", "q3 experts (default)"],
        "deltas": ["+32%", "−28%", "no loss"],
    },
    "zh": {
        "eyebrow": "本地大模型推理  ·  APPLE SILICON  ·  METAL",
        "title": "nano-metal-moe",
        "subtitle": "在 16GB Mac mini 上运行 Qwen3.6-35B-A3B",
        "tiles": [("9.0", "tok/s", "生成速度"), ("35B", "", "参数 · 激活 3B"),
                  ("16 GB", "", "统一内存"), ("1", "个", "原生程序 · Metal")],
        "grid_caption": "每层有 256 个专家，",
        "grid_caption2": "每个 token 只读路由器选中的 8 个。",

        "arch_title": "35B 模型如何装进 16GB",
        "arch_sub": "共享权重常驻内存；专家放在 SSD 上，按需读取。",
        "mem_title": "16GB 统一内存",
        "mem_rows": [("macOS 与其他应用", "≈ 8 GB"), ("页缓存：热门专家", "≈ 6 GB"),
                     ("Metal 缓冲区", "0.3 GB"), ("共享权重（常驻）", "1.4 GB")],
        "flow_title": "每个 token，经过 40 层",
        "flow": [("GPU", "注意力 + 路由器", "共享权重，始终在内存中"),
                 ("CPU", "从 256 个专家中选 8 个", "softmax + top-k"),
                 ("CPU", "并行 pread，约 11 MB", "85% 命中页缓存"),
                 ("GPU", "8 个专家 + 共享专家", "合并结果，进入下一层")],
        "token": "token", "next": "下一个 token", "loop": "× 40 层",
        "ssd_title": "SSD 上的专家包",
        "ssd_sub": "40 个文件 × 256 个专家",
        "ssd_sub2": "q3：13GB，每个专家 1.38MB",
        "hit": "命中", "miss": "未命中 → SSD 2.8 GB/s",

        "pipe_title": "每层的流水线：三条泳道，没有空等",
        "pipe_sub": "每层一个 Metal command buffer，GPU 与 CPU 在执行中途通过 MTLSharedEvent 交接。",
        "lanes": ["GPU", "CPU", "SSD"],
        "p_attn": "注意力 + 路由器", "p_topk": "top-k", "p_w1": "读取 gate + up",
        "p_w2": "读取 down", "p_gu": "gate/up", "p_down": "down + 合并",
        "p_next": "下一层", "p_miss1": "缓存未命中", "p_miss2": "未命中",
        "p_route": "路由就绪", "p_rel1": "放行", "p_rel2": "放行",
        "p_wait": "等待事件", "p_axis": "毫秒（示意，q3 在 M4 上每层约 2.4 ms）",

        "perf_title": "q3 专家包：更快、更小，精度不变",
        "perf_sub": "M4 Mac mini 16GB。困惑度在 scripts/eval/mixed.txt（364 token）上测得。",
        "panels": [("生成速度", "tok/s，越高越好"),
                   ("专家包大小", "GB，越小越好"),
                   ("困惑度", "越低越好")],
        "legend": ["q4 专家包", "q3 专家包（默认）"],
        "deltas": ["+32%", "−28%", "无损失"],
    },
}


# --------------------------------------------------------------------------
# tiny SVG helpers
# --------------------------------------------------------------------------

def text(x, y, s, size, fill, weight=400, anchor="start", family=FONT, spacing=None, opacity=None):
    extra = ""
    if spacing is not None:
        extra += f' letter-spacing="{spacing}"'
    if opacity is not None:
        extra += f' opacity="{opacity}"'
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-family="{family}" font-size="{size}" '
            f'font-weight="{weight}" fill="{fill}" text-anchor="{anchor}"{extra}>{escape(s)}</text>')


def rect(x, y, w, h, fill, rx=0, opacity=None, stroke=None, sw=1):
    extra = f' fill-opacity="{opacity}"' if opacity is not None else ""
    if stroke:
        extra += f' stroke="{stroke}" stroke-width="{sw}"'
    return f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{rx}" fill="{fill}"{extra}/>'


def est_width(s, size):
    """Rough rendered width: CJK glyphs are ~1em, Latin ~0.56em."""
    return sum(1.0 if ord(ch) >= 0x2E80 else 0.56 for ch in s) * size


def bar_h(x, y, w, h, fill, r=4):
    """Horizontal bar: square at the baseline (left), rounded data end (right)."""
    r = min(r, w / 2, h / 2)
    return (f'<path d="M{x:.1f},{y:.1f} H{x + w - r:.1f} Q{x + w:.1f},{y:.1f} {x + w:.1f},{y + r:.1f} '
            f'V{y + h - r:.1f} Q{x + w:.1f},{y + h:.1f} {x + w - r:.1f},{y + h:.1f} H{x:.1f} Z" fill="{fill}"/>')


def arrow_defs(c):
    out = ['<defs>']
    for name, color in (("ink", c["muted"]), ("gpu", c["gpu"]), ("cpu", c["cpu"]), ("ssd", c["ssd"])):
        out.append(f'<marker id="ah-{name}" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="7" '
                   f'markerHeight="7" orient="auto-start-reverse"><path d="M0,0.8 L9,5 L0,9.2 Z" fill="{color}"/></marker>')
    out.append('</defs>')
    return "".join(out)


def line(x1, y1, x2, y2, color, width=1.6, head=None, dash=None):
    extra = f' marker-end="url(#ah-{head})"' if head else ""
    if dash:
        extra += f' stroke-dasharray="{dash}"'
    return (f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{color}" '
            f'stroke-width="{width}" stroke-linecap="round"{extra}/>')


def path(d, color, width=1.6, head=None, dash=None, fill="none"):
    extra = f' marker-end="url(#ah-{head})"' if head else ""
    if dash:
        extra += f' stroke-dasharray="{dash}"'
    return (f'<path d="{d}" fill="{fill}" stroke="{color}" stroke-width="{width}" '
            f'stroke-linecap="round" stroke-linejoin="round"{extra}/>')


def svg(w, h, body, c, title):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}" '
            f'role="img" aria-label="{escape(title)}"><title>{escape(title)}</title>'
            f'{arrow_defs(c)}'
            f'{rect(0.5, 0.5, w - 1, h - 1, c["surface"], rx=18, stroke=c["border"], sw=1)}'
            f'{body}</svg>\n')


def expert_grid(x0, y0, cell, gap, lit, c, faded=False):
    """16x16 grid of 256 experts; `lit` indices highlighted in the SSD hue."""
    out = []
    for i in range(256):
        r, col = divmod(i, 16)
        x = x0 + col * (cell + gap)
        y = y0 + r * (cell + gap)
        if i in lit and not faded:
            out.append(f'<circle cx="{x + cell / 2:.1f}" cy="{y + cell / 2:.1f}" r="{cell * 0.95:.1f}" '
                       f'fill="{c["ssd"]}" fill-opacity="0.18"/>')
            out.append(f'<circle cx="{x + cell / 2:.1f}" cy="{y + cell / 2:.1f}" r="{cell / 2:.1f}" fill="{c["ssd"]}"/>')
        else:
            fade = ' fill-opacity="0.6"' if faded else ""
            out.append(f'<circle cx="{x + cell / 2:.1f}" cy="{y + cell / 2:.1f}" r="{cell / 2 * 0.78:.1f}" '
                       f'fill="{c["dot"]}"{fade}/>')
    return "".join(out)


LIT = {21, 46, 93, 115, 138, 167, 200, 243}


# --------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------

def hero(c, t):
    w, h = 1200, 380
    b = []
    # soft accent wash on the right half
    b.append(f'<defs><radialGradient id="glow" cx="0.78" cy="0.45" r="0.55">'
             f'<stop offset="0" stop-color="{c["ssd"]}" stop-opacity="{c["wash"] * 0.9}"/>'
             f'<stop offset="1" stop-color="{c["ssd"]}" stop-opacity="0"/></radialGradient></defs>')
    b.append(rect(1, 1, w - 2, h - 2, "url(#glow)", rx=18))
    b.append(text(56, 74, t["eyebrow"], 13, c["muted"], 600, spacing=1.6))
    b.append(text(54, 132, t["title"], 50, c["ink"], 750, spacing=-1))
    b.append(text(56, 172, t["subtitle"], 22, c["ink2"], 500))
    # stat tiles
    tx, ty, tw, th, tg = 56, 214, 156, 112, 14
    accents = [c["gpu"], c["mem"], c["cpu"], c["ssd"]]
    for i, (big, unit, label) in enumerate(t["tiles"]):
        x = tx + i * (tw + tg)
        b.append(rect(x, ty, tw, th, c["plane"], rx=12))
        b.append(rect(x + 16, ty + 18, 22, 4, accents[i], rx=2))
        b.append(f'<text x="{x + 16}" y="{ty + 66}" font-family="{FONT}" fill="{c["ink"]}">'
                 f'<tspan font-size="32" font-weight="700">{escape(big)}</tspan>'
                 f'<tspan font-size="15" font-weight="500" fill="{c["ink2"]}" dx="5">{escape(unit)}</tspan></text>')
        b.append(text(x + 16, ty + 92, label, 13, c["ink2"], 500))
    # expert grid motif: two faded layers behind, the live layer in front
    cell, gap = 10, 5
    span = 16 * cell + 15 * gap
    gx, gy = 1200 - 92 - span, 62
    b.append(f'<g opacity="0.35">{expert_grid(gx + 28, gy - 22, cell, gap, LIT, c, faded=True)}</g>')
    b.append(f'<g opacity="0.6">{expert_grid(gx + 14, gy - 11, cell, gap, LIT, c, faded=True)}</g>')
    b.append(rect(gx - 14, gy - 14, span + 28, span + 28, c["surface"], rx=14, opacity=0.86))
    b.append(expert_grid(gx, gy, cell, gap, LIT, c))
    b.append(text(gx + span / 2, gy + span + 42, t["grid_caption"], 14, c["ink2"], 500, anchor="middle"))
    b.append(text(gx + span / 2, gy + span + 63, t["grid_caption2"], 14, c["ink2"], 500, anchor="middle"))
    return svg(w, h, "".join(b), c, f'{t["title"]}: {t["subtitle"]}')


def card(x, y, w, h, role, color, title, sub, c, tag_w=46):
    out = [rect(x, y, w, h, color, rx=12, opacity=c["wash"]),
           rect(x, y + 12, 4, h - 24, color, rx=2),
           rect(x + 18, y + h / 2 - 11, tag_w, 22, color, rx=11),
           text(x + 18 + tag_w / 2, y + h / 2 + 4.5, role, 11.5, "#ffffff", 700, anchor="middle", spacing=0.6),
           text(x + 18 + tag_w + 14, y + h / 2 - 3, title, 16, c["ink"], 650),
           text(x + 18 + tag_w + 14, y + h / 2 + 17, sub, 12.5, c["ink2"], 450)]
    return "".join(out)


def architecture(c, t):
    w, h = 1200, 600
    b = [text(48, 62, t["arch_title"], 26, c["ink"], 720),
         text(48, 90, t["arch_sub"], 15, c["ink2"], 450)]

    # ---- memory column: stacked bar, 16 GB tall ----
    mx, my, mw, mh = 64, 168, 60, 360
    b.append(text(48, 146, t["mem_title"], 15, c["ink"], 650))
    gb = mh / 16.0
    segs = [(8.3, c["base"], c["seg"]), (6.0, c["ssd"], c["seg"]), (0.3, c["mem"], 1.0), (1.4, c["gpu"], 1.0)]
    y = my
    centers = []
    for i, (size, color, op) in enumerate(segs):
        hh = size * gb
        top_r = 8 if i == 0 else 0
        bot_r = 8 if i == len(segs) - 1 else 0
        y0, y1 = y, y + hh - (2 if i < len(segs) - 1 else 0)
        d = (f"M{mx},{y0 + top_r} Q{mx},{y0} {mx + top_r},{y0} H{mx + mw - top_r} Q{mx + mw},{y0} {mx + mw},{y0 + top_r} "
             f"V{y1 - bot_r} Q{mx + mw},{y1} {mx + mw - bot_r},{y1} H{mx + bot_r} Q{mx},{y1} {mx},{y1 - bot_r} Z")
        b.append(f'<path d="{d}" fill="{color}" fill-opacity="{op}"/>')
        centers.append((y0 + y1) / 2)
        y += hh
    label_y = [centers[0], centers[1], my + mh - 1.4 * gb - 30, my + mh - 0.7 * gb + 6]
    for i, ((name, val), cy) in enumerate(zip(t["mem_rows"], label_y)):
        b.append(text(mx + mw + 18, cy - 2, name, 13.5, c["ink"], 600))
        b.append(text(mx + mw + 18, cy + 16, val, 12.5, c["ink2"], 450, family=MONO))
    # tick from the small segments to their labels
    b.append(line(mx + mw + 4, my + mh - 1.4 * gb - 0.15 * gb, mx + mw + 12, label_y[2] - 6, c["muted"], 1))
    page_cache_y = centers[1]

    # ---- flow column ----
    fx, fw, fh = 392, 388, 66
    b.append(text(fx, 146, t["flow_title"], 15, c["ink"], 650))
    tok_y = 168
    b.append(rect(fx, tok_y, 92, 30, c["plane"], rx=15))
    b.append(text(fx + 46, tok_y + 20, t["token"], 13, c["ink2"], 600, anchor="middle", family=MONO))
    ys = [tok_y + 52 + i * (fh + 22) for i in range(4)]
    colors = {"GPU": c["gpu"], "CPU": c["cpu"]}
    b.append(line(fx + 46, tok_y + 30, fx + 46, ys[0] - 3, c["muted"], head="ink"))
    for i, (role, title, sub) in enumerate(t["flow"]):
        b.append(card(fx, ys[i], fw, fh, role, colors[role], title, sub, c))
        if i < 3:
            b.append(line(fx + 46, ys[i] + fh, fx + 46, ys[i + 1] - 3, c["muted"], head="ink"))
    nt_y = ys[3] + fh + 18
    b.append(line(fx + 46, ys[3] + fh, fx + 46, nt_y + 14, c["muted"], head="ink"))
    b.append(text(fx + 64, nt_y + 12, t["next"], 13, c["ink2"], 600, family=MONO))

    # ---- SSD column ----
    sx, sy, sw_, sh = 924, 168, 248, 360
    b.append(text(sx, 146, t["ssd_title"], 15, c["ink"], 650))
    # stacked layer cards
    card_w, card_h = sw_ - 28, 262
    for k in (2, 1):
        b.append(rect(sx + 12 * k, sy + 12 * (2 - k), card_w, card_h, c["ssd"], rx=12, opacity=c["wash"] * 0.7))
    cx0, cy0 = sx, sy + 24
    b.append(rect(cx0, cy0, card_w, card_h, c["surface"], rx=12, stroke=c["ssd"], sw=1.5))
    b.append(text(cx0 + 16, cy0 + 26, "layer_07.bin", 12.5, c["ink2"], 600, family=MONO))
    b.append(expert_grid(cx0 + (card_w - (16 * 8.6 + 15 * 3.6)) / 2, cy0 + 44, 8.6, 3.6, LIT, c))
    b.append(text(sx, cy0 + card_h + 34, t["ssd_sub"], 14, c["ink"], 600))
    b.append(text(sx, cy0 + card_h + 56, t["ssd_sub2"], 13, c["ink2"], 450))

    # ---- arrows into the pread card ----
    pr_y = ys[2] + fh / 2
    b.append(line(cx0 - 4, pr_y, fx + fw + 6, pr_y, c["ssd"], 1.8, head="ssd"))
    mid = (cx0 + fx + fw) / 2
    b.append(text(mid, pr_y - 12, t["miss"].split(" → ")[0], 12.5, c["ink2"], 650, anchor="middle"))
    b.append(text(mid, pr_y + 22, "SSD · 2.8 GB/s", 12, c["ink2"], 500, anchor="middle"))
    hit_x0 = mx + mw + 18 + 182
    b.append(line(hit_x0, pr_y, fx - 6, pr_y, c["ssd"], 1.8, head="ssd"))
    b.append(text((hit_x0 + fx) / 2, pr_y - 12, t["hit"], 12.5, c["ink2"], 650, anchor="middle"))
    return svg(w, h, "".join(b), c, t["arch_title"])


def pipeline(c, t):
    w, h = 1200, 400
    b = [text(48, 62, t["pipe_title"], 26, c["ink"], 720),
         text(48, 90, t["pipe_sub"], 15, c["ink2"], 450)]
    x0, x1 = 150, 1150
    tmax = 2.7
    sx = lambda ms: x0 + (x1 - x0) * ms / tmax
    lane_y = {"GPU": 132, "CPU": 212, "SSD": 292}
    lh = 46
    lane_color = {"GPU": c["gpu"], "CPU": c["cpu"], "SSD": c["ssd"]}
    for name, label in zip(("GPU", "CPU", "SSD"), t["lanes"]):
        y = lane_y[name]
        b.append(rect(x0 - 6, y - 8, x1 - x0 + 12, lh + 16, c["plane"], rx=12))
        b.append(rect(52, y + lh / 2 - 13, 64, 26, lane_color[name], rx=13))
        b.append(text(84, y + lh / 2 + 5, label, 12.5, "#ffffff", 700, anchor="middle", spacing=0.6))

    def block(lane, a, z, label, solid=True, small=False, faded=False):
        y = lane_y[lane]
        col = lane_color[lane]
        x, ww = sx(a), sx(z) - sx(a)
        op = 1.0 if solid else c["seg"] * 0.5
        if faded:
            op = 0.35
        out = [rect(x + 1, y, ww - 2, lh, col, rx=8, opacity=op)]
        ink = "#ffffff" if solid and not faded else c["ink"]
        if label:
            out.append(text(x + ww / 2, y + lh / 2 + 5, label, 12 if small else 13.5, ink, 650, anchor="middle"))
        return "".join(out)

    # GPU lane
    b.append(block("GPU", 0.0, 1.1, t["p_attn"]))
    b.append(rect(sx(1.1) + 2, lane_y["GPU"] + lh / 2 - 1, sx(1.9) - sx(1.1) - 4, 2, c["axis"], rx=1))
    b.append(text((sx(1.1) + sx(1.9)) / 2, lane_y["GPU"] + lh / 2 - 9, t["p_wait"], 12, c["muted"], 500, anchor="middle"))
    b.append(block("GPU", 1.9, 2.08, t["p_gu"], small=True))
    b.append(block("GPU", 2.3, 2.46, "", small=True))
    b.append(text(sx(2.38), lane_y["GPU"] - 16, t["p_down"], 12, c["ink2"], 600, anchor="middle"))
    b.append(line(sx(2.38), lane_y["GPU"] - 11, sx(2.38), lane_y["GPU"] - 2, c["muted"], 1))
    b.append(block("GPU", 2.46, 2.7, t["p_next"], faded=True, small=True))
    # CPU lane
    b.append(block("CPU", 1.1, 1.16, "", small=True))
    b.append(text(sx(1.13), lane_y["CPU"] + lh + 22, t["p_topk"], 12, c["ink2"], 600, anchor="middle"))
    b.append(line(sx(1.13), lane_y["CPU"] + lh + 8, sx(1.13), lane_y["CPU"] + lh + 1, c["muted"], 1))
    b.append(block("CPU", 1.16, 1.9, t["p_w1"]))
    b.append(block("CPU", 1.9, 2.3, t["p_w2"]))
    # SSD lane: only cache misses touch the SSD
    b.append(block("SSD", 1.3, 1.75, t["p_miss1"], solid=False))
    b.append(block("SSD", 2.0, 2.2, t["p_miss2"], solid=False, small=True))
    # events
    def event(ms, y_from, y_to, label, label_dx=8):
        x = sx(ms)
        out = [line(x, y_from, x, y_to, c["ink2"], 1.4, head="ink", dash="3 4"),
               f'<rect x="{x - 5}" y="{y_from - 5}" width="10" height="10" fill="{c["ink"]}" transform="rotate(45 {x} {y_from})"/>']
        out.append(text(x + label_dx, (y_from + y_to) / 2 + 4, label, 12, c["ink"], 650))
        return "".join(out)
    b.append(event(1.1, lane_y["GPU"] + lh + 4, lane_y["CPU"] - 6, t["p_route"]))
    b.append(event(1.9, lane_y["CPU"] - 4, lane_y["GPU"] + lh + 6, t["p_rel1"]))
    b.append(event(2.3, lane_y["CPU"] - 4, lane_y["GPU"] + lh + 6, t["p_rel2"]))
    # time axis
    ay = 370
    b.append(line(x0, ay, x1, ay, c["axis"], 1))
    for ms in (0, 0.5, 1.0, 1.5, 2.0, 2.5):
        b.append(line(sx(ms), ay, sx(ms), ay + 5, c["axis"], 1))
        b.append(text(sx(ms), ay + 20, f"{ms:g}", 11.5, c["muted"], 500, anchor="middle", family=MONO))
    b.append(text(x1, ay - 8, t["p_axis"], 12, c["muted"], 500, anchor="end"))
    return svg(w, h + 10, "".join(b), c, t["pipe_title"])


def performance(c, t):
    w, h = 1200, 330
    b = [text(48, 62, t["perf_title"], 26, c["ink"], 720),
         text(48, 90, t["perf_sub"], 15, c["ink2"], 450)]
    # legend
    lx = w - 48
    items = list(zip(t["legend"], (c["base"], c["gpu"])))
    widths = [est_width(s, 13) + 40 for s, _ in items]
    x = lx - sum(widths)
    for (label, col), ww in zip(items, widths):
        b.append(rect(x, 54, 12, 12, col, rx=3))
        b.append(text(x + 18, 64.5, label, 13, c["ink2"], 500))
        x += ww
    vals = [(DATA["decode_q4"], DATA["decode_q3"], 10.0, "{:.1f}"),
            (DATA["size_q4"], DATA["size_q3"], 20.0, "{:.0f} GB"),
            (DATA["ppl_q4"], DATA["ppl_q3"], 6.0, "{:.2f}")]
    pw, pg, px0, py = 352, 24, 48, 120
    for i, ((title, sub), (a, q, vmax, fmt), delta) in enumerate(zip(t["panels"], vals, t["deltas"])):
        x0 = px0 + i * (pw + pg)
        b.append(rect(x0, py, pw, 180, c["plane"], rx=14))
        b.append(text(x0 + 22, py + 34, title, 16, c["ink"], 680))
        b.append(text(x0 + 22, py + 56, sub, 12.5, c["muted"], 500))
        b.append(text(x0 + pw - 22, py + 36, delta, 20, c["good"], 750, anchor="end"))
        bx, bw_max = x0 + 52, pw - 52 - 80
        for j, (v, col, lab) in enumerate(((a, c["base"], "q4"), (q, c["gpu"], "q3"))):
            yy = py + 88 + j * 40
            b.append(text(x0 + 22, yy + 15, lab, 13, c["ink2"], 650, family=MONO))
            b.append(bar_h(bx, yy, bw_max * v / vmax, 22, col))
            b.append(text(bx + bw_max * v / vmax + 8, yy + 16, fmt.format(v), 14, c["ink"], 650))
        b.append(line(bx, py + 82, bx, py + 156, c["axis"], 1))
    return svg(w, h, "".join(b), c, t["perf_title"])


FIGURES = {"hero": hero, "architecture": architecture, "pipeline": pipeline, "performance": performance}


def main():
    for lang, strings in T.items():
        for theme, colors in THEMES.items():
            for name, fn in FIGURES.items():
                path_ = os.path.join(OUT, f"{name}-{lang}-{theme}.svg")
                with open(path_, "w", encoding="utf-8") as f:
                    f.write(fn(colors, strings))
    print(f"wrote {len(T) * len(THEMES) * len(FIGURES)} figures to {OUT}")


if __name__ == "__main__":
    main()
