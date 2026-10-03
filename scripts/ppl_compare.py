#!/usr/bin/env python3
"""Compare `nmoe ppl` dumps (NMOE_PPL_DUMP) against a baseline dump.

Each record: u32 target, f32 target_logprob, u32 ids[32], f32 logprobs[32].
KL(base || test) is approximated over the baseline's top-32 tokens plus one
bucket holding the remaining probability mass.

Usage: python3 scripts/ppl_compare.py base.bin test1.bin [test2.bin ...]
"""

import sys

import numpy as np

TOPN = 32
REC = np.dtype([("target", "<u4"), ("lp", "<f4"), ("ids", "<u4", TOPN), ("lps", "<f4", TOPN)])


def load(path):
    return np.fromfile(path, dtype=REC)


def summarize(name, d):
    nll = -d["lp"].mean()
    top1 = (d["ids"][:, 0] == d["target"]).mean()
    return f"{name}: n={len(d)} ppl={np.exp(nll):.4f} nll={nll:.5f} top1={top1:.4f}"


def compare(base, test):
    n = min(len(base), len(test))
    base, test = base[:n], test[:n]
    agree = (base["ids"][:, 0] == test["ids"][:, 0]).mean()
    kls = []
    for b, t in zip(base, test):
        pb = np.exp(b["lps"].astype(np.float64))
        tmap = dict(zip(t["ids"].tolist(), t["lps"].tolist()))
        floor = float(t["lps"][-1])  # unseen tokens get at most the test's 32nd logprob
        lt = np.array([tmap.get(i, floor) for i in b["ids"].tolist()], dtype=np.float64)
        pt = np.exp(lt)
        kl = float((pb * (b["lps"] - lt)).sum())
        rb, rt = max(1e-12, 1 - pb.sum()), max(1e-12, 1 - min(pt.sum(), 1 - 1e-12))
        kl += rb * np.log(rb / rt)
        kls.append(max(kl, 0.0))
    kls = np.array(kls)
    dnll = (-test["lp"]).mean() - (-base["lp"]).mean()
    return f"  vs base: top1_agree={agree:.4f} mean_kl={kls.mean():.5f} p99_kl={np.percentile(kls, 99):.4f} dNLL={dnll:+.5f}"


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    base = load(sys.argv[1])
    print(summarize(sys.argv[1], base))
    for path in sys.argv[2:]:
        d = load(path)
        print(summarize(path, d))
        print(compare(base, d))


if __name__ == "__main__":
    main()
