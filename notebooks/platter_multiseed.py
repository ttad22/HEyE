#!/usr/bin/env python3
"""Multi-seed confidence intervals for the whole-platter result.

Reuses real_platter.py's exact Enc/Dec/Chan/train/whole_platter definitions by
importing it as a module, so the only thing that changes versus the committed
single-seed run is that training AND evaluation are both seeded and repeated.

Writes results_platter/platter_multiseed.json. Does not overwrite platter.json.
"""
import json
import math
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import real_platter as rp  # module-level code loads the real disorder images

SEEDS = [1, 2, 3, 4, 5]
BITS = 2
STEPS = 1500

out = {"bits": BITS, "steps": STEPS, "seeds": SEEDS, "runs": []}
t0 = time.time()

for s in SEEDS:
    torch.manual_seed(s)
    np.random.seed(s)
    torch.cuda.manual_seed_all(s)
    e, d, c, m = rp.train(BITS, steps=STEPS)
    st, _viz = rp.whole_platter(e, d, c, m, T=8, seed=s)
    st["seed"] = s
    out["runs"].append(st)
    print(f"[seed {s}] ber={st['whole_platter_ber']:.6f} "
          f"byte_acc={st['whole_platter_byte_acc']:.4f} "
          f"addr={st['block_addressing_recovery']:.4f} "
          f"ser_mean={st['per_block_ser_mean']:.4f}", flush=True)
    del e, d, c
    torch.cuda.empty_cache()


def ci95(vals):
    n = len(vals)
    mean = sum(vals) / n
    if n < 2:
        return mean, 0.0, 0.0
    sd = math.sqrt(sum((v - mean) ** 2 for v in vals) / (n - 1))
    # t critical values for 95% two-sided, df = n-1
    tcrit = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
             6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262}.get(n - 1, 1.96)
    half = tcrit * sd / math.sqrt(n)
    return mean, sd, half


summary = {}
for k in ("whole_platter_ber", "whole_platter_byte_acc",
          "per_block_ser_mean", "per_block_ser_max",
          "block_addressing_recovery"):
    vals = [r[k] for r in out["runs"]]
    mean, sd, half = ci95(vals)
    summary[k] = {"mean": mean, "sd": sd, "ci95_halfwidth": half,
                  "lo": mean - half, "hi": mean + half,
                  "min": min(vals), "max": max(vals), "n": len(vals)}
out["summary"] = summary
out["minutes"] = round((time.time() - t0) / 60, 1)

os.makedirs("../results/multiseed_final", exist_ok=True)
with open("../results/multiseed_final/platter_multiseed.json", "w") as f:
    json.dump(out, f, indent=2)

print("\n=== SUMMARY (n=%d seeds) ===" % len(SEEDS))
for k, v in summary.items():
    print(f"  {k:32s} {v['mean']:.6f} +/- {v['ci95_halfwidth']:.6f} "
          f"(95% CI [{v['lo']:.6f}, {v['hi']:.6f}], sd={v['sd']:.6f})")
print("minutes:", out["minutes"])
