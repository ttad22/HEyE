#!/usr/bin/env python3
"""Multi-seed confidence intervals for the imperfection-conditioning ablation.

Reuses rw_codec_real.py's exact train()/evalat() definitions. Config matches the
committed ablation_cond.json: bits=4, signal_e=2600, cond=True vs cond=False.
Writes results_rw/ablation_cond_multiseed.json; does not overwrite the original.
"""
import json
import math
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rw_codec_real as rc

SEEDS = [1, 2, 3, 4, 5]
BITS = 4
SIGNAL = 2600
STEPS = 2000

out = {"bits": BITS, "signal_e": SIGNAL, "steps": STEPS, "seeds": SEEDS, "runs": []}
t0 = time.time()

for s in SEEDS:
    row = {"seed": s}
    for cond in (True, False):
        torch.manual_seed(s)
        np.random.seed(s)
        torch.cuda.manual_seed_all(s)
        e, d, c, m = rc.train(BITS, steps=STEPS, use_encoder=True, cond=cond)
        ber_val = rc.evalat(e, d, c, m, SIGNAL)
        key = "cond" if cond else "no_cond"
        row[key] = {"ber": ber_val}
        del e, d, c
        torch.cuda.empty_cache()
    row["delta_ber"] = row["no_cond"]["ber"] - row["cond"]["ber"]
    out["runs"].append(row)
    print(f"[seed {s}] cond_ber={row['cond']['ber']:.6f} "
          f"no_cond_ber={row['no_cond']['ber']:.6f} "
          f"delta={row['delta_ber']:.6f}", flush=True)


def ci95(vals):
    n = len(vals)
    mean = sum(vals) / n
    if n < 2:
        return mean, 0.0, 0.0
    sd = math.sqrt(sum((v - mean) ** 2 for v in vals) / (n - 1))
    tcrit = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
             6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262}.get(n - 1, 1.96)
    half = tcrit * sd / math.sqrt(n)
    return mean, sd, half


summary = {}
for key, getter in (("cond_ber", lambda r: r["cond"]["ber"]),
                    ("no_cond_ber", lambda r: r["no_cond"]["ber"]),
                    ("delta_ber", lambda r: r["delta_ber"])):
    vals = [getter(r) for r in out["runs"]]
    mean, sd, half = ci95(vals)
    summary[key] = {"mean": mean, "sd": sd, "ci95_halfwidth": half,
                    "lo": mean - half, "hi": mean + half,
                    "min": min(vals), "max": max(vals), "n": len(vals)}
out["summary"] = summary
out["minutes"] = round((time.time() - t0) / 60, 1)

os.makedirs("../results/multiseed_final", exist_ok=True)
with open("../results/multiseed_final/ablation_cond_multiseed.json", "w") as f:
    json.dump(out, f, indent=2)

print("\n=== ABLATION SUMMARY (n=%d seeds) ===" % len(SEEDS))
for k, v in summary.items():
    print(f"  {k:14s} {v['mean']:.6f} +/- {v['ci95_halfwidth']:.6f} "
          f"(95% CI [{v['lo']:.6f}, {v['hi']:.6f}], sd={v['sd']:.6f})")
print("minutes:", out["minutes"])
