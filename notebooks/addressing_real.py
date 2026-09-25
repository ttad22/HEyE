#!/usr/bin/env python3
"""Corrected block-addressing-recovery test.

The committed metric (real_platter.py:118-120) computes cosine similarity
between the enrolled ground-truth disorder tensor `dc` and a permutation of
itself, never touching the simulated camera image. Each shuffled block's best
match is bit-identical to itself, so the result is 1.0 by construction,
independent of noise, blur, ADC, or the codec. It is not a test of anything.

This script re-derives block identity from the ACTUAL simulated camera capture
(`cam_full`, post-blur/shot-noise/read-noise/ADC quantization), tiled into
blocks (`cam_bl`), correlated against the enrolled reference fingerprints
(`dc`, captured once at enrollment). This is what the addressing claim in the
paper describes; the committed code does not implement it.

Reports BOTH numbers side by side across seeds so the delta is documented, not
asserted. Writes results_platter/addressing_real.json.
"""
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import real_platter as rp

SEEDS = [1, 2, 3, 4, 5]
BITS = 2
STEPS = 1500


def corrected_whole_platter(enc, dec, chan, meta, T=8, n=rp.NCELL, se=6000, seed=1):
    """Same construction as real_platter.whole_platter, plus a corrected
    addressing metric computed from the post-channel camera image."""
    torch.manual_seed(seed)
    M = meta["M"]
    bits = meta["bits"]
    g = torch.Generator(device="cpu").manual_seed(seed)
    bi = int(torch.randint(0, rp.IMG.shape[0], (1,), generator=g))
    big = rp.IMG[bi]
    Hc = T * n
    d_cell_full = F.interpolate(big[None, None], size=(Hc, Hc), mode="area")
    d_img_full = F.interpolate(big[None, None], size=(Hc * rp.PPV, Hc * rp.PPV), mode="area")
    dc = d_cell_full.reshape(1, 1, T, n, T, n).permute(0, 2, 4, 1, 3, 5).reshape(T * T, 1, n, n)
    npx = n * rp.PPV
    total_sym = T * T * n * n
    nbytes = (total_sym * bits) // 8
    rng = np.random.default_rng(seed)
    payload = rng.integers(0, 256, nbytes, dtype=np.uint8).tobytes()
    bs = np.unpackbits(np.frombuffer(payload, np.uint8))[: total_sym * bits]
    w = 1 << np.arange(bits - 1, -1, -1)
    syms = (bs.reshape(-1, bits) * w).sum(1).reshape(T * T, n, n)
    sym = torch.from_numpy(syms).long().to(rp.DEV)
    sp = rp.snr_plane(se, (T * T, 1, n, n))
    lvl = enc(sym, dc, sp)
    lvl_img = F.interpolate(lvl, scale_factor=rp.PPV, mode="nearest")
    lvl_full = lvl_img.reshape(1, T, T, 1, npx, npx).permute(0, 3, 1, 4, 2, 5).reshape(1, 1, Hc * rp.PPV, Hc * rp.PPV)
    cam_full = chan(lvl_full, d_img_full, se)  # actual simulated camera capture, post channel
    cam_bl = cam_full.reshape(1, 1, T, npx, T, npx).permute(0, 2, 4, 1, 3, 5).reshape(T * T, 1, npx, npx)
    pred = dec(cam_bl, dc, sp).argmax(1)
    ber = rp.berf(dec(cam_bl, dc, sp), sym, bits)
    pr = pred.reshape(T * T, n * n).reshape(-1).cpu().numpy()[:total_sym]
    mb = ((pr[:, None] >> np.arange(bits - 1, -1, -1)) & 1).astype(np.uint8)
    rec = np.packbits(mb.reshape(-1)[: nbytes * 8]).tobytes()
    byte_acc = float(np.mean(np.frombuffer(rec, np.uint8) == np.frombuffer(payload, np.uint8)))

    # -- ORIGINAL (tautological) metric: enrolled tensor vs itself, permuted --
    feats_enrolled = dc.reshape(T * T, -1)
    feats_enrolled = feats_enrolled - feats_enrolled.mean(1, keepdim=True)
    feats_enrolled = feats_enrolled / (feats_enrolled.norm(dim=1, keepdim=True) + 1e-8)
    perm = torch.randperm(T * T, device=rp.DEV)
    shuf_enrolled = feats_enrolled[perm]
    sim_orig = shuf_enrolled @ feats_enrolled.t()
    match_orig = sim_orig.argmax(1)
    recovered_original_tautological = (match_orig == perm).float().mean().item()

    # -- CORRECTED metric: query = actual noisy camera-observed block,        --
    # -- reference = enrolled clean fingerprint dc. Shuffle the QUERY blocks  --
    # -- (simulating out-of-order capture) and match each against the        --
    # -- enrolled reference database.                                        --
    # Downsample the observed camera block back to cell resolution so it is
    # directly comparable to the enrolled reference resolution.
    cam_cell = F.interpolate(cam_bl, size=(n, n), mode="area")
    feats_query = cam_cell.reshape(T * T, -1)
    feats_query = feats_query - feats_query.mean(1, keepdim=True)
    feats_query = feats_query / (feats_query.norm(dim=1, keepdim=True) + 1e-8)
    feats_ref = dc.reshape(T * T, -1)
    feats_ref = feats_ref - feats_ref.mean(1, keepdim=True)
    feats_ref = feats_ref / (feats_ref.norm(dim=1, keepdim=True) + 1e-8)
    perm2 = torch.randperm(T * T, device=rp.DEV)
    shuf_query = feats_query[perm2]  # simulate the captured blocks arriving out of order
    sim_real = shuf_query @ feats_ref.t()
    match_real = sim_real.argmax(1)
    recovered_from_camera = (match_real == perm2).float().mean().item()

    return dict(
        seed=seed,
        whole_platter_ber=round(ber, 6),
        whole_platter_byte_acc=round(byte_acc, 4),
        addressing_recovery_TAUTOLOGICAL_original_code=round(recovered_original_tautological, 4),
        addressing_recovery_FROM_CAMERA_IMAGE_corrected=round(recovered_from_camera, 4),
    )


out = {"bits": BITS, "steps": STEPS, "seeds": SEEDS, "runs": [],
       "note": "addressing_recovery_TAUTOLOGICAL_original_code reproduces the "
               "committed metric (real_platter.py:118-120): it self-matches the "
               "enrolled ground-truth tensor against a permutation of itself and "
               "never touches the simulated camera image, so it is 1.0 by "
               "construction. addressing_recovery_FROM_CAMERA_IMAGE_corrected "
               "matches the actual simulated camera-observed block (post blur, "
               "shot noise, read noise, ADC quantization) against the enrolled "
               "reference database, which is what the paper's claim describes."}
t0 = time.time()

for s in SEEDS:
    torch.manual_seed(s)
    np.random.seed(s)
    torch.cuda.manual_seed_all(s)
    e, d, c, m = rp.train(BITS, steps=STEPS)
    r = corrected_whole_platter(e, d, c, m, T=8, seed=s)
    out["runs"].append(r)
    print(f"[seed {s}] ber={r['whole_platter_ber']:.6f} "
          f"tautological={r['addressing_recovery_TAUTOLOGICAL_original_code']:.4f} "
          f"CORRECTED_from_camera={r['addressing_recovery_FROM_CAMERA_IMAGE_corrected']:.4f}",
          flush=True)
    del e, d, c
    torch.cuda.empty_cache()


def ci95(vals):
    n = len(vals)
    mean = sum(vals) / n
    if n < 2:
        return mean, 0.0, 0.0
    sd = math.sqrt(sum((v - mean) ** 2 for v in vals) / (n - 1))
    tcrit = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571}.get(n - 1, 1.96)
    return mean, sd, tcrit * sd / math.sqrt(n)


summary = {}
for k in ("addressing_recovery_TAUTOLOGICAL_original_code",
          "addressing_recovery_FROM_CAMERA_IMAGE_corrected",
          "whole_platter_ber"):
    vals = [r[k] for r in out["runs"]]
    mean, sd, half = ci95(vals)
    summary[k] = {"mean": mean, "sd": sd, "ci95_halfwidth": half, "n": len(vals)}
out["summary"] = summary
out["minutes"] = round((time.time() - t0) / 60, 1)

os.makedirs("../results/multiseed_final", exist_ok=True)
with open("../results/multiseed_final/addressing_real.json", "w") as f:
    json.dump(out, f, indent=2)

print("\n=== ADDRESSING RECOVERY: tautological vs corrected (n=%d seeds) ===" % len(SEEDS))
for k, v in summary.items():
    print(f"  {k:48s} {v['mean']:.4f} +/- {v['ci95_halfwidth']:.4f}")
print("minutes:", out["minutes"])
