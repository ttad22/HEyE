# Heaven's Eye

A rewritable optical block device on commodity glass. A fixed camera and a small
on-device neural codec read and write data on ordinary glass, using the glass's own
manufacturing imperfections as a shared write/read coordinate frame. No laser, no
etching, no moving parts.

This repository contains the paper source, the learned codec, the differentiable
optical channel simulator, and every committed result artifact behind the paper's
claims, each traceable to the script that produced it.

## What's real vs. simulated

The disorder maps used throughout are real (a public optical-PUF speckle-image
response set). Payload modulation and camera-frame formation are calibrated
simulation. This distinction is stated explicitly everywhere a number is reported;
see `paper/fixed_optics_glass_fast.tex` §4 and §10 for the full scope statement.

## Repository layout

```
paper/                  LaTeX source, bibliography, compiled figures
  fixed_optics_glass_fast.tex
  refs.bib
notebooks/               training / evaluation scripts (GPU required)
  real_platter.py         whole-platter experiment (the platter.json result)
  rw_codec_real.py        learned codec sweep + conditioning ablation
  platter_multiseed.py    5-seed re-run of the platter experiment
  ablation_multiseed.py   5-seed re-run of the conditioning ablation
  addressing_real.py      corrected block-addressing-recovery test
figures/                  matplotlib figure generators (CPU, no GPU needed)
  device.py                system architecture diagram
  headline_figures.py      latency, BER/SNR, frontier charts
results/                  committed result artifacts (JSON), one per experiment
  rw_final/                 codec sweep, conditioning ablation
  platter_final/            whole-platter result
  stack_final/               multi-layer stacking
  multiseed_final/          5-seed re-runs, see below
sim/                      analytic/system-level models (latency, energy, landmarks)
```

## Getting Started Instructions

A ~5-minute check that the artifact works, no GPU required:

```bash
pip install -r requirements.txt
python3 figures/headline_figures.py   # regenerates the latency-ladder and BER/SNR figures
python3 figures/device.py             # regenerates the architecture diagram
```

Each command reads a committed result file under `results/` or `sim/` and writes a
`.png` next to the script. If both run without error and produce non-empty PNGs, the
artifact's non-GPU path is working.

## Claims

Each numbered item is a specific claim in the paper, with the exact file or command
that backs it.

1. **Whole-platter read: BER $6.5\times10^{-3}$, 4,096-byte payload, 64 blocks**
   (paper §6, Fig. 4). Source: `results/platter_final/platter.json`. Regenerate:
   `python3 notebooks/real_platter.py` (GPU required).
2. **Conditioning ablation: BER 0.13 (conditioned) vs. 0.40 (unconditioned) at
   4 bits/cell** (paper §5). Source: `results/rw_final/ablation_cond.json`.
   Regenerate: `python3 notebooks/rw_codec_real.py` (GPU required).
3. **Learned write beats identity write across the SNR sweep** (paper §5, Fig. 3).
   Source: `results/rw_final/codec_sweep.csv`. Figure: `python3
   figures/headline_figures.py` (no GPU; reads the committed CSV).
4. **Random-read latency $\approx$17 ms, modeled** (paper §6, Fig. 2). This is an
   analytic model, not a training result. Source: `sim/results/
   fixed_optics_frontier.csv`. Figure: `python3 figures/headline_figures.py`.
5. **Block-addressing recovery: known discrepancy between the originally reported
   metric and a corrected one.** The paper and this repo report both numbers rather
   than resolving to one. The original `block_addressing_recovery` field in
   `results/platter_final/platter.json` (1.0, i.e. 100%) is a metric that matches the
   enrolled ground-truth tensor against a permutation of itself and returns 1.0 by
   construction, independent of the actual channel or codec. `results/
   multiseed_final/addressing_real.json` reports a corrected version, matching the
   actual simulated camera-observed block against the enrolled reference: roughly
   8% recovery (95% CI, 5 seeds). Regenerate either with `python3
   notebooks/addressing_real.py` (GPU required; reports both metrics side by side).
6. **5-seed confidence intervals for the whole-platter and conditioning-ablation
   results.** Source: `results/multiseed_final/platter_multiseed.json` and
   `ablation_cond_multiseed.json`. These re-run the experiments in claims 1 and 2
   with both training and evaluation seeded (the single-seed originals seeded only
   evaluation). Regenerate: `python3 notebooks/platter_multiseed.py` (66 min,
   measured) and `ablation_multiseed.py` (114 min, measured) on the hardware below.
7. **Codec footprint: 0.17 MB, ~43k parameters, ~0.38 MFLOP/cell.** Source:
   `results/rw_final/codec_summary.json`. These are static counts, not something that
   needs a GPU to verify: inspect the JSON directly, or recompute from the model
   definition in `notebooks/rw_codec_real.py`.

## Detailed Instructions

To regenerate a GPU result from scratch instead of trusting the committed JSON:

```bash
# GPU required (CUDA); training scripts assert this and refuse to run on CPU
python3 notebooks/real_platter.py        # -> results/platter_final/platter.json  (9 min, measured)
python3 notebooks/rw_codec_real.py       # -> results/rw_final/codec_summary.json, ablation_cond.json  (runtime not recorded by the script)
python3 notebooks/platter_multiseed.py   # -> results/multiseed_final/platter_multiseed.json  (66 min, measured, 5 seeds)
python3 notebooks/ablation_multiseed.py  # -> results/multiseed_final/ablation_cond_multiseed.json  (114 min, measured, 5 seeds x 2 trainings)
python3 notebooks/addressing_real.py     # -> results/multiseed_final/addressing_real.json  (62 min, measured, 5 seeds)
```

Runtimes above were measured on 2x NVIDIA GTX 1080 Ti (11 GB), CUDA 12.4, PyTorch
2.5.1, Python 3.10; a single GPU is sufficient, the scripts do not require multi-GPU.
Expect different wall-clock time on other hardware, but the same relative ordering.

Figures regenerate from the committed JSON/CSV without a GPU:

```bash
python3 figures/headline_figures.py
python3 figures/device.py
```

## Multi-seed results

`results/multiseed_final/` holds 5-seed re-runs (both training and evaluation seeded,
unlike the single-seed committed results elsewhere) with 95% confidence intervals, for
the whole-platter experiment, the conditioning ablation, and the corrected addressing
test. See Claims 5 and 6 above.

## Requirements

Tested on:
- Python 3.10
- PyTorch 2.5.1 with CUDA 12.4, on 2x NVIDIA GTX 1080 Ti (11 GB): for anything in
  `notebooks/`. A single GPU with at least ~4 GB free is expected to be sufficient
  (the models are small; see Claim 7), but only the above was actually tested.
- matplotlib, numpy, pandas, for anything in `figures/` or `sim/` (no GPU needed)

See `requirements.txt` for exact package versions.

## Data

The platter experiment's disorder maps are drawn unmodified from a public
optical-PUF response image set (Zenodo record 8377156) and are not redistributed
here; `notebooks/real_train.py` fetches them.

## Contributing

Issues and pull requests are welcome: see `CONTRIBUTING.md`. In particular:

- A physical light-driven write on a coated glass coupon (specified as a low-cost,
  falsifiable test) is the open experimental question this paper leaves for future
  work.
- A learned (rather than naive correlation-based) block-addressing matcher is an open
  problem; see Claim 5 above.
- A held-out-glass train/test split for every learned result (train and eval
  currently draw from the same 300-image disorder pool) is not yet done; see the
  paper's Limitations section.

## License

MIT. See `LICENSE`.
