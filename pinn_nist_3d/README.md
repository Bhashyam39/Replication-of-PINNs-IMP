# PINN NIST 3D — public-data-only starter

A runnable **PyTorch** starter for a 3D physics-informed neural network (PINN)
inspired by Zhu, Liu & Yan ("Machine learning for metal additive
manufacturing: predicting temperature and melt pool fluid dynamics using
physics-informed neural networks"), targeted at **NIST AM-Bench 2018-02**
single-track IN625 bare-plate scans (cases A/B/C).

> **Public data only — not the paper's protocol.** The paper trained its PINN
> with dense FEM field labels (u, v, w, p, T). This repository uses **only
> public NIST AM-Bench summary data** (melt-pool lengths and cooling-rate
> definitions); the main training constraint is the PDE residual. Expect
> method-level behavior and trend validation, **not** reproduction of the
> paper's Tables 4/5. The paper used TensorFlow; this is a PyTorch
> re-implementation. All settings the paper does not report are marked
> `ASSUMPTION` in the config.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Requires Python >= 3.9 and CPU-only PyTorch is sufficient.

## Cases and coordinates

Programmed cases (paper/NIST scan plan), see `src/config/settings.py`:

| Case | Programmed power | Scan speed | AMMT actual power |
|------|------------------|------------|-------------------|
| A    | 150 W            | 0.4 m/s    | 137.9 W           |
| B    | 195 W            | 0.8 m/s    | 179.2 W           |
| C    | 195 W            | 1.2 m/s    | 179.2 W           |

`power_source: programmed | ammt_actual | cbm` selects which power is used
(CBM powers equal the programmed ones). NIST spot sizes (AMMT D4sigma
170 um / FWHM 100 um; CBM D4sigma 100 um / FWHM 59 um) can override the
paper beam radius (rb = 50 um) via `LaserConfig.rb_override` /
`apply_nist_spot`.

SI units throughout (m, s, K, W, Pa). Default local domain: x in
[-2, 2] mm (scan direction), y in [-0.3, 0.3] mm, z in [-0.2, 0] mm
(z = 0 is the top surface), t in [0, 2] ms. Moving Gaussian heat flux on the
top surface: `q = 2 Q eta / (pi rb^2) * exp(-2 ((x - x0 - V t)^2 + y^2)/rb^2)`
with `eta = 0.43`.

## Data placement (optional)

No data download happens automatically — importing any module performs no
network or disk I/O.

* Without any files, `src.data.nist_public.load_summary_csv(None)` returns
  **built-in public summary targets** (NIST experimental melt-pool lengths
  for A/B/C as tabulated in the paper's Table 4, plus the paper's PINN
  numbers labeled as model values, not targets).
* To use the actual public NIST CSVs, download them yourself (e.g. DOI
  10.18434/M31931, DOI 10.18434/mds2-3830) and place them under `data/`.
  Pass the path to `load_summary_csv(path)` or to `eval_case
  --summary-csv` (used as the comparison targets in the eval JSON). Expected
  columns are documented by `src.data.nist_public.expected_schema()` (case,
  source, power_w, speed_m_per_s, length_m, width_m, depth_m,
  cooling_rate_1290_1000_K_per_s, cooling_rate_1290_1190_K_per_s, notes;
  SI units). Missing files only fall back to the built-ins for the *summary*
  loader; explicitly requested raw files fail loudly
  (`require_raw_files` raises `FileNotFoundError`).

## Commands

```bash
# tests (CPU, < 60 s)
pytest -q

# quick smoke training run (tiny model, a few steps, writes checkpoints)
python -m src.scripts.train_case --case B --iterations 5 --smoke

# realistic short run (paper-size model 5x250, default batch sizes)
python -m src.scripts.train_case --case B --iterations 200

# evaluate a checkpoint -> JSON metrics on stdout
python -m src.scripts.eval_case --case B --checkpoint latest --smoke
python -m src.scripts.eval_case --case B \
    --checkpoint outputs/checkpoints/ckpt_B_final.pt
```

Training saves `ckpt_<CASE>_it*.pt`, `ckpt_<CASE>_final.pt` and
`latest_<CASE>.pt` under `outputs/checkpoints/` (override with
`--output-dir`).

## Expected outputs

* Training logs one line every `log_every` iterations with the total and
  per-term losses (data / momentum / continuity / energy / top Neumann /
  initial condition).
* Eval prints a JSON document with the melt-pool `length_m` / `width_m` /
  `depth_m` (T >= 1290 C solidus mask), centerline cooling rates for the two
  NIST ranges (1290 C -> 1000 C and 1290 C -> 1190 C), the checkpoint used,
  and the built-in public NIST/paper targets for comparison. All values are
  finite; if the predicted pool never reaches the NIST isotherms (e.g. a
  smoke-trained model), the cooling rates fall back to the peak centerline
  material derivative and are flagged with `"fallback": true`.

## Layout

```
src/config/settings.py    dataclasses + case registry (A/B/C, power sources)
src/materials/in625.py    IN625 props: liquid_fraction, cp, kappa, rho, mu, L
src/physics/residuals.py  momentum/continuity/energy residuals (autograd)
src/physics/laser.py      Gaussian heat flux, top Neumann residual
src/physics/bc.py         corrected smooth Heaviside, hard Dirichlet BCs
src/models/fcnn.py        FCNN (4 -> 5), Swish, Xavier init
src/losses/total.py       total loss assembly (data/PDE/top/IC/summary)
src/data/sampling.py      interior/top/dirichlet/initial batch samplers
src/data/nist_public.py   public summary schema + built-in targets (no scraping)
src/train/trainer.py      Adam training loop + checkpoints
src/eval/metrics.py       melt-pool dims, centerline cooling rates
src/scripts/              train_case / eval_case CLIs
tests/                    CPU-safe tests
```

## Deviations from the paper (summary)

1. **PyTorch** instead of TensorFlow.
2. **Public NIST summary data only** for supervision — no FEM u,p,T labels;
   PDE residuals carry the training. The loss assembly supports an optional
   sparse summary term for pointwise targets, but the public NIST data are
   melt-pool dimension/cooling-rate summaries, so the CLIs use those for
   evaluation-time comparison instead of training supervision.
3. Physics simplifications (all ASSUMPTIONs, see module docstrings):
   constant density/viscosity, no buoyancy/Marangoni/Darcy terms, kappa
   frozen inside the divergence, laser enters via the top Neumann BC only.
4. Paper-missing hyperparameters (learning rate, iterations, weights, h_conv,
   emissivity, T_inf, ...) are config values marked ASSUMPTION.
