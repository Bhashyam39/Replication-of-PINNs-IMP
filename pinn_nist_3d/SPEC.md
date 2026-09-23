# SPEC — Public-data-only 3D NIST PINN starter

## Goal
Implement a runnable PyTorch starter for a 3D PINN inspired by Zhu/Liu/Yan, targeted at NIST AM-Bench 2018-02 single-track IN625 bare-plate scans. This is **not** the paper's exact FEM-trained protocol. Because the user selected public NIST data only, labeled supervision is sparse experimental/summary data; PDE residuals provide the main constraint.

## Hard deviations from the paper
- Framework: PyTorch, although the paper used TensorFlow.
- Supervision: public NIST AM-Bench summary/CSV data only. No FEM u,p,T field labels.
- Expected outcome: method-level implementation and trend validation, not exact Table 4/5 reproduction.
- Missing paper settings are config values marked `ASSUMPTION`.

## Coordinate and units
- SI units internally: m, s, K, W, Pa.
- Domain default: x along scan, y transverse, z depth (z<=0 into substrate, top surface z=0).
- Default local domain: x in [-2e-3, 2e-3] m, y in [-3e-4, 3e-4] m, z in [-2e-4, 0] m, t in [0, 2e-3] s. Configurable.
- Laser moves along +x: q(x,y,z=0,t)=2*Q*eta/(pi*rb^2)*exp(-2*((x-x0-V*t)^2+y^2)/rb^2).
- Default use paper beam radius rb=50e-6 m and eta=0.43; provide NIST spot-size override because NIST reports AMMT D4sigma 170 um/FWHM 100 um and CBM D4sigma 100 um/FWHM 59 um.

## Cases
Programmed cases from paper/NIST: A=150 W,0.4 m/s; B=195 W,0.8 m/s; C=195 W,1.2 m/s. Public NIST note: AMMT actual powers were 137.9/179.2/179.2 W; CBM powers as programmed. Config must expose `power_source: programmed|ammt_actual|cbm`.

## Modules and required files
1. `src/config/settings.py`
   - Dataclasses: `MaterialProps`, `LaserConfig`, `DomainConfig`, `TrainConfig`, `CaseConfig`.
   - Function `load_config(path)->dict` and `get_case(name, power_source)->CaseConfig`.
2. `src/materials/in625.py`
   - `liquid_fraction(T, Ts, Tl)`, `cp(T,fL)`, `kappa(T,fL)`, `rho()`, `mu()`, `latent_heat()`, all torch-tensor compatible and differentiable where sensible.
3. `src/physics/residuals.py`
   - `grad(outputs, inputs)` helper using torch.autograd.
   - `residuals(model, batch, cfg)->dict` with keys `r_mom` (3,), `r_cont` (1,), `r_energy` (1,), plus derived `fL`.
   - Energy residual must include latent heat term using apparent capacity form to remain AD-stable: d[(cp*T + L*fL)]/dt + u.grad(cp*T + L*fL) - div(kappa grad T) - Q = 0. Document if simplified.
4. `src/physics/laser.py`
   - `gaussian_heat_flux(x,y,t,cfg)` and `top_heat_residual(model,batch,cfg)`.
5. `src/physics/bc.py`
   - `hard_heaviside(d, eps)` using corrected `(1-cos(pi*d/eps))/2` ramp.
   - `apply_hard_bc(raw, batch, cfg)` for fixed T/no-slip on bottom/side Dirichlet faces; top surface remains Neumann.
6. `src/models/fcnn.py`
   - `FCNN(in_dim=4,out_dim=5,hidden=250,layers=5)` with Swish; output order u,v,w,p,T.
   - `init_weights` using Xavier.
7. `src/losses/total.py`
   - `compute_losses(model,batch,cfg)->(total, parts)` including data, pde interior, Neumann top, initial condition, optional sparse public summary losses.
8. `src/data/sampling.py`
   - `sample_interior(n,cfg)`, `sample_top(n,cfg)`, `sample_dirichlet(n,cfg)`, `sample_initial(n,cfg)` returning dict of torch tensors with requires_grad.
9. `src/data/nist_public.py`
   - `expected_schema()` documenting CSV columns.
   - `load_summary_csv(path)` tolerant loader; if file absent, return built-in paper/NIST summary targets for A/B/C with source labels.
   - No automatic scraping required; fail loudly if user-requested raw files are missing.
10. `src/train/trainer.py`
   - `train(cfg)` Adam loop, configurable iterations, batch sizes, weights, lr; logs every `log_every`; saves checkpoint to `outputs/checkpoints/`.
11. `src/eval/metrics.py`
   - `melt_pool_mask(T,cfg)`, `length_width_depth(mask, coords)`, `cooling_rate_centerline(model,cfg)` using paper Eq. 29 style and NIST 1290C→1000C/1190C variants.
12. `src/scripts/train_case.py`
   - CLI: `--case B --iterations 200 --smoke`.
13. `src/scripts/eval_case.py`
   - CLI loads checkpoint and prints metrics JSON.
14. `tests/test_core.py`
   - CPU-safe tests: swish sign, hard_heaviside endpoints, liquid_fraction bounds, laser flux peak, residual shapes on tiny random batch, summary loader fallback.
15. `README.md`
   - setup, data placement, commands, deviations, expected outputs.

## Interface contract
- Batch dict keys: `x,y,z,t` each shape `(N,1)`; optional labels `u,v,w,p,T` same shapes; `surface` string in {`interior`,`top`,`dirichlet`,`initial`}.
- Model forward accepts tensor `(N,4)` ordered `(x,y,z,t)` and returns `(N,5)` ordered `(u,v,w,p,T)`.
- All public functions must be importable without downloading data.
- Tests must run with `pytest -q` in under 60 seconds on CPU.

## Validation gate
- `pytest -q` passes.
- `python -m src.scripts.train_case --case B --iterations 5 --smoke` runs without error and writes a checkpoint.
- `python -m src.scripts.eval_case --case B --checkpoint <latest> --smoke` prints JSON with finite values.
