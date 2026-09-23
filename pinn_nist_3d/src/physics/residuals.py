"""PDE residuals for the 3D IN625 melt-pool PINN.

Simplifications relative to the full paper FEM model (all ASSUMPTIONs,
kept AD-stable on CPU):

* constant density ``rho`` and constant liquid viscosity ``mu``
  (Boussinesq-style; no buoyancy, Marangoni, or mushy-zone Darcy terms —
  the paper does not publish those coefficients),
* energy equation in apparent-capacity form,
  ``d(cp*T + L*fL)/dt + u.grad(cp*T + L*fL) - div(kappa grad T) - Q = 0``,
  differentiated directly with autograd so latent heat stays AD-stable,
* ``div(kappa grad T)`` uses kappa evaluated at the local state but
  treated as frozen (stop-gradient) in the divergence to avoid
  third-order derivatives,
* the laser enters only through the top-surface Neumann condition
  (``src.physics.laser``), so the interior volumetric source Q = 0.
"""

from __future__ import annotations

from typing import Dict

import torch

from src.materials.in625 import cp as _cp
from src.materials.in625 import kappa as _kappa
from src.materials.in625 import liquid_fraction


def grad(outputs: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
    """d(outputs)/d(inputs) with the graph kept for higher-order derivatives."""
    return torch.autograd.grad(
        outputs,
        inputs,
        grad_outputs=torch.ones_like(outputs),
        create_graph=True,
        retain_graph=True,
    )[0]


def _model_outputs(model, batch: Dict[str, torch.Tensor]):
    x, y, z, t = batch["x"], batch["y"], batch["z"], batch["t"]
    xt = torch.cat([x, y, z, t], dim=1)  # (N,4) ordered (x,y,z,t)
    out = model(xt)  # (N,5) ordered (u,v,w,p,T)
    u, v, w, p, T = out[:, 0:1], out[:, 1:2], out[:, 2:3], out[:, 3:4], out[:, 4:5]
    return x, y, z, t, u, v, w, p, T


def _laplacian(f, x, y, z):
    return grad(grad(f, x), x) + grad(grad(f, y), y) + grad(grad(f, z), z)


def residuals(model, batch: Dict[str, torch.Tensor], cfg) -> Dict[str, torch.Tensor]:
    """Interior PDE residuals for an (x, y, z, t) batch.

    Returns a dict with keys ``r_mom`` (N,3), ``r_cont`` (N,1),
    ``r_energy`` (N,1) and the derived liquid fraction ``fL`` (N,1).
    """
    mat = cfg.material
    x, y, z, t, u, v, w, p, T = _model_outputs(model, batch)

    fL = liquid_fraction(T, mat.Ts, mat.Tl)
    rho = torch.as_tensor(float(mat.rho), dtype=T.dtype, device=T.device)
    mu = torch.as_tensor(float(mat.mu), dtype=T.dtype, device=T.device)
    L = torch.as_tensor(float(mat.latent_heat), dtype=T.dtype, device=T.device)

    # --- continuity: div(u) = 0 ---
    r_cont = grad(u, x) + grad(v, y) + grad(w, z)

    # --- momentum (constant rho/mu, no body-force terms; see module docstring) ---
    def momentum_comp(f, p_grad):
        adv = u * grad(f, x) + v * grad(f, y) + w * grad(f, z)
        return rho * (grad(f, t) + adv) + p_grad - mu * _laplacian(f, x, y, z)

    r_mx = momentum_comp(u, grad(p, x))
    r_my = momentum_comp(v, grad(p, y))
    r_mz = momentum_comp(w, grad(p, z))
    r_mom = torch.cat([r_mx, r_my, r_mz], dim=1)  # (N,3)

    # --- energy: apparent-capacity form, latent heat inside the derivative ---
    cp_T = _cp(T, fL, mat)
    h_app = cp_T * T + L * fL  # J/kg, differentiable apparent enthalpy
    dh_dt = grad(h_app, t)
    dh_dx = grad(h_app, x)
    dh_dy = grad(h_app, y)
    dh_dz = grad(h_app, z)
    # kappa frozen (stop-grad) in the divergence -> second-order AD only.
    kappa = _kappa(T, fL, mat).detach()
    conduction = kappa * _laplacian(T, x, y, z)
    Q_vol = torch.zeros_like(T)  # laser enters via top Neumann BC, not volumetric
    r_energy = rho * (dh_dt + u * dh_dx + v * dh_dy + w * dh_dz) - conduction - Q_vol

    return {"r_mom": r_mom, "r_cont": r_cont, "r_energy": r_energy, "fL": fL}
