"""Moving Gaussian laser heat flux and top-surface Neumann condition.

SPEC flux (top surface z = 0, laser moving along +x):

    q(x, y, t) = 2*Q*eta / (pi*rb^2) * exp(-2*((x - x0 - V*t)^2 + y^2) / rb^2)

``rb`` defaults to the paper value 50 um; a NIST D4sigma/FWHM spot-size
override can be set via ``LaserConfig.rb_override`` (see
``src.config.settings``).
"""

from __future__ import annotations

import math
from typing import Dict

import torch

from src.materials.in625 import kappa as _kappa
from src.materials.in625 import liquid_fraction
from src.physics.residuals import grad


def gaussian_heat_flux(x: torch.Tensor, y: torch.Tensor, t: torch.Tensor, cfg) -> torch.Tensor:
    """Surface heat flux W/m^2 at top-surface points (x, y, z=0, t)."""
    laser = cfg.laser
    Q = torch.as_tensor(float(cfg.power), dtype=x.dtype, device=x.device)
    V = torch.as_tensor(float(cfg.velocity), dtype=x.dtype, device=x.device)
    rb = torch.as_tensor(laser.effective_rb(), dtype=x.dtype, device=x.device)
    eta = torch.as_tensor(float(laser.eta), dtype=x.dtype, device=x.device)
    x0 = torch.as_tensor(float(laser.x0), dtype=x.dtype, device=x.device)

    r2 = (x - x0 - V * t) ** 2 + y**2
    peak = 2.0 * Q * eta / (math.pi * rb**2)
    return peak * torch.exp(-2.0 * r2 / rb**2)


def top_heat_residual(model, batch: Dict[str, torch.Tensor], cfg) -> torch.Tensor:
    """Neumann top-surface residual (N,1) for z = 0 points.

    Energy balance at the surface (z <= 0 into the substrate, outward
    normal +z): conduction into the material equals deposited laser flux
    minus convection/radiation losses,

        kappa * dT/dz = q_laser - h*(T - T_inf) - eps*sigma*(T^4 - T_inf^4)

    so the residual is

        r = kappa*dT/dz - q_laser + h*(T - T_inf) + eps*sigma*(T^4 - T_inf^4).

    Simplification (ASSUMPTION): kappa uses the local liquid fraction but
    is treated as frozen in the normal derivative, matching the interior
    conduction treatment.
    """
    x, y, z, t = batch["x"], batch["y"], batch["z"], batch["t"]
    out = model(torch.cat([x, y, z, t], dim=1))
    T = out[:, 4:5]

    dT_dz = grad(T, z)
    fL = liquid_fraction(T, cfg.material.Ts, cfg.material.Tl)
    kappa = _kappa(T, fL, cfg.material).detach()

    laser = cfg.laser
    q = gaussian_heat_flux(x, y, t, cfg)
    h = torch.as_tensor(float(laser.h_conv), dtype=T.dtype, device=T.device)
    eps = torch.as_tensor(float(laser.emissivity), dtype=T.dtype, device=T.device)
    sigma = torch.as_tensor(float(laser.stefan_boltzmann), dtype=T.dtype, device=T.device)
    t_inf = torch.as_tensor(float(laser.t_inf), dtype=T.dtype, device=T.device)

    conv = h * (T - t_inf)
    rad = eps * sigma * (T**4 - t_inf**4)
    return kappa * dT_dz - q + conv + rad
