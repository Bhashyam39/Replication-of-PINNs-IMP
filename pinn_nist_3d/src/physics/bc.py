"""Hard boundary conditions via a smooth Heaviside ramp.

Dirichlet faces: bottom (z = z_min) and the four side walls
(x = x_min/x_max, y = y_min/y_max). On those faces velocity is no-slip
(u = v = w = 0) and temperature is fixed at the substrate/ambient value;
pressure is left free. The top surface (z = 0) stays Neumann (laser
flux, see ``src.physics.laser``) and is never included in the ramp.
"""

from __future__ import annotations

import math
from typing import Dict, Union

import torch


def hard_heaviside(d: Union[float, torch.Tensor], eps: float) -> torch.Tensor:
    """Corrected smooth Heaviside: 0 for d<=0, 1 for d>=eps, with the
    C1 ramp ``(1 - cos(pi*d/eps)) / 2`` in between."""
    if not isinstance(d, torch.Tensor):
        d = torch.tensor(float(d), dtype=torch.float32)
    s = torch.clamp(d / float(eps), 0.0, 1.0)
    return (1.0 - torch.cos(math.pi * s)) / 2.0


def _distance_to_dirichlet(batch: Dict[str, torch.Tensor], cfg) -> torch.Tensor:
    """Distance (N,1) to the nearest Dirichlet face (bottom or side)."""
    dom = cfg.domain
    x, y, z = batch["x"], batch["y"], batch["z"]
    d_bottom = z - float(dom.z_min)
    d_xmin = x - float(dom.x_min)
    d_xmax = float(dom.x_max) - x
    d_ymin = y - float(dom.y_min)
    d_ymax = float(dom.y_max) - y
    d = torch.cat([d_bottom, d_xmin, d_xmax, d_ymin, d_ymax], dim=1)
    return d.min(dim=1, keepdim=True).values


def apply_hard_bc(raw: torch.Tensor, batch: Dict[str, torch.Tensor], cfg) -> torch.Tensor:
    """Blend raw network outputs (N,5 ordered u,v,w,p,T) with Dirichlet values.

    Returns ``h * raw + (1 - h) * bc`` componentwise where
    ``h = hard_heaviside(distance_to_dirichlet_face, bc_eps)``; the ramp
    is exactly the Dirichlet value on the face and exactly the raw output
    farther than ``bc_eps`` from it. Pressure passes through unmodified,
    and no constraint is applied at the (Neumann) top surface.
    """
    d = _distance_to_dirichlet(batch, cfg)
    h = hard_heaviside(d, cfg.domain.bc_eps)

    t_bc = torch.as_tensor(float(cfg.domain.t_ambient), dtype=raw.dtype, device=raw.device)

    out = raw.clone()
    out[:, 0:3] = h * raw[:, 0:3]  # no-slip u=v=w=0
    out[:, 3:4] = raw[:, 3:4]  # pressure unconstrained
    out[:, 4:5] = h * raw[:, 4:5] + (1.0 - h) * t_bc  # fixed T
    return out
