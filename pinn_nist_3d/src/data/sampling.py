"""Collocation-point samplers for the 3D NIST PINN (SPEC item 8).

Each sampler returns a SPEC batch dict:

    {"x": (N,1), "y": (N,1), "z": (N,1), "t": (N,1), "surface": <str>}

with torch float32 tensors that have ``requires_grad=True`` (needed by the
autograd-based PDE residuals). All coordinates are SI (m, s) drawn from the
``cfg.domain`` box:

    x in [x_min, x_max]  (scan direction)
    y in [y_min, y_max]  (transverse)
    z in [z_min, z_max]  (depth; z = z_max = 0 is the top surface)
    t in [t_min, t_max]

``cfg`` may be a ``config.settings.CaseConfig`` (attribute access) or a dict
with a ``"domain"`` section (mapping access), matching the tolerance of the
trainer/loss modules.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

import torch

__all__ = [
    "sample_interior",
    "sample_top",
    "sample_dirichlet",
    "sample_initial",
]

#: Dirichlet faces (bottom + four sides); the top face stays Neumann.
_DIRICHLET_FACES = ("bottom", "x_min", "x_max", "y_min", "y_max")


def _get(obj: Any, key: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, Mapping):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _domain(cfg: Any):
    dom = _get(cfg, "domain", cfg)
    return {
        "x_min": float(_get(dom, "x_min", -2e-3)),
        "x_max": float(_get(dom, "x_max", 2e-3)),
        "y_min": float(_get(dom, "y_min", -3e-4)),
        "y_max": float(_get(dom, "y_max", 3e-4)),
        "z_min": float(_get(dom, "z_min", -2e-4)),
        "z_max": float(_get(dom, "z_max", 0.0)),
        "t_min": float(_get(dom, "t_min", 0.0)),
        "t_max": float(_get(dom, "t_max", 2e-3)),
    }


def _uniform(
    n: int,
    lo: float,
    hi: float,
    generator: Optional[torch.Generator],
    requires_grad: bool = True,
) -> torch.Tensor:
    u = torch.rand(n, 1, generator=generator, dtype=torch.float32)
    out = lo + (hi - lo) * u
    return out.requires_grad_(requires_grad)


def _batch(
    x: torch.Tensor,
    y: torch.Tensor,
    z: torch.Tensor,
    t: torch.Tensor,
    surface: str,
) -> dict:
    return {"x": x, "y": y, "z": z, "t": t, "surface": surface}


def sample_interior(
    n: int, cfg: Any, generator: Optional[torch.Generator] = None
) -> dict:
    """Uniform random points in the full 4D (x, y, z, t) domain interior."""
    d = _domain(cfg)
    return _batch(
        _uniform(n, d["x_min"], d["x_max"], generator),
        _uniform(n, d["y_min"], d["y_max"], generator),
        _uniform(n, d["z_min"], d["z_max"], generator),
        _uniform(n, d["t_min"], d["t_max"], generator),
        "interior",
    )


def sample_top(
    n: int, cfg: Any, generator: Optional[torch.Generator] = None
) -> dict:
    """Points on the top (Neumann) surface z = z_max with laser flux BC."""
    d = _domain(cfg)
    z = torch.full((n, 1), d["z_max"], dtype=torch.float32).requires_grad_(True)
    return _batch(
        _uniform(n, d["x_min"], d["x_max"], generator),
        _uniform(n, d["y_min"], d["y_max"], generator),
        z,
        _uniform(n, d["t_min"], d["t_max"], generator),
        "top",
    )


def sample_dirichlet(
    n: int, cfg: Any, generator: Optional[torch.Generator] = None
) -> dict:
    """Points on the Dirichlet faces (bottom + four side walls).

    Each point is placed on a uniformly chosen face: the face coordinate is
    fixed at the boundary value, the other spatial coordinates are uniform
    over the face, and ``t`` is uniform over the time window. The top surface
    (z = z_max) is never sampled here; it is Neumann.
    """
    d = _domain(cfg)
    x = _uniform(n, d["x_min"], d["x_max"], generator)
    y = _uniform(n, d["y_min"], d["y_max"], generator)
    z = _uniform(n, d["z_min"], d["z_max"], generator)
    t = _uniform(n, d["t_min"], d["t_max"], generator)

    face_idx = torch.randint(
        low=0, high=len(_DIRICHLET_FACES), size=(n,), generator=generator
    )
    bounds = {
        "bottom": ("z", d["z_min"]),
        "x_min": ("x", d["x_min"]),
        "x_max": ("x", d["x_max"]),
        "y_min": ("y", d["y_min"]),
        "y_max": ("y", d["y_max"]),
    }
    tensors = {"x": x, "y": y, "z": z}
    for i, face in enumerate(_DIRICHLET_FACES):
        key, value = bounds[face]
        mask = face_idx == i
        if mask.any():
            tensors[key] = tensors[key].clone()
            tensors[key][mask] = float(value)
    return _batch(tensors["x"], tensors["y"], tensors["z"], t, "dirichlet")


def sample_initial(
    n: int, cfg: Any, generator: Optional[torch.Generator] = None
) -> dict:
    """Points at the initial time t = t_min, uniform over the 3D domain."""
    d = _domain(cfg)
    t = torch.full((n, 1), d["t_min"], dtype=torch.float32).requires_grad_(True)
    return _batch(
        _uniform(n, d["x_min"], d["x_max"], generator),
        _uniform(n, d["y_min"], d["y_max"], generator),
        _uniform(n, d["z_min"], d["z_max"], generator),
        t,
        "initial",
    )
