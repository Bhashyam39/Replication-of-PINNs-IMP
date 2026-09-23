"""Evaluation metrics for the 3D NIST PINN (SPEC item 11).

Public API:

* :func:`melt_pool_mask` -- boolean mask of molten points (``T >= Ts``,
  solidus = 1290 C, the NIST melt-pool threshold).
* :func:`length_width_depth` -- melt-pool dimensions (m) from a mask and
  coordinates, matching the NIST CHAL-AMB2018-02-MP definitions: length is
  the x extent, width the y extent, depth the z extent below the top surface.
* :func:`cooling_rate_centerline` -- centerline cooling rate (K/s), paper
  Eq. 29 style: in the quasi-steady moving frame the thermal history of a
  material point on the centerline follows ``T(x - V t)``, so the cooling
  rate between two isotherms is ``V * (T_hi - T_lo) / (x_hi - x_lo)`` and the
  local rate is the material derivative ``dT/dt + V * dT/dx``. Both NIST
  CHAL-AMB2018-02-CR ranges are reported: 1290 C -> 1000 C and
  1290 C -> 1190 C.
* :func:`evaluate_model` -- convenience wrapper assembling a JSON-able
  metrics dict (melt-pool dimensions + cooling rates + public targets).

All functions are CPU-safe and differentiable only where noted; the eval
helpers run under ``torch.enable_grad`` where autograd is needed and avoid
network/disk access entirely.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Mapping, Optional, Tuple, Union

import torch

from src.physics.residuals import grad

__all__ = [
    "melt_pool_mask",
    "length_width_depth",
    "cooling_rate_centerline",
    "evaluate_model",
]

#: NIST cooling-rate ranges (K), see src.data.nist_public.
_THRESHOLDS_K = {
    "1290_1000": (1290.0 + 273.15, 1000.0 + 273.15),
    "1290_1190": (1290.0 + 273.15, 1190.0 + 273.15),
}


def _get(obj: Any, key: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, Mapping):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _solidus(cfg: Any) -> float:
    mat = _get(cfg, "material", cfg)
    return float(_get(mat, "Ts", 1290.0 + 273.15))


def _domain(cfg: Any) -> Dict[str, float]:
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


def _velocity(cfg: Any) -> float:
    return float(_get(cfg, "velocity", _get(_get(cfg, "laser"), "velocity", 0.8)))


# ---------------------------------------------------------------------------
# melt pool geometry
# ---------------------------------------------------------------------------
def melt_pool_mask(T: torch.Tensor, cfg: Any) -> torch.Tensor:
    """Boolean mask (N,) of molten points: ``T >= Ts`` (solidus, 1290 C).

    This is the NIST AM-Bench melt-pool definition (solidus isotherm as the
    pool boundary).
    """
    if not isinstance(T, torch.Tensor):
        T = torch.as_tensor(T, dtype=torch.float32)
    return (T.reshape(-1) >= _solidus(cfg)).bool()


def _coords_xyz(coords: Any) -> torch.Tensor:
    """Normalize coords to an (N, 3) tensor ordered (x, y, z).

    Accepts a tensor of shape (N, 3) or (N, 4) (extra trailing columns such
    as ``t`` are dropped), or a batch dict with ``x``/``y``/``z`` keys.
    """
    if isinstance(coords, Mapping):
        return torch.cat(
            [coords["x"].reshape(-1, 1), coords["y"].reshape(-1, 1), coords["z"].reshape(-1, 1)],
            dim=1,
        )
    c = torch.as_tensor(coords, dtype=torch.float32)
    if c.dim() != 2 or c.size(1) < 3:
        raise ValueError(f"coords must be (N,3)/(N,4) or a batch dict, got {tuple(c.shape)}")
    return c[:, :3]


def length_width_depth(mask: torch.Tensor, coords: Any) -> Dict[str, float]:
    """Melt-pool dimensions in meters from a boolean mask and coordinates.

    length = x extent, width = y extent, depth = z extent of the masked
    (molten) points. Returns zeros when the mask is empty (no melt pool).
    """
    m = torch.as_tensor(mask).bool().reshape(-1)
    xyz = _coords_xyz(coords)
    if xyz.shape[0] != m.shape[0]:
        raise ValueError(
            f"mask and coords disagree: {m.shape[0]} vs {xyz.shape[0]} points"
        )
    if not bool(m.any()):
        return {"length": 0.0, "width": 0.0, "depth": 0.0}
    pts = xyz[m]
    span = pts.max(dim=0).values - pts.min(dim=0).values
    return {
        "length": float(span[0]),
        "width": float(span[1]),
        "depth": float(span[2]),
    }


# ---------------------------------------------------------------------------
# cooling rate
# ---------------------------------------------------------------------------
def _find_crossing(
    xs: torch.Tensor, Ts: torch.Tensor, level: float, x_laser: float
) -> Optional[float]:
    """x position where the centerline profile crosses ``level`` behind the
    laser (x < x_laser), by linear interpolation. ``None`` if not found."""
    x_list = xs.tolist()
    t_list = Ts.tolist()
    best: Optional[float] = None
    for i in range(len(x_list) - 1):
        t0, t1 = t_list[i], t_list[i + 1]
        if (t0 - level) * (t1 - level) > 0.0:
            continue
        if t0 == t1:
            continue
        frac = (level - t0) / (t1 - t0)
        xc = x_list[i] + frac * (x_list[i + 1] - x_list[i])
        if xc <= x_laser and (best is None or xc > best):
            best = xc  # keep the crossing closest to the laser (pool tail)
    return best


def _model_device_dtype(model: torch.nn.Module):
    try:
        p = next(model.parameters())
        return p.device, p.dtype
    except StopIteration:  # parameter-free surrogate model (e.g. in tests)
        return torch.device("cpu"), torch.float32


def cooling_rate_centerline(
    model: torch.nn.Module,
    cfg: Any,
    n: int = 512,
    t_eval: Optional[float] = None,
) -> Dict[str, Any]:
    """Centerline cooling-rate metrics (paper Eq. 29 style, NIST variants).

    Samples the top-surface centerline (y = 0, z = z_max) at ``t_eval``
    (default: ``domain.t_max``, the quasi-steady state used by the paper),
    evaluates ``T(x)`` with autograd, and returns:

    * ``cooling_rate_1290_1000_K_per_s`` / ``cooling_rate_1290_1190_K_per_s``:
      ``V * dT/dx`` based rates between the NIST isotherm pairs behind the
      laser, i.e. the material-frame rate ``dT/dt + V*dT/dx`` in the
      quasi-steady limit;
    * ``material_rate_at_solidus_K_per_s``: the autograd material derivative
      ``dT/dt + V*dT/dx`` interpolated at the solidus (1290 C) crossing;
    * bookkeeping fields (laser position, eval time, crossing locations).

    If the profile never reaches the thresholds (e.g. an untrained/smoke
    model), a finite fallback is returned: the maximum material-derivative
    magnitude along the centerline, with ``"fallback": True``. This keeps
    eval JSON finite while flagging that no true pool crossing existed.
    """
    device, dtype = _model_device_dtype(model)
    d = _domain(cfg)
    V = _velocity(cfg)
    laser = _get(cfg, "laser")
    x0 = float(_get(laser, "x0", -1e-3))
    t_val = float(d["t_max"] if t_eval is None else t_eval)
    x_laser = x0 + V * t_val

    was_training = model.training
    model.eval()
    with torch.enable_grad():
        x = torch.linspace(d["x_min"], d["x_max"], n, dtype=dtype, device=device)
        x = x.reshape(-1, 1).requires_grad_(True)
        y = torch.zeros_like(x)
        z = torch.full_like(x, d["z_max"])
        t = torch.full_like(x, t_val).requires_grad_(True)
        out = model(torch.cat([x, y, z, t], dim=1))
        T = out[:, 4:5]
        dT_dt = grad(T, t)
        dT_dx = grad(T, x)
        material_rate = (dT_dt + V * dT_dx).detach().reshape(-1)
        x_det = x.detach().reshape(-1)
        T_det = T.detach().reshape(-1)
    if was_training:
        model.train()

    result: Dict[str, Any] = {
        "eval_time_s": t_val,
        "laser_x_m": x_laser,
        "scan_speed_m_per_s": V,
        "fallback": False,
    }

    solidus = _solidus(cfg)
    x_solidus = _find_crossing(x_det, T_det, solidus, x_laser)
    result["solidus_crossing_x_m"] = x_solidus

    # Material derivative interpolated at the solidus crossing.
    if x_solidus is not None:
        rate_at = torch_interp1d(x_det, material_rate, x_solidus)
        result["material_rate_at_solidus_K_per_s"] = float(abs(rate_at))
    else:
        result["material_rate_at_solidus_K_per_s"] = None

    ok = True
    for name, (T_hi, T_lo) in _THRESHOLDS_K.items():
        x_hi = _find_crossing(x_det, T_det, T_hi, x_laser)
        x_lo = _find_crossing(x_det, T_det, T_lo, x_laser)
        result[f"x_{name}_hi_m"] = x_hi
        result[f"x_{name}_lo_m"] = x_lo
        if x_hi is not None and x_lo is not None and x_hi > x_lo:
            rate = V * (T_hi - T_lo) / (x_hi - x_lo)
            result[f"cooling_rate_{name}_K_per_s"] = float(rate)
        else:
            result[f"cooling_rate_{name}_K_per_s"] = None
            ok = False

    if not ok:
        # Finite fallback: peak material-derivative magnitude on the line.
        fallback_rate = float(material_rate.abs().max()) if n > 0 else 0.0
        result["fallback"] = True
        for name in _THRESHOLDS_K:
            if result[f"cooling_rate_{name}_K_per_s"] is None:
                result[f"cooling_rate_{name}_K_per_s"] = fallback_rate
        if result["material_rate_at_solidus_K_per_s"] is None:
            result["material_rate_at_solidus_K_per_s"] = fallback_rate
    return result


def torch_interp1d(xs: torch.Tensor, ys: torch.Tensor, x0: float) -> float:
    """Linear interpolation of ys(xs) at x0 (xs ascending), pure torch."""
    x_list = xs.tolist()
    y_list = ys.tolist()
    if x0 <= x_list[0]:
        return float(y_list[0])
    if x0 >= x_list[-1]:
        return float(y_list[-1])
    for i in range(len(x_list) - 1):
        if x_list[i] <= x0 <= x_list[i + 1]:
            frac = (x0 - x_list[i]) / max(x_list[i + 1] - x_list[i], 1e-30)
            return float(y_list[i] + frac * (y_list[i + 1] - y_list[i]))
    return float(y_list[-1])


# ---------------------------------------------------------------------------
# convenience: full eval dict
# ---------------------------------------------------------------------------
def evaluate_model(
    model: torch.nn.Module,
    cfg: Any,
    grid: Tuple[int, int, int] = (256, 48, 32),
    n_centerline: int = 512,
) -> Dict[str, Any]:
    """Assemble a JSON-able metrics dict for a trained model.

    Evaluates the melt-pool mask on a regular (nx, ny, nz) grid at
    ``t = domain.t_max`` and the centerline cooling rates. Returns plain
    floats (meters, K/s). No disk or network access.
    """
    device, dtype = _model_device_dtype(model)
    d = _domain(cfg)
    nx, ny, nz = grid

    xs = torch.linspace(d["x_min"], d["x_max"], nx, dtype=dtype, device=device)
    ys = torch.linspace(d["y_min"], d["y_max"], ny, dtype=dtype, device=device)
    zs = torch.linspace(d["z_min"], d["z_max"], nz, dtype=dtype, device=device)
    gx, gy, gz = torch.meshgrid(xs, ys, zs, indexing="ij")
    coords = torch.stack([gx, gy, gz], dim=-1).reshape(-1, 3)

    was_training = model.training
    model.eval()
    with torch.no_grad():
        t_col = torch.full(
            (coords.shape[0], 1), d["t_max"], dtype=dtype, device=device
        )
        out = model(torch.cat([coords, t_col], dim=1))
        T = out[:, 4]
    if was_training:
        model.train()

    mask = melt_pool_mask(T, cfg)
    dims = length_width_depth(mask, coords)
    cooling = cooling_rate_centerline(model, cfg, n=n_centerline)

    return {
        "melt_pool": {
            "n_molten_points": int(mask.sum()),
            "length_m": dims["length"],
            "width_m": dims["width"],
            "depth_m": dims["depth"],
        },
        "cooling_rate": cooling,
    }
