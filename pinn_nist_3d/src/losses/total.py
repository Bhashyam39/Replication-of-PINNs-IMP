"""Total loss assembly for the 3D NIST PINN.

Implements ``compute_losses(model, batch, cfg) -> (total, parts)`` per SPEC:
data loss, PDE interior residual loss, Neumann top (heat-flux) loss, initial
condition loss, and optional sparse public-summary supervision.

Notes / assumptions
-------------------
* Loss weights come from ``cfg["train"]["weights"]`` (dict or dataclass-like).
  Recognized keys with defaults:
    ``data=1.0``, ``pde=1.0`` (umbrella for momentum/continuity/energy),
    ``mom``/``cont``/``energy`` (default to the ``pde`` value),
    ``top=1.0``, ``ic=1.0``, ``summary=0.0``.
  ASSUMPTION: the paper does not report loss weights, so all default to 1.0
  and the summary (sparse NIST public data) term defaults to 0.0 (off).
* ASSUMPTION: the initial condition enforces u=v=w=0 and T=T0; pressure is
  left unconstrained at t=0. T0 is read from ``case.T0`` / ``domain.T0`` /
  ``initial_temp`` with default 300 K (paper does not state it).
* Hard Dirichlet BCs are assumed to be enforced by ``physics.bc.apply_hard_bc``
  inside the physics residual functions; this module uses raw model outputs.
* Physics modules are imported lazily so that this file stays importable even
  while sibling modules are being written concurrently.
"""
from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Tuple

import torch

__all__ = ["compute_losses"]

#: Output channel index for each predicted field (SPEC contract order).
_OUT_IDX = {"u": 0, "v": 1, "w": 2, "p": 3, "T": 4}
_LABEL_KEYS = ("u", "v", "w", "p", "T")

#: Recognized surface names when ``batch`` is a collection of sub-batches.
_SURFACES = ("interior", "top", "dirichlet", "initial", "data", "summary")

#: ASSUMPTION: ambient/initial temperature (paper does not state it).
_DEFAULT_T0 = 300.0


# ---------------------------------------------------------------------------
# cfg helpers (tolerant of dicts and dataclass-like objects)
# ---------------------------------------------------------------------------
def _get(obj: Any, key: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, Mapping):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _as_mapping(obj: Any) -> dict:
    if obj is None:
        return {}
    if isinstance(obj, Mapping):
        return dict(obj)
    if hasattr(obj, "__dict__"):
        return dict(vars(obj))
    return {}


def _weights(cfg: Any) -> Dict[str, float]:
    w = _as_mapping(_get(_get(cfg, "train"), "weights"))
    pde = float(w.get("pde", 1.0))
    return {
        "data": float(w.get("data", 1.0)),
        "pde": pde,
        "mom": float(w.get("mom", pde)),
        "cont": float(w.get("cont", pde)),
        "energy": float(w.get("energy", pde)),
        "top": float(w.get("top", 1.0)),
        "ic": float(w.get("ic", 1.0)),
        "summary": float(w.get("summary", 0.0)),
    }


def _initial_temp(cfg: Any) -> float:
    for holder in (_get(cfg, "case"), _get(cfg, "domain"), cfg):
        for key in ("T0", "initial_temp", "T_initial", "ambient_temp"):
            val = _get(holder, key)
            if val is not None:
                return float(val)
    return _DEFAULT_T0


# ---------------------------------------------------------------------------
# lazy physics imports (sibling modules may be written concurrently)
# ---------------------------------------------------------------------------
def _residuals_fn():
    try:
        from ..physics.residuals import residuals
    except ImportError:  # pragma: no cover - fallback for non-package usage
        from src.physics.residuals import residuals
    return residuals


def _top_heat_residual_fn():
    try:
        from ..physics.laser import top_heat_residual
    except ImportError:  # pragma: no cover
        from src.physics.laser import top_heat_residual
    return top_heat_residual


# ---------------------------------------------------------------------------
# batch handling
# ---------------------------------------------------------------------------
def _normalize_batches(batch: Any) -> Dict[str, dict]:
    """Accept either a single batch dict (with a ``surface`` key) or a dict of
    ``surface -> batch`` sub-batches, and return the latter form."""
    if not isinstance(batch, Mapping):
        raise TypeError(f"batch must be a dict, got {type(batch)!r}")
    if any(isinstance(batch.get(k), Mapping) for k in _SURFACES):
        return {k: v for k, v in batch.items() if isinstance(v, Mapping)}
    surface = batch.get("surface", "interior")
    return {str(surface): dict(batch)}


def _coords(b: Mapping[str, torch.Tensor]) -> torch.Tensor:
    return torch.cat([b["x"], b["y"], b["z"], b["t"]], dim=1)


def _mse(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (a - b).pow(2).mean()


def _data_loss_single(
    model: torch.nn.Module, b: Mapping[str, torch.Tensor]
) -> Optional[torch.Tensor]:
    """Supervised MSE over whichever label keys (u,v,w,p,T) are present."""
    labels = [k for k in _LABEL_KEYS if b.get(k) is not None]
    if not labels:
        return None
    pred = model(_coords(b))
    losses = [_mse(pred[:, _OUT_IDX[k] : _OUT_IDX[k] + 1], b[k]) for k in labels]
    return torch.stack(losses).mean()


def _initial_loss(
    model: torch.nn.Module, b: Mapping[str, torch.Tensor], T0: float
) -> torch.Tensor:
    """u=v=w=0 and T=T0 at t=0 (pressure unconstrained; ASSUMPTION)."""
    pred = model(_coords(b))
    ref = torch.zeros_like(pred[:, 0:1])
    loss = _mse(pred[:, 0:1], ref) + _mse(pred[:, 1:2], ref) + _mse(pred[:, 2:3], ref)
    loss = loss + _mse(pred[:, 4:5], torch.full_like(ref, T0))
    return loss


def _summary_loss(model: torch.nn.Module, cfg: Any) -> Optional[torch.Tensor]:
    """Optional sparse supervision from public NIST/paper summary data.

    Expected cfg layout (all optional; term skipped when absent)::

        cfg["data"]["summary"] = {
            "points":  [[x, y, z, t], ...],          # SI units
            "targets": {"T": [...], "u": [...], ...} # subset of fields
        }
    """
    spec = _get(_get(cfg, "data"), "summary")
    if not spec:
        return None
    points = spec.get("points")
    targets = spec.get("targets")
    if not points or not targets:
        return None
    pts = torch.as_tensor(points, dtype=torch.float32)
    try:
        param = next(model.parameters())
        pts = pts.to(device=param.device, dtype=param.dtype)
    except StopIteration:  # pragma: no cover
        pass
    pred = model(pts)
    losses = []
    for key, vals in targets.items():
        if key not in _OUT_IDX:
            continue
        tgt = torch.as_tensor(vals, dtype=pred.dtype, device=pred.device)
        tgt = tgt.reshape(-1, 1)
        losses.append(_mse(pred[:, _OUT_IDX[key] : _OUT_IDX[key] + 1], tgt))
    if not losses:
        return None
    return torch.stack(losses).mean()


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------
def compute_losses(
    model: torch.nn.Module,
    batch: Any,
    cfg: Any = None,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Assemble the total PINN loss.

    Parameters
    ----------
    model:
        Network mapping (N, 4) -> (N, 5), e.g. ``models.fcnn.FCNN``.
    batch:
        Either a single batch dict (SPEC contract keys ``x,y,z,t`` plus
        optional labels and a ``surface`` string) or a dict mapping surface
        names (``interior``, ``top``, ``dirichlet``, ``initial``, ``data``)
        to such batch dicts.
    cfg:
        Config dict (as returned by ``config.settings.load_config``) or
        dataclass-like object.

    Returns
    -------
    total:
        Scalar weighted-sum loss tensor (differentiable).
    parts:
        Dict of named scalar loss tensors (``data``, ``mom``, ``cont``,
        ``energy``, ``pde``, ``top``, ``ic``, ``summary`` as available) plus
        ``total``.
    """
    cfg = {} if cfg is None else cfg
    w = _weights(cfg)
    batches = _normalize_batches(batch)
    parts: Dict[str, torch.Tensor] = {}
    total: Optional[torch.Tensor] = None

    def _accumulate(name: str, value: Optional[torch.Tensor], weight: float) -> None:
        nonlocal total
        if value is None:
            return
        parts[name] = value
        term = value * weight
        total = term if total is None else total + term

    # --- supervised data loss (any sub-batch carrying labels) -------------
    data_terms = [
        _data_loss_single(model, b) for b in batches.values()
    ]
    data_terms = [t for t in data_terms if t is not None]
    data_loss = torch.stack(data_terms).mean() if data_terms else None
    _accumulate("data", data_loss, w["data"])

    # --- PDE interior residuals -------------------------------------------
    interior = batches.get("interior")
    if interior is not None and (w["mom"] or w["cont"] or w["energy"]):
        res = _residuals_fn()(model, interior, cfg)
        r_mom = res["r_mom"]
        r_cont = res["r_cont"]
        r_energy = res["r_energy"]
        mom_loss = r_mom.pow(2).mean()
        cont_loss = r_cont.pow(2).mean()
        energy_loss = r_energy.pow(2).mean()
        _accumulate("mom", mom_loss, w["mom"])
        _accumulate("cont", cont_loss, w["cont"])
        _accumulate("energy", energy_loss, w["energy"])
        parts["pde"] = mom_loss + cont_loss + energy_loss

    # --- Neumann top surface (laser heat flux) ----------------------------
    top = batches.get("top")
    if top is not None and w["top"]:
        r_top = _top_heat_residual_fn()(model, top, cfg)
        if isinstance(r_top, (tuple, list)):
            r_top = r_top[0]
        _accumulate("top", r_top.pow(2).mean(), w["top"])

    # --- initial condition -------------------------------------------------
    initial = batches.get("initial")
    if initial is not None and w["ic"]:
        _accumulate("ic", _initial_loss(model, initial, _initial_temp(cfg)), w["ic"])

    # --- optional sparse public summary supervision -----------------------
    if w["summary"]:
        _accumulate("summary", _summary_loss(model, cfg), w["summary"])

    if total is None:
        # No active terms: return a differentiable zero tied to the model.
        try:
            total = sum(p.sum() for p in model.parameters()) * 0.0
        except StopIteration:  # pragma: no cover
            total = torch.zeros((), requires_grad=True)
    parts["total"] = total
    return total, parts
