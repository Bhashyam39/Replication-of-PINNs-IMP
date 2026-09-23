"""Training loop for the 3D NIST PINN.

``train(cfg)`` runs a configurable Adam loop (SPEC item 10): configurable
iterations, batch sizes, loss weights and learning rate; logs every
``log_every``; saves checkpoints under ``outputs/checkpoints/``.

ASSUMPTIONS (paper does not report these; all configurable via
``cfg["train"]``):
    lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0,
    iterations=20000, log_every=100, save_every=1000, seed=1234,
    device="cpu", grad_clip=None (optional max grad norm),
    batch_sizes: interior=512, top=256, dirichlet=256, initial=256, data=0.

Sibling modules (``src.config``, ``src.data``) are imported lazily so this
file stays importable while they are being written concurrently.
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import torch

try:  # package-relative first, absolute fallback for robustness
    from ..losses.total import compute_losses
    from ..models.fcnn import FCNN
except ImportError:  # pragma: no cover
    from src.losses.total import compute_losses
    from src.models.fcnn import FCNN

__all__ = ["train", "load_checkpoint", "latest_checkpoint"]

#: ASSUMPTION defaults (paper is missing these hyperparameters).
_DEFAULTS = {
    "iterations": 20000,
    "lr": 1e-3,
    "betas": (0.9, 0.999),
    "eps": 1e-8,
    "weight_decay": 0.0,
    "grad_clip": None,
    "log_every": 100,
    "save_every": 1000,
    "seed": 1234,
    "device": "cpu",
    "output_dir": "outputs/checkpoints",
}
_DEFAULT_BATCH_SIZES = {
    "interior": 512,
    "top": 256,
    "dirichlet": 256,
    "initial": 256,
    "data": 0,
}


# ---------------------------------------------------------------------------
# cfg helpers (tolerant of dicts and dataclass-like objects)
# ---------------------------------------------------------------------------
def _get(obj: Any, key: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, Mapping):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _train_cfg(cfg: Any) -> dict:
    raw = _get(cfg, "train")
    if raw is None:
        raw = cfg  # allow a flat cfg dict
    out = dict(_DEFAULTS)
    for key in out:
        val = _get(raw, key)
        if val is not None:
            out[key] = val
    return out


def _batch_sizes(cfg: Any) -> dict:
    raw = _get(_get(cfg, "train"), "batch_sizes")
    if raw is None:
        raw = _get(cfg, "batch_sizes")
    out = dict(_DEFAULT_BATCH_SIZES)
    if raw is not None:
        for key in out:
            val = _get(raw, key)
            if val is not None:
                out[key] = int(val)
    return out


def _model_kwargs(cfg: Any) -> dict:
    raw = _get(cfg, "model")
    return {
        "in_dim": int(_get(raw, "in_dim", 4)),
        "out_dim": int(_get(raw, "out_dim", 5)),
        "hidden": int(_get(raw, "hidden", 250)),
        "layers": int(_get(raw, "layers", 5)),
    }


def _case_name(cfg: Any) -> str:
    for holder in (_get(cfg, "case"), cfg):
        name = _get(holder, "name")
        if name:
            return str(name)
    return "case"


# ---------------------------------------------------------------------------
# lazy sibling imports
# ---------------------------------------------------------------------------
def _load_config_fn():
    try:
        from ..config.settings import load_config
    except ImportError:  # pragma: no cover
        from src.config.settings import load_config
    return load_config


def _sampling_module():
    try:
        from ..data import sampling
    except ImportError:  # pragma: no cover
        from src.data import sampling
    return sampling


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _to_device(batch: Any, device: torch.device) -> Any:
    if not isinstance(batch, Mapping):
        return batch
    out = dict(batch)
    for key, val in out.items():
        if torch.is_tensor(val):
            out[key] = val.to(device)
    return out


def _sample_batches(cfg: Any, device: torch.device) -> Dict[str, dict]:
    """Draw one fresh batch per surface using ``src.data.sampling``."""
    sampling = _sampling_module()
    sizes = _batch_sizes(cfg)
    batches: Dict[str, dict] = {}
    draws = (
        ("interior", "sample_interior"),
        ("top", "sample_top"),
        ("dirichlet", "sample_dirichlet"),
        ("initial", "sample_initial"),
    )
    for surface, fn_name in draws:
        n = sizes.get(surface, 0)
        if n and n > 0:
            batches[surface] = _to_device(getattr(sampling, fn_name)(n, cfg), device)
    return batches


def _save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    case: str,
    history: list,
    cfg: Any,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "iteration": iteration,
        "case": case,
        "history": history,
        "model_kwargs": {
            "in_dim": model.in_dim,
            "out_dim": model.out_dim,
            "hidden": model.hidden,
            "layers": model.layers,
        }
        if isinstance(model, FCNN)
        else None,
        "config": cfg,
    }
    torch.save(payload, path)


def latest_checkpoint(output_dir: Any, case: str) -> Path:
    """Path of the rolling 'latest' checkpoint for a case."""
    return Path(output_dir) / f"latest_{case}.pt"


def load_checkpoint(path: Any, device: Any = "cpu") -> dict:
    """Load a checkpoint written by :func:`train`."""
    return torch.load(str(path), map_location=device, weights_only=False)


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------
def train(cfg: Any) -> Dict[str, Any]:
    """Run the Adam training loop.

    Parameters
    ----------
    cfg:
        Config dict (from ``config.settings.load_config`` / ``get_case``),
        a dataclass-like object, or a path to a config file.

    Returns
    -------
    dict with keys ``model``, ``optimizer``, ``history``, ``checkpoint``
    (final checkpoint path), ``latest_checkpoint`` and ``iterations``.
    """
    if isinstance(cfg, (str, os.PathLike)):
        cfg = _load_config_fn()(str(cfg))

    tc = _train_cfg(cfg)
    torch.manual_seed(int(tc["seed"]))

    device = torch.device(str(tc["device"]))
    model = FCNN(**_model_kwargs(cfg)).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(tc["lr"]),
        betas=tuple(tc["betas"]),
        eps=float(tc["eps"]),
        weight_decay=float(tc["weight_decay"]),
    )

    iterations = int(tc["iterations"])
    log_every = max(1, int(tc["log_every"]))
    save_every = max(1, int(tc["save_every"]))
    grad_clip = tc["grad_clip"]
    output_dir = Path(str(tc["output_dir"]))
    case = _case_name(cfg)

    history: list = []
    t_start = time.time()
    total: Optional[torch.Tensor] = None
    parts: Dict[str, torch.Tensor] = {}

    for it in range(1, iterations + 1):
        batches = _sample_batches(cfg, device)
        total, parts = compute_losses(model, batches, cfg)

        optimizer.zero_grad(set_to_none=True)
        total.backward()
        if grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
        optimizer.step()

        if it % log_every == 0 or it == 1 or it == iterations:
            entry = {"iteration": it, "total": float(total.detach().cpu())}
            entry.update(
                {k: float(v.detach().cpu()) for k, v in parts.items() if k != "total"}
            )
            history.append(entry)
            msg = " ".join(
                f"{k}={v:.3e}" for k, v in entry.items() if k != "iteration"
            )
            print(f"[train] case={case} it={it}/{iterations} {msg}", flush=True)

        if it % save_every == 0 or it == iterations:
            ckpt = output_dir / f"ckpt_{case}_it{it:07d}.pt"
            _save_checkpoint(ckpt, model, optimizer, it, case, history, cfg)
            _save_checkpoint(
                latest_checkpoint(output_dir, case),
                model, optimizer, it, case, history, cfg,
            )

    final_ckpt = output_dir / f"ckpt_{case}_final.pt"
    _save_checkpoint(final_ckpt, model, optimizer, iterations, case, history, cfg)
    _save_checkpoint(
        latest_checkpoint(output_dir, case), model, optimizer, iterations, case,
        history, cfg,
    )
    elapsed = time.time() - t_start
    print(
        f"[train] done case={case} iterations={iterations} "
        f"elapsed={elapsed:.1f}s checkpoint={final_ckpt}",
        flush=True,
    )
    return {
        "model": model,
        "optimizer": optimizer,
        "history": history,
        "checkpoint": str(final_ckpt),
        "latest_checkpoint": str(latest_checkpoint(output_dir, case)),
        "iterations": iterations,
    }
