"""Evaluate a trained checkpoint for one NIST AM-Bench case (SPEC item 13).

Usage:
    python -m src.scripts.eval_case --case B --checkpoint latest --smoke
    python -m src.scripts.eval_case --case B \
        --checkpoint outputs/checkpoints/ckpt_B_final.pt

Loads the checkpoint, computes melt-pool dimensions and centerline cooling
rates (public-data-only metrics; no FEM fields), compares against the
built-in public NIST/paper summary targets, and prints a JSON document with
finite values to stdout.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import Any, Dict, List, Optional

import torch

from src.config.settings import POWER_SOURCES, get_case

_SMOKE_GRID = (64, 16, 12)
_SMOKE_N_CENTERLINE = 128
_FULL_GRID = (256, 48, 32)
_FULL_N_CENTERLINE = 512


def _resolve_checkpoint(args: argparse.Namespace) -> str:
    from src.train.trainer import latest_checkpoint

    if args.checkpoint in (None, "latest"):
        path = str(latest_checkpoint(args.output_dir, args.case.upper()))
    else:
        path = os.fspath(args.checkpoint)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"checkpoint not found: {path!r}. Train first, e.g. "
            f"python -m src.scripts.train_case --case {args.case.upper()} --smoke"
        )
    return path


def _load_model(path: str, device: torch.device) -> torch.nn.Module:
    from src.models.fcnn import FCNN
    from src.train.trainer import load_checkpoint

    payload = load_checkpoint(path, device=device)
    kwargs = payload.get("model_kwargs") or {}
    model = FCNN(
        in_dim=int(kwargs.get("in_dim", 4)),
        out_dim=int(kwargs.get("out_dim", 5)),
        hidden=int(kwargs.get("hidden", 250)),
        layers=int(kwargs.get("layers", 5)),
    )
    model.load_state_dict(payload["model_state_dict"])
    return model.to(device).eval(), payload


def _public_targets(case: str, csv_path: Optional[str]) -> List[Dict[str, Any]]:
    """Public NIST/paper summary rows for this case.

    Uses the built-in targets by default (no download); a user-supplied CSV
    (see ``nist_public.expected_schema``) replaces them.
    """
    from src.data.nist_public import load_summary_csv

    return [r for r in load_summary_csv(csv_path) if r["case"] == case.upper()]


def _sanitize(obj: Any) -> Any:
    """Recursively coerce tensors/numpy-ish values to finite JSON floats."""
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, bool) or obj is None or isinstance(obj, (str, int)):
        return obj
    val = float(obj)
    if not math.isfinite(val):
        return None  # should not happen: metrics guarantee finite values
    return val


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m src.scripts.eval_case",
        description=__doc__.splitlines()[0] if __doc__ else None,
    )
    p.add_argument("--case", default="B", choices=["A", "B", "C", "a", "b", "c"])
    p.add_argument("--power-source", default="programmed", choices=POWER_SOURCES)
    p.add_argument(
        "--checkpoint",
        default="latest",
        help="checkpoint path, or 'latest' for <output-dir>/latest_<CASE>.pt",
    )
    p.add_argument("--output-dir", default="outputs/checkpoints")
    p.add_argument("--device", default="cpu")
    p.add_argument(
        "--summary-csv",
        default=None,
        help="optional user-supplied NIST summary CSV for comparison targets "
        "(default: built-in public targets; never downloaded)",
    )
    p.add_argument(
        "--smoke",
        action="store_true",
        help="small evaluation grid for a fast CPU sanity run",
    )
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    case = args.case.upper()
    device = torch.device(args.device)

    cfg = get_case(case, power_source=args.power_source)
    ckpt_path = _resolve_checkpoint(args)
    model, payload = _load_model(ckpt_path, device)

    from src.eval.metrics import evaluate_model

    grid = _SMOKE_GRID if args.smoke else _FULL_GRID
    n_cl = _SMOKE_N_CENTERLINE if args.smoke else _FULL_N_CENTERLINE
    metrics = evaluate_model(model, cfg, grid=grid, n_centerline=n_cl)

    report = {
        "case": case,
        "power_source": args.power_source,
        "power_w": cfg.power,
        "velocity_m_per_s": cfg.velocity,
        "beam_radius_m": cfg.laser.effective_rb(),
        "checkpoint": ckpt_path,
        "checkpoint_iteration": payload.get("iteration"),
        "smoke": bool(args.smoke),
        "metrics": metrics,
        "public_targets": _public_targets(case, args.summary_csv),
        "notes": (
            "Public-data-only evaluation (NIST AM-Bench 2018-02 summaries); "
            "NOT the paper's FEM-trained protocol. Cooling-rate 'fallback' "
            "means the predicted pool never reached the NIST isotherms."
        ),
    }
    print(json.dumps(_sanitize(report), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
