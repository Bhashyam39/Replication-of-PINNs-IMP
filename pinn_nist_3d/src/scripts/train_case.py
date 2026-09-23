"""Train one NIST AM-Bench case (SPEC item 12).

Usage:
    python -m src.scripts.train_case --case B --iterations 200 --smoke
    python -m src.scripts.train_case --case A --power-source ammt_actual \
        --iterations 10000 --output-dir outputs/checkpoints

Public-data-only: training uses PDE residuals + boundary/initial conditions.
``src.losses.total.compute_losses`` additionally supports an optional sparse
public-summary supervision term (``cfg["data"]["summary"]`` point targets,
weight ``cfg.train.weights["summary"]``); the public NIST data ship only
melt-pool dimension/cooling-rate summaries, not pointwise fields, so this CLI
leaves that term off. This is NOT the paper's FEM-trained protocol.
"""

from __future__ import annotations

import argparse
import sys
from typing import Any, List, Optional

from src.config.settings import POWER_SOURCES, get_case, load_config

# Smoke-mode overrides: tiny model + batches so CPU runs finish in seconds.
_SMOKE_MODEL = {"in_dim": 4, "out_dim": 5, "hidden": 32, "layers": 2}
_SMOKE_BATCH_SIZES = {"interior": 64, "top": 32, "dirichlet": 32, "initial": 32, "data": 0}


def _apply_config_file(cfg: Any, path: str) -> Any:
    """Merge a JSON/YAML config file (``load_config`` schema) onto a CaseConfig."""
    user = load_config(path)
    for section in ("material", "laser", "domain", "train"):
        vals = user.get(section)
        if not isinstance(vals, dict):
            continue
        target = getattr(cfg, section)
        for key, val in vals.items():
            if hasattr(target, key):
                setattr(target, key, val)
            elif section == "train":
                # forward-compatible extras the trainer understands
                # (output_dir, save_every, batch_sizes, weights, ...)
                setattr(target, key, val)
    return cfg


def build_case_config(args: argparse.Namespace):
    """Assemble the CaseConfig consumed by trainer/losses/physics."""
    cfg = get_case(args.case, power_source=args.power_source)
    if args.config:
        _apply_config_file(cfg, args.config)

    cfg.train.iterations = int(args.iterations)
    cfg.train.seed = int(args.seed)
    cfg.train.device = args.device
    if args.log_every is not None:
        cfg.train.log_every = int(args.log_every)

    # Trainer-specific keys that are not TrainConfig dataclass fields.
    cfg.train.output_dir = args.output_dir  # noqa: attribute-defined-outside-init
    cfg.train.save_every = int(args.save_every)  # noqa

    if args.smoke:
        cfg.train.log_every = 1
        cfg.train.batch_sizes = dict(_SMOKE_BATCH_SIZES)  # noqa
        cfg.model = dict(_SMOKE_MODEL)  # noqa: read by trainer._model_kwargs
    return cfg


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m src.scripts.train_case",
        description=__doc__.splitlines()[0] if __doc__ else None,
    )
    p.add_argument("--case", default="B", choices=["A", "B", "C", "a", "b", "c"])
    p.add_argument("--power-source", default="programmed", choices=POWER_SOURCES)
    p.add_argument("--iterations", type=int, default=200)
    p.add_argument("--lr", type=float, default=None, help="Adam learning rate override")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--device", default="cpu")
    p.add_argument("--log-every", type=int, default=None)
    p.add_argument("--save-every", type=int, default=1000)
    p.add_argument("--output-dir", default="outputs/checkpoints")
    p.add_argument("--config", default=None, help="optional JSON/YAML config file")
    p.add_argument(
        "--smoke",
        action="store_true",
        help="tiny model/batches and log every step (CPU sanity run)",
    )
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    cfg = build_case_config(args)
    if args.lr is not None:
        cfg.train.lr = float(args.lr)

    print(
        f"[train_case] case={cfg.name} power={cfg.power:.1f} W "
        f"({cfg.power_source}) velocity={cfg.velocity} m/s "
        f"rb={cfg.laser.effective_rb() * 1e6:.1f} um iterations={cfg.train.iterations} "
        f"smoke={args.smoke}",
        flush=True,
    )

    from src.train.trainer import train

    result = train(cfg)
    print(f"[train_case] checkpoint: {result['checkpoint']}", flush=True)
    print(f"[train_case] latest:     {result['latest_checkpoint']}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
