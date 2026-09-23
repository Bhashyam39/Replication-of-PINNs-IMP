"""Configuration dataclasses and case registry for the NIST AM-Bench IN625 PINN.

All values are SI (m, s, K, W, Pa). Values not given by the paper
(Zhu/Liu/Yan) or by the public NIST AM-Bench 2018-02 metadata are marked
``ASSUMPTION`` so they can be tuned or replaced later.
"""

from __future__ import annotations

import dataclasses
import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class MaterialProps:
    """IN625 thermophysical properties (SI).

    Solidus/liquidus are the standard IN625 values used by NIST AM-Bench
    (solidus 1290 C is also the NIST melt-pool threshold). Temperature-
    dependent fits are not public in the paper, so constant / two-phase
    blend values are ASSUMPTIONs.
    """

    # IN625 solidus / liquidus (NIST AM-Bench uses 1290 C as melt threshold).
    Ts: float = 1290.0 + 273.15  # K, solidus
    Tl: float = 1350.0 + 273.15  # K, liquidus
    # ASSUMPTION: paper does not report cp; typical IN625 values near melt.
    cp_solid: float = 650.0  # J/(kg K)
    cp_liquid: float = 740.0  # J/(kg K)
    # ASSUMPTION: paper does not report kappa; typical IN625 high-T values.
    kappa_solid: float = 21.0  # W/(m K)
    kappa_liquid: float = 28.0  # W/(m K)
    # ASSUMPTION: constant density (Boussinesq-style), IN625 near melt.
    rho: float = 8440.0  # kg/m^3
    # ASSUMPTION: constant liquid dynamic viscosity, typical Ni-superalloy melt.
    mu: float = 7.0e-3  # Pa s
    # ASSUMPTION: latent heat of fusion for IN625 (~290 kJ/kg literature).
    latent_heat: float = 2.9e5  # J/kg


@dataclass
class LaserConfig:
    """Moving Gaussian surface heat source.

    Paper defaults: rb = 50 um, eta = 0.43. NIST reports spot sizes as
    D4sigma (AMMT 170 um, CBM 100 um) and FWHM (AMMT 100 um, CBM 59 um);
    for q ~ exp(-2 r^2 / rb^2), rb = D4sigma / 2 = FWHM / sqrt(2 ln 2),
    so AMMT -> rb 85 um and CBM -> rb 50 um. Use ``rb_override`` (or the
    ``nist_spot`` presets) to replace the paper beam radius.
    """

    power: float = 195.0  # W (overridden by CaseConfig resolution)
    velocity: float = 0.8  # m/s along +x (overridden by CaseConfig)
    rb: float = 50e-6  # m, paper 1/e^2 beam radius
    eta: float = 0.43  # absorptivity (paper value per SPEC)
    x0: float = -1e-3  # m, ASSUMPTION: laser start x at t=0 (paper not explicit)
    rb_override: Optional[float] = None  # m, e.g. 85e-6 for AMMT D4sigma 170 um
    # Surface loss model (top Neumann BC).
    h_conv: float = 20.0  # W/(m^2 K), ASSUMPTION: paper value not public
    emissivity: float = 0.4  # ASSUMPTION: typical IN625 high-T emissivity
    t_inf: float = 300.0  # K, ASSUMPTION: ambient/gas temperature
    stefan_boltzmann: float = 5.670374419e-8  # W/(m^2 K^4), exact constant

    # NIST reported spot sizes (m), kept for provenance / overrides.
    NIST_SPOTS: Dict[str, Dict[str, float]] = field(
        default_factory=lambda: {
            "ammt": {"d4sigma": 170e-6, "fwhm": 100e-6},
            "cbm": {"d4sigma": 100e-6, "fwhm": 59e-6},
        }
    )

    def effective_rb(self) -> float:
        """Beam radius actually used (NIST override wins over paper default)."""
        return float(self.rb_override) if self.rb_override is not None else float(self.rb)

    def apply_nist_spot(self, source: str) -> None:
        """Set ``rb_override`` from a NIST D4sigma spot-size report."""
        spot = self.NIST_SPOTS[source.lower()]
        self.rb_override = spot["d4sigma"] / 2.0


@dataclass
class DomainConfig:
    """Local moving-frame-ish domain per SPEC (SI)."""

    x_min: float = -2e-3
    x_max: float = 2e-3
    y_min: float = -3e-4
    y_max: float = 3e-4
    z_min: float = -2e-4  # z <= 0 into substrate; top surface z = 0
    z_max: float = 0.0
    t_min: float = 0.0
    t_max: float = 2e-3
    # ASSUMPTION: substrate/ambient initial temperature (room temperature).
    t_ambient: float = 300.0  # K
    # ASSUMPTION: smoothing length of the hard-BC Heaviside ramp.
    bc_eps: float = 5e-5  # m


@dataclass
class TrainConfig:
    """Training hyperparameters. All paper-missing values are ASSUMPTIONs."""

    iterations: int = 10000  # ASSUMPTION
    lr: float = 1e-3  # ASSUMPTION
    n_interior: int = 4096  # ASSUMPTION
    n_top: int = 1024  # ASSUMPTION
    n_dirichlet: int = 1024  # ASSUMPTION
    n_initial: int = 1024  # ASSUMPTION
    w_data: float = 1.0  # ASSUMPTION
    w_pde: float = 1.0  # ASSUMPTION
    w_top: float = 1.0  # ASSUMPTION
    w_ic: float = 1.0  # ASSUMPTION
    w_summary: float = 1.0  # ASSUMPTION
    log_every: int = 100
    seed: int = 1234
    device: str = "cpu"
    checkpoint_dir: str = "outputs/checkpoints"


# Programmed (commanded) case parameters from the paper / NIST scan plan.
_CASES_PROGRAMMED: Dict[str, Dict[str, float]] = {
    "A": {"power": 150.0, "velocity": 0.4},
    "B": {"power": 195.0, "velocity": 0.8},
    "C": {"power": 195.0, "velocity": 1.2},
}

# Public NIST note: AMMT *actual* measured powers differ from programmed.
_CASES_AMMT_ACTUAL: Dict[str, float] = {"A": 137.9, "B": 179.2, "C": 179.2}

POWER_SOURCES = ("programmed", "ammt_actual", "cbm")


@dataclass
class CaseConfig:
    """Self-contained config for one scan case (what residuals/losses consume)."""

    name: str = "B"
    power_programmed: float = 195.0  # W
    velocity: float = 0.8  # m/s
    power_source: str = "programmed"  # one of POWER_SOURCES
    material: MaterialProps = field(default_factory=MaterialProps)
    laser: LaserConfig = field(default_factory=LaserConfig)
    domain: DomainConfig = field(default_factory=DomainConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    @property
    def power(self) -> float:
        """Resolved laser power in W for the selected ``power_source``."""
        if self.power_source == "ammt_actual":
            return _CASES_AMMT_ACTUAL[self.name]
        # "programmed" and "cbm" both use the commanded power (public NIST note).
        return float(self.power_programmed)


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def _defaults_dict() -> Dict[str, Any]:
    case = get_case("B")
    return {
        "material": dataclasses.asdict(case.material),
        "laser": dataclasses.asdict(case.laser),
        "domain": dataclasses.asdict(case.domain),
        "train": dataclasses.asdict(case.train),
        "cases": {k: dict(v) for k, v in _CASES_PROGRAMMED.items()},
        "ammt_actual_power": dict(_CASES_AMMT_ACTUAL),
        "power_sources": list(POWER_SOURCES),
    }


def load_config(path: Optional[str] = None) -> Dict[str, Any]:
    """Load a JSON/YAML config file, merged over built-in defaults.

    ``path=None`` returns the defaults. Missing files raise
    ``FileNotFoundError`` (fail loudly per SPEC). File values override
    defaults section-by-section.
    """
    cfg = _defaults_dict()
    if path is None:
        return cfg
    if not os.path.exists(path):
        raise FileNotFoundError(f"config file not found: {path}")
    with open(path, "r", encoding="utf-8") as fh:
        text = fh.read()
    if path.endswith(".json"):
        user = json.loads(text)
    else:
        try:
            import yaml

            user = yaml.safe_load(text)
        except ImportError:  # pragma: no cover - yaml is expected to exist
            user = json.loads(text)
    if not isinstance(user, dict):
        raise ValueError(f"config file must contain a mapping, got {type(user)}")
    for key, val in user.items():
        if isinstance(val, dict) and isinstance(cfg.get(key), dict):
            cfg[key].update(val)
        else:
            cfg[key] = val
    return cfg


def get_case(name: str, power_source: str = "programmed") -> CaseConfig:
    """Build a :class:`CaseConfig` for case A/B/C with the given power source."""
    key = str(name).upper()
    if key not in _CASES_PROGRAMMED:
        raise ValueError(f"unknown case {name!r}; expected one of {sorted(_CASES_PROGRAMMED)}")
    if power_source not in POWER_SOURCES:
        raise ValueError(f"unknown power_source {power_source!r}; expected {POWER_SOURCES}")
    prog = _CASES_PROGRAMMED[key]
    laser = LaserConfig(power=prog["power"], velocity=prog["velocity"])
    case = CaseConfig(
        name=key,
        power_programmed=prog["power"],
        velocity=prog["velocity"],
        power_source=power_source,
        laser=laser,
    )
    return case
