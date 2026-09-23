"""IN625 material property models (torch-tensor compatible).

All functions accept floats or torch tensors and return torch tensors.
Where a function is used inside a PDE residual it is differentiable
(almost everywhere); piecewise-constant helpers (rho/mu/latent_heat) are
constants by design.

Values come from ``src.config.settings.MaterialProps``; anything not
reported by the paper or public NIST metadata is an ASSUMPTION there.
"""

from __future__ import annotations

from typing import Optional, Union

import torch

from src.config.settings import MaterialProps

Number = Union[float, torch.Tensor]

_DEFAULT = MaterialProps()


def _as_tensor(x: Number, like: Optional[torch.Tensor] = None) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        out = x
    else:
        out = torch.tensor(float(x), dtype=torch.float32)
    if like is not None:
        out = out.to(dtype=like.dtype, device=like.device)
    return out


def _to_T(T: Number) -> torch.Tensor:
    return T if isinstance(T, torch.Tensor) else torch.tensor(float(T), dtype=torch.float32)


def liquid_fraction(T: Number, Ts: Number = _DEFAULT.Ts, Tl: Number = _DEFAULT.Tl) -> torch.Tensor:
    """Smooth(er) liquid fraction: 0 at/below solidus, 1 at/above liquidus.

    Linear ramp in the mushy zone, clamped to [0, 1]. Differentiable
    almost everywhere (kinks at Ts/Tl), which keeps autograd stable for
    the apparent-capacity energy residual.
    """
    Tt = _to_T(T)
    Ts_t = _as_tensor(Ts, like=Tt)
    Tl_t = _as_tensor(Tl, like=Tt)
    f = (Tt - Ts_t) / (Tl_t - Ts_t)
    return torch.clamp(f, 0.0, 1.0)


def cp(T: Number, fL: Optional[Number] = None, props: MaterialProps = _DEFAULT) -> torch.Tensor:
    """Specific heat J/(kg K), linear two-phase blend on liquid fraction."""
    Tt = _to_T(T)
    if fL is None:
        fL = liquid_fraction(Tt, props.Ts, props.Tl)
    f = _as_tensor(fL, like=Tt)
    cp_s = _as_tensor(props.cp_solid, like=Tt)
    cp_l = _as_tensor(props.cp_liquid, like=Tt)
    return cp_s * (1.0 - f) + cp_l * f


def kappa(T: Number, fL: Optional[Number] = None, props: MaterialProps = _DEFAULT) -> torch.Tensor:
    """Thermal conductivity W/(m K), linear two-phase blend on liquid fraction."""
    Tt = _to_T(T)
    if fL is None:
        fL = liquid_fraction(Tt, props.Ts, props.Tl)
    f = _as_tensor(fL, like=Tt)
    k_s = _as_tensor(props.kappa_solid, like=Tt)
    k_l = _as_tensor(props.kappa_liquid, like=Tt)
    return k_s * (1.0 - f) + k_l * f


def rho(T: Optional[Number] = None, props: MaterialProps = _DEFAULT) -> torch.Tensor:
    """Density kg/m^3 (constant ASSUMPTION)."""
    out = torch.tensor(float(props.rho), dtype=torch.float32)
    if isinstance(T, torch.Tensor):
        out = out.to(dtype=T.dtype, device=T.device)
    return out


def mu(T: Optional[Number] = None, props: MaterialProps = _DEFAULT) -> torch.Tensor:
    """Dynamic viscosity Pa s (constant liquid value, ASSUMPTION)."""
    out = torch.tensor(float(props.mu), dtype=torch.float32)
    if isinstance(T, torch.Tensor):
        out = out.to(dtype=T.dtype, device=T.device)
    return out


def latent_heat(props: MaterialProps = _DEFAULT) -> torch.Tensor:
    """Latent heat of fusion J/kg (constant ASSUMPTION)."""
    return torch.tensor(float(props.latent_heat), dtype=torch.float32)
