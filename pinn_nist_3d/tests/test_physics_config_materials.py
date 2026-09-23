"""Focused CPU tests for config/materials/physics (complements test_core.py)."""

import math

import pytest
import torch

from src.config.settings import LaserConfig, MaterialProps, get_case, load_config
from src.materials.in625 import cp, kappa, latent_heat, liquid_fraction, mu, rho
from src.physics.bc import apply_hard_bc, hard_heaviside
from src.physics.laser import gaussian_heat_flux, top_heat_residual
from src.physics.residuals import grad, residuals

MAT = MaterialProps()


class TinyNet(torch.nn.Module):
    """Stand-in for models.FCNN so these tests do not depend on other modules."""

    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(4, 16),
            torch.nn.SiLU(),
            torch.nn.Linear(16, 5),
        )

    def forward(self, xt):
        return self.net(xt)


def make_batch(n=8, cfg=None, surface="interior"):
    cfg = cfg or get_case("B")
    d = cfg.domain
    g = torch.Generator().manual_seed(0)

    def lo_hi(lo, hi):
        return (torch.rand(n, 1, generator=g) * (hi - lo) + lo).requires_grad_(True)

    return {
        "x": lo_hi(d.x_min, d.x_max),
        "y": lo_hi(d.y_min, d.y_max),
        "z": lo_hi(d.z_min, d.z_max),
        "t": lo_hi(d.t_min, d.t_max),
        "surface": surface,
    }


def test_get_case_power_sources():
    assert get_case("A").power == pytest.approx(150.0)
    assert get_case("B", "cbm").power == pytest.approx(195.0)
    assert get_case("B", "ammt_actual").power == pytest.approx(179.2)
    assert get_case("A", "ammt_actual").power == pytest.approx(137.9)
    assert get_case("C", "ammt_actual").power == pytest.approx(179.2)
    with pytest.raises(ValueError):
        get_case("D")
    with pytest.raises(ValueError):
        get_case("B", "bogus")


def test_load_config_defaults_and_missing(tmp_path):
    cfg = load_config(None)
    for key in ("material", "laser", "domain", "train", "cases"):
        assert key in cfg
    assert cfg["cases"]["B"]["power"] == pytest.approx(195.0)
    with pytest.raises(FileNotFoundError):
        load_config(str(tmp_path / "nope.yaml"))
    p = tmp_path / "c.yaml"
    p.write_text("train:\n  iterations: 7\n")
    assert load_config(str(p))["train"]["iterations"] == 7


def test_liquid_fraction_bounds_and_blend():
    assert liquid_fraction(MAT.Ts - 50.0, MAT.Ts, MAT.Tl).item() == 0.0
    assert liquid_fraction(MAT.Tl + 50.0, MAT.Ts, MAT.Tl).item() == 1.0
    assert liquid_fraction(0.5 * (MAT.Ts + MAT.Tl), MAT.Ts, MAT.Tl).item() == pytest.approx(0.5)
    T = torch.tensor([[MAT.Ts - 10.0], [MAT.Ts + 30.0], [MAT.Tl + 10.0]])
    f = liquid_fraction(T, MAT.Ts, MAT.Tl)
    assert torch.all((f >= 0) & (f <= 1))
    # differentiability through the mushy zone
    Td = torch.tensor([[0.5 * (MAT.Ts + MAT.Tl)]], requires_grad=True)
    liquid_fraction(Td, MAT.Ts, MAT.Tl).backward()
    assert Td.grad.item() == pytest.approx(1.0 / (MAT.Tl - MAT.Ts))


def test_material_constants_and_blends():
    T = torch.tensor([[MAT.Ts - 1.0], [MAT.Tl + 1.0]])
    f = torch.tensor([[0.0], [1.0]])
    assert torch.allclose(cp(T, f), torch.tensor([[MAT.cp_solid], [MAT.cp_liquid]]))
    assert torch.allclose(kappa(T, f), torch.tensor([[MAT.kappa_solid], [MAT.kappa_liquid]]))
    assert rho().item() == pytest.approx(MAT.rho)
    assert mu().item() == pytest.approx(MAT.mu)
    assert latent_heat().item() == pytest.approx(MAT.latent_heat)


def test_hard_heaviside_endpoints():
    d = torch.tensor([-1.0, 0.0, 0.5, 1.0, 2.0])
    h = hard_heaviside(d, eps=1.0)
    assert h[0].item() == 0.0 and h[1].item() == 0.0
    assert h[2].item() == pytest.approx(0.5)
    assert h[3].item() == pytest.approx(1.0) and h[4].item() == pytest.approx(1.0)


def test_laser_flux_peak_and_override():
    cfg = get_case("B")
    laser, rb = cfg.laser, cfg.laser.rb
    t = torch.tensor([[1e-3]])
    x_c = laser.x0 + cfg.velocity * 1e-3
    q_peak = gaussian_heat_flux(torch.tensor([[x_c]]), torch.zeros(1, 1), t, cfg)
    expected = 2.0 * cfg.power * laser.eta / (math.pi * rb**2)
    assert q_peak.item() == pytest.approx(expected, rel=1e-5)
    # decays to exp(-2) at r = rb, transverse and along scan
    q_rb = gaussian_heat_flux(torch.tensor([[x_c]]), torch.tensor([[rb]]), t, cfg)
    assert q_rb.item() == pytest.approx(expected * math.exp(-2.0), rel=1e-5)
    # NIST AMMT D4sigma 170 um -> rb 85 um override widens and lowers peak
    laser.apply_nist_spot("ammt")
    q_ammt = gaussian_heat_flux(torch.tensor([[x_c]]), torch.zeros(1, 1), t, cfg)
    assert laser.effective_rb() == pytest.approx(85e-6)
    assert q_ammt.item() == pytest.approx(expected * (rb / 85e-6) ** 2, rel=1e-5)


def test_residual_shapes_and_grad():
    torch.manual_seed(0)
    cfg = get_case("B")
    batch = make_batch(8, cfg)
    out = residuals(TinyNet(), batch, cfg)
    assert out["r_mom"].shape == (8, 3)
    assert out["r_cont"].shape == (8, 1)
    assert out["r_energy"].shape == (8, 1)
    assert out["fL"].shape == (8, 1)
    for k in ("r_mom", "r_cont", "r_energy", "fL"):
        assert torch.all(torch.isfinite(out[k]))
    y = torch.tensor([[2.0]], requires_grad=True)
    assert torch.allclose(grad(y**3, y), torch.tensor([[12.0]]))


def test_top_heat_residual_shape_and_finite():
    torch.manual_seed(0)
    cfg = get_case("B")
    batch = make_batch(8, cfg, surface="top")
    batch["z"] = torch.zeros(8, 1, requires_grad=True)
    r = top_heat_residual(TinyNet(), batch, cfg)
    assert r.shape == (8, 1)
    assert torch.all(torch.isfinite(r))


def test_apply_hard_bc_dirichlet_faces():
    cfg = get_case("B")
    d = cfg.domain
    n = 6
    batch = {
        # points exactly on: bottom, xmin, xmax, ymin, ymax, and one interior
        "x": torch.tensor([[0.0], [d.x_min], [d.x_max], [0.0], [0.0], [0.0]]),
        "y": torch.tensor([[0.0], [0.0], [0.0], [d.y_min], [d.y_max], [0.0]]),
        "z": torch.tensor([[d.z_min], [-1e-4], [-1e-4], [-1e-4], [-1e-4], [-1e-4]]),
        "t": torch.zeros(n, 1),
    }
    raw = torch.full((n, 5), 7.0)
    out = apply_hard_bc(raw, batch, cfg)
    # on-face points: no-slip + fixed T, pressure untouched
    assert torch.allclose(out[:5, 0:3], torch.zeros(5, 3))
    assert torch.allclose(out[:5, 4], torch.full((5,), d.t_ambient))
    assert torch.allclose(out[:, 3], raw[:, 3])
    # interior point (dist > bc_eps from all Dirichlet faces): unchanged
    assert torch.allclose(out[5], raw[5])
    # top surface (z=0, away from side walls) is not constrained
    top_batch = {
        "x": torch.tensor([[0.0]]),
        "y": torch.tensor([[0.0]]),
        "z": torch.tensor([[0.0]]),
        "t": torch.zeros(1, 1),
    }
    assert torch.allclose(apply_hard_bc(raw[:1], top_batch, cfg), raw[:1])
