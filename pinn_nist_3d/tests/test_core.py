"""CPU-safe core tests for the 3D NIST PINN (SPEC item 14).

Covers: swish sign, hard_heaviside endpoints, liquid_fraction bounds, laser
flux peak, residual shapes on a tiny random batch, summary loader fallback,
plus the data samplers and eval helpers owned with this file. Everything
runs on CPU in well under 60 s; no network or disk access is required.
"""

import math
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config.settings import MaterialProps, get_case  # noqa: E402
from src.data.nist_public import (  # noqa: E402
    builtin_summary_targets,
    expected_schema,
    load_summary_csv,
    require_raw_files,
)
from src.data.sampling import (  # noqa: E402
    sample_dirichlet,
    sample_initial,
    sample_interior,
    sample_top,
)
from src.eval.metrics import (  # noqa: E402
    cooling_rate_centerline,
    length_width_depth,
    melt_pool_mask,
)
from src.materials.in625 import liquid_fraction  # noqa: E402
from src.models.fcnn import FCNN, Swish  # noqa: E402
from src.physics.bc import hard_heaviside  # noqa: E402
from src.physics.laser import gaussian_heat_flux  # noqa: E402
from src.physics.residuals import residuals  # noqa: E402

MAT = MaterialProps()


def _tiny_model(seed: int = 0) -> FCNN:
    torch.manual_seed(seed)
    return FCNN(in_dim=4, out_dim=5, hidden=8, layers=2)


# ---------------------------------------------------------------------------
# SPEC-required tests
# ---------------------------------------------------------------------------
def test_swish_sign():
    swish = Swish()
    x = torch.linspace(-5.0, 5.0, 101)
    y = swish(x)
    assert torch.all(y[x > 0] > 0)
    assert torch.all(y[x < 0] < 0)
    assert swish(torch.tensor(0.0)).item() == pytest.approx(0.0)
    # Swish(x) = x * sigmoid(x)
    assert torch.allclose(y, x * torch.sigmoid(x))


def test_hard_heaviside_endpoints():
    eps = 1e-4
    assert hard_heaviside(torch.tensor(0.0), eps).item() == pytest.approx(0.0)
    assert hard_heaviside(torch.tensor(eps), eps).item() == pytest.approx(1.0)
    assert hard_heaviside(torch.tensor(-eps), eps).item() == pytest.approx(0.0)
    assert hard_heaviside(torch.tensor(10.0 * eps), eps).item() == pytest.approx(1.0)
    # midpoint of the corrected (1 - cos(pi d/eps))/2 ramp
    assert hard_heaviside(torch.tensor(0.5 * eps), eps).item() == pytest.approx(0.5)
    # monotone on [0, eps]
    d = torch.linspace(0.0, eps, 50)
    h = hard_heaviside(d, eps)
    assert torch.all(h[1:] >= h[:-1])


def test_liquid_fraction_bounds():
    Ts, Tl = MAT.Ts, MAT.Tl
    assert liquid_fraction(torch.tensor(Ts - 100.0)).item() == pytest.approx(0.0)
    assert liquid_fraction(torch.tensor(Ts)).item() == pytest.approx(0.0)
    assert liquid_fraction(torch.tensor(Tl)).item() == pytest.approx(1.0)
    assert liquid_fraction(torch.tensor(Tl + 100.0)).item() == pytest.approx(1.0)
    assert liquid_fraction(torch.tensor(0.5 * (Ts + Tl))).item() == pytest.approx(0.5)
    grid = torch.linspace(Ts - 500.0, Tl + 500.0, 1001)
    f = liquid_fraction(grid)
    assert torch.all(f >= 0.0) and torch.all(f <= 1.0)


def test_laser_flux_peak():
    cfg = get_case("B")  # Q=195 W, V=0.8 m/s, rb=50 um, eta=0.43, x0=-1e-3
    rb = cfg.laser.effective_rb()
    t = torch.tensor([[5e-4]])
    x_center = cfg.laser.x0 + cfg.velocity * 5e-4
    x = torch.tensor([[x_center]])
    y = torch.tensor([[0.0]])
    q_peak = gaussian_heat_flux(x, y, t, cfg)
    expected = 2.0 * cfg.power * cfg.laser.eta / (math.pi * rb**2)
    assert q_peak.item() == pytest.approx(expected, rel=1e-5)
    # one beam radius off-axis in y -> peak * exp(-2)
    q_off = gaussian_heat_flux(x, torch.tensor([[rb]]), t, cfg)
    assert q_off.item() == pytest.approx(expected * math.exp(-2.0), rel=1e-5)


def test_residual_shapes_tiny_batch():
    cfg = get_case("B")
    model = _tiny_model()
    batch = sample_interior(6, cfg, generator=torch.Generator().manual_seed(0))
    res = residuals(model, batch, cfg)
    assert res["r_mom"].shape == (6, 3)
    assert res["r_cont"].shape == (6, 1)
    assert res["r_energy"].shape == (6, 1)
    assert res["fL"].shape == (6, 1)
    for key in ("r_mom", "r_cont", "r_energy", "fL"):
        assert torch.all(torch.isfinite(res[key]))
    assert torch.all(res["fL"] >= 0.0) and torch.all(res["fL"] <= 1.0)


def test_summary_loader_fallback():
    schema = expected_schema()
    assert "case" in schema and "length_m" in schema
    # path=None -> built-ins
    rows = load_summary_csv(None)
    cases = {r["case"] for r in rows}
    assert cases == {"A", "B", "C"}
    assert all(r.get("source") for r in rows)
    # missing explicit path -> warns and falls back (no scraping)
    with pytest.warns(UserWarning):
        rows_missing = load_summary_csv("/nonexistent/nist_summary.csv")
    assert [r["case"] for r in rows_missing] == [r["case"] for r in rows]
    # built-in NIST experimental lengths (Zhu/Liu/Yan 2021 Table 4)
    nist = {r["case"]: r for r in builtin_summary_targets() if "NIST" in r["source"]}
    assert nist["A"]["length_m"] == pytest.approx(659e-6)
    assert nist["B"]["length_m"] == pytest.approx(782e-6)
    assert nist["C"]["length_m"] == pytest.approx(754e-6)


# ---------------------------------------------------------------------------
# additional coverage for owned modules
# ---------------------------------------------------------------------------
def test_load_summary_csv_parses_user_file(tmp_path):
    csv_path = tmp_path / "summary.csv"
    csv_path.write_text(
        "case,source,power_w,speed_m_per_s,length_m,width_m,depth_m\n"
        "B,NIST test,195,0.8,7.82e-4,-,5.28e-5\n"
        "junk row without enough columns\n"
    )
    rows = load_summary_csv(csv_path)
    assert len(rows) == 1
    row = rows[0]
    assert row["case"] == "B"
    assert row["length_m"] == pytest.approx(782e-6)
    assert row["width_m"] is None  # '-' tolerated as missing
    assert row["depth_m"] == pytest.approx(52.8e-6)


def test_require_raw_files_fails_loudly(tmp_path):
    ok = tmp_path / "present.csv"
    ok.write_text("case\n")
    assert require_raw_files(ok) == [str(ok)]
    with pytest.raises(FileNotFoundError):
        require_raw_files([str(ok), str(tmp_path / "missing.csv")])


def test_samplers_contract():
    cfg = get_case("A")
    g = torch.Generator().manual_seed(1)
    d = cfg.domain
    n = 32
    batches = {
        "interior": sample_interior(n, cfg, generator=g),
        "top": sample_top(n, cfg, generator=g),
        "dirichlet": sample_dirichlet(n, cfg, generator=g),
        "initial": sample_initial(n, cfg, generator=g),
    }
    for surface, b in batches.items():
        assert b["surface"] == surface
        for key in ("x", "y", "z", "t"):
            assert b[key].shape == (n, 1)
            assert b[key].dtype == torch.float32
            assert b[key].requires_grad
        assert torch.all(b["x"] >= d.x_min) and torch.all(b["x"] <= d.x_max)
        assert torch.all(b["y"] >= d.y_min) and torch.all(b["y"] <= d.y_max)
        assert torch.all(b["z"] >= d.z_min) and torch.all(b["z"] <= d.z_max)
        assert torch.all(b["t"] >= d.t_min) and torch.all(b["t"] <= d.t_max)
    # surface constraints
    assert torch.all(batches["top"]["z"] == d.z_max)
    assert torch.all(batches["initial"]["t"] == d.t_min)
    db = batches["dirichlet"]
    on_face = (
        (db["z"] == d.z_min)
        | (db["x"] == d.x_min)
        | (db["x"] == d.x_max)
        | (db["y"] == d.y_min)
        | (db["y"] == d.y_max)
    )
    assert torch.all(on_face)


def test_melt_pool_mask_and_dimensions():
    cfg = get_case("B")
    Ts = cfg.material.Ts
    T = torch.tensor([[Ts - 1.0], [Ts], [Ts + 50.0]])
    mask = melt_pool_mask(T, cfg)
    assert mask.dtype == torch.bool
    assert mask.tolist() == [False, True, True]

    coords = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1e-3, 2e-4, -1e-5],
            [2e-3, -1e-4, -5e-5],
        ]
    )
    dims = length_width_depth(mask, coords)
    assert dims["length"] == pytest.approx(1e-3)
    assert dims["width"] == pytest.approx(3e-4)
    assert dims["depth"] == pytest.approx(4e-5)
    # empty mask -> zeros (finite)
    empty = length_width_depth(torch.zeros(3, dtype=torch.bool), coords)
    assert empty == {"length": 0.0, "width": 0.0, "depth": 0.0}


def test_cooling_rate_centerline_finite_on_untrained_model():
    cfg = get_case("B")
    model = _tiny_model(seed=3)
    out = cooling_rate_centerline(model, cfg, n=64)
    for key in ("cooling_rate_1290_1000_K_per_s", "cooling_rate_1290_1190_K_per_s"):
        assert key in out
        assert math.isfinite(out[key])  # fallback keeps values finite
    assert math.isfinite(out["material_rate_at_solidus_K_per_s"])


def test_cooling_rate_centerline_recovers_imposed_profile():
    """Quasi-steady profile T(x - V t): metric must recover V * dT/dx."""

    class HotProfile(torch.nn.Module):
        """Quasi-steady ramp: T hot at the laser, cooling linearly behind it.

        T = 300 + relu(2200 + 2e6 * (x - x_laser)), so dT/dx = 2e6 K/m on
        the cooling side (x < x_laser) and the expected NIST cooling rate is
        V * dT/dx = 0.8 * 2e6 = 1.6e6 K/s for both isotherm pairs.
        """

        def forward(self, xt):
            x, t = xt[:, 0:1], xt[:, 3:4]
            x_laser = -1e-3 + 0.8 * t
            T = 300.0 + torch.relu(2200.0 + 2.0e6 * (x - x_laser))
            return torch.cat(
                [torch.zeros_like(T)] * 4 + [T], dim=1
            )  # (u, v, w, p, T)

    cfg = get_case("B")
    model = HotProfile()
    out = cooling_rate_centerline(model, cfg, n=1024)
    assert out["fallback"] is False
    expected = cfg.velocity * 2.0e6  # K/s
    assert out["cooling_rate_1290_1000_K_per_s"] == pytest.approx(expected, rel=1e-2)
    assert out["cooling_rate_1290_1190_K_per_s"] == pytest.approx(expected, rel=1e-2)
