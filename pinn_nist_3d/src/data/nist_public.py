"""Public NIST AM-Bench 2018-02 summary data access (SPEC item 9).

Public-data-only policy
-----------------------
This module never scrapes or downloads anything, and importing it performs
no network or disk I/O. Two ways to get data:

* :func:`load_summary_csv` with ``path=None`` (or a missing file) returns the
  built-in summary targets below, transcribed from public sources:

  - NIST AM-Bench 2018-02 experimental melt-pool *lengths* (cases A/B/C),
    as tabulated in Zhu, Liu & Yan (2021), Table 4 ("Experiment [77]").
    Width/depth were not reported experimentally in that table (``None``).
  - The paper's own PINN predictions (Table 4) are included as separate rows
    for trend reference only -- they are *not* experimental targets.

  Actual AMMT laser powers (137.9/179.2/179.2 W for A/B/C) and spot sizes
  (AMMT D4sigma 170 um / FWHM 100 um; CBM D4sigma 100 um / FWHM 59 um) come
  from the NIST AMB2018-02 test description and are already encoded in
  ``src.config.settings``.

* Point the loader at a CSV you downloaded yourself from the public NIST
  data repository (e.g. DOI 10.18434/M31931, DOI 10.18434/mds2-3830). See
  :func:`expected_schema` for the expected columns. If you explicitly request
  raw files that do not exist, :func:`require_raw_files` raises
  ``FileNotFoundError`` (fail loudly, never silently substitute).
"""

from __future__ import annotations

import csv
import os
import warnings
from typing import Any, Dict, List, Mapping, Optional, Union

__all__ = [
    "expected_schema",
    "load_summary_csv",
    "require_raw_files",
    "builtin_summary_targets",
    "BUILTIN_SUMMARY",
]

#: Cooling-rate temperature ranges used by NIST CHAL-AMB2018-02-CR (Kelvin).
COOLING_RANGE_WARM_K = (1290.0 + 273.15, 1190.0 + 273.15)  # 1290 C -> 1190 C
COOLING_RANGE_FULL_K = (1290.0 + 273.15, 1000.0 + 273.15)  # 1290 C -> 1000 C

_SRC_NIST_EXP = (
    "NIST AM-Bench 2018-02 experiment, as reported in "
    "Zhu/Liu/Yan 2021 Table 4 ('Experiment')"
)
_SRC_PAPER_PINN = (
    "Zhu/Liu/Yan 2021 Table 4 PINN prediction "
    "(reference trend only, NOT an experimental target)"
)

#: Built-in summary targets. SI units (m, s, K, W). ``*_unc`` are the
#: published +/- values; ``None`` means "not publicly reported".
BUILTIN_SUMMARY: List[Dict[str, Any]] = [
    {
        "case": "A",
        "source": _SRC_NIST_EXP,
        "power_w": 150.0,  # programmed; AMMT actual was 137.9 W
        "speed_m_per_s": 0.4,
        "length_m": 659e-6,
        "length_unc_m": 21e-6,
        "width_m": None,
        "depth_m": None,
        "cooling_rate_1290_1000_K_per_s": None,
        "cooling_rate_1290_1190_K_per_s": None,
    },
    {
        "case": "B",
        "source": _SRC_NIST_EXP,
        "power_w": 195.0,  # programmed; AMMT actual was 179.2 W
        "speed_m_per_s": 0.8,
        "length_m": 782e-6,
        "length_unc_m": 21e-6,
        "width_m": None,
        "depth_m": None,
        "cooling_rate_1290_1000_K_per_s": None,
        "cooling_rate_1290_1190_K_per_s": None,
    },
    {
        "case": "C",
        "source": _SRC_NIST_EXP,
        "power_w": 195.0,  # programmed; AMMT actual was 179.2 W
        "speed_m_per_s": 1.2,
        "length_m": 754e-6,
        "length_unc_m": 46e-6,
        "width_m": None,
        "depth_m": None,
        "cooling_rate_1290_1000_K_per_s": None,
        "cooling_rate_1290_1190_K_per_s": None,
    },
    {
        "case": "A",
        "source": _SRC_PAPER_PINN,
        "power_w": 150.0,
        "speed_m_per_s": 0.4,
        "length_m": 594.8e-6,
        "length_unc_m": None,
        "width_m": 193.3e-6,
        "depth_m": 64.0e-6,
        "cooling_rate_1290_1000_K_per_s": None,
        "cooling_rate_1290_1190_K_per_s": None,
    },
    {
        "case": "B",
        "source": _SRC_PAPER_PINN,
        "power_w": 195.0,
        "speed_m_per_s": 0.8,
        "length_m": 740.3e-6,
        "length_unc_m": None,
        "width_m": 160.0e-6,
        "depth_m": 52.8e-6,
        "cooling_rate_1290_1000_K_per_s": None,
        "cooling_rate_1290_1190_K_per_s": None,
    },
    {
        "case": "C",
        "source": _SRC_PAPER_PINN,
        "power_w": 195.0,
        "speed_m_per_s": 1.2,
        "length_m": 732.5e-6,
        "length_unc_m": None,
        "width_m": 131.5e-6,
        "depth_m": 43.2e-6,
        "cooling_rate_1290_1000_K_per_s": None,
        "cooling_rate_1290_1190_K_per_s": None,
    },
]


def expected_schema() -> Dict[str, str]:
    """Document the CSV columns understood by :func:`load_summary_csv`.

    Returns an ordered mapping ``column -> description``. All quantities are
    SI. Unknown extra columns are ignored by the loader; missing optional
    columns become ``None``. Only ``case`` is required.
    """
    return {
        "case": "REQUIRED. Case label, one of 'A', 'B', 'C' (AMB2018-02 scan cases).",
        "source": "Provenance label, e.g. 'NIST AM-Bench 2018-02 experiment' or "
        "'paper PINN'. Free text; used to distinguish experiments from model values.",
        "power_w": "Laser power in watts (programmed or actual; say which in 'source').",
        "speed_m_per_s": "Scan speed in m/s along +x.",
        "length_m": "Melt-pool length (x extent of the T >= solidus region) in m.",
        "length_unc_m": "Reported +/- uncertainty on length_m in m (optional).",
        "width_m": "Melt-pool width (y extent of the melt region) in m (optional).",
        "depth_m": "Melt-pool depth (z extent below z=0) in m (optional).",
        "cooling_rate_1290_1000_K_per_s": "Centerline cooling rate over "
        "1290 C -> 1000 C in K/s (optional; NIST CHAL-AMB2018-02-CR).",
        "cooling_rate_1290_1190_K_per_s": "Centerline cooling rate over "
        "1290 C -> 1190 C in K/s (optional; NIST CHAL-AMB2018-02-CR).",
        "notes": "Free-text notes (optional).",
    }


def builtin_summary_targets() -> List[Dict[str, Any]]:
    """A copy of the built-in paper/NIST summary targets (rows for A/B/C)."""
    return [dict(row) for row in BUILTIN_SUMMARY]


def _to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    text = str(value).strip()
    if text == "" or text.lower() in {"none", "nan", "na", "n/a", "-"}:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _normalize_row(raw: Mapping) -> Dict[str, Any]:
    row: Dict[str, Any] = {}
    for col in expected_schema():
        row[col] = raw.get(col)
    row["case"] = str(row.get("case") or "").strip().upper()
    row["source"] = str(row.get("source") or "user CSV").strip()
    for col in expected_schema():
        if col in ("case", "source", "notes"):
            continue
        row[col] = _to_float(row[col])
    return row


def load_summary_csv(path: Optional[Union[str, os.PathLike]] = None) -> List[Dict[str, Any]]:
    """Load melt-pool summary targets from a user-supplied CSV.

    Tolerant behavior (per SPEC): if ``path`` is ``None`` **or the file is
    absent**, return the built-in paper/NIST summary targets for cases A/B/C
    (with source labels) instead of failing. A missing *explicit* path emits
    a ``UserWarning`` so the substitution is never silent; use
    :func:`require_raw_files` when silent substitution must be an error.

    Parsing is deliberately forgiving: unknown columns are ignored, blank /
    ``-`` / ``NA`` numeric cells become ``None``, and rows without a case
    label are skipped. Returns a list of row dicts with the
    :func:`expected_schema` keys (SI units).
    """
    if path is None:
        return builtin_summary_targets()
    path = os.fspath(path)
    if not os.path.exists(path):
        warnings.warn(
            f"summary CSV not found: {path!r}; falling back to built-in "
            "paper/NIST summary targets (public-data-only mode)",
            UserWarning,
            stacklevel=2,
        )
        return builtin_summary_targets()

    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        for raw in reader:
            if raw is None:
                continue
            row = _normalize_row({k.strip().lower(): v for k, v in raw.items() if k})
            if row["case"] not in ("A", "B", "C"):
                continue  # tolerant: skip header/junk/unknown-case rows
            rows.append(row)
    if not rows:
        warnings.warn(
            f"no valid A/B/C rows parsed from {path!r}; falling back to "
            "built-in paper/NIST summary targets",
            UserWarning,
            stacklevel=2,
        )
        return builtin_summary_targets()
    return rows


def require_raw_files(paths: Union[str, os.PathLike, List[Union[str, os.PathLike]]]) -> List[str]:
    """Fail loudly if user-requested raw data files are missing.

    Returns the validated path strings; raises ``FileNotFoundError`` listing
    every missing path. No automatic download/scraping is ever attempted.
    """
    if isinstance(paths, (str, os.PathLike)):
        paths = [paths]
    checked = [os.fspath(p) for p in paths]
    missing = [p for p in checked if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(
            "requested raw NIST data file(s) not found (no automatic "
            "download is performed; obtain them from the public NIST "
            "AM-Bench repository, e.g. DOI 10.18434/M31931): "
            + ", ".join(repr(p) for p in missing)
        )
    return checked
