"""Data access for the 3D NIST PINN (SPEC items 8-9).

Importing this package performs no network or disk I/O: the public NIST
summary targets are built-in constants, and samplers only draw random
collocation points.
"""
from .nist_public import (
    BUILTIN_SUMMARY,
    builtin_summary_targets,
    expected_schema,
    load_summary_csv,
    require_raw_files,
)
from .sampling import (
    sample_dirichlet,
    sample_initial,
    sample_interior,
    sample_top,
)

__all__ = [
    "BUILTIN_SUMMARY",
    "builtin_summary_targets",
    "expected_schema",
    "load_summary_csv",
    "require_raw_files",
    "sample_dirichlet",
    "sample_initial",
    "sample_interior",
    "sample_top",
]
