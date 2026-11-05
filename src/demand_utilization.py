"""
demand_utilization.py - The NYC entitlement-utilization factor ``u`` of E_test.

``u`` is the average fraction of the 800 MGD Decree allocation that NYC calls
on in a state of the world, the fourth deeply uncertain factor of the held-out
re-evaluation ensemble (``docs/notes/methods/forcing_parameterization.md``,
"Demand-utilization axis"). It is sampled as one Latin hypercube column over
:data:`DEMAND_UTILIZATION_BOUNDS`, one value per SOW, appended to the
three-axis climate design (``scripts/supplemental/append_demand_utilization.py``),
and realized through a multiplicative scale ``k(u)`` on each realization's
whole regression-structured demand series with the 800 MGD monthly-mean cap
enforced AFTER scaling (Pywr-DRB ``ExtrapolatedDiversionPreprocessor.demand_scale``;
staged per chunk by :func:`src.ensemble_prep.stage_nyc_demand`).

``u`` is the NOMINAL utilization on the reference record (the historic
extrapolation on the reconstruction's NYC reservoir inflows, 1945 to 2023, the
covariate the regressions are trained on): the uncapped series is generated
once, ``u(k) = mean(cap_800(k * D_ref)) / 800``
is tabulated on a dense ``k`` grid (:func:`build_calibration_table`), and the
monotone table is inverted by a shape-preserving PCHIP interpolant
(:func:`k_from_u`). The table is regenerable, gitignored output
(:data:`CALIBRATION_CSV`, written by ``scripts/supplemental/demand_utilization_calibration.py``);
``k_from_u`` fails loudly when it is missing rather than fall back to a guess.
The realized utilization of a staged realization differs from nominal by
design (dry SOWs draw more through the negative regression slopes, and the
residual draws add scatter); it is reported by :func:`realized_utilization`
and never forced to nominal.

Search ensembles are never scaled (``k = 1``); only E_test carries the factor.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

import config
from src.scenario_designs import SEED_ROOT

###############################################################################
# Constants (single source of truth; src.etest imports these)
###############################################################################

#: Admissible range of the nominal utilization. The floor sits just below the
#: lowest River Master year on record (378 MGD in 2005-06 and 2018-19, 0.47);
#: the ceiling is the legal cap, nothing above 1.0 is admissible.
DEMAND_UTILIZATION_BOUNDS: tuple[float, float] = (0.45, 1.0)

#: Largest multiplicative scale applied before the cap (u(6.0) = 0.998 on the
#: reference record). ``k_from_u`` clips here and records the clipped flag;
#: u = 1.0 is the flat-800 limit.
DEMAND_SCALE_MAX: float = 6.0

#: Smallest scale of the calibration grid (u(0.3) = 0.21, well below the floor).
DEMAND_SCALE_MIN: float = 0.3

#: Reserved seed domain of the campaign E_test's utilization column
#: (``scengen.seeds.SEED_DOMAINS``); :func:`demand_seed_domain` derives the
#: domain of any variant from its streamflow seed domain.
DEMAND_UTILIZATION_SEED_DOMAIN: str = "etest:kn:demand"

#: The NYC Decree allocation the utilization is a fraction of (MGD).
NYC_ALLOCATION_MGD: float = float(config.NYC_DECREE_DIVERSION_CAP_MGD)

#: Where the calibration artifacts live (gitignored, regenerable).
DEMAND_UTILIZATION_DIR: Path = config.REFERENCE_DIR / "demand_utilization"
CALIBRATION_CSV: Path = DEMAND_UTILIZATION_DIR / "demand_utilization_calibration.csv"
REFERENCE_SERIES_CSV: Path = DEMAND_UTILIZATION_DIR / "demand_reference_uncapped_mgd.csv"

#: Per-realization arrays the append step adds to ``forcing_profiles.npz``
#: (each SOW value repeated ``realizations_per_profile`` times, like ``theta_params``).
NPZ_KEYS: tuple[str, ...] = ("demand_utilization", "demand_scale_k", "demand_scale_clipped")

#: Scalar provenance the append step writes to the npz and to every ``_meta.json``.
PROVENANCE_KEYS: tuple[str, ...] = (
    "demand_utilization_bounds",
    "demand_utilization_seed",
    "demand_utilization_seed_domain",
    "demand_scale_max",
    "demand_calibration_sha256",
)

#: Utilization bands of the subset re-scoring: u < 0.6, 0.6 <= u < 0.8, u >= 0.8.
UTILIZATION_BAND_EDGES: tuple[float, ...] = (0.6, 0.8)
UTILIZATION_BAND_LABELS: tuple[str, ...] = ("u_lt_0.6", "u_0.6_to_0.8", "u_ge_0.8")

#: Equal-width bins of the regret-versus-u summary over DEMAND_UTILIZATION_BOUNDS.
UTILIZATION_REGRET_BINS: int = 10

#: Dense ``k`` grid of the calibration table (step 0.02 to 2.0, then 0.1 to k_max).
CALIBRATION_K_GRID: np.ndarray = np.round(
    np.concatenate([np.arange(DEMAND_SCALE_MIN, 2.0, 0.02),
                    np.arange(2.0, DEMAND_SCALE_MAX + 1e-9, 0.1)]), 4)

#: Seed of the reference extrapolation's residual draws (the preprocessor's own
#: default, reseeded explicitly so the table never depends on prior RNG use).
CALIBRATION_DRAW_SEED: int = 1

#: Effectively no cap: the reference extrapolation is generated uncapped.
_UNCAPPED: float = 1e9


###############################################################################
# Seeds
###############################################################################

def demand_seed_domain(etest_seed_domain: str) -> str:
    """The reserved seed domain of an E_test variant's utilization column.

    Args:
        etest_seed_domain: The variant's streamflow seed domain (``"etest:kn"``).

    Returns:
        ``"{etest_seed_domain}:demand"``; :data:`DEMAND_UTILIZATION_SEED_DOMAIN`
        for the campaign variant.

    Raises:
        ValueError: If the domain is not a reserved ``etest:*`` domain.
    """
    if not str(etest_seed_domain).startswith("etest:"):
        raise ValueError(
            f"the utilization column is an E_test factor; got seed domain "
            f"{etest_seed_domain!r}, expected an 'etest:*' domain."
        )
    return f"{etest_seed_domain}:demand"


def demand_utilization_seed(etest_seed_domain: str = "etest:kn") -> int:
    """Seed of the utilization Latin hypercube column of one E_test variant.

    ``design_seed(SEED_ROOT, demand_seed_domain(etest_seed_domain), 0)``.
    """
    from scengen.seeds import design_seed

    return design_seed(SEED_ROOT, demand_seed_domain(etest_seed_domain), 0)


###############################################################################
# The monthly-mean cap and the reference record
###############################################################################

def cap_monthly_mean(series: pd.Series, cap: float = NYC_ALLOCATION_MGD) -> pd.Series:
    """Scale every calendar month whose mean exceeds ``cap`` down to a ``cap`` mean.

    The vectorized equivalent of Pywr-DRB's ``enforce_max_mean_demand``: daily
    values inside an offending month are multiplied uniformly by
    ``cap / monthly_mean``; other months are untouched.

    Args:
        series: Daily series with a ``DatetimeIndex`` (MGD).
        cap: Monthly-mean ceiling (MGD).

    Returns:
        The capped daily series (same index).
    """
    idx = pd.DatetimeIndex(series.index)
    monthly = series.groupby([idx.year, idx.month]).transform("mean").to_numpy(dtype=float)
    factor = np.where(monthly > cap, cap / monthly, 1.0)
    return series * factor


def reference_uncapped_series() -> pd.Series:
    """The uncapped historic-flow NYC extrapolation the calibration is measured on.

    Runs Pywr-DRB's ``ExtrapolatedDiversionPreprocessor(loc="nyc")`` on the
    reconstruction's NYC reservoir inflows (``pub_nhmv10_BC_withObsScaled``,
    1945 to 2023, the covariate the regressions are trained on) with the cap
    disabled, reseeding the residual draws at :data:`CALIBRATION_DRAW_SEED` so
    the series is deterministic.

    Returns:
        Daily ``aggregate`` demand (MGD) over the full reference window.
    """
    from pywrdrb.pre.extrapolate_nyc_nj_diversions import ExtrapolatedDiversionPreprocessor

    pp = ExtrapolatedDiversionPreprocessor(loc="nyc", max_mean_demand=_UNCAPPED)
    pp.load()
    np.random.seed(CALIBRATION_DRAW_SEED)
    pp.process()
    out = pp.processed_data["aggregate"].astype(float)
    out.index = pd.DatetimeIndex(out.index)
    out.name = "aggregate"
    return out


###############################################################################
# Calibration table: u(k) and its inverse
###############################################################################

def calibration_table_from_series(reference: pd.Series,
                                  k_grid: Iterable[float] | None = None,
                                  cap: float = NYC_ALLOCATION_MGD) -> pd.DataFrame:
    """Tabulate ``u(k) = mean(cap(k * D_ref)) / allocation`` on a ``k`` grid.

    Args:
        reference: Uncapped daily reference demand (MGD).
        k_grid: Scales to tabulate (default :data:`CALIBRATION_K_GRID`).
        cap: Monthly-mean cap and the allocation the utilization is a fraction of.

    Returns:
        Frame with columns ``k`` and ``u``, strictly increasing in both. ``u(k)``
        is non-decreasing and flat at 1.0 once every month of the reference is
        capped; the flat tail (where the inverse is undefined) is dropped, so
        the table ends at the smallest ``k`` that realizes its last ``u``. On
        the reference record ``u(DEMAND_SCALE_MAX) = 0.998 < 1`` and the whole
        grid is kept (the calibration driver asserts this).
    """
    ks = np.asarray(CALIBRATION_K_GRID if k_grid is None else list(k_grid), dtype=float)
    ref = reference.astype(float)
    idx = pd.DatetimeIndex(ref.index)
    monthly = ref.groupby([idx.year, idx.month]).transform("mean").to_numpy(dtype=float)
    daily = ref.to_numpy(dtype=float)
    us = []
    for k in ks:
        factor = np.minimum(1.0, cap / (k * monthly))
        us.append(float(np.nanmean(k * daily * factor)) / cap)
    us = np.asarray(us, dtype=float)
    rising = np.concatenate(([True], np.diff(us) > 0))
    n_keep = len(us) if rising.all() else int(np.argmin(rising))
    table = pd.DataFrame({"k": ks[:n_keep], "u": us[:n_keep]})
    _validate_table(table)
    return table


def build_calibration_table(k_grid: Iterable[float] | None = None
                            ) -> tuple[pd.DataFrame, pd.Series]:
    """Generate the reference extrapolation and tabulate ``u(k)`` on it.

    Returns:
        ``(table, reference)``: the ``k``/``u`` frame and the uncapped daily
        reference series it was measured on.
    """
    reference = reference_uncapped_series()
    return calibration_table_from_series(reference, k_grid), reference


def _validate_table(table: pd.DataFrame) -> None:
    """Hard-fail on a table that is not a strictly monotone, bounded ``u(k)``."""
    if list(table.columns[:2]) != ["k", "u"] or len(table) < 3:
        raise ValueError("calibration table must hold columns k, u with >= 3 rows.")
    k, u = table["k"].to_numpy(dtype=float), table["u"].to_numpy(dtype=float)
    if not (np.all(np.diff(k) > 0) and np.all(np.diff(u) > 0)):
        raise ValueError("calibration table must be strictly increasing in k and u.")
    if u.min() <= 0.0 or u.max() > 1.0:
        raise ValueError("calibration u must lie in (0, 1].")


def load_calibration_table(path: Path | None = None) -> pd.DataFrame:
    """Read the persisted calibration table, failing loudly when it is absent.

    Raises:
        FileNotFoundError: Naming the driver that writes the table.
    """
    path = CALIBRATION_CSV if path is None else Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"demand-utilization calibration table not found: {path}. Run "
            f"`python scripts/supplemental/demand_utilization_calibration.py` "
            f"first; k_from_u never falls back to an internal guess."
        )
    table = pd.read_csv(path)
    _validate_table(table)
    return table


def calibration_sha256(path: Path | None = None) -> str:
    """SHA-256 of the persisted calibration table (provenance of every ``k``)."""
    path = CALIBRATION_CSV if path is None else Path(path)
    if not path.exists():
        load_calibration_table(path)  # raises the named error
    return hashlib.sha256(path.read_bytes()).hexdigest()


def u_from_k(k, table: pd.DataFrame | None = None) -> np.ndarray:
    """Nominal utilization of a scale, by PCHIP on the calibration table.

    Args:
        k: Scalar or array of scales within ``[k_min, DEMAND_SCALE_MAX]``.
        table: Calibration table (default: the persisted one).

    Returns:
        ``u(k)`` as a float array of the same shape.

    Raises:
        ValueError: For a scale outside the tabulated range.
    """
    from scipy.interpolate import PchipInterpolator

    table = load_calibration_table() if table is None else table
    k = np.asarray(k, dtype=float)
    k_min, k_max = float(table["k"].iloc[0]), float(table["k"].iloc[-1])
    if np.any(k < k_min - 1e-12) or np.any(k > k_max + 1e-12):
        raise ValueError(f"k must lie in [{k_min}, {k_max}]; got range "
                         f"[{k.min()}, {k.max()}].")
    f = PchipInterpolator(table["k"].to_numpy(dtype=float),
                          table["u"].to_numpy(dtype=float), extrapolate=False)
    return np.asarray(f(np.clip(k, k_min, k_max)), dtype=float)


def k_from_u(u, table: pd.DataFrame | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Scale that realizes a nominal utilization, clipped at ``DEMAND_SCALE_MAX``.

    Inverts the monotone calibration table with a shape-preserving PCHIP
    interpolant of ``k(u)``. A ``u`` above ``u(DEMAND_SCALE_MAX)`` (0.998 on
    the reference record; ``u = 1.0`` is the flat-800 limit) maps to
    ``DEMAND_SCALE_MAX`` with the clipped flag set.

    Args:
        u: Scalar or array of nominal utilizations in ``[u(k_min), 1]``.
        table: Calibration table (default: the persisted one).

    Returns:
        ``(k, clipped)``: float and bool arrays of ``u``'s shape.

    Raises:
        ValueError: For a ``u`` below the tabulated range or above 1.
    """
    from scipy.interpolate import PchipInterpolator

    table = load_calibration_table() if table is None else table
    u = np.asarray(u, dtype=float)
    ks = table["k"].to_numpy(dtype=float)
    us = table["u"].to_numpy(dtype=float)
    if np.any(u < us[0] - 1e-12) or np.any(u > 1.0 + 1e-12):
        raise ValueError(f"u must lie in [{us[0]:.4f}, 1.0]; got range "
                         f"[{u.min()}, {u.max()}].")
    clipped = u > us[-1]
    inv = PchipInterpolator(us, ks, extrapolate=False)
    k = np.asarray(inv(np.clip(u, us[0], us[-1])), dtype=float)
    k = np.where(clipped, float(ks[-1]), k)
    return k, clipped


###############################################################################
# Sampling and the realized diagnostic
###############################################################################

def sample_utilization(n_sow: int, seed: int,
                       bounds: tuple[float, float] = DEMAND_UTILIZATION_BOUNDS
                       ) -> np.ndarray:
    """One scrambled Latin hypercube column of ``u``, one value per SOW.

    Appending an independent stratified column to the climate design keeps the
    joint design a Latin hypercube, so no streamflow is regenerated.

    Args:
        n_sow: Number of SOWs (strata).
        seed: ``design_seed`` of the variant's ``*:demand`` domain.
        bounds: ``(lo, hi)`` of the column.

    Returns:
        ``(n_sow,)`` float array in SOW order.
    """
    from scipy.stats import qmc

    lo, hi = bounds
    sampler = qmc.LatinHypercube(d=1, scramble=True, seed=int(seed))
    return qmc.scale(sampler.random(int(n_sow)), [lo], [hi])[:, 0]


def realized_utilization(series: pd.Series,
                         allocation: float = NYC_ALLOCATION_MGD) -> float:
    """Mean demand over complete FFMP years (June to May) divided by the allocation.

    The unit matches the objectives (``objectives_ensemble.ffmp_year_unit_slices``):
    the spin-up months and any trailing partial year are excluded.

    Args:
        series: Daily demand (MGD) with a ``DatetimeIndex``.
        allocation: The allocation the utilization is a fraction of (MGD).

    Raises:
        ValueError: If the series holds no complete FFMP year.
    """
    from src.objectives_ensemble import ffmp_year_unit_slices

    idx = pd.DatetimeIndex(series.index)
    slices = ffmp_year_unit_slices(idx)
    if not slices:
        raise ValueError("realized_utilization needs at least one complete "
                         "June-May FFMP year after the spin-up.")
    values = series.to_numpy(dtype=float)
    pooled = np.concatenate([values[s] for s in slices])
    return float(np.nanmean(pooled)) / allocation


###############################################################################
# Bands and bins of the analysis layer
###############################################################################

def utilization_band_masks(u) -> dict[str, np.ndarray]:
    """Boolean masks of the three utilization bands, in :data:`UTILIZATION_BAND_LABELS` order."""
    u = np.asarray(u, dtype=float)
    lo, hi = UTILIZATION_BAND_EDGES
    return {
        UTILIZATION_BAND_LABELS[0]: u < lo,
        UTILIZATION_BAND_LABELS[1]: (u >= lo) & (u < hi),
        UTILIZATION_BAND_LABELS[2]: u >= hi,
    }


def utilization_bin_labels(u, n_bins: int = UTILIZATION_REGRET_BINS,
                           bounds: tuple[float, float] = DEMAND_UTILIZATION_BOUNDS
                           ) -> tuple[np.ndarray, np.ndarray]:
    """Equal-width bin index of each ``u`` over ``bounds``.

    Returns:
        ``(labels, edges)``: int labels in ``[0, n_bins)`` and the ``n_bins + 1`` edges.
    """
    u = np.asarray(u, dtype=float)
    edges = np.linspace(bounds[0], bounds[1], int(n_bins) + 1)
    labels = np.clip(np.searchsorted(edges[1:-1], u, side="right"), 0, int(n_bins) - 1)
    return labels.astype(int), edges
