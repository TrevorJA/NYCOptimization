"""hazard_metric_image.py - The 21-metric hazard image and its pure helpers.

Every hazard image stores the eight event metrics and the remaining metrics in
two arrays; :func:`descriptor_image` stacks them into the single (M, 21)
candidate-metric matrix every diagnostic works on, drops the two truncation
flags, and records each metric's hazardous direction. The other helpers are
the rank-to-normal transform, the Spearman clusters and normal-score PCA, the
hazard-direction tail mask, the exact-snap tail-share limit, and the
nearest-member effective sample size. Pure numpy/scipy/pandas; no matplotlib,
no ``config``, no environment reads, so worker processes can import it.
Settings: ``supplemental_config.SELDIAG_*``. Tests:
``tests/test_hazard_selector_descriptors.py``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata

import supplemental_config as scfg
from scengen import subsample as ss
from scengen.diagnostics import pca_effective_dimension, spearman_clusters


def descriptor_image(
    H: np.ndarray, hazard_axes: list[str], supplement: np.ndarray, supplement_names,
) -> dict:
    """The descriptor matrix: candidate axes plus the supplement minus its flags.

    Args:
        H: ``(M, 8)`` candidate-axis image.
        hazard_axes: Candidate-axis names.
        supplement: ``(M, 15)`` supplement aligned with ``H``.
        supplement_names: Supplement column names.

    Returns:
        Dict with ``D`` ``(M, k)``, ``names``, ``sign`` (+1 where high values are
        hazardous, -1 for ``SELDIAG_LOW_TAIL_DESCRIPTORS``), and the truncation
        flags ``flags`` ``(M, 2)`` with ``flag_names``.
    """
    names = [str(a) for a in supplement_names]
    flag_idx = [names.index(f) for f in scfg.SELDIAG_TRUNCATION_FLAGS]
    keep = [i for i in range(len(names)) if i not in flag_idx]
    all_names = list(hazard_axes) + [names[i] for i in keep]
    return {
        "D": np.hstack([np.asarray(H, dtype=float), np.asarray(supplement, dtype=float)[:, keep]]),
        "names": all_names,
        "sign": np.array([-1.0 if a in scfg.SELDIAG_LOW_TAIL_DESCRIPTORS else 1.0
                          for a in all_names]),
        "flags": np.asarray(supplement, dtype=float)[:, flag_idx],
        "flag_names": list(scfg.SELDIAG_TRUNCATION_FLAGS),
    }


def normal_scores(D: np.ndarray) -> np.ndarray:
    """Rank-to-normal transform of every column (average ranks for ties)."""
    D = np.asarray(D, dtype=float)
    M = D.shape[0]
    return np.column_stack([norm.ppf((rankdata(D[:, k]) - 0.5) / M) for k in range(D.shape[1])])


def descriptor_redundancy(
    D: np.ndarray, names: list[str], *, threshold: float, variance_share: float,
) -> dict:
    """Spearman clusters and normal-score PCA of a descriptor set.

    Clusters are ``scengen.diagnostics.spearman_clusters`` (average linkage on
    ``1 - |rho_S|``, cut so every pair at ``|rho_S| >= threshold`` shares a
    cluster). The PCA is on the correlation matrix of the normal scores
    (``scengen.diagnostics.pca_effective_dimension`` for the spectrum and the
    participation ratio ``(sum lambda)^2 / sum lambda^2``). Constant columns
    carry no rank information and are left out (listed in ``constant``).

    Args:
        D: ``(M, k)`` descriptor values.
        names: Descriptor names (columns of ``D``).
        threshold: Spearman ``|rho|`` of the cluster cut.
        variance_share: Share of variance the reported leading components reach.

    Returns:
        Dict with ``names`` (non-constant descriptors), ``constant``, ``rho``,
        ``clusters``, ``eigenvalues``, ``explained``, ``participation_ratio``,
        ``n_components`` (the leading count reaching ``variance_share``), and
        ``top_loadings`` (``(component, descriptor, loading)`` of each leading
        component's highest-``|loading|`` descriptor).
    """
    D = np.asarray(D, dtype=float)
    varying = D.std(axis=0) > 0
    kept = [n for n, v in zip(names, varying) if v]
    D = D[:, varying]
    cl = spearman_clusters(D, kept, threshold=threshold)
    Z = normal_scores(D)
    pca = pca_effective_dimension(Z)
    evr = np.asarray(pca["explained_variance_ratio"], dtype=float)
    n_comp = int(min(np.searchsorted(np.cumsum(evr), variance_share - 1e-12) + 1, len(evr)))
    evals, evecs = np.linalg.eigh(np.corrcoef(Z, rowvar=False))
    order = np.argsort(evals)[::-1]
    evals = np.clip(evals[order], 0.0, None)
    loadings = evecs[:, order] * np.sqrt(evals)
    top = []
    for c in range(n_comp):
        i = int(np.argmax(np.abs(loadings[:, c])))
        top.append((c + 1, kept[i], float(loadings[i, c])))
    return {
        "names": kept,
        "constant": [n for n, v in zip(names, varying) if not v],
        "rho": np.atleast_2d(cl["rho"]),
        "clusters": cl["clusters"],
        "eigenvalues": evals,
        "explained": evr,
        "participation_ratio": float(pca["participation_ratio"]),
        "n_components": n_comp,
        "top_loadings": top,
    }


def ssi_fit_check(ssi: pd.Series, *, level: float) -> pd.DataFrame:
    """Spread and lower-tail count of an SSI series by calendar month.

    Under a well-fitted distribution every calendar month's SSI is standard
    normal: its standard deviation is 1 and ``n * Phi(level)`` of its ``n``
    values lie at or below ``level``.

    Args:
        ssi: SSI series on a DatetimeIndex.
        level: SSI level of the lower-tail count.

    Returns:
        One row per calendar month: ``month``, ``n``, ``sd`` (``ddof=0``, the
        spread about the month's mean with no sample correction),
        ``n_at_or_below`` and ``n_expected``.
    """
    months = ssi.groupby(ssi.index.month)
    n = months.size()
    return pd.DataFrame({
        "month": n.index.to_numpy(),
        "n": n.to_numpy(),
        "sd": months.std(ddof=0).to_numpy(),
        "n_at_or_below": months.apply(lambda v: int((v <= level).sum())).to_numpy(),
        "n_expected": n.to_numpy() * norm.cdf(level),
    })


def hazard_tail_mask(D: np.ndarray, sign: np.ndarray, *, pct: float) -> np.ndarray:
    """Members in each descriptor's hazard-direction pool tail.

    Above the pool ``pct`` percentile where ``sign`` is +1, below the pool
    ``100 - pct`` percentile where it is -1 (strict inequalities).

    Returns:
        ``(M, k)`` boolean mask.
    """
    Do = np.asarray(D, dtype=float) * np.asarray(sign, dtype=float)
    return Do > np.percentile(Do, pct, axis=0)


def exact_snap_limit(
    H: np.ndarray, *, tail_pct: float,
    lo_pct: float = ss.ROBUST_LO_PCT, hi_pct: float = ss.ROBUST_HI_PCT,
) -> np.ndarray:
    """Tail share an exact snap to uniform targets on the clipped box would give.

    Uniform targets on the ``[p_lo, p_hi]`` scaled axis land above the pool
    ``tail_pct`` percentile with probability ``(p_hi - p_tail) / (p_hi - p_lo)``.

    Returns:
        Length-``d`` limits in ``[0, 1]``.
    """
    H = np.asarray(H, dtype=float)
    lo, hi = ss.robust_range_bounds(H, lo_pct=lo_pct, hi_pct=hi_pct)
    tail = np.percentile(H, tail_pct, axis=0)
    return np.clip((hi - tail) / (hi - lo), 0.0, 1.0)


def nearest_member_ess_ratio(X: np.ndarray, rows: np.ndarray, *, workers: int = -1) -> float:
    """``n_eff / N`` of the nearest-member (Voronoi) weights of a selection.

    Reuses the HF design-metric helpers (``hf_design_metrics_run``): every pool
    row's mass goes to its nearest selected member, and ``n_eff`` is the Kish
    effective sample size of those weights.

    Args:
        X: ``(M, d)`` scaled pool coordinates.
        rows: Selected row indices.
        workers: Threads of the nearest-member query (``-1`` = all cores; pass
            ``1`` inside worker processes).
    """
    from scripts.supplemental.hf_design_metrics_run import (
        effective_sample_size, nearest_member, voronoi_masses,
    )

    rows = np.asarray(rows, dtype=int)
    _, idx = nearest_member(X, X[rows], workers=workers)
    return effective_sample_size(voronoi_masses(idx, len(rows))) / len(rows)
