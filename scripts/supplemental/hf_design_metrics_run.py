"""hf_design_metrics_run.py - Design metrics of the Hazard Filling (HF) search ensemble.

Supplemental diagnostic (docs/notes/methods/hf_design_metrics.md; SI Text S4).
Computes, on hazard images alone (no simulation), the measurable properties
that define the HF design, every one in the selector's own p1/p99 range-scaled
coordinates and against the candidate ensemble the selection drew from:

  1. Target displacement. The selection is replayed from its recorded seed
     (identity-checked against the staged ``selected_rows``) to recover the
     target-to-member pairing of the greedy rule, then the exact
     minimum-total-displacement assignment is solved on a k-nearest-candidate
     graph with a certificate of global optimality. Reported: the free lower
     bound (nearest candidate per target), the exact optimum, the greedy total,
     their gap, and the Jaccard overlap of the two selected sets.
  2. Coverage. The minimax distance of the selected set relative to the
     candidate ensemble (the largest candidate-to-nearest-member distance) with
     its mean and quantiles, and the per-axis Kolmogorov-Smirnov distance of
     each scaled marginal to uniform.
  3. Diversity. Minimum-spanning-tree edge lengths of the selected set (mean;
     minimum = the maximin distance).
  4. Range. Per-axis span in scaled units and the count of members beyond the
     historical windows' maximum.
  5. Measure. The nearest-member redistribution of the candidate ensemble's
     mass onto the selected set (Voronoi masses) and its effective sample
     size, plus the per-axis tail share above the candidate p90 and the
     Kolmogorov-Smirnov distance to the candidate marginal.

Sets scored per draw: the realized HF ensemble, its Latin hypercube targets,
the MC ensemble, R random N-subsets of the candidate ensemble (the same-N
reference), and the historical 10-year windows (range and tail statistics).
The per-axis p1/p99 bounds are also recomputed on nested prefixes of the
candidate ensemble (bound stability).

Settings in ``supplemental_config.py`` (``HFM_*``); no CLI value flags. Smoke
mode (``NYCOPT_HFM_SMOKE=1``) uses the locally staged P = 300 / N = 40 images
and prefixes every artifact with ``smoke_``. Missing inputs are skipped with a
notice; an identity or provenance failure is an error.

Run (wrapper: ``workflow/supplemental/hf_design_metrics.sh``)::

    python scripts/supplemental/hf_design_metrics_run.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial import cKDTree
from scipy.spatial.distance import pdist, squareform

PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import supplemental_config as scfg  # noqa: E402

scfg.configure_hfm_env()

import config  # noqa: E402
from scengen import selector_diagnostics as sd  # noqa: E402
from scengen import subsample as ss  # noqa: E402
from scengen.diagnostics import check_hazard_image_provenance, load_hazard_image  # noqa: E402
from src.ensembles import staged_ensemble_dir  # noqa: E402
from src.scenario_designs import get_scenario_design  # noqa: E402


###############################################################################
# Pure helpers (tested in tests/test_hf_design_metrics.py)
###############################################################################

def scaled_coordinates(H: np.ndarray, lo: np.ndarray, hi: np.ndarray, *, clip: bool) -> np.ndarray:
    """Selector-space coordinates of raw hazard values under given bounds.

    Args:
        H: ``(n, m)`` raw hazard values.
        lo, hi: Per-axis bounds (the candidate ensemble's p1/p99).
        clip: True reproduces the selector's clipped geometry; False keeps
            excursions beyond the box, for beyond-box counts.

    Returns:
        ``(n, m)`` scaled coordinates.
    """
    Z = (np.asarray(H, dtype=float) - lo[None, :]) / (hi - lo)[None, :]
    return np.clip(Z, 0.0, 1.0) if clip else Z


def nearest_member(Z_query: np.ndarray, Z_ref: np.ndarray, *, workers: int = -1) -> tuple:
    """Distance and index of each query row's nearest reference row.

    Returns:
        ``(distance, index)`` arrays of length ``len(Z_query)``.
    """
    d, i = cKDTree(Z_ref).query(Z_query, k=1, workers=workers)
    return np.asarray(d, dtype=float).ravel(), np.asarray(i, dtype=int).ravel()


def coverage_summary(d_cover: np.ndarray, *, quantiles) -> dict:
    """Minimax distance, mean, and quantiles of candidate-to-nearest-member distances."""
    d = np.asarray(d_cover, dtype=float)
    out = {
        "minimax_distance": float(d.max()),
        "mean_nearest_member_distance": float(d.mean()),
    }
    for q in quantiles:
        out[f"cover_q{int(round(q * 100)):02d}"] = float(np.quantile(d, q))
    return out


def mst_edges(Z: np.ndarray) -> np.ndarray:
    """Sorted edge lengths of the Euclidean minimum spanning tree of a point set.

    A constant offset is added to every pairwise distance before the tree is
    built so that coincident points (zero distance) keep their edge; every
    spanning tree has ``n - 1`` edges, so the tree itself is unchanged.
    """
    Z = np.asarray(Z, dtype=float)
    if len(Z) < 2:
        return np.zeros(0)
    W = squareform(pdist(Z)) + 1.0
    np.fill_diagonal(W, 0.0)
    tree = minimum_spanning_tree(csr_matrix(W)).tocoo()
    return np.sort(np.maximum(tree.data - 1.0, 0.0))


def mst_edge_stats(edges: np.ndarray) -> dict:
    """Mean, minimum (the maximin distance), and standard deviation of MST edges."""
    e = np.asarray(edges, dtype=float)
    if e.size == 0:
        return {"mst_edge_mean": np.nan, "mst_edge_min": np.nan, "mst_edge_sd": np.nan}
    return {
        "mst_edge_mean": float(e.mean()),
        "mst_edge_min": float(e.min()),
        "mst_edge_sd": float(e.std()),
    }


def voronoi_masses(nearest_idx: np.ndarray, n_members: int) -> np.ndarray:
    """Share of candidates whose nearest member is each member (sums to 1)."""
    idx = np.asarray(nearest_idx, dtype=int)
    return np.bincount(idx, minlength=n_members).astype(float) / len(idx)


def effective_sample_size(w: np.ndarray) -> float:
    """Kish effective sample size ``(sum w)^2 / sum w^2``."""
    w = np.asarray(w, dtype=float)
    s = w.sum()
    return float(s * s / np.sum(w * w))


def ks_to_uniform(x: np.ndarray) -> float:
    """Kolmogorov distance of a sample in [0, 1] to Uniform(0, 1)."""
    x = np.sort(np.asarray(x, dtype=float))
    n = len(x)
    up = np.arange(1, n + 1) / n
    lo = np.arange(0, n) / n
    return float(max(np.max(up - x), np.max(x - lo)))


def ks_two_sample(x: np.ndarray, y_sorted: np.ndarray) -> float:
    """Exact two-sample Kolmogorov distance (both ECDFs at every jump of either)."""
    x = np.sort(np.asarray(x, dtype=float))
    y = np.asarray(y_sorted, dtype=float)
    pts = np.concatenate([x, y])
    fx = np.searchsorted(x, pts, side="right") / len(x)
    fy = np.searchsorted(y, pts, side="right") / len(y)
    return float(np.max(np.abs(fx - fy)))


def axis_marginal_stats(
    z: np.ndarray,
    z_cand_sorted: np.ndarray,
    *,
    p90_scaled: float,
    z_unclipped: np.ndarray | None = None,
    raw: np.ndarray | None = None,
    raw_hist_max: float | None = None,
) -> dict:
    """Per-axis range, uniformity, candidate-marginal distance, and tail statistics.

    Args:
        z: Scaled (clipped) coordinates of one set on one axis.
        z_cand_sorted: Sorted scaled coordinates of the candidate ensemble.
        p90_scaled: The candidate p90 in scaled units.
        z_unclipped: Unclipped coordinates, for the beyond-box count.
        raw: Raw values of the set, for the beyond-historic count.
        raw_hist_max: Largest historical-window value on this axis.
    """
    z = np.asarray(z, dtype=float)
    out = {"min": float(z.min()), "max": float(z.max())}
    out["span"] = out["max"] - out["min"]
    out["ks_uniform"] = ks_to_uniform(z)
    out["ks_candidate"] = ks_two_sample(z, z_cand_sorted)
    out["tail_share_p90"] = float(np.mean(z > p90_scaled))
    out["n_beyond_box"] = (
        int(np.sum((z_unclipped < 0.0) | (z_unclipped > 1.0)))
        if z_unclipped is not None else 0
    )
    out["n_beyond_historic_max"] = (
        int(np.sum(np.asarray(raw, dtype=float) > raw_hist_max))
        if raw is not None and raw_hist_max is not None else np.nan
    )
    return out


def geometry_record(Z_ens: np.ndarray, Z_cand: np.ndarray, *, quantiles) -> tuple[dict, dict]:
    """Coverage, diversity, and measure statistics of one set against the candidates.

    One query of every candidate against the set gives the coverage distances
    and the nearest-member index, hence the Voronoi masses and the effective
    sample size; the MST is built on the set alone.

    Returns:
        ``(scalars, arrays)`` with arrays ``cover_d``, ``weights``, ``mst_edges``.
    """
    d, idx = nearest_member(Z_cand, Z_ens)
    out = coverage_summary(d, quantiles=quantiles)
    w = voronoi_masses(idx, len(Z_ens))
    out["ess"] = effective_sample_size(w)
    out["ess_over_n"] = out["ess"] / len(Z_ens)
    edges = mst_edges(Z_ens)
    out.update(mst_edge_stats(edges))
    return out, {"cover_d": d, "weights": w, "mst_edges": edges}


def bound_stability(H: np.ndarray, prefixes, *, lo_pct: float, hi_pct: float) -> list[dict]:
    """Per-axis p_lo/p_hi on nested prefixes, as deviations from the full image in span units."""
    H = np.asarray(H, dtype=float)
    lo_full, hi_full = ss.robust_range_bounds(H, lo_pct=lo_pct, hi_pct=hi_pct)
    span = hi_full - lo_full
    rows: list[dict] = []
    for p in prefixes:
        p = int(min(p, len(H)))
        lo, hi = ss.robust_range_bounds(H[:p], lo_pct=lo_pct, hi_pct=hi_pct)
        for k in range(H.shape[1]):
            rows.append({
                "prefix": p, "axis_index": k,
                "p_lo": float(lo[k]), "p_hi": float(hi[k]),
                "p_lo_full": float(lo_full[k]), "p_hi_full": float(hi_full[k]),
                "dev_lo_span": float((lo[k] - lo_full[k]) / span[k]),
                "dev_hi_span": float((hi[k] - hi_full[k]) / span[k]),
            })
    return rows


def ecdf_levels(values: np.ndarray, n_levels: int) -> np.ndarray:
    """Quantiles of a sample on a fixed probability grid (persisted for ECDF figures)."""
    return np.quantile(np.asarray(values, dtype=float), np.linspace(0.0, 1.0, n_levels))


def _band(level_rows: list[np.ndarray], lo_q: float = 0.05, hi_q: float = 0.95) -> dict:
    """Pointwise quantile band across replicate level arrays."""
    A = np.vstack(level_rows)
    return {"lo": np.quantile(A, lo_q, axis=0).tolist(),
            "hi": np.quantile(A, hi_q, axis=0).tolist(),
            "mean": A.mean(axis=0).tolist()}


###############################################################################
# Inputs
###############################################################################

def _axis_columns(image_axes, axes) -> list[int]:
    missing = [a for a in axes if a not in image_axes]
    if missing:
        raise RuntimeError(f"hazard image lacks selection axes {missing}")
    return [list(image_axes).index(a) for a in axes]


def load_candidate(draw: int) -> dict | None:
    """Candidate ensemble on the selection axes, with the selector geometry.

    Reads the pool image; if absent, the HF image of the same draw (which
    embeds the full candidate H). Returns None with a notice if neither exists.
    """
    axes = list(config.HAZARD_SELECTION_AXES)
    pool_slug = scfg.hfm_pool_slug(draw)
    hf_slug = scfg.hfm_hf_slug(draw)
    pool_path = staged_ensemble_dir(pool_slug) / "hazard_image.npz"
    hf_path = staged_ensemble_dir(hf_slug) / "hazard_image.npz"
    if pool_path.exists():
        img, source = load_hazard_image(pool_path), pool_slug
    elif hf_path.exists():
        img, source = load_hazard_image(hf_path), hf_slug
        print(f"[hfm] pool image not staged ({pool_path}); candidate H read from {hf_slug}")
    else:
        print(f"[hfm] no candidate image for draw {draw} ({pool_path}); skipping draw")
        return None
    H = np.asarray(img["H"], dtype=float)[:, _axis_columns(img["hazard_axes"], axes)]
    lo_pct, hi_pct = scfg.HFM_BOUND_PCT
    lo, hi = ss.robust_range_bounds(H, lo_pct=lo_pct, hi_pct=hi_pct)
    Z = scaled_coordinates(H, lo, hi, clip=True)
    p90 = np.percentile(H, scfg.HFM_TAIL_PCT, axis=0)
    return {
        "slug": source, "draw": draw, "axes": axes, "H": H, "lo": lo, "hi": hi,
        "Z": Z, "Z_sorted": np.sort(Z, axis=0), "p90_scaled": (p90 - lo) / (hi - lo),
        "tree": cKDTree(Z), "P": int(len(H)),
    }


def load_hf(draw: int, cand: dict) -> dict | None:
    """The staged HF ensemble of ``draw``: selected rows, seed, and consistency checks."""
    slug = scfg.hfm_hf_slug(draw)
    root = staged_ensemble_dir(slug)
    path = root / "hazard_image.npz"
    if not path.exists():
        print(f"[hfm] HF image not staged ({path}); skipping HF for draw {draw}")
        return None
    design = get_scenario_design(scfg.HFM_HF_DESIGN)
    if not scfg.HFM_SMOKE and design.search_ensemble_slug(draw) != slug:
        raise RuntimeError(
            f"HFM slug {slug} differs from the registry's {design.search_ensemble_slug(draw)}; "
            "check NYCOPT_SEARCH_N / NYCOPT_CANDIDATE_POOL_N"
        )
    img = load_hazard_image(path)
    rows = np.asarray(img["selected_rows"], dtype=int)
    if rows.size == 0:
        raise RuntimeError(f"{path} carries no selected_rows; an HF image must record its selection")
    if list(img["chosen_axes"]) != cand["axes"]:
        raise RuntimeError(
            f"{slug} chosen_axes {list(img['chosen_axes'])} differ from "
            f"config.HAZARD_SELECTION_AXES {cand['axes']}"
        )
    H_img = np.asarray(img["H"], dtype=float)[:, _axis_columns(img["hazard_axes"], cand["axes"])]
    if H_img.shape != cand["H"].shape or not np.array_equal(H_img, cand["H"]):
        raise RuntimeError(f"{slug} embeds a candidate H that differs from {cand['slug']}")
    meta = json.loads((root / "_meta.json").read_text(encoding="utf-8"))
    seed = int(meta["selector_seed"])
    if not scfg.HFM_SMOKE and seed != design.selector_seed(draw):
        raise RuntimeError(
            f"{slug} selector_seed {seed} differs from the registry's "
            f"{design.selector_seed(draw)} (NYCOPT_SEED_ROOT drift?)"
        )
    norm = meta.get("normalization", {}).get("axes", {})
    for k, a in enumerate(cand["axes"]):
        if a in norm:
            for key, val in (("lo", cand["lo"][k]), ("hi", cand["hi"][k])):
                if not np.isclose(norm[a][key], val, rtol=1e-6, atol=1e-9):
                    raise RuntimeError(
                        f"{slug} recorded {key}[{a}]={norm[a][key]} but the candidate "
                        f"image gives {val}"
                    )
    return {"slug": slug, "rows": rows, "seed": seed, "n": int(rows.size)}


def load_mc(draw: int, cand: dict) -> tuple[str, np.ndarray] | None:
    """Raw hazard values of the staged MC ensemble on the selection axes, or None."""
    slug = scfg.hfm_mc_slug(draw)
    path = staged_ensemble_dir(slug) / "hazard_image.npz"
    if not path.exists():
        print(f"[hfm] MC image not staged ({path}); skipping MC for draw {draw}")
        return None
    img = load_hazard_image(path)
    rows = np.asarray(img["selected_rows"], dtype=int)
    H = np.asarray(img["H"], dtype=float)
    H = H[rows] if rows.size else H
    return slug, H[:, _axis_columns(img["hazard_axes"], cand["axes"])]


def load_historic(axes) -> dict | None:
    """Historical 10-year window hazard values on the selection axes, or None."""
    path = scfg.HFM_HISTORIC_WINDOWS_PATH
    if not path.exists():
        print(f"[hfm] historic windows cache not found ({path}); skipping historic rows")
        return None
    data = np.load(path, allow_pickle=True)
    check_hazard_image_provenance(data, path)
    H = np.asarray(data["H"], dtype=float)[:, _axis_columns([str(a) for a in data["hazard_axes"]], axes)]
    starts = [str(s) for s in data["window_starts"]] if "window_starts" in data else [str(i) for i in range(len(H))]
    return {"H": H, "window_starts": starts}


###############################################################################
# Scoring
###############################################################################

def replay_selection(cand: dict, hf: dict) -> dict:
    """Greedy replay (identity-asserted), exact assignment, and their gap."""
    a = ss.lhs_nn_assignment(cand["Z"], hf["n"], seed=hf["seed"])
    if not np.array_equal(np.sort(a.rows), hf["rows"]):
        raise RuntimeError(
            f"replayed selection of {hf['slug']} (seed {hf['seed']}) does not reproduce "
            "the staged selected_rows"
        )
    exact = sd.knn_min_sum_assignment(a.targets, cand["Z"], k_ladder=scfg.HFM_KNN_LADDER,
                                      tree=cand["tree"])
    d_free = np.atleast_1d(cand["tree"].query(a.targets, k=1)[0])
    greedy_total = float(a.displacement.sum())
    summary = {
        "greedy_total": greedy_total,
        "greedy_mean": float(a.displacement.mean()),
        "greedy_max": float(a.displacement.max()),
        "exact_total": exact.total,
        "exact_mean": float(exact.displacement.mean()),
        "exact_max": float(exact.displacement.max()),
        "free_total": float(d_free.sum()),
        "free_mean": float(d_free.mean()),
        "gap_rel": (greedy_total - exact.total) / exact.total if exact.total > 0 else np.nan,
        "jaccard_greedy_exact": sd.jaccard(a.rows, exact.rows),
        "exact_k": int(exact.k),
        "exact_certified": bool(exact.certified),
        "exact_slack": float(exact.slack),
        "n_fallback": int(a.n_fallback),
    }
    return {"assignment": a, "exact": exact, "d_free": d_free, "summary": summary}


def score_member_set(
    name: str, draw: int, replicate: int, Z: np.ndarray, cand: dict, *,
    raw: np.ndarray | None, z_unclipped: np.ndarray | None, hist_max: np.ndarray | None,
) -> tuple[dict, list[dict], dict]:
    """Summary row, per-axis rows, and persisted arrays of one member set."""
    scalars, arrays = geometry_record(Z, cand["Z"], quantiles=scfg.HFM_COVERAGE_QUANTILES)
    summary = {"set": name, "draw": draw, "replicate": replicate, "n": int(len(Z)),
               "P": cand["P"], **scalars}
    axis_rows = []
    for k, axis in enumerate(cand["axes"]):
        stats = axis_marginal_stats(
            Z[:, k], cand["Z_sorted"][:, k], p90_scaled=float(cand["p90_scaled"][k]),
            z_unclipped=None if z_unclipped is None else z_unclipped[:, k],
            raw=None if raw is None else raw[:, k],
            raw_hist_max=None if hist_max is None else float(hist_max[k]),
        )
        axis_rows.append({"set": name, "draw": draw, "replicate": replicate, "axis": axis,
                          "lo": float(cand["lo"][k]), "hi": float(cand["hi"][k]), **stats})
    return summary, axis_rows, arrays


def _levels(values: np.ndarray) -> list[float]:
    return ecdf_levels(values, scfg.HFM_ECDF_LEVELS).tolist()


def run_draw(draw: int, hist: dict | None) -> dict | None:
    """Every table row and persisted array for one draw."""
    cand = load_candidate(draw)
    if cand is None:
        return None
    axes = cand["axes"]
    N = scfg.HFM_N
    hist_max = None if hist is None else hist["H"].max(axis=0)
    out = {"summary": [], "axes": [], "points": [], "bounds": [], "dist": {}, "manifest": {}}
    manifest = out["manifest"]
    manifest.update({"pool_slug": cand["slug"], "P": cand["P"], "N": N,
                     "lo": cand["lo"].tolist(), "hi": cand["hi"].tolist(),
                     "p90_scaled": cand["p90_scaled"].tolist(), "skipped": []})
    t0 = time.time()

    # Candidate ensemble: per-axis rows only (its geometry against itself is trivial).
    for k, axis in enumerate(axes):
        stats = axis_marginal_stats(cand["Z"][:, k], cand["Z_sorted"][:, k],
                                    p90_scaled=float(cand["p90_scaled"][k]),
                                    raw=cand["H"][:, k], raw_hist_max=None if hist_max is None else float(hist_max[k]))
        out["axes"].append({"set": "candidate", "draw": draw, "replicate": 0, "axis": axis,
                            "lo": float(cand["lo"][k]), "hi": float(cand["hi"][k]), **stats})
    out["dist"]["candidate_axis_levels"] = {a: _levels(cand["Z"][:, k]) for k, a in enumerate(axes)}

    # Random same-N reference.
    rand_axis_levels = {a: [] for a in axes}
    rand_cover, rand_mst, rand_w = [], [], []
    for r in range(scfg.HFM_RANDOM_REPLICATES):
        rows = ss.random_subsample(cand["H"], N, seed=scfg.HFM_RANDOM_SEED_BASE + r)
        Zr = cand["Z"][rows]
        s, ax_rows, arr = score_member_set("random", draw, r, Zr, cand, raw=cand["H"][rows],
                                           z_unclipped=scaled_coordinates(cand["H"][rows], cand["lo"], cand["hi"], clip=False),
                                           hist_max=hist_max)
        out["summary"].append(s)
        out["axes"].extend(ax_rows)
        for k, a in enumerate(axes):
            rand_axis_levels[a].append(ecdf_levels(Zr[:, k], scfg.HFM_ECDF_LEVELS))
        rand_cover.append(ecdf_levels(arr["cover_d"], scfg.HFM_ECDF_LEVELS))
        rand_mst.append(ecdf_levels(arr["mst_edges"], scfg.HFM_ECDF_LEVELS))
        rand_w.append(np.sort(arr["weights"] * N)[::-1])
    out["dist"]["random_axis_band"] = {a: _band(v) for a, v in rand_axis_levels.items()}
    out["dist"]["random_cover_band"] = _band(rand_cover)
    out["dist"]["random_mst_band"] = _band(rand_mst)
    out["dist"]["random_weights_band"] = _band(rand_w)
    rand_mean = pd.DataFrame([s for s in out["summary"] if s["set"] == "random"]).mean(numeric_only=True)
    print(f"[hfm] draw {draw}: {scfg.HFM_RANDOM_REPLICATES} random references scored "
          f"({time.time() - t0:.0f}s)", flush=True)

    def _named_set(name: str, Z: np.ndarray, *, raw, z_unclipped) -> dict:
        s, ax_rows, arr = score_member_set(name, draw, 0, Z, cand, raw=raw,
                                           z_unclipped=z_unclipped, hist_max=hist_max)
        for m in ("minimax_distance", "mean_nearest_member_distance", "mst_edge_mean",
                  "mst_edge_min", "ess_over_n"):
            s[f"ratio_to_random_{m}"] = s[m] / rand_mean[m] if rand_mean[m] else np.nan
        out["summary"].append(s)
        out["axes"].extend(ax_rows)
        out["dist"].setdefault("sets", {})[f"{name}_d{draw}"] = {
            "axis_values": {a: np.sort(Z[:, k]).tolist() for k, a in enumerate(axes)},
            "cover_levels": _levels(arr["cover_d"]),
            "mst_levels": _levels(arr["mst_edges"]),
            "weights_sorted": np.sort(arr["weights"] * len(Z))[::-1].tolist(),
            "ess_over_n": s["ess_over_n"],
        }
        return arr

    # HF ensemble, its targets, and the assignment gap.
    hf = load_hf(draw, cand)
    if hf is None:
        manifest["skipped"].append("hf")
    else:
        rep = replay_selection(cand, hf)
        a, exact = rep["assignment"], rep["exact"]
        Zh = cand["Z"][hf["rows"]]
        arr_hf = _named_set("hf", Zh, raw=cand["H"][hf["rows"]],
                            z_unclipped=scaled_coordinates(cand["H"][hf["rows"]], cand["lo"], cand["hi"], clip=False))
        out["summary"][-1].update(rep["summary"])
        _named_set("lhs_targets", a.targets, raw=None, z_unclipped=a.targets)
        manifest.update({"hf_slug": hf["slug"], "seed": hf["seed"], "identity_ok": True,
                         "assignment": {"k": exact.k, "certified": exact.certified,
                                        "slack": exact.slack, "ladder": list(exact.ladder),
                                        "n_fallback": a.n_fallback}})
        center = np.full(len(axes), 0.5)
        for i in range(hf["n"]):
            row = {"kind": "target", "draw": draw, "index": i,
                   **{f"z_{a}": float(a_val) for a, a_val in zip(axes, a.targets[i])},
                   "greedy_row": int(a.rows[i]), "greedy_disp": float(a.displacement[i]),
                   "exact_row": int(exact.rows[i]), "exact_disp": float(exact.displacement[i]),
                   "free_disp": float(rep["d_free"][i]),
                   "dist_to_center": float(np.linalg.norm(a.targets[i] - center))}
            out["points"].append(row)
        w_hf = arr_hf["weights"]
        for j, r in enumerate(hf["rows"]):
            out["points"].append({"kind": "hf_member", "draw": draw, "index": j, "pool_row": int(r),
                                  **{f"z_{a}": float(v) for a, v in zip(axes, Zh[j])},
                                  "weight": float(w_hf[j]), "n_weight": float(w_hf[j] * hf["n"])})
        print(f"[hfm] draw {draw}: HF replay ok; greedy {rep['summary']['greedy_total']:.4f} "
              f"exact {exact.total:.4f} (k={exact.k}, certified={exact.certified}) "
              f"jaccard {rep['summary']['jaccard_greedy_exact']:.3f}", flush=True)

    # MC ensemble (scaled with the candidate bounds; may lie beyond the box).
    mc = load_mc(draw, cand)
    if mc is None:
        manifest["skipped"].append("mc")
    else:
        mc_slug, H_mc = mc
        Zu = scaled_coordinates(H_mc, cand["lo"], cand["hi"], clip=False)
        arr_mc = _named_set("mc", np.clip(Zu, 0.0, 1.0), raw=H_mc, z_unclipped=Zu)
        manifest["mc_slug"] = mc_slug
        for j in range(len(H_mc)):
            out["points"].append({"kind": "mc_member", "draw": draw, "index": j,
                                  **{f"z_{a}": float(v) for a, v in zip(axes, np.clip(Zu[j], 0, 1))},
                                  "weight": float(arr_mc["weights"][j]),
                                  "n_weight": float(arr_mc["weights"][j] * len(H_mc))})

    # Historical windows: range and tail rows plus per-window distances.
    if hist is not None:
        Zu_h = scaled_coordinates(hist["H"], cand["lo"], cand["hi"], clip=False)
        Zc_h = np.clip(Zu_h, 0.0, 1.0)
        for k, axis in enumerate(axes):
            stats = axis_marginal_stats(Zc_h[:, k], cand["Z_sorted"][:, k],
                                        p90_scaled=float(cand["p90_scaled"][k]),
                                        z_unclipped=Zu_h[:, k], raw=hist["H"][:, k])
            out["axes"].append({"set": "historic", "draw": draw, "replicate": 0, "axis": axis,
                                "lo": float(cand["lo"][k]), "hi": float(cand["hi"][k]), **stats})
        d_cand = nearest_member(Zc_h, cand["Z"])[0]
        d_hf = nearest_member(Zc_h, cand["Z"][hf["rows"]])[0] if hf is not None else np.full(len(Zc_h), np.nan)
        d_mc = nearest_member(Zc_h, np.clip(scaled_coordinates(mc[1], cand["lo"], cand["hi"], clip=False), 0, 1))[0] if mc is not None else np.full(len(Zc_h), np.nan)
        for j in range(len(Zc_h)):
            out["points"].append({"kind": "historic_window", "draw": draw, "index": j,
                                  "window_start": hist["window_starts"][j],
                                  **{f"z_{a}": float(v) for a, v in zip(axes, Zc_h[j])},
                                  **{f"zu_{a}": float(v) for a, v in zip(axes, Zu_h[j])},
                                  "beyond_box": bool(np.any((Zu_h[j] < 0) | (Zu_h[j] > 1))),
                                  "nearest_cand_d": float(d_cand[j]),
                                  "nearest_hf_d": float(d_hf[j]), "nearest_mc_d": float(d_mc[j])})

    # Nested candidate-ensemble bound stability.
    lo_pct, hi_pct = scfg.HFM_BOUND_PCT
    for row in bound_stability(cand["H"], scfg.HFM_NP_PREFIXES, lo_pct=lo_pct, hi_pct=hi_pct):
        out["bounds"].append({"draw": draw, "axis": axes[row.pop("axis_index")], **row})
    print(f"[hfm] draw {draw} done ({time.time() - t0:.0f}s)", flush=True)
    return out


###############################################################################
# Driver
###############################################################################

def main() -> None:
    """Score every configured draw and persist tables, distributions, and the manifest."""
    t0 = time.time()
    if scfg.HFM_BOUND_PCT != (ss.ROBUST_LO_PCT, ss.ROBUST_HI_PCT):
        sys.exit(f"[hfm] HFM_BOUND_PCT {scfg.HFM_BOUND_PCT} differs from the selector's "
                 f"({ss.ROBUST_LO_PCT}, {ss.ROBUST_HI_PCT})")
    scfg.HFM_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    axes = list(config.HAZARD_SELECTION_AXES)
    print(f"[hfm] P={scfg.HFM_POOL_P} N={scfg.HFM_N} draws={scfg.HFM_DRAWS} axes={axes} "
          f"smoke={scfg.HFM_SMOKE}", flush=True)
    hist = load_historic(axes)

    summary, axis_rows, points, bounds = [], [], [], []
    dist: dict = {"levels": np.linspace(0.0, 1.0, scfg.HFM_ECDF_LEVELS).tolist(), "draws": {}}
    manifest: dict = {"smoke": scfg.HFM_SMOKE, "P": scfg.HFM_POOL_P, "N": scfg.HFM_N, "axes": axes,
                      "bound_pct": list(scfg.HFM_BOUND_PCT), "knn_ladder": list(scfg.HFM_KNN_LADDER),
                      "random_replicates": scfg.HFM_RANDOM_REPLICATES,
                      "historic_windows": None if hist is None else len(hist["H"]),
                      "draws": {}}
    for draw in scfg.HFM_DRAWS:
        res = run_draw(draw, hist)
        if res is None:
            manifest["draws"][str(draw)] = {"skipped": ["candidate"]}
            continue
        summary.extend(res["summary"])
        axis_rows.extend(res["axes"])
        points.extend(res["points"])
        bounds.extend(res["bounds"])
        dist["draws"][str(draw)] = res["dist"]
        manifest["draws"][str(draw)] = res["manifest"]
    if not summary:
        sys.exit("[hfm] no draw could be scored; stage the candidate and HF images first")

    pd.DataFrame(summary).to_csv(scfg.hfm_table_path("hfm_summary"), index=False)
    pd.DataFrame(axis_rows).to_csv(scfg.hfm_table_path("hfm_axes"), index=False)
    pd.DataFrame(points).to_csv(scfg.hfm_table_path("hfm_points"), index=False)
    pd.DataFrame(bounds).to_csv(scfg.hfm_table_path("hfm_bound_stability"), index=False)
    scfg.hfm_json_path("hfm_distributions").write_text(json.dumps(dist), encoding="utf-8")
    scfg.hfm_json_path("hfm_manifest").write_text(json.dumps(manifest, indent=2, default=str),
                                                  encoding="utf-8")
    print(f"[hfm] tables -> {scfg.HFM_TABLES_DIR} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
