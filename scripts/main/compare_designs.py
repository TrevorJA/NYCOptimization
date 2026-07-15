"""compare_designs.py - Cross-design comparison of re-evaluated Pareto sets.

Every scenario design is optimized independently and its final Pareto set is
re-evaluated on ONE common held-out test ensemble E_test. Re-evaluated
performance is the SOLE basis of cross-design comparison
(``docs/notes/methods/experimental_design.md`` §"Controls for fair comparison",
item 3). This script is the only consumer of the per-run robustness artifacts
written by ``src.robustness`` -- it reads them across designs, draws and seeds
and produces the cross-design tables and figures.

What it produces, and why each piece exists
-------------------------------------------
1. **Satisficing-threshold sweep (the main-text figure).** Quinn et al. (2020)
   found that robustness-rank agreement ACROSS scenario designs *degrades as the
   satisficing criterion becomes more stringent*, so the design effect is largest
   at the conservative end and a single threshold could manufacture or hide the
   entire result. Our thresholds are still provisional, which makes this the
   analysis that renders the comparison credible. The per-run
   ``robustness_threshold_spectrum.csv`` cannot answer this: it is univariate,
   and the primary metric is the MULTIVARIATE (all-criteria conjunction) Starr
   domain criterion. So the sweep is recomputed from the raw cube
   (``robustness.load_raw`` + ``robustness.satisficing_multivariate`` with a
   swept threshold dict).

   The sweep is driven by ONE scalar stringency knob ``s``: each objective's
   threshold is the ``s``-quantile of the *pooled across designs* E_test cell
   distribution (oriented, so ``s`` = the marginal fraction of pooled cells the
   criterion excludes). Pooling across designs is load-bearing -- a per-design
   quantile would make "the same stringency" a different magnitude for each
   design and the comparison at fixed ``s`` would be meaningless. The
   registry-default thresholds are located on the same axis and marked.

2. **Cross-design scorecard aggregation + ranking stability.** Metric choice
   changes rankings (Herman et al. 2015; McPhail et al. 2018), so if satisficing
   and maximin rank the campaign designs differently that is itself a result.
   This is Kendall tau_b among the DESIGN rankings induced by each metric (not
   among solution rankings -- ``robustness.ranking_stability`` already does those).

3. **Variance components.** ``outcome ~ design (fixed) + draw(design) (random) +
   seed(draw) (random)``. The unit of analysis for between-design tests is the
   DRAW; seeds within a draw are pseudoreplicates, so effective n ~= K, not K*S
   (experimental_design.md §"Replication"). The F-test for design accordingly
   uses the draw mean-square as its denominator, never the seed residual.

4. **Raw performance distributions (mandatory sanity check).** A robustness
   scalar can be stable, optimizable and still perverse: Huang et al. (2025) show
   a deviation metric driven to zero by being *uniformly terrible* (a policy is
   "robust" because it is consistently awful), and Bonham et al. (2024) show a
   saturated criterion ties everything. So the raw re-evaluated distributions are
   co-reported next to every robustness number, with the satisficing threshold
   drawn on the same axis (the threshold-margin diagnostic of Gold et al. 2023),
   and degeneracy flags are printed loudly.

5. **Attainability screen.** Pooling the cubes of ALL designs answers "is this
   E_test realization winnable by ANY policy from ANY design?". Shavazipour et
   al. (2021) found 23% of their test scenarios unwinnable by any feasible
   policy; without this you cannot separate "this design searched badly" from
   "this test realization is impossible".

Inputs (per design, per moea slug, per re-eval tag, optionally per seed):
    outputs/{design}/{moea_slug}/reeval/{reeval_tag}[/seed_NN]/
        reeval_raw.parquet | reeval_raw.csv.gz   (+ reeval_raw_meta.json)
        robustness_scorecard.csv
        robustness_quantiles.csv
        robustness_attainability.csv

Outputs:
    outputs/comparison/{slug}/{reeval_tag}/*.csv
    figures/comparison/{slug}/robustness/*.png

Settings come from module constants + ``NYCOPT_*`` env overrides (repo rule: no
CLI value flags). ``--formulation`` / ``--reeval-tag`` / ``--seed`` are
identifiers.

Run::

    NYCOPT_REEVAL_ENSEMBLE_PRESET=etest_kn_10yr_n1000 \
        python scripts/main/compare_designs.py --formulation ffmp
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import matplotlib

if not os.environ.get("DISPLAY") and sys.platform != "win32":
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import config  # noqa: E402
import src.robustness as rob  # noqa: E402
from src.plotting import style  # noqa: E402
from src.reeval_core import reeval_tag as reeval_tag_of  # noqa: E402
from src.scenario_designs import SCENARIO_DESIGNS, campaign_designs  # noqa: E402


###############################################################################
# Settings (env-overridable; never CLI value flags)
###############################################################################

def _parse_float_list_env(name: str, default: tuple[float, ...]) -> tuple[float, ...]:
    raw = os.environ.get(name)
    if not raw:
        return tuple(default)
    return tuple(float(x) for x in raw.split(",") if x.strip())


#: Stringency grid ``s``. ``s`` is the marginal fraction of the pooled E_test cell
#: distribution that each objective's criterion excludes, so ``s`` increases from
#: lenient to conservative and moves every criterion together.
STRINGENCY_GRID: tuple[float, ...] = _parse_float_list_env(
    "NYCOPT_COMPARE_STRINGENCY",
    tuple(round(float(s), 2) for s in np.arange(0.10, 0.901, 0.05)),
)

#: The primary robustness metric (Starr multivariate domain criterion).
PRIMARY_METRIC = "sat_multivariate"

#: Saturation bounds for the degeneracy screen. An objective whose satisficing
#: fraction sits outside these for EVERY design cannot discriminate designs
#: (Bonham et al. 2024: a saturated criterion ties everything).
SATURATION_HI = float(os.environ.get("NYCOPT_COMPARE_SAT_HI", "0.98"))
SATURATION_LO = float(os.environ.get("NYCOPT_COMPARE_SAT_LO", "0.02"))

#: A design is "highly robust" above this; if its median raw performance is
#: simultaneously in the pooled worst tail, that is the Huang et al. (2025)
#: uniformly-terrible pathology and is flagged.
HIGH_ROBUSTNESS = float(os.environ.get("NYCOPT_COMPARE_HIGH_ROBUSTNESS", "0.5"))
CATASTROPHIC_TAIL_Q = float(os.environ.get("NYCOPT_COMPARE_TAIL_Q", "0.05"))

#: Cap on cells sampled per (run, objective) when building the pooled reference
#: distribution, so the quantile grid stays cheap for a large E_test.
POOL_SAMPLE_PER_RUN = int(os.environ.get("NYCOPT_COMPARE_POOL_SAMPLE", "200000"))
POOL_SAMPLE_SEED = 0

_QCOL_RE = re.compile(r"^q\d{2}$")
_DRAW_RE = re.compile(r"_d(\d+)(?=_|$)")


###############################################################################
# Run discovery
###############################################################################

@dataclass(frozen=True)
class ReevalRun:
    """One re-evaluated Pareto set: a (design, draw, seed) replicate on E_test.

    Attributes:
        design: Scenario-design name (top-level output partition).
        slug: The moea slug the run was optimized under.
        draw: Independent ensemble-draw index k, parsed from the slug's ``_d{k}``
            token (0 when absent).
        seed: MOEA seed, parsed from a ``seed_NN`` subdir (None when the re-eval
            was written seed-pooled).
        path: The re-eval output directory holding the robustness artifacts.
    """

    design: str
    slug: str
    draw: int
    seed: Optional[int]
    path: Path

    @property
    def key(self) -> tuple:
        return (self.design, self.draw, self.seed)


def _has_raw(d: Path) -> bool:
    return (d / "reeval_raw_meta.json").exists() and (
        (d / "reeval_raw.parquet").exists() or (d / "reeval_raw.csv.gz").exists()
    )


def _leaf_dirs(tag_dir: Path, seed: Optional[int]) -> list[tuple[Path, Optional[int]]]:
    """Return ``(dir, seed)`` for every re-eval leaf under one re-eval tag dir."""
    leaves: list[tuple[Path, Optional[int]]] = []
    if seed is None and _has_raw(tag_dir):
        leaves.append((tag_dir, None))
    for sub in sorted(tag_dir.glob("seed_*")):
        if not sub.is_dir() or not _has_raw(sub):
            continue
        try:
            s = int(sub.name.split("_", 1)[1])
        except ValueError:
            continue
        if seed is None or s == seed:
            leaves.append((sub, s))
    return leaves


def discover_runs(formulation: str, reeval_tag: str, seed: Optional[int] = None,
                  outputs_root: Optional[Path] = None) -> list[ReevalRun]:
    """Find every campaign design's re-eval leaf for one formulation + E_test tag.

    A design that has not been run (or has not been re-evaluated on this tag) is
    warned about and skipped -- never fatal, so the comparison can be assembled
    incrementally as the campaign lands.

    Args:
        formulation: Formulation identifier (e.g. ``"ffmp"``, ``"ffmp_8"``).
        reeval_tag: The common held-out ensemble's tag (``reeval_core.reeval_tag``).
        seed: Restrict to one MOEA seed; None keeps all.
        outputs_root: Root of the output tree. Defaults to ``config.OUTPUTS_DIR``.

    Returns:
        The discovered runs, sorted by (design, draw, seed).
    """
    root = Path(outputs_root) if outputs_root is not None else config.OUTPUTS_DIR
    slug_re = re.compile(rf"^{re.escape(formulation)}_obj\d+")

    runs: list[ReevalRun] = []
    for design in campaign_designs():
        ddir = root / design
        if not ddir.is_dir():
            warnings.warn(f"[compare] design '{design}' has no output dir; skipped.")
            continue
        found = 0
        for slug_dir in sorted(p for p in ddir.iterdir() if p.is_dir()):
            if not slug_re.match(slug_dir.name):
                continue
            tag_dir = slug_dir / "reeval" / reeval_tag
            if not tag_dir.is_dir():
                continue
            m = _DRAW_RE.search(slug_dir.name)
            draw = int(m.group(1)) if m else 0
            for leaf, s in _leaf_dirs(tag_dir, seed):
                runs.append(ReevalRun(design, slug_dir.name, draw, s, leaf))
                found += 1
        if not found:
            warnings.warn(
                f"[compare] design '{design}' has no re-eval on tag "
                f"'{reeval_tag}' for formulation '{formulation}'; skipped."
            )
    return sorted(runs, key=lambda r: (r.design, r.draw, -1 if r.seed is None else r.seed))


def common_slug(runs: Iterable[ReevalRun], formulation: str) -> str:
    """The slug shared by the runs, with the ``_d{k}`` draw token stripped.

    Used to name the comparison's own output dirs. Falls back to
    ``config.derive_slug(formulation)`` when the runs disagree (e.g. a mixed
    MOEA-config campaign), so output naming never depends silently on the
    ambient env.
    """
    bases = {_DRAW_RE.sub("", r.slug) for r in runs}
    if len(bases) == 1:
        return bases.pop()
    return config.derive_slug(formulation)


###############################################################################
# Loading + metric orientation
###############################################################################

@dataclass
class LoadedRun:
    """A run's scorecard, its per-objective metadata, and metric orientations."""

    run: ReevalRun
    scorecard: pd.DataFrame
    higher_better: dict
    thresholds: dict
    kinds: dict
    directions: dict


def _orientations(directions: dict, columns: Iterable[str]) -> dict:
    """Map every scorecard column to whether larger = more robust.

    Satisficing fractions are higher-better; ``vs_baseline__*`` is a shortfall
    (lower-better); ``laplace__*`` / ``maximin__*`` are in natural units, so they
    follow their objective's own direction.
    """
    hb: dict = {}
    for col in columns:
        if col == PRIMARY_METRIC or col.startswith("sat_uni__"):
            hb[col] = True
        elif col.startswith("vs_baseline__"):
            hb[col] = False
        elif "__" in col:
            _, name = col.split("__", 1)
            hb[col] = directions.get(name) == "maximize"
        else:
            hb[col] = True
    return hb


def load_runs(runs: Iterable[ReevalRun]) -> list[LoadedRun]:
    """Load each run's ``robustness_scorecard.csv`` + its raw-matrix metadata."""
    import json

    loaded: list[LoadedRun] = []
    for r in runs:
        sc_path = r.path / "robustness_scorecard.csv"
        if not sc_path.exists():
            warnings.warn(
                f"[compare] {r.design} draw={r.draw} seed={r.seed}: no "
                f"robustness_scorecard.csv (run `python -m src.robustness`); skipped."
            )
            continue
        sc = pd.read_csv(sc_path, index_col="solution_id")
        meta = json.loads((r.path / "reeval_raw_meta.json").read_text())
        base = list(meta["base_names"])
        loaded.append(LoadedRun(
            run=r,
            scorecard=sc,
            higher_better=_orientations(meta.get("directions", {}), sc.columns),
            thresholds={k: meta.get("thresholds", {}).get(k) for k in base},
            kinds={k: meta.get("kinds", {}).get(k) for k in base},
            directions={k: meta.get("directions", {}).get(k) for k in base},
        ))
    return loaded


def _best(values: np.ndarray, higher: bool) -> float:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return float("nan")
    return float(v.max() if higher else v.min())


def _median(values: np.ndarray) -> float:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    return float(np.median(v)) if v.size else float("nan")


###############################################################################
# 2. Cross-design scorecard aggregation
###############################################################################

def design_summary(loaded: Iterable[LoadedRun]) -> pd.DataFrame:
    """Per (design, draw, seed) best-solution and median value of every metric.

    "Best" is the decision-relevant number (a decision maker deploys ONE policy);
    "median" guards against a design winning the comparison on one lucky policy.

    Returns:
        Tidy frame: design, draw, seed, slug, metric, higher_better, n_solutions,
        best, median.
    """
    rows = []
    for lr in loaded:
        r = lr.run
        for col in lr.scorecard.columns:
            hb = lr.higher_better.get(col, True)
            v = lr.scorecard[col].to_numpy(dtype=float)
            rows.append({
                "design": r.design, "draw": r.draw, "seed": r.seed, "slug": r.slug,
                "metric": col, "higher_better": hb,
                "n_solutions": int(np.isfinite(v).sum()),
                "best": _best(v, hb), "median": _median(v),
            })
    return pd.DataFrame(rows)


def design_level(summary: pd.DataFrame, statistic: str = "best") -> pd.DataFrame:
    """Collapse the per-run summary to one value per (design, metric).

    The mean over that design's runs -- NOT a pool of all its solutions, which
    would reward a design merely for having been replicated more often.
    """
    if summary.empty:
        return pd.DataFrame()
    wide = summary.pivot_table(index="design", columns="metric", values=statistic,
                               aggfunc="mean")
    return wide.dropna(axis=1, how="all")


def design_ranking_stability(level: pd.DataFrame,
                             higher_better: dict) -> pd.DataFrame:
    """Kendall tau_b between the DESIGN rankings induced by each metric.

    Not solution rankings -- ``robustness.ranking_stability`` already does those.
    If satisficing and maximin order the designs differently, the design ranking
    is metric-dependent and that is a headline result, not a footnote (Herman et
    al. 2015; McPhail et al. 2018).
    """
    if level.empty or level.shape[0] < 3:
        warnings.warn(
            "[compare] fewer than 3 designs available; design-ranking tau_b is "
            "undefined and will be all-NaN."
        )
    return rob.ranking_stability(level, higher_better)


###############################################################################
# 2b. Variance components
###############################################################################

def variance_components(summary: pd.DataFrame, metric: str = PRIMARY_METRIC,
                        statistic: str = "best") -> pd.DataFrame:
    """Nested decomposition of ``outcome ~ design + draw(design) + seed(draw)``.

    The unit of analysis for between-design inference is the DRAW: seeds within a
    draw are pseudoreplicates, so the effective sample size is ~K (the number of
    draws), never K*S (experimental_design.md §"Replication"). The design F-test
    therefore uses MS_draw as its denominator -- using the seed residual would
    silently pretend n = K*S and inflate significance.

    A ``statsmodels`` MixedLM (``value ~ C(design)``, random intercept for draw
    nested in design) is fitted alongside when available; its variance estimates
    are reported as extra rows with ``method="mixedlm"``. Absent statsmodels the
    ANOVA-style sums-of-squares decomposition stands alone.

    Args:
        summary: The tidy frame from :func:`design_summary`.
        metric: Scorecard column to decompose.
        statistic: ``"best"`` or ``"median"``.

    Returns:
        Tidy frame: component, method, n, df, ss, ms, var_component, f_stat,
        p_value, note.
    """
    df = summary[summary["metric"] == metric][["design", "draw", "seed", statistic]]
    df = df.rename(columns={statistic: "value"}).dropna(subset=["value"])
    rows: list[dict] = []

    n_designs = df["design"].nunique()
    cell = df.groupby(["design", "draw"])["value"]
    n_draws = len(cell)
    n_obs = len(df)

    if n_obs == 0 or n_designs < 2:
        return pd.DataFrame([{
            "component": "design", "method": "anova", "n": n_obs, "df": np.nan,
            "ss": np.nan, "ms": np.nan, "var_component": np.nan, "f_stat": np.nan,
            "p_value": np.nan,
            "note": "insufficient data (need >=2 designs with a finite outcome)",
        }])

    grand = df["value"].mean()
    dmean = df.groupby("design")["value"].transform("mean")
    cmean = df.groupby(["design", "draw"])["value"].transform("mean")

    ss_design = float(((dmean - grand) ** 2).sum())
    ss_draw = float(((cmean - dmean) ** 2).sum())
    ss_seed = float(((df["value"] - cmean) ** 2).sum())

    df_design = n_designs - 1
    df_draw = n_draws - n_designs
    df_seed = n_obs - n_draws

    ms_design = ss_design / df_design if df_design > 0 else np.nan
    ms_draw = ss_draw / df_draw if df_draw > 0 else np.nan
    ms_seed = ss_seed / df_seed if df_seed > 0 else np.nan

    # Balanced-EMS variance components. s_bar = mean seeds per draw; k_bar = mean
    # draws per design.
    s_bar = n_obs / n_draws if n_draws else np.nan
    k_bar = n_draws / n_designs if n_designs else np.nan
    var_seed = ms_seed
    var_draw = ((ms_draw - ms_seed) / s_bar
                if np.isfinite(ms_draw) and np.isfinite(ms_seed) and s_bar
                else np.nan)

    f_design, p_design = np.nan, np.nan
    if np.isfinite(ms_design) and np.isfinite(ms_draw) and ms_draw > 0:
        f_design = ms_design / ms_draw
        try:
            from scipy.stats import f as f_dist
            p_design = float(f_dist.sf(f_design, df_design, df_draw))
        except Exception:  # noqa: BLE001 - scipy absent: report F without p
            p_design = np.nan

    rows.append({
        "component": "design (fixed)", "method": "anova", "n": n_designs,
        "df": df_design, "ss": ss_design, "ms": ms_design,
        "var_component": np.nan, "f_stat": f_design, "p_value": p_design,
        "note": ("F = MS_design / MS_draw; the DRAW is the unit of analysis, so "
                 f"effective n = {n_draws} draws, NOT {n_obs} (design, draw, seed) "
                 "observations"),
    })
    draw_note = f"mean draws per design K_bar = {k_bar:.2f}" if np.isfinite(k_bar) else ""
    if np.isfinite(var_draw) and var_draw < 0:
        # MS_draw < MS_seed. The method-of-moments estimator is unbiased, not
        # non-negative, so a negative estimate is the honest reading "no detectable
        # ensemble-draw variance beyond seed noise" -- reported, not clipped to 0.
        draw_note += (" | negative MoM estimate: draw variance is not distinguishable "
                      "from seed noise (interpret as ~0)")
    rows.append({
        "component": "draw(design) (random)", "method": "anova", "n": n_draws,
        "df": df_draw, "ss": ss_draw, "ms": ms_draw, "var_component": var_draw,
        "f_stat": np.nan, "p_value": np.nan, "note": draw_note,
    })
    rows.append({
        "component": "seed(draw) (random / residual)", "method": "anova",
        "n": n_obs, "df": df_seed, "ss": ss_seed, "ms": ms_seed,
        "var_component": var_seed, "f_stat": np.nan, "p_value": np.nan,
        "note": (f"mean seeds per draw S_bar = {s_bar:.2f}"
                 if np.isfinite(s_bar) else "")
                + ("" if df_seed > 0 else " | no seed replication: residual "
                                          "variance is not identified"),
    })

    rows.extend(_mixedlm_rows(df, n_obs, n_draws))
    out = pd.DataFrame(rows)
    out.insert(0, "metric", metric)
    out.insert(1, "statistic", statistic)
    return out


def _mixedlm_rows(df: pd.DataFrame, n_obs: int, n_draws: int) -> list[dict]:
    """MixedLM variance estimates for the same nested model, when available."""
    if df["seed"].nunique(dropna=False) < 2 or n_draws <= df["design"].nunique():
        return []  # no seed replication or no draw replication: nothing to fit
    try:
        import statsmodels.formula.api as smf
    except Exception:  # noqa: BLE001 - statsmodels absent
        return []
    d = df.copy()
    d["group"] = d["design"].astype(str) + "|d" + d["draw"].astype(str)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = smf.mixedlm("value ~ C(design)", d, groups=d["group"]).fit()
        var_draw = float(np.asarray(fit.cov_re).ravel()[0])
        var_resid = float(fit.scale)
    except Exception as exc:  # noqa: BLE001 - singular fit on a tiny design
        warnings.warn(f"[compare] MixedLM did not converge ({exc}); ANOVA only.")
        return []
    return [
        {"component": "draw(design) (random)", "method": "mixedlm", "n": n_draws,
         "df": np.nan, "ss": np.nan, "ms": np.nan, "var_component": var_draw,
         "f_stat": np.nan, "p_value": np.nan,
         "note": "MixedLM: value ~ C(design), random intercept for draw in design"},
        {"component": "seed(draw) (random / residual)", "method": "mixedlm",
         "n": n_obs, "df": np.nan, "ss": np.nan, "ms": np.nan,
         "var_component": var_resid, "f_stat": np.nan, "p_value": np.nan,
         "note": "MixedLM residual scale"},
    ]


###############################################################################
# 1. Satisficing-threshold sweep
###############################################################################

def pooled_cells(runs: Iterable[ReevalRun]) -> dict[str, np.ndarray]:
    """Pool every design's finite E_test cells, per base objective.

    The pooled distribution is the common yardstick the stringency knob is
    defined against. It MUST be pooled across designs: a per-design quantile
    would make "the same stringency" a different magnitude for each design, and
    the design comparison at fixed stringency would be meaningless.
    """
    rng = np.random.default_rng(POOL_SAMPLE_SEED)
    acc: dict[str, list[np.ndarray]] = {}
    for r in runs:
        raw = rob.load_raw(r.path)
        for k, name in enumerate(raw.base_names):
            v = raw.cube[:, :, k]
            v = v[np.isfinite(v)]
            if v.size > POOL_SAMPLE_PER_RUN:
                v = rng.choice(v, POOL_SAMPLE_PER_RUN, replace=False)
            acc.setdefault(name, []).append(v)
    return {n: np.concatenate(v) if v else np.array([]) for n, v in acc.items()}


def thresholds_at(stringency: float, pooled: dict[str, np.ndarray],
                  kinds: dict) -> dict:
    """The threshold dict at one stringency level.

    For a ``ge`` criterion (satisfy if value >= thr) a HIGHER threshold is more
    stringent, so ``thr = Q(s)``; for ``le`` a LOWER threshold is more stringent,
    so ``thr = Q(1 - s)``. In both cases exactly a fraction ``s`` of the pooled
    cells fails that objective's criterion marginally -- one knob, moving every
    criterion together from lenient to conservative.

    An objective with no pooled data gets a permissive infinite threshold (rather
    than ``None``, which ``robustness._satisfaction_cube`` would treat as an
    always-false column and silently zero the multivariate metric).
    """
    out: dict = {}
    for name, kind in kinds.items():
        vals = pooled.get(name, np.array([]))
        if vals.size == 0:
            out[name] = -np.inf if kind == "ge" else np.inf
            continue
        q = stringency if kind == "ge" else 1.0 - stringency
        out[name] = float(np.quantile(vals, q))
    return out


def default_stringency(pooled: dict[str, np.ndarray], thresholds: dict,
                       kinds: dict) -> pd.DataFrame:
    """Locate each registry-default threshold on the pooled stringency axis.

    Returns:
        Frame: objective, kind, default_threshold, default_stringency (the
        marginal fraction of pooled cells the default criterion excludes).
    """
    rows = []
    for name, kind in kinds.items():
        thr = thresholds.get(name)
        vals = pooled.get(name, np.array([]))
        if thr is None or vals.size == 0:
            s = np.nan
        elif kind == "ge":
            s = float(np.mean(vals < thr))
        else:
            s = float(np.mean(vals > thr))
        rows.append({"objective": name, "kind": kind, "default_threshold": thr,
                     "default_stringency": s})
    return pd.DataFrame(rows)


def threshold_sweep(runs: list[ReevalRun], pooled: dict[str, np.ndarray],
                    kinds: dict, grid: Iterable[float] = STRINGENCY_GRID,
                    ) -> pd.DataFrame:
    """Multivariate satisficing vs stringency, per run.

    Recomputed from the raw cube because the per-run
    ``robustness_threshold_spectrum.csv`` is univariate and the primary metric is
    the all-criteria conjunction.

    Returns:
        Tidy frame: design, draw, seed, stringency, is_default, best, median.
    """
    grid = list(grid)
    thr_by_s = {s: thresholds_at(s, pooled, kinds) for s in grid}
    rows = []
    for r in runs:
        raw = rob.load_raw(r.path)
        for s in grid:
            v = rob.satisficing_multivariate(raw, thresholds=thr_by_s[s]).to_numpy()
            rows.append({"design": r.design, "draw": r.draw, "seed": r.seed,
                         "stringency": s, "is_default": False,
                         "best": _best(v, True), "median": _median(v)})
        v = rob.satisficing_multivariate(raw).to_numpy()   # registry defaults
        rows.append({"design": r.design, "draw": r.draw, "seed": r.seed,
                     "stringency": np.nan, "is_default": True,
                     "best": _best(v, True), "median": _median(v)})
    return pd.DataFrame(rows)


def sweep_design_level(sweep: pd.DataFrame, statistic: str = "best") -> pd.DataFrame:
    """Design x stringency matrix of the primary metric (mean over that design's runs)."""
    grid_rows = sweep[~sweep["is_default"]]
    return grid_rows.pivot_table(index="design", columns="stringency",
                                 values=statistic, aggfunc="mean")


def rank_agreement(sweep: pd.DataFrame) -> pd.DataFrame:
    """Kendall tau_b of the DESIGN ranking at each stringency vs at the default.

    The direct test of whether the design ranking is threshold-invariant. If the
    lines cross -- if tau_b falls away from 1 as the criterion tightens -- the
    design effect is threshold-dependent (Quinn et al. 2020) and MUST be reported
    as such rather than quoted at one convenient threshold.
    """
    rows = []
    for statistic in ("best", "median"):
        ref = (sweep[sweep["is_default"]]
               .groupby("design")[statistic].mean())
        mat = sweep_design_level(sweep, statistic)
        for s in mat.columns:
            common = ref.index.intersection(mat.index)
            tau = rob._kendall_tau(mat.loc[common, s].to_numpy(dtype=float),
                                   ref.loc[common].to_numpy(dtype=float))
            rows.append({"statistic": statistic, "stringency": float(s),
                         "n_designs": int(len(common)), "tau_b_vs_default": tau})
    return pd.DataFrame(rows)


###############################################################################
# 3. Raw performance distributions + degeneracy screen
###############################################################################

def performance_distributions(loaded: list[LoadedRun]) -> pd.DataFrame:
    """Re-evaluated raw performance quantiles, per design and objective.

    Two solution groups per design: ``best`` (the design's best solution by the
    primary metric -- the policy a decision maker would deploy) and ``median``
    (the across-solution median of each quantile -- the design's typical policy).
    Averaged over the design's runs.

    Returns:
        Frame: design, objective, solution_group, q05, q25, q50, q75, q95.
    """
    frames = []
    for lr in loaded:
        qpath = lr.run.path / "robustness_quantiles.csv"
        if not qpath.exists():
            continue
        q = pd.read_csv(qpath)
        qcols = [c for c in q.columns if _QCOL_RE.match(c)]
        if not qcols:
            continue
        sc = lr.scorecard
        if PRIMARY_METRIC in sc.columns and sc[PRIMARY_METRIC].notna().any():
            best_sid = sc[PRIMARY_METRIC].idxmax()
            b = q.loc[q["solution_id"] == best_sid, ["objective"] + qcols].copy()
            b["solution_group"] = "best"
            b["design"] = lr.run.design
            frames.append(b)
        m = q.groupby("objective", as_index=False)[qcols].median()
        m["solution_group"] = "median"
        m["design"] = lr.run.design
        frames.append(m)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    qcols = [c for c in out.columns if _QCOL_RE.match(c)]
    return (out.groupby(["design", "objective", "solution_group"], as_index=False)[qcols]
            .mean())


def degeneracy_flags(summary: pd.DataFrame, perf: pd.DataFrame,
                     pooled: dict[str, np.ndarray], kinds: dict,
                     directions: dict) -> pd.DataFrame:
    """Screen for robustness scalars that are stable, optimizable and perverse.

    Two failure modes, both of which would invalidate a design ranking computed
    on top of them:

    * **Saturation** -- an objective whose satisficing fraction is above
      ``SATURATION_HI`` (or below ``SATURATION_LO``) for EVERY design cannot
      discriminate designs; the criterion ties everything (Bonham et al. 2024).
    * **Uniformly terrible** -- a design that scores highly robust while its
      median re-evaluated performance sits in the pooled worst tail. The
      robustness scalar is then measuring consistency, not quality (Huang et al.
      2025).

    Returns:
        Frame: flag, objective, design, value, detail. Empty when nothing fires.
    """
    flags: list[dict] = []

    med = design_level(summary, "median")
    for name in kinds:
        col = f"sat_uni__{name}"
        if col not in med.columns:
            continue
        v = med[col].dropna()
        if v.empty:
            continue
        if (v > SATURATION_HI).all() or (v < SATURATION_LO).all():
            flags.append({
                "flag": "saturated_objective", "objective": name, "design": "ALL",
                "value": float(v.mean()),
                "detail": (f"median-solution satisficing fraction is "
                           f"{'>' if (v > SATURATION_HI).all() else '<'} "
                           f"{SATURATION_HI if (v > SATURATION_HI).all() else SATURATION_LO} "
                           f"for every design -- non-discriminating"),
            })

    best = design_level(summary, "best")
    if PRIMARY_METRIC in best.columns and not perf.empty:
        for design, rob_val in best[PRIMARY_METRIC].dropna().items():
            if rob_val < HIGH_ROBUSTNESS:
                continue
            sub = perf[(perf["design"] == design) & (perf["solution_group"] == "best")]
            for _, row in sub.iterrows():
                name = row["objective"]
                vals = pooled.get(name, np.array([]))
                if vals.size == 0 or not np.isfinite(row.get("q50", np.nan)):
                    continue
                maximize = directions.get(name) == "maximize"
                cut = float(np.quantile(
                    vals, CATASTROPHIC_TAIL_Q if maximize else 1 - CATASTROPHIC_TAIL_Q))
                worse = row["q50"] < cut if maximize else row["q50"] > cut
                if worse:
                    flags.append({
                        "flag": "high_robustness_catastrophic_median",
                        "objective": name, "design": design,
                        "value": float(row["q50"]),
                        "detail": (f"{PRIMARY_METRIC}={rob_val:.3f} >= "
                                   f"{HIGH_ROBUSTNESS} but the best solution's "
                                   f"median E_test performance is worse than the "
                                   f"pooled {CATASTROPHIC_TAIL_Q:.0%} tail "
                                   f"({cut:.4g})"),
                    })
    return pd.DataFrame(flags, columns=["flag", "objective", "design", "value",
                                        "detail"])


###############################################################################
# 4. Attainability screen
###############################################################################

def attainability(runs: list[ReevalRun]) -> pd.DataFrame:
    """Per-design and POOLED attainability of the E_test realizations.

    A realization is attainable for a design if ANY of that design's re-evaluated
    solutions meets all criteria jointly; the POOLED row ORs across every design,
    answering "is this realization winnable by ANY policy from ANY design?" The
    complement is structurally unwinnable -- Shavazipour et al. (2021) found 23%
    of their test scenarios in that class, and without the screen a design that
    searched badly is indistinguishable from a test realization that is
    impossible.

    This is an EMPIRICAL bound, not a true ceiling: it says no policy *in this
    pooled set* wins the realization, not that none exists.

    Returns:
        Frame: design ("POOLED_ALL_DESIGNS" for the pooled row), n_realizations,
        n_attainable, frac_attainable, plus ``unmet__{objective}`` = the fraction
        of realizations no solution could satisfy on that criterion alone (which
        criterion is binding where nothing attains the joint criterion).
    """
    per_design: dict[str, pd.DataFrame] = {}
    for r in runs:
        path = r.path / "robustness_attainability.csv"
        if not path.exists():
            continue
        a = pd.read_csv(path).set_index("realization_id")
        bool_cols = [c for c in a.columns if c == "attainable" or c.startswith("anysat__")]
        a = a[bool_cols].astype(bool)
        cur = per_design.get(r.design)
        # OR across a design's runs (draws/seeds): a realization is attainable for
        # the design if any of its replicates produced a policy that wins it.
        per_design[r.design] = a if cur is None else (
            cur.reindex(cur.index.union(a.index), fill_value=False)
            | a.reindex(cur.index.union(a.index), fill_value=False))

    if not per_design:
        return pd.DataFrame()

    rows = []
    pooled = None
    for design, a in per_design.items():
        rows.append(_attain_row(design, a))
        pooled = a if pooled is None else (
            pooled.reindex(pooled.index.union(a.index), fill_value=False)
            | a.reindex(pooled.index.union(a.index), fill_value=False))
    rows.append(_attain_row("POOLED_ALL_DESIGNS", pooled))
    return pd.DataFrame(rows)


def _attain_row(design: str, a: pd.DataFrame) -> dict:
    n = len(a)
    n_att = int(a["attainable"].sum())
    row = {"design": design, "n_realizations": n, "n_attainable": n_att,
           "frac_attainable": n_att / n if n else np.nan,
           "frac_unwinnable": 1 - n_att / n if n else np.nan}
    for c in a.columns:
        if c.startswith("anysat__"):
            row[f"unmet__{c[len('anysat__'):]}"] = float((~a[c]).mean()) if n else np.nan
    return row


###############################################################################
# Figures
###############################################################################

def _design_colors(designs: Iterable[str]) -> dict:
    """One color per design, derived from the registry (never a hardcoded list)."""
    designs = list(designs)
    cmap = plt.get_cmap("tab10" if len(designs) <= 10 else "tab20")
    return {d: cmap(i % cmap.N) for i, d in enumerate(designs)}


def _design_labels(designs: Iterable[str]) -> dict:
    """Human-readable design labels from the scenario-design registry."""
    return {d: SCENARIO_DESIGNS[d].name.replace("_", " ") if d in SCENARIO_DESIGNS
            else str(d).replace("_", " ") for d in designs}


def fig_threshold_sweep(sweep: pd.DataFrame, agreement: pd.DataFrame,
                        defaults: pd.DataFrame, fig_dir: Path) -> Path:
    """THE main-text figure: design robustness vs stringency + rank agreement.

    Left: the primary metric per design across the stringency grid (solid =
    best solution, faint dashed = median solution), with the registry-default
    criterion marked. Crossing lines mean the design effect is threshold-
    dependent. Right: Kendall tau_b of the design ranking at each stringency
    against the ranking at the default -- the direct test of threshold
    invariance (Quinn et al. 2020).
    """
    best = sweep_design_level(sweep, "best")
    med = sweep_design_level(sweep, "median")
    designs = list(best.index)
    colors, labels = _design_colors(designs), _design_labels(designs)
    s_default = float(defaults["default_stringency"].mean(skipna=True))
    dref = sweep[sweep["is_default"]].groupby("design")["best"].mean()

    fig, axes = plt.subplots(1, 2, figsize=style.FIGSIZE_WIDE)
    ax = axes[0]
    for d in designs:
        ax.plot(best.columns, best.loc[d], "-o", ms=3.5, lw=1.6,
                color=colors[d], label=labels[d])
        if d in med.index:
            ax.plot(med.columns, med.loc[d], "--", lw=1.0, alpha=0.45,
                    color=colors[d])
    if np.isfinite(s_default):
        ax.axvline(s_default, color="0.35", ls=":", lw=1.2)
        ax.annotate("registry-default\ncriterion", xy=(s_default, 1.0),
                    xytext=(3, -2), textcoords="offset points",
                    va="top", fontsize=8, color="0.35")
        for d in designs:
            if d in dref.index and np.isfinite(dref[d]):
                ax.plot([s_default], [dref[d]], marker="*", ms=11,
                        color=colors[d], mec="black", mew=0.5, zorder=5)
    ax.set_xlabel("Satisficing stringency $s$\n(pooled-$E_{test}$ fraction excluded per criterion)")
    ax.set_ylabel("Multivariate satisficing robustness\n(solid: best solution; dashed: median solution)")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("Design robustness vs criterion stringency")
    ax.legend(frameon=False, fontsize=8, loc="best")

    ax = axes[1]
    for statistic, ls in (("best", "-o"), ("median", "--s")):
        sub = agreement[agreement["statistic"] == statistic].sort_values("stringency")
        ax.plot(sub["stringency"], sub["tau_b_vs_default"], ls, ms=3.5, lw=1.5,
                label=f"{statistic}-solution ranking")
    ax.axhline(1.0, color="0.6", lw=0.8)
    ax.axhline(0.975, color="0.6", ls=":", lw=0.8)
    ax.annotate("stable (Bonham et al. 2024)", xy=(0.02, 0.975),
                xycoords=("axes fraction", "data"), xytext=(0, -9),
                textcoords="offset points", fontsize=7, color="0.4")
    if np.isfinite(s_default):
        ax.axvline(s_default, color="0.35", ls=":", lw=1.2)
    ax.set_xlabel("Satisficing stringency $s$")
    ax.set_ylabel(r"Kendall $\tau_b$ of the DESIGN ranking" "\n" r"vs the ranking at the default criterion")
    ax.set_ylim(-1.05, 1.05)
    ax.set_title("Is the design ranking threshold-invariant?")
    ax.legend(frameon=False, fontsize=8, loc="lower left")

    fig.tight_layout()
    out = fig_dir / "threshold_sweep"
    style.save_figure(fig, out)
    plt.close(fig)
    return out.with_suffix(".png")


#: Display prefix for each robustness-metric family.
_METRIC_FAMILY = {
    "sat_uni": "satisficing", "laplace": "mean", "maximin": "worst-case",
    "vs_baseline": "vs status quo",
}


def metric_label(col: str) -> str:
    """Readable label for a scorecard column (``family: objective``)."""
    if col == PRIMARY_METRIC:
        return "satisficing (multivariate)"
    if "__" not in col:
        return col
    family, name = col.split("__", 1)
    return f"{_METRIC_FAMILY.get(family, family)}: {style.label_for(name)}"


def fig_design_ranking_stability(tau: pd.DataFrame, fig_dir: Path) -> Path:
    """Heatmap of Kendall tau_b among the design rankings induced by each metric."""
    fig, ax = plt.subplots(figsize=(8.0, 7.0))
    im = style.annotated_corr_heatmap(
        ax, tau.to_numpy(dtype=float), list(tau.columns),
        label_fn=metric_label, box_threshold=None, fontsize=7,
    )
    ax.set_title("Agreement among the DESIGN rankings induced by each metric\n"
                 r"(Kendall $\tau_b$; low = metric choice changes the design ranking)",
                 fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=r"$\tau_b$")
    fig.tight_layout()
    out = fig_dir / "design_ranking_stability"
    style.save_figure(fig, out)
    plt.close(fig)
    return out.with_suffix(".png")


def fig_performance_distributions(perf: pd.DataFrame, thresholds: dict,
                                  fig_dir: Path) -> Path:
    """Raw re-evaluated performance per objective, with the threshold drawn on.

    The mandatory co-report: a robustness scalar alone cannot distinguish a good
    policy from a uniformly terrible one (Huang et al. 2025) or a discriminating
    criterion from a saturated one (Bonham et al. 2024). Drawing the satisficing
    threshold on the same axis makes the threshold margin visible (Gold et al.
    2023, Fig. 5).
    """
    objectives = sorted(perf["objective"].unique())
    designs = sorted(perf["design"].unique())
    colors, labels = _design_colors(designs), _design_labels(designs)

    ncol = min(3, max(1, len(objectives)))
    nrow = int(np.ceil(len(objectives) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.6 * ncol, 1.15 * len(designs) * nrow + 1.4),
                             squeeze=False)
    for i, name in enumerate(objectives):
        ax = axes[i // ncol][i % ncol]
        sub = perf[perf["objective"] == name]
        for j, d in enumerate(designs):
            for group, off, alpha in (("best", 0.16, 1.0), ("median", -0.16, 0.42)):
                row = sub[(sub["design"] == d) & (sub["solution_group"] == group)]
                if row.empty:
                    continue
                r = row.iloc[0]
                y = j + off
                ax.plot([r["q05"], r["q95"]], [y, y], lw=1.0, color=colors[d],
                        alpha=alpha, solid_capstyle="butt")
                ax.plot([r["q25"], r["q75"]], [y, y], lw=6.0, color=colors[d],
                        alpha=alpha * 0.65, solid_capstyle="butt")
                ax.plot([r["q50"]], [y], marker="|", ms=9, mew=1.6,
                        color="black", alpha=alpha)
        thr = thresholds.get(name)
        if thr is not None and np.isfinite(thr):
            ax.axvline(thr, color="crimson", ls="--", lw=1.2)
        ax.set_yticks(range(len(designs)))
        # Only the leftmost column carries design names; repeating them on every
        # panel would crowd out the data.
        ax.set_yticklabels([labels[d] for d in designs] if i % ncol == 0
                           else [""] * len(designs), fontsize=8)
        ax.set_ylim(-0.7, len(designs) - 0.3)
        ax.invert_yaxis()
        ax.set_xlabel(style.label_for(name), fontsize=9)
    for k in range(len(objectives), nrow * ncol):
        axes[k // ncol][k % ncol].axis("off")
    fig.suptitle("Re-evaluated performance on $E_{test}$ (median, IQR, 5-95%)\n"
                 "dark = best solution, faint = median solution; dashed red = satisficing threshold",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = fig_dir / "performance_distributions"
    style.save_figure(fig, out)
    plt.close(fig)
    return out.with_suffix(".png")


###############################################################################
# Orchestration
###############################################################################

def run_comparison(formulation: str = "ffmp", reeval_tag: Optional[str] = None,
                   seed: Optional[int] = None,
                   outputs_root: Optional[Path] = None,
                   table_dir: Optional[Path] = None,
                   fig_dir: Optional[Path] = None) -> dict:
    """Assemble every cross-design table and figure for one (formulation, E_test).

    Args:
        formulation: Formulation identifier.
        reeval_tag: The common held-out ensemble's tag. Defaults to the tag of
            ``config.REEVAL_ENSEMBLE_SPEC`` (i.e. ``NYCOPT_REEVAL_ENSEMBLE_PRESET``).
        seed: Restrict to one MOEA seed; None keeps all.
        outputs_root: Root of the run output tree (default ``config.OUTPUTS_DIR``).
        table_dir: Where the CSV tables go (default
            ``outputs/comparison/{slug}/{reeval_tag}``).
        fig_dir: Where the figures go (default
            ``figures/comparison/{slug}/robustness``).

    Returns:
        Mapping of artifact name -> written path.
    """
    tag = reeval_tag or reeval_tag_of(config.REEVAL_ENSEMBLE_SPEC)
    runs = discover_runs(formulation, tag, seed, outputs_root)
    if not runs:
        raise SystemExit(
            f"[compare] no re-eval runs found for formulation='{formulation}', "
            f"reeval tag='{tag}'. Run workflow step 08 (+ `python -m src.robustness`) "
            f"for at least one campaign design first."
        )

    slug = common_slug(runs, formulation)
    if table_dir is None:
        table_dir = config.OUTPUTS_DIR / "comparison" / slug / tag
    if fig_dir is None:
        fig_dir = config.figure_dir_for("comparison", slug, "robustness")
    table_dir = Path(table_dir)
    fig_dir = Path(fig_dir)
    table_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    loaded = load_runs(runs)
    if not loaded:
        raise SystemExit("[compare] runs found but none carries a robustness_scorecard.csv.")
    runs = [lr.run for lr in loaded]
    kinds = loaded[0].kinds
    thresholds = loaded[0].thresholds
    directions = loaded[0].directions
    if any(k is None for k in kinds.values()):
        raise SystemExit(
            "[compare] the re-eval metadata carries no satisficing kind for some "
            "objectives (single-trace re-eval?). The multivariate criterion is "
            "undefined; re-evaluate on an ensemble E_test."
        )

    written: dict[str, Path] = {}
    style.apply_style()

    # --- 2. scorecard aggregation, design ranking stability, variance -------
    summary = design_summary(loaded)
    written["design_summary"] = table_dir / "design_summary.csv"
    summary.to_csv(written["design_summary"], index=False)

    level_best = design_level(summary, "best")
    hb = {}
    for lr in loaded:
        hb.update(lr.higher_better)
    tau = design_ranking_stability(level_best, hb)
    written["design_ranking_stability"] = table_dir / "design_ranking_stability.csv"
    tau.to_csv(written["design_ranking_stability"])
    written["design_metric_matrix"] = table_dir / "design_metric_matrix.csv"
    level_best.to_csv(written["design_metric_matrix"])

    vc = pd.concat([variance_components(summary, PRIMARY_METRIC, "best"),
                    variance_components(summary, PRIMARY_METRIC, "median")],
                   ignore_index=True)
    written["design_variance_components"] = table_dir / "design_variance_components.csv"
    vc.to_csv(written["design_variance_components"], index=False)

    # --- 1. threshold sweep (the deliverable) ------------------------------
    pooled = pooled_cells(runs)
    defaults = default_stringency(pooled, thresholds, kinds)
    written["default_thresholds"] = table_dir / "default_thresholds.csv"
    defaults.to_csv(written["default_thresholds"], index=False)

    sweep = threshold_sweep(runs, pooled, kinds)
    written["design_threshold_sweep"] = table_dir / "design_threshold_sweep.csv"
    sweep.to_csv(written["design_threshold_sweep"], index=False)

    mats = []
    for statistic in ("best", "median"):
        m = sweep_design_level(sweep, statistic).reset_index()
        m.insert(1, "statistic", statistic)
        mats.append(m)
    written["design_stringency_matrix"] = table_dir / "design_stringency_matrix.csv"
    pd.concat(mats, ignore_index=True).to_csv(written["design_stringency_matrix"],
                                              index=False)

    agreement = rank_agreement(sweep)
    written["design_rank_agreement"] = table_dir / "design_rank_agreement.csv"
    agreement.to_csv(written["design_rank_agreement"], index=False)

    # --- 3. raw distributions + degeneracy screen --------------------------
    perf = performance_distributions(loaded)
    written["design_performance_quantiles"] = table_dir / "design_performance_quantiles.csv"
    perf.to_csv(written["design_performance_quantiles"], index=False)

    flags = degeneracy_flags(summary, perf, pooled, kinds, directions)
    written["degeneracy_flags"] = table_dir / "degeneracy_flags.csv"
    flags.to_csv(written["degeneracy_flags"], index=False)

    # --- 4. attainability ---------------------------------------------------
    attain = attainability(runs)
    written["design_attainability"] = table_dir / "design_attainability.csv"
    attain.to_csv(written["design_attainability"], index=False)

    # --- figures ------------------------------------------------------------
    written["fig_threshold_sweep"] = fig_threshold_sweep(sweep, agreement, defaults, fig_dir)
    written["fig_design_ranking_stability"] = fig_design_ranking_stability(tau, fig_dir)
    if not perf.empty:
        written["fig_performance_distributions"] = fig_performance_distributions(
            perf, thresholds, fig_dir)

    _report(runs, sweep, agreement, flags, attain, table_dir, fig_dir)
    return written


def _report(runs, sweep, agreement, flags, attain, table_dir, fig_dir) -> None:
    """Print the headline findings the caller must not miss."""
    designs = sorted({r.design for r in runs})
    print(f"[compare] {len(runs)} run(s) across {len(designs)} design(s): "
          f"{', '.join(designs)}")

    tb = agreement[agreement["statistic"] == "best"]["tau_b_vs_default"]
    if tb.notna().any() and float(tb.min()) < 0.975:
        print(f"[compare] THRESHOLD-DEPENDENT DESIGN RANKING: Kendall tau_b vs the "
              f"default criterion falls to {float(tb.min()):.3f} across the "
              f"stringency grid. The ranking is NOT threshold-invariant and must "
              f"be reported across the sweep, not at one threshold "
              f"(Quinn et al. 2020).")

    if not attain.empty:
        pooled_row = attain[attain["design"] == "POOLED_ALL_DESIGNS"]
        if not pooled_row.empty:
            frac = float(pooled_row["frac_unwinnable"].iloc[0])
            print(f"[compare] ATTAINABILITY: {frac:.1%} of E_test is unwinnable by "
                  f"ANY policy from ANY design (empirical bound; "
                  f"cf. Shavazipour et al. 2021, 23%).")

    if not flags.empty:
        print("[compare] " + "!" * 62)
        print(f"[compare] DEGENERACY FLAGS ({len(flags)}): the robustness scalar may "
              f"be perverse. Do not rank designs on a flagged objective.")
        for _, f in flags.iterrows():
            print(f"[compare]   [{f['flag']}] objective={f['objective']} "
                  f"design={f['design']} value={f['value']:.4g} :: {f['detail']}")
        print("[compare] " + "!" * 62)

    print(f"[compare] tables  -> {table_dir}")
    print(f"[compare] figures -> {fig_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formulation", default="ffmp",
                        help="Formulation identifier (e.g. ffmp, ffmp_8).")
    parser.add_argument("--reeval-tag", default=None,
                        help="E_test tag. Default: the tag of config.REEVAL_ENSEMBLE_SPEC "
                             "(set NYCOPT_REEVAL_ENSEMBLE_PRESET).")
    parser.add_argument("--seed", type=int, default=None,
                        help="Restrict to one MOEA seed (default: all seeds).")
    args = parser.parse_args()
    run_comparison(formulation=args.formulation, reeval_tag=args.reeval_tag,
                   seed=args.seed)


if __name__ == "__main__":
    main()
