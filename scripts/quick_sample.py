"""
scripts/quick_sample.py - LHS sampling pipeline for any formulation or policy architecture.

For each architecture, evaluates N random LHS parameter sets, finds the approximate
Pareto front, saves .set files, and generates parallel coordinates + pairwise scatter
figures.  All figures include the FFMP default-parameter baseline as a reference line.

Usage (from NYCOptimization/ root):
    python scripts/quick_sample.py [--arch ARCH] [--n N] [--seed SEED]
    python scripts/quick_sample.py --all [--n N] [--seed SEED]

Examples:
    python scripts/quick_sample.py --arch ffmp --n 200
    python scripts/quick_sample.py --arch rbf  --n 50
    python scripts/quick_sample.py --all        --n 50
"""

import sys
import argparse
import csv
import time
import numpy as np
from pathlib import Path

# Force UTF-8 stdout/stderr (avoids cp1252 failures on Windows)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    get_n_objs, get_bounds, get_var_names, get_obj_names,
    get_obj_directions, get_baseline_values, make_objective_function,
    is_external_policy, OUTPUTS_DIR,
)


# ---------------------------------------------------------------------------
# Display metadata
# ---------------------------------------------------------------------------

#: Human-readable name for each architecture, used in figure titles.
ARCH_LABELS = {
    "ffmp":  "Parameterized FFMP",
    "rbf":   "RBF Policy (6 centers, 15 inputs)",
    "tree":  "Oblique Tree (depth 3, 15 inputs)",
    "ann":   "ANN (2×8 hidden, 15 inputs)",
}

#: Short per-objective labels for scatter-plot axes (ordered to match DEFAULT_OBJECTIVES).
OBJ_SHORT = [
    "NYC Rel.",
    "NYC Vuln. %",
    "NJ Rel.",
    "Montague\nRel.",
    "Trenton\nRel.",
    "Flood Days",
    "Min Stor. %",
]

#: Parallel-coordinates axis labels (multi-line, direction hint on third line).
OBJ_AXIS_LABELS = {
    "nyc_reliability_weekly":          "NYC\nReliability\n(max)",
    "nyc_vulnerability":               "NYC\nVuln. %\n(min)",
    "nj_reliability_weekly":           "NJ\nReliability\n(max)",
    "montague_reliability_weekly":     "Montague\nReliability\n(max)",
    "trenton_reliability_weekly":      "Trenton\nReliability\n(max)",
    "flood_risk_downstream_flow_days": "Flood\nDays\n(min)",
    "storage_min_combined_pct":        "Min\nStorage\n(max)",
}

#: Six pairwise scatter pairs (0-based objective indices).
SCATTER_PAIRS = [
    (0, 3),   # NYC Rel. vs Montague
    (0, 5),   # NYC Rel. vs Flood Days
    (3, 5),   # Montague vs Flood Days
    (0, 6),   # NYC Rel. vs Min Storage
    (3, 6),   # Montague vs Min Storage
    (2, 5),   # NJ Rel. vs Flood Days
]


# ---------------------------------------------------------------------------
# Math utilities
# ---------------------------------------------------------------------------

def lhs_sample(n_samples: int, lower: np.ndarray, upper: np.ndarray,
               rng: np.random.Generator) -> np.ndarray:
    """Latin Hypercube Sample in [lower, upper], shape (n_samples, n_vars)."""
    n_vars = len(lower)
    cut = np.linspace(0, 1, n_samples + 1)
    u = rng.uniform(size=(n_samples, n_vars))
    pts = np.zeros((n_samples, n_vars))
    for j in range(n_vars):
        perm = rng.permutation(n_samples)
        pts[:, j] = cut[perm] + u[:, j] * (cut[1] - cut[0])
    return lower + pts * (upper - lower)


def pareto_filter(obj_matrix: np.ndarray) -> np.ndarray:
    """Return indices of non-dominated rows (pure minimization)."""
    n = obj_matrix.shape[0]
    dominated = np.zeros(n, dtype=bool)
    for i in range(n):
        if dominated[i]:
            continue
        for j in range(n):
            if i == j or dominated[j]:
                continue
            if np.all(obj_matrix[j] <= obj_matrix[i]) and np.any(obj_matrix[j] < obj_matrix[i]):
                dominated[i] = True
                break
    return np.where(~dominated)[0]


def unegate(arr: np.ndarray, directions: list) -> np.ndarray:
    """Un-negate Borg-minimized objectives back to raw values."""
    out = arr.copy()
    for i, d in enumerate(directions):
        if d == 1:
            if out.ndim == 1:
                out[i] = -out[i]
            else:
                out[:, i] = -out[:, i]
    return out


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_ffmp_baseline() -> np.ndarray | None:
    """Load FFMP default-parameter baseline objectives from saved CSV.

    Returns raw (un-negated) objective values as a 1-D array, or None if
    the baseline CSV does not exist yet.
    """
    baseline_csv = OUTPUTS_DIR / "baseline" / "ffmp_baseline_objectives.csv"
    if not baseline_csv.exists():
        return None
    with open(baseline_csv, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        values = next(reader)
    obj_names = get_obj_names()
    # Re-order columns to match current objective set (CSV may have different order)
    name_to_val = dict(zip(header, [float(v) for v in values]))
    raw = np.array([name_to_val[n] for n in obj_names if n in name_to_val])
    if len(raw) != len(obj_names):
        return None  # Column mismatch — caller will skip baseline overlay
    return raw


def save_ffmp_baseline(obj_raw: np.ndarray) -> None:
    """Save raw FFMP baseline objectives to CSV."""
    baseline_csv = OUTPUTS_DIR / "baseline" / "ffmp_baseline_objectives.csv"
    baseline_csv.parent.mkdir(parents=True, exist_ok=True)
    obj_names = get_obj_names()
    with open(baseline_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(obj_names)
        writer.writerow([f"{v:.10f}" for v in obj_raw])
    print(f"  Saved FFMP baseline -> {baseline_csv}")


def write_set_file(filepath: Path, dv_matrix: np.ndarray, obj_matrix: np.ndarray,
                   arch: str, note: str = "") -> None:
    """Write a Borg-format .set file (vars then objs, whitespace-delimited)."""
    var_names = get_var_names(arch)
    obj_names = get_obj_names()
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(f"# Architecture: {arch}\n")
        if note:
            f.write(f"# Note: {note}\n")
        f.write(f"# Variables ({dv_matrix.shape[1]}): {','.join(var_names)}\n")
        f.write(f"# Objectives ({obj_matrix.shape[1]}): {','.join(obj_names)}\n")
        for dvs, objs in zip(dv_matrix, obj_matrix):
            row = " ".join(f"{v:.6e}" for v in dvs) + " " + " ".join(f"{v:.6e}" for v in objs)
            f.write(row + "\n")
    print(f"  Saved {dv_matrix.shape[0]} solutions -> {filepath}")


# ---------------------------------------------------------------------------
# Sampling loop
# ---------------------------------------------------------------------------

def run_lhs(arch: str, n_samples: int, seed: int) -> tuple[np.ndarray, np.ndarray, int]:
    """Draw n_samples LHS points, evaluate, return (dv_matrix, obj_borg, n_penalty).

    obj_borg is Borg-minimized (maximization objectives are negated).
    """
    lower, upper = get_bounds(arch)
    n_objs = get_n_objs()
    obj_fn = make_objective_function(arch)
    rng = np.random.default_rng(seed)

    samples = lhs_sample(n_samples, lower, upper, rng)
    penalty_threshold = 1e5

    all_dvs, all_objs = [], []
    n_penalty = 0
    t_start = time.time()

    for k, dv in enumerate(samples):
        objs = list(obj_fn(dv))
        if any(o >= penalty_threshold for o in objs):
            n_penalty += 1
        else:
            all_dvs.append(dv)
            all_objs.append(objs)

        if (k + 1) % 10 == 0:
            elapsed = time.time() - t_start
            rate = (k + 1) / elapsed
            eta = (n_samples - k - 1) / rate
            print(f"  [{k+1:>4}/{n_samples}]  valid={len(all_dvs)}  "
                  f"penalty={n_penalty}  {rate:.2f}/s  ETA={eta:.0f}s")

    t_total = time.time() - t_start
    print(f"\n  Done: {len(all_dvs)} valid, {n_penalty} penalized, "
          f"{t_total:.1f}s ({t_total / n_samples:.2f}s/eval)")

    if not all_dvs:
        raise RuntimeError("All evaluations returned penalty — check simulation setup.")

    return np.array(all_dvs), np.array(all_objs), n_penalty


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _fmt_val(v: float) -> str:
    """Format a raw objective value for axis annotation."""
    if abs(v) >= 100:
        return f"{v:.0f}"
    elif abs(v) >= 1:
        return f"{v:.3f}"
    else:
        return f"{v:.4f}"


def plot_parallel_coordinates(
    obj_pareto_raw: np.ndarray,
    baseline_raw: np.ndarray | None,
    arch: str,
    n_samples: int,
    n_pareto: int,
    out_path: Path,
) -> None:
    """Parallel coordinates of Pareto-approximate objective set.

    All axes oriented so up = preferred.  FFMP baseline drawn in red if provided.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    obj_names = get_obj_names()
    directions = get_obj_directions()
    n_objs = len(obj_names)
    arch_label = ARCH_LABELS.get(arch, arch)

    # Normalization range (include baseline if present)
    all_data = (np.vstack([obj_pareto_raw, baseline_raw.reshape(1, -1)])
                if baseline_raw is not None else obj_pareto_raw)
    col_min = all_data.min(axis=0)
    col_max = all_data.max(axis=0)
    col_range = np.where(col_max > col_min, col_max - col_min, 1.0)

    normed = (obj_pareto_raw - col_min) / col_range

    # Flip minimize objectives: low raw value -> top of axis
    for i, d in enumerate(directions):
        if d == -1:
            normed[:, i] = 1.0 - normed[:, i]

    fig, ax = plt.subplots(figsize=(13, 5))
    x = np.arange(n_objs)

    for row in normed:
        ax.plot(x, row, alpha=0.2, color="steelblue", linewidth=0.8)

    # FFMP baseline overlay
    if baseline_raw is not None:
        baseline_normed = (baseline_raw - col_min) / col_range
        for i, d in enumerate(directions):
            if d == -1:
                baseline_normed[i] = 1.0 - baseline_normed[i]
        ax.plot(x, baseline_normed, color="firebrick", linewidth=2.5,
                marker="o", markersize=5, label="FFMP default params", zorder=10)
        ax.legend(loc="lower right", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(
        [OBJ_AXIS_LABELS.get(n, n) for n in obj_names],
        fontsize=9, ha="center",
    )

    # Annotate best/worst raw values at top/bottom of each axis
    for i, d in enumerate(directions):
        top_val = col_max[i] if d == 1 else col_min[i]
        bot_val = col_min[i] if d == 1 else col_max[i]
        ax.text(i, 1.05, _fmt_val(top_val), ha="center", va="bottom", fontsize=7, color="0.4")
        ax.text(i, -0.05, _fmt_val(bot_val), ha="center", va="top",   fontsize=7, color="0.4")

    ax.set_ylabel("Normalized preference  (up = better for all axes)", fontsize=9)
    ax.set_title(
        f"{arch_label}  |  LHS {n_samples} samples  |  {n_pareto} non-dominated solutions",
        fontsize=11,
    )
    ax.set_ylim(-0.15, 1.15)
    ax.axhline(0, color="0.85", linewidth=0.6)
    ax.axhline(1, color="0.85", linewidth=0.6)
    ax.grid(True, alpha=0.25, axis="x")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_objective_scatter(
    obj_all_raw: np.ndarray,
    obj_pareto_raw: np.ndarray,
    baseline_raw: np.ndarray | None,
    arch: str,
    out_path: Path,
) -> None:
    """2×3 grid of pairwise objective scatter plots."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_objs = obj_all_raw.shape[1]
    arch_label = ARCH_LABELS.get(arch, arch)
    pairs = [(i, j) for (i, j) in SCATTER_PAIRS if i < n_objs and j < n_objs]
    short = OBJ_SHORT[:n_objs]

    ncols = 3
    nrows = (len(pairs) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.5 * nrows))
    axes = np.array(axes).flatten()

    for ax_idx, (i, j) in enumerate(pairs):
        ax = axes[ax_idx]
        ax.scatter(obj_all_raw[:, i], obj_all_raw[:, j],
                   alpha=0.18, s=14, color="lightsteelblue", label="All valid")
        ax.scatter(obj_pareto_raw[:, i], obj_pareto_raw[:, j],
                   alpha=0.75, s=28, color="steelblue", label="Pareto approx.")
        if baseline_raw is not None:
            ax.scatter([baseline_raw[i]], [baseline_raw[j]],
                       color="firebrick", s=100, zorder=10, marker="*",
                       label="FFMP default")
        ax.set_xlabel(short[i], fontsize=9)
        ax.set_ylabel(short[j], fontsize=9)
        if ax_idx == 0:
            ax.legend(fontsize=7, loc="best")

    for k in range(len(pairs), len(axes)):
        axes[k].set_visible(False)

    fig.suptitle(f"Pairwise Objective Scatter  |  {arch_label}", fontsize=11, y=1.01)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def run(arch: str, n_samples: int, seed: int) -> None:
    """Run LHS sampling for one architecture, save results, generate figures."""
    directions = get_obj_directions()

    sets_dir = OUTPUTS_DIR / "optimization" / arch / "sets"
    figs_dir = OUTPUTS_DIR / "figures"

    print(f"\n{'='*60}")
    print(f"Architecture: {ARCH_LABELS.get(arch, arch)}  (--arch {arch})")
    print(f"Samples: {n_samples}  |  Seed: {seed}")
    print(f"{'='*60}")

    # ------------------------------------------------------------------
    # 1. Baseline: evaluate FFMP default params (FFMP only) or load from CSV
    # ------------------------------------------------------------------
    baseline_raw = None

    if not is_external_policy(arch):
        print("\n--- Evaluating FFMP default parameters ---")
        obj_fn = make_objective_function(arch)
        t0 = time.time()
        baseline_borg = np.array(obj_fn(get_baseline_values(arch)))
        print(f"  Time: {time.time() - t0:.1f}s")
        baseline_raw = unegate(baseline_borg, directions)
        for name, val in zip(get_obj_names(), baseline_raw):
            print(f"  {name}: {val:.4f}")
        save_ffmp_baseline(baseline_raw)
    else:
        baseline_raw = load_ffmp_baseline()
        if baseline_raw is not None:
            print("\n--- FFMP baseline loaded from CSV (shown as reference) ---")
        else:
            print("\n--- No FFMP baseline found; run --arch ffmp first for reference line ---")

    # ------------------------------------------------------------------
    # 2. LHS sampling
    # ------------------------------------------------------------------
    print(f"\n--- LHS sampling: {n_samples} evaluations ---")
    dv_matrix, obj_borg, n_penalty = run_lhs(arch, n_samples, seed)
    obj_raw = unegate(obj_borg, directions)

    # Save full sample immediately (before any potential post-processing crash)
    all_file = sets_dir / f"lhs_all_{n_samples}_{arch}.set"
    write_set_file(all_file, dv_matrix, obj_borg, arch,
                   note=f"LHS all valid samples  seed={seed}")

    # ------------------------------------------------------------------
    # 3. Pareto filter
    # ------------------------------------------------------------------
    print("\n--- Pareto filtering ---")
    pareto_idx = pareto_filter(obj_borg)
    n_pareto = len(pareto_idx)
    print(f"  {n_pareto} non-dominated from {len(dv_matrix)} valid")

    dv_pareto    = dv_matrix[pareto_idx]
    obj_pareto_borg = obj_borg[pareto_idx]
    obj_pareto_raw  = obj_raw[pareto_idx]

    dir_sym = {1: "(max)", -1: "(min)"}
    for i, name in enumerate(get_obj_names()):
        lo, hi = obj_pareto_raw[:, i].min(), obj_pareto_raw[:, i].max()
        print(f"  {name:42s} [{lo:.4f}, {hi:.4f}]  {dir_sym[directions[i]]}")

    pareto_file = sets_dir / f"lhs_pareto_{n_samples}_{arch}.set"
    write_set_file(pareto_file, dv_pareto, obj_pareto_borg, arch,
                   note=f"LHS Pareto approx  {len(dv_matrix)} valid samples  seed={seed}")

    # ------------------------------------------------------------------
    # 4. Figures
    # ------------------------------------------------------------------
    print("\n--- Generating figures ---")
    plot_parallel_coordinates(
        obj_pareto_raw, baseline_raw, arch, n_samples, n_pareto,
        figs_dir / f"parallel_coords_{arch}_lhs{n_samples}.png",
    )
    plot_objective_scatter(
        obj_raw, obj_pareto_raw, baseline_raw, arch,
        figs_dir / f"scatter_pairs_{arch}_lhs{n_samples}.png",
    )

    print(f"\nDone  ({arch}, n={n_samples}, seed={seed})")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

ALL_ARCHITECTURES = ["ffmp", "rbf", "tree", "ann"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="LHS sampling for any formulation or policy architecture.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="\n".join([
            "Examples:",
            "  python scripts/quick_sample.py --arch ffmp --n 200",
            "  python scripts/quick_sample.py --arch rbf  --n 50",
            "  python scripts/quick_sample.py --all       --n 50",
        ]),
    )
    parser.add_argument("--arch", type=str, default="ffmp",
                        choices=ALL_ARCHITECTURES,
                        help="Architecture/formulation name (default: ffmp)")
    parser.add_argument("--all", dest="run_all", action="store_true",
                        help="Run all architectures sequentially")
    parser.add_argument("--n", type=int, default=50,
                        help="LHS samples per architecture (default: 50)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    args = parser.parse_args()

    archs = ALL_ARCHITECTURES if args.run_all else [args.arch]
    for arch in archs:
        run(arch, args.n, args.seed)
