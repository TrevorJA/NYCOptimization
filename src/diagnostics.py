"""
diagnostics.py - MOEA runtime diagnostics using MOEAFramework v5.0 CLI.

Convergence is read from the MM Borg coordinator archive, one trace per seed
(Reed, 2026-10-23), never from the island archives individually: islands
exchange solutions through the controller (periodic updates, guidance after
stagnation), so their archives are coupled and a line per island overstates
the number of convergence trajectories. The controller writes no runtime
file (``BORG_Controller_run`` in lib/borg/borgmm.c only returns its archive
at shutdown), so step 07 reconstructs the coordinator archive at every
snapshot from the island runtime files, which all record the same NFE grid.

Workflow (``run_full_diagnostics``):

    1. Coordinator series per seed: the seed's island runtime files ->
       ``runtime_coordinator/seed_SS_{slug}.runtime``, whose entry k is the
       epsilon-box merge of every island's archive at snapshot k
       (``src.runtime_archive.coordinator_series``), annotated with
       ``//NFE=`` (total NFE = per-island NFE x islands) and ``//IslandNFE=``.
       When the seed's controller ``.set`` exists and records its search
       epsilons (``# Epsilons:``, written by ``src.mmborg`` since
       2026-10-23), the merge of the final island archives under those
       epsilons is checked against it box by box: the controller's archive
       is the ground truth for the merge, and the two must agree exactly.
    2. Diagnostics reference set ``sets/{slug}_diagnostics_reference.set``:
       the epsilon-box merge of every coordinator snapshot of every seed. This
       is the best-known front of the design's own runs, including seed 1's
       tail beyond the reporting NFE, and it is distinct from the campaign
       reference set ``{slug}_merged.set`` that
       ``scripts/main/extract_runtime_archive.py --merge --install`` writes
       for the step 08/09 re-evaluation; step 07 never writes that name.
    3. MetricsEvaluator: coordinator series + reference set -> one
       ``metrics/seed_SS_{slug}.metrics`` per seed (hypervolume, generational
       distance, inverted generational distance, spacing, additive
       epsilon-indicator, maximum Pareto-front error per snapshot).
    4. The search-diagnostics figure suite (``src.plotting.search_diagnostics``).

One epsilon-box archive implementation serves every merge in the project
(``src.sensitivity_common.epsilon_nondominated``, validated against Borg's
own archive membership); MOEAFramework's mergers are deliberately not used
(in v5.0 ``ResultFileMerger`` ignores ``--epsilon`` and
``ResultFileSeedMerger`` retains boxes Borg's rule rejects), so MOEAFramework
only scores.

MOEAFramework v5.0 requires a registered problem definition (JAR in lib/)
to parse runtime files correctly (knowing #DVs vs #objectives). The JARs
are built by workflow/00_setup_borg_jars.sh, which derives the counts from
get_n_vars()/get_n_objs() at build time (one JAR per formulation,
registered as "drb_{formulation}" in MOEAFramework-5.0/lib/).

The problem definitions declare 0 constraints even though the Borg search
uses formal constraints (get_n_constrs() == 2): the runtime/.set files
parsed here contain only feasible solutions in vars+objs format — the C
writer (BORG_Archive_append) strips constraint violators and never writes
constraint columns. See the header of 00_setup_borg_jars.sh.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

from config import (
    get_epsilons,
    DIAGNOSTICS_SETTINGS,
    FFMP_VR_N_SWEEP,
    active_scenario_name,
    run_output_dir,
)
from src.runtime_archive import (
    compare_archives,
    coordinator_series,
    epsilon_merge,
    moea_result_header,
    parse_result_file,
    write_result_file,
)

#: Artifact subdirectory of the per-seed coordinator series (kept apart from
#: ``runtime/`` so the ``seed_SS_{slug}_*`` island globs never match it).
COORDINATOR_ARTIFACT = "runtime_coordinator"

# Per-formulation MOEAFramework problem registrations. Each is built as a
# separate JAR in MOEAFramework-5.0/lib/ with the correct (nvars, nobjs).
# All use the active objective set.
#   drb_ffmp     -> 36 DVs
#   drb_ffmp_{N} -> per-N DV count (varies with number of zones).
#                   Built automatically by workflow/00_setup_borg_jars.sh.
# Slugs can be any string; we infer the formulation family by finding the
# longest contiguous token-substring that matches a known formulation. This
# handles both prefix-tag slugs (``smoke_ffmp``) and suffix-tag variant slugs
# (``ffmp_8_extended``).
_FORMULATION_TO_PROBLEM = {
    "ffmp": "drb_ffmp",
}
for _n in FFMP_VR_N_SWEEP:
    _FORMULATION_TO_PROBLEM[f"ffmp_{_n}"] = f"drb_ffmp_{_n}"


def problem_name_for(slug: str) -> str:
    """Return the MOEAFramework problem name matching a slug's formulation family.

    Strategy: split the slug on ``_`` and search every contiguous token-window
    (longest first) for a match in ``_FORMULATION_TO_PROBLEM``. Handles:
      - plain slugs: ``ffmp`` -> drb_ffmp
      - prefix-tag (smoke/version): ``smoke_ffmp`` -> drb_ffmp
      - suffix-tag (variant): ``ffmp_weekly_only`` -> drb_ffmp
      - multi-token formulation names: ``ffmp_8`` -> drb_ffmp_8
        (matched in preference to the shorter ``ffmp`` token within the same slug)
    """
    tokens = slug.split("_")
    n = len(tokens)
    # Try every window from longest to shortest so multi-token names like
    # ``ffmp_8`` win over the bare ``ffmp`` substring.
    for window_len in range(n, 0, -1):
        for start in range(0, n - window_len + 1):
            candidate = "_".join(tokens[start : start + window_len])
            if candidate in _FORMULATION_TO_PROBLEM:
                return _FORMULATION_TO_PROBLEM[candidate]
    raise ValueError(
        f"Cannot resolve MOEAFramework problem name for slug '{slug}'. "
        f"Expected the slug to contain one of: "
        f"{sorted(_FORMULATION_TO_PROBLEM)}."
    )


def formulation_for(slug: str) -> str:
    """The formulation name behind a slug (``problem_name_for`` without ``drb_``)."""
    return problem_name_for(slug).removeprefix("drb_")


def get_cli_path() -> str:
    """Return path to MOEAFramework CLI executable."""
    return DIAGNOSTICS_SETTINGS["moea_framework_jar"]


def _epsilon_arg() -> str:
    return ",".join(str(e) for e in get_epsilons())


# ---------------------------------------------------------------------------
# Paths and discovery
# ---------------------------------------------------------------------------

def island_runtime_files(slug: str, seed: int, runtime_dir: Path = None,
                         scenario: str = None) -> list[Path]:
    """The seed's island runtime files, ordered by island index."""
    if scenario is None:
        scenario = active_scenario_name()
    if runtime_dir is None:
        runtime_dir = run_output_dir(scenario, slug, "runtime")
    pattern = re.compile(rf"^seed_{seed:02d}_{re.escape(slug)}_(\d+)\.runtime$")
    found = []
    for f in Path(runtime_dir).iterdir():
        m = pattern.match(f.name)
        if m:
            found.append((int(m.group(1)), f))
    return [f for _, f in sorted(found)]


def coordinator_series_path(slug: str, seed: int, scenario: str = None) -> Path:
    """``outputs/{scenario}/{slug}/runtime_coordinator/seed_SS_{slug}.runtime``."""
    if scenario is None:
        scenario = active_scenario_name()
    return run_output_dir(scenario, slug, COORDINATOR_ARTIFACT) / f"seed_{seed:02d}_{slug}.runtime"


def coordinator_series_files(slug: str, scenario: str = None) -> list[Path]:
    """Every seed's coordinator series under the run, sorted by seed."""
    if scenario is None:
        scenario = active_scenario_name()
    d = run_output_dir(scenario, slug, COORDINATOR_ARTIFACT)
    pattern = re.compile(rf"^seed_(\d+)_{re.escape(slug)}\.runtime$")
    return [f for f in sorted(d.iterdir()) if pattern.match(f.name)]


def diagnostics_reference_set_path(slug: str, scenario: str = None) -> Path:
    """``sets/{slug}_diagnostics_reference.set``, the step-07 reference set.

    Never the campaign reference set ``{slug}_merged.set`` (written only by
    ``extract_runtime_archive.py --merge --install`` and consumed by step
    08/09); the two names are asserted distinct.
    """
    if scenario is None:
        scenario = active_scenario_name()
    sets_dir = run_output_dir(scenario, slug, "sets")
    out = sets_dir / f"{slug}_diagnostics_reference.set"
    assert out.name != f"{slug}_merged.set"
    return out


def discover_seeds(slug: str, runtime_dir: Path = None, scenario: str = None) -> list[int]:
    """Return sorted list of seed numbers found in the slug's runtime directory.

    Looks for island files matching ``seed_NN_{slug}_{i}.runtime`` and
    extracts NN.
    """
    if scenario is None:
        scenario = active_scenario_name()
    if runtime_dir is None:
        runtime_dir = run_output_dir(scenario, slug, "runtime")

    pattern = re.compile(rf"^seed_(\d+)_{re.escape(slug)}_\d+\.runtime$")
    seeds = set()
    if runtime_dir.exists():
        for f in runtime_dir.iterdir():
            m = pattern.match(f.name)
            if m:
                seeds.add(int(m.group(1)))
    return sorted(seeds)


# ---------------------------------------------------------------------------
# Step 1: coordinator series per seed
# ---------------------------------------------------------------------------

def _problem_header(slug: str) -> list[str]:
    from src.formulations import get_n_objs, get_n_vars
    return moea_result_header(problem_name_for(slug),
                              get_n_vars(formulation_for(slug)), get_n_objs())


def check_series_against_set(island_files: list[Path], set_file: Path,
                             n_vars: int) -> dict | None:
    """Check the merge of the final island archives against the controller.

    The controller archive at shutdown (``set_file``) is the ground truth
    for the merge of the island archives at the final snapshot, but only
    under the epsilon vector the search ran with, which the ``.set`` header
    records (``# Epsilons:``). The final island snapshots are merged under
    that vector and must occupy exactly the controller's boxes; box
    representatives may differ only on distance ties, which is reported.

    Returns:
        ``compare_archives`` result, or None when the ``.set`` carries no
        epsilon header (an archive written before 2026-10-23).

    Raises:
        RuntimeError: On any box-set difference.
    """
    from src.load.reference_set import load_set_file
    from src.runtime_archive import island_nfe_grid, read_set_epsilons, snapshot_at

    eps_run = read_set_epsilons(set_file)
    if eps_run is None:
        return None
    final_nfe = island_nfe_grid(island_files)[-1]
    merged = epsilon_merge(snapshot_at(island_files, final_nfe), n_vars, eps_run)
    set_rows = load_set_file(set_file)
    cmp = compare_archives(merged, set_rows, n_vars, eps_run)
    if cmp["boxes_only_a"] or cmp["boxes_only_b"]:
        raise RuntimeError(
            f"merge of the final island archives ({cmp['n_a']} rows) and the "
            f"controller archive {Path(set_file).name} ({cmp['n_b']} rows) occupy "
            f"different epsilon boxes under the search vector {eps_run}: "
            f"{len(cmp['boxes_only_a'])} only in the merge, "
            f"{len(cmp['boxes_only_b'])} only in the controller archive; "
            + _box_mismatch_detail(cmp, merged, set_rows, n_vars, eps_run))
    return cmp


def _box_mismatch_detail(cmp: dict, rows_a, rows_b, n_vars: int, epsilons) -> str:
    """Name the mismatched boxes and, for each, the box that dominates it."""
    import numpy as np
    from src.runtime_archive import epsilon_boxes, round_to_set_precision

    boxes_a = epsilon_boxes(round_to_set_precision(np.asarray(rows_a, dtype=float)),
                            n_vars, epsilons)
    boxes_b = epsilon_boxes(round_to_set_precision(np.asarray(rows_b, dtype=float)),
                            n_vars, epsilons)
    parts = []
    for label, extra, other in (("series", cmp["boxes_only_a"], boxes_b),
                                ("controller", cmp["boxes_only_b"], boxes_a)):
        for box in extra[:5]:
            b = np.asarray(box)
            dominators = [o for o in other
                          if (np.asarray(o) <= b).all() and (np.asarray(o) < b).any()]
            parts.append(f"only-in-{label} box {box}: dominated by "
                         f"{dominators[:3] if dominators else 'no box of the other archive'}")
    return "; ".join(parts)


def build_coordinator_series(slug: str, seed: int, runtime_dir: Path = None,
                             scenario: str = None,
                             output_file: Path = None,
                             set_file: Path = None) -> Path:
    """Reconstruct one seed's coordinator archive at every runtime snapshot.

    Entry k of the written series is the epsilon-box merge, under the
    campaign epsilons, of every island's archive at the k-th snapshot
    (``src.runtime_archive.coordinator_series``); each entry carries
    ``//NFE=`` (total: per-island NFE x islands), ``//IslandNFE=`` and
    ``//Islands=``. When the seed's controller ``.set`` exists and records
    its search epsilons, the merge of the final island archives under those
    epsilons is checked against it (``check_series_against_set``).

    Args:
        slug: Run moea slug.
        seed: Seed number.
        runtime_dir: Directory holding the island ``.runtime`` files.
        scenario: Scenario-design name; defaults to the active one.
        output_file: Override of the series path (tests).
        set_file: Override of the controller ``.set`` path (tests); defaults
            to ``sets/seed_SS_{slug}.set`` and is skipped when absent (a
            seed still running).

    Returns:
        Path of the coordinator series.
    """
    from src.formulations import get_n_vars

    if scenario is None:
        scenario = active_scenario_name()
    files = island_runtime_files(slug, seed, runtime_dir=runtime_dir, scenario=scenario)
    if not files:
        raise FileNotFoundError(f"no island runtime files for seed {seed} of {scenario}/{slug}")
    n_vars = get_n_vars(formulation_for(slug))
    epsilons = get_epsilons()
    series = coordinator_series(files, n_vars, epsilons)
    n_islands = len(files)
    out = (Path(output_file) if output_file is not None
           else coordinator_series_path(slug, seed, scenario=scenario))
    write_result_file(out, [
        ({"NFE": nfe * n_islands, "IslandNFE": nfe, "Islands": n_islands}, rows)
        for nfe, rows in series
    ], header_lines=_problem_header(slug))
    print(f"  Coordinator series: {n_islands} islands x {len(series)} snapshots "
          f"({series[-1][0] * n_islands} total NFE, {len(series[-1][1])} rows at the end) "
          f"-> {out}")

    if set_file is None:
        set_file = run_output_dir(scenario, slug, "sets") / f"seed_{seed:02d}_{slug}.set"
    if not Path(set_file).exists():
        print(f"  {Path(set_file).name} absent (seed still running): merge unchecked")
    else:
        cmp = check_series_against_set(files, Path(set_file), n_vars)
        if cmp is None:
            print(f"  {Path(set_file).name} carries no '# Epsilons:' header (archive "
                  f"written before 2026-10-23): merge unchecked")
        else:
            print(f"  verified against {Path(set_file).name} under its search "
                  f"epsilons: boxes identical, {cmp['representatives_differ']} "
                  f"tie-broken representatives differ")
    return out


# ---------------------------------------------------------------------------
# Step 2: diagnostics reference set
# ---------------------------------------------------------------------------

def merge_reference_set(slug: str, scenario: str = None,
                        series_files: list[Path] = None,
                        output_file: Path = None) -> Path:
    """Merge every coordinator snapshot of every seed into one reference set.

    The epsilon-box merge of the union of all entries of all series, written
    in the MOEAFramework result-file layout for ``MetricsEvaluator``.

    Returns:
        Path to the reference set.
    """
    import numpy as np
    from src.formulations import get_n_vars

    if scenario is None:
        scenario = active_scenario_name()
    if series_files is None:
        series_files = coordinator_series_files(slug, scenario=scenario)
    if not series_files:
        raise FileNotFoundError(f"no coordinator series under {scenario}/{slug}")
    if output_file is None:
        output_file = diagnostics_reference_set_path(slug, scenario=scenario)

    blocks = [rows for f in series_files
              for _, rows in parse_result_file(Path(f).read_text()) if rows.size]
    union = np.vstack(blocks) if blocks else np.empty((0, 0))
    front = epsilon_merge(union, get_n_vars(formulation_for(slug)), get_epsilons())
    write_result_file(output_file, [({}, front)], header_lines=_problem_header(slug))
    print(f"  Reference set: {len(series_files)} series, {len(union)} rows unioned, "
          f"{len(front)} retained under the campaign epsilons -> {output_file.name}")
    return output_file


# ---------------------------------------------------------------------------
# Step 3: metrics per seed
# ---------------------------------------------------------------------------

def compute_metrics(series_file: Path, reference_file: Path, slug: str,
                    output_file: Path = None) -> Path:
    """Score one coordinator series against the reference set.

    Uses MetricsEvaluator to compute hypervolume, generational distance,
    inverted generational distance, spacing, additive epsilon-indicator, and
    maximum Pareto-front error at each snapshot.

    Args:
        series_file: Input coordinator series (``.runtime``).
        reference_file: Reference set (.set or .ref file).
        slug: Run slug (picks the MOEAFramework problem registration).
        output_file: Output .metrics file; defaults to
            ``metrics/{series stem}.metrics`` beside the series directory.

    Returns:
        Path to the .metrics file.
    """
    series_file = Path(series_file)
    if output_file is None:
        metrics_dir = series_file.parent.parent / "metrics"
        metrics_dir.mkdir(parents=True, exist_ok=True)
        output_file = metrics_dir / f"{series_file.stem}.metrics"

    cmd = [
        get_cli_path(), "MetricsEvaluator",
        "--problem", problem_name_for(slug),
        "--epsilon", _epsilon_arg(),
        "--input", str(series_file),
        "--reference", str(reference_file),
        "--output", str(output_file),
        "--force", "--overwrite",
    ]
    print(f"  Computing metrics: {series_file.name} -> {output_file.name}")
    subprocess.run(cmd, check=True)
    return output_file


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def _notice_retired_artifacts(slug: str, scenario: str) -> None:
    """Point out leftovers of the per-island workflow; nothing is deleted."""
    run_dir = run_output_dir(scenario, slug, "metrics").parent
    legacy = sorted((run_dir / "metrics").glob(f"seed_*_{slug}_[0-9]*.metrics"))
    legacy += sorted((run_dir / "sets").glob(f"{slug}_seed*_merged.set"))
    if legacy:
        print(f"[{scenario}/{slug}] {len(legacy)} per-island artifact(s) from the "
              f"retired workflow are present and ignored (e.g. {legacy[0].name}).")


def run_full_diagnostics(
    slug: str, seeds: list = None, runtime_dir: Path = None, scenario: str = None,
):
    """Run the coordinator-archive diagnostics for every seed of one run.

    Args:
        slug: Run moea slug (inner output subdirectory + filename prefix).
        seeds: Optional explicit list of seed numbers. If None, auto-discovers
            every seed found in ``outputs/{scenario}/{slug}/runtime``.
        runtime_dir: Optional explicit runtime directory override.
        scenario: Scenario-design name (top-level output partition). Defaults
            to the active scenario design.

    Returns:
        Dict of {seed: metrics_path}; empty when no runtime files exist.
    """
    if scenario is None:
        scenario = active_scenario_name()
    if seeds is None:
        seeds = discover_seeds(slug, runtime_dir=runtime_dir, scenario=scenario)
    if not seeds:
        print(f"[{scenario}/{slug}] no runtime files found; skipping.")
        return {}

    print(f"\n=== MOEAFramework Diagnostics: {scenario}/{slug}, seeds {seeds} ===\n")
    _notice_retired_artifacts(slug, scenario)

    series = {}
    for seed in seeds:
        print(f"[{scenario}/{slug}] seed {seed}: coordinator series")
        series[seed] = build_coordinator_series(
            slug, seed, runtime_dir=runtime_dir, scenario=scenario)

    print(f"[{scenario}/{slug}] diagnostics reference set")
    reference = merge_reference_set(
        slug, scenario=scenario, series_files=[series[s] for s in seeds])

    results = {}
    for seed in seeds:
        results[seed] = compute_metrics(series[seed], reference, slug=slug)

    # Standard post-search figure suite; a figure failure never fails the
    # diagnostics.
    try:
        from src.plotting.search_diagnostics import render_search_diagnostics
        print(f"[{scenario}/{slug}] rendering search-diagnostics figures...")
        render_search_diagnostics(slug, scenario=scenario)
    except Exception as e:
        print(f"[{scenario}/{slug}] figure suite FAILED "
              f"(metrics/sets unaffected): {e}")
    print(f"\n=== Complete: reference {reference.name}; metrics "
          f"{[p.name for p in results.values()]} ===")
    return results
