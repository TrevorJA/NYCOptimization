"""
diagnostics.py - MOEA runtime diagnostics using MOEAFramework v5.0 CLI.

Three-step workflow for computing runtime metrics from Borg output:

    1. ResultFileMerger:     runtime files -> merged reference set (.set)
    2. MetricsEvaluator:     runtime files + .set -> per-snapshot metrics

MOEAFramework v5.0 requires a registered problem definition (JAR in lib/)
to parse runtime files correctly (knowing #DVs vs #objectives). The problem
was created via:

    ./MOEAFramework-5.0/cli BuildProblem \\
        --problemName drb_ffmp \\
        --language python \\
        --numberOfVariables 24 \\
        --numberOfObjectives 6 \\
        --lowerBound -1000000.0 \\
        --upperBound 1000000.0 \\
        --directory MOEAFramework-5.0/native

    cd MOEAFramework-5.0/native/drb_ffmp
    mkdir -p bin && cp -r META-INF bin
    javac -classpath "../../lib/*:." -d bin \\
        src/drb_ffmp/drb_ffmp.java src/drb_ffmp/drb_ffmpProvider.java
    cp drb_ffmp.py bin/drb_ffmp
    jar -cf drb_ffmp.jar -C bin META-INF/ -C bin drb_ffmp
    cp drb_ffmp.jar ../../lib/

The resulting JAR (drb_ffmp.jar) is in MOEAFramework-5.0/lib/ and registers
the problem name "drb_ffmp" with 24 real-valued DVs and 6 objectives.

References:
    - WaterProgramming: "MM Borg MOEA Python Wrapper - Checkpointing,
      Runtime and Operator Dynamics using MOEA Framework-5.0" (Aug 2025)
    - WaterProgramming: "Performing runtime diagnostics" (Apr 2024)
    - WaterProgramming: "MM Borg Training Part 2" (Sep 2024)
"""

import subprocess
from pathlib import Path

from config import (
    get_epsilons,
    DIAGNOSTICS_SETTINGS,
    FFMP_VR_N_SWEEP,
    active_scenario_name,
    run_output_dir,
)

# Per-formulation MOEAFramework problem registrations. Each is built as a
# separate JAR in MOEAFramework-5.0/lib/ with the correct (nvars, nobjs).
# All use the current 7-objective set.
#   drb_ffmp     -> 24 DVs
#   drb_ffmp_{N} -> per-N DV count (varies with number of zones).
#                   Built automatically by slurm/main/build_jars.sh.
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


# Back-compat default (unused when problem_name_for is called)
MOEA_PROBLEM_NAME = "drb_ffmp"


def get_cli_path() -> str:
    """Return path to MOEAFramework CLI executable."""
    return DIAGNOSTICS_SETTINGS["moea_framework_jar"]


def merge_reference_set(
    slug: str,
    seed: int = None,
    runtime_dir: Path = None,
    output_file: Path = None,
    scenario: str = None,
) -> Path:
    """Merge runtime files into an epsilon-dominated reference set.

    Uses ResultFileMerger to combine all snapshots across islands
    into a single non-dominated set.

    Args:
        slug: Run moea slug (inner output subdirectory + filename prefix).
        seed: Seed number (if None, merges all seeds).
        runtime_dir: Directory containing .runtime files.
        output_file: Output .set file path.
        scenario: Scenario-design name (top-level output partition). Defaults
            to the active scenario design.

    Returns:
        Path to the merged .set file.
    """
    if scenario is None:
        scenario = active_scenario_name()
    if runtime_dir is None:
        runtime_dir = run_output_dir(scenario, slug, "runtime")
    if output_file is None:
        sets_dir = run_output_dir(scenario, slug, "sets")
        suffix = f"_seed{seed:02d}" if seed else ""
        output_file = sets_dir / f"{slug}{suffix}_merged.set"

    epsilons = ",".join(str(e) for e in get_epsilons())

    # Find runtime files (filter by seed if specified)
    if seed is not None:
        pattern = f"seed_{seed:02d}_{slug}_*.runtime"
    else:
        pattern = f"*_{slug}_*.runtime"
    runtime_files = sorted(runtime_dir.glob(pattern))

    if not runtime_files:
        raise FileNotFoundError(
            f"No runtime files matching {pattern} in {runtime_dir}"
        )

    cmd = [
        get_cli_path(), "ResultFileMerger",
        "--problem", problem_name_for(slug),
        "--epsilon", epsilons,
        "--output", str(output_file),
    ] + [str(f) for f in runtime_files]

    print(f"  Merging {len(runtime_files)} runtime files -> {output_file.name}")
    subprocess.run(cmd, check=True)
    return output_file


def compute_metrics(
    runtime_file: Path,
    reference_file: Path,
    slug: str,
    output_file: Path = None,
) -> Path:
    """Compute runtime metrics for a single runtime file.

    Uses MetricsEvaluator to compute hypervolume, generational distance,
    inverted generational distance, spacing, epsilon indicator, and
    maximum Pareto front error at each NFE snapshot.

    Args:
        runtime_file: Input .runtime file.
        reference_file: Reference set (.set or .ref file).
        slug: Run slug (picks the MOEAFramework problem registration).
        output_file: Output .metrics file.

    Returns:
        Path to the .metrics file.
    """
    if output_file is None:
        metrics_dir = runtime_file.parent.parent / "metrics"
        metrics_dir.mkdir(parents=True, exist_ok=True)
        output_file = metrics_dir / f"{runtime_file.stem}.metrics"

    epsilons = ",".join(str(e) for e in get_epsilons())

    cmd = [
        get_cli_path(), "MetricsEvaluator",
        "--problem", problem_name_for(slug),
        "--epsilon", epsilons,
        "--input", str(runtime_file),
        "--reference", str(reference_file),
        "--output", str(output_file),
        "--force",
    ]

    print(f"  Computing metrics: {runtime_file.name} -> {output_file.name}")
    subprocess.run(cmd, check=True)
    return output_file


def run_diagnostics(
    slug: str,
    seed: int = 1,
    runtime_dir: Path = None,
    scenario: str = None,
):
    """Run the full diagnostics workflow for a single seed.

    Steps:
        1. Merge all island runtime files into a reference set
        2. Compute metrics for each island's runtime file

    Args:
        slug: Run moea slug (inner output subdirectory + filename prefix).
        seed: Seed number.
        runtime_dir: Directory containing .runtime files.
        scenario: Scenario-design name (top-level output partition). Defaults
            to the active scenario design.

    Returns:
        Tuple of (reference_set_path, list_of_metrics_paths).
    """
    if scenario is None:
        scenario = active_scenario_name()
    if runtime_dir is None:
        runtime_dir = run_output_dir(scenario, slug, "runtime")

    print(f"\n=== MOEAFramework Diagnostics: {scenario}/{slug}, seed {seed} ===\n")

    # Step 1: Merge runtime files into reference set
    print("Step 1: Merging reference set...")
    ref_file = merge_reference_set(
        slug, seed=seed, runtime_dir=runtime_dir, scenario=scenario,
    )

    # Step 2: Compute metrics for each island runtime file
    print("\nStep 2: Computing runtime metrics...")
    pattern = f"seed_{seed:02d}_{slug}_*.runtime"
    runtime_files = sorted(runtime_dir.glob(pattern))

    metrics_files = []
    for rf in runtime_files:
        mf = compute_metrics(rf, ref_file, slug=slug)
        metrics_files.append(mf)

    print(f"\n=== Complete ===")
    print(f"  Reference set: {ref_file}")
    print(f"  Metrics files: {[f.name for f in metrics_files]}")

    return ref_file, metrics_files


def discover_seeds(slug: str, runtime_dir: Path = None, scenario: str = None) -> list[int]:
    """Return sorted list of seed numbers found in the slug's runtime directory.

    Looks for files matching ``seed_NN_{slug}_*.runtime`` and extracts NN.
    """
    import re

    if scenario is None:
        scenario = active_scenario_name()
    if runtime_dir is None:
        runtime_dir = run_output_dir(scenario, slug, "runtime")

    pattern = re.compile(rf"^seed_(\d+)_{re.escape(slug)}_.*\.runtime$")
    seeds = set()
    if runtime_dir.exists():
        for f in runtime_dir.iterdir():
            m = pattern.match(f.name)
            if m:
                seeds.add(int(m.group(1)))
    return sorted(seeds)


def run_full_diagnostics(
    slug: str, seeds: list = None, runtime_dir: Path = None, scenario: str = None,
):
    """Run per-seed diagnostics for every seed present in the slug's output.

    Args:
        slug: Run moea slug (inner output subdirectory + filename prefix).
        seeds: Optional explicit list of seed numbers. If None, auto-discovers
            every seed found in ``outputs/{scenario}/{slug}/runtime``.
        runtime_dir: Optional explicit runtime directory override.
        scenario: Scenario-design name (top-level output partition). Defaults
            to the active scenario design.

    Returns:
        Dict of {seed: (ref_file, metrics_files)}.
    """
    if scenario is None:
        scenario = active_scenario_name()
    if seeds is None:
        seeds = discover_seeds(slug, runtime_dir=runtime_dir, scenario=scenario)
    if not seeds:
        print(f"[{scenario}/{slug}] no runtime files found; skipping.")
        return {}

    results = {}
    for seed in seeds:
        results[seed] = run_diagnostics(
            slug, seed=seed, runtime_dir=runtime_dir, scenario=scenario,
        )
    return results
