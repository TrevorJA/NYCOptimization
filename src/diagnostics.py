"""
diagnostics.py - MOEA runtime diagnostics using MOEAFramework v5.0 CLI.

Provides functions to generate shell commands for the three-step
MOEAFramework diagnostic workflow:

    1. ResultFileMerger:     runtime files -> per-seed .set files
    2. ResultFileSeedMerger: per-seed .set files -> cross-seed .ref file
    3. MetricsEvaluator:     runtime files + .ref -> metrics files

MOEAFramework v5.0 CLI reference:
    https://github.com/MOEAFramework/MOEAFramework/blob/master/docs/commandLineTools.md

Key v5.0 changes from v4.x:
    - ReferenceSetMerger   -> ResultFileSeedMerger
    - ResultFileEvaluator  -> MetricsEvaluator
    - New --overwrite flag (do NOT use with ResultFileMerger)
    - CLI invoked via ./cli instead of java -cp ... org.moeaframework...

References:
    - WaterProgramming: "Introducing MOEAFramework v5.0" (Mar 2025)
    - WaterProgramming: "MM Borg Training Part 2" (Sep 2024)
    - WaterProgramming: "Performing runtime diagnostics" (Apr 2024)
"""

import subprocess
from pathlib import Path

from config import (
    get_n_objs,
    get_epsilons,
    OUTPUTS_DIR,
    DIAGNOSTICS_SETTINGS,
)


def get_cli_path() -> str:
    """Return path to MOEAFramework CLI executable."""
    return DIAGNOSTICS_SETTINGS["moea_framework_jar"]


def extract_sets(
    formulation: str,
    runtime_dir: Path = None,
    sets_dir: Path = None,
):
    """Step 1: Extract per-seed approximation sets from runtime files.

    Uses ResultFileMerger to merge all snapshots within a single
    runtime file into one epsilon-dominated approximation set.

    NOTE: For MM Borg, there may be multiple runtime files per seed
    (one per island, e.g., seed_01_ffmp_0.runtime, seed_01_ffmp_1.runtime).
    These should be merged together per seed.
    """
    if runtime_dir is None:
        runtime_dir = OUTPUTS_DIR / "optimization" / formulation / "runtime"
    if sets_dir is None:
        sets_dir = OUTPUTS_DIR / "optimization" / formulation / "sets"
    sets_dir.mkdir(parents=True, exist_ok=True)

    cli = get_cli_path()
    n_objs = get_n_objs()
    epsilons = ",".join(str(e) for e in get_epsilons())

    runtime_files = sorted(runtime_dir.glob(f"*_{formulation}*.runtime"))
    if not runtime_files:
        raise FileNotFoundError(f"No runtime files in {runtime_dir}")

    # Group runtime files by seed
    seed_groups = {}
    for rf in runtime_files:
        # Extract seed from filename: seed_01_ffmp_0.runtime -> 01
        parts = rf.stem.split("_")
        seed = parts[1]  # e.g., "01"
        seed_groups.setdefault(seed, []).append(rf)

    for seed, files in sorted(seed_groups.items()):
        set_file = sets_dir / f"seed_{seed}_{formulation}.set"
        cmd = [
            cli, "ResultFileMerger",
            "--dimension", str(n_objs),
            "--output", str(set_file),
            "--epsilon", epsilons,
        ] + [str(f) for f in files]

        print(f"  Merging seed {seed}: {len(files)} runtime file(s)")
        subprocess.run(cmd, check=True)

    return sets_dir


def generate_reference_set(
    formulation: str,
    sets_dir: Path = None,
    ref_file: Path = None,
):
    """Step 2: Generate cross-seed reference set.

    Uses ResultFileSeedMerger (v5.0; replaces old ReferenceSetMerger).
    WARNING: Do NOT use --overwrite flag.
    """
    if sets_dir is None:
        sets_dir = OUTPUTS_DIR / "optimization" / formulation / "sets"
    if ref_file is None:
        ref_dir = OUTPUTS_DIR / "reference_sets"
        ref_dir.mkdir(parents=True, exist_ok=True)
        ref_file = ref_dir / f"{formulation}.ref"

    cli = get_cli_path()
    n_objs = get_n_objs()
    epsilons = ",".join(str(e) for e in get_epsilons())

    set_files = sorted(sets_dir.glob(f"seed_*_{formulation}.set"))
    if not set_files:
        raise FileNotFoundError(f"No .set files in {sets_dir}")

    cmd = [
        cli, "ResultFileSeedMerger",
        "--dimension", str(n_objs),
        "--output", str(ref_file),
        "--epsilon", epsilons,
    ] + [str(f) for f in set_files]

    print(f"  Merging {len(set_files)} seed sets -> {ref_file}")
    subprocess.run(cmd, check=True)

    # Count solutions
    n_solutions = sum(
        1 for line in open(ref_file) if line.strip() and not line.startswith("#")
    )
    print(f"  Reference set: {n_solutions} solutions")
    return ref_file


def compute_metrics(
    formulation: str,
    runtime_dir: Path = None,
    ref_file: Path = None,
    metrics_dir: Path = None,
):
    """Step 3: Compute runtime metrics (hypervolume, GD, epsilon indicator).

    Uses MetricsEvaluator (v5.0; replaces old ResultFileEvaluator).
    """
    if runtime_dir is None:
        runtime_dir = OUTPUTS_DIR / "optimization" / formulation / "runtime"
    if ref_file is None:
        ref_file = OUTPUTS_DIR / "reference_sets" / f"{formulation}.ref"
    if metrics_dir is None:
        metrics_dir = OUTPUTS_DIR / "diagnostics" / formulation
    metrics_dir.mkdir(parents=True, exist_ok=True)

    cli = get_cli_path()
    n_objs = get_n_objs()
    epsilons = ",".join(str(e) for e in get_epsilons())

    runtime_files = sorted(runtime_dir.glob(f"*_{formulation}*.runtime"))
    if not runtime_files:
        raise FileNotFoundError(f"No runtime files in {runtime_dir}")

    for rf in runtime_files:
        metrics_file = metrics_dir / f"{rf.stem}.metrics"
        cmd = [
            cli, "MetricsEvaluator",
            "--dimension", str(n_objs),
            "--input", str(rf),
            "--reference", str(ref_file),
            "--epsilon", epsilons,
            "--output", str(metrics_file),
        ]
        print(f"  Evaluating: {rf.stem}")
        subprocess.run(cmd, check=True)

    return metrics_dir


def run_full_diagnostics(formulation: str):
    """Run the complete 3-step diagnostic workflow."""
    print(f"=== MOEA Diagnostics: {formulation} ===\n")

    print("Step 1: Extracting approximation sets...")
    sets_dir = extract_sets(formulation)

    print("\nStep 2: Generating reference set...")
    ref_file = generate_reference_set(formulation)

    print("\nStep 3: Computing runtime metrics...")
    metrics_dir = compute_metrics(formulation, ref_file=ref_file)

    print(f"\n=== Complete ===")
    print(f"  Sets:       {sets_dir}")
    print(f"  Reference:  {ref_file}")
    print(f"  Metrics:    {metrics_dir}")
