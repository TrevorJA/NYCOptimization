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

from config import get_epsilons, OUTPUTS_DIR, DIAGNOSTICS_SETTINGS

# MOEAFramework problem name (must match JAR in lib/)
MOEA_PROBLEM_NAME = "drb_ffmp"


def get_cli_path() -> str:
    """Return path to MOEAFramework CLI executable."""
    return DIAGNOSTICS_SETTINGS["moea_framework_jar"]


def merge_reference_set(
    formulation: str,
    seed: int = None,
    runtime_dir: Path = None,
    output_file: Path = None,
) -> Path:
    """Merge runtime files into an epsilon-dominated reference set.

    Uses ResultFileMerger to combine all snapshots across islands
    into a single non-dominated set.

    Args:
        formulation: Formulation name.
        seed: Seed number (if None, merges all seeds).
        runtime_dir: Directory containing .runtime files.
        output_file: Output .set file path.

    Returns:
        Path to the merged .set file.
    """
    if runtime_dir is None:
        runtime_dir = OUTPUTS_DIR / "optimization" / formulation / "runtime"
    if output_file is None:
        sets_dir = OUTPUTS_DIR / "optimization" / formulation / "sets"
        sets_dir.mkdir(parents=True, exist_ok=True)
        suffix = f"_seed{seed:02d}" if seed else ""
        output_file = sets_dir / f"{formulation}{suffix}_merged.set"

    epsilons = ",".join(str(e) for e in get_epsilons())

    # Find runtime files (filter by seed if specified)
    if seed is not None:
        pattern = f"seed_{seed:02d}_{formulation}_*.runtime"
    else:
        pattern = f"*_{formulation}_*.runtime"
    runtime_files = sorted(runtime_dir.glob(pattern))

    if not runtime_files:
        raise FileNotFoundError(
            f"No runtime files matching {pattern} in {runtime_dir}"
        )

    cmd = [
        get_cli_path(), "ResultFileMerger",
        "--problem", MOEA_PROBLEM_NAME,
        "--epsilon", epsilons,
        "--output", str(output_file),
    ] + [str(f) for f in runtime_files]

    print(f"  Merging {len(runtime_files)} runtime files -> {output_file.name}")
    subprocess.run(cmd, check=True)
    return output_file


def compute_metrics(
    runtime_file: Path,
    reference_file: Path,
    output_file: Path = None,
) -> Path:
    """Compute runtime metrics for a single runtime file.

    Uses MetricsEvaluator to compute hypervolume, generational distance,
    inverted generational distance, spacing, epsilon indicator, and
    maximum Pareto front error at each NFE snapshot.

    Args:
        runtime_file: Input .runtime file.
        reference_file: Reference set (.set or .ref file).
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
        "--problem", MOEA_PROBLEM_NAME,
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
    formulation: str,
    seed: int = 1,
    runtime_dir: Path = None,
):
    """Run the full diagnostics workflow for a single seed.

    Steps:
        1. Merge all island runtime files into a reference set
        2. Compute metrics for each island's runtime file

    Args:
        formulation: Formulation name.
        seed: Seed number.
        runtime_dir: Directory containing .runtime files.

    Returns:
        Tuple of (reference_set_path, list_of_metrics_paths).
    """
    if runtime_dir is None:
        runtime_dir = OUTPUTS_DIR / "optimization" / formulation / "runtime"

    print(f"\n=== MOEAFramework Diagnostics: {formulation}, seed {seed} ===\n")

    # Step 1: Merge runtime files into reference set
    print("Step 1: Merging reference set...")
    ref_file = merge_reference_set(formulation, seed=seed, runtime_dir=runtime_dir)

    # Step 2: Compute metrics for each island runtime file
    print("\nStep 2: Computing runtime metrics...")
    pattern = f"seed_{seed:02d}_{formulation}_*.runtime"
    runtime_files = sorted(runtime_dir.glob(pattern))

    metrics_files = []
    for rf in runtime_files:
        mf = compute_metrics(rf, ref_file)
        metrics_files.append(mf)

    print(f"\n=== Complete ===")
    print(f"  Reference set: {ref_file}")
    print(f"  Metrics files: {[f.name for f in metrics_files]}")

    return ref_file, metrics_files
