"""
generate_presim.py - Generate pre-simulated releases for the trimmed model.

This is a one-time setup step that must be run before Borg optimization.
It runs a single full-model simulation with default NYC operations and extracts
releases from the independent STARFIT reservoirs. The trimmed model replaces
these reservoirs with simple input nodes that read the pre-simulated values,
reducing per-evaluation runtime during optimization.

What the trimmed model skips:
    The DRB contains ~11 independent STARFIT reservoirs (Beltzville, Fewalter,
    etc.) whose operations are unaffected by NYC decisions. Simulating them
    repeatedly is pure overhead during optimization. The trimmed model uses
    pre-computed releases for these reservoirs, keeping only the NYC-coupled
    nodes active.

Runtime:
    Full model simulation over 1945-2022 takes ~3-10 minutes on a laptop.
    This only needs to be run once per inflow_type / date range combination.

Usage:
    python scripts/generate_presim.py

Outputs:
    outputs/presim/presimulated_releases_mgd.csv
    outputs/presim/presimulated_releases_mgd_metadata.json
    outputs/presim/full_model_baseline.json     (kept for reference)
    outputs/presim/full_model_baseline.hdf5     (kept for reference)
"""

import sys
import time
from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from config import (
    START_DATE,
    END_DATE,
    INFLOW_TYPE,
    INITIAL_VOLUME_FRAC,
    PRESIM_DIR,
    PRESIM_FILE,
)


def generate_presim():
    """Run full model and extract presimulated releases."""
    import pywrdrb
    from pywrdrb.pre import generate_presimulated_releases

    print("=" * 60)
    print("  Generate Pre-Simulated Releases (trimmed model setup)")
    print("=" * 60)
    print(f"\n  Inflow type : {INFLOW_TYPE}")
    print(f"  Period      : {START_DATE} to {END_DATE}")
    print(f"  Output dir  : {PRESIM_DIR}")

    if PRESIM_FILE.exists():
        print(f"\n  Overwriting existing file: {PRESIM_FILE}")

    PRESIM_DIR.mkdir(parents=True, exist_ok=True)

    model_json = PRESIM_DIR / "full_model_baseline.json"
    output_hdf5 = PRESIM_DIR / "full_model_baseline.hdf5"

    # --- Step 1: Build full model (no trimming, no presim needed) ---
    print("\n--- Step 1: Building full model ---")
    t0 = time.perf_counter()

    mb = pywrdrb.ModelBuilder(
        inflow_type=INFLOW_TYPE,
        start_date=START_DATE,
        end_date=END_DATE,
        options={
            "nyc_nj_demand_source": "historical",
            "use_trimmed_model": False,
            "initial_volume_frac": INITIAL_VOLUME_FRAC,
        },
    )
    mb.make_model()
    mb.write_model(str(model_json))

    elapsed = time.perf_counter() - t0
    print(f"  Nodes:      {len(mb.model_dict['nodes'])}")
    print(f"  Parameters: {len(mb.model_dict['parameters'])}")
    print(f"  Built in {elapsed:.1f}s")

    # --- Step 2: Run simulation ---
    print("\n--- Step 2: Running full model simulation ---")
    print("  (this takes several minutes for the full 1945-2022 period)")
    t0 = time.perf_counter()

    model = pywrdrb.Model.load(str(model_json))
    recorder = pywrdrb.OutputRecorder(
        model=model,
        output_filename=str(output_hdf5),
    )
    model.run()

    elapsed = time.perf_counter() - t0
    print(f"  Simulation complete in {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"  Output: {output_hdf5}")

    # --- Step 3: Extract presimulated releases ---
    print("\n--- Step 3: Extracting presimulated releases ---")
    t0 = time.perf_counter()

    metadata = generate_presimulated_releases(
        output_filename=str(output_hdf5),
        inflow_type=INFLOW_TYPE,
        output_dir=str(PRESIM_DIR),
    )

    elapsed = time.perf_counter() - t0
    print(f"  Extraction complete in {elapsed:.1f}s")
    print(f"\n  Reservoirs ({len(metadata['reservoirs'])}):")
    for r in metadata["reservoirs"]:
        print(f"    - {r}")
    print(f"\n  CSV:      {metadata['output_file']}")
    print(f"  Metadata: {metadata['metadata_file']}")

    print("\n" + "=" * 60)
    print("  Setup complete. You can now run Borg optimization:")
    print("    bash 02_run_mmborg.sh")
    print("=" * 60)


if __name__ == "__main__":
    generate_presim()
