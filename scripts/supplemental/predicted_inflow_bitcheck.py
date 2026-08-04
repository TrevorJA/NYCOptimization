"""predicted_inflow_bitcheck.py - Production-scale acceptance gate for the
vectorized perfect-foresight predicted-inflow kernel.

Regenerates one staged E_test chunk's predicted inflows through the VECTORIZED
PredictedInflowEnsemblePreprocessor (pywrdrb, MPI across realizations exactly
like step 04) into a SCRATCH file, then compares every dataset against the
already-staged production artifact (which the scalar path produced). The staged
chunk is never opened for writing.

Pass criteria (the presim precedent, job 19660330, came back bit-identical
cross-job): exact equality expected. Cross-era comparisons on Anvil formally
tolerate <= 1% of each column's robust range (p99-p1) per the
verify_shard_boundaries convention; ANY nonzero diff is reported per column for
manual review before the vectorized path is accepted. The scratch file is
deleted afterwards on success (kept on mismatch for inspection).

Run only under sbatch/srun (compute nodes), via
workflow/supplemental/predicted_inflow_bitcheck.sh.

Configuration via env (no CLI value flags):

    NYCOPT_BITCHECK_PRESET   staged chunk slug to check
                             (default etest_kn_50yr_n25000__chunk000)
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_DIR))

import h5py  # noqa: E402
import numpy as np  # noqa: E402

import config  # noqa: E402
from src.ensembles import (  # noqa: E402
    get_ensemble_spec,
    register_ensemble_path,
    staged_ensemble_dir,
)

_DEFAULT_PRESET = "etest_kn_50yr_n25000__chunk000"


def _get_mpi_context():
    """Return (comm, rank, size). Falls back to (None, 0, 1) without a runtime."""
    try:
        from mpi4py import MPI

        comm = MPI.COMM_WORLD
        return comm, comm.Get_rank(), comm.Get_size()
    except Exception:
        return None, 0, 1


def _compare(staged_path: Path, scratch_path: Path) -> tuple[int, list[str]]:
    """Compare every dataset; return (n_mismatched_columns, report lines)."""
    lines: list[str] = []
    n_bad = 0
    with h5py.File(staged_path, "r") as f_ref, h5py.File(scratch_path, "r") as f_new:
        ref_groups, new_groups = set(f_ref.keys()), set(f_new.keys())
        if ref_groups != new_groups:
            return 1, [f"GROUP SET MISMATCH: staged-only={sorted(ref_groups - new_groups)[:5]} "
                       f"scratch-only={sorted(new_groups - ref_groups)[:5]}"]
        worst = {}
        for rid in sorted(ref_groups, key=lambda x: (len(x), x)):
            g_ref, g_new = f_ref[rid], f_new[rid]
            if set(g_ref.keys()) != set(g_new.keys()):
                return 1, [f"DATASET SET MISMATCH in realization {rid}"]
            for ds in g_ref.keys():
                a, b = g_ref[ds][:], g_new[ds][:]
                if ds == "datetime":
                    if not (a == b).all():
                        n_bad += 1
                        lines.append(f"realization {rid}: datetime mismatch")
                    continue
                if np.array_equal(a, b):
                    continue
                rng = np.percentile(a, 99) - np.percentile(a, 1)
                denom = rng if rng > 0 else 1.0
                rel = float(np.max(np.abs(a - b)) / denom)
                prev = worst.get(ds, 0.0)
                worst[ds] = max(prev, rel)
        for ds, rel in sorted(worst.items()):
            n_bad += 1
            lines.append(f"column {ds}: max |diff| = {rel:.3e} of robust range "
                         f"({'within' if rel <= 0.01 else 'BEYOND'} the 1% cross-era tolerance)")
    if not lines:
        lines.append("EXACT: every dataset of every realization is bit-identical.")
    return n_bad, lines


def main() -> None:
    from pywrdrb.pre.predict_inflows import PredictedInflowEnsemblePreprocessor

    preset = os.environ.get("NYCOPT_BITCHECK_PRESET", _DEFAULT_PRESET)
    comm, rank, size = _get_mpi_context()

    spec = get_ensemble_spec(preset)
    slug = spec.inflow_type
    register_ensemble_path(slug)
    flows_dir = staged_ensemble_dir(slug)
    base_inflow = flows_dir / "catchment_inflow_mgd.hdf5"
    staged_out = flows_dir / "predicted_inflows_mgd.hdf5"
    if not staged_out.exists():
        raise FileNotFoundError(f"No staged predicted inflows to compare against: {staged_out}")

    scratch_dir = Path(config.OUTPUTS_DIR) / "tmp_bitcheck" / slug
    scratch_out = scratch_dir / "predicted_inflows_mgd.hdf5"
    if rank == 0:
        scratch_dir.mkdir(parents=True, exist_ok=True)
        print(f"[bitcheck] preset={preset} ranks={size} "
              f"staged={staged_out} scratch={scratch_out}", flush=True)

    pp = PredictedInflowEnsemblePreprocessor(
        flow_type=slug,
        ensemble_hdf5_file=str(base_inflow),
        realization_ids=list(spec.realization_indices),
        modes=(config.PYWRDRB_FLOW_PREDICTION_MODE,),
        use_mpi=size > 1,
        comm=comm,
    )
    assert pp._vectorize_perfect_foresight, "expected the vectorized default"
    # The staged production artifact must never be opened for writing.
    pp.output_dirs = {"predicted_inflows_mgd.hdf5": scratch_out}
    assert Path(pp.output_dirs["predicted_inflows_mgd.hdf5"]).resolve() \
        != staged_out.resolve()

    pp.load()
    pp.process()
    pp.save()

    if rank != 0:
        return
    n_bad, lines = _compare(staged_out, scratch_out)
    for ln in lines:
        print(f"[bitcheck] {ln}", flush=True)
    if n_bad == 0:
        scratch_out.unlink()
        try:
            scratch_dir.rmdir()
        except OSError:
            pass
        print("[bitcheck] PASS (scratch removed)", flush=True)
    else:
        print(f"[bitcheck] {n_bad} mismatched column(s); scratch kept at "
              f"{scratch_out} for inspection.", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
