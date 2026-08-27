"""
make_etest_subset.py - Stage a metadata-only chunk-PREFIX subset of a staged, chunked E_test.

Purpose: the campaign re-evaluates every Pareto set on the leading ``E_TEST_REEVAL_N_THETA``
SOWs of the generated 1,000-point E_test design (``src.etest``; 500 SOWs = the first 25 chunks,
``etest_kn_50yr_n25000_first25ch``), and interim, cost-bounded re-evaluations used shorter
prefixes (``first10ch``, 200 SOWs). This tool stages either. The subset keeps R (realizations per
SOW) untouched, so every per-SOW objective value is computed in exactly the final metric currency —
only the cross-SOW Monte Carlo error depends on the prefix length (worst-case SE of a satisficing
fraction is 0.5/sqrt(N_theta): +/-3.5 pp at 200 SOWs, +/-2.2 pp at 500, +/-1.6 pp at 1,000).

NOTHING is regenerated, copied, or re-prepped. The subset directory holds only three metadata
files — ``_meta.json``, ``chunk_index.json``, ``forcing_profiles.npz`` — and its chunk entries
point at the parent pool's ALREADY-STAGED chunk directories (step-04 presim included). The
chunked re-eval (``src.chunk_reeval`` via ``pool_chunk_specs``) then runs on it end-to-end with
no code changes; outputs land under a distinct ``reeval/{subset_slug}/`` tag, so interim cubes
can never be confused with (or clobber) the eventual full-E_test cube.

PREFIX-ONLY, by construction: unit rows are keyed by GLOBAL SOW id (``global_realization_id //
R``) while the reduced spec's index space is ``range(K * chunk_size)``, so the selected chunks
must be the leading chunks (global ids starting at 0, contiguous). A non-prefix selection would
key rows past the reduced SOW space and break the merge. This is statistically sound because
the E_test LHS theta rows are randomly ordered (scipy ``LatinHypercube`` assigns strata by
independent random permutation per axis), so a prefix is an unbiased, well-spread subsample —
verified 2026-08-12 on ``etest_kn_50yr_n25000``: the first-200-SOW axis means/ranges match the
full 1,000 (e.g. axis ``m``: 0.076 vs 0.067 over a (-0.144, 0.278) range).

Scope note: ``src.etest``'s "never subsampled" contract concerns the search-side control
argument (uniform subsets of i.i.d. pools vs LHS); it does not preclude an interim theta-subset
for re-evaluation. The subset still passes ``assert_staged_etest_contract`` (LHS, R > 1,
``etest:*`` seed domain), and this script stamps full provenance into the subset ``_meta.json``.
Manuscript numbers must come from ONE cube per claim — never mix subset-cube and full-cube
values; the separate re-eval tag enforces the separation on disk.

Usage (login node is fine — pure metadata I/O). The campaign subset (``--n-chunks`` defaults to
the campaign value, 25):
    python3 -m scripts.supplemental.make_etest_subset --pool etest_kn_50yr_n25000

Then point every re-eval step at it (steps 05, 08, 09, 09b, 10):
    NYCOPT_REEVAL_ENSEMBLE_PRESET=etest_kn_50yr_n25000_first25ch \
        sbatch ... workflow/09_simulate_test_chunks.sh
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def _load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing — is the pool staged?")
    return json.loads(path.read_text())


def _validate_prefix(entries: list[dict], n_chunks: int, chunk_size: int) -> None:
    """Assert the first ``n_chunks`` chunk-index entries are the contiguous global prefix."""
    if n_chunks < 1 or n_chunks > len(entries):
        raise ValueError(
            f"--n-chunks must be in [1, {len(entries)}] for this pool, got {n_chunks}."
        )
    expect_start = 0
    for i, e in enumerate(entries[:n_chunks]):
        if int(e["chunk_index"]) != i:
            raise ValueError(
                f"chunk_index.json entry {i} carries chunk_index={e['chunk_index']}; the "
                f"subset must be the leading prefix (rows keyed by global SOW id)."
            )
        if int(e["global_start"]) != expect_start:
            raise ValueError(
                f"chunk {i} starts at global {e['global_start']}, expected {expect_start}: "
                f"the prefix is not contiguous from 0."
            )
        if int(e["global_end"]) - int(e["global_start"]) != int(e["n_realizations"]):
            raise ValueError(f"chunk {i}: global range does not match n_realizations.")
        if int(e["n_realizations"]) != chunk_size:
            # Tolerated (a short tail chunk can never be in a valid prefix anyway),
            # but worth failing loudly on rather than mis-sizing the subset.
            raise ValueError(
                f"chunk {i} has {e['n_realizations']} realizations, expected chunk_size="
                f"{chunk_size}."
            )
        expect_start = int(e["global_end"])


def _subset_npz(pool_npz: Path, out_npz: Path, n_pool: int, n_theta_pool: int,
                n_sub: int, n_theta_sub: int) -> list[str]:
    """Write the sliced ``forcing_profiles.npz``; return a description of what was done."""
    notes = []
    payload: dict[str, np.ndarray] = {}
    with np.load(pool_npz, allow_pickle=True) as z:
        for key in z.files:
            arr = z[key]
            if arr.ndim >= 1 and arr.shape[0] == n_pool:
                payload[key] = arr[:n_sub]
                notes.append(f"{key}: sliced per-realization axis {n_pool} -> {n_sub}")
            elif arr.ndim >= 1 and arr.shape[0] == n_theta_pool:
                payload[key] = arr[:n_theta_sub]
                notes.append(f"{key}: sliced per-theta axis {n_theta_pool} -> {n_theta_sub}")
            elif key == "n_forcing_profiles":
                payload[key] = np.asarray(n_theta_sub)
                notes.append(f"{key}: {n_theta_pool} -> {n_theta_sub}")
            else:
                payload[key] = arr
    np.savez(out_npz, **payload)
    return notes


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage a metadata-only chunk-prefix subset of a chunked E_test.")
    parser.add_argument("--pool", required=True,
                        help="Staged chunked pool slug, e.g. etest_kn_50yr_n25000.")
    parser.add_argument("--n-chunks", type=int, default=None,
                        help="Number of LEADING chunks to include (default: the campaign "
                             "re-evaluation subset, src.etest.campaign_etest_variant()"
                             ".reeval_n_chunks).")
    parser.add_argument("--out-slug", default=None,
                        help="Subset slug (default: {pool}_first{K}ch).")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite an existing subset directory.")
    args = parser.parse_args()
    if args.n_chunks is None:
        from src.etest import campaign_etest_variant
        args.n_chunks = campaign_etest_variant().reeval_n_chunks

    from src.ensembles import staged_ensemble_dir, staged_ensemble_missing

    pool_dir = staged_ensemble_dir(args.pool)
    meta = _load_json(pool_dir / "_meta.json")
    idx = _load_json(pool_dir / "chunk_index.json")
    entries = idx.get("chunks") or []
    if not entries:
        raise ValueError(f"pool '{args.pool}' has no chunk_index.json chunk entries; "
                         f"the subset mechanism is chunk-based.")

    chunk_size = int(idx.get("chunk_size", meta.get("chunk_size", 0)) or
                     entries[0]["n_realizations"])
    r_per_sow = int(meta["realizations_per_profile"])
    if chunk_size % r_per_sow != 0:
        raise ValueError(f"chunk_size ({chunk_size}) is not a multiple of R ({r_per_sow}).")
    _validate_prefix(entries, args.n_chunks, chunk_size)

    n_pool = int(meta["n_realizations"])
    n_theta_pool = int(meta["n_forcing_profiles"])
    n_sub = args.n_chunks * chunk_size
    n_theta_sub = n_sub // r_per_sow
    out_slug = args.out_slug or f"{args.pool}_first{args.n_chunks}ch"
    out_dir = staged_ensemble_dir(out_slug)

    # Every referenced chunk must be FULLY staged (incl. step-04 presim): the whole
    # point is to reuse existing prep, so absence is a hard error, never a re-prep.
    for e in entries[:args.n_chunks]:
        missing = staged_ensemble_missing(e["slug"])
        if missing:
            raise FileNotFoundError(
                f"chunk '{e['slug']}' is missing staged files {missing}; refusing — this "
                f"tool never regenerates or re-preps data.")

    if out_dir.exists():
        if not args.force:
            raise FileExistsError(f"{out_dir} exists; pass --force to overwrite.")
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    # --- _meta.json: parent meta with sizes reduced + provenance stamped. ---
    sub_meta = dict(meta)
    sub_meta["slug"] = out_slug
    sub_meta["n_realizations"] = n_sub
    sub_meta["n_forcing_profiles"] = n_theta_sub
    if "n_chunks" in sub_meta:
        sub_meta["n_chunks"] = args.n_chunks
    sub_meta["subset_of"] = args.pool
    sub_meta["subset_n_chunks"] = args.n_chunks
    sub_meta["subset_note"] = (
        "Metadata-only chunk-prefix subset for re-evaluation; chunk entries "
        "reference the parent pool's staged chunk directories (no data copied). "
        "Prefix-only: rows are keyed by global SOW id. LHS rows are randomly ordered, "
        "so the prefix is an unbiased theta subsample."
    )
    sub_meta["subset_created"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    (out_dir / "_meta.json").write_text(json.dumps(sub_meta, indent=2))

    # --- chunk_index.json: the leading entries VERBATIM (slugs -> parent chunk dirs). ---
    sub_idx = {
        "pool_slug": out_slug,
        "source_pool": args.pool,
        "n_realizations": n_sub,
        "chunk_size": chunk_size,
        "n_chunks": args.n_chunks,
        "chunks": entries[:args.n_chunks],
    }
    (out_dir / "chunk_index.json").write_text(json.dumps(sub_idx, indent=2))

    # --- Per-realization npz sidecars sliced to the prefix: forcing_profiles.npz
    # (SOW grouping + step-11 theta joins) and hazard_image.npz (step-11 scenario
    # discovery runs in HAZARD space and loads the image from the re-eval spec's
    # staged dir; rows/ids are per-realization, so the prefix slice stays aligned). ---
    npz_notes = _subset_npz(pool_dir / "forcing_profiles.npz",
                            out_dir / "forcing_profiles.npz",
                            n_pool, n_theta_pool, n_sub, n_theta_sub)
    if (pool_dir / "hazard_image.npz").exists():
        npz_notes += _subset_npz(pool_dir / "hazard_image.npz",
                                 out_dir / "hazard_image.npz",
                                 n_pool, n_theta_pool, n_sub, n_theta_sub)

    # --- Verify the subset resolves exactly as the re-eval stack will consume it. ---
    from src.ensembles import get_ensemble_spec, pool_chunk_specs
    from src.etest import assert_staged_etest_contract
    from src.reeval_core import sow_grouping

    spec = get_ensemble_spec(out_slug)
    assert spec.is_ensemble and spec.n_realizations == n_sub, (
        f"subset spec resolved wrong: n={spec.n_realizations}, expected {n_sub}")
    chunks = pool_chunk_specs(out_slug)
    gids = [g for _spec, ids in chunks for g in ids]
    assert gids == list(range(n_sub)), "subset chunk global ids are not range(n_sub)"
    sow_ids, n_sow, r = sow_grouping(spec, spec.realization_indices)
    assert sow_ids is not None and n_sow == n_theta_sub and r == r_per_sow, (
        f"SOW grouping mismatch: n_sow={n_sow} (expected {n_theta_sub}), "
        f"R={r} (expected {r_per_sow})")
    assert_staged_etest_contract(out_slug)

    print(f"[etest-subset] staged '{out_slug}' at {out_dir}")
    print(f"[etest-subset]   {args.n_chunks}/{len(entries)} chunks -> "
          f"{n_sub} realizations = {n_theta_sub} SOWs x R={r_per_sow} "
          f"(parent: {n_theta_pool} SOWs)")
    for n in npz_notes:
        print(f"[etest-subset]   npz: {n}")
    print(f"[etest-subset] verified: spec resolution, chunk global ids, SOW grouping, "
          f"E_test staging contract.")
    print(f"[etest-subset] use with: NYCOPT_REEVAL_ENSEMBLE_PRESET={out_slug}")


if __name__ == "__main__":
    sys.exit(main())
