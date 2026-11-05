"""
make_etest_subset.py - Stage E_test as the chunk PREFIX of a larger generated design (metadata only).

E_test is the registry object of ``src.etest``: N_theta = 500 LHS SOWs x R = 25 x 50 yr in 25
chunks of 500 realizations (``etest_kn_50yr_n12500``). When the design generated on disk is
larger (the campaign E_test is the leading 25 chunks of the 1,000-SOW design generated
2026-08-19, ``etest_kn_50yr_n25000``), this tool cuts E_test from it without regenerating,
copying or re-prepping anything. The E_test directory holds metadata only (``_meta.json``,
``chunk_index.json``, the sliced ``forcing_profiles.npz`` / ``hazard_image.npz`` and
``demand_utilization_nominal.csv``) whose chunk entries point at the parent's already-staged
chunk directories (step-04 presim and the scaled NYC demand series included). Every
downstream reader (the chunked re-eval, the hazard-image pass, the diagnostics and the
figures) resolves the E_test slug alone; the parent is never read again. A design generated
at the registry size needs no cut.

PREFIX-ONLY, by construction: unit rows are keyed by GLOBAL SOW id (``global_realization_id //
R``) while the reduced spec's index space is ``range(K * chunk_size)``, so the selected chunks
must be the leading chunks. The LHS theta rows are randomly ordered (scipy ``LatinHypercube``
permutes strata independently per axis), so a prefix is an unbiased, well-spread subsample. R
is untouched, so every per-SOW objective value is computed in the final metric currency. The
staged E_test passes ``assert_staged_etest_contract`` and records ``subset_of`` in its
``_meta.json``.

Usage (login node is fine; pure metadata I/O). The defaults are the registry's chunk count
and slug; ``--n-chunks`` / ``--out-slug`` exist for fixtures and checks only:
    python3 -m scripts.supplemental.make_etest_subset --pool etest_kn_50yr_n25000

Then every re-eval step (05, 08, 09, 09b, 10, 11) takes:
    NYCOPT_REEVAL_ENSEMBLE_PRESET=etest_kn_50yr_n12500
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

#: Per-SOW nominal utilization table the append step writes beside ``forcing_profiles.npz``.
_NOMINAL_CSV = "demand_utilization_nominal.csv"


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


def _subset_nominal_table(pool_dir: Path, out_dir: Path, n_theta_sub: int) -> str | None:
    """Copy the leading ``n_theta_sub`` rows of the parent's nominal utilization table.

    The table (``sow, u, k, clipped``; one row per SOW) is written by
    ``append_demand_utilization.py`` beside the parent's npz; the E_test
    directory carries its own prefix so no reader has to follow ``subset_of``.

    Returns:
        A note describing the copy, or ``None`` when the parent has no table.
    """
    import pandas as pd

    src = pool_dir / _NOMINAL_CSV
    if not src.is_file():
        return None
    table = pd.read_csv(src)
    head = table[table["sow"] < n_theta_sub].sort_values("sow").reset_index(drop=True)
    if len(head) != n_theta_sub:
        raise ValueError(
            f"{src} holds {len(head)} rows for SOWs < {n_theta_sub}, expected {n_theta_sub}.")
    head.to_csv(out_dir / _NOMINAL_CSV, index=False)
    return f"{_NOMINAL_CSV}: {len(table)} -> {len(head)} SOW rows"


def _verify_demand_utilization_slice(pool_npz: Path, out_npz: Path, n_sub: int) -> None:
    """Assert the sliced npz carries the NYC utilization keys as a length-``n_sub`` prefix.

    Raises:
        ValueError: If the parent lacks the keys (append them first) or the
            slice is not the parent's leading ``n_sub`` values.
    """
    from src.demand_utilization import NPZ_KEYS

    with np.load(pool_npz, allow_pickle=True) as z:
        missing = [k for k in NPZ_KEYS if k not in z.files]
        if missing:
            raise ValueError(
                f"{pool_npz} lacks the NYC utilization keys {missing}; run "
                f"scripts/supplemental/append_demand_utilization.py on the parent before "
                f"cutting E_test from it.")
        parent = {k: np.asarray(z[k])[:n_sub] for k in NPZ_KEYS}
    with np.load(out_npz, allow_pickle=True) as z:
        for key in NPZ_KEYS:
            if key not in z.files:
                raise ValueError(f"{out_npz} lost the utilization key {key!r} in the slice.")
            arr = np.asarray(z[key])
            if arr.shape[0] != n_sub or not np.array_equal(arr, parent[key]):
                raise ValueError(
                    f"{out_npz}: {key} is not the parent's leading {n_sub} values "
                    f"(got shape {arr.shape}).")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage E_test as a metadata-only chunk prefix of a larger generated design.")
    parser.add_argument("--pool", required=True,
                        help="Staged chunked parent slug, e.g. etest_kn_50yr_n25000.")
    parser.add_argument("--variant", default=None,
                        help="E_test variant whose registry size and slug are the defaults "
                             "(default: src.etest.E_TEST_VARIANT).")
    parser.add_argument("--n-chunks", type=int, default=None,
                        help="Number of LEADING chunks to include (default: the registry's "
                             "chunk count, ETestVariant.n_chunks).")
    parser.add_argument("--out-slug", default=None,
                        help="Staged slug to write (default: the registry's E_test slug, "
                             "ETestVariant.slug).")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite an existing staged directory.")
    args = parser.parse_args()

    from src.etest import E_TEST_VARIANT, get_etest_variant

    variant = get_etest_variant(args.variant or E_TEST_VARIANT)
    if args.n_chunks is None:
        args.n_chunks = variant.n_chunks
    out_slug = args.out_slug or variant.slug

    from src.ensembles import staged_ensemble_dir, staged_ensemble_missing

    pool_dir = staged_ensemble_dir(args.pool)
    meta = _load_json(pool_dir / "_meta.json")
    idx = _load_json(pool_dir / "chunk_index.json")
    entries = idx.get("chunks") or []
    if not entries:
        raise ValueError(f"pool '{args.pool}' has no chunk_index.json chunk entries; "
                         f"the prefix mechanism is chunk-based.")

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
    if out_slug == variant.slug and (n_theta_sub, r_per_sow) != (
            variant.n_theta, variant.realizations_per_theta):
        raise ValueError(
            f"'{out_slug}' is the registry's E_test slug, which names N_theta = "
            f"{variant.n_theta} x R = {variant.realizations_per_theta}; the requested prefix "
            f"is {n_theta_sub} SOWs x R = {r_per_sow}. Fix --n-chunks or name another slug.")
    out_dir = staged_ensemble_dir(out_slug)

    # Every referenced chunk must be FULLY staged (incl. step-04 presim and the
    # scaled demand series): the whole point is to reuse existing prep, so
    # absence is a hard error, never a re-prep.
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
        "Metadata-only chunk prefix of a larger generated design; chunk entries "
        "reference the parent's staged chunk directories (no data copied). "
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

    # --- Per-realization sidecars sliced to the prefix: forcing_profiles.npz
    # (SOW grouping, theta joins, the utilization column), hazard_image.npz
    # (step-11 scenario discovery loads the realization-level image from the
    # re-eval spec's staged dir) and the per-SOW nominal utilization table. ---
    npz_notes = _subset_npz(pool_dir / "forcing_profiles.npz",
                            out_dir / "forcing_profiles.npz",
                            n_pool, n_theta_pool, n_sub, n_theta_sub)
    if (pool_dir / "hazard_image.npz").exists():
        npz_notes += _subset_npz(pool_dir / "hazard_image.npz",
                                 out_dir / "hazard_image.npz",
                                 n_pool, n_theta_pool, n_sub, n_theta_sub)
    nominal_note = _subset_nominal_table(pool_dir, out_dir, n_theta_sub)
    if nominal_note:
        npz_notes.append(nominal_note)

    # --- Verify the staged E_test resolves exactly as the re-eval stack will consume it. ---
    from src.ensembles import get_ensemble_spec, pool_chunk_specs
    from src.etest import assert_staged_etest_contract
    from src.reeval_core import sow_grouping

    spec = get_ensemble_spec(out_slug)
    assert spec.is_ensemble and spec.n_realizations == n_sub, (
        f"staged spec resolved wrong: n={spec.n_realizations}, expected {n_sub}")
    chunks = pool_chunk_specs(out_slug)
    gids = [g for _spec, ids in chunks for g in ids]
    assert gids == list(range(n_sub)), "chunk global ids are not range(n_sub)"
    sow_ids, n_sow, r = sow_grouping(spec, spec.realization_indices)
    assert sow_ids is not None and n_sow == n_theta_sub and r == r_per_sow, (
        f"SOW grouping mismatch: n_sow={n_sow} (expected {n_theta_sub}), "
        f"R={r} (expected {r_per_sow})")
    # The NYC utilization column is per realization, so the slicer carried it;
    # E_test inherits the leading SOWs' values and the parent's provenance.
    _verify_demand_utilization_slice(pool_dir / "forcing_profiles.npz",
                                     out_dir / "forcing_profiles.npz", n_sub)
    assert_staged_etest_contract(out_slug)

    print(f"[etest-prefix] staged '{out_slug}' at {out_dir}")
    print(f"[etest-prefix]   {args.n_chunks}/{len(entries)} chunks -> "
          f"{n_sub} realizations = {n_theta_sub} SOWs x R={r_per_sow} "
          f"(parent: {n_theta_pool} SOWs)")
    for n in npz_notes:
        print(f"[etest-prefix]   {n}")
    print(f"[etest-prefix] verified: spec resolution, chunk global ids, SOW grouping, "
          f"E_test staging contract.")
    print(f"[etest-prefix] use with: NYCOPT_REEVAL_ENSEMBLE_PRESET={out_slug}")


if __name__ == "__main__":
    sys.exit(main())
