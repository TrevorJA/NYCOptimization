"""append_demand_utilization.py - Append the NYC utilization column to a staged E_test.

Metadata only, seconds, no streamflow touched. For the variant's parent slug
(``src.etest.get_etest_variant(variant).slug``) this draws one scrambled Latin
hypercube column of the nominal utilization ``u`` over
``DEMAND_UTILIZATION_BOUNDS``, one value per SOW in SOW order, from the
variant's reserved ``{seed_domain}:demand`` seed stream, maps every value to
its demand scale ``k(u)`` through the persisted calibration table
(``scripts/supplemental/demand_utilization_calibration.py``), and writes:

    forcing_profiles.npz        += demand_utilization, demand_scale_k, demand_scale_clipped
                                   (per realization, each SOW value repeated
                                   realizations_per_profile times, exactly like theta_params)
                                   + the scalar provenance of PROVENANCE_KEYS
    _meta.json                  += the same provenance (parent AND every chunk)
    demand_utilization_nominal.csv   sow, u, k, clipped

``u`` is stored as its own key, never appended to ``theta_params``: the
forcing-space figure and ``assert_box_contains`` check ``theta_params``
against the harmonic box. Idempotent (same seed, same values); a slug whose
npz already carries the keys is refused unless ``NYCOPT_FORCE=1``. The prefix
subset is NOT appended here: rebuild it afterwards with
``python -m scripts.supplemental.make_etest_subset --pool {slug} --force`` so
it inherits the first SOWs' values. Then stage the scaled demand per chunk
(``workflow/supplemental/stage_nyc_demand_etest_chunks.sh``).

Usage (identifier flag only):
    python scripts/supplemental/append_demand_utilization.py --variant kn
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src import demand_utilization as du  # noqa: E402


def _load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing; is the E_test staged?")
    return json.loads(path.read_text())


def _write_json(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    os.replace(tmp, path)


def demand_provenance(seed: int, seed_domain: str, sha256: str) -> dict:
    """The scalar provenance block written to the npz and to every ``_meta.json``."""
    return {
        "demand_utilization_bounds": [float(b) for b in du.DEMAND_UTILIZATION_BOUNDS],
        "demand_utilization_seed": int(seed),
        "demand_utilization_seed_domain": str(seed_domain),
        "demand_scale_max": float(du.DEMAND_SCALE_MAX),
        "demand_calibration_sha256": str(sha256),
    }


def append_demand_utilization(slug: str, *, force: bool = False) -> pd.DataFrame:
    """Append the utilization column to the staged E_test ``slug``.

    Args:
        slug: Staged parent E_test slug (a ``kind == "forcing_pool"`` directory
            carrying ``forcing_profiles.npz``; never a prefix subset).
        force: Overwrite an already-appended slug (the values are identical by
            construction; only the provenance timestamp changes).

    Returns:
        The per-SOW table written to ``demand_utilization_nominal.csv``.

    Raises:
        FileExistsError: If the npz already carries the keys and not ``force``.
        ValueError: If ``slug`` is a prefix subset or its npz is not SOW-major.
    """
    from src.ensembles import staged_ensemble_dir

    staged = staged_ensemble_dir(slug)
    meta = _load_json(staged / "_meta.json")
    if meta.get("subset_of"):
        raise ValueError(
            f"'{slug}' is a prefix subset of '{meta['subset_of']}'. Append to the "
            f"parent, then rebuild the subset with make_etest_subset.py --force."
        )
    npz_path = staged / "forcing_profiles.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"{npz_path} is missing; '{slug}' carries no forcing profiles.")
    with np.load(npz_path, allow_pickle=True) as z:
        payload = {key: z[key] for key in z.files}
    present = [k for k in du.NPZ_KEYS if k in payload]
    if present and not force:
        raise FileExistsError(
            f"'{slug}' already carries {present} in forcing_profiles.npz. The column is "
            f"deterministic in the seed, so re-running changes nothing; set NYCOPT_FORCE=1 "
            f"to rewrite it anyway."
        )

    n_theta = int(payload["n_forcing_profiles"])
    rpp = int(payload["realizations_per_profile"])
    n = int(len(payload["realization_ids"]))
    if n != n_theta * rpp:
        raise ValueError(f"'{slug}': {n} realizations != {n_theta} SOWs x R={rpp}.")
    if int(meta.get("n_forcing_profiles", n_theta)) != n_theta:
        raise ValueError(f"'{slug}': npz and _meta.json disagree on n_forcing_profiles.")

    seed_domain = du.demand_seed_domain(str(meta["seed_domain"]))
    seed = du.demand_utilization_seed(str(meta["seed_domain"]))
    u = du.sample_utilization(n_theta, seed)
    k, clipped = du.k_from_u(u)
    sha = du.calibration_sha256()
    provenance = demand_provenance(seed, seed_domain, sha)

    if present and force:
        old = np.asarray(payload["demand_utilization"], dtype=float)[::rpp]
        if old.shape == u.shape and not np.allclose(old, u):
            print(f"[append-u] WARNING: existing u differs from the regenerated column "
                  f"(max |diff| {np.max(np.abs(old - u)):.3e}); overwriting.")

    payload["demand_utilization"] = np.repeat(u, rpp).astype(float)
    payload["demand_scale_k"] = np.repeat(k, rpp).astype(float)
    payload["demand_scale_clipped"] = np.repeat(clipped, rpp).astype(bool)
    payload["demand_utilization_bounds"] = np.asarray(provenance["demand_utilization_bounds"],
                                                     dtype=float)
    payload["demand_utilization_seed"] = np.asarray(seed, dtype=np.int64)
    payload["demand_utilization_seed_domain"] = np.asarray(seed_domain)
    payload["demand_scale_max"] = np.asarray(float(du.DEMAND_SCALE_MAX))
    payload["demand_calibration_sha256"] = np.asarray(sha)

    tmp = npz_path.with_suffix(".npz.tmp")
    with open(tmp, "wb") as fh:
        np.savez(fh, **payload)
    os.replace(tmp, npz_path)

    table = pd.DataFrame({"sow": np.arange(n_theta, dtype=int), "u": u, "k": k,
                          "clipped": clipped.astype(bool)})
    table.to_csv(staged / "demand_utilization_nominal.csv", index=False)

    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    meta.update(provenance)
    meta["demand_utilization_appended"] = stamp
    _write_json(staged / "_meta.json", meta)

    n_chunks = 0
    idx_path = staged / "chunk_index.json"
    if idx_path.exists():
        for entry in _load_json(idx_path).get("chunks") or []:
            chunk_meta_path = staged_ensemble_dir(entry["slug"]) / "_meta.json"
            chunk_meta = _load_json(chunk_meta_path)
            chunk_meta.update(provenance)
            chunk_meta["demand_utilization_appended"] = stamp
            _write_json(chunk_meta_path, chunk_meta)
            n_chunks += 1

    print(f"[append-u] '{slug}': {n_theta} SOWs x R={rpp}; u in "
          f"[{u.min():.3f}, {u.max():.3f}], k in [{k.min():.3f}, {k.max():.3f}], "
          f"{int(clipped.sum())} clipped at k={du.DEMAND_SCALE_MAX}; seed {seed} "
          f"({seed_domain}); calibration sha256 {sha[:12]}...; "
          f"{n_chunks} chunk metas stamped.")
    return table


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Append the NYC utilization column to a staged E_test (metadata only).")
    parser.add_argument("--variant", default=None,
                        help="E_test variant (default: src.etest.E_TEST_VARIANT).")
    args = parser.parse_args()

    from src.etest import E_TEST_VARIANT, get_etest_variant

    variant = get_etest_variant(args.variant or E_TEST_VARIANT)
    force = os.environ.get("NYCOPT_FORCE", "0") == "1"
    append_demand_utilization(variant.slug, force=force)
    print(f"[append-u] next: python -m scripts.supplemental.make_etest_subset "
          f"--pool {variant.slug} --force, then "
          f"sbatch workflow/supplemental/stage_nyc_demand_etest_chunks.sh")


if __name__ == "__main__":
    main()
