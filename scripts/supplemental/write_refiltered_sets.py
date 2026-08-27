"""
write_refiltered_sets.py - Re-filter of production reference sets under the registry epsilons.

Writes `{slug}_merged_{suffix}.set` per scenario: the epsilon-box nondominated
archive of the step-07 raw union (`{slug}_merged_raw.set`) under the current
registry epsilon vector (config.get_epsilons()); the originals are untouched.
Follows the line-preserving rewrite convention of
`src.diagnostics.epsilon_box_filter_set` (header comments kept, retained data
lines copied byte-for-byte, trailing `#` terminator).

Usage (from repo root, venv; submit via the epsilon_ensemble_refilter.sh
launcher or any `shared` job):
    python3 scripts/supplemental/write_refiltered_sets.py \
        --slug ffmp_obj8 --suffix eps20260812 \
        --scenarios historic fixed_probabilistic hazard_filling_stationary
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from epsilon_ensemble_refilter import fast_epsilon_nondominated  # noqa: E402

from src.formulations import get_n_vars, get_n_objs  # noqa: E402
from src.load.reference_set import load_reference_set  # noqa: E402
from src.sensitivity_common import epsilon_nondominated  # noqa: E402
from config import get_epsilons  # noqa: E402

import numpy as np  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--slug", default="ffmp_obj8")
    ap.add_argument("--suffix", default="eps20260812")
    ap.add_argument("--scenarios", nargs="+",
                    default=["historic", "fixed_probabilistic",
                             "hazard_filling_stationary"])
    args = ap.parse_args()

    eps = np.array(get_epsilons(), dtype=float)
    n_vars, n_objs = get_n_vars("ffmp"), get_n_objs()
    print(f"registry epsilons: {[float(e) for e in eps]}", flush=True)

    for scenario in args.scenarios:
        raw = (Path("outputs") / scenario / args.slug / "sets"
               / f"{args.slug}_merged_raw.set")
        if not raw.exists():
            print(f"[skip] {scenario}: {raw} missing", flush=True)
            continue
        out = raw.with_name(f"{args.slug}_merged_{args.suffix}.set")
        if out.exists():
            sys.exit(f"{out} already exists — refusing to overwrite")

        _, objs = load_reference_set(raw, n_vars, n_objs=n_objs)
        keep = set(fast_epsilon_nondominated(objs, eps).tolist())
        # cross-check on the small result against the validated reference
        ref = set(epsilon_nondominated(objs[sorted(keep)], eps).tolist())
        if len(ref) != len(keep):
            sys.exit(f"{scenario}: fast/reference filter disagree — aborting")

        lines = raw.read_text().splitlines(keepends=True)
        header = [l for l in lines if l.startswith("#")]
        data = [l for l in lines if not l.startswith("#") and l.strip()]
        if len(data) != len(objs):
            sys.exit(f"{raw}: parsed {len(objs)} rows but found {len(data)} "
                     "data lines — aborting")
        filtered = [l for i, l in enumerate(data) if i in keep]
        body = header[:-1] if header and header[-1].strip() == "#" else header
        tail = ["#\n"] if body is not header else [header[-1]]
        out.write_text("".join(body + filtered + tail))
        print(f"[{scenario}] {raw.name} ({len(data)}) -> {out.name} "
              f"({len(filtered)})", flush=True)


if __name__ == "__main__":
    main()
