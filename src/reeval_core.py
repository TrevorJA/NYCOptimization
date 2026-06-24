"""reeval_core.py - shared helpers for re-evaluating Pareto-optimal policies on
a COMMON (held-out) streamflow ensemble.

Both ``src.reevaluate`` (multiprocessing) and ``src.reevaluate_mpi`` (MPI) use
these so the two execution paths score solutions identically.

Why this exists
---------------
Comparing the objective scores an MOEA produced for *different* search
ensembles is invalid — each arm's scores are computed on its own ensemble. The
only sound cross-design comparison re-evaluates every arm's Pareto **policies**
(decision-variable vectors) on ONE common ensemble and compares those scores.

Flexibility
-----------
The common ensemble is whatever ``config.REEVAL_ENSEMBLE_SPEC`` resolves to,
selected by the ``NYCOPT_REEVAL_ENSEMBLE_PRESET`` env var. It can be ANY
registered preset, a ``kn_{Y}yr_n{N}`` slug, or any staged ensemble directory
carrying a ``_meta.json`` (see ``src.ensembles.get_ensemble_spec``). Swap the
re-eval ensemble by changing that one env var — nothing else. Outputs are
written under a per-ensemble subdir (``reeval/{tag}/``) so re-evals on different
common ensembles never clobber each other.

The objective *set* is resolved against the re-eval ensemble (ensemble
satisficing objectives when it ``is_ensemble``, else the single-trace set),
using the same objective names (``config.ACTIVE_OBJECTIVES``) for every arm —
so scores are directly comparable across arms.
"""
from __future__ import annotations

import re

# Lazily-built, process-local cache of (objective_set, ensemble_spec, is_ensemble)
# for the default (config-driven) re-eval target. Avoids rebuilding per solution
# and sidesteps pickling the objective set into multiprocessing workers (each
# spawned worker re-imports config and rebuilds from inherited env vars).
_REEVAL_CACHE = None


def resolve_reeval(objectives=None, reeval_spec=None):
    """Resolve the (objective_set, ensemble_spec, is_ensemble) for re-eval.

    With no args, reads ``config.REEVAL_ENSEMBLE_SPEC`` and
    ``config.ACTIVE_OBJECTIVES`` and caches the result. Pass explicit
    ``objectives`` / ``reeval_spec`` to override (not cached).
    """
    global _REEVAL_CACHE
    if _REEVAL_CACHE is not None and objectives is None and reeval_spec is None:
        return _REEVAL_CACHE

    from config import REEVAL_ENSEMBLE_SPEC, ACTIVE_OBJECTIVES
    spec = reeval_spec if reeval_spec is not None else REEVAL_ENSEMBLE_SPEC
    names = objectives if objectives is not None else ACTIVE_OBJECTIVES

    is_ensemble = bool(spec is not None and spec.is_ensemble)
    if is_ensemble:
        from src.objectives_ensemble import build_ensemble_objective_set
        obj_set = build_ensemble_objective_set(names)
    else:
        from src.objectives import build_objective_set
        obj_set = build_objective_set(names)

    result = (obj_set, spec, is_ensemble)
    if objectives is None and reeval_spec is None:
        _REEVAL_CACHE = result
    return result


def reeval_tag(spec) -> str:
    """Filesystem-safe label for the re-eval ensemble (its preset name)."""
    name = getattr(spec, "preset_name", None) or "historic_single"
    return re.sub(r"[^A-Za-z0-9_.-]", "_", str(name))


def reeval_output_dir(scenario: str, slug: str, spec, seed=None):
    """``outputs/{scenario}/{slug}/reeval/{reeval_tag}[/seed_NN]`` (created)."""
    from config import run_output_dir
    d = run_output_dir(scenario, slug, "reeval") / reeval_tag(spec)
    if seed is not None:
        d = d / f"seed_{seed:02d}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def reeval_obj_names() -> list:
    """Objective names for the re-eval objective set (CSV columns)."""
    obj_set, _, _ = resolve_reeval()
    return list(obj_set.names)


def evaluate_solution(solution_id: int, dv_vector, formulation: str,
                      out_path=None):
    """Re-evaluate one policy on the common re-eval ensemble.

    Ensemble re-eval reuses the search-path ``evaluate()`` (in-memory ensemble
    simulation + the identical objective aggregation), so re-eval and search
    compute objectives byte-for-byte the same way — only the ensemble differs.
    Returns objectives in **natural units** (not Borg-minimized): satisficing
    fractions in [0, 1] for the ensemble path, raw metric values otherwise.

    Single-trace re-eval keeps the legacy per-solution HDF5 path. Ensemble
    re-eval is in-memory (no giant per-solution HDF5s) and ignores ``out_path``.

    Returns:
        (solution_id, natural_objectives | None, error_message | None)
    """
    try:
        obj_set, spec, is_ensemble = resolve_reeval()
        if is_ensemble:
            from src.simulation import evaluate
            borg = evaluate(dv_vector, formulation_name=formulation,
                            objective_set=obj_set, ensemble_spec=spec)
            # Borg minimizes; undo the sign flip to recover natural values.
            dirs = obj_set.directions
            natural = [(-v if d == 1 else v) for v, d in zip(borg, dirs)]
        else:
            from src.simulation import dvs_to_config, run_simulation_to_disk
            cfg = dvs_to_config(dv_vector, formulation)
            data = run_simulation_to_disk(cfg, out_path)
            natural = list(obj_set.compute(data))
        return solution_id, [float(x) for x in natural], None
    except Exception as e:
        return solution_id, None, f"{type(e).__name__}: {e}"
