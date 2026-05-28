"""
tests/test_ensembles.py - Unit tests for the ensemble-evaluation registry.

Covers:
  1. EnsembleSpec immutability and derived properties.
  2. PRESETS registry contents (v1 ships three: historic_single,
     wcu_kirsch_n5, reeval_wcu_kirsch_n300).
  3. get_ensemble_spec resolver behavior (hit / miss).
  4. with_indices_override returns a new spec without mutating the original.
  5. derive_slug integration: ensemble fragment is inserted only when the
     active search preset is an ensemble (legacy slugs preserved).

Run:
    venv/Scripts/python -m pytest tests/test_ensembles.py -v
"""

import os
import subprocess
import sys
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

PROJECT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from src.ensembles import (
    EnsembleSpec,
    PRESETS,
    get_ensemble_spec,
    list_presets,
    with_indices_override,
)


# ---------------------------------------------------------------------------
# EnsembleSpec
# ---------------------------------------------------------------------------

def test_spec_is_frozen():
    spec = EnsembleSpec(
        preset_name="x",
        inflow_type="y",
        realization_indices=(0, 1),
    )
    with pytest.raises(FrozenInstanceError):
        spec.preset_name = "z"  # type: ignore[misc]


def test_spec_n_realizations():
    spec = EnsembleSpec(
        preset_name="x",
        inflow_type="y",
        realization_indices=tuple(range(7)),
    )
    assert spec.n_realizations == 7


def test_du_factor_signature_empty_by_default():
    spec = EnsembleSpec(
        preset_name="x",
        inflow_type="y",
        realization_indices=(0,),
    )
    assert spec.du_factor_signature == ""


def test_du_factor_signature_is_sorted_deterministic():
    spec = EnsembleSpec(
        preset_name="x",
        inflow_type="y",
        realization_indices=(0,),
        du_factors={"b": 2, "a": 1},
    )
    # Sorted by key: a then b.
    assert spec.du_factor_signature == "a=1|b=2"


# ---------------------------------------------------------------------------
# PRESETS registry
# ---------------------------------------------------------------------------

def test_v1_presets_present():
    expected = {"historic_single", "wcu_kirsch_n5", "reeval_wcu_kirsch_n300"}
    assert expected.issubset(set(PRESETS))


def test_historic_single_is_legacy_passthrough():
    spec = PRESETS["historic_single"]
    assert spec.is_ensemble is False
    assert spec.realization_indices == (0,)
    assert spec.slug_fragment == "", "historic_single must have empty slug fragment to preserve legacy paths"
    assert spec.inflow_type == "pub_nhmv10_BC_withObsScaled"


def test_wcu_kirsch_n5_shape():
    spec = PRESETS["wcu_kirsch_n5"]
    assert spec.is_ensemble is True
    assert spec.n_realizations == 5
    assert spec.slug_fragment == "wcu5"
    assert spec.source_kind == "synhydro_kn"


def test_reeval_wcu_kirsch_n300_shape():
    spec = PRESETS["reeval_wcu_kirsch_n300"]
    assert spec.is_ensemble is True
    assert spec.n_realizations == 300
    assert spec.slug_fragment == "reeval_wcu300"
    # Independent seed from search preset (selection-bias guard).
    assert spec.seed != PRESETS["wcu_kirsch_n5"].seed


# ---------------------------------------------------------------------------
# Resolver helpers
# ---------------------------------------------------------------------------

def test_get_ensemble_spec_hit():
    spec = get_ensemble_spec("historic_single")
    assert spec is PRESETS["historic_single"]


def test_get_ensemble_spec_miss_raises_keyerror():
    with pytest.raises(KeyError) as excinfo:
        get_ensemble_spec("nope_no_such_preset")
    assert "Available presets" in str(excinfo.value)


def test_list_presets_is_sorted():
    presets = list_presets()
    assert presets == sorted(presets)


def test_with_indices_override_does_not_mutate_original():
    original = PRESETS["wcu_kirsch_n5"]
    overridden = with_indices_override(original, [0, 2, 4])
    assert overridden.realization_indices == (0, 2, 4)
    assert overridden.n_realizations == 3
    # Original is untouched.
    assert original.realization_indices == tuple(range(5))
    # Other fields carry over.
    assert overridden.preset_name == original.preset_name
    assert overridden.slug_fragment == original.slug_fragment


# ---------------------------------------------------------------------------
# derive_slug integration (subprocess so env vars are read at config import)
# ---------------------------------------------------------------------------

def _slug_with_env(env_overrides: dict) -> str:
    """Spawn a subprocess that imports config and prints derive_slug('ffmp')."""
    env = os.environ.copy()
    # Strip any pre-existing NYCOPT_* state that could pollute the test.
    for k in list(env):
        if k.startswith("NYCOPT_") or k in ("RUN_SLUG", "RUN_SLUG_TAG"):
            env.pop(k, None)
    env.update({k: str(v) for k, v in env_overrides.items()})
    code = (
        "import sys; sys.path.insert(0, '.'); "
        "from config import derive_slug; "
        "print(derive_slug('ffmp'))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(PROJECT_DIR),
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, (
        f"subprocess failed:\nstdout={result.stdout}\nstderr={result.stderr}"
    )
    # Last non-empty line is the slug; earlier lines may be config-import warnings.
    lines = [ln for ln in result.stdout.splitlines() if ln.strip()]
    return lines[-1]


@pytest.mark.slow
def test_derive_slug_legacy_unchanged():
    """historic_single (default) emits no ensemble fragment."""
    slug = _slug_with_env({})
    # Default config: ffmp_obj7 (no T/S, no sfdv, no state mismatch, no ensemble fragment)
    assert slug.startswith("ffmp_obj7")
    assert "wcu" not in slug


@pytest.mark.slow
def test_derive_slug_wcu_fragment_appears():
    slug = _slug_with_env({"NYCOPT_ENSEMBLE_PRESET": "wcu_kirsch_n5"})
    assert "wcu5" in slug


@pytest.mark.slow
def test_derive_slug_indices_override():
    """NYCOPT_ENSEMBLE_INDICES subsets the active search preset."""
    env = {
        "NYCOPT_ENSEMBLE_PRESET": "wcu_kirsch_n5",
        "NYCOPT_ENSEMBLE_INDICES": "0,2",
    }
    code = (
        "import sys; sys.path.insert(0, '.'); "
        "from config import SEARCH_ENSEMBLE_SPEC; "
        "print(SEARCH_ENSEMBLE_SPEC.realization_indices)"
    )
    full_env = os.environ.copy()
    for k in list(full_env):
        if k.startswith("NYCOPT_"):
            full_env.pop(k, None)
    full_env.update(env)
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(PROJECT_DIR),
        env=full_env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    last = [ln for ln in result.stdout.splitlines() if ln.strip()][-1]
    assert last == "(0, 2)"
