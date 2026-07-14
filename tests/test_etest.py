"""
test_etest.py - The held-out test ensemble E_test: construction contract and SOW grouping.

E_test is the measuring stick. Three things about it are load-bearing and would fail SILENTLY if
broken, which is why they are asserted here rather than left to review:

1. **It is LHS over the FULL DU range, with R > 1 realizations per LHS point.** The i.i.d.
   requirement of ``src/scenario_designs.py`` belongs to the SEARCH-side candidate pools, which are
   subsampled; E_test is never subsampled and is never a control, so it is exempt. If someone
   "fixes" E_test to be i.i.d. with R = 1, nothing breaks except that the SOW-unit robustness
   metric quietly becomes the realization-unit metric.

2. **The SOW grouping survives generation.** ``sow_ids`` must round-trip from the staged ensemble
   into ``reeval_raw_meta.json``, or the SOW unit (Herman 2014; Trindade 2017; Gold 2022) cannot be
   computed offline from the persisted cube — which is the whole point of persisting the cube.

3. **Its seed stream is disjoint from every search ensemble's.** Otherwise the held-out
   re-evaluation is not held out (Bonham et al. 2024).

Staging here is a real ``_meta.json`` + ``forcing_profiles.npz`` pair (the artifacts the pipeline
actually reads) rather than a full generation, which would take minutes of Kirsch fitting to test a
metadata contract.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

import config
from src.etest import (
    E_TEST_DEFAULT_VARIANT,
    E_TEST_VARIANTS,
    ETestVariant,
    assert_etest_contract,
    assert_staged_etest_contract,
    campaign_etest_variant,
    get_etest_variant,
    list_etest_variants,
)
from src.ensembles import get_ensemble_spec
from src.reeval_core import sow_grouping
from src.scenario_designs import SCENARIO_DESIGNS, SCENARIO_YEARS, assert_iid_pools


# ---------------------------------------------------------------------------
# Staging helpers
# ---------------------------------------------------------------------------

def _stage_etest(root: Path, slug: str, *, n_theta: int, r: int, years: int = 10,
                 theta_sampler: str = "lhs", seed_domain: str = "etest:kn",
                 population: str = "du_forced", with_npz: bool = True) -> Path:
    """Stage the two artifacts an E_test directory is resolved and grouped from."""
    d = root / slug
    d.mkdir(parents=True, exist_ok=True)
    n = n_theta * r
    (d / "_meta.json").write_text(json.dumps({
        "slug": slug,
        "kind": "forcing_pool",
        "population": population,
        "generator": "kn",
        "seed_domain": seed_domain,
        "theta_sampler": theta_sampler if population == "du_forced" else None,
        "n_realizations": n,
        "n_forcing_profiles": n_theta,
        "realizations_per_profile": r,
        "realization_years": years,
        "n_years": years,
    }))
    if with_npz and population == "du_forced":
        np.savez(
            d / "forcing_profiles.npz",
            realization_ids=np.arange(n),
            theta_params=np.repeat(np.arange(n_theta * 3, dtype=float).reshape(n_theta, 3),
                                   r, axis=0),
            theta_param_names=np.array(["m", "r1", "r2"]),
            theta_sampler=theta_sampler,
            n_forcing_profiles=n_theta,
            realizations_per_profile=r,
        )
    return d


# ---------------------------------------------------------------------------
# Registry + construction contract
# ---------------------------------------------------------------------------

def test_registry_resolves_and_lists():
    assert list_etest_variants() == sorted(E_TEST_VARIANTS)
    for name in E_TEST_VARIANTS:
        v = get_etest_variant(name)
        assert isinstance(v, ETestVariant)
        assert v.name == name
    with pytest.raises(KeyError):
        get_etest_variant("does_not_exist")


def test_exactly_one_campaign_variant_and_it_is_the_default():
    """The campaign has ONE measuring stick; every other variant is opt-in."""
    v = campaign_etest_variant()
    assert v.name == E_TEST_DEFAULT_VARIANT == "kn"
    assert v.generator == "kn"
    assert [n for n, x in E_TEST_VARIANTS.items() if not x.campaign] == ["hmm"]


def test_construction_contract_holds():
    assert_etest_contract()  # the import-time assertion, exercised explicitly


def test_every_variant_replicates_within_each_sow():
    """R_test > 1 is what makes the SOW unit a DIFFERENT quantity from the realization unit.

    With R = 1 each SOW holds one realization, the within-SOW collapse is the identity, and
    ``sat_multivariate_sow`` silently equals ``sat_multivariate``.
    """
    for name, v in E_TEST_VARIANTS.items():
        assert v.realizations_per_theta > 1, f"'{name}' has R_test == 1"
        assert v.n_realizations == v.n_theta * v.realizations_per_theta


def test_test_records_are_at_least_as_long_as_search_records():
    for name, v in E_TEST_VARIANTS.items():
        assert v.realization_years >= SCENARIO_YEARS, f"'{name}' L_test < search L"


def test_etest_box_is_wider_than_the_search_box():
    """A measuring stick contained in the search box can only reward interpolation."""
    s_lo, s_hi = config.ENSEMBLE_FORCING_BOUND_PCT
    for name, v in E_TEST_VARIANTS.items():
        e_lo, e_hi = v.bound_pct
        assert e_lo <= s_lo and e_hi >= s_hi, f"'{name}' DU percentile box is not wider"
        assert v.margin >= config.ENSEMBLE_FORCING_MARGIN, f"'{name}' has no extra margin"


def test_chunks_never_split_a_sow():
    for name, v in E_TEST_VARIANTS.items():
        if v.chunk_size:
            assert v.chunk_size % v.realizations_per_theta == 0, f"'{name}' splits a SOW"


def test_seed_domains_are_reserved_and_disjoint_from_every_design():
    """If E_test shared a seed stream with a search ensemble it would not be held out."""
    search_domains = {d.seed_domain for d in SCENARIO_DESIGNS.values() if d.seed_domain}
    seeds = {}
    for name, v in E_TEST_VARIANTS.items():
        assert v.seed_domain.startswith("etest:")
        assert v.seed_domain not in search_domains
        assert v.seed not in seeds, f"'{name}' collides with '{seeds.get(v.seed)}'"
        seeds[v.seed] = name


def test_etest_is_not_a_scenario_design_and_is_exempt_from_the_iid_invariant():
    """E_test is an EnsembleSpec, never a ScenarioDesign -- so ``assert_iid_pools`` (which
    forbids LHS + R>1) does not, and must not, reach it. It is never subsampled and is never
    a control, so nothing about it requires i.i.d. sampling."""
    assert not any(n.startswith("etest") for n in SCENARIO_DESIGNS)
    assert_iid_pools()  # still passes while E_test is registered as LHS with R > 1
    assert all(v.realizations_per_theta > 1 for v in E_TEST_VARIANTS.values())


# ---------------------------------------------------------------------------
# The generator config the script actually builds
# ---------------------------------------------------------------------------

def test_etest_config_is_lhs_replicated_chunked_and_hazard_imaged():
    from scripts.main.generate_test_ensemble import etest_config

    v = campaign_etest_variant()
    cfg = etest_config(v)
    assert cfg.theta_sampler == "lhs"          # never subsampled -> LHS is correct
    assert cfg.population == "du_forced"
    assert cfg.realizations_per_profile > 1    # the SOW replication
    assert cfg.compute_hazard_image is True    # step 11 hard-fails without it
    assert cfg.chunk_size > 0                  # E_test is the biggest ensemble in the study
    assert cfg.store_daily is True
    assert cfg.seed_domain == v.seed_domain
    assert cfg.generator == v.generator
    assert cfg.n_realizations == v.n_realizations


# ---------------------------------------------------------------------------
# The STAGED contract (what the generator actually wrote)
# ---------------------------------------------------------------------------

def test_staged_meta_records_lhs_and_replication(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "STAGED_ENSEMBLE_DIR", tmp_path)
    _stage_etest(tmp_path, "etest_kn_10yr_n40", n_theta=10, r=4)
    meta = assert_staged_etest_contract("etest_kn_10yr_n40")
    assert meta["theta_sampler"] == "lhs"
    assert meta["realizations_per_profile"] > 1


@pytest.mark.parametrize("bad", [
    {"theta_sampler": "iid"},          # the search-side pool construction, not E_test's
    {"r": 1},                          # no within-SOW sample -> SOW metric undefined
    {"seed_domain": "fixed"},          # a SEARCH seed stream -> not held out
])
def test_staged_contract_rejects_a_search_side_construction(bad, tmp_path, monkeypatch):
    monkeypatch.setattr(config, "STAGED_ENSEMBLE_DIR", tmp_path)
    kwargs = {"n_theta": 10, "r": 4} | bad
    _stage_etest(tmp_path, "etest_kn_10yr_n40", **kwargs)
    with pytest.raises(AssertionError):
        assert_staged_etest_contract("etest_kn_10yr_n40")


def test_staged_contract_raises_when_not_staged(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "STAGED_ENSEMBLE_DIR", tmp_path)
    with pytest.raises(FileNotFoundError):
        assert_staged_etest_contract("etest_kn_10yr_n40")


# ---------------------------------------------------------------------------
# SOW grouping: the round trip that makes the SOW-unit metric computable
# ---------------------------------------------------------------------------

def test_sow_grouping_round_trips_from_the_staged_ensemble(tmp_path, monkeypatch):
    """realization k belongs to SOW k // R, recovered from what generation persisted."""
    monkeypatch.setattr(config, "STAGED_ENSEMBLE_DIR", tmp_path)
    _stage_etest(tmp_path, "etest_kn_10yr_n12", n_theta=3, r=4)
    spec = get_ensemble_spec("etest_kn_10yr_n12")
    assert spec.n_realizations == 12

    sow_ids, n_sow, r_per_sow = sow_grouping(spec, spec.realization_indices)
    assert sow_ids == [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]
    assert n_sow == 3
    assert r_per_sow == 4


def test_sow_grouping_falls_back_to_meta_without_the_npz(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "STAGED_ENSEMBLE_DIR", tmp_path)
    _stage_etest(tmp_path, "etest_kn_10yr_n12", n_theta=3, r=4, with_npz=False)
    spec = get_ensemble_spec("etest_kn_10yr_n12")
    sow_ids, n_sow, r_per_sow = sow_grouping(spec, spec.realization_indices)
    assert (n_sow, r_per_sow) == (3, 4)
    assert sow_ids[:5] == [0, 0, 0, 0, 1]


def test_no_forcing_profiles_means_no_grouping_not_a_fabricated_one(tmp_path, monkeypatch):
    """A stationary ensemble has no SOWs. Inventing one (e.g. one realization each) would
    make the SOW metric silently equal the realization metric."""
    monkeypatch.setattr(config, "STAGED_ENSEMBLE_DIR", tmp_path)
    _stage_etest(tmp_path, "fixprob_10yr_n8_d0", n_theta=8, r=1,
                 population="stationary", seed_domain="fixed", with_npz=False)
    spec = get_ensemble_spec("fixprob_10yr_n8_d0")
    assert sow_grouping(spec, spec.realization_indices) == (None, None, None)


def test_single_trace_spec_has_no_grouping():
    spec = get_ensemble_spec("historic_single")
    assert sow_grouping(spec, [0]) == (None, None, None)


# ---------------------------------------------------------------------------
# The selection-bias guard
# ---------------------------------------------------------------------------

def test_seed_domain_guard_fires_on_a_shared_stream(tmp_path, monkeypatch):
    """Pointing the re-eval preset at an ensemble from a SEARCH seed stream must HARD-ERROR."""
    monkeypatch.setattr(config, "STAGED_ENSEMBLE_DIR", tmp_path)
    _stage_etest(tmp_path, "fixprob_10yr_n8_d0", n_theta=8, r=1, population="stationary",
                 seed_domain="fixed", with_npz=False)
    _stage_etest(tmp_path, "fixprob_10yr_n8_d1", n_theta=8, r=1, population="stationary",
                 seed_domain="fixed", with_npz=False)
    search = get_ensemble_spec("fixprob_10yr_n8_d0")
    same_stream = get_ensemble_spec("fixprob_10yr_n8_d1")
    with pytest.raises(RuntimeError, match="seed domain"):
        config.assert_search_test_seed_domains_disjoint(search, same_stream)


def test_seed_domain_guard_passes_for_etest(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "STAGED_ENSEMBLE_DIR", tmp_path)
    _stage_etest(tmp_path, "fixprob_10yr_n8_d0", n_theta=8, r=1, population="stationary",
                 seed_domain="fixed", with_npz=False)
    _stage_etest(tmp_path, "etest_kn_10yr_n12", n_theta=3, r=4, seed_domain="etest:kn")
    config.assert_search_test_seed_domains_disjoint(
        get_ensemble_spec("fixprob_10yr_n8_d0"), get_ensemble_spec("etest_kn_10yr_n12"),
    )  # does not raise
