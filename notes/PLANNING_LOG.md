# NYCOptimization Planning Log

Internal session log for Claude-assisted development.
Each entry records what was done, decisions made, and issues encountered.

---

## Session 3 — 2026-03-06

**Focus:** Full code review, pywrdrb API verification against source, objectives refactoring.

### Code Review Completed

Reviewed all source files against pywrdrb `nyc_opt` branch source via MCP tools. Key findings documented in HANDOFF.md "Code Review Findings (2026-03-06)" section.

**API verification results:**
- `NYCOperationsConfig` method signatures: ALL CONFIRMED correct
- `update_mrf_baselines(cannonsville, pepacton, neversink, montague, trenton)`: correct
- `update_delivery_constraints(max_nyc_delivery, drought_factors_nyc[7], drought_factors_nj[7])`: correct
- `update_flood_limits(max_release_cannonsville, max_release_pepacton, max_release_neversink)`: correct
- `DROUGHT_LEVELS = ['level1a', 'level1b', 'level1c', 'level2', 'level3', 'level4', 'level5']`: confirmed
- Constants CSV keys match code expectations (e.g., `level1a_factor_delivery_nyc`)

**Critical discovery — default NYC drought factors:**
- L1a-L2 NYC factors = 1,000,000 (unconstrained), NOT 1.0
- L1a-L2 NJ factors = 1.0 (no reduction)
- The `.get(..., 1.0)` fallback in `_apply_ffmp_params` masks this difference

**Critical discovery — `mrf_factors_daily_df` initialization:**
- `from_defaults()` does `mrf_factors_daily_df = storage_zones_df.copy()` (line 80)
- Both loaded from same CSV: `ffmp_reservoir_operation_daily_profiles.csv`
- The CSV likely has BOTH storage zone rows and MRF factor rows indexed by `profile`
- `_apply_mrf_profile_scaling` modifies this DataFrame with seasonal scaling

**DV count corrected:** 25 DVs, not 19 as previously documented.

**Objective metric discrepancy found:** `nyc_max_deficit_pct` implements max monthly shortage %, but HANDOFF described it as max consecutive shortfall days.

### Objectives Refactored

Rewrote `src/objectives.py` with class-based architecture:
- `Objective` class: wraps a metric function with name, direction, epsilon, description
- `ObjectiveSet` class: ordered collection with `compute()`, `compute_for_borg()`, `names`, `epsilons`, `directions`
- Pre-built objective instances with verbose names (`obj_nyc_supply_reliability_daily`, `obj_flood_risk_storage_spill_days`, etc.)
- Pre-built objective sets: `DEFAULT_OBJECTIVES`, `DROUGHT_DURATION_OBJECTIVES`, `DOWNSTREAM_FLOOD_OBJECTIVES`, `COMPREHENSIVE_OBJECTIVES`, `COMPACT_OBJECTIVES`
- All callers updated to use `DEFAULT_OBJECTIVES.compute(data)` directly (no backward-compat wrappers)

New alternative metrics added:
- `nyc_max_consecutive_shortfall`: max streak of shortfall days (alternative to `nyc_max_deficit_pct`)
- `flood_days_downstream`: Montague flow exceedance (alternative to storage-proxy `flood_days`)
- `avg_storage_pct`: average storage fraction
- `montague_deficit_mgd`: average Montague shortfall on violation days

Updated `config.py`:
- Removed `OBJECTIVES` OrderedDict (replaced by `ACTIVE_OBJECTIVE_SET` selector)
- `get_n_objs()`, `get_epsilons()`, `get_obj_names()`, `get_obj_directions()` now delegate to `ObjectiveSet`
- Added `get_objective_set()` helper

Updated `src/simulation.py`:
- `evaluate()` accepts optional `objective_set` parameter

### Notes Created

- `notes/objectives.md`: Mathematical definitions of all objective functions with LaTeX notation, epsilon rationale, regulatory context, and open questions
- `notes/decision_vars.md`: All DVs for Formulations A/B/C with pywrdrb method mapping, regulatory context, bound rationale, flat vector indexing, and RBF/MLP architecture sizing

### Session Outcome

Code review complete with 14 findings (5 critical, 5 medium, 4 minor). Objectives refactored to class-based architecture supporting swappable objective sets. Two living-document notes created for objectives and decision variables. All backward-compatible APIs preserved.

---

## Session 2 — 2026-02-27

**Focus:** Repo scaffolding, pywrdrb API verification, simulation pipeline implementation, presim handling.

### Work Completed

**Repo structure finalized:**
- Numbered bash workflow: `00_generate_presim.sh`, `01_run_baseline.sh`, `02_run_mmborg.sh`, `03_run_diagnostics.sh`, `04_plot_diagnostics.sh`, `05_reevaluate.sh`
- Each is a thin wrapper calling `scripts/<name>.py`
- `src/mmborg_cli.py` created to separate argparse from `mmborg.py`

**pywrdrb API verified (nyc_opt branch, read from source):**
- `ModelBuilder.make_model()` populates `mb.model_dict` (not `mb.model`)
- Must write JSON then `pywrdrb.Model.load(json_path)` — no dict constructor
- `NYCOperationsConfig` attributes: `storage_zones_df`, `mrf_factors_daily_df`
- `update_mrf_baselines(cannonsville, pepacton, neversink, montague, trenton)`
- `update_delivery_constraints(max_nyc_delivery, drought_factors_nyc[7], drought_factors_nj[7])`
- `update_flood_limits(max_release_cannonsville, max_release_pepacton, max_release_neversink)`
- `OutputRecorder.recorder_dict[key].data` → numpy array `[timesteps, scenarios]`

**Critical bugs fixed in `src/simulation.py`:**
- `mb.model` → `mb.model_dict`
- `pywrdrb.Model(model_dict)` → write JSON + `Model.load(json)`
- `config.storage_zones` → `config.storage_zones_df`
- `config.mrf_daily_factors` → `config.mrf_factors_daily_df`
- `update_mrf_baselines(delMontague=...)` → `montague=...`
- `update_delivery_constraints(nyc_drought_factor_L3=...)` → `drought_factors_nyc=np.array([7 vals])`
- `update_flood_limits(cannonsville=...)` → `max_release_cannonsville=...`

**New components created:**
- `InMemoryRecorder` class (wraps OutputRecorder, overrides `finish()` to skip HDF5)
- `_extract_results_from_recorder()` (maps recorder keys to standard pywrdrb names)
- `scripts/generate_presim.py` (one-time presim generation)
- `scripts/run_baseline.py` (proper Python baseline runner)
- `tests/test_simulation_api.py` (full API test — run locally with pywrdrb installed)

**Presim resolution:**
- Error occurred because `use_trimmed_model=True` requires presim CSV
- Fix: `run_baseline.py` defaults to `use_trimmed=False` (full model, single run OK)
- `_require_presim_file()` raises clear `FileNotFoundError` pointing to `00_generate_presim.sh`
- Borg evaluations still use trimmed model — requires presim to be generated first

**README updates:**
- Blog URLs corrected: `wpcomstaging.com` → `waterprogramming.wordpress.com`
- Added Aug 2025 WaterProgramming post on MM Borg + MOEAFramework 5.0

### Open Issues / Unknowns

1. **`InMemoryRecorder.__init__`** — Pywr `Recorder.__init__` registers the recorder with the model. The current subclass pattern should work, but needs runtime confirmation.
2. **`_apply_zone_shifts` column indexing** — `storage_zones_df` has `doy` + date columns. Filter `[c for c in zones.columns if c != "doy"]` needs runtime confirmation.
3. **`_apply_mrf_profile_scaling` DOY indexing** — `date_cols[d-1]` mapping needs confirmation against actual CSV structure.
4. **pywrdrb `nyc_opt` install** — sandbox has no network access; all tests must run on user's local machine.

### Session Outcome

Phase 1 infrastructure is complete on paper. All scripts exist and are logically consistent. End-to-end validation requires running on a machine with pywrdrb installed.

---

## Session 1 — (Pre-compaction, Feb 2026)

**Focus:** Initial repo setup, config.py, objective functions, Borg wrapper scaffolding.

Summary only (details in compacted context):
- Created `config.py`, `src/objectives.py`, `src/mmborg.py`, `src/diagnostics.py`
- Set up `src/plotting/`, `src/load/` modules
- Defined Formulation A (ffmp) decision variables and bounds
- Scaffolded bash workflow (pre-numbered)
- Initial `src/simulation.py` stub (API not yet verified against source)
