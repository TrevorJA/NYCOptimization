# Claude Handoff Document — NYCOptimization

**Project:** Multi-objective optimization of NYC reservoir operations (PhD dissertation chapter)
**Owner:** Trevor Amestoy, Cornell CEE / Reed Research Group
**Date:** 2026-03-06 (updated from 2026-02-27)
**Repo root:** `Pywr-DRB/NYCOptimization/`

---

## Current State

Phase 1 infrastructure is complete. All scripts exist and are logically consistent. **Nothing has been run end-to-end yet** — pywrdrb is not installed in the Claude sandbox. All execution must happen on Trevor's machine.

### Workflow Sequence (numbered bash scripts)

```
00_generate_presim.sh   # One-time setup. Run FIRST. ~5-10 min.
01_run_baseline.sh      # Historic FFMP baseline. Full model. ~10-30 min.
02_run_mmborg.sh        # Borg optimization. Trimmed model. Hours on HPC.
03_run_diagnostics.sh   # MOEA diagnostics (hypervolume, epsilon progress).
04_plot_diagnostics.sh  # Generate diagnostic plots.
05_reevaluate.sh        # Re-evaluate Pareto set under stochastic conditions.
```

Each script is a thin bash wrapper calling `scripts/<name>.py`.

---

## Immediate TODOs (in order)

1. **Run `python tests/test_simulation_api.py`** — validates the full pywrdrb API pipeline locally. Uses `use_trimmed=False` so no presim needed. Fix any errors before proceeding.

2. **Run `bash 00_generate_presim.sh`** — generates `outputs/presim/presimulated_releases_mgd.csv`. Required for Borg evaluations. Takes ~5-10 min (full 1945-2022 simulation).

3. **Run `bash 01_run_baseline.sh`** — runs full-model historic baseline, saves HDF5 + objectives CSV to `outputs/baseline/`. Verify all 6 objective values are reasonable.

4. **Compile Borg** — place `borgmm.c` and `mt19937ar.c` in `borg/` and run:
   ```bash
   mpicc -shared -fPIC -O3 -o borg/libborgmm.so borg/borgmm.c borg/mt19937ar.c -lm
   ```
   Then test with `bash 02_run_mmborg.sh --nfe 1000 --seeds 1` locally before HPC submission.

5. **HPC submission** — use `02_submit_mmborg.slurm` for Anvil. Adjust walltime/nodes based on local timing.

---

## Key Files

| File | Purpose |
|------|---------|
| `config.py` | All constants: dates, paths, DV bounds, objective specs |
| `src/simulation.py` | `dvs_to_config()`, `run_simulation_inmemory()`, `run_simulation_to_disk()` |
| `src/objectives.py` | `DEFAULT_OBJECTIVES.compute(data)` → list of 6 floats |
| `src/mmborg.py` | MM Borg wrapper: `BorgOptimizer` class |
| `src/mmborg_cli.py` | CLI entry point for `02_run_mmborg.sh` |
| `src/diagnostics.py` | Post-optimization analysis using MOEAFramework |
| `scripts/generate_presim.py` | One-time presim generation |
| `scripts/run_baseline.py` | Baseline evaluation |
| `tests/test_simulation_api.py` | Full pipeline test (run locally) |

---

## pywrdrb API — Verified Patterns

These were confirmed by reading the `nyc_opt` branch source directly.

```python
# Build model
mb = pywrdrb.ModelBuilder(inflow_type=..., start_date=..., end_date=..., options={...})
mb.make_model()                          # populates mb.model_dict (not mb.model)
mb.write_model(str(json_path))           # must write JSON first
model = pywrdrb.Model.load(str(json_path))  # then load

# Run with recorder
recorder = pywrdrb.OutputRecorder(model=model, output_filename=str(hdf5_path))
model.run()
# recorder.recorder_dict[key].data  → numpy [timesteps, scenarios]

# NYCOperationsConfig
config = NYCOperationsConfig.from_defaults()
config.storage_zones_df               # not storage_zones
config.mrf_factors_daily_df           # not mrf_daily_factors
config.update_mrf_baselines(cannonsville=..., pepacton=..., neversink=...,
                             montague=..., trenton=...)  # not delMontague
config.update_delivery_constraints(max_nyc_delivery=...,
                                    drought_factors_nyc=np.array([7 vals]),
                                    drought_factors_nj=np.array([7 vals]))
config.update_flood_limits(max_release_cannonsville=...,
                            max_release_pepacton=...,
                            max_release_neversink=...)
```

---

## Code Review Findings (2026-03-06)

### Critical — FIXED

1. **~~`nyc_max_deficit_pct` metric contradicts documentation~~** — FIXED: Both metrics now exist as separate objectives with verbose names: `nyc_drought_max_monthly_deficit_pct` and `nyc_drought_max_consecutive_shortfall_days`. Multiple `ObjectiveSet` configurations include different combinations. No premature choice needed.

2. **~~`flood_days` is a crude proxy~~** — FIXED: Both flood metrics now exist: `flood_risk_storage_spill_days` (storage proxy) and `flood_risk_downstream_flow_days` (Montague flow > 25,000 CFS). The `comprehensive` objective set includes both; `default` and `downstream_flood` sets use one each.

3. **~~NYC L1a-L2 drought factor defaults are 1,000,000, not 1.0~~** — FIXED: Replaced `.get(..., 1.0)` with explicit `float(defaults["key"])` and added a `KeyError` check that validates all expected keys exist in `config.constants` before use. If the constants CSV fails to load, the error is now immediate and descriptive.

4. **~~`mrf_factors_daily_df` initialized from `storage_zones_df.copy()`~~** — FIXED: Added documentation in `_apply_mrf_profile_scaling()` explaining the pywrdrb design (both DFs from same CSV). Added a runtime warning if the DataFrame has fewer than 365 data columns, which would indicate a structural problem. Column filtering is now more defensive (excludes "doy", "profile", "type" metadata columns).

5. **`_extract_results_from_recorder` may miss data**: The recorder key matching uses prefix patterns (`"reservoir_"`, `"link_"`, `"demand_"`, `"delivery_"`) that are fragile. If pywrdrb recorder naming changes, data silently becomes empty DataFrames. Also does not extract `res_level` despite it being in `RESULTS_SETS`.

### Medium — Fix Before Optimization

6. **No constraints in Borg** (`n_constrs = 0`): STUDY_PLAN discusses 1954 Decree constraints and running-average diversion limits, but none are implemented. The optimizer can freely violate legal constraints.

7. **Ecological flow objective dropped**: STUDY_PLAN lists 7 objectives, only 6 implemented. No replacement metric for ecological stakeholders.

8. **`InMemoryRecorder` is fragile**: Uses `OutputRecorder.__new__()` bypass, `/dev/null` path (doesn't exist on Windows), and monkey-patches `finish()`. Will break with pywrdrb updates. Consider adding an `InMemoryRecorder` class to pywrdrb itself.

9. **No `__init__.py` for `src/`**: All imports use `sys.path.insert()` hacks. Should use proper package structure.

10. **`_apply_zone_shifts` "doy" filter is a no-op**: The `storage_zones_df` uses `index_col='profile'` with date string columns. There is no "doy" column, so the filter `[c for c in zones.columns if c != "doy"]` returns all columns. Harmless but misleading.

11. **Season boundary off-by-one in `_apply_mrf_profile_scaling`**: DOY 335 maps to Dec 1 in non-leap years but Nov 30 in leap years. Minor impact.

### Previously Listed (Status Updated)

12. **`InMemoryRecorder.__init__` registration** — Pywr `Recorder.__init__` called on `self._inner`. Should work but still needs runtime confirmation. (UNCHANGED)

13. **Trimmed model speedup** — Expected ~5-10s vs ~30s. Needs empirical confirmation. (UNCHANGED)

14. **DV count mismatch with docs** — Config has 25 DVs (5 MRF + 1 delivery + 3 NYC drought + 2 NJ drought + 6 zone shifts + 3 flood + 4 MRF seasonal). Previous HANDOFF said 19. Updated below.

---

## Formulation A (ffmp) Decision Variables

Defined in `config.py`. Currently: **25 DVs** total.
- MRF baselines: 5 (cannonsville, pepacton, neversink, montague, trenton)
- NYC max delivery: 1
- Drought factors NYC: 3 (L3, L4, L5 only; L1a-L2 unconstrained at 1,000,000)
- Drought factors NJ: 2 (L4, L5 only; L1a-L3 at defaults)
- Storage zone vertical shifts: 6 (level1b through level5)
- Flood release maximums: 3 (cannonsville, pepacton, neversink)
- MRF seasonal profile scaling: 4 (winter, spring, summer, fall)

Bounds, names, and baseline values all defined in `config.py` `FORMULATIONS["ffmp"]`.

---

## Objectives

Defined in `src/objectives.py` using the `Objective` class. Multiple `ObjectiveSet` configurations available.
See `notes/objectives.md` for full mathematical definitions.

### Available Objective Sets

| Set Name | Objectives | Notes |
|----------|-----------|-------|
| `default` | 6 | Monthly deficit, storage-proxy flood |
| `drought_duration` | 6 | Consecutive shortfall days instead of monthly deficit |
| `downstream_flood` | 6 | Montague flow exceedance instead of storage proxy |
| `comprehensive` | 8 | Both drought metrics + both flood metrics |
| `compact` | 4 | Quick diagnostic runs |

Select via `config.ACTIVE_OBJECTIVE_SET = "default"`.

### Default Set (6 objectives)

1. `nyc_supply_reliability_daily` — maximize (eps=0.005)
2. `nyc_drought_max_monthly_deficit_pct` — minimize (eps=1.0)
3. `montague_flow_reliability_daily` — maximize (eps=0.005)
4. `trenton_flow_reliability_seasonal` — maximize (eps=0.005)
5. `flood_risk_storage_spill_days` — minimize (eps=5.0)
6. `storage_min_combined_pct` — maximize (eps=0.5)

All directions normalized so Borg minimizes all. Signs handled in `ObjectiveSet.compute_for_borg()`.

---

## Phase 2 Readiness Checklist

Before starting Borg optimization runs:
- [ ] `test_simulation_api.py` passes locally
- [ ] `00_generate_presim.sh` complete (presim CSV exists)
- [ ] `01_run_baseline.sh` complete (baseline objectives look reasonable)
- [ ] Borg compiled (`borg/libborgmm.so` exists)
- [ ] Quick local Borg test passes (1000 NFE, 1 seed)
- [ ] HPC account and allocation confirmed (Anvil/ACCESS)
- [ ] Walltime estimate from local timing

---

## Notes on Repo History

- This repo was built from scratch in Feb 2026 as the optimization harness for Trevor's dissertation chapter.
- The `nyc_opt` branch of Pywr-DRB (not yet merged to main) provides the parameterized `NYCOperationsConfig` interface. This branch must be installed.
- Reference implementations: `NYCOperationExploration/` (sensitivity analysis, same pywrdrb patterns).
- MOEAFramework v5.0 API is used in `src/diagnostics.py` (`ResultFileSeedMerger`, `MetricsEvaluator`).
