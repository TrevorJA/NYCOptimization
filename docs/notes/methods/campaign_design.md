# Campaign Design at Scale

*The production campaign as it will be run, with its expected budget. Single-source constants: `src/scenario_designs.py` (N, L, draws), `src/moea_config.py::production` (islands, workers, NFE per seed, snapshot cadence), `src/etest.py` (E_test), `workflow/envs/*_production.env` (batch, τ, submit lines), `config.py` (memory model). Sizing evidence: `ensemble_size_diagnostics.md`; cost provenance: SI Text S8. Where this note and the code disagree, the code is the record.*

---

## 1. Designs

| Design | Search ensemble | Draws searched | Seeds | Role |
|---|---|---|---|---|
| `monte_carlo` (MC) | N = 300 i.i.d. realizations, L = 10 yr | 1 (d0) | 2 | exact control |
| `hazard_filling_stationary` (HF) | N = 300 selected from the d0 P = 10⁶ pool, L = 10 yr | 1 (d0) | 2 | proposed method |
| `historic` | one 78-yr record | 1 | 2 | prevailing-practice reference, matched NFE |

Two draws (d0, d1) are staged for both matched designs. The search runs on d0 only (K = 1). d1 is the replicate for the SI draw-sensitivity re-evaluation (§5). The unit of analysis is the seed, and the design comparison is conditional on one draw per design. Re-simulating each design's final set on d1 measures how far its objective estimates shift between draws. It does not show what policies a search on another draw would produce.

S = 2 is a floor. A third seed for both matched designs costs ~135–160k SU of search plus the re-evaluation of its policies and is not planned (§6).

## 2. NFE scheme

| Seed | NFE per island | Total NFE | Runtime snapshots (every 2,500) | Reported as |
|---|---|---|---|---|
| 1 | 187,500 | 750,000 | 75 | equal-NFE result = the snapshot at 125,000 per island (snapshot 50); the 750k tail is SI convergence evidence |
| 2 | 125,000 | 500,000 | 50 | final archive |
| 2 (`historic`, 2026-11-11) | 312,500 | 1,250,000 | 125 | equal-NFE result = the snapshot at 125,000 per island; the 1.25M tail is the convergence test below |

The campaign result for every design is the ε-nondominated merge of both seeds at 125,000 NFE per island (`scripts/main/extract_runtime_archive.py` builds `seed_01_{slug}_nfe125000.set` from the island runtime files and the equal-NFE `{slug}_merged_nfe125000.set`, installed as `{slug}_merged.set`). Convergence is read per seed from the coordinator archive: the controller writes no runtime file, so step 09 reconstructs its archive at every snapshot as the ε-box merge of the four island archives (the project's one ε-archive implementation, `src/sensitivity_common.py`; the final entry is checked against the seed's controller archive) and scores that series on total NFE; island archives are never read individually (Reed, 2026-10-23). If seed 1's coordinator hypervolume is not converged by 500,000 total NFE, seed 2 is extended to 187,500 per island by editing `max_evaluations_by_seed` before it is submitted. A design-scoped extension goes through `NYCOPT_MAX_EVALUATIONS_BY_SEED` in that design's env file instead, which leaves the shared `production` entry untouched. The `historic` seed-1 rerun of 2026-11-11 was still gaining coordinator hypervolume at 750k (about 2.5 % per 100k NFE after 500k, and 35 % of the 750k front in ε-boxes empty at 500k), so `historic` seed 2 runs to 312,500 per island (1.25M total) to locate the plateau before the reporting NFE is confirmed for every design; its 125,000-per-island snapshot remains the seed-2 equal-NFE set. Searches are NFE-bounded (no Borg maxTime); the SLURM wall is the only cap.

## 3. Geometry

| Item | Value |
|---|---|
| Nodes | 12 Anvil `wholenode` (1,536 cores) |
| Ranks | 1 controller + 4 islands × (382 workers + 1 master) = 1,533 (3 idle cores) |
| Ranks per node | 128 |
| Realization batch | `NYCOPT_SEARCH_REALIZATION_BATCH=150` in both matched env files (two model runs per evaluation); unset for `historic` |
| Estimated node RSS | 167 GB at N = 300 batched (envelope model 600 + 0.49 MB per simulated year per rank; 259 GB unbatched, above the 217 GB line) |
| Pre-flight | `nycopt_check_allocation` (ranks) and `nycopt_check_memory` (node RSS vs 85 % of 256 GB) both abort before Borg starts |
| `--time` | seed 1 96 h (matched), 8 h (`historic`); seed 2 72 h (matched), 12 h (`historic`, 1.25M NFE) |
| Resume | none. Runtime files are diagnostic archive dumps and the Borg checkpoint is disabled (race-prone across islands, never run). Every search must finish inside one job |

Twelve nodes are required because a 750k-NFE search at N = 300 is projected at 99 h on eight. Node scaling beyond eight nodes is unmeasured (single-island curve loses ~9 % per worker doubling; the one cross-node production pair shows +30 % SU per NFE per doubling, confounded with NFE), so every 12-node number below carries a factor g ∈ [1.00, 1.17]. The seed-1 job is itself the measurement. Its 125,000-per-island snapshot lands at 44–51 h on the measured basis and at 67–79 h even on the model basis, so the equal-NFE result is recoverable from a job killed at the wall.

## 4. Pre-search steps

Before the searches, E_test is generated over the mid-century box and staged (§5: the 500-SOW build, the demand-utilization column, the per-chunk scaled NYC demand series, the sub-window hazard image under the June 1 rules and the current-policy cube, about 0.2k SU inside the staging allowance), both matched designs are staged at N = 300 for draws d0 and d1 (workflow steps 02–04 on the P = 10⁶ pools), the step-06 current FFMP policy baselines are simulated on each d0 ensemble, and the batched-search memory smoke (`workflow/submit_search_memory_smoke.sh`) confirms node RSS and evaluation time on one node. The search ε vector [0.025, 5.0, 0.025, 5.0, 5.0, 0.3, 5.0, 0.025] is fixed (decision 2026-11-09): each entry is a practically significant difference in native units at or above its calibration floor, and the reported cardinality is set after the search by the grouped re-filter (`epsilon_calibration_experiment.md`).

## 5. E_test re-evaluation

| Item | Value |
|---|---|
| E_test | N_θ = 500 LHS SOWs over the full amplitude box of the 27 mid-century (2020 to 2059) CMIP6 projections widened by 25 % on every axis (annual-volume multiplier e^m = 0.899 to 1.28, no per-axis override; `config.ENSEMBLE_FORCING_PERIOD`, `src/etest.py::E_TEST_BOUND_PCT`, `E_TEST_MARGIN`) and, as a fourth Latin hypercube column, the NYC entitlement utilization u ∈ [0.45, 1.0] (row below) × R = 25 × L_test = 50 yr: 12,500 realizations, 625k simulated years, 25 staged chunks of 500 realizations (`etest_kn_50yr_n12500`). `src/etest.py::E_TEST_N_THETA` is the single source and `NYCOPT_REEVAL_ENSEMBLE_PRESET` names the slug in every step-10/11/12 submission. The one object every re-evaluation, every E_test diagnostic (sub-window hazard image, forcing profiles, dry envelope, hazard support, threshold placement) and every figure reads |
| Staging | generated directly at the registry size: a 500-point Latin hypercube over the box, 25 shards of one 500-realization chunk each, through the one chain `workflow/submit_etest_staging.sh all` (`staging/etest_shards.sh` → `etest_merge.sh` → `etest_demand_utilization.sh` → `etest_prep_chunks.sh` → `etest_realized_utilization.sh` → the hazard-image shards and merge); `assert_staged_etest_contract` refuses a staging generated with a per-axis override or without the box's `forcing_period` in its metadata. Every staging of this slug generated before 2026-11-12 holds different SOWs and is superseded, with every cube computed on it |
| Demand utilization u | one LHS value per SOW over [0.45, 1.0] (seed domain `etest:kn:demand`; `src/demand_utilization.py`, `forcing_parameterization.md` "Demand-utilization axis"), appended to the generated design by `scripts/supplemental/append_demand_utilization.py` without regenerating streamflow. Implemented as a multiplicative scale k(u) on each realization's regression-structured NYC demand series with the 800 MGD monthly-mean cap enforced after scaling (k from `outputs/reference/demand_utilization/demand_utilization_calibration.csv`: k = 0.8 → u = 0.55, 1.0 → 0.68, 1.5 → 0.90, 3.0 → 0.99, clipped at k_max = 6.0, u = 0.998). Staged per chunk by `staging/etest_prep_chunks.sh` from the generated design's `demand_utilization_nominal.csv`; realized u per SOW is reported as a diagnostic, never forced to nominal. Search ensembles stay at k = 1 (u ≈ 0.68) |
| Why 500 | the re-evaluation space has 4 factors (three climate amplitudes and the utilization u). Published re-evaluation ensembles are 10,000 LHS SOWs in 13-factor spaces (Herman et al. 2014; Trindade et al. 2017) and the 5-factor lake problem (Bartholomew & Kwakkel 2020), 1,000–2,000 at 5–14 factors (Eker & Kwakkel 2018; Hadjimichael et al. 2020; Gold et al. 2023), and 500 in the one 3-axis space with a measured convergence curve (Bonham et al. 2024), where satisficing rankings stabilize from 50–300 SOWs and regret-type metrics need 400–500. With precedents at 5–14 factors on 1,000–2,000 SOWs and at 3 factors on 500, 500 sits at that density for four factors and at the lower edge of the regret range, so the count stands |
| Why the dry bound | the box's lower bound is the widened mid-century CMIP6 minimum, e^m = 0.899. It reaches beyond the stationary drought envelope: at that bound 8.3 % of a SOW's decades lie beyond the stationary generator's 99th-percentile controlling-event drought magnitude (13 % on the magnitude summed over all events), against none at the driest mid-century run (0.953) and 1 % by construction (`scripts/supplemental/dry_envelope_run.py`, `dryenv_levels.csv`, 60 paired windows per level, SI Figure S18). Drier levels down to 0.754 (0.80 is the dry limit of Herman et al. 2014 and Trindade et al. 2017; at 0.799, 22 % and 53 % summed) are reported by the same diagnostic as a sensitivity (`forcing_parameterization.md`, "Box width"). The production confirmation is the dry-envelope production leg on the staged images |
| Cross-SOW precision | worst-case SE of a satisficing fraction 0.5/√500 = 2.2 pp; the RQ1 discrimination band δ = 2 × paired SE is measured on the production cubes |
| Per-SOW precision | 1,225 pooled annual units per SOW (R = 25). The measured per-SOW noise on E_test (paired floors 0.017–0.024 reliabilities, 1.2 pp deficit, 0.04 ft·d/yr flood, 3.0 pp storage) is below ε on every axis (0.48, 0.12, 0.13 and 0.60 of ε). Realizations within a SOW are independent, so R = 50 would scale every floor by 1/√2 (0.017, 0.85 pp, 0.028, 2.1 pp) and lower the current policy's share of SOWs misclassified at the adopted criteria from at most 1.1 pp (storage) to 0.7 pp, against the 2.2 pp cross-SOW standard error it leaves unchanged, for +66.5k SU (`scripts/supplemental/etest_realization_precision.py`, tables `etr_per_sow_noise.csv`, `etr_options.csv`; SI Table S1). Published re-evaluations use 1 to 1,000 realizations per SOW and none tests the count at a fixed SOW count, so R is set by this noise argument |
| Current FFMP policy baseline | simulated on E_test by step 10 with `NYCOPT_CHUNK_POLICIES=baseline`, once for all designs (the cube is design-independent: one policy on the common ensemble), after the per-chunk demand staging so the cube carries the per-SOW utilization scaling; the run lands in `outputs/baseline/reeval/{tag}/`, which every design's regret scoring resolves (`src.reeval_core.find_incumbent_cube`; one policy × 25 chunks, ~33 SU) |
| Path | step 10 chunked metrics-only re-evaluation through `workflow/submit_reeval_chunks.sh <design>`: `wholenode`, 128 single-threaded ranks per node, realization batch 25, then the chained step 11 merge; ~50k (policy, chunk) units at the cap. The one-policy incumbent cube runs on `shared`, one rank per chunk × 2 cpus (`submit_reeval_chunks.sh baseline`, ~75 SU) |
| Measured cost | 33 SU per policy (1.27–1.33 SU per policy-chunk unit × 25 chunks; ~80 min per unit per rank; measured at 128 ranks per node on 2026-08-13, jobs 19845823/19845825/19859233, node RSS 197 GB at batch 50; `ANVIL_NOTES.md`). The 16 ranks × 8 cpus `shared` layout recorded here until 2026-11-05 takes the same wall per unit and bills 8× |
| Policies | the equal-NFE merged set per design after the post-search ε re-filter; the total is capped at 2,000 (`epsilon_calibration_experiment.md` Stage 2); unmeasured at N = 300 and S = 2 |
| Cap | 2,000 policies (~66k SU). If the union exceeds it, the post-hoc ε re-filter is coarsened to the cardinality target and applied identically to every design |
| Stability check | θ-subsample (250 vs 500) ranking-stability curve scored offline from the persisted per-SOW matrix |
| Draw sensitivity (SI) | a thinned subset of each matched design's final set re-simulated on its own d1 at N = 300 (~70 SU staging per design, ~66 SU per 100 policies), sized to 250 SU. Paired per-policy shifts are reported against ε with no variance estimate |

## 6. Budget

Measured basis: 21,850 SU and 21.3 h per N = 100 / 500k-NFE search on 8 × 128 (two production runs), scaled by (N/100)^0.951 (fitted N = 10–200, r² 0.999; ±8 % at N = 300) and NFE/500k, with a ×1.09 batch penalty (upper bound, measured at N = 20). The model basis of 33,400 SU and 32.6 h (173.8 s per evaluation ÷ 0.729 efficiency) is not the cost basis: it was refuted 1.53× by the two measured runs and is kept only as the stress case. The campaign is budgeted against a remaining Anvil balance of about 600k SU, with the reserve below.

| Item | Basis | Measured, g = 1.00 | Measured, g = 1.17 | Model, g = 1.00 |
|---|---|---|---|---|
| Matched search, seed 1 (750k) | extrapolated in N, batch, nodes | 102k, 66 h | 118k, 77 h | 155k, 101 h |
| Matched search, seed 2 (500k) | extrapolated | 68k, 44 h | 79k, 51 h | 104k, 67 h |
| Both matched designs, both seeds | | 340k | 394k | 518k |
| `historic`, both seeds | measured 4,200 per 500k | 10.5k | 12.3k | 15.3k |
| Staging, E_test regeneration, ε calibration, smoke | allowance | 5k | 5k | 5k |
| E_test re-evaluation at the cap | measured 33 SU per policy (500 SOWs) | 66k | 66k | 66k |
| Draw-sensitivity re-evaluation | sized to the limit | 0.25k | 0.25k | 0.25k |
| **Total** | | **422k** | **478k** | **605k** |
| Reserve against 600k | | 178k (30 %) | 122k (20 %) | none (−5k) |

Decision points. After seed 1 of both matched designs: read SU per NFE and the coordinator-archive hypervolume at 500,000 total NFE (the snapshot at 125,000 per island). If the pair prices at or below the measured basis, submit seed 2 as planned; if it prices at the model basis, seed 2 runs at 500k only if the remaining balance covers it plus the 66k re-evaluation, otherwise the campaign reports S = 1 at equal NFE and S = 2 for `historic`. A third seed for both matched designs (~135–160k plus its re-evaluation) fits only on the measured basis with g = 1 and would consume the whole reserve, so none is planned.
