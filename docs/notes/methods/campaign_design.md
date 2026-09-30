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

The campaign result for every design is the ε-nondominated merge of both seeds at 125,000 NFE per island (`scripts/main/extract_runtime_archive.py` builds `seed_01_{slug}_nfe125000.set` from the island runtime files and the equal-NFE `{slug}_merged_nfe125000.set`). If seed 1's runtime hypervolume is not converged by 125,000 per island, seed 2 is extended to 187,500 by editing `max_evaluations_by_seed` before it is submitted. Searches are NFE-bounded (no Borg maxTime); the SLURM wall is the only cap.

## 3. Geometry

| Item | Value |
|---|---|
| Nodes | 12 Anvil `wholenode` (1,536 cores) |
| Ranks | 1 controller + 4 islands × (382 workers + 1 master) = 1,533 (3 idle cores) |
| Ranks per node | 128 |
| Realization batch | `NYCOPT_SEARCH_REALIZATION_BATCH=150` in both matched env files (two model runs per evaluation); unset for `historic` |
| Estimated node RSS | 167 GB at N = 300 batched (envelope model 600 + 0.49 MB per simulated year per rank; 259 GB unbatched, above the 217 GB line) |
| Pre-flight | `nycopt_check_allocation` (ranks) and `nycopt_check_memory` (node RSS vs 85 % of 256 GB) both abort before Borg starts |
| `--time` | seed 1 96 h (matched), 12 h (`historic`); seed 2 72 h (matched), 8 h (`historic`) |
| Resume | none. Runtime files are diagnostic archive dumps and the Borg checkpoint is disabled (race-prone across islands, never run). Every search must finish inside one job |

Twelve nodes are required because a 750k-NFE search at N = 300 is projected at 99 h on eight. Node scaling beyond eight nodes is unmeasured (single-island curve loses ~9 % per worker doubling; the one cross-node production pair shows +30 % SU per NFE per doubling, confounded with NFE), so every 12-node number below carries a factor g ∈ [1.00, 1.17]. The seed-1 job is itself the measurement. Its 125,000-per-island snapshot lands at 44–51 h on the measured basis and at 67–79 h even on the model basis, so the equal-NFE result is recoverable from a job killed at the wall.

## 4. Pre-search steps

Before the searches, E_test is regenerated over the extended forcing box (§5: sharded generation, the presim pass, the sub-window hazard image, the prefix subset and the current-policy cubes, about 0.35k SU inside the staging allowance), both matched designs are staged at N = 300 for draws d0 and d1 (workflow steps 02–04 on the P = 10⁶ pools), the step-05 current FFMP policy baselines are simulated on each d0 ensemble, the ε vector [0.05, 10.0, 0.05, 10.0, 10.0, 0.3, 5.0, 0.05] is re-verified against the N = 300 floors (`workflow/supplemental/epsilon_calibration.sh`; τ is re-pinned only if ε changes), and the batched-search memory smoke (`workflow/submit_search_memory_smoke.sh`) confirms node RSS and evaluation time on one node.

## 5. E_test re-evaluation

| Item | Value |
|---|---|
| Generated E_test | N_θ = 1,000 LHS SOWs over the full CMIP6 amplitude box widened by 25 % with the annual-volume lower bound extended to e^m = 0.80 (`src/etest.py::E_TEST_VOLUME_MULTIPLIER_MIN`; wet bound 1.32) × R = 25 × L_test = 50 yr, 50 staged chunks of 500 realizations (`etest_kn_50yr_n25000`). Regenerated over the extended box before the campaign; the pre-extension staging and its current-policy cubes are superseded and `assert_staged_etest_contract` refuses them. The hazard-image and forcing-profile source |
| Re-evaluated E_test | the leading 500 SOWs = the first 25 chunks, 12,500 realizations, 625k simulated years (`etest_kn_50yr_n25000_first25ch`, a metadata-only prefix subset staged by `scripts/supplemental/make_etest_subset.py`; LHS rows are randomly ordered, so the prefix is an unbiased half of the design). `src/etest.py::E_TEST_REEVAL_N_THETA` is the single source; `NYCOPT_REEVAL_ENSEMBLE_PRESET` names it in every step-05/08/09/10 submission |
| Why 500 | the forcing space has 3 axes. Published re-evaluation ensembles are 10,000 LHS SOWs in 13-factor spaces (Herman et al. 2014; Trindade et al. 2017) and the 5-factor lake problem (Bartholomew & Kwakkel 2020), 1,000–2,000 at 5–14 factors (Eker & Kwakkel 2018; Hadjimichael et al. 2020; Gold et al. 2023), and 500 in the one 3-axis space with a measured convergence curve (Bonham et al. 2024), where satisficing rankings stabilize from 50–300 SOWs and regret-type metrics need 400–500. 500 sits at that precedent's density and at the lower edge of its regret range |
| Why the dry bound | the margin box's lower bound (0.866) leaves the stationary drought envelope only at its edge: 10 % of a SOW's decades beyond the stationary generator's 99th-percentile controlling-event drought magnitude (22 % on the magnitude summed over all events), and the driest CMIP6 run (0.929) 1.7 %, against 1 % by construction (`scripts/supplemental/dry_envelope_run.py`, `dryenv_levels.csv`, SI Figure S18). At 0.807, the measured level nearest 0.80, 17 % (48 % summed) with the median decade at the stationary 90th percentile. 0.80 is the dry limit of the inflow multiplier of Herman et al. (2014) and Trindade et al. (2017), and Hadjimichael et al. (2020) set their envelope to span beyond the stationary synthetic flows on both sides. 18 % of the 500 SOWs lie below the former bound and 35 % below the driest CMIP6 run. The production confirmation is the dry-envelope production leg on the regenerated images |
| Cross-SOW precision | worst-case SE of a satisficing fraction 0.5/√500 = 2.2 pp; the RQ1 discrimination band δ = 2 × paired SE is measured on the production cubes |
| Per-SOW precision | 1,225 pooled annual units per SOW (R = 25). The measured per-SOW noise on E_test (paired floors 0.017–0.024 reliabilities, 1.2 pp deficit, 0.04 ft·d/yr flood, 3.0 pp storage) is below ε on every axis (0.48, 0.12, 0.13 and 0.60 of ε). Realizations within a SOW are independent, so R = 50 would scale every floor by 1/√2 (0.017, 0.85 pp, 0.028, 2.1 pp) and lower the current policy's share of SOWs misclassified at the adopted criteria from at most 1.1 pp (storage) to 0.7 pp, against the 2.2 pp cross-SOW standard error it leaves unchanged, for +66.5k SU (`scripts/supplemental/etest_realization_precision.py`, tables `etr_per_sow_noise.csv`, `etr_options.csv`; SI Table S1). Published re-evaluations use 1 to 1,000 realizations per SOW and none tests the count at a fixed SOW count, so R is set by this noise argument |
| Current FFMP policy baseline | simulated once on the full regenerated 1,000-SOW E_test by step 09 with `NYCOPT_CHUNK_POLICIES=baseline` under the historic env (~66 SU); that cube is a superset joined by SOW label, which `stage_etest_subset_baseline.py` symlinks under the subset tag per design (no re-simulation) |
| Path | step 09 chunked metrics-only re-evaluation, `shared`, 16 ranks × 8 cpus per node, batch 50, then 09b merge; ~50k (policy, chunk) units at the cap |
| Measured cost | 33 SU per policy (1.33 SU per policy-chunk unit, measured on the full pool, × 25 chunks) |
| Policies | the equal-NFE merged set per design; expected ≈ 2,000 in total (measured at N = 100, S = 1: 1,040 + 833 + 335 after the ε re-filter; unmeasured at N = 300 and S = 2) |
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

Decision points. After seed 1 of both matched designs: read SU per NFE and the runtime hypervolume at 125,000 per island. If the pair prices at or below the measured basis, submit seed 2 as planned; if it prices at the model basis, seed 2 runs at 500k only if the remaining balance covers it plus the 66k re-evaluation, otherwise the campaign reports S = 1 at equal NFE and S = 2 for `historic`. A third seed for both matched designs (~135–160k plus its re-evaluation) fits only on the measured basis with g = 1 and would consume the whole reserve, so none is planned.
