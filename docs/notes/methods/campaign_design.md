# Campaign Design at Scale

*The production campaign as it will be run, with its expected budget. Single-source constants: `src/scenario_designs.py` (N, L, draws), `src/moea_config.py::production` (islands, workers, NFE per seed, snapshot cadence), `src/etest.py` (E_test), `workflow/envs/*_production.env` (batch, τ, submit lines), `config.py` (memory model). Sizing evidence: `ensemble_size_diagnostics.md`; cost provenance: SI Text S8. Where this note and the code disagree, the code is the record.*

---

## 1. Arms

| Arm | Search ensemble | Draws searched | Seeds | Role |
|---|---|---|---|---|
| `fixed_probabilistic` (PS) | N = 300 i.i.d. realizations, L = 10 yr | 1 (d0) | 2 | exact control |
| `hazard_filling_stationary` (HF) | N = 300 selected from the d0 P = 10⁶ pool, L = 10 yr | 1 (d0) | 2 | proposed method |
| `historic` | one 78-yr trace | 1 | 2 | prevailing-practice reference, matched NFE |

Three draws (d0, d1, d2) are staged for both matched designs. The search runs on d0 only (K = 1). d1 and d2 exist for the SI draw-sensitivity re-evaluation (§5). The unit of analysis is the seed; the design comparison is conditional on one draw per design, and draw-dependence is measured by re-evaluating each design's final set on its own other draws.

S = 2 is a floor. A third seed for both matched designs costs ~135–160k SU of search plus the re-evaluation of its policies and does not fit the balance in §6 unless the seed-1 runs price at least 25 % below projection.

## 2. NFE scheme

| Seed | NFE per island | Total NFE | Runtime snapshots (every 2,500) | Reported as |
|---|---|---|---|---|
| 1 | 187,500 | 750,000 | 75 | equal-NFE result = the snapshot at 125,000 per island (snapshot 50); the 750k tail is SI convergence evidence |
| 2 | 125,000 | 500,000 | 50 | final archive |

The campaign result for every arm is the ε-nondominated merge of both seeds at 125,000 NFE per island (`scripts/main/extract_runtime_archive.py` builds `seed_01_{slug}_nfe125000.set` from the island runtime files and the equal-NFE `{slug}_merged_nfe125000.set`). If seed 1's runtime hypervolume is not converged by 125,000 per island, seed 2 is extended to 187,500 by editing `max_evaluations_by_seed` before it is submitted. Searches are NFE-bounded (no Borg maxTime); the SLURM wall is the only cap.

## 3. Geometry

| Item | Value |
|---|---|
| Nodes | 12 Anvil `wholenode` (1,536 cores) |
| Ranks | 1 controller + 4 islands × (382 workers + 1 master) = 1,533 (3 idle cores) |
| Ranks per node | 128 |
| Realization batch | `NYCOPT_SEARCH_REALIZATION_BATCH=150` in both matched env files (two model runs per evaluation); unset for `historic` |
| Estimated node RSS | 167 GB at N = 300 batched (envelope model 600 + 0.49 MB per scenario-year per rank; 259 GB unbatched, above the 217 GB line) |
| Pre-flight | `nycopt_check_allocation` (ranks) and `nycopt_check_memory` (node RSS vs 85 % of 256 GB) both abort before Borg starts |
| `--time` | seed 1 96 h (matched), 12 h (`historic`); seed 2 72 h (matched), 8 h (`historic`) |
| Resume | none. Runtime files are diagnostic archive dumps and the Borg checkpoint is disabled (race-prone across islands, never run). Every search must finish inside one job |

Twelve nodes are required because a 750k-NFE search at N = 300 is projected at 99 h on eight. Node scaling beyond eight nodes is unmeasured (single-island curve loses ~9 % per worker doubling; the one cross-node production pair shows +30 % SU per NFE per doubling, confounded with NFE), so every 12-node number below carries a factor g ∈ [1.00, 1.17]. The seed-1 job is itself the measurement. Its 125,000-per-island snapshot lands at 44–51 h on the measured basis and at 67–79 h even on the model basis, so the equal-NFE result is recoverable from a job killed at the wall.

## 4. Staging and pre-search steps

1. Steps 02–04 for both matched designs at N = 300, draws 0–2 (step 03 with `NYCOPT_CANDIDATE_POOL_N=1000000`; the P = 10⁶ pools are draw-indexed and N-independent, so only the selections and the pywrdrb inputs are new); `validate_staged_seasonality` on each; step 05 baselines scenario-matched to the new d0 ensembles. Step 01 presim and E_test are unchanged.
2. ε calibration at N = 300 per design (`workflow/supplemental/epsilon_calibration.sh`, `EPS_REALIZATION_BATCH=150`). The adopted vector [0.05, 10.0, 0.05, 10.0, 0.05, 0.3, 5.0, 0.05] stays in force for the searches provided its entries lie above the N = 300 floors (expected, floors fall with N); the archive-cardinality re-filter is repeated on the first N = 300 archives after seed 1. τ is re-pinned only if ε changes.
3. Batched-search memory smoke (`workflow/submit_search_memory_smoke.sh`): one wholenode node, 127 ranks, N = 300, batch 150, ~400 NFE, node-memory sampler on. Go criterion: sampled node RSS ≤ 217 GB and warm evaluation time within +20 % of 540 s (174 s × 2.84 × 1.09).

## 5. E_test re-evaluation

| Item | Value |
|---|---|
| E_test | N_θ = 1,000 LHS SOWs × R = 25 × L_test = 50 yr, 50 staged chunks of 500 realizations (`etest_kn_50yr_n25000`), unchanged |
| Path | step 09 chunked metrics-only re-evaluation, `shared`, 16 ranks × 8 cpus per node, batch 50, then 09b merge |
| Measured cost | 66 SU per policy (1.33 SU per policy-chunk unit × 50 chunks) |
| Policies | the equal-NFE merged set per arm; expected ≈ 2,000 in total (measured at N = 100, S = 1: 1,040 + 833 + 335 after the ε re-filter; unmeasured at N = 300 and S = 2) |
| Cap | 2,000 policies (~132k SU). If the union exceeds it, the post-hoc ε re-filter is coarsened to the cardinality target and applied identically to every arm |
| Per-SOW precision | 1,225 pooled annual units per SOW; the measured per-SOW noise on E_test (paired floors 0.017–0.024 reliabilities, 1.2 pp deficit, 0.04 ft·d/yr flood, 3.0 pp storage) is below ε on every axis, and comparison precision is set by the 1,000-SOW count |
| Draw sensitivity (SI) | each matched design's final set re-simulated on its own d1 and d2 at N = 300 (~70 SU staging each, ~66 SU per 100 policies), ~1k SU |

## 6. Budget

Measured basis: 21,850 SU and 21.3 h per N = 100 / 500k-NFE search on 8 × 128 (two production runs), scaled by (N/100)^0.951 (fitted N = 10–200, r² 0.999; ±8 % at N = 300) and NFE/500k, with a ×1.09 batch penalty (upper bound, measured at N = 20). Model basis: 33,400 SU and 32.6 h (173.8 s per evaluation ÷ 0.729 efficiency), refuted 1.53× by the two measured runs and kept as the stress case. Balance ≈ 590–600k SU (no ledger in the repo; 679k on 2026-08-10 less ~80k spent since; verify before launch).

| Item | Basis | Measured, g = 1.00 | Measured, g = 1.17 | Model, g = 1.00 |
|---|---|---|---|---|
| Matched search, seed 1 (750k) | extrapolated in N, batch, nodes | 102k, 66 h | 118k, 77 h | 155k, 101 h |
| Matched search, seed 2 (500k) | extrapolated | 68k, 44 h | 79k, 51 h | 104k, 67 h |
| Both matched designs, both seeds | | 340k | 394k | 518k |
| `historic`, both seeds | measured 4,200 per 500k | 10.5k | 12.3k | 15.3k |
| Staging, ε calibration, smoke | allowance | 5k | 5k | 5k |
| E_test re-evaluation at the cap | measured 66 SU per policy | 132k | 132k | 132k |
| Draw-sensitivity re-evaluation | extrapolated | 1k | 1k | 1k |
| **Total** | | **489k** | **544k** | **671k** |
| Reserve against 600k | | 111k (19 %) | 56k (9 %) | none |

Decision points. After seed 1 of both matched designs: read SU per NFE and the runtime hypervolume at 125,000 per island. If the pair prices at or below the measured basis, submit seed 2 as planned; if it prices at the model basis, seed 2 runs at 500k only if the remaining balance covers it plus the 132k re-evaluation, otherwise the campaign reports S = 1 at equal NFE and S = 2 for `historic`. No third seed is planned.
