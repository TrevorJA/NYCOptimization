# Anvil notes

Running record of SU balance, measured costs, and runtimes on Anvil (account
`ees260021`). Update the balance line whenever a job family lands. Budget
rationale lives in `docs/notes/methods/campaign_design.md` §6; this file holds
only the numbers needed to make submission decisions.

## Balance

| Date | `mybalance` | Used | Note |
|---|---|---|---|
| 2026-09-30 | 595,542 | 104,458 | limit shown 700,000; total award 750,000, registered in tranches |
| 2026-10-01 | 591,772 | 108,228 | after the June-1 pool regeneration (3,769 SU) |

Campaign budget (campaign_design.md §6): 422k measured basis, 478k with the
node-scaling factor, 605k model basis, against ~600k remaining.

## Billing rules

- `shared` partition bills `AllocCPUS × Elapsed`. Memory is converted to cores
  at roughly 1.9 GB per core, rounded up: `--mem=4G` bills 3 cores (not 2),
  `--mem=16G` bills 9, `--mem=32G` bills 18. Set memory with that in mind.
- `wholenode` bills 128 cores per node for the full elapsed time.
- Small 2-3 core jobs backfill within minutes even with ~20k jobs pending in
  `shared`; multi-node `wholenode` jobs can wait hours.

## Log hygiene

Every wrapper logs to `logs/{script_stem}_%j.out` (`_%A_%a` for arrays; the
MM-Borg launcher adds the job name). At each archive event, tar the logs older
than that event into `archive/<date>_<reason>/logs/` so `logs/` holds only the
current campaign.

## Measured costs (sacct, July–Sept 2026)

| Job family | Per unit | Runtime | Notes |
|---|---|---|---|
| Matched-design search, N=100, 500k NFE, 8 nodes | 21.6–23.1k SU | 21.3 h | two production runs; scale by (N/100)^0.951 and NFE/500k |
| `historic` search, 500k NFE, 8 nodes | 4,172 SU | 4.07 h | job 19770937_1 (`prod_hist`, 2026-08-10), 1,024 ranks, one seed; ≈8.35 SU per 1,000 NFE, within 3 % of 31 s × NFE, so SU is linear in NFE and seeds and node count sets only the wall |
| P=10⁶ candidate pool, one draw, 50 shards | 1.85–1.9k SU | 12.4 h median shard (11.8–15.1 h) | 3 cores billed per shard at 4 GB; one shard OOM at 4 GB in Aug |
| P=10⁶ candidate pool, one draw, 100 shards | 1.85–1.87k SU | 6.15 h median shard (5.9–7.3 h) | 2026-10-01; same SU, half the wall; peak RSS 1.6 GB |
| Pool merge + verify | ~10 SU | 1–2 min | 9 cores billed |
| Selector diagnostic, full battery, P=10⁶, N=300 | 3–3.6 SU per draw | 35–43 min | 2026-10-01; 4 cpus / 8 GB header bills 5 cores; `hazard_selector_diagnostics.sh` |
| P=2,000 smoke pool | ~18 SU | ~1 h | 8 cpus / 32 GB header |
| E_test generation, 100 shards | 1.4k SU | | `gen_etest_shard` |
| E_test presim chunks, 100 tasks | 1.6k SU | | `prep_etest_chunk` |
| Chunked re-evaluation (`sim_master_chunks`) | 33 SU per policy on 500 SOWs | | 16 jobs, 28.6k SU to date |
| Historic single-trace evaluation | ~31 s | | for step-08 reeval sizing |

## Runtime rules of thumb

- Pool generation is single-threaded at ~2.2 s per 10-yr realization
  (20,000 realizations per shard at 50 shards). Wall time per shard scales
  as P / shard count; total SU is nearly independent of the count.
- Step-09 chunk re-evaluation OOMs under dense packing; use ~8 cpus per rank
  with an explicit batch (16 × 8, batch = 50 works).
- Full-node searches at N = 300 project to 44–51 h to the 125k-per-island
  snapshot on 12 nodes (measured basis).

## Log

- 2026-09-30: June-1 hazard-image pool regeneration started (TODO §2).
  Balance before 595,542. Smoke array 20986172 (started 18:02 EDT). Shard
  count raised from 50 to 100 per draw to halve the wall (same SU; rows are
  keyed to the global index so the image is unchanged). Smoke pools done
  19:16 EDT (73 min each, 18 cores billed, ~44 SU together). Submitted 19:51
  EDT: d0 shard array 20987592 → merge 20987593; d1 shard array 20987594 →
  merge 20987595 (afterok). 100 tasks per array, 3 cores each.
- 2026-10-01: all 200 shards completed (5.9–7.3 h each); both merges and
  verifiers passed by 03:16 EDT. Total 3,769 SU; balance 591,772 after.
- 2026-10-01: production selector diagnostic, array 20996434 (d0 43 min, d1
  35 min, 5 cores each, ~6.5 SU). Outputs under
  `outputs/supplemental/hazard_selector_diagnostics/statpool_10yr_n1000000_d{0,1}/`.
- 2026-10-16: full-model historic HDF5 runs of the HIST `ffmp_obj8` representatives
  (`sim_selected_policies.sh`, shared, 1 task, 2 cpus). Job 21392961 (all 13 table
  rows + incumbent) cancelled after 2 min: home quota at 95% and ~140 MB per run;
  `outputs/historic/ffmp_obj8/timeseries/selected_policies/` was made a symlink into
  projects space (moved to `archive/2026-11-04_pre_demand_update/` as a real directory
  on 2026-11-04). Resubmitted as job 21393031
  with the three highlighted policies (rows 1393, 1682, 115) + incumbent. Measured:
  one full-model historic run to HDF5 = 40 s. Balance before 591,766.
- 2026-10-16: job 21393031 completed in 2 min 51 s (4 runs, ~40 s each, 2 cores;
  ~0.1 SU). Outputs 4 x 140 MB HDF5 under projects space.
- 2026-10-16: selection regenerated under the current baseline (rows 1947, 2205,
  2013); job 21393749 simulated the three new rows (2 min 35 s, ~0.1 SU), baseline
  reused; the superseded row_1393/1682/115 runs were deleted.
- 2026-10-23: coordinator-archive diagnostics (step 07 rewrite; Reed guidance).
  Shared partition, 2-4 cores: test jobs 21595978, 21596972, 21597370, 21597990
  (<1 min each), 21597369 (first step-07 attempt, 4 min), 21597752 (tests + full
  step 07 on `monte_carlo/ffmp_obj8` seed 1: 6 min 23 s on 4 cores; series build
  ~3 min, MetricsEvaluator 50 snapshots at ~2.4 s each, figures), 21598097
  (legacy merge check, 49 s: under the August-11 search vector the Python merge
  of the four final island archives reproduces the controller `.set` exactly,
  6,353 rows, identical epsilon boxes, 50 within-box representatives differ).
  Family total ~15 core-minutes, ~1 SU. Step 07 per seed is now ~6 min on 4
  cores, so the production diagnostics are negligible.
- 2026-10-23: historic step 07 rerun (job 21599089, 1 min 52 s, 4 cores) and the
  `explore_*` suite of `figures/historic/ffmp_obj8` regenerated (job 21600863, 57 s,
  2 cores) under the August objective list (`trenton_flow_reliability_weekly` fifth)
  that the archive and `outputs/baseline/ffmp_baseline_objectives.csv` share; the
  selection reproduced rows 1947/2205/2013, so figures 05/06 reused the HDF5 traces.
- 2026-11-04: balance 591,765 before the demand-model rerun. Every pre-2026-11-04 simulation
  output moved to `archive/2026-11-04_pre_demand_update/` (projects space, repo symlink
  `archive/`; `archive/README.md` lists the contents), the pre-June-1 pools to
  `archive/2026-10-01_pre_june1_hazard_recompute/`; home quota 89 % -> 80 %. The
  pre-extension E_test (290 GB) sits in the archive and can be deleted once the
  regenerated one passes `assert_staged_etest_contract`. Measured costs of the previous
  E_test builds (sacct): generation 50 shards x 92 min x 9 cores = ~690 SU
  (19654577, 20033555); presim `prep_etest_chunk` 49 tasks x 55 min x 33 cores = ~1.5k SU
  on 2026-08-04 (19660403, two OOM reruns) but 50 x 0.7 min = ~19 SU on 2026-08-19
  (20034847) after the STARFIT presim parallelism change; hazard-image shards ~21 SU.
  Budget the chain at ~1k SU, not the 0.35k in `campaign_design.md` §4.
  `demand_utilization_calibration.py` ran on the login node (6 s).
- 2026-11-04 (later): the 0.80 annual-volume extension of E_test was reverted (the box is the
  widened CMIP6 range, e^m = 0.866 to 1.32), so the 2026-08-19 E_test is reused: moved back
  from the archive (projects-space rename, no copy), its retired-scoring sub-window image set
  aside in the archive, utilization column appended by `srun` job 21947556 (2 cores, seconds;
  u in [0.450, 0.999], k in [0.653, 6.0], 4 SOWs clipped at k = 6). Remaining E_test chain is
  ~0.1k SU (chunk demand staging, sub-window image) plus the 66 SU baseline cube.
- 2026-11-05: E_test completion chain. `stage_nyc_demand_etest_chunks.sh` array 21950437
  (50 x 33 ranks, shared): all 50 tasks COMPLETED by 00:14 UTC, ~2-3 min each (~100 SU).
  `etest_subset_stage.sh` job 21950524 FAILED in 6 s (pre-merge wrapper order; no cost);
  resubmitted as 21951643, COMPLETED in 8 s: `etest_kn_50yr_n12500` staged (leading 25
  chunks of `etest_kn_50yr_n25000`), contract passed, realized utilization merged.
  Balance unchanged in substance (~591.6k).
