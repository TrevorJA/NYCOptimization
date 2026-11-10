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

Every wrapper logs to `logs/{job_name}_%j.out` (`_%A_%a` for arrays; the MM-Borg
launcher adds the submit-time job name). At each archive event, tar the logs older
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
| Chunked re-evaluation (`sim_master_chunks`, 128 ranks per node on `wholenode`) | 1.27–1.33 SU per (policy, chunk) unit = 33 SU per policy on 500 SOWs | ~80 min per unit per rank | jobs 19845823/19845825/19859233 (2026-08-13, 8–14 nodes); 28.6k SU to date; node RSS 197 GB at batch 50 |
| `historic` search, 750k NFE, 12 nodes (per-realization NYC demand build) | 3,673 SU | 2.39 h | job 21956470_1 (`prod_hist`, 2026-11-06), 1,533 ranks, one seed; ≈4.90 SU per 1,000 NFE (~17.6 core-s per evaluation), 41 % below the August 8.35 basis |
| Historic single-trace evaluation | ~31 s | | for re-evaluation sizing |

## Runtime rules of thumb

- Pool generation is single-threaded at ~2.2 s per 10-yr realization
  (20,000 realizations per shard at 50 shards). Wall time per shard scales
  as P / shard count; total SU is nearly independent of the count.
- Step-10 chunk re-evaluation: 128 single-threaded ranks per node on `wholenode`
  with an explicit realization batch (25 adopted; 50 measured at 1.54 GB per
  rank). The 2026-08-08 OOMs (jobs 19733674/19733773) were batch 100 (~2.8 GB
  per rank), not the rank count. The 16 ranks × 8 cpus `shared` layout (jobs
  19738752, 20037455) takes the same ~80 min per unit but bills 8 cores per
  rank: 10.8 SU per unit. One-policy cubes: `shared`, one rank per chunk × 2
  cpus. Geometry lives in `workflow/submit_reeval_chunks.sh`.
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
- 2026-10-23: coordinator-archive diagnostics (step 09 rewrite; Reed guidance).
  Shared partition, 2-4 cores: test jobs 21595978, 21596972, 21597370, 21597990
  (<1 min each), 21597369 (first step-09 attempt, 4 min), 21597752 (tests + full
  step 09 on `monte_carlo/ffmp_obj8` seed 1: 6 min 23 s on 4 cores; series build
  ~3 min, MetricsEvaluator 50 snapshots at ~2.4 s each, figures), 21598097
  (legacy merge check, 49 s: under the August-11 search vector the Python merge
  of the four final island archives reproduces the controller `.set` exactly,
  6,353 rows, identical epsilon boxes, 50 within-box representatives differ).
  Family total ~15 core-minutes, ~1 SU. Step 09 per seed is now ~6 min on 4
  cores, so the production diagnostics are negligible.
- 2026-10-23: historic step 09 rerun (job 21599089, 1 min 52 s, 4 cores) and the
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
- 2026-11-05: E_test completion chain. `staging/etest_stage_nyc_demand.sh` array 21950437
  (50 x 33 ranks, shared): all 50 tasks COMPLETED by 00:14 UTC, ~2-3 min each (~100 SU).
  `staging/etest_cut.sh` job 21950524 FAILED in 6 s (pre-merge wrapper order; no cost);
  resubmitted as 21951643, COMPLETED in 8 s: `etest_kn_50yr_n12500` staged (leading 25
  chunks of `etest_kn_50yr_n25000`), contract passed, realized utilization merged.
  Balance unchanged in substance (~591.6k).
- 2026-11-05: step-10 geometry settled from sacct, no new jobs. The three dense
  `wholenode` re-evaluations of 2026-08-13 (19845823: 8 nodes, 12.3 h, 991 policies x
  10 chunks; 19845825: 8 nodes, 9.8 h, 784 x 10; 19859233: 14 nodes, 2.5 h, 335 x 10)
  cost 1.27/1.28/1.33 SU per (policy, chunk) unit at ~76-80 min per unit per rank with
  node RSS 197 GB (1.54 GB per rank): that is the "33 SU per policy" basis. The
  `shared` 16 x 8 jobs (19738752, 20037455: 50 units in 4 h 14 m, 541 SU) run the same
  81 min per unit but bill 10.8 SU per unit. The 2026-08-08 OOMs were batch 100 RSS,
  not page cache (the dense nodes exceeded physical memory in RSS + cache and ran).
  Adopted: `submit_reeval_chunks.sh` (wholenode 128 ranks/node, batch 25, chained
  step 11; baseline cube on `shared` 25 x 2 cpus, ~75 SU instead of ~350 at 16 x 8).
  Balance 591,730.
- 2026-11-05: HIST production search, seed 1, submitted as job 21956470_1 (`submit_production_search.sh historic 1`: 12 x 128 `wholenode`, 12 h, 1,533 ranks, 750k total NFE; expected ~6.3k SU at 8.35 SU per 1,000 NFE). First pywrdrb build on Anvil under the regenerated historic NYC demand CSV (Pywr-DRB `nyc_opt` 9619a90). Companion build check `submit_smoke.sh ffmp` as job 21956471_1 (debug, 2 x 40, 79 ranks). Balance at submission 591,730.
- 2026-11-05: current-FFMP baseline cube on `etest_kn_50yr_n12500` submitted as job 21956562 (`submit_reeval_chunks.sh baseline`: `shared`, 1 node, 25 ranks x 2 cpus, batch 25, 3 h, in-job merge; expected one wave of 25 units, ~65-75 SU). First Anvil simulation that reads the per-realization NYC demand series. The helper's first submission was rejected ("Requested node configuration is not available") because `--ntasks=25` does not override the step-10 header's `--ntasks-per-node=16` on one node; the helper now passes `--ntasks-per-node`. Balance at submission 591,730.
- 2026-11-05: baseline cube job 21956562 COMPLETED (shared, 25 ranks x 2 cpus, batch 25): 1 h 14 m 35 s wall, 62 SU (50 cpus x 1.24 h; balance 591,730 -> 591,668), 25/25 units in one wave, no failed sidecars, in-job merge and robustness ran; cube at `outputs/baseline/reeval/etest_kn_50yr_n12500/` (n_solutions 1, 500 SOWs x 25 replicates). Measured per-unit basis under the per-realization demand series, from the `[unit]` lines: elapsed_s min 4138 / median 4174 / max 4463 (69.0 / 69.6 / 74.4 min; ranks 0 and 1 were the slow pair), ru_maxrss 1.02-1.03 GB per rank at batch 25, i.e. ~2.5 SU per unit at 2 cpus billed and ~1.25 SU at one cpu per rank, so the 80 min / 1.3 SU `wholenode` basis in `submit_reeval_chunks.sh` holds with ~15 % margin. Log oddities, none affecting results: Pywr-DRB `ffmp.py:2120` prints "fully constrained [...]" per timestep from every rank (~19 MB of .out); `src/robustness.py` warns that regret metrics were skipped because no baseline re-eval was found, which is expected for the incumbent's own cube.
- 2026-11-05: single-policy profile figures of the incumbent cube (`src/plotting/policy_profile.py`, registry kind `policy_profile`) rendered through `14_results_figures.sh` with `FIGURES=policy_profile`: job 21974594 FAILED in 6 s (the figure driver defaulted the E_test tag to config's `historic_single` without an env file; the driver now defaults to the campaign tag), job 21974641 COMPLETED in 17 s (4 cpus, negligible SU). Five figures in `figures/si/policy_profile_*_baseline.png`, tables under `outputs/comparison/ffmp_obj8/etest_kn_50yr_n12500/figure_tables/policy_profile/`.
- 2026-11-05: historic step-06 baseline rerun under the historical NYC demand series, job 21989910 (`06_run_baseline.sh`, historic env, shared 1 core): COMPLETED in 50 s, negligible SU; `outputs/baseline/ffmp_baseline.hdf5` (141 MB) and the historic objective vector (the script wrote it flat beside the HDF5; moved to the `outputs/baseline/historic/` locator `config.baseline_objectives_csv` names and the writer corrected). Threshold-anchor diagnostics `robustness_threshold_diagnostics.sh` chained as job 21989911 (`NYCOPT_RTD_REFRESH=1`).
- 2026-11-05: threshold-anchor diagnostics job 21989911 COMPLETED in 18 s (shared, 4 cpus, negligible SU): historic anchors recomputed from the new baseline HDF5 (identical to the step-06 vector), tables and figures under `outputs/supplemental/robustness_threshold_diagnostics/`.
- 2026-11-06: HIST production search seed 1, job 21956470_1, COMPLETED (exit 0): started 03:46 EST (queued ~28 h), ended 06:10, 2 h 23 m 28 s wall (ElapsedRaw 8,608 s) on 12 x 128 `wholenode`, 1,536 cores billed = 3,673 SU (balance 591,668 -> 587,995, matches sacct). All four islands reached 187,500 NFE (750k total); MM-Borg wall 8,583 s; `outputs/historic/ffmp_obj8/sets/seed_01_ffmp_obj8.set` (42 solutions) and four island runtime files written; stderr holds only the `flood_stage.py` rating-curve extrapolation warnings. Measured 4.90 SU per 1,000 NFE against the 8.35 August basis (~17.6 core-s per historic evaluation instead of ~31 s), so the seed-2 estimate drops from ~6.3k to ~3.7k SU at the same NFE. Companion smoke job 21956471_1 (debug) still pending.
- 2026-11-06: HIST seed-1 post-processing, all `shared`, negligible SU: step 08 `--seed 1` job 22005400 (12 s; 161 island rows at NFE 125,000 per island -> 41 after the epsilon merge, `seed_01_ffmp_obj8_nfe125000.set`); step 09 job 22005402 (14 s; 75 coordinator snapshots, controller `.set` boxes reproduced, figures `outputs/figures/historic/ffmp_obj8/search_0{1,2,3}_*.png`); explore suite job 22005404 (25 s, new wrapper `workflow/supplemental/figures/explore_results.sh`, `--skip-timeseries`, on `seed_01_ffmp_obj8.set`); baseline-dominance layer test job 22006829 (8 s, new `workflow/supplemental/results_diagnostics/baseline_dominance_layers.sh`): no row in any archive layer (final set, island finals, all 13,069 island snapshot rows, all 3,444 coordinator rows) dominates the historic baseline.
- 2026-11-06: satisficing thresholds re-specified (NYC/Montague/NJ rel >= 0.80, deficits <= 50/25/10 %, flood <= 2.0 ft-d/yr, storage P01 >= 20 %; criteria variant `v3_20261106`). Incumbent cube re-snapshotted and rescored by `rescore_thresholds.sh` job 22008035 (7 s), threshold diagnostics job 22008036 (14 s) and policy-profile figures job 22008037 (17 s, `14_results_figures.sh` with `FIGURES=policy_profile`), all `shared`, all COMPLETED, negligible SU. No design cube exists on the tag yet, so the design-policy scenario-discovery drivers were not run.
- 2026-11-06: policy-profile failure maps re-rendered with the satisficing condition in each panel title (`14_results_figures.sh`, jobs 22008229, 22008331, 22008383; `shared`, 8 to 19 s each, negligible SU). A comma-separated `FIGURES` list cannot pass through `sbatch --export` (the comma is the export separator); use one figure name or the kind `policy_profile`.
- 2026-11-06: smoke job 21956471_1 (`smoke_ffmp`, debug) CANCELLED while still pending, never ran; no cost. The HIST seed-1 search completed under the same build, so the separate build check was no longer needed.
- 2026-11-09: epsilon re-filter estimate on the union of every HIST seed-1 island snapshot (new `workflow/supplemental/results_diagnostics/epsilon_union_refilter.sh`): jobs 22116718 and 22116821, `shared` 2 cpus, 14 s each, negligible SU; tables under `outputs/supplemental/epsilon_union_refilter/historic/ffmp_obj8/`.
- 2026-11-09: epsilon union re-filter job 22116766 (14 s, shared 2 cpus, negligible SU; new `workflow/supplemental/results_diagnostics/epsilon_union_refilter.sh`): epsilon-box size of the union of every HIST seed-1 island snapshot (7,049 distinct rows) under candidate vectors; tables under `outputs/supplemental/epsilon_union_refilter/historic/ffmp_obj8/`.
