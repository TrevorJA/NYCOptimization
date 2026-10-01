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

## Measured costs (sacct, July–Sept 2026)

| Job family | Per unit | Runtime | Notes |
|---|---|---|---|
| Matched-design search, N=100, 500k NFE, 8 nodes | 21.6–23.1k SU | 21.3 h | two production runs; scale by (N/100)^0.951 and NFE/500k |
| `historic` search, 500k NFE | 4.2k SU | | single trace |
| P=10⁶ candidate pool, one draw, 50 shards | 1.85–1.9k SU | 12.4 h median shard (11.8–15.1 h) | 3 cores billed per shard at 4 GB; one shard OOM at 4 GB in Aug |
| P=10⁶ candidate pool, one draw, 100 shards | 1.85–1.87k SU | 6.15 h median shard (5.9–7.3 h) | 2026-10-01; same SU, half the wall; peak RSS 1.6 GB |
| Pool merge + verify | ~10 SU | 1–2 min | 9 cores billed |
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
