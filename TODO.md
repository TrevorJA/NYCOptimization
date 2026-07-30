# Pre-campaign TODO

Concise action items between now and the HPC optimization campaign, in rough
priority order. Each item is a pointer, not a spec — investigate and plan the
details when picked up. Delete items as they land.

Venue tags: **[local]** laptop-only, **[HPC]** needs the cluster,
**[local→HPC]** decide/dry-run locally, execute at scale on the cluster.

## 1. Artifact regeneration

Sequence after §3: JARs encode the DV **and** objective counts, so regenerating
them before the objective set is final means doing it twice.

- [ ] **[local]** Regenerate everything downstream of the 39-DV scheme: problem JARs (step 00) + `.set` artifacts (baseline regenerated 2026-07-29).
- [ ] **[local]** Regenerate any `du_forced` ensemble staged before 2026-07-28 with the variance axis off: they carry the fixed `c = 1` convention (absolute-SD) instead of the CV-preserving `c = a` (bug fixed in `src/ensemble_generation.py`).
- [ ] **[local→HPC]** Regenerate everything downstream of the 6-month metric window and the flood days/yr normalization; discard any objective values computed under the old 365-day warm-up or the whole-trace flood count. 2026-07-29: the step-05 baseline is regenerated locally under the current method (skip-reeval); the baseline re-eval matrix half stays open pending E_test (run step 05 with `--reeval` once E_test is staged).

## 2. Sizing decisions (evidence-based, before generation)

- [ ] **[HPC — TOP PRIORITY, run before JARs/baselines/searches]** Nested-P saturation diagnostic on Anvil: stage streamed hazard images at nested P (e.g. 5k / 20k / 1e5 / 1e6), run `diagnose_hazard_selectors.py` per pool, and fit min-per-axis tail share + snap-concentration ratio vs P. Gate: min per-axis tail share ≥ ~0.30 at (m = 8, N = 100). Pre-registered fallback ladder if the fitted scaling says the gate is unreachable at any affordable P: (1) reduce to concept-group representatives from the cluster tree (m ≈ 4–5; sanctioned by the sparsity clause of the axis policy — m4 showed ~0.35 mean enrichment at P=2k); (2) dry-block/wet-block split fill (two independent LHS fills of N/2; licensed by dry–wet |ρ_S| ≤ 0.15) — only if (1) fails. Never a weighted/optimized snap. Context: at P=2,000 the full set fails the gate through snap concentration + tail exhaustion, both P-limited; intrinsic dimension ≈ 5.1 so improvement scales between P^(−1/8) and P^(−1/5) (`hazard_selector_diagnostics.md` §5). Generation SU is not binding even at P=1e6 with the K=3 pool re-roll (≤~9k SU total; `scenario_design_methods.md` §6) — the diagnostic decides, not budget.
- [ ] **[HPC]** Confirm K=3 / S=2 against the allocation.
- [ ] **[local→HPC]** E_test sizing: pick the grid (candidate table computed 2026-07-28 — A/B/C/D, N_theta favored per the cost surface; recommendation N_theta=500, R=10, L=10 at ~6–10k SU), set the `src/etest.py` constants, then stage. Persistence axis DECIDED OFF 2026-07-29 — L_test = 10 yr, no gate remains.
- [ ] **[local]** Decide whether the variable-resolution `ffmp_N` sweep is in scope against the allocation; if so, which scenario design it runs under.

## 3. Remaining method closures

Most are gated by the ensemble objective-sensitivity experiment (scripts exist;
its own full-scale sizes in `supplemental_config.py` are still placeholders).

- [ ] **[HPC]** Flood unit operator (mean vs P99) + confirm the shipped failure-week counts k (`_DEFAULT_FAILURE_K`: 3 NYC/Montague, 1 Trenton/NJ) against the saturation screen.
- [ ] **[local→HPC]** Run the framing-convention diagnostics (`docs/notes/methods/framing_convention_diagnostics.md`): failure-week k sweep, weekly satisfaction factor, satisficing-criterion OAT stringency + threshold-margin CDFs, flood-days controllability. SI material; the criterion diagnostics may be promoted to main text.
- [ ] **[HPC]** Satisficing criterion values + sweep grid (`_DEFAULT_THRESHOLDS`).
- [ ] **[HPC]** Epsilons for the campaign objective set (annual-unit values are placeholders). Resolve the near-saturating Trenton reliability ε (0.0003) before JAR generation — epsilons are baked into the archive.
- [ ] **[HPC]** 8th objective (`nj_delivery_reliability_annual`): activate or drop, from the redundancy screen.
- [ ] **[local]** Flood-tail structure — potential method change: the Nowak fragment disaggregation bounds daily-extreme structure in both the candidate pool and E_test, so the wet hazard axes and the flood objective vary within a generator-limited tail. Assess whether a method change (e.g., fragment/tail perturbation) is needed or whether a scoped claim suffices.
- [ ] **[local]** E_test hazard-space overlay (simulation-free; **candidate main-text figure**, run as soon as E_test is staged): compute E_test's hazard image and overlay it on the candidate pool AND each realized search ensemble (E_d per design/draw) in the retained axes / p1–p99 box. Answers whether E_test's severe droughts (forced mean reduction) occupy the same hazard coordinates as the pool's natural-variability corners the selector enriches; also feeds the registered hazard-restricted E_test composition-sensitivity checks and the generalization claim.
- [ ] **[local]** Basin-coverage SI diagnostic: correlation of NYC-aggregate-inflow SSI-6 hazard metrics with Montague/Trenton deficits and NYC storage drawdown (strong correlation already observed in ../StochasticExploratoryExperiment) — pre-empts the aggregate-inflow scoping critique; flag in Discussion.

## 4. Pipeline shakeout on Anvil (before production submissions)

- [ ] **[local→HPC]** End-to-end smoke of `hazard_filling_stationary` with the new code (wet-exclusion, robust p1/p99 bounds, QC meta): steps 02→03→04 run locally at small P; step 06 needs the cluster (`submit_smoke.sh` = 79 MPI ranks).
- [x] **[local]** `check_objective_determinism` on the new metric window — 2026-07-29: DETERMINISTIC on all four paths (historic/ensemble × trimmed/full; 4 policies × 5 fresh-process repeats). Worst max relative deviation 2.1e-13 (Montague deficit CVaR90); LP state jitter (NYC storage differing up to 0.79 %-capacity points on 80% of days, historic_trimmed) does not propagate through the annual-unit aggregation. `scripts/supplemental/check_objective_determinism.py`; outputs under `outputs/supplemental/objective_determinism/`.
- [ ] **[HPC]** `pilot` MOEA config go/no-go run.

## 5. Production gates

- [ ] **[HPC]** Generate production pools (K draws) + `fixed_probabilistic` draws + E_test.
- [ ] **[local]** Rerun `scripts/supplemental/diagnose_hazard_selectors.py` on the production pool image (copy `hazard_image.npz` down — tens of MB, minutes to analyze): confirm the retained descriptor set (near-duplicate prune; all 8 at laptop scale) and pass the (m = 8, N = 100) adequacy gate — min per-axis tail share ≥ ~0.30 on every axis; final SI figures (F1–F10).
- [ ] **[HPC]** Launch campaign searches.

## 6. Post-campaign deliverables

- [ ] **[local]** Results figure plan + the scripts to build them (none exist yet).
- [ ] **[local]** Manuscript Results / Discussion / Conclusions; SI sections beyond S8 are outline-only.
- [ ] **[local]** Import the manuscript's † references into Zotero before submission, plus the five persistence-literature DOIs listed in `docs/notes/literature/persistence_and_low_frequency_variability.md` (MCP was read-only when they were annotated).

## Parked (scope decisions, not blockers)

- [ ] Optional HMM E_test variant — deprioritized (it introduces a second DU factor set, and the DRB-fitted HMM is near-memoryless so it is not a persistence stress). Persistence stressing was decided against 2026-07-29 — disclosure only (`persistence_axis_diagnostics.md`); a persistence-stressed test ensemble (Mechanism A, prototyped in `scengen.persistence`) is future-work material.
- [ ] Hazard descriptors beyond the controlling event (event multiplicity / frequency within a window) — independent follow-up investigation, own session.
- [ ] Manuscript scoping sentences (hazard-axis NYC-only + single-event scope LANDED in §4.3/§8 2026-07-28; remaining): demand is Decree-capped and held stationary (no demand-growth DU axis — consider other justifiable socio-economic parameterizations); the Nowak daily disaggregation limits daily-extreme (flood-tail) structure in both the search population and E_test.
- [ ] Per-realization diversion staging (`src/ensemble_prep.py` supports only `constant_max`) and the deferred DU-factor presets (`src/ensembles.py`) — both forward hooks, not campaign blockers.
