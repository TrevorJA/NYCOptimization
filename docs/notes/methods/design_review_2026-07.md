# Design Review — 2026-07-09

*Session deliverable: critique and refinement of the experimental design ahead of the
full-scale Anvil campaign. Sources: all `docs/notes/` methods+literature notes, the code
as of `main@07a46ec` (verified this session), the input-vs-hazard diagnostic
(`outputs/diagnostics/input_vs_hazard_coverage/FINDINGS.md`), the Zotero collection
ISYGLK35 (56 items), and a verified web scan of 2020–2026 literature. Docs are treated
as working drafts, not commitments; doc-vs-code divergences are findings, not errors.
Sections: A critique · B Zotero additions · C code edits · D discussion points.*

---

## A. Methods critique and recommended refinements (ranked by severity)

**A1. The probabilistic designs do not currently draw from the shared master — the
central control is broken in code.**
`scenario_design_methods.md` §2.1 states all non-historical designs draw from the *same*
master ensemble so differences are attributable to selection. In code,
`fixed_probabilistic_short/_long` and `resampled_probabilistic` are generated directly
from the **stationary baseline** Kirsch–Nowak generator (`master_kind="stationary"`,
`src/scenario_designs.py`), while `hazard_filling`/`input_stratified` use the
**CMIP6-forced** master (`master_kind="forcing"`). As implemented, a hazard-filling win
could be attributed to its scenarios containing climate-perturbed hydrology at all,
not to hazard-space selection. *Recommendation:* before the campaign, re-wire the
probabilistic designs as random index-draws from the same forcing master (the
subsample-from-master step the methods note already anticipates). This also makes the
"representative-in-probability" framing precise: the reference measure is the master's
empirical measure — which is itself a designed DMDU object. State that explicitly in the
manuscript; a stochastic-programming reviewer will otherwise read "probabilistic
baseline" as stationary well-characterized sampling.

**A2. Satisficing-fraction search objectives risk a degenerate fitness landscape.**
All ensemble designs collapse each objective to a satisficing fraction over N
realizations (`objectives_ensemble.py`, satisficing-only by design). Three problems:
(i) granularity is 1/N — with N≈100–300 that is 0.003–0.01 steps, but for
`fixed_probabilistic_long` (N′≈2–20) the objective takes ≤21 distinct values, and at the
current provisional N′=2 it takes three. Borg's ε-dominance archive and its
auto-adaptive operators need gradient; plateaued objectives stall search and would make
the long-record design lose *by construction of the operator*, not by scenario
composition — a fatal confound for the many-short-vs-few-long contrast. (ii) The ε
values in `objective_definitions.md` §1 are in temporal-metric units; fraction-valued
objectives need their own ε (sensible floor: ε ≥ 1/N, else the archive resolves noise).
(iii) Ties across policies are common when thresholds are slack, weakening selection
pressure everywhere. *Recommendation:* let the planned ensemble objective-sensitivity
experiment decide, but between two defensible options: (a) keep satisficing everywhere
but require N ≥ ~100 for every budget-matched design (which kills the current long-record
construction — see D3), or (b) use a continuous across-realization operator (mean, or
mean of the per-realization metric with CVaR reported) during search and reserve
satisficing for re-evaluation. Option (b) preserves the "one operator family for all
designs" commensurability argument and is the safer default for Borg.

**A3. Overfitting-gap: two notes prescribe opposite designs.**
`scenario_design_methods.md` §6b pre-registers a coverage→overfitting-gap regression as
the *central hypothesis*; `objective_definitions.md` §3 states the gap "is **not** used
as a comparison measure." Both are dated within a day of each other. This is the single
most important unresolved methods decision because it fixes what the study claims to
test. *Recommendation:* keep robustness + regret as the primary comparison (per
objective_definitions) and retain the coverage→gap regression as a clearly-labeled
secondary *mechanism* analysis — it is the only analysis that explains *why* a design
wins, and the confound guard in §6b is already well thought through. Whichever way this
lands, edit the losing passage; a reviewer who reads both notes' descendants in the
manuscript will find the contradiction. (Decision: D1.)

**A4. The deeply uncertain forcing space omits the mechanism that dominates NYC
drought.** The forcing hypercube perturbs monthly mean and CV only
(`forcing_parameterization.md`, flagged there as a limitation). Interannual persistence
— which controls multi-year drought, the hazard NYC storage actually responds to — is
inherited unchanged from the historically-fitted Kirsch correlation structure in every
scenario. Consequences: (i) hazard-axis reachability from input space is weak (max
|Spearman| 0.21–0.33 in the coverage diagnostic), so part of hazard-filling's coverage
advantage may be coverage of *generator noise* rather than of climate-driven hazard;
(ii) the "deeply uncertain" test ensemble is not deeply uncertain in the dimension that
matters most. *Recommendation:* either add one persistence axis (e.g., a bounded
inflation factor on the Kirsch interannual/lag-1 structure, or an HMM-style wet/dry
persistence parameter) to the forcing space before generating the production master, or
explicitly scope every claim to "under historical persistence" and defend it. The first
is stronger and cheap relative to the campaign. (Decision: D5.)

**A5. Scenario length L = 5 yr truncates the design-basis drought.** The 1960s drought
of record is ~4–5 yr; the methods note's own rule is that L must exceed the longest
within-window drought, and the coverage diagnostic already flags duration-type axes as
truncation-limited. This gates master generation, so it must be decided now.
*Recommendation:* L = 10 yr for the constant-L designs (halves N at fixed scenario-years
but preserves whole events and drought-duration axes); keep L′ = 50 for the length
contrast. If 5 yr is retained for cost, the duration axis should be dropped from the
hazard set and the truncation bias quantified on the master. (Decision: D3.)

**A6. Test-ensemble conditioning: out-of-sample in draw, not in structure.** E_test is
generated by the same Kirsch–Nowak generator over the same forcing space as the search
master, so "generalization" is tested within the generator family; the planned ≥2
E_test variants (wider envelope, historically-anchored tail) share that family.
*Recommendation:* make one of the ≥2 variants structurally different — e.g., a
weather-regime/HMM or wavelet-based generation (Steinschneider 2019; Brunner 2020,
both already in the collection) or the temperature-informed generation of Ji & Ahn
2023. Cross-generator ranking stability is a far stronger robustness statement and
directly answers the predictable reviewer question. Also note E_test does not exist in
code at all yet (reeval default is the single historic trace) — see C6.

**A7. The campaign SU arithmetic is not yet written down.**
Methods §5 targets K=10 draws × S=2–3 seeds per design, ×7–8 designs, ×2 budget arms,
×≥2 budget levels ⇒ ~300–500 independent optimizations, plus re-evaluation of every
Pareto set on N_test ≥ 1000 × ≥30-yr × ≥2 E_test variants. Whether that is affordable
is unknown until the running Anvil scaling experiment reports — no feasibility claim is
made here. *Recommendation:* once the scaling results land, build a one-page SU budget
table (per design: s/eval × NFE × K × S × arms/levels + re-eval cost) and size the
campaign against it (a tiering option is sketched in D4, to be used only if needed).
That table is also the pre-registration-style artifact a reviewer of a compute-heavy
paper looks for.

**A8. "Equal scenario-years" is not equal compute because of warm-up.** Every scenario
carries a 365-day warm-up excluded from metrics but not from simulation cost: overhead
is 20% at L=5, 3.9% at L=25, 2% at L=50, plus per-evaluation fixed costs (model build)
that scale with N, not N·L. Under equal metric-bearing scenario-years, short-window
designs are 15–20% more expensive than the budget model assumes, undermining the
"approximately equal wall-clock" claim. *Recommendation:* define B in *simulated* years
including warm-up (compute-honest) and report metric-bearing years separately; or fold
the correction into NFE_d. One sentence in the methods, but it must be defined before
NFE_d is derived.

**A9. Regret has the same self-reference problem the reference set already fixes.**
Best-in-scenario f*(s) is computed across all re-evaluated policies pooled over designs,
so each design contributes to the yardstick it is scored against (and designs granted
more NFE contribute more points — the same winner's-curse noted for reference sets in
methods §4.7). *Recommendation:* report regret against both pooled and
design-leave-one-out f*(s), mirroring the hypervolume treatment.

**A10. Satisficing thresholds θᵢ are being calibrated on the wrong distribution.**
The plan sets θᵢ from the random-DV experiment on the 77-yr historic trace, but the
search-time objectives are computed on short (5–10 yr) windows whose metric
distributions differ materially (a 5-yr window has no 77-yr tail; reliability metrics
concentrate near 0 or 1). Thresholds tuned on the long trace can saturate the
satisficing fraction (all-pass or all-fail) on short windows — no gradient.
*Recommendation:* set θᵢ from per-realization metric distributions computed on the
master ensemble at the chosen L (the ensemble sensitivity experiment already stores
exactly the needed matrix), not from the historic-trace experiment alone.

**A11. Archive contamination in the resampled design should be named and handled.**
Per-evaluation redraws make Borg's ε-archive retain solutions whose recorded objectives
were lucky draws; MM-Borg island migration propagates stale values. Trindade et al.
(2017) is precedent, but the noisy-EMO literature (Jin & Branke 2005; Fieldsend &
Everson 2015; Rakshit et al. 2017 — see B) is the standard framing a reviewer will
expect. *Recommendation:* no algorithm change; add one manuscript paragraph
acknowledging optimistic-bias archive selection under noise, and report the
search-vs-re-eval gap for this design as partially noise-driven (already anticipated in
experimental_design.md).

**A12. Smaller items.**
- *Historic design commensurability:* its search objectives are continuous temporal
  metrics while ensemble designs' are fractions; harmless (comparison is at re-eval)
  but state it once, plainly.
- *Hazard-filling K-replication:* the current selector is deterministic LHS+NN
  (`scengen`), not the stochastic annealer of methods §4.6 — so "K annealer replicates"
  is currently meaningless in code. Either implement the SA selector or replicate via
  master-draw/axis-bootstrap; do not let K=1-by-construction masquerade as stability
  (the docs themselves warn of exactly this).
- *Support points:* both prior reviews recommended it; it is the single cheapest
  addition that converts "hazard-filling beats random" into an identifiable mechanism
  claim (coverage vs designed selection). Include it at least in the equal-NFE arm.
- *Doc hygiene:* `experimental_design.md` still cites the retired note name
  `optimization_scenario_sampling_review.md`; `terminology.md` says re-evaluation is
  "workflow step 07" (now 08); `figure_sequence.md` is empty — a figure plan is a real
  gap for a results-heavy manuscript. (Mechanical fixes applied this session; figure
  plan → D7.)

---

## B. Zotero additions (collection ISYGLK35; recommend-only, verified against the current 56 items)

**Group 1 — cited in project docs but missing from the collection** (mostly from
`scenario_design_methods.md` §8): Minasny & McBratney 2006 (cLHS); Mak & Joseph 2018
(support points); Morris & Mitchell 1995 + Johnson et al. 1990 (maximin); Kennard et
al. 2010 (index selection); Kleywegt et al. 2002 (SAA); Hashimoto et al. 1982
(reliability/resilience/vulnerability); Herman et al. 2015 JWRPM ("How should
robustness be defined") and Herman et al. 2014 ("beyond optimality"); Law 2015
(*Simulation Modeling & Analysis*, terminating simulations); Rockafellar & Uryasev 2000
(CVaR); Fang et al. 2000 (L2 discrepancy); Székely & Rizzo 2013 (energy distance);
Vicente-Serrano et al. 2012 (SSI); Yevjevich 1967 (run theory); Richter et al. 1996
(IHA); Kirsch et al. 2013; Nowak et al. 2010; Vogel & Stedinger 1988 (record-length
statistics); Borgomeo et al. 2015 WRR 10.1002/2014WR016827 (+ the 2016 copula
companion) — duration–deficit hazard-space generation, the direct ancestor of the
hazard axes; Vakayil & Joseph 2022 (twinning); one variance-components/mixed-models
reference (e.g., Searle, Casella & McCulloch 2006) for the §5 replication model.

**Group 2 — from the verified 2020–2026 web scan (ranked):**

1. **Cohen, Zeff & Herman 2021**, *EMS* 141:105047, 10.1016/j.envsoft.2021.105047 —
   **top priority; scooping-adjacent.** Asks how *training-scenario properties* shape
   out-of-sample reservoir-policy robustness. Now in the collection (added 2026-07-09)
   with a full per-paper note (`literature/notes/Cohen et al. (2021).md`); the
   remaining action is engaging it in the gap statement — cite as motivation
   (composition moves robustness) and differentiate on the four deltas in the note
   (coverage design vs cluster unions; simulation-free hazard axes vs problem-driven
   baseline regret; master-scale synthetic pool vs 97 GCM traces; held-out DU test
   ensemble vs complementary halves of the same ensemble).
2. Zaniolo, Fletcher & Mauter 2023, *ERL* 18:054014, 10.1088/1748-9326/acceb5 —
   nearest generation-control relative. Now in the collection with a full per-paper
   note (`literature/notes/Zaniolo et al. (2023).md`); engaged in the gap statement,
   taxonomy (III.3), and methods novelty paragraphs. Deltas: hazard properties imposed
   at generation at four discrete SRI-typed levels vs selected from a realized master
   toward continuous coverage; OOS testing from the same designed families vs a
   held-out DU test ensemble. Its intensity-dominance finding supports hazard-axis
   screening. (Note: its generator is modified Borgomeo 2015 — raises the priority of
   the Borgomeo Group-1 addition; FIND 2024 is the generator descendant.)
3. Taylor, Brodeur, Steinschneider & Herman 2026, *JWRPM* 152(4),
   10.1061/JWRMD5.WRENG-7250 — training-forcing choice drives out-of-sample policy
   robustness; newest evidence for the premise.
4. Jin & Branke 2005, *IEEE TEVC* 9(3), 10.1109/TEVC.2005.846356 — canonical noisy-EA
   survey (grounds the resampled design, A11).
5. Fieldsend & Everson 2015, *IEEE TEVC* 19(1), 10.1109/TEVC.2014.2304415 — Pareto
   archives under noise (already cited in methods §4.4; not in collection).
6. Rakshit, Konar & Das 2017, *Swarm & Evol. Comput.* 33:18–45,
   10.1016/j.swevo.2016.09.002 — comprehensive noisy-optimization survey.
7. Steinmann, Auping & Kwakkel 2020, *TFSC* 156:120052, 10.1016/j.techfore.2020.120052
   — behavior-based (realized-dynamics) scenario discovery; exploratory-modeling analog
   of the input-vs-hazard distinction.
8. Guivarch et al. 2022, *Nature Climate Change* 12:428–435, 10.1038/s41558-022-01349-x
   — selecting/subsampling large scenario ensembles for robust insight.
9. Bertsimas & Mundru 2023, *Operations Research* 71(4), 10.1287/opre.2022.2265 —
   state-of-the-art problem-driven scenario reduction (already row IV.1 of the
   taxonomy; not in collection).
10. Hilbers, Brayshaw & Gandy 2023, *Applied Energy* 334:120624,
    10.1016/j.apenergy.2022.120624 — a-posteriori importance/time-series aggregation;
    follow-up to Hilbers 2019.
11. Brodeur, Delaney, Whitin & Steinschneider 2024, *WRR* 60:e2023WR034898,
    10.1029/2023WR034898 — synthetic ensembles built for policy training/evaluation.
12. Gozini et al. 2026, *WRR* 62:e2025WR042355, 10.1029/2025WR042355 — target-property
    multisite Kirsch-based generation (adjacent to the master-generation step).
13. Steinmann et al. 2025, *Socio-Environmental Systems Modelling* 7:18823 — diverse
    scenario-set selection as explicit search; methodological neighbor of the selector.
14. Optional: Dinot et al. 2023 (IJCAI, noisy-MOEA runtime theory); Bonham, Kasprzyk &
    Zagona 2025 (EMS vulnerability-analysis taxonomy — verify final DOI); van der
    Heijden et al. 2025, *WRR* 10.1029/2024WR037115 (scenario generation + optimal
    control).

**Scooping assessment.** No published study was found that space-fills a large master
ensemble of *realized* synthetic sequences in a multi-dimensional hazard-metric space
and uses the result as the MOEA *search* ensemble with held-out cross-design
re-evaluation. The three nearest neighbors each miss an element: Cohen 2021 (property
clustering, not coverage design), Bonham 2024 (space-filling subsampling, but post-hoc
ranking, already differentiated in the docs), Zaniolo 2023/2024 (hazard control at
generation, not subsampling of realized sequences). The gap statement in
`literature/scenario_design.md` should add the Cohen-2021 differentiation sentence and
name both distinctions (search-phase vs evaluation-phase; generation-control vs
realized-sequence subsampling). Herman-group and Zaniolo/Fletcher-group follow-ups are
the plausible scooping vectors — argues for a timely submission.

---

## C. Code edits needed for full HPC scale (verified file:line evidence)

**Blocking the RQ1 campaign (ordered by dependency):**

1. **Probabilistic designs → shared forcing master** (A1): re-wire
   `fixed_probabilistic_short/_long` and `resampled_probabilistic` in
   `src/scenario_designs.py` to index-draw from the forcing master (via
   `materialize_subset_from_master` / `with_indices_override`) instead of direct
   stationary KN generation.
2. **Multi-draw replication**: `resolve_search_spec(draw != 0)` raises
   (`src/scenario_designs.py:283-290`); K>1 ensemble draws — the unit of analysis — are
   impossible until draw-indexed subsampling + slug provenance are wired.
3. **Production master staging**: no `master_*` directory exists; sizes are test-scale
   env constants (`_MASTER_*`, `src/scenario_designs.py:53-56`: 200×1×5yr, subset 64).
   Needs the L/N_Θ/R decision (D3), then a chunked, `stream_only` step-02 run —
   generation is currently serial (should MPI-distribute chunks; determinism is
   partition-invariant by design).
4. **Budget→NFE derivation**: `MOEAConfig.budget_scenario_years` is `None` everywhere
   and nothing computes `NFE_d = B/(N_d·L_d)` (methods §5); `production` config is
   schema-only (`src/moea_config.py:255-266`) while real runs use `mm_full` sized for
   single-trace evals. Implement per-design NFE derivation + a production MOEA config
   sized from the Anvil scaling results.
5. **Ensemble aggregation + thresholds**: `objectives_ensemble.py` is satisficing-only
   with placeholder thresholds (`:138-148`); implement the operator decided in D2 (mean
   /percentile operators are currently absent) and fraction-appropriate epsilons.
6. **Held-out test ensemble**: no production reeval preset exists (defaults to
   `historic_single`); build E_test generation + staging + `NYCOPT_REEVAL_ENSEMBLE_PRESET`
   entry, including the ≥2 variants (one structurally different, A6).
7. **`input_stratified` staging path**: expects `master_5yr_n200/forcing_profiles.npz`
   (`src/scenario_designs.py:334-339`); resolved by item 3 (regenerate under the
   production slug) — note the older `kn_inputlhs_*` masters use an incompatible slug.
8. **Support-points selector** (A12): net-new in `scengen/subsample.py` + a
   `support_points` design registry entry.
9. **Chunk-pipeline hardening before first big run**: real Pywr-DRB simulation over a
   chunk has never run (chunk re-eval is mock-tested only); per-chunk pywrdrb input
   staging (predicted inflows) not wired; `await_all_done` 1800 s deadline too short for
   HPC sims.
10. **Env files for the comparison designs**: only `historic`, `hazard_filling`,
    `scaling_stationary` have envs; add one per campaign design so runs stay
    identifier-only (no ad-hoc `--export`).

**Cheap cleanups:** `_DEFAULT_MOEA_SLUG_CONFIG="production"` vs actual `mm_full` means
every production slug gets an `_mm_full` suffix (`config.py:606`, `:834-835`) — either
rename mm_full→production once numbers are final, or change the default; stale
runnable-designs message in `src/mmborg.py:92-95` (omits input_stratified /
hazard_filling); provisional sizing duplicated between `scenario_designs.py` and
`supplemental_config.py` (single source of truth); stale Hopper references in
`moea_config.py` notes.

---

## D. Discussion points — decision log (updated 2026-07-09)

1. **Primary comparison metric set** (A3) — **DECIDED.** Robustness (satisficing) +
   regret on the held-out ensemble are primary. The overfitting-gap regression is
   demoted to the supplement and reframed as a coverage → re-evaluated-robustness
   association (drops the coverage-weighted in-sample term). Notes reconciled:
   `objective_definitions.md` §3, `scenario_design_methods.md` §6b.
2. **Search-time across-realization operator** (A2) — **DECIDED: satisficing
   fraction confirmed** (threshold forms more stable than absolute values, Quinn et
   al. 2017; fastest-converging, Bonham 2024; magnitude carried by temporal metrics,
   frequency-of-unacceptable by the across-realization stage; Borg precedent in the
   Research Triangle lineage). Guardrails attached (`objective_definitions.md` §2):
   (b) common fraction-unit ε across designs — agreed; (c) θ_i calibrated on the
   master at L=10 and saturation-screened under probabilistic AND hazard-filled arms
   — revised two-arm sensitivity experiment planned
   (`ensemble_objective_sensitivity_experiment.md`); (d) operator-agreement panel
   retained as an SI diagnostic. **Open sub-decision D2(a)** — REFRAMED 2026-07-09
   after the user rejected the initial options (larger NL / pooled-year / 10-yr
   block-scoring as proposed) and relaxed the constraint: search objectives need
   NOT be commensurable across designs (re-evaluation on E_test is the only
   comparison point), and formulations should follow citable published practice.
   **Literature extraction (three agents, 2026-07-09; Zotero full texts + web):**
   - During search, the literature uses a **per-objective operator mix**, not one
     family: fraction-of-realizations *frequency* for reliability/restriction
     objectives (Zeff 2014 Eq. 2; Trindade 2017 Eq. 16; Gold 2023 Table 4;
     Kasprzyk 2013 "likelihood"; Hamilton 2022 J_hedge); **mean** across
     realizations for cost/magnitude objectives (everyone); exactly **one tail
     objective** as a percentile across realizations (worst-1% VaR in
     Zeff/Trindade/Gold; Q95 in Hamilton 2022; worst-1st-percentile in Quinn
     2017/2018). No paper optimizes a multi-criteria satisficing fraction over all
     objectives during search (purest satisficing-in-search precedent = domain
     criterion over 50 LHS scenarios, Bartholomew & Kwakkel 2020; Giuliani &
     Castelletti 2016 explicitly reject satisficing for operations problems).
   - **Year-unit pooling is precedented**: Trindade restriction frequency divides
     by (N_r · N_y); Quinn 2018 slices one continuous 1000-yr record into
     consecutive 1-yr scoring units with inherited state ("representative"
     initial-condition distributions) and applies worst-1st-percentile across
     units; Borgomeo et al. 2016 optimize restriction-frequency risk pooled across
     long traces. Hamilton 2022 supplies the vocabulary: within-record **time
     aggregation** + across-record **noise filtering** as separable choices.
   - **Few-record caveat**: no verified study optimizes a percentile/tail operator
     over fewer than ~50 records; with few records the precedents are mean,
     pooled frequency, worst-record, or domain criterion (Mortazavi-Naeini 2015
     worst-scenario/regret; Quinn 2017 WC arm shows within-record mean +
     across-record extreme is pathological: "may mask particularly bad years").
   **RESOLVED (user-confirmed 2026-07-09): two-layer, literature-anchored scheme on
   pooled annual units** — per (realization × water-year) unit compute annual
   metrics; across the pooled NL unit-years apply per-objective operators mirroring
   published forms (reliability = frequency of failure-years, Zeff 2014/Trindade
   2017/Gold 2023; deficit tails + storage = worst-1st-percentile unit-year, Quinn
   2017 WP1/2018; flood = expected annual count with P99 variant registered). Long
   records are sliced into consecutive annual units with inherited state (Quinn
   2018) — no special-casing, identical denominator NL across all designs, native-
   unit ε for the mean/percentile objectives, thresholds reduced to Decree-anchored
   annual failure criteria. Within-record unit dependence disclosed (ESS/clustered
   SE). Authoritative spec: `objective_definitions.md` §2. This supersedes the
   earlier all-objective satisficing-fraction family; the sensitivity experiment's
   role shrinks to saturation screening of the failure criteria per composition,
   the flood operator choice, ε values, P99 stability at campaign NL, and
   annual-vs-realization unit validation.
3. **L and window construction** (A5) — **DECIDED: L = 10 yr**, kept editable as a
   single config constant (code edit: unify the scenario-length constant, currently
   split across `scenario_designs.py` env constants and `supplemental_config.py`).
   Disjoint windows, fixed 0.80 initial storage + 365-day warm-up unchanged; L′ = 50
   retained for the length contrast (pending D2(a)). Caveat recorded: downstream
   artifacts are L-conditional (changing L ⇒ regenerate master); settle D5 before
   generating the L=10 master.
4. **Campaign scope** (A7) — open; size against the SU table once the Anvil scaling
   results land. Sketch if trimming is needed: Tier 1 = 5 designs (historic,
   fixed_short, resampled, input_stratified, hazard_filling) × K=5 × S=2 × equal-NFE
   at 2 budget levels, `ffmp` only; Tier 2 = +fixed_long, +support_points,
   equal-scenario-years arm; RQ3 (`ffmp_N`) on the winning design afterward. Also:
   is `hazard_filling_absolute` a campaign member or a sensitivity?
5. **Persistence axis** (A4) — **DECIDED 2026-07-09: keep historical persistence.**
   The forcing space remains monthly mean + CV only; interannual persistence is
   inherited from the historically-fitted Kirsch structure in every scenario.
   Consequences accepted and to be stated in the manuscript: (i) all claims are
   scoped to "under historical persistence" (deep uncertainty spans volume,
   seasonality, and variability — not multi-year drought persistence); (ii) the
   hazard-axis screen should verify which axes the forcing space can actually move
   (reachability), and drought-persistence-driven axes are expected to reflect
   generator variability rather than forcing; (iii) a persistence axis is named
   future work. This unblocks master generation (with D3's L = 10).
6. **Test-ensemble family** (A6) — open; include one non-Kirsch–Nowak E_test variant?
   Which generator?
7. **Figure plan** — open; `figure_sequence.md` is empty post-pivot; draft the
   results-figure skeleton before the campaign so the runs persist what the figures
   need (especially runtime diagnostics and coverage-vs-outcome panels).
