# Figure 2 (Methods Diagram) — Rough-Draft Design Plans

Three deliberately distinct design concepts for the graphical experimental-design
overview referenced in Section 3 of the manuscript. Rough drafts only: each SVG
scopes layout and content, not final styling. Finalization happens in Inkscape.

## Content inventory (what any variant must communicate)

Derived from Section 3 of the manuscript draft:

1. **Shared stationary population.** Kirsch-Nowak generator fitted to the
   reconstructed 77-year record (WY 1946-2022), unperturbed.
2. **Three scenario designs for the search ensemble** (the varied factor):
   - **HIST** — the observed record, 1 realization x 77 yr (reference, unmatched).
   - **PS** — N = 100 x L = 10 yr, i.i.d. from the generator (exact control).
   - **HF** — N = 100 x L = 10 yr selected from a P = 10^6-member i.i.d.
     candidate ensemble by LHS anchors + nearest-neighbor assignment in the
     range-scaled 6-D hazard space (the contribution).
3. **Hazard space.** 6 selection axes (drought: deficit volume, peak depth,
   onset rate, recovery rate; flood: peak magnitude, pulse duration), computed
   on aggregate NYC inflow, simulation-free, robust p1/p99 range scaling;
   tail over-representation (>= 31% of members above the candidate p90 vs 10%
   under i.i.d.).
4. **Matched many-objective search.** 36 FFMP decision variables, 8 objectives,
   trimmed Pywr-DRB, MM Borg, 500k NFE; K = 3 draws x S = 2 seeds per matched
   design; seeds merged to per-draw Pareto-approximate sets.
5. **Common DU re-evaluation.** One re-evaluation ensemble: 1,000 SOWs (LHS
   over the 3-D CMIP6 harmonic forcing-amplitude space) x 25 realizations x
   50 yr, trimmed model; never seen during search; DU enters HERE only.
6. **Robustness comparison.** Satisficing fraction (Starr domain criterion),
   per-objective decomposition, criterion sweep; signed improvement vs the
   default 2017 FFMP policy; comparison at the draw level.

Key structural claims the geometry itself should carry:
- PS and HF differ ONLY in the selection rule (exact-control property).
- Everything else (generator, N, L, objectives, algorithm, NFE) is held fixed.
- All designs funnel into ONE common re-evaluation ensemble — the only
  cross-design comparison point.

## Shared conventions (all three variants)

- Terminology: "scenario design" (never arm/treatment), "re-evaluation
  ensemble" (never master ensemble, never E_test), "flood exceedance"
  (never severity). Tokens: HIST, PS, HF.
- Persistent design colors (colorblind-safe, carried into later figures):
  HIST = gray (#666666), PS = blue (#4477AA), HF = vermillion (#EE7733),
  DU re-evaluation = purple (#AA3377), held-fixed/neutral structure = light gray.
- Minimal text: bold short titles, math/cardinality tokens (N = 100, L = 10 yr,
  P = 10^6, 500k NFE, K = 3 x S = 2, 1,000 SOWs x 25 x 50 yr), no sentences.
- Show data objects as stylized miniatures (dot clouds, envelope bands, Pareto
  mini-scatters), not empty labeled boxes.
- Plain SVG, Inkscape-editable: `<text>` elements (no paths-as-text), grouped
  layers with descriptive ids, no external assets or fonts beyond
  Arial/Helvetica fallback.

## Alternative A — Lettered panel pipeline (Reed-lineage classic)

*Lineage: Zatarain Salazar 2017 Fig. 7 + Bonham 2024 Fig. 1 + Hamilton 2024 Fig. 3.*

Three lettered panels in one composition:
- **(a) Scenario designs** (top band, left-to-right): generator glyph
  (envelope band around historic trace) -> three parallel design tracks
  (HIST / PS / HF), each ending in a small "search ensemble" tile stack;
  HF's track passes through a candidate-pool glyph (large gray dot cloud).
- **(b) Hazard-filling selection** (zoom inset from HF track): 2-D stylized
  hazard-space scatter — gray candidate cloud dense at origin, LHS anchor
  crosses on a coarse grid, arrows snapping anchors to nearest candidates,
  selected points in vermillion filling the severe corners.
- **(c) Search -> common re-evaluation -> robustness** (bottom band):
  per-design search blocks (36 DVs, 8 obj, 500k NFE; K x S replication as
  stacked cards) -> per-draw Pareto mini-scatters -> all arrows converging
  into ONE purple re-evaluation ensemble block (SOW LHS mini-cube) ->
  robustness bars by design.

Emphasis: faithful graphical outline of Section 3; each panel maps to a
subsection. The most conventional, lowest-risk option.

## Alternative B — Three-spaces conceptual backbone

*Lineage: the manuscript's own input/hazard/outcome-space taxonomy (Section 1
P4, 3.2.3); decision-scaling exposure-space diagrams; Herman 2020's
conceptual altitude.*

Three large column panels ARE the figure: **input space | hazard space |
outcome space**, with the workflow drawn as flows through the spaces.
- **Input space** (left): generator parameter box with the stationary point
  marked; below it, the 3-D CMIP6 forcing amplitude box (purple) with LHS
  points — visually separated, connecting ONLY to re-evaluation (a long
  purple arrow that bypasses search), making "DU enters only at re-evaluation"
  geometric.
- **Hazard space** (center, hero panel): one large scatter shown twice
  (small-multiple pair, identical frames): PS subset = blue dots following the
  gray population density (center-heavy); HF subset = vermillion dots covering
  the frame including severe corners. HIST appears as a single gray point/track.
  Six axis tokens listed beneath.
- **Outcome space** (right): search-objective mini-scatter with Pareto fronts
  per design -> re-evaluated satisficing axis (robustness), with the common
  re-evaluation ensemble block between them.
Flow arrows run left-to-right THROUGH the spaces: generate -> characterize ->
select -> simulate/search -> re-evaluate.

Emphasis: the conceptual contribution (where the hazard space sits, and that
selection happens there before any simulation). Most distinctive; least like
a step-by-step pipeline.

## Alternative C — Parallel-lane controlled experiment (funnel)

*Lineage: Hadjimichael 2020 Fig. 3 (columns + cardinalities) rotated to
vertical; Trindade 2017's held-fixed vs varied encoding; CONSORT-style
experiment flow.*

Vertical top-to-bottom flow through full-width horizontal stage bands; three
lanes (HIST | PS | HF) descend through them:
1. **Stationary population** band (shared, spans all lanes): generator glyph.
2. **Search-ensemble construction** band — the ONLY stage where lanes differ;
   highlighted border/tint. HIST: record strip; PS: i.i.d. draw tile stack;
   HF: candidate cloud -> hazard-space selection mini-glyph -> tile stack.
   Cardinality chips (1 x 77 yr | 100 x 10 yr | 10^6 -> 100 x 10 yr).
3. **Matched search** band (shared settings printed once across lanes: 36 DVs,
   8 objectives, MM Borg, 500k NFE); K = 3 draws x S = 2 seeds drawn as
   stacked cards per matched lane; per-draw Pareto set glyphs.
4. **Funnel**: all lanes converge into one purple **common re-evaluation
   ensemble** block (1,000 SOWs x 25 x 50 yr).
5. **Robustness comparison** band: satisficing bars per design per draw +
   signed-improvement reference line for the 2017 FFMP baseline.
A thin left-edge annotation rail marks "held fixed" (gray) vs "varied"
(accent) per band.

Emphasis: the controlled-experiment logic — what is matched, what varies,
where the single comparison point is. Strongest at carrying the
exact-control claim.

## Deliverables

- `alt_A_pipeline_panels.svg`
- `alt_B_three_spaces.svg`
- `alt_C_parallel_lanes.svg`
- one short `alt_*_notes.md` per variant (content decisions + open questions)
