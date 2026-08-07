# Alternative C — Parallel-lane controlled experiment (funnel): notes

## Layout rationale

- Portrait 1300 x 1700, five full-width horizontal stage bands read top to
  bottom; three lanes (HIST gray | PS blue | HF vermillion) descend through
  bands 1-3 at fixed x-centers (285 / 675 / 1065), so lane identity is carried
  by both position and color from construction through search.
- Band 2 is the only visually "hot" band: warm tint (#FFF6EF) + vermillion
  border + right-aligned tag "only stage that differs". Bands 1, 3, 5 use the
  neutral light-gray band style; band 4 is an isolated purple block rather
  than a full-width band, so the funnel convergence reads as a narrowing.
- Left annotation rail: gray "held fixed" ticks beside bands 1 and 3,
  vermillion "varied" beside band 2, one purple "common test" tick spanning
  bands 4-5 (re-evaluation and comparison share the common-test status).
- Lane arrows are colored by destination design; the three funnel arrows
  converge onto a single purple block, and only one purple arrow continues to
  band 5 — the geometry alone carries "one comparison point".

## Content placement decisions

- Generator glyph = one wiggly trace with a thick light-gray envelope stroke
  (cheap to restyle in Inkscape: two paths sharing identical d).
- HF lane in band 2 is a 3-step vertical mini-pipeline (cloud -> selection
  frame -> tile stack); HIST and PS glyphs are vertically centered in the
  same band so the extra HF machinery is visible as extra machinery.
- Selection glyph: stylized 2-D projection of the 6-D space — gray candidates
  dense in the low-hazard corner with sparse tail members, 3x3 grid of black
  LHS crosses, 9 vermillion dots offset slightly from the crosses (NN snap)
  and occupying the severe corners. Labels ("LHS anchors / + NN snap / 6-D
  hazard space") sit left of the frame to keep the frame clean.
- Cardinality chips are pill-shaped, stroke-colored per design, aligned on one
  row at the bottom of band 2: 1 x 77 yr | N = 100 x L = 10 yr |
  P = 10^6 -> 100 x 10 yr.
- Band 3 replication: K = 3 is drawn literally as a 3-deep card stack (front
  card + two offset backs) for PS and HF; the front card is opened up to show
  the S = 2 seed sub-cards merging into a per-draw Pareto mini-scatter. No
  ellipsis needed — 3 stacked cards is exactly K. HIST gets a single card
  (no backs). Matched settings (36 DVs, 8 objectives, MM Borg, 500k NFE,
  trimmed Pywr-DRB) appear once in a full-width pill above the cards.
- Purple block: title + "1,000 SOWs x 25 x 50 yr" + "DU enters here only" +
  "held out of search - trimmed model", with a 3-D LHS-cube glyph (CMIP6
  forcing amplitudes) on the right.
- Band 5: one gray bar (HIST), three blue and three vermillion draw-level
  bars, dashed "2017 FFMP baseline" reference line crossing all groups, and a
  small "one bar per draw" note. Bar heights are placeholders and encode no
  claimed result ordering beyond illustration.

## Open questions for the author

1. HIST replication chip reads "1 draw x S = 2 seeds" — confirm HIST is run
   with 2 seeds; otherwise change to "1 search" and drop the seed sub-cards
   from the HIST card.
2. Bar heights in band 5 are placeholders; decide whether the rough draft
   should stay deliberately neutral (equal heights) instead of the current
   mild HF > PS > HIST suggestion.
3. Tail over-representation token (>= 31% above candidate p90 vs 10% i.i.d.)
   is omitted for space; it could be added as a small annotation beside the
   HF selection frame if wanted.
4. Band-1 arrows imply all three lanes descend from the generator; strictly,
   HIST is the fitted record itself. If that nuance matters, dash the HIST
   arrow or start it from a small record mark inside band 1.
5. The six hazard-axis names (deficit volume, peak depth, onset rate,
   recovery rate, flood peak, pulse duration) are not listed; add as a
   two-line micro-caption under the selection frame if Section 3 cross-
   reference is not considered sufficient.
