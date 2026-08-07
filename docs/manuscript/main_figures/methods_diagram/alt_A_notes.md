# Alternative A — Lettered panel pipeline: draft notes

## Layout

- 1800 x 1300 landscape, three panels: (a) top-left band (scenario designs),
  (b) top-right framed inset (hazard-filling selection zoom), (c) full-width
  bottom band (matched search -> common re-evaluation -> robustness).
- (a) reads left-to-right: generator glyph (envelope band + gray historic
  trace) -> branch -> three horizontal tracks (HIST / PS / HF, top to bottom)
  -> aligned "search ensembles" column at a common x, so the three outputs
  are visually commensurable.
- (b) is connected to the HF selection box by two light zoom-wedge lines
  anchored on the box's right corners and the inset frame's left corners.
- (c) uses one row per design with identical row anatomy (mini ensemble icon
  -> search-card stack -> per-draw Pareto mini-scatter), then all three rows'
  arrows converge on the single purple re-evaluation block, then one purple
  arrow to the robustness bars.

## How the two structural claims are carried

1. **PS/HF exact control**: PS and HF rows share the dashed
   "only difference between PS and HF" frame spanning the construction step.
   Inside it, PS is a bare arrow ("direct i.i.d. draw x N"); HF inserts the
   candidate pool (P = 10^6 dot cloud) and the vermillion selection box.
   Everything outside the dashed frame is identical between the two tracks
   (same generator, same tile-stack geometry, same N = 100 x L = 10 yr token).
2. **Single comparison point**: exactly one purple block in the whole figure;
   HIST/PS/HF arrows physically converge into it before the robustness bars,
   and "DU enters here only" is printed inside it.

## Content placement decisions

- Held-fixed search settings (36 DVs, 8 objectives, trimmed Pywr-DRB, MM Borg,
  500k NFE) printed once in a neutral gray banner above the (c) rows rather
  than repeated per card — repetition would dilute the matched-design claim.
- HIST search block drawn as a single card labeled "S = 2 seeds" +
  "unmatched reference" (no K = 3 stack), vs three offset cards for PS/HF.
- Tail over-representation token (>= 31% vs 10%) and the six named hazard
  axes live in (b) captions; (a)'s selection box carries only the compressed
  token "P = 10^6 -> N = 100, 6-D hazard space".
- Panel (b) axes are labeled generically ("hazard axis 1/2, range-scaled
  p1–p99") since the 2-D scatter is a stylization of the 6-D space; the
  caption names all six axes.
- LHS anchors drawn as 8 crosses on an 8x8 faint grid (one per row/column, a
  genuine Latin-hypercube pattern); 4 have explicit snap arrows to their
  nearest candidate; 14 vermillion selected dots cover all four frame corners.
- Robustness bars: one gray bar (HIST), three blue and three vermillion bars
  (one per draw), dashed horizontal reference for the default 2017 FFMP
  policy (signed improvement read as height above the line). Bar heights are
  placeholders, not data.
- Re-evaluation block shows a 3-D cube glyph (LHS over the 3-D CMIP6 forcing
  box) with purple SOW dots, plus the 1,000 SOWs x 25 x 50 yr token.

## Open questions for the author

1. HIST replication labeling: is "S = 2 seeds / unmatched reference" the
   intended treatment, or does HIST also get K-style re-rolls of nothing but
   seeds? Adjust the card stack if so.
2. Panel (b) shows 2 of 6 axes with generic labels — prefer naming two real
   axes (e.g., deficit volume vs flood peak) on the frame instead?
3. Should the (a) ensemble tile stacks visually connect down into the (c)
   mini ensemble icons (small connector arrows across panels), or is the
   icon-echo sufficient?
4. The dashed only-difference frame currently encloses PS's entire arrow
   span; it could be tightened to just the pool + selection column width.
5. Robustness caption mentions "criterion sweep" — keep, or drop until the
   satisficing criterion values are finalized?
6. Bar ordering within groups is draw index; if draws should be visually
   paired across PS/HF (same draw = same pool re-roll is false — draws are
   independent per design), the current independent ordering is correct.
