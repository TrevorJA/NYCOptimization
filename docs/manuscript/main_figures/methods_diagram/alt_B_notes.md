# Alt B — Three-spaces conceptual backbone: draft notes

Rough draft of `alt_B_three_spaces.svg` (1800 x 1300 landscape). All elements are
plain SVG primitives with `<text>` labels; groups are named per column
(`input-space`, `hazard-space`, `outcome-space`) and per major element
(`generator-box`, `cmip6-box`, `ps-frame`, `hf-frame`, `pareto-scatter`,
`reevaluation-block`, `robustness-bars`, `du-bypass-arrow`) for Inkscape layer work.

## Layout rationale

- **Column widths encode emphasis**: hazard space gets ~40% of the width (720 px)
  as the hero panel; input and outcome flank it. The main workflow reads
  left-to-right at mid-height: `generate -> (characterize / select) -> search ->
  re-evaluate -> compare`.
- **Small-multiple pair carries the exact-control claim**: the PS and HF frames
  are pixel-identical (same 300 x 300 frame, same gray candidate cloud via a
  single `<use>` of `#pool-cloud`, same HIST diamond at the same coordinates).
  Only the colored overlay differs — PS blue dots follow the corner-heavy
  population density; HF vermillion dots are a jittered 40-stratum LHS covering
  the frame including the severe corners. Nothing else about the frames varies.
- **"Selection before any simulation"** is carried by position (both frames sit
  upstream of the `search` arrow) and by the sub-caption token
  `characterize . select — simulation-free` under the column title.
- **"DU enters only at re-evaluation"** is carried by the purple bypass: the
  CMIP6 box sits below a dashed divider in the input panel (visually severed
  from the stationary generator machinery) and its only outgoing edge is the
  long purple arrow routed along the figure bottom, entering the common
  re-evaluation block's left edge. It never touches the hazard panel or the
  search arrow. Label: `re-evaluation only`.
- **One comparison point**: a single purple block between the Pareto scatter and
  the robustness bars; the search-side arrow (top) and the DU arrow (left) meet
  only there.

## Content placement decisions

- HIST appears three times, consistently gray: the observed-record strip in
  input space (1 x 77 yr token), a diamond marker inside both hazard frames
  (diamond, not circle, so shape — not color alone — separates it from PS), and
  a gray Pareto front / bar group downstream. A thin gray arrow from the record
  strip into the hazard panel shows the record is characterized in the same
  space.
- PS sub-label reads `i.i.d. from generator` (not "from pool") — PS is the exact
  i.i.d. control drawn from the stationary generator; the gray cloud under the
  PS frame therefore depicts the population density, while the same cloud under
  HF depicts the P = 10^6 candidate pool. One legend entry
  (`candidate pool . P = 10^6`) covers both readings; flag if that conflation
  bothers you.
- Matched-search tokens (`36 DVs . 8 objectives`, `trimmed Pywr-DRB . MM Borg .
  500k NFE`, `K = 3 draws x S = 2 seeds`) sit once at the top of the outcome
  column, i.e., stated once because they are held fixed across designs.
- The six selection axes are pill tokens grouped `drought` (deficit volume, peak
  depth, onset rate, recovery rate) and `flood` (peak flow magnitude, pulse
  duration), with a scaling token (`6-D . robust p1/p99 range scaling .
  aggregate NYC inflow`) beneath.
- Robustness glyph: three bars per design = the K = 3 draws (comparison at the
  draw level), on a 0-1 satisficing-fraction axis with a dashed `2017 FFMP`
  reference line. Bar lengths and the reference position are placeholders, not
  results.
- Pareto fronts are offset/jittered to interleave (PS and HF cross) so the
  methods figure does not pre-announce a winner; HIST drawn slightly outward.
- All dot positions are seeded pseudo-random stylizations, no real data.

## Open questions for the author

1. **Hazard-frame axes**: currently abstract (`severe ->` on both axes), i.e., a
   generic 2-D projection of the 6-D space. Alternative: name two real axes
   (e.g., deficit volume vs. peak flow magnitude) at the cost of implying a
   specific projection.
2. **Tail over-representation token**: the >= 31% vs 10% above-p90 contrast is
   not shown. It could go under the HF frame as one more token if you want the
   quantitative selling point in the figure.
3. **HIST record -> hazard arrow**: keep or drop? It is the least essential edge
   and the panel works without it.
4. **LHS anchor mechanics**: the HF frame shows only the selected points; anchor
   crosses + snap arrows were left to Alt A's zoom inset. Add a few anchors here
   if you want the mechanism visible in this variant too.
5. **Placeholder values**: satisficing bar lengths, the 2017 FFMP reference
   position, Pareto front shapes, and the HIST marker location in hazard space
   are all invented — replace or restyle at will.
6. **Palette caveat** (carried from the shared conventions): HIST gray vs. PS
   blue is a low-separation pair for grayscale printing; the diamond-vs-circle
   shape coding and direct labels are the mitigations. Consider a lighter gray
   for the candidate cloud if the HIST gray ever reads as pool members.
