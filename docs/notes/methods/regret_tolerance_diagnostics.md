# Regret-Tolerance Diagnostics: fixing $\tau$ and $\delta$ before the result

*Sets the two free parameters of the incumbent-relative regret comparison
(`objective_definitions.md` §3.2b): the no-harm tolerance $\tau_i = k\,\epsilon_i$
($\epsilon_i$ = the ANNUAL-UNIT epsilon of
`objectives_ensemble.ENSEMBLE_OBJECTIVES`) and the non-inferiority margin
$\delta$ on `no_harm_freq_tau`. The metric substrate is the per-SOW
annual-unit objective value — the search objective recomputed per
deeply-uncertain state. Implementation:
`scripts/supplemental/regret_tolerance_diagnostics.py`, settings in
`supplemental_config.py` (`RTOL_*`), launcher
`workflow/supplemental/regret_tolerance_diagnostics.sh`.*

---

## 0. Why this note exists

$\tau$ and $\delta$ are the only quantities in the regret comparison that the
data do not determine. They are therefore pre-registration quantities, and the
whole of the RQ2 answer can be moved by choosing them badly or late.

Two failure modes, and they pull in opposite directions.

**Circularity.** A tolerance read off the distribution of candidate-policy regret
guarantees its own answer. This project has already ruled on the identical
hazard once: `robustness_threshold_diagnostics.md` §0b rule 4 reports the
baseline's own SOW-mean quantiles as candidate satisficing thresholds in every
table and **never adopts them**, because a threshold placed on a feature of the
distribution it is meant to measure fixes the fraction it will return. The same
rule binds here, and binds harder, because $\tau$ enters a conjunction over eight
objectives.

**Insensitivity.** The RQ2 hypothesis is a *non-inferiority* claim — hazard
filling is *not worse* in regret — and non-inferiority claims are flattered by
anything that blunts the measurement. A loose tolerance drives $\Pi_\tau
\rightarrow 1$ for every design, at which point "not worse" is true by
construction and means nothing. Every choice below is therefore biased toward
the *discriminating* end, and §4 adds the positive control that a non-inferiority
claim requires.

The resolution is that **$k$ is not chosen at all in the main text — the
$k$-curve is the deliverable**, exactly as the satisficing criterion is reported
as a sweep rather than a value (Quinn et al. 2020). One rung still has to carry
the headline sentence, and §3 fixes *which* rung by a rule that touches no
campaign result. $\delta$ is likewise derived by a pre-registered rule from
nuisance variance only (§5).

---

## 1. Admissibility: what may anchor what

| Tier | Anchor depends on | $\tau$ | $\delta$ | Why |
|---|---|---|---|---|
| **A** | Estimator noise and measurement resolution of the incumbent's own cube | ✅ | ✅ | A *floor*, not a preference. Below it the metric reports Monte Carlo noise as harm. Computable before any policy exists. |
| **B** | External decision increments (Decree quantities, the observed record) | ✅ | ❌ | Legitimate for a magnitude; says nothing about how large a *difference between designs* must be to matter. |
| **C** | The distribution of candidate-policy regret | ❌ | ❌ | **Circular.** This is the quantity under test. |
| **D** | Within-design nuisance variance (seed pairs, draw pairs) | — | ✅ | Uses campaign data but only its *null* part; it fixes the scale of "no difference" without touching the direction of the answer. |

Tier D is the one that needs stating carefully, because it is data-dependent and
still admissible. A within-design difference — two seeds of one draw, or two
draws of one design — carries **no design effect by construction**. Anchoring
$\delta$ on it is a Least-Significant-Difference construction: the pre-registered
object is the *rule*, and the number it returns is a function of the nuisance
variance alone. It cannot flip the sign of the comparison, only its resolution.

---

## 2. Pass A — the noise floor (needs only the incumbent)

Every quantity here is a reduction of the step-05 incumbent cube (plus the
staged $E_{\text{test}}$ forcing profiles for the $m$ join), so **pass A can
and should run as soon as that cube lands, long before any search finishes.**
Running it early is not a convenience: a tolerance fixed after the campaign
contrast is visible is not a pre-registered tolerance, whatever value it takes.

The question is *how small can $\tau$ be before it measures the estimator?* Under
the null that a policy is operationally identical to the incumbent, the per-SOW
difference $D_i(x,\theta)$ is pure estimator noise (finite $R$ realizations
pooled per SOW), and any $\tau$ below its scale reports that noise as harm.

**Floor estimator.** The persisted cube carries ONE value per (SOW, objective)
— the SOW's $R$ realizations already pooled through the §2 unit operator — so
the Monte Carlo noise of $J(\theta)$ cannot be recovered from per-realization
spread (none is persisted). It is instead BOUNDED from the incumbent's per-SOW
values: sort the SOWs on the dominant forcing axis $m$ (theta joined on the
SOW label), partition into consecutive bins of `RTOL_M_BIN_SIZE` SOWs, and
take $\sigma_i$ = the median across bins of the within-bin standard deviation
— the local (binned-in-$\theta$) spread of $J$ around its forcing response.
Then

$$
\tau_i^{\text{floor}} \;=\; z \cdot \sqrt{2} \cdot \sigma_i,
\qquad z = 1.645 \;\Rightarrow\; \text{5\% false-harm per (objective, SOW)}.
$$

**The floor is an unpaired UPPER BOUND — this must be stated wherever it is
used.** Two stacked conservatisms: the within-bin spread still contains real
structure (the $r_1$/$r_2$ response and the residual $m$-trend inside each
narrow bin), and the real comparison is paired — a policy and the incumbent
are simulated on the *same* inflow sequences, so their difference cancels most
of the shared natural variability and the true floor is smaller, possibly much
smaller. The paired floor cannot be estimated without a second policy on the
test ensemble, so pass A reports the unpaired bound explicitly labelled as an
upper bound and pass B replaces it as soon as any policy cube exists.

The direction of that conservatism matters and is not benign: an overstated floor
pushes $k$ *up*, toward less discrimination, which is the direction that flatters
a non-inferiority claim. It is a bound to be tightened, not a safe default.

**Outputs.** `rtol_noise_floor.csv` (per objective: $\sigma_i$, the unpaired
null SD, $\tau^{\text{floor}}$, $\epsilon_i$, and
$k^{\text{floor}} = \tau^{\text{floor}}/\epsilon_i$),
`rtol_ladder_shapes.csv` (§2b), `rtol_floors.json` (the floor vector, consumed by
`robustness.tau_ladder(floors=...)`).

### 2b. The ladder's *shape* is a separate choice from its scale

$\tau_i = k\,u_i$ shares **one** $k$ across eight objectives, so the unit $u_i$
decides what a rung means on each axis. Using $u_i = \epsilon_i$ tacitly assumes
every epsilon is comparably placed relative to its own estimator noise. It need
not be: illustrative pass-A numbers on a synthetic incumbent with realistic
spreads give

| objective | $\epsilon_i$ | $\tau_i^{\text{floor}}$ | $\tau^{\text{floor}}/\epsilon$ |
|---|---|---|---|
| `nyc_delivery_reliability_annual` | 0.02 | 0.009 | 0.5 |
| `downstream_flood_exceedance_annual` | 0.3 | 1.2 | **3.9** |

A flood epsilon sitting well *below* its noise floor means that on that axis
$k = 1$ is inside the estimator's noise while on the reliability axis it is
far outside it. A single rung then means two different things, and the
eps-shape rule of §3 responds by pushing the headline rung upward — defensible
for flood and loose for everything else.

Three candidate shapes, reported side by side in `rtol_ladder_shapes.csv`:

| shape | $u_i$ | Behaviour |
|---|---|---|
| `eps` | $\epsilon_i$ | fails wherever $\epsilon_i < \tau_i^{\text{floor}}$ |
| `floor` | $\tau_i^{\text{floor}}$ | equalizes the false-harm rate, discards resolution |
| **`max`** | $\max(\epsilon_i, \tau_i^{\text{floor}})$ | **recommended** |

`max` keeps epsilon where resolution binds and the floor where noise binds, so
$k$ means the same thing on every axis, and it can never fall below $\epsilon_i$.
It stays admissible under §1 because the floor is a Tier-A quantity — a property
of the incumbent's cube alone, computed before any policy exists.

Adopting it means passing `rtol_floors.json` to
`robustness.tau_ladder(floors=...)`, or recording the resulting vector explicitly
in `NYCOPT_REGRET_TAU` (a whole-vector JSON override; partial vectors raise, since
they would leave some objectives on a different basis without saying so).

The numbers above are from a synthetic cube and are illustrative of the
*mechanism*, not of the campaign. The real ratios come from the step-05 incumbent.
The finding that matters and will not change is structural: **the annual
epsilons were calibrated for Borg archiving resolution, not against per-SOW
estimator noise, so their ratios across objectives cannot be assumed to carry
a shared tolerance.**

---

## 3. The pre-registered tolerance rule

$$
k_{\text{headline}} \;=\; \min\left\{\,k \in \text{ladder} \;:\; k\,\epsilon_i \ge
\tau_i^{\text{floor}} \ \ \forall i \,\right\}
$$

**Smallest, not largest.** For a non-inferiority claim the conservative choice is
the most discriminating defensible tolerance, because looseness is what makes the
claim cheap. The binding objective — the one with the largest $k^{\text{floor}}$
— is reported alongside, since it alone sets the rung.

If no rung clears every floor, that is itself the finding: the ladder needs
extending upward, and every rung below the binding $k^{\text{floor}}$ is
measuring estimator noise rather than harm. The script says so explicitly rather
than returning the top rung.

The full $k$-curve is reported regardless. $k_{\text{headline}}$ fixes only which
rung carries the headline sentence; the claim reported in the results is *the
range of tolerances over which the design ordering holds*, not a value at one
rung. That is the same discipline already applied to the satisficing criteria.

---

## 4. Pass B — is the comparison informative at all?

Four checks, all on the re-evaluated policy sets, none of which uses the
between-design contrast to set a parameter.

**4.1 Discrimination band.** A rung is *saturated* when every design's
$\Pi_\tau$ exceeds `RTOL_SATURATION_HI` (0.95) — the non-inferiority claim is then
trivially true — and *starved* when every design falls below
`RTOL_SATURATION_LO` (0.05). Only the informative band between them can support
the claim, and the reported band is a property of the metric, not of any design.

**4.2 Empirical nulls.** Two levels, both within a design and therefore free of
design effect:

- **seed**: two seeds of one draw — pure MOEA stochasticity;
- **draw**: two draws of one design — search *and* ensemble construction.

$\delta$ is anchored on the **draw** level. The draw is the declared unit of
analysis (`experimental_design.md`), and `compare_designs.variance_components`
already F-tests the primary metric against the draw mean-square; using the seed
level would omit construction variance and make the comparison look more decisive
than the replication scheme supports. With $K = 3$ draws and $S = 2$ seeds there
are three within-draw seed pairs and three within-design draw pairs per design —
few, and reported as such.

**4.3 Paired bootstrap.** Both designs are scored on the *same* states of the
world, so the between-design difference is paired. Resampling SOWs and
recomputing **both** designs on each resample preserves that pairing and gives a
standard error strictly smaller than differencing two independently bootstrapped
margins. Differencing the margins would overstate the error and, again, flatter
the non-inferiority claim.

**4.4 Assay sensitivity — the check a non-inferiority claim cannot skip.** A
"no difference" result is only interpretable if the comparison could have found a
difference had one existed. `historic` is the unmatched prevailing-practice
reference, expected to be worse and present in the campaign for other reasons, so
it is a free positive control. If $\Pi_\tau$ cannot separate `historic` from the
matched designs at a given tolerance, then a null between the matched designs at
that tolerance is evidence of an insensitive measurement, not of equivalent
designs, and must be reported that way. The script prints this in the imperative
when it fails at every rung.

**4.5 Co-occurrence.** $\Pi_\tau$ is a conjunction over eight objectives, where
small per-objective harm rates compound. Comparing the observed $\Pi_\tau$ with
$\prod_i (1 - \phi_i)$ — what it would be if harms were independent across
objectives — says whether the joint metric is driven by one binding objective
(harms co-occur; observed $\gg$ independent) or by accumulation across many
(observed $\approx$ independent). The two readings are different claims about the
system and should not be conflated in the text.

---

## 5. The pre-registered margin rule

$$
\delta \;=\; \max\Big(\, 2 \times \mathrm{SE}_{\text{paired bootstrap}}\big[\Delta \Pi_\tau\big],
\;\; \operatorname{spread}_{\text{draw}}\big[\Pi_\tau\big] \,\Big)
$$

evaluated at $k_{\text{headline}}$. The first term is the Monte Carlo resolution
of the endpoint on 1,000 SOWs; the second is the replication-level noise the
design contrast must beat. Both are nuisance quantities. Neither can determine
the direction of the answer, which is what makes the rule pre-registerable even
though the number it returns cannot be computed until the campaign exists.

Stated plainly in the manuscript: with $K = 3$ draws there is **no inferential
test**, and none is claimed. $\delta$ is a practical-equivalence bound whose
credibility rests on being larger than both noise sources and on being derived by
a rule fixed in advance — not on a $p$-value.

---

## 6. Reporting plan (SI)

One SI text section, three panels, plus the two rules quoted verbatim from
`supplemental_config.RTOL_TAU_RULE` / `RTOL_MARGIN_RULE` so that what was fixed in
advance is legible after the fact.

| Panel | Content | Answers |
|---|---|---|
| A | Per-objective $\tau^{\text{floor}}$ against $\epsilon_i$ and the ladder rungs, natural units, with the $\tau^{\text{floor}}/\epsilon$ ratio annotated | which rungs measure noise, and which axes need the `max` unit |
| B | $\Pi_\tau(k)$ per design and draw, with the starved / informative / saturated bands shaded | over what range the comparison is informative, and whether the ordering holds across it |
| C | The two empirical null distributions and the paired bootstrap CI at $k_{\text{headline}}$, with $\delta$ marked | how large a difference means nothing |

The assay-sensitivity verdict is a sentence in the main text, not an SI panel: if
it fails, the RQ2 result is not reportable as a null and the reader must be told
in the same place the result is.

---

## 7. Order of operations (the part that is easy to get wrong)

1. Step 05 lands the incumbent cube → **run pass A** → adopt
   $k_{\text{headline}}$ into `RTOL_ADOPTED_K` and `NYCOPT_REGRET_TAU_K`.
   *This must happen before any re-evaluated policy set is inspected.*
2. Campaign runs; step 08 re-evaluation completes.
3. **Run pass B.** It computes the margin only when `RTOL_ADOPTED_K` is already
   set, and refuses otherwise — deriving the margin at a rung chosen after seeing
   the profile is precisely the circularity in §1.
4. Report the $k$-curve, the band, the assay verdict, and the result at
   $k_{\text{headline}}$ against $\delta$.

---

## 8. Open items

1. ~~Adopt the ladder **shape** (§2b) and then $k_{\text{headline}}$ (§3) from
   pass A.~~ **ADOPTED 2026-08-08**, from pass A on the regenerated step-05
   incumbent cube (1,000 SOWs × R = 25, per-SOW annual-unit substrate), before
   any re-evaluated policy set existed on the new substrate. Shape = **`max`**:
   6 of 8 epsilons sit below their measured noise floors (reliability axes
   5.5–8.1×, flood 3.1×, storage 1.2×; `rtol_ladder_shapes.csv`), confirming
   §2b's structural prediction. Headline rung under the max-shape units:
   $k_{\text{headline}} = 1$ (the smallest grid rung whose $\tau_i = k\,u_i$
   clears every floor, since $u_i$ absorbs the floors). The eps-shape answer
   ($k = 10$, binding `trenton_flow_reliability_annual`, $k^{\text{floor}} =
   8.12$) is reported in `rtol_noise_floor.csv` and rejected per §2b — a rung
   that large is far outside the noise on every other axis. Recorded in
   `supplemental_config.RTOL_ADOPTED_K = 1.0` and as the whole-vector
   `NYCOPT_REGRET_TAU` (= the `unit_max` column at k = 1) in the replication
   env files; floors artifact `rtol_floors.json`. NOTE for pass B: the env
   vector pins one τ for scoring — the §4 sweep must be run with
   `NYCOPT_REGRET_TAU` unset (or via `tau_ladder(k, floors=...)`) so the
   $k$-curve does not degenerate to a single rung.
   **CORRECTION 2026-08-14:** the floor/epsilon ratios quoted above
   (5.5–8.1× / 3.1× / 1.2×) were computed against the SUPERSEDED 2026-08-05
   epsilons. Against the adopted 2026-08-12 vector they are 2.44–3.09 on the
   reliabilities and flood and 1.15 on storage. The conclusion ("6 of 8
   epsilons sit below their floors", shape = `max`) is unchanged; the
   magnitudes are not. `rtol_noise_floor.csv` / `rtol_ladder_shapes.csv` still
   carry the stale epsilon column and pass A should be re-run to refresh them.

2. ~~Replace the unpaired noise floor with the paired estimate once any policy
   cube exists on the test ensemble (§2), and record whether it moves the
   rung.~~ **DONE 2026-08-14 — and it moved the τ VECTOR, though not the rung.**
   Paired construction: for the policies nearest a tie with the incumbent in
   mean $D_i$, the median across those policies of the SD across SOWs of the
   per-SOW $D_i$. This nets out the shared inflow sequence the unpaired
   estimator double-counts. Result ($z = 1.645$, stable across 2/5/10 %
   near-tie sets; `rtolB_paired_floor_check.csv`):

   | axis | paired floor | unpaired pass-A floor | inflation |
   |---|---|---|---|
   | NYC / Montague / Trenton / NJ reliability | 0.0197 / 0.0176 / 0.0171 / 0.0240 | 0.134 / 0.130 / 0.122 / 0.137 | 5.7–7.4× |
   | downstream flood exceedance | 0.0426 | 0.9265 | **21.8×** |
   | NYC storage P1 % | 3.02 | 5.76 | 1.9× |

   The §2 caution ("a bound to be tightened, not a safe default") is confirmed
   quantitatively. Both estimators remain UPPER bounds: only pooled per-SOW
   values are persisted, so the true Monte-Carlo noise of $J(\theta)$ is not
   recoverable from the cubes, and the near-tie construction still contains
   real policy × SOW response. A true paired null needs two independent
   simulations of the SAME policy on $E_{\text{test}}$.

   **Consequence — τ RE-ADOPTED 2026-08-14 on round values** (k = 1 unchanged):
   reliabilities 0.02, deficit-P99 2 pp, flood 0.25 ft·d/yr, storage 5 pp. At
   the old vector the RQ2 comparison was inert: every design at
   $\Pi_\tau = 1.000$, all pairwise differences exactly 0.000 with paired
   bootstrap SE 0.0000, and **assay sensitivity FAILED** — the metric could not
   separate the unmatched `historic` control, so the non-inferiority null was
   produced by the tolerance, not measured. At the round vector assay
   sensitivity holds on both endpoints and neither is saturated or starved.
   Per-axis anchoring is in the `workflow/envs/*.env` comment block.

3. **Report with the result, not as an appendix:** on the all-8 metric the
   ordering of the two MATCHED designs is a function of the flood tolerance
   alone — `fixed_probabilistic` is flat at 0.620 across $\tau_{\text{flood}}$
   while `hazard_filling_stationary` sweeps 0.065 → 0.850, crossing it between
   0.25 and 0.30. Show the 0.10 and 0.50 variants alongside. On the
   compromise-3 framing the ordering is stable.

4. **Not estimable at K = 1 draw / S = 1 seed:** both §4.2 empirical nulls.
   δ therefore reduces to `2 × paired bootstrap SE` (0.102 all-8, 0.065
   compromise-3), which is a LOWER BOUND on δ and must be labelled as one.
   `run_pass_b()` currently raises `KeyError: 'level'` in this situation
   instead of degrading to "not estimable" — fix before the production pass.

5. The all-8 conjunction is effectively a one-objective metric:
   `montague_flow_deficit_p99_pct` is degraded in 90 % of policy × SOW cells
   (median −20.9 pp) and alone caps $\Pi_\tau$ at ~0.19–0.29. Observed
   $\Pi_\tau$ ≈ the independence product, so this is accumulation, not
   co-occurrence (§4.5).
