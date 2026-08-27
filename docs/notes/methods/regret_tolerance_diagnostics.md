# Regret-Tolerance Diagnostics

Fixes the two free parameters of the incumbent-relative regret comparison
(`objective_definitions.md` §3.2b): the no-harm tolerance
$\tau_i = k\,u_i$ with $u_i = \max(\epsilon_i, \tau_i^{\text{floor}})$, and the
non-inferiority margin $\delta$ on `no_harm_freq_tau`. The substrate is the
per-SOW annual-unit objective value, the search objective recomputed per
deeply-uncertain state. Code: `scripts/supplemental/regret_tolerance_diagnostics.py`
(pass A and pass B), `regret_tolerance_passb_candidates.py` and
`regret_tolerance_passb_rank.py` (paired floors and round candidate vectors,
measurement only), settings `RTOL_*` in `supplemental_config.py`, launcher
`workflow/supplemental/regret_tolerance_diagnostics.sh`, outputs
`outputs/supplemental/regret_tolerance_diagnostics/tables/`. Manuscript
statement in Section 3.4.3.

## 1. Why the parameters are pre-registered

$\tau$ and $\delta$ are the only quantities in the regret comparison the data
do not determine, and the RQ1 answer can be moved by choosing them badly or
late. Two failure modes pull in opposite directions. Circularity, since a
tolerance read off the distribution of candidate-policy regret guarantees its
own answer (the same rule keeps the satisficing thresholds off the baseline's
own quantiles, `robustness_threshold_diagnostics.md` rule 4). Insensitivity,
since a loose tolerance drives $\Pi_\tau \to 1$ for every design, at which
point "not worse" is true by construction. Every choice is therefore biased
toward the discriminating end, and the $k$-curve rather than one rung is the
deliverable, as the satisficing criterion is reported as a sweep (Quinn et al.
2020). One rung carries the headline sentence and is fixed by a rule that
touches no campaign result.

Admissible anchors. Tier A, estimator noise and measurement resolution of the
incumbent's own cube, for both $\tau$ and $\delta$. Tier B, external decision
increments, for $\tau$ only. Tier C, the candidate-policy regret distribution,
never. Tier D, within-design nuisance variance (seed pairs), for $\delta$ only,
a least-significant-difference construction that fixes the scale of "no
difference" and cannot flip the comparison's sign.

## 2. Pass A, the noise floor (incumbent cube only)

Under the null that a policy is operationally identical to the incumbent, the
per-SOW difference $D_i(x,\theta)$ is estimator noise, and any $\tau$ below its
scale reports noise as harm. The cube carries one value per (SOW, objective),
so the noise is bounded from the incumbent's per-SOW values. SOWs are sorted on
the dominant forcing axis $m$ ($\theta$ joined on the SOW label), partitioned
into consecutive bins of `RTOL_M_BIN_SIZE` = 10, and $\sigma_i$ is the median
across bins of the within-bin standard deviation. Then

$$
\tau_i^{\text{floor}} = z \sqrt{2}\, \sigma_i, \qquad z = 1.645
\;\Rightarrow\; 5\,\% \text{ false-harm per (objective, SOW)}.
$$

This unpaired floor is an upper bound (the within-bin spread still holds real
$r_1$/$r_2$ response, and the real comparison is paired on common inflow
sequences) and is labelled as such wherever it is used. Overstating it pushes
$k$ up, the direction that flatters a non-inferiority claim. Outputs
`rtol_noise_floor.csv` ($\sigma_i$, $\tau^{\text{floor}}$, $\epsilon_i$,
$k^{\text{floor}} = \tau^{\text{floor}}/\epsilon_i$), `rtol_ladder_shapes.csv`,
`rtol_floors.json` (consumed by `robustness.tau_ladder(floors=...)`).

**Paired floor.** Once policy cubes exist on $E_{\text{test}}$ the floor is
re-estimated paired. For the policies nearest a tie with the incumbent in mean
$D_i$, it is the median across those policies of the SD across SOWs of the
per-SOW $D_i$, stable across 2/5/10 % near-tie sets
(`rtolB_paired_floor_check.csv`). The paired floors are 0.017 to 0.024 on the
reliabilities, 1.2 pp on the deficit operators, 0.04 ft·d/yr on flood and
3.0 pp on storage, 2 to 22× below the unpaired bound. Both estimators remain
upper bounds, since a true paired null needs two independent simulations of
the same policy on $E_{\text{test}}$.

## 3. Ladder shape and headline rung

One $k$ is shared across eight objectives, so the unit $u_i$ decides what a
rung means on each axis. With $u_i = \epsilon_i$ a rung sits inside the noise
on any axis whose epsilon lies below its floor and far outside it elsewhere,
because the annual epsilons were calibrated for archiving resolution, not
against per-SOW estimator noise. The adopted shape is
$u_i = \max(\epsilon_i, \tau_i^{\text{floor}})$, epsilon where resolution binds
and the floor where noise binds, so $k$ means the same thing on every axis and
never falls below $\epsilon_i$. It is admissible because the floor is a Tier-A
quantity. The three shapes (`eps`, `floor`, `max`) are reported side by side in
`rtol_ladder_shapes.csv`.

Headline rule (`RTOL_TAU_RULE`). $k_{\text{headline}}$ is the smallest rung of
`RTOL_TAU_GRID` = (0, 0.5, 1, 2, 5, 10) whose $\tau_i$ clears every floor.
Smallest, not largest, because looseness is what makes a non-inferiority claim
cheap. Under the max shape $k_{\text{headline}} = 1$ (`RTOL_ADOPTED_K`); the
binding objective is reported alongside, and the full $k$-curve is reported
regardless. The claim in the results is the range of tolerances over which the
design ordering holds.

**Adopted vector.** The headline tolerance is pinned whole as
`NYCOPT_REGRET_TAU` in the run env files (`NYCOPT_REGRET_TAU_K` = 1;
`robustness.tau_ladder` scales the pinned vector by $k$ for the sweep, and a
partial vector raises): reliabilities 0.02, deficit-P99s 2.0 pp, flood
0.25 ft·d/yr, storage 5.0 pp. Each entry is a round value anchored on the
paired floor. Reliabilities sit at the floor (24.5× the 1/1225 metric
granularity, 6 % of the incumbent's own q10–q90 spread). Deficits clear the
Montague floor (1.2 pp) while staying far below the degradations present, so
the binding Montague harm is reported rather than hidden. Flood is 5.9× its
floor and 0.83 ε (the rounder 0.10 starves the compromise panel). Storage is
where ε and both floors agree. At the unpaired pass-A vector every design
scored $\Pi_\tau = 1.000$ with a paired-bootstrap SE of zero and assay
sensitivity failed, so that vector is rejected. The per-axis anchoring is in
the env-file comment block, evidence in `rtolB_*.csv`.

## 4. Pass B, is the comparison informative

Four checks on the re-evaluated policy sets, none using the between-design
contrast to set a parameter.

- **Discrimination band.** A rung is saturated when every design's $\Pi_\tau$
  exceeds `RTOL_SATURATION_HI` = 0.95 and starved when every design falls below
  `RTOL_SATURATION_LO` = 0.05. Only the band between supports the claim, and it
  is a property of the metric, not of any design.
- **Empirical null.** The seed level, the two seeds of a design's one searched
  draw (K = 1, so no within-design draw pair exists). Draw dependence is
  measured separately by the SI draw-sensitivity re-evaluation and reported
  beside the result, not folded into $\delta$.
- **Paired bootstrap.** Both designs are scored on the same SOWs, so SOWs are
  resampled (`RTOL_BOOTSTRAP_N` = 2000) and both designs recomputed per
  resample. Differencing two independent margins would overstate the error.
- **Assay sensitivity.** `historic` (`RTOL_ASSAY_CONTROL_DESIGN`) is the
  unmatched reference expected to be worse. If $\Pi_\tau$ cannot separate it at
  a rung, a matched-design null at that rung is an insensitive measurement and
  is reported as such, in the main text beside the result.
- **Co-occurrence.** Observed $\Pi_\tau$ against $\prod_i (1 - \phi_i)$ says
  whether the conjunction is driven by one binding objective or by
  accumulation across many. Where a single axis binds the all-axes conjunction,
  pass B scores that metric on the best policy and the compromise subset on the
  median policy (`regret_tolerance_passb_rank.py`), and the matched-design
  ordering on the all-axes metric is reported together with its
  flood-tolerance sensitivity.

## 5. Margin rule

$$
\delta = \max\Big(2 \times \mathrm{SE}_{\text{paired bootstrap}}[\Delta \Pi_\tau],\;
\operatorname{spread}_{\text{seed}}[\Pi_\tau]\Big)
$$

at $k_{\text{headline}}$ (`RTOL_MARGIN_RULE`). Both terms are nuisance
quantities, so the rule is pre-registerable although the number arrives with
the campaign. With one searched draw per design and $S = 2$ seeds there is no
inferential test and none is claimed; $\delta$ is a practical-equivalence
bound whose credibility rests on exceeding both noise sources.

## 6. Reporting and order of operations

SI panels: (A) $\tau^{\text{floor}}$ against $\epsilon_i$ and the rungs per
objective, (B) $\Pi_\tau(k)$ per design and seed with the starved, informative
and saturated bands shaded, (C) the seed-level null and the paired-bootstrap CI
at $k_{\text{headline}}$ with $\delta$ marked, plus the two rules quoted
verbatim from `supplemental_config`. Order: pass A on the step-05 incumbent
cube before any re-evaluated policy set is inspected, then the campaign and
step 08, then pass B, which computes $\delta$ only when `RTOL_ADOPTED_K` is
already set.
