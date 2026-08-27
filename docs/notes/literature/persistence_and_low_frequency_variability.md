# Interannual persistence & low-frequency hydroclimate variability (NE US)

*Paper 3 literature note. Annotated summaries of items in the Zotero collection "Paper 3 NYC
Reoptimization" (`ISYGLK35`). Last updated 2026-07-29.*

Scope: the evidence base for the persistence-axis decision
(`../methods/persistence_axis_diagnostics.md`) — observed and paleo low-frequency
variability of NYC-watershed hydroclimate, projected changes in multi-year drought risk for
the northeastern US, and the credibility of GCM chains for interannual-to-decadal
persistence. Generator-side persistence mechanics (weighted/tilted bootstraps,
wavelet-AR, attribute-targeted generation) are in
[stochastic streamflow generation](stochastic_streamflow_generation.md).

---

## Observed & paleo record: persistence is epochal, and the fitted record is pluvial-dominated

- **Pederson, N., Bell, A. R., Cook, E. R., Lall, U., Devineni, N., Seager, R., Eggleston,
  K. & Vranes, K. P. (2013).** Is an epic pluvial masking the water insecurity of the
  greater New York City region? *Journal of Climate*, 26(4), 1339–1354.
  doi:10.1175/JCLI-D-11-00723.1
  472-yr (1531–2003) nested tree-ring reconstruction of May–August PDSI for the NYC
  watershed (32 chronologies, 12 species, up to 66% variance). The mid-1960s drought is
  severe in the full record, but the sixteenth and seventeenth centuries saw repeated
  droughts of **similar intensity and greater duration**; the record trends pluvial since
  ~1800, capped by an **unprecedented 43-yr pluvial through 2011**. Directly load-bearing
  here: the 1945–2023 fitted record is drawn almost entirely from that pluvial, so
  multi-year drought *duration* risk is understated by the instrumental record itself —
  independent of any climate-change signal. This motivates stressing persistence at least
  to the historical-record level, and is the citation for the "under historical
  persistence" scoping limitation.

- **Seager, R., Pederson, N., Kushnir, Y., Nakamura, J. & Jurburg, S. (2012).** The 1960s
  drought and the subsequent shift to a wetter climate in the Catskill Mountains region of
  the New York City watershed. *Journal of Climate*. doi:10.1175/JCLI-D-11-00518.1
  Attribution study of the exact epochal structure measured in our fit record (annual
  water-year ρ₁: 0.51 over WY1946–80, 0.06 over WY1981–2019). Neither the 1960s drought
  nor the subsequent pluvial is reproduced by SST-forced atmosphere GCMs, and the
  long-term wetting is not a simulated response to radiative forcing: both were most
  likely **internal atmospheric variability** — unpredictable, and "a drought like the
  1960s one could return while the long-term wetting trend need not continue." Two
  consequences for the axis decision: (i) a 1960s-type multi-year event is a live hazard
  regardless of projections; (ii) the mechanism that generated observed persistence is one
  GCMs do not capture, so CMIP-ensemble Δρ₁ statistics carry little mechanistic
  credibility in either direction.

- **Devineni, N., Lall, U., Pederson, N. & Cook, E. (2013).** A tree-ring-based
  reconstruction of Delaware River basin streamflow using hierarchical Bayesian
  regression. *Journal of Climate*. doi:10.1175/JCLI-D-11-00675.1
  Hierarchical-Bayes reconstruction of summer streamflow at five DRB gauges (1754–2000)
  with explicit parameter and cross-site residual uncertainty; long simulations from the
  posterior give duration/severity probabilities of regional drought; no monotonic trend
  in reconstructed drought events (90% level). The basin-specific companion to Pederson:
  pre-instrumental DRB flow carried longer-duration droughts than the gauged era, and
  reconstruction-based duration-risk estimates are the natural external check on any
  persistence-stressed E_test.

## Projections: NE multi-year drought risk is not projected to grow

- **Cook, B. I., Mankin, J. S., Marvel, K., Williams, A. P., Smerdon, J. E. &
  Anchukaitis, K. J. (2020).** Twenty-first century drought projections in the CMIP6
  forcing scenarios. *Earth's Future*, 8, e2019EF001461. doi:10.1029/2019EF001461
  CMIP6 multimodel drought assessment across precipitation, soil moisture, and runoff.
  Robust end-of-century drying in the usual hotspots (western North America,
  Mediterranean, southern Africa), but the **eastern US is one of the few regions whose
  runoff response reverses to wetting in CMIP6** relative to CMIP5. The ensemble mean
  state for the NE is wetter, not drier — the projection case for a persistence axis
  cannot rest on mean drying.

- **Xue, Z.-M. & Ullrich, P. A. (2021).** A retrospective and prospective examination of
  the 1960s U.S. Northeast drought. *Earth's Future*, 9, e2020EF001930.
  doi:10.1029/2020EF001930
  Pseudo-global-warming WRF experiments replaying the 1960s drought's dynamical
  conditions under early/mid/late-century RCP8.5 thermodynamics. The same circulation
  generally produces **more** precipitation, soil moisture, and ET (less snowpack); wet
  months wetten strongly while the driest months are essentially unchanged. A recurrence
  of 1960s-type dynamics remains possible (internal variability), but warming does not
  amplify its hydrologic deficit — weakening the case that a *climate-change* axis must
  reach beyond historical persistence.

- **Xue, Z.-M. & Ullrich, P. A. (2022).** Changing trends in drought patterns over the
  northeastern United States using multiple large ensemble datasets. *Journal of
  Climate*, 35(22). doi:10.1175/JCLI-D-21-0810.1
  Seven single-model large ensembles (internal variability explicitly sampled): the NEUS
  continues a long-term wetting trend with more extremely wet months, while the drought
  risk that grows is **short-term / rapidly developing (flash) drought** — spring
  intensification driven by ET, growing-season extension, and increasing P–ET
  anti-correlation. Projected NE drought character moves *away* from multi-year
  persistence, toward sub-seasonal onset — the opposite of what a persistence DU axis
  stresses.

## GCM credibility for interannual-to-decadal persistence

- **Moon, H., Gudmundsson, L. & Seneviratne, S. I. (2018).** Drought persistence errors
  in global climate models. *Journal of Geophysical Research: Atmospheres*, 123,
  3483–3496. doi:10.1002/2017JD027577
  CMIP5 dry-to-dry transition probabilities at monthly and annual scales vs
  observation-based products (1901–2010): despite substantial spread, most simulations
  **systematically underestimate drought persistence** at the global scale. The
  class-level citation for treating GCM-derived persistence statistics — including our
  sibling Δρ₁ ensemble — as low-credibility for magnitude.

- **Vieira, M. J. F. & Stadnyk, T. A. (2023).** Leveraging global climate models to
  assess multi-year hydrologic drought. *npj Climate and Atmospheric Science*, 6, 179.
  doi:10.1038/s41612-023-00496-y
  Global GCM-runoff assessment aimed exactly at multi-year drought: even after bias
  adjustment, **errors in lag-1 autocorrelation and in the mean number of cumulative dry
  years remain**, and unprecedented drought severity/duration projections separate from
  internal variability over only ~23–28% of land area. Explicit that GCM runoff is not
  yet "a high-confidence, design-suitable variable" for multi-year drought — the direct
  caution against anchoring an axis range on CMIP6-chain ρ₁ values.

---

Reading of the base for the axis decision: the *paleo/observed* evidence (Pederson,
Seager, Devineni) supports stressing persistence up to roughly historical-record level —
the record itself sits in an epic pluvial, and a 1960s-type event needs no
climate-change argument. The *projection* evidence (Cook, Xue & Ullrich ×2) does not
support a persistence increase beyond that for the NE, and the *credibility* evidence
(Moon, Vieira & Stadnyk, Seager) says CMIP-chain persistence statistics are too biased
and too noisy to anchor an upper bound — consistent with the in-house finding that the
CMIP6 sibling-Δρ₁ upper tail is sampling noise
(`../methods/persistence_axis_diagnostics.md`, anchor-noise check).

**Related notes:** [scenario design overview](scenario_design.md) ·
[stochastic streamflow generation](stochastic_streamflow_generation.md) ·
[hydrologic hazard metrics](hydrologic_hazard_metrics.md)
