# Hydrologic hazard metrics

*Paper 3 literature note. Annotated summaries of items in the Zotero collection "Paper 3 NYC Reoptimization" (`ISYGLK35`). Last updated 2026-06-16.*

Scope: the definition and selection of the **hazard-space axes** themselves — drought and flow indices, and the redundancy screening needed to pick a parsimonious, non-redundant set. Directly supports the hazard-metric-axis decision. The metrics here are computed on the output of the generators in [stochastic streamflow generation](stochastic_streamflow_generation.md). Foundational index references not yet in the collection (Vicente-Serrano SSI, Yevjevich run theory, Richter IHA) are tracked in the [overview](scenario_design.md).

---

- **Olden, J. D. & Poff, N. L. (2003).** Redundancy and the choice of hydrologic indices for characterizing streamflow regimes. *River Research and Applications*, 19(2), 101-121. doi:10.1002/rra.700
  Canonical index-redundancy screening reference. Compiles 171 hydrologic indices (including the Indicators of Hydrologic Alteration, IHA) spanning the five flow-regime facets — magnitude, frequency, duration, timing, and rate of change — from long-term daily records at 420 US gages, grouped into six stream types. Method: principal component analysis (PCA) on the 171×171 correlation matrix; significant axes retained via the broken-stick rule; loadings on each retained orthogonal PC identify a low-redundancy subset, choosing indices from different axes to minimize multicollinearity, with transferability checked across stream types. Direct precedent for selecting non-redundant hazard-space dimensions and the mirror of the project's scenario-redundancy argument (redundancy among indices vs among scenarios in index space).

- **McKee, T. B., Doesken, N. J. & Kleist, J. (1993).** The relationship of drought frequency and duration to time scales. *Proceedings of the 8th Conference on Applied Climatology*, 179-184. American Meteorological Society.
  Origin of the Standardized Precipitation Index (SPI). SPI standardizes accumulated precipitation over a moving window of *j* = 3, 6, 12, 24, or 48 months by gamma-fitting the empirical distribution then back-transforming to a standard normal, so the index is probability-calibrated and comparable across sites and time scales. Drought-event definition by run theory: an event runs while SPI is continuously negative and reaches ≤ −1.0; intensity classes (mild/moderate/severe/extreme) at SPI thresholds; magnitude = summed SPI over the run. Shows event frequency decreases inversely and duration increases linearly with accumulation time scale. The methodological template the Standardized Streamflow Index (SSI) adapts to streamflow for the planned drought-event axes; longer scales (12–24 months) match hydrologic/reservoir drought.

- **Brunner, M. I., Slater, L., Tallaksen, L. M. & Clark, M. (2021).** Challenges in modeling and predicting floods and droughts: A review. *WIREs Water*, 8(3), e1520. doi:10.1002/wat2.1520
  Review organizing flood/drought prediction challenges into four categories — data, process understanding, modeling, and human–water interactions — across forecasts, frequency estimates, and projections. Relevant subthemes: event definition (threshold/run-theory choices), the multivariate and spatial characterization of extremes, and non-stationarity. Argues for jointly assessing droughts and floods in one framework rather than treating them independently. Context for choosing hazard axes that span both low- and high-flow extremes in a single coordinate system and for being explicit about event-definition and time-scale choices when constructing those axes.

---

**Related notes:** [scenario design overview](scenario_design.md) · [stochastic streamflow generation](stochastic_streamflow_generation.md) · [bottom-up & scenario-neutral design](bottom_up_scenario_neutral.md)
