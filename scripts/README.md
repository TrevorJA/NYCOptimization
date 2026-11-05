# scripts/

Three buckets, by purpose:

- **`main/`** — Every script a numbered `workflow/` step calls: presim,
  ensemble generation / selection / prep, baseline, test-ensemble generation,
  chunked re-evaluation and merge, design comparison, scenario discovery,
  runtime-archive extraction (`extract_runtime_archive.py`), and figure
  rendering (`figures.py` over `src/figures/registry.py`). These compose the
  manuscript-relevant pipeline.

- **`supplemental/`** — Manuscript-relevant but not part of the linear
  pipeline: E_test staging from a larger generated design
  (`make_etest_subset.py`), the E_test demand-utilization axis
  (`demand_utilization_calibration.py`, `append_demand_utilization.py`,
  `merge_demand_utilization_realized.py`), the NYC demand-covariate
  diagnostics (`demand_covariate/run.py`, `demand_covariate/figures.py`), build QC (`validate_staged_seasonality.py`,
  `verify_*.py`), calibration and diagnostic runners with their figure
  scripts, benchmarks. Driven by `workflow/supplemental/` and the root
  `supplemental_config.py`.

- **`temporary/`** — Ad-hoc / non-manuscript material. Sandbox for
  exploratory work. Nothing here should be referenced by `workflow/`
  or `src/`.
