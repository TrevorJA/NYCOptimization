# scripts/

Three buckets, by purpose:

- **`main/`** — Production pipeline scripts called by `workflow/` (presim,
  baseline, diagnostics, the 3-step ensemble pipeline). These are the
  scripts that compose the manuscript-relevant pipeline.

- **`supplemental/`** — Manuscript-relevant but not part of the linear
  pipeline: benchmarks, diagnostic samplers, smoke-testing tools. Useful
  while iterating but not run on every campaign.

- **`temporary/`** — Ad-hoc / non-manuscript material. Sandbox for
  exploratory work. Nothing here should be referenced by `workflow/`,
  `slurm/`, or `src/`.
