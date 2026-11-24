# Data layout

- `raw/results/` contains append-only game and model-pick logs. Generated
  runtime files are ignored by Git.
- `processed/target_pool_v1.json` is the canonical, versioned target manifest.
- `interim/` is reserved for derived datasets between raw logs and final
  analysis.
