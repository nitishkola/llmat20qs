# Archived pilot

This directory contains legacy pilot code, small-model results, Slurm scripts, and logs. These files are preserved for provenance and are not the current experiment implementation. Do not use the archived job scripts or runner for new experiments without reviewing and updating them.

- [`code/`](./code/) contains the pilot runner.
- [`notebooks/`](./notebooks/) contains the original pilot notebook.
- [`jobs/`](./jobs/) and [`tests/`](./tests/) contain its Slurm scripts and tests.
- [`results/`](./results/) contains the tracked pilot JSONL outputs.
- [`logs/`](./logs/) contains the corresponding archived cluster logs.
- [`README_legacy.md`](./README_legacy.md) and [`project_proposal_legacy.md`](./project_proposal_legacy.md) preserve previous documentation snapshots.

The previously generated, ignored files `results/gpu-smoke-93366.jsonl` and `results/original-pilot.jsonl` were deleted before the archive was created and could not be recovered from Git. They are therefore not included here.
