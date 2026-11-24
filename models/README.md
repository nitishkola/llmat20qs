# Model artifacts

This project runs pinned Hugging Face checkpoints from the user's external
cache; model weights are not copied into the repository. The model registry
and per-checkpoint inference precision live in [`src/models/registry.py`](../src/models/registry.py).
