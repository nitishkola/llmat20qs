PYTHON ?= python3

.PHONY: test check mock-e1 smoke-e1 run-e1 run-followups

test:
	$(PYTHON) -m unittest discover -s tests -v

check:
	$(PYTHON) -m py_compile src/experiments/e1.py src/experiments/followups.py src/models/registry.py
	bash -n jobs/slurm/*.job

mock-e1:
	$(PYTHON) -m src.experiments.e1 --mock --max-games 8 --results-file /tmp/e1-mock.jsonl

smoke-e1:
	sbatch jobs/slurm/run_e1_smoke.job

run-e1:
	sbatch jobs/slurm/run_e1.job
	sbatch jobs/slurm/run_e1_self.job

run-followups:
	sbatch jobs/slurm/run_e2_predictable.job
