# How Good Are LLMs at 20 Questions?

This project studies language models in both roles of 20 Questions: the answerer, who knows the secret, and the guesser, who asks questions and guesses. `twenty_questions_pilot.py` runs a directed round robin, records turn-level JSONL logs, and validates that all scheduled pairings and targets were completed. Existing small-model pilot logs are exploratory and are not part of the current model cohort.

## Selected model cohort

The runner uses only these four locally hosted, instruction-tuned Hugging Face models. Each checkpoint is pinned to a specific revision for reproducibility.

| Model | Hugging Face checkpoint | Revision |
|---|---|---|
| Qwen3 8B | `Qwen/Qwen3-8B` | `b968826d9c46dd6066d109eabc6255188de91218` |
| Gemma 3 12B IT | `google/gemma-3-12b-it` | `96b6f1eccf38110c56df3a15bffe176da04bfd80` |
| Mistral Nemo 12B Instruct | `mistralai/Mistral-Nemo-Instruct-2407` | `04d8a90549d23fc6bd7f642064003592df51e9b3` |
| Granite 3.3 8B Instruct | `ibm-granite/granite-3.3-8b-instruct` | `51dd4bc2ade4059a6bd87649d68aa11e4fb2529b` |

This cohort addresses the concern that the earlier 0.6B–3B pilot models may be too weak for meaningful comparisons, while avoiding the substantially larger 20B–31B candidates in the proposal draft. The models are from four different developers and use instruction tuning, supporting a more relevant role-by-role comparison. Their 8B–12B sizes are a practical compromise for Tesla1's 32 GB V100 GPUs. The runner uses FP16 on CUDA and assigns each distinct model in a pairing to its own GPU; the round-robin job therefore requests two GPUs.

Gemma 3 is gated: accept Google's terms on the [Hugging Face model page](https://huggingface.co/google/gemma-3-12b-it), then authenticate on AI3 (for example, with `hf auth login`). Keep credentials in the Hugging Face user cache or a protected environment variable. Never put tokens in source, job files, or Git.

## Local checks

Use Python with `torch`, `transformers`, and `accelerate` installed. Install dependencies with:

```bash
python -m pip install -r requirements.txt
```

Run a mock 48-game schedule without downloading models:

```bash
python twenty_questions_pilot.py --mock --results-file /tmp/20q-mock.jsonl
python twenty_questions_pilot.py --validate-only /tmp/20q-mock.jsonl --repeats 1
```

Run the test suite:

```bash
python -m unittest discover -s tests -v
```

Mock results are pipeline checks, not model results. Real inference downloads pinned checkpoints from Hugging Face on first use and runs them locally; no hosted inference API is used.

## Run on Stony Brook AI3

AI3 provides CUDA-enabled PyTorch, Transformers, and Accelerate through the `pytorch-cuda12` module. Set up the user environment once on a login node:

```bash
module load pytorch-cuda12
mkdir -p "$HOME/.venvs"
python -m venv --system-site-packages "$HOME/.venvs/llmat20qs"
source "$HOME/.venvs/llmat20qs/bin/activate"
python -m pip install -r requirements.txt
hf auth login
```

Accept the Gemma terms and configure Hugging Face access before submitting either job. From the repository root, create output folders and submit the four-model load/generation check:

```bash
mkdir -p logs results
sbatch run_gpu_smoke.job
```

After the smoke job succeeds, submit the 48-game round robin:

```bash
sbatch run_pilot.job
```

`run_pilot.job` pins execution to Tesla1, requests two GPUs (one per player model), and writes to a new cohort-specific file at `results/tesla1-qwen3-8b-gemma3-12b-mistral-nemo-granite33-r1.jsonl`. The runner refuses to mix results from a different cohort, revision, or repeat schedule; rerunning the same job/file resumes completed game IDs. For more repetitions, edit `--repeats` and use a distinct output file.

Check job status with `squeue -u "$USER"` and accounting with `sacct -u "$USER"`. The job scripts sample GPU utilization once per minute and save the log under `logs/`. Do not request GPU resources for mock tests.

## Preliminary pilot results

| Pilot output | Games | Wins | Invalid answer attempts / answer turns |
|---|---:|---:|---:|
| `free_hf_ollama_games.jsonl` (original cohort) | 48 | 2 (4.2%) | 91 / 729 (12.5%) |
| `additional_hf_ollama_games.jsonl` (additional cohort) | 48 | 10 (20.8%) | 64 / 754 (8.5%) |

These are one-repeat preflights across three fixed targets, not a powered comparison. The 50 picks in `picks.jsonl` are a separate sample: “desk lamp” appears 16 times and “toaster oven” 13 times. Neither the game pilots nor the pick sample establish which model is better or whether model selection is statistically non-uniform.

## Research plan

The planned study compares all 16 ordered role pairings among four models, including self-play with isolated role histories. It adds a defined target-sampling experiment, information-gain scoring for questions, and independent answer-key evaluation for oracle accuracy. See [the proposal](./project_proposal.md) for the study design and references. The proposal's earlier large-model roster is a draft; the currently executable cohort is listed above.
