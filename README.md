# How Good Are LLMs at 20 Questions?

This repository implements the first experiment phase (E1) of the project proposal: a controlled round robin in which four language models play both answerer and guesser roles. The archived pilot is preserved under [`archive/pilot/`](./archive/pilot/) and is not used by the new runner.

## E1 design

- Four models produce 16 directed answerer/guesser pairings, including four self-play pairings.
- Each pairing plays one game against the same fixed set of 40 secret concepts: **640 games total**.
- The 40 concepts are stratified across objects, animals, places, and famous people; each category has five common and five rare concepts.
- The versioned 128-concept pool and E1 target selection are in [`data/target_pool_v1.json`](./data/target_pool_v1.json). Common/rare labels are project-curated strata, not measured word-frequency estimates.
- A game is capped at 20 turns. The answerer and guesser receive separate conversation histories. Malformed moves and answers receive at most two formatting retries; an unresolved malformed response is logged and spends the turn rather than silently becoming a Yes/No answer.

The shared target set makes pairing comparisons matched by secret. The report gives win rates and 95% target-clustered bootstrap intervals, plus paired model-role comparisons using sign-flip permutation tests with Holm correction across the six model comparisons in each role. It also reports turn and formatting-error summaries. The runner does **not** report question information gain or factual answer accuracy yet: those metrics require a reviewed answer key for the free-form questions.

## Selected models

The players are instruction-tuned open-weight checkpoints from four developers. The 8B–12B size range was chosen as a practical compromise between the pilot feedback about very small models and AI3's 48-GiB Quadro RTX 8000 GPUs. Two 8B and two 12B models are included; cross-model results do not isolate model size as a causal factor.

| Model | Hugging Face checkpoint | Pinned revision |
|---|---|---|
| Qwen3 8B | `Qwen/Qwen3-8B` | `b968826d9c46dd6066d109eabc6255188de91218` |
| Gemma 3 12B IT | `google/gemma-3-12b-it` | `96b6f1eccf38110c56df3a15bffe176da04bfd80` |
| Mistral Nemo 12B Instruct | `mistralai/Mistral-Nemo-Instruct-2407` | `04d8a90549d23fc6bd7f642064003592df51e9b3` |
| Granite 3.3 8B Instruct | `ibm-granite/granite-3.3-8b-instruct` | `51dd4bc2ade4059a6bd87649d68aa11e4fb2529b` |

Models run locally through Transformers. Gemma uses FP32 because it produced NaN logits in FP16 on both V100 and Quadro RTX 8000; Qwen3, Mistral Nemo, and Granite use FP16. This mixed-precision cohort is a practical compromise for the available Quadro RTX 8000 hardware, but precision is a potential confound when comparing Gemma with the other models. The 12B FP32 Gemma checkpoint nearly fills a 48-GiB card, so its smoke test must confirm both load and generation before E1 is run. Per-model precision is recorded in each result's run configuration. Each distinct model in a pairing is assigned a separate allocated GPU. The models are pinned to revisions, and each generated response uses a seed derived from the run seed, game ID, role, turn, and formatting retry. Exact reproducibility across GPU hardware/software versions is not guaranteed.

Gemma 3 is gated. Accept Google's terms on the [Hugging Face model page](https://huggingface.co/google/gemma-3-12b-it), then authenticate securely on AI3 (for example, `hf auth login`). Do not put access tokens in source files, job scripts, or Git.

## Environment and local validation

The project uses the AI3 `pytorch-cuda12` module for CUDA-enabled PyTorch, Transformers, and Accelerate. On an AI3 login node, create the environment once:

```bash
module load pytorch-cuda12
mkdir -p "$HOME/.venvs"
python -m venv --system-site-packages "$HOME/.venvs/llmat20qs"
source "$HOME/.venvs/llmat20qs/bin/activate"
python -m pip install -r requirements.txt
hf auth login
```

Run the unit tests and validate the target manifest/schedule without a GPU:

```bash
python -m unittest discover -s tests -v
python -c 'import e1_experiment as e; pool, targets = e.load_targets(); print(pool, len(targets), len(e.build_schedule(targets, 519, pool)))'
```

Mock mode tests the execution and resume path only; its output is not experiment data:

```bash
python e1_experiment.py --mock --max-games 8 --results-file /tmp/e1-mock.jsonl
```

`--max-games` is restricted to mock mode. Re-running that command continues with the next missing scheduled games.

## Run on Stony Brook AI3

From the repository root, create the Slurm log directory **before submission** and first run the model smoke test:

```bash
mkdir -p logs results
sbatch run_e1_smoke.job
```

The smoke job loads each selected checkpoint and tests the two 12B models together on separate Quadro RTX 8000 GPUs. Review the Slurm output for gated-model access or GPU-memory failures before continuing.

After the smoke test succeeds, submit E1's cross-play and self-play batches:

```bash
sbatch run_e1.job
sbatch run_e1_self.job
```

Cross-play uses 12 directed pairings and requests two Quadro RTX 8000 GPUs (one per distinct model). Self-play uses four pairings and requests one Quadro RTX 8000 GPU, avoiding an idle second GPU. Both jobs use four CPUs and 50 GB host memory. They save separate JSONL files and flush each completed game, allowing a rerun with the same command to resume missing game IDs. If a job is interrupted during a game, that incomplete game is retried on resume. The scripts log GPU utilization once per minute under `logs/`.

Check jobs with `squeue -u "$USER"` and accounting with `sacct -u "$USER"`. Do not launch the 640-game run until model loading and the paired 12B memory test have succeeded.

Validate both batches, merge them into the full 640-game E1 file, then generate the report:

```bash
python e1_experiment.py --validate-only --pairing-group cross-play \
  --results-file results/e1-cross-play-r1.jsonl
python e1_experiment.py --validate-only --pairing-group self-play \
  --results-file results/e1-self-play-r1.jsonl
python e1_experiment.py \
  --merge-files results/e1-cross-play-r1.jsonl results/e1-self-play-r1.jsonl \
  --results-file results/e1-qwen3-8b-gemma3-12b-mistral-nemo-granite33-r1.jsonl
```

## Validate and summarize

The merged result can also be validated on its own:

```bash
python e1_experiment.py --validate-only \
  --results-file results/e1-qwen3-8b-gemma3-12b-mistral-nemo-granite33-r1.jsonl
python e1_experiment.py --report-only \
  --results-file results/e1-qwen3-8b-gemma3-12b-mistral-nemo-granite33-r1.jsonl
```

The runner writes a `.summary.json` file beside the JSONL results. Merge refuses to overwrite an existing destination and validates the full schedule before writing. Runtime outputs under `results/` and `logs/` are ignored by Git; keep them backed up according to the project's data-management plan.

## Additional experiments

The four follow-up studies are implemented in [`followup_experiments.py`](./followup_experiments.py). They write separate resumable JSONL logs and summaries under `results/`, leaving E1 records untouched. All use seed 519 by default and the same four pinned models/precision configuration as E1.

| Study | Games | Design |
|---|---:|---|
| E2 predictable minds | 640 | Collect 100 constrained target picks per model (25 per target category), then run 16 directed pairings × 40 games using answerer-model picks. Target lists are matched across each answerer's four guessers. Reports pick entropy, distinct picks, favorite picks, and a Monte Carlo uniform-pool baseline. |
| E2 self-secret/scouting | 640 | Four self-play pairings, 80 matched model-picked secrets per condition, with and without the opponent model's top-five pick report. Reuses the predictable-minds pick file. |
| E3 liar | 500 | Qwen3 is the fixed baseline answerer. Four guessers are tested with zero, one, or two scheduled answer flips, with/without a warning that gives no count. The schedule balances 24 cells at 20 or 21 games per cell and matches targets and flip schedules across warning conditions. |
| E4 noisy oracle | 300 | Qwen3 is the fixed baseline answerer. Four guessers receive independently flipped answers at 5%, 10%, or 17%, with 25 games per model/rate cell. |

Qwen3's responses are a model-generated baseline, not a reviewed factual answer key. E3/E4 measure additional deliberate or probabilistic flips on top of that answer behavior. E3 schedules flips by answer-turn index; early-ending games may realize fewer flips than scheduled, and both counts are logged. The 100-pick distribution per model is descriptive and has limited power to establish uniformity.

These are real GPU experiments, not mock runs. Wait for the active E1 jobs to finish before submitting the follow-up jobs, since they share Quadro2 resources. On AI3, from the repository root:

```bash
# Predictable minds first; also collects and summarizes the model-picked secrets.
sbatch run_e2_predictable.job

# Submit after predictable minds has completed, because this reuses its pick file.
sbatch run_e2_scouting.job
sbatch run_e3_liar.job
sbatch run_e4_noisy.job
```

Each job resumes from completed JSONL game IDs if rerun with the same seed. Outputs are `results/e2-predictable-picks-s519.jsonl`, `results/e2-predictable-games-s519.jsonl`, `results/e2-scouting-s519.jsonl`, `results/e3-liar-s519.jsonl`, and `results/e4-noisy-s519.jsonl`, each with adjacent summaries. The total plan is 2,720 games including E1.
