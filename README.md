# How Good Are LLMs at 20 Questions?

This repository implements a controlled round robin (E1) and four follow-up
studies with four locally hosted language models. The pilot is preserved in
[`archive/pilot/`](./archive/pilot/) and is not used by the current runners.

## Repository layout

```text
data/
  raw/results/          Append-only model picks and turn-level game logs
  interim/              Reserved for intermediate analysis datasets
  processed/            Versioned target manifest
src/
  data/                 Canonical project data paths
  experiments/          E1 and E2-E4 runners
  features/             Reserved for answer-key and feature work
  models/               Pinned model registry and inference precision
  visualization/        Reserved for charts and presentation utilities
jobs/slurm/             Canonical Slurm submission scripts
models/                 Notes about externally cached model checkpoints
notebooks/              Active exploratory notebooks (none yet)
reports/                Generated summaries and figures
tests/                  Unit and protocol tests
```

Slurm stdout/stderr and GPU utilization logs remain under root `logs/`.
Python entry points live in `src/experiments/`, and all Slurm scripts live in
`jobs/slurm/`. The root `results/` and `data/target_pool_v1.json` paths remain
compatibility symlinks to their canonical locations; `results/` is retained
while the active E1 jobs finish. Model weights stay in the user's Hugging Face
cache and are not stored in the repository.

## E1: assigned-secret round robin

- Four models produce 16 directed answerer/guesser pairings, including four
  self-play pairings.
- Each pairing plays against the same fixed set of 40 secret concepts:
  **640 games total**.
- The 40 concepts are stratified across objects, animals, places, and famous
  people; each category has five common and five rare concepts.
- The versioned 128-concept pool is
  [`data/processed/target_pool_v1.json`](./data/processed/target_pool_v1.json).
  Common/rare labels are project-curated strata, not measured word-frequency
  estimates.
- A game is capped at 20 turns. The answerer and guesser receive separate
  conversation histories. Malformed moves and answers receive at most two
  formatting retries; an unresolved malformed response is logged and spends
  the turn.

Pairing comparisons are matched by secret. The E1 report includes win rates,
target-clustered 95% bootstrap intervals, paired sign-flip tests with Holm
correction, turns, and formatting errors. It does not report question
information gain or factual answer accuracy because free-form questions need
a reviewed answer key.

## Selected models

| Model | Hugging Face checkpoint | Pinned revision | Precision |
|---|---|---|---|
| Qwen3 8B | `Qwen/Qwen3-8B` | `b968826d9c46dd6066d109eabc6255188de91218` | FP16 |
| Gemma 3 12B IT | `google/gemma-3-12b-it` | `96b6f1eccf38110c56df3a15bffe176da04bfd80` | FP32 |
| Mistral Nemo 12B Instruct | `mistralai/Mistral-Nemo-Instruct-2407` | `04d8a90549d23fc6bd7f642064003592df51e9b3` | FP16 |
| Granite 3.3 8B Instruct | `ibm-granite/granite-3.3-8b-instruct` | `51dd4bc2ade4059a6bd87649d68aa11e4fb2529b` | FP16 |

Gemma uses FP32 because it produced NaN logits in FP16 on V100 and Quadro RTX
8000 hardware. This mixed-precision cohort is a practical compromise for
Quadro2, but precision may confound comparisons involving Gemma. Its 12B FP32
checkpoint uses about 45.5 GiB of a 48-GiB card, leaving limited memory
headroom. Each distinct model in a pairing is assigned a separate GPU.
Checkpoints and generation parameters are pinned/recorded; exact reproducibility
across hardware and software versions is not guaranteed.

Gemma 3 is gated. Accept Google's terms on the
[Hugging Face model page](https://huggingface.co/google/gemma-3-12b-it), then
authenticate on AI3 with `hf auth login`. Do not put access tokens in source,
job scripts, or Git.

## Environment and local checks

On an AI3 login node, create the environment once:

```bash
module load pytorch-cuda12
mkdir -p "$HOME/.venvs"
python -m venv --system-site-packages "$HOME/.venvs/llmat20qs"
source "$HOME/.venvs/llmat20qs/bin/activate"
python -m pip install -r requirements.txt
hf auth login
```

For local development, the `Makefile` provides `make test`, `make check`, and
`make mock-e1`. Mock output is a pipeline check, never experiment data.

```bash
make test
make check
make mock-e1
```

Equivalent direct commands:

```bash
python -m unittest discover -s tests -v
python -m src.experiments.e1 --mock --max-games 8 \
  --results-file /tmp/e1-mock.jsonl
```

## Run E1 on AI3

Run the model smoke test first. From the repository root:

```bash
mkdir -p logs data/raw/results reports
sbatch jobs/slurm/run_e1_smoke.job
```

The smoke job loads every selected checkpoint and tests Gemma+Mistral together
on separate Quadro RTX 8000 GPUs. Review the Slurm output for gated-model or
memory failures before proceeding.

After the smoke test succeeds, submit the cross-play and self-play batches:

```bash
sbatch jobs/slurm/run_e1.job
sbatch jobs/slurm/run_e1_self.job
```

Cross-play uses 12 directed pairings and requests two GPUs; self-play uses four
pairings and requests one. Both jobs use four CPUs and 50 GB host memory. Game
records are flushed after each completed game, allowing reruns to resume
missing IDs. GPU samples and Slurm logs are written under `logs/`.

Check jobs with `squeue -u "$USER"` and accounting with `sacct -u "$USER"`.
Do not launch the 640-game run until the smoke test succeeds.

E1 game records live under `data/raw/results/`; summaries are written under
`reports/`. Validate and merge the batches after they complete:

```bash
python -m src.experiments.e1 --validate-only --pairing-group cross-play \
  --results-file data/raw/results/e1-cross-play-r1.jsonl
python -m src.experiments.e1 --validate-only --pairing-group self-play \
  --results-file data/raw/results/e1-self-play-r1.jsonl
python -m src.experiments.e1 \
  --merge-files data/raw/results/e1-cross-play-r1.jsonl \
               data/raw/results/e1-self-play-r1.jsonl \
  --results-file data/raw/results/e1-qwen3-8b-gemma3-12b-mistral-nemo-granite33-r1.jsonl
python -m src.experiments.e1 --report-only \
  --results-file data/raw/results/e1-qwen3-8b-gemma3-12b-mistral-nemo-granite33-r1.jsonl
```

## Additional experiments

The four follow-up studies live in
[`src/experiments/followups.py`](./src/experiments/followups.py). Their
resumable game logs go to `data/raw/results/`; generated JSON summaries go to
`reports/`.

| Study | Games | Design |
|---|---:|---|
| E2 predictable minds | 640 | Collect 100 constrained target picks per model (25 per target category), then run 16 directed pairings × 40 games using answerer-model picks. Reports pick entropy and a Monte Carlo uniform-pool baseline. |
| E2 self-secret/scouting | 640 | Four self-play pairings, 80 matched model-picked secrets per condition, with and without the opponent model's top-five pick report. Reuses the predictable-minds pick file. |
| E3 liar | 500 | Qwen3 is the fixed baseline answerer. Four guessers, 0/1/2 scheduled answer flips, and warned/unwarned conditions; 20 or 21 matched games per cell. |
| E4 noisy oracle | 300 | Qwen3 is the fixed baseline answerer. Four guessers × 5%/10%/17% flip probability × 25 games. |

Qwen3's responses are a model-generated baseline, not a reviewed factual answer
key. E3/E4 measure additional deliberate or probabilistic flips on top of that
answer behavior. E3 warns that some answers may be changed without revealing
how many or when. Early-ending games may realize fewer flips than scheduled;
both scheduled and applied counts are recorded. The 100-pick distribution per
model is descriptive and has limited power to establish uniformity.

Wait for active E1 jobs to finish before submitting follow-ups because they
share Quadro2 resources. Run predictable minds first; scouting reuses its pick
file. Then submit the remaining jobs:

```bash
sbatch jobs/slurm/run_e2_predictable.job
# After it completes:
sbatch jobs/slurm/run_e2_scouting.job
sbatch jobs/slurm/run_e3_liar.job
sbatch jobs/slurm/run_e4_noisy.job
```

The follow-up studies total 2,080 games; together with E1 the current plan is
2,720 games.
