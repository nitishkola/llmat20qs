# How Good Are LLMs at 20 Questions?

**Project proposal**

**Instructor:** Professor Steven Skiena

## Problem

20 Questions is a great test of many skills, such as choosing a secret, asking good questions, being truthful about the secret, and choosing when to guess. There are many ways a model can lose: being misled by a lie from the answerer, dealing with a particularly difficult secret, or asking uninformative questions. These causes are not easily separated from the win rate.

We will play all 16 directed pairings of 4 LLMs, using each model as both guesser and answerer. We will separate the individual factors that affect performance.

### Research questions

1. **Guesser, answerer, and combination:** How do each model's guesser ability, answerer ability, and pairing affect winning? Does a model play better against itself?
2. **Randomness:** To what extent are the model's "random" secret choices truly random? Is there a bias that an opponent can exploit?
3. **Information in questions:** How much information is gained by each question compared with the ideal 1 bit from a perfect yes/no split?
4. **Answer accuracy:** How often are answers wrong? How costly is a wrong answer to the guesser?

This game is useful because we know the optimal play. For *N* equally likely secrets, a perfect guesser needs approximately log₂ *N* questions. We also know the approximate cost of each lie from the answerer. If the answerer is wrong with fraction ε, it has at most 1 − h(ε) bits per answer. We can compare model behavior with these theoretical optima rather than only with other models.

## Preliminary work

We built a game master that plays 20-turn games between two models. It separates the conversations, retries malformed answers, and logs every turn.

We used this game master to play two round robins of all 16 pairings of 4 small open models on 3 secrets: banana, bicycle, and elephant. We evaluated the logs using the proposed metrics.

| Round robin | Models | Games | Win guesser (95% CI) | Malformed answers |
|---|---|---:|---|---|
| A | Qwen3 0.6B, Qwen2.5 1.5B, SmolLM2 1.7B, Llama 3.1 8B | 48 | 2 (4%; 1–14%) | 12.5% |
| B | Phi-3.5 Mini, Granite 3.3 2B, Qwen2.5 3B, Llama 3.1 8B | 48 | 10 (21%; 12–34%) | 8.5% |

### Findings

- Some answerers provide almost no information. For example, Qwen3 0.6B answered "No" to 99% of questions. This is approximately 0.09 bits per answer, so no guesser could beat this model in an informative search.
- When guessing, Qwen3 0.6B repeated 92 of its 154 questions word for word.
- One wrong answer can be decisive. In a hand-labeled game against a pool of 16 candidates, Granite told Llama that an elephant is a "domesticated mammal." After that answer, the correct target became unreachable. Llama guessed farm animals for its final 8 turns.
- Across 3 hand-labeled games, the models gave 18 correct answers out of 20, for an error rate of 0.10.
- Most questions are uninformative. Across those games, the average information gain was 0.28 bits per question, compared with an ideal of 1 bit. Fifteen of 24 questions had almost no information gain. For example, "Is it a physical object you can touch?" provided little separation.
- Random choices are predictable. When asked for a random object 50 times, Llama gave only 17 distinct answers. "Desk lamp" appeared 16 times and "toaster oven" appeared 13 times. In 2,000 simulated uniform draws, we did not see 16 occurrences of the same object. If an opponent knew these favorites, it could exploit them by guessing those two objects first.
- Different secrets have different difficulty. Guesser could find "bicycle" in 9 of 32 games, but "banana" in only 1 of 32 games.
- With only 1 game per pairing and secret, the confidence interval for a 12-game secret-specific win rate is approximately ±25 percentage points. We cannot rank models from this pilot, but we have shown that the pipeline works and that its metrics can separate the causes of losing.

These findings set the stage for a larger design with many secrets, repeated games, and a labeled answer key.

## Approach and evaluation

### Models

We will use 4 open models of similar size from 4 different developers: Qwen3.8 27B, Gemma 4 31B, GPT-OSS 20B, and Mistral Small 24B.

We will use the Stony Brook AI Institute GPU cluster: two nodes with 8 × Tesla V100 GPUs (32 GB each), or Quadro RTX 8000 nodes (48 GB each) when available. Each player may use up to 4 GPUs with 16-bit precision.

We have a fifth model, Llama 3.3 70B, which we will use only as a judge. All players use the same prompts and settings. We record exact model versions. Since the players are open-weight models, we can also inspect probabilities rather than only generated text.

### Secrets

We will use a fixed pool of 128 concepts:

- 32 objects
- 32 animals
- 32 places
- 32 famous people

Half will be common concepts and half rare concepts. We will build the pool using WordNet and Wikidata. With 128 secrets, a perfect guesser would need approximately 7 questions.

## Experiments: 2080 games

| Experiment | Design | Games |
|---|---|---:|
| E1. Round robin | All 16 pairings × 40 assigned secrets | 640 |
| E2. Predictable minds | Each model picks a random noun, place, person, or word 100 times; 16 pairings × 20 games | 640 |
| E2. Self-secret comparison | 4 guessers × 40 games with and without a scouting report of the opponent's favorite picks | 640 |
| E3. Liar's 20 questions | The game master secretly flips 0, 1, or 2 answers; guesser is warned or not | 500 |
| E4. Noisy oracle | Each answer is flipped with probability 5%, 10%, or 17% | 300 |

The total is 2,080 games. E1 answers the role and pairing question. E2 tests whether performance against a model's own secret reflects shared selection bias: assigned-secret games should not show the same advantage. E3 and E4 test how guessers respond to wrong answers and compare their behavior with known optima.

The E2 description requires clarification before implementation: an answerer can pick its own secret, or the secret can be assigned from the fixed pool. Scouting-report and no-scouting conditions should be counted separately so that the source of any performance gain is observable.

## Evaluation

### Information gain

If a fraction p of the remaining secrets would answer Yes, question information gain is:

H(p) = −p log₂(p) − (1 − p) log₂(1 − p).

The maximum is 1 bit for an even split. We will report mean information gain, wasted questions below 0.1 bits, and questions asked after the remaining answer was already determined. We will consider results against random-question and ideal-partition baselines.

### Answer accuracy

The judge will label every question as Yes or No for all 128 concepts. Disagreements will be resolved by a person. We will hand-check 300 labels.

For each answerer, we will estimate:

- The error rate ε
- Information capacity 1 − h(ε)
- The turn on which a wrong answer made the secret unreachable

The 70B Llama judge will score factual labels, while independent human review resolves disagreements.

### Winning

We will report win rates with 95% confidence intervals. Logistic regression will include guesser, answerer, same-model pairing, secret type, and concept category. Unsolved games stop after 20 turns, so survival analysis will supplement the win-rate analysis.

### Randomness

We will measure:

- Distinct picks
- Shannon entropy
- Chi-squared deviation from a seeded uniform sampler

The gap log₂ N − H estimates how many questions an informed opponent could save. We will compare that value with the actual savings produced by a scouting report.

### Lies and noise

For deliberate lies and noisy answers, we will measure extra questions compared with the theoretical reference of approximately 2.8 extra questions for 128 secrets. We will also measure how often guessers recheck an answer after a lie.

## Future work

- **Larger open models:** Use additional cluster memory to compare players such as Llama 3.3 70B with the 4 main models on the same secrets.
- **Reasoning budgets:** Vary the reasoning budget and measure its effect on information gain.
- **Strategic liars:** Let the answerer decide when to lie to cause the most damage instead of introducing random lies.
- **Committees of guessers:** Have several models discuss one question and test whether a mixed team outperforms the best individual model.
- **Model size:** Compare different sizes from the same model family.
- **Human baseline:** Let humans play as both guesser and answerer against the models.

## References

1. Bertolazzi, L., Mazzaccara, D., Merlo, F., & Bernardi, R. (2023). "ChatGPT's Information Seeking Strategy: Insights from the 20-Questions Game." *Proceedings of INLG 2023*, 153–162. ACL Anthology.
2. Hutson, D., Vennemeyer, D., Deshmukh, Zhan, & Jiang (2025). "GuessingGame: Measuring the Informativeness of Open-Ended Questions in Large Language Models." *Proceedings of EMNLP 2025*. ACL Anthology.
3. Dagan, Y., Filmus, Y., Kane, D., & Moran, S. (2021). "The Entropy of Lies: Playing Twenty Questions with a Liar." *ITCS 2021*. arXiv:1811.02177.
4. Choudhury, D., et al. (2026). "BED-LLM: Intelligent Information Gathering with LLMs and Bayesian Experimental Design." *ICLR 2026*. arXiv:2508.21184.
5. Miller, G. A. (1995). "WordNet: A Lexical Database for English." *Communications of the ACM*, 38(11), 39–41. DOI:10.1145/219717.219748.
6. Wikimedia Foundation. *Wikidata: Introduction*. https://www.wikidata.org/
7. Stony Brook University AI Institute. *Computing Resources*. https://ai.stonybrook.edu/resources/computingresources
