How Good Are LLMs at 20 Questions?

In the game of 20 questions, one player selects a secret word or thing, which a second player must identify by asking a series of true/false questions.   The winner (guesser or questioner) is determined by failure or success after twenty questions.    LLMs are in principle capable of suggesting random-ish words, and asking/answering questions.    How good are they at 20 questions?   Getting two models to play 20 questions with each other will not be difficult: the interesting thing is to carefully evaluate what they are doing and how well they are doing it.

A successful project here will convincingly evaluate different pairs of models against each other.   Perhaps this is done through an agent-type architecture, but I will want to see results of all possible pairs of at least four different LLM models, each playing the role of guesser and questioner against each other.
Evaluation is a critical thing here.   You need to design and perform experiments which convincingly compare models in different roles, answering questions like:
Do models perform better when the same model is used in both roles in different sessions, or is one model better than others in a particular role?
How well do models do selecting words/concepts uniformly at random?   Compare them to a truly random generator for nouns, places, people, and general vocabulary worlds.
How good are the questions the models ask?  In theory each question should try to half the space of remaining possibilities.
How good are they at accurately answering questions about the hidden word?  The guesser can mislead the questioner by giving misleading answers, which is bad/cheating.
There is previous work here.   Do a literature search on papers which have tried to evaluate models at 20 questions.   I will expect to see a research summary in your progress report.
