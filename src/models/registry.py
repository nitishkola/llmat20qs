"""Pinned model checkpoints and their inference precision."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    key: str
    name: str
    model_id: str
    revision: str
    inference_dtype: str


MODELS = (
    ModelConfig(
        "qwen3-8b",
        "Qwen3 8B",
        "Qwen/Qwen3-8B",
        "b968826d9c46dd6066d109eabc6255188de91218",
        "float16",
    ),
    ModelConfig(
        "gemma3-12b-it",
        "Gemma 3 12B IT",
        "google/gemma-3-12b-it",
        "96b6f1eccf38110c56df3a15bffe176da04bfd80",
        "float32",
    ),
    ModelConfig(
        "mistral-nemo-12b",
        "Mistral Nemo 12B Instruct",
        "mistralai/Mistral-Nemo-Instruct-2407",
        "04d8a90549d23fc6bd7f642064003592df51e9b3",
        "float16",
    ),
    ModelConfig(
        "granite-3.3-8b",
        "Granite 3.3 8B Instruct",
        "ibm-granite/granite-3.3-8b-instruct",
        "51dd4bc2ade4059a6bd87649d68aa11e4fb2529b",
        "float16",
    ),
)
