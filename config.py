from dataclasses import dataclass


@dataclass
class ExperimentConfig:
    lm: str = "openai-community/gpt2"
    activation_dim: int = 768
    layer: int = 8
    hf: str = "Skylion007/openwebtext"
    steps: int = 100_000
    n_ctxs: int = 10_000


cfg = ExperimentConfig()
