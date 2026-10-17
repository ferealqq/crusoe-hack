from functools import lru_cache

from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    crusoe_api_key: str = ""
    researcher_model: str = "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B"
    curator_model: str = "deepseek-ai/DeepSeek-V4-Pro"
    # Modal workspace for sandboxes. Empty: use the active ~/.modal.toml profile.
    modal_token_id: str = ""
    modal_token_secret: str = ""
    sandbox_timeout_seconds: int = 120
    sandbox_cpu: float = 2
    sandbox_memory_mb: int = 4096


class ResearchConfig(BaseModel):
    """Budget for one research run (README "Budget"). The CLI can override some fields."""

    generations: int = 5
    researchers: int = 4
    researcher_runs: int = 7  # No memory; Shared Brain researchers give the curator its runs
    curator_runs: int = 4  # 3 seeds to verify the top claim + 1 for a conflict or stale entry
    researcher_turns: int = 30
    curator_turns: int = 20
    memory_cap_tokens: int = 2000
    baseline_seeds: list[int] = [0, 1, 2, 3, 4]
    research_seed: int = 0
    verify_seeds: list[int] = [1, 2, 3]
    test_seeds: list[int] = [1, 2, 3]
    focuses: list[str] = ["features", "model type", "tuning", "regularization", "ensembling"]


@lru_cache
def get_settings() -> Settings:
    return Settings()
