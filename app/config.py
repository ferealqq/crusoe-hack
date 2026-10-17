from functools import lru_cache
from typing import Literal
from urllib.parse import urlsplit, urlunsplit

from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict

Setup = Literal["one_shot", "no_memory", "raw_log", "shared_brain"]
SETUPS: tuple[Setup, ...] = ("one_shot", "no_memory", "raw_log", "shared_brain")


class ResearchConfig(BaseModel):
    """Budget for one research run (README: Budget). Override from the orchestrator CLI."""

    generations: int = 5
    researchers: int = 4
    researcher_runs: int = 7  # No memory, Raw log
    researcher_runs_shared_brain: int = 6  # the curator gets the other 4 runs
    curator_runs: int = 4
    curator_seeds: list[int] = [1, 2, 3]
    test_seeds: list[int] = [1, 2, 3]
    baseline_seeds: list[int] = [0, 1, 2, 3, 4]
    research_seed: int = 0
    web_searches: int = 2
    web_results: int = 5
    max_turns: int = 30
    curator_max_turns: int = 20
    memory_cap_tokens: int = 2000
    raw_log_cap_tokens: int = 100_000

    def generations_for(self, setup: Setup) -> int:
        return 1 if setup == "one_shot" else self.generations

    def researchers_for(self, setup: Setup) -> int:
        return 1 if setup == "one_shot" else self.researchers

    def runs_per_researcher(self, setup: Setup) -> int:
        if setup == "one_shot":
            return 1
        if setup == "shared_brain":
            return self.researcher_runs_shared_brain
        return self.researcher_runs

    def runs_per_generation(self, setup: Setup) -> int:
        curator = self.curator_runs if setup == "shared_brain" else 0
        return self.researchers_for(setup) * self.runs_per_researcher(setup) + curator


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=(".env", ".env.local"), extra="ignore")

    # Neon Postgres. `neon env pull` writes a pooled DATABASE_URL (app) and a direct
    # DATABASE_URL_UNPOOLED (migrations).
    database_url: str = "postgresql://localhost/crusoe_hack"
    database_url_unpooled: str = ""

    # Neon Object Storage (S3 compatible)
    aws_access_key_id: str = ""
    aws_secret_access_key: str = ""
    aws_endpoint_url_s3: str = ""
    aws_region: str = "us-east-2"
    s3_bucket: str = "artifacts"

    # Crusoe inference
    crusoe_api_key: str = ""
    crusoe_base_url: str = "https://api.inference.crusoecloud.com/v1"
    crusoe_default_model: str = "deepseek-ai/Deepseek-V4-Flash"
    crusoe_judge_model: str = "openai/gpt-oss-120b"
    # SDK retries 429/5xx with exponential backoff (honours Retry-After)
    crusoe_max_retries: int = 10

    # Modal sandboxes (also read from ~/.modal.toml if unset)
    modal_token_id: str = ""
    modal_token_secret: str = ""
    # Overrides the environment of the active ~/.modal.toml profile (Modal's default is "main")
    modal_environment: str = ""
    modal_app_name: str = "crusoe-hack-sandboxes"
    sandbox_timeout_seconds: int = 120
    sandbox_cpu: float = 2.0
    sandbox_memory_mb: int = 4096
    # Sandboxes running at once across all setups
    sandbox_max_concurrency: int = 24

    # Shared Brain research agents
    researcher_model: str = "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B"
    curator_model: str = "deepseek-ai/DeepSeek-V4-Pro"
    # Requests per minute per model: 30 without a Crusoe payment method, 600 with one
    crusoe_max_rpm: int = 30
    # You.com Search API key; researchers get web_search only when it's set
    ydc_api_key: str = ""
    # Prepared datasets and research results (git-ignored)
    data_dir: str = "data"

    @property
    def migration_database_url(self) -> str:
        return self.database_url_unpooled or self.database_url


def asyncpg_engine_args(url: str) -> tuple[str, dict]:
    """Turn a Neon connection string into (SQLAlchemy URL, connect_args) for asyncpg.

    Neon URLs carry `?sslmode=require&channel_binding=require`, which asyncpg rejects, so
    strip the query and pass SSL via connect_args. The `-pooler` host is PgBouncer in
    transaction mode, where asyncpg's prepared-statement caches must be off.
    """
    parts = urlsplit(url)
    connect_args: dict = {}
    if "neon.tech" in parts.netloc or "sslmode=require" in parts.query:
        connect_args["ssl"] = "require"
    query = ""
    if "-pooler" in parts.netloc:
        connect_args["statement_cache_size"] = 0
        query = "prepared_statement_cache_size=0"
    return urlunsplit(("postgresql+asyncpg", parts.netloc, parts.path, query, "")), connect_args


@lru_cache
def get_settings() -> Settings:
    return Settings()
