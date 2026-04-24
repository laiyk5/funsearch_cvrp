"""Configuration classes for FunSearch."""
import dataclasses


@dataclasses.dataclass
class ProgramsDatabaseConfig:
    """Configuration for ProgramsDatabase."""
    num_islands: int = 10
    reset_period: int = 600  # seconds
    functions_per_prompt: int = 3
    cluster_sampling_temperature_init: float = 0.1
    cluster_sampling_temperature_period: int = 1000
    score_bucket_precision: int | None = None
    """Round each per-test score to this many decimal places before forming the
    signature.  e.g. 2 buckets 0.4612 and 0.4598 into the same cluster (0.46).
    None disables bucketing (exact match required)."""


@dataclasses.dataclass
class LLMConfig:
    """Configuration for LLM."""
    model: str = "gpt-4"
    base_url: str | None = None  # Custom API base URL (e.g., for proxy)
    api_key: str | None = None
    temperature: float = 0.7
    max_tokens: int = 1000
    timeout: int = 30


@dataclasses.dataclass
class Config:
    """Main FunSearch configuration."""
    programs_database: ProgramsDatabaseConfig = dataclasses.field(
        default_factory=ProgramsDatabaseConfig
    )
    llm: LLMConfig = dataclasses.field(default_factory=LLMConfig)
    num_evaluators: int = 1
    num_samplers: int = 1
    samples_per_prompt: int = 5
