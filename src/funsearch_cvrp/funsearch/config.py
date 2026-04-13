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


@dataclasses.dataclass
class Config:
    """Main FunSearch configuration."""
    programs_database: ProgramsDatabaseConfig = dataclasses.field(
        default_factory=ProgramsDatabaseConfig
    )
    num_evaluators: int = 1
    num_samplers: int = 1
    samples_per_prompt: int = 5
