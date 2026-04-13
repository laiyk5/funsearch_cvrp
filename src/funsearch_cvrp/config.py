"""Configuration for FunSearch CVRP project.

This module automatically loads default configurations and overrides them with any user-provided values from a `config.ini` file. It provides a centralized place to manage all configuration
parameters related to LLM interactions and the search process, ensuring that the rest of the code can access these settings in a consistent way.
"""

from typing import Any

# You can specify the endpoint and prepare all API_KEY

__default_llm_configs: dict[str, Any] = dict(
    MODEL="qwen3-max-2025-09-23",  # Default model
    ENDPOINT = "ALI_TONGYI",  # Default endpoint
    OPENAI_API_KEY="",    # at least one of these API_KEYs must be set
    DASHSCOPE_API_KEY="",
    OPENAI_BASE_URL="", # OPENAI_BASE_URL is automatically determined by the model name, but can be overridden if needed
    LLM_TEMPERATURE=0.7,
)

__default_search_configs: dict[str, Any] = dict(
    LLM_MAX_TOKENS=2000,
    N_ITERATIONS=10,
    MIN_HEURISTICS_PER_ITER=50,
    MAX_HEURISTICS_PER_ITER=100,
    EARLY_PRUNING_THRESHOLD=1.5,
    LARGE_SCALE_THRESHOLD=4.0,
    API_CALL_DELAY=2.0,
    MAX_RETRIES=3,
    MAX_PARALLEL_CALLS=5
)

import configparser

config = configparser.ConfigParser()
config.read_dict(
    {
        "LLM": __default_llm_configs,
        "SEARCH": __default_search_configs
    }
)

import os

if os.path.exists("config.ini"):
    config.read("config.ini")
