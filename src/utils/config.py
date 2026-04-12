"""Configuration for FunSearch CVRP project.

This module loads configuration from environment variables.
Create a .env file in the project root (see .env.example for template).

Both OpenAI and Alibaba DashScope use the openai Python package,
as DashScope provides an OpenAI-compatible API.
"""

import os
from pathlib import Path

# Try to load from .env file if exists
def _load_env_file():
    """Load environment variables from .env file."""
    env_path = Path(__file__).parent.parent.parent / ".env"
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    key, value = line.split("=", 1)
                    key = key.strip()
                    value = value.strip().strip('"').strip("'")
                    if key not in os.environ:  # Don't override existing env vars
                        os.environ[key] = value

_load_env_file()

# =============================================================================
# API CONFIGURATION (Both use openai package)
# =============================================================================

# API Key - OpenAI key takes priority
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
DASHSCOPE_API_KEY = os.environ.get("DASHSCOPE_API_KEY", "")

# Determine which service to use
if OPENAI_API_KEY:
    # Using OpenAI
    API_KEY = OPENAI_API_KEY
    API_BASE_URL = "https://api.openai.com/v1"
    DEFAULT_MODEL = "gpt-4"
elif DASHSCOPE_API_KEY:
    # Using Alibaba DashScope (OpenAI-compatible)
    API_KEY = DASHSCOPE_API_KEY
    API_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    DEFAULT_MODEL = "qwen3-max-2025-09-23"
else:
    # No API key configured
    API_KEY = ""
    API_BASE_URL = ""
    DEFAULT_MODEL = ""

# =============================================================================
# LLM MODEL CONFIGURATION
# =============================================================================

OPENAI_MODEL = os.environ.get("OPENAI_MODEL", DEFAULT_MODEL)
LLM_TEMPERATURE = float(os.environ.get("LLM_TEMPERATURE", "0.7"))
LLM_MAX_TOKENS = int(os.environ.get("LLM_MAX_TOKENS", "2000"))

# =============================================================================
# SEARCH CONFIGURATION
# =============================================================================

N_ITERATIONS = int(os.environ.get("N_ITERATIONS", "10"))
MIN_HEURISTICS_PER_ITER = int(os.environ.get("MIN_HEURISTICS_PER_ITER", "50"))
MAX_HEURISTICS_PER_ITER = int(os.environ.get("MAX_HEURISTICS_PER_ITER", "100"))

# =============================================================================
# DATASET CONFIGURATION
# =============================================================================

DATASET_SIZES = [
    int(x.strip()) 
    for x in os.environ.get("DATASET_SIZES", "20,50,100").split(",")
]
DATASET_SEED = int(os.environ.get("DATASET_SEED", "2026"))

# =============================================================================
# EVALUATION CONFIGURATION
# =============================================================================

EARLY_PRUNING_THRESHOLD = float(os.environ.get("EARLY_PRUNING_THRESHOLD", "1.5"))
LARGE_SCALE_THRESHOLD = float(os.environ.get("LARGE_SCALE_THRESHOLD", "4.0"))

# =============================================================================
# API RATE LIMITING
# =============================================================================

API_CALL_DELAY = float(os.environ.get("API_CALL_DELAY", "2.0"))
MAX_RETRIES = int(os.environ.get("MAX_RETRIES", "3"))
MAX_PARALLEL_CALLS = int(os.environ.get("MAX_PARALLEL_CALLS", "5"))
