"""Configuration for FunSearch CVRP project.

This module loads configuration from environment variables.
Create a .env file in the project root (see .env.example for template).

Both OpenAI and Alibaba DashScope use the openai Python package,
as DashScope provides an OpenAI-compatible API.

API endpoint is determined by the model name:
- gpt-* models → OpenAI API
- qwen-* models → Alibaba DashScope API
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
# API KEYS
# =============================================================================

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
DASHSCOPE_API_KEY = os.environ.get("DASHSCOPE_API_KEY", "")

# =============================================================================
# LLM MODEL CONFIGURATION
# =============================================================================

OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4")
LLM_TEMPERATURE = float(os.environ.get("LLM_TEMPERATURE", "0.7"))
LLM_MAX_TOKENS = int(os.environ.get("LLM_MAX_TOKENS", "2000"))

# =============================================================================
# API ENDPOINT DETECTION (based on model name)
# =============================================================================

def get_api_config(model: str = None):
    """Get API configuration based on model name.
    
    Args:
        model: Model name. If None, uses OPENAI_MODEL from config.
        
    Returns:
        tuple: (api_key, base_url, service_name)
        
    Raises:
        ValueError: If required API key is not set.
    """
    model = model or OPENAI_MODEL
    model_lower = model.lower()
    
    # Determine API endpoint based on model name
    if model_lower.startswith("gpt-") or model_lower.startswith("o1") or model_lower.startswith("o3"):
        # OpenAI models
        if not OPENAI_API_KEY:
            raise ValueError(
                f"模型 {model} 需要 OpenAI API Key。"
                f"请设置 OPENAI_API_KEY 环境变量。"
            )
        return (
            OPENAI_API_KEY,
            "https://api.openai.com/v1",
            "OpenAI"
        )
    elif model_lower.startswith("qwen-") or model_lower.startswith("deepseek-"):
        # Alibaba DashScope models
        if not DASHSCOPE_API_KEY:
            raise ValueError(
                f"模型 {model} 需要阿里云 DashScope API Key。"
                f"请设置 DASHSCOPE_API_KEY 环境变量。"
            )
        return (
            DASHSCOPE_API_KEY,
            "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "阿里云 DashScope"
        )
    else:
        # Unknown model, default to OpenAI if key is available
        if OPENAI_API_KEY:
            return (
                OPENAI_API_KEY,
                "https://api.openai.com/v1",
                "OpenAI (unknown model)"
            )
        elif DASHSCOPE_API_KEY:
            return (
                DASHSCOPE_API_KEY,
                "https://dashscope.aliyuncs.com/compatible-mode/v1",
                "阿里云 DashScope (unknown model)"
            )
        else:
            raise ValueError(
                f"未知模型 {model}，且未配置任何 API Key。"
                f"请设置 OPENAI_API_KEY 或 DASHSCOPE_API_KEY。"
            )

# =============================================================================
# COMPATIBILITY: Old config attributes
# =============================================================================

API_KEY, API_BASE_URL, SERVICE_NAME = get_api_config()

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
