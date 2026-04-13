"""
A verified list of LLM API endpoints and their corresponding model identifiers,
along with utility functions to parse model names and create API clients.

This module serves as a central place to manage different LLM providers and their
models, ensuring that the rest of the code can work with a consistent interface
regardless of the underlying API.
"""

ENDPOINTS = {
    "OPENAI": {
        "API_KEY_ENV_VAR": "OPENAI_API_KEY",
        "BASE_URL": "https://api.openai.com/v1",
        "MODELS": {
            "GPT_4": "gpt-4",
            "GPT_4_TURBO": "gpt-4-turbo",
            "GPT_3_5_TURBO": "gpt-3.5-turbo",
            "GPT_3_5": "gpt-3.5",
        },
    },
    "ALI_TONGYI": {
        "API_KEY_ENV_VAR": "DASHSCOPE_API_KEY",
        "BASE_URL": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "MODELS": {
            "MAX_MODEL": "qwen3-max-2025-09-23",
            "TURBO_MODEL": "qwen3-max-2025-09-23",
            "DEEPSEEK_R1": "deepseek-r1",
            "DEEPSEEK_R10528": "deepseek-r1-0528",
            "DEEPSEEK_V3": "deepseek-v3",
            "REASONER_MODEL": "qvq-max-latest",
            "EMBEDDING_V3": "text-embedding-v3",
            "EMBEDDING_V4": "text-embedding-v4"
        }
    },
    "DEEPSEEK": {
        "API_KEY_ENV_VAR": "DEEPSEEK_API_KEY",
        "BASE_URL": "https://api.deepseek.com/v1",
        "MODELS": {
            "CHAT_MODEL": "deepseek-chat",
            "REASONER_MODEL": "deepseek-reasoner"
        }
    },
    "TENCENT_HUNYUAN": {
        "API_KEY_ENV_VAR": "HUNYUAN_API_KEY",
        "BASE_URL": "https://api.hunyuan.cloud.tencent.com/v1",
        "MODELS": {
            "TURBO_MODEL": "hunyuan-turbos-latest",
            "REASONER_MODEL": "hunyuan-t1-latest",
            "LONGCONTEXT_MODEL": "hunyuan-large-longcontext"
        }
    }
}


def parse_model(model_name: str):
    """根据模型名称解析出平台和具体模型标识
    
    Args:
        model_name: 模型名称，如"gpt-4"、"qwen3-max-2025-09-23"
        
    Returns:
        (平台名称, 模型标识)，如("OPENAI", "gpt-4")，失败返回(None, None)
    """
    for platform, info in ENDPOINTS.items():
        for key, model_id in info["MODELS"].items():
            if model_id == model_name:
                return platform, model_id
    return None, None


import logging

from ..config import config
from openai import OpenAI

def get_client():
    """根据配置创建并返回一个 OpenAI 客户端实例"""
    if config["LLM"].get("OPENAI_BASE_URL"):
        # 如果用户指定了 OPENAI_BASE_URL，尝试从模型名称解析平台
        found = False
        for platform, info in ENDPOINTS.items():
            if config["LLM"]["OPENAI_BASE_URL"] == info["BASE_URL"]:
                logging.info(f"根据 OPENAI_BASE_URL 识别平台: {platform}")
                found = True
                break
        if not found:
            logging.warning(f"无法根据 OPENAI_BASE_URL 识别平台, 将直接使用指定的 URL: {config['LLM']['OPENAI_BASE_URL']}")
            base_url = config["LLM"]["OPENAI_BASE_URL"]
    else:
        platform, model_id = parse_model(config["LLM"].get("MODEL", ""))
        if platform is None:
            raise ValueError(f"无法识别的模型名称: {config['LLM'].get('MODEL', '')}")
    
        api_key = config["LLM"].get(ENDPOINTS[platform]["API_KEY_ENV_VAR"])
        if not api_key:
            raise ValueError(f"缺少 API Key，请设置环境变量 {ENDPOINTS[platform]['API_KEY_ENV_VAR']}")

        base_url = ENDPOINTS[platform]["BASE_URL"]
    
    client = OpenAI(api_key=api_key, base_url=base_url)
    return client