from funsearch_cvrp.config import config

from funsearch_cvrp.utils.check_llm import check_llm

import logging
logging.basicConfig(level=logging.INFO)

import sys

config.write(sys.stdout)

OPENAI_MODEL = config.get("LLM", "openai_model")
OPENAI_API_KEY = config.get("LLM", "openai_api_key")
OPENAI_BASE_URL = config.get("LLM", "openai_base_url")

print(
f"""You are using these:
{OPENAI_MODEL=}
{OPENAI_BASE_URL=}
{OPENAI_API_KEY=}
"""
)

def check():
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)
    check_llm(client=client, model=OPENAI_MODEL)
        
try:
    check()
    print("OK.")
except Exception as e:
    print("failed.")
    raise e