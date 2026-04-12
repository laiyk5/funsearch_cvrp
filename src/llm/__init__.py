"""LLM integration package."""

from src.llm.interface import LLMInterface
from src.llm.equivalence import FunctionEquivalenceDetector

__all__ = ["LLMInterface", "FunctionEquivalenceDetector"]
