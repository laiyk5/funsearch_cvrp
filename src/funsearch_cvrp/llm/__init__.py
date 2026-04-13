"""LLM integration package."""

from .interface import LLMInterface
from .equivalence import FunctionEquivalenceDetector

__all__ = ["LLMInterface", "FunctionEquivalenceDetector"]
