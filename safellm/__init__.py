"""SafeLLM: Tamper-Proof Mathematical Confidence and Uncertainty Quantification for LLMs."""

from safellm.client import OllamaClient, OllamaClientError
from safellm.confidence import (
    AlternativeToken,
    AnalysisResult,
    ConfidenceMetrics,
    TokenAnalysis,
    analyze_token_sequence,
    parse_ollama_completion,
)

__version__ = "0.2.0"

__all__ = [
    "OllamaClient",
    "OllamaClientError",
    "AlternativeToken",
    "TokenAnalysis",
    "ConfidenceMetrics",
    "AnalysisResult",
    "analyze_token_sequence",
    "parse_ollama_completion",
]
