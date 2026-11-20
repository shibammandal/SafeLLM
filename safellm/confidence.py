"""SafeLLM: Core Mathematical Uncertainty and Confidence Engine."""

import math
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class AlternativeToken(BaseModel):
    """An alternative token candidate considered by the model during sampling."""
    token: str
    logprob: float
    probability: float


class TokenAnalysis(BaseModel):
    """Detailed probability metrics for a single generated token."""
    token: str
    logprob: float
    probability: float = Field(description="Linear probability percentage (0-100%)")
    entropy: Optional[float] = Field(default=None, description="Local Shannon entropy across top candidates")
    alternatives: List[AlternativeToken] = Field(default_factory=list)


class ConfidenceMetrics(BaseModel):
    """Aggregated mathematical uncertainty metrics across a token sequence."""
    text: str
    total_tokens: int
    mean_confidence: float = Field(description="Average linear probability (0-100%)")
    median_confidence: float = Field(default=0.0)
    perplexity: float = Field(description="Sequence perplexity: exp(-mean(logprob))")
    min_token: Optional[TokenAnalysis] = None
    max_token: Optional[TokenAnalysis] = None
    low_confidence_count: int = Field(default=0, description="Tokens with probability < 50%")
    low_confidence_ratio: float = Field(default=0.0, description="Percentage of tokens with probability < 50%")
    mean_entropy: Optional[float] = None
    tokens: List[TokenAnalysis] = Field(default_factory=list)


class AnalysisResult(BaseModel):
    """Full analysis output separating reasoning / thinking from final answer."""
    answer: ConfidenceMetrics
    reasoning: Optional[ConfidenceMetrics] = None
    raw_text: str = ""


def _calculate_token_entropy(top_logprobs: List[Dict[str, Any]]) -> float:
    """Computes Shannon entropy H(X) = -sum(p * log2(p)) over top candidate tokens."""
    if not top_logprobs:
        return 0.0
    entropy = 0.0
    for item in top_logprobs:
        logp = item.get("logprob", -999.0)
        p = math.exp(logp)
        if p > 1e-12:
            entropy -= p * math.log2(p)
    return max(entropy, 0.0)


def analyze_token_sequence(raw_tokens: List[Dict[str, Any]]) -> ConfidenceMetrics:
    """Computes unforgeable mathematical certainty metrics from a list of token logprobs."""
    if not raw_tokens:
        return ConfidenceMetrics(
            text="",
            total_tokens=0,
            mean_confidence=0.0,
            median_confidence=0.0,
            perplexity=1.0,
        )

    analyzed: List[TokenAnalysis] = []
    probabilities: List[float] = []
    logprobs: List[float] = []
    entropies: List[float] = []

    for item in raw_tokens:
        tok_str = item.get("token", "")
        logp = float(item.get("logprob", -999.0))

        # Clamp and convert logprob to linear probability percentage
        prob = math.exp(max(logp, -50.0)) * 100.0
        prob = min(max(prob, 0.0), 100.0)

        # Parse alternative candidates if provided
        top_items = item.get("top_logprobs", [])
        alts: List[AlternativeToken] = []
        for alt in top_items:
            alt_tok = alt.get("token", "")
            alt_logp = float(alt.get("logprob", -999.0))
            alt_prob = min(max(math.exp(max(alt_logp, -50.0)) * 100.0, 0.0), 100.0)
            alts.append(AlternativeToken(token=alt_tok, logprob=alt_logp, probability=round(alt_prob, 2)))

        token_entropy = _calculate_token_entropy(top_items) if top_items else None
        if token_entropy is not None:
            entropies.append(token_entropy)

        analyzed.append(
            TokenAnalysis(
                token=tok_str,
                logprob=logp,
                probability=round(prob, 2),
                entropy=round(token_entropy, 3) if token_entropy is not None else None,
                alternatives=alts,
            )
        )
        probabilities.append(prob)
        logprobs.append(logp)

    total_count = len(analyzed)
    mean_conf = sum(probabilities) / total_count if total_count else 0.0

    # Perplexity: exp(-sum(logp) / N)
    mean_neg_logp = -(sum(logprobs) / total_count) if total_count else 0.0
    perplexity = math.exp(min(mean_neg_logp, 50.0))

    sorted_probs = sorted(probabilities)
    median_conf = sorted_probs[total_count // 2] if total_count else 0.0

    min_tok = min(analyzed, key=lambda x: x.probability) if analyzed else None
    max_tok = max(analyzed, key=lambda x: x.probability) if analyzed else None

    low_conf_items = [p for p in probabilities if p < 50.0]
    low_conf_count = len(low_conf_items)
    low_conf_ratio = (low_conf_count / total_count * 100.0) if total_count else 0.0

    mean_ent = (sum(entropies) / len(entropies)) if entropies else None
    full_text = "".join(t.token for t in analyzed)

    return ConfidenceMetrics(
        text=full_text,
        total_tokens=total_count,
        mean_confidence=round(mean_conf, 2),
        median_confidence=round(median_conf, 2),
        perplexity=round(perplexity, 2),
        min_token=min_tok,
        max_token=max_tok,
        low_confidence_count=low_conf_count,
        low_confidence_ratio=round(low_conf_ratio, 2),
        mean_entropy=round(mean_ent, 3) if mean_ent is not None else None,
        tokens=analyzed,
    )


def parse_ollama_completion(api_data: Dict[str, Any]) -> AnalysisResult:
    """Parses Ollama chat completion data, isolating reasoning and final content."""
    choices = api_data.get("choices", [])
    if not choices:
        empty = ConfidenceMetrics(text="", total_tokens=0, mean_confidence=0.0, perplexity=1.0)
        return AnalysisResult(answer=empty, raw_text="")

    choice = choices[0]
    message = choice.get("message", {})
    raw_content = message.get("content", "")
    logprobs_obj = choice.get("logprobs", {})
    content_tokens = logprobs_obj.get("content", [])

    # If logprobs are available, analyze them
    if content_tokens:
        answer_metrics = analyze_token_sequence(content_tokens)
    else:
        answer_metrics = ConfidenceMetrics(
            text=raw_content,
            total_tokens=len(raw_content.split()),
            mean_confidence=0.0,
            perplexity=1.0,
        )

    return AnalysisResult(
        answer=answer_metrics,
        reasoning=None,
        raw_text=raw_content,
    )
