"""SafeLLM: Displays the raw, unmanipulated mathematical confidence under every LLM answer."""

import math
import sys
from typing import Any, Dict, List
import httpx
from rich.console import Console
from rich.panel import Panel

console = Console()

OLLAMA_URL = "http://127.0.0.1:11434/v1/chat/completions"
MODEL_NAME = "qwen3:8b"  # Local Ollama model


def query_ollama(prompt: str) -> Dict[str, Any]:
    """Queries Ollama and requests raw token log-probabilities."""
    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "logprobs": True,
        "top_logprobs": 1,
        "temperature": 0.2,
    }

    try:
        with httpx.Client(timeout=90.0) as client:
            response = client.post(OLLAMA_URL, json=payload)
            response.raise_for_status()
            return response.json()
    except httpx.ConnectError:
        console.print("[bold red]Error:[/bold red] Could not connect to Ollama. Run 'ollama serve' in a terminal.")
        sys.exit(1)


def compute_raw_confidence(api_response: Dict[str, Any]) -> Dict[str, Any]:
    """Extracts tokens and computes exact linear probabilities: P(t) = e^(logprob)."""
    choice = api_response["choices"][0]
    content_logprobs = choice.get("logprobs", {}).get("content", [])

    if not content_logprobs:
        text = choice["message"].get("content", "")
        return {"text": text, "confidence": 0.0, "min_token": None}

    tokens: List[Dict[str, Any]] = []
    probabilities: List[float] = []

    for item in content_logprobs:
        token = item["token"]
        logprob = item.get("logprob", -999.0)

        # Mathematical probability conversion: P = e^(logprob)
        prob = math.exp(logprob) * 100.0
        prob = min(max(prob, 0.0), 100.0)

        tokens.append({"token": token, "prob": prob})
        probabilities.append(prob)

    avg_confidence = sum(probabilities) / len(probabilities) if probabilities else 0.0
    min_token = min(tokens, key=lambda x: x["prob"]) if tokens else None
    full_text = "".join(t["token"] for t in tokens)

    return {
        "text": full_text.strip(),
        "confidence": avg_confidence,
        "min_token": min_token,
        "total_tokens": len(tokens),
    }


def main():
    console.print("[bold cyan]SafeLLM: Raw Mathematical Confidence Display[/bold cyan]")
    console.print("[dim]The model cannot see or manipulate these confidence metrics.[/dim]\n")

    while True:
        try:
            prompt = console.input("[bold yellow]Enter your prompt[/bold yellow] (or 'exit' to quit): ").strip()
            if not prompt or prompt.lower() in ("exit", "quit", "q"):
                break

            with console.status("[bold green]Generating answer and calculating token probabilities..."):
                raw_response = query_ollama(prompt)
                result = compute_raw_confidence(raw_response)

            # 1. Output the Model's Response
            console.print(Panel(result["text"], title="[bold white]Model Response[/bold white]", border_style="blue"))

            # 2. Display Raw Confidence Below Every Output
            confidence_str = f"[bold green]{result['confidence']:.2f}%[/bold green]"
            if result["confidence"] < 60.0:
                confidence_str = f"[bold red]{result['confidence']:.2f}%[/bold red]"
            elif result["confidence"] < 80.0:
                confidence_str = f"[bold yellow]{result['confidence']:.2f}%[/bold yellow]"

            min_info = ""
            if result["min_token"]:
                min_tok = result["min_token"]["token"].strip() or repr(result["min_token"]["token"])
                min_info = f" | Lowest certainty token: [italic]{min_tok}[/italic] ({result['min_token']['prob']:.1f}%)"

            console.print(f"📊 [bold]Raw Mathematical Confidence:[/bold] {confidence_str} (across {result['total_tokens']} tokens){min_info}\n")

        except KeyboardInterrupt:
            break

    console.print("\n[dim]SafeLLM session ended.[/dim]")


if __name__ == "__main__":
    main()

