"""SafeLLM: Client abstraction for interacting with local Ollama service."""

from typing import Any, Dict, List, Optional
import httpx


class OllamaClientError(Exception):
    """Raised when Ollama connection or generation encounters an error."""
    pass


class OllamaClient:
    """Client for local Ollama OpenAI-compatible and native endpoints."""

    def __init__(self, base_url: str = "http://127.0.0.1:11434", default_model: Optional[str] = None):
        self.base_url = base_url.rstrip("/")
        self.chat_url = f"{self.base_url}/v1/chat/completions"
        self.tags_url = f"{self.base_url}/api/tags"
        self._default_model = default_model

    def is_available(self, timeout: float = 3.0) -> bool:
        """Checks if the local Ollama daemon is reachable and responding."""
        try:
            with httpx.Client(timeout=timeout) as client:
                res = client.get(self.tags_url)
                return res.status_code == 200
        except Exception:
            return False

    def list_models(self, timeout: float = 5.0) -> List[str]:
        """Returns a list of all model names currently pulled and available in Ollama."""
        try:
            with httpx.Client(timeout=timeout) as client:
                res = client.get(self.tags_url)
                res.raise_for_status()
                data = res.json()
                models = data.get("models", [])
                return [m.get("name") for m in models if m.get("name")]
        except Exception as exc:
            raise OllamaClientError(f"Failed to retrieve installed models from Ollama: {exc}") from exc

    def get_preferred_model(self) -> str:
        """Finds the best available model on the host, prioritizing Qwen/Llama/Deepseek."""
        if self._default_model:
            return self._default_model

        available = self.list_models()
        if not available:
            raise OllamaClientError(
                "No models found in Ollama. Pull a model using 'ollama pull qwen3:8b' or similar."
            )

        # Prioritization list
        preferred_prefixes = ["qwen3", "qwen", "deepseek-r1", "dolphin3", "llama3"]
        for prefix in preferred_prefixes:
            for model_name in available:
                if prefix in model_name.lower():
                    return model_name

        return available[0]

    def generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        system_prompt: str = "",
        temperature: float = 0.2,
        top_logprobs: int = 3,
        timeout: float = 120.0,
    ) -> Dict[str, Any]:
        """Generates completion text with full token log-probabilities requested."""
        selected_model = model or self.get_preferred_model()

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": selected_model,
            "messages": messages,
            "logprobs": True,
            "top_logprobs": top_logprobs,
            "temperature": temperature,
        }

        try:
            with httpx.Client(timeout=timeout) as client:
                res = client.post(self.chat_url, json=payload)
                res.raise_for_status()
                return res.json()
        except httpx.ConnectError as exc:
            raise OllamaClientError(
                f"Could not connect to Ollama at {self.base_url}. Ensure 'ollama serve' is running."
            ) from exc
        except httpx.HTTPStatusError as exc:
            raise OllamaClientError(
                f"Ollama returned HTTP error {exc.response.status_code}: {exc.response.text}"
            ) from exc
        except Exception as exc:
            raise OllamaClientError(f"Unexpected error while querying Ollama: {exc}") from exc
