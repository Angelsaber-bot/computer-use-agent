"""Provider-neutral boundary for LLM text generation."""

from __future__ import annotations

from typing import Protocol


class LLMClient(Protocol):
    """Minimal LLM client interface used by the reasoner."""

    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        """Return the provider response text for the supplied prompts."""
