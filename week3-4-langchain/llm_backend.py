"""
LLM backend factory for the Week 3-4 LangChain work.

`get_chat_model()` returns a LangChain Runnable that can be dropped straight
into an LCEL chain (`prompt | model | parser`):

  * If GROQ_API_KEY is set in the environment -> a real `ChatGroq` model.
  * Otherwise                                 -> a deterministic, offline
    `LocalFallbackChat` that still produces grounded, inspectable answers so
    the chains and tests run with no network access and no API key.

The fallback is intentionally simple and explainable: it is an *extractive*
responder. When the prompt contains a "Context:" block (the RAG case) it scores
each sentence in that context by word overlap with the question and returns the
best-matching sentence(s). This makes the RAG pipeline genuinely demonstrable
offline -- the answer provably comes from the retrieved context.
"""

from __future__ import annotations

import os
import re

from langchain_core.messages import AIMessage
from langchain_core.prompt_values import PromptValue
from langchain_core.runnables import Runnable, RunnableLambda

_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "do", "does", "did", "of",
    "to", "in", "on", "for", "and", "or", "what", "which", "who", "how",
    "when", "where", "why", "i", "you", "it", "this", "that", "my", "your",
    "with", "can", "please", "me", "we", "our", "be", "as", "at", "by",
}


def _tokens(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", text.lower())
            if w not in _STOPWORDS and len(w) > 1}


def _prompt_to_text(prompt_input) -> str:
    """Normalise whatever the prompt step produced into a plain string."""
    if isinstance(prompt_input, PromptValue):
        return prompt_input.to_string()
    if isinstance(prompt_input, str):
        return prompt_input
    if isinstance(prompt_input, list):  # list of BaseMessages
        return "\n".join(getattr(m, "content", str(m)) for m in prompt_input)
    return str(prompt_input)


def _extract_section(text: str, label: str) -> str | None:
    """Pull the text after a 'Label:' marker up to the next ALL-CAPS-ish marker."""
    m = re.search(rf"{label}:\s*(.*?)(?:\n[A-Z][a-z]+:|\Z)", text,
                  re.DOTALL | re.IGNORECASE)
    return m.group(1).strip() if m else None


def _local_answer(prompt_input) -> AIMessage:
    """Deterministic offline 'LLM'. Returns an AIMessage like ChatGroq would."""
    text = _prompt_to_text(prompt_input)
    question = _extract_section(text, "Question") or text
    context = _extract_section(text, "Context")

    if context:
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n", context)
                     if s.strip()]
        q_tokens = _tokens(question)
        scored = sorted(
            sentences,
            key=lambda s: len(_tokens(s) & q_tokens),
            reverse=True,
        )
        best = [s for s in scored if _tokens(s) & q_tokens][:2]
        if best:
            answer = " ".join(best)
            return AIMessage(content=f"[offline] {answer}")
        return AIMessage(content="[offline] I don't find that in the "
                                 "provided context.")

    # No context -> echo a helpful, deterministic transformation of the question.
    return AIMessage(
        content=f"[offline] (no GROQ_API_KEY set) You asked: {question.strip()}"
    )


class _LocalFallbackChat(RunnableLambda):
    """A RunnableLambda tagged so callers can tell which backend is active."""

    is_fallback = True


def get_chat_model(temperature: float = 0.0, model: str = "llama-3.1-8b-instant"
                   ) -> Runnable:
    """Return a chat model Runnable: real ChatGroq if keyed, else offline fallback."""
    api_key = os.environ.get("GROQ_API_KEY")
    if api_key:
        from langchain_groq import ChatGroq  # imported lazily
        return ChatGroq(model=model, temperature=temperature, api_key=api_key)
    return _LocalFallbackChat(_local_answer)


def backend_name() -> str:
    return "ChatGroq" if os.environ.get("GROQ_API_KEY") else "LocalFallbackChat (offline)"
