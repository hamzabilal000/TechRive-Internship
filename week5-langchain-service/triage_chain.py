"""
Triage logic for the Week 5 service.

Reuses the Week 3-4 LangChain building blocks (the Groq/offline backend and the
knowledge base) and wraps them into a support-ticket triage pipeline:

  1. rule-based routing  -> category + priority (fast, deterministic, testable)
  2. RAG reply chain      -> a grounded suggested reply (prompt | model | parser)

Keeping routing rule-based means the endpoint behaves identically with or
without a GROQ_API_KEY, while the suggested reply still flows through a real
LCEL chain (ChatGroq when keyed, offline fallback otherwise).
"""

from __future__ import annotations

import pathlib
import sys

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Reuse the Week 3-4 backend + knowledge base.
_WEEK34 = pathlib.Path(__file__).resolve().parent.parent / "week3-4-langchain"
sys.path.insert(0, str(_WEEK34))
from knowledge_base import DOCUMENTS          # noqa: E402
from llm_backend import backend_name, get_chat_model  # noqa: E402

# --------------------------------------------------------------------------- #
# 1. Rule-based routing
# --------------------------------------------------------------------------- #
_CATEGORY_KEYWORDS = {
    "billing": ["bill", "charge", "invoice", "refund", "payment", "subscription",
                "price", "cost", "overcharg", "card"],
    "technical": ["error", "crash", "bug", "login", "log in", "loading", "500",
                  "timeout", "broken", "freeze", "not working", "fail", "sync"],
    "general": ["how", "what", "where", "plan", "trial", "demo", "documentation",
                "integration", "upgrade", "feature"],
}
_URGENT = ["urgent", "asap", "immediately", "down", "outage", "cannot access",
           "can't access", "lost", "critical", "emergency", "charged twice"]


def route(message: str) -> tuple[str, str]:
    """Return (category, priority) for a support message."""
    text = message.lower()
    scores = {
        cat: sum(kw in text for kw in kws)
        for cat, kws in _CATEGORY_KEYWORDS.items()
    }
    best = max(scores, key=scores.get)
    category = best if scores[best] > 0 else "other"
    priority = "high" if any(kw in text for kw in _URGENT) else "normal"
    # Billing disputes and technical outages skew urgent.
    if category in {"billing", "technical"} and priority == "normal":
        if any(kw in text for kw in ["refund", "overcharg", "crash", "error",
                                     "cannot", "can't", "fail"]):
            priority = "high"
    return category, priority


# --------------------------------------------------------------------------- #
# 2. RAG reply chain (reuses the Week 3-4 pattern)
# --------------------------------------------------------------------------- #
class _TfidfRetriever:
    def __init__(self, documents: list[str], k: int = 3):
        self.documents = documents
        self.k = k
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.matrix = self.vectorizer.fit_transform(documents)

    def retrieve(self, query: str) -> list[str]:
        q = self.vectorizer.transform([query])
        sims = cosine_similarity(q, self.matrix).ravel()
        idx = sims.argsort()[::-1][: self.k]
        return [self.documents[i] for i in idx if sims[i] > 0]


_retriever = _TfidfRetriever(DOCUMENTS, k=3)


def _format_context(question: str) -> str:
    docs = _retriever.retrieve(question)
    return "\n".join(f"- {d}" for d in docs) if docs else "(no documents found)"


def build_reply_chain():
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system",
             "You are a support assistant. Draft a short reply to the customer "
             "using ONLY the context below. If the context does not cover it, "
             "say a human agent will follow up."),
            ("human", "Context:\n{context}\n\nQuestion: {question}"),
        ]
    )
    model = get_chat_model()
    return (
        {"context": RunnableLambda(_format_context),
         "question": RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
    )


_reply_chain = build_reply_chain()


def triage(message: str) -> dict:
    """Full triage: route + retrieve + draft a grounded reply."""
    category, priority = route(message)
    retrieved = _retriever.retrieve(message)
    suggested_reply = _reply_chain.invoke(message)
    return {
        "category": category,
        "priority": priority,
        "suggested_reply": suggested_reply,
        "retrieved": retrieved,
        "backend": backend_name(),
    }
