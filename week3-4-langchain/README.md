# Weeks 3–4 — LangChain

LLM orchestration with LangChain: prompt templates, LCEL chains, and a
retrieval-augmented generation (RAG) pipeline. The LLM backend is **Groq**, with
an **offline fallback** so everything runs and is testable without an API key or
network access.

## What was done

### Backend factory — `llm_backend.py`
- `get_chat_model()` returns a LangChain `Runnable` that drops straight into an
  LCEL chain:
  - **`GROQ_API_KEY` set** → a real `ChatGroq` model (`llama-3.1-8b-instant`).
  - **key absent** → `LocalFallbackChat`, a deterministic, offline, *extractive*
    responder that returns an `AIMessage` just like `ChatGroq` does.
- The fallback reads the prompt's `Context:`/`Question:` sections and answers by
  word-overlap scoring against the retrieved context — so the RAG answer is
  provably grounded in the knowledge base even offline.

### 1. Prompt templates + LCEL — `01_lcel_chain.py`
- The canonical pattern: `chain = prompt | model | parser`.
- `ChatPromptTemplate` (system + human) filled with a `{question}` variable,
  `StrOutputParser` to extract plain text.
- Shows both `.invoke()` and `.batch()`.

### 2. RAG pipeline — `02_rag_pipeline.py` + `knowledge_base.py`
- **Knowledge base**: 10 short "documentation" snippets about a fictional SaaS.
- **Retriever**: `TfidfRetriever` — TF-IDF vectors + cosine similarity, top-k.
  (Chosen over an embeddings API so retrieval works fully offline.)
- **Chain**:
  ```python
  {"context": RunnableLambda(format_context), "question": RunnablePassthrough()}
      | prompt | model | parser
  ```
- Correctly answers in-KB questions from retrieved context and **declines**
  out-of-KB questions ("What is the meaning of life?" → 0 docs → "I don't know").

## Groq + offline fallback

The whole point of the fallback is that this repo is runnable and testable by
anyone, immediately. To use the real Groq LLM:

```bash
export GROQ_API_KEY=sk-...        # or copy .env.example -> .env
python 02_rag_pipeline.py         # now routes through ChatGroq
```

With no key set, the same scripts run against `LocalFallbackChat` and print
`Backend: LocalFallbackChat (offline)`.

## Tools used
`Python 3.11`, `langchain`, `langchain-core`, `langchain-groq`, `scikit-learn`.

## Key learnings
- **LCEL** composes `Runnable`s with `|`; each step's output feeds the next, and
  the same chain supports `.invoke()`, `.batch()`, and streaming for free.
- A dict of runnables (`{"context": ..., "question": ...}`) runs its branches and
  assembles a mapping — the clean idiom for feeding retrieved context into a
  prompt.
- **RAG = retrieve → stuff context → generate**. Grounding answers in retrieved
  documents reduces hallucination and lets you cite sources; instructing the model
  to say "I don't know" when the context lacks the answer is essential.
- Designing an LLM-backed system around a **swappable backend** (real API vs.
  offline stub) makes it testable in CI and portable across environments — a
  pattern carried straight into the Week 5 service.

## How to run
```bash
pip install -r requirements.txt
python 01_lcel_chain.py
python 02_rag_pipeline.py
```

---
AI assistance (Claude) was used for debugging, explaining concepts, and code review while completing this week's tasks.
