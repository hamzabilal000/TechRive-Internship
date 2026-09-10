# TechRive Internship

A 6-week, hands-on machine-learning and LLM-engineering program. Each week lives
in its own folder with runnable code and a README describing what was done, the
tools used, and the key learnings. The work builds up from classical ML
fundamentals to a deployed, orchestrated LLM pipeline.

## Roadmap

| Week | Folder | Topic |
|------|--------|-------|
| 1 | [`week1-python-ml/`](week1-python-ml/) | Python, NumPy/Pandas & Classical ML |
| 2 | [`week2-dl-nlp/`](week2-dl-nlp/) | Deep Learning (ANN, CNN) & NLP |
| 3–4 | [`week3-4-langchain/`](week3-4-langchain/) | LangChain: prompt templates, LCEL chains & RAG |
| 5 | [`week5-langchain-service/`](week5-langchain-service/) | LangChain as a FastAPI service |
| 6 | [`week6-n8n-capstone/`](week6-n8n-capstone/) | n8n workflow + capstone architecture |

## How it fits together

Weeks 1–2 establish the modelling fundamentals (data handling, classical ML,
neural nets, NLP). Weeks 3–4 move to LLM orchestration with LangChain (prompts,
LCEL chains, and retrieval-augmented generation). Week 5 wraps that chain in a
FastAPI service with a `/triage` endpoint. Week 6 orchestrates the service from an
n8n workflow and documents the end-to-end architecture in
[`week6-n8n-capstone/architecture.md`](week6-n8n-capstone/architecture.md).

## Running the code

Each week is self-contained. From a week's folder:

```bash
pip install -r requirements.txt   # where present
python <script>.py
```

Core dependencies across the program: `numpy`, `pandas`, `scikit-learn`,
`langchain`, `langchain-groq`, `fastapi`, `uvicorn`, `pytest`.

## A note on AI assistance

AI assistance (Claude) was used for debugging, explaining concepts, and code
review throughout this internship. Each week's README repeats this note for that
week's work.
