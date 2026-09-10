# Capstone Architecture — How Weeks 1–5 Feed the Final Pipeline

The internship builds, week by week, toward one working system: an **automated
support-ticket triage pipeline**. A customer message comes in, an LLM-powered
service classifies and drafts a grounded reply, and an n8n workflow routes the
ticket based on that decision. This document traces how each week contributes.

## The end-to-end pipeline

```
                                   ┌─────────────────────────────────────────┐
   Customer message                │            n8n workflow (Week 6)         │
        │                          │                                          │
        ▼                          │   ┌──────────┐    ┌───────────────────┐  │
  ┌───────────┐   HTTP POST        │   │ Webhook  │───▶│ HTTP Request node │──┼──┐
  │  Webhook  │◀──────────────────▶│   │ (intake) │    │  → /triage        │  │  │
  └───────────┘                    │   └──────────┘    └───────────────────┘  │  │
                                   │                            │             │  │
                                   │                            ▼             │  │
                                   │                     ┌─────────────┐      │  │
                                   │                     │ IF priority │      │  │
                                   │                     │   == high   │      │  │
                                   │                     └─────────────┘      │  │
                                   │                     true │   │ false     │  │
                                   │              ┌───────────┘   └────────┐  │  │
                                   │              ▼                        ▼  │  │
                                   │      ┌──────────────┐        ┌────────────┐ │
                                   │      │ Escalate to  │        │ Auto-reply │ │
                                   │      │ human agent  │        │ to customer│ │
                                   │      └──────────────┘        └────────────┘ │
                                   └───────────────────────────────────│────────┘
                                                                        ▼
                                                              Respond to Webhook
        ┌───────────────────────────────────────────────────────────────────────┐
        │                    Week 5 FastAPI service  (POST /triage)              │
        │                                                                        │
        │   route(message) ── rule-based ──▶ category + priority                 │
        │        │                                                               │
        │        └── RAG chain (Weeks 3–4):                                      │
        │             retrieve(KB) ─▶ prompt | model | parser ─▶ suggested_reply │
        │                                    │                                   │
        │                          ChatGroq  ▲  or  offline fallback             │
        └────────────────────────────────────────────────────────────────────────┘
```

## Week-by-week contribution

### Week 1 — Python, NumPy/Pandas & Classical ML
The foundation. Data handling (NumPy/Pandas), the train/test/cross-validate
discipline, and the habit of comparing a simple baseline against a stronger model.
The **TF-IDF retriever** used in the RAG service is classical ML at heart —
vectorise text, measure similarity — and the evaluation mindset from Week 1 is
what tells us whether any of this actually works.

### Week 2 — Deep Learning (ANN, CNN) & NLP
Two threads feed forward:
- **NLP**: the Week 2 **TF-IDF + Logistic Regression** support-message classifier
  is the direct ancestor of the triage service's routing step. The three
  categories (billing / technical / general) are the same taxonomy.
- **DL intuition**: building an ANN and a from-scratch CNN demystifies what a
  "model" is — essential background for using LLMs as components rather than
  magic boxes.

### Weeks 3–4 — LangChain
The core reasoning engine.
- **Prompt templates + LCEL** (`prompt | model | parser`) become the reusable
  unit of composition.
- **RAG** (retriever → context → chain) is exactly what produces the
  `suggested_reply`: the answer is grounded in the TechRive knowledge base, not
  hallucinated.
- The **Groq backend with an offline fallback** is the design decision that makes
  everything downstream testable — the Week 5 service and its CI tests run with no
  API key.

### Week 5 — LangChain as a service
The chain becomes a network-accessible **FastAPI** endpoint (`POST /triage`).
This is the integration boundary: everything before Week 5 is a Python library;
Week 5 turns it into a service any system can call over HTTP. The response
contract — `{category, priority, suggested_reply, retrieved, backend}` — is the
API the orchestration layer depends on. The **smoke test** guarantees that
contract holds.

### Week 6 — n8n orchestration (this week)
The business logic and automation layer. n8n doesn't do any ML itself — it
**orchestrates**:
1. **Webhook** receives an incoming ticket (`POST /webhook/support-triage`).
2. **HTTP Request** node calls the Week 5 service's `/triage` endpoint.
3. **IF** node branches on the returned `priority`:
   - `high` → **Escalate to Human Agent** (routed to a `<category>-urgent` queue).
   - `normal` → **Auto-Reply to Customer** using the grounded `suggested_reply`.
4. **Respond to Webhook** returns the decision to the caller.

This is where the ML/LLM work becomes an actual *workflow* with side effects and
routing — the payoff of the whole internship.

## Data / control flow summary

| Stage | Owner | Input | Output |
|-------|-------|-------|--------|
| Vectorise & retrieve | Weeks 1–2 techniques | message | top-k KB docs |
| Generate reply | Weeks 3–4 (LangChain RAG) | message + context | `suggested_reply` |
| Serve over HTTP | Week 5 (FastAPI) | `{message}` | `{category, priority, reply, ...}` |
| Route & act | Week 6 (n8n) | triage response | escalate **or** auto-reply |

## Running the full pipeline locally

1. **Start the Week 5 service:**
   ```bash
   cd ../week5-langchain-service
   uvicorn app:app --port 8000
   ```
2. **Import the workflow** `support_triage_workflow.json` into n8n
   (Workflows → Import from File) and **activate** it.
3. **Networking note:** the HTTP node targets `http://host.docker.internal:8000/triage`
   (correct when n8n runs in Docker and the service on the host). If n8n and the
   service run on the same host without Docker, change it to
   `http://localhost:8000/triage`.
4. **Fire a ticket** at the n8n webhook:
   ```bash
   curl -X POST http://localhost:5678/webhook/support-triage \
        -H 'content-type: application/json' \
        -d '{"message": "I was charged twice this month, please refund me"}'
   ```
   → the service returns `priority: high` → the workflow takes the **escalate**
   branch.

For a real deployment, swap the offline fallback for Groq by setting
`GROQ_API_KEY` on the Week 5 service, and replace the Set-node stubs (escalate /
auto-reply) with real actions (create a Zendesk ticket, send an email, post to
Slack).
