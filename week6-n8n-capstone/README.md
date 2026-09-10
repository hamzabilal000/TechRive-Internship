# Week 6 — n8n + Capstone

The capstone: an **n8n workflow** that turns the Week 5 triage service into an
end-to-end automated support pipeline, plus an **architecture document** tying
Weeks 1–5 together.

## What was done

### 1. n8n workflow — `support_triage_workflow.json`
An importable n8n workflow (6 nodes) that:
1. **Webhook — Ticket Intake** — receives a support message via
   `POST /webhook/support-triage`.
2. **Call Week 5 Triage Service** — an HTTP Request node that POSTs the message
   to the Week 5 `/triage` endpoint.
3. **IF High Priority** — branches on the service's returned `priority`:
   - **true** → **Escalate to Human Agent** (routes to a `<category>-urgent` queue).
   - **false** → **Auto-Reply to Customer** (uses the grounded `suggested_reply`).
4. **Respond to Webhook** — returns the decision to the caller.

The JSON validates and every connection resolves to a real node. Import it via
**Workflows → Import from File** in n8n.

### 2. Capstone architecture — `architecture.md`
A full write-up (with an ASCII pipeline diagram) explaining how each week feeds
the final system: Week 1's data/eval discipline and the TF-IDF idea, Week 2's NLP
classifier → the routing taxonomy, Weeks 3–4's LangChain RAG → the reply
generator, Week 5's FastAPI service → the HTTP integration boundary, and Week 6's
n8n → the orchestration/automation layer.

## How the branching works

The Week 5 service returns `priority: "high" | "normal"`. The IF node's condition
is `{{ $json.priority }} equals "high"`:

| Message | Service verdict | n8n branch |
|---------|-----------------|------------|
| "I was charged twice, please refund me" | `billing` / **high** | Escalate to human agent |
| "How do I turn on dark mode?" | `general` / normal | Auto-reply to customer |

## Tools used
`n8n` (self-hosted or cloud), the Week 5 FastAPI service, `curl` for testing.

## Key learnings
- **n8n is the orchestration layer, not the intelligence** — it calls the ML/LLM
  service and acts on the result. Keeping the "thinking" in the service and the
  "routing/side-effects" in n8n is a clean separation of concerns.
- A stable **API contract** (`{category, priority, suggested_reply, ...}`) is what
  makes this decoupling possible — n8n only needs the JSON shape, not the model.
- Expression syntax (`{{ $json.field }}`) and the IF node's true/false outputs
  make conditional automation straightforward.
- The **Docker networking gotcha**: from a container, the host is
  `host.docker.internal`, not `localhost` — a common first stumble when wiring
  n8n to a locally-served API.

## How to run
See [`architecture.md`](architecture.md) → "Running the full pipeline locally"
for the complete end-to-end steps (start the Week 5 service, import + activate
the workflow, POST to the webhook).

---
AI assistance (Claude) was used for debugging, explaining concepts, and code review while completing this week's tasks.
