"""
Week 5 - LangChain as a service.

A FastAPI app that exposes the Week 3-4 chain behind a `/triage` endpoint.
Send a customer message; get back a category, a priority, a knowledge-base-
grounded suggested reply, and the documents used.

Run the server:
    uvicorn app:app --reload --port 8000

Then:
    curl -X POST localhost:8000/triage \
         -H 'content-type: application/json' \
         -d '{"message": "I was charged twice this month, please refund me"}'
"""

from fastapi import FastAPI
from pydantic import BaseModel, Field

from triage_chain import backend_name, triage

app = FastAPI(
    title="TechRive Support Triage",
    description="LangChain-powered support-ticket triage service.",
    version="1.0.0",
)


class TriageRequest(BaseModel):
    message: str = Field(..., min_length=1, description="The customer's message.")


class TriageResponse(BaseModel):
    category: str
    priority: str
    suggested_reply: str
    retrieved: list[str]
    backend: str


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "backend": backend_name()}


@app.post("/triage", response_model=TriageResponse)
def triage_endpoint(request: TriageRequest) -> TriageResponse:
    result = triage(request.message)
    return TriageResponse(**result)
