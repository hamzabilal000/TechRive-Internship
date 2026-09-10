"""
Week 3-4 - Task 1: Prompt templates + LCEL chains.

Demonstrates the canonical LangChain Expression Language pattern:

    chain = prompt | model | parser

The chain is invoked with variables that fill the prompt template. The model is
Groq (if GROQ_API_KEY is set) or the offline fallback. `StrOutputParser`
extracts the plain-text content from the model's message.

Run:
    python 01_lcel_chain.py
"""

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from llm_backend import backend_name, get_chat_model


def build_chain():
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a concise support assistant. Answer in one sentence."),
            ("human", "Question: {question}"),
        ]
    )
    model = get_chat_model()
    parser = StrOutputParser()
    # The LCEL pipe: each `|` feeds the previous step's output into the next.
    return prompt | model | parser


def main() -> None:
    print(f"Backend: {backend_name()}\n")
    chain = build_chain()

    questions = [
        "How do I reset my password?",
        "What payment methods do you accept?",
        "Can I export my data?",
    ]
    for q in questions:
        answer = chain.invoke({"question": q})
        print(f"Q: {q}\nA: {answer}\n")

    # `batch` runs multiple inputs through the same chain.
    print("--- batch() over all questions ---")
    for q, a in zip(questions, chain.batch([{"question": q} for q in questions])):
        print(f"* {q} -> {a}")


if __name__ == "__main__":
    main()
