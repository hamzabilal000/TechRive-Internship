"""
Week 3-4 - Task 2: Retrieval-Augmented Generation (RAG) pipeline.

Pieces:
  * a small knowledge base (knowledge_base.DOCUMENTS)
  * a retriever: TF-IDF vectors + cosine similarity (offline, no embeddings API)
  * an LCEL chain that stuffs the retrieved context into the prompt:

        {"context": retriever, "question": passthrough} | prompt | model | parser

The model is Groq if GROQ_API_KEY is set, else the offline fallback. Either way
the answer is grounded in the retrieved documents.

Run:
    python 02_rag_pipeline.py
"""

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from knowledge_base import DOCUMENTS
from llm_backend import backend_name, get_chat_model


class TfidfRetriever:
    """A minimal vector retriever: TF-IDF + cosine similarity top-k."""

    def __init__(self, documents: list[str], k: int = 3):
        self.documents = documents
        self.k = k
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.doc_matrix = self.vectorizer.fit_transform(documents)

    def retrieve(self, query: str) -> list[str]:
        q_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(q_vec, self.doc_matrix).ravel()
        top_idx = sims.argsort()[::-1][: self.k]
        return [self.documents[i] for i in top_idx if sims[i] > 0]


def build_rag_chain(retriever: TfidfRetriever):
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system",
             "You are a support assistant. Answer the question using ONLY the "
             "context below. If the answer is not in the context, say you "
             "don't know."),
            ("human", "Context:\n{context}\n\nQuestion: {question}"),
        ]
    )
    model = get_chat_model()
    parser = StrOutputParser()

    def format_context(question: str) -> str:
        docs = retriever.retrieve(question)
        return "\n".join(f"- {d}" for d in docs) if docs else "(no documents found)"

    # Parallel mapping: build {context, question} from the incoming question,
    # then pipe into prompt | model | parser.
    return (
        {"context": RunnableLambda(format_context), "question": RunnablePassthrough()}
        | prompt
        | model
        | parser
    )


def main() -> None:
    print(f"Backend: {backend_name()}")
    retriever = TfidfRetriever(DOCUMENTS, k=3)
    chain = build_rag_chain(retriever)

    questions = [
        "How much does the Pro plan cost?",
        "How do I reset my password?",
        "How is my data secured?",
        "Which integrations are supported?",
        "What is the meaning of life?",  # not in the KB -> should decline
    ]

    for q in questions:
        print("\n" + "=" * 68)
        print(f"Q: {q}")
        retrieved = retriever.retrieve(q)
        print(f"Retrieved {len(retrieved)} doc(s):")
        for d in retrieved:
            print(f"   - {d[:70]}{'...' if len(d) > 70 else ''}")
        print(f"A: {chain.invoke(q)}")


if __name__ == "__main__":
    main()
