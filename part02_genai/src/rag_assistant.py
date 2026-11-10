
from pathlib import Path

from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.vectorstores import FAISS

# Locate the saved FAISS index
PROJECT_DIR = Path(__file__).resolve().parents[1]
INDEX_DIR = PROJECT_DIR / "data" / "faiss_index"

# Load the local embedding model and knowledge base
embeddings = OllamaEmbeddings(model="nomic-embed-text")

vector_store = FAISS.load_local(
    str(INDEX_DIR),
    embeddings,
    allow_dangerous_deserialization=True,
)

# Load the local language model
llm = ChatOllama(
    model="llama3.2:3b",
    temperature=0,
)

def ask_weather_question(question: str) -> str:
    # Retrieve the most relevant knowledge passages
    documents = vector_store.similarity_search(question, k=3)

    context = "\n\n".join(
        f"Source: {doc.metadata.get('source', 'Unknown')}\n"
        f"{doc.page_content}"
        for doc in documents
    )

    prompt = f"""
You are a Zimbabwe Weather Research Assistant.

Answer the user's question using the supplied context.
Do not invent facts or claim that the context says something it does not.
If the context is insufficient, clearly say so.
Distinguish observations from model predictions.

KNOWLEDGE CONTEXT:
{context}

USER QUESTION:
{question}

Provide a clear, concise answer.
"""

    response = llm.invoke(prompt)
    return response.content


if __name__ == "__main__":
    question = input("Ask a weather research question: ").strip()

    if question:
        print("\nGenerating answer...\n")
        answer = ask_weather_question(question)
        print("ANSWER")
        print(answer)
