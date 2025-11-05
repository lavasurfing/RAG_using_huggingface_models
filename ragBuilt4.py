"""
ragBuilt4.py - Simple Chroma vector store helper (based on splitChunk.py)

Minimal helpers to build and query a Chroma vector store.
"""

from uuid import uuid4
from langchain_chroma import Chroma
from embedding import get_embedding_model
from loader import load_pdf
from ragBuilt3 import split_documents


def build_vector_store() -> Chroma:
    thepdf = load_pdf()
    chunks = split_documents(thepdf)

    ids = [str(uuid4()) for _ in range(len(chunks))]
    embedding_model = get_embedding_model()

    vector_store = Chroma(
        collection_name="thebook",
        embedding_function=embedding_model,
        persist_directory="./chroma_db"
    )

    vector_store.add_documents(documents=chunks, ids=ids)
    print("Added document chunks to vector store.")
    return vector_store


def get_retriever(k: int = 5):
    embedding_model = get_embedding_model()
    vector_store = Chroma(
        collection_name="thebook",
        embedding_function=embedding_model,
        persist_directory="./chroma_db"
    )
    return vector_store.as_retriever(search_type="similarity", search_kwargs={"k": k})


if __name__ == "__main__":
    vs = build_vector_store()
    retriever = vs.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    print(retriever.invoke("What is Machine Learning?"))
