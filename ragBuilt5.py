"""
ragBuilt5.py - Minimal Runnables-based RAG pipeline

This file intentionally keeps only the essentials:
- initialize embeddings and LLM
- build/load vector store and retriever
- compose a LangChain Runnable chain (retriever → prompt → LLM → text)
- simple test in __main__
"""

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_huggingface import HuggingFacePipeline
from transformers import pipeline as hf_pipeline
from operator import itemgetter

from ragBuilt0 import create_embeddings
from ragBuilt4 import build_vector_store


def build_runnable_chain(question: str, k: int = 4):
    vectorstore = build_vector_store()

    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": k})

    docs = retriever.invoke(question)
    context = "\n\n".join(d.page_content for d in docs)
    combined_input = f"{context}\n\nQuestion: {question}"

    # Initialize QA pipeline
    qa_pipeline = hf_pipeline(
        task="summarization",
        model="facebook/bart-large-cnn",
        device=0,  # CPU by default; set to 0 for GPU if available
        truncation=True,
        max_new_tokens=128,
        max_length=128,
    )
    
    # Run pipeline on the combined input and parse output text
    result = qa_pipeline(combined_input)
    if isinstance(result, list) and result and isinstance(result[0], dict) and "summary_text" in result[0]:
        return result[0]["summary_text"]
    if isinstance(result, str):
        return result
    return str(result)


if __name__ == "__main__":
    # Initialize (embeddings are used inside vector DB and retriever)
    print("Testing Hugging Face Pipeline...")
    _ = create_embeddings()
    question = "What is prerequisites for Machine Learning?"
    print(f"\n🧪 Query: {question}\n")
    print("Answer:\n" + (build_runnable_chain(question, k=4) or "<no answer>"))
