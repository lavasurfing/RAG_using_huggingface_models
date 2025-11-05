"""
ragbuilt5_uTest.py - Runnable RAG test using persisted Chroma DB and BART

This script builds a runnable chain that:
- loads a persisted Chroma vector store (./chroma_db, collection "thebook")
- retrieves context for a Machine Learning question
- formats context + question into a prompt
- uses a facebook/bart model via Hugging Face pipeline
- parses the model output to a plain string
"""

from operator import itemgetter

from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFacePipeline
from transformers import pipeline as hf_pipeline
from langchain_chroma import Chroma
from embedding import get_embedding_model


def load_persisted_vector_store(persist_directory: str = "./chroma_db", collection_name: str = "thebook") -> Chroma:
    embedding_model = get_embedding_model()
    vector_store = Chroma(
        collection_name=collection_name,
        embedding_function=embedding_model,
        persist_directory=persist_directory,
    )
    return vector_store


def build_chain(k: int = 5):
    # Retriever over persisted DB
    vector_store = load_persisted_vector_store()
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": k})

    def format_docs(docs):
        return "\n\n".join(d.page_content for d in docs)

    # Initialize facebook/bart model as a HF pipeline and wrap for LangChain
    text_gen = hf_pipeline(
        task="summarization",
        model="facebook/bart-large-cnn",
        device=-1,  # CPU for compatibility; set 0 for GPU if available
        max_new_tokens=128,
    )
    llm = HuggingFacePipeline(pipeline=text_gen)

    # question -> retrieve context -> concat context + question -> llm -> text
    chain = (
        {
            "context": itemgetter("question") | retriever | format_docs,
            "question": itemgetter("question"),
        }

        | RunnableLambda(lambda d: f"{d['context']}\n\nQuestion: {d['question']}")
        | llm
        | StrOutputParser()
    )
    return chain


def test_retrieval(question: str, k: int = 5) -> str:
    """
    Retrieve context for the given question, run it through a simple
    summarization pipeline (no custom prompt), and return the answer text.
    """
    # Setup retriever
    vector_store = load_persisted_vector_store()
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": k})
    docs = retriever.invoke(question)
    context = "\n\n".join(d.page_content for d in docs)
    combined_input = f"{context}\n\nQuestion: {question}"

    # Initialize QA pipeline
    qa_pipeline = hf_pipeline(
        task="summarization",
        model="text-generation",
        device=-1,  # CPU by default; set to 0 for GPU if available
    )
    
    # Run pipeline on the combined input and parse output text
    result = qa_pipeline(combined_input)
    if isinstance(result, list) and result and isinstance(result[0], dict) and "summary_text" in result[0]:
        return result[0]["summary_text"]
    if isinstance(result, str):
        return result
    return str(result)


if __name__ == "__main__":
    # chain = build_chain(k=5)
    question = "What are the prerqueisties for Machine Learning?"
    # print("\n🧪 Asking with retrieved context:\n")
    # answer = chain.invoke({"question": question})
    # print("Answer:\n" + (answer or "<no answer>"))

    # Also test direct retrieval + prompt + LLM function
    print("\n🧪 Testing direct retrieval function:\n")
    direct_answer = test_retrieval(question, k=5)
    print("Answer (direct retrieval):\n" + (direct_answer or "<no answer>"))

