"""
ragBuilt6.py - Query Functionality Module

WHAT THIS FILE DOES:
This file handles asking questions to your RAG system. It's the interface between 
you and the AI. You give it a question, and it returns an answer along with the 
documents that were used to answer it.

Think of it as the question-asking function:
- You: "What is artificial intelligence?"
- System: Finds relevant documents, generates answer, returns both

HOW IT WORKS:
- Takes a question as input
- Passes it to the retrieval QA system (from ragBuilt5.py)
- Gets back an answer and source documents
- Returns everything in a clean dictionary format
- Handles errors gracefully

EXAMPLE:
Input: question = "What is machine learning?"
Output: {
    "answer": "Machine learning is a subset of AI...",
    "source_documents": [doc1, doc2, doc3]
}
"""


def query_rag_system(retrieval_qa, question: str) -> dict:
    """
    Ask a question to the RAG system and get an answer.
    
    Think of this function as asking a question to a smart librarian:
    - You ask: "What is machine learning?"
    - Librarian searches the database (vector store)
    - Finds relevant books/documents
    - Reads them and gives you an answer
    - Also tells you which books/documents were used
    
    Args:
        retrieval_qa: The Q&A system set up in ragBuilt5.py
                     Can be a full QA chain (with LLM) or just a retriever
        question: Your question as a string
                 Example: "What is artificial intelligence?"
    
    Returns:
        A dictionary with:
        - "answer": The generated answer (or notice if no LLM)
        - "source_documents": List of documents used to answer
        - "error": Error message if something went wrong
    """
    try:
        if not retrieval_qa:
            return {"error": "Retrieval QA system not initialized"}
        
        if not question or not question.strip():
            return {"error": "Question cannot be empty"}
        
        # Check if we have a full QA chain (with LLM) or just a retriever
        # Full QA chains have a "__call__" method that takes {"query": question}
        # Simple retrievers have "get_relevant_documents" method
        
        if hasattr(retrieval_qa, "__call__") or hasattr(retrieval_qa, "invoke"):
            # Full RAG system with answer generation
            try:
                # Try new LangChain API first
                if hasattr(retrieval_qa, "invoke"):
                    result = retrieval_qa.invoke({"query": question})
                else:
                    # Fallback to old API
                    result = retrieval_qa({"query": question})
                
                return {
                    "answer": result["result"],
                    "source_documents": result.get("source_documents", [])
                }
            except Exception as e:
                return {"error": f"Error during query processing: {e}"}
        else:
            # Just retrieval, no answer generation
            try:
                docs = retrieval_qa.get_relevant_documents(question)
                return {
                    "answer": "LLM not available. Here are the most relevant document chunks:",
                    "source_documents": docs
                }
            except Exception as e:
                return {"error": f"Error during retrieval: {e}"}
                
    except Exception as e:
        return {"error": f"Unexpected error: {e}"}


# Example usage (if running this file directly)
if __name__ == "__main__":
    print("Testing query functionality...")
    print("\nTo use this module:")
    print("   1. Set up retrieval_qa from ragBuilt5")
    print("   2. Ask questions:")
    print("      result = query_rag_system(retrieval_qa, 'What is AI?')")
    print("      print(result['answer'])")
    print("\n✅ Query module is ready!")
