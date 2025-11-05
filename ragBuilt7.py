"""
ragBuilt7.py - Main RAG Chatbot Class

WHAT THIS FILE DOES:
This file brings everything together into one easy-to-use class. Think of it as 
the main control center - it combines all the individual modules (embeddings, LLM, 
document loading, etc.) into a single RAGChatbot class that you can use easily.

HOW IT WORKS:
- Combines all the separate modules we created (ragBuilt0 through ragBuilt6)
- Provides a simple interface: create chatbot → load documents → ask questions
- Handles all the complexity behind the scenes
- Provides methods for common tasks (load docs, create vector store, query, chat)

EXAMPLE:
    chatbot = RAGChatbot()
    documents = chatbot.load_documents("my_file.txt")
    chunks = chatbot.split_documents(documents)
    chatbot.create_vectorstore(chunks)
    chatbot.setup_retrieval_qa(k=3)
    answer = chatbot.query("What is AI?")
"""

from typing import List, Optional
from ragBuilt0 import create_embeddings
from ragBuilt1 import create_llm
from ragBuilt2 import load_documents
from ragBuilt3 import split_documents
from ragBuilt4 import create_vectorstore, load_vectorstore
from ragBuilt5 import setup_retrieval_qa
from ragBuilt6 import query_rag_system


class RAGChatbot:
    """
    A complete RAG chatbot that combines all modules.
    
    This class is like a smart assistant that:
    - Can read and understand your documents
    - Can answer questions based on those documents
    - Handles all the technical stuff for you
    """
    
    def __init__(self, 
                 embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
                 llm_model: str = "microsoft/DialoGPT-medium",
                 chunk_size: int = 1000,
                 chunk_overlap: int = 200):
        """
        Initialize the RAG chatbot.
        
        This sets up everything needed for the chatbot:
        - Creates embeddings (text to numbers converter)
        - Creates LLM (answer generator)
        - Sets chunking parameters
        
        Args:
            embedding_model: Which model to use for converting text to numbers
            llm_model: Which language model to use for generating answers
            chunk_size: How big each document chunk should be
            chunk_overlap: How much chunks should overlap
        """
        self.embedding_model = embedding_model
        self.llm_model = llm_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Initialize components (using our modular functions)
        print("🚀 Initializing RAG Chatbot components...")
        self.embeddings = create_embeddings(embedding_model)
        self.llm = create_llm(llm_model)
        self.vectorstore = None
        self.retrieval_qa = None
        
        if self.embeddings:
            print("✅ Chatbot initialization complete!\n")
        else:
            print("⚠️  Chatbot initialized but embeddings failed")
    
    def load_documents(self, file_path: str) -> List:
        """
        Load documents from a file or directory.
        
        Uses the document loading module (ragBuilt2.py).
        """
        return load_documents(file_path)
    
    def split_documents(self, documents: List) -> List:
        """
        Split documents into chunks.
        
        Uses the document splitting module (ragBuilt3.py).
        """
        return split_documents(documents, self.chunk_size, self.chunk_overlap)
    
    def create_vectorstore(self, documents: List, save_path: Optional[str] = None):
        """
        Create a vector store from document chunks.
        
        Uses the vector store module (ragBuilt4.py).
        Stores it in self.vectorstore for later use.
        """
        if not self.embeddings:
            print("❌ Embeddings not initialized")
            return
        
        self.vectorstore = create_vectorstore(documents, self.embeddings, save_path)
    
    def load_vectorstore(self, path: str):
        """
        Load a previously saved vector store.
        
        Uses the vector store module (ragBuilt4.py).
        """
        if not self.embeddings:
            print("❌ Embeddings not initialized")
            return
        
        self.vectorstore = load_vectorstore(path, self.embeddings)
    
    def setup_retrieval_qa(self, k: int = 4):
        """
        Set up the question-answering system.
        
        Uses the retrieval QA module (ragBuilt5.py).
        Connects the vector store with the LLM.
        """
        if not self.vectorstore:
            print("❌ Vector store not initialized. Load or create documents first.")
            return
        
        self.retrieval_qa = setup_retrieval_qa(self.vectorstore, self.llm, k)
    
    def query(self, question: str) -> dict:
        """
        Ask a question and get an answer.
        
        Uses the query module (ragBuilt6.py).
        
        Args:
            question: Your question as a string
            
        Returns:
            Dictionary with "answer" and "source_documents"
        """
        if not self.retrieval_qa:
            return {"error": "Retrieval QA not initialized. Run setup_retrieval_qa() first."}
        
        return query_rag_system(self.retrieval_qa, question)
    
    def chat(self):
        """
        Start an interactive chat session.
        
        This is a simple command-line chat interface.
        Type your questions and get answers!
        """
        if not self.retrieval_qa:
            print("❌ Retrieval QA not initialized. Cannot start chat.")
            print("   Make sure you've loaded documents and run setup_retrieval_qa()")
            return
        
        print("\n🤖 RAG Chatbot Ready!")
        print("Type 'quit' to exit, 'help' for commands\n")
        
        while True:
            try:
                question = input("You: ").strip()
                
                if question.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                elif question.lower() == 'help':
                    print("\n📋 Available commands:")
                    print("- Ask any question about your documents")
                    print("- 'quit' or 'exit' to end the chat")
                    print("- 'help' to show this message\n")
                    continue
                elif not question:
                    continue
                
                print("🔄 Thinking...")
                result = self.query(question)
                
                if "error" in result:
                    print(f"❌ {result['error']}")
                else:
                    print(f"\n🤖 Bot: {result['answer']}")
                    
                    if result.get('source_documents'):
                        print(f"\n📚 Sources ({len(result['source_documents'])} documents):")
                        for i, doc in enumerate(result['source_documents'][:3], 1):
                            content = doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
                            print(f"  {i}. {content}")
                
                print("\n" + "="*50 + "\n")
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")


# Example usage
if __name__ == "__main__":
    print("This is the main RAG Chatbot class.")
    print("Import it and use like this:")
    print("\nfrom ragBuilt7 import RAGChatbot")
    print("\nchatbot = RAGChatbot()")
    print("docs = chatbot.load_documents('file.txt')")
    print("chunks = chatbot.split_documents(docs)")
    print("chatbot.create_vectorstore(chunks)")
    print("chatbot.setup_retrieval_qa(k=3)")
    print("chatbot.chat()")
