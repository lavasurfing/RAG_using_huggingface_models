"""
RAG Chatbot using Hugging Face Embeddings with LangChain
This implementation creates a Retrieval-Augmented Generation (RAG) chatbot
that uses Hugging Face embeddings for semantic search and retrieval.
"""

import os
import streamlit as st
from typing import List, Optional
from langchain_community.document_loaders import TextLoader, PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from transformers import pipeline
import torch

class RAGChatbot:
    """
    A RAG chatbot that uses Hugging Face embeddings for document retrieval
    and generation.
    """
    
    def __init__(self, 
                 embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
                 llm_model: str = "microsoft/DialoGPT-medium",
                 chunk_size: int = 1000,
                 chunk_overlap: int = 200):
        """
        Initialize the RAG chatbot.
        
        Args:
            embedding_model: Hugging Face model for embeddings
            llm_model: Hugging Face model for text generation
            chunk_size: Size of text chunks for splitting documents
            chunk_overlap: Overlap between chunks
        """
        self.embedding_model = embedding_model
        self.llm_model = llm_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Initialize components
        self.embeddings = None
        self.vectorstore = None
        self.retrieval_qa = None
        self.llm = None
        
        # Initialize the components
        self._initialize_embeddings()
        self._initialize_llm()
    
    def _initialize_embeddings(self):
        """Initialize Hugging Face embeddings."""
        try:
            self.embeddings = HuggingFaceEmbeddings(
                model_name=self.embedding_model,
                model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
            print(f"✅ Embeddings initialized with model: {self.embedding_model}")
        except Exception as e:
            print(f"❌ Error initializing embeddings: {e}")
            # Fallback to CPU
            self.embeddings = HuggingFaceEmbeddings(
                model_name=self.embedding_model,
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
            print(f"✅ Embeddings initialized with CPU fallback")
    
    def _initialize_llm(self):
        """Initialize the language model."""
        try:
            # Create a text generation pipeline
            pipe = pipeline(
                "text-generation",
                model=self.llm_model,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
                max_length=512,
                do_sample=True,
                temperature=0.7,
                pad_token_id=50256
            )
            
            self.llm = HuggingFacePipeline(pipeline=pipe)
            print(f"✅ LLM initialized with model: {self.llm_model}")
        except Exception as e:
            print(f"❌ Error initializing LLM: {e}")
            # Use a simpler fallback
            self.llm = None
            print("⚠️ LLM initialization failed, will use retrieval only")
    
    def load_documents(self, file_path: str) -> List:
        """
        Load documents from a file or directory.
        
        Args:
            file_path: Path to file or directory
            
        Returns:
            List of loaded documents
        """
        documents = []
        
        try:
            if os.path.isfile(file_path):
                # Single file
                if file_path.endswith('.pdf'):
                    loader = PyPDFLoader(file_path)
                else:
                    loader = TextLoader(file_path, encoding='utf-8')
                documents = loader.load()
            elif os.path.isdir(file_path):
                # Directory
                loader = DirectoryLoader(
                    file_path,
                    glob="**/*.txt",
                    loader_cls=TextLoader,
                    loader_kwargs={'encoding': 'utf-8'}
                )
                documents = loader.load()
            else:
                print(f"❌ Path not found: {file_path}")
                return []
                
            print(f"✅ Loaded {len(documents)} documents from {file_path}")
            return documents
            
        except Exception as e:
            print(f"❌ Error loading documents: {e}")
            return []
    
    def split_documents(self, documents: List) -> List:
        """
        Split documents into chunks for better retrieval.
        
        Args:
            documents: List of documents to split
            
        Returns:
            List of document chunks
        """
        try:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                length_function=len,
                separators=["\n\n", "\n", " ", ""]
            )
            
            chunks = text_splitter.split_documents(documents)
            print(f"✅ Split documents into {len(chunks)} chunks")
            return chunks
            
        except Exception as e:
            print(f"❌ Error splitting documents: {e}")
            return []
    
    def create_vectorstore(self, documents: List, save_path: Optional[str] = None):
        """
        Create a vector store from documents.
        
        Args:
            documents: List of document chunks
            save_path: Optional path to save the vector store
        """
        try:
            if not documents:
                print("❌ No documents provided for vector store creation")
                return
            
            print("🔄 Creating vector store...")
            self.vectorstore = FAISS.from_documents(
                documents=documents,
                embedding=self.embeddings
            )
            
            if save_path:
                self.vectorstore.save_local(save_path)
                print(f"✅ Vector store saved to {save_path}")
            else:
                print("✅ Vector store created in memory")
                
        except Exception as e:
            print(f"❌ Error creating vector store: {e}")
    
    def load_vectorstore(self, path: str):
        """
        Load an existing vector store.
        
        Args:
            path: Path to the saved vector store
        """
        try:
            self.vectorstore = FAISS.load_local(
                path, 
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            print(f"✅ Vector store loaded from {path}")
        except Exception as e:
            print(f"❌ Error loading vector store: {e}")
    
    def setup_retrieval_qa(self, k: int = 4):
        """
        Set up the retrieval QA chain.
        
        Args:
            k: Number of documents to retrieve
        """
        try:
            if not self.vectorstore:
                print("❌ Vector store not initialized")
                return
            
            # Create a custom prompt template
            prompt_template = """Use the following pieces of context to answer the question at the end. 
            If you don't know the answer based on the context, just say that you don't know, don't try to make up an answer.

            Context:
            {context}

            Question: {question}

            Answer:"""
            
            PROMPT = PromptTemplate(
                template=prompt_template,
                input_variables=["context", "question"]
            )
            
            # Create retriever
            retriever = self.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": k}
            )
            
            if self.llm:
                # Create QA chain with LLM
                self.retrieval_qa = RetrievalQA.from_chain_type(
                    llm=self.llm,
                    chain_type="stuff",
                    retriever=retriever,
                    chain_type_kwargs={"prompt": PROMPT},
                    return_source_documents=True
                )
                print("✅ Retrieval QA chain with LLM created")
            else:
                # Create simple retrieval without generation
                self.retrieval_qa = retriever
                print("✅ Simple retriever created (no LLM)")
                
        except Exception as e:
            print(f"❌ Error setting up retrieval QA: {e}")
    
    def query(self, question: str) -> dict:
        """
        Query the RAG system.
        
        Args:
            question: The question to ask
            
        Returns:
            Dictionary containing answer and source documents
        """
        try:
            if not self.retrieval_qa:
                return {"error": "Retrieval QA not initialized"}
            
            if self.llm:
                # Full RAG with generation
                result = self.retrieval_qa({"query": question})
                return {
                    "answer": result["result"],
                    "source_documents": result["source_documents"]
                }
            else:
                # Simple retrieval only
                docs = self.retrieval_qa.get_relevant_documents(question)
                return {
                    "answer": "LLM not available. Retrieved relevant documents:",
                    "source_documents": docs
                }
                
        except Exception as e:
            return {"error": f"Error querying: {e}"}
    
    def chat(self):
        """Interactive chat interface."""
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


def main():
    """Main function to demonstrate the RAG chatbot."""
    
    # Initialize the chatbot
    print("🚀 Initializing RAG Chatbot...")
    chatbot = RAGChatbot(
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        llm_model="microsoft/DialoGPT-medium",
        chunk_size=1000,
        chunk_overlap=200
    )
    
    # Example usage
    print("\n📝 Example: Loading documents...")
    
    # Create a sample document for demonstration
    sample_text = """
    Artificial Intelligence (AI) is intelligence demonstrated by machines, in contrast to the natural intelligence displayed by humans and animals. 
    Leading AI textbooks define the field as the study of "intelligent agents": any device that perceives its environment and takes actions that maximize its chance of successfully achieving its goals.
    
    Machine Learning (ML) is a subset of AI that focuses on algorithms that can learn from and make predictions or decisions based on data. 
    It has applications in many fields including computer vision, speech recognition, email filtering, agriculture, and medicine.
    
    Natural Language Processing (NLP) is a subfield of linguistics, computer science, and AI concerned with the interactions between computers and human language. 
    It focuses on how to program computers to process and analyze large amounts of natural language data.
    
    Deep Learning is part of a broader family of machine learning methods based on artificial neural networks with representation learning. 
    Learning can be supervised, semi-supervised or unsupervised.
    """
    
    # Save sample text to a file
    with open("sample_document.txt", "w", encoding="utf-8") as f:
        f.write(sample_text)
    
    # Load and process documents
    documents = chatbot.load_documents("sample_document.txt")
    if documents:
        chunks = chatbot.split_documents(documents)
        chatbot.create_vectorstore(chunks)
        chatbot.setup_retrieval_qa(k=3)
        
        # Start interactive chat
        chatbot.chat()
    
    # Clean up
    if os.path.exists("sample_document.txt"):
        os.remove("sample_document.txt")


if __name__ == "__main__":
    main()
