"""
ragBuilt8.py - Streamlit Web Interface for RAG Chatbot

WHAT THIS FILE DOES:
This file creates a beautiful web interface using Streamlit. Instead of using the 
command line, users can interact with the chatbot through a web browser. It's like 
creating a website for your chatbot!

HOW IT WORKS:
- Uses Streamlit to create a web page
- Users can upload documents through the web interface
- Configure models and parameters with sliders and dropdowns
- Chat with the bot in a nice visual interface
- See source documents used for each answer

EXAMPLE:
Run: streamlit run ragBuilt8.py
Then open browser to see the interface
"""

import streamlit as st
import os
import tempfile
from typing import List, Optional
from ragBuilt0 import create_embeddings
from ragBuilt1 import create_llm
from ragBuilt2 import load_documents
from ragBuilt3 import split_documents
from ragBuilt4 import create_vectorstore
from ragBuilt5 import setup_retrieval_qa
from ragBuilt6 import query_rag_system
from langchain_community.document_loaders import TextLoader, PyPDFLoader

# Configure the web page
st.set_page_config(
    page_title="RAG Chatbot with Hugging Face",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add some nice styling to the page
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .user-message {
        background-color: #e3f2fd;
        margin-left: 20%;
    }
    .bot-message {
        background-color: #f3e5f5;
        margin-right: 20%;
    }
</style>
""", unsafe_allow_html=True)


def process_uploaded_files(uploaded_files, embeddings, chunk_size: int, chunk_overlap: int):
    """
    Process files uploaded through the web interface.
    
    This function takes files uploaded by the user, saves them temporarily,
    loads them as documents, splits them, and creates a vector store.
    
    Returns the vectorstore if successful, None otherwise.
    """
    if not uploaded_files:
        return None
    
    try:
        documents = []
        
        # For each uploaded file, save it temporarily and load it
        for uploaded_file in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{uploaded_file.name}") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name
            
            try:
                # Load based on file type
                if uploaded_file.name.endswith('.pdf'):
                    loader = PyPDFLoader(tmp_file_path)
                else:
                    loader = TextLoader(tmp_file_path, encoding='utf-8')
                
                docs = loader.load()
                documents.extend(docs)
                
            finally:
                # Clean up temporary file
                os.unlink(tmp_file_path)
        
        if not documents:
            st.error("No documents could be loaded.")
            return None
        
        # Split documents
        with st.spinner("Splitting documents into chunks..."):
            chunks = split_documents(documents, chunk_size, chunk_overlap)
        
        # Create vector store
        with st.spinner("Creating vector store (converting to searchable format)..."):
            vectorstore = create_vectorstore(chunks, embeddings)
        
        if vectorstore:
            st.success(f"✅ Processed {len(documents)} documents into {len(chunks)} chunks")
        
        return vectorstore
        
    except Exception as e:
        st.error(f"Error processing files: {e}")
        return None


def main():
    """Main Streamlit application."""
    
    # Page header
    st.markdown('<h1 class="main-header">🤖 RAG Chatbot with Hugging Face</h1>', unsafe_allow_html=True)
    
    # Initialize session state (keeps data between page refreshes)
    if 'embeddings' not in st.session_state:
        st.session_state.embeddings = None
    if 'llm' not in st.session_state:
        st.session_state.llm = None
    if 'vectorstore' not in st.session_state:
        st.session_state.vectorstore = None
    if 'retrieval_qa' not in st.session_state:
        st.session_state.retrieval_qa = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Model selection
        embedding_model = st.selectbox(
            "Embedding Model (converts text to numbers)",
            [
                "sentence-transformers/all-MiniLM-L6-v2",
                "sentence-transformers/all-mpnet-base-v2",
                "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
            ],
            index=0
        )
        
        llm_model = st.selectbox(
            "Language Model (generates answers)",
            [
                "microsoft/DialoGPT-medium",
                "gpt2",
                "distilgpt2"
            ],
            index=0
        )
        
        # Document processing settings
        st.subheader("📄 Document Processing")
        chunk_size = st.slider("Chunk Size (characters)", 500, 2000, 1000)
        chunk_overlap = st.slider("Chunk Overlap (characters)", 50, 500, 200)
        k_docs = st.slider("Number of documents to retrieve", 1, 10, 4)
        
        # Initialize models button
        if st.button("🚀 Initialize Models", type="primary"):
            with st.spinner("Loading models (this may take a few minutes on first run)..."):
                st.session_state.embeddings = create_embeddings(embedding_model)
                st.session_state.llm = create_llm(llm_model)
                
                if st.session_state.embeddings:
                    st.success("✅ Embeddings loaded!")
                if st.session_state.llm:
                    st.success("✅ Language model loaded!")
                elif not st.session_state.llm:
                    st.info("ℹ️ Using retrieval only (no LLM)")
    
    # Main content area - two columns
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📁 Upload Documents")
        
        # File uploader
        uploaded_files = st.file_uploader(
            "Choose files",
            type=['txt', 'pdf'],
            accept_multiple_files=True,
            help="Upload text or PDF files to create your knowledge base"
        )
        
        if uploaded_files and st.button("📚 Process Documents"):
            if st.session_state.embeddings:
                st.session_state.vectorstore = process_uploaded_files(
                    uploaded_files, 
                    st.session_state.embeddings, 
                    chunk_size, 
                    chunk_overlap
                )
                if st.session_state.vectorstore:
                    st.session_state.retrieval_qa = setup_retrieval_qa(
                        st.session_state.vectorstore, 
                        st.session_state.llm, 
                        k_docs
                    )
                    st.session_state.chat_history = []  # Clear chat history
                    st.success("✅ Documents processed! You can now ask questions.")
            else:
                st.error("Please initialize the embedding model first!")
    
    with col2:
        st.header("💬 Chat Interface")
        
        if not st.session_state.retrieval_qa:
            st.info("👆 Please upload and process documents first!")
        else:
            # Chat input
            user_question = st.text_input(
                "Ask a question about your documents:",
                placeholder="What is artificial intelligence?",
                key="user_input"
            )
            
            if st.button("🔍 Ask", type="primary") and user_question:
                # Add user message to chat history
                st.session_state.chat_history.append({
                    "type": "user",
                    "content": user_question
                })
                
                # Get response
                with st.spinner("Thinking..."):
                    result = query_rag_system(st.session_state.retrieval_qa, user_question)
                
                if "error" in result:
                    st.session_state.chat_history.append({
                        "type": "bot",
                        "content": f"Error: {result['error']}"
                    })
                else:
                    st.session_state.chat_history.append({
                        "type": "bot",
                        "content": result["answer"],
                        "sources": result.get("source_documents", [])
                    })
    
    # Display chat history
    if st.session_state.chat_history:
        st.header("💭 Chat History")
        
        for message in st.session_state.chat_history:
            if message["type"] == "user":
                st.markdown(f"""
                <div class="chat-message user-message">
                    <strong>You:</strong> {message["content"]}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="chat-message bot-message">
                    <strong>🤖 Bot:</strong> {message["content"]}
                </div>
                """, unsafe_allow_html=True)
                
                # Show sources if available
                if "sources" in message and message["sources"]:
                    with st.expander(f"📚 Sources ({len(message['sources'])} documents)"):
                        for j, doc in enumerate(message["sources"][:3], 1):
                            content = doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content
                            st.markdown(f"""
                            **Source {j}:**
                            {content}
                            """)
        
        # Clear chat button
        if st.button("🗑️ Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>Built with ❤️ using Streamlit, LangChain, and Hugging Face Transformers</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
