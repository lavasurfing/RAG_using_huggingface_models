"""
ragBuilt9.py - Command Line Interface / Main Entry Point

WHAT THIS FILE DOES:
This is the main file you run from the command line. It demonstrates how to use 
the RAG chatbot and provides a simple example workflow. Think of it as the 
"main program" that shows everything working together.

HOW IT WORKS:
- Imports the RAGChatbot class from ragBuilt7.py
- Creates a sample document for demonstration
- Loads and processes the document
- Starts an interactive chat session
- Cleans up temporary files when done

EXAMPLE:
Run: python ragBuilt9.py
This will start an interactive chat session with the bot.
"""

import os
from ragBuilt7 import RAGChatbot


def main():
    """
    Main function to demonstrate the RAG chatbot.
    
    This function:
    1. Creates a chatbot instance
    2. Creates a sample document
    3. Loads and processes it
    4. Starts chatting
    5. Cleans up
    """
    
    print("="*60)
    print("🚀 Initializing RAG Chatbot...")
    print("="*60)
    
    # Initialize the chatbot
    # This automatically loads embeddings and LLM
    chatbot = RAGChatbot(
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        llm_model="microsoft/DialoGPT-medium",
        chunk_size=1000,
        chunk_overlap=200
    )
    
    # Check if chatbot initialized properly
    if not chatbot.embeddings:
        print("❌ Failed to initialize chatbot. Exiting.")
        return
    
    # Example usage: Create a sample document
    print("\n📝 Creating sample document for demonstration...")
    
    sample_text = """
    Artificial Intelligence (AI) is intelligence demonstrated by machines, in contrast to the natural intelligence displayed by humans and animals. 
    Leading AI textbooks define the field as the study of "intelligent agents": any device that perceives its environment and takes actions that maximize its chance of successfully achieving its goals.
    
    Machine Learning (ML) is a subset of AI that focuses on algorithms that can learn from and make predictions or decisions based on data. 
    It has applications in many fields including computer vision, speech recognition, email filtering, agriculture, and medicine.
    
    Natural Language Processing (NLP) is a subfield of linguistics, computer science, and AI concerned with the interactions between computers and human language. 
    It focuses on how to program computers to process and analyze large amounts of natural language data.
    
    Deep Learning is part of a broader family of machine learning methods based on artificial neural networks with representation learning. 
    Learning can be supervised, semi-supervised or unsupervised.
    
    Neural Networks are computing systems inspired by biological neural networks. They consist of interconnected nodes (neurons) that process information.
    These networks can learn patterns from data through a process called training.
    
    Computer Vision is a field of AI that enables machines to interpret and understand visual information from the world. 
    It involves techniques for acquiring, processing, analyzing, and understanding digital images and videos.
    """
    
    # Save sample text to a temporary file
    sample_file = "sample_document.txt"
    with open(sample_file, "w", encoding="utf-8") as f:
        f.write(sample_text)
    
    print(f"✅ Sample document created: {sample_file}")
    
    # Load and process documents
    print("\n📚 Loading and processing documents...")
    documents = chatbot.load_documents(sample_file)
    
    if documents:
        # Split into chunks
        chunks = chatbot.split_documents(documents)
        
        # Create vector store
        chatbot.create_vectorstore(chunks)
        
        # Set up Q&A system
        chatbot.setup_retrieval_qa(k=3)
        
        print("\n" + "="*60)
        print("✅ Setup complete! Ready to answer questions.")
        print("="*60)
        
        # Start interactive chat
        chatbot.chat()
    else:
        print("❌ Failed to load documents. Cannot proceed.")
    
    # Clean up temporary file
    print("\n🧹 Cleaning up...")
    if os.path.exists(sample_file):
        os.remove(sample_file)
        print(f"✅ Removed temporary file: {sample_file}")
    
    print("\n👋 Thank you for using RAG Chatbot!")


if __name__ == "__main__":
    main()
