"""
ragBuilt.py - Main RAG Chatbot (Uses Modular Components)

WHAT THIS FILE DOES:
This file imports and uses all the modular components we created. It's kept 
for backward compatibility. The actual implementation is now in separate modules:
- ragBuilt0.py through ragBuilt9.py

For new projects, use ragBuilt7.py (main class) and ragBuilt9.py (CLI) or 
ragBuilt8.py (Streamlit web interface).
"""

import os
from typing import List, Optional

# Import all our modular components
from ragBuilt0 import create_embeddings
from ragBuilt1 import create_llm
from ragBuilt2 import load_documents
from ragBuilt3 import split_documents
from ragBuilt4 import create_vectorstore, load_vectorstore
from ragBuilt5 import setup_retrieval_qa
from ragBuilt6 import query_rag_system

# Import the main class
from ragBuilt7 import RAGChatbot

# The RAGChatbot class is now imported from ragBuilt7.py
# This file is kept for backward compatibility
# Use ragBuilt7.RAGChatbot for the main class


def main():
    """
    Main function - redirects to ragBuilt9.py for CLI interface.
    
    For command line usage, run: python ragBuilt9.py
    For web interface, run: streamlit run ragBuilt8.py
    """
    print("="*60)
    print("RAG Chatbot - Modular Version")
    print("="*60)
    print("\nThis file is for backward compatibility.")
    print("For command line interface, use: python ragBuilt9.py")
    print("For web interface, use: streamlit run ragBuilt8.py")
    print("\nOr import and use directly:")
    print("  from ragBuilt7 import RAGChatbot")
    print("  chatbot = RAGChatbot()")
    print("="*60)
    
    # Optionally, you can still use it like before
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--run":
        # Actually run if --run flag is provided
        from ragBuilt9 import main as run_main
        run_main()
    else:
        print("\n💡 Tip: Run with --run flag to execute, or use ragBuilt9.py directly")


if __name__ == "__main__":
    main()

