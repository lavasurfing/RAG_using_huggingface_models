"""
ragBuilt0.py - Embeddings Initialization Module

WHAT THIS FILE DOES:
Think of this file as a translator. It takes regular text (words) and converts them 
into numbers that the computer can understand and compare. 

HOW IT WORKS:
- When you give it text like "What is artificial intelligence?", it converts each word 
  or sentence into a list of numbers (called a vector or embedding)
- These numbers represent the meaning of the text
- Similar meanings get similar numbers, so the computer can find related documents
- Uses Hugging Face's pre-trained models (like a smart dictionary that knows meanings)

EXAMPLE:
Input: "machine learning"
Output: [0.23, -0.45, 0.67, ...] (a list of numbers representing the meaning)
"""

import torch
from langchain_huggingface import HuggingFaceEmbeddings


def create_embeddings(embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
    """
    Create and return an embedding model that converts text to numbers.
    
    Think of this function as setting up the translator:
    - It loads a pre-trained model from Hugging Face
    - The model knows how to convert text to meaningful numbers
    - It tries to use GPU (graphics card) if available for speed, otherwise uses CPU
    
    Args:
        embedding_model: The name of the Hugging Face model to use
                         Default is a fast, efficient model
        
    Returns:
        An embedding object that can convert text to vectors (numbers)
    """
    try:
        # Try to create embeddings with GPU if available (faster)
        # Otherwise use CPU (slower but works on any computer)
        embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'},
            encode_kwargs={'normalize_embeddings': True}  # Makes comparisons more accurate
        )
        print(f"✅ Embeddings initialized with model: {embedding_model}")
        print(f"   Using: {'GPU (fast!)' if torch.cuda.is_available() else 'CPU (slower but works)'}")
        return embeddings
        
    except Exception as e:
        print(f"❌ Error initializing embeddings: {e}")
        # If something goes wrong, try again with CPU as backup
        print("   Trying again with CPU as backup...")
        try:
            embeddings = HuggingFaceEmbeddings(
                model_name=embedding_model,
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
            print(f"✅ Embeddings initialized with CPU fallback")
            return embeddings
        except Exception as e2:
            print(f"❌ Failed to initialize embeddings: {e2}")
            return None


# Example usage (if running this file directly)
if __name__ == "__main__":
    print("Testing embeddings initialization...")
    emb = create_embeddings()
    if emb:
        # Test: convert some text to numbers
        test_text = "Hello, this is a test"
        result = emb.embed_query(test_text)
        print(f"\nTest embedding created! Vector length: {len(result)}")
        print(f"First 5 numbers: {result[:5]}")
        print("\n✅ Embeddings module works correctly!")
