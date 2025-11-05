"""
ragBuilt1.py - Language Model (LLM) Initialization Module

WHAT THIS FILE DOES:
This file sets up the AI "brain" that can understand questions and generate answers.
Think of it as the conversation partner - it reads context from documents and creates 
human-like responses.

HOW IT WORKS:
- Loads a pre-trained language model (like GPT-style models) from Hugging Face
- This model has been trained on lots of text to understand language patterns
- When given a question and context, it can generate a coherent answer
- The model converts input text into predictions about what words should come next

EXAMPLE:
Input Question: "What is machine learning?"
Input Context: "Machine Learning is a subset of AI that learns from data..."
Output Answer: "Machine learning is a subset of artificial intelligence that focuses 
                on algorithms that can learn from and make predictions based on data..."
"""

import torch
from transformers import pipeline
from langchain_community.llms import HuggingFacePipeline
from sample_story import send_story


def create_llm(llm_model: str = "facebook/bart-large-cnn"):
    """
    Create and return a language model that can generate text answers.
    
    Think of this function as waking up the AI brain:
    - It loads a pre-trained language model (has learned from millions of text examples)
    - The model can take a question + context and generate an answer
    - It tries to use GPU if available for faster generation
    
    Args:
        llm_model: The name of the Hugging Face language model to use
                   Default is a conversational model good for Q&A
    
    Returns:
        A language model object that can generate answers, or None if it fails
    """
    try:
        print(f"🔄 Loading language model: {llm_model}")
        print("   This may take a few minutes on first run (downloading model)...")
        
        # Create a text generation pipeline
        # This is like setting up a factory that converts input to output
        model = pipeline(
            task="summarization",
            model=llm_model,
            device=0,
            max_new_tokens=120,
            truncation=True,  # Number of new tokens to generate
        )
        
        # Wrap it in LangChain's interface so we can use it easily
        # llm = HuggingFacePipeline(pipeline=model)
        llm = model
        print(f"✅ LLM initialized with model: {llm_model}")
        print(f"   Using: {'GPU (fast!)' if torch.cuda.is_available() else 'CPU (slower)'}")
        return llm
        
    except Exception as e:
        print(f"❌ Error initializing LLM: {e}")
        print("⚠️  LLM initialization failed, will use retrieval only")
        print("   (This means we can find relevant documents but can't generate fancy answers)")
        return None


# Example usage (if running this file directly)
if __name__ == "__main__":
    print("Testing LLM initialization...")
    print("Note: This will download the model on first run (can take time)...\n")
    
    llm = create_llm('facebook/bart-large-cnn')
    if llm:

# Test Data for the pipe
        res = llm(send_story())       
        
        print(res)
        
        print("\n✅ LLM module works correctly!")
        print("\nNote: You can now use this LLM to generate answers from context.")
    else:
        print("\n⚠️  LLM not available, but the RAG system can still work with retrieval only")
