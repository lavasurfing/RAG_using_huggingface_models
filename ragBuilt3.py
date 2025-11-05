"""
ragBuilt3.py - Document Splitting Module

WHAT THIS FILE DOES:
This file breaks long documents into smaller pieces (chunks). Think of it like 
cutting a long article into paragraphs or sections. This is important because:
- Large documents are too big for the AI to process all at once
- Smaller chunks make it easier to find the exact relevant information
- Overlapping chunks ensure no information is lost at the boundaries

HOW IT WORKS:
- Takes a list of documents as input
- Splits each document into smaller chunks based on size
- Keeps some overlap between chunks (so words at boundaries aren't lost)
- Uses smart separators (paragraphs, sentences, words) to split naturally
- Returns all the chunks in a list

EXAMPLE:
Input: One 5000-word document
Output: 5 chunks of ~1000 words each, with 200 words overlapping between adjacent chunks
"""

from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from ragBuilt2 import load_pdf

def split_documents(documents: List, chunk_size: int = 1000, chunk_overlap: int = 200) -> List:
    """
    Split documents into smaller chunks for better processing.
    
    Think of this function as a smart paper cutter:
    - Takes long documents and cuts them into smaller, manageable pieces
    - Each chunk is about 'chunk_size' characters long
    - Adjacent chunks overlap by 'chunk_overlap' characters (so nothing is lost)
    - Uses natural break points (paragraphs, sentences) when possible
    
    Args:
        documents: List of documents to split (from ragBuilt2.py)
        chunk_size: How big each chunk should be (default: 1000 characters)
                   Smaller = more precise, but more chunks to process
                   Larger = fewer chunks, but each chunk is harder to process
        chunk_overlap: How much adjacent chunks should overlap (default: 200 characters)
                      This ensures sentences split at boundaries aren't lost
    
    Returns:
        A list of document chunks, each ready to be converted to vectors
        Returns empty list [] if something goes wrong
    """
    try:
        if not documents:
            print("❌ No documents provided to split")
            return []
        
        print(f"✂️  Splitting {len(documents)} document(s) into chunks...")
        print(f"   Chunk size: {chunk_size} characters")
        print(f"   Overlap: {chunk_overlap} characters")
        
        # Create a text splitter that's smart about where to cut
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,      # Target size for each chunk
            chunk_overlap=chunk_overlap, # Overlap between chunks
            length_function=len,         # How to measure length (character count)
            separators=["\n\n", "\n", " ", ""]  # Try to split at paragraphs first, 
                                                  # then sentences, then words, 
                                                  # finally anywhere if needed
        )
        
        # Actually split the documents
        chunks = text_splitter.split_documents(documents)
        
        print(f"✅ Split into {len(chunks)} chunks")
        if len(chunks) > 0:
            avg_size = sum(len(chunk.page_content) for chunk in chunks) / len(chunks)
            print(f"   Average chunk size: {int(avg_size)} characters")
        
        return chunks
        
    except Exception as e:
        print(f"❌ Error splitting documents: {e}")
        return []


# Example usage (if running this file directly)
if __name__ == "__main__":
    print("Testing document splitting...")
    print("\nExample:")
    print("   from ragBuilt2 import load_documents")
    print("   from ragBuilt3 import split_documents")
    doc = load_pdf()
    spliter = split_documents(doc, chunk_size=1000, chunk_overlap=200)
    for i, chunk in enumerate(spliter):
        if i == 10:
            break
        print(f"Chunk {i+1}: {chunk.page_content[:100]}...")
    print("\n✅ Document splitting module is ready!")
