# RAG Chatbot - Modular Structure Guide

## Overview

The RAG chatbot has been broken down into **10 modular files** (ragBuilt0.py through ragBuilt9.py), each handling a specific function. This makes the code easier to understand, maintain, and modify.

---

## File Structure

### **ragBuilt0.py** - Embeddings Initialization
**What it does:** Converts text to numbers (vectors)
- Function: `create_embeddings()`
- Loads Hugging Face embedding models
- Converts words/sentences into numerical representations
- Handles GPU/CPU automatically

**Simple explanation:** Like a translator that converts human words into computer-understandable numbers.

---

### **ragBuilt1.py** - Language Model (LLM) Initialization
**What it does:** Sets up the AI brain that generates answers
- Function: `create_llm()`
- Loads pre-trained language models
- Handles text generation
- Can work without this (retrieval-only mode)

**Simple explanation:** The "smart assistant" that reads context and writes answers.

---

### **ragBuilt2.py** - Document Loading
**What it does:** Reads files from your computer
- Function: `load_documents()`
- Handles .txt and .pdf files
- Can load single files or entire folders
- Returns document objects

**Simple explanation:** A file reader that gets your documents into the program.

---

### **ragBuilt3.py** - Document Splitting
**What it does:** Breaks long documents into smaller chunks
- Function: `split_documents()`
- Splits documents intelligently (at paragraphs, sentences)
- Adds overlap between chunks
- Makes documents easier to search

**Simple explanation:** Like cutting a long article into paragraphs for easier reading.

---

### **ragBuilt4.py** - Vector Store Creation
**What it does:** Creates a searchable database of your documents
- Functions: `create_vectorstore()`, `load_vectorstore()`
- Converts document chunks to vectors
- Stores them in a fast-search database (FAISS)
- Can save/load to avoid rebuilding

**Simple explanation:** Like creating a Google search index for your documents.

---

### **ragBuilt5.py** - Retrieval QA Setup
**What it does:** Connects everything together for Q&A
- Function: `setup_retrieval_qa()`
- Links vector store with language model
- Creates prompt templates
- Sets up the question-answering pipeline

**Simple explanation:** The glue that connects document search with answer generation.

---

### **ragBuilt6.py** - Query Functionality
**What it does:** Handles asking questions and getting answers
- Function: `query_rag_system()`
- Takes a question as input
- Returns answer + source documents
- Handles errors gracefully

**Simple explanation:** The "ask a question" function - you give it a question, it gives you an answer.

---

### **ragBuilt7.py** - Main RAGChatbot Class
**What it does:** Brings all modules together in one class
- Class: `RAGChatbot`
- Combines all individual modules
- Provides easy-to-use methods
- Main interface for using the system

**Simple explanation:** The main control center that uses all the other modules.

---

### **ragBuilt8.py** - Streamlit Web Interface
**What it does:** Creates a beautiful web interface
- Full web application
- File upload interface
- Chat interface
- Configuration options

**How to run:** `streamlit run ragBuilt8.py`

**Simple explanation:** A website version of the chatbot - no command line needed!

---

### **ragBuilt9.py** - Command Line Interface
**What it does:** Command-line entry point
- Main function with example
- Interactive chat session
- Demonstrates full workflow
- Creates sample documents

**How to run:** `python ragBuilt9.py`

**Simple explanation:** The command-line version you run from terminal.

---

### **ragBuilt.py** - Backward Compatibility
**What it does:** Imports all modules for backward compatibility
- Kept for existing code that imports from ragBuilt.py
- Imports from modular files
- Redirects to ragBuilt9.py

**Simple explanation:** For old code that still uses ragBuilt.py - redirects to new modules.

---

## Usage Examples

### Basic Usage (Using the Class)
```python
from ragBuilt7 import RAGChatbot

# Create chatbot
chatbot = RAGChatbot()

# Load documents
docs = chatbot.load_documents("my_file.txt")
chunks = chatbot.split_documents(docs)

# Create vector store
chatbot.create_vectorstore(chunks)

# Set up Q&A
chatbot.setup_retrieval_qa(k=3)

# Ask questions
result = chatbot.query("What is AI?")
print(result['answer'])

# Or start interactive chat
chatbot.chat()
```

### Using Individual Modules
```python
from ragBuilt0 import create_embeddings
from ragBuilt2 import load_documents
from ragBuilt3 import split_documents
from ragBuilt4 import create_vectorstore

# Use each module independently
embeddings = create_embeddings()
docs = load_documents("file.txt")
chunks = split_documents(docs)
vectorstore = create_vectorstore(chunks, embeddings)
```

---

## File Dependencies

```
ragBuilt9.py (CLI)
    └── uses ragBuilt7.py
            ├── uses ragBuilt0.py (embeddings)
            ├── uses ragBuilt1.py (LLM)
            ├── uses ragBuilt2.py (load docs)
            ├── uses ragBuilt3.py (split docs)
            ├── uses ragBuilt4.py (vector store)
            ├── uses ragBuilt5.py (QA setup)
            └── uses ragBuilt6.py (query)

ragBuilt8.py (Web UI)
    ├── uses ragBuilt0.py
    ├── uses ragBuilt1.py
    ├── uses ragBuilt2.py
    ├── uses ragBuilt3.py
    ├── uses ragBuilt4.py
    ├── uses ragBuilt5.py
    └── uses ragBuilt6.py
```

---

## Benefits of Modular Structure

1. **Easy to Understand:** Each file does one thing clearly
2. **Easy to Modify:** Change one module without affecting others
3. **Easy to Test:** Test each component independently
4. **Easy to Reuse:** Use individual modules in other projects
5. **Better Documentation:** Each file explains its own purpose

---

## Quick Start

**Command Line:**
```bash
python ragBuilt9.py
```

**Web Interface:**
```bash
streamlit run ragBuilt8.py
```

**Programmatic Use:**
```python
from ragBuilt7 import RAGChatbot
chatbot = RAGChatbot()
# ... use chatbot methods
```

---

## Notes

- All files have extensive comments in simple, layman terms
- Each file can be run independently for testing
- The modular structure makes it easy to customize for your needs
- Backward compatibility maintained with ragBuilt.py
