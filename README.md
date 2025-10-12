# RAG Chatbot with Hugging Face Embeddings

A powerful Retrieval-Augmented Generation (RAG) chatbot built using Hugging Face embeddings and LangChain. This implementation allows you to create a conversational AI that can answer questions based on your own documents.

## 🚀 Features

- **Hugging Face Embeddings**: Uses state-of-the-art sentence transformers for semantic search
- **Document Processing**: Supports both text and PDF files
- **Vector Storage**: FAISS-based vector store for efficient similarity search
- **Multiple Interfaces**: Command-line and Streamlit web interface
- **Flexible Models**: Support for various embedding and language models
- **Source Attribution**: Shows which documents were used to generate answers

## 📋 Requirements

- Python 3.8+
- CUDA-compatible GPU (optional, for faster processing)
- At least 4GB RAM (8GB+ recommended)

## 🛠️ Installation

1. **Clone or download the repository**
2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## 🎯 Quick Start

### Option 1: Command Line Interface

Run the basic RAG chatbot with a sample document:

```bash
python ragBuilt.py
```

This will:
- Initialize the embedding and language models
- Create a sample document about AI/ML
- Start an interactive chat session

### Option 2: Streamlit Web Interface

For a more user-friendly experience:

```bash
streamlit run rag_streamlit.py
```

Then open your browser to `http://localhost:8501`

## 📖 Usage Guide

### Using the Command Line Version

1. **Run the script**:
   ```bash
   python ragBuilt.py
   ```

2. **Follow the prompts**:
   - The system will initialize models (this may take a few minutes on first run)
   - A sample document will be created and processed
   - You can start asking questions

3. **Example questions**:
   - "What is artificial intelligence?"
   - "How does machine learning work?"
   - "What is natural language processing?"

### Using the Streamlit Web Interface

1. **Start the web app**:
   ```bash
   streamlit run rag_streamlit.py
   ```

2. **Configure models** (in the sidebar):
   - Choose your embedding model
   - Select a language model
   - Adjust chunk size and overlap parameters

3. **Upload documents**:
   - Click "Choose files" and select your text or PDF files
   - Click "Process Documents" to create the vector store

4. **Start chatting**:
   - Type your questions in the chat interface
   - View source documents used for each answer

## 🔧 Configuration

### Available Embedding Models

- `sentence-transformers/all-MiniLM-L6-v2` (default, fast and efficient)
- `sentence-transformers/all-mpnet-base-v2` (higher quality, slower)
- `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (multilingual support)

### Available Language Models

- `microsoft/DialoGPT-medium` (default, conversational)
- `gpt2` (general purpose)
- `distilgpt2` (faster, smaller)

### Customizing the RAG System

You can modify the `RAGChatbot` class parameters:

```python
chatbot = RAGChatbot(
    embedding_model="sentence-transformers/all-mpnet-base-v2",  # Better quality
    llm_model="gpt2",                                          # Different LLM
    chunk_size=1500,                                           # Larger chunks
    chunk_overlap=300                                          # More overlap
)
```

## 📁 File Structure

```
├── ragBuilt.py          # Main command-line RAG chatbot
├── rag_streamlit.py     # Streamlit web interface
├── requirements.txt     # Python dependencies
├── README.md           # This file
└── sample_document.txt # Auto-generated sample (temporary)
```

## 🔍 How It Works

1. **Document Loading**: Text and PDF files are loaded and processed
2. **Text Splitting**: Documents are split into overlapping chunks for better retrieval
3. **Embedding Creation**: Each chunk is converted to a vector using Hugging Face embeddings
4. **Vector Storage**: Embeddings are stored in a FAISS vector database
5. **Query Processing**: User questions are embedded and used to find similar document chunks
6. **Answer Generation**: Relevant chunks are passed to the language model to generate answers

## 🚨 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**:
   - The system will automatically fall back to CPU
   - Reduce chunk size or use smaller models

2. **Model Download Issues**:
   - Ensure you have internet connection for first-time model downloads
   - Models are cached locally after first download

3. **Slow Performance**:
   - Use smaller embedding models (e.g., `all-MiniLM-L6-v2`)
   - Reduce chunk size and overlap
   - Use CPU-optimized models if no GPU available

### Performance Tips

- **GPU Usage**: Install PyTorch with CUDA support for faster processing
- **Model Selection**: Balance between quality and speed based on your needs
- **Chunk Size**: Larger chunks provide more context but slower processing
- **Document Size**: Very large documents may need to be split into smaller files

## 🔒 Security Notes

- The system processes documents locally
- No data is sent to external services (except for model downloads)
- Vector stores can be saved locally for reuse

## 🤝 Contributing

Feel free to submit issues, feature requests, or pull requests to improve this RAG chatbot implementation.

## 📄 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- [Hugging Face](https://huggingface.co/) for the transformer models
- [LangChain](https://langchain.com/) for the RAG framework
- [Streamlit](https://streamlit.io/) for the web interface
- [FAISS](https://github.com/facebookresearch/faiss) for vector similarity search


# AI Model
 - Summerization : facebook/bart-large-cnn
 - Ques/Ans : deepset/roberta-base-squad2
 - Embeddings : sentence-transformers/all-MiniLM-L6-v2

### RAG Chatbot built

## Files
### loader.py : loading pdf
### embedding_model.py : Return Hugginface embedding model
### splitChunkVec.py : splitting chunking and setting up vector store
### llm_prompts.py
### prompts.py