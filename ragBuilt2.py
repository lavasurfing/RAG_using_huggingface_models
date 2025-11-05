"""
ragBuilt2.py - Simple PDF loader (based on loader.py)

This script provides a minimal PDF loading helper using LangChain's PyPDFLoader.
"""

from langchain_community.document_loaders import PyPDFLoader


def load_pdf(pdf_path: str = "./data/thebook.pdf"):
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    print(f"Loaded {len(docs)} documents.")
    return docs


if __name__ == "__main__":
    pdf = load_pdf()
    print(pdf[0].page_content)
