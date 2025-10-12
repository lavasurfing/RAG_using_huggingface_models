from langchain_community.document_loaders import PyPDFLoader
# load your PDF
def load_pdf():
    # pdf_path = Path("/data/thebook.pdf")
    # print(f"Loading PDF from: {pdf_path}")
    loader = PyPDFLoader('./data/thebook.pdf')
    docs = loader.load()
    print(f"Loaded {len(docs)} documents.")

    # print Documents
    # for i, doc in enumerate(docs):
    #     if i == 6:
    #         break
    #     print(f"\nDocument {i+1}:\n")
    #     print(doc.page_content[:500])  # Print first 500 characters of each document
    return docs
