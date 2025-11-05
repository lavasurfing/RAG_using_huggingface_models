from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from loader import load_pdf
from uuid import uuid4
from embedding import get_embedding_model

def split_text_vec(doc, chunk_size=1000, chunk_overlap=200):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_documents(doc)
    return chunks

def adding_data_toVecStore()->None:
    thepdf = load_pdf()

    chunks = split_text_vec(thepdf)

    # creating unique ids for each chunk
    ids = [str(uuid4()) for _ in range(len(chunks))]



    # embedding setup
    embedding_model = get_embedding_model()


    # setting up vector store
    vector_store = Chroma(
        collection_name="thebook",
        embedding_function=embedding_model,
        persist_directory="./chroma_db"
    )



    # adding document to vector store
    vector_store.add_documents(documents=chunks, ids=ids)
    print('Document chunks added to vector store with unique IDs:')

# function to get retrival vector store
def get_retrieval_vector_store():
    embedding_model = get_embedding_model()
    vector_store = Chroma(
        collection_name="thebook",
        embedding_function=embedding_model,
        persist_directory="./chroma_db"
    )
    retrieval_vector_store = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 50})
    return retrieval_vector_store


def read_from_vector_store():
    embedding_model = get_embedding_model()
    vector_store = Chroma(
        collection_name="thebook",
        embedding_function=embedding_model,
        persist_directory="./chroma_db"
    )

    # res = vector_store.similarity_search(
    #     "Machine learning",
    #     k=3,
    #     filter={"source": "thebook.pdf"}
    #     )
    # print('Single Res obj', res)
    
    # for i, doc in enumerate(res):
        
    #     print(f"\nResult {i+1}:\n")
    #     print(doc.page_content[:500])  # Print first 500 characters of each document

    retrieval_vector_store = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    print(retrieval_vector_store.invoke('What is Machine Learning?'))


def sample_ReadFromPersistedDb():
    embedding_model = get_embedding_model()
    db = Chroma(
        collection_name="thebook",
        embedding_function=embedding_model,
        persist_directory="./chroma_db"
    )
    results = db.similarity_search("What is Machine Learning?", k=3)
    for i, doc in enumerate(results):
        print(f"\nResult {i+1}:\n")
        print(doc.page_content)
    









   


# DEBUGER: document chunking 
def run():
    ids = [str(f'{i}-{uuid4()}') for i in range(3)]
    print(f"Generated UUIDs: {ids}")

# retrieval_vector_store()

# run_uuid()

# run()

# sample_ReadFromPersistedDb()

read_from_vector_store()
