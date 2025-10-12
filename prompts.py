from embedding import get_embedding_model
from splitChunk import get_retrieval_vector_store
from transformers import pipeline   

def get_generative_chain() -> pipeline:


    generator = pipeline(
        task="question-answering", 
        model="deepset/roberta-base-squad2",
        device=0,
        top_k=5,
        top_p=0.95,

    )

    return generator


def rag_answer_prompt(question: str):
    retriever = get_retrieval_vector_store()
    docs = retriever.get_relevant_documents(question)

    context = "\n\n".join([d.page_content for d in docs])

    prompt = '''
    you are a helpful AI assistant. Use the following context to answer the question.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    Context: {context}


    Question: {question}

    Answer:
    '''

    generator = get_generative_chain()
    response = generator.invoke({
        "question": question,
        "context": context
    })

    return response

res = rag_answer_prompt("What the easiest way to learn Machine Learning?")

print(res)