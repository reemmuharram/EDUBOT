#RETRIEVER CHAIN
from langchain.chains import RetrievalQA

def create_chain(llm, retriever, prompt):
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt}
)
