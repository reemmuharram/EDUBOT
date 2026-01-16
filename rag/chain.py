# chain.py
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

def create_chain(llm, retriever, prompt):
    """Create a retrieval QA chain using LCEL"""
    
    def format_docs(docs):
        return "\n\n".join([doc.page_content for doc in docs])
    
    # Build chain using LCEL (LangChain Expression Language)
    chain = (
        {
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
    )
    
    return chain