from langchain_community.vectorstores import FAISS

def load_vectorstore(path, embeddings):
    return FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)

def save_vectorstore(docs, embeddings, path):
    vs = FAISS.from_documents(docs, embeddings)
    vs.save_local(path)
    return vs