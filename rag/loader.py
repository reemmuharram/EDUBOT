from langchain_core.documents import Document

def load_q_a_data(df):
    documents = []
    for q, a in zip(df["input"], df["target"]):
        content = f"Question: {q}. Answer: {a}."
        documents.append(Document(page_content=content))
    return documents
