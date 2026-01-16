from langchain_core.documents import Document

def load_q_a_data(df):
    documents = []
    for _, row in df.iterrows():
        q = str(row["question"])
        a = str(row["answer"])
        content = f"Question: {q}\nAnswer: {a}"
        documents.append(Document(page_content=content))
    return documents

