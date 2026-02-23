# main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

# RAG imports
from rag.loader import load_q_a_data
from rag.splitter import split_docs
from rag.embeddings import get_embeddings
from rag.vectorstore import load_vectorstore
from rag.retriever import get_retriever
from rag.prompts import prompt_temp
from rag.chain import create_chain
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline


class QueryRequest(BaseModel):
    query: str

app = FastAPI(title="RAG Chatbot API")
emb_model = get_embeddings()

# vectorstore
vector_store = load_vectorstore("vectorstore", emb_model)
retriever = get_retriever(vector_store, k=3)

# prompt template
prompt = prompt_temp()
# LLM
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer, max_new_tokens=100)
llm = HuggingFacePipeline(pipeline=pipe)

# chain
retrieval_chain = create_chain(llm, retriever, prompt)


@app.get("/")
def root():
    return {"message": "RAG Chatbot API is alive!"}

@app.post("/query")
def query(request: QueryRequest):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    try:
        result = retrieval_chain.invoke({"query": request.query})
        return {"answer": result["result"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
