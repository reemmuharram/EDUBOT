# main.py
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pathlib import Path

# RAG imports
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

# Add CORS middleware to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize models and chain
print("Loading embeddings model...")
emb_model = get_embeddings()

print("Loading vector store...")
vector_store = load_vectorstore("vectorstore", emb_model)
retriever = get_retriever(vector_store, k=3)

print("Loading prompt template...")
prompt = prompt_temp()

print("Loading Flan-T5 model...")
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base", cache_dir="./model")
pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer, max_new_tokens=100)
llm = HuggingFacePipeline(pipeline=pipe)

print("Creating retrieval chain...")
retrieval_chain = create_chain(llm, retriever, prompt)

print("✅ All models loaded successfully!")


@app.get("/", response_class=HTMLResponse)
def serve_html():
    """Serve the chatbot UI from templates folder"""
    template_path = Path("templates/index.html")
    if template_path.exists():
        return template_path.read_text()
    else:
        return HTMLResponse(
            content="<h1>Error: templates/index.html not found</h1>", 
            status_code=404
        )


@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {"status": "ok", "message": "RAG Chatbot API is running"}


@app.post("/query")
def query(request: QueryRequest):
    """Process query through RAG chain"""
    try:
        if not request.query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")
        
        # LCEL chain expects just the question string
        result = retrieval_chain.invoke(request.query)
        
        # LCEL chain returns a string directly
        return {"answer": result}
    except Exception as e:
        print(f"Error processing query: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)