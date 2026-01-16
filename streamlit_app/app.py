import streamlit as st
from pathlib import Path
import sys

# Add parent directory to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from rag.embeddings import get_embeddings
from rag.vectorstore import load_vectorstore, save_vectorstore
from rag.retriever import get_retriever
from rag.prompts import prompt_temp
from rag.chain import create_chain
from rag.loader import load_q_a_data
from rag.splitter import split_docs
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import pandas as pd

st.set_page_config(
    page_title="RAG Student Chatbot",
    page_icon="📚",
    layout="centered",
    initial_sidebar_state="collapsed"
)

if "messages" not in st.session_state:
    st.session_state.messages = []

if "chain_loaded" not in st.session_state:
    st.session_state.chain_loaded = False
    st.session_state.retrieval_chain = None


@st.cache_resource
def load_models():
    """Load all models once and cache them"""
    try:
        print("Loading embeddings model...")
        emb_model = get_embeddings()

        print("Checking for vectorstore...")
        vectorstore_path = Path("vectorstore")
        
        # Check if vectorstore exists
        if not vectorstore_path.exists() or not (vectorstore_path / "index.faiss").exists():
            st.info("🔨 Building vectorstore for first time... This will take 5-10 minutes.")
            print("Building vectorstore from dataset...")
            
            # Load dataset
            csv_path = Path("dataset/full_dataset.csv")
            if not csv_path.exists():
                raise FileNotFoundError(f"Dataset not found at {csv_path}")
            
            df = pd.read_csv(csv_path)
            
            # Rename columns if needed
            if 'input' in df.columns and 'target' in df.columns:
                df = df.rename(columns={'input': 'question', 'target': 'answer'})
            
            # Create documents
            print(f"Creating documents from {len(df)} rows...")
            documents = load_q_a_data(df)
            
            # Split documents
            print("Splitting documents...")
            split_documents = split_docs(documents, chunk_size=500, chunk_overlap=50)
            print(f"Created {len(split_documents)} chunks")
            
            # Build and save vectorstore
            print("Building vectorstore (this takes time)...")
            save_vectorstore(split_documents, emb_model, "vectorstore")
            st.success("✅ Vectorstore built successfully!")
            print("Vectorstore saved!")
        
        print("Loading vector store...")
        vector_store = load_vectorstore("vectorstore", emb_model)
        retriever = get_retriever(vector_store, k=3)

        print("Loading prompt template...")
        prompt = prompt_temp()

        print("Loading Flan-T5 model...")
        tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
        model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base", cache_dir="./model")
        pipe = pipeline(
            "text2text-generation", 
            model=model, 
            tokenizer=tokenizer, 
            max_new_tokens=100
        )
        llm = HuggingFacePipeline(pipeline=pipe)

        print("Creating retrieval chain...")
        retrieval_chain = create_chain(llm, retriever, prompt)
        
        if retrieval_chain is None:
            raise ValueError("Chain creation returned None")
        
        print("✅ All models loaded successfully!")
        return retrieval_chain
        
    except Exception as e:
        print(f"Error loading models: {str(e)}")
        import traceback
        traceback.print_exc()
        st.error(f"❌ Error loading models: {str(e)}")
        return None


if not st.session_state.chain_loaded:
    with st.spinner("✨ Loading AI models... This may take 10-15 minutes on first run"):
        st.session_state.retrieval_chain = load_models()
        if st.session_state.retrieval_chain is not None:
            st.session_state.chain_loaded = True
        else:
            st.error("Failed to load models. Please refresh the page.")
            st.stop()


st.markdown("# 📚 Student Assistant")
st.markdown('<p class="subtitle">Ask me anything and I\'ll help you learn!</p>', unsafe_allow_html=True)

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Type your question here..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        try:
            with st.spinner("🤔 Thinking..."):
                if st.session_state.retrieval_chain is None:
                    raise ValueError("Chain not loaded properly")
                answer = st.session_state.retrieval_chain.invoke(prompt)
                answer = str(answer)
            
            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
        except Exception as e:
            error_msg = f"❌ Oops! Something went wrong: {str(e)}"
            print(f"Query error: {e}")
            import traceback
            traceback.print_exc()
            st.error(error_msg)
