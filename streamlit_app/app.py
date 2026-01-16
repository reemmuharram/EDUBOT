# streamlit_app/app.py
import streamlit as st
from pathlib import Path
import sys

# Add parent directory to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# RAG imports
from rag.embeddings import get_embeddings
from rag.vectorstore import load_vectorstore
from rag.retriever import get_retriever
from rag.prompts import prompt_temp
from rag.chain import create_chain
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline


# Set page config
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
        
        return retrieval_chain
    except Exception as e:
        print(f"Error loading models: {str(e)}")
        import traceback
        traceback.print_exc()
        st.error(f"Error loading models: {str(e)}")
        return None


# Load models
if not st.session_state.chain_loaded:
    with st.spinner("✨ Loading AI models... This may take a moment"):
        st.session_state.retrieval_chain = load_models()
        st.session_state.chain_loaded = True


# Beautiful header
st.markdown("# 📚 Student Assistant")
st.markdown('<p class="subtitle">Ask me anything and I\'ll help you learn!</p>', unsafe_allow_html=True)

# Display welcome message only if no messages yet
if len(st.session_state.messages) == 0:
    st.markdown("""
        <div class="welcome-card">
            <p>👋 Hi! I'm your AI study assistant. I can help you with questions about your courses, 
            concepts, assignments, and more. Just type your question below!</p>
        </div>
    """, unsafe_allow_html=True)

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Type your question here..."):
    # Add user message to chat
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get response from chain
    with st.chat_message("assistant"):
        try:
            with st.spinner("🤔 Thinking..."):
                if st.session_state.retrieval_chain is None:
                    raise ValueError("Chain not loaded properly")
                
                # LCEL chain expects just the question string
                answer = st.session_state.retrieval_chain.invoke(prompt)
                
                # LCEL returns string directly
                answer = str(answer)
            
            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
        except Exception as e:
            error_msg = f"❌ Oops! Something went wrong: {str(e)}"
            print(f"Query error: {e}")
            import traceback
            traceback.print_exc()
            st.error(error_msg)
