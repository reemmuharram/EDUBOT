# 📚 Student Assistant Chatbot

 Intelligent RAG chatbot designed to help students with their questions. Built with FastAPI, Streamlit, and powered by Google's Flan-T5 model.

![Python](https://img.shields.io/badge/python-3.11+-blue.svg)

## 🚀 Streamlit Demo
[You can try it here 🤖](https://edubot-ebc2frxtf7jw2su44si2ks.streamlit.app/)



## ✨ Features

- 🤖 **AI-Powered Responses**: Uses Google Flan-T5 for natural language understanding
- 🔍 **Smart Retrieval**: FAISS vector search finds the most relevant information
- 📊 **22,000+ Q&A Pairs**: Comprehensive educational dataset
- ⚡ **Fast & Efficient**: Optimized for CPU inference
---

## 🎬 Quick Start

### Prerequisites

- Python 3.11 or higher
- 4GB+ RAM recommended

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/chatbot-assistant.git
   cd chatbot-assistant
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv rag_env
   
   # On Windows:
   rag_env\Scripts\activate
   
   # On Mac/Linux:
   source rag_env/bin/activate
   ```

3. **Install PyTorch (CPU version)**
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
   ```

4. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

5. **Build the vectorstore** (first time only)
   ```bash
   python rebuild_vectorstore.py
   ```
   *This will take 5-10 minutes. It processes your dataset and creates searchable embeddings.*

---

## 🚀 Running the Application

### Option 1: Streamlit Interface

```bash
streamlit run streamlit_app/app.py
```

Open your browser at: **http://localhost:8501**

### Option 2: FastAPI Web Interface

```bash
python main.py
```

Open your browser at: **http://localhost:8000**

---

## 📁 Project Structure

```
chatbot-assistant/
│
├── dataset/
│   └── full_dataset.csv          # Your Q&A training data (22,571 pairs)
│
├── model/                         # Downloaded Flan-T5 model (auto-created)
│
├── vectorstore/                   # FAISS vector database
│   ├── index.faiss               # Vector indices
│   └── index.pkl                 # Metadata
│
├── rag/                          # RAG pipeline modules
│   ├── __init__.py
│   ├── embeddings.py             # Sentence transformers setup
│   ├── vectorstore.py            # FAISS operations
│   ├── retriever.py              # Search logic
│   ├── prompts.py                # Prompt templates
│   ├── chain.py                  # LangChain integration
│   ├── loader.py                 # Data loading utilities
│   └── splitter.py               # Text chunking
│
├── streamlit_app/
│   └── app.py                    # Streamlit chat interface
│
├── templates/
│   └── index.html                # FastAPI web UI
│
├── main.py                       # FastAPI application
├── rebuild_vectorstore.py        # Vectorstore builder script
├── requirements.txt              # Python dependencies
└── README.md                     # You are here! 👋
```

---

## 🛠️ How It Works

### The RAG Pipeline

1. **User Question** → You ask: "How do I apply for admission?"

2. **Embedding** → Your question is converted to a vector (numerical representation)

3. **Retrieval** → FAISS searches the vectorstore for the 3 most similar Q&A pairs

4. **Context Building** → Retrieved information is formatted as context

5. **Generation** → Flan-T5 generates a friendly, natural answer based on the context

6. **Response** → You get: "You can apply for admission by filling out the application form."

### Tech Stack

- **LangChain**: Orchestrates the RAG pipeline
- **FAISS**: Lightning-fast vector similarity search
- **Sentence Transformers**: Creates semantic embeddings
- **Google Flan-T5**: Generates natural language responses
- **FastAPI**: Modern, fast web framework
- **Streamlit**: Interactive data apps

---
### Update the Dataset

1. Replace `dataset/full_dataset.csv` with your data
2. Ensure columns are named: `question` and `answer` (or `input` and `target`)
3. Run: `python rebuild_vectorstore.py`

### Adjust AI Parameters

In `rag/chain.py`, modify the LLM settings:

```python
pipe = pipeline(
    "text2text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=150,  # Longer responses
    temperature=0.8,     # More creative (0.0 = deterministic, 1.0 = creative)
)
```

---

## 🐛 Troubleshooting

### "Module not found" errors
```bash
pip install -r requirements.txt
```

### "Vectorstore not found"
```bash
python rebuild_vectorstore.py
```
### Slow model downloads
Models download on first run (~900MB). Be patient or use faster internet.

### Out of memory
Reduce `k=3` to `k=2` in `rag/retriever.py` for less context

---

## 📊 Performance

- **Response Time**: 2-5 seconds per query
- **Memory Usage**: ~2GB RAM
- **Dataset Size**: 22,571 Q&A pairs
- **Model Size**: ~900MB (Flan-T5-base)
- **Embedding Model**: ~90MB (all-MiniLM-L6-v2)
## ⭐ Star History

If you find this project helpful, please consider giving it a star! ⭐

<div align="center">
  Made with ☕ 
</div>
