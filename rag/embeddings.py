from langchain_community.embeddings import HuggingFaceEmbeddings
import os

def get_embeddings():
    # Set longer timeout for downloading models
    os.environ['HF_HUB_DOWNLOAD_TIMEOUT'] = '300'  # 5 minutes
    
    # Option 1: Standard model (90MB) - best quality
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    
    # Option 2: Smaller model (30MB) - faster download, slightly lower quality
    # model_name = "sentence-transformers/paraphrase-MiniLM-L3-v2"
    
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )