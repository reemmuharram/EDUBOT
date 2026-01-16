"""
Script to rebuild the vectorstore with current environment
Run this once to fix the Pydantic compatibility issue
"""
import pandas as pd
from pathlib import Path
from rag.embeddings import get_embeddings
from rag.loader import load_q_a_data
from rag.splitter import split_docs
from rag.vectorstore import save_vectorstore

def rebuild_vectorstore():
    """Rebuild vectorstore from CSV data"""
    
    print("🔄 Rebuilding vectorstore...")
    
    # 1. Load your dataset
    print("📁 Loading dataset...")
    csv_path = Path("dataset/full_dataset.csv")
    
    if not csv_path.exists():
        print(f"❌ Error: {csv_path} not found!")
        return
    
    df = pd.read_csv(csv_path)
    print(f"✅ Loaded {len(df)} rows from dataset")
    
    # Rename columns if needed (adjust based on your CSV structure)
    # If your CSV has 'input' and 'target' columns:
    if 'input' in df.columns and 'target' in df.columns:
        df = df.rename(columns={'input': 'question', 'target': 'answer'})
    
    # 2. Create documents
    print("📄 Creating documents...")
    documents = load_q_a_data(df)
    print(f"✅ Created {len(documents)} documents")
    
    # 3. Split documents
    print("✂️  Splitting documents...")
    split_documents = split_docs(documents, chunk_size=500, chunk_overlap=50)
    print(f"✅ Split into {len(split_documents)} chunks")
    
    # 4. Get embeddings model
    print("🔢 Loading embeddings model...")
    embeddings = get_embeddings()
    print("✅ Embeddings model loaded")
    
    # 5. Create and save vectorstore
    print("💾 Creating vectorstore (this may take a few minutes)...")
    vectorstore_path = "vectorstore"
    save_vectorstore(split_documents, embeddings, vectorstore_path)
    print(f"✅ Vectorstore saved to {vectorstore_path}/")
    
    print("\n🎉 Done! You can now run your app:")
    print("   streamlit run streamlit_app/app.py")
    print("   OR")
    print("   python main.py")


if __name__ == "__main__":
    try:
        rebuild_vectorstore()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()