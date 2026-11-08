import os
from pathlib import Path
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import chromadb

# 1. Setup Paths and DB (relative to this file, so the backend works from any cwd)
PDF_PATH = str(Path(__file__).parent / "data" / "rulebook.pdf")
DB_PATH = str(Path(__file__).parent / "data" / "chroma_db")

# Initialize ChromaDB (Local Vector Database)
client = chromadb.PersistentClient(path=DB_PATH)
# Use a very fast, lightweight embedding model locally
embedding_model = SentenceTransformer('all-MiniLM-L6-v2', device='cuda') 

def get_embedding(text):
    return embedding_model.encode(text).tolist()

def chunk_text(text, chunk_size=500, overlap=50):
    """Splits text into chunks with a bit of overlap to preserve context."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

def process_rulebook():
    if not os.path.exists(PDF_PATH):
        print(f"Error: Could not find {PDF_PATH}. Please add the PDF to the data folder.")
        return

    print("Reading PDF...")
    reader = PdfReader(PDF_PATH)
    collection_name = "primal_rules"
    
    # Delete existing collection if we are re-running
    try:
        client.delete_collection(name=collection_name)
    except:
        pass
    
    collection = client.create_collection(name=collection_name)

    documents = []
    metadatas = []
    ids = []

    print(f"Found {len(reader.pages)} pages. Processing...")
    
    for page_num, page in enumerate(reader.pages):
        text = page.extract_text()
        if not text:
            continue
            
        chunks = chunk_text(text)
        for i, chunk in enumerate(chunks):
            documents.append(chunk)
            metadatas.append({"source": "Rulebook", "page": page_num + 1})
            ids.append(f"page_{page_num + 1}_chunk_{i}")

    # Embed and add to database in batches
    print("Embedding text (this may take a moment)...")
    batch_size = 100
    for i in range(0, len(documents), batch_size):
        batch_docs = documents[i:i+batch_size]
        batch_metas = metadatas[i:i+batch_size]
        batch_ids = ids[i:i+batch_size]
        
        # Generate embeddings
        embeddings = [get_embedding(doc) for doc in batch_docs]
        
        collection.add(
            documents=batch_docs,
            metadatas=batch_metas,
            ids=batch_ids,
            embeddings=embeddings
        )
        print(f"Processed {min(i+batch_size, len(documents))}/{len(documents)} chunks")

    print("Rulebook successfully ingested into ChromaDB!")

if __name__ == "__main__":
    process_rulebook()