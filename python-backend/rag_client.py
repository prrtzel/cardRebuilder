# rag_client.py
from pathlib import Path
import chromadb
from sentence_transformers import SentenceTransformer

DB_PATH = str(Path(__file__).parent / "data" / "chroma_db")
_collection = None
_embedding_model = None


def _init():
    """Lazily initialize ChromaDB + embedding model. Returns False (never raises)
    when the DB is missing/empty or initialization fails."""
    global _collection, _embedding_model
    if _collection is not None:
        return True
    db_path = Path(DB_PATH)
    if not db_path.exists() or not any(p.is_file() for p in db_path.iterdir()):
        print(f"RAG Error: ChromaDB not found or empty at {DB_PATH}. AI context will be unavailable.")
        return False
    try:
        client = chromadb.PersistentClient(path=DB_PATH)
        _embedding_model = SentenceTransformer('all-MiniLM-L6-v2', device='cuda')
        _collection = client.get_collection(name="primal_rules")
        return True
    except Exception as e:
        print(f"RAG Error: {e}")
        _collection = None
        _embedding_model = None
        return False


def query_rules(query_text, n_results=3):
    """Queries the rulebook for relevant text.

    Never raises. Returns (context_str, chunks_list); on failure returns ("", []).
    """
    if not _init():
        return "", []

    try:
        if _embedding_model is None or _collection is None:
            return "", []
        query_embedding = _embedding_model.encode(query_text).tolist()
        results = _collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )

        chunks = []
        context = ""
        docs = results.get('documents', [[]])[0] or []
        metas = results.get('metadatas', [[]])[0] or []
        for doc, meta in zip(docs, metas):
            page = meta.get('page') if meta else None
            chunks.append({
                "page": int(page) if page is not None else -1,
                "text": doc,
            })
            context += f"[Page {page}]: {doc}\n\n"
        return context, chunks
    except Exception as e:
        print(f"RAG Error: {e}")
        return "", []
