# app/memory.py
import os, uuid
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from chromadb import PersistentClient
import numpy as np

load_dotenv()

CHROMA_DIR = os.getenv("CHROMA_DIR", "./chroma_store")
SIM_THRESHOLD = float(os.getenv("DUPLICATE_SIMILARITY_THRESHOLD", 0.88))

# Embedding model (local, nhanh & nhẹ)
EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")

client = chromadb.Client(Settings(persist_directory=CHROMA_DIR))
collection = client.get_or_create_collection("memory_collection")

def embed_text(text):
    vec = EMBED_MODEL.encode(text, show_progress_bar=False)
    # normalize for cosine similarity
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec.tolist()

def is_duplicate(text, top_k=3, threshold=SIM_THRESHOLD):
    q_emb = embed_text(text)
    res = collection.query(
        query_embeddings=[q_emb],
        n_results=top_k,
        include=["embeddings", "metadatas", "documents"]  # Bỏ "ids"
    )
    if len(res["documents"]) == 0 or len(res["documents"][0]) == 0:
        return False, None, 0.0
    for idx, emb in enumerate(res["embeddings"][0]):
        emb_np = np.array(emb)
        q_np = np.array(q_emb)
        sim = float(np.dot(q_np, emb_np) / (np.linalg.norm(q_np)*np.linalg.norm(emb_np) + 1e-10))
        if sim >= threshold:
            return True, None, sim
    return False, None, 0.0
# def is_duplicate(text, top_k=3, threshold=SIM_THRESHOLD):
#     q_emb = embed_text(text)
#     res = collection.query(query_embeddings=[q_emb], n_results=top_k, include=["embeddings","ids","metadatas","documents"])
#     if len(res["ids"]) == 0 or len(res["ids"][0]) == 0:
#         return False, None, 0.0
#     # res["embeddings"][0] is list of embeddings returned
#     # compute cosine similarity manually (in case chroma distances differ)
#     for idx, emb in enumerate(res["embeddings"][0]):
#         emb_np = np.array(emb)
#         q_np = np.array(q_emb)
#         sim = float(np.dot(q_np, emb_np) / (np.linalg.norm(q_np)*np.linalg.norm(emb_np) + 1e-10))
#         if sim >= threshold:
#             return True, res["ids"][0][idx], sim
#     return False, None, 0.0

def add_memory(text, metadata=None):
    doc_id = str(uuid.uuid4())
    emb = embed_text(text)
    collection.add(documents=[text], metadatas=[metadata or {}], ids=[doc_id], embeddings=[emb])
    # client.persist()
    return doc_id

def query_memory(query, k=5):
    q_emb = embed_text(query)
    res = collection.query(query_embeddings=[q_emb], n_results=k, include=["documents","metadatas","distances"])
    # returns dict with lists, convert to convenient format
    docs = []
    if res and "documents" in res:
        for i, doc in enumerate(res["documents"][0]):
            docs.append({
                "document": doc,
                "metadata": res["metadatas"][0][i] if "metadatas" in res else {},
                "distance": res["distances"][0][i] if "distances" in res else None
            })
    return docs
