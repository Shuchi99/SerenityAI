from pathlib import Path
from typing import List, Tuple
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from ..config import settings

def _load_vectorstore() -> FAISS:
    emb = HuggingFaceEmbeddings(model_name=settings.embed_model)
    index_path = Path(settings.index_dir)
    if not index_path.exists():
        raise FileNotFoundError(f"Vector index not found at '{index_path}'. Run ingestion first.")
    return FAISS.load_local(str(index_path), emb, allow_dangerous_deserialization=True)

def search(query: str, k: int = 4) -> List[Tuple[Document, float]]:
    vs = _load_vectorstore()
    return vs.similarity_search_with_score(query, k=k)
