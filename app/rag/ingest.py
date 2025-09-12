from pathlib import Path
from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from ..config import settings

def load_files(kb_dir: str) -> List[Document]:
    docs: List[Document] = []
    for p in Path(kb_dir).rglob("*"):
        if p.is_file() and p.suffix.lower() in {".txt", ".md"}:
            text = p.read_text(encoding="utf-8", errors="ignore")
            docs.append(Document(page_content=text, metadata={"source": p.name, "path": str(p)}))
    return docs

def build_index(docs: List[Document], index_dir: str) -> int:
    if not docs:
        raise ValueError("No documents found in knowledge base.")
    splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=80)
    chunks = splitter.split_documents(docs)
    emb = HuggingFaceEmbeddings(model_name=settings.embed_model)
    vs = FAISS.from_documents(chunks, emb)
    Path(index_dir).mkdir(parents=True, exist_ok=True)
    vs.save_local(index_dir)
    return len(chunks)

def main():
    kb_dir = settings.kb_dir
    index_dir = settings.index_dir
    docs = load_files(kb_dir)
    n_chunks = build_index(docs, index_dir)
    print(f"Ingested {len(docs)} docs into {n_chunks} chunks.")
    print(f"Index saved to: {index_dir}")

if __name__ == "__main__":
    main()
