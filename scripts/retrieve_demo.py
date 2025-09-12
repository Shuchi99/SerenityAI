import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from app.rag.retriever import search

def _pretty(results):
    for i, (doc, score) in enumerate(results, 1):
        src = doc.metadata.get("source")
        path = doc.metadata.get("path")
        preview = doc.page_content[:300].replace("\\n", " ")
        if len(doc.page_content) > 300:
            preview += "..."
        print(f"[{i}] score={score:.4f}  source={src}  path={path}")
        print(f"    {preview}")

if __name__ == "__main__":
    query = "How can I calm anxiety quickly?"
    print(f"Query: {query}")
    results = search(query, k=4)
    _pretty(results)
