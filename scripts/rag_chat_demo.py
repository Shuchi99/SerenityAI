import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from app.llm import chat_once
from app.rag.retriever import search

def build_context(query: str, k: int = 4):
    results = search(query, k=k)
    context = []
    cites = []
    for doc, _ in results:
        context.append(doc.page_content)
        cites.append(doc.metadata.get("source"))
    return "\n\n---\n\n".join(context), cites

if __name__ == "__main__":
    user = "I feel anxious and can't sleep. What can I try right now?"
    ctx, cites = build_context(user, k=4)
    system = (
        "You are a supportive, non-clinical companion. Be brief and kind. "
        "Use only the CONTEXT to suggest 2–3 actionable, safe techniques. "
        "Explain why each helps in one line. If context is thin, say so.\n\n"
        f"CONTEXT:\n{ctx}"
    )
    print("User:", user)
    print("----")
    out = chat_once(user, system=system)
    print(out)
    print("\nSources:", ", ".join(dict.fromkeys(cites)))
