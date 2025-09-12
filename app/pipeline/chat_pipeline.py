from typing import List, Tuple
from app.rag.retriever import search
from app.llm import chat_once

def _build_context(query: str, k: int = 4) -> tuple[str, list[str]]:
    results = search(query, k=k)
    texts, cites = [], []
    for doc, _ in results:
        texts.append(doc.page_content)
        cites.append(doc.metadata.get("source"))
    # de-dupe cites while preserving order
    cites = list(dict.fromkeys(cites))
    return "\n\n---\n\n".join(texts), cites

def answer_with_rag(user_text: str, k: int = 2) -> tuple[str, List[str]]:
    ctx, cites = _build_context(user_text, k=k)
    system = (
    "You are a supportive, non-clinical companion. "
    "From the CONTEXT below, extract and present EXACTLY three safe, actionable techniques. "
    "Write them as a numbered list with one-sentence explanations.\n\n"
    f"CONTEXT:\n{ctx}"
)
    # system = (
    #     "You are a supportive, non-clinical companion. Be brief and kind. "
    #     "Use only the CONTEXT to suggest up to 3 safe, actionable techniques. "
    #     "Explain why each helps in one short line. If context is thin, say so.\n\n"
    #     f"CONTEXT:\n{ctx}"
    # )
    reply = chat_once(user_text, system=system)
    return reply, cites
