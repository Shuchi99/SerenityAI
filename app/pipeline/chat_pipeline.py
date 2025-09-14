# app/pipeline/chat_pipeline.py
from typing import List, Tuple
from app.rag.retriever import search
from app.llm import chat_once
from app.services.emotion.model import classify_emotion  # <- make sure Step 5 model is in place

def _build_context(query: str, k: int = 4) -> tuple[str, list[str]]:
    results = search(query, k=k)
    texts, cites = [], []
    for doc, _ in results:
        texts.append(doc.page_content)
        cites.append(doc.metadata.get("source"))
    cites = list(dict.fromkeys(cites))  # de-dupe
    return "\n\n---\n\n".join(texts), cites

def answer_with_rag(user_text: str, k: int = 4) -> tuple[str, List[str], str]:
    # Build retrieval context
    ctx, cites = _build_context(user_text, k=k)

    # Classify emotion
    emotion, conf, _ = classify_emotion(user_text)

    # Condition the system prompt with emotion + context
    system = (
        f"You are a supportive, non-clinical companion. "
        f"The user seems to feel {emotion} (confidence {conf:.2f}). Acknowledge their feeling gently. "
        f"From the CONTEXT, suggest up to 3 safe, actionable techniques. "
        f"Explain why each helps in one short line. If context is thin, say so.\n\n"
        f"CONTEXT:\n{ctx}"
    )

    # Generate
    reply = chat_once(user_text, system=system)
    return reply, cites, emotion
