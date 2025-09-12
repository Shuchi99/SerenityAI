from fastapi import FastAPI
from pydantic import BaseModel, Field
from app.pipeline.chat_pipeline import answer_with_rag

app = FastAPI(title="MH Companion API")

class ChatRequest(BaseModel):
    message: str
    k: int = Field(default=4, ge=1, le=8)

class ChatResponse(BaseModel):
    reply: str
    sources: list[str]

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    reply, sources = answer_with_rag(req.message, k=req.k)
    return ChatResponse(reply=reply, sources=sources)
