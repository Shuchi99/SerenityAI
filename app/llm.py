from typing import Optional
from langchain_core.messages import SystemMessage, HumanMessage
from .config import settings

SYSTEM_PROMPT = (
    "You are a supportive, non-clinical chat companion. "
    "Be brief and kind. Do not diagnose or prescribe."
)

def _get_llm():
    # Lazy import so the package isn’t required until used
    try:
        from langchain_ollama import ChatOllama
    except Exception as e:
        raise RuntimeError(
            "Missing dependency 'langchain-ollama'. Install with:\n"
            "  python -m pip install -U langchain-ollama"
        ) from e

    return ChatOllama(
        model=settings.ollama_model,
        temperature=settings.temperature,
        base_url=settings.ollama_base_url,  # default Ollama server
    )

def chat_once(user_text: str, system: Optional[str] = None) -> str:
    msgs = [SystemMessage(content=system or SYSTEM_PROMPT),
            HumanMessage(content=user_text)]
    try:
        llm = _get_llm()
        resp = llm.invoke(msgs)
        return getattr(resp, "content", str(resp))
    except Exception as e:
        hint = (
            "\nTips:"
            "\n • Ensure the Ollama service is running (try `ollama --version` or `ollama serve`)."
            "\n • Confirm the model is pulled: `ollama pull llama3.1:8b`."
            "\n • Check OLLAMA_BASE_URL in your .env (default http://localhost:11434)."
        )
        return f"[LLM error] {e}{hint}"
