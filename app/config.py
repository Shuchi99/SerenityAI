from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

class Settings(BaseSettings):
    # LLM (Ollama)
    temperature: float = Field(default=0.1, env="TEMPERATURE")
    ollama_model: str = Field(default="llama3.1:8b", env="OLLAMA_MODEL")
    ollama_base_url: str = Field(default="http://localhost:11434", env="OLLAMA_BASE_URL")

    # RAG settings
    kb_dir: str = Field(default="knowledge_base", env="KB_DIR")
    index_dir: str = Field(default=".storage/index", env="INDEX_DIR")
    embed_model: str = Field(default="sentence-transformers/all-MiniLM-L6-v2", env="EMBED_MODEL")

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

settings = Settings()
