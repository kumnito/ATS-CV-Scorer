from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    anthropic_api_key: str = ""
    vision_llm_enabled: bool = False
    claude_model: str = "claude-sonnet-4-6"
    spacy_model: str = "en_core_web_sm"
    sentence_transformer_model: str = "all-MiniLM-L6-v2"
    max_pdf_size_mb: int = 10

    adzuna_id: str = ""
    adzuna_api_key: str = ""
    adzuna_country: str = "fr"


settings = Settings()
