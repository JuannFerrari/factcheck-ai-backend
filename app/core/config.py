import os
from typing import Optional, List
from pydantic_settings import BaseSettings
from pydantic import Field, ConfigDict
from dotenv import load_dotenv

load_dotenv()


class Settings(BaseSettings):
    # Hugging Face OpenAI-compatible endpoint
    huggingface_api_key: str
    huggingface_base_url: str = "https://router.huggingface.co/v1"
    huggingface_model: str = "meta-llama/Meta-Llama-3-8B-Instruct"

    # Serper.dev Configuration
    serper_api_key: str

    # Application Configuration
    app_name: str = "factcheck-ai-backend"
    app_version: str = "1.0.0"
    environment: str = "development"

    # API Configuration
    max_requests_per_minute: int = 60

    # Security & Production Configuration
    cors_origins: List[str] = ["http://localhost:3000"]
    api_key_header: str = "X-API-Key"
    api_key: Optional[str] = None
    rate_limit_per_minute: int = 60

    # Logging Configuration
    log_level: str = "INFO"

    model_config = ConfigDict(
        env_file=".env",
        case_sensitive=False,
        extra="ignore"
    )


settings = Settings()
