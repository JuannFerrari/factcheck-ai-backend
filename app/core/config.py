from typing import Optional, List
from pydantic_settings import BaseSettings
from pydantic import ConfigDict, Field, field_validator
from dotenv import load_dotenv
import json

load_dotenv()


class Settings(BaseSettings):
    # Hugging Face OpenAI-compatible endpoint
    huggingface_api_key: str
    huggingface_base_url: str = "https://router.huggingface.co/v1"
    huggingface_model: str = "meta-llama/Meta-Llama-3-8B-Instruct"

    # Serper.dev Configuration
    serper_api_key: str

    # Neon PostgreSQL Configuration
    neon_database_url: Optional[str] = None
    neon_database_host: Optional[str] = None
    neon_database_port: int = 5432
    neon_database_name: str = "factcheck_db"
    neon_database_user: Optional[str] = None
    neon_database_password: Optional[str] = None

    # Vector Database Configuration
    vector_similarity_threshold: float = Field(default=0.55, ge=0.0, le=1.0)
    max_vector_results: int = Field(default=3, ge=1, le=10)
    enable_vector_search: bool = True
    enable_vector_storage: bool = True

    # Application Configuration
    app_name: str = "factcheck-ai-backend"
    app_version: str = "1.0.0"
    environment: str = "development"

    # Security & Production Configuration
    cors_origins: List[str] = ["http://localhost:3000"]
    api_key_header: str = "X-API-Key"
    api_key: Optional[str] = None
    rate_limit_per_minute: int = Field(default=60, ge=1, le=1000)

    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, v):
        """Parse CORS origins from various formats"""
        if isinstance(v, str):
            # Try to parse as JSON first
            try:
                return json.loads(v)
            except json.JSONDecodeError:
                # If not JSON, treat as comma-separated string
                return [origin.strip() for origin in v.split(",") if origin.strip()]
        return v

    # Logging Configuration
    log_level: str = Field(
        default="INFO", pattern="^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$"
    )

    # Content Moderation Configuration
    enable_content_moderation: bool = True
    content_moderation_strictness: str = Field(
        default="medium", pattern="^(low|medium|high)$"
    )
    skip_web_search_for_inappropriate: bool = True

    # Performance & Resilience Configuration
    # Database connection pooling
    db_pool_size: int = Field(default=10, ge=1, le=50)
    db_max_overflow: int = Field(default=20, ge=0, le=100)
    db_pool_timeout: int = Field(default=30, ge=5, le=60)

    # AI Model Configuration
    model_timeout: float = Field(default=30.0, ge=5.0, le=120.0)
    model_max_retries: int = Field(default=3, ge=1, le=5)
    model_max_tokens: int = Field(default=512, ge=100, le=2048)
    model_temperature: float = Field(default=0.1, ge=0.0, le=1.0)

    # Search Configuration
    web_search_results: int = Field(default=5, ge=1, le=10)
    max_combined_sources: int = Field(default=8, ge=3, le=15)

    # Caching Configuration
    cache_timeout: float = Field(default=25.0, ge=10.0, le=60.0)
    similarity_threshold: float = Field(default=0.55, ge=0.0, le=1.0)

    # Monitoring & Health Checks
    enable_health_checks: bool = True
    health_check_interval: int = Field(default=300, ge=60, le=3600)  # 5 minutes
    enable_metrics: bool = Field(default=False)  # For future Prometheus integration

    # Error Handling
    max_retry_attempts: int = Field(default=3, ge=1, le=5)
    retry_backoff_multiplier: float = Field(default=1.0, ge=0.5, le=3.0)
    retry_min_wait: float = Field(default=4.0, ge=1.0, le=10.0)
    retry_max_wait: float = Field(default=10.0, ge=5.0, le=30.0)

    model_config = ConfigDict(env_file=".env", case_sensitive=False, extra="ignore")


settings = Settings()
