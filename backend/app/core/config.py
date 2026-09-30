"""
AI Multi-Factor Stock Intelligence & Portfolio Decision-Support System
backend/app/core/config.py

Application configuration loaded from environment variables via Pydantic Settings.
All later modules import settings from here — never read os.environ directly.
"""

from enum import StrEnum

from pydantic_settings import BaseSettings, SettingsConfigDict


class Environment(StrEnum):
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


class Settings(BaseSettings):
    """
    Central application configuration.

    Values are read from environment variables (case-insensitive).
    A .env file in the working directory is loaded automatically during development.
    In production, inject variables via Docker / platform secrets.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ------------------------------------------------------------------
    # Application
    # ------------------------------------------------------------------
    APP_NAME: str = "AI Stock Intelligence System"
    APP_VERSION: str = "0.1.0"
    ENVIRONMENT: Environment = Environment.DEVELOPMENT
    DEBUG: bool = False

    # ------------------------------------------------------------------
    # API
    # ------------------------------------------------------------------
    API_V1_PREFIX: str = "/api/v1"

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------
    LOG_LEVEL: str = "INFO"
    LOG_JSON: bool = False  # Set True in production for structured JSON logs

    # ------------------------------------------------------------------
    # Server (used by Uvicorn at startup)
    # ------------------------------------------------------------------
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # ------------------------------------------------------------------
    # Database
    # ------------------------------------------------------------------
    database_url: str = "postgresql+psycopg://stockvue:stockvue@localhost:5432/stockvue"

    @property
    def is_development(self) -> bool:
        return self.ENVIRONMENT == Environment.DEVELOPMENT

    @property
    def is_production(self) -> bool:
        return self.ENVIRONMENT == Environment.PRODUCTION


# Module-level singleton — import this everywhere.
settings = Settings()
