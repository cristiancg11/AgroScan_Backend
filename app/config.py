"""
Application Configuration Module using Pydantic Settings.
Loads environment variables for Gemini API, MongoDB Atlas, JWT authentication, and CORS origins.
"""

from typing import List
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application Settings class.
    Reads configuration values from environment variables or a .env file.
    """
    GEMINI_API_KEY: str = ""
    MONGODB_URI: str = ""
    MONGODB_DB_NAME: str = "agroscan_db"
    SECRET_KEY: str = "agroscan-secure-jwt-key-andean-highlands-2026"
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 10080  # 7 days
    CORS_ORIGINS: str = "http://localhost:5173,http://127.0.0.1:5173,http://localhost:3000,http://localhost:5180,http://127.0.0.1:5180"
    PORT: int = 8000
    HOST: str = "0.0.0.0"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    @property
    def cors_origins_list(self) -> List[str]:
        """
        Parses comma-separated CORS_ORIGINS string into a list of cleaned strings.
        """
        if not self.CORS_ORIGINS:
            return ["*"]
        return [origin.strip() for origin in self.CORS_ORIGINS.split(",") if origin.strip()]


# Global singleton instance of application settings
settings = Settings()
