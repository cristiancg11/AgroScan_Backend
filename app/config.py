"""
Configuration settings for AgroScan AI backend.
Reads configuration from environment variables and .env file.
"""

from typing import List
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings model powered by Pydantic Settings.
    """
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    gemini_api_key: str = ""
    database_url: str = "postgresql://postgres.dthmcpvbbhqvbywpjpre:3GVS?rqGD%zBGy@@aws-0-us-east-2.pooler.supabase.com:6543/postgres?pgbouncer=true"
    host: str = "0.0.0.0"
    port: int = 8000
    cors_origins: str = "http://localhost:5173"

    @property
    def cors_origins_list(self) -> List[str]:
        """
        Parses comma-separated CORS origins into a list of strings.
        """
        if not self.cors_origins:
            return ["*"]
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]


settings = Settings()
