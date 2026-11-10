"""
Database connection and session management for Supabase PostgreSQL with PgBouncer.
Configures SQLAlchemy engine, session maker, and session dependency.
"""

import logging
from typing import Generator
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker, Session

from app.config import settings

logger = logging.getLogger("agroscan.db")


def get_clean_database_url(raw_url: str) -> str:
    """
    Sanitizes database URL for SQLAlchemy and psycopg2.
    Strips surrounding quotes and removes '?pgbouncer=true' (Prisma-specific option not supported by libpq).
    Ensures postgresql+psycopg2:// scheme is used for stable connectivity across environments.
    """
    cleaned = raw_url.strip().strip('"').strip("'")
    if "?pgbouncer=true" in cleaned:
        cleaned = cleaned.replace("?pgbouncer=true", "")
    elif "&pgbouncer=true" in cleaned:
        cleaned = cleaned.replace("&pgbouncer=true", "")

    if cleaned.startswith("postgresql://"):
        cleaned = cleaned.replace("postgresql://", "postgresql+psycopg2://", 1)

    return cleaned


# Supabase PgBouncer pooler connection engine
# pool_pre_ping checks connection liveness before checking out from pool
# pool_recycle handles connection timeouts in cloud poolers
engine = create_engine(
    get_clean_database_url(settings.database_url),
    pool_pre_ping=True,
    pool_recycle=300,
    pool_size=10,
    max_overflow=20,
    pool_timeout=30,
)

# Thread-local session factory
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
)

# Declarative ORM base
Base = declarative_base()


def init_db_schema():
    """
    Ensures all metadata tables and entities are registered in Supabase PostgreSQL.
    """
    Base.metadata.create_all(bind=engine)


def get_db() -> Generator[Session, None, None]:
    """
    FastAPI dependency that yields a SQLAlchemy database session per request
    and guarantees proper closure and cleanup upon completion.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
