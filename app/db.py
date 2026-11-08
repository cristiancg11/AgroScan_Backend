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

# Supabase PgBouncer pooler connection engine
# pool_pre_ping checks connection liveness before checking out from pool
# pool_recycle handles connection timeouts in cloud poolers
engine = create_engine(
    settings.database_url,
    pool_pre_ping=True,
    pool_recycle=300,
)

# Thread-local session factory
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
)

# Declarative ORM base
Base = declarative_base()


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
