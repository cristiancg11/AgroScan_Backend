"""
MongoDB Atlas Persistence Layer for AgroScan AI.
Configures AsyncIOMotorClient with Motor and TLS certification using certifi.
Manages database connection and exports 'users' and 'diagnostics' collections.
"""

import logging
from typing import Optional
import certifi
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorCollection, AsyncIOMotorDatabase
from app.config import settings

logger = logging.getLogger("agroscan.db")

client: Optional[AsyncIOMotorClient] = None
db: Optional[AsyncIOMotorDatabase] = None
users_collection: Optional[AsyncIOMotorCollection] = None
diagnostics_collection: Optional[AsyncIOMotorCollection] = None

# In-memory fallbacks when MongoDB URI is not configured or in offline test environments
_in_memory_users: dict = {}
_in_memory_diagnostics: list = []


def is_mongo_connected() -> bool:
    """
    Checks if MongoDB client is initialized with a configured URI.
    """
    return client is not None and bool(settings.MONGODB_URI)


def get_client() -> Optional[AsyncIOMotorClient]:
    """
    Returns the active AsyncIOMotorClient instance.
    """
    global client
    if client is None and settings.MONGODB_URI:
        try:
            logger.info("Connecting to MongoDB Atlas with Motor...")
            client = AsyncIOMotorClient(
                settings.MONGODB_URI,
                tlsCAFile=certifi.where(),
                serverSelectionTimeoutMS=5000,
                connectTimeoutMS=5000,
            )
            logger.info("MongoDB Atlas client created successfully.")
        except Exception as exc:
            logger.error("Failed to initialize MongoDB client: %s", exc)
            client = None
    return client


def get_database() -> Optional[AsyncIOMotorDatabase]:
    """
    Returns the AgroScan database handle.
    """
    global db
    cli = get_client()
    if cli is not None:
        db = cli[settings.MONGODB_DB_NAME]
    return db


# Initialize default collections
if settings.MONGODB_URI:
    try:
        client = AsyncIOMotorClient(
            settings.MONGODB_URI,
            tlsCAFile=certifi.where(),
            serverSelectionTimeoutMS=5000,
            connectTimeoutMS=5000,
        )
        db = client[settings.MONGODB_DB_NAME]
        users_collection = db["users"]
        diagnostics_collection = db["diagnostics"]
        logger.info("MongoDB Atlas collections 'users' and 'diagnostics' initialized.")
    except Exception as exc:
        logger.warning("MongoDB Atlas initialization deferred: %s", exc)
else:
    logger.info("MONGODB_URI not configured. Operating with in-memory persistence fallback.")


async def init_db():
    """
    Ensures optimal indexes exist in MongoDB Atlas collections.
    """
    if users_collection is not None:
        try:
            await users_collection.create_index("email", unique=True)
            await users_collection.create_index("id", unique=True)
            await diagnostics_collection.create_index("user_id")
            await diagnostics_collection.create_index("plot_identifier")
            await diagnostics_collection.create_index("created_at")
            logger.info("MongoDB Atlas indexes verified.")
        except Exception as exc:
            logger.warning("Index verification skipped: %s", exc)


async def ping_db() -> bool:
    """
    Executes a ping command against MongoDB Atlas to verify live connectivity.
    """
    if db is None:
        return False
    try:
        await db.command("ping")
        return True
    except Exception as exc:
        logger.debug("MongoDB Atlas ping failed: %s", exc)
        return False
