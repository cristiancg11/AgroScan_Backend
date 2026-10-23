"""
Authentication and Security Module for AgroScan AI.
Provides password hashing via passlib bcrypt and JWT token issuance / verification.
Manages user lookup from MongoDB Atlas and authenticated farmer contexts.
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import Optional
import jwt
from fastapi import Depends, Header, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from passlib.context import CryptContext

from app.config import settings
from app.db import users_collection, _in_memory_users
from app.schemas import FarmerRole, UserProfile

logger = logging.getLogger("agroscan.auth")

# Password hashing context with bcrypt
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# OAuth2 scheme for Swagger UI integration
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/login", auto_error=False)

# Seed demo users for instantaneous field testing and testing resilience
DEMO_USERS = [
    {
        "id": "usr-carlos-guancha",
        "email": "carlos@agroscan.co",
        "password_hash": pwd_context.hash("narino2026"),
        "full_name": "Don Carlos Guancha",
        "farm_name": "Finca Bella Vista",
        "municipality": "Túquerres (Plateau)",
        "role": FarmerRole.SMALLHOLDER,
        "avatar_url": "https://images.unsplash.com/photo-1544717305-2782549b5136?w=150&auto=format&fit=crop&q=80",
        "created_at": "2026-01-15T08:00:00Z",
    },
    {
        "id": "usr-elena-bastidas",
        "email": "elena@agrosavia.co",
        "password_hash": pwd_context.hash("narino2026"),
        "full_name": "Dra. Elena Bastidas",
        "farm_name": "Centro Experimental Obonuco",
        "municipality": "Pasto (Galeras Slopes)",
        "role": FarmerRole.AGRONOMIST,
        "avatar_url": "https://images.unsplash.com/photo-1573496359142-b8d87734a5a2?w=150&auto=format&fit=crop&q=80",
        "created_at": "2026-02-10T09:30:00Z",
    },
]

# Populate in-memory map
for du in DEMO_USERS:
    _in_memory_users[du["email"].lower()] = du


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Verifies a plaintext password against a stored bcrypt hash.
    """
    try:
        return pwd_context.verify(plain_password, hashed_password)
    except Exception as exc:
        logger.warning("Password verification check error: %s", exc)
        return False


def get_password_hash(password: str) -> str:
    """
    Generates a secure bcrypt hash of the provided password.
    """
    return pwd_context.hash(password)


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """
    Generates a signed JWT access token encoding user identity and expiration.
    """
    to_encode = data.copy()
    now_utc = datetime.now(timezone.utc)
    if expires_delta:
        expire = now_utc + expires_delta
    else:
        expire = now_utc + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire, "iat": now_utc})
    encoded_jwt = jwt.encode(
        to_encode,
        settings.SECRET_KEY,
        algorithm=settings.JWT_ALGORITHM,
    )
    return encoded_jwt


def decode_access_token(token: str) -> Optional[dict]:
    """
    Decodes and validates a JWT access token using the configured secret key.
    """
    try:
        payload = jwt.decode(
            token,
            settings.SECRET_KEY,
            algorithms=[settings.JWT_ALGORITHM],
        )
        return payload
    except jwt.PyJWTError as exc:
        logger.debug("JWT decode failure: %s", exc)
        return None


async def get_user_by_email(email: str) -> Optional[dict]:
    """
    Fetches a user record by email from MongoDB Atlas, falling back to in-memory store.
    """
    clean_email = email.strip().lower()
    if users_collection is not None:
        try:
            doc = await users_collection.find_one({"email": clean_email})
            if doc:
                return doc
        except Exception as exc:
            logger.warning("MongoDB user query failed: %s", exc)
    return _in_memory_users.get(clean_email)


async def get_user_by_id(user_id: str) -> Optional[dict]:
    """
    Fetches a user record by ID from MongoDB Atlas, falling back to in-memory store.
    """
    if users_collection is not None:
        try:
            doc = await users_collection.find_one({"id": user_id})
            if doc:
                return doc
        except Exception as exc:
            logger.warning("MongoDB user query by id failed: %s", exc)

    for user in _in_memory_users.values():
        if user.get("id") == user_id:
            return user
    return None


async def save_user(user_record: dict) -> None:
    """
    Persists a new user record to MongoDB Atlas and local memory.
    """
    clean_email = user_record["email"].strip().lower()
    _in_memory_users[clean_email] = user_record

    if users_collection is not None:
        try:
            # Upsert into MongoDB Atlas
            await users_collection.update_one(
                {"email": clean_email},
                {"$set": user_record},
                upsert=True,
            )
            logger.info("User persisted to MongoDB Atlas: %s", clean_email)
        except Exception as exc:
            logger.error("Failed to persist user in MongoDB Atlas: %s", exc)


def extract_token_from_header(authorization: Optional[str], token_bearer: Optional[str]) -> Optional[str]:
    """
    Resolves JWT token string from Authorization header or OAuth2 bearer.
    """
    if authorization and authorization.lower().startswith("bearer "):
        return authorization.split(" ", 1)[1].strip()
    if token_bearer:
        return token_bearer.strip()
    return None


async def get_current_user_optional(
    authorization: Optional[str] = Header(None),
    token_bearer: Optional[str] = Depends(oauth2_scheme),
) -> Optional[UserProfile]:
    """
    Resolves the authenticated UserProfile if a valid token is provided, or None.
    """
    raw_token = extract_token_from_header(authorization, token_bearer)
    if not raw_token:
        return None

    # Handle demo tokens
    if raw_token.startswith("demo-"):
        for demo in DEMO_USERS:
            if demo["id"] in raw_token or "carlos" in raw_token:
                return UserProfile(
                    id=demo["id"],
                    email=demo["email"],
                    full_name=demo["full_name"],
                    farm_name=demo["farm_name"],
                    municipality=demo["municipality"],
                    role=demo["role"],
                    avatar_url=demo.get("avatar_url"),
                    created_at=demo["created_at"],
                )

    payload = decode_access_token(raw_token)
    if not payload:
        return None

    email = payload.get("email") or payload.get("sub")
    if not email:
        return None

    user_doc = await get_user_by_email(email)
    if not user_doc:
        user_doc = await get_user_by_id(email)

    if not user_doc:
        return None

    return UserProfile(
        id=user_doc["id"],
        email=user_doc["email"],
        full_name=user_doc["full_name"],
        farm_name=user_doc["farm_name"],
        municipality=user_doc["municipality"],
        role=FarmerRole(user_doc["role"]) if isinstance(user_doc["role"], str) else user_doc["role"],
        avatar_url=user_doc.get("avatar_url"),
        created_at=user_doc["created_at"],
    )


async def get_current_user(
    current_user: Optional[UserProfile] = Depends(get_current_user_optional),
) -> UserProfile:
    """
    Strict dependency requiring a valid authenticated user session.
    """
    if not current_user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication credentials were not provided or session has expired.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return current_user
