"""
Security and Cryptography Module for AgroScan AI.
Provides bcrypt password hashing, credential verification, and signed JWT issuance.
"""

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional
import jwt
from passlib.context import CryptContext

from app.config import settings

# Cryptographic password hashing context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def get_password_hash(password: str) -> str:
    """
    Computes secure one-way bcrypt salt and hash of plaintext password.
    """
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Verifies plaintext password against stored bcrypt hash.
    """
    return pwd_context.verify(plain_password, hashed_password)


def create_access_token(
    data: Dict[str, Any],
    expires_delta: Optional[timedelta] = None,
) -> str:
    """
    Generates signed HMAC-SHA256 (HS256) JSON Web Token (JWT) with expiration.
    """
    to_encode = data.copy()
    now_utc = datetime.now(timezone.utc)
    if expires_delta:
        expire = now_utc + expires_delta
    else:
        expire = now_utc + timedelta(minutes=settings.access_token_expire_minutes)

    to_encode.update({"exp": expire, "iat": now_utc})
    encoded_jwt = jwt.encode(
        to_encode,
        settings.secret_key,
        algorithm=settings.algorithm,
    )
    return encoded_jwt


def decode_access_token(token: str) -> Optional[Dict[str, Any]]:
    """
    Decodes and validates JWT token signature and expiration.
    Returns token payload dictionary or None if invalid.
    """
    try:
        payload = jwt.decode(
            token,
            settings.secret_key,
            algorithms=[settings.algorithm],
        )
        return payload
    except (jwt.PyJWTError, Exception):
        return None
