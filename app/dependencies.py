"""
FastAPI request dependencies for authentication and user extraction.
"""

import logging
from typing import Optional
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session

from app.db import get_db
from app.models import User
from app.security import decode_access_token

logger = logging.getLogger("agroscan.auth")

# OAuth2 password bearer token extractor (auto_error=False allows optional authentication)
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/login", auto_error=False)


def get_current_user(
    token: Optional[str] = Depends(oauth2_scheme),
    db: Session = Depends(get_db),
) -> User:
    """
    Mandatory dependency that extracts and verifies Bearer JWT token from request header.
    Returns authenticated User model instance or raises HTTP 401 Unauthorized.
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials or session has expired.",
        headers={"WWW-Authenticate": "Bearer"},
    )

    if not token:
        raise credentials_exception

    payload = decode_access_token(token)
    if not payload:
        raise credentials_exception

    user_id: Optional[str] = payload.get("sub")
    if not user_id:
        raise credentials_exception

    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise credentials_exception

    return user


def get_current_user_optional(
    token: Optional[str] = Depends(oauth2_scheme),
    db: Session = Depends(get_db),
) -> Optional[User]:
    """
    Optional dependency allowing public/guest endpoints to capture user context
    when Bearer token is provided without failing when omitted.
    """
    if not token:
        return None

    payload = decode_access_token(token)
    if not payload:
        return None

    user_id: Optional[str] = payload.get("sub")
    if not user_id:
        return None

    return db.query(User).filter(User.id == user_id).first()
