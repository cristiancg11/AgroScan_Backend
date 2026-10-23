"""
AgroScan AI - Andean Crop Phytosanitary Diagnostic API.
Built with FastAPI, Pydantic, MongoDB Atlas (Motor), and Google Gemini 2.5 Flash Vision.
"""

import logging
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import List, Optional
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.auth import (
    create_access_token,
    get_current_user,
    get_current_user_optional,
    get_password_hash,
    get_user_by_email,
    save_user,
    verify_password,
)
from app.config import settings
from app.db import (
    diagnostics_collection,
    init_db,
    is_mongo_connected,
    ping_db,
    users_collection,
)
from app.schemas import (
    AuthResponse,
    ChatMessage,
    ChatFollowUpRequest,
    ChatFollowUpResponse,
    CropType,
    DiagnosticListResponse,
    DiagnosticResponse,
    FarmerRole,
    HealthCheckResponse,
    UserLoginRequest,
    UserProfile,
    UserRegisterRequest,
)
from app.services.gemini_service import gemini_service

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("agroscan.api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifecycle management: initializes database indexes on startup.
    """
    logger.info("Starting up AgroScan AI API...")
    try:
        await init_db()
        logger.info("Database indexes initialized successfully.")
    except Exception as exc:
        logger.warning("Initial DB index creation skipped: %s", exc)
    yield
    logger.info("Shutting down AgroScan AI API...")


# Initialize FastAPI Application
app = FastAPI(
    title="AgroScan AI API",
    description=(
        "Phytosanitary diagnostic REST API powered by Google Gemini 2.5 Flash vision and "
        "MongoDB Atlas persistence with JWT farmer authentication. "
        "Designed for smallholder Andean farmers in Nariño, Colombia."
    ),
    version="1.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# Configure CORS Middleware for web frontend integration
cors_origins = settings.cors_origins_list
logger.info("Configuring CORS middleware with origins: %s", cors_origins)
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1)(:[0-9]+)?$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory store for diagnostic history fallback across offline test sessions
in_memory_history: List[DiagnosticResponse] = []


# ============================================================================
# AUTHENTICATION ENDPOINTS (JWT + MongoDB Atlas)
# ============================================================================

@app.post(
    "/api/v1/auth/register",
    response_model=AuthResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["Authentication"],
    summary="Register New Farmer Account",
)
async def register(payload: UserRegisterRequest):
    """
    Creates a new farmer or agronomist account, hashes the password with bcrypt,
    persists into MongoDB Atlas, and returns a signed JWT access token.
    """
    email_clean = payload.email.strip().lower()

    # Check for existing account
    existing = await get_user_by_email(email_clean)
    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="An account with this email address already exists.",
        )

    if len(payload.password) < 6:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Password must be at least 6 characters long.",
        )

    now_iso = datetime.now(timezone.utc).isoformat()
    new_user_id = f"usr-{uuid.uuid4().hex[:10]}"
    avatar = f"https://api.dicebear.com/7.x/bottts/svg?seed={email_clean}"

    user_record = {
        "id": new_user_id,
        "email": email_clean,
        "password_hash": get_password_hash(payload.password),
        "full_name": payload.full_name.strip(),
        "farm_name": payload.farm_name.strip(),
        "municipality": payload.municipality.strip(),
        "role": payload.role.value if isinstance(payload.role, FarmerRole) else str(payload.role),
        "avatar_url": avatar,
        "created_at": now_iso,
    }

    # Save to MongoDB Atlas / fallback
    await save_user(user_record)

    # Issue JWT token
    token = create_access_token({"sub": new_user_id, "email": email_clean})

    profile = UserProfile(
        id=user_record["id"],
        email=user_record["email"],
        full_name=user_record["full_name"],
        farm_name=user_record["farm_name"],
        municipality=user_record["municipality"],
        role=FarmerRole(user_record["role"]) if isinstance(user_record["role"], str) else user_record["role"],
        avatar_url=user_record["avatar_url"],
        created_at=user_record["created_at"],
    )

    logger.info("New farmer registered: %s (%s)", email_clean, profile.role.value)
    return AuthResponse(access_token=token, token_type="bearer", user=profile)


@app.post(
    "/api/v1/auth/login",
    response_model=AuthResponse,
    status_code=status.HTTP_200_OK,
    tags=["Authentication"],
    summary="Authenticate Farmer Account",
)
async def login(payload: UserLoginRequest):
    """
    Validates farmer credentials against MongoDB Atlas bcrypt hash and returns a JWT access token.
    """
    email_clean = payload.email.strip().lower()
    user_record = await get_user_by_email(email_clean)

    if not user_record or not verify_password(payload.password, user_record.get("password_hash", "")):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password. Please verify your credentials.",
        )

    # Issue JWT access token
    token = create_access_token({"sub": user_record["id"], "email": user_record["email"]})

    profile = UserProfile(
        id=user_record["id"],
        email=user_record["email"],
        full_name=user_record["full_name"],
        farm_name=user_record["farm_name"],
        municipality=user_record["municipality"],
        role=FarmerRole(user_record["role"]) if isinstance(user_record["role"], str) else user_record["role"],
        avatar_url=user_record.get("avatar_url"),
        created_at=user_record["created_at"],
    )

    logger.info("Farmer authenticated: %s", email_clean)
    return AuthResponse(access_token=token, token_type="bearer", user=profile)


@app.get(
    "/api/v1/auth/me",
    response_model=UserProfile,
    tags=["Authentication"],
    summary="Get Current User Profile",
)
async def get_me(current_user: UserProfile = Depends(get_current_user)):
    """
    Returns the profile of the currently authenticated farmer based on JWT validation.
    """
    return current_user


# ============================================================================
# SYSTEM HEALTH & ROOT
# ============================================================================

@app.get("/", tags=["Root"])
async def root():
    """
    Root entrypoint providing service info and documentation links.
    """
    return {
        "service": "AgroScan AI Phytosanitary API",
        "version": "1.1.0",
        "target_region": "Nariño, Colombia (Andean Highlands)",
        "supported_crops": [c.value for c in CropType],
        "database": "MongoDB Atlas",
        "model": "gemini-2.5-flash",
        "docs": "/docs",
    }


@app.get("/health", response_model=HealthCheckResponse, tags=["Health"])
@app.get("/api/v1/health", response_model=HealthCheckResponse, tags=["Health"])
async def health_check():
    """
    Service health check endpoint reporting status, Gemini API, and MongoDB Atlas connectivity.
    """
    mongo_ok = await ping_db() if is_mongo_connected() else False
    return HealthCheckResponse(
        status="healthy",
        service="AgroScan AI",
        version="1.1.0",
        gemini_configured=gemini_service.is_configured,
        mongodb_connected=mongo_ok,
        model="gemini-2.5-flash",
    )


# ============================================================================
# DIAGNOSTICS & PERSISTENCE (Gemini 2.5 Flash + MongoDB Atlas)
# ============================================================================

@app.post(
    "/api/v1/diagnose",
    response_model=DiagnosticResponse,
    status_code=status.HTTP_200_OK,
    tags=["Diagnostics"],
    summary="Diagnose Andean Crop Image",
    description=(
        "Receives a multipart/form-data upload with plant image bytes and plot metadata. "
        "Processes the visual symptoms using Gemini 2.5 Flash vision, associates the diagnostic "
        "with the authenticated farmer, and persists the record into MongoDB Atlas."
    ),
)
async def diagnose_crop(
    image: UploadFile = File(..., description="High-resolution or worker-optimized plant photo"),
    crop_type: str = Form(..., description="Crop kind: Potato, Coffee, Corn, or Tomato"),
    plot_identifier: str = Form("Plot A - Main Terrace", description="Farm plot / lot identifier"),
    current_user: Optional[UserProfile] = Depends(get_current_user_optional),
):
    """
    Handles image analysis and phytosanitary diagnosis generation, persisting output to MongoDB Atlas.
    """
    # Validate crop type input against supported Andean crop enums
    normalized_crop = None
    for member in CropType:
        if member.value.lower() == crop_type.strip().lower():
            normalized_crop = member
            break

    if not normalized_crop:
        valid_options = ", ".join([c.value for c in CropType])
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid crop_type '{crop_type}'. Supported Andean crops are: {valid_options}",
        )

    # Validate image MIME content type
    allowed_mime_types = ["image/jpeg", "image/png", "image/webp", "image/jpg"]
    content_type = image.content_type or "image/jpeg"
    if content_type.lower() not in allowed_mime_types:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported media type '{content_type}'. Must be one of: {allowed_mime_types}",
        )

    try:
        # Read uploaded image bytes
        image_bytes = await image.read()
        if not image_bytes or len(image_bytes) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Empty image payload received. Please provide a valid crop photo.",
            )

        # Enforce maximum payload limit (15MB uncompressed limit)
        if len(image_bytes) > 15 * 1024 * 1024:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail="Image size exceeds maximum limit of 15MB. Optimize on client thread before upload.",
            )

        # Call Gemini 2.5 Flash vision service
        diagnostic_result = await gemini_service.diagnose_crop(
            image_bytes=image_bytes,
            mime_type=content_type,
            crop_type=normalized_crop,
            plot_identifier=plot_identifier.strip() or "General Plot",
        )

        # Associate authenticated user ID
        user_id = current_user.id if current_user else "usr-guest"
        diagnostic_result.user_id = user_id

        # Persist diagnostic directly into MongoDB Atlas
        if diagnostics_collection is not None:
            try:
                diag_doc = diagnostic_result.model_dump()
                await diagnostics_collection.insert_one(diag_doc)
                logger.info(
                    "Persisted diagnosis %s to MongoDB Atlas for user %s",
                    diagnostic_result.id,
                    user_id,
                )
            except Exception as exc:
                logger.error("Failed to persist diagnosis to MongoDB Atlas: %s", exc)

        # Prepend to in-memory history cache as fallback
        in_memory_history.insert(0, diagnostic_result)
        if len(in_memory_history) > 100:
            in_memory_history.pop()

        return diagnostic_result

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Unexpected error processing diagnostic request: %s", exc)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": f"An unexpected diagnostic error occurred: {str(exc)}"},
        )


@app.get(
    "/api/v1/history",
    response_model=DiagnosticListResponse,
    tags=["Diagnostics"],
    summary="Retrieve Diagnostic History",
    description="Returns phytosanitary diagnostic sessions queried from MongoDB Atlas filtered by user and plot.",
)
async def get_diagnostic_history(
    plot_identifier: Optional[str] = None,
    limit: int = 20,
    current_user: Optional[UserProfile] = Depends(get_current_user_optional),
):
    """
    Retrieves previous diagnoses stored in MongoDB Atlas, filtered by user plots.
    """
    items: List[DiagnosticResponse] = []

    # Query MongoDB Atlas if configured
    if diagnostics_collection is not None:
        try:
            query_filter: dict = {}
            if current_user:
                query_filter["user_id"] = current_user.id
            if plot_identifier and plot_identifier.strip():
                query_filter["plot_identifier"] = {
                    "$regex": plot_identifier.strip(),
                    "$options": "i",
                }

            cursor = (
                diagnostics_collection.find(query_filter, {"_id": 0})
                .sort("created_at", -1)
                .limit(limit)
            )
            raw_docs = await cursor.to_list(length=limit)
            if raw_docs:
                items = [DiagnosticResponse(**doc) for doc in raw_docs]
                total = await diagnostics_collection.count_documents(query_filter)
                return DiagnosticListResponse(total=total, items=items)
        except Exception as exc:
            logger.warning("MongoDB Atlas history query error: %s. Falling back to local cache.", exc)

    # Fallback to in-memory history
    filtered = in_memory_history
    if current_user:
        filtered = [item for item in filtered if item.user_id == current_user.id or not item.user_id]
    if plot_identifier and plot_identifier.strip():
        plot_clean = plot_identifier.strip().lower()
        filtered = [item for item in filtered if plot_clean in item.plot_identifier.lower()]

    paginated = filtered[:limit]
    return DiagnosticListResponse(
        total=len(filtered),
        items=paginated,
    )


@app.post(
    "/api/v1/diagnose/{diagnostic_id}/chat",
    response_model=ChatFollowUpResponse,
    tags=["Diagnostics", "Chat"],
    summary="Interactive Agronomic Follow-up Consultation",
    description=(
        "Enables interactive follow-up technical consultation grounded on a specific "
        "crop diagnostic report. Powered by Gemini 2.5 Flash with persistent message history in MongoDB Atlas."
    ),
)
async def agronomic_followup_chat(
    diagnostic_id: str,
    payload: ChatFollowUpRequest,
    current_user: Optional[UserProfile] = Depends(get_current_user_optional),
):
    """
    Handles follow-up technical questions grounded on a specific diagnostic session.
    Retrieves ground truth context from MongoDB Atlas, invokes Gemini 2.5 Flash,
    persists message thread turns, and returns practical field recommendations.
    """
    if not payload.message.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Message content cannot be empty.",
        )

    # 1. Retrieve diagnostic context from MongoDB Atlas or local memory
    diag_context: Optional[dict] = None
    if diagnostics_collection is not None:
        try:
            doc = await diagnostics_collection.find_one({"id": diagnostic_id}, {"_id": 0})
            if doc:
                diag_context = doc
        except Exception as exc:
            logger.warning("MongoDB diagnostic lookup error for chat: %s", exc)

    if not diag_context:
        for item in in_memory_history:
            if item.id == diagnostic_id:
                diag_context = item.model_dump()
                break

    # If diagnostic is still not located (e.g., local mock or offline session), synthesize context
    if not diag_context:
        diag_context = {
            "id": diagnostic_id,
            "crop_type": "Potato",
            "plot_identifier": "Plot A - Monitored Lot",
            "disease_name": "Phytosanitary Assessment",
            "scientific_name": "Solanum tuberosum condition",
            "pathogen_type": "FUNGUS",
            "severity_level": "MODERATE",
            "symptoms": ["Observable foliar spots", "Leaf chlorosis"],
            "organic_treatment": ["Bordeaux mixture 1%", "Trichoderma harzianum bio-fungicide"],
            "chemical_treatment": ["Metalaxyl-M + Mancozeb curative spray"],
            "preventive_measures": ["Sterilize farm pruning tools", "Optimize terrace drainage"],
        }

    # 2. Invoke Gemini 2.5 Flash follow-up engine
    followup_response = await gemini_service.generate_agronomic_followup(
        diagnostic_context=diag_context,
        user_question=payload.message.strip(),
        history=payload.chat_history,
    )

    # 3. Persist message thread turns to MongoDB Atlas
    user_turn = ChatMessage(
        role="user",
        content=payload.message.strip(),
    )
    model_turn = ChatMessage(
        role="model",
        content=followup_response.reply,
        timestamp=followup_response.timestamp,
    )

    if diagnostics_collection is not None:
        try:
            await diagnostics_collection.update_one(
                {"id": diagnostic_id},
                {
                    "$push": {
                        "chat_thread": {
                            "$each": [user_turn.model_dump(), model_turn.model_dump()]
                        }
                    }
                },
                upsert=False,
            )
            logger.info("Persisted conversational thread turns for diagnostic %s", diagnostic_id)
        except Exception as exc:
            logger.error("Failed to append chat turn to MongoDB Atlas: %s", exc)

    return followup_response


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host=settings.HOST, port=settings.PORT, reload=True)

