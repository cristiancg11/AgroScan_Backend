"""
AgroScan AI - Andean Crop Phytosanitary Diagnostic & Advisory Engine.
FastAPI REST API powered by Google Gemini 2.5 Flash Vision, Supabase PostgreSQL (SQLAlchemy + PgBouncer),
and an interactive agronomic follow-up consultation engine.
"""

import logging
from contextlib import asynccontextmanager
from typing import List, Optional
from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.config import settings
from app.db import Base, engine, get_db
from app.models import ChatThreadMessage, DiagnosticRecord
from app.schemas import (
    ChatMessage,
    ChatFollowUpRequest,
    ChatFollowUpResponse,
    CropType,
    DiagnosticResponse,
    HealthCheckResponse,
)
from app.services.gemini_service import gemini_service

# Configure application logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("agroscan.api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifecycle manager: creates relational tables on Supabase PostgreSQL upon startup.
    """
    logger.info("Starting AgroScan AI Backend Service...")
    try:
        logger.info("Initializing relational schema in Supabase PostgreSQL...")
        Base.metadata.create_all(bind=engine)
        logger.info("Supabase PostgreSQL schema verified successfully.")
    except Exception as exc:
        logger.error("Failed to initialize database tables: %s", exc)
    yield
    logger.info("Shutting down AgroScan AI Backend Service...")


# Initialize FastAPI application
app = FastAPI(
    title="AgroScan AI - Andean Phytosanitary API",
    description=(
        "Production-grade phytosanitary diagnostic and parcel management API for Andean smallholders in Nariño, Colombia. "
        "Powered by Google Gemini 2.5 Flash multimodal vision and Supabase PostgreSQL relational persistence."
    ),
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# Configure CORS Middleware
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


# ============================================================================
# SYSTEM HEALTH & ROOT ENDPOINTS
# ============================================================================

@app.get("/", tags=["System"])
def root_endpoint():
    """
    Root endpoint providing service metadata and documentation links.
    """
    return {
        "service": "AgroScan AI - Andean Phytosanitary Engine",
        "version": "2.0.0",
        "target_region": "Nariño, Colombia (Andean Highlands)",
        "supported_crops": [c.value for c in CropType],
        "database": "Supabase PostgreSQL (PgBouncer)",
        "vision_model": "gemini-2.5-flash",
        "docs": "/docs",
    }


@app.get("/health", response_model=HealthCheckResponse, tags=["System"])
def health_check(db: Session = Depends(get_db)):
    """
    Comprehensive health check verifying FastAPI service, Gemini API configuration,
    and live connectivity to Supabase PostgreSQL via PgBouncer.
    """
    database_ok = False
    try:
        db.execute(text("SELECT 1"))
        database_ok = True
    except Exception as exc:
        logger.error("Database health check ping failed: %s", exc)

    return HealthCheckResponse(
        status="healthy" if database_ok else "degraded",
        service="AgroScan AI Backend",
        version="2.0.0",
        gemini_configured=gemini_service.is_configured,
        database_connected=database_ok,
    )


# ============================================================================
# PHYTOSANITARY DIAGNOSTIC ENDPOINTS
# ============================================================================

@app.post(
    "/api/v1/diagnose",
    response_model=DiagnosticResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["Diagnostics"],
    summary="Diagnose Andean Crop Image",
    description="Receives crop leaf photo and plot metadata, runs Gemini 2.5 Flash vision, and persists result in Supabase.",
)
async def diagnose_crop(
    image: UploadFile = File(..., description="Crop leaf, stem, or plant photo (JPEG/PNG/WebP)"),
    crop_type: str = Form(..., description="Crop kind: Potato, Coffee, Corn, or Tomato"),
    plot_name: str = Form("Plot A - Main Lot", description="Farm parcel / lot identifier"),
    notes: Optional[str] = Form(None, description="Optional field observations or symptoms noted by the farmer"),
    db: Session = Depends(get_db),
):
    """
    Performs visual phytosanitary analysis using Gemini 2.5 Flash, stores the diagnostic
    record in Supabase PostgreSQL, and returns the complete agronomic assessment.
    """
    # 1. Validate crop type against Andean crop enum
    normalized_crop = None
    for member in CropType:
        if member.value.lower() == crop_type.strip().lower():
            normalized_crop = member
            break

    if not normalized_crop:
        valid_crops = ", ".join([c.value for c in CropType])
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid crop_type '{crop_type}'. Supported Andean crops are: {valid_crops}",
        )

    # 2. Validate MIME content type
    allowed_mime = ["image/jpeg", "image/png", "image/webp", "image/jpg"]
    content_type = image.content_type or "image/jpeg"
    if content_type.lower() not in allowed_mime:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported file type '{content_type}'. Must be one of: {allowed_mime}",
        )

    # 3. Read image bytes and validate payload
    try:
        image_bytes = await image.read()
        if not image_bytes or len(image_bytes) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Empty image received. Please upload a valid crop photo.",
            )

        if len(image_bytes) > 15 * 1024 * 1024:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail="Image exceeds maximum allowed size of 15MB.",
            )

        # 4. Execute multimodal analysis with Gemini 2.5 Flash
        diagnostic_result = await gemini_service.analyze_crop_health(
            image_bytes=image_bytes,
            mime_type=content_type,
            crop_type=normalized_crop,
            plot_name=plot_name.strip() or "General Lot",
            notes=notes.strip() if notes else None,
        )

        # 5. Persist record into Supabase PostgreSQL
        db_record = DiagnosticRecord(
            id=diagnostic_result.id,
            crop_type=diagnostic_result.crop_type.value,
            plot_name=diagnostic_result.plot_name,
            disease_name=diagnostic_result.disease_name,
            scientific_name=diagnostic_result.scientific_name,
            pathogen_type=diagnostic_result.pathogen_type.value,
            confidence_score=diagnostic_result.confidence_score,
            severity_level=diagnostic_result.severity_level.value,
            symptoms_analysis=diagnostic_result.symptoms_analysis,
            organic_treatment=diagnostic_result.organic_treatment,
            chemical_treatment=diagnostic_result.chemical_treatment,
            preventive_actions=diagnostic_result.preventive_actions,
            disclaimer=diagnostic_result.disclaimer,
            timestamp=diagnostic_result.timestamp,
        )

        db.add(db_record)
        db.commit()
        db.refresh(db_record)
        logger.info("Successfully persisted diagnostic record %s to Supabase PostgreSQL.", db_record.id)

        return diagnostic_result

    except HTTPException:
        db.rollback()
        raise
    except Exception as exc:
        db.rollback()
        logger.exception("Unexpected error diagnosing crop: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while processing diagnosis: {str(exc)}",
        )


@app.get(
    "/api/v1/history",
    response_model=List[DiagnosticResponse],
    tags=["Diagnostics"],
    summary="Retrieve Diagnostic History",
    description="Fetches previous phytosanitary diagnostics from Supabase PostgreSQL ordered by timestamp desc.",
)
def get_diagnostic_history(
    limit: int = 50,
    db: Session = Depends(get_db),
):
    """
    Queries all diagnostic records stored in Supabase PostgreSQL ordered from newest to oldest.
    """
    try:
        records = (
            db.query(DiagnosticRecord)
            .order_by(DiagnosticRecord.timestamp.desc())
            .limit(limit)
            .all()
        )
        return [DiagnosticResponse.model_validate(rec) for rec in records]
    except Exception as exc:
        logger.error("Failed to query diagnostic history from Supabase: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch diagnostic history: {str(exc)}",
        )


# ============================================================================
# AGRONOMIC CHAT FOLLOW-UP ENDPOINT
# ============================================================================

@app.post(
    "/api/v1/diagnose/{diagnostic_id}/chat",
    response_model=ChatFollowUpResponse,
    tags=["Consultation"],
    summary="Interactive Agronomic Follow-Up Consultation",
    description="Enables interactive consultation with field agronomist AI grounded on a specific diagnostic session.",
)
async def agronomic_followup_chat(
    diagnostic_id: str,
    payload: ChatFollowUpRequest,
    db: Session = Depends(get_db),
):
    """
    Handles follow-up technical questions grounded on a specific diagnostic record.
    Retrieves diagnostic ground truth, fetches chat history from Supabase, calls Gemini service,
    and records both question and answer turns into the chat_thread_messages table.
    """
    clean_message = payload.message.strip()
    if not clean_message:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Message content cannot be empty.",
        )

    # 1. Fetch diagnostic context from Supabase PostgreSQL
    record = db.query(DiagnosticRecord).filter(DiagnosticRecord.id == diagnostic_id).first()
    if not record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Diagnostic record '{diagnostic_id}' was not found.",
        )

    diagnostic_context = {
        "id": record.id,
        "crop_type": record.crop_type,
        "plot_name": record.plot_name,
        "disease_name": record.disease_name,
        "scientific_name": record.scientific_name,
        "pathogen_type": record.pathogen_type,
        "severity_level": record.severity_level,
        "symptoms_analysis": record.symptoms_analysis,
        "organic_treatment": record.organic_treatment,
        "chemical_treatment": record.chemical_treatment,
        "preventive_actions": record.preventive_actions,
    }

    # 2. Fetch past conversation turns for this diagnostic
    past_messages = (
        db.query(ChatThreadMessage)
        .filter(ChatThreadMessage.diagnostic_id == diagnostic_id)
        .order_by(ChatThreadMessage.timestamp.asc())
        .all()
    )

    history = [
        ChatMessage(role=msg.role, content=msg.content, timestamp=msg.timestamp)
        for msg in past_messages
    ]

    # Combine with client-supplied history if any
    if payload.chat_history:
        history.extend(payload.chat_history)

    # 3. Generate response via Gemini agronomic service
    try:
        followup_response = await gemini_service.generate_agronomic_followup(
            diagnostic_context=diagnostic_context,
            user_question=clean_message,
            history=history,
        )

        # 4. Save both user and model message turns into Supabase
        user_turn = ChatThreadMessage(
            diagnostic_id=diagnostic_id,
            role="user",
            content=clean_message,
        )
        model_turn = ChatThreadMessage(
            diagnostic_id=diagnostic_id,
            role="model",
            content=followup_response.reply,
            timestamp=followup_response.timestamp,
        )

        db.add(user_turn)
        db.add(model_turn)
        db.commit()
        logger.info("Saved conversation turns for diagnostic %s into Supabase.", diagnostic_id)

        return followup_response

    except Exception as exc:
        db.rollback()
        logger.exception("Failed to process agronomic follow-up consultation: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Agronomic chat consultation error: {str(exc)}",
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host=settings.host, port=settings.port, reload=True)
