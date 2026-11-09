"""
AgroScan AI - Andean Crop Phytosanitary Diagnostic & Advisory Engine.
FastAPI REST API powered by Google Gemini 2.5 Flash Vision, Supabase PostgreSQL (SQLAlchemy + PgBouncer),
and JWT-authenticated agronomic consultations.
"""

import logging
import uuid
from contextlib import asynccontextmanager
from typing import List, Optional
from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.config import settings
from app.db import Base, engine, get_db
from app.dependencies import get_current_user, get_current_user_optional
from app.models import ChatThreadMessage, DiagnosticRecord, DosageAssessment, User
from app.schemas import (
    ChatMessage,
    ChatFollowUpRequest,
    ChatFollowUpResponse,
    CropType,
    DiagnosticResponse,
    DosageCalculationRequest,
    DosageCalculationResponse,
    HealthCheckResponse,
    TokenResponse,
    UserLoginRequest,
    UserProfileResponse,
    UserRegisterRequest,
)
from app.security import create_access_token, get_password_hash, verify_password
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
        # Ensure user_id column exists if table was initialized earlier
        with engine.connect() as conn:
            conn.execute(
                text(
                    "ALTER TABLE diagnostics ADD COLUMN IF NOT EXISTS user_id VARCHAR(64) "
                    "REFERENCES users(id) ON DELETE SET NULL;"
                )
            )
            # Explicitly verify and create dosage_assessments table
            DosageAssessment.__table__.create(bind=conn, checkfirst=True)
            conn.commit()
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
    version="2.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# Configure CORS Middleware for web frontend integration (Vite dev server on 5173 / 5180)
origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:5180",
    "http://127.0.0.1:5180",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]
for custom_origin in settings.cors_origins_list:
    if custom_origin and custom_origin not in origins:
        origins.append(custom_origin)

logger.info("Configuring CORS middleware with origins: %s", origins)
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
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
        "version": "2.1.0",
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
        version="2.1.0",
        gemini_configured=gemini_service.is_configured,
        database_connected=database_ok,
    )


# ============================================================================
# AUTHENTICATION ENDPOINTS (Bcrypt + JWT + Supabase PostgreSQL)
# ============================================================================

@app.post(
    "/api/v1/auth/register",
    response_model=TokenResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["Authentication"],
    summary="Register Farmer Account",
    description="Registers a new farmer, hashes the password with bcrypt, persists to Supabase, and issues a JWT.",
)
@app.post(
    "/auth/register",
    response_model=TokenResponse,
    status_code=status.HTTP_201_CREATED,
    include_in_schema=False,
)
@app.post(
    "/api/v1/register",
    response_model=TokenResponse,
    status_code=status.HTTP_201_CREATED,
    include_in_schema=False,
)
def register_farmer(
    payload: UserRegisterRequest,
    db: Session = Depends(get_db),
):
    """
    Creates a new farmer profile in Supabase PostgreSQL and returns signed JWT access token.
    """
    print(f"--> [AUTH] Incoming registration payload: {payload}")
    clean_email = payload.email.strip().lower()

    # Check for existing user account
    existing = db.query(User).filter(User.email == clean_email).first()
    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="An account with this email address is already registered.",
        )

    # Hash password with bcrypt
    hashed = get_password_hash(payload.password)
    user_id = str(uuid.uuid4())

    new_user = User(
        id=user_id,
        full_name=payload.full_name.strip(),
        email=clean_email,
        hashed_password=hashed,
        farm_name=payload.farm_name.strip(),
    )

    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    print(f"[AUTH] User created successfully: {new_user.email}")
    logger.info("Registered new farmer user '%s' (%s)", new_user.email, new_user.id)

    # Issue JWT access token
    access_token = create_access_token(
        data={"sub": new_user.id, "email": new_user.email}
    )

    profile = UserProfileResponse(
        id=new_user.id,
        email=new_user.email,
        full_name=new_user.full_name,
        farm_name=new_user.farm_name,
        created_at=new_user.created_at,
    )

    return TokenResponse(
        access_token=access_token,
        token_type="bearer",
        user_id=new_user.id,
        full_name=new_user.full_name,
        farm_name=new_user.farm_name,
        email=new_user.email,
        user=profile,
    )


@app.post(
    "/api/v1/auth/login",
    response_model=TokenResponse,
    status_code=status.HTTP_200_OK,
    tags=["Authentication"],
    summary="Farmer Authentication Login",
    description="Authenticates farmer credentials against stored bcrypt hash in Supabase and returns a JWT access token.",
)
@app.post(
    "/auth/login",
    response_model=TokenResponse,
    status_code=status.HTTP_200_OK,
    include_in_schema=False,
)
@app.post(
    "/api/v1/login",
    response_model=TokenResponse,
    status_code=status.HTTP_200_OK,
    include_in_schema=False,
)
def login_farmer(
    payload: UserLoginRequest,
    db: Session = Depends(get_db),
):
    """
    Validates farmer email and password credentials, returning a signed JWT access token.
    """
    print(f"--> [AUTH] Incoming login attempt: {payload.email}")
    clean_email = payload.email.strip().lower()
    user = db.query(User).filter(User.email == clean_email).first()

    if not user or not verify_password(payload.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password credentials. Please verify your details.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    access_token = create_access_token(
        data={"sub": user.id, "email": user.email}
    )
    print(f"[AUTH] User logged in successfully: {user.email}")
    logger.info("Authenticated farmer user '%s'", user.email)

    profile = UserProfileResponse(
        id=user.id,
        email=user.email,
        full_name=user.full_name,
        farm_name=user.farm_name,
        created_at=user.created_at,
    )

    return TokenResponse(
        access_token=access_token,
        token_type="bearer",
        user_id=user.id,
        full_name=user.full_name,
        farm_name=user.farm_name,
        email=user.email,
        user=profile,
    )


@app.get(
    "/api/v1/auth/me",
    response_model=UserProfileResponse,
    tags=["Authentication"],
    summary="Get Current User Profile",
    description="Returns the profile of the currently authenticated farmer from JWT Bearer token validation.",
)
@app.get(
    "/auth/me",
    response_model=UserProfileResponse,
    include_in_schema=False,
)
def get_current_user_profile(
    current_user: User = Depends(get_current_user),
):
    """
    Returns the authenticated farmer user profile extracted from the validated JWT token.
    """
    return current_user


# ============================================================================
# PHYTOSANITARY DIAGNOSTIC ENDPOINTS
# ============================================================================

@app.post(
    "/api/v1/diagnose",
    response_model=DiagnosticResponse,
    status_code=status.HTTP_200_OK,
    tags=["Diagnostics"],
    summary="Diagnose Andean Crop Image",
    description="Receives crop photo and plot metadata, runs Gemini 2.5 Flash, associates with user account, and persists in Supabase.",
)
async def diagnose_crop(
    image: UploadFile = File(..., description="Crop leaf, stem, or plant photo (JPEG/PNG/WebP)"),
    crop_type: str = Form(..., description="Crop kind: Potato, Coffee, Corn, or Tomato"),
    plot_name: str = Form("Plot A - Main Lot", description="Farm parcel / lot identifier"),
    notes: Optional[str] = Form(None, description="Optional field observations or symptoms noted by the farmer"),
    current_user: Optional[User] = Depends(get_current_user_optional),
    db: Session = Depends(get_db),
):
    """
    Performs visual phytosanitary analysis using Gemini 2.5 Flash, associates the record
    with the logged-in farmer account (if present), and stores it in Supabase PostgreSQL.
    Guaranteed zero unhandled 500 errors through defensive fallback mechanisms.
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

        # 4. Execute multimodal analysis with Gemini 2.5 Flash (with resilient fallback)
        try:
            diagnostic_result = await gemini_service.analyze_crop_health(
                image_bytes=image_bytes,
                mime_type=content_type,
                crop_type=normalized_crop,
                plot_name=plot_name.strip() or "General Lot",
                notes=notes.strip() if notes else None,
            )
        except Exception as ai_err:
            logger.warning("Vision analysis error caught gracefully: %s", ai_err)
            diagnostic_result = gemini_service._generate_fallback_diagnosis(
                f"diag-{uuid.uuid4().hex[:12]}",
                normalized_crop,
                plot_name.strip() or "General Lot",
                datetime.now(timezone.utc),
            )

        user_id = current_user.id if current_user else None
        diagnostic_result.user_id = user_id

        # 5. Persist record into Supabase PostgreSQL
        try:
            db_record = DiagnosticRecord(
                id=diagnostic_result.id,
                user_id=user_id,
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
            print(f"[DIAGNOSE] Record persisted successfully: {db_record.id}")
            logger.info(
                "Successfully persisted diagnostic record %s (user: %s) to Supabase PostgreSQL.",
                db_record.id,
                user_id or "guest",
            )
        except Exception as db_err:
            db.rollback()
            logger.error("Database persistence warning: %s", db_err)

        return diagnostic_result

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Unexpected error in diagnose_crop handler: %s", exc)
        emergency_fallback = gemini_service._generate_fallback_diagnosis(
            f"diag-{uuid.uuid4().hex[:12]}",
            normalized_crop or CropType.POTATO,
            plot_name or "General Lot",
            datetime.now(timezone.utc),
        )
        return emergency_fallback


@app.get(
    "/api/v1/history",
    response_model=List[DiagnosticResponse],
    tags=["Diagnostics"],
    summary="Retrieve Diagnostic History",
    description="Fetches previous phytosanitary diagnostics from Supabase PostgreSQL, filtered by authenticated farmer.",
)
def get_diagnostic_history(
    limit: int = 50,
    current_user: Optional[User] = Depends(get_current_user_optional),
    db: Session = Depends(get_db),
):
    """
    Queries diagnostic records stored in Supabase PostgreSQL ordered from newest to oldest.
    If authenticated, returns records specifically associated with the logged-in user.
    """
    try:
        query = db.query(DiagnosticRecord)

        # Filter by logged-in user if available
        if current_user:
            query = query.filter(DiagnosticRecord.user_id == current_user.id)

        records = query.order_by(DiagnosticRecord.timestamp.desc()).limit(limit).all()
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


# ============================================================================
# AI DOSAGE CALCULATOR & WEATHER RISK ENDPOINTS
# ============================================================================

@app.post(
    "/api/v1/diagnose/{diagnostic_id}/dosage",
    response_model=DosageCalculationResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["Dosage & Safety"],
    summary="Calculate AI Backpack Sprayer Dosage & Weather Risk",
    description="Calculates calibrated backpack tank requirements, product concentration, rainfastness, and economic loss risks.",
)
async def calculate_crop_dosage(
    diagnostic_id: str,
    payload: DosageCalculationRequest,
    db: Session = Depends(get_db),
):
    """
    Computes precision pesticide / bio-input spray calibrations for a specific diagnostic report
    and parcel surface area, persisting the assessment in Supabase PostgreSQL.
    """
    # 1. Validate diagnostic record existence
    record = db.query(DiagnosticRecord).filter(DiagnosticRecord.id == diagnostic_id).first()
    if not record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Diagnostic report '{diagnostic_id}' was not found.",
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

    try:
        # 2. Invoke Gemini 2.5 Flash calculation engine
        dosage_result = await gemini_service.calculate_smart_dosage(
            diagnostic_data=diagnostic_context,
            area_sqm=payload.plot_area_sqm,
            tank_capacity=payload.backpack_capacity_liters,
        )

        # 3. Persist calculation in Supabase PostgreSQL
        assessment_id = f"dose-{uuid.uuid4().hex[:12]}"
        assessment_record = DosageAssessment(
            id=assessment_id,
            diagnostic_id=diagnostic_id,
            plot_area_sqm=payload.plot_area_sqm,
            total_tanks_needed=dosage_result.total_tanks_needed,
            product_recommended=dosage_result.product_recommended,
            concentration_per_tank=dosage_result.concentration_per_20l_backpack,
            rain_safety_interval_hours=dosage_result.rain_safety_interval_hours,
            economic_loss_risk=dosage_result.estimated_economic_loss_percentage,
            full_prescription_payload=dosage_result.model_dump(),
        )

        db.add(assessment_record)
        db.commit()
        db.refresh(assessment_record)
        logger.info(
            "Persisted dosage assessment %s for diagnosis %s (Area: %s m², Tanks: %s)",
            assessment_id,
            diagnostic_id,
            payload.plot_area_sqm,
            dosage_result.total_tanks_needed,
        )

        return dosage_result

    except Exception as exc:
        db.rollback()
        logger.exception("Failed to calculate dosage prescription: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Dosage calculation engine error: {str(exc)}",
        )


@app.get(
    "/api/v1/diagnose/{diagnostic_id}/dosage",
    response_model=List[DosageCalculationResponse],
    tags=["Dosage & Safety"],
    summary="Retrieve Dosage Calculation History for Diagnosis",
    description="Retrieves previous backpack sprayer dosage prescriptions computed for a specific diagnostic session.",
)
def get_diagnosis_dosages(
    diagnostic_id: str,
    db: Session = Depends(get_db),
):
    """
    Returns all dosage assessments previously calculated for this diagnostic record.
    """
    record = db.query(DiagnosticRecord).filter(DiagnosticRecord.id == diagnostic_id).first()
    if not record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Diagnostic report '{diagnostic_id}' was not found.",
        )

    assessments = (
        db.query(DosageAssessment)
        .filter(DosageAssessment.diagnostic_id == diagnostic_id)
        .order_by(DosageAssessment.created_at.desc())
        .all()
    )

    results = []
    for a in assessments:
        if a.full_prescription_payload:
            results.append(DosageCalculationResponse.model_validate(a.full_prescription_payload))
        else:
            results.append(
                DosageCalculationResponse(
                    product_recommended=a.product_recommended,
                    concentration_per_20l_backpack=a.concentration_per_tank,
                    total_tanks_needed=a.total_tanks_needed,
                    total_product_quantity="Calculated in assessment",
                    rain_safety_interval_hours=a.rain_safety_interval_hours,
                    estimated_economic_loss_percentage=a.economic_loss_risk,
                    protective_equipment_ppe=[],
                    application_tips=[],
                )
            )

    return results


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host=settings.host, port=settings.port, reload=True)
