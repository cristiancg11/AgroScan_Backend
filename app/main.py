"""
AgroScan AI - Andean Crop Phytosanitary Diagnostic & Advisory Engine.
FastAPI REST API powered by Google Gemini 2.5 Flash Vision, Supabase PostgreSQL (SQLAlchemy + PgBouncer),
and JWT-authenticated agronomic consultations.
"""

import logging
import traceback
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
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
    ChatRequestSchema,
    ChatResponseSchema,
    ClearDiagnosticResponse,
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
from app.services import diagnostic_engine, telemetry_service
from app.services.gemini_service import gemini_service, generate_dynamic_agronomic_chat

# Model alias for flexible diagnostic querying
Diagnostic = DiagnosticRecord

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
            conn.execute(
                text(
                    "ALTER TABLE diagnostics ADD COLUMN IF NOT EXISTS part_analyzed VARCHAR(64) DEFAULT 'Hoja / Follaje';"
                )
            )
            conn.execute(
                text(
                    "ALTER TABLE diagnostics ADD COLUMN IF NOT EXISTS dosage_backpack_20l VARCHAR(255) DEFAULT '';"
                )
            )
            # Explicitly verify and create dosage_assessments table
            DosageAssessment.__table__.create(bind=conn, checkfirst=True)
            conn.commit()
        logger.info("Supabase PostgreSQL schema verified successfully.")
    except Exception as exc:
        logger.error("Failed to initialize database tables: %s", exc)

    # Initialize non-blocking gRPC microservice server
    grpc_server_instance = None
    try:
        from app.grpc_server import start_grpc_server

        grpc_server_instance = start_grpc_server(port=50051)
        logger.info("gRPC Telemetry & Evaluation Microservice active on port 50051.")
    except Exception as g_err:
        logger.warning("Could not launch gRPC microservice server: %s", g_err)

    yield

    if grpc_server_instance:
        try:
            grpc_server_instance.stop(grace=1.0)
            logger.info("gRPC Microservice server terminated cleanly.")
        except Exception:
            pass
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

# Configure CORS Middleware for web frontend integration (Vite dev server on 5180 / 5173)
# Explicitly lists dev origins without wildcards to fully support browser credentials (cookies / Auth headers)
origins = [
    "http://localhost:5180",
    "http://127.0.0.1:5180",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]
for custom_origin in settings.cors_origins_list:
    if custom_origin and custom_origin != "*" and custom_origin not in origins:
        origins.append(custom_origin)

logger.info("Configuring CORS middleware with explicit origins: %s", origins)
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
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
    Defensively handles errors to return structured JSON without 500 crashes.
    """
    print(f"--> [AUTH] Incoming registration payload: {payload}")
    try:
        clean_email = payload.email.strip().lower()

        # Check for existing user account
        existing = db.query(User).filter(User.email == clean_email).first()
        if existing:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="An account with this email address is already registered.",
            )

        # Hash password safely with native bcrypt
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
        print(f"[AUTH] User created successfully: {new_user.email} ({new_user.id})")
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

    except HTTPException:
        db.rollback()
        raise
    except Exception as exc:
        db.rollback()
        tb_str = traceback.format_exc()
        print(f"--> [AUTH ERROR] Registration failed with traceback:\n{tb_str}")
        logger.error("Unexpected error in register_farmer: %s\n%s", exc, tb_str)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Registration failed: {str(exc)}",
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
    try:
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

    except HTTPException:
        raise
    except Exception as exc:
        print(f"--> [AUTH ERROR] Login failed: {exc}")
        logger.exception("Unexpected error in login_farmer: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Login authentication failed: {str(exc)}",
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

def _normalize_andean_crop(crop_input_str: str) -> Optional[CropType]:
    """
    Normalizes Andean crop name variants in Spanish and English to the CropType enum.
    Supports Papa, Tomate, Café, Maíz, Uchuva, Mora de Castilla, Arveja, and Frutales.
    """
    clean_input = crop_input_str.strip().lower()
    mapping = {
        "papa": CropType.POTATO,
        "patata": CropType.POTATO,
        "potato": CropType.POTATO,
        "solanum tuberosum": CropType.POTATO,
        "cafe": CropType.COFFEE,
        "café": CropType.COFFEE,
        "coffee": CropType.COFFEE,
        "coffea arabica": CropType.COFFEE,
        "maiz": CropType.CORN,
        "maíz": CropType.CORN,
        "corn": CropType.CORN,
        "zea mays": CropType.CORN,
        "tomate": CropType.TOMATO,
        "tomato": CropType.TOMATO,
        "solanum lycopersicum": CropType.TOMATO,
        "uchuva": CropType.UCHUVA,
        "physalis": CropType.UCHUVA,
        "physalis peruviana": CropType.UCHUVA,
        "cape gooseberry": CropType.UCHUVA,
        "mora": CropType.MORA,
        "mora de castilla": CropType.MORA,
        "rubus glaucus": CropType.MORA,
        "blackberry": CropType.MORA,
        "arveja": CropType.PEA,
        "arvejas": CropType.PEA,
        "pea": CropType.PEA,
        "pisum sativum": CropType.PEA,
        "guisante": CropType.PEA,
        "frutal": CropType.FRUIT_TREES,
        "frutales": CropType.FRUIT_TREES,
        "frutales andinos": CropType.FRUIT_TREES,
        "fruit": CropType.FRUIT_TREES,
        "fruit trees": CropType.FRUIT_TREES,
    }
    for key, member in mapping.items():
        if key == clean_input or key in clean_input:
            return member

    for member in CropType:
        if member.value.lower() == clean_input:
            return member

    return None


@app.post(
    "/api/v1/diagnose",
    response_model=ClearDiagnosticResponse,
    status_code=status.HTTP_200_OK,
    tags=["Diagnostics"],
    summary="Diagnose Andean Crop Image",
    description=(
        "Receives crop photo and plot metadata, performs Gemini 2.5 Flash multimodal vision analysis "
        "structured into 4 ultra-clear farmer sections, associates with user account, and persists in Supabase."
    ),
)
async def diagnose_crop(
    image: UploadFile = File(..., description="Crop leaf, stem, fruit, or plant photo (JPEG/PNG/WebP)"),
    crop_type: str = Form(..., description="Andean crop: Papa, Café, Tomate, Maíz, Uchuva, Mora, Arveja, Frutales"),
    plot_name: str = Form("Lote Principal", description="Farm parcel / lot identifier"),
    notes: Optional[str] = Form(None, description="Optional field observations or symptoms noted by the farmer"),
    current_user: Optional[User] = Depends(get_current_user_optional),
    db: Session = Depends(get_db),
):
    """
    Performs visual phytosanitary analysis using Gemini 2.5 Flash, associates the record
    with the logged-in farmer account (if present), and stores it in Supabase PostgreSQL.
    Guaranteed zero unhandled 500 errors through defensive fallback mechanisms.
    """
    # 1. Validate and normalize crop type against supported Andean crops
    normalized_crop = _normalize_andean_crop(crop_type)
    if not normalized_crop:
        valid_crops = "Papa, Café, Tomate, Maíz, Uchuva, Mora de Castilla, Arveja, Frutales"
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

        # 4. Execute multimodal analysis delegating to OOP DiagnosticEngine
        try:
            diag_dict = await diagnostic_engine.analyze_sample(
                image_bytes=image_bytes,
                crop=normalized_crop.value,
                mime_type=content_type,
                plot_name=plot_name.strip() or "Lote Principal",
                notes=notes.strip() if notes else None,
            )
            diagnostic_result = ClearDiagnosticResponse.model_validate(diag_dict)
        except Exception as ai_err:
            logger.warning("Diagnostic engine analysis caught gracefully: %s", ai_err)
            diagnostic_result = gemini_service._generate_fallback_diagnosis(
                f"diag-{uuid.uuid4().hex[:12]}",
                normalized_crop,
                plot_name.strip() or "Lote Principal",
                datetime.now(timezone.utc),
            )

        user_id = current_user.id if current_user else None
        diagnostic_result.user_id = user_id
        diagnostic_result.plot_name = plot_name.strip() or "Lote Principal"

        # 5. Persist record using OOP AgronomicTelemetryService
        try:
            telemetry_service.record_diagnostic_history(
                user_id=user_id,
                payload=diagnostic_result.model_dump(),
                db=db,
            )
            print(f"[DIAGNOSE] Record persisted successfully: {diagnostic_result.id}")
            logger.info(
                "Successfully persisted clear diagnostic record %s (user: %s) to Supabase PostgreSQL.",
                diagnostic_result.id,
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
            plot_name or "Lote Principal",
            datetime.now(timezone.utc),
        )
        emergency_fallback.user_id = current_user.id if current_user else None
        return emergency_fallback


@app.get(
    "/api/v1/diagnose/history",
    response_model=List[ClearDiagnosticResponse],
    tags=["Diagnostics"],
    summary="Retrieve Persistent Diagnostic History",
    description="Fetches previous phytosanitary diagnostics from Supabase PostgreSQL, filtered by authenticated farmer and sorted newest first.",
)
@app.get(
    "/api/v1/history",
    response_model=List[ClearDiagnosticResponse],
    include_in_schema=False,
)
def get_diagnostic_history(
    limit: int = 50,
    current_user: Optional[User] = Depends(get_current_user_optional),
    db: Session = Depends(get_db),
):
    """
    Queries diagnostic records stored in Supabase PostgreSQL ordered from newest to oldest (created_at DESC).
    If authenticated, returns records specifically associated with the logged-in user.
    """
    try:
        query = db.query(DiagnosticRecord)

        # Filter by logged-in user if available
        if current_user:
            query = query.filter(DiagnosticRecord.user_id == current_user.id)

        records = query.order_by(DiagnosticRecord.timestamp.desc()).limit(limit).all()
        return [rec.to_clear_response() for rec in records]
    except Exception as exc:
        logger.error("Failed to query diagnostic history from Supabase: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch diagnostic history: {str(exc)}",
        )


# ============================================================================
# AGRONOMIC CHAT ENDPOINTS (LIVE GEMINI 2.5 FLASH)
# ============================================================================

@app.post(
    "/api/v1/diagnose/{diagnostic_id}/chat",
    response_model=ChatResponseSchema,
    tags=["Consultation"],
    summary="Interactive Diagnostic-Linked Dynamic Agronomic Chat",
    description="Connects directly to Google Gemini 2.5 Flash to answer any agronomic question dynamically without hardcoded limits.",
)
def chat_diagnostic_endpoint(
    diagnostic_id: str,
    payload: ChatRequestSchema,
    db: Session = Depends(get_db),
):
    """
    Diagnostic-grounded consultation endpoint powered dynamically by Gemini 2.5 Flash.
    Pulls crop and disease context from Supabase PostgreSQL if available.
    """
    crop = payload.crop or "General"
    disease = payload.disease_context or "Ninguno"
    try:
        diag = db.query(Diagnostic).filter(Diagnostic.id == diagnostic_id).first()
        if diag:
            crop = getattr(diag, "crop_name", None) or getattr(diag, "crop_type", None) or crop
            disease = getattr(diag, "common_disease_name", None) or getattr(diag, "disease_name", None) or disease
    except Exception:
        pass

    history_turns = payload.history or []

    try:
        reply_text = generate_dynamic_agronomic_chat(
            user_message=payload.message,
            crop=crop,
            disease_context=disease,
            history=history_turns,
        )
    except ValueError as ve:
        logger.warning("Dynamic chat fallback due to unconfigured GEMINI_API_KEY: %s", ve)
        reply_text = (
            f"Asesoría técnica para {crop} ({disease}): Para la bomba de espalda de 20 litros, "
            f"calibre la dosis recomendada según el ingrediente activo, asegure condiciones secas de al menos 3 horas "
            f"antes de la lluvia y utilice equipo de protección completo. "
            f"(Nota de sistema: configure GEMINI_API_KEY para habilitar razonamiento en tiempo real con Gemini 2.5 Flash)."
        )

    # Persist conversational turns to Supabase if diagnostic exists
    try:
        user_turn = ChatThreadMessage(
            diagnostic_id=diagnostic_id,
            role="user",
            content=payload.message,
        )
        model_turn = ChatThreadMessage(
            diagnostic_id=diagnostic_id,
            role="model",
            content=reply_text,
        )
        db.add(user_turn)
        db.add(model_turn)
        db.commit()
    except Exception:
        db.rollback()

    suggested = [
        "¿Cuál es el intervalo de horas sin lluvia recomendado?",
        "¿Qué equipo de protección personal debo utilizar?",
        "¿Puedo mezclar este tratamiento con biofertilizante foliar?",
    ]

    return {"reply": reply_text, "suggested_followups": suggested}


@app.post(
    "/api/v1/chat",
    response_model=ChatResponseSchema,
    tags=["Consultation"],
    summary="General Dynamic Agronomic Chat",
    description="Direct endpoint connecting to Google Gemini 2.5 Flash to answer any agricultural question openly.",
)
def chat_general_endpoint(payload: ChatRequestSchema):
    """
    Open agronomic chat consultation endpoint powered dynamically by Gemini 2.5 Flash.
    """
    history_turns = payload.history or []

    try:
        reply_text = generate_dynamic_agronomic_chat(
            user_message=payload.message,
            crop=payload.crop or "General",
            disease_context=payload.disease_context or "General",
            history=history_turns,
        )
    except ValueError as ve:
        logger.warning("Dynamic chat fallback due to unconfigured GEMINI_API_KEY: %s", ve)
        reply_text = (
            f"Asesoría técnica para {payload.crop or 'cultivo general'}: "
            f"Para la bomba de espalda de 20 litros, calibre la dosis recomendada y asegure tiempo seco "
            f"antes de la lluvia. (Configure GEMINI_API_KEY para habilitar respuestas directas con Gemini 2.5 Flash)."
        )

    suggested = [
        "¿Cuál es la dosis recomendada por bomba de 20L?",
        "¿Cómo afecta el clima andino la aplicación?",
        "¿Qué alternativas orgánicas existen para este problema?",
    ]

    return {"reply": reply_text, "suggested_followups": suggested}


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
        # 2. Invoke calculation delegating to OOP DiagnosticEngine
        dosage_dict = await diagnostic_engine.calculate_dosage(
            area_sqm=payload.plot_area_sqm,
            severity=record.severity_level,
            crop=record.crop_type,
            disease=record.disease_name,
            tank_capacity=payload.backpack_capacity_liters,
        )
        dosage_result = DosageCalculationResponse.model_validate(dosage_dict)

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
    "/api/v1/telemetry/microclimate",
    tags=["Telemetry & Safety"],
    summary="Evaluate Andean Spraying Microclimate Window",
    description="Evaluates temperature, humidity, and rainfall probability to determine the optimal agricultural spraying window.",
)
def evaluate_microclimate_endpoint(
    humidity: float = 65.0,
    rain_probability: float = 20.0,
):
    """
    Evaluates microclimate spraying conditions using AgronomicTelemetryService.
    """
    advisory = telemetry_service.evaluate_microclimate_window(
        humidity=humidity,
        rain_prob=rain_probability,
    )
    return {
        "humidity": humidity,
        "rain_probability": rain_probability,
        "advisory": advisory,
    }


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
