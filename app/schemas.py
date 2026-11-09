"""
Pydantic schemas and enums for AgroScan AI backend.
Provides data structures for API request/response validation,
authentication, and structured output extraction with Google Gemini 2.5 Flash.
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, EmailStr, Field


class CropType(str, Enum):
    """
    Supported Andean crops grown predominantly in Nariño, Colombia.
    Expanded to support potatoes, coffee, corn, tomatoes, uchuva, mora de Castilla, peas, and fruit trees.
    """
    POTATO = "Potato"
    COFFEE = "Coffee"
    CORN = "Corn"
    TOMATO = "Tomato"
    UCHUVA = "Uchuva"
    MORA = "Mora de Castilla"
    PEA = "Arveja"
    FRUIT_TREES = "Frutales"


class PathogenType(str, Enum):
    """
    Phytopathological pathogen classification categories.
    """
    FUNGUS = "FUNGUS"
    BACTERIA = "BACTERIA"
    VIRUS = "VIRUS"
    PEST = "PEST"
    HEALTHY = "HEALTHY"


class SeverityLevel(str, Enum):
    """
    Agronomic infestation or infection severity grading.
    """
    LOW = "LOW"
    MODERATE = "MODERATE"
    CRITICAL = "CRITICAL"


# ============================================================================
# AUTHENTICATION SCHEMAS
# ============================================================================

class UserRegisterRequest(BaseModel):
    """
    Payload for new farmer account registration.
    """
    email: str = Field(..., description="Farmer or agronomist email address")
    password: str = Field(..., min_length=6, description="Account password (min 6 characters)")
    full_name: str = Field(..., min_length=2, description="Farmer full name")
    farm_name: str = Field(..., min_length=2, description="Farm or parcel designation")


class UserLoginRequest(BaseModel):
    """
    Payload for farmer authentication.
    """
    email: str = Field(..., description="Registered email address")
    password: str = Field(..., description="Account password")


class TokenResponse(BaseModel):
    """
    JWT token response payload returning session token and user profile.
    """
    access_token: str = Field(..., description="Signed JWT Bearer access token")
    token_type: str = Field(default="bearer", description="Token authentication scheme")
    user_id: str = Field(..., description="Authenticated user UUID")
    full_name: str = Field(..., description="Farmer full name")
    farm_name: str = Field(..., description="Farm or parcel designation")
    email: Optional[str] = Field(default=None, description="Registered email address")
    user: Optional[UserProfileResponse] = Field(
        default=None,
        description="Nested user profile object for instant frontend state initialization"
    )


class UserProfileResponse(BaseModel):
    """
    Farmer profile response schema.
    """
    model_config = ConfigDict(from_attributes=True)

    id: str = Field(..., description="User unique identifier")
    email: str = Field(..., description="User email address")
    full_name: str = Field(..., description="Farmer full name")
    farm_name: str = Field(..., description="Farm or parcel designation")
    created_at: Optional[datetime] = Field(default=None, description="Account registration timestamp")


# ============================================================================
# DIAGNOSTICS SCHEMAS
# ============================================================================

class ClearDiagnosticResponse(BaseModel):
    """
    Ultra-clear, farmer-friendly phytosanitary diagnostic response schema.
    Structured into 4 direct sections for smallholders in the Andean region:
    1. ¿Qué tiene la planta? (crop_name, part_analyzed, common_disease_name, scientific_name)
    2. ¿Qué tan grave es? (severity_level, confidence_percentage, summary_what_happened)
    3. ¿Qué se debe hacer de inmediato? (immediate_actions)
    4. Solución recomendada (organic_solution, chemical_solution, dosage_backpack_20l)
    """
    model_config = ConfigDict(from_attributes=True, populate_by_name=True)

    id: str = Field(..., description="Unique diagnostic session identifier")
    crop_name: str = Field(..., description="Identified Andean crop name (Papa, Tomate, Café, etc.)")
    part_analyzed: str = Field(
        default="Hoja / Follaje",
        description="Analyzed plant organ: 'Hoja / Follaje' or 'Fruto / Tubérculo'"
    )
    common_disease_name: str = Field(..., description="Common agricultural disease, pest or deficiency name")
    scientific_name: str = Field(..., description="Latin binomial scientific name of pathogen/pest")
    severity_level: str = Field(
        ...,
        description="Infestation or infection severity classification: 'Baja', 'Moderada', or 'Crítica'"
    )
    confidence_percentage: int = Field(
        ...,
        ge=0,
        le=100,
        description="AI diagnostic confidence percentage (0 to 100)"
    )
    summary_what_happened: str = Field(
        ...,
        description="Two direct, jargon-free sentences explaining the damage to the farmer"
    )
    immediate_actions: List[str] = Field(
        ...,
        description="Three concrete, immediate steps to contain the damage (Pasos 1, 2 y 3)"
    )
    organic_solution: str = Field(
        ...,
        description="Accessible ecological or biological control solution"
    )
    chemical_solution: str = Field(
        ...,
        description="Targeted chemical active ingredient solution with safety instructions"
    )
    dosage_backpack_20l: str = Field(
        ...,
        description="Calibrated recipe dosage per standard 20-liter agricultural backpack sprayer"
    )
    created_at: str = Field(
        ...,
        description="ISO 8601 creation timestamp string"
    )
    user_id: Optional[str] = Field(
        default=None,
        description="Associated farmer user UUID"
    )
    plot_name: Optional[str] = Field(
        default="Lote Principal",
        description="Parcel or lot identifier"
    )


class DiagnosticResponse(BaseModel):
    """
    Full phytosanitary diagnostic report schema generated by Gemini 2.5 Flash
    and persisted in Supabase PostgreSQL.
    """
    model_config = ConfigDict(from_attributes=True)

    id: str = Field(
        ...,
        description="Unique identifier for the diagnostic record (UUID string)"
    )
    user_id: Optional[str] = Field(
        default=None,
        description="Owner farmer user identifier"
    )
    crop_type: CropType = Field(
        ...,
        description="Type of Andean crop evaluated (Potato, Coffee, Corn, Tomato)"
    )
    plot_name: str = Field(
        ...,
        description="Farmer plot or lot identifier (e.g., 'Plot A - Upper Terrace')"
    )
    disease_name: str = Field(
        ...,
        description="Common identified disease, pest, or physiological condition name"
    )
    scientific_name: str = Field(
        ...,
        description="Binomial Latin name of the pathogen or pest organism"
    )
    pathogen_type: PathogenType = Field(
        ...,
        description="Categorical pathogen classification"
    )
    confidence_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Diagnostic confidence assessment score between 0.0 and 1.0"
    )
    severity_level: SeverityLevel = Field(
        ...,
        description="Infestation or infection severity classification"
    )
    symptoms_analysis: str = Field(
        ...,
        description="Detailed agronomic analysis of foliar and stem visual symptoms"
    )
    organic_treatment: List[str] = Field(
        default_factory=list,
        description="Organic, biological, and eco-friendly treatment actions"
    )
    chemical_treatment: List[str] = Field(
        default_factory=list,
        description="Targeted chemical active ingredients, dosages, and safety precautions"
    )
    preventive_actions: List[str] = Field(
        default_factory=list,
        description="Cultural practices, soil drainage, and preventive farm sanitation steps"
    )
    disclaimer: str = Field(
        default="AI-assisted phytosanitary diagnostic for agricultural guidance. Verify with a registered agronomist.",
        description="Agronomic advisory disclaimer"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp of the diagnostic generation"
    )


class ChatMessage(BaseModel):
    """
    Single conversational turn in an agronomic technical consultation thread.
    """
    model_config = ConfigDict(from_attributes=True)

    role: str = Field(..., description="Role of the speaker: 'user' or 'model'")
    content: str = Field(..., description="Message text content")
    timestamp: Optional[datetime] = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Timestamp of the message turn"
    )


class ChatFollowUpRequest(BaseModel):
    """
    Payload for interactive follow-up technical consultation grounded on a diagnostic record.
    """
    message: str = Field(
        ...,
        min_length=1,
        description="User question regarding treatments, dosages, or field safety"
    )
    chat_history: Optional[List[ChatMessage]] = Field(
        default=None,
        description="Prior conversational turns for dialogue context continuity"
    )


class ChatFollowUpResponse(BaseModel):
    """
    Agronomic follow-up consultation response generated by Gemini 2.5 Flash.
    """
    reply: str = Field(
        ...,
        description="Field agronomist response with dosage and application instructions"
    )
    suggested_followups: List[str] = Field(
        default_factory=list,
        description="Two or three recommended follow-up questions for the farmer"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp of response generation"
    )


class ChatRequestSchema(BaseModel):
    """
    Dynamic live agronomic chat request payload.
    """
    message: str = Field(..., description="Farmer question or technical agronomic consultation")
    history: Optional[List[Dict[str, Any]]] = Field(default_factory=list, description="Prior conversational turns")
    crop: Optional[str] = Field(default="General", description="Target Andean crop or general context")
    disease_context: Optional[str] = Field(default="Ninguno", description="Diagnosed phytopathological condition")


class ChatResponseSchema(BaseModel):
    """
    Dynamic live agronomic chat response generated directly by Gemini 2.5 Flash.
    """
    reply: str = Field(..., description="Direct, helpful, practical agronomic advice for the farmer")


class HealthCheckResponse(BaseModel):
    """
    Health check response model reporting system status.
    """
    status: str
    service: str
    version: str
    gemini_configured: bool
    database_connected: bool


# ============================================================================
# DOSAGE & WEATHER RISK SCHEMAS
# ============================================================================

class DosageCalculationRequest(BaseModel):
    """
    Payload for AI backpack sprayer dosage calculation and weather risk assessment.
    """
    diagnostic_id: str = Field(..., description="Target diagnostic session UUID")
    plot_area_sqm: float = Field(..., gt=0.0, description="Plot area in square meters (e.g., 2500 for 0.25 ha)")
    backpack_capacity_liters: float = Field(default=20.0, gt=0.0, description="Backpack sprayer tank volume in liters")
    application_method: str = Field(
        default="manual_backpack",
        description="Sprayer mechanism: 'manual_backpack' or 'motorized_sprayer'"
    )


class DosageCalculationResponse(BaseModel):
    """
    Structured prescription response model for precision Andean phytosanitary spraying.
    """
    model_config = ConfigDict(from_attributes=True)

    product_recommended: str = Field(
        ...,
        description="Selected chemical active ingredient or biological bio-input formulation"
    )
    concentration_per_20l_backpack: str = Field(
        ...,
        description="Calibrated dilution dose per 20-liter tank (e.g., '50g or 30cc per 20L tank')"
    )
    total_tanks_needed: float = Field(
        ...,
        description="Total number of 20L backpack loads required to cover the specified parcel area"
    )
    total_product_quantity: str = Field(
        ...,
        description="Total quantity of commercial product required for the full plot treatment"
    )
    rain_safety_interval_hours: int = Field(
        ...,
        description="Minimum rainfast drying window in hours to prevent wash-off by Andean rainfall"
    )
    estimated_economic_loss_percentage: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Estimated harvest yield economic loss (0-100%) if the disease remains untreated within 5 days"
    )
    protective_equipment_ppe: List[str] = Field(
        default_factory=list,
        description="Mandatory personal protective equipment items required during preparation and application"
    )
    application_tips: List[str] = Field(
        default_factory=list,
        description="Highland agronomic application tips (sprayer pressure, nozzle type, time of day)"
    )

