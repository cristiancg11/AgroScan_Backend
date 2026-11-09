"""
SQLAlchemy ORM models for Supabase PostgreSQL.
Defines tables for users, phytosanitary diagnostics, and agronomic chat threads.
"""

from datetime import datetime, timezone
from sqlalchemy import (
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    JSON,
    String,
    Text,
)
from sqlalchemy.orm import relationship

from app.db import Base


def utc_now() -> datetime:
    """Returns current datetime in UTC timezone."""
    return datetime.now(timezone.utc)


class User(Base):
    """
    User model representing registered Andean farmers and agronomists.
    Stored in the 'users' table in Supabase PostgreSQL.
    """
    __tablename__ = "users"

    id = Column(String(64), primary_key=True, index=True)
    full_name = Column(String(255), nullable=False)
    email = Column(String(255), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    farm_name = Column(String(255), nullable=False)
    created_at = Column(DateTime(timezone=True), default=utc_now, nullable=False)

    # One-to-many relationship with diagnostics
    diagnostics = relationship(
        "DiagnosticRecord",
        back_populates="user",
        cascade="all, delete-orphan",
        order_by="DiagnosticRecord.timestamp.desc()",
    )


class DiagnosticRecord(Base):
    """
    DiagnosticRecord model representing crop phytosanitary evaluation results.
    Stored in the 'diagnostics' table in Supabase PostgreSQL.
    """
    __tablename__ = "diagnostics"

    id = Column(String(64), primary_key=True, index=True)
    user_id = Column(
        String(64),
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    crop_type = Column(String(64), nullable=False)
    plot_name = Column(String(128), nullable=False, index=True)
    disease_name = Column(String(255), nullable=False)
    scientific_name = Column(String(255), nullable=False)
    pathogen_type = Column(String(64), nullable=False)
    confidence_score = Column(Float, nullable=False, default=0.0)
    severity_level = Column(String(32), nullable=False)
    symptoms_analysis = Column(Text, nullable=False)
    organic_treatment = Column(JSON, nullable=False, default=list)
    chemical_treatment = Column(JSON, nullable=False, default=list)
    preventive_actions = Column(JSON, nullable=False, default=list)
    disclaimer = Column(String(500), nullable=False, default="")
    timestamp = Column(DateTime(timezone=True), default=utc_now, nullable=False)

    # Relationships
    user = relationship("User", back_populates="diagnostics")
    chat_messages = relationship(
        "ChatThreadMessage",
        back_populates="diagnostic",
        cascade="all, delete-orphan",
        order_by="ChatThreadMessage.timestamp.asc()",
    )
    dosage_assessments = relationship(
        "DosageAssessment",
        back_populates="diagnostic",
        cascade="all, delete-orphan",
        order_by="DosageAssessment.created_at.desc()",
    )


class ChatThreadMessage(Base):
    """
    ChatThreadMessage model storing interactive technical follow-up exchanges
    grounded on a specific diagnostic evaluation.
    """
    __tablename__ = "chat_thread_messages"

    id = Column(Integer, primary_key=True, autoincrement=True, index=True)
    diagnostic_id = Column(
        String(64),
        ForeignKey("diagnostics.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    role = Column(String(32), nullable=False)  # "user" | "model"
    content = Column(Text, nullable=False)
    timestamp = Column(DateTime(timezone=True), default=utc_now, nullable=False)

    diagnostic = relationship("DiagnosticRecord", back_populates="chat_messages")


class DosageAssessment(Base):
    """
    DosageAssessment model storing AI-computed backpack sprayer dosage calibrations
    and Andean weather rainfastness risk assessments.
    """
    __tablename__ = "dosage_assessments"

    id = Column(String(64), primary_key=True, index=True)
    diagnostic_id = Column(
        String(64),
        ForeignKey("diagnostics.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    plot_area_sqm = Column(Float, nullable=False)
    total_tanks_needed = Column(Float, nullable=False)
    product_recommended = Column(String(255), nullable=False)
    concentration_per_tank = Column(String(255), nullable=False)
    rain_safety_interval_hours = Column(Integer, nullable=False)
    economic_loss_risk = Column(Float, nullable=False)
    full_prescription_payload = Column(JSON, nullable=False, default=dict)
    created_at = Column(DateTime(timezone=True), default=utc_now, nullable=False)

    diagnostic = relationship("DiagnosticRecord", back_populates="dosage_assessments")

