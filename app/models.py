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
    part_analyzed = Column(String(64), nullable=True, default="Hoja / Follaje")
    dosage_backpack_20l = Column(String(255), nullable=True, default="")
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

    def to_clear_response(self):
        """
        Transforms relational DiagnosticRecord into a farmer-friendly ClearDiagnosticResponse.
        Ensures backwards compatibility with records created under legacy formats.
        """
        from app.schemas import ClearDiagnosticResponse

        conf_pct = (
            int(round(self.confidence_score * 100))
            if self.confidence_score <= 1.0
            else int(self.confidence_score)
        )

        sev_map = {
            "LOW": "Baja",
            "MODERATE": "Moderada",
            "CRITICAL": "Crítica",
            "BAJA": "Baja",
            "MODERADA": "Moderada",
            "CRÍTICA": "Crítica",
            "CRITICA": "Crítica",
        }
        mapped_sev = sev_map.get(str(self.severity_level).upper(), str(self.severity_level))

        if isinstance(self.organic_treatment, list):
            org_str = " ".join(self.organic_treatment) if self.organic_treatment else "Aplicar biofertilizante foliar orgánico."
        else:
            org_str = str(self.organic_treatment)

        if isinstance(self.chemical_treatment, list):
            chem_str = " ".join(self.chemical_treatment) if self.chemical_treatment else "Consulte con el agrónomo de confianza para el ingrediente activo."
        else:
            chem_str = str(self.chemical_treatment)

        imm_act = (
            list(self.preventive_actions)
            if isinstance(self.preventive_actions, list)
            else [str(self.preventive_actions)]
        )
        if not imm_act or len(imm_act) == 0:
            imm_act = [
                "Podar y retirar de inmediato el material vegetal infectado.",
                "Mejorar drenajes y aireación en el lote.",
                "Fumigar en horas tempranas de la mañana con equipo de protección.",
            ]
        while len(imm_act) < 3:
            imm_act.append("Monitorear el avance del cultivo cada 3 días.")

        dosage_val = self.dosage_backpack_20l or "50g o 30cc por bomba de 20L"

        return ClearDiagnosticResponse(
            id=self.id,
            crop_name=self.crop_type,
            part_analyzed=self.part_analyzed or "Hoja / Follaje",
            common_disease_name=self.disease_name,
            scientific_name=self.scientific_name,
            severity_level=mapped_sev,
            confidence_percentage=conf_pct,
            summary_what_happened=self.symptoms_analysis,
            immediate_actions=imm_act[:3],
            organic_solution=org_str,
            chemical_solution=chem_str,
            dosage_backpack_20l=dosage_val,
            created_at=self.timestamp.isoformat() if self.timestamp else utc_now().isoformat(),
            user_id=self.user_id,
            plot_name=self.plot_name,
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

