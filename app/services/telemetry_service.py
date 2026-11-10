"""
Agronomic Telemetry Service providing microclimate spraying window evaluation
and persistent diagnostic history tracking.
Follows Object-Oriented Architecture (OOP) principles.
"""

import logging
from typing import Any, Dict, Optional
from sqlalchemy.orm import Session

from app.models import DiagnosticRecord, utc_now

logger = logging.getLogger("agroscan.telemetry")


class AgronomicTelemetryService:
    """
    Service responsible for microclimatic spraying window assessments
    and historical diagnostic record telemetry tracking.
    """

    def evaluate_microclimate_window(self, humidity: float, rain_prob: float) -> str:
        """
        Evaluates Andean mountain microclimatic conditions to determine spray safety window.

        :param humidity: Relative atmospheric humidity percentage (0-100%).
        :param rain_prob: Precipitation probability percentage within next 4 hours (0-100%).
        :return: Spraying window advisory rating ('OPTIMAL', 'CAUTION', or 'UNFAVORABLE').
        """
        if rain_prob > 60.0 or humidity > 85.0:
            return (
                "UNFAVORABLE: High risk of rainfall wash-off or fungal spore condensation. "
                "Postpone foliar applications until morning drying."
            )
        elif rain_prob <= 30.0 and 45.0 <= humidity <= 75.0:
            return (
                "OPTIMAL: Ideal microclimate window. Favorable leaf cuticle absorption "
                "with low drift and zero immediate rain risk."
            )
        else:
            return (
                "CAUTION: Moderate humidity or rain probability. "
                "Include organosilicone adherent surfactant and monitor cloud cover closely."
            )

    def record_diagnostic_history(
        self,
        user_id: Optional[str],
        payload: Dict[str, Any],
        db: Optional[Session] = None,
    ) -> bool:
        """
        Records or logs a diagnostic evaluation entry in persistent storage.

        :param user_id: Farmer or user identifier.
        :param payload: Diagnostic data dictionary.
        :param db: SQLAlchemy Session (optional).
        :return: Boolean indicating whether telemetry was successfully recorded.
        """
        try:
            diag_id = payload.get("id")
            if not diag_id:
                logger.warning("Telemetry payload missing unique record id.")
                return False

            if db:
                existing = db.query(DiagnosticRecord).filter(DiagnosticRecord.id == diag_id).first()
                if not existing:
                    conf_raw = payload.get("confidence_percentage")
                    if conf_raw is not None:
                        conf_score = float(conf_raw) / 100.0 if conf_raw > 1.0 else float(conf_raw)
                    else:
                        conf_score = float(payload.get("confidence_score", 0.8))

                    org_sol = payload.get("organic_solution")
                    if isinstance(org_sol, list):
                        org_list = org_sol
                    elif isinstance(org_sol, str):
                        org_list = [org_sol]
                    else:
                        org_list = payload.get("organic_treatment", [])

                    chem_sol = payload.get("chemical_solution")
                    if isinstance(chem_sol, list):
                        chem_list = chem_sol
                    elif isinstance(chem_sol, str):
                        chem_list = [chem_sol]
                    else:
                        chem_list = payload.get("chemical_treatment", [])

                    imm_act = payload.get("immediate_actions") or payload.get("preventive_actions", [])
                    if isinstance(imm_act, str):
                        imm_act = [imm_act]

                    new_record = DiagnosticRecord(
                        id=diag_id,
                        user_id=user_id,
                        crop_type=payload.get("crop_name") or payload.get("crop_type", "General"),
                        part_analyzed=payload.get("part_analyzed", "Hoja / Follaje"),
                        plot_name=payload.get("plot_name", "Main Lot"),
                        disease_name=payload.get("common_disease_name") or payload.get("disease_name", "Unknown"),
                        scientific_name=payload.get("scientific_name", "Unknown"),
                        pathogen_type=payload.get("pathogen_type", "FUNGUS"),
                        confidence_score=conf_score,
                        severity_level=str(payload.get("severity_level", "MODERATE")),
                        symptoms_analysis=payload.get("summary_what_happened") or payload.get("symptoms_analysis", ""),
                        organic_treatment=org_list,
                        chemical_treatment=chem_list,
                        preventive_actions=imm_act,
                        dosage_backpack_20l=payload.get("dosage_backpack_20l", ""),
                        timestamp=utc_now(),
                    )
                    db.add(new_record)
                    db.commit()
                    logger.info("Recorded diagnostic telemetry %s for user %s.", diag_id, user_id or "guest")
            else:
                logger.info(
                    "Logged diagnostic telemetry %s for user %s without active DB session.",
                    diag_id,
                    user_id or "guest",
                )

            return True
        except Exception as exc:
            logger.error("Failed to record diagnostic telemetry: %s", exc)
            if db:
                db.rollback()
            return False


# Default singleton instance for dependency injection
telemetry_service = AgronomicTelemetryService()
