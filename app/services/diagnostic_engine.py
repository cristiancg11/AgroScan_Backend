"""
Diagnostic engine abstraction and concrete Gemini 2.5 Flash implementation.
Follows Object-Oriented Architecture (OOP) and Service Layer design patterns.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from app.schemas import CropType
from app.services.gemini_service import gemini_service

logger = logging.getLogger("agroscan.diagnostic_engine")


class BaseDiagnosticEngine(ABC):
    """
    Abstract Base Class defining the contract for phytosanitary diagnostic
    and precision agricultural dosage calculation engines.
    """

    @abstractmethod
    async def analyze_sample(
        self,
        image_bytes: bytes,
        crop: str,
        mime_type: str = "image/jpeg",
        plot_name: str = "Main Plot",
        notes: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Analyzes an image sample of a crop organ and produces a phytosanitary assessment dictionary.

        :param image_bytes: Raw binary image payload.
        :param crop: Target crop designation string.
        :param mime_type: Image MIME type (e.g., 'image/jpeg', 'image/png').
        :param plot_name: Farm lot or plot designation.
        :param notes: Optional observations from the field.
        :return: Standardized diagnostic dictionary.
        """
        pass

    @abstractmethod
    async def calculate_dosage(
        self,
        area_sqm: float,
        severity: str = "MODERATE",
        crop: str = "Potato",
        disease: str = "Late Blight",
        tank_capacity: float = 20.0,
    ) -> Dict[str, Any]:
        """
        Computes backpack sprayer tank counts, product quantity, and rainfastness intervals.

        :param area_sqm: Parcel surface area in square meters.
        :param severity: Agronomic infection severity grading.
        :param crop: Target Andean crop.
        :param disease: Identified pathogen or disease condition.
        :param tank_capacity: Tank capacity in liters (default 20L).
        :return: Standardized prescription dictionary.
        """
        pass


class GeminiDiagnosticEngine(BaseDiagnosticEngine):
    """
    Concrete implementation of BaseDiagnosticEngine utilizing Google Gemini 2.5 Flash
    multimodal vision and precision agronomic calibrations.
    """

    def __init__(self, service=None):
        self._service = service or gemini_service

    async def analyze_sample(
        self,
        image_bytes: bytes,
        crop: str,
        mime_type: str = "image/jpeg",
        plot_name: str = "Main Plot",
        notes: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Executes multimodal visual diagnosis on the crop leaf/fruit sample.
        """
        crop_clean = crop.strip().lower()
        matched_crop = CropType.POTATO
        for member in CropType:
            if member.value.lower() == crop_clean or member.value.lower() in crop_clean:
                matched_crop = member
                break

        result = await self._service.analyze_crop_health(
            image_bytes=image_bytes,
            mime_type=mime_type,
            crop_type=matched_crop,
            plot_name=plot_name,
            notes=notes,
        )
        return result.model_dump()

    async def calculate_dosage(
        self,
        area_sqm: float,
        severity: str = "MODERATE",
        crop: str = "Potato",
        disease: str = "Late Blight",
        tank_capacity: float = 20.0,
    ) -> Dict[str, Any]:
        """
        Calculates calibrated sprayer requirements and rain risk assessment.
        """
        diagnostic_data = {
            "crop_type": crop,
            "disease_name": disease,
            "severity_level": severity,
        }
        response = await self._service.calculate_smart_dosage(
            diagnostic_data=diagnostic_data,
            area_sqm=area_sqm,
            tank_capacity=tank_capacity,
        )
        return response.model_dump()


# Default singleton instance for dependency injection
diagnostic_engine = GeminiDiagnosticEngine()
