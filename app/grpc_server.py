"""
gRPC server implementation for high-performance phytosanitary telemetry and evaluation.
Concurrently runs alongside FastAPI / Uvicorn using a non-blocking thread pool.
"""

import asyncio
from concurrent import futures
import logging
import time
from typing import Optional

import grpc
from sqlalchemy import text

from app.db import SessionLocal
from app.proto import agroscan_pb2, agroscan_pb2_grpc
from app.services.diagnostic_engine import diagnostic_engine
from app.services.gemini_service import gemini_service
from app.services.telemetry_service import telemetry_service

logger = logging.getLogger("agroscan.grpc")


class DiagnosticServicer(agroscan_pb2_grpc.DiagnosticServiceServicer):
    """
    gRPC Servicer implementing DiagnosticService RPC endpoints.
    Delegates domain evaluation to the Object-Oriented DiagnosticEngine
    and records operational telemetry via AgronomicTelemetryService.
    """

    def EvaluateSample(
        self, request: agroscan_pb2.SampleRequest, context
    ) -> agroscan_pb2.SampleResponse:
        """
        RPC handler to evaluate an agricultural crop sample image.
        """
        try:
            if not request.image_data:
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details("Sample image_data payload cannot be empty.")
                return agroscan_pb2.SampleResponse(
                    success=False,
                    error_message="Sample image_data payload cannot be empty.",
                )

            crop_target = request.crop_name or "Potato"
            plot_name = request.plot_name or "Main Plot"
            notes = request.notes or None
            user_id = request.user_id or None

            # Execute sample diagnosis through the OOP DiagnosticEngine
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                diag_dict = loop.run_until_complete(
                    diagnostic_engine.analyze_sample(
                        image_bytes=request.image_data,
                        crop=crop_target,
                        plot_name=plot_name,
                        notes=notes,
                    )
                )
            finally:
                loop.close()

            # Record telemetry in persistent store if DB session is reachable
            try:
                with SessionLocal() as db:
                    telemetry_service.record_diagnostic_history(
                        user_id=user_id,
                        payload=diag_dict,
                        db=db,
                    )
            except Exception as db_err:
                logger.warning("Telemetry recording warning in gRPC call: %s", db_err)

            return agroscan_pb2.SampleResponse(
                diagnostic_id=str(diag_dict.get("id", "")),
                crop_name=str(diag_dict.get("crop_name", crop_target)),
                part_analyzed=str(diag_dict.get("part_analyzed", "Hoja / Follaje")),
                common_disease_name=str(diag_dict.get("common_disease_name", "")),
                scientific_name=str(diag_dict.get("scientific_name", "")),
                severity_level=str(diag_dict.get("severity_level", "MODERATE")),
                confidence_percentage=int(diag_dict.get("confidence_percentage", 80)),
                summary_what_happened=str(diag_dict.get("summary_what_happened", "")),
                immediate_actions=diag_dict.get("immediate_actions", []),
                organic_solution=str(diag_dict.get("organic_solution", "")),
                chemical_solution=str(diag_dict.get("chemical_solution", "")),
                dosage_backpack_20l=str(diag_dict.get("dosage_backpack_20l", "")),
                created_at=str(diag_dict.get("created_at", "")),
                success=True,
                error_message="",
            )

        except Exception as exc:
            logger.exception("Error in gRPC EvaluateSample: %s", exc)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(exc))
            return agroscan_pb2.SampleResponse(
                success=False,
                error_message=f"Evaluation failed: {str(exc)}",
            )

    def GetHealthStatus(
        self, request: agroscan_pb2.HealthRequest, context
    ) -> agroscan_pb2.HealthResponse:
        """
        RPC handler returning live system health, database connection, and AI readiness.
        """
        db_ok = False
        try:
            with SessionLocal() as db:
                db.execute(text("SELECT 1"))
                db_ok = True
        except Exception as exc:
            logger.warning("gRPC Health check DB probe failed: %s", exc)

        return agroscan_pb2.HealthResponse(
            status="healthy" if db_ok else "degraded",
            version="2.2.0",
            gemini_configured=gemini_service.is_configured,
            database_connected=db_ok,
            timestamp_epoch=int(time.time()),
        )


def create_grpc_server(port: int = 50051, max_workers: int = 10) -> grpc.Server:
    """
    Creates and configures a non-blocking multithreaded gRPC server.
    """
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers))
    agroscan_pb2_grpc.add_DiagnosticServiceServicer_to_server(DiagnosticServicer(), server)
    server.add_insecure_port(f"[::]:{port}")
    logger.info("Configured gRPC DiagnosticService server on port %d", port)
    return server


def start_grpc_server(port: int = 50051) -> grpc.Server:
    """
    Starts the gRPC server in non-blocking mode.
    """
    server = create_grpc_server(port=port)
    server.start()
    logger.info("gRPC server started and listening on port %d", port)
    return server


def serve(port: int = 50051):
    """
    Blocking entry point for running gRPC server as an independent microservice process.
    """
    server = start_grpc_server(port=port)
    print(f"[gRPC] AgroScan DiagnosticService server active on port {port}. Press Ctrl+C to terminate.")
    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        logger.info("Stopping gRPC server...")
        server.stop(grace=2.0)
        print("[gRPC] Server stopped successfully.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    serve()
