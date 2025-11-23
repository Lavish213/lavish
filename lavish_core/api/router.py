# lavish_core/api/router.py  (FINAL BUFFED VERSION)

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
import logging

# -------------------------------------------
# Logging
# -------------------------------------------
logger = logging.getLogger("LavishBot.API")

# -------------------------------------------
# Request Models
# -------------------------------------------
class ProcessPayload(BaseModel):
    """Incoming API payload model."""
    data: dict | None = None
    source: str | None = "unknown"

# -------------------------------------------
# API Router
# -------------------------------------------
api_router = APIRouter(
    prefix="/api",
    tags=["API"]
)

# -------------------------------------------
# Health Check
# -------------------------------------------
@api_router.get("/status")
async def api_status():
    """
    Basic API heartbeat.
    """
    return {
        "service": "LavishBot API",
        "status": "online",
        "version": "1.0"
    }

# -------------------------------------------
# Processing Endpoint
# -------------------------------------------
@api_router.post("/process")
async def api_process(payload: ProcessPayload, request: Request):
    """
    Universal processing endpoint.
    Accepts JSON payloads and routes them internally later.
    """
    try:
        client_ip = request.client.host
        logger.info(f"[PROCESS] Request from {client_ip}: {payload.dict()}")

        # ------------------------
        # Placeholder Logic
        # (Later you plug in ML, automation, commands, LLM...)
        # ------------------------
        response = {
            "success": True,
            "message": "Payload received",
            "received": payload.dict(),
        }

        return response

    except Exception as e:
        logger.error(f"[PROCESS_ERROR] {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal Processing Error: {str(e)}"
        )