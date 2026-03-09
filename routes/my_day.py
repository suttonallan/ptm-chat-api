from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import logging
from services.my_day import generate_my_day
from limiter import limiter

logger = logging.getLogger("piano-tek-ai")

router = APIRouter()


class PianoInfo(BaseModel):
    brand: Optional[str] = None
    model: Optional[str] = None
    type: Optional[str] = None  # "à queue", "droit", etc.
    year: Optional[int] = None
    age: Optional[str] = None
    serial: Optional[str] = None


class EnvironmentInfo(BaseModel):
    temperature: Optional[str] = None
    humidity: Optional[str] = None


class DayEntry(BaseModel):
    type: str = "appointment"  # "appointment" or "personal"
    time: Optional[str] = None
    title: Optional[str] = None

    # Professional appointment fields
    client_name: Optional[str] = None
    client_since: Optional[str] = None
    location: Optional[str] = None
    piano: Optional[PianoInfo] = None
    services: Optional[List[str]] = None
    previous_services: Optional[List[str]] = None
    environment: Optional[EnvironmentInfo] = None
    tuning_pitch: Optional[str] = None
    notes: Optional[str] = None
    parking: Optional[str] = None  # accepted but never shown in output


class MyDayRequest(BaseModel):
    date: Optional[str] = None
    entries: List[DayEntry]


class MyDayResponse(BaseModel):
    briefing: str


@router.post("/my-day", response_model=MyDayResponse)
@limiter.limit("20/day")
async def my_day_endpoint(request: Request, body: MyDayRequest):
    """Generate the technician's daily briefing from appointments and personal events."""
    if not body.entries:
        raise HTTPException(status_code=400, detail="Aucun rendez-vous ou événement fourni.")

    try:
        entries_data = [entry.model_dump(exclude_none=True) for entry in body.entries]
        briefing = await generate_my_day(entries_data)
        return MyDayResponse(briefing=briefing)
    except Exception as e:
        logger.error(f"Erreur lors de la génération du résumé de journée: {e}")
        raise HTTPException(status_code=500, detail=str(e))
