from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any
from services.openai_chat import get_chat_response
from main import limiter

router = APIRouter()

class ChatRequest(BaseModel):
    message: str
    session_id: str
    expertise_result: Optional[Dict[str, Any]] = None

class ChatResponse(BaseModel):
    reply: str

@router.post("/chat", response_model=ChatResponse)
@limiter.limit("20/hour")
async def chat_endpoint(request: ChatRequest):
    try:
        reply = await get_chat_response(
            message=request.message,
            session_id=request.session_id,
            expertise_result=request.expertise_result
        )
        return ChatResponse(reply=reply)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
