import base64
from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
from services.openai_chat import get_chat_response
from services.piano_analysis import analyze_piano_images
from limiter import limiter

router = APIRouter()

ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp", "image/gif"}
MAX_IMAGES = 3

class ChatRequest(BaseModel):
    message: str
    session_id: str
    expertise_result: Optional[Dict[str, Any]] = None

class ChatResponse(BaseModel):
    reply: str

class ChatUploadResponse(BaseModel):
    reply: str
    expertise_result: Optional[Dict[str, Any]] = None

@router.post("/chat", response_model=ChatResponse)
@limiter.limit("20/hour")
async def chat_endpoint(request: Request, body: ChatRequest):
    try:
        reply = await get_chat_response(
            message=body.message,
            session_id=body.session_id,
            expertise_result=body.expertise_result
        )
        return ChatResponse(reply=reply)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/chat-upload", response_model=ChatUploadResponse)
@limiter.limit("10/hour")
async def chat_upload_endpoint(
    request: Request,
    session_id: str = Form(...),
    message: str = Form(""),
    images: List[UploadFile] = File(...),
):
    """Chat endpoint that accepts images. Analyzes photos via Gemini then responds via GPT-4o."""
    if len(images) < 1 or len(images) > MAX_IMAGES:
        raise HTTPException(
            status_code=400,
            detail=f"Veuillez envoyer entre 1 et {MAX_IMAGES} photos.",
        )

    for img in images:
        if img.content_type not in ALLOWED_IMAGE_TYPES:
            raise HTTPException(
                status_code=400,
                detail=f"Type de fichier non supporté : {img.content_type}.",
            )

    try:
        # 1. Encode images for Gemini
        images_data = []
        for img in images:
            content = await img.read()
            images_data.append({
                "data": base64.b64encode(content).decode("utf-8"),
                "mime_type": img.content_type or "image/jpeg",
            })

        # 2. Run piano analysis via Gemini
        expertise_result = await analyze_piano_images(images_data, notes=message or None)

        # 3. Build a user message that tells GPT-4o the client sent photos
        user_message = message if message else "J'ai envoyé des photos de mon piano pour une évaluation."

        # 4. Get chat response with analysis context
        reply = await get_chat_response(
            message=user_message,
            session_id=session_id,
            expertise_result=expertise_result,
        )

        return ChatUploadResponse(reply=reply, expertise_result=expertise_result)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur : {str(e)}")
