import base64
from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from typing import Optional
from services.piano_analysis import analyze_piano_images
from main import limiter

router = APIRouter()

ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp", "image/gif"}
MAX_IMAGES = 3


@router.post("/analyze")
@limiter.limit("10/hour")
async def analyze_endpoint(
    request: Request,
    images: list[UploadFile] = File(...),
    email: Optional[str] = Form(None),
    nom: Optional[str] = Form(None),
    telephone: Optional[str] = Form(None),
    notes: Optional[str] = Form(None),
):
    if len(images) < 1 or len(images) > MAX_IMAGES:
        raise HTTPException(
            status_code=400,
            detail=f"Veuillez fournir entre 1 et {MAX_IMAGES} images.",
        )

    for img in images:
        if img.content_type not in ALLOWED_TYPES:
            raise HTTPException(
                status_code=400,
                detail=f"Type de fichier non supporté : {img.content_type}. "
                f"Types acceptés : {', '.join(ALLOWED_TYPES)}",
            )

    try:
        images_data = []
        for img in images:
            content = await img.read()
            images_data.append({
                "data": base64.b64encode(content).decode("utf-8"),
                "mime_type": img.content_type or "image/jpeg",
            })

        result = await analyze_piano_images(images_data, notes=notes)
        return result

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors de l'analyse : {str(e)}",
        )
