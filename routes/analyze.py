from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from typing import Optional
from services.piano_analysis import analyze_piano
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
        contents = [await img.read() for img in images]
        filenames = [img.filename or "image.jpg" for img in images]
        content_types = [img.content_type or "image/jpeg" for img in images]

        result = await analyze_piano(
            images=contents,
            filenames=filenames,
            content_types=content_types,
            email=email,
            nom=nom,
            telephone=telephone,
            notes=notes,
        )
        return result

    except ValueError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=502,
            detail=f"Erreur lors de l'analyse : {str(e)}",
        )
