import base64
from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import logging
from services.openai_chat import get_chat_response
from services.piano_analysis import analyze_piano_images
from services.url_scraper import find_urls, scrape_listing, format_listing_context, download_listing_images
from limiter import limiter

logger = logging.getLogger("piano-tek-ai")

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

@router.post("/chat", response_model=ChatUploadResponse)
@limiter.limit("20/hour")
async def chat_endpoint(request: Request, body: ChatRequest):
    try:
        # Detect URLs in the user message and scrape listing data
        listing_context = None
        expertise_result = body.expertise_result
        urls = find_urls(body.message)
        if urls:
            logger.info(f"URL détectée dans le message: {urls[0][:80]}")
            listing = await scrape_listing(urls[0])
            if listing:
                listing_context = format_listing_context(listing)
                logger.info(f"Contexte d'annonce généré ({len(listing_context)} chars)")

                # Download and analyze listing images via Gemini
                image_urls = listing.get("images", [])
                if image_urls and not expertise_result:
                    logger.info(f"Téléchargement de {len(image_urls)} images de l'annonce...")
                    images_data = await download_listing_images(image_urls)
                    if images_data:
                        notes = f"Achat potentiel — annonce en ligne ({listing.get('source', 'web')})"
                        if listing.get("title"):
                            notes += f"\nTitre: {listing['title']}"
                        if listing.get("price"):
                            notes += f"\nPrix demandé: {listing['price']}"
                        if listing.get("description"):
                            notes += f"\nDescription: {listing['description'][:500]}"
                        try:
                            expertise_result = await analyze_piano_images(images_data, notes=notes)
                            logger.info("Analyse Gemini des photos de l'annonce réussie")
                        except Exception as e:
                            logger.warning(f"Analyse Gemini échouée pour l'annonce: {e}")
                    else:
                        logger.warning("Aucune image n'a pu être téléchargée de l'annonce")
                elif not image_urls:
                    logger.info("Aucune image trouvée dans l'annonce")
            else:
                # Scraping failed — provide minimal fallback context so GPT-4o
                # knows a listing URL was shared and can still comment on it
                logger.warning(f"Scraping échoué pour {urls[0][:80]}, utilisation du contexte minimal")
                domain = urls[0].split("/")[2] if len(urls[0].split("/")) > 2 else "web"
                source = "Kijiji" if "kijiji" in domain.lower() else domain
                listing_context = (
                    f"Le client a partagé un lien d'annonce ({source}) : {urls[0]}\n"
                    f"IMPORTANT : Nous n'avons pas pu récupérer les détails de l'annonce automatiquement. "
                    f"Demande au client de t'envoyer des photos du piano avec le bouton 📎 "
                    f"pour que tu puisses faire une évaluation. Mentionne aussi qu'une inspection "
                    f"avant achat est toujours recommandée."
                )

        reply = await get_chat_response(
            message=body.message,
            session_id=body.session_id,
            expertise_result=expertise_result,
            listing_context=listing_context,
        )
        return ChatUploadResponse(reply=reply, expertise_result=expertise_result)
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
