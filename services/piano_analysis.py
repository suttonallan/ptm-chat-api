import os
import httpx
from dotenv import load_dotenv
from typing import Optional

load_dotenv()

PIANO_TEK_AI_URL = os.getenv("PIANO_TEK_AI_URL", "")


async def analyze_piano(
    images: list[bytes],
    filenames: list[str],
    content_types: list[str],
    email: Optional[str] = None,
    nom: Optional[str] = None,
    telephone: Optional[str] = None,
    notes: Optional[str] = None,
) -> dict:
    """
    Forward piano images to piano-tek-ai for analysis.

    Args:
        images: List of image file contents (1-3 images)
        filenames: Original filenames for each image
        content_types: MIME types for each image
        email: Optional client email
        nom: Optional client name
        telephone: Optional client phone
        notes: Optional client notes about the piano

    Returns:
        Analysis result dict from piano-tek-ai
    """
    if not PIANO_TEK_AI_URL:
        raise ValueError("PIANO_TEK_AI_URL is not configured")

    url = f"{PIANO_TEK_AI_URL.rstrip('/')}/analyze-piano-wordpress"

    files = [
        ("images", (filenames[i], images[i], content_types[i]))
        for i in range(len(images))
    ]

    data = {}
    if email:
        data["email"] = email
    if nom:
        data["nom"] = nom
    if telephone:
        data["telephone"] = telephone
    if notes:
        data["notes"] = notes

    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(url, files=files, data=data)
        response.raise_for_status()
        return response.json()
