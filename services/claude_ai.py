"""
Accès unique à l'IA du chat : Claude (Anthropic), le même fournisseur que l'Assistant Gazelle.
Sert à la conversation ET à l'analyse des photos.

Le modèle n'est jamais codé en dur de façon définitive :
- CLAUDE_MODEL (variable d'environnement Render) l'impose s'il est défini ;
- sinon on utilise le modèle par défaut ci-dessous ;
- si Anthropic le retire (erreur « not found »), on choisit tout seul le Sonnet le plus récent.
"""
import os
import re
import logging
from typing import List, Dict, Optional

logger = logging.getLogger("piano-tek-ai")

DEFAULT_MODEL = "claude-sonnet-5-5"
_model: Optional[str] = None
_client = None


def get_client():
    global _client
    if _client is None:
        key = os.getenv("ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError("ANTHROPIC_API_KEY non configurée")
        from anthropic import AsyncAnthropic
        _client = AsyncAnthropic(api_key=key)
    return _client


def is_model_gone(err: Exception) -> bool:
    msg = str(err).lower()
    return "not_found" in msg or "not found" in msg or "404" in msg or "deprecated" in msg


async def _discover_model() -> str:
    try:
        page = await get_client().models.list(limit=100)
        ids = [m.id for m in page.data]
    except Exception as e:
        logger.error(f"Impossible de lister les modèles Claude: {e}")
        return DEFAULT_MODEL
    sonnets = [i for i in ids if "sonnet" in i]
    if not sonnets:
        return ids[0] if ids else DEFAULT_MODEL

    def version(i: str):
        nums = re.findall(r"\d+", i.replace("sonnet", ""))
        # ignore les dates (8 chiffres) pour ne garder que le numéro de version
        return tuple(int(n) for n in nums if len(n) < 8)

    return sorted(sonnets, key=version, reverse=True)[0]


async def get_model(force_discovery: bool = False) -> str:
    global _model
    if force_discovery:
        _model = await _discover_model()
        logger.info(f"Modèle Claude choisi automatiquement : {_model}")
    elif not _model:
        _model = os.getenv("CLAUDE_MODEL") or DEFAULT_MODEL
    return _model


async def complete(system: str, messages: List[Dict], max_tokens: int = 800,
                   temperature: float = 0.5) -> str:
    """Appelle Claude et retourne le texte de la réponse. Rebascule seul si le modèle a été retiré."""
    async def call(model: str):
        return await get_client().messages.create(
            model=model, system=system, messages=messages,
            max_tokens=max_tokens, temperature=temperature,
        )
    try:
        resp = await call(await get_model())
    except Exception as e:
        if not is_model_gone(e):
            raise
        logger.warning(f"Modèle Claude retiré ({e}) — recherche d'un modèle à jour")
        resp = await call(await get_model(force_discovery=True))
    return "".join(b.text for b in resp.content if getattr(b, "type", "") == "text").strip()
