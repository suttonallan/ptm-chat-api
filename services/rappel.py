"""Demandes de rappel : quand le client a donné son prénom et son numéro dans le chat,
l'IA ajoute un marqueur invisible [[RAPPEL]]{…} à la fin de sa réponse. On le retire
avant d'afficher la réponse et on transmet la demande à l'équipe (Front, info@)
par l'Assistant Gazelle, de serveur à serveur.
"""
import json
import logging
import os
import re
from typing import Dict, List, Optional, Tuple

import httpx

logger = logging.getLogger("piano-tek-ai")

RELAIS = "https://assistant-gazelle-v5-api.onrender.com/api/chat-public/rappel"
_MARQUEUR = re.compile(r"\[\[RAPPEL\]\]\s*(\{.*?\})\s*$", re.S)

ECHEC_FR = ("\n\nDésolé, je n'ai pas réussi à transmettre votre demande à l'équipe. "
            "Le plus sûr est d'appeler au 514 344-8008 ou d'écrire à info@piano-tek.com.")
ECHEC_EN = ("\n\nSorry, I couldn't pass your request on to the team. "
            "Please call 514 344-8008 or write to info@piano-tek.com.")


def extraire(reponse: str) -> Tuple[str, Optional[Dict]]:
    """Retourne (réponse sans marqueur, demande ou None)."""
    m = _MARQUEUR.search(reponse or "")
    if not m:
        return reponse, None
    propre = reponse[: m.start()].rstrip()
    try:
        d = json.loads(m.group(1))
    except ValueError:
        return propre, None
    if not isinstance(d, dict):
        return propre, None
    return propre, d


async def transmettre(demande: Dict, historique: List[Dict]) -> bool:
    secret = os.getenv("CHAT_RELAY_SECRET")
    prenom = str(demande.get("prenom") or "").strip()[:80]
    tel = str(demande.get("telephone") or "").strip()[:40]
    if not secret or not prenom or len(re.sub(r"\D", "", tel)) < 10:
        logger.warning("Demande de rappel non transmise (secret absent ou données incomplètes)")
        return False
    corps = {
        "prenom": prenom,
        "telephone": tel,
        "moment": (str(demande.get("moment") or "").strip()[:200] or None),
        "langue": "en" if demande.get("langue") == "en" else "fr",
        "conversation": [
            {"role": "user" if h["role"] == "user" else "chat", "texte": str(h["content"])[:3000]}
            for h in historique[-12:] if isinstance(h.get("content"), str)
        ],
    }
    try:
        async with httpx.AsyncClient(timeout=30) as c:
            r = await c.post(RELAIS, json=corps, headers={"X-Chat-Secret": secret})
        if r.status_code == 200:
            logger.info("Demande de rappel transmise à l'équipe")
            return True
        logger.error(f"Relais de rappel : HTTP {r.status_code} {r.text[:200]}")
    except httpx.HTTPError as e:
        logger.error(f"Relais de rappel injoignable : {e}")
    return False
