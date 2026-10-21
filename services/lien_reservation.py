"""Lien de réservation prérempli : l'IA écrit le marqueur [[LIEN_RESERVATION]]{…} avec ce
qu'elle sait déjà ; on le remplace par l'adresse de la page de réservation, renseignements
placés après le « # » (jamais envoyés aux serveurs ni conservés dans leurs journaux)."""
import json
import re
from urllib.parse import urlencode, quote

BASE = "https://reserver.pianotechniquemontreal.com/"
_MARQUEUR = re.compile(r"\[\[LIEN_RESERVATION\]\]\s*(\{.*?\})?", re.S)
_CHAMPS = {"rue": 200, "ville": 100, "cp": 12, "type": 10, "marque": 60, "annee": 10, "besoin": 300}


def construire(d: dict) -> str:
    params = {"source": "chat"}
    for k, maxi in _CHAMPS.items():
        v = str(d.get(k) or "").strip()[:maxi]
        if k == "type":
            v = {"droit": "UPRIGHT", "upright": "UPRIGHT", "queue": "GRAND", "grand": "GRAND"}.get(v.lower(), v.upper())
            if v not in ("UPRIGHT", "GRAND"):
                continue
        if v:
            params[k] = v
    langue = "?lang=en" if str(d.get("langue", "")).lower() == "en" else ""
    return BASE + langue + "#" + urlencode(params, quote_via=quote)


def remplacer(reponse: str) -> str:
    def sub(m):
        try:
            d = json.loads(m.group(1)) if m.group(1) else {}
        except ValueError:
            d = {}
        return construire(d if isinstance(d, dict) else {})
    return _MARQUEUR.sub(sub, reponse or "")
