from datetime import date
from slowapi import Limiter
from slowapi.util import get_remote_address


def client_ip(request) -> str:
    """Vraie adresse du visiteur : Render place le serveur derrière un relais,
    qui transmet l'adresse d'origine dans X-Forwarded-For. Sans ça, tous les
    visiteurs partageraient la même limite."""
    fwd = [x.strip() for x in request.headers.get("x-forwarded-for", "").split(",") if x.strip()]
    if fwd:
        return fwd[-1]  # le relais ajoute l'adresse réelle à la fin ; le début est falsifiable
    return get_remote_address(request)


limiter = Limiter(key_func=client_ip, default_limits=["100/day"])

# --- Photo analysis limiter (Gemini calls are expensive) ---

MAX_ANALYSES_PER_DAY = 3

# {ip: {"date": date, "count": int}}
_analysis_counts: dict = {}


# Plafond global quotidien (tous visiteurs confondus) : protège la facture d'IA.
MAX_ANALYSES_GLOBAL_PAR_JOUR = 150
_global = {"date": None, "count": 0}


def can_analyze(ip: str) -> bool:
    """Check if this IP still has photo analyses remaining today."""
    today = date.today()
    if _global["date"] == today and _global["count"] >= MAX_ANALYSES_GLOBAL_PAR_JOUR:
        return False
    record = _analysis_counts.get(ip)
    if not record or record["date"] != today:
        return True
    return record["count"] < MAX_ANALYSES_PER_DAY


def record_analysis(ip: str):
    """Record one photo analysis for this IP."""
    today = date.today()
    if _global["date"] != today:
        _global["date"], _global["count"] = today, 0
    _global["count"] += 1
    record = _analysis_counts.get(ip)
    if not record or record["date"] != today:
        _analysis_counts[ip] = {"date": today, "count": 1}
    else:
        record["count"] += 1


def analyses_remaining(ip: str) -> int:
    """Return how many analyses this IP has left today."""
    today = date.today()
    record = _analysis_counts.get(ip)
    if not record or record["date"] != today:
        return MAX_ANALYSES_PER_DAY
    return max(0, MAX_ANALYSES_PER_DAY - record["count"])
