from datetime import date
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address, default_limits=["100/day"])

# --- Photo analysis limiter (Gemini calls are expensive) ---

MAX_ANALYSES_PER_DAY = 3

# {ip: {"date": date, "count": int}}
_analysis_counts: dict = {}


def can_analyze(ip: str) -> bool:
    """Check if this IP still has photo analyses remaining today."""
    today = date.today()
    record = _analysis_counts.get(ip)
    if not record or record["date"] != today:
        return True
    return record["count"] < MAX_ANALYSES_PER_DAY


def record_analysis(ip: str):
    """Record one photo analysis for this IP."""
    today = date.today()
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
