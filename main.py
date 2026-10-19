import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from starlette.responses import JSONResponse
from limiter import limiter

logging.basicConfig(level=logging.INFO)

app = FastAPI(title="PTM Chat API")

app.state.limiter = limiter
app.add_middleware(SlowAPIMiddleware)

@app.exception_handler(RateLimitExceeded)
def rate_limit_handler(request, exc):
    return JSONResponse(
        status_code=429,
        content={"detail": "Vous avez atteint la limite de messages pour aujourd'hui. Appelez-nous au 514 344-8008 pour poursuivre la conversation."}
    )

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "https://ptm-chat.onrender.com",
        "https://pianotechniquemontreal.com",
        "https://www.pianotechniquemontreal.com"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
from routes import chat, analyze
app.include_router(chat.router, prefix="/api", tags=["chat"])
app.include_router(analyze.router, prefix="/api", tags=["analyze"])

@app.get("/")
def root():
    return {"message": "PTM Chat API"}

@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/health/ai")
async def health_ai():
    """Vérifie que l'IA (Claude) répond. Ne révèle aucune clé : seulement le modèle
    utilisé et, en cas d'échec, le début du message d'erreur du fournisseur."""
    import os
    from services.claude_ai import complete, get_model
    try:
        r = await complete("Réponds seulement : ok", [{"role": "user", "content": "test"}], max_tokens=5)
        return {"ia": {"ok": True, "fournisseur": "Claude", "model": await get_model(), "reponse": r[:20]}}
    except Exception as e:
        return {"ia": {"ok": False, "erreur": str(e)[:300], "cle_presente": bool(os.getenv("ANTHROPIC_API_KEY"))}}
