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
    """Vérifie que les deux IA répondent. Ne révèle aucune clé : seulement le modèle
    utilisé et, en cas d'échec, le début du message d'erreur du fournisseur."""
    import os
    from services.openai_chat import client, get_chat_model, _is_model_gone
    out = {}
    try:
        model = get_chat_model()
        try:
            client.chat.completions.create(model=model, messages=[{"role": "user", "content": "ok"}], max_tokens=3)
        except Exception as e:
            if not _is_model_gone(e):
                raise
            model = get_chat_model(force_discovery=True)
            client.chat.completions.create(model=model, messages=[{"role": "user", "content": "ok"}], max_tokens=3)
        out["conversation"] = {"ok": True, "model": model}
    except Exception as e:
        out["conversation"] = {"ok": False, "erreur": str(e)[:300], "cle_presente": bool(os.getenv("OPENAI_API_KEY"))}
    try:
        from services.piano_analysis import get_gemini_model
        m = get_gemini_model()
        r = await m.generate_content_async("Réponds seulement : ok")
        out["analyse_photo"] = {"ok": True, "model": m.model_name, "reponse": (r.text or "")[:20]}
    except Exception as e:
        out["analyse_photo"] = {"ok": False, "erreur": str(e)[:300]}
    return out
