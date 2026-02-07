from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes import chat

app = FastAPI(title="PTM Chat API")

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "https://ptm-chat.onrender.com"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(chat.router, prefix="/api", tags=["chat"])

@app.get("/")
def root():
    return {"message": "PTM Chat API"}

@app.get("/health")
def health():
    return {"status": "ok"}
