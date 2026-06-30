from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.state import lifespan
from app.routes import router

app = FastAPI(
    title="Sentiment Analysis API",
    description="Predicts positive/negative sentiment for movie reviews.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

app.include_router(router)
