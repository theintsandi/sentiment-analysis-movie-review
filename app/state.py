import pickle
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI

state: dict = {}


@asynccontextmanager
async def lifespan(app:FastAPI):
    model_path   = Path("models/sentiment_model.pkl")
    encoder_path = Path("models/label_encoder.pkl")

    if not model_path.exists() or not encoder_path.exists():
        raise RuntimeError(
            "Model files not found in models/. Run 'python train.py' first."
        )

    state["model"]   = pickle.load(model_path.open("rb"))
    state["encoder"] = pickle.load(encoder_path.open("rb"))

    print("Model loaded and ready.")
    yield
    state.clear()
