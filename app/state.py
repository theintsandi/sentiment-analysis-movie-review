import pickle
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI

state: dict = {}

MODEL_PATH   = Path("models/sentiment_model.pkl")
ENCODER_PATH = Path("models/label_encoder.pkl")


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not MODEL_PATH.exists() or not ENCODER_PATH.exists():
        raise RuntimeError(
            "Model files not found in models/. Run 'python train_mlflow.py' first."
        )

    with open(MODEL_PATH, "rb") as f:
        state["model"] = pickle.load(f)

    with open(ENCODER_PATH, "rb") as f:
        state["encoder"] = pickle.load(f)

    print("Model loaded and ready.")
    yield
    state.clear()