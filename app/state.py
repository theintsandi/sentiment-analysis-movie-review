import pickle
from contextlib import asynccontextmanager
from pathlib import Path

import mlflow.sklearn
from fastapi import FastAPI

state: dict = {}

MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
REGISTERED_MODEL    = "models:/sentiment-model/Production"
ENCODER_PATH        = Path("models/label_encoder.pkl")


@asynccontextmanager
async def lifespan(app: FastAPI):
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    print(f"Loading model from registry: {REGISTERED_MODEL}")
    state["model"] = mlflow.sklearn.load_model(REGISTERED_MODEL)

    if not ENCODER_PATH.exists():
        raise RuntimeError(
            "label_encoder.pkl not found in models/. Run 'python train_mlflow.py' first."
        )
    state["encoder"] = pickle.load(ENCODER_PATH.open("rb"))

    print("Model and encoder loaded and ready.")
    yield
    state.clear()