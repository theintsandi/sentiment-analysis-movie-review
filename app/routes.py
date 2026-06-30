from fastapi import APIRouter, HTTPException

from app.schemas import ReviewInput, SentimentOutput
from utils import clean_text
from app.state import state

router = APIRouter()


@router.get("/")
def root():
    return {"message": "Sentiment Analysis API is running. POST to /predict"}


@router.get("/health")
def health():
    return {"status": "ok"}


@router.post("/predict", response_model=SentimentOutput)
def predict(body: ReviewInput):
    if not body.review.strip():
        raise HTTPException(status_code=400, detail="review must be a non-empty string")

    cleaned    = clean_text(body.review)
    model      = state["model"]
    encoder    = state["encoder"]

    prediction = model.predict([cleaned])
    sentiment  = encoder.inverse_transform(prediction)[0]

    confidence = None
    try:
        proba      = model.predict_proba([cleaned])[0]
        confidence = round(float(max(proba)) * 100, 2)
    except AttributeError:
        pass

    return SentimentOutput(sentiment=sentiment, confidence=confidence)
