from pydantic import BaseModel


class ReviewInput(BaseModel):
    review: str

    model_config = {
        "json_schema_extra": {
            "examples": [{"review": "This movie was absolutely fantastic!"}]
        }
    }


class SentimentOutput(BaseModel):
    sentiment:   str
    confidence:  float | None = None
