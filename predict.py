import pickle
import sys

from utils import clean_text

# Load model

try:
    model = pickle.load(open("models/sentiment_model.pkl", "rb"))
    encoder = pickle.load(open("models/label_encoder.pkl", "rb"))
except FileNotFoundError:
    sys.exit(
        "[ERROR] Model files not found in models/. "
        "Run 'python train.py' first."
    )

# ----------------------------------
# Prediction Loop
# ----------------------------------

while True:

    review = input(
        "\nEnter Review (or type quit): "
    )

    if review.lower() == "quit":
        break

    cleaned = clean_text(review)
    prediction = model.predict([cleaned])

    sentiment = encoder.inverse_transform(
        prediction
    )[0]

    print(
        f"Predicted Sentiment: {sentiment}"
    )