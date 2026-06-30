# Sentiment Analysis — Movie Reviews

Train ML models on the IMDB dataset to classify movie reviews as **positive** or **negative**.

## Project Structure

```
├── train.py         — Trains and evaluates 3 models, saves the best
├── predict.py       — CLI tool for interactive predictions
├── app.py           — FastAPI REST API (POST /predict)
├── utils.py         — Text cleaning pipeline
├── index.html       — Dark-themed web UI (served separately)
├── data/
│   └── IMDB_Dataset.csv
└── models/
    ├── sentiment_model.pkl
    └── label_encoder.pkl
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

**Train models:**
```bash
python train.py
```

**CLI prediction:**
```bash
python predict.py
```

**API server:**
```bash
uvicorn app:app --reload
```

**Web UI:** Open `index.html` in your browser while the API is running, then paste a review and click **Analyse**.

## Benchmarks

Trained on 40,000 reviews, tested on 10,000 (IMDB 50K dataset).

| Model                | Accuracy |
|----------------------|----------|
| Logistic Regression  | 0.8964   |
| Linear SVM           | 0.8863   |
| Naive Bayes          | 0.8661   |

**Best model:** Logistic Regression

## API

```http
POST http://localhost:8000/predict
Content-Type: application/json

{"review": "A slow-burn masterpiece — every frame drips with intention."}
```

Response:
```json
{"sentiment": "positive", "confidence": 94.32}
```
