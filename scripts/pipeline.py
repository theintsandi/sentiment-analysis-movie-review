import pickle
from datetime import timedelta
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from prefect import flow, task
from prefect.tasks import task_input_hash
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

from utils import clean_text

MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
EXPERIMENT_NAME     = "movie-sentiment"
MODEL_NAME          = "sentiment-model"
ACCURACY_THRESHOLD  = 0.88


@task(cache_key_fn=task_input_hash, cache_expiration=timedelta(hours=24))
def load_and_preprocess(path: str):
    df = pd.read_csv(path)
    df["clean"] = df["review"].apply(clean_text)
    encoder = LabelEncoder()
    df["label"] = encoder.fit_transform(df["sentiment"])

    # Save encoder for the API
    Path("models").mkdir(exist_ok=True)
    with open("models/label_encoder.pkl", "wb") as f:
        pickle.dump(encoder, f)

    X = df["clean"]
    y = df["label"]
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


@task(retries=3, retry_delay_seconds=30)
def train_model(X_train, y_train):
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=10_000, ngram_range=(1, 2))),
        ("model", LogisticRegression(C=1.0, max_iter=1000)),
    ])
    pipeline.fit(X_train, y_train)
    return pipeline


@task
def evaluate_model(pipeline, X_test, y_test):
    preds = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, preds)
    print(f"Accuracy: {accuracy:.4f}")
    return accuracy


@task
def register_if_better(pipeline, accuracy: float):
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    if accuracy <= ACCURACY_THRESHOLD:
        print(f"Accuracy {accuracy:.4f} below threshold — skipping registration")
        return

    with mlflow.start_run(run_name="prefect-auto-retrain"):
        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_param("C", 1.0)
        mlflow.log_param("max_iter", 1000)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.sklearn.log_model(pipeline, name="model")

        # Save model locally for Docker
        Path("models").mkdir(exist_ok=True)
        with open("models/sentiment_model.pkl", "wb") as f:
            pickle.dump(pipeline, f)

        print(f"Model registered — accuracy: {accuracy:.4f}")


@flow(name="movieops-retraining-pipeline")
def retraining_pipeline(data_path: str = "data/IMDB_Dataset.csv"):
    X_train, X_test, y_train, y_test = load_and_preprocess(data_path)
    pipeline = train_model(X_train, y_train)
    accuracy = evaluate_model(pipeline, X_test, y_test)
    register_if_better(pipeline, accuracy)


if __name__ == "__main__":
    retraining_pipeline()