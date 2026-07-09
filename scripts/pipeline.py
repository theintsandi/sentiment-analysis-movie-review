import mlflow
import mlflow.sklearn
import pandas as pd
from prefect import flow, task
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Your existing cleaning function
from utils import clean_text


# -----------------------------
# Task 1: Load and preprocess
# -----------------------------
@task
def load_and_preprocess(path: str):
    df = pd.read_csv(path)
    df["clean"] = df["review"].apply(clean_text)
    X = df["clean"]
    y = df["sentiment"]
    return train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )
# -----------------------------------------
# Task 2: Train model (most failure-prone)
# -----------------------------------------
@task(retries=3, retry_delay_seconds=30)
def train_model(X_train, y_train):
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_vec = vectorizer.fit_transform(X_train)
    model = LogisticRegression(C=1.0, max_iter=1000)
    model.fit(X_train_vec, y_train)
    return model, vectorizer
# -----------------------------
# Task 3: Evaluate
# -----------------------------
@task
def evaluate_model(model, vectorizer, X_test, y_test):
    X_test_vec = vectorizer.transform(X_test)
    accuracy = model.score(X_test_vec, y_test)
    print(f"Accuracy: {accuracy:.4f}")
    return accuracy
# -----------------------------
# Task 4: Register model
# -----------------------------
@task
def register_model(model, accuracy):
    with mlflow.start_run(run_name="prefect-pipeline"):
        mlflow.log_metric("accuracy", accuracy)
        mlflow.sklearn.log_model(model, artifact_path="model")
        print("Model registered in MLflow")
# -----------------------------
# Flow
# -----------------------------
@flow(name="movieops-training-pipeline")
def training_pipeline():
    X_train, X_test, y_train, y_test = load_and_preprocess(
        "data/IMDB_Dataset.csv"
    )
    model, vectorizer = train_model(
        X_train,
        y_train
    )
    accuracy = evaluate_model(
        model,
        vectorizer,
        X_test,
        y_test
    )
    if accuracy > 0.88:
        register_model(model)
        print("✅ Accuracy exceeded threshold. Model registered.")
    else:
        print(
            f"⚠ Accuracy ({accuracy:.4f}) below threshold (0.88). "
            "Keeping current production model."
        )
if __name__ == "__main__":
    training_pipeline()