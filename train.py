import os
import pickle
import sys

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
from utils import clean_text

# ----------------------------------
# Load Dataset
# ----------------------------------
 
DATA_PATH = "data/IMDB_Dataset.csv"
 
if not os.path.exists(DATA_PATH):
    sys.exit(f"[ERROR] Dataset not found at '{DATA_PATH}'. "
             "Place 'IMDB_Dataset.csv' inside a 'data/' folder.")
 
df = pd.read_csv(DATA_PATH)
 
required_cols = {"review", "sentiment"}
if not required_cols.issubset(df.columns):
    sys.exit(f"[ERROR] CSV must contain columns: {required_cols}. "
             f"Found: {set(df.columns)}")
 
print(f"Loaded {len(df):,} rows.")
print(df["sentiment"].value_counts(), "\n")
 
# ----------------------------------
# Clean Text
# ----------------------------------
 
print("Cleaning reviews …")
df["clean_review"] = df["review"].apply(clean_text)
 
# ----------------------------------
# Encode Labels
# ----------------------------------
 
encoder = LabelEncoder()
df["label"] = encoder.fit_transform(df["sentiment"])   # positive=1, negative=0
 
print("Classes:", list(encoder.classes_))
 
# ----------------------------------
# Split Data
# ----------------------------------
 
X = df["clean_review"]
y = df["label"]
 
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y,          # keeps class balance in both splits
)
 
print(f"Train: {len(X_train):,}  |  Test: {len(X_test):,}\n")
 
# ----------------------------------
# Models
# ----------------------------------
 
TFIDF_PARAMS = dict(max_features=10_000, ngram_range=(1, 2), sublinear_tf=True)
 
models = {
    "Naive Bayes": Pipeline([
        ("tfidf", TfidfVectorizer(**TFIDF_PARAMS)),
        ("model", MultinomialNB()),
    ]),
    "Logistic Regression": Pipeline([
        ("tfidf", TfidfVectorizer(**TFIDF_PARAMS)),
        ("model", LogisticRegression(max_iter=1000, C=1.0)),
    ]),
    "Linear SVM": Pipeline([
        ("tfidf", TfidfVectorizer(**TFIDF_PARAMS)),
        ("model", LinearSVC(max_iter=2000, C=1.0)),
    ]),
}
 
# ----------------------------------
# Train & Evaluate
# ----------------------------------
 
results = {}
 
for name, pipeline in models.items():
    print(f"{name}")
    print("=" * 50)
 
    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)
    acc = accuracy_score(y_test, preds)
    results[name] = acc
 
    print(f"Accuracy : {acc:.4f}")
    print(classification_report(y_test, preds, target_names=encoder.classes_))
 
# ----------------------------------
# Save Best Model + Encoder
# ----------------------------------
 
best_name = max(results, key=results.get)
best_model = models[best_name]
 
print(f"\n Best Model : {best_name}  (accuracy={results[best_name]:.4f})")
 
os.makedirs("models", exist_ok=True)
 
with open("models/sentiment_model.pkl", "wb") as f:
    pickle.dump(best_model, f)
 
with open("models/label_encoder.pkl", "wb") as f:
    pickle.dump(encoder, f)
 
print("Model and encoder saved to models/")
