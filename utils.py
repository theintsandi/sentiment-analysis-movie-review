import re
import ssl

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

ssl._create_default_https_context = ssl._create_unverified_context
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
 
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()
 
 
def clean_text(text: str) -> str:
    """Normalize a review string for TF-IDF vectorization."""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)   # URLs
    text = re.sub(r"<.*?>", "", text)             # HTML tags
    text = re.sub(r"[^a-zA-Z\s]", "", text)       # non-alpha chars
    words = text.split()
    words = [
        lemmatizer.lemmatize(w)
        for w in words
        if w not in stop_words and len(w) > 1    # drop single-char noise
    ]
    return " ".join(words)
