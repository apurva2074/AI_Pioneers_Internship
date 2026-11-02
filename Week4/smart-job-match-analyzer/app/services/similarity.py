"""TF-IDF and cosine-similarity service."""
from pathlib import Path
from typing import Any

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.core.config import MODEL_PATH
from app.services.text_processing import normalize_for_similarity


def build_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(
        preprocessor=normalize_for_similarity,
        lowercase=False,
        token_pattern=r"(?u)\b\w+\b",
        ngram_range=(1, 2),
        sublinear_tf=True,
    )


def save_vectorizer(vectorizer: TfidfVectorizer, path: Path = MODEL_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(vectorizer, path)


def load_vectorizer(path: Path = MODEL_PATH) -> TfidfVectorizer:
    if not path.exists():
        raise FileNotFoundError(
            f"Model not found at {path}. Run `python train_model.py` before starting the API."
        )
    loaded: Any = joblib.load(path)
    return loaded


def calculate_similarity(resume_text: str, job_description: str, vectorizer: TfidfVectorizer) -> float:
    matrix = vectorizer.transform([resume_text, job_description])
    score = float(cosine_similarity(matrix[0:1], matrix[1:2])[0][0])
    return round(score * 100, 2)
