"""Fit and save the TF-IDF vectorizer from the bundled text corpus."""
import json

from app.core.config import TRAINING_CORPUS_PATH, MODEL_PATH
from app.services.similarity import build_vectorizer, save_vectorizer


def main() -> None:
    with TRAINING_CORPUS_PATH.open("r", encoding="utf-8") as file:
        documents = json.load(file)

    if not isinstance(documents, list) or len(documents) < 2:
        raise ValueError("Training corpus must contain at least two text documents.")

    vectorizer = build_vectorizer()
    vectorizer.fit(documents)
    save_vectorizer(vectorizer, MODEL_PATH)
    print(f"TF-IDF vectorizer trained on {len(documents)} documents.")
    print(f"Saved model to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
