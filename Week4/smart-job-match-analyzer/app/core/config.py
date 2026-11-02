from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "app" / "models"
MODEL_PATH = MODEL_DIR / "tfidf_vectorizer.joblib"
TRAINING_CORPUS_PATH = DATA_DIR / "training_corpus.json"
MAX_UPLOAD_SIZE_BYTES = 5 * 1024 * 1024  # 5 MB
ALLOWED_EXTENSIONS = {".pdf", ".txt"}
MIN_TEXT_LENGTH = 20
