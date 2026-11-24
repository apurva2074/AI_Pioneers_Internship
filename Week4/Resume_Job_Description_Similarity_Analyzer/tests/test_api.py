from fastapi.testclient import TestClient

from app.main import app
from app.services.similarity import build_vectorizer


client = TestClient(app)


def setup_module():
    # Tests use an in-memory fitted vectorizer and do not require a prior training command.
    corpus = [
        "Python developer with FastAPI, SQL, PostgreSQL, Git and Docker.",
        "Frontend engineer with JavaScript, React.js, HTML and CSS.",
        "Machine learning engineer with Python, pandas, NumPy and scikit-learn.",
    ]
    vectorizer = build_vectorizer()
    vectorizer.fit(corpus)
    app.state.vectorizer = vectorizer


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_analyze_valid_text():
    response = client.post(
        "/analyze",
        json={
            "resume_text": "Python developer experienced with FastAPI, SQL, PostgreSQL, Git and Docker. Built backend APIs and tested services.",
            "job_description": "We need a Python developer with FastAPI, SQL, PostgreSQL, Git, Docker and AWS experience. Build backend APIs and test services.",
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert 0 <= body["match_score"] <= 100
    assert "python" in body["matched_skills"]
    assert "aws" in body["missing_skills"]


def test_reject_short_text():
    response = client.post(
        "/analyze",
        json={"resume_text": "short", "job_description": "also short"},
    )
    assert response.status_code == 422


def test_reject_unsupported_upload_extension():
    response = client.post(
        "/analyze/upload",
        data={"job_description": "We need a Python developer with backend API experience and SQL skills."},
        files={"resume_file": ("resume.docx", b"sample", "application/octet-stream")},
    )
    assert response.status_code == 400
