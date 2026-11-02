from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import RedirectResponse

from app.core.config import ALLOWED_EXTENSIONS, MAX_UPLOAD_SIZE_BYTES, MIN_TEXT_LENGTH
from app.schemas import AnalyzeTextRequest
from app.services.document_parser import extract_text_from_file
from app.services.similarity import calculate_similarity, load_vectorizer
from app.services.skill_extractor import extract_skills, flatten_skills


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        app.state.vectorizer = load_vectorizer()
    except FileNotFoundError:
        app.state.vectorizer = None
    yield


app = FastAPI(
    title="Smart Job Match & Skill Gap Analyzer",
    description=(
        "Compares resume text with a job description using TF-IDF cosine similarity "
        "and identifies matched and missing skills. The score is not a hiring probability."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": app.state.vectorizer is not None}


def _analyze(resume_text: str, job_description: str):
    if app.state.vectorizer is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not trained. Run `python train_model.py` and restart the API.",
        )
    if len(resume_text.strip()) < MIN_TEXT_LENGTH:
        raise HTTPException(status_code=422, detail="Resume text is too short.")
    if len(job_description.strip()) < MIN_TEXT_LENGTH:
        raise HTTPException(status_code=422, detail="Job description is too short.")

    resume_skills_by_category = extract_skills(resume_text)
    job_skills_by_category = extract_skills(job_description)
    resume_skills = set(flatten_skills(resume_skills_by_category))
    job_skills = set(flatten_skills(job_skills_by_category))

    matched = sorted(resume_skills & job_skills)
    missing = sorted(job_skills - resume_skills)
    extra = sorted(resume_skills - job_skills)

    score = calculate_similarity(resume_text, job_description, app.state.vectorizer)
    skill_coverage = round((len(matched) / len(job_skills)) * 100, 2) if job_skills else 0.0

    return {
        "match_score": score,
        "score_explanation": "TF-IDF cosine similarity between the resume and job description; not a hiring probability.",
        "skill_coverage_percent": skill_coverage,
        "matched_skills": matched,
        "missing_skills": missing,
        "additional_resume_skills": extra,
        "resume_skills_by_category": resume_skills_by_category,
        "job_skills_by_category": job_skills_by_category,
        "note": "Dictionary-based skill extraction can miss synonyms, context, and skills not included in its vocabulary.",
    }


@app.post("/analyze")
def analyze_text(request: AnalyzeTextRequest):
    return _analyze(request.resume_text, request.job_description)


@app.post("/analyze/upload")
async def analyze_upload(
    job_description: str = Form(...),
    resume_file: UploadFile = File(...),
):
    extension = Path(resume_file.filename or "").suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Only PDF and TXT resumes are supported.")

    content = await resume_file.read()
    if len(content) > MAX_UPLOAD_SIZE_BYTES:
        raise HTTPException(status_code=413, detail="Resume file exceeds the 5 MB limit.")
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded resume file is empty.")

    try:
        resume_text = extract_text_from_file(resume_file.filename or "resume", content)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not read resume: {exc}") from exc

    if not resume_text.strip():
        raise HTTPException(
            status_code=422,
            detail="No readable text found. Scanned/image-only PDFs need OCR, which this version does not include.",
        )
    return _analyze(resume_text, job_description)
