# Smart Job Match & Skill Gap Analyzer

A Python NLP project that compares resume content with a job description. It returns a TF-IDF cosine-similarity score and a transparent skill comparison (matched skills, missing job skills, and additional resume skills).

> **Important:** `match_score` is a text-similarity measure, not a probability of getting an interview or job. Skill extraction uses a curated dictionary and regex, so it can miss synonyms, implied skills, and terms outside the dictionary.

## Features

- Accepts resume text directly or uploads a PDF/TXT resume.
- Cleans and normalizes text while preserving the meaning of technical terms such as C++, C#, .NET, Node.js and CI/CD.
- Extracts skills by category using literal-safe regular expressions.
- Fits and persists a TF-IDF vectorizer with Joblib.
- Calculates cosine similarity between a resume and a job description.
- Reports matched, missing, and additional skills.
- Exposes a FastAPI API with interactive Swagger documentation.
- Includes tests for valid requests and invalid inputs.

## Project structure

```text
smart-job-match-analyzer/
├── app/
│   ├── core/
│   │   └── config.py
│   ├── services/
│   │   ├── document_parser.py
│   │   ├── similarity.py
│   │   ├── skill_extractor.py
│   │   └── text_processing.py
│   ├── main.py
│   └── schemas.py
├── data/
│   └── training_corpus.json
├── tests/
│   └── test_api.py
├── train_model.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Requirements

- Python 3.10 or newer
- pip

## Setup on Windows (PowerShell)

Open a terminal in this project directory and run:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation, use Command Prompt instead:

```bat
.venv\Scripts\activate.bat
```

## Train and save the TF-IDF model

Run this once after installing dependencies:

```bash
python train_model.py
```

This creates `app/models/tfidf_vectorizer.joblib`. The model directory is generated automatically. The bundled corpus is a small demonstration corpus for fitting the vocabulary and inverse-document-frequency weights; it is not a large, benchmarked training dataset.

## Start the API

```bash
uvicorn app.main:app --reload
```

Open:

- API documentation: http://127.0.0.1:8000/docs
- Health check: http://127.0.0.1:8000/health

The health endpoint should return `model_loaded: true` after training and starting the API.

## API usage

### 1. Analyze text (JSON)

`POST /analyze`

Example request:

```json
{
  "resume_text": "Python developer with FastAPI, SQL, PostgreSQL, Git and Docker experience. Built backend APIs and wrote unit tests.",
  "job_description": "Hiring a Python backend developer with FastAPI, SQL, PostgreSQL, Git, Docker and AWS. Experience building APIs and writing tests is required."
}
```

### 2. Analyze an uploaded resume (PDF/TXT)

`POST /analyze/upload`

Use `multipart/form-data` with:

- `resume_file`: a PDF or TXT file (maximum 5 MB)
- `job_description`: job description text

Scanned/image-only PDFs are not supported because OCR is not included in this version.

### Example response

```json
{
  "match_score": 61.42,
  "score_explanation": "TF-IDF cosine similarity between the resume and job description; not a hiring probability.",
  "skill_coverage_percent": 83.33,
  "matched_skills": ["docker", "fastapi", "git", "postgresql", "python", "sql"],
  "missing_skills": ["aws"],
  "additional_resume_skills": [],
  "resume_skills_by_category": {"programming_languages": ["python"]},
  "job_skills_by_category": {"programming_languages": ["python"]},
  "note": "Dictionary-based skill extraction can miss synonyms, context, and skills not included in its vocabulary."
}
```

Numbers above are illustrative; actual results depend on the input text and fitted corpus.

## Run tests

```bash
pytest -q
```

The tests fit a small vectorizer in memory, so they do not require the generated model file. The API itself does require you to run `python train_model.py` before starting it.

## How it works

1. **Input:** Resume text (or PDF/TXT upload) and a job description.
2. **Preprocessing:** Lowercase, normalize whitespace, remove irrelevant punctuation, and map technical terms with meaningful symbols to stable tokens for similarity calculation.
3. **Skill extraction:** Match known skills with escaped regular expressions and word boundaries. Extract resume and job-description skills independently.
4. **Vectorization:** Transform both documents using the persisted TF-IDF vocabulary and IDF weights.
5. **Similarity:** Calculate cosine similarity and express it as a percentage.
6. **Gap analysis:** Compare the extracted skill sets to identify overlap and skills mentioned in the job description but not detected in the resume.
7. **API response:** Return the score, skill coverage, matched skills, missing skills, and extra resume skills.

## Limitations and future improvements

- The bundled corpus is small and intended for a reproducible baseline, not production-grade model quality.
- TF-IDF similarity is affected by wording and document length. It does not understand experience quality or semantic equivalence.
- Skill matching is dictionary-based; extend `SKILLS` in `app/services/skill_extractor.py` as you evaluate more job descriptions.
- PDF extraction works for text-based PDFs, not scanned documents.
- A future version could add synonym mapping, OCR, a larger representative corpus, ranking evaluation, and a user interface.

## Week 4 completion checklist

- [x] End-to-end preprocessing pipeline
- [x] TF-IDF vectorizer fitting and Joblib serialization
- [x] Prediction/analyze API
- [x] Resume and job-description analysis
- [x] Matched and missing skill analysis
- [x] Valid and invalid input tests
- [x] Setup, API, and limitation documentation
- [ ] Run the setup, training, API, and test commands on your machine
- [ ] Capture API demo screenshots and prepare a short project presentation

The final two items require execution and evidence from your own environment; they should not be marked complete until you have done them.
