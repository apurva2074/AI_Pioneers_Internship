from pydantic import BaseModel, Field, field_validator

from app.core.config import MIN_TEXT_LENGTH


class AnalyzeTextRequest(BaseModel):
    resume_text: str = Field(..., min_length=MIN_TEXT_LENGTH)
    job_description: str = Field(..., min_length=MIN_TEXT_LENGTH)

    @field_validator("resume_text", "job_description")
    @classmethod
    def reject_blank_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Text cannot be blank.")
        return value.strip()
