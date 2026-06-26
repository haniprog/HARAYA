from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .pipeline import HarayaPipeline


# Layer 1: API Input
# Dito dumarating ang mga message  from browser.
class AnalyzeRequest(BaseModel):
    messages: list[str] = Field(default_factory=list)


class AnalyzeResponse(BaseModel):
    conversation: str
    label: str
    finalScore: float
    confidence: float
    features: dict
    reasons: list[str]
    legalBasis: list[str]
    recommendations: list[str]


@lru_cache(maxsize=1)
def get_pipeline() -> HarayaPipeline:
    # this will load the model para hindi bumigat ang bawat request.
    model_path = Path(__file__).resolve().parents[1] / "artifacts" / "haraya_bert.pt"
    return HarayaPipeline(model_path=model_path if model_path.exists() else None)


app = FastAPI(title="HARAYA API", version="0.1.0")

# this will allow to call the api from any frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health_check() -> dict[str, str]:
    # Simpleng check lng kung buhay ang API.
    return {"status": "ok"}


# Layer 7: API Response
@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(request: AnalyzeRequest) -> AnalyzeResponse:
    # Ito ang actual prediction flow mula request hanggang result.
    result = get_pipeline().predict(request.messages)
    return AnalyzeResponse(**result.__dict__)
