from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


Category = Literal[
    "Hygiene",
    "Processing",
    "Storage/Temperature",
    "Packaging/Labeling",
    "Documentation/Records",
    "Pest Control",
    "Personnel/Training",
    "Other",
]


class OperationalClaim(BaseModel):
    category: Category
    text: str
    source_excerpt: str | None = None


class RetrievedClause(BaseModel):
    text: str
    source: str
    chunk_id: str
    score: float | None = None


class GapFinding(BaseModel):
    category: Category
    claim: str
    clause: str
    source: str
    severity: Literal["Critical", "Major", "Minor"]
    recommendation: str


class Scorecard(BaseModel):
    overall_percent: float = Field(ge=0, le=100)
    by_category: dict[Category, float]


class AuditReport(BaseModel):
    facility_filename: str
    created_at: datetime
    claims: list[OperationalClaim]
    gaps: list[GapFinding]
    scorecard: Scorecard
    notes: list[str] = Field(default_factory=list)
