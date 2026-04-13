from pydantic import BaseModel, Field


class HopfieldRecallRequest(BaseModel):
    pattern: list[int] = Field(default_factory=list)
    side: int = Field(default=15, ge=2, le=25)
    steps: int = Field(default=500, ge=1, le=5000)
    confidence_threshold: float = Field(default=0.4, ge=0.0, le=1.0)


class HopfieldRecallResponse(BaseModel):
    label: str
    confidence: float
    matched_label: str
    is_unknown: bool
    iterations: int
    converged: bool
    energy_before: float
    energy_after: float
    recalled_pattern: list[int]


class HopfieldLibraryResponse(BaseModel):
    side: int
    labels: list[str]
    total_patterns: int
