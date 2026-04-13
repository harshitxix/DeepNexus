from pydantic import BaseModel, Field


class PerceptronRequest(BaseModel):
    x1: float = Field(default=0.3)
    x2: float = Field(default=0.3)
    w1: float = Field(default=0.5)
    w2: float = Field(default=0.5)
    bias: float = Field(default=0.0)
    activation: str = Field(default="sigmoid")
    threshold: float = Field(default=0.5)


class PerceptronResponse(BaseModel):
    z: float
    a: float
    predicted_class: int
