from pydantic import BaseModel, Field


class ForwardRequest(BaseModel):
    inputs: list[float] = Field(default=[0.3, 0.7])
    weights: list[list[float]] = Field(default=[[0.5, -0.2], [0.1, 0.4]])
    bias: list[float] = Field(default=[0.0, 0.1])
    activation: str = Field(default="sigmoid")


class ForwardResponse(BaseModel):
    z: list[float]
    a: list[float]
