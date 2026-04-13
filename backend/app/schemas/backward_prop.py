from pydantic import BaseModel, Field


class BackwardRequest(BaseModel):
    x: float = Field(default=0.3)
    y_true: float = Field(default=1.0)
    w: float = Field(default=0.5)
    b: float = Field(default=0.0)
    learning_rate: float = Field(default=0.1)


class BackwardResponse(BaseModel):
    y_pred: float
    loss: float
    grad_w: float
    grad_b: float
    new_w: float
    new_b: float
