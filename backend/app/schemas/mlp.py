from pydantic import BaseModel, Field


class MLPConfigRequest(BaseModel):
    layer_sizes: list[int] = Field(default=[2, 4, 1])


class MLPConfigResponse(BaseModel):
    total_parameters: int
    summary: str
