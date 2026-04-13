from fastapi import APIRouter

from app.schemas.mlp import MLPConfigRequest, MLPConfigResponse
from app.services.mlp_service import summarize_mlp

router = APIRouter()


@router.post("/summary", response_model=MLPConfigResponse)
def summary(req: MLPConfigRequest) -> MLPConfigResponse:
    return summarize_mlp(req)
