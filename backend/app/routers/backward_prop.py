from fastapi import APIRouter

from app.schemas.backward_prop import BackwardRequest, BackwardResponse
from app.services.backward_service import run_backward_step

router = APIRouter()


@router.post("/step", response_model=BackwardResponse)
def step(req: BackwardRequest) -> BackwardResponse:
    return run_backward_step(req)
