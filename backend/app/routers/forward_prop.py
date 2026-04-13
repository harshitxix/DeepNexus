from fastapi import APIRouter

from app.schemas.forward_prop import ForwardRequest, ForwardResponse
from app.services.forward_service import run_forward

router = APIRouter()


@router.post("/run", response_model=ForwardResponse)
def run(req: ForwardRequest) -> ForwardResponse:
    return run_forward(req)
