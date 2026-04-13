from fastapi import APIRouter

from app.schemas.perceptron import PerceptronRequest, PerceptronResponse
from app.services.perceptron_service import run_perceptron

router = APIRouter()


@router.post("/predict", response_model=PerceptronResponse)
def predict(req: PerceptronRequest) -> PerceptronResponse:
    return run_perceptron(req)
