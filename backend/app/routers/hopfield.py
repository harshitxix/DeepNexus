from fastapi import APIRouter, HTTPException

from app.schemas.hopfield import HopfieldLibraryResponse, HopfieldRecallRequest, HopfieldRecallResponse
from app.services.hopfield_service import get_hopfield_library, recall_hopfield

router = APIRouter()


@router.get("/library", response_model=HopfieldLibraryResponse)
def library(side: int = 10) -> HopfieldLibraryResponse:
    return get_hopfield_library(side)


@router.post("/recall", response_model=HopfieldRecallResponse)
def recall(req: HopfieldRecallRequest) -> HopfieldRecallResponse:
    try:
        return recall_hopfield(req)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
