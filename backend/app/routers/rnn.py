from fastapi import APIRouter

from app.schemas.rnn import (
    EmotionAnalysisRequest,
    EmotionAnalysisResponse,
    RNNLabInfoResponse,
    TextGenerationRequest,
    TextGenerationResponse,
)
from app.services.rnn_service import (
    analyze_emotion,
    generate_next_words,
    get_rnn_lab_info,
)

router = APIRouter()


@router.get("/info", response_model=RNNLabInfoResponse)
def info() -> RNNLabInfoResponse:
    return get_rnn_lab_info()


@router.post("/generate-text", response_model=TextGenerationResponse)
def generate_text(req: TextGenerationRequest) -> TextGenerationResponse:
    return generate_next_words(req)


@router.post("/analyze-emotion", response_model=EmotionAnalysisResponse)
def analyze(req: EmotionAnalysisRequest) -> EmotionAnalysisResponse:
    return analyze_emotion(req)
