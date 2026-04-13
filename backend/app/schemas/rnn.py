from pydantic import BaseModel, Field


class RNNLabInfoResponse(BaseModel):
    module_name: str
    tasks: list[str]
    lstm_model_type: str
    bert_model_id: str
    emotion_labels: list[str]


class TextGenerationRequest(BaseModel):
    text: str = Field(default="the deep learning model", min_length=1, max_length=500)
    top_k: int = Field(default=5, ge=1, le=5)
    generate_words: int = Field(default=5, ge=1, le=10)
    temperature: float = Field(default=1.0, gt=0.0, le=2.0)


class NextWordSuggestion(BaseModel):
    word: str
    confidence: float


class TextGenerationResponse(BaseModel):
    engine: str
    input_text: str
    suggestions: list[NextWordSuggestion]
    generated_text: str


class EmotionAnalysisRequest(BaseModel):
    text: str = Field(default="I am feeling great today", min_length=1, max_length=500)


class EmotionProbability(BaseModel):
    label: str
    probability: float


class EmotionAnalysisResponse(BaseModel):
    engine: str
    predicted_emotion: str
    confidence: float
    probabilities: list[EmotionProbability]
