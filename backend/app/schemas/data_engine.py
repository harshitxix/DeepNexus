from pydantic import BaseModel, ConfigDict, Field


class DataEngineColumnInfo(BaseModel):
    name: str
    type: str
    non_null_count: int
    null_count: int


class DataEnginePreviewResponse(BaseModel):
    session_id: str
    rows: int
    columns: int
    columns_info: list[DataEngineColumnInfo]
    preview_data: list[dict]
    suggested_target_column: str | None


class DataEngineFeatureDetection(BaseModel):
    numerical_features: list[str]
    categorical_features: list[str]
    datetime_features: list[str]
    total_features: int


class DataEngineRecommendations(BaseModel):
    primary_problem_type: str
    suggested_models: list[str]
    feature_engineering_suggestions: str
    data_quality_issues: list[str]


class DataEngineAnalysisResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    analysis_id: str
    understanding: str
    problem_type: str
    feature_engineering: str
    preprocessing: str
    model_recommendations: str
    recommendations: DataEngineRecommendations


class DataEngineAnalysisRequest(BaseModel):
    file_content: bytes
    target_column: str


class DataEngineResultsResponse(BaseModel):
    rows: int
    columns: int
    target_column: str
    analysis: DataEngineAnalysisResponse
    features: DataEngineFeatureDetection


class DataEngineAskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)
    analysis_id: str
    conversation_history: list[dict] = Field(default_factory=list)


class DataEngineAskResponse(BaseModel):
    answer: str
    analysis_id: str
