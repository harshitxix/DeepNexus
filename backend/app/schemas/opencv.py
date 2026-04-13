from pydantic import BaseModel


class CNNLabInfoResponse(BaseModel):
    module_name: str
    supported_models: list[str]
    input_size: int
    endpoints: list[str]


class PredictionItem(BaseModel):
    class_label: str
    confidence: float


class CNNLabPredictResponse(BaseModel):
    model_id: str
    top_predictions: list[PredictionItem]
    input_image: str


class FeatureMapItem(BaseModel):
    layer_name: str
    stage: str
    grid_image: str


class CNNLabFeatureMapsResponse(BaseModel):
    model_id: str
    feature_maps: list[FeatureMapItem]


class CNNLabGradCamResponse(BaseModel):
    model_id: str
    predicted_label: str
    confidence: float
    heatmap_image: str
    overlay_image: str
