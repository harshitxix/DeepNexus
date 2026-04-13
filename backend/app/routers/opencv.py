from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.schemas.opencv import (
    CNNLabFeatureMapsResponse,
    CNNLabGradCamResponse,
    CNNLabInfoResponse,
    CNNLabPredictResponse,
)
from app.services import cnn_lab_service

router = APIRouter()


@router.get("/info", response_model=CNNLabInfoResponse)
def info() -> CNNLabInfoResponse:
    return CNNLabInfoResponse(
        module_name="CNN Lab",
        supported_models=["efficientnet_b0", "resnet50", "mobilenet_v2"],
        input_size=224,
        endpoints=["/predict", "/feature-maps", "/grad-cam"],
    )


@router.post("/predict", response_model=CNNLabPredictResponse)
async def predict(
    image: UploadFile = File(...),
    model_id: str = Form("efficientnet_b0"),
    top_k: int = Form(3),
) -> CNNLabPredictResponse:
    try:
        image_bytes = await image.read()
        payload = cnn_lab_service.predict_image(image_bytes, model_name=model_id, top_k=top_k)
        return CNNLabPredictResponse(**payload)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/feature-maps", response_model=CNNLabFeatureMapsResponse)
async def feature_maps(
    image: UploadFile = File(...),
    model_id: str = Form("efficientnet_b0"),
    max_maps: int = Form(16),
) -> CNNLabFeatureMapsResponse:
    try:
        image_bytes = await image.read()
        payload = cnn_lab_service.extract_feature_maps(image_bytes, model_name=model_id, max_maps=max_maps)
        return CNNLabFeatureMapsResponse(**payload)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/grad-cam", response_model=CNNLabGradCamResponse)
async def grad_cam(
    image: UploadFile = File(...),
    model_id: str = Form("efficientnet_b0"),
) -> CNNLabGradCamResponse:
    try:
        image_bytes = await image.read()
        payload = cnn_lab_service.generate_grad_cam(image_bytes, model_name=model_id)
        return CNNLabGradCamResponse(**payload)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
