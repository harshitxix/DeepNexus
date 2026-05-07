from fastapi import APIRouter

router = APIRouter()


@router.get("/health")
def health_check() -> dict:
    return {"status": "ok", "service": "deep-learning-api"}


@router.get("/ping")
def ping() -> dict:
    return {"status": "alive"}
