from fastapi import APIRouter
from fastapi.responses import PlainTextResponse

router = APIRouter()


@router.get("/health")
def health_check() -> dict:
    return {"status": "ok", "service": "deep-learning-api"}


@router.api_route("/ping", methods=["GET", "HEAD"])
async def ping():
    return PlainTextResponse("ok", status_code=200)
