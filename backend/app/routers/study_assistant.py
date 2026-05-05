from fastapi import APIRouter, HTTPException

from app.schemas.study_assistant import StudyAssistantAskRequest, StudyAssistantAskResponse
from app.services.study_assistant_service import ask_study_assistant

router = APIRouter()


@router.post("/ask", response_model=StudyAssistantAskResponse)
def ask(req: StudyAssistantAskRequest) -> StudyAssistantAskResponse:
    try:
        return ask_study_assistant(req)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Study assistant upstream failure: {exc}") from exc
