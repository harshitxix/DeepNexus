from pydantic import BaseModel, Field


class StudyAssistantAskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    module_key: str = Field(min_length=1, max_length=80)
    detailed: bool = Field(default=False)
    session_id: str | None = Field(default=None, max_length=120)


class StudyAssistantAskResponse(BaseModel):
    session_id: str
    answer: str
    sources: list[str]
    max_tokens_used: int
    relevant_module_key: str
    relevant_module_title: str
    is_current_module: bool
