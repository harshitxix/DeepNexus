from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware

from app.routers import (
    backward_prop,
    forward_prop,
    health,
    hopfield,
    mlp,
    opencv,
    perceptron,
    rnn,
    study_assistant,
    data_engine,
)

app = FastAPI(title="Deep Learning API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router, prefix="/api")
app.include_router(perceptron.router, prefix="/api/perceptron", tags=["perceptron"])
app.include_router(forward_prop.router, prefix="/api/forward", tags=["forward"])
app.include_router(backward_prop.router, prefix="/api/backward", tags=["backward"])
app.include_router(mlp.router, prefix="/api/mlp", tags=["mlp"])
app.include_router(rnn.router, prefix="/api/rnn", tags=["rnn"])
app.include_router(opencv.router, prefix="/api/cnn-lab", tags=["cnn-lab"])
app.include_router(hopfield.router, prefix="/api/hopfield", tags=["hopfield"])
app.include_router(study_assistant.router, prefix="/api/study-assistant", tags=["study-assistant"])
app.include_router(data_engine.router, prefix="/api/data-engine", tags=["data-engine"])


@app.get("/")
def root() -> dict:
    return {
        "message": "DeepNexus backend is running.",
        "health": "/api/health",
        "docs": "/docs",
    }


@app.get("/favicon.ico", include_in_schema=False)
def favicon() -> Response:
    return Response(status_code=204)
