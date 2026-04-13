# Backend (FastAPI)

## Setup

```powershell
cd GDstreamlit/backend
pip install -r requirements.txt
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

## Base URL

- `http://127.0.0.1:8000`
- Health check: `GET /api/health`

## Endpoints

- `POST /api/perceptron/predict`
- `POST /api/forward/run`
- `POST /api/backward/step`
- `POST /api/mlp/summary`
- `POST /api/rnn/sentiment`
- `GET /api/opencv/info`

## Legacy Streamlit Code

Existing Streamlit implementation is preserved in `backend/legacy_streamlit/`.
