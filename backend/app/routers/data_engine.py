import os
import uuid
from fastapi import APIRouter, HTTPException, UploadFile, File

from app.schemas.data_engine import (
    DataEngineAnalysisRequest,
    DataEngineAskRequest,
    DataEngineAskResponse,
    DataEnginePreviewResponse,
    DataEngineResultsResponse,
    DataEngineFeatureDetection,
)
from app.services.data_engine_service import (
    parse_csv_file,
    generate_preview,
    detect_features,
    perform_analysis,
    _detect_column_types,
    generate_report,
    ask_followup,
    _read_local_env,
)

router = APIRouter()

# Store DataFrames and analyses temporarily (in production, use database)
_temp_dataframes = {}
_temp_analyses = {}

@router.post("/upload", response_model=DataEnginePreviewResponse)
async def upload_csv(file: UploadFile = File(...)) -> DataEnginePreviewResponse:
    """Upload and preview CSV file."""
    try:
        if not file.filename or not file.filename.lower().endswith('.csv'):
            raise HTTPException(status_code=400, detail="Only CSV files are allowed")
        
        content = await file.read()
        df = parse_csv_file(content)
        
        if df.empty:
            raise HTTPException(status_code=400, detail="CSV file is empty")
        
        # Store DataFrame temporarily
        session_id = str(uuid.uuid4())
        _temp_dataframes[session_id] = df
        
        # Detect column types
        col_types = _detect_column_types(df)
        
        # Generate preview
        preview = generate_preview(df, col_types, session_id)
        
        return preview
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@router.post("/detect-features")
def get_feature_detection(session_id: str, target_column: str) -> DataEngineFeatureDetection:
    """Detect features for the dataset."""
    try:
        if session_id not in _temp_dataframes:
            raise HTTPException(status_code=404, detail="Session not found")
        
        df = _temp_dataframes[session_id]
        col_types = _detect_column_types(df)
        
        features = detect_features(df, target_column, col_types)
        return features
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Feature detection failed: {str(e)}")


@router.post("/analyze", response_model=DataEngineResultsResponse)
def analyze_dataset(session_id: str, target_column: str) -> DataEngineResultsResponse:
    """Perform comprehensive analysis on the dataset."""
    try:
        if session_id not in _temp_dataframes:
            raise HTTPException(status_code=404, detail="Session not found")
        
        df = _temp_dataframes[session_id]
        
        if target_column not in df.columns:
            raise HTTPException(status_code=400, detail="Target column not found in dataset")
        
        # Perform analysis
        req = DataEngineAnalysisRequest(
            file_content=b"",  # Not needed here
            target_column=target_column
        )
        analysis = perform_analysis(req, df)
        
        # Store analysis
        _temp_analyses[analysis.analysis_id] = {
            "df": df,
            "analysis": analysis,
            "target_column": target_column
        }
        
        # Detect features
        col_types = _detect_column_types(df)
        features = detect_features(df, target_column, col_types)
        
        return DataEngineResultsResponse(
            rows=len(df),
            columns=len(df.columns),
            target_column=target_column,
            analysis=analysis,
            features=features
        )
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@router.post("/ask", response_model=DataEngineAskResponse)
def ask_data_engine(req: DataEngineAskRequest) -> DataEngineAskResponse:
    """Interactive AI assistant for data exploration."""
    try:
        if req.analysis_id not in _temp_analyses:
            raise HTTPException(status_code=404, detail="Analysis not found")
        
        api_key = (os.getenv("NVIDIA_API_KEY", "") or _read_local_env("NVIDIA_API_KEY")).strip()
        if not api_key:
            raise HTTPException(status_code=503, detail="NVIDIA API not configured")
        
        stored_data = _temp_analyses[req.analysis_id]
        analysis = stored_data["analysis"]
        
        # Build conversation context
        system_context = f"""You are a data science expert helping users explore and understand their dataset.
The dataset has {stored_data['df'].shape[0]} rows and {stored_data['df'].shape[1]} columns.
Target column: {stored_data['target_column']}

Previous analysis:
- Problem Type: {analysis.recommendations.primary_problem_type}
- Suggested Models: {', '.join(analysis.recommendations.suggested_models)}
- Key Issues: {', '.join(analysis.recommendations.data_quality_issues[:2])}

Help the user with data exploration, answer questions about the data, and provide actionable insights."""
        
        answer = ask_followup(
            api_key=api_key,
            system_context=system_context,
            conversation_history=req.conversation_history,
            question=req.question,
        )
        
        return DataEngineAskResponse(
            answer=answer,
            analysis_id=req.analysis_id
        )
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ask failed: {str(e)}")


@router.get("/download-report/{analysis_id}")
def download_report(analysis_id: str):
    """Generate and download analysis report."""
    try:
        if analysis_id not in _temp_analyses:
            raise HTTPException(status_code=404, detail="Analysis not found")
        
        stored_data = _temp_analyses[analysis_id]
        df = stored_data["df"]
        analysis = stored_data["analysis"]
        target_col = stored_data["target_column"]
        
        report = generate_report(df, analysis, target_col)
        
        return {
            "content": report,
            "filename": f"data_analysis_{analysis_id}.md"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Report generation failed: {str(e)}")
