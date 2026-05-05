import io
import json
import os
import uuid
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from app.schemas.data_engine import (
    DataEngineAnalysisRequest,
    DataEngineAnalysisResponse,
    DataEngineColumnInfo,
    DataEngineFeatureDetection,
    DataEnginePreviewResponse,
    DataEngineRecommendations,
)

NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
NVIDIA_MODEL = "nvidia/nemotron-3-nano-30b-a3b"
ENV_FILE_PATH = Path(__file__).resolve().parents[2] / ".env"

# Predefined analysis prompts for NVIDIA API
ANALYSIS_PROMPTS = {
    "understanding": """Provide a CONCISE dataset overview in 3-4 bullet points:
- Key dimensions and data completeness (rows, columns, null count)
- Data types breakdown (numeric, categorical, datetime)
- One standout statistical observation (e.g., price range, distribution skew)
- Any immediate data quality flags (if present)

Keep each bullet to ONE line. No verbose explanations.""",
    
    "problem_type": """Identify the problem type in 2-3 sentences MAX:
1. State the problem type (Regression/Classification/Clustering/Time Series/Anomaly Detection)
2. Give ONE key reason from the data (target type, cardinality, feature count, etc.)
3. Mention ONE relevant consequence for model selection (if applicable)

Be direct and concise.""",
    
    "feature_engineering": """Suggest 3-5 practical feature engineering steps as a SHORT list:
- Each step as ONE line (e.g., "Log-transform size to stabilize variance")
- Include the reason if it applies to THIS dataset specifically
- Focus on high-impact transformations only
- Skip generic advice; be data-driven

NO detailed code examples; just action items.""",
    
    "preprocessing": """Recommend 3-4 preprocessing steps as a SHORT checklist:
- Missing value handling (if any, by column)
- Outlier treatment (if needed, method only)
- Scaling/normalization (type: standardize, min-max, none?)
- One validation check to perform before training

ONE sentence per item. Tailored to actual data characteristics.""",
    
    "model_recommendations": """Recommend TOP 3 models in a crisp format:
For each model: "Model Name - Why it fits (1 reason only)"
Example: "Linear Regression - Few features and linear pattern expected"

Then add ONE line: "Best cross-validation strategy: [KFold/StratifiedKFold/TimeSeriesSplit]"

No hyperparameter ranges or lengthy explanations; just model names and one-line justifications.""",
}


def _read_local_env(name: str) -> str:
    if not ENV_FILE_PATH.exists():
        return ""

    try:
        for raw_line in ENV_FILE_PATH.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            if key.strip() == name:
                return value.strip().strip('"').strip("'")
    except OSError:
        return ""

    return ""


def _detect_column_types(df: pd.DataFrame) -> dict[str, str]:
    """Detect data types for each column with strict datetime detection."""
    types = {}
    for col in df.columns:
        series = df[col].dropna()
        
        if series.empty:
            types[col] = "unknown"
            continue
        
        # Check for numerical FIRST (before datetime, to avoid misclassifying numeric as datetime)
        if pd.api.types.is_numeric_dtype(series):
            if pd.api.types.is_integer_dtype(series):
                types[col] = "integer"
            else:
                types[col] = "float"
            continue
        
        # Check for datetime ONLY on object/string columns (not numeric)
        if pd.api.types.is_object_dtype(series) or pd.api.types.is_categorical_dtype(series):
            # Try to parse as datetime, but only if it looks date-like (contains common separators)
            sample = series.head(10)
            date_like_patterns = ['/', '-', ':', 'T', ' ', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            
            looks_like_date = any(
                any(pattern in str(val) for pattern in date_like_patterns)
                for val in sample if pd.notna(val)
            )
            
            if looks_like_date:
                try:
                    pd.to_datetime(sample)
                    types[col] = "datetime"
                    continue
                except (ValueError, TypeError):
                    pass
            
            # Check for categorical vs text
            unique_ratio = len(series.unique()) / len(series)
            if unique_ratio < 0.05 or len(series.unique()) < 20:
                types[col] = "categorical"
            else:
                types[col] = "text"
        else:
            types[col] = "other"
    
    return types


def _calculate_column_stats(df: pd.DataFrame, col_type: str, col: str) -> dict[str, Any]:
    """Calculate statistics for a column based on its type."""
    stats = {
        "missing_count": df[col].isna().sum(),
        "missing_percentage": (df[col].isna().sum() / len(df)) * 100,
        "unique_count": df[col].nunique(),
    }
    
    if col_type in ["integer", "float"]:
        stats["mean"] = float(df[col].mean())
        stats["median"] = float(df[col].median())
        stats["std"] = float(df[col].std())
        stats["min"] = float(df[col].min())
        stats["max"] = float(df[col].max())
        stats["q25"] = float(df[col].quantile(0.25))
        stats["q75"] = float(df[col].quantile(0.75))
    elif col_type in ["categorical", "text"]:
        value_counts = df[col].value_counts().head(10)
        stats["top_values"] = value_counts.to_dict()
    
    return stats


def _auto_detect_target_column(df: pd.DataFrame, col_types: dict[str, str]) -> str | None:
    """Auto-detect the most likely target column based on heuristics."""
    candidates = []
    
    for col in df.columns:
        col_type = col_types[col]
        
        # Prioritize columns that look like targets
        col_lower = col.lower()
        
        # Check if column name suggests it's a target
        target_keywords = ["target", "label", "output", "prediction", "result", "class", "category", 
                          "diagnosis", "outcome", "result_label"]
        if any(keyword in col_lower for keyword in target_keywords):
            candidates.append((col, 100))
            continue
        
        # Categorical columns with reasonable cardinality are good classification targets
        if col_type == "categorical":
            unique_count = df[col].nunique()
            if 2 <= unique_count <= 50:
                candidates.append((col, 80 - unique_count + 2))
                continue
        
        # Numerical columns might be regression targets
        if col_type in ["integer", "float"]:
            candidates.append((col, 50))
            continue
    
    if candidates:
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[0][0]
    
    return None


def _analyze_dataset_for_ai(df: pd.DataFrame, target_col: str | None, col_types: dict[str, str]) -> str:
    """Format dataset information for AI analysis."""
    info = {
        "shape": df.shape,
        "columns": list(df.columns),
        "column_types": col_types,
        "target_column": target_col,
        "sample_data": df.head(5).to_dict("records"),
        "missing_summary": df.isna().sum().to_dict(),
        "statistics": {}
    }
    
    for col in df.columns:
        info["statistics"][col] = _calculate_column_stats(df, col_types[col], col)
    
    return json.dumps(info, indent=2, default=str)


def _call_nvidia_api(api_key: str, analysis_type: str, dataset_info: str) -> str:
    """Call NVIDIA API for analysis."""
    prompt = ANALYSIS_PROMPTS.get(analysis_type, "Analyze this dataset and provide insights.")
    
    messages = [
        {
            "role": "system",
            "content": "You are an expert data scientist and machine learning engineer. Provide specific, actionable insights based on data properties. Avoid generic advice."
        },
        {
            "role": "user",
            "content": f"{prompt}\n\nDataset Information:\n{dataset_info}"
        }
    ]
    
    return _request_chat_completion(api_key=api_key, messages=messages, temperature=0.5, max_tokens=2000)


def _request_chat_completion(api_key: str, messages: list[dict[str, str]], temperature: float, max_tokens: int) -> str:
    payload = {
        "model": NVIDIA_MODEL,
        "messages": messages,
        "temperature": temperature,
        "top_p": 0.95,
        "max_tokens": max_tokens,
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    try:
        response = requests.post(
            f"{NVIDIA_BASE_URL}/chat/completions",
            headers=headers,
            json=payload,
            timeout=60,
        )
        response.raise_for_status()
        body = response.json()
        
        choices = body.get("choices", [])
        if not choices or not isinstance(choices[0], dict):
            return "Unable to generate analysis."

        choice = choices[0]
        message = choice.get("message", {})
        return (message.get("content") or "").strip()
    except requests.RequestException as e:
        raise RuntimeError(f"NVIDIA API request failed: {str(e)}")


def ask_followup(api_key: str, system_context: str, conversation_history: list[dict[str, str]], question: str) -> str:
    messages: list[dict[str, str]] = [{"role": "system", "content": system_context}]
    messages.extend(conversation_history)
    messages.append({"role": "user", "content": question})
    return _request_chat_completion(api_key=api_key, messages=messages, temperature=0.65, max_tokens=1500)


def parse_csv_file(file_content: bytes) -> pd.DataFrame:
    """Parse CSV file content into DataFrame."""
    try:
        df = pd.read_csv(io.BytesIO(file_content))
        return df
    except Exception as e:
        raise ValueError(f"Failed to parse CSV: {str(e)}")


def generate_preview(df: pd.DataFrame, col_types: dict[str, str], session_id: str) -> DataEnginePreviewResponse:
    """Generate data preview response."""
    columns_info = []
    for col in df.columns:
        col_info = DataEngineColumnInfo(
            name=col,
            type=col_types[col],
            non_null_count=int(df[col].notna().sum()),
            null_count=int(df[col].isna().sum()),
        )
        columns_info.append(col_info)
    
    preview_rows = df.head(10).to_dict("records")
    
    target_col = _auto_detect_target_column(df, col_types)
    
    return DataEnginePreviewResponse(
        session_id=session_id,
        rows=len(df),
        columns=len(df.columns),
        columns_info=columns_info,
        preview_data=preview_rows,
        suggested_target_column=target_col,
    )


def detect_features(df: pd.DataFrame, target_col: str, col_types: dict[str, str]) -> DataEngineFeatureDetection:
    """Detect and categorize features."""
    numerical_features = []
    categorical_features = []
    datetime_features = []
    
    for col in df.columns:
        if col == target_col:
            continue
        
        col_type = col_types[col]
        if col_type in ["integer", "float"]:
            numerical_features.append(col)
        elif col_type == "categorical":
            categorical_features.append(col)
        elif col_type == "datetime":
            datetime_features.append(col)
    
    return DataEngineFeatureDetection(
        numerical_features=numerical_features,
        categorical_features=categorical_features,
        datetime_features=datetime_features,
        total_features=len(numerical_features) + len(categorical_features) + len(datetime_features),
    )


def perform_analysis(req: DataEngineAnalysisRequest, df: pd.DataFrame) -> DataEngineAnalysisResponse:
    """Perform comprehensive analysis using NVIDIA API."""
    api_key = (os.getenv("NVIDIA_API_KEY", "") or _read_local_env("NVIDIA_API_KEY")).strip()
    if not api_key:
        raise RuntimeError("NVIDIA_API_KEY is not configured.")
    
    col_types = _detect_column_types(df)
    dataset_info = _analyze_dataset_for_ai(df, req.target_column, col_types)
    
    # Perform all predefined analyses
    analyses = {}
    for analysis_type in ["understanding", "problem_type", "feature_engineering", "preprocessing", "model_recommendations"]:
        analyses[analysis_type] = _call_nvidia_api(api_key, analysis_type, dataset_info)
    
    # Generate recommendations
    recommendations = DataEngineRecommendations(
        primary_problem_type=_extract_primary_problem_type(analyses["problem_type"]),
        suggested_models=_extract_suggested_models(analyses["model_recommendations"]),
        feature_engineering_suggestions=analyses["feature_engineering"][:500],
        data_quality_issues=_extract_data_quality_issues(df, col_types),
    )
    
    return DataEngineAnalysisResponse(
        analysis_id=str(uuid.uuid4()),
        understanding=analyses["understanding"],
        problem_type=analyses["problem_type"],
        feature_engineering=analyses["feature_engineering"],
        preprocessing=analyses["preprocessing"],
        model_recommendations=analyses["model_recommendations"],
        recommendations=recommendations,
    )


def _extract_primary_problem_type(problem_analysis: str) -> str:
    """Extract primary problem type from analysis."""
    types = ["Classification", "Regression", "Clustering", "Time Series", "Anomaly Detection"]
    for ptype in types:
        if ptype.lower() in problem_analysis.lower():
            return ptype
    return "Unsupervised Learning"


def _extract_suggested_models(model_analysis: str) -> list[str]:
    """Extract suggested models from analysis."""
    models = []
    common_models = [
        "Random Forest", "XGBoost", "LightGBM", "Gradient Boosting",
        "Linear Regression", "Ridge", "Lasso", "SVM", "Logistic Regression",
        "Neural Network", "Deep Learning", "LSTM", "CNN", "Ensemble",
        "KMeans", "DBSCAN", "Isolation Forest", "Decision Tree", "Naive Bayes"
    ]
    
    for model in common_models:
        if model.lower() in model_analysis.lower():
            models.append(model)
    
    return models[:5]


def _extract_data_quality_issues(df: pd.DataFrame, col_types: dict[str, str]) -> list[str]:
    """Identify data quality issues."""
    issues = []
    
    # Check for missing values
    missing_threshold = 0.3
    for col in df.columns:
        missing_pct = df[col].isna().sum() / len(df)
        if missing_pct > missing_threshold:
            issues.append(f"High missing value rate ({missing_pct*100:.1f}%) in {col}")
    
    # Check for duplicates
    dup_pct = df.duplicated().sum() / len(df)
    if dup_pct > 0.05:
        issues.append(f"Duplicate rows detected ({dup_pct*100:.1f}%)")
    
    # Check for potential outliers (numerical columns)
    for col in df.columns:
        if col_types[col] in ["integer", "float"]:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
            if len(outliers) > 0:
                outlier_pct = len(outliers) / len(df)
                if outlier_pct > 0.05:
                    issues.append(f"Potential outliers ({outlier_pct*100:.1f}%) in {col}")
    
    return issues if issues else ["No major data quality issues detected"]


def generate_report(df: pd.DataFrame, analysis: DataEngineAnalysisResponse, target_col: str) -> str:
    """Generate a downloadable report."""
    report = f"""# Data Analysis Report
    
Generated Analysis ID: {analysis.analysis_id}
Target Column: {target_col}

## Dataset Overview
- Rows: {len(df)}
- Columns: {len(df.columns)}

## Understanding
{analysis.understanding}

## Problem Type
{analysis.problem_type}

## Feature Engineering Recommendations
{analysis.feature_engineering}

## Preprocessing Steps
{analysis.preprocessing}

## Model Recommendations
{analysis.model_recommendations}

## Key Recommendations
- Primary Problem Type: {analysis.recommendations.primary_problem_type}
- Suggested Models: {', '.join(analysis.recommendations.suggested_models)}
- Data Quality Issues: {', '.join(analysis.recommendations.data_quality_issues)}

---
Generated by DeepNexus Data Engine
"""
    return report
