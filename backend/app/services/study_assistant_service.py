import os
import re
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any

import requests

from app.schemas.study_assistant import StudyAssistantAskRequest, StudyAssistantAskResponse

NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
NVIDIA_MODEL = "nvidia/nemotron-3-nano-30b-a3b"
HISTORY_RETENTION_SECONDS = 7 * 24 * 60 * 60
ENV_FILE_PATH = Path(__file__).resolve().parents[2] / ".env"

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
HISTORY_DB_PATH = DATA_DIR / "study_assistant_history.db"

# Study-only grounded knowledge base.
STUDY_CONTENT: dict[str, dict[str, Any]] = {
    "deep_learning_foundations": {
        "title": "Deep Learning Foundations",
        "chunks": [
            "Deep learning combines multiple neural layers to learn hierarchical representations from data.",
            "Backpropagation and gradient descent are core training mechanisms for neural networks.",
            "Overfitting, underfitting, normalization, dropout, and regularization are common training concerns.",
            "Activation functions such as ReLU, sigmoid, and tanh shape how neural units learn non-linear patterns.",
        ],
        "tags": {
            "keywords": ["deep learning", "neural network", "gradient descent", "dropout", "normalization", "activation"],
            "difficulty": "Beginner",
        },
    },
    "perceptron_mlp": {
        "title": "Perceptron + MLP",
        "chunks": [
            "Perceptron is a basic neural unit for binary classification using weighted sum, bias, and activation.",
            "Single-layer perceptron learns linear boundaries only; MLP uses hidden layers for non-linear patterns.",
            "Key formula: y = f(sum(w_i * x_i) + b).",
            "Common issues include overfitting with too many layers and feature scaling problems.",
        ],
        "tags": {
            "keywords": ["perceptron", "MLP", "decision boundary", "relu", "sigmoid"],
            "difficulty": "Beginner",
        },
    },
    "neural_flow_engine": {
        "title": "Neural Cycle",
        "chunks": [
            "Neural learning cycle follows forward pass, loss computation, backpropagation, and weight update.",
            "Loss curves across epochs indicate learning quality and training stability.",
            "Weight update formula: W = W - eta * dL/dW.",
            "Learning rate and data normalization strongly affect convergence.",
        ],
        "tags": {
            "keywords": ["forward pass", "loss", "backpropagation", "epochs", "weight update"],
            "difficulty": "Beginner",
        },
    },
    "backpropagation_deep_dive": {
        "title": "Backpropagation: Deep Dive",
        "chunks": [
            "Backpropagation computes gradients of loss with respect to each weight using chain rule.",
            "Gradient flow can vanish or explode across many layers, affecting training stability.",
            "Backprop provides gradients; optimizer applies parameter updates.",
            "ReLU often helps reduce vanishing gradient issues compared to sigmoid in deep stacks.",
        ],
        "tags": {
            "keywords": ["backpropagation", "chain rule", "gradient", "vanishing", "exploding"],
            "difficulty": "Intermediate",
        },
    },
    "cnn_lab": {
        "title": "CNN",
        "chunks": [
            "CNNs use convolution filters to learn spatial features from images such as edges and textures.",
            "Pooling reduces spatial dimensions while retaining salient information.",
            "Feature hierarchy evolves from low-level edges to object-level representations.",
            "CNNs are effective for image classification and object detection, but computationally expensive.",
        ],
        "tags": {
            "keywords": ["cnn", "convolution", "pooling", "feature map", "image classification"],
            "difficulty": "Intermediate",
        },
    },
    "rnn": {
        "title": "RNN",
        "chunks": [
            "RNNs process sequential data by carrying hidden state memory through time steps.",
            "Hidden state update combines current input and previous state: h_t = f(W_x x_t + W_h h_(t-1) + b).",
            "RNNs are useful for text, speech, and time-series, but struggle with long dependencies.",
            "LSTM and GRU are common upgrades to improve long-range memory retention.",
        ],
        "tags": {
            "keywords": ["rnn", "sequence", "hidden state", "lstm", "time series"],
            "difficulty": "Intermediate",
        },
    },
    "hopfield": {
        "title": "Hopfield Network",
        "chunks": [
            "Hopfield networks are associative memories that converge to stored stable patterns.",
            "Recall works by iterative updates that reduce network energy.",
            "Hebbian-style storage builds symmetric weights without self-connections.",
            "Hopfield models are useful for pattern completion under noise.",
        ],
        "tags": {
            "keywords": ["hopfield", "associative memory", "energy", "hebbian", "recall"],
            "difficulty": "Intermediate",
        },
    },
}


def _ensure_history_db() -> None:
    with sqlite3.connect(HISTORY_DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS assistant_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                module_key TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                ts INTEGER NOT NULL
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_assistant_history_session_ts ON assistant_history(session_id, ts)")
        conn.commit()


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


def _prune_old_history() -> None:
    cutoff = int(time.time()) - HISTORY_RETENTION_SECONDS
    with sqlite3.connect(HISTORY_DB_PATH) as conn:
        conn.execute("DELETE FROM assistant_history WHERE ts < ?", (cutoff,))
        conn.commit()


def _save_message(session_id: str, module_key: str, role: str, content: str) -> None:
    with sqlite3.connect(HISTORY_DB_PATH) as conn:
        conn.execute(
            "INSERT INTO assistant_history(session_id, module_key, role, content, ts) VALUES (?, ?, ?, ?, ?)",
            (session_id, module_key, role, content, int(time.time())),
        )
        conn.commit()


def _load_recent_history(session_id: str, limit: int = 12) -> list[dict[str, str]]:
    with sqlite3.connect(HISTORY_DB_PATH) as conn:
        rows = conn.execute(
            "SELECT role, content FROM assistant_history WHERE session_id = ? ORDER BY ts DESC LIMIT ?",
            (session_id, limit),
        ).fetchall()

    rows.reverse()
    return [{"role": role, "content": content} for role, content in rows]


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9_]+", text.lower()))


def _module_search_text(module_key: str) -> str:
    module = STUDY_CONTENT[module_key]
    tags = module.get("tags", {})
    keyword_text = " ".join(tags.get("keywords", []))
    chunk_text = " ".join(module.get("chunks", []))
    return f"{module.get('title', '')} {keyword_text} {chunk_text}"


def _module_score(question: str, module_key: str) -> float:
    question_terms = _tokenize(question)
    module = STUDY_CONTENT[module_key]
    module_terms = _tokenize(_module_search_text(module_key))

    score = float(len(question_terms & module_terms))

    title_terms = _tokenize(module.get("title", ""))
    score += 2.0 * float(len(question_terms & title_terms))

    tags = module.get("tags", {})
    keyword_terms = set()
    for keyword in tags.get("keywords", []):
        keyword_terms.update(_tokenize(keyword))
    score += 1.5 * float(len(question_terms & keyword_terms))

    if module_key == "deep_learning_foundations":
        foundation_terms = {
            "deep",
            "learning",
            "neural",
            "network",
            "activation",
            "gradient",
            "loss",
            "dropout",
            "normalization",
            "regularization",
            "training",
        }
        score += 1.2 * float(len(question_terms & foundation_terms))

    return score


def _identify_relevant_module(question: str, current_module_key: str) -> tuple[str, float]:
    scored = [(_module_score(question, module_key), module_key) for module_key in STUDY_CONTENT]
    scored.sort(key=lambda item: item[0], reverse=True)

    best_score, best_module_key = scored[0]
    current_score = _module_score(question, current_module_key) if current_module_key in STUDY_CONTENT else 0.0

    if current_score >= best_score - 0.5:
        return current_module_key, current_score

    return best_module_key, best_score


def _retrieve_context(module_key: str, question: str, top_k: int = 4) -> list[str]:
    module = STUDY_CONTENT[module_key]
    question_terms = _tokenize(question)

    scored: list[tuple[float, str]] = []
    for chunk in module["chunks"]:
        terms = _tokenize(chunk)
        overlap = len(question_terms & terms)
        score = float(overlap)
        scored.append((score, chunk))

    scored.sort(key=lambda item: item[0], reverse=True)
    selected = [chunk for _, chunk in scored[:top_k]]
    # Ensure deterministic non-empty context
    if not selected:
        selected = module["chunks"][:top_k]
    return selected


def _build_system_prompt(current_module_title: str, relevant_module_title: str, detailed: bool, is_current_module: bool) -> str:
    tone = "concise, clear, and practical" if not detailed else "detailed, structured, and pedagogical"
    module_note = (
        f"The question belongs to the current module: {current_module_title}."
        if is_current_module
        else f"The question belongs most closely to: {relevant_module_title}."
    )
    return (
        "You are DeepNexus Study Tutor for deep learning and neural networks. "
        "You can answer general DL/NN questions, not only the currently opened module. "
        "Use the supplied context as grounding, but do not refuse a DL/NN question just because it is outside the current module. "
        f"{module_note} "
        "If the question is better matched to another module, state that module first, then answer. "
        "Format the response in markdown with short headings like ## Short answer, ## Why it happens, ## Example. "
        "Use bullets for steps and wrap equations in $...$ or $$...$$. "
        f"Style must be {tone}. Current module: {current_module_title}. Relevant module: {relevant_module_title}."
    )


def _build_user_prompt(question: str, context_chunks: list[str], tags: dict[str, Any], current_module_title: str, relevant_module_title: str, is_current_module: bool) -> str:
    tags_blob = "\n".join(
        [
            f"topic: {tags.get('topic', '')}",
            f"subtopic: {tags.get('subtopic', '')}",
            f"difficulty: {tags.get('difficulty', '')}",
            f"keywords: {', '.join(tags.get('keywords', []))}",
        ]
    )
    context_blob = "\n".join([f"- {c}" for c in context_chunks])
    module_blob = "current module" if is_current_module else f"relevant module: {relevant_module_title}"
    return (
        f"Current module: {current_module_title}\n"
        f"Module routing: {module_blob}\n\n"
        f"Study tags:\n{tags_blob}\n\n"
        f"Context:\n{context_blob}\n\n"
        f"Question:\n{question}"
    )


def _max_tokens_for(question: str, detailed: bool) -> int:
    q_len = len(question)
    if detailed:
        if q_len > 320:
            return 1800
        if q_len > 160:
            return 1500
        return 1300

    if q_len > 320:
        return 1200
    if q_len > 160:
        return 1000
    return 850


def _request_completion(api_key: str, messages: list[dict[str, str]], max_tokens: int, detailed: bool) -> dict[str, Any]:
    payload = {
        "model": NVIDIA_MODEL,
        "messages": messages,
        "temperature": 0.35 if not detailed else 0.55,
        "top_p": 0.95,
        "max_tokens": max_tokens,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    response = requests.post(
        f"{NVIDIA_BASE_URL}/chat/completions",
        headers=headers,
        json=payload,
        timeout=60,
    )
    response.raise_for_status()
    body = response.json()

    choices = body.get("choices", []) if isinstance(body, dict) else []
    if not choices:
        raise RuntimeError("NVIDIA API returned no completion choices.")

    choice = choices[0] if isinstance(choices[0], dict) else {}
    message = choice.get("message", {}) if isinstance(choice, dict) else {}
    answer = (message.get("content") or "").strip()
    finish_reason = (choice.get("finish_reason") or "").strip().lower()

    return {
        "answer": answer,
        "finish_reason": finish_reason,
    }


def ask_study_assistant(req: StudyAssistantAskRequest) -> StudyAssistantAskResponse:
    _ensure_history_db()
    _prune_old_history()

    module_key = req.module_key.strip()
    if module_key not in STUDY_CONTENT:
        valid = ", ".join(sorted(STUDY_CONTENT.keys()))
        raise ValueError(f"Unknown module_key '{module_key}'. Valid values: {valid}")

    api_key = (os.getenv("NVIDIA_API_KEY", "") or _read_local_env("NVIDIA_API_KEY")).strip()
    if not api_key:
        raise RuntimeError("NVIDIA_API_KEY is not configured on the server environment.")

    session_id = (req.session_id or str(uuid.uuid4())).strip()
    relevant_module_key, _ = _identify_relevant_module(req.question, module_key)
    relevant_module = STUDY_CONTENT[relevant_module_key]
    current_module = STUDY_CONTENT[module_key]
    retrieved = _retrieve_context(relevant_module_key, req.question)
    is_current_module = relevant_module_key == module_key

    messages: list[dict[str, str]] = [
        {
            "role": "system",
            "content": _build_system_prompt(current_module["title"], relevant_module["title"], req.detailed, is_current_module),
        },
    ]

    # Load limited one-week chat history for continuity.
    history = _load_recent_history(session_id=session_id, limit=10)
    for item in history:
        if item["role"] in {"user", "assistant"}:
            messages.append(item)

    messages.append(
        {
            "role": "user",
            "content": _build_user_prompt(
                req.question,
                retrieved,
                relevant_module.get("tags", {}),
                current_module["title"],
                relevant_module["title"],
                is_current_module,
            ),
        }
    )

    max_tokens = _max_tokens_for(req.question, req.detailed)

    try:
        first_pass = _request_completion(api_key, messages, max_tokens, req.detailed)
    except requests.RequestException as exc:
        raise RuntimeError(f"NVIDIA API request failed: {exc}") from exc

    answer = first_pass["answer"]

    # If response is cut by token limit, ask once to continue and merge.
    if first_pass["finish_reason"] == "length" and answer:
        continuation_messages = messages + [
            {"role": "assistant", "content": answer},
            {
                "role": "user",
                "content": "Continue from exactly where you stopped. Do not repeat earlier sections. Finish the answer cleanly.",
            },
        ]
        try:
            second_budget = min(900, max(400, max_tokens // 2))
            second_pass = _request_completion(api_key, continuation_messages, second_budget, req.detailed)
            if second_pass["answer"]:
                answer = f"{answer}\n\n{second_pass['answer']}".strip()
        except requests.RequestException:
            # Keep first pass if continuation fails.
            pass

    if not answer:
        answer = "I could not generate a response from Study Mode context right now."

    if not is_current_module:
        answer = f"## Module fit\nThis question belongs to **{relevant_module['title']}**.\n\n{answer}"

    _save_message(session_id=session_id, module_key=module_key, role="user", content=req.question)
    _save_message(session_id=session_id, module_key=module_key, role="assistant", content=answer)

    return StudyAssistantAskResponse(
        session_id=session_id,
        answer=answer,
        sources=retrieved,
        max_tokens_used=max_tokens,
        relevant_module_key=relevant_module_key,
        relevant_module_title=relevant_module["title"],
        is_current_module=is_current_module,
    )
