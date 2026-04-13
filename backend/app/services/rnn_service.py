import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer

from app.schemas.rnn import (
    EmotionAnalysisRequest,
    EmotionAnalysisResponse,
    EmotionProbability,
    NextWordSuggestion,
    RNNLabInfoResponse,
    TextGenerationRequest,
    TextGenerationResponse,
)

MODEL_DIR = Path(__file__).resolve().parents[2] / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

LSTM_STATE_PATH = MODEL_DIR / "lstm_model.pt"
LSTM_TOKENIZER_PATH = MODEL_DIR / "tokenizer.json"

EMOTION_MODEL_ID = "j-hartmann/emotion-english-distilroberta-base"
GENERATOR_MODEL_ID = "distilgpt2"
EMOTION_LABELS = ["Happy", "Sad", "Angry", "Neutral"]


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _tokenize_words(text: str) -> list[str]:
    return re.findall(r"[a-z']+|[.!?]", _normalize_text(text))


def _build_training_corpus() -> str:
    return " ".join(
        [
            "deep learning helps computers learn patterns from data",
            "recurrent neural networks process sequences over time",
            "lstm models remember long term dependencies in language",
            "students learn machine learning by experimenting with examples",
            "natural language processing powers chatbots and assistants",
            "transformers provide strong text understanding in real tasks",
            "emotion analysis can detect happy sad angry or neutral tone",
            "the weather is bright and the day feels joyful and calm",
            "this result is frustrating and makes users feel angry",
            "the review sounds neutral and objective in tone",
            "careful preprocessing improves model stability and performance",
            "sequence models predict the next word from context",
            "good explanations make ai outputs easier to trust",
            "a learner can compare model confidence across predictions",
            "the system should be fast accurate and interactive",
            "language models suggest probable words from learned patterns",
            "emotion classifiers map nuanced labels into simple categories",
            "practice and iteration improve understanding of neural networks",
            "text generation demonstrates how sequence memory works",
            "probability distributions reveal model certainty",
        ]
    )


class LSTMNextWordModel(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(x)
        out, _ = self.lstm(emb)
        logits = self.fc(out[:, -1, :])
        return logits


_LSTM_RUNTIME: dict[str, Any] | None = None
_EMOTION_RUNTIME: dict[str, Any] | None = None
_GENERATION_RUNTIME: dict[str, Any] | None = None


def _save_tokenizer(vocab: dict[str, int], context_size: int) -> None:
    payload = {"vocab": vocab, "context_size": context_size}
    LSTM_TOKENIZER_PATH.write_text(json.dumps(payload), encoding="utf-8")


def _load_tokenizer() -> tuple[dict[str, int], int]:
    payload = json.loads(LSTM_TOKENIZER_PATH.read_text(encoding="utf-8"))
    return payload["vocab"], int(payload["context_size"])


def _build_sequences(tokens: list[str], vocab: dict[str, int], context_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    ids = [vocab.get(t, 1) for t in tokens]
    xs: list[list[int]] = []
    ys: list[int] = []

    for i in range(context_size, len(ids)):
        xs.append(ids[i - context_size : i])
        ys.append(ids[i])

    x_tensor = torch.tensor(xs, dtype=torch.long)
    y_tensor = torch.tensor(ys, dtype=torch.long)
    return x_tensor, y_tensor


def _train_and_cache_lstm() -> dict[str, Any]:
    context_size = 4
    embedding_dim = 48
    hidden_dim = 96

    corpus_tokens = _tokenize_words(_build_training_corpus())
    vocab_tokens = sorted(set(corpus_tokens))
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for token in vocab_tokens:
        if token not in vocab:
            vocab[token] = len(vocab)

    x_train, y_train = _build_sequences(corpus_tokens, vocab, context_size)
    model = LSTMNextWordModel(vocab_size=len(vocab), embedding_dim=embedding_dim, hidden_dim=hidden_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    model.train()
    for _ in range(120):
        optimizer.zero_grad()
        logits = model(x_train)
        loss = F.cross_entropy(logits, y_train)
        loss.backward()
        optimizer.step()

    torch.save(
        {
            "model_state": model.state_dict(),
            "vocab_size": len(vocab),
            "embedding_dim": embedding_dim,
            "hidden_dim": hidden_dim,
        },
        LSTM_STATE_PATH,
    )
    _save_tokenizer(vocab, context_size)

    idx_to_word = {idx: word for word, idx in vocab.items()}
    model.eval()
    return {
        "model": model,
        "vocab": vocab,
        "idx_to_word": idx_to_word,
        "context_size": context_size,
    }


def _load_lstm_runtime() -> dict[str, Any]:
    global _LSTM_RUNTIME
    if _LSTM_RUNTIME is not None:
        return _LSTM_RUNTIME

    if not (LSTM_STATE_PATH.exists() and LSTM_TOKENIZER_PATH.exists()):
        _LSTM_RUNTIME = _train_and_cache_lstm()
        return _LSTM_RUNTIME

    vocab, context_size = _load_tokenizer()
    checkpoint = torch.load(LSTM_STATE_PATH, map_location="cpu")
    model = LSTMNextWordModel(
        vocab_size=int(checkpoint["vocab_size"]),
        embedding_dim=int(checkpoint["embedding_dim"]),
        hidden_dim=int(checkpoint["hidden_dim"]),
    )
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    idx_to_word = {idx: word for word, idx in vocab.items()}
    _LSTM_RUNTIME = {
        "model": model,
        "vocab": vocab,
        "idx_to_word": idx_to_word,
        "context_size": context_size,
    }
    return _LSTM_RUNTIME


def _load_emotion_runtime() -> dict[str, Any]:
    global _EMOTION_RUNTIME
    if _EMOTION_RUNTIME is not None:
        return _EMOTION_RUNTIME

    tokenizer = AutoTokenizer.from_pretrained(EMOTION_MODEL_ID)
    model = AutoModelForSequenceClassification.from_pretrained(EMOTION_MODEL_ID)
    model.eval()
    id2label = {int(k): str(v) for k, v in model.config.id2label.items()}

    _EMOTION_RUNTIME = {
        "tokenizer": tokenizer,
        "model": model,
        "id2label": id2label,
    }
    return _EMOTION_RUNTIME


def _load_generation_runtime() -> dict[str, Any]:
    global _GENERATION_RUNTIME
    if _GENERATION_RUNTIME is not None:
        return _GENERATION_RUNTIME

    tokenizer = AutoTokenizer.from_pretrained(GENERATOR_MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(GENERATOR_MODEL_ID)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()

    _GENERATION_RUNTIME = {
        "tokenizer": tokenizer,
        "model": model,
    }
    return _GENERATION_RUNTIME


def _map_emotion_label(raw_label: str) -> str:
    label = raw_label.strip().lower()
    if label in {"joy", "love"}:
        return "Happy"
    if label == "sadness":
        return "Sad"
    if label == "anger":
        return "Angry"
    if label in {"neutral", "fear", "surprise"}:
        return "Neutral"
    return "Neutral"


def get_rnn_lab_info() -> RNNLabInfoResponse:
    return RNNLabInfoResponse(
        module_name="RNN Lab",
        tasks=["Text Generation (Next-Token LM)", "Emotion Analysis (DistilRoBERTa)"],
        lstm_model_type="DistilGPT2 next-word predictor with probability ranking",
        bert_model_id=EMOTION_MODEL_ID,
        emotion_labels=EMOTION_LABELS,
    )


def _make_context_ids(tokens: list[str], vocab: dict[str, int], context_size: int) -> list[int]:
    ids = [vocab.get(t, 1) for t in tokens]
    if len(ids) < context_size:
        ids = [0] * (context_size - len(ids)) + ids
    else:
        ids = ids[-context_size:]
    return ids


def generate_next_words(req: TextGenerationRequest) -> TextGenerationResponse:
    runtime = _load_generation_runtime()
    tokenizer = runtime["tokenizer"]
    model = runtime["model"]

    clean_text = _normalize_text(req.text)
    encoded = tokenizer(
        clean_text,
        return_tensors="pt",
        truncation=True,
        max_length=128,
    )
    input_ids = encoded["input_ids"]

    with torch.no_grad():
        logits = model(input_ids=input_ids).logits[0, -1, :]
        probs = torch.softmax(logits / max(1e-6, req.temperature), dim=-1)

    top_pool = min(400, int(probs.shape[0]))
    pool_values, pool_indices = torch.topk(probs, k=top_pool)

    suggestions: list[NextWordSuggestion] = []
    used_words: set[str] = set()
    for conf, token_id in zip(pool_values.tolist(), pool_indices.tolist()):
        token_text = tokenizer.decode([int(token_id)], clean_up_tokenization_spaces=True).strip()
        word = re.sub(r"^[^A-Za-z']+|[^A-Za-z']+$", "", token_text).lower()
        if not word or word in used_words:
            continue
        if not re.match(r"^[a-z][a-z']*$", word):
            continue

        used_words.add(word)
        suggestions.append(NextWordSuggestion(word=word, confidence=float(conf)))
        if len(suggestions) >= req.top_k:
            break

    if not suggestions:
        fallback_word = tokenizer.decode([int(pool_indices[0].item())], clean_up_tokenization_spaces=True).strip() or "next"
        suggestions = [NextWordSuggestion(word=fallback_word, confidence=float(pool_values[0].item()))]

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            attention_mask=encoded.get("attention_mask"),
            max_new_tokens=req.generate_words,
            do_sample=True,
            temperature=float(req.temperature),
            top_p=0.92,
            top_k=50,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
    if generated_text.lower() == clean_text.lower():
        generated_text = f"{clean_text} {suggestions[0].word}".strip()

    return TextGenerationResponse(
        engine="DistilGPT2",
        input_text=req.text,
        suggestions=suggestions,
        generated_text=generated_text,
    )


def analyze_emotion(req: EmotionAnalysisRequest) -> EmotionAnalysisResponse:
    runtime = _load_emotion_runtime()
    tokenizer = runtime["tokenizer"]
    model = runtime["model"]
    id2label: dict[int, str] = runtime["id2label"]

    encoded = tokenizer(
        req.text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128,
    )

    with torch.no_grad():
        logits = model(**encoded).logits[0]
        probs = torch.softmax(logits, dim=0).cpu().numpy()

    mapped = {label: 0.0 for label in EMOTION_LABELS}
    for idx, prob in enumerate(probs.tolist()):
        raw_label = id2label.get(idx, "neutral")
        mapped_label = _map_emotion_label(raw_label)
        mapped[mapped_label] += float(prob)

    total = max(1e-9, float(sum(mapped.values())))
    normalized = {k: v / total for k, v in mapped.items()}

    ordered_probs = [EmotionProbability(label=label, probability=normalized[label]) for label in EMOTION_LABELS]
    predicted_label = max(EMOTION_LABELS, key=lambda label: normalized[label])

    return EmotionAnalysisResponse(
        engine="DistilRoBERTa",
        predicted_emotion=predicted_label,
        confidence=normalized[predicted_label],
        probabilities=ordered_probs,
    )
