import json
from pathlib import Path
import string

import cv2
import numpy as np

from app.schemas.hopfield import HopfieldLibraryResponse, HopfieldRecallRequest, HopfieldRecallResponse


_CANVAS_SIZE = 180
_PATTERN_STORE = Path(__file__).resolve().parent.parent / "data" / "hopfield_patterns.json"


LETTER_BITMAPS_5X7 = {
    "A": ["01110", "10001", "10001", "11111", "10001", "10001", "10001"],
    "B": ["11110", "10001", "10001", "11110", "10001", "10001", "11110"],
    "C": ["01111", "10000", "10000", "10000", "10000", "10000", "01111"],
    "D": ["11110", "10001", "10001", "10001", "10001", "10001", "11110"],
    "E": ["11111", "10000", "10000", "11110", "10000", "10000", "11111"],
    "F": ["11111", "10000", "10000", "11110", "10000", "10000", "10000"],
    "G": ["01111", "10000", "10000", "10011", "10001", "10001", "01110"],
    "H": ["10001", "10001", "10001", "11111", "10001", "10001", "10001"],
    "I": ["11111", "00100", "00100", "00100", "00100", "00100", "11111"],
    "J": ["00111", "00010", "00010", "00010", "10010", "10010", "01100"],
    "K": ["10001", "10010", "10100", "11000", "10100", "10010", "10001"],
    "L": ["10000", "10000", "10000", "10000", "10000", "10000", "11111"],
    "M": ["10001", "11011", "10101", "10101", "10001", "10001", "10001"],
    "N": ["10001", "10001", "11001", "10101", "10011", "10001", "10001"],
    "O": ["01110", "10001", "10001", "10001", "10001", "10001", "01110"],
    "P": ["11110", "10001", "10001", "11110", "10000", "10000", "10000"],
    "Q": ["01110", "10001", "10001", "10001", "10101", "10010", "01101"],
    "R": ["11110", "10001", "10001", "11110", "10100", "10010", "10001"],
    "S": ["01111", "10000", "10000", "01110", "00001", "00001", "11110"],
    "T": ["11111", "00100", "00100", "00100", "00100", "00100", "00100"],
    "U": ["10001", "10001", "10001", "10001", "10001", "10001", "01110"],
    "V": ["10001", "10001", "10001", "10001", "10001", "01010", "00100"],
    "W": ["10001", "10001", "10001", "10101", "10101", "10101", "01010"],
    "X": ["10001", "10001", "01010", "00100", "01010", "10001", "10001"],
    "Y": ["10001", "10001", "01010", "00100", "00100", "00100", "00100"],
    "Z": ["11111", "00001", "00010", "00100", "01000", "10000", "11111"],
}

DIGIT_BITMAPS_5X7 = {
    "0": ["01110", "10001", "10011", "10101", "11001", "10001", "01110"],
    "1": ["00100", "01100", "00100", "00100", "00100", "00100", "01110"],
    "2": ["01110", "10001", "00001", "00010", "00100", "01000", "11111"],
    "3": ["11110", "00001", "00001", "01110", "00001", "00001", "11110"],
    "4": ["00010", "00110", "01010", "10010", "11111", "00010", "00010"],
    "5": ["11111", "10000", "11110", "00001", "00001", "10001", "01110"],
    "6": ["00110", "01000", "10000", "11110", "10001", "10001", "01110"],
    "7": ["11111", "00001", "00010", "00100", "01000", "01000", "01000"],
    "8": ["01110", "10001", "10001", "01110", "10001", "10001", "01110"],
    "9": ["01110", "10001", "10001", "01111", "00001", "00010", "11100"],
}


def _normalize_binary_pattern(pattern: list[int], side: int) -> np.ndarray:
    expected = side * side
    if len(pattern) != expected:
        raise ValueError(f"pattern size mismatch: expected {expected}, got {len(pattern)}")

    arr = np.array(pattern, dtype=np.int8)
    arr = np.where(arr > 0, 1, 0).astype(np.int8)
    return arr


def _center_binary_grid(binary: np.ndarray, side: int) -> np.ndarray:
    grid = np.asarray(binary, dtype=np.int8).reshape(side, side)
    ink = grid > 0
    if not np.any(ink):
        return grid.reshape(-1)

    rows = np.where(np.any(ink, axis=1))[0]
    cols = np.where(np.any(ink, axis=0))[0]
    r0, r1 = int(rows[0]), int(rows[-1])
    c0, c1 = int(cols[0]), int(cols[-1])
    glyph = grid[r0 : r1 + 1, c0 : c1 + 1]

    centered = np.zeros_like(grid)
    gh, gw = glyph.shape
    start_r = max(0, (side - gh) // 2)
    start_c = max(0, (side - gw) // 2)
    end_r = min(side, start_r + gh)
    end_c = min(side, start_c + gw)
    centered[start_r:end_r, start_c:end_c] = glyph[: end_r - start_r, : end_c - start_c]
    return centered.reshape(-1)


def _to_bipolar(binary: np.ndarray) -> np.ndarray:
    return np.where(binary > 0, 1, -1).astype(np.int8)


def _energy(state: np.ndarray, weights: np.ndarray) -> float:
    return float(-0.5 * state @ weights @ state)


def _charset_labels() -> list[str]:
    return list(string.digits + string.ascii_lowercase + string.ascii_uppercase)


def _bitmap_to_pattern(bitmap_rows: list[str], side: int) -> np.ndarray:
    small = np.array([[1 if ch == "1" else -1 for ch in row] for row in bitmap_rows], dtype=np.int8)
    scaled = np.kron(small, np.ones((2, 2), dtype=np.int8))
    grid = np.full((side, side), -1, dtype=np.int8)
    start_row = (side - scaled.shape[0]) // 2
    start_col = (side - scaled.shape[1]) // 2
    grid[start_row : start_row + scaled.shape[0], start_col : start_col + scaled.shape[1]] = scaled
    return grid.flatten()


def _render_char_pattern(label: str, side: int) -> np.ndarray:
    if label in LETTER_BITMAPS_5X7:
        return _bitmap_to_pattern(LETTER_BITMAPS_5X7[label], side)
    if label in DIGIT_BITMAPS_5X7:
        return _bitmap_to_pattern(DIGIT_BITMAPS_5X7[label], side)

    img = np.zeros((_CANVAS_SIZE, _CANVAS_SIZE), dtype=np.uint8)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 2.8 if side >= 15 else 2.2
    thickness = 4 if side >= 15 else 3

    (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)
    x = max(0, (_CANVAS_SIZE - text_w) // 2)
    y = min(_CANVAS_SIZE - baseline - 2, (_CANVAS_SIZE + text_h) // 2)

    cv2.putText(img, label, (x, y), font, font_scale, 255, thickness, cv2.LINE_AA)
    small = cv2.resize(img, (side, side), interpolation=cv2.INTER_AREA)

    _, binary = cv2.threshold(small, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary = binary.astype(np.int8)

    if int(np.sum(binary)) == 0:
        binary = (small > 0).astype(np.int8)

    return binary.flatten()


def _default_pattern_map(side: int) -> dict[str, np.ndarray]:
    return {label: (_render_char_pattern(label, side) > 0).astype(np.int8) for label in _charset_labels()}


def _load_user_patterns(side: int) -> dict[str, np.ndarray]:
    if not _PATTERN_STORE.exists():
        return {}

    try:
        payload = json.loads(_PATTERN_STORE.read_text(encoding="utf-8"))
    except Exception:
        return {}

    if payload.get("side") != side:
        return {}

    user_patterns = {}
    raw_patterns = payload.get("patterns", {})
    if not isinstance(raw_patterns, dict):
        return {}

    for label, values in raw_patterns.items():
        if not isinstance(label, str) or not isinstance(values, list):
            continue
        try:
            user_patterns[label] = _normalize_binary_pattern(values, side)
        except ValueError:
            continue

    return user_patterns


def _build_pattern_map(side: int) -> dict[str, np.ndarray]:
    defaults = _default_pattern_map(side)
    custom = _load_user_patterns(side)
    defaults.update(custom)
    return {k: (np.asarray(v) > 0).astype(np.int8) for k, v in defaults.items()}


def _preprocess_input_binary(binary: np.ndarray, side: int) -> np.ndarray:
    grid = (np.asarray(binary).reshape(side, side) > 0).astype(np.uint8)
    kernel = np.ones((3, 3), dtype=np.uint8)
    cleaned = cv2.morphologyEx(grid, cv2.MORPH_OPEN, kernel, iterations=1)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=1)
    if int(np.sum(cleaned)) == 0:
        cleaned = grid
    return _center_binary_grid(cleaned.astype(np.int8).reshape(-1), side)


def _hebbian_weights(patterns_bipolar: list[np.ndarray]) -> np.ndarray:
    n = patterns_bipolar[0].shape[0]
    weights = np.zeros((n, n), dtype=np.float64)
    for pattern in patterns_bipolar:
        p = pattern.reshape(-1, 1)
        weights += p @ p.T

    np.fill_diagonal(weights, 0.0)
    weights /= n
    return weights


def _pseudoinverse_weights(patterns_bipolar: list[np.ndarray]) -> np.ndarray:
    patterns = np.stack(patterns_bipolar, axis=1).astype(np.float64)
    gram = patterns.T @ patterns
    weights = patterns @ np.linalg.pinv(gram) @ patterns.T
    np.fill_diagonal(weights, 0.0)
    return weights


def _shift_bipolar_grid(grid: np.ndarray, dr: int, dc: int) -> np.ndarray:
    shifted = np.full_like(grid, -1)

    if dr >= 0:
        src_r = slice(0, grid.shape[0] - dr)
        dst_r = slice(dr, grid.shape[0])
    else:
        src_r = slice(-dr, grid.shape[0])
        dst_r = slice(0, grid.shape[0] + dr)

    if dc >= 0:
        src_c = slice(0, grid.shape[1] - dc)
        dst_c = slice(dc, grid.shape[1])
    else:
        src_c = slice(-dc, grid.shape[1])
        dst_c = slice(0, grid.shape[1] + dc)

    shifted[dst_r, dst_c] = grid[src_r, src_c]
    return shifted


def _dilate_bipolar_grid(grid: np.ndarray) -> np.ndarray:
    ink = grid == 1
    padded = np.pad(ink, 1, mode="constant", constant_values=False)
    dilated = np.zeros_like(ink)

    for r in range(3):
        for c in range(3):
            dilated |= padded[r : r + grid.shape[0], c : c + grid.shape[1]]

    return np.where(dilated, 1, -1).astype(np.int8)


def _build_classifier_variants(base_patterns: np.ndarray, side: int) -> tuple[np.ndarray, np.ndarray]:
    variants = []
    variant_label_idx = []
    shifts = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    for label_idx, flat_pattern in enumerate(base_patterns):
        grid = flat_pattern.reshape(side, side)
        pool = [grid]
        pool.extend(_shift_bipolar_grid(grid, dr, dc) for dr, dc in shifts)

        thick = _dilate_bipolar_grid(grid)
        pool.append(thick)
        pool.extend(_shift_bipolar_grid(thick, dr, dc) for dr, dc in shifts)

        seen = set()
        for variant in pool:
            key = variant.tobytes()
            if key in seen:
                continue
            seen.add(key)
            variants.append(variant.flatten())
            variant_label_idx.append(label_idx)

    return np.array(variants, dtype=np.int8), np.array(variant_label_idx, dtype=np.int8)


def _aggregate_variant_scores(variant_scores: np.ndarray, variant_to_label: np.ndarray, num_labels: int) -> np.ndarray:
    label_scores = np.full(num_labels, -np.inf, dtype=np.float64)
    for idx, score in enumerate(variant_scores):
        label_idx = int(variant_to_label[idx])
        if score > label_scores[label_idx]:
            label_scores[label_idx] = score
    return label_scores


def _shape_similarity_scores(state: np.ndarray, patterns: np.ndarray, side: int) -> np.ndarray:
    s = (np.asarray(state) > 0).astype(np.int8).reshape(1, -1)
    p = (np.asarray(patterns) > 0).astype(np.int8)

    s_ink = s == 1
    p_ink = p == 1

    intersection = np.sum(p_ink & s_ink, axis=1).astype(np.float64)
    union = np.sum(p_ink | s_ink, axis=1).astype(np.float64)
    iou = intersection / np.maximum(union, 1.0)

    s_ink_count = float(np.sum(s_ink))
    precision = intersection / max(s_ink_count, 1.0)
    p_ink_count = np.sum(p_ink, axis=1).astype(np.float64)
    recall = intersection / np.maximum(p_ink_count, 1.0)
    f1 = (2.0 * precision * recall) / np.maximum(precision + recall, 1e-9)

    dot = (p @ s.ravel()) / s.shape[1]
    dot01 = (dot + 1.0) / 2.0

    p_grid = p.reshape(p.shape[0], side, side)
    s_grid = s.reshape(side, side)
    p_ink_float = (p_grid == 1).astype(np.float64)
    s_ink_float = (s_grid == 1).astype(np.float64)

    p_row = np.sum(p_ink_float, axis=2)
    s_row = np.sum(s_ink_float, axis=1)
    row_dot = p_row @ s_row
    row_norm = np.linalg.norm(p_row, axis=1) * max(np.linalg.norm(s_row), 1e-9)
    row_sim = row_dot / np.maximum(row_norm, 1e-9)

    p_col = np.sum(p_ink_float, axis=1)
    s_col = np.sum(s_ink_float, axis=0)
    col_dot = p_col @ s_col
    col_norm = np.linalg.norm(p_col, axis=1) * max(np.linalg.norm(s_col), 1e-9)
    col_sim = col_dot / np.maximum(col_norm, 1e-9)

    return 0.34 * iou + 0.20 * f1 + 0.10 * dot01 + 0.18 * row_sim + 0.18 * col_sim


def _classify_with_consensus(input_state: np.ndarray, recalled_state: np.ndarray, patterns: np.ndarray, variant_to_label: np.ndarray, labels: list[str], side: int):
    input_variant_scores = _shape_similarity_scores(input_state, patterns, side)
    recall_variant_scores = _shape_similarity_scores(recalled_state, patterns, side)

    input_scores = _aggregate_variant_scores(input_variant_scores, variant_to_label, len(labels))
    recall_scores = _aggregate_variant_scores(recall_variant_scores, variant_to_label, len(labels))

    agreement = float(np.mean(np.asarray(input_state) == np.asarray(recalled_state)))
    recall_weight = 0.15 + 0.25 * agreement
    input_weight = 1.0 - recall_weight

    combined_scores = (input_weight * input_scores) + (recall_weight * recall_scores)
    best_idx = int(np.argmax(combined_scores))

    sorted_scores = np.sort(combined_scores)
    margin = float(sorted_scores[-1] - sorted_scores[-2]) if sorted_scores.size > 1 else float(sorted_scores[-1])

    return best_idx, combined_scores, input_scores, recall_scores, margin, agreement


def get_hopfield_library(side: int = 15) -> HopfieldLibraryResponse:
    pattern_map = _build_pattern_map(side)
    labels = sorted([label for label in pattern_map.keys() if label.isdigit() or label.isupper()])
    return HopfieldLibraryResponse(side=side, labels=labels, total_patterns=len(labels))


def recall_hopfield(req: HopfieldRecallRequest) -> HopfieldRecallResponse:
    side = req.side
    pattern_map = _build_pattern_map(side)
    if not pattern_map:
        raise ValueError("no patterns available for hopfield recall")

    input_binary = _preprocess_input_binary(_normalize_binary_pattern(req.pattern, side), side)
    input_state = _to_bipolar(input_binary).astype(np.int16)

    recognition_labels = [label for label in pattern_map.keys() if label.isdigit() or label.isupper()]
    stored_binary = [pattern_map[label] for label in recognition_labels]
    stored_bipolar = [_to_bipolar(p).astype(np.int16) for p in stored_binary]

    weights = _pseudoinverse_weights(stored_bipolar)

    state = input_state.copy()
    previous_state = state.copy()
    unchanged_count = 0
    converged = False
    executed_steps = 0
    n = state.shape[0]
    indices = np.arange(n)
    rng = np.random.default_rng(int(np.sum(input_binary)) + side * 131)

    for step in range(req.steps):
        executed_steps = step + 1
        if step % n == 0:
            rng.shuffle(indices)
        idx = int(indices[step % n])
        net = float(weights[idx] @ state)
        new_value = 1 if net >= 0 else -1

        if state[idx] == new_value:
            unchanged_count += 1
        else:
            unchanged_count = 0
            state[idx] = new_value

        if unchanged_count >= state.shape[0]:
            converged = True
            break

        if (step + 1) % n == 0:
            if np.array_equal(state, previous_state):
                converged = True
                break
            previous_state = state.copy()

    recalled_binary = np.where(state > 0, 1, 0).astype(np.int8)
    recalled_binary = _center_binary_grid(recalled_binary, side)

    labels = recognition_labels
    reference_matrix = np.stack(stored_binary, axis=0).astype(np.int8)
    variant_patterns, variant_to_label = _build_classifier_variants(_to_bipolar(reference_matrix), side)
    best_idx, combined_scores, input_scores, recall_scores, margin, agreement = _classify_with_consensus(
        input_binary,
        recalled_binary,
        variant_patterns,
        variant_to_label,
        labels,
        side,
    )

    confidence = float(combined_scores[best_idx])
    matched_label = labels[best_idx]
    ink_count = int(np.sum(input_binary))
    min_ink = max(6, side // 2)
    is_low_conf = confidence < req.confidence_threshold
    is_ambiguous = margin < 0.03
    is_unknown = ink_count < min_ink or (is_low_conf and is_ambiguous)
    label = "Unknown" if is_unknown else matched_label
    recalled_binary = reference_matrix[best_idx]

    energy_before = _energy(input_state.astype(np.float64), weights)
    energy_after = _energy(state.astype(np.float64), weights)

    return HopfieldRecallResponse(
        label=label,
        confidence=confidence,
        matched_label=matched_label,
        is_unknown=is_unknown,
        iterations=executed_steps,
        converged=converged,
        energy_before=energy_before,
        energy_after=energy_after,
        recalled_pattern=recalled_binary.tolist(),
    )
