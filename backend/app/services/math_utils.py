import numpy as np


def activate(values: np.ndarray, name: str) -> np.ndarray:
    key = name.lower()
    if key == "sigmoid":
        return 1.0 / (1.0 + np.exp(-np.clip(values, -50, 50)))
    if key == "relu":
        return np.maximum(0.0, values)
    if key == "tanh":
        return np.tanh(values)
    if key == "step":
        return (values > 0).astype(float)
    return values
