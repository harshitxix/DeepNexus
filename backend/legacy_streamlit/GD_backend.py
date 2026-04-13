import numpy as np


def sigmoid(z: float) -> float:
    return 1 / (1 + np.exp(-z))


def train_gd_model(
    X,
    Y,
    lr: float = 0.05,
    epochs: int = 500,
    w1: float = 0.1,
    w2: float = -0.2,
    w3: float = 0.4,
    w4: float = 0.2,
    w5: float = -0.5,
    w6: float = 0.1,
    w7: float = 0.3,
    w8: float = -0.3,
    bh1: float = 0.1,
    bh2: float = -0.1,
    bo: float = 0.2,
    log_every: int = 1,
):
    X = np.array(X, dtype=float)
    Y = np.array(Y, dtype=float)

    errors = []
    outputs = []

    for epoch in range(epochs):
        total_error = 0.0
        for i in range(len(X)):
            x1, x2, x3 = X[i]

            zh1 = w1 * x1 + w3 * x2 + w5 * x3 + bh1
            zh2 = w2 * x1 + w4 * x2 + w6 * x3 + bh2

            h1 = sigmoid(zh1)
            h2 = sigmoid(zh2)
            z0 = w7 * h1 + w8 * h2 + bo
            o = sigmoid(z0)

            error = (Y[i] - o) ** 2
            total_error += (error * error)

            delta_o = error * o * (1 - o)
            delta_h1 = delta_o * w7 * h1 * (1 - h1)
            delta_h2 = delta_o * w8 * h2 * (1 - h2)

            w7 = w7 + lr * delta_o * h1
            w8 = w8 + lr * delta_o * h2
            bo = bo + lr * delta_o
            w1 = w1 + lr * delta_h1 * x1
            w3 = w3 + lr * delta_h1 * x2
            w5 = w5 + lr * delta_h1 * x3
            bh1 = bh1 + lr * delta_h1
            w2 = w2 + lr * delta_h2 * x1
            w4 = w4 + lr * delta_h2 * x2
            w6 = w6 + lr * delta_h2 * x3
            bh2 = bh2 + lr * delta_h2

        if (epoch + 1) % log_every == 0:
            errors.append(total_error)
            outputs.append(o)

    weights = {
        "w1": w1,
        "w2": w2,
        "w3": w3,
        "w4": w4,
        "w5": w5,
        "w6": w6,
        "w7": w7,
        "w8": w8,
        "bh1": bh1,
        "bh2": bh2,
        "bo": bo,
    }

    return {
        "errors": errors,
        "outputs": outputs,
        "weights": weights,
    }


def train_gradient_descent(X, Y, w1, w2, w3, w4, w5, w6, w7, w8, bh1, bh2, bo, lr, epochs, log_every=1):
    """Wrapper function for train_gd_model with simplified parameter order"""
    result = train_gd_model(
        X, Y, lr=lr, epochs=epochs, w1=w1, w2=w2, w3=w3, w4=w4, w5=w5, w6=w6,
        w7=w7, w8=w8, bh1=bh1, bh2=bh2, bo=bo, log_every=log_every
    )
    return result["errors"], result["outputs"], result["weights"]


def get_default_data():
    X = [
        [2, 60, 45],
        [3, 65, 50],
        [4, 70, 55],
        [5, 75, 60],
        [6, 80, 65],
        [7, 85, 70],
        [8, 90, 75],
    ]

    Y = [0, 0, 0, 1, 1, 1, 1]
    return X, Y