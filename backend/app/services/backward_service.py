import math
from app.schemas.backward_prop import BackwardRequest, BackwardResponse


def run_backward_step(req: BackwardRequest) -> BackwardResponse:
    y_pred = 1.0 / (1.0 + math.exp(-(req.w * req.x + req.b)))
    loss = (y_pred - req.y_true) ** 2

    dloss_dypred = 2.0 * (y_pred - req.y_true)
    dypred_dz = y_pred * (1.0 - y_pred)
    dz_dw = req.x
    dz_db = 1.0

    grad_w = dloss_dypred * dypred_dz * dz_dw
    grad_b = dloss_dypred * dypred_dz * dz_db

    new_w = req.w - req.learning_rate * grad_w
    new_b = req.b - req.learning_rate * grad_b

    return BackwardResponse(
        y_pred=y_pred,
        loss=loss,
        grad_w=grad_w,
        grad_b=grad_b,
        new_w=new_w,
        new_b=new_b,
    )
