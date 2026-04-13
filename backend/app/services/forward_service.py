import numpy as np
from app.schemas.forward_prop import ForwardRequest, ForwardResponse
from app.services.math_utils import activate


def run_forward(req: ForwardRequest) -> ForwardResponse:
    x = np.array(req.inputs, dtype=float)
    w = np.array(req.weights, dtype=float)
    b = np.array(req.bias, dtype=float)
    z = (w @ x) + b
    a = activate(z, req.activation)
    return ForwardResponse(z=z.tolist(), a=a.tolist())
