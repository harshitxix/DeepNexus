from app.schemas.perceptron import PerceptronRequest, PerceptronResponse
from app.services.math_utils import activate
import numpy as np


def run_perceptron(req: PerceptronRequest) -> PerceptronResponse:
    z = req.w1 * req.x1 + req.w2 * req.x2 + req.bias
    a = float(activate(np.array([z]), req.activation)[0])
    predicted_class = 1 if a >= req.threshold else 0
    return PerceptronResponse(z=z, a=a, predicted_class=predicted_class)
