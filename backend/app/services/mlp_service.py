from app.schemas.mlp import MLPConfigRequest, MLPConfigResponse


def summarize_mlp(req: MLPConfigRequest) -> MLPConfigResponse:
    sizes = req.layer_sizes
    total = 0
    parts: list[str] = []
    for i in range(len(sizes) - 1):
        layer_params = sizes[i] * sizes[i + 1] + sizes[i + 1]
        total += layer_params
        parts.append(f"L{i+1}: {sizes[i]} -> {sizes[i+1]} = {layer_params} params")
    return MLPConfigResponse(total_parameters=total, summary=" | ".join(parts))
