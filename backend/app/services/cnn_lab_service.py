import base64
import io
from typing import Any

import cv2
import numpy as np
from PIL import Image

try:
    import torch
    import torch.nn.functional as F
    from torchvision import transforms
    from torchvision.models import (
        EfficientNet_B0_Weights,
        MobileNet_V2_Weights,
        ResNet50_Weights,
        efficientnet_b0,
        mobilenet_v2,
        resnet50,
    )

    TORCH_AVAILABLE = True
    TORCH_IMPORT_ERROR = ""
except Exception as exc:  # pragma: no cover
    TORCH_AVAILABLE = False
    TORCH_IMPORT_ERROR = str(exc)


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

_MODEL_CACHE: dict[str, dict[str, Any]] = {}


def _ensure_torch_available() -> None:
    if not TORCH_AVAILABLE:
        raise RuntimeError(
            "CNN Lab requires torch and torchvision. "
            f"Import failed with: {TORCH_IMPORT_ERROR}"
        )


def _to_data_url_rgb(image_rgb: np.ndarray) -> str:
    ok, encoded = cv2.imencode(".png", cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
    if not ok:
        raise RuntimeError("Failed to encode RGB image")
    b64 = base64.b64encode(encoded.tobytes()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def _to_data_url_gray(image_gray: np.ndarray) -> str:
    ok, encoded = cv2.imencode(".png", image_gray)
    if not ok:
        raise RuntimeError("Failed to encode grayscale image")
    b64 = base64.b64encode(encoded.tobytes()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def _read_image_rgb(image_bytes: bytes) -> Image.Image:
    pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return pil


def _build_preprocess() -> Any:
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def _load_model_bundle(model_name: str) -> dict[str, Any]:
    _ensure_torch_available()
    key = model_name.lower()
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]

    if key == "efficientnet_b0":
        weights = EfficientNet_B0_Weights.DEFAULT
        model = efficientnet_b0(weights=weights)
        final_conv = model.features[-1][0]
        feature_layers = {
            "early_conv": model.features[0],
            "early_blocks": model.features[2],
            "mid_blocks": model.features[4],
            "late_blocks": model.features[6],
            "deep_conv": model.features[8],
        }
    elif key == "resnet50":
        weights = ResNet50_Weights.DEFAULT
        model = resnet50(weights=weights)
        final_conv = model.layer4[-1].conv3
        feature_layers = {
            "early_conv": model.conv1,
            "early_blocks": model.layer1[0].conv1,
            "mid_blocks": model.layer2[0].conv1,
            "late_blocks": model.layer3[0].conv1,
            "deep_conv": model.layer4[0].conv1,
        }
    elif key == "mobilenet_v2":
        weights = MobileNet_V2_Weights.DEFAULT
        model = mobilenet_v2(weights=weights)
        final_conv = model.features[-1][0]
        feature_layers = {
            "early_conv": model.features[0],
            "early_blocks": model.features[3],
            "mid_blocks": model.features[6],
            "late_blocks": model.features[13],
            "deep_conv": model.features[18],
        }
    else:
        raise ValueError(f"Unsupported model_name: {model_name}")

    model.eval()
    bundle = {
        "model": model,
        "categories": list(weights.meta.get("categories", [])),
        "preprocess": _build_preprocess(),
        "final_conv": final_conv,
        "feature_layers": feature_layers,
    }
    _MODEL_CACHE[key] = bundle
    return bundle


def _prepare_input_tensor(image_bytes: bytes, preprocess: Any) -> tuple[Any, np.ndarray]:
    pil = _read_image_rgb(image_bytes)
    preview = np.array(pil.resize((224, 224), Image.Resampling.BILINEAR))
    tensor = preprocess(pil).unsqueeze(0)
    return tensor, preview


def predict_image(image_bytes: bytes, model_name: str, top_k: int = 3) -> dict[str, Any]:
    bundle = _load_model_bundle(model_name)
    model = bundle["model"]
    categories = bundle["categories"]
    preprocess = bundle["preprocess"]

    x, preview = _prepare_input_tensor(image_bytes, preprocess)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)

    k = max(1, min(int(top_k), 5))
    values, indices = torch.topk(probs[0], k=k)

    predictions = []
    for conf, idx in zip(values.tolist(), indices.tolist()):
        label = categories[idx] if idx < len(categories) else f"class_{idx}"
        predictions.append({"class_label": label, "confidence": float(conf)})

    return {
        "model_id": model_name,
        "top_predictions": predictions,
        "input_image": _to_data_url_rgb(preview),
    }


def _activation_to_grid(activation: Any, max_maps: int = 16, tile_size: int = 84) -> np.ndarray:
    fmap = activation.detach().cpu().squeeze(0).numpy()  # [C,H,W]
    channels = max(1, min(int(max_maps), fmap.shape[0]))
    subset = fmap[:channels]

    normalized_maps = []
    for m in subset:
        m_min, m_max = float(m.min()), float(m.max())
        if m_max - m_min < 1e-9:
            norm = np.zeros_like(m, dtype=np.uint8)
        else:
            norm = ((m - m_min) / (m_max - m_min) * 255.0).astype(np.uint8)
        norm = cv2.resize(norm, (tile_size, tile_size), interpolation=cv2.INTER_NEAREST)
        normalized_maps.append(norm)

    cols = 4
    rows = (len(normalized_maps) + cols - 1) // cols
    canvas = np.full((rows * tile_size + (rows - 1) * 6, cols * tile_size + (cols - 1) * 6), 245, dtype=np.uint8)

    for i, m in enumerate(normalized_maps):
        r = i // cols
        c = i % cols
        y = r * (tile_size + 6)
        x = c * (tile_size + 6)
        canvas[y : y + tile_size, x : x + tile_size] = m

    return canvas


def extract_feature_maps(image_bytes: bytes, model_name: str, max_maps: int = 16) -> dict[str, Any]:
    bundle = _load_model_bundle(model_name)
    model = bundle["model"]
    preprocess = bundle["preprocess"]
    layer_map = bundle["feature_layers"]

    activations: dict[str, Any] = {}
    hooks = []

    for layer_name, module in layer_map.items():
        def _capture(_module: Any, _inputs: Any, output: Any, name: str = layer_name) -> None:
            activations[name] = output

        hooks.append(module.register_forward_hook(_capture))

    try:
        x, _ = _prepare_input_tensor(image_bytes, preprocess)
        with torch.no_grad():
            _ = model(x)
    finally:
        for h in hooks:
            h.remove()

    feature_maps = []
    for layer_name in ["early_conv", "early_blocks", "mid_blocks", "late_blocks", "deep_conv"]:
        act = activations.get(layer_name)
        if act is None:
            continue
        grid = _activation_to_grid(act, max_maps=max_maps)
        feature_maps.append(
            {
                "layer_name": layer_name,
                "stage": layer_name.replace("_", " "),
                "grid_image": _to_data_url_gray(grid),
            }
        )

    return {"model_id": model_name, "feature_maps": feature_maps}


def generate_grad_cam(image_bytes: bytes, model_name: str) -> dict[str, Any]:
    bundle = _load_model_bundle(model_name)
    model = bundle["model"]
    preprocess = bundle["preprocess"]
    categories = bundle["categories"]
    target_layer = bundle["final_conv"]

    x, preview = _prepare_input_tensor(image_bytes, preprocess)
    x.requires_grad_(True)

    storage: dict[str, Any] = {"acts": None, "grads": None}

    def _fwd_hook(_module: Any, _inputs: Any, output: Any) -> None:
        storage["acts"] = output

    def _bwd_hook(_module: Any, grad_input: Any, grad_output: Any) -> None:
        del grad_input
        storage["grads"] = grad_output[0]

    h1 = target_layer.register_forward_hook(_fwd_hook)
    h2 = target_layer.register_full_backward_hook(_bwd_hook)

    try:
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        class_idx = int(torch.argmax(probs[0]).item())
        score = logits[0, class_idx]

        model.zero_grad(set_to_none=True)
        score.backward()

        acts = storage["acts"]
        grads = storage["grads"]
        if acts is None or grads is None:
            raise RuntimeError("Grad-CAM hooks failed to capture tensors")

        weights = grads.mean(dim=(2, 3), keepdim=True)
        cam = (weights * acts).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=(224, 224), mode="bilinear", align_corners=False)
        cam = cam.squeeze().detach().cpu().numpy()

        cam_min, cam_max = float(cam.min()), float(cam.max())
        if cam_max - cam_min < 1e-9:
            cam_norm = np.zeros_like(cam, dtype=np.float32)
        else:
            cam_norm = (cam - cam_min) / (cam_max - cam_min)

        heat_uint8 = (cam_norm * 255.0).astype(np.uint8)
        heat_bgr = cv2.applyColorMap(heat_uint8, cv2.COLORMAP_JET)
        heat_rgb = cv2.cvtColor(heat_bgr, cv2.COLOR_BGR2RGB)
        overlay = (0.55 * preview + 0.45 * heat_rgb).astype(np.uint8)

        confidence = float(probs[0, class_idx].item())
        label = categories[class_idx] if class_idx < len(categories) else f"class_{class_idx}"

        return {
            "model_id": model_name,
            "predicted_label": label,
            "confidence": confidence,
            "heatmap_image": _to_data_url_rgb(heat_rgb),
            "overlay_image": _to_data_url_rgb(overlay),
        }
    finally:
        h1.remove()
        h2.remove()
