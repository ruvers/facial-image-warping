"""
FaceWarp Lab — Face Parsing Module
BiSeNet face segmentation wrapper.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image


# =========================================================
# PATHS
# =========================================================

BACKEND_DIR = Path(__file__).resolve().parent
ROOT_DIR = BACKEND_DIR.parent


def _candidate_repo_paths() -> list[Path]:
    return [
        ROOT_DIR / "local_models" / "face-parsing.PyTorch",
        ROOT_DIR.parent / "face-parsing.PyTorch",
        BACKEND_DIR / "local_models" / "face-parsing.PyTorch",
    ]


def _resolve_bisenet_path() -> Path | None:
    for candidate in _candidate_repo_paths():
        if candidate.exists():
            return candidate.resolve()
    return None


def _resolve_model_path(repo_path: Path | None) -> Path | None:
    if repo_path is None:
        return None

    model_path = repo_path / "res" / "cp" / "79999_iter.pth"
    if model_path.exists():
        return model_path.resolve()

    return model_path


BISENET_PATH = _resolve_bisenet_path()
MODEL_PATH = _resolve_model_path(BISENET_PATH)


# =========================================================
# LABELS
# =========================================================

LABELS = {
    0: "background",
    1: "skin",
    2: "left_eyebrow",
    3: "right_eyebrow",
    4: "left_eye",
    5: "right_eye",
    6: "glasses",
    7: "left_ear",
    8: "right_ear",
    9: "earring",
    10: "nose",
    11: "mouth",
    12: "upper_lip",
    13: "lower_lip",
    14: "neck",
    15: "necklace",
    16: "cloth",
    17: "hair",
    18: "hat",
}


# =========================================================
# MODEL
# =========================================================

_model = None
_model_device = None
_model_device_info: dict[str, Any] | None = None
_last_error: str | None = None


def get_face_parsing_status() -> dict[str, Any]:
    repo_candidates = [str(path) for path in _candidate_repo_paths()]
    repo_ok = BISENET_PATH is not None and BISENET_PATH.exists()
    weights_ok = MODEL_PATH is not None and MODEL_PATH.exists()
    diagnostic_error = _last_error

    if not repo_ok:
        diagnostic_error = diagnostic_error or "face-parsing.PyTorch repo not found"
    elif not weights_ok:
        diagnostic_error = diagnostic_error or f"BiSeNet weights not found: {MODEL_PATH}"

    return {
        "name": "BiSeNet Face Parsing",
        "provider": "face-parsing.PyTorch",
        "local_only": True,
        "available": bool(repo_ok and weights_ok and _last_error is None),
        "repo_ok": bool(repo_ok),
        "weights_ok": bool(weights_ok),
        "loaded": _model is not None,
        "device": str(_model_device) if _model_device is not None else None,
        "device_info": _model_device_info,
        "repo_path": str(BISENET_PATH) if BISENET_PATH else None,
        "weights_path": str(MODEL_PATH) if MODEL_PATH else None,
        "searched_paths": repo_candidates,
        "last_error": diagnostic_error,
        "fallback": "zero background parsing map when unavailable",
    }


def get_model():
    global _model, _model_device, _model_device_info, _last_error

    if _model is not None:
        return _model

    if BISENET_PATH is None:
        _last_error = "face-parsing.PyTorch repo not found"
        return None

    if MODEL_PATH is None or not MODEL_PATH.exists():
        _last_error = f"BiSeNet weights not found: {MODEL_PATH}"
        return None

    bisenet_path_str = str(BISENET_PATH)
    if bisenet_path_str not in sys.path:
        sys.path.insert(0, bisenet_path_str)

    try:
        import torch
        import torchvision.transforms as transforms
        from model import BiSeNet
        from backend.local_models.torch_device import select_torch_device
    except Exception as e:
        _last_error = f"BiSeNet import failed: {e}"
        return None

    try:
        device, device_info = select_torch_device(prefer_gpu=True)
        net = BiSeNet(n_classes=19)

        state = torch.load(
            MODEL_PATH,
            map_location=device,
        )

        net.load_state_dict(state)
        try:
            net.to(device)
        except Exception as exc:
            if str(device) != "cpu":
                device = torch.device("cpu")
                device_info = {
                    **device_info,
                    "backend": "cpu",
                    "fallback_reason": f"model_to_gpu_failed: {exc!r}",
                }
                net.to(device)
            else:
                raise
        net.eval()

        _model = net
        _model_device = device
        _model_device_info = device_info
        _last_error = None

        return _model

    except Exception as e:
        _last_error = f"BiSeNet load failed: {e}"
        return None


# =========================================================
# TRANSFORM
# =========================================================

def _to_tensor_transform():
    import torchvision.transforms as transforms

    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            (0.485, 0.456, 0.406),
            (0.229, 0.224, 0.225),
        ),
    ])


# =========================================================
# FACE PARSING
# =========================================================

def parse_face(image_rgb: np.ndarray) -> np.ndarray:
    """
    Parameters
    ----------
    image_rgb : np.ndarray
        RGB uint8 image

    Returns
    -------
    parsing : np.ndarray
        (H,W) uint8 segmentation map
    """

    h, w = image_rgb.shape[:2]
    global _model_device, _model_device_info, _last_error

    net = get_model()

    if net is None:
        return np.zeros((h, w), dtype=np.uint8)

    import torch

    pil = Image.fromarray(image_rgb)

    pil_512 = pil.resize(
        (512, 512),
        Image.BILINEAR,
    )

    device = _model_device or torch.device("cpu")
    tensor = _to_tensor_transform()(
        pil_512
    ).unsqueeze(0).to(device)

    try:
        with torch.no_grad():
            out = net(tensor)[0]
    except Exception as exc:
        backend = (_model_device_info or {}).get("backend")
        if backend != "cpu":
            _last_error = f"BiSeNet GPU inference failed; falling back to CPU: {exc!r}"
            device = torch.device("cpu")
            net.to(device)
            _model_device = device
            _model_device_info = {
                **(_model_device_info or {}),
                "backend": "cpu",
                "fallback_reason": f"gpu_inference_failed: {exc!r}",
            }
            tensor = tensor.to(device)
            with torch.no_grad():
                out = net(tensor)[0]
        else:
            raise

    parsing = (
        out.squeeze(0)
        .argmax(0)
        .cpu()
        .numpy()
        .astype(np.uint8)
    )

    parsing = cv2.resize(
        parsing,
        (w, h),
        interpolation=cv2.INTER_NEAREST,
    )

    return parsing


# =========================================================
# BEARD REFINEMENT
# =========================================================

def detect_beard_region(
    image_rgb: np.ndarray,
    skin_mask: np.ndarray,
) -> np.ndarray:
    """
    Detect dark textured beard-like regions.
    Better than geometric landmark masking.
    """

    hsv = cv2.cvtColor(
        image_rgb,
        cv2.COLOR_RGB2HSV,
    )

    _, _, v = cv2.split(hsv)

    # Dark regions
    dark = v < 110

    gray = cv2.cvtColor(
        image_rgb,
        cv2.COLOR_RGB2GRAY,
    )

    # Texture detection
    edges = cv2.Laplacian(
        gray,
        cv2.CV_64F,
    )

    edges = np.abs(edges)

    texture = edges > 12

    beard = (
        dark &
        texture &
        (skin_mask > 0)
    )

    beard = (
        beard.astype(np.uint8)
        * 255
    )

    beard = cv2.medianBlur(
        beard,
        5,
    )

    return beard


# =========================================================
# MASK HELPERS
# =========================================================

def get_mask(
    parsing: np.ndarray,
    labels: list[int],
) -> np.ndarray:

    mask = (
        np.isin(parsing, labels)
        .astype(np.uint8)
        * 255
    )

    return mask


def smooth_mask(
    mask: np.ndarray,
    ksize: int = 11,
) -> np.ndarray:

    mask = cv2.GaussianBlur(
        mask,
        (ksize, ksize),
        0,
    )

    return mask


def feather_mask(
    mask: np.ndarray,
    blur: int = 31,
) -> np.ndarray:
    """
    Create soft alpha mask.
    Critical for realistic blending.
    """

    mask = cv2.GaussianBlur(
        mask,
        (blur, blur),
        0,
    )

    mask = (
        mask.astype(np.float32)
        / 255.0
    )
    
    return mask


def subtract_mask(
    base_mask: np.ndarray,
    remove_mask: np.ndarray,
) -> np.ndarray:
    """
    Remove one mask from another.
    """

    result = base_mask.copy()

    result[
        remove_mask > 0
    ] = 0

    return result


_FACE_PARSING_PRELOAD_ERROR: str | None = None


def is_face_parsing_loaded() -> bool:
    return globals().get("_model") is not None


def preload_face_parsing() -> dict[str, Any]:
    """
    Load BiSeNet face parsing model once at FastAPI startup.

    This function is defensive because existing loader names can vary.
    It tries known internal loader functions first, then falls back to
    parse_face(dummy_image).
    """

    import time
    import threading

    global _FACE_PARSING_PRELOAD_ERROR

    started = time.perf_counter()

    lock = globals().get("_FACE_PARSING_PRELOAD_LOCK")
    if lock is None:
        lock = threading.Lock()
        globals()["_FACE_PARSING_PRELOAD_LOCK"] = lock

    with lock:
        if globals().get("_model") is not None:
            return {
                "ok": True,
                "loaded": True,
                "cache_hit": True,
                "provider": "face-parsing.PyTorch/BiSeNet",
                "seconds": round(time.perf_counter() - started, 3),
                "status": get_face_parsing_status() if "get_face_parsing_status" in globals() else None,
            }

        try:
            # Prefer existing internal loader names if present.
            for loader_name in (
                "_get_model",
                "get_model",
                "_load_model",
                "load_model",
                "_get_face_parsing_model",
                "get_face_parsing_model",
            ):
                loader = globals().get(loader_name)
                if callable(loader):
                    loader()
                    break

            # If loader name was unknown, trigger lazy load through parse_face.
            if globals().get("_model") is None:
                parse_fn = globals().get("parse_face")
                if callable(parse_fn):
                    import numpy as np

                    dummy = np.zeros((512, 512, 3), dtype=np.uint8)
                    parse_fn(dummy)

            loaded = globals().get("_model") is not None
            _FACE_PARSING_PRELOAD_ERROR = None if loaded else "model_not_loaded_after_preload"

            return {
                "ok": bool(loaded),
                "loaded": bool(loaded),
                "cache_hit": False,
                "provider": "face-parsing.PyTorch/BiSeNet",
                "seconds": round(time.perf_counter() - started, 3),
                "error": _FACE_PARSING_PRELOAD_ERROR,
                "status": get_face_parsing_status() if "get_face_parsing_status" in globals() else None,
            }

        except Exception as exc:
            _FACE_PARSING_PRELOAD_ERROR = repr(exc)

            return {
                "ok": False,
                "loaded": False,
                "cache_hit": False,
                "provider": "face-parsing.PyTorch/BiSeNet",
                "reason": "preload_failed",
                "error": _FACE_PARSING_PRELOAD_ERROR,
                "seconds": round(time.perf_counter() - started, 3),
                "status": get_face_parsing_status() if "get_face_parsing_status" in globals() else None,
            }
