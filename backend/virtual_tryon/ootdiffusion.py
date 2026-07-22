from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
import time
import uuid
import hashlib
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from backend.archive_pairs import (
    archive_exists,
    get_archive_root,
    get_archive_stats,
    get_preprocess_paths,
    find_pair_person,
    is_archive_cloth,
    get_archive_person_image,
)


DEFAULT_OOTDIFFUSION_ROOT = Path(r"D:\2Testfile\OOTDiffusion")
_RUN_LOCK = threading.Lock()

MODEL_TYPES = {"hd", "dc"}
CATEGORIES = {
    "upperbody": 0,
    "upper_body": 0,
    "upper-body": 0,
    "lowerbody": 1,
    "lower_body": 1,
    "lower-body": 1,
    "dress": 2,
    "dresses": 2,
}


def get_ootdiffusion_root() -> Path:
    return Path(os.getenv("FACEWARP_OOTDIFFUSION_ROOT", str(DEFAULT_OOTDIFFUSION_ROOT))).resolve()


def _path_status(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "exists": path.exists(),
        "is_dir": path.is_dir(),
    }


def _check_python_import(module: str) -> dict[str, Any]:
    try:
        __import__(module)
        return {"module": module, "available": True, "error": None}
    except Exception as exc:
        return {"module": module, "available": False, "error": repr(exc)}


def _get_cuda_device_info() -> dict[str, Any]:
    """Return detailed CUDA device information for performance tuning."""
    info: dict[str, Any] = {
        "available": False,
        "device_count": 0,
        "torch_version": None,
    }
    try:
        import torch

        info["torch_version"] = str(torch.__version__)
        info["available"] = bool(torch.cuda.is_available())
        if info["available"]:
            info["device_count"] = torch.cuda.device_count()
            info["current_device"] = torch.cuda.current_device()
            info["device_name"] = torch.cuda.get_device_name(0)
            mem = torch.cuda.get_device_properties(0)
            info["total_memory_gb"] = round(mem.total_mem / (1024 ** 3), 2)
            info["compute_capability"] = f"{mem.major}.{mem.minor}"
            info["supports_fp16"] = mem.major >= 7
            info["supports_tf32"] = mem.major >= 8
            info["supports_bf16"] = mem.major >= 8
    except Exception:
        pass
    return info


def _apply_cuda_optimizations() -> None:
    """Apply CUDA performance optimizations for maximum throughput."""
    try:
        import torch

        if not torch.cuda.is_available():
            return

        # Enable TF32 for Ampere+ GPUs (massive speedup, negligible quality loss)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        # CuDNN auto-tuner picks the fastest convolution algorithms
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True

        # Deterministic off for speed
        torch.backends.cudnn.deterministic = False

    except Exception:
        pass


def get_virtual_tryon_status() -> dict[str, Any]:
    root = get_ootdiffusion_root()
    checkpoints = root / "checkpoints"
    required_paths = {
        "root": root,
        "run_script": root / "run" / "run_ootd.py",
        "ootd_checkpoint": checkpoints / "ootd",
        "clip_checkpoint": checkpoints / "clip-vit-large-patch14",
        "openpose_checkpoint": checkpoints / "openpose",
        "humanparsing_atr": checkpoints / "humanparsing" / "parsing_atr.onnx",
        "humanparsing_lip": checkpoints / "humanparsing" / "parsing_lip.onnx",
    }
    paths = {name: _path_status(path) for name, path in required_paths.items()}

    deps = {
        name: _check_python_import(name)
        for name in ("torch", "diffusers", "transformers", "accelerate", "onnxruntime")
    }

    cuda_info = _get_cuda_device_info()
    cuda_available = cuda_info["available"]
    torch_version = cuda_info["torch_version"]

    missing_paths = [
        name
        for name, item in paths.items()
        if not item["exists"]
    ]
    missing_deps = [
        name
        for name, item in deps.items()
        if not item["available"]
    ]

    installed = root.exists() and not missing_paths and not missing_deps
    available = installed and cuda_available
    if not root.exists():
        reason = "ootdiffusion_repo_missing"
    elif missing_paths:
        reason = "ootdiffusion_checkpoints_missing"
    elif missing_deps:
        reason = "ootdiffusion_dependencies_missing"
    elif not cuda_available:
        reason = "ootdiffusion_requires_cuda"
    else:
        reason = None

    # Archive data status
    archive_status = get_archive_stats() if archive_exists() else {"available": False}

    return {
        "provider": "OOTDiffusion",
        "available": available,
        "installed": installed,
        "reason": reason,
        "runtime_blocker": None if available else reason,
        "root": str(root),
        "cuda_available": cuda_available,
        "cuda_device": cuda_info,
        "torch_version": torch_version,
        "requires": {
            "cuda": True,
            "nvidia_gpu": True,
            "cpu_fallback": _cpu_preview_fallback_enabled(),
        },
        "archive": archive_status,
        "preview_cpu_fallback_available": _cpu_preview_fallback_enabled(),
        "preview_cpu_fallback_note": (
            "CPU preview is a temporary store/UI test compositor, not realistic virtual try-on."
        ),
        "paths": paths,
        "dependencies": deps,
        "supported": {
            "model_types": ["hd", "dc"],
            "categories": ["upperbody", "lowerbody", "dress"],
            "image_size": [768, 1024],
        },
        "performance": {
            "cuda_optimizations": [
                "TF32 matmul (Ampere+)",
                "cuDNN auto-tuner",
                "fp16 inference",
                "CUDA memory caching",
            ],
            "recommended_steps": 20,
            "recommended_scale": 2.0,
            "recommended_model_type": "dc",
        },
        "notes": [
            "OOTDiffusion is used as an external local runtime.",
            "The source person image should be full-body or half-body depending on model_type.",
            "CUDA with NVIDIA GPU is required for real virtual try-on.",
            "Archive VITON-HD data provides pre-processed inputs for faster inference.",
        ],
    }


def _normalize_model_type(model_type: str) -> str:
    value = str(model_type or "dc").strip().lower()
    if value not in MODEL_TYPES:
        raise ValueError("model_type must be 'hd' or 'dc'.")
    return value


def _normalize_category(category: str | int, model_type: str) -> int:
    if isinstance(category, int):
        category_idx = int(category)
    else:
        value = str(category or "upperbody").strip().lower()
        category_idx = CATEGORIES.get(value, 0)

    if category_idx not in {0, 1, 2}:
        raise ValueError("category must be upperbody, lowerbody, or dress.")
    if model_type == "hd" and category_idx != 0:
        raise ValueError("model_type 'hd' only supports upperbody garments.")
    return category_idx





def _pad_image_to_aspect(image: Image.Image, target_ratio: float = 0.75) -> Image.Image:
    """Pad image to target aspect ratio (W/H) with white background to prevent stretching."""
    w, h = image.size
    current_ratio = w / h
    if abs(current_ratio - target_ratio) < 0.01:
        return image.copy()
    
    if current_ratio > target_ratio:
        new_w = w
        new_h = int(w / target_ratio)
    else:
        new_h = h
        new_w = int(h * target_ratio)
        
    pad_left = (new_w - w) // 2
    pad_top = (new_h - h) // 2
    
    new_img = Image.new("RGB", (new_w, new_h), (255, 255, 255))
    new_img.paste(image, (pad_left, pad_top))
    return new_img

def _fit_image_to_aspect(image: Image.Image, target_ratio: float = 0.75) -> tuple[Image.Image, dict[str, int]]:
    """Crop image to target aspect ratio (W/H) to prevent AI hallucination from stretching."""
    w, h = image.size
    current_ratio = w / h
    if abs(current_ratio - target_ratio) < 0.01:
        return image.copy(), {"left": 0, "top": 0, "right": w, "bottom": h, "orig_w": w, "orig_h": h, "new_w": w, "new_h": h}
    
    if current_ratio > target_ratio:
        new_h = h
        new_w = int(h * target_ratio)
        left = (w - new_w) // 2
        top = 0
        right = left + new_w
        bottom = h
        new_img = image.crop((left, top, right, bottom))
        return new_img, {"left": left, "top": top, "right": right, "bottom": bottom, "orig_w": w, "orig_h": h, "new_w": new_w, "new_h": new_h}
    else:
        new_w = w
        new_h = int(w / target_ratio)
        left = 0
        top = (h - new_h) // 2
        right = w
        bottom = top + new_h
        new_img = image.crop((left, top, right, bottom))
        return new_img, {"left": left, "top": top, "right": right, "bottom": bottom, "orig_w": w, "orig_h": h, "new_w": new_w, "new_h": new_h}

def _verify_image(path: Path) -> dict[str, Any]:
    with Image.open(path) as image:
        return {
            "path": str(path),
            "width": int(image.width),
            "height": int(image.height),
            "mode": image.mode,
        }


def _cpu_preview_fallback_enabled() -> bool:
    value = os.getenv("FACEWARP_TRYON_CPU_PREVIEW", "1").strip().lower()
    return value not in {"0", "false", "no", "off"}


def _crop_alpha(rgba: np.ndarray) -> np.ndarray:
    alpha = rgba[:, :, 3]
    ys, xs = np.where(alpha > 8)
    if len(xs) == 0 or len(ys) == 0:
        return rgba
    pad = 3
    x1 = max(0, int(xs.min()) - pad)
    x2 = min(rgba.shape[1], int(xs.max()) + pad + 1)
    y1 = max(0, int(ys.min()) - pad)
    y2 = min(rgba.shape[0], int(ys.max()) + pad + 1)
    return rgba[y1:y2, x1:x2]


def _refine_prepared_alpha(rgba: np.ndarray) -> np.ndarray:
    alpha = rgba[:, :, 3].astype(np.uint8)
    visible = alpha > 8
    if float(np.mean(visible)) < 0.03:
        return rgba

    ys, xs = np.where(visible)
    if len(xs) == 0 or len(ys) == 0:
        return rgba

    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())
    pad = max(8, int(max(x2 - x1, y2 - y1) * 0.08))
    border = visible & (
        (np.indices(alpha.shape)[1] <= x1 + pad)
        | (np.indices(alpha.shape)[1] >= x2 - pad)
        | (np.indices(alpha.shape)[0] <= y1 + pad)
        | (np.indices(alpha.shape)[0] >= y2 - pad)
    )

    rgb = rgba[:, :, :3].astype(np.uint8)
    samples = rgb[border]
    if samples.shape[0] < 64:
        return rgba

    bg_color = np.median(samples.astype(np.float32), axis=0)
    dist = np.linalg.norm(rgb.astype(np.float32) - bg_color[None, None, :], axis=2)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    bg_like = visible & (
        (dist < 48)
        | ((saturation < 58) & (value > 135) & (dist < 92))
    )

    fg = (visible & ~bg_like).astype(np.uint8) * 255
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
    if num_labels > 1:
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        fg = (labels == largest).astype(np.uint8) * 255

    if np.count_nonzero(fg) < np.count_nonzero(visible) * 0.18:
        return rgba

    refined = rgba.copy()
    refined_alpha = cv2.GaussianBlur(fg, (5, 5), 0)
    refined[:, :, 3] = np.minimum(alpha, refined_alpha)
    return refined


def _rgba_with_estimated_alpha(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        rgba = np.array(image.convert("RGBA"), dtype=np.uint8)

    alpha = rgba[:, :, 3]
    if np.any(alpha < 245):
        return _crop_alpha(_refine_prepared_alpha(rgba))

    rgb = rgba[:, :, :3].copy()
    h, w = rgb.shape[:2]
    mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
    flood = rgb.copy()
    for seed in ((0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)):
        cv2.floodFill(
            flood,
            mask,
            seed,
            (255, 0, 255),
            (16, 16, 16),
            (16, 16, 16),
            cv2.FLOODFILL_MASK_ONLY,
        )

    bg = (mask[1:-1, 1:-1] > 0).astype(np.uint8) * 255
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    low_saturation_bright = ((hsv[:, :, 1] < 28) & (hsv[:, :, 2] > 225)).astype(np.uint8) * 255
    bg = cv2.bitwise_or(bg, low_saturation_bright)
    bg = cv2.morphologyEx(bg, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    estimated_alpha = 255 - cv2.GaussianBlur(bg, (5, 5), 0)
    rgba[:, :, 3] = estimated_alpha
    return _crop_alpha(rgba)


def _subject_bbox(person_rgb: np.ndarray) -> tuple[int, int, int, int]:
    h, w = person_rgb.shape[:2]
    hsv = cv2.cvtColor(person_rgb, cv2.COLOR_RGB2HSV)
    # Works well enough for catalog/person test photos on white or simple backgrounds.
    foreground = ~((hsv[:, :, 1] < 35) & (hsv[:, :, 2] > 232))
    foreground[:, : int(w * 0.03)] = False
    foreground[:, int(w * 0.97) :] = False
    foreground[: int(h * 0.02), :] = False
    mask = foreground.astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return int(w * 0.2), int(h * 0.12), int(w * 0.8), int(h * 0.95)

    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    x = int(stats[largest, cv2.CC_STAT_LEFT])
    y = int(stats[largest, cv2.CC_STAT_TOP])
    bw = int(stats[largest, cv2.CC_STAT_WIDTH])
    bh = int(stats[largest, cv2.CC_STAT_HEIGHT])
    if bw < w * 0.12 or bh < h * 0.20:
        return int(w * 0.2), int(h * 0.12), int(w * 0.8), int(h * 0.95)
    return x, y, min(w, x + bw), min(h, y + bh)


def _detect_pose_points(person_rgb: np.ndarray) -> dict[int, tuple[float, float, float]] | None:
    """Return MediaPipe Pose landmarks as pixel coordinates when available."""
    try:
        import mediapipe as mp

        h, w = person_rgb.shape[:2]
        with mp.solutions.pose.Pose(
            static_image_mode=True,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.45,
        ) as pose:
            result = pose.process(person_rgb)
        if not result.pose_landmarks:
            return None
        points: dict[int, tuple[float, float, float]] = {}
        for idx, lm in enumerate(result.pose_landmarks.landmark):
            points[idx] = (
                float(np.clip(lm.x * w, 0, w - 1)),
                float(np.clip(lm.y * h, 0, h - 1)),
                float(getattr(lm, "visibility", 1.0)),
            )
        return points
    except Exception:
        return None


def _pose_point(
    points: dict[int, tuple[float, float, float]] | None,
    idx: int,
    min_visibility: float = 0.25,
) -> np.ndarray | None:
    if not points or idx not in points:
        return None
    x, y, visibility = points[idx]
    if visibility < min_visibility:
        return None
    return np.array([x, y], dtype=np.float32)


def _body_quad_from_pose(
    person_rgb: np.ndarray,
    category: str,
    pose_points: dict[int, tuple[float, float, float]] | None,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    h, w = person_rgb.shape[:2]
    debug: dict[str, Any] = {"pose_used": False}

    left_shoulder = _pose_point(pose_points, 11)
    right_shoulder = _pose_point(pose_points, 12)
    left_hip = _pose_point(pose_points, 23)
    right_hip = _pose_point(pose_points, 24)
    left_knee = _pose_point(pose_points, 25)
    right_knee = _pose_point(pose_points, 26)
    left_ankle = _pose_point(pose_points, 27, min_visibility=0.15)
    right_ankle = _pose_point(pose_points, 28, min_visibility=0.15)

    if left_shoulder is None or right_shoulder is None or left_hip is None or right_hip is None:
        return None, debug

    shoulder_w = float(np.linalg.norm(right_shoulder - left_shoulder))
    hip_w = float(np.linalg.norm(right_hip - left_hip))
    torso_h = float(np.linalg.norm((left_hip + right_hip) * 0.5 - (left_shoulder + right_shoulder) * 0.5))
    if shoulder_w < w * 0.08 or torso_h < h * 0.08:
        return None, debug

    shoulder_pad = max(6.0, shoulder_w * 0.18)
    hip_pad = max(4.0, hip_w * 0.14)
    top_l = left_shoulder + np.array([-shoulder_pad, -torso_h * 0.08], dtype=np.float32)
    top_r = right_shoulder + np.array([shoulder_pad, -torso_h * 0.08], dtype=np.float32)

    if category == "lowerbody":
        if left_ankle is not None and right_ankle is not None:
            bottom_l = left_ankle + np.array([-hip_w * 0.12, 0.0], dtype=np.float32)
            bottom_r = right_ankle + np.array([hip_w * 0.12, 0.0], dtype=np.float32)
        elif left_knee is not None and right_knee is not None:
            bottom_l = left_knee + np.array([-hip_w * 0.18, torso_h * 0.75], dtype=np.float32)
            bottom_r = right_knee + np.array([hip_w * 0.18, torso_h * 0.75], dtype=np.float32)
        else:
            bottom_l = left_hip + np.array([-hip_w * 0.10, torso_h * 1.65], dtype=np.float32)
            bottom_r = right_hip + np.array([hip_w * 0.10, torso_h * 1.65], dtype=np.float32)
        top_l = left_hip + np.array([-hip_pad, -torso_h * 0.02], dtype=np.float32)
        top_r = right_hip + np.array([hip_pad, -torso_h * 0.02], dtype=np.float32)
    elif category == "dress":
        if left_knee is not None and right_knee is not None:
            bottom_l = left_knee + np.array([-hip_w * 0.20, torso_h * 0.18], dtype=np.float32)
            bottom_r = right_knee + np.array([hip_w * 0.20, torso_h * 0.18], dtype=np.float32)
        else:
            bottom_l = left_hip + np.array([-hip_w * 0.24, torso_h * 1.05], dtype=np.float32)
            bottom_r = right_hip + np.array([hip_w * 0.24, torso_h * 1.05], dtype=np.float32)
    else:
        bottom_l = left_hip + np.array([-hip_pad, torso_h * 0.05], dtype=np.float32)
        bottom_r = right_hip + np.array([hip_pad, torso_h * 0.05], dtype=np.float32)

    quad = np.array([top_l, top_r, bottom_r, bottom_l], dtype=np.float32)
    quad[:, 0] = np.clip(quad[:, 0], 0, w - 1)
    quad[:, 1] = np.clip(quad[:, 1], 0, h - 1)
    debug.update(
        {
            "pose_used": True,
            "target_quad": np.round(quad, 2).tolist(),
            "shoulder_width": round(shoulder_w, 2),
            "torso_height": round(torso_h, 2),
        }
    )
    return quad, debug


def _category_name(category: str | int) -> str:
    if isinstance(category, int):
        return {0: "upperbody", 1: "lowerbody", 2: "dress"}.get(category, "upperbody")
    value = str(category or "upperbody").strip().lower().replace("-", "_")
    if value in {"upper_body", "top"}:
        return "upperbody"
    if value in {"lower_body", "bottom"}:
        return "lowerbody"
    if value in {"dresses"}:
        return "dress"
    return value if value in {"upperbody", "lowerbody", "dress"} else "upperbody"


def _target_rect(person_rgb: np.ndarray, category: str | int) -> tuple[int, int, int, int, dict[str, Any]]:
    h, w = person_rgb.shape[:2]
    x1, y1, x2, y2 = _subject_bbox(person_rgb)
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    cx = x1 + bw * 0.5
    category_name = _category_name(category)

    if category_name == "lowerbody":
        tw = bw * 0.62
        th = bh * 0.45
        ty = y1 + bh * 0.50
    elif category_name == "dress":
        tw = bw * 0.78
        th = bh * 0.68
        ty = y1 + bh * 0.17
    else:
        tw = bw * 0.78
        th = bh * 0.39
        ty = y1 + bh * 0.11

    tx1 = int(np.clip(cx - tw / 2, 0, w - 1))
    ty1 = int(np.clip(ty, 0, h - 1))
    tx2 = int(np.clip(cx + tw / 2, tx1 + 1, w))
    ty2 = int(np.clip(ty + th, ty1 + 1, h))
    return tx1, ty1, tx2, ty2, {
        "subject_bbox": [int(x1), int(y1), int(x2), int(y2)],
        "target_rect": [tx1, ty1, tx2, ty2],
        "category": category_name,
    }


def _fallback_quad_from_rect(rect: tuple[int, int, int, int]) -> np.ndarray:
    x1, y1, x2, y2 = rect
    return np.array(
        [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
        dtype=np.float32,
    )


def _protect_mask_from_pose(
    image_shape: tuple[int, int, int],
    pose_points: dict[int, tuple[float, float, float]] | None,
    category: str,
) -> np.ndarray:
    h, w = image_shape[:2]
    mask = np.zeros((h, w), dtype=np.float32)
    if not pose_points:
        return mask

    left_shoulder = _pose_point(pose_points, 11)
    right_shoulder = _pose_point(pose_points, 12)
    if left_shoulder is None or right_shoulder is None:
        return mask
    shoulder_w = float(np.linalg.norm(right_shoulder - left_shoulder))
    thickness = max(8, int(shoulder_w * 0.20))

    def draw_limb(indices: tuple[int, int, int]) -> None:
        pts = [_pose_point(pose_points, idx, min_visibility=0.18) for idx in indices]
        pts = [pt for pt in pts if pt is not None]
        if len(pts) < 2:
            return
        for a, b in zip(pts[:-1], pts[1:]):
            cv2.line(mask, tuple(np.round(a).astype(int)), tuple(np.round(b).astype(int)), 1.0, thickness, cv2.LINE_AA)
        for pt in pts:
            cv2.circle(mask, tuple(np.round(pt).astype(int)), max(4, thickness // 2), 1.0, -1, cv2.LINE_AA)

    if category in {"upperbody", "dress"}:
        draw_limb((11, 13, 15))
        draw_limb((12, 14, 16))

    nose = _pose_point(pose_points, 0, min_visibility=0.15)
    if nose is not None:
        cv2.circle(mask, tuple(np.round(nose).astype(int)), max(12, int(shoulder_w * 0.30)), 1.0, -1, cv2.LINE_AA)

    return cv2.GaussianBlur(mask, (31, 31), 0)


def _warp_cloth_to_quad(cloth_rgba: np.ndarray, quad: np.ndarray, output_shape: tuple[int, int, int]) -> np.ndarray:
    h, w = output_shape[:2]
    ch, cw = cloth_rgba.shape[:2]
    src = np.array([[0, 0], [cw - 1, 0], [cw - 1, ch - 1], [0, ch - 1]], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(src, quad.astype(np.float32))
    return cv2.warpPerspective(
        cloth_rgba,
        matrix,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )


def _dense_texture_rgba(cloth_rgba: np.ndarray) -> np.ndarray:
    alpha = cloth_rgba[:, :, 3].astype(np.float32) / 255.0
    rgb = cloth_rgba[:, :, :3].astype(np.float32)
    fg = alpha > 0.08
    if np.count_nonzero(fg) > 32:
        fill = np.median(rgb[fg], axis=0)
    else:
        fill = np.median(rgb.reshape(-1, 3), axis=0)
    dense_rgb = rgb.copy()
    dense_rgb[~fg] = fill
    dense = np.dstack(
        [
            np.clip(dense_rgb, 0, 255).astype(np.uint8),
            np.full(alpha.shape, 255, dtype=np.uint8),
        ]
    )
    return dense


def _composite_cloth_preview(person_rgb: np.ndarray, cloth_rgba: np.ndarray, category: str | int) -> tuple[np.ndarray, dict[str, Any]]:
    category_name = _category_name(category)
    x1, y1, x2, y2, rect_debug = _target_rect(person_rgb, category_name)
    pose_points = _detect_pose_points(person_rgb)
    quad, pose_debug = _body_quad_from_pose(person_rgb, category_name, pose_points)
    if quad is None:
        quad = _fallback_quad_from_rect((x1, y1, x2, y2))
    warped = _warp_cloth_to_quad(cloth_rgba, quad, person_rgb.shape)
    warped_texture = _warp_cloth_to_quad(_dense_texture_rgba(cloth_rgba), quad, person_rgb.shape)

    out = person_rgb.copy().astype(np.float32)
    rgb = warped_texture[:, :, :3].astype(np.float32)
    item_alpha = warped[:, :, 3].astype(np.float32) / 255.0
    alpha = cv2.GaussianBlur(item_alpha, (7, 7), 0) * 0.82

    protect = _protect_mask_from_pose(person_rgb.shape, pose_points, category_name)
    alpha = np.clip(alpha * (1.0 - protect * 0.92), 0.0, 0.90)

    # Let a little original luminance/shadow through so the preview is less sticker-like.
    person_gray = cv2.cvtColor(person_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    local_shadow = cv2.GaussianBlur(person_gray, (41, 41), 0) / 255.0
    shade = np.clip(0.82 + local_shadow * 0.34, 0.78, 1.08)[:, :, None]
    rgb = np.clip(rgb * shade, 0, 255)

    alpha3 = alpha[:, :, None]
    contact_shadow = cv2.GaussianBlur(alpha, (35, 35), 0)[:, :, None] * 10.0
    base = np.clip(out - contact_shadow, 0, 255)
    out = rgb * alpha3 + base * (1.0 - alpha3)
    debug = {
        **rect_debug,
        **pose_debug,
        "placed_quad": np.round(quad, 2).tolist(),
        "cloth_size": [int(cloth_rgba.shape[1]), int(cloth_rgba.shape[0])],
        "alpha_pixels": int(np.count_nonzero(alpha > 0.05)),
        "protect_pixels": int(np.count_nonzero(protect > 0.05)),
    }
    return np.clip(out, 0, 255).astype(np.uint8), debug


def run_cpu_preview_tryon(
    *,
    person_path: str | Path,
    cloth_path: str | Path,
    output_dir: str | Path,
    category: str | int = "upperbody",
    status: dict[str, Any] | None = None,
) -> dict[str, Any]:
    started = time.time()
    person_path = Path(person_path).resolve()
    cloth_path = Path(cloth_path).resolve()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    person_info = _verify_image(person_path)
    cloth_info = _verify_image(cloth_path)

    with Image.open(person_path) as person_image:
        person_rgb = np.array(person_image.convert("RGB"), dtype=np.uint8)
    cloth_rgba = _rgba_with_estimated_alpha(cloth_path)
    preview_rgb, debug = _composite_cloth_preview(person_rgb, cloth_rgba, category)

    job_id = uuid.uuid4().hex
    output_path = output_dir / f"tryon_cpu_preview_{job_id}.png"
    Image.fromarray(preview_rgb).save(output_path)
    elapsed = time.time() - started

    return {
        "success": True,
        "applied": True,
        "fallback_used": True,
        "error": "cpu_preview_fallback",
        "provider": "CPU Preview Try-On",
        "quality_warning": "Temporary geometric overlay for store/UI testing only; not realistic virtual try-on.",
        "model_type": "cpu_preview",
        "category": _category_name(category),
        "elapsed_seconds": round(elapsed, 3),
        "person": person_info,
        "cloth": cloth_info,
        "debug": debug,
        "status": status,
        "outputs": [
            {
                "path": str(output_path),
                "filename": output_path.name,
            }
        ],
    }


def _prepare_archive_inputs(
    cloth_path: Path,
    person_path: Path,
    job_input_dir: Path,
) -> dict[str, Any] | None:
    """If cloth is from the archive, prepare pre-processed data for OOTDiffusion.

    Copies agnostic, openpose, parsing, and densepose data to the job input
    directory for direct use by the OOTDiffusion inference script, skipping
    the expensive on-the-fly preprocessing step.
    """
    is_archive, cloth_stem = is_archive_cloth(cloth_path)
    if not is_archive or cloth_stem is None:
        return None

    # Find matching person from archive pairs
    person_stem = find_pair_person(cloth_stem)
    if person_stem is None:
        return None

    preprocess = get_preprocess_paths(person_stem, split="test")
    available_keys = [k for k, v in preprocess.items() if v is not None]
    if not available_keys:
        return None

    # Copy pre-processed data to job directory
    copied: dict[str, str] = {}
    for key, src_path in preprocess.items():
        if src_path is None:
            continue
        dest = job_input_dir / f"{key}{src_path.suffix}"
        try:
            shutil.copy2(src_path, dest)
            copied[key] = str(dest)
        except Exception:
            pass

    # Use archive person image if available for better results
    archive_person = get_archive_person_image(person_stem)

    return {
        "cloth_stem": cloth_stem,
        "person_stem": person_stem,
        "archive_person_path": str(archive_person) if archive_person else None,
        "preprocess_copied": copied,
        "available_keys": available_keys,
    }


def _compute_file_hash(filepath: Path) -> str:
    hasher = hashlib.md5()
    try:
        with open(filepath, 'rb') as f:
            buf = f.read(65536)
            while len(buf) > 0:
                hasher.update(buf)
                buf = f.read(65536)
        return hasher.hexdigest()
    except Exception:
        return ""

def _get_tryon_cache_dir() -> Path:
    assets_dir = Path(__file__).resolve().parent.parent.parent / "assets"
    cache_dir = assets_dir / "cache" / "tryon"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir

def run_virtual_tryon(
    *,
    person_path: str | Path,
    cloth_path: str | Path,
    output_dir: str | Path,
    model_type: str = "dc",
    category: str | int = "upperbody",
    sample: int = 1,
    steps: int = 20,
    scale: float = 2.0,
    seed: int = -1,
    timeout_seconds: int = 900,
) -> dict[str, Any]:
    # Apply CUDA optimizations on first run
    _apply_cuda_optimizations()

    status = get_virtual_tryon_status()
    if not status["available"]:
        if _cpu_preview_fallback_enabled():
            return run_cpu_preview_tryon(
                person_path=person_path,
                cloth_path=cloth_path,
                output_dir=output_dir,
                category=category,
                status=status,
            )
        return {
            "success": False,
            "applied": False,
            "fallback_used": True,
            "error": status["reason"],
            "status": status,
            "outputs": [],
        }

    root = Path(status["root"])
    run_dir = root / "run"
    script = run_dir / "run_ootd.py"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_type = _normalize_model_type(model_type)
    category_idx = _normalize_category(category, model_type)
    sample = int(max(1, min(int(sample), 4)))
    steps = int(max(5, min(int(steps), 50)))
    scale = float(max(0.5, min(float(scale), 8.0)))
    seed = int(seed)

    person_path = Path(person_path).resolve()
    cloth_path = Path(cloth_path).resolve()
    person_info = _verify_image(person_path)
    cloth_info = _verify_image(cloth_path)

    # ------------------ CACHE CHECK ------------------
    cache_dir = _get_tryon_cache_dir()
    p_hash = _compute_file_hash(person_path)
    c_hash = _compute_file_hash(cloth_path)
    cache_key = f"{p_hash}_{c_hash}_{model_type}_{category_idx}_{sample}_{steps}_{scale}_{seed}"
    cache_hash = hashlib.md5(cache_key.encode()).hexdigest()

    cached_outputs = []
    # Check if cache exists for this exact combination
    for idx in range(sample):
        cached_file = cache_dir / f"{cache_hash}_{idx}.png"
        if cached_file.exists():
            dest = output_dir / f"tryon_cached_{cache_hash}_{idx}.png"
            shutil.copy2(cached_file, dest)
            cached_outputs.append({
                "path": str(dest),
                "filename": dest.name,
                "cached": True
            })
    
    if len(cached_outputs) == sample:
        # Cache hit! Return immediately
        mask_cached = cache_dir / f"{cache_hash}_mask.jpg"
        mask_output = None
        if mask_cached.exists():
            mask_dest = output_dir / f"tryon_cached_{cache_hash}_mask.jpg"
            shutil.copy2(mask_cached, mask_dest)
            mask_output = str(mask_dest)
            
        return {
            "success": True,
            "applied": True,
            "fallback_used": False,
            "error": None,
            "provider": "OOTDiffusion (Cached)",
            "model_type": model_type,
            "category": _category_name(category),
            "elapsed_seconds": 0.05,
            "person": person_info,
            "cloth": cloth_info,
            "status": status,
            "outputs": cached_outputs,
            "mask": mask_output,
        }
    # -------------------------------------------------

    job_id = uuid.uuid4().hex
    job_input_dir = output_dir / f"tryon_inputs_{job_id}"
    job_input_dir.mkdir(parents=True, exist_ok=True)
    person_job_path = job_input_dir / "person.png"
    cloth_job_path = job_input_dir / "garment.png"

    # Check for archive pre-processed data
    archive_info = _prepare_archive_inputs(
        cloth_path, person_path, job_input_dir,
    )

    target_ratio = 0.75 if model_type == "hd" else 1.0
    with Image.open(person_path) as image:
        orig_size = image.size
        cropped_person, crop_info = _fit_image_to_aspect(image, target_ratio)
        cropped_person.convert("RGB").save(person_job_path)
    with Image.open(cloth_path) as image:
        # Pad the cloth to prevent vertical stretching which causes AI to hallucinate long sleeves
        # However, for lower body and dresses, they should be tall, so stretching is desired to prevent them from becoming shorts
        if category in ("upperbody", "upper_body"):
            padded_cloth = _pad_image_to_aspect(image, target_ratio)
        else:
            padded_cloth = image
        padded_cloth.convert("RGB").save(cloth_job_path)

    # Build command with CUDA-optimized environment
    cmd = [
        os.sys.executable,
        str(script),
        "--model_path",
        str(person_job_path),
        "--cloth_path",
        str(cloth_job_path),
        "--model_type",
        model_type,
        "--category",
        str(category_idx),
        "--scale",
        str(scale),
        "--step",
        str(steps),
        "--sample",
        str(sample),
        "--seed",
        str(seed),
    ]

    # Set CUDA environment variables for maximum performance
    env = os.environ.copy()
    env.update({
        "CUDA_LAUNCH_BLOCKING": "0",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "TORCH_CUDNN_V8_API_ENABLED": "1",
    })
    # Pass archive pre-processed data paths if available
    if archive_info and archive_info.get("preprocess_copied"):
        env["FACEWARP_ARCHIVE_PREPROCESS"] = json.dumps(
            archive_info["preprocess_copied"]
        )

    started = time.time()
    with _RUN_LOCK:
        oot_output_dir = run_dir / "images_output"
        oot_output_dir.mkdir(parents=True, exist_ok=True)
        for old in oot_output_dir.glob(f"out_{model_type}_*.png"):
            try:
                old.unlink()
            except Exception:
                pass

        completed = subprocess.run(
            cmd,
            cwd=str(run_dir),
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            env=env,
        )

        outputs: list[dict[str, Any]] = []
        for idx, src in enumerate(sorted(oot_output_dir.glob(f"out_{model_type}_*.png"))):
            dest = output_dir / f"tryon_{job_id}_{idx}.png"
            
            with Image.open(src) as out_img:
                # Resize back to crop size
                out_img = out_img.resize((crop_info["new_w"], crop_info["new_h"]), Image.LANCZOS)
                
                # Paste back onto original uncropped image
                # Because run_ootd preserved the background, this paste is seamless!
                with Image.open(person_path) as orig_image:
                    final_img = orig_image.convert("RGB")
                    final_img.paste(out_img, (crop_info["left"], crop_info["top"]))
                    final_img.save(dest)
            
            outputs.append(
                {
                    "path": str(dest),
                    "filename": dest.name,
                }
            )

            # SAVE TO CACHE
            try:
                cached_file = cache_dir / f"{cache_hash}_{idx}.png"
                shutil.copy2(dest, cached_file)
            except Exception:
                pass

        mask_src = oot_output_dir / "mask.jpg"
        mask_output = None
        if mask_src.exists():
            mask_dest = output_dir / f"tryon_{job_id}_mask.jpg"
            shutil.copy2(mask_src, mask_dest)
            mask_output = str(mask_dest)
            
            # SAVE MASK TO CACHE
            try:
                mask_cached = cache_dir / f"{cache_hash}_mask.jpg"
                shutil.copy2(mask_dest, mask_cached)
            except Exception:
                pass

    elapsed = time.time() - started
    if completed.returncode != 0:
        return {
            "success": False,
            "applied": False,
            "fallback_used": True,
            "error": "ootdiffusion_runtime_failed",
            "returncode": completed.returncode,
            "stdout": completed.stdout[-4000:],
            "stderr": completed.stderr[-4000:],
            "elapsed_seconds": round(elapsed, 3),
            "person": person_info,
            "cloth": cloth_info,
            "outputs": outputs,
        }

    if not outputs:
        return {
            "success": False,
            "applied": False,
            "fallback_used": True,
            "error": "ootdiffusion_no_output",
            "stdout": completed.stdout[-4000:],
            "stderr": completed.stderr[-4000:],
            "elapsed_seconds": round(elapsed, 3),
            "person": person_info,
            "cloth": cloth_info,
            "outputs": [],
        }

    return {
        "success": True,
        "applied": True,
        "fallback_used": False,
        "error": None,
        "provider": "OOTDiffusion",
        "model_type": model_type,
        "category": category_idx,
        "sample": sample,
        "steps": steps,
        "scale": scale,
        "seed": seed,
        "elapsed_seconds": round(elapsed, 3),
        "person": person_info,
        "cloth": cloth_info,
        "mask_path": mask_output,
        "outputs": outputs,
        "cuda_optimized": True,
        "archive_preprocess_used": archive_info is not None,
        "archive_info": archive_info,
    }
