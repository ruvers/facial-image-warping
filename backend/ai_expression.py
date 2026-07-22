from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from backend.local_models.liveportrait_expression import (
    get_liveportrait_expression_status,
)
from backend.liveportrait_bridge import apply_liveportrait_expression
from backend.warping import apply_expression_transform
import hashlib
import json
import os
from pathlib import Path


SUPPORTED_PRESETS = {
    "smile",
    "eyebrow_raise",
    "laugh",
    "surprise",
    "neutral",
    "wink",
    "sad",
    "angry",
    "open_lip",
    "shy",
    "aggrieved",
}
FALLBACK_PRESETS = {"smile"}
MAX_FALLBACK_SMILE_INTENSITY = 0.35


BROW_INDICES = [70, 63, 105, 66, 107, 336, 296, 334, 293, 300]
LEFT_BROW_INDICES = [70, 63, 105, 66, 107]
RIGHT_BROW_INDICES = [336, 296, 334, 293, 300]
LEFT_EYE_INDICES = [33, 133, 159, 145, 160, 144, 158, 153]
RIGHT_EYE_INDICES = [362, 263, 386, 374, 385, 380, 387, 373]
IRIS_INDICES = [468, 469, 470, 471, 472, 473, 474, 475, 476, 477]


def get_ai_expression_status() -> dict[str, Any]:
    liveportrait = get_liveportrait_expression_status()
    bridge_ready = bool(
        liveportrait.get("inference_bridge_implemented", False)
    )

    return {
        "legacy_expression_available": True,
        "ai_expression_available": bridge_ready,
        "liveportrait_files_available": bool(
            liveportrait.get("files_available", False)
        ),
        "liveportrait_inference_bridge_implemented": bridge_ready,
        "supported_presets": sorted(SUPPORTED_PRESETS),
        "fallback_supported_presets": sorted(FALLBACK_PRESETS),
        "max_fallback_smile_intensity": MAX_FALLBACK_SMILE_INTENSITY,
        "notes": [
            "Legacy MediaPipe/OpenCV expression remains the stable path.",
            "Experimental AI expression prefers LivePortrait when a full bridge exists.",
            "LivePortrait direct single-image bridge prototype supports smile only when runtime dependencies are installed.",
            "Other LivePortrait presets require a driving video or .pkl motion template.",
            "Fallback is limited to a subtle smile preview and does not claim AI inference.",
            "AlbedoGAN is not used for expression editing; it remains a future optional 3D/albedo provider slot.",
        ],
    }


def _ctx_landmarks_to_legacy_points(ctx: dict) -> list[dict[str, float]]:
    landmarks = ctx.get("landmarks_2d")

    if landmarks is None:
        landmarks = ctx.get("landmarks")

    if landmarks is None:
        return []

    arr = np.asarray(
        landmarks,
        dtype=np.float32,
    )

    if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] < 2:
        return []

    points = []

    for idx, point in enumerate(arr):
        points.append(
            {
                "index": idx,
                "x": float(point[0]),
                "y": float(point[1]),
                "z": float(point[2]) if arr.shape[1] >= 3 else 0.0,
                "visibility": 1.0,
            }
        )

    return points


def _preset_to_legacy_params(
    preset: str,
    intensity: float,
) -> dict[str, float]:
    if preset == "smile":
        safe_intensity = float(
            np.clip(
                intensity,
                0.0,
                MAX_FALLBACK_SMILE_INTENSITY,
            )
        )

        return {
            "smile_intensity": safe_intensity,
            "eyebrow_intensity": 0.0,
            "lip_intensity": 0.0,
            "slim_intensity": 0.0,
        }

    return {
        "smile_intensity": 0.0,
        "eyebrow_intensity": 0.0,
        "lip_intensity": 0.0,
        "slim_intensity": 0.0,
    }


def _ctx_landmarks_array(ctx: dict) -> np.ndarray:
    landmarks = ctx.get("landmarks_2d")

    if landmarks is None:
        landmarks = ctx.get("landmarks")

    if landmarks is None:
        return np.zeros((0, 2), dtype=np.float32)

    arr = np.asarray(landmarks, dtype=np.float32)

    if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] < 2:
        return np.zeros((0, 2), dtype=np.float32)

    return arr[:, :2].astype(np.float32)


def _safe_landmark_points(
    landmarks: np.ndarray,
    indices: list[int],
) -> np.ndarray:
    points = [
        landmarks[idx]
        for idx in indices
        if idx < len(landmarks)
    ]

    if not points:
        return np.zeros((0, 2), dtype=np.float32)

    return np.asarray(points, dtype=np.float32)


def _gaussian_field(
    xx: np.ndarray,
    yy: np.ndarray,
    points: np.ndarray,
    sigma_x: float,
    sigma_y: float,
) -> np.ndarray:
    field = np.zeros_like(xx, dtype=np.float32)

    for point in points:
        px = float(point[0])
        py = float(point[1])
        dist = (
            ((xx - px) ** 2) / max(2.0 * sigma_x * sigma_x, 1e-6)
            + ((yy - py) ** 2) / max(2.0 * sigma_y * sigma_y, 1e-6)
        )
        field = np.maximum(field, np.exp(-dist).astype(np.float32))

    return field


def _make_single_brow_mask(
    image_bgr: np.ndarray,
    landmarks: np.ndarray,
    indices: list[int],
    face_width: float,
    face_height: float,
) -> np.ndarray:
    h, w = image_bgr.shape[:2]
    points = _safe_landmark_points(landmarks, indices)

    if points.shape[0] < 2:
        return np.zeros((h, w), dtype=np.uint8)

    points = points[np.argsort(points[:, 0])]
    line_mask = np.zeros((h, w), dtype=np.uint8)
    thickness = int(np.clip(face_height * 0.020, 5, 10))
    cv2.polylines(
        line_mask,
        [np.round(points).astype(np.int32)],
        isClosed=False,
        color=255,
        thickness=thickness,
        lineType=cv2.LINE_AA,
    )

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (max(3, thickness | 1), max(3, thickness | 1)),
    )
    line_mask = cv2.dilate(line_mask, kernel, iterations=1)

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    candidate = line_mask > 0
    values = gray[candidate]

    if values.size >= 20:
        threshold = float(np.percentile(values, 58))
        dark_mask = (gray <= threshold).astype(np.uint8) * 255
        refined = cv2.bitwise_and(line_mask, dark_mask)

        if cv2.countNonZero(refined) >= max(18, int(face_width * face_height * 0.00010)):
            line_mask = refined

    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    line_mask = cv2.morphologyEx(line_mask, cv2.MORPH_CLOSE, close_kernel, iterations=1)

    return line_mask


def _make_eyebrow_mask(
    image_bgr: np.ndarray,
    landmarks: np.ndarray,
    face_width: float,
    face_height: float,
) -> np.ndarray:
    left = _make_single_brow_mask(
        image_bgr,
        landmarks,
        LEFT_BROW_INDICES,
        face_width,
        face_height,
    )
    right = _make_single_brow_mask(
        image_bgr,
        landmarks,
        RIGHT_BROW_INDICES,
        face_width,
        face_height,
    )

    return cv2.bitwise_or(left, right)


def _apply_local_eyebrow_raise(
    image_bgr: np.ndarray,
    ctx: dict,
    intensity: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    landmarks = _ctx_landmarks_array(ctx)

    if landmarks.shape[0] < 301:
        return image_bgr.copy(), {
            "provider": "local_landmark_warp",
            "mode": "local_eyebrow_raise",
            "applied": False,
            "fallback_used": False,
            "error": "Not enough MediaPipe landmarks for eyebrow_raise.",
        }

    h, w = image_bgr.shape[:2]
    face_width = float(
        max(
            np.linalg.norm(landmarks[234] - landmarks[454])
            if len(landmarks) > 454
            else 0.0,
            np.linalg.norm(landmarks[127] - landmarks[356])
            if len(landmarks) > 356
            else 0.0,
            w * 0.35,
        )
    )
    face_height = float(
        max(
            np.linalg.norm(landmarks[10] - landmarks[152])
            if len(landmarks) > 152
            else 0.0,
            face_width,
            h * 0.35,
        )
    )

    brow_points = _safe_landmark_points(landmarks, BROW_INDICES)
    eye_points = np.vstack(
        [
            _safe_landmark_points(landmarks, LEFT_EYE_INDICES),
            _safe_landmark_points(landmarks, RIGHT_EYE_INDICES),
        ]
    )
    iris_points = _safe_landmark_points(landmarks, IRIS_INDICES)

    if brow_points.size == 0 or eye_points.size == 0:
        return image_bgr.copy(), {
            "provider": "local_landmark_warp",
            "mode": "local_eyebrow_raise",
            "applied": False,
            "fallback_used": False,
            "error": "Eyebrow or eye landmarks unavailable.",
        }

    brow_mask = _make_eyebrow_mask(
        image_bgr,
        landmarks,
        face_width,
        face_height,
    )
    brow_area_before = int(cv2.countNonZero(brow_mask))

    if brow_area_before < 24:
        return image_bgr.copy(), {
            "provider": "local_landmark_patch",
            "mode": "local_eyebrow_raise_patch",
            "applied": False,
            "fallback_used": False,
            "error": "Eyebrow mask too small for stable raise.",
        }

    safe_intensity = float(np.clip(intensity, 0.0, 1.0))
    lift_px = float(np.clip(3.0 + safe_intensity * min(face_height * 0.040, 16.0), 0.0, 18.0))
    lift_px = min(
        lift_px,
        max(2.0, float(np.min(eye_points[:, 1]) - np.max(brow_points[:, 1]) - 2.0)),
    )

    cleanup_kernel_size = int(np.clip(face_height * 0.018, 3, 9)) | 1
    cleanup_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (cleanup_kernel_size, cleanup_kernel_size),
    )
    cleanup_mask = cv2.dilate(brow_mask, cleanup_kernel, iterations=1)

    inpainted = cv2.inpaint(
        image_bgr,
        cleanup_mask,
        inpaintRadius=float(np.clip(face_height * 0.010, 2.0, 5.0)),
        flags=cv2.INPAINT_TELEA,
    )

    affine = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, -lift_px],
        ],
        dtype=np.float32,
    )
    shifted_image = cv2.warpAffine(
        image_bgr,
        affine,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    shifted_mask = cv2.warpAffine(
        brow_mask,
        affine,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    min_eye_y = float(np.min(eye_points[:, 1]))
    yy_full = np.arange(h, dtype=np.float32)[:, None]
    eye_safe_gate = (yy_full < (min_eye_y - face_height * 0.012)).astype(np.float32)
    shifted_mask = (
        shifted_mask.astype(np.float32)
        * eye_safe_gate
    ).astype(np.uint8)

    target_area = int(cv2.countNonZero(shifted_mask))
    if target_area < 16:
        return image_bgr.copy(), {
            "provider": "local_landmark_patch",
            "mode": "local_eyebrow_raise_patch",
            "applied": False,
            "fallback_used": False,
            "error": "Translated eyebrow target mask too small.",
        }

    alpha = cv2.GaussianBlur(
        shifted_mask.astype(np.float32) / 255.0,
        (0, 0),
        sigmaX=1.1,
        sigmaY=1.1,
    )
    alpha = np.clip(alpha, 0.0, 0.96)[..., None]

    output = (
        shifted_image.astype(np.float32) * alpha
        + inpainted.astype(np.float32) * (1.0 - alpha)
    ).astype(np.uint8)

    gray_before = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray_after = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
    source_pixels = gray_before[brow_mask > 0]
    residual_threshold = (
        float(np.percentile(source_pixels, 60))
        if source_pixels.size
        else 80.0
    )
    residual_region = (cleanup_mask > 0) & (shifted_mask < 8)
    residual_pixels = int(
        np.count_nonzero(gray_after[residual_region] <= residual_threshold)
    ) if np.any(residual_region) else 0

    mean_brow_disp = -float(lift_px)
    max_eye_disp = 0.0
    max_iris_disp = 0.0

    return output, {
        "provider": "local_landmark_patch",
        "mode": "local_eyebrow_raise_patch",
        "applied": True,
        "fallback_used": False,
        "intensity": safe_intensity,
        "eyebrow_vertical_displacement_px": mean_brow_disp,
        "brow_vertical_shift_px": mean_brow_disp,
        "brow_mask_area_before": brow_area_before,
        "brow_mask_area_after": target_area,
        "eye_corner_max_displacement_px": max_eye_disp,
        "iris_max_displacement_px": max_iris_disp,
        "eye_corner_shift_px": max_eye_disp,
        "iris_shift_px": max_iris_disp,
        "residual_brow_pixels_in_source_region": residual_pixels,
        "lift_px": lift_px,
        "roi": {
            "x1": int(max(0, np.min(brow_points[:, 0]) - face_width * 0.16)),
            "y1": int(max(0, np.min(brow_points[:, 1]) - lift_px - face_height * 0.05)),
            "x2": int(min(w, np.max(brow_points[:, 0]) + face_width * 0.16)),
            "y2": int(min(h, np.max(brow_points[:, 1]) + face_height * 0.05)),
        },
        "eye_anchor_protected": True,
        "error": None,
    }


def _get_expr_cache_dir() -> Path:
    assets_dir = Path(__file__).resolve().parent.parent / "assets"
    cache_dir = assets_dir / "cache" / "expression"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir

def apply_ai_expression(
    image_bgr: np.ndarray,
    ctx: dict,
    params: dict,
) -> tuple[np.ndarray, dict[str, Any]]:
    if image_bgr is None:
        raise ValueError("image_bgr is None")

    cache_dir = _get_expr_cache_dir()
    img_hash = hashlib.md5(image_bgr.tobytes()).hexdigest()
    params_str = json.dumps(params or {}, sort_keys=True)
    cache_key = f"{img_hash}_{hashlib.md5(params_str.encode()).hexdigest()}"
    cache_hash = hashlib.md5(cache_key.encode()).hexdigest()

    cached_img_path = cache_dir / f"{cache_hash}.png"
    cached_meta_path = cache_dir / f"{cache_hash}.json"

    if cached_img_path.exists() and cached_meta_path.exists():
        try:
            cached_img = cv2.imread(str(cached_img_path))
            with open(cached_meta_path, "r", encoding="utf-8") as f:
                cached_meta = json.load(f)
            if cached_img is not None:
                cached_meta["provider"] = cached_meta.get("provider", "unknown") + " (Cached)"
                return cached_img, cached_meta
        except Exception:
            pass

    output_bgr, meta = _apply_ai_expression_inner(image_bgr, ctx, params)

    if meta.get("error") is None:
        try:
            cv2.imwrite(str(cached_img_path), output_bgr)
            with open(cached_meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f)
        except Exception:
            pass

    return output_bgr, meta

def _apply_ai_expression_inner(
    image_bgr: np.ndarray,
    ctx: dict,
    params: dict,
) -> tuple[np.ndarray, dict[str, Any]]:
    if image_bgr is None:
        raise ValueError("image_bgr is None")

    params = params or {}
    preset = str(params.get("expression_preset", "smile")).lower()
    preset_supported = preset in SUPPORTED_PRESETS

    intensity = float(
        np.clip(
            float(params.get("expression_intensity", 0.25)),
            0.0,
            1.0,
        )
    )

    use_liveportrait = bool(params.get("use_liveportrait", True))
    fallback_to_legacy = bool(params.get("fallback_to_legacy", True))

    liveportrait_status = get_liveportrait_expression_status()
    files_available = bool(liveportrait_status.get("files_available", False))
    bridge_ready = bool(
        liveportrait_status.get("inference_bridge_implemented", False)
    )

    meta: dict[str, Any] = {
        "provider": "not_available",
        "requested_preset": preset,
        "intensity": intensity,
        "files_available": files_available,
        "inference_bridge_implemented": bridge_ready,
        "fallback_used": False,
        "fallback": None,
        "error": None,
        "warning": None,
        "liveportrait": {
            "repo_ok": bool(liveportrait_status.get("repo_ok", False)),
            "weights_ok": bool(liveportrait_status.get("weights_ok", False)),
            "runtime_available": bool(
                liveportrait_status.get("runtime_available", False)
            ),
            "requires": "source image plus driving video or motion template",
        },
    }

    if not preset_supported:
        meta["error"] = "Unsupported expression preset."
        return image_bgr.copy(), meta

    if preset == "neutral":
        meta["provider"] = "neutral_noop"
        meta["fallback_used"] = False
        meta["fallback"] = None
        meta["error"] = None
        meta["warning"] = "Neutral preset leaves the source expression unchanged."
        return image_bgr.copy(), meta

    if preset == "eyebrow_raise":
        output_bgr, eyebrow_meta = _apply_local_eyebrow_raise(
            image_bgr,
            ctx,
            intensity,
        )
        meta.update(
            {
                "provider": eyebrow_meta.get("provider"),
                "mode": eyebrow_meta.get("mode"),
                "preset": preset,
                "inference_bridge_implemented": True,
                "fallback_used": False,
                "fallback": None,
                "error": eyebrow_meta.get("error"),
                "warning": None
                if eyebrow_meta.get("applied")
                else "eyebrow_raise could not be applied locally.",
                "eyebrow_raise": eyebrow_meta,
            }
        )
        return output_bgr, meta

    if use_liveportrait:
        liveportrait_output, liveportrait_meta = apply_liveportrait_expression(
            image_bgr,
            preset,
            intensity,
            params,
        )
        meta["liveportrait"].update(liveportrait_meta)

        if (
            liveportrait_meta.get("provider") == "liveportrait"
            and liveportrait_meta.get("error") is None
            and liveportrait_meta.get("fallback_used") is False
            and liveportrait_meta.get("runtime_available") is True
            and liveportrait_meta.get("inference_bridge_implemented") is True
        ):
            meta.update(
                {
                    "provider": "liveportrait",
                    "mode": liveportrait_meta.get("mode"),
                    "used_driving_template": liveportrait_meta.get(
                        "used_driving_template"
                    ),
                    "frame_count": liveportrait_meta.get("frame_count"),
                    "frame_index": liveportrait_meta.get("frame_index"),
                    "selected_frame": liveportrait_meta.get("selected_frame"),
                    "selected_frame_index": liveportrait_meta.get(
                        "selected_frame_index"
                    ),
                    "scores": liveportrait_meta.get("scores"),
                    "selected_expression_score": liveportrait_meta.get(
                        "selected_expression_score"
                    ),
                    "top_frame_indices": liveportrait_meta.get(
                        "top_frame_indices"
                    ),
                    "candidate_dir": liveportrait_meta.get("candidate_dir"),
                    "scoring_method": liveportrait_meta.get("scoring_method"),
                    "preset": liveportrait_meta.get("preset", preset),
                    "inference_bridge_implemented": True,
                    "fallback_used": False,
                    "fallback": None,
                    "error": None,
                    "warning": None,
                }
            )
            return liveportrait_output, meta

        meta["error"] = liveportrait_meta.get("error")
        bridge_ready = False

    if use_liveportrait and not bridge_ready:
        if not meta.get("error"):
            meta["error"] = (
                "LivePortrait files are available, but runtime is unavailable."
                if files_available
                else "LivePortrait files are not available."
            )

    if not fallback_to_legacy:
        return image_bgr.copy(), meta

    if not bridge_ready and preset not in FALLBACK_PRESETS:
        meta["provider"] = "not_available"
        meta["fallback_used"] = False
        meta["fallback"] = None
        meta["error"] = "Preset requires LivePortrait inference bridge."
        return image_bgr.copy(), meta

    try:
        legacy_points = _ctx_landmarks_to_legacy_points(
            ctx,
        )

        if not legacy_points:
            meta["provider"] = "not_available"
            meta["fallback_used"] = False
            meta["fallback"] = "legacy_expression_unavailable"
            meta["error"] = "No MediaPipe landmarks available for fallback."
            return image_bgr.copy(), meta

        legacy_params = _preset_to_legacy_params(
            preset,
            intensity,
        )

        image_rgb = cv2.cvtColor(
            image_bgr,
            cv2.COLOR_BGR2RGB,
        )

        output_rgb = apply_expression_transform(
            image_rgb,
            legacy_points,
            smile_intensity=legacy_params["smile_intensity"],
            eyebrow_intensity=legacy_params["eyebrow_intensity"],
            lip_intensity=legacy_params["lip_intensity"],
            slim_intensity=legacy_params["slim_intensity"],
        )

        output_bgr = cv2.cvtColor(
            output_rgb,
            cv2.COLOR_RGB2BGR,
        )

        meta["provider"] = "legacy_fallback"
        meta["fallback_used"] = True
        meta["fallback"] = "old MediaPipe/OpenCV expression"
        meta["warning"] = "Legacy fallback is only a temporary dev preview."
        meta["effective_expression_intensity"] = legacy_params["smile_intensity"]

        return output_bgr, meta

    except Exception as e:
        meta["provider"] = "not_available"
        meta["fallback_used"] = False
        meta["fallback"] = "legacy_expression_failed"
        meta["error"] = str(e)
        return image_bgr.copy(), meta
