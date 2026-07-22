from __future__ import annotations

import time
import traceback
from typing import Any

import cv2
import numpy as np

from backend.three_d.deca_runtime import get_deca_runtime


def is_deca_available() -> bool:
    runtime = get_deca_runtime()
    return runtime.is_available()


def get_deca_runtime_status() -> dict[str, Any]:
    runtime = get_deca_runtime()
    return runtime.status()


def preload_deca() -> dict[str, Any]:
    """
    Load the shared DECA runtime once for startup preloading.

    This keeps renderer disabled through DecaRuntime.load().
    It does not run reconstruction, so startup preload only pays model load cost.
    """

    started = time.perf_counter()
    runtime = get_deca_runtime()

    if runtime.loaded and runtime.deca is not None:
        return {
            "ok": True,
            "loaded": True,
            "cache_hit": True,
            "provider": "deca_flame",
            "seconds": round(time.perf_counter() - started, 3),
            "status": runtime.status(),
        }

    loaded = runtime.load()
    status = runtime.status()

    return {
        "ok": bool(loaded),
        "loaded": bool(loaded),
        "cache_hit": False,
        "provider": "deca_flame",
        "seconds": round(time.perf_counter() - started, 3),
        "status": status,
        "error": None if loaded else status.get("last_error"),
    }


def _get_face_bbox_from_ctx(
    ctx: dict,
    image_bgr: np.ndarray,
) -> tuple[float, float, float, float]:
    """
    Get face bbox from MediaPipe landmarks.

    Returns:
        x1, y1, x2, y2
    """

    h, w = image_bgr.shape[:2]

    pts = ctx.get("landmarks_2d")

    if pts is None:
        pts = ctx.get("landmarks")

    if not isinstance(pts, np.ndarray) or pts.size == 0:
        return (
            float(w * 0.25),
            float(h * 0.12),
            float(w * 0.75),
            float(h * 0.88),
        )

    pts = np.asarray(
        pts,
        dtype=np.float32,
    )

    x1 = float(np.percentile(pts[:, 0], 2))
    y1 = float(np.percentile(pts[:, 1], 2))
    x2 = float(np.percentile(pts[:, 0], 98))
    y2 = float(np.percentile(pts[:, 1], 98))

    bw = x2 - x1
    bh = y2 - y1

    pad_x = bw * 0.10
    pad_y_top = bh * 0.20
    pad_y_bottom = bh * 0.10

    x1 -= pad_x
    x2 += pad_x
    y1 -= pad_y_top
    y2 += pad_y_bottom

    x1 = max(0.0, min(float(w - 1), x1))
    y1 = max(0.0, min(float(h - 1), y1))
    x2 = max(0.0, min(float(w - 1), x2))
    y2 = max(0.0, min(float(h - 1), y2))

    return x1, y1, x2, y2


def _make_square_bbox(
    bbox: tuple[float, float, float, float],
    image_shape,
    scale: float = 1.25,
) -> tuple[int, int, int, int]:
    """
    Convert face bbox to square crop bbox.
    DECA expects a face-centered crop.
    """

    h, w = image_shape[:2]

    x1, y1, x2, y2 = bbox

    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5

    bw = x2 - x1
    bh = y2 - y1

    side = max(bw, bh) * scale

    sx1 = int(round(cx - side * 0.5))
    sy1 = int(round(cy - side * 0.5))
    sx2 = int(round(cx + side * 0.5))
    sy2 = int(round(cy + side * 0.5))

    sx1 = max(0, min(w - 1, sx1))
    sy1 = max(0, min(h - 1, sy1))
    sx2 = max(0, min(w, sx2))
    sy2 = max(0, min(h, sy2))

    if sx2 <= sx1 + 2 or sy2 <= sy1 + 2:
        return 0, 0, w, h

    return sx1, sy1, sx2, sy2


def _crop_for_deca(
    ctx: dict,
    image_bgr: np.ndarray,
) -> tuple[np.ndarray, tuple[int, int, int, int], tuple[float, float, float, float]]:
    """
    Crop face before sending to DECA.

    Returns:
        crop_bgr,
        crop_box,
        face_bbox
    """

    face_bbox = _get_face_bbox_from_ctx(
        ctx,
        image_bgr,
    )

    crop_box = _make_square_bbox(
        face_bbox,
        image_bgr.shape,
        scale=1.25,
    )

    x1, y1, x2, y2 = crop_box

    crop = image_bgr[y1:y2, x1:x2].copy()

    if crop.size == 0:
        return image_bgr.copy(), (0, 0, image_bgr.shape[1], image_bgr.shape[0]), face_bbox

    return crop, crop_box, face_bbox

def _fit_deca_points_to_target_bbox(
    points: np.ndarray,
    target_bbox: tuple[float, float, float, float],
    image_bgr: np.ndarray,
    keep_z: bool = False,
) -> np.ndarray:
    """
    Project raw DECA landmarks into original image space.

    DECA landmarks now come mostly as normalized [-1, 1] coordinates.
    Since DECA is run on face crop, map those coordinates directly into crop_box.
    """

    if points is None:
        if keep_z:
            return np.zeros((0, 3), dtype=np.float32)
        return np.zeros((0, 2), dtype=np.float32)

    arr = np.asarray(points, dtype=np.float32)

    if arr.ndim != 2 or arr.shape[0] == 0:
        if keep_z:
            return np.zeros((0, 3), dtype=np.float32)
        return np.zeros((0, 2), dtype=np.float32)

    xy = arr[:, :2].copy()

    x1, y1, x2, y2 = target_bbox
    tw = float(x2 - x1)
    th = float(y2 - y1)

    finite_xy = xy[np.isfinite(xy)]

    if finite_xy.size == 0:
        if keep_z:
            return np.zeros((arr.shape[0], 3), dtype=np.float32)
        return np.zeros((arr.shape[0], 2), dtype=np.float32)

    min_v = float(np.min(finite_xy))
    max_v = float(np.max(finite_xy))
    max_abs = float(np.max(np.abs(finite_xy)))

    out_xy = np.zeros_like(xy, dtype=np.float32)

    # Case 1: normalized DECA coordinates [-1, 1]
    if max_abs <= 2.5:
        out_xy[:, 0] = float(x1) + ((xy[:, 0] + 1.0) * 0.5) * tw
        out_xy[:, 1] = float(y1) + ((xy[:, 1] + 1.0) * 0.5) * th

    # Case 2: 224 crop coordinates
    elif min_v >= -10.0 and max_v <= 234.0:
        out_xy[:, 0] = float(x1) + (xy[:, 0] / 224.0) * tw
        out_xy[:, 1] = float(y1) + (xy[:, 1] / 224.0) * th

    # Case 3: fallback min/max fit
    else:
        sx1 = float(np.min(xy[:, 0]))
        sy1 = float(np.min(xy[:, 1]))
        sx2 = float(np.max(xy[:, 0]))
        sy2 = float(np.max(xy[:, 1]))

        sw = sx2 - sx1
        sh = sy2 - sy1

        if sw < 1e-6 or sh < 1e-6:
            if keep_z:
                return np.zeros((arr.shape[0], 3), dtype=np.float32)
            return np.zeros((arr.shape[0], 2), dtype=np.float32)

        out_xy[:, 0] = float(x1) + ((xy[:, 0] - sx1) / sw) * tw
        out_xy[:, 1] = float(y1) + ((xy[:, 1] - sy1) / sh) * th

    h, w = image_bgr.shape[:2]

    out_xy[:, 0] = np.clip(out_xy[:, 0], 0, w - 1)
    out_xy[:, 1] = np.clip(out_xy[:, 1], 0, h - 1)

    if keep_z:
        out = np.zeros((arr.shape[0], 3), dtype=np.float32)
        out[:, 0:2] = out_xy

        if arr.shape[1] >= 3:
            out[:, 2] = arr[:, 2]

        return out

    return out_xy.astype(np.float32)

def _make_deca_anchor_points(
    landmarks2d: np.ndarray,
    landmarks3d: np.ndarray,
) -> dict:
    """
    Build simple 68-landmark DECA anchor points.
    """

    def p2(idx: int) -> list[float]:
        if landmarks2d is None or idx >= len(landmarks2d):
            return [0.0, 0.0]

        return [
            float(landmarks2d[idx][0]),
            float(landmarks2d[idx][1]),
        ]

    def p3(idx: int) -> list[float]:
        if landmarks3d is None or idx >= len(landmarks3d):
            return [0.0, 0.0, 0.0]

        p = landmarks3d[idx]

        if len(p) < 3:
            return [float(p[0]), float(p[1]), 0.0]

        return [
            float(p[0]),
            float(p[1]),
            float(p[2]),
        ]

    return {
        "landmark_type": "deca_68",

        "jaw_left": p2(0),
        "jaw_right": p2(16),
        "chin": p2(8),

        "nose_bridge": p2(27),
        "nose_tip": p2(30),

        "left_eye_outer": p2(36),
        "left_eye_inner": p2(39),
        "right_eye_inner": p2(42),
        "right_eye_outer": p2(45),

        "mouth_left": p2(48),
        "mouth_right": p2(54),

        "chin_3d": p3(8),
        "nose_tip_3d": p3(30),
        "left_eye_outer_3d": p3(36),
        "right_eye_outer_3d": p3(45),
    }


def _fallback_depth_map(
    ctx: dict,
    image_bgr: np.ndarray,
) -> np.ndarray:
    h, w = image_bgr.shape[:2]

    depth = ctx.get("depth_map")

    if isinstance(depth, np.ndarray):
        if depth.shape[:2] != (h, w):
            depth = cv2.resize(
                depth.astype(np.float32),
                (w, h),
                interpolation=cv2.INTER_LINEAR,
            )

        return depth.astype(np.float32)

    return np.zeros(
        (h, w),
        dtype=np.float32,
    )


def enrich_with_deca_context(
    ctx: dict,
    image_bgr: np.ndarray,
) -> dict | None:
    """
    DECA/FLAME true-3D provider.

    If successful:
        fills ctx["three_d"] with DECA mesh.
    """

    try:
        runtime = get_deca_runtime()

        status = runtime.status()

        ctx.setdefault("three_d_model_status", {})
        ctx["three_d_model_status"]["deca"] = status

        if not status.get("available", False):
            return None

        crop_bgr, crop_box, face_bbox = _crop_for_deca(
            ctx,
            image_bgr,
        )

        result = runtime.reconstruct(
            crop_bgr,
        )

        if result is None:
            ctx["three_d_model_status"]["deca"] = runtime.status()
            return None

        vertices = result.get("vertices")
        faces = result.get("faces")

        landmarks2d_raw = result.get("landmarks2d")
        landmarks3d_raw = result.get("landmarks3d")

        # Fit DECA raw 68 points to original image face bbox.
        landmarks2d = _fit_deca_points_to_target_bbox(
            landmarks2d_raw,
            crop_box,
            image_bgr,
            keep_z=False,
        )

        landmarks3d = _fit_deca_points_to_target_bbox(
            landmarks3d_raw,
            crop_box,
            image_bgr,
            keep_z=True,
        )

        if not isinstance(vertices, np.ndarray) or vertices.size == 0:
            return None

        if not isinstance(faces, np.ndarray):
            faces = np.zeros((0, 3), dtype=np.int32)

        if not isinstance(landmarks2d, np.ndarray):
            landmarks2d = np.zeros((0, 2), dtype=np.float32)

        if not isinstance(landmarks3d, np.ndarray):
            landmarks3d = np.zeros((0, 3), dtype=np.float32)

        depth_map = result.get("depth_map")

        if not isinstance(depth_map, np.ndarray):
            depth_map = _fallback_depth_map(
                ctx,
                image_bgr,
            )

        anchor_points = _make_deca_anchor_points(
            landmarks2d,
            landmarks3d,
        )

        ctx["three_d"] = {
            "provider": "deca_flame",
            "provider_priority": "deca_first",
            "is_true_3d": True,

            "camera": result.get("camera", {}),

            "crop_box": {
                "x1": int(crop_box[0]),
                "y1": int(crop_box[1]),
                "x2": int(crop_box[2]),
                "y2": int(crop_box[3]),
            },

            "face_bbox": {
                "x1": float(face_bbox[0]),
                "y1": float(face_bbox[1]),
                "x2": float(face_bbox[2]),
                "y2": float(face_bbox[3]),
            },

            "mesh": {
                "type": "deca_flame_mesh",
                "vertices": vertices.astype(np.float32),
                "faces": faces.astype(np.int32),
                "is_true_3d": True,
                "vertex_count": int(vertices.shape[0]),
                "face_count": int(faces.shape[0]),
            },

            "vertices": vertices.astype(np.float32),
            "faces": faces.astype(np.int32),

            "landmarks2d": landmarks2d.astype(np.float32),
            "landmarks3d": landmarks3d.astype(np.float32),

            "landmarks2d_raw": landmarks2d_raw.astype(np.float32)
            if isinstance(landmarks2d_raw, np.ndarray)
            else np.zeros((0, 2), dtype=np.float32),

            "landmarks3d_raw": landmarks3d_raw.astype(np.float32)
            if isinstance(landmarks3d_raw, np.ndarray)
            else np.zeros((0, 3), dtype=np.float32),

            "depth_map": depth_map.astype(np.float32),

            "anchor_points": anchor_points,

            "flame_params": result.get("flame_params", {}),

            "renderer_enabled": bool(
                result.get("renderer_enabled", False)
            ),
        }

        ctx["three_d_provider"] = "deca_flame"

        ctx["three_d_model_status"]["deca"] = runtime.status()

        return ctx

    except Exception as e:
        traceback.print_exc()

        ctx.setdefault("three_d_model_status", {})
        ctx["three_d_model_status"]["deca"] = {
            "provider": "deca_flame",
            "available": False,
            "runtime_ready": False,
            "error": str(e),
        }

        return None
