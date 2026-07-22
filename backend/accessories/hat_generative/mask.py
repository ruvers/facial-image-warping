from __future__ import annotations

from typing import Any

import cv2
import numpy as np


FACE_OVAL = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
    397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
    172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109,
]
LEFT_BROW = [276, 283, 282, 295, 285, 300, 293, 334, 296, 336]
RIGHT_BROW = [46, 53, 52, 65, 55, 70, 63, 105, 66, 107]
EYE_POINTS = [33, 133, 159, 145, 263, 362, 386, 374]


def _landmarks(ctx: dict[str, Any] | None) -> np.ndarray | None:
    if not isinstance(ctx, dict):
        return None
    pts = ctx.get("landmarks_2d")
    if isinstance(pts, np.ndarray) and pts.ndim == 2 and pts.shape[0] > 386:
        return pts[:, :2].astype(np.float32, copy=False)
    landmarks = ctx.get("landmarks")
    if isinstance(landmarks, np.ndarray) and landmarks.ndim == 2 and landmarks.shape[0] > 386:
        return landmarks[:, :2].astype(np.float32, copy=False)
    return None


def _safe_points(points: np.ndarray, indices: list[int]) -> np.ndarray:
    rows = [points[i] for i in indices if i < points.shape[0]]
    if not rows:
        return np.zeros((0, 2), dtype=np.float32)
    return np.asarray(rows, dtype=np.float32)


def build_hat_placement_mask(
    image_shape: tuple[int, ...],
    ctx: dict[str, Any] | None,
    params: dict[str, Any] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    params = params or {}
    h, w = int(image_shape[0]), int(image_shape[1])
    mask = np.zeros((h, w), dtype=np.uint8)
    pts = _landmarks(ctx)

    if pts is not None:
        oval = _safe_points(pts, FACE_OVAL)
        brows = _safe_points(pts, LEFT_BROW + RIGHT_BROW)
        eyes = _safe_points(pts, EYE_POINTS)
    else:
        oval = np.zeros((0, 2), dtype=np.float32)
        brows = np.zeros((0, 2), dtype=np.float32)
        eyes = np.zeros((0, 2), dtype=np.float32)

    if oval.shape[0] >= 8 and brows.shape[0] >= 4:
        face_x1 = float(np.percentile(oval[:, 0], 3))
        face_x2 = float(np.percentile(oval[:, 0], 97))
        face_top_y = float(np.percentile(oval[:, 1], 2))
        face_bottom_y = float(np.percentile(oval[:, 1], 98))
        face_width = max(1.0, face_x2 - face_x1)
        face_height = max(1.0, face_bottom_y - face_top_y)
        cx = float((face_x1 + face_x2) * 0.5)
        brow_y = float(np.mean(brows[:, 1]))
        eye_y = float(np.mean(eyes[:, 1])) if eyes.shape[0] else brow_y + face_height * 0.12
        forehead_y = float(face_top_y + face_height * 0.12)
        hairline_y = float(face_top_y + face_height * 0.08)
        source = "landmarks"
        confidence = 0.74
    else:
        anchors = ctx.get("anchors", {}) if isinstance(ctx, dict) else {}
        metrics = anchors.get("metrics", {}) if isinstance(anchors, dict) else {}
        face_width = float(metrics.get("face_width", w * 0.32))
        face_height = float(metrics.get("face_height", h * 0.42))
        cx = float(w * 0.5)
        face_top_y = float(h * 0.20)
        brow_y = float(face_top_y + face_height * 0.30)
        eye_y = float(face_top_y + face_height * 0.40)
        forehead_y = float(face_top_y + face_height * 0.14)
        hairline_y = float(face_top_y + face_height * 0.10)
        source = "heuristic_fallback"
        confidence = 0.35

    bottom_limit = min(
        brow_y - 20.0,
        eye_y - max(32.0, face_height * 0.12),
    )
    requested_offset_y = float(params.get("offset_y", 0.0))
    bottom_y = float(min(forehead_y + requested_offset_y, bottom_limit))
    top_y = float(max(0.0, bottom_y - max(face_height * 0.34, face_width * 0.58)))
    rx = float(face_width * 0.72 * float(params.get("width_scale", 1.0)))

    # Rounded target region above forehead. It intentionally excludes the lower
    # face center by clamping bottom_y above eyebrows and only filling the dome.
    dome_h = max(4.0, bottom_y - top_y)
    polygon: list[list[float]] = []
    for t in np.linspace(-1.0, 1.0, 42):
        x = cx + t * rx
        y = bottom_y - dome_h * (0.22 + 0.78 * np.sqrt(max(0.0, 1.0 - t * t)))
        polygon.append([x, y])
    for t in np.linspace(1.0, -1.0, 42):
        x = cx + t * rx * 0.96
        y = bottom_y - 3.0 * (1.0 - t * t)
        polygon.append([x, y])

    poly = np.asarray(polygon, dtype=np.float32)
    poly[:, 0] = np.clip(poly[:, 0], 0, w - 1)
    poly[:, 1] = np.clip(poly[:, 1], 0, h - 1)
    cv2.fillPoly(mask, [poly.astype(np.int32)], 255, cv2.LINE_AA)

    # Hard-clear everything under the safe bottom line so mask cannot enter eye
    # or face-center regions even after antialiasing.
    clear_y = int(max(0, min(h, np.floor(bottom_y + 1))))
    if clear_y < h:
        mask[clear_y:, :] = 0

    k = int(max(5, min(h, w) * 0.015)) | 1
    mask = cv2.GaussianBlur(mask, (k, k), 0)
    if clear_y < h:
        mask[clear_y:, :] = 0

    mask_area = int(np.count_nonzero(mask > 10))
    debug = {
        "source": source,
        "confidence": float(confidence),
        "mask_area": mask_area,
        "forehead_y": float(forehead_y),
        "hairline_y": float(hairline_y),
        "brow_y": float(brow_y),
        "eye_y": float(eye_y),
        "bottom_y": float(bottom_y),
        "top_y": float(top_y),
        "bottom_above_brow": bool(bottom_y <= brow_y - 20.0),
    }
    return mask, debug
