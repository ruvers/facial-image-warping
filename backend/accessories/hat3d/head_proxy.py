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
LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]


def _landmarks(ctx: dict[str, Any]) -> np.ndarray | None:
    pts = ctx.get("landmarks_2d")
    if isinstance(pts, np.ndarray) and pts.ndim == 2 and pts.shape[0] > 454:
        return pts.astype(np.float32)
    return None


def _safe_points(points: np.ndarray, indices: list[int]) -> np.ndarray:
    return np.asarray([points[i] for i in indices if i < points.shape[0]], dtype=np.float32)


def _mask_bbox(mask: np.ndarray | None) -> dict[str, float] | None:
    if mask is None:
        return None
    ys, xs = np.where(mask > 20)
    if xs.size == 0 or ys.size == 0:
        return None
    x1, x2 = float(xs.min()), float(xs.max())
    y1, y2 = float(ys.min()), float(ys.max())
    return {
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2,
        "cx": (x1 + x2) * 0.5,
        "cy": (y1 + y2) * 0.5,
        "w": max(1.0, x2 - x1),
        "h": max(1.0, y2 - y1),
    }


def _make_forehead_mask(shape: tuple[int, int], center_x: float, forehead_y: float, rx: float, ry: float) -> np.ndarray:
    h, w = shape
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(
        mask,
        (int(center_x), int(forehead_y)),
        (max(4, int(rx * 0.68)), max(4, int(ry * 0.16))),
        0,
        0,
        360,
        255,
        -1,
        cv2.LINE_AA,
    )
    return mask


def _estimate_hairline_y(
    hair_mask: np.ndarray | None,
    cx: float,
    face_width: float,
    face_top_y: float,
    face_height: float,
    brow_y: float,
) -> float:
    fallback = face_top_y + face_height * 0.16
    if hair_mask is None:
        return float(min(fallback, brow_y - max(24.0, face_height * 0.12)))

    h, w = hair_mask.shape[:2]
    x1 = int(max(0, cx - face_width * 0.34))
    x2 = int(min(w, cx + face_width * 0.34))
    y1 = int(max(0, face_top_y - face_height * 0.12))
    y2 = int(min(h, brow_y - max(16.0, face_height * 0.07)))
    if x2 <= x1 or y2 <= y1:
        return float(min(fallback, brow_y - max(24.0, face_height * 0.12)))

    central = hair_mask[y1:y2, x1:x2]
    ys, _xs = np.where(central > 20)
    if ys.size == 0:
        return float(min(fallback, brow_y - max(24.0, face_height * 0.12)))

    hairline = float(y1 + np.percentile(ys, 88))
    lower_limit = brow_y - max(24.0, face_height * 0.12)
    upper_limit = max(0.0, face_top_y - face_height * 0.04)
    return float(np.clip(hairline, upper_limit, lower_limit))


def build_head_proxy(
    image_bgr: np.ndarray,
    ctx: dict[str, Any],
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if image_bgr is None:
        raise ValueError("image_bgr is None")

    h, w = image_bgr.shape[:2]
    metadata = metadata or {}
    masks = ctx.get("masks", {}) if isinstance(ctx, dict) else {}
    pose = ctx.get("pose", {}) if isinstance(ctx, dict) else {}
    anchors = ctx.get("anchors", {}) if isinstance(ctx, dict) else {}
    metrics = anchors.get("metrics", {}) if isinstance(anchors, dict) else {}
    pts = _landmarks(ctx)

    hair_mask = masks.get("hair")
    face_mask = masks.get("skin")
    hair_box = _mask_bbox(hair_mask)
    fallback_used = pts is None

    if pts is not None:
        oval = _safe_points(pts, FACE_OVAL)
        brows = _safe_points(pts, LEFT_BROW + RIGHT_BROW)
        eyes = _safe_points(pts, [33, 263, 159, 386, 145, 374])
        if oval.shape[0] >= 8:
            x1, y1 = np.percentile(oval[:, 0], 3), np.percentile(oval[:, 1], 2)
            x2, y2 = np.percentile(oval[:, 0], 97), np.percentile(oval[:, 1], 98)
            face_top_candidates = [float(y1)]
            for idx in (10, 151):
                if idx < pts.shape[0] and pts[idx, 1] > 1:
                    face_top_candidates.append(float(pts[idx, 1]))
            face_top_y = float(min(face_top_candidates))
            chin_y = float(pts[152, 1]) if pts.shape[0] > 152 and pts[152, 1] > face_top_y else float(y2)
            side_width = (
                float(np.linalg.norm(pts[454] - pts[234]))
                if pts.shape[0] > 454 and pts[454, 0] > 1 and pts[234, 0] > 1
                else 0.0
            )
            face_width = float(max(metrics.get("face_width", 0.0), x2 - x1, side_width))
            face_height = float(max(metrics.get("face_height", 0.0), y2 - y1, chin_y - face_top_y))
            cx = float((x1 + x2) * 0.5)
            brow_y = float(np.mean(brows[:, 1])) if brows.shape[0] else float(y1 + face_height * 0.30)
            eye_y = float(np.percentile(eyes[:, 1], 35)) if eyes.shape[0] else float(brow_y + face_height * 0.12)
            hairline_y = _estimate_hairline_y(hair_mask, cx, face_width, face_top_y, face_height, brow_y)
            safe_bottom_limit = min(
                brow_y - max(24.0, face_height * 0.12),
                eye_y - max(38.0, face_height * 0.18),
            )
            min_bottom = float(face_top_y + face_height * 0.02)
            if min_bottom >= safe_bottom_limit:
                min_bottom = float(safe_bottom_limit - 1.0)
            bottom_y = float(np.clip(hairline_y - face_height * 0.035, min_bottom, safe_bottom_limit))
            forehead_y = bottom_y
            skull_top_y = float(max(0.0, face_top_y - face_height * 0.28))
            rx = float(face_width * 0.62)
            ry = float(max(face_height * 0.50, bottom_y - skull_top_y + face_height * 0.12))
            source = "landmarks_and_masks" if hair_box else "landmarks"
            confidence = 0.74 if hair_box else 0.62
        else:
            fallback_used = True
            cx = w * 0.5
            face_width = float(metrics.get("face_width", w * 0.30))
            face_height = float(metrics.get("face_height", h * 0.28))
            face_top_y = h * 0.22
            brow_y = h * 0.36
            eye_y = h * 0.42
            hairline_y = face_top_y + face_height * 0.16
            forehead_y = min(hairline_y, brow_y - 24.0, eye_y - 38.0)
            skull_top_y = h * 0.16
            rx = face_width * 0.65
            ry = face_height * 0.72
            source = "metrics_fallback"
            confidence = 0.42
    else:
        face_box = _mask_bbox(face_mask)
        if face_box:
            cx = face_box["cx"]
            face_width = face_box["w"]
            face_height = face_box["h"]
            face_top_y = face_box["y1"]
            brow_y = face_top_y + face_height * 0.28
            eye_y = face_top_y + face_height * 0.38
            hairline_y = _estimate_hairline_y(hair_mask, cx, face_width, face_top_y, face_height, brow_y)
            forehead_y = min(hairline_y + face_height * 0.02, brow_y - 24.0, eye_y - 38.0)
            skull_top_y = max(0.0, face_box["y1"] - face_height * 0.18)
            rx = face_width * 0.62
            ry = face_height * 0.46
            source = "face_mask_fallback"
            confidence = 0.48
        else:
            cx = w * 0.5
            face_width = w * 0.30
            face_height = h * 0.34
            face_top_y = h * 0.22
            brow_y = h * 0.36
            eye_y = h * 0.42
            hairline_y = face_top_y + face_height * 0.16
            forehead_y = min(hairline_y, brow_y - 24.0, eye_y - 38.0)
            skull_top_y = h * 0.16
            rx = face_width * 0.65
            ry = face_height * 0.66
            source = "image_center_fallback"
            confidence = 0.30

    fit = float(metadata.get("skull_fit", 1.05))
    rx *= fit * float(metadata.get("scale", 1.0))
    offset_y = float(metadata.get("offset_y", 0.0))
    offset_x = float(metadata.get("offset_x", 0.0))
    cx += offset_x
    forehead_y += offset_y
    safe_lower = min(
        brow_y - max(22.0, face_height * 0.11),
        eye_y - max(36.0, face_height * 0.17),
    )
    safe_upper = max(0.0, skull_top_y + face_height * 0.08)
    if safe_upper >= safe_lower:
        safe_upper = max(0.0, safe_lower - 1.0)
    forehead_y = float(np.clip(forehead_y, safe_upper, safe_lower))
    cy = float((forehead_y + skull_top_y) * 0.5 + ry * 0.10)
    forehead_mask = _make_forehead_mask((h, w), cx, forehead_y, rx, ry)

    return {
        "ok": True,
        "source": source,
        "confidence": float(confidence),
        "pose": {
            "yaw": float(pose.get("yaw", 0.0)),
            "pitch": float(pose.get("pitch", 0.0)),
            "roll": float(pose.get("roll", 0.0)),
        },
        "head_ellipse": {
            "cx": float(cx),
            "cy": float(cy),
            "rx": float(rx),
            "ry": float(ry),
        },
        "face_width": float(face_width),
        "face_height": float(face_height),
        "face_top_y": float(face_top_y),
        "brow_y": float(brow_y),
        "eye_y": float(eye_y),
        "forehead_y": float(forehead_y),
        "bottom_y": float(forehead_y),
        "skull_top_y": float(skull_top_y),
        "hairline_y": float(hairline_y),
        "hairline_estimate": hair_box,
        "hair_mask": hair_mask,
        "face_mask": face_mask,
        "forehead_mask": forehead_mask,
        "fallback_used": bool(fallback_used),
    }
