from __future__ import annotations

from typing import Any

import cv2
import numpy as np


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
        "w": max(1.0, x2 - x1),
        "h": max(1.0, y2 - y1),
        "cx": (x1 + x2) * 0.5,
        "cy": (y1 + y2) * 0.5,
    }


def _point(ctx: dict[str, Any], idx: int, fallback: tuple[float, float]) -> np.ndarray:
    pts = ctx.get("landmarks_2d")
    if isinstance(pts, np.ndarray) and pts.ndim == 2 and pts.shape[0] > idx:
        return pts[idx].astype(np.float32)
    return np.array(fallback, dtype=np.float32)


def _content_bbox(image_bgr: np.ndarray) -> dict[str, float]:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    row_mask = np.mean(gray, axis=1) > 12
    col_mask = np.mean(gray, axis=0) > 12
    ys = np.where(row_mask)[0]
    xs = np.where(col_mask)[0]
    if xs.size == 0 or ys.size == 0:
        h, w = image_bgr.shape[:2]
        return {"x1": 0.0, "y1": 0.0, "x2": float(w - 1), "y2": float(h - 1)}
    return {
        "x1": float(xs.min()),
        "y1": float(ys.min()),
        "x2": float(xs.max()),
        "y2": float(ys.max()),
    }


def build_neck_chest_proxy(
    image_bgr: np.ndarray,
    ctx: dict[str, Any],
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Build a lightweight 2.5D neck/chest proxy for necklace draping.

    This deliberately uses available face/neck information only. It does not
    pretend to be a full body model.
    """
    if image_bgr is None:
        raise ValueError("image_bgr is None")

    metadata = metadata or {}
    h, w = image_bgr.shape[:2]
    masks = ctx.get("masks", {}) if isinstance(ctx, dict) else {}
    anchors = ctx.get("anchors", {}) if isinstance(ctx, dict) else {}
    metrics = anchors.get("metrics", {}) if isinstance(anchors, dict) else {}
    content_box = _content_bbox(image_bgr)

    face_width = float(metrics.get("face_width") or max(80.0, w * 0.28))
    chin = np.array(
        anchors.get("necklace", {}).get("chin")
        or _point(ctx, 152, (w * 0.5, h * 0.52)),
        dtype=np.float32,
    )

    neck_mask = masks.get("neck")
    neck_box = _mask_bbox(neck_mask)
    fallback_used = neck_box is None

    if neck_box:
        neck_center = np.array([neck_box["cx"], neck_box["y1"] + neck_box["h"] * 0.28], dtype=np.float32)
        neck_width = float(max(min(neck_box["w"] * 0.72, face_width * 0.72), face_width * 0.44))
        neck_height = float(max(min(neck_box["h"], face_width * 0.62), face_width * 0.34))
        source = "face_parsing_neck_mask"
        confidence = 0.78
    else:
        neck_center = np.array([float(chin[0]), float(chin[1] + face_width * 0.20)], dtype=np.float32)
        neck_width = float(face_width * 0.50)
        neck_height = float(face_width * 0.42)
        source = "chin_face_width_heuristic"
        confidence = 0.45

    chain_length = float(metadata.get("chain_length", 1.0))
    visible_bottom = float(min(h - 2, content_box["y2"] - 10.0))
    max_drape_y = float(min(visible_bottom, neck_center[1] + face_width * (0.28 + 0.09 * chain_length)))
    shoulder_y = float(min(max_drape_y, neck_center[1] + face_width * (0.24 + 0.04 * chain_length)))
    shoulder_half = float(face_width * (0.54 + 0.08 * chain_length))

    sternum = np.array(
        [
            float(neck_center[0]),
            float(min(max_drape_y, neck_center[1] + face_width * (0.26 + 0.08 * chain_length))),
        ],
        dtype=np.float32,
    )

    neck_left = np.array([neck_center[0] - neck_width * 0.50, neck_center[1]], dtype=np.float32)
    neck_right = np.array([neck_center[0] + neck_width * 0.50, neck_center[1]], dtype=np.float32)
    neck_base = np.array([neck_center[0], neck_center[1] + neck_height * 0.35], dtype=np.float32)
    shoulder_left = np.array([max(0.0, neck_center[0] - shoulder_half), shoulder_y], dtype=np.float32)
    shoulder_right = np.array([min(float(w - 1), neck_center[0] + shoulder_half), shoulder_y], dtype=np.float32)

    chest_rx = float(max(face_width * 0.62, neck_width * 1.35))
    chest_ry = float(max(face_width * 0.46, neck_height * 1.10))

    body_mask = masks.get("skin")
    if body_mask is None:
        body_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.ellipse(
            body_mask,
            (int(sternum[0]), int(sternum[1])),
            (int(chest_rx), int(chest_ry)),
            0,
            180,
            360,
            255,
            -1,
            cv2.LINE_AA,
        )

    return {
        "neck_left": neck_left.tolist(),
        "neck_right": neck_right.tolist(),
        "neck_base": neck_base.tolist(),
        "sternum_center": sternum.tolist(),
        "shoulder_line": [shoulder_left.tolist(), shoulder_right.tolist()],
        "upper_chest_proxy": {
            "type": "ellipsoid_2d",
            "center": sternum.tolist(),
            "radius_x": chest_rx,
            "radius_y": chest_ry,
            "neck_radius_x": neck_width * 0.48,
            "neck_radius_y": neck_height * 0.46,
        },
        "image_size": [int(w), int(h)],
        "body_mask": body_mask,
        "hair_occlusion_mask": masks.get("hair"),
        "debug": {
            "body_proxy_source": source,
            "confidence": confidence,
            "fallback_used": bool(fallback_used),
            "face_width": face_width,
            "neck_width": neck_width,
            "neck_height": neck_height,
            "image_size": [int(w), int(h)],
            "content_box": content_box,
            "max_drape_y": max_drape_y,
        },
    }
