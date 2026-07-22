from __future__ import annotations

from typing import Any

import numpy as np


def _rot(points: np.ndarray, center: np.ndarray, angle_deg: float) -> np.ndarray:
    rad = np.deg2rad(angle_deg)
    c, s = np.cos(rad), np.sin(rad)
    rel = points - center[None, :]
    out = np.empty_like(rel)
    out[:, 0] = rel[:, 0] * c - rel[:, 1] * s
    out[:, 1] = rel[:, 0] * s + rel[:, 1] * c
    return out + center[None, :]


def fit_beanie_to_head(
    head_proxy: dict[str, Any],
    template: dict[str, Any],
) -> dict[str, Any]:
    ellipse = head_proxy["head_ellipse"]
    meta = template["metadata"]
    pose = head_proxy.get("pose", {})

    cx = float(ellipse["cx"])
    cy = float(ellipse["cy"])
    rx = float(ellipse["rx"])
    ry = float(ellipse["ry"])
    roll = float(pose.get("roll", 0.0))
    yaw = float(np.clip(float(pose.get("yaw", 0.0)), -35.0, 35.0))
    pitch = float(np.clip(float(pose.get("pitch", 0.0)), -30.0, 30.0))

    yaw_scale_left = 1.0 + max(0.0, yaw) / 120.0
    yaw_scale_right = 1.0 + max(0.0, -yaw) / 120.0
    pitch_shift = float(np.clip(pitch * 0.25, -8.0, 8.0))
    top_sag = float(np.clip(float(meta.get("top_sag", 0.08)), 0.0, 0.14))
    fold_ratio = float(np.clip(float(meta.get("fold_height", 0.18)), 0.10, 0.22))

    face_height = float(head_proxy.get("face_height", ry * 1.8))
    brow_y = float(head_proxy.get("brow_y", head_proxy.get("bottom_y", cy) + face_height * 0.20))
    eye_y = float(head_proxy.get("eye_y", brow_y + face_height * 0.12))
    hairline_y = float(head_proxy.get("hairline_y", head_proxy.get("bottom_y", cy)))
    face_top_y = float(head_proxy.get("face_top_y", hairline_y - face_height * 0.16))
    safe_limit = min(
        brow_y - max(20.0, face_height * 0.10),
        eye_y - max(34.0, face_height * 0.16),
    )
    base_y = float(min(float(head_proxy.get("bottom_y", hairline_y)) + pitch_shift, safe_limit))
    base_y = float(max(base_y, face_top_y + face_height * 0.02))

    beanie_height = float(np.clip(ry * 0.78, face_height * 0.32, face_height * 0.54))
    top_y = float(max(0.0, base_y - beanie_height))
    fold_height = float(min(beanie_height * fold_ratio, beanie_height * 0.24))
    fold_height = float(max(8.0, fold_height))
    center = np.array([cx, cy], dtype=np.float32)

    xs = np.linspace(-1.0, 1.0, 42)
    top_arc = []
    bottom_arc = []
    for t in xs:
        side_scale = yaw_scale_left if t < 0 else yaw_scale_right
        x = cx + t * rx * side_scale
        arch = np.sqrt(max(0.0, 1.0 - t * t))
        y = base_y - beanie_height * (0.12 + 0.88 * arch)
        y += top_sag * beanie_height * np.exp(-(t / 0.42) ** 2)
        top_arc.append([x, y])

        lower_curve = 3.0 * (1.0 - t * t)
        edge_lift = 8.0 * abs(t)
        bottom_arc.append([x, base_y + lower_curve - edge_lift])

    outer = np.asarray(top_arc + bottom_arc[::-1], dtype=np.float32)
    outer = _rot(outer, center, roll)

    fold_top = []
    fold_bottom = []
    for t in xs:
        side_scale = yaw_scale_left if t < 0 else yaw_scale_right
        x = cx + t * rx * 0.98 * side_scale
        lower_curve = 2.5 * (1.0 - t * t)
        edge_lift = 7.0 * abs(t)
        y_bottom = base_y + lower_curve - edge_lift
        y_top = y_bottom - fold_height * (0.86 + 0.08 * (1.0 - t * t))
        fold_top.append([x, y_top])
        fold_bottom.append([x, y_bottom])
    fold_polygon = np.asarray(fold_top + fold_bottom[::-1], dtype=np.float32)
    fold_polygon = _rot(fold_polygon, center, roll)

    contact = np.asarray(
        [
            [cx - rx * yaw_scale_left * 0.84, base_y - 2.0],
            [cx, base_y + 3.0],
            [cx + rx * yaw_scale_right * 0.84, base_y - 2.0],
        ],
        dtype=np.float32,
    )
    contact = _rot(contact, center, roll)

    return {
        "placement": {
            "center": [cx, cy],
            "rx": rx,
            "ry": ry,
            "roll": roll,
            "yaw": yaw,
            "pitch": pitch,
            "base_y": base_y,
            "bottom_y": base_y,
            "top_y": top_y,
            "fold_height": fold_height,
            "face_top_y": face_top_y,
            "brow_y": brow_y,
            "eye_y": eye_y,
            "hairline_y": hairline_y,
            "bottom_above_brow": bool(base_y < brow_y - 15.0),
            "shape": "beanie_dome",
        },
        "outer_polygon": outer,
        "fold_polygon": fold_polygon,
        "contact_line": contact,
        "occlusion_masks": {
            "hair": head_proxy.get("hair_mask"),
            "forehead": head_proxy.get("forehead_mask"),
        },
    }
