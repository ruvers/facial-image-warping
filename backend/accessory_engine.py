from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from backend.assets_manager import (
    asset_path_is_manifest_allowed,
    validate_asset_path,
)
from backend.accessories.base import load_asset_meta
from backend.accessories.earring_motion import (
    compute_ear_anchor,
    compute_visibility,
    update_earring_motion,
)


@dataclass
class AccessoryPlacement:
    center: tuple[float, float]
    width: float
    roll_deg: float = 0.0
    offset_x: float = 0.0
    offset_y: float = 0.0
    alpha: float = 1.0
    yaw_deg: float | None = None
    pitch_deg: float | None = None


def _remove_corner_matte(asset: np.ndarray) -> np.ndarray:
    if asset.ndim != 3:
        return asset

    if asset.shape[2] == 4:
        alpha = asset[:, :, 3]
        if float(np.mean(alpha > 250)) < 0.98:
            return asset
        bgr = asset[:, :, :3].copy()
    else:
        bgr = asset[:, :, :3].copy()

    h, w = bgr.shape[:2]
    if h < 4 or w < 4:
        return asset

    mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
    corners = [
        (0, 0),
        (w - 1, 0),
        (0, h - 1),
        (w - 1, h - 1),
    ]
    for corner in corners:
        try:
            cv2.floodFill(
                bgr,
                mask,
                corner,
                (255, 0, 255),
                (10, 10, 10),
                (10, 10, 10),
                cv2.FLOODFILL_MASK_ONLY,
            )
        except Exception:
            continue

    matte = mask[1:-1, 1:-1] > 0
    if float(np.mean(matte)) < 0.04:
        return asset

    alpha_new = np.where(matte, 0, 255).astype(np.uint8)
    alpha_new = cv2.GaussianBlur(alpha_new, (3, 3), 0)
    base_bgr = asset[:, :, :3] if asset.shape[2] >= 3 else bgr
    return np.dstack([base_bgr, alpha_new])


def _remove_checkerboard_matte(asset: np.ndarray) -> np.ndarray:
    if asset.ndim != 3 or asset.shape[2] != 4:
        return asset

    alpha = asset[:, :, 3]
    if float(np.mean(alpha > 250)) < 0.98:
        return asset

    bgr = asset[:, :, :3].astype(np.float32)
    max_ch = np.max(bgr, axis=2)
    min_ch = np.min(bgr, axis=2)
    mean_ch = np.mean(bgr, axis=2)
    low_saturation = (max_ch - min_ch) <= 12.0
    bright_checker = low_saturation & (mean_ch >= 205.0)

    if float(np.mean(bright_checker)) < 0.12:
        return asset

    keep = (~bright_checker).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    keep = cv2.morphologyEx(keep, cv2.MORPH_OPEN, kernel)
    keep = cv2.GaussianBlur(keep, (3, 3), 0)

    cleaned = asset.copy()
    cleaned[:, :, 3] = keep
    return cleaned


def _trim_alpha_bbox(asset: np.ndarray, margin: int = 4) -> np.ndarray:
    if asset.ndim != 3 or asset.shape[2] != 4:
        return asset

    alpha = asset[:, :, 3]
    ys, xs = np.where(alpha > 8)
    if xs.size == 0 or ys.size == 0:
        return asset

    h, w = alpha.shape
    x1 = max(0, int(xs.min()) - margin)
    y1 = max(0, int(ys.min()) - margin)
    x2 = min(w, int(xs.max()) + margin + 1)
    y2 = min(h, int(ys.max()) + margin + 1)

    if x1 <= 0 and y1 <= 0 and x2 >= w and y2 >= h:
        return asset

    return asset[y1:y2, x1:x2].copy()


def _prepare_overlay_asset(asset: np.ndarray, accessory_type: str) -> np.ndarray:
    prepared = _remove_corner_matte(asset)
    prepared = _remove_checkerboard_matte(prepared)
    prepared = _trim_alpha_bbox(prepared, margin=6)
    return prepared


def _read_rgba(asset_path: str, accessory_type: str = "") -> np.ndarray | None:
    if not asset_path:
        return None

    if not asset_path_is_manifest_allowed(asset_path):
        return None

    try:
        asset_path = validate_asset_path(asset_path)
    except Exception:
        return None

    if not os.path.exists(asset_path):
        return None

    try:
        data = np.fromfile(
            asset_path,
            dtype=np.uint8,
        )
        asset = cv2.imdecode(
            data,
            cv2.IMREAD_UNCHANGED,
        )
    except Exception:
        asset = None

    if asset is None:
        return None

    if asset.ndim == 2:
        asset = cv2.cvtColor(asset, cv2.COLOR_GRAY2BGRA)

    if asset.shape[2] == 3:
        alpha = np.full(
            asset.shape[:2],
            255,
            dtype=np.uint8,
        )
        asset = np.dstack([asset, alpha])

    return _prepare_overlay_asset(asset, accessory_type)


def _rotate_and_resize_rgba(
    asset_rgba: np.ndarray,
    target_width: float,
    roll_deg: float = 0.0,
) -> np.ndarray:
    h, w = asset_rgba.shape[:2]

    if w <= 0 or h <= 0:
        return asset_rgba

    scale = float(target_width) / float(w)

    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    resized = cv2.resize(
        asset_rgba,
        (new_w, new_h),
        interpolation=cv2.INTER_AREA,
    )

    if abs(roll_deg) < 0.01:
        return resized

    center = (
        new_w / 2.0,
        new_h / 2.0,
    )

    matrix = cv2.getRotationMatrix2D(
        center,
        roll_deg,
        1.0,
    )

    cos = abs(matrix[0, 0])
    sin = abs(matrix[0, 1])

    bound_w = int((new_h * sin) + (new_w * cos))
    bound_h = int((new_h * cos) + (new_w * sin))

    matrix[0, 2] += (bound_w / 2.0) - center[0]
    matrix[1, 2] += (bound_h / 2.0) - center[1]

    rotated = cv2.warpAffine(
        resized,
        matrix,
        (bound_w, bound_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )

    return rotated


def _rotate_rgba_about_anchor(
    asset_rgba: np.ndarray,
    angle_deg: float,
    anchor_x: float,
    anchor_y: float,
) -> tuple[np.ndarray, tuple[float, float]]:
    if asset_rgba.ndim != 3 or asset_rgba.shape[2] != 4 or abs(angle_deg) < 0.01:
        return asset_rgba, (anchor_x, anchor_y)

    h, w = asset_rgba.shape[:2]
    if h <= 1 or w <= 1:
        return asset_rgba, (anchor_x, anchor_y)

    pivot = (
        float(np.clip(anchor_x, 0.0, 1.0)) * float(w),
        float(np.clip(anchor_y, 0.0, 1.0)) * float(h),
    )
    matrix = cv2.getRotationMatrix2D(
        pivot,
        float(angle_deg),
        1.0,
    )

    corners = np.array(
        [
            [0.0, 0.0, 1.0],
            [float(w), 0.0, 1.0],
            [float(w), float(h), 1.0],
            [0.0, float(h), 1.0],
            [pivot[0], pivot[1], 1.0],
        ],
        dtype=np.float32,
    )
    transformed = corners @ matrix.T
    min_x = float(np.min(transformed[:4, 0]))
    min_y = float(np.min(transformed[:4, 1]))
    max_x = float(np.max(transformed[:4, 0]))
    max_y = float(np.max(transformed[:4, 1]))

    out_w = max(1, int(np.ceil(max_x - min_x)))
    out_h = max(1, int(np.ceil(max_y - min_y)))
    matrix[0, 2] -= min_x
    matrix[1, 2] -= min_y

    rotated = cv2.warpAffine(
        asset_rgba,
        matrix,
        (out_w, out_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )
    new_pivot = np.array([pivot[0], pivot[1], 1.0], dtype=np.float32) @ matrix.T
    return rotated, (
        float(np.clip(new_pivot[0] / max(1.0, float(out_w)), 0.0, 1.0)),
        float(np.clip(new_pivot[1] / max(1.0, float(out_h)), 0.0, 1.0)),
    )


def _alpha_blend_rgba(
    image_bgr: np.ndarray,
    asset_rgba: np.ndarray,
    center: tuple[float, float],
    alpha_multiplier: float = 1.0,
    occlusion_mask: np.ndarray | None = None,
    occlusion_strength: float = 0.0,
) -> np.ndarray:
    out = image_bgr.copy()

    ih, iw = out.shape[:2]
    ah, aw = asset_rgba.shape[:2]

    cx, cy = center

    x1 = int(round(cx - aw / 2.0))
    y1 = int(round(cy - ah / 2.0))
    x2 = x1 + aw
    y2 = y1 + ah

    src_x1 = max(0, -x1)
    src_y1 = max(0, -y1)
    src_x2 = aw - max(0, x2 - iw)
    src_y2 = ah - max(0, y2 - ih)

    dst_x1 = max(0, x1)
    dst_y1 = max(0, y1)
    dst_x2 = min(iw, x2)
    dst_y2 = min(ih, y2)

    if dst_x1 >= dst_x2 or dst_y1 >= dst_y2:
        return out

    asset_crop = asset_rgba[
        src_y1:src_y2,
        src_x1:src_x2,
    ]

    rgb = asset_crop[:, :, :3].astype(np.float32)
    alpha = asset_crop[:, :, 3].astype(np.float32) / 255.0
    alpha = np.clip(
        alpha * float(alpha_multiplier),
        0.0,
        1.0,
    )

    if occlusion_mask is not None and occlusion_strength > 0.0:
        try:
            mask_crop = occlusion_mask[
                dst_y1:dst_y2,
                dst_x1:dst_x2,
            ].astype(np.float32)
            if mask_crop.shape[:2] == alpha.shape[:2]:
                occ = cv2.GaussianBlur(
                    (mask_crop > 20).astype(np.float32),
                    (0, 0),
                    sigmaX=1.6,
                    sigmaY=1.6,
                )
                alpha *= np.clip(1.0 - occ * float(occlusion_strength), 0.0, 1.0)
        except Exception:
            pass

    alpha_3 = alpha[:, :, None]

    roi = out[
        dst_y1:dst_y2,
        dst_x1:dst_x2,
    ].astype(np.float32)

    blended = (
        rgb * alpha_3
        + roi * (1.0 - alpha_3)
    )

    out[
        dst_y1:dst_y2,
        dst_x1:dst_x2,
    ] = np.clip(
        blended,
        0,
        255,
    ).astype(np.uint8)

    return out


def _default_anchor_ratio(accessory_type: str) -> tuple[float, float]:
    if accessory_type == "hat":
        return 0.5, 0.82
    if accessory_type == "necklace":
        return 0.5, 0.04
    return 0.5, 0.5


def _warp_perspective_rgba(
    asset_rgba: np.ndarray,
    yaw_deg: float,
    pitch_deg: float,
    accessory_type: str,
) -> np.ndarray:
    if asset_rgba.ndim != 3 or asset_rgba.shape[2] != 4:
        return asset_rgba

    if accessory_type not in {"glasses", "hat", "necklace"}:
        return asset_rgba

    h, w = asset_rgba.shape[:2]
    if h <= 1 or w <= 1:
        return asset_rgba

    yaw = float(np.clip(yaw_deg, -35.0, 35.0)) / 35.0
    pitch = float(np.clip(pitch_deg, -25.0, 25.0)) / 25.0
    if abs(yaw) < 0.03 and abs(pitch) < 0.03:
        return asset_rgba

    yaw_shift = yaw * w * 0.10
    pitch_shift = pitch * h * 0.06
    top_squeeze = abs(pitch) * w * 0.035

    src = np.array(
        [
            [0.0, 0.0],
            [float(w - 1), 0.0],
            [float(w - 1), float(h - 1)],
            [0.0, float(h - 1)],
        ],
        dtype=np.float32,
    )
    dst = np.array(
        [
            [top_squeeze + yaw_shift, pitch_shift],
            [float(w - 1) - top_squeeze + yaw_shift, -pitch_shift],
            [float(w - 1) - yaw_shift, float(h - 1) - pitch_shift],
            [-yaw_shift, float(h - 1) + pitch_shift],
        ],
        dtype=np.float32,
    )

    min_x = float(np.min(dst[:, 0]))
    min_y = float(np.min(dst[:, 1]))
    max_x = float(np.max(dst[:, 0]))
    max_y = float(np.max(dst[:, 1]))
    pad_x = int(max(0.0, -min_x) + 4)
    pad_y = int(max(0.0, -min_y) + 4)
    out_w = int(np.ceil(max_x + pad_x + 4))
    out_h = int(np.ceil(max_y + pad_y + 4))
    dst[:, 0] += pad_x
    dst[:, 1] += pad_y

    matrix = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(
        asset_rgba,
        matrix,
        (max(out_w, 1), max(out_h, 1)),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )


def _draw_placeholder(
    image_bgr: np.ndarray,
    placement: AccessoryPlacement,
    accessory_type: str,
) -> np.ndarray:
    out = image_bgr.copy()

    x = int(round(placement.center[0] + placement.offset_x))
    y = int(round(placement.center[1] + placement.offset_y))

    color = (255, 0, 255)

    if "earring" in accessory_type:
        color = (255, 0, 0)
    elif "necklace" in accessory_type:
        color = (0, 255, 0)
    elif "glasses" in accessory_type:
        color = (0, 0, 255)
    elif "hair_clip" in accessory_type:
        color = (0, 215, 255)

    cv2.circle(
        out,
        (x, y),
        8,
        color,
        -1,
        cv2.LINE_AA,
    )

    cv2.putText(
        out,
        accessory_type,
        (x + 10, y - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        2,
        cv2.LINE_AA,
    )

    return out


def _face_metrics(ctx: dict) -> dict:
    anchors = ctx.get("anchors", {})
    metrics = anchors.get("metrics", {})

    return {
        "face_width": float(metrics.get("face_width", 180.0)),
        "face_height": float(metrics.get("face_height", 220.0)),
    }


def _ctx_landmarks_2d(ctx: dict) -> np.ndarray | None:
    landmarks = ctx.get("landmarks_2d")
    if landmarks is None:
        landmarks = ctx.get("landmarks")

    if isinstance(landmarks, np.ndarray):
        if landmarks.ndim == 2 and landmarks.shape[0] >= 10 and landmarks.shape[1] >= 2:
            return landmarks[:, :2].astype(np.float32, copy=False)
        return None

    if not isinstance(landmarks, list) or len(landmarks) < 10:
        return None

    pts = np.zeros((len(landmarks), 2), dtype=np.float32)
    for i, item in enumerate(landmarks):
        if isinstance(item, dict):
            x = item.get("x", 0.0)
            y = item.get("y", 0.0)
            idx = item.get("index", i)
            if 0 <= int(idx) < len(landmarks):
                pts[int(idx)] = (float(x), float(y))
        else:
            pts[i] = (float(item[0]), float(item[1]))

    return pts


def _landmark_bbox_and_center(ctx: dict) -> tuple[dict[str, float], tuple[float, float]] | None:
    pts = _ctx_landmarks_2d(ctx)
    if pts is None or pts.shape[0] < 455:
        return None

    face_indices = [
        10,
        21,
        54,
        67,
        93,
        103,
        109,
        127,
        132,
        136,
        148,
        149,
        150,
        152,
        162,
        172,
        176,
        234,
        251,
        284,
        297,
        323,
        332,
        338,
        356,
        361,
        365,
        377,
        378,
        379,
        389,
        397,
        400,
        454,
    ]
    valid = pts[[idx for idx in face_indices if idx < pts.shape[0]]]
    if valid.size == 0:
        return None

    x1 = float(np.min(valid[:, 0]))
    y1 = float(np.min(valid[:, 1]))
    x2 = float(np.max(valid[:, 0]))
    y2 = float(np.max(valid[:, 1]))

    bbox = {
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2,
        "cx": float((x1 + x2) * 0.5),
        "cy": float((y1 + y2) * 0.5),
    }

    center = (
        bbox["cx"],
        bbox["cy"],
    )

    return bbox, center


def get_accessory_placement(
    ctx: dict,
    accessory_type: str,
    params: dict | None = None,
) -> AccessoryPlacement:
    params = params or {}

    anchors = ctx.get("anchors", {})
    metrics = _face_metrics(ctx)

    face_width = metrics["face_width"]
    face_height = metrics["face_height"]

    if accessory_type == "glasses":
        a = anchors.get("glasses", {})

        center = a.get("center", (0.0, 0.0))
        width = float(a.get("width", face_width * 1.35))
        roll = float(a.get("roll_deg", 0.0))

        return AccessoryPlacement(
            center=center,
            width=width * float(params.get("scale", 1.0)),
            roll_deg=roll + float(params.get("rotation", 0.0)),
            offset_x=float(params.get("offset_x", 0.0)),
            offset_y=(
                float(params.get("offset_y", 0.0))
                + face_height * float(params.get("offset_y_ratio", 0.0))
            ),
            alpha=float(params.get("alpha", params.get("opacity", 1.0))),
            # Glasses are a rigid 2D overlay — only roll rotation is needed.
            # Setting yaw/pitch to 0 prevents the perspective warp from
            # distorting the glasses angle.
            yaw_deg=0.0,
            pitch_deg=0.0,
        )

    if accessory_type == "left_earring":
        a = anchors.get("earrings", {})
        center = a.get("left", (0.0, 0.0))

        return AccessoryPlacement(
            center=center,
            width=face_width * float(params.get("scale", 0.12)),
            roll_deg=float(params.get("rotation", 0.0)),
            offset_x=(
                float(params.get("offset_x", 0.0))
                - face_width * float(params.get("offset_x_ratio", 0.015))
            ),
            offset_y=(
                float(params.get("offset_y", 0.0))
                + face_height * float(params.get("offset_y_ratio", 0.015))
            ),
            alpha=float(params.get("alpha", params.get("opacity", 1.0))),
        )

    if accessory_type == "right_earring":
        a = anchors.get("earrings", {})
        center = a.get("right", (0.0, 0.0))

        return AccessoryPlacement(
            center=center,
            width=face_width * float(params.get("scale", 0.12)),
            roll_deg=float(params.get("rotation", 0.0)),
            offset_x=(
                float(params.get("offset_x", 0.0))
                + face_width * float(params.get("offset_x_ratio", 0.015))
            ),
            offset_y=(
                float(params.get("offset_y", 0.0))
                + face_height * float(params.get("offset_y_ratio", 0.015))
            ),
            alpha=float(params.get("alpha", params.get("opacity", 1.0))),
        )

    if accessory_type == "necklace":
        a = anchors.get("necklace", {})
        center = a.get("center", (0.0, 0.0))
        base_width = float(a.get("width", face_width * 0.85))
        anchor_face_width = float(a.get("face_width", face_width))

        pose = ctx.get("pose", {})
        euler = pose.get("euler", {}) if isinstance(pose, dict) else {}
        face_yaw = float(euler.get("yaw", 0.0))
        face_roll = float(euler.get("roll", 0.0))
        face_pitch = float(euler.get("pitch", 0.0))

        return AccessoryPlacement(
            center=(
                float(center[0]),
                float(center[1]),
            ),
            width=base_width * float(params.get("scale", 1.0)),
            roll_deg=(face_roll * 0.15) + float(params.get("rotation", 0.0)),
            offset_x=float(params.get("offset_x", 0.0)),
            offset_y=(
                float(params.get("offset_y", 0.0))
                + anchor_face_width * float(params.get("offset_y_ratio", 0.0))
            ),
            alpha=float(params.get("alpha", params.get("opacity", 1.0))),
            yaw_deg=face_yaw * 0.25,
            pitch_deg=face_pitch * 0.20,
        )

    if accessory_type == "hair_clip":
        a = anchors.get("hair_clip", {})
        center = a.get("center")
        base_width = float(a.get("width", face_width * 0.5))
        roll = float(a.get("roll_deg", -12.0))

        if center:
            return AccessoryPlacement(
                center=(
                    float(center[0]),
                    float(center[1]),
                ),
                width=base_width * float(params.get("scale", 1.0)),
                roll_deg=roll + float(params.get("rotation", 0.0)),
                offset_x=float(params.get("offset_x", 0.0)),
                offset_y=float(params.get("offset_y", 0.0)),
                alpha=float(params.get("alpha", params.get("opacity", 1.0))),
            )

        face_center = anchors.get("face_center")
        if not face_center:
            bbox = anchors.get("bbox", {})
            face_center = (
                float(bbox.get("cx", 0.0)),
                float(bbox.get("cy", 0.0)),
            )
        cx, cy = face_center
        center = (
            float(cx) + face_width * float(params.get("anchor_x_ratio", 0.32)),
            float(cy) - face_height * float(params.get("anchor_y_ratio", 0.34)),
        )

        return AccessoryPlacement(
            center=center,
            width=face_width * float(params.get("scale", 0.34)),
            roll_deg=float(params.get("rotation", -12.0)),
            offset_x=float(params.get("offset_x", 0.0)),
            offset_y=float(params.get("offset_y", 0.0)),
            alpha=float(params.get("alpha", params.get("opacity", 1.0))),
        )

    if accessory_type == "hat":
        a = anchors.get("hat", {})
        center = a.get("center")
        if center:
            return AccessoryPlacement(
                center=(
                    float(center[0]),
                    float(center[1]),
                ),
                width=float(a.get("width", face_width * 1.35)) * float(params.get("scale", 1.0)),
                roll_deg=float(a.get("roll_deg", 0.0)) + float(params.get("rotation", 0.0)),
                offset_x=float(params.get("offset_x", 0.0)),
                offset_y=(
                    float(params.get("offset_y", 0.0))
                    + face_height * float(params.get("offset_y_ratio", 0.0))
                ),
                alpha=float(params.get("alpha", params.get("opacity", 1.0))),
            )

        face_center = anchors.get("face_center")
        bbox = anchors.get("bbox", {})
        landmark_anchor = _landmark_bbox_and_center(ctx)
        if landmark_anchor is not None:
            landmark_bbox, landmark_center = landmark_anchor
            if not bbox:
                bbox = landmark_bbox
            if not face_center:
                face_center = landmark_center

        if not face_center:
            face_center = (
                float(bbox.get("cx", 0.0)),
                float(bbox.get("cy", 0.0)),
            )

        cx, cy = face_center
        top_y = float(bbox.get("y1", cy - face_height * 0.58))
        center = (
            float(cx),
            top_y + face_height * 0.16,
        )

        return AccessoryPlacement(
            center=center,
            width=face_width * float(params.get("scale", 0.95)),
            roll_deg=float(params.get("rotation", 0.0)),
            offset_x=float(params.get("offset_x", 0.0)),
            offset_y=float(params.get("offset_y", -18.0)),
            alpha=float(params.get("alpha", params.get("opacity", 1.0))),
        )

    raise ValueError(f"Unknown accessory_type: {accessory_type}")


def place_accessory(
    image_bgr: np.ndarray,
    ctx: dict,
    accessory_type: str,
    params: dict | None = None,
) -> np.ndarray:
    """
    Generic accessory placement.

    Supported accessory_type:
    - glasses
    - left_earring
    - right_earring
    - necklace
    - hair_clip
    - hat

    params:
    {
        "asset_path": "assets/glasses/model.png",
        "scale": 1.0,
        "offset_x": 0,
        "offset_y": 0,
        "offset_y_ratio": 0.10,
        "rotation": 0,
        "alpha": 1.0,
        "debug_placeholder": true
    }
    """

    if image_bgr is None:
        raise ValueError("image_bgr is None")

    params = params or {}

    placement = get_accessory_placement(
        ctx,
        accessory_type,
        params,
    )

    asset_path = params.get("asset_path")
    asset = _read_rgba(asset_path, accessory_type=accessory_type) if asset_path else None

    if asset is None:
        if bool(params.get("debug_placeholder", False)):
            return _draw_placeholder(
                image_bgr,
                placement,
                accessory_type,
            )

        return image_bgr

    if accessory_type == "right_earring":
        asset = cv2.flip(asset, 1)

    earring_debug: dict[str, Any] | None = None
    earring_visibility = 1.0
    earring_occlusion_mask = None
    earring_occlusion_strength = 0.0
    motion_angle = 0.0
    motion_offset = (0.0, 0.0)
    motion_anchor_ratio: tuple[float, float] | None = None

    if accessory_type in {"left_earring", "right_earring"}:
        side = "left" if accessory_type == "left_earring" else "right"
        refined_anchor, anchor_confidence = compute_ear_anchor(
            ctx,
            side,
            placement.center,
        )
        placement.center = refined_anchor

        earring_visibility, visibility_debug = compute_visibility(
            ctx,
            side,
            refined_anchor,
            anchor_confidence,
        )

        motion = update_earring_motion(
            ctx.get("accessory_motion_state"),
            side,
            refined_anchor,
            ctx,
            params,
        )
        motion_angle = float(motion.get("angle_deg", 0.0))
        motion_offset = motion.get("offset", (0.0, 0.0))

        metadata = params.get("metadata") if isinstance(params.get("metadata"), dict) else {}
        if str(metadata.get("motion_preset", "")).lower() in {"off", "static"}:
            motion_angle = 0.0
            motion_offset = (0.0, 0.0)

        placement.offset_x += float(motion_offset[0])
        placement.offset_y += float(motion_offset[1])
        placement.alpha *= earring_visibility

        pose = ctx.get("pose", {})
        euler = pose.get("euler", {}) if isinstance(pose, dict) else {}
        yaw = float(euler.get("yaw", 0.0))
        side_sign = -1.0 if side == "left" else 1.0
        far_side = float(np.clip((side_sign * yaw - 4.0) / 45.0, 0.0, 0.35))

        ah, aw = asset.shape[:2]
        face_width = _face_metrics(ctx)["face_width"]
        scale = float(params.get("scale", 0.12))
        if scale < 0.6:
            target_height = face_width * scale
        else:
            target_height = face_width * 0.25 * scale
        target_height *= 1.0 - far_side
        scale_factor = target_height / max(1.0, float(ah))
        placement.width = float(aw * scale_factor)
        placement.offset_y += float(target_height) * 0.2
        placement.offset_x += side_sign * face_width * far_side * 0.015

        earring_occlusion_mask = ctx.get("masks", {}).get("hair")
        earring_occlusion_strength = 0.88
        motion_anchor_ratio = (
            float(metadata.get("anchor_x", params.get("anchor_x", 0.50))),
            float(metadata.get("anchor_y", params.get("anchor_y", 0.08))),
        )
        earring_debug = {
            "side": side,
            "anchor": [float(refined_anchor[0]), float(refined_anchor[1])],
            "anchor_confidence": float(anchor_confidence),
            "visibility": float(earring_visibility),
            "motion": {
                "angle_deg": motion_angle,
                "offset": [float(motion_offset[0]), float(motion_offset[1])],
                "earring_type": motion.get("earring_type"),
                "fallback_static": bool(motion.get("fallback_static", False)),
            },
            "visibility_debug": visibility_debug,
        }

    if accessory_type == "glasses" and asset_path:
        try:
            meta = load_asset_meta(
                str(validate_asset_path(asset_path)),
                asset.shape,
            )
            src_left = np.array(meta["anchors"]["left_eye"], dtype=np.float32)
            src_right = np.array(meta["anchors"]["right_eye"], dtype=np.float32)
            asset_eye_center = (src_left + src_right) / 2.0
            asset_eye_dist = float(np.linalg.norm(src_right - src_left))
            face_eye_dist = float(
                ctx.get("anchors", {})
                .get("glasses", {})
                .get("eye_distance", 0.0)
            )

            # Use temple distance from anchors (already computed in build_glasses_anchor)
            glasses_anchor = ctx.get("anchors", {}).get("glasses", {})
            temple_distance = float(glasses_anchor.get("temple_distance", 0.0))
            if temple_distance < 1.0:
                # Fallback: compute from landmarks
                landmarks = ctx.get("landmarks_2d")
                if landmarks is None:
                    landmarks = ctx.get("landmarks")
                if landmarks is not None and len(landmarks) > 356:
                    left_temple = np.array(landmarks[127], dtype=np.float32)
                    right_temple = np.array(landmarks[356], dtype=np.float32)
                    temple_distance = float(np.linalg.norm(right_temple - left_temple))
                else:
                    temple_distance = face_eye_dist * 2.0

            if face_eye_dist > 1.0 and asset_eye_dist > 1.0:
                # Scale so that the full glasses asset spans the temple distance
                # with a small margin so the arms reach toward the ears
                target_width = temple_distance * 1.05
                scale_factor = target_width / max(1.0, float(asset.shape[1]))
                scale_factor *= float(params.get("scale", 1.0))
                placement.width = float(asset.shape[1] * scale_factor)

                # Offset so the asset's eye center aligns with the face eye center
                asset_center = np.array(
                    [
                        asset.shape[1] / 2.0,
                        asset.shape[0] / 2.0,
                    ],
                    dtype=np.float32,
                )
                center_offset = (asset_center - asset_eye_center) * scale_factor
                placement.offset_x += float(center_offset[0])
                placement.offset_y += float(center_offset[1])
        except Exception:
            pass

    anchor_point = (
        placement.center[0] + placement.offset_x,
        placement.center[1] + placement.offset_y,
    )

    transformed = _rotate_and_resize_rgba(
        asset,
        target_width=placement.width,
        roll_deg=placement.roll_deg,
    )

    pose = ctx.get("pose", {})
    euler = pose.get("euler", {}) if isinstance(pose, dict) else {}
    final_yaw = placement.yaw_deg if placement.yaw_deg is not None else float(euler.get("yaw", 0.0))
    final_pitch = placement.pitch_deg if placement.pitch_deg is not None else float(euler.get("pitch", 0.0))
    transformed = _warp_perspective_rgba(
        transformed,
        yaw_deg=float(final_yaw),
        pitch_deg=float(final_pitch),
        accessory_type=accessory_type,
    )

    default_anchor_x, default_anchor_y = _default_anchor_ratio(accessory_type)
    anchor_x = float(params.get("anchor_x", params.get("anchor_x_ratio", default_anchor_x)))
    anchor_y = float(params.get("anchor_y", params.get("anchor_y_ratio", default_anchor_y)))
    if motion_anchor_ratio is not None:
        anchor_x, anchor_y = motion_anchor_ratio
    anchor_x = float(np.clip(anchor_x, 0.0, 1.0))
    anchor_y = float(np.clip(anchor_y, 0.0, 1.0))

    if accessory_type in {"left_earring", "right_earring"} and abs(motion_angle) > 0.01:
        transformed, rotated_anchor = _rotate_rgba_about_anchor(
            transformed,
            motion_angle,
            anchor_x,
            anchor_y,
        )
        anchor_x, anchor_y = rotated_anchor

    th, tw = transformed.shape[:2]
    final_center = (
        anchor_point[0] + (0.5 - anchor_x) * float(tw),
        anchor_point[1] + (0.5 - anchor_y) * float(th),
    )

    out = _alpha_blend_rgba(
        image_bgr,
        transformed,
        center=final_center,
        alpha_multiplier=placement.alpha,
        occlusion_mask=earring_occlusion_mask,
        occlusion_strength=earring_occlusion_strength,
    )
    if earring_debug is not None:
        ctx.setdefault("effect_debug_meta", {}).setdefault("earring_motion", []).append(earring_debug)
    return out


def apply_accessory_pack(
    image_bgr: np.ndarray,
    ctx: dict,
    params: dict | None = None,
) -> np.ndarray:
    """
    Apply multiple accessories.

    params example:
    {
        "enabled": true,
        "items": [
            {"type": "glasses", "asset_path": "...", "scale": 1.0},
            {"type": "left_earring", "asset_path": "..."},
            {"type": "right_earring", "asset_path": "..."},
            {"type": "necklace", "asset_path": "...", "offset_y_ratio": 0.12}
        ]
    }
    """

    params = params or {}

    if not params.get("enabled", False):
        return image_bgr

    result = image_bgr.copy()
    debug_items: list[dict] = []

    for item in params.get("items", []):
        accessory_type = item.get("type")

        if not accessory_type:
            continue

        if accessory_type == "necklaces":
            accessory_type = "necklace"
        elif accessory_type == "hats":
            accessory_type = "hat"
        elif accessory_type == "hair_clips":
            accessory_type = "hair_clip"

        if accessory_type == "earrings":
            for side_type in ("left_earring", "right_earring"):
                side_item = {
                    **item,
                    "type": side_type,
                }
                before = result
                reason = side_item.get("fallback_reason") or side_item.get("debug_missing_reason")
                error = None

                try:
                    result = place_accessory(
                        result,
                        ctx,
                        side_type,
                        side_item,
                    )
                except Exception as exc:
                    result = before
                    error = str(exc)
                    reason = reason or error

                changed_pixels = int(np.count_nonzero(np.any(before != result, axis=2)))
                if changed_pixels == 0 and not reason:
                    if not side_item.get("asset_path"):
                        reason = "missing_asset_path"
                    else:
                        reason = "asset_missing_or_not_drawn"

                debug_items.append(
                    {
                        "type": side_type,
                        "category": side_item.get("category"),
                        "asset_id": side_item.get("asset_id"),
                        "render_mode": side_item.get("render_mode", "overlay_2d"),
                        "applied": changed_pixels > 0,
                        "changed_pixels": changed_pixels,
                        "fallback_used": changed_pixels == 0,
                        "reason": reason,
                        "error": error,
                    }
                )
            continue

        before = result
        reason = item.get("fallback_reason") or item.get("debug_missing_reason")
        error = None

        try:
            result = place_accessory(
                result,
                ctx,
                accessory_type,
                item,
            )
        except Exception as exc:
            result = before
            error = str(exc)
            reason = reason or error

        changed_pixels = int(np.count_nonzero(np.any(before != result, axis=2)))
        if changed_pixels == 0 and not reason:
            if not item.get("asset_path"):
                reason = "missing_asset_path"
            else:
                reason = "asset_missing_or_not_drawn"

        debug_items.append(
            {
                "type": accessory_type,
                "category": item.get("category"),
                "asset_id": item.get("asset_id"),
                "render_mode": item.get("render_mode", "overlay_2d"),
                "applied": changed_pixels > 0,
                "changed_pixels": changed_pixels,
                "fallback_used": changed_pixels == 0,
                "reason": reason,
                "error": error,
            }
        )

    earring_motion_debug = ctx.get("effect_debug_meta", {}).get("earring_motion")
    ctx.setdefault("effect_debug_meta", {})["accessories"] = {
        "items": debug_items,
        "earring_motion": earring_motion_debug if isinstance(earring_motion_debug, list) else [],
        "fallback_used": any(item.get("fallback_used") for item in debug_items),
        "reason": next(
            (item.get("reason") for item in debug_items if item.get("reason")),
            None,
        ),
    }

    return result
