from __future__ import annotations

import json
import os
from typing import Any

import cv2
import numpy as np


def load_rgba_asset(asset_path: str) -> np.ndarray:
    asset = cv2.imread(asset_path, cv2.IMREAD_UNCHANGED)
    if asset is None:
        raise FileNotFoundError(f"Asset not found: {asset_path}")

    if asset.ndim == 2:
        asset = cv2.cvtColor(asset, cv2.COLOR_GRAY2BGRA)

    if asset.shape[2] == 3:
        alpha = np.full(asset.shape[:2], 255, dtype=np.uint8)
        asset = np.dstack([asset, alpha])

    return asset


def load_asset_meta(asset_path: str, asset_shape=None) -> dict[str, Any]:
    """
    Companion JSON format:
    same/path/glasses.png
    same/path/glasses.json

    Example:
    {
      "anchors": {
        "left_eye": [120, 110],
        "right_eye": [280, 110],
        "nose_bridge": [200, 120]
      }
    }
    """
    meta_path = os.path.splitext(asset_path)[0] + ".json"

    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            return json.load(f)

    if asset_shape is None:
        raise ValueError("asset_shape required for fallback meta")

    h, w = asset_shape[:2]

    # fallback guess
    return {
        "anchors": {
            "left_eye": [int(w * 0.32), int(h * 0.48)],
            "right_eye": [int(w * 0.68), int(h * 0.48)],
            "nose_bridge": [int(w * 0.50), int(h * 0.50)],
        }
    }


def warp_asset_affine(
    asset_rgba: np.ndarray,
    src_points: np.ndarray,
    dst_points: np.ndarray,
    output_size: tuple[int, int],
) -> np.ndarray:
    """
    output_size = (width, height)
    """
    src = np.asarray(src_points, dtype=np.float32)
    dst = np.asarray(dst_points, dtype=np.float32)

    M, _ = cv2.estimateAffinePartial2D(src, dst, method=cv2.LMEDS)
    if M is None:
        raise RuntimeError("Could not estimate affine transform for asset")

    out_w, out_h = output_size

    warped = cv2.warpAffine(
        asset_rgba,
        M,
        (out_w, out_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )

    return warped


def alpha_blend_rgba(
    background_bgr: np.ndarray,
    overlay_rgba: np.ndarray,
) -> np.ndarray:
    if background_bgr.shape[:2] != overlay_rgba.shape[:2]:
        raise ValueError("Background and overlay must have same spatial size")

    bg = background_bgr.astype(np.float32)
    fg = overlay_rgba[:, :, :3].astype(np.float32)
    alpha = overlay_rgba[:, :, 3:4].astype(np.float32) / 255.0

    out = fg * alpha + bg * (1.0 - alpha)
    out = np.clip(out, 0, 255).astype(np.uint8)
    return out


def mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def clamp_glasses_side_arms(
    warped_rgba: np.ndarray,
    face_mask: np.ndarray | None,
    extra_ratio: float = 0.08,
) -> np.ndarray:
    """
    Prevent the glasses arms from sticking out too much.
    This is a practical clamp, not a true 3D solution.
    """
    if face_mask is None:
        return warped_rgba

    bbox = mask_bbox(face_mask)
    if bbox is None:
        return warped_rgba

    x1, _, x2, _ = bbox
    face_width = max(1, x2 - x1)
    pad = int(face_width * extra_ratio)

    keep_left = max(0, x1 - pad)
    keep_right = min(warped_rgba.shape[1] - 1, x2 + pad)

    result = warped_rgba.copy()
    alpha = result[:, :, 3]

    alpha[:, :keep_left] = 0
    alpha[:, keep_right + 1:] = 0

    result[:, :, 3] = alpha
    return result