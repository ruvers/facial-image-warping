from __future__ import annotations

import cv2
import numpy as np

from backend.warping import apply_expression_transform


def ctx_landmarks_to_legacy_list(ctx: dict) -> list[dict]:
    """
    Convert new ctx landmarks to old warping.py landmark format.
    warping.py expects dict landmarks with pixel x/y.
    """

    landmarks_2d = ctx.get("landmarks_2d")
    landmarks_3d = ctx.get("landmarks_3d")

    if landmarks_2d is None:
        raise ValueError("ctx['landmarks_2d'] missing")

    out = []

    for i, p in enumerate(landmarks_2d):
        z = 0.0

        if landmarks_3d is not None and i < len(landmarks_3d):
            z = float(landmarks_3d[i][2])

        out.append(
            {
                "index": int(i),
                "x": float(p[0]),
                "y": float(p[1]),
                "z": z,
                "visibility": 1.0,
            }
        )

    return out


def apply_warp_from_ctx(
    image_bgr: np.ndarray,
    ctx: dict,
    *,
    smile_intensity: float = 0.0,
    eyebrow_intensity: float = 0.0,
    lip_intensity: float = 0.0,
    slim_intensity: float = 0.0,
) -> np.ndarray:
    """
    Bridge new effect engine -> old warping.py.

    Engine uses BGR.
    warping.py was written for RGB.
    """

    if image_bgr is None:
        raise ValueError("image_bgr is None")

    landmarks = ctx_landmarks_to_legacy_list(ctx)

    image_rgb = cv2.cvtColor(
        image_bgr,
        cv2.COLOR_BGR2RGB,
    )

    warped_rgb = apply_expression_transform(
        image_rgb,
        landmarks,
        smile_intensity=float(smile_intensity),
        eyebrow_intensity=float(eyebrow_intensity),
        lip_intensity=float(lip_intensity),
        slim_intensity=float(slim_intensity),
    )

    warped_bgr = cv2.cvtColor(
        warped_rgb,
        cv2.COLOR_RGB2BGR,
    )

    return warped_bgr