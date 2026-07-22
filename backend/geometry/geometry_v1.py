from __future__ import annotations

import numpy as np

from backend.geometry.warp_adapter import apply_warp_from_ctx


def apply_face_reshape(
    image_bgr: np.ndarray,
    ctx: dict,
    params: dict | None = None,
) -> np.ndarray:
    """
    Stage 4 — Geometry Warp Engine.

    Supported now:
        - face_slimming / slim_intensity
        - lip_widening / lip_intensity

    Later extension:
        - nose_slim
        - chin_reshape
        - jawline
        - cheekbone
        - eye_enlarge

    Params example:
    {
        "enabled": true,
        "face_slimming": 0.35,
        "lip_intensity": 0.20
    }
    """

    params = params or {}

    slim_intensity = float(
        params.get(
            "slim_intensity",
            params.get("face_slimming", 0.0),
        )
    )

    lip_intensity = float(
        params.get(
            "lip_intensity",
            params.get("lip_widening", 0.0),
        )
    )

    slim_intensity = float(np.clip(slim_intensity, -1.0, 1.0))
    lip_intensity = float(np.clip(lip_intensity, -1.0, 1.0))

    if abs(slim_intensity) < 1e-6 and abs(lip_intensity) < 1e-6:
        return image_bgr

    return apply_warp_from_ctx(
        image_bgr,
        ctx,
        smile_intensity=0.0,
        eyebrow_intensity=0.0,
        lip_intensity=lip_intensity,
        slim_intensity=slim_intensity,
    )