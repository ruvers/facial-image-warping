from __future__ import annotations

import numpy as np

from backend.geometry.warp_adapter import apply_warp_from_ctx


def apply_expression_effect(
    image_bgr: np.ndarray,
    ctx: dict,
    params: dict | None = None,
) -> np.ndarray:
    """
    Stage 5 — Expression Engine.

    Supported now:
        - smile
        - eyebrow raise
        - lip widening

    Params example:
    {
        "enabled": true,
        "smile_intensity": 0.45,
        "eyebrow_intensity": 0.20,
        "lip_intensity": 0.10
    }
    """

    params = params or {}

    smile_intensity = float(
        params.get("smile_intensity", params.get("smile", 0.0))
    )

    eyebrow_intensity = float(
        params.get(
            "eyebrow_intensity",
            params.get("eyebrow_height", 0.0),
        )
    )

    lip_intensity = float(
        params.get(
            "lip_intensity",
            params.get("lip_widening", 0.0),
        )
    )

    smile_intensity = float(np.clip(smile_intensity, -1.0, 1.0))
    eyebrow_intensity = float(np.clip(eyebrow_intensity, -1.0, 1.0))
    lip_intensity = float(np.clip(lip_intensity, -1.0, 1.0))

    if (
        abs(smile_intensity) < 1e-6
        and abs(eyebrow_intensity) < 1e-6
        and abs(lip_intensity) < 1e-6
    ):
        return image_bgr

    return apply_warp_from_ctx(
        image_bgr,
        ctx,
        smile_intensity=smile_intensity,
        eyebrow_intensity=eyebrow_intensity,
        lip_intensity=lip_intensity,
        slim_intensity=0.0,
    )