from __future__ import annotations

import traceback

import numpy as np

from backend.three_d.provider import enrich_with_3d_context
from backend.three_d.deca_provider import (
    is_deca_available,
    enrich_with_deca_context,
)


def enrich_with_best_available_3d(
    ctx: dict,
    image_bgr: np.ndarray,
) -> dict:
    """
    Main 3D provider manager.

    Policy:
        1. If DECA/FLAME is available, use it.
        2. Otherwise use MediaPipe pseudo-3D.
        3. If all fails, attach a failed three_d block but do not crash pipeline.

    The frontend does not choose provider.
    The backend uses the best available provider automatically.
    """

    if ctx is None:
        raise ValueError("ctx is None")

    if image_bgr is None:
        raise ValueError("image_bgr is None")

    # =====================================================
    # Future true 3D provider: DECA / FLAME
    # =====================================================

    try:
        if is_deca_available():
            deca_ctx = enrich_with_deca_context(
                ctx,
                image_bgr,
            )

            if deca_ctx is not None:
                deca_ctx.setdefault("three_d", {})
                deca_ctx["three_d"]["provider_priority"] = "deca_first"
                return deca_ctx

    except Exception as e:
        print("[WARN] DECA/FLAME provider failed. Falling back to MediaPipe pseudo-3D.")
        print(e)
        traceback.print_exc()

    # =====================================================
    # Current working provider: MediaPipe pseudo-3D
    # =====================================================

    try:
        mp_ctx = enrich_with_3d_context(
            ctx,
            image_bgr,
        )

        mp_ctx.setdefault("three_d", {})
        mp_ctx["three_d"]["provider_priority"] = "mediapipe_fallback"

        return mp_ctx

    except Exception as e:
        print("[WARN] MediaPipe pseudo-3D provider failed.")
        print(e)
        traceback.print_exc()

        ctx["three_d"] = {
            "provider": "failed",
            "is_true_3d": False,
            "error": str(e),
            "provider_priority": "none",
        }

        return ctx