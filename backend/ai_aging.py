from __future__ import annotations

from typing import Any

import cv2
import numpy as np


def get_ai_aging_status() -> dict[str, Any]:
    try:
        from backend.local_models.sam_aging import get_sam_aging_status

        sam_status = get_sam_aging_status()

        return {
            "provider": "local_sam",
            "remote_api": False,
            "replicate_used": False,
            "available": bool(sam_status.get("available", False)),
            "sam": sam_status,
            "notes": [
                "Replicate aging path has been removed.",
                "This compatibility wrapper delegates to the local SAM aging plugin.",
            ],
        }

    except Exception as exc:
        return {
            "provider": "local_sam",
            "remote_api": False,
            "replicate_used": False,
            "available": False,
            "error": repr(exc),
        }


def apply_ai_aging(
    image_rgb: np.ndarray,
    target_age: float = 60.0,
) -> np.ndarray:
    """
    Backward-compatible local aging entrypoint.

    Old code used Replicate here.
    Current code must stay local-only and delegate to the SAM aging plugin.
    """

    if image_rgb is None:
        raise ValueError("image_rgb is None")

    try:
        from backend.local_models.custom_plugins.aging_model import apply as apply_aging_model

        image_bgr = cv2.cvtColor(
            image_rgb,
            cv2.COLOR_RGB2BGR,
        )

        params = {
            "enabled": True,
            "target_age": int(float(target_age)),
            "intensity": 1.0,
            "aging_model": {
                "enabled": True,
                "target_age": int(float(target_age)),
                "intensity": 1.0,
            },
        }

        output_bgr = apply_aging_model(
            image_bgr,
            {},
            params,
        )

        if output_bgr is None:
            return image_rgb.copy()

        return cv2.cvtColor(
            output_bgr,
            cv2.COLOR_BGR2RGB,
        )

    except Exception as exc:
        print(f"[AI Aging][WARN] local SAM wrapper failed: {exc}")
        return image_rgb.copy()