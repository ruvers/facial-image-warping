from __future__ import annotations

import numpy as np

from backend.effect_catalog import normalize_params
from backend.effect_engine import run_photo_engine
from backend.register_default_effects import register_default_effects


_DEFAULTS_REGISTERED = False


def apply_photo_pipeline(
    image_bgr: np.ndarray,
    params: dict | None = None,
) -> tuple[np.ndarray, dict]:
    """
    Main photo-mode pipeline.

    Responsibilities:
    - register default effects once
    - normalize frontend params
    - run face analysis once
    - enrich context with 3D-ready data
    - apply enabled effects in stage order
    - return final image + context
    """

    global _DEFAULTS_REGISTERED

    if not _DEFAULTS_REGISTERED:
        register_default_effects()
        _DEFAULTS_REGISTERED = True

    normalized_params = normalize_params(
        params,
    )

    accessories = normalized_params.get("accessories")
    if isinstance(accessories, dict):
        items = accessories.get("items")
        if isinstance(items, list):
            has_3d_item = any(
                str(item.get("render_mode", "")).lower() in {"physics_3d", "parametric_3d", "hybrid_3d_refine"}
                for item in items
                if isinstance(item, dict)
            )
            if has_3d_item:
                normalized_params["accessory_3d"] = {
                    "enabled": True,
                    "items": items,
                }

    return run_photo_engine(
        image_bgr,
        normalized_params,
    )
