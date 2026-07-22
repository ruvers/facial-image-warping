from __future__ import annotations

import os
import numpy as np

from backend.accessory_engine import apply_accessory_pack
from backend.accessories.metadata import PHYSICS_RENDER_MODES
from backend.accessories.glasses_v2 import apply_glasses_v2
from backend.accessories.earrings_v2 import apply_earrings_v2
from backend.accessories.necklace_v2 import apply_necklace_v2


def apply_accessories(
    image_bgr: np.ndarray,
    ctx: dict,
    params: dict | None = None,
) -> np.ndarray:
    """
    Central accessory dispatcher.

    Example params:

    {
        "glasses": {
            "enabled": true,
            "asset_path": "assets/glasses/black.png",
            "width_scale": 1.0,
            "y_offset_ratio": 0.0
        },
        "earrings": {
            "enabled": true,
            "side": "both",
            "style": "diamond",
            "color": "gold",
            "scale": 1.0
        },
        "necklace": {
            "enabled": true,
            "style": "diamond",
            "color": "gold",
            "scale": 1.0
        }
    }
    """

    params = params or {}

    result = image_bgr.copy()

    if params.get("enabled", False) and isinstance(params.get("items"), list):
        overlay_items = [
            item
            for item in params.get("items", [])
            if not (
                isinstance(item, dict)
                and str(item.get("render_mode", "")).lower() in PHYSICS_RENDER_MODES
            )
        ]
        result = apply_accessory_pack(
            result,
            ctx,
            {
                **params,
                "enabled": bool(overlay_items),
                "items": overlay_items,
            },
        )

    # =====================================================
    # GLASSES
    # =====================================================

    glasses_params = params.get("glasses", {})

    if glasses_params.get("enabled", False):
        asset_path = glasses_params.get("asset_path")

        if asset_path and os.path.exists(asset_path):
            result = apply_glasses_v2(
                image_bgr=result,
                analysis=ctx,
                asset_path=asset_path,
                width_scale=float(glasses_params.get("width_scale", 1.0)),
                y_offset_ratio=float(glasses_params.get("y_offset_ratio", 0.0)),
            )

    # =====================================================
    # EARRINGS
    # =====================================================

    earrings_params = params.get("earrings", {})

    if earrings_params.get("enabled", False):
        result = apply_earrings_v2(
            image_bgr=result,
            ctx=ctx,
            params=earrings_params,
        )

    # =====================================================
    # NECKLACE
    # =====================================================

    necklace_params = params.get("necklace", {})

    if necklace_params.get("enabled", False):
        result = apply_necklace_v2(
            image_bgr=result,
            ctx=ctx,
            params=necklace_params,
        )

    return result
