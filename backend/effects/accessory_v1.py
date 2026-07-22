from __future__ import annotations

import numpy as np

from backend.accessory_engine import apply_accessory_pack


def apply_accessories(
    image_bgr: np.ndarray,
    ctx: dict,
    params: dict | None = None,
) -> np.ndarray:
    return apply_accessory_pack(
        image_bgr,
        ctx,
        params,
    )