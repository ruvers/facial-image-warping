from __future__ import annotations

import numpy as np

from backend.local_models.slots import apply_local_model_slot


def apply_aging_model_slot(
    image_bgr: np.ndarray,
    ctx: dict,
    params: dict | None = None,
) -> np.ndarray:
    return apply_local_model_slot(
        "aging_model",
        image_bgr,
        ctx,
        params,
    )


def apply_makeup_model_slot(
    image_bgr: np.ndarray,
    ctx: dict,
    params: dict | None = None,
) -> np.ndarray:
    return apply_local_model_slot(
        "makeup_model",
        image_bgr,
        ctx,
        params,
    )


def apply_expression_model_slot(
    image_bgr: np.ndarray,
    ctx: dict,
    params: dict | None = None,
) -> np.ndarray:
    return apply_local_model_slot(
        "expression_model",
        image_bgr,
        ctx,
        params,
    )


def apply_accessory_model_slot(
    image_bgr: np.ndarray,
    ctx: dict,
    params: dict | None = None,
) -> np.ndarray:
    return apply_local_model_slot(
        "accessory_model",
        image_bgr,
        ctx,
        params,
    )


def apply_face_restore_slot(
    image_bgr: np.ndarray,
    ctx: dict,
    params: dict | None = None,
) -> np.ndarray:
    return apply_local_model_slot(
        "face_restore",
        image_bgr,
        ctx,
        params,
    )