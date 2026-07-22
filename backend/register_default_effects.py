from __future__ import annotations

import cv2
import numpy as np

from backend.effect_engine import photo_engine

from backend.effects.beauty_v1 import (
    apply_blush,
    apply_eyebrow_enhance,
    apply_lipstick,
    apply_skin_smooth,
    apply_beard,
)
from backend.effects.color_v1 import (
    apply_eye_color,
    apply_eyeliner,
    apply_eyeshadow,
)

from backend.effects.model_slots_v1 import (
    apply_accessory_model_slot,
    apply_aging_model_slot,
    apply_expression_model_slot,
    apply_face_restore_slot,
    apply_makeup_model_slot,
)
from backend.effects.accessory_3d_v1 import apply_accessory_3d

from backend.effects.geometry_v1 import apply_face_reshape
from backend.effects.expression_v1 import apply_expression_effect
from backend.effects.hair_color_v2 import apply_faceapp_hair_color

from backend.accessories.manager import apply_accessories
from backend.ai_enhancement.manager import apply_ai_enhancement


_DEFAULT_EFFECTS_REGISTERED = False


def _hex_to_bgr(hex_color: str) -> tuple[int, int, int]:
    hex_color = hex_color.strip().replace("#", "")

    if len(hex_color) != 6:
        raise ValueError("hex_color must be like '#6633AA'")

    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)

    return b, g, r


def _reinforce_visible_hair_tint(
    image_rgb: np.ndarray,
    hair_mask: np.ndarray,
    target_bgr: tuple[int, int, int],
    intensity: float,
) -> np.ndarray:
    h, w = image_rgb.shape[:2]
    if hair_mask.shape[:2] != (h, w):
        hair_mask = cv2.resize(
            hair_mask,
            (w, h),
            interpolation=cv2.INTER_NEAREST,
        )

    alpha = cv2.GaussianBlur(
        (hair_mask > 20).astype(np.float32),
        (0, 0),
        sigmaX=5.0,
        sigmaY=5.0,
    )
    alpha = np.clip(alpha, 0.0, 1.0)[..., None]

    tint_rgb = np.array(
        [target_bgr[2], target_bgr[1], target_bgr[0]],
        dtype=np.float32,
    )
    tint_strength = float(np.clip(intensity, 0.0, 1.0)) * 0.28
    image_f = image_rgb.astype(np.float32)
    blended = image_f * (1.0 - alpha * tint_strength) + tint_rgb * (alpha * tint_strength)
    return np.clip(blended, 0, 255).astype(np.uint8)


# =========================================================
# HAIR COLOR WRAPPER
# =========================================================

def hair_color_effect(
    image_bgr: np.ndarray,
    ctx: dict,
    params: dict,
) -> np.ndarray:
    """
    FaceApp-style hair recoloring.

    Engine works in BGR. The v2 implementation preserves luminance and texture,
    and falls back to the local BiSeNet parser when ctx does not include a
    usable hair mask.
    """

    hair_mask = ctx.get("masks", {}).get("hair")

    return apply_faceapp_hair_color(
        image_bgr,
        hair_mask,
        color=params.get("color", "#7B3FE4"),
        intensity=float(params.get("intensity", 0.75)),
    )


# =========================================================
# ACCESSORY WRAPPERS
# =========================================================

def accessories_effect(
    image_bgr: np.ndarray,
    ctx: dict,
    params: dict,
) -> np.ndarray:
    """
    Generic accessory dispatcher.

    Params example:
    {
        "enabled": true,
        "glasses": {...},
        "earrings": {...},
        "necklace": {...}
    }
    """

    return apply_accessories(
        image_bgr=image_bgr,
        ctx=ctx,
        params=params,
    )


def glasses_effect(
    image_bgr: np.ndarray,
    ctx: dict,
    params: dict,
) -> np.ndarray:
    """
    Direct top-level glasses effect.

    Params example:
    {
        "enabled": true,
        "asset_path": "assets/glasses/black.png",
        "width_scale": 1.0,
        "y_offset_ratio": 0.0
    }
    """

    return apply_accessories(
        image_bgr=image_bgr,
        ctx=ctx,
        params={
            "glasses": params,
        },
    )


def earrings_effect(
    image_bgr: np.ndarray,
    ctx: dict,
    params: dict,
) -> np.ndarray:
    """
    Direct top-level earrings effect.
    """

    return apply_accessories(
        image_bgr=image_bgr,
        ctx=ctx,
        params={
            "earrings": params,
        },
    )


def necklace_effect(
    image_bgr: np.ndarray,
    ctx: dict,
    params: dict,
) -> np.ndarray:
    """
    Direct top-level necklace effect.
    """

    return apply_accessories(
        image_bgr=image_bgr,
        ctx=ctx,
        params={
            "necklace": params,
        },
    )


# =========================================================
# DEFAULT REGISTRATION
# =========================================================

def register_default_effects() -> None:
    """
    Register built-in effects.

    Effect interface:
        def apply_xxx(image_bgr, ctx, params) -> image_bgr

    Stage order comes from effect_engine.py:
        pre_geometry
        geometry
        beauty
        hair
        accessory
        postprocess
    """

    global _DEFAULT_EFFECTS_REGISTERED

    if _DEFAULT_EFFECTS_REGISTERED:
        return

    # =====================================================
    # STAGE 4 — GEOMETRY WARP ENGINE
    # =====================================================

    photo_engine.register(
        name="face_reshape",
        stage="geometry",
        fn=apply_face_reshape,
    )

    # =====================================================
    # STAGE 5 — EXPRESSION ENGINE
    # =====================================================

    photo_engine.register(
        name="expression",
        stage="geometry",
        fn=apply_expression_effect,
    )

    photo_engine.register(
        name="expression_model",
        stage="geometry",
        fn=apply_expression_model_slot,
    )

    # =====================================================
    # STAGE 3 — BEAUTY EFFECTS
    # =====================================================

    photo_engine.register(
        name="skin_smooth",
        stage="beauty",
        fn=apply_skin_smooth,
    )

    photo_engine.register(
        name="blush",
        stage="beauty",
        fn=apply_blush,
    )

    photo_engine.register(
        name="lipstick",
        stage="beauty",
        fn=apply_lipstick,
    )

    photo_engine.register(
        name="eye_color",
        stage="beauty",
        fn=apply_eye_color,
    )

    photo_engine.register(
        name="eyeshadow",
        stage="beauty",
        fn=apply_eyeshadow,
    )

    photo_engine.register(
        name="eyeliner",
        stage="beauty",
        fn=apply_eyeliner,
    )

    photo_engine.register(
        name="eyebrow",
        stage="beauty",
        fn=apply_eyebrow_enhance,
    )

    photo_engine.register(
        name="beard",
        stage="beauty",
        fn=apply_beard,
    )

    photo_engine.register(
        name="makeup_model",
        stage="beauty",
        fn=apply_makeup_model_slot,
    )
    # =====================================================
    # HAIR
    # =====================================================

    photo_engine.register(
        name="hair_color",
        stage="hair",
        fn=hair_color_effect,
    )

    # =====================================================
    # STAGE 2 — ACCESSORY ENGINE
    # =====================================================
    # İki kullanım da destekli:
    #
    # 1) Tek dispatcher:
    #    "accessories": {
    #        "enabled": true,
    #        "earrings": {...},
    #        "necklace": {...}
    #    }
    #
    # 2) Direkt top-level:
    #    "earrings": {"enabled": true, ...}
    #    "necklace": {"enabled": true, ...}
    #    "glasses": {"enabled": true, ...}

    photo_engine.register(
        name="accessory_3d",
        stage="accessory",
        fn=apply_accessory_3d,
    )

    photo_engine.register(
        name="accessories",
        stage="accessory",
        fn=accessories_effect,
    )

    photo_engine.register(
        name="glasses",
        stage="accessory",
        fn=glasses_effect,
    )

    photo_engine.register(
        name="earrings",
        stage="accessory",
        fn=earrings_effect,
    )

    photo_engine.register(
        name="necklace",
        stage="accessory",
        fn=necklace_effect,
    )

    photo_engine.register(
        name="accessory_model",
        stage="accessory",
        fn=apply_accessory_model_slot,
    )

    _DEFAULT_EFFECTS_REGISTERED = True

    # =====================================================
    # AI / POSTPROCESS
    # =====================================================
    # Real AI model is not active yet.
    # Current behavior: optional lightweight CV polish fallback.

    photo_engine.register(
        name="ai_enhance",
        stage="postprocess",
        fn=apply_ai_enhancement,
    )

    photo_engine.register(
        name="aging_model",
        stage="postprocess",
        fn=apply_aging_model_slot,
    )

    photo_engine.register(
        name="face_restore",
        stage="postprocess",
        fn=apply_face_restore_slot,
    )
