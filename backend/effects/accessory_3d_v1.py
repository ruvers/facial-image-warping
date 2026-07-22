from __future__ import annotations

from typing import Any

import numpy as np

from backend.accessories.metadata import is_physics_necklace_item
from backend.accessories.hat_generative.pipeline import apply_generative_hat
from backend.accessories.necklace3d.pipeline import apply_physics_necklace


GENERATIVE_HAT_MODES = {
    "hat_light_inpaint",
    "anydoor_inpaint",
    "generative_hat_inpaint",
    "sdxl_ip_adapter_inpaint",
    "hybrid_reference_inpaint",
}


PARAMETRIC_HAT_MODES = {
    "parametric_3d",
    "hybrid_3d_refine",
}


def _is_hat_item(item: dict[str, Any]) -> bool:
    kind = str(item.get("type") or "").lower()
    category = str(item.get("category") or "").lower()
    return kind in {"hat", "hats"} or category in {
        "hat",
        "hats",
        "beanie",
        "baseball_cap",
        "bucket_hat",
        "fedora",
    }


def apply_accessory_3d(
    image_bgr: np.ndarray,
    ctx: dict,
    params: dict | None = None,
) -> np.ndarray:
    """
    Physics/generative accessory dispatcher.

    Current supported paths:
    - necklace + render_mode physics_3d/parametric_3d
    - hat + render_mode anydoor_inpaint/generative_hat_inpaint

    Procedural/parametric hats are intentionally experimental-only and do not
    modify the output image.

    Other items are intentionally left untouched so the existing overlay
    accessory system can continue to handle glasses and legacy assets.
    """
    params = params or {}
    if not params.get("enabled", False):
        return image_bgr

    items = params.get("items", [])
    if not isinstance(items, list):
        return image_bgr

    result = image_bgr.copy()
    item_meta: list[dict[str, Any]] = []

    for item in items:
        if not isinstance(item, dict):
            continue

        if is_physics_necklace_item(item):
            result, meta = apply_physics_necklace(
                result,
                ctx,
                item,
            )
            item_meta.append(meta)
            continue

        render_mode = str(item.get("render_mode") or "overlay_2d").lower()

        if _is_hat_item(item) and render_mode in PARAMETRIC_HAT_MODES:
            item_meta.append(
                {
                    "effect": "accessory_3d",
                    "type": "hat",
                    "category": item.get("category"),
                    "render_mode": render_mode,
                    "provider": "procedural_parametric",
                    "experimental": True,
                    "applied": False,
                    "fallback_used": True,
                    "error": "parametric_hat_disabled_experimental_only",
                    "debug": {
                        "reason": "Procedural/parametric hat is disabled for production output.",
                    },
                }
            )
            continue

        if _is_hat_item(item) and render_mode in GENERATIVE_HAT_MODES:
            result, meta = apply_generative_hat(
                result,
                ctx,
                item,
            )
            item_meta.append(meta)
            continue

        if render_mode in {
            "physics_3d",
            "parametric_3d",
            "hybrid_3d_refine",
            "anydoor_inpaint",
            "hat_light_inpaint",
            "sdxl_ip_adapter_inpaint",
            "generative_hat_inpaint",
            "hybrid_reference_inpaint",
        }:
            item_meta.append(
                {
                    "effect": "accessory_3d",
                    "type": item.get("type"),
                    "category": item.get("category"),
                    "render_mode": render_mode,
                    "provider": None,
                    "applied": False,
                    "fallback_used": True,
                    "error": "unsupported_3d_accessory_item",
                    "debug": {},
                }
            )

    if item_meta:
        ctx.setdefault("effect_debug_meta", {})
        ctx["effect_debug_meta"]["accessory_3d"] = {
            "items": item_meta,
            "type": item_meta[0].get("type") if len(item_meta) == 1 else "multi",
            "category": item_meta[0].get("category") if len(item_meta) == 1 else "multi",
            "render_mode": item_meta[0].get("render_mode") if len(item_meta) == 1 else "mixed",
            "applied_items": [
                item for item in item_meta if bool(item.get("applied", False))
            ],
            "fallback_used": any(bool(item.get("fallback_used", False)) for item in item_meta),
            "provider": item_meta[0].get("provider") if len(item_meta) == 1 else "mixed",
            "error": item_meta[0].get("error") if len(item_meta) == 1 else None,
            "debug": item_meta[0].get("debug", {}) if len(item_meta) == 1 else {"items": item_meta},
        }
    else:
        ctx.setdefault("effect_debug_meta", {})
        ctx["effect_debug_meta"]["accessory_3d"] = {
            "items": [],
            "fallback_used": False,
            "reason": "no physics_3d or parametric_3d accessory items requested",
        }

    return result
