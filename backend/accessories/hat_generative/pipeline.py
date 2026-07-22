from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from backend.accessories.hat_generative.mask import build_hat_placement_mask
from backend.assets_manager import validate_asset_path
from backend.local_models.generative_refiner import apply_generative_hat_inpaint


GENERATIVE_HAT_MODES = {
    "hat_light_inpaint",
    "anydoor_inpaint",
    "generative_hat_inpaint",
    "sdxl_ip_adapter_inpaint",
    "hybrid_reference_inpaint",
}


def _load_hat_reference(item: dict[str, Any]) -> tuple[np.ndarray | None, dict[str, Any]]:
    asset_path = str(item.get("asset_path") or item.get("reference_path") or "").strip()
    if not asset_path:
        return None, {
            "loaded": False,
            "reason": "hat_reference_missing",
        }

    try:
        absolute = validate_asset_path(asset_path)
        ref = cv2.imread(absolute, cv2.IMREAD_COLOR)
        if ref is None:
            return None, {
                "loaded": False,
                "path": asset_path,
                "reason": "hat_reference_decode_failed",
            }
        return ref, {
            "loaded": True,
            "path": asset_path,
            "shape": list(ref.shape),
        }
    except Exception as exc:
        return None, {
            "loaded": False,
            "path": asset_path,
            "reason": str(exc),
        }


def apply_generative_hat(
    image_bgr: np.ndarray,
    ctx: dict[str, Any],
    item: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    render_mode = str(item.get("render_mode") or "").strip().lower()
    if render_mode not in GENERATIVE_HAT_MODES:
        return image_bgr.copy(), {
            "effect": "accessory_3d",
            "type": "hat",
            "category": item.get("category"),
            "render_mode": render_mode,
            "provider": None,
            "applied": False,
            "fallback_used": True,
            "error": "unsupported_generative_hat_render_mode",
            "debug": {},
        }

    try:
        metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
        mask, mask_debug = build_hat_placement_mask(
            image_bgr.shape,
            ctx,
            metadata,
        )
        hat_ref, reference_debug = _load_hat_reference(item)

        provider = "anydoor" if render_mode == "anydoor_inpaint" else "hat_light_inpaint"

        output, refiner_meta = apply_generative_hat_inpaint(
            image_bgr,
            hat_ref,
            mask,
            {
                **metadata,
                "provider": provider,
                "render_mode": render_mode,
                "asset_id": item.get("asset_id"),
            },
        )
        changed_pixels = int(np.count_nonzero(np.any(output != image_bgr, axis=2)))

        return output, {
            "effect": "accessory_3d",
            "type": "hat",
            "category": item.get("category") or "hat",
            "render_mode": render_mode,
            "provider": refiner_meta.get("provider", provider),
            "applied": changed_pixels > 0,
            "fallback_used": bool(refiner_meta.get("fallback_used", True)),
            "error": refiner_meta.get("error"),
            "changed_pixels": changed_pixels,
            "debug": {
                "mask": mask_debug,
                "reference": reference_debug,
                "refiner": refiner_meta,
            },
        }
    except Exception as exc:
        return image_bgr.copy(), {
            "effect": "accessory_3d",
            "type": "hat",
            "category": item.get("category"),
            "render_mode": render_mode,
            "provider": None,
            "applied": False,
            "fallback_used": True,
            "error": str(exc),
            "debug": {},
        }
