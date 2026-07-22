from __future__ import annotations

from typing import Any

import numpy as np

from backend.accessories.hat3d.fitting import fit_beanie_to_head
from backend.accessories.hat3d.head_proxy import build_head_proxy
from backend.accessories.hat3d.render import render_beanie
from backend.accessories.hat3d.templates import build_beanie_template


def apply_parametric_hat(
    image_bgr: np.ndarray,
    ctx: dict[str, Any],
    item: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    category = str(item.get("category") or "beanie")
    render_mode = str(item.get("render_mode") or "parametric_3d")

    if category not in {"beanie", "hat", "hats"}:
        return image_bgr, {
            "effect": "accessory_3d",
            "type": "hat",
            "category": category,
            "render_mode": render_mode,
            "applied": False,
            "fallback_used": True,
            "reason": "unsupported_hat_category",
            "error": None,
            "debug": {},
        }

    try:
        template = build_beanie_template(item)
        head_proxy = build_head_proxy(image_bgr, ctx, template["metadata"])
        fitted = fit_beanie_to_head(head_proxy, template)
        output, render_debug = render_beanie(image_bgr, head_proxy, fitted, template)
        changed_pixels = int(np.count_nonzero(np.any(output != image_bgr, axis=2)))

        confidence = float(head_proxy.get("confidence", 0.0))
        fallback_used = bool(head_proxy.get("fallback_used", False))

        return output, {
            "effect": "accessory_3d",
            "type": "hat",
            "category": "beanie",
            "render_mode": "parametric_3d",
            "applied": changed_pixels > 0,
            "fallback_used": fallback_used,
            "fallback_reason": "low_head_proxy_confidence" if confidence < 0.50 else None,
            "reason": None if changed_pixels > 0 else "No visible pixel change.",
            "error": None,
            "debug": {
                "head_proxy": {
                    "source": head_proxy.get("source"),
                    "confidence": confidence,
                    "pose": head_proxy.get("pose"),
                    "head_ellipse": head_proxy.get("head_ellipse"),
                    "face_top_y": head_proxy.get("face_top_y"),
                    "brow_y": head_proxy.get("brow_y"),
                    "eye_y": head_proxy.get("eye_y"),
                    "hairline_y": head_proxy.get("hairline_y"),
                    "forehead_y": head_proxy.get("forehead_y"),
                    "bottom_y": head_proxy.get("bottom_y"),
                    "skull_top_y": head_proxy.get("skull_top_y"),
                    "fallback_used": fallback_used,
                },
                "placement": {
                    **fitted.get("placement", {}),
                    "contact_line": np.asarray(fitted.get("contact_line")).round(2).tolist(),
                },
                "render": render_debug,
                "hat_debug": {
                    "category": "beanie",
                    "render_mode": "parametric_3d",
                    "face_top_y": head_proxy.get("face_top_y"),
                    "brow_y": head_proxy.get("brow_y"),
                    "eye_y": head_proxy.get("eye_y"),
                    "hairline_y": head_proxy.get("hairline_y"),
                    "bottom_y": fitted.get("placement", {}).get("bottom_y"),
                    "top_y": fitted.get("placement", {}).get("top_y"),
                    "bottom_above_brow": fitted.get("placement", {}).get("bottom_above_brow"),
                    "shape": "beanie_dome",
                    "fallback_used": fallback_used,
                    "error": None,
                },
                "refinement": {
                    "available": False,
                    "used": False,
                    "reason": "generative refinement intentionally not implemented in MVP",
                },
            },
        }

    except Exception as exc:
        return image_bgr, {
            "effect": "accessory_3d",
            "type": "hat",
            "category": category,
            "render_mode": render_mode,
            "applied": False,
            "fallback_used": True,
            "reason": "parametric_hat_failed",
            "error": str(exc),
            "debug": {},
        }
