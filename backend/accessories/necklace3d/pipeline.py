from __future__ import annotations

from typing import Any

import numpy as np

from backend.accessories.metadata import normalize_necklace_metadata
from backend.accessories.necklace3d.body_proxy import build_neck_chest_proxy
from backend.accessories.necklace3d.physics import simulate_necklace_chain
from backend.accessories.necklace3d.render import render_necklace


def apply_physics_necklace(
    image_bgr: np.ndarray,
    ctx: dict[str, Any],
    item: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    metadata = normalize_necklace_metadata(item)
    category = str(item.get("category") or "pendant_necklace")
    render_mode = str(item.get("render_mode") or "physics_3d")

    try:
        proxy = build_neck_chest_proxy(image_bgr, ctx, metadata)
        simulation = simulate_necklace_chain(proxy, metadata)
        output, render_debug = render_necklace(image_bgr, proxy, simulation, metadata)
        changed_pixels = int(np.count_nonzero(np.any(output != image_bgr, axis=2)))
        proxy_debug = proxy.get("debug", {})
        proxy_confidence = float(proxy_debug.get("confidence", 0.0))
        proxy_fallback = bool(proxy_debug.get("fallback_used", False))

        meta = {
            "effect": "accessory_3d",
            "type": "necklace",
            "category": category,
            "render_mode": render_mode,
            "applied": changed_pixels > 0,
            "fallback_used": proxy_fallback,
            "fallback_reason": "low_body_proxy_confidence" if proxy_confidence < 0.50 else None,
            "reason": None if changed_pixels > 0 else "No visible pixel change.",
            "debug": {
                "body_proxy": proxy_debug,
                "physics": {
                    **simulation.get("debug", {}),
                    "collision_count": simulation.get("collision_count", 0),
                    "simulation_iterations": simulation.get("simulation_iterations", 0),
                    "pendant_position": np.asarray(simulation.get("pendant_position")).round(2).tolist(),
                    "nodes_preview": np.asarray(simulation.get("nodes_2d"))[:: max(1, len(simulation.get("nodes_2d")) // 8)].round(2).tolist(),
                },
                "render": render_debug,
            },
        }
        return output, meta

    except Exception as exc:
        return image_bgr, {
            "effect": "accessory_3d",
            "type": "necklace",
            "category": category,
            "render_mode": render_mode,
            "applied": False,
            "fallback_used": True,
            "reason": "physics_necklace_failed",
            "error": str(exc),
            "debug": {
                "body_proxy": {},
                "physics": {},
                "render": {},
            },
        }
