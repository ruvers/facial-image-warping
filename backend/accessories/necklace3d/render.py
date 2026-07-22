from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from backend.accessories.common.compositing import (
    alpha_blend_rgba,
    apply_alpha_occlusion,
    apply_contact_shadow,
)


MATERIALS: dict[str, dict[str, tuple[int, int, int]]] = {
    "gold": {
        "base": (38, 178, 232),
        "dark": (20, 110, 155),
        "highlight": (180, 238, 255),
    },
    "silver": {
        "base": (205, 205, 205),
        "dark": (120, 120, 130),
        "highlight": (255, 255, 255),
    },
    "rose_gold": {
        "base": (128, 162, 220),
        "dark": (82, 96, 150),
        "highlight": (215, 225, 255),
    },
    "black": {
        "base": (24, 24, 28),
        "dark": (6, 6, 8),
        "highlight": (80, 80, 88),
    },
    "pearl": {
        "base": (218, 222, 235),
        "dark": (150, 154, 168),
        "highlight": (255, 255, 255),
    },
}


def _material(name: str) -> dict[str, tuple[int, int, int]]:
    return MATERIALS.get(str(name or "gold").lower(), MATERIALS["gold"])


def _to_rgba(color_bgr: tuple[int, int, int], alpha: int) -> tuple[int, int, int, int]:
    return int(color_bgr[0]), int(color_bgr[1]), int(color_bgr[2]), int(alpha)


def _draw_teardrop(
    overlay: np.ndarray,
    center: np.ndarray,
    size: float,
    mat: dict[str, tuple[int, int, int]],
    alpha: int,
) -> None:
    cx, cy = int(round(center[0])), int(round(center[1]))
    r = max(4, int(round(size)))
    pts = np.array(
        [
            [cx, cy - r],
            [cx + int(r * 0.82), cy],
            [cx, cy + int(r * 1.28)],
            [cx - int(r * 0.82), cy],
        ],
        dtype=np.int32,
    )
    cv2.fillPoly(overlay, [pts], _to_rgba(mat["base"], alpha), cv2.LINE_AA)
    cv2.polylines(overlay, [pts], True, _to_rgba(mat["dark"], min(255, alpha)), max(1, r // 5), cv2.LINE_AA)
    cv2.circle(
        overlay,
        (cx - max(1, r // 4), cy - max(1, r // 4)),
        max(1, r // 5),
        _to_rgba(mat["highlight"], min(180, alpha)),
        -1,
        cv2.LINE_AA,
    )


def render_necklace(
    image_bgr: np.ndarray,
    proxy: dict[str, Any],
    simulation: dict[str, Any],
    metadata: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    if image_bgr is None:
        raise ValueError("image_bgr is None")

    h, w = image_bgr.shape[:2]
    overlay = np.zeros((h, w, 4), dtype=np.uint8)
    nodes = np.asarray(simulation["nodes_2d"], dtype=np.float32)
    if nodes.ndim != 2 or nodes.shape[0] < 2:
        return image_bgr, {"applied": False, "reason": "insufficient_nodes"}

    mat = _material(str(metadata.get("material", "gold")))
    thickness = max(1, int(round(float(metadata.get("chain_thickness", 2.0)))))
    alpha = int(np.clip(float(metadata.get("opacity", 0.86)), 0.0, 1.0) * 255)
    pts = np.round(nodes).astype(np.int32).reshape((-1, 1, 2))

    # Layered strokes give a lightweight metallic/pearl bevel without a renderer.
    cv2.polylines(overlay, [pts], False, _to_rgba(mat["dark"], min(210, alpha)), thickness + 1, cv2.LINE_AA)
    cv2.polylines(overlay, [pts], False, _to_rgba(mat["base"], alpha), thickness, cv2.LINE_AA)

    highlight_nodes = nodes.copy()
    highlight_nodes[:, 1] -= max(1.0, thickness * 0.45)
    hpts = np.round(highlight_nodes).astype(np.int32).reshape((-1, 1, 2))
    cv2.polylines(overlay, [hpts], False, _to_rgba(mat["highlight"], min(118, alpha)), 1, cv2.LINE_AA)

    bead_step = max(3, nodes.shape[0] // 22)
    for p in nodes[::bead_step]:
        cv2.circle(
            overlay,
            (int(round(p[0])), int(round(p[1]))),
            max(1, thickness - 1),
            _to_rgba(mat["highlight"], min(135, alpha)),
            -1,
            cv2.LINE_AA,
        )

    pendant_enabled = bool(metadata.get("pendant_enabled", True))
    pendant_position = np.asarray(simulation.get("pendant_position"), dtype=np.float32)
    pendant_size_px = 0
    if pendant_enabled and pendant_position.shape == (2,):
        span = float(np.linalg.norm(nodes[-1] - nodes[0]))
        pendant_size_px = max(5, int(round(span * float(metadata.get("pendant_size", 0.12)))))
        _draw_teardrop(overlay, pendant_position, pendant_size_px, mat, alpha)

    overlay = apply_alpha_occlusion(
        overlay,
        proxy.get("hair_occlusion_mask"),
        strength=float(metadata.get("hair_occlusion_strength", 0.25)),
    )

    if bool(metadata.get("contact_shadow", True)):
        shadowed = apply_contact_shadow(
            image_bgr,
            overlay[:, :, 3],
            dx=1,
            dy=2,
            blur=17,
            opacity=0.09,
        )
    else:
        shadowed = image_bgr
    out = alpha_blend_rgba(shadowed, overlay)

    changed_pixels = int(np.count_nonzero(np.any(out != image_bgr, axis=2)))

    return out, {
        "applied": changed_pixels > 0,
        "changed_pixels": changed_pixels,
        "material": str(metadata.get("material", "gold")),
        "chain_thickness_px": thickness,
        "pendant_enabled": pendant_enabled,
        "pendant_size_px": pendant_size_px,
        "occlusion_applied": proxy.get("hair_occlusion_mask") is not None,
    }
