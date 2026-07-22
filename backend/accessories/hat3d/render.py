from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from backend.accessories.common.compositing import (
    alpha_blend_rgba,
    apply_alpha_occlusion,
    apply_contact_shadow,
)


def _hex_to_bgr(hex_color: str) -> tuple[int, int, int]:
    raw = str(hex_color or "#222222").strip().replace("#", "")
    if len(raw) != 6:
        raw = "222222"
    r = int(raw[0:2], 16)
    g = int(raw[2:4], 16)
    b = int(raw[4:6], 16)
    return b, g, r


def _rgba(color: tuple[int, int, int], alpha: int) -> tuple[int, int, int, int]:
    return int(color[0]), int(color[1]), int(color[2]), int(alpha)


def _blend_color(c1: tuple[int, int, int], scale: float) -> tuple[int, int, int]:
    return tuple(int(np.clip(v * scale, 0, 255)) for v in c1)


def render_beanie(
    image_bgr: np.ndarray,
    head_proxy: dict[str, Any],
    fitted: dict[str, Any],
    template: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    if image_bgr is None:
        raise ValueError("image_bgr is None")

    h, w = image_bgr.shape[:2]
    overlay = np.zeros((h, w, 4), dtype=np.uint8)
    meta = template["metadata"]
    base = _hex_to_bgr(meta.get("color", "#222222"))
    dark = _blend_color(base, 0.68)
    light = tuple(int(np.clip(v + 34, 0, 255)) for v in base)
    alpha = int(np.clip(float(meta.get("opacity", 0.78)), 0.55, 0.90) * 255)

    outer = fitted["outer_polygon"].astype(np.int32)
    fold = fitted["fold_polygon"].astype(np.int32)

    cv2.fillPoly(overlay, [outer], _rgba(base, alpha), cv2.LINE_AA)

    placement = fitted["placement"]
    cx, cy = int(placement["center"][0]), int(placement["center"][1])
    rx, ry = float(placement["rx"]), float(placement["ry"])
    roll = float(placement["roll"])

    # Fabric curvature: wide soft highlight plus darker side, constrained by alpha.
    highlight = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(
        highlight,
        (cx - int(rx * 0.18), int(cy - ry * 0.20)),
        (max(3, int(rx * 0.30)), max(3, int(ry * 0.34))),
        roll,
        0,
        360,
        70,
        -1,
        cv2.LINE_AA,
    )
    hat_alpha = overlay[:, :, 3] > 0
    overlay[:, :, :3] = np.where(
        (highlight[:, :, None] > 0) & hat_alpha[:, :, None],
        (overlay[:, :, :3].astype(np.float32) * 0.82 + np.array(light, dtype=np.float32) * 0.18).astype(np.uint8),
        overlay[:, :, :3],
    )

    side_shadow = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(
        side_shadow,
        (cx + int(rx * 0.28), int(cy - ry * 0.05)),
        (max(3, int(rx * 0.26)), max(3, int(ry * 0.48))),
        roll,
        0,
        360,
        70,
        -1,
        cv2.LINE_AA,
    )
    overlay[:, :, :3] = np.where(
        (side_shadow[:, :, None] > 0) & hat_alpha[:, :, None],
        (overlay[:, :, :3].astype(np.float32) * 0.84 + np.array(dark, dtype=np.float32) * 0.16).astype(np.uint8),
        overlay[:, :, :3],
    )

    cv2.fillPoly(overlay, [fold], _rgba(_blend_color(base, 0.76), min(255, alpha + 14)), cv2.LINE_AA)

    # Subtle fabric variation without visible barcode/vertex/debug lines.
    texture_strength = float(np.clip(float(meta.get("fabric_texture_strength", 0.25)), 0.0, 0.55))
    if texture_strength > 1e-3:
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        weave = (
            np.sin((xx + yy * 0.18) / max(5.0, rx * 0.12))
            + np.sin((yy - xx * 0.10) / max(6.0, ry * 0.16))
        ) * 0.5
        weave = cv2.GaussianBlur(weave.astype(np.float32), (0, 0), 1.2)
        factor = 1.0 + weave[:, :, None] * (0.035 * texture_strength)
        rgb = overlay[:, :, :3].astype(np.float32)
        rgb = np.where(hat_alpha[:, :, None], rgb * factor, rgb)
        overlay[:, :, :3] = np.clip(rgb, 0, 255).astype(np.uint8)

    edge_softness = int(max(3, min(11, round(float(meta.get("edge_softness", 0.65)) * 9)))) | 1
    overlay[:, :, 3] = cv2.GaussianBlur(overlay[:, :, 3], (edge_softness, edge_softness), 0)

    # Let visible front hair soften the lower edge only.
    hair_mask = fitted.get("occlusion_masks", {}).get("hair")
    if hair_mask is not None:
        lower_band = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(lower_band, [fold], 255, cv2.LINE_AA)
        occ = cv2.bitwise_and((hair_mask > 20).astype(np.uint8) * 255, lower_band)
        overlay = apply_alpha_occlusion(overlay, occ, strength=0.24)

    use_shadow = bool(meta.get("contact_shadow", True))
    shadowed = (
        apply_contact_shadow(image_bgr, overlay[:, :, 3], dx=1, dy=2, blur=19, opacity=0.045)
        if use_shadow
        else image_bgr
    )
    out = alpha_blend_rgba(shadowed, overlay)
    changed_pixels = int(np.count_nonzero(np.any(out != image_bgr, axis=2)))
    ys, xs = np.where(overlay[:, :, 3] > 4)
    alpha_bbox = None
    if xs.size and ys.size:
        alpha_bbox = {
            "x1": int(xs.min()),
            "y1": int(ys.min()),
            "x2": int(xs.max()),
            "y2": int(ys.max()),
        }

    return out, {
        "changed_pixels": changed_pixels,
        "used_hair_occlusion": hair_mask is not None,
        "contact_shadow": use_shadow,
        "color": meta.get("color", "#222222"),
        "fold_height_px": float(placement["fold_height"]),
        "mask_area": int(np.count_nonzero(overlay[:, :, 3] > 4)),
        "alpha_bbox": alpha_bbox,
        "shape": "beanie_dome",
    }
