from __future__ import annotations

import cv2
import numpy as np

from backend.accessories.base import alpha_blend_rgba


def _draw_earring(
    overlay: np.ndarray,
    anchor: tuple[float, float],
    size: int,
    color_bgr: tuple[int, int, int],
    opacity: int,
    style: str,
):
    x, y = int(anchor[0]), int(anchor[1])

    b, g, r = color_bgr
    color = (b, g, r, opacity)

    # Stud
    cv2.circle(
        overlay,
        (x, y),
        max(2, size // 5),
        color,
        -1,
        cv2.LINE_AA,
    )

    # Connector
    drop_y = y + int(size * 0.55)

    cv2.line(
        overlay,
        (x, y + size // 6),
        (x, drop_y),
        color,
        max(1, size // 12),
        cv2.LINE_AA,
    )

    if style == "diamond":
        pts = np.array([
            [x, drop_y - size // 3],
            [x + size // 3, drop_y],
            [x, drop_y + size // 3],
            [x - size // 3, drop_y],
        ], dtype=np.int32)

        cv2.fillPoly(
            overlay,
            [pts],
            color,
            cv2.LINE_AA,
        )

        cv2.polylines(
            overlay,
            [pts],
            True,
            (255, 255, 255, min(120, opacity)),
            max(1, size // 18),
            cv2.LINE_AA,
        )

    elif style == "hoop":
        cv2.ellipse(
            overlay,
            (x, drop_y),
            (size // 3, size // 2),
            0,
            0,
            360,
            color,
            max(2, size // 10),
            cv2.LINE_AA,
        )

    else:
        # round drop
        cv2.circle(
            overlay,
            (x, drop_y),
            max(3, size // 3),
            color,
            -1,
            cv2.LINE_AA,
        )

        cv2.circle(
            overlay,
            (x - size // 10, drop_y - size // 10),
            max(1, size // 10),
            (255, 255, 255, min(100, opacity)),
            -1,
            cv2.LINE_AA,
        )


def _apply_hair_occlusion(
    overlay: np.ndarray,
    hair_mask: np.ndarray | None,
    strength: float = 0.25,
) -> np.ndarray:
    if hair_mask is None:
        return overlay

    if hair_mask.shape[:2] != overlay.shape[:2]:
        return overlay

    result = overlay.copy()
    alpha = result[:, :, 3].astype(np.float32)

    occ = (
        (hair_mask > 25)
        & (alpha > 0)
    )

    alpha[occ] *= strength

    result[:, :, 3] = np.clip(
        alpha,
        0,
        255,
    ).astype(np.uint8)

    return result


def _apply_soft_shadow(
    image_bgr: np.ndarray,
    alpha: np.ndarray,
    dx: int = 2,
    dy: int = 3,
    blur: int = 13,
    opacity: float = 0.18,
) -> np.ndarray:
    if blur % 2 == 0:
        blur += 1

    h, w = alpha.shape[:2]

    shadow = cv2.GaussianBlur(
        alpha,
        (blur, blur),
        0,
    ).astype(np.float32) / 255.0

    M = np.float32([
        [1, 0, dx],
        [0, 1, dy],
    ])

    shadow = cv2.warpAffine(
        shadow,
        M,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    shadow = np.clip(
        shadow * opacity,
        0.0,
        1.0,
    )

    out = image_bgr.astype(np.float32)
    out = out * (1.0 - shadow[:, :, None])

    return np.clip(out, 0, 255).astype(np.uint8)


def apply_earrings_v2(
    image_bgr: np.ndarray,
    ctx: dict,
    params: dict | None = None,
) -> np.ndarray:
    """
    Procedural earrings using ctx['anchors']['earrings'].

    params example:
    {
        "enabled": true,
        "side": "both",          # both | left | right
        "style": "diamond",      # diamond | hoop | round
        "color": "gold",         # gold | silver | rose
        "scale": 1.0,
        "opacity": 0.95,
        "hair_occlusion": true
    }
    """

    params = params or {}

    if image_bgr is None:
        raise ValueError("image_bgr is None")

    h, w = image_bgr.shape[:2]

    overlay = np.zeros(
        (h, w, 4),
        dtype=np.uint8,
    )

    anchors = ctx["anchors"]["earrings"]
    face_width = ctx["anchors"]["metrics"]["face_width"]

    side = params.get("side", "both")
    style = params.get("style", "diamond")
    scale = float(params.get("scale", 1.0))
    opacity = float(params.get("opacity", 0.95))

    opacity_i = int(np.clip(opacity, 0.0, 1.0) * 255)

    color_name = params.get("color", "gold")

    colors = {
        "gold": (35, 185, 235),
        "silver": (210, 210, 210),
        "rose": (120, 150, 230),
        "black": (20, 20, 24),
    }

    color_bgr = colors.get(color_name, colors["gold"])

    size = max(
        10,
        int(face_width * 0.075 * scale),
    )

    if side in ("both", "left"):
        _draw_earring(
            overlay,
            anchors["left"],
            size,
            color_bgr,
            opacity_i,
            style,
        )

    if side in ("both", "right"):
        _draw_earring(
            overlay,
            anchors["right"],
            size,
            color_bgr,
            opacity_i,
            style,
        )

    if params.get("hair_occlusion", True):
        overlay = _apply_hair_occlusion(
            overlay,
            ctx["masks"].get("hair"),
            strength=0.28,
        )

    result = _apply_soft_shadow(
        image_bgr,
        overlay[:, :, 3],
        dx=2,
        dy=3,
        blur=13,
        opacity=0.16,
    )

    result = alpha_blend_rgba(
        result,
        overlay,
    )

    return result