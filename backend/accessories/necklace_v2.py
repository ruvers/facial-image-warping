from __future__ import annotations

import cv2
import numpy as np

from backend.accessories.base import alpha_blend_rgba


def _soft_shadow(
    image_bgr: np.ndarray,
    alpha: np.ndarray,
    dx: int = 2,
    dy: int = 4,
    blur: int = 19,
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


def _draw_chain(
    overlay: np.ndarray,
    center: tuple[float, float],
    width: float,
    color_bgr: tuple[int, int, int],
    opacity: int,
    thickness: int,
    sag_ratio: float,
):
    cx, cy = center
    b, g, r = color_bgr
    color = (b, g, r, opacity)

    half = width / 2.0
    sag = width * sag_ratio

    points = []

    for i in range(80):
        t = i / 79.0
        u = (t - 0.5) * 2.0

        x = cx + u * half
        y = cy + sag * (1.0 - u * u)

        points.append([int(x), int(y)])

    pts = np.array(points, dtype=np.int32)

    cv2.polylines(
        overlay,
        [pts],
        False,
        color,
        thickness,
        cv2.LINE_AA,
    )

    # small chain beads
    step = max(4, len(points) // 18)

    for p in points[::step]:
        cv2.circle(
            overlay,
            tuple(p),
            max(1, thickness),
            color,
            -1,
            cv2.LINE_AA,
        )

    return points


def _draw_pendant(
    overlay: np.ndarray,
    center: tuple[float, float],
    width: float,
    color_bgr: tuple[int, int, int],
    opacity: int,
    style: str,
):
    cx, cy = int(center[0]), int(center[1])

    b, g, r = color_bgr
    color = (b, g, r, opacity)

    size = max(8, int(width * 0.075))

    if style == "diamond":
        pts = np.array([
            [cx, cy - size],
            [cx + size, cy],
            [cx, cy + size],
            [cx - size, cy],
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
            max(1, size // 6),
            cv2.LINE_AA,
        )

    elif style == "round":
        cv2.circle(
            overlay,
            (cx, cy),
            size,
            color,
            -1,
            cv2.LINE_AA,
        )

        cv2.circle(
            overlay,
            (cx - size // 3, cy - size // 3),
            max(1, size // 4),
            (255, 255, 255, min(120, opacity)),
            -1,
            cv2.LINE_AA,
        )

    else:
        # simple vertical charm
        cv2.ellipse(
            overlay,
            (cx, cy),
            (size // 2, size),
            0,
            0,
            360,
            color,
            -1,
            cv2.LINE_AA,
        )


def apply_necklace_v2(
    image_bgr: np.ndarray,
    ctx: dict,
    params: dict | None = None,
) -> np.ndarray:
    """
    Procedural necklace using ctx['anchors']['necklace'].

    params example:
    {
        "enabled": true,
        "style": "diamond",
        "color": "gold",
        "scale": 1.0,
        "opacity": 0.90,
        "sag": 0.22
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

    anchor = ctx["anchors"]["necklace"]

    center = anchor["center"]
    base_width = float(anchor["width"])

    scale = float(params.get("scale", 1.0))
    width = base_width * scale

    opacity = int(
        np.clip(
            float(params.get("opacity", 0.90)),
            0.0,
            1.0,
        ) * 255
    )

    color_name = params.get("color", "gold")

    colors = {
        "gold": (35, 185, 235),
        "silver": (210, 210, 210),
        "rose": (120, 150, 230),
        "black": (20, 20, 24),
    }

    color_bgr = colors.get(color_name, colors["gold"])

    thickness = max(
        2,
        int(width * 0.018),
    )

    sag_ratio = float(params.get("sag", 0.22))
    style = params.get("style", "diamond")

    points = _draw_chain(
        overlay,
        center,
        width,
        color_bgr,
        opacity,
        thickness,
        sag_ratio,
    )

    # pendant at lowest point
    if points:
        lowest = max(
            points,
            key=lambda p: p[1],
        )

        pendant_center = (
            float(lowest[0]),
            float(lowest[1] + width * 0.055),
        )

        _draw_pendant(
            overlay,
            pendant_center,
            width,
            color_bgr,
            opacity,
            style,
        )

    # Prevent necklace appearing above chin too much
    chin = anchor.get("chin")

    if chin:
        chin_y = int(chin[1])
        alpha = overlay[:, :, 3]

        cutoff = max(
            0,
            chin_y - int(width * 0.02),
        )

        alpha[:cutoff, :] = 0
        overlay[:, :, 3] = alpha

    result = _soft_shadow(
        image_bgr,
        overlay[:, :, 3],
        dx=2,
        dy=4,
        blur=19,
        opacity=0.16,
    )

    result = alpha_blend_rgba(
        result,
        overlay,
    )

    return result