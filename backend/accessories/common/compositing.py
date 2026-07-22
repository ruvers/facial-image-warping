from __future__ import annotations

import cv2
import numpy as np


def alpha_blend_rgba(
    image_bgr: np.ndarray,
    overlay_rgba: np.ndarray,
) -> np.ndarray:
    if image_bgr.shape[:2] != overlay_rgba.shape[:2]:
        raise ValueError("image and overlay sizes must match")

    base = image_bgr.astype(np.float32)
    fg = overlay_rgba[:, :, :3].astype(np.float32)
    alpha = np.clip(overlay_rgba[:, :, 3:4].astype(np.float32) / 255.0, 0.0, 1.0)
    out = fg * alpha + base * (1.0 - alpha)
    return np.clip(out, 0, 255).astype(np.uint8)


def apply_contact_shadow(
    image_bgr: np.ndarray,
    alpha: np.ndarray,
    dx: int = 1,
    dy: int = 3,
    blur: int = 21,
    opacity: float = 0.20,
) -> np.ndarray:
    if alpha is None or not np.any(alpha > 0):
        return image_bgr

    if blur % 2 == 0:
        blur += 1

    h, w = alpha.shape[:2]
    shadow = cv2.GaussianBlur(alpha.astype(np.float32) / 255.0, (blur, blur), 0)
    matrix = np.float32([[1, 0, dx], [0, 1, dy]])
    shadow = cv2.warpAffine(
        shadow,
        matrix,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    shadow = np.clip(shadow * float(opacity), 0.0, 1.0)

    out = image_bgr.astype(np.float32) * (1.0 - shadow[:, :, None])
    return np.clip(out, 0, 255).astype(np.uint8)


def apply_alpha_occlusion(
    overlay_rgba: np.ndarray,
    occlusion_mask: np.ndarray | None,
    strength: float = 0.20,
) -> np.ndarray:
    if occlusion_mask is None:
        return overlay_rgba

    if occlusion_mask.shape[:2] != overlay_rgba.shape[:2]:
        return overlay_rgba

    result = overlay_rgba.copy()
    alpha = result[:, :, 3].astype(np.float32)
    occ = (occlusion_mask > 20) & (alpha > 0)
    alpha[occ] *= float(max(0.0, min(1.0, strength)))
    result[:, :, 3] = np.clip(alpha, 0, 255).astype(np.uint8)
    return result
