from __future__ import annotations

import cv2
import numpy as np


def is_ai_enhancer_available() -> bool:
    """
    Future AI model availability check.

    Later examples:
    - GFPGAN
    - CodeFormer
    - diffusion/inpainting
    - custom makeup/accessory model

    For now returns False.
    """

    return False


def apply_cv_polish(
    image_bgr: np.ndarray,
    intensity: float = 0.25,
) -> np.ndarray:
    """
    Lightweight non-AI fallback polish.

    This is not a replacement for real AI.
    It only gives a mild finished look:
    - small contrast correction
    - mild denoise
    - mild sharpening

    Safe for now because it does not change face structure.
    """

    intensity = float(np.clip(intensity, 0.0, 1.0))

    if intensity <= 0.0:
        return image_bgr

    img = image_bgr.copy()

    # Mild denoise / smoothing
    denoised = cv2.bilateralFilter(
        img,
        d=5,
        sigmaColor=25,
        sigmaSpace=25,
    )

    # Mild sharpening
    blur = cv2.GaussianBlur(
        denoised,
        (0, 0),
        1.2,
    )

    sharpened = cv2.addWeighted(
        denoised,
        1.15,
        blur,
        -0.15,
        0,
    )

    # Mild local contrast using LAB
    lab = cv2.cvtColor(
        sharpened,
        cv2.COLOR_BGR2LAB,
    )

    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=1.2,
        tileGridSize=(8, 8),
    )

    l2 = clahe.apply(l)

    contrast = cv2.merge([l2, a, b])

    contrast_bgr = cv2.cvtColor(
        contrast,
        cv2.COLOR_LAB2BGR,
    )

    result = (
        image_bgr.astype(np.float32) * (1.0 - intensity)
        + contrast_bgr.astype(np.float32) * intensity
    )

    return np.clip(result, 0, 255).astype(np.uint8)


def apply_real_ai_enhancement(
    image_bgr: np.ndarray,
    ctx: dict,
    params: dict,
) -> np.ndarray:
    """
    Future real AI enhancement entry.

    This function is intentionally a placeholder.
    When model is installed, implementation goes here.

    Expected future behavior:
    - face restoration
    - AI makeup refinement
    - accessory realism pass
    - hair boundary refinement
    - skin retouch model
    """

    raise RuntimeError("Real AI enhancer is not installed yet.")


def apply_ai_enhancement(
    image_bgr: np.ndarray,
    ctx: dict,
    params: dict | None = None,
) -> np.ndarray:
    """
    Main AI enhancement manager.

    Params example:
    {
        "enabled": true,
        "use_real_ai": false,
        "fallback_cv_polish": true,
        "intensity": 0.25
    }
    """

    params = params or {}

    if image_bgr is None:
        raise ValueError("image_bgr is None")

    use_real_ai = bool(
        params.get("use_real_ai", False)
    )

    fallback_cv_polish = bool(
        params.get("fallback_cv_polish", True)
    )

    intensity = float(
        params.get("intensity", 0.25)
    )

    # Future real model path
    if use_real_ai and is_ai_enhancer_available():
        try:
            return apply_real_ai_enhancement(
                image_bgr,
                ctx,
                params,
            )

        except Exception as e:
            print(f"[WARN] Real AI enhancement failed: {e}")

            if not fallback_cv_polish:
                return image_bgr

    # Current safe fallback
    if fallback_cv_polish:
        return apply_cv_polish(
            image_bgr,
            intensity=intensity,
        )

    return image_bgr