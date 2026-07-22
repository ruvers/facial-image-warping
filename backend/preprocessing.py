"""
FaceWarp Lab — Image preprocessing pipeline.

Pure functions that transform a raw OpenCV image array through the
standard pipeline: BGR/GRAY/BGRA → RGB → 512×512 resize → normalize.
No endpoint logic; designed to be imported by main.py.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import cv2
import numpy as np


# ── Colour-space conversion ──────────────────────────────────────────────────


def ensure_rgb(image: np.ndarray) -> np.ndarray:
    """Convert any OpenCV image to RGB uint8 (H, W, 3).

    Handles grayscale (2-D), BGR (3-ch), and BGRA (4-ch) inputs.
    Raises ValueError for None, empty, or unsupported layouts.
    """
    if image is None:
        raise ValueError("Input image is None.")
    if image.size == 0:
        raise ValueError("Input image is empty (zero elements).")

    # Ensure uint8 first
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)

    if image.ndim == 2:
        # Grayscale
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    if image.ndim == 3:
        channels = image.shape[2]
        if channels == 3:
            # OpenCV default is BGR
            return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if channels == 4:
            # BGRA
            return cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        if channels == 1:
            # Single-channel stored as (H, W, 1)
            return cv2.cvtColor(image[:, :, 0], cv2.COLOR_GRAY2RGB)

    raise ValueError(
        f"Unsupported image layout: ndim={image.ndim}, "
        f"shape={image.shape}, dtype={image.dtype}."
    )


# ── Resize ────────────────────────────────────────────────────────────────────

# resize with letterbox + Padding metadata Otherwise it breake pictures aspect ratio

def resize_to_target(image: np.ndarray, size: int = 512) -> tuple[np.ndarray, dict]:
    """Resize image to (size × size) with letterbox padding.

    Aspect ratio is preserved. Empty areas are filled with black.
    Returns (resized_image, padding_info).
    """
    if size <= 0:
        raise ValueError(f"Target size must be positive, got {size}.")

    h, w = image.shape[:2]
    scale = size / max(h, w)
    new_w = int(w * scale)
    new_h = int(h * scale)

    if h > size or w > size:
        interpolation = cv2.INTER_AREA
    else:
        interpolation = cv2.INTER_LANCZOS4

    resized = cv2.resize(image, (new_w, new_h), interpolation=interpolation)

    pad_top    = (size - new_h) // 2
    pad_bottom = size - new_h - pad_top
    pad_left   = (size - new_w) // 2
    pad_right  = size - new_w - pad_left

    letterboxed = cv2.copyMakeBorder(
        resized, pad_top, pad_bottom, pad_left, pad_right,
        cv2.BORDER_CONSTANT, value=(0, 0, 0)
    )

    padding_info = {
        "scale": float(scale),
        "pad_top": pad_top,
        "pad_bottom": pad_bottom,
        "pad_left": pad_left,
        "pad_right": pad_right,
        "letterbox": True,
    }

    return letterboxed, padding_info


# ── Normalisation ─────────────────────────────────────────────────────────────


def normalize_rgb(image: np.ndarray) -> np.ndarray:
    """Normalise uint8 image to float32 in [0.0, 1.0]."""
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    return image.astype(np.float32) / 255.0


# ── Grayscale (float32, legacy) ───────────────────────────────────────────────


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert an RGB image to single-channel grayscale float32 (H, W, 1).

    Input must be RGB.  Output range is [0.0, 1.0].
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    if gray.dtype != np.float32:
        gray = gray.astype(np.float32) / 255.0

    return gray[:, :, np.newaxis]


# ── Grayscale (uint8, FFT-ready) ──────────────────────────────────────────────


def to_grayscale_uint8(image: np.ndarray) -> np.ndarray:
    """Convert a resized RGB uint8 image to grayscale uint8 (H, W, 1)
    with histogram equalization applied for contrast normalization.

    Steps (per SDD §4.2):
      1. RGB → single-channel GRAY (cv2.COLOR_RGB2GRAY)
      2. Contrast normalization via cv2.equalizeHist
      3. Shape expanded to (H, W, 1) for consistent array handling

    Input:  RGB uint8 array, shape (H, W, 3)
    Output: uint8 array,     shape (H, W, 1), values in [0, 255]

    The output is suitable for direct use in numpy.fft.fft2() and other
    frequency-domain operations that expect a 2-D uint8/float array.
    Pass output[:, :, 0] to numpy.fft.fft2() to strip the channel axis.
    """
    if image is None:
        raise ValueError("Input image is None.")
    if image.size == 0:
        raise ValueError("Input image is empty (zero elements).")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(
            f"Expected RGB image with shape (H, W, 3), "
            f"got shape={image.shape}."
        )
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)

    # Step 1: colour → luminance
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)  # (H, W), uint8

    # Step 2: histogram equalization for contrast normalization
    gray_eq = cv2.equalizeHist(gray)               # (H, W), uint8

    # Step 3: restore channel axis so callers get a consistent (H, W, 1) array
    return gray_eq[:, :, np.newaxis]


# ── BGR helper (for cv2.imwrite) ──────────────────────────────────────────────


def image_uint8_rgb_to_bgr(image: np.ndarray) -> np.ndarray:
    """Convert RGB uint8 image to BGR uint8 for OpenCV file I/O."""
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


# ── Metadata builder ─────────────────────────────────────────────────────────


def build_preprocess_metadata(
    original_image: np.ndarray,
    processed_image_uint8: np.ndarray,
    channels: int = 3,
) -> Dict[str, int]:
    """Build the ``metadata`` dict for ProcessResponse.

    Extracts width/height from the original and processed arrays.
    """
    orig_h, orig_w = original_image.shape[:2]
    proc_h, proc_w = processed_image_uint8.shape[:2]
    return {
        "original_width": int(orig_w),
        "original_height": int(orig_h),
        "processed_width": int(proc_w),
        "processed_height": int(proc_h),
        "channels": channels,
    }


# ── Full pipeline ─────────────────────────────────────────────────────────────

def preprocess_pipeline(
    image: np.ndarray,
    target_size: int = 512,
) -> Dict[str, Any]:
    rgb = ensure_rgb(image)
    resized, padding_info = resize_to_target(rgb, size=target_size)
    normalized = normalize_rgb(resized)
    grayscale = to_grayscale_uint8(resized)

    return {
        "processed_image": normalized,
        "processed_image_uint8": resized,
        "grayscale_image": grayscale,
        "preprocess_info": {
            "target_size": target_size,
            "resized_width": target_size,
            "resized_height": target_size,
            "color_space": "RGB",
            "normalized": True,
            "normalization_range": [0.0, 1.0],
            "grayscale_generated": True,
            "grayscale_dtype": "uint8",
            "histogram_equalized": True,
            **padding_info,
        },
    }