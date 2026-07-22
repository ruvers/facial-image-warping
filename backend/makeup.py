"""
FaceWarp Lab - Landmark-based Makeup Application.
Applies lipstick, blush, and eye shadow using MediaPipe face landmarks.
No external API needed - runs fully offline with OpenCV.
"""
from __future__ import annotations
import cv2
import numpy as np
from typing import List, Optional


# ── MediaPipe 468 landmark indices for facial regions ──

# Lips (outer contour)
UPPER_LIP_OUTER = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291]
LOWER_LIP_OUTER = [291, 375, 321, 405, 314, 17, 84, 181, 91, 146, 61]

# Lips (inner contour)
UPPER_LIP_INNER = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308]
LOWER_LIP_INNER = [308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 78]

# Left cheek region (approximate)
LEFT_CHEEK = [50, 101, 118, 119, 120, 100, 142, 36, 205, 187, 123, 116, 117]

# Right cheek region (approximate)
RIGHT_CHEEK = [280, 330, 347, 348, 349, 329, 371, 266, 425, 411, 352, 345, 346]

# Left eye shadow region (upper eyelid area)
LEFT_EYE_SHADOW = [33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7]

# Right eye shadow region (upper eyelid area)
RIGHT_EYE_SHADOW = [263, 466, 388, 387, 386, 385, 384, 398, 362, 382, 381, 380, 374, 373, 390, 249]


def _landmarks_to_pixel_points(landmarks: list, w: int, h: int) -> np.ndarray:
    """Convert landmark list to Nx2 pixel coordinates."""
    pts = []
    for lm in landmarks:
        x = lm['x'] if isinstance(lm, dict) else getattr(lm, 'x')
        y = lm['y'] if isinstance(lm, dict) else getattr(lm, 'y')
        # Check if normalized (0-1) or pixel coords
        if float(x) <= 1.5:
            pts.append([int(x * w), int(y * h)])
        else:
            pts.append([int(x), int(y)])
    return np.array(pts)


def _get_region_mask(shape: tuple, all_pts: np.ndarray, indices: List[int],
                     blur_size: int = 15) -> np.ndarray:
    """Create a smooth mask for a facial region defined by landmark indices."""
    mask = np.zeros(shape[:2], dtype=np.float32)
    region_pts = all_pts[indices].reshape((-1, 1, 2)).astype(np.int32)
    cv2.fillPoly(mask, [region_pts], 1.0)
    if blur_size > 0:
        mask = cv2.GaussianBlur(mask, (blur_size, blur_size), 0)
    return mask


def _apply_color_overlay(image: np.ndarray, mask: np.ndarray,
                         color: tuple, alpha: float) -> np.ndarray:
    """Apply a colored overlay on image through a mask with given alpha."""
    overlay = np.full_like(image, color, dtype=np.uint8)
    mask_3ch = np.stack([mask] * 3, axis=-1)
    blended = image.astype(np.float32) * (1 - mask_3ch * alpha) + \
              overlay.astype(np.float32) * (mask_3ch * alpha)
    return np.clip(blended, 0, 255).astype(np.uint8)


def apply_makeup(image_rgb: np.ndarray, landmarks: list,
                 lipstick: bool = True,
                 blush: bool = True,
                 eye_shadow: bool = True,
                 lip_color: tuple = (200, 30, 50),
                 blush_color: tuple = (210, 120, 130),
                 shadow_color: tuple = (140, 90, 160),
                 intensity: float = 0.6) -> np.ndarray:
    """
    Apply makeup to a face image using landmark positions.
    
    Args:
        image_rgb: Input image in RGB format (uint8).
        landmarks: MediaPipe 468 face landmarks.
        lipstick: Whether to apply lipstick.
        blush: Whether to apply blush.
        eye_shadow: Whether to apply eye shadow.
        lip_color: RGB color for lipstick.
        blush_color: RGB color for blush.
        shadow_color: RGB color for eye shadow.
        intensity: Overall makeup intensity (0.0 - 1.0).
        
    Returns:
        Image with makeup applied (RGB uint8).
    """
    h, w = image_rgb.shape[:2]
    all_pts = _landmarks_to_pixel_points(landmarks, w, h)
    result = image_rgb.copy()

    if len(all_pts) < 468:
        print(f"[Makeup] Not enough landmarks ({len(all_pts)}), skipping.")
        return result

    # ── Lipstick ──
    if lipstick:
        # Outer lip mask
        lip_mask = np.zeros((h, w), dtype=np.float32)
        outer_pts = all_pts[UPPER_LIP_OUTER + LOWER_LIP_OUTER[1:]].reshape((-1, 1, 2)).astype(np.int32)
        cv2.fillPoly(lip_mask, [outer_pts], 1.0)
        
        # Subtract inner lip (mouth opening) for more natural look
        inner_pts = all_pts[UPPER_LIP_INNER + LOWER_LIP_INNER[1:]].reshape((-1, 1, 2)).astype(np.int32)
        inner_mask = np.zeros((h, w), dtype=np.float32)
        cv2.fillPoly(inner_mask, [inner_pts], 1.0)
        lip_mask = np.clip(lip_mask - inner_mask * 0.3, 0, 1)
        
        # Smooth edges
        lip_mask = cv2.GaussianBlur(lip_mask, (7, 7), 0)
        
        # Apply with controlled alpha
        lip_alpha = 0.45 * intensity
        result = _apply_color_overlay(result, lip_mask, lip_color, lip_alpha)

    # ── Blush ──
    if blush:
        left_blush_mask = _get_region_mask((h, w), all_pts, LEFT_CHEEK, blur_size=31)
        right_blush_mask = _get_region_mask((h, w), all_pts, RIGHT_CHEEK, blur_size=31)
        blush_mask = np.clip(left_blush_mask + right_blush_mask, 0, 1)
        
        blush_alpha = 0.25 * intensity
        result = _apply_color_overlay(result, blush_mask, blush_color, blush_alpha)

    # ── Eye Shadow ──
    if eye_shadow:
        left_shadow = _get_region_mask((h, w), all_pts, LEFT_EYE_SHADOW, blur_size=11)
        right_shadow = _get_region_mask((h, w), all_pts, RIGHT_EYE_SHADOW, blur_size=11)
        shadow_mask = np.clip(left_shadow + right_shadow, 0, 1)
        
        shadow_alpha = 0.30 * intensity
        result = _apply_color_overlay(result, shadow_mask, shadow_color, shadow_alpha)

    print(f"[Makeup] Applied successfully (intensity={intensity:.2f})")
    return result
