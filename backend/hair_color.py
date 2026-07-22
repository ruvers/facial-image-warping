"""
FaceWarp Lab — Hair Color Transformation Module.

Uses OpenCV GrabCut + face landmarks for robust hair segmentation,
then applies HSV-based color shifting for natural-looking hair recoloring.
"""

from __future__ import annotations

import cv2
import numpy as np
from typing import Optional, Tuple


def _extract_point(lm, w: int, h: int) -> Tuple[int, int]:
    """Extract pixel coords from a landmark dict."""
    x = lm['x'] if isinstance(lm, dict) else getattr(lm, 'x')
    y = lm['y'] if isinstance(lm, dict) else getattr(lm, 'y')
    if float(x) <= 1.5:
        return int(float(x) * w), int(float(y) * h)
    return int(float(x)), int(float(y))


def segment_hair(
    image_rgb: np.ndarray,
    landmarks: Optional[list] = None,
) -> np.ndarray:
    """
    Return a float32 hair mask [0..1] using GrabCut + face landmarks.
    """
    h, w = image_rgb.shape[:2]

    if not landmarks or len(landmarks) < 300:
        return np.zeros((h, w), dtype=np.float32)

    FACE_OVAL = [
        10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
        397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
        172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109, 10
    ]
    FOREHEAD = [10, 338, 297, 332, 284, 251, 21, 54, 103, 67, 109]

    face_pts = []
    for idx in FACE_OVAL:
        if idx < len(landmarks):
            face_pts.append(_extract_point(landmarks[idx], w, h))

    forehead_pts = []
    for idx in FOREHEAD:
        if idx < len(landmarks):
            forehead_pts.append(_extract_point(landmarks[idx], w, h))

    if len(face_pts) < 5 or len(forehead_pts) < 3:
        return np.zeros((h, w), dtype=np.float32)

    forehead_y = min(p[1] for p in forehead_pts)
    chin_y = _extract_point(landmarks[152], w, h)[1]
    left_x = _extract_point(landmarks[234], w, h)[0]
    right_x = _extract_point(landmarks[454], w, h)[0]
    face_height = max(chin_y - forehead_y, 50)
    face_width = max(right_x - left_x, 50)
    face_cx = (left_x + right_x) // 2

    # Face oval mask
    face_mask = np.zeros((h, w), dtype=np.uint8)
    face_poly = np.array(face_pts, dtype=np.int32)
    cv2.fillPoly(face_mask, [face_poly], 255)
    dil_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    face_mask = cv2.dilate(face_mask, dil_k, iterations=1)

    # GrabCut mask
    gc_mask = np.full((h, w), cv2.GC_PR_BGD, dtype=np.uint8)

    # Definite BG: edges + bottom
    border = 5
    gc_mask[:border, :] = cv2.GC_BGD
    gc_mask[-border:, :] = cv2.GC_BGD
    gc_mask[:, :border] = cv2.GC_BGD
    gc_mask[:, -border:] = cv2.GC_BGD
    gc_mask[int(h * 0.75):, :] = cv2.GC_BGD

    # Definite FG: face oval (eroded)
    face_fg = cv2.erode(face_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)), iterations=2)
    gc_mask[face_fg > 0] = cv2.GC_FGD

    # Probable FG: head region above/around face
    prob_fg = np.zeros((h, w), dtype=np.uint8)
    ell_cy = forehead_y - int(face_height * 0.1)
    cv2.ellipse(prob_fg, (face_cx, ell_cy),
                (int(face_width * 1.2), int(face_height * 1.0)),
                0, 0, 360, 255, -1)
    top_x1 = max(0, face_cx - int(face_width * 0.9))
    top_x2 = min(w, face_cx + int(face_width * 0.9))
    cv2.rectangle(prob_fg, (top_x1, 0), (top_x2, forehead_y + 10), 255, -1)

    side_w = int(face_width * 0.4)
    eye_y = _extract_point(landmarks[33], w, h)[1]
    mouth_y = _extract_point(landmarks[13], w, h)[1]
    cv2.rectangle(prob_fg, (max(0, left_x - side_w), forehead_y), (left_x, mouth_y), 255, -1)
    cv2.rectangle(prob_fg, (right_x, forehead_y), (min(w, right_x + side_w), mouth_y), 255, -1)

    prob_region = (prob_fg > 0) & (gc_mask != cv2.GC_FGD) & (gc_mask != cv2.GC_BGD)
    gc_mask[prob_region] = cv2.GC_PR_FGD

    # Run GrabCut
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    bgd_model = np.zeros((1, 65), dtype=np.float64)
    fgd_model = np.zeros((1, 65), dtype=np.float64)

    try:
        cv2.grabCut(image_bgr, gc_mask, None, bgd_model, fgd_model, 3, cv2.GC_INIT_WITH_MASK)
    except cv2.error:
        return _fallback_mask(face_pts, face_mask, forehead_y, chin_y, face_cx, face_width, face_height, w, h)

    fg_mask = np.where((gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)

    # Subtract face → hair
    hair_mask = cv2.subtract(fg_mask, face_mask)
    hair_mask[min(chin_y + int(face_height * 0.08), h):, :] = 0

    # Cleanup
    hair_mask = cv2.morphologyEx(hair_mask, cv2.MORPH_CLOSE,
                                 cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13)), iterations=2)
    hair_mask = cv2.morphologyEx(hair_mask, cv2.MORPH_OPEN,
                                 cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)

    hair_float = hair_mask.astype(np.float32) / 255.0
    hair_float = cv2.GaussianBlur(hair_float, (21, 21), 6)
    return np.clip(hair_float, 0, 1)


def _fallback_mask(face_pts, face_mask, forehead_y, chin_y, face_cx, face_width, face_height, w, h):
    """Geometric fallback if GrabCut fails."""
    region = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(region, (face_cx, forehead_y - int(face_height * 0.15)),
                (int(face_width * 1.1), int(face_height * 0.9)), 0, 0, 360, 255, -1)
    cv2.rectangle(region, (max(0, face_cx - int(face_width * 0.8)), 0),
                  (min(w, face_cx + int(face_width * 0.8)), forehead_y), 255, -1)
    hair = cv2.subtract(region, face_mask)
    hair[chin_y:, :] = 0
    hair_float = hair.astype(np.float32) / 255.0
    return cv2.GaussianBlur(hair_float, (21, 21), 6)


# ── Color Presets ──

HAIR_COLOR_PRESETS = {
    "blonde":     (25, 180, 200),
    "platinum":   (22, 40, 235),
    "red":        (5, 220, 175),
    "auburn":     (12, 190, 150),
    "brown":      (15, 140, 110),
    "dark_brown": (12, 110, 65),
    "black":      (0, 25, 25),
    "blue":       (105, 210, 155),
    "pink":       (170, 175, 200),
    "purple":     (138, 200, 155),
    "green":      (70, 200, 145),
    "silver":     (0, 12, 195),
}


def apply_hair_color(
    image_rgb: np.ndarray,
    landmarks: Optional[list],
    target_color: str = "blonde",
    intensity: float = 0.7,
    custom_hue: Optional[int] = None,
    custom_saturation: Optional[int] = None,
) -> np.ndarray:
    """Change hair color in the image."""
    if intensity <= 0:
        return image_rgb.copy()
    intensity = min(intensity, 1.0)

    hair_mask = segment_hair(image_rgb, landmarks)

    hair_area = int(np.sum(hair_mask > 0.3))
    total_area = hair_mask.shape[0] * hair_mask.shape[1]
    print(f"[HairColor] Hair mask coverage: {hair_area}/{total_area} ({100*hair_area/total_area:.1f}%)")

    if hair_area < total_area * 0.005:
        print("[HairColor] Warning: Too little hair detected. Skipping.")
        return image_rgb.copy()

    # Convert to HSV
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)

    # Target color
    if custom_hue is not None:
        target_h = float(custom_hue)
        target_s = float(custom_saturation if custom_saturation is not None else 180)
        target_v = None
    elif target_color in HAIR_COLOR_PRESETS:
        target_h, target_s, target_v = [float(v) for v in HAIR_COLOR_PRESETS[target_color]]
    else:
        target_h, target_s, target_v = [float(v) for v in HAIR_COLOR_PRESETS["blonde"]]

    # Build recolored HSV
    recolored_hsv = hsv.copy()
    recolored_hsv[:, :, 0] = target_h
    recolored_hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 0.2 + target_s * 0.8, 0, 255)

    if target_v is not None:
        orig_val = hsv[:, :, 2]
        hair_pixels = hair_mask > 0.3
        mean_val = float(np.mean(orig_val[hair_pixels])) if np.any(hair_pixels) else 128.0
        val_ratio = np.clip(target_v / (mean_val + 1e-6), 0.3, 2.5)
        recolored_hsv[:, :, 2] = np.clip(orig_val * val_ratio * 0.5 + orig_val * 0.5, 0, 255)

    # Convert back
    recolored_bgr = cv2.cvtColor(np.clip(recolored_hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)
    recolored_rgb = cv2.cvtColor(recolored_bgr, cv2.COLOR_BGR2RGB)

    # Blend
    mask_3ch = (hair_mask * intensity)[:, :, np.newaxis]
    result = recolored_rgb.astype(np.float32) * mask_3ch + image_rgb.astype(np.float32) * (1.0 - mask_3ch)
    return np.clip(result, 0, 255).astype(np.uint8)
