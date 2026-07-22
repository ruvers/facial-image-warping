"""
FaceWarp Lab - Landmark-based Glasses Overlay.
Draws realistic sunglasses on the face using MediaPipe landmarks.
Uses perspective-correct polygon shapes instead of simple ellipses.
"""
from __future__ import annotations
import cv2
import numpy as np
from typing import Tuple


def _get_pt(lm, w: int = 1, h: int = 1) -> Tuple[int, int]:
    """Extract (x, y) pixel coords from a landmark dict or object."""
    if isinstance(lm, dict):
        x, y = lm['x'], lm['y']
    else:
        x, y = getattr(lm, 'x'), getattr(lm, 'y')
    if float(x) <= 1.5:
        return int(x * w), int(y * h)
    return int(x), int(y)


def _midpoint(p1: Tuple[int, int], p2: Tuple[int, int]) -> Tuple[int, int]:
    return ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)


def _lerp(p1: Tuple[int, int], p2: Tuple[int, int], t: float) -> Tuple[int, int]:
    return (int(p1[0] + (p2[0] - p1[0]) * t), int(p1[1] + (p2[1] - p1[1]) * t))


def apply_glasses(image_rgb: np.ndarray, landmarks: list) -> np.ndarray:
    """
    Draw realistic sunglasses on a face using landmark positions.
    
    Uses landmark-guided polygons for lens shapes that follow the actual
    eye/brow contours, producing a natural-looking overlay.
    """
    if len(landmarks) < 468:
        print("[Glasses] Not enough landmarks, skipping.")
        return image_rgb.copy()

    result = image_rgb.copy()
    h, w = result.shape[:2]

    # ── Key landmarks ──
    # Eye corners
    L_outer = _get_pt(landmarks[33], w, h)
    L_inner = _get_pt(landmarks[133], w, h)
    R_inner = _get_pt(landmarks[362], w, h)
    R_outer = _get_pt(landmarks[263], w, h)

    # Eye centers (iris)
    L_center = _get_pt(landmarks[468], w, h) if len(landmarks) > 468 else _midpoint(L_outer, L_inner)
    R_center = _get_pt(landmarks[473], w, h) if len(landmarks) > 473 else _midpoint(R_inner, R_outer)

    # Eyebrow points (top edge guide)
    L_brow_inner = _get_pt(landmarks[107], w, h)
    L_brow_mid   = _get_pt(landmarks[105], w, h)
    L_brow_outer = _get_pt(landmarks[70], w, h)
    R_brow_inner = _get_pt(landmarks[336], w, h)
    R_brow_mid   = _get_pt(landmarks[334], w, h)
    R_brow_outer = _get_pt(landmarks[300], w, h)

    # Below-eye / cheek points (bottom edge guide)
    L_below_outer = _get_pt(landmarks[143], w, h)
    L_below_mid   = _get_pt(landmarks[111], w, h)
    L_below_inner = _get_pt(landmarks[117], w, h)
    R_below_inner = _get_pt(landmarks[346], w, h)
    R_below_mid   = _get_pt(landmarks[340], w, h)
    R_below_outer = _get_pt(landmarks[372], w, h)

    # Nose bridge
    nose_bridge = _get_pt(landmarks[6], w, h)

    # Temple endpoints (for arms)
    L_temple = _get_pt(landmarks[127], w, h)
    R_temple = _get_pt(landmarks[356], w, h)

    # ── Geometry ──
    eye_dist = np.sqrt((R_center[0] - L_center[0])**2 + (R_center[1] - L_center[1])**2)
    angle = np.arctan2(R_center[1] - L_center[1], R_center[0] - L_center[0])

    # Padding to extend lens beyond eye corners
    pad_x = int(eye_dist * 0.12)
    pad_y_top = int(eye_dist * 0.06)     # above brow
    pad_y_bot = int(eye_dist * 0.08)     # below cheek

    frame_thick = max(3, int(eye_dist * 0.035))
    bridge_thick = max(2, int(eye_dist * 0.030))
    arm_thick = max(2, int(eye_dist * 0.028))

    # ── Build lens polygons using landmark-guided contours ──
    # Left lens: polygon that traces brow line on top, under-eye on bottom
    L_lens_pts = np.array([
        # Top edge: follows eyebrow line (outer → inner)
        (L_brow_outer[0] - pad_x, L_brow_outer[1] - pad_y_top),
        (L_brow_mid[0], L_brow_mid[1] - pad_y_top),
        (L_brow_inner[0], L_brow_inner[1] - pad_y_top),
        # Inner edge (toward nose)
        (L_inner[0] + int(pad_x * 0.3), L_inner[1]),
        # Bottom edge: follows under-eye line (inner → outer)
        (L_below_inner[0], L_below_inner[1] + pad_y_bot),
        (L_below_mid[0], L_below_mid[1] + pad_y_bot),
        (L_below_outer[0] - pad_x, L_below_outer[1] + pad_y_bot),
    ], dtype=np.int32)

    # Right lens
    R_lens_pts = np.array([
        (R_brow_inner[0], R_brow_inner[1] - pad_y_top),
        (R_brow_mid[0], R_brow_mid[1] - pad_y_top),
        (R_brow_outer[0] + pad_x, R_brow_outer[1] - pad_y_top),
        # Outer edge
        (R_below_outer[0] + pad_x, R_below_outer[1] + pad_y_bot),
        (R_below_mid[0], R_below_mid[1] + pad_y_bot),
        (R_below_inner[0], R_below_inner[1] + pad_y_bot),
        # Inner edge (toward nose)
        (R_inner[0] - int(pad_x * 0.3), R_inner[1]),
    ], dtype=np.int32)

    # Smooth the polygons using approxPolyDP for natural curves
    epsilon = eye_dist * 0.02
    L_lens_smooth = cv2.approxPolyDP(L_lens_pts.reshape(-1, 1, 2), epsilon, True)
    R_lens_smooth = cv2.approxPolyDP(R_lens_pts.reshape(-1, 1, 2), epsilon, True)

    # ── Draw on overlay ──
    overlay = result.copy()

    # Lens fill: dark tinted gradient
    # Create gradient mask for each lens
    lens_color_top = (25, 25, 35)     # slightly blue-tinted dark
    lens_color_bot = (10, 10, 15)     # darker at bottom

    # Left lens fill with gradient
    lens_mask_L = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(lens_mask_L, [L_lens_smooth], 255)

    lens_mask_R = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(lens_mask_R, [R_lens_smooth], 255)

    combined_lens_mask = cv2.bitwise_or(lens_mask_L, lens_mask_R)

    # Create vertical gradient for lens tint
    gradient = np.zeros((h, w, 3), dtype=np.float32)
    for row in range(h):
        t = row / h  # 0 at top, 1 at bottom
        r = lens_color_top[0] * (1 - t) + lens_color_bot[0] * t
        g = lens_color_top[1] * (1 - t) + lens_color_bot[1] * t
        b = lens_color_top[2] * (1 - t) + lens_color_bot[2] * t
        gradient[row, :] = [r, g, b]

    # Apply gradient as lens fill
    lens_fill = gradient.astype(np.uint8)

    # ── Frames ──
    frame_color = (35, 35, 40)
    # Left frame outline
    cv2.polylines(overlay, [L_lens_smooth], True, frame_color, frame_thick, cv2.LINE_AA)
    # Right frame outline
    cv2.polylines(overlay, [R_lens_smooth], True, frame_color, frame_thick, cv2.LINE_AA)

    # Bridge: smooth curve through nose bridge point
    bridge_L = (L_inner[0] + int(pad_x * 0.3), L_inner[1])
    bridge_R = (R_inner[0] - int(pad_x * 0.3), R_inner[1])
    bridge_mid = (nose_bridge[0], nose_bridge[1] - int(eye_dist * 0.02))

    # Draw bridge as a smooth Bezier-like curve through 3 points
    bridge_pts = []
    for t_val in np.linspace(0, 1, 20):
        # Quadratic Bezier
        x = (1 - t_val)**2 * bridge_L[0] + 2 * (1 - t_val) * t_val * bridge_mid[0] + t_val**2 * bridge_R[0]
        y = (1 - t_val)**2 * bridge_L[1] + 2 * (1 - t_val) * t_val * bridge_mid[1] + t_val**2 * bridge_R[1]
        bridge_pts.append([int(x), int(y)])
    bridge_pts = np.array(bridge_pts, dtype=np.int32).reshape(-1, 1, 2)
    cv2.polylines(overlay, [bridge_pts], False, frame_color, bridge_thick, cv2.LINE_AA)

    # Temple arms
    L_arm_start = (L_lens_smooth[0][0][0], L_lens_smooth[0][0][1])
    R_arm_start = (R_lens_smooth[2 if len(R_lens_smooth) > 2 else 0][0][0],
                   R_lens_smooth[2 if len(R_lens_smooth) > 2 else 0][0][1])

    # Left arm: from outer-top of left lens to temple
    cv2.line(overlay, L_arm_start, L_temple, frame_color, arm_thick, cv2.LINE_AA)
    # Right arm
    cv2.line(overlay, R_arm_start, R_temple, frame_color, arm_thick, cv2.LINE_AA)

    # ── Compositing with proper alpha ──
    # Lens regions: semi-transparent tinted
    lens_alpha_val = 0.72
    frame_alpha_val = 0.92

    # Start with frame blending
    frame_mask_full = np.zeros((h, w), dtype=np.uint8)
    cv2.polylines(frame_mask_full, [L_lens_smooth], True, 255, frame_thick + 2, cv2.LINE_AA)
    cv2.polylines(frame_mask_full, [R_lens_smooth], True, 255, frame_thick + 2, cv2.LINE_AA)
    cv2.polylines(frame_mask_full, [bridge_pts], False, 255, bridge_thick + 2, cv2.LINE_AA)
    cv2.line(frame_mask_full, L_arm_start, L_temple, 255, arm_thick + 2, cv2.LINE_AA)
    cv2.line(frame_mask_full, R_arm_start, R_temple, 255, arm_thick + 2, cv2.LINE_AA)

    # Build final composite
    result_f = result.astype(np.float32)

    # 1. Apply lens tint
    lens_mask_f = combined_lens_mask.astype(np.float32) / 255.0
    lens_3ch = np.stack([lens_mask_f] * 3, axis=-1) * lens_alpha_val
    result_f = result_f * (1.0 - lens_3ch) + lens_fill.astype(np.float32) * lens_3ch

    # 2. Apply frame on top
    frame_mask_f = frame_mask_full.astype(np.float32) / 255.0
    frame_3ch = np.stack([frame_mask_f] * 3, axis=-1) * frame_alpha_val
    overlay_f = overlay.astype(np.float32)
    result_f = result_f * (1.0 - frame_3ch) + overlay_f * frame_3ch

    result = np.clip(result_f, 0, 255).astype(np.uint8)

    # ── Subtle lens reflection ──
    for center in [L_center, R_center]:
        ref_x = center[0] - int(eye_dist * 0.06)
        ref_y = center[1] - int(eye_dist * 0.08)
        ref_rx = int(eye_dist * 0.05)
        ref_ry = int(eye_dist * 0.03)

        ref_overlay = result.copy()
        cv2.ellipse(ref_overlay, (ref_x, ref_y), (ref_rx, ref_ry),
                    angle * 180 / np.pi + 25, 0, 360, (55, 55, 65), -1, cv2.LINE_AA)
        # Soft blend the reflection
        ref_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.ellipse(ref_mask, (ref_x, ref_y), (ref_rx, ref_ry),
                    angle * 180 / np.pi + 25, 0, 360, 255, -1, cv2.LINE_AA)
        ref_mask = cv2.GaussianBlur(ref_mask, (11, 11), 3)
        ref_alpha = ref_mask.astype(np.float32) / 255.0 * 0.25
        ref_3ch = np.stack([ref_alpha] * 3, axis=-1)
        result = np.clip(
            result.astype(np.float32) * (1 - ref_3ch) + ref_overlay.astype(np.float32) * ref_3ch,
            0, 255
        ).astype(np.uint8)

    print(f"[Glasses] Applied successfully (eye_dist={eye_dist:.0f}px)")
    return result
