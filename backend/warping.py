"""
FaceWarp Lab — Natural Anatomical Warping.
Fixes "Joker/Grimace" effect by applying strict U-curve lip targeting,
reduced Gaussian radii, and vertical-dominant lifting.
"""

from __future__ import annotations
import cv2
import numpy as np
import copy
from scipy.spatial import Delaunay
from typing import List, Tuple

# ── Helper Functions ──────────────────────────────────────────────────────────

def landmarks_to_points(landmarks: list, width: int, height: int) -> List[List[float]]:
    if not landmarks: return []
    first_lm = landmarks[0]
    first_x = first_lm['x'] if isinstance(first_lm, dict) else getattr(first_lm, 'x')
    is_normalized = float(first_x) <= 1.5 
    
    pts = []
    for lm in landmarks:
        x = lm['x'] if isinstance(lm, dict) else getattr(lm, 'x')
        y = lm['y'] if isinstance(lm, dict) else getattr(lm, 'y')
        if is_normalized:
            pts.append([x * width, y * height])
        else:
            pts.append([x, y])
    return pts

def get_face_properties(points: List[List[float]]) -> Tuple[float, np.ndarray]:
    try:
        left_cheek = np.array(points[234])
        right_cheek = np.array(points[454])
        scale = float(np.linalg.norm(left_cheek - right_cheek))
        nose_tip = np.array(points[1])
        return scale, nose_tip
    except IndexError:
        return 100.0, np.array([0.0, 0.0])


# ── Core: Gaussian-Weighted Deformation Field ─────────────────────────────────

def apply_gaussian_field(src_points: List[List[float]], dst_points: List[List[float]], 
                         center_idx: int, vec_x: float, vec_y: float, radius: float):
    if center_idx >= len(src_points): return
    
    center = np.array(src_points[center_idx])
    sigma = radius / 2.0
    denom = 2 * (sigma ** 2)

    for i, pt in enumerate(src_points):
        dist_sq = (pt[0] - center[0])**2 + (pt[1] - center[1])**2
        if dist_sq < radius**2:
            weight = np.exp(-dist_sq / denom)
            dst_points[i][0] += vec_x * weight
            dst_points[i][1] += vec_y * weight

# ── Anatomik Mimikler (Facial Expressions) ────────────────────────────────────

def shift_for_duchenne_smile(src_points: List[List[float]], dst_points: List[List[float]], intensity: float, scale: float):
    """
    Natural Duchenne smile.

    Anatomy reference:
    - Zygomaticus major pulls mouth corners UP and slightly OUT
    - Orbicularis oculi contracts (Duchenne marker — crow's-feet)
    - Upper lip lifts slightly at cupid's bow
    - Nasolabial fold deepens
    - Jaw and chin must NOT move
    """
    # ── Pin jaw / chin so they never shift ──
    JAW_PINS = [
        132, 172, 136, 150, 149, 176, 148, 152,  # left jaw + chin
        361, 397, 365, 379, 378, 400, 377,         # right jaw
        58, 215, 170, 169, 135,                     # left lower jaw
        288, 435, 395, 394, 364,                    # right lower jaw
    ]
    saved_jaw = {}
    for idx in JAW_PINS:
        if idx < len(dst_points):
            saved_jaw[idx] = [dst_points[idx][0], dst_points[idx][1]]

    # 1. Mouth corners — primarily UP, minimal outward
    lift   = intensity * (scale * 0.085)
    widen  = intensity * (scale * 0.025)
    corner_r = scale * 0.18

    apply_gaussian_field(src_points, dst_points, 61,  -widen, -lift, corner_r)   # left corner
    apply_gaussian_field(src_points, dst_points, 291,  widen, -lift, corner_r)   # right corner

    # 2. U-curve — lift mid-lip slightly so mouth isn't a flat line
    mid_lift = lift * 0.45
    mid_r    = scale * 0.12
    apply_gaussian_field(src_points, dst_points, 13, 0, -mid_lift, mid_r)  # upper lip center
    apply_gaussian_field(src_points, dst_points, 14, 0,  mid_lift * 0.25, mid_r)  # lower lip drops a tiny bit

    # 3. Cheeks — gentle lift only (no outward push)
    cheek_lift = intensity * (scale * 0.045)
    cheek_r    = scale * 0.20
    apply_gaussian_field(src_points, dst_points, 205, 0, -cheek_lift, cheek_r)
    apply_gaussian_field(src_points, dst_points, 425, 0, -cheek_lift, cheek_r)

    # 4. Duchenne eye crinkle — very subtle
    eye_lift = intensity * (scale * 0.020)
    eye_r    = scale * 0.12
    apply_gaussian_field(src_points, dst_points, 145, 0, -eye_lift, eye_r)
    apply_gaussian_field(src_points, dst_points, 374, 0, -eye_lift, eye_r)

    # 5. Nasolabial fold — slight inward pull for realism
    naso_shift = intensity * (scale * 0.015)
    naso_r     = scale * 0.10
    apply_gaussian_field(src_points, dst_points, 92,  naso_shift, 0, naso_r)   # left
    apply_gaussian_field(src_points, dst_points, 322, -naso_shift, 0, naso_r)  # right

    # ── Restore jaw / chin ──
    for idx, pos in saved_jaw.items():
        if idx < len(dst_points):
            dst_points[idx][0] = pos[0]
            dst_points[idx][1] = pos[1]


def shift_for_face_slimming(src_points: List[List[float]], dst_points: List[List[float]], intensity: float, scale: float, nose_tip: np.ndarray):
    jaw_left = [132, 172, 136, 150, 149]
    jaw_right = [361, 397, 365, 379, 378]
    
    slim_strength = intensity * (scale * 0.035)
    radius = scale * 0.20 # Etki alanını dar tuttuk

    for idx in jaw_left:
        apply_gaussian_field(src_points, dst_points, idx, slim_strength, -slim_strength * 0.1, radius)
        
    for idx in jaw_right:
        apply_gaussian_field(src_points, dst_points, idx, -slim_strength, -slim_strength * 0.1, radius)

    apply_gaussian_field(src_points, dst_points, 152, 0, -slim_strength * 0.5, scale * 0.25)


def shift_for_eyebrow(src_points: List[List[float]], dst_points: List[List[float]], intensity: float, scale: float):
    """
    Anatomically correct eyebrow raise/lower.
    
    Key improvements over the old version:
    1. Uses ALL MediaPipe eyebrow landmark indices (not just 2 points)
    2. Much smaller Gaussian radius to avoid bleeding into the eye area
    3. Explicitly PINS all eye contour and iris points so they cannot move
    4. Graduated intensity: inner brow moves more, outer brow moves less
    """
    # ── MediaPipe 468 landmark indices ──
    # Left eyebrow (inner → outer)
    LEFT_BROW  = [107, 66, 105, 63, 70,   # upper edge
                  336, 296, 334, 293, 300] # lower edge (near eye)
    # Right eyebrow (inner → outer) 
    RIGHT_BROW = [336, 296, 334, 293, 300,  # upper edge
                  107, 66, 105, 63, 70]     # lower edge

    # Correct MediaPipe eyebrow indices:
    LEFT_BROW_UPPER  = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]
    LEFT_BROW_LOWER  = [156, 143, 111, 117, 118, 119, 120, 121, 128, 245]
    RIGHT_BROW_UPPER = [300, 293, 334, 296, 336, 285, 295, 282, 283, 276]
    RIGHT_BROW_LOWER = [383, 372, 340, 346, 347, 348, 349, 350, 357, 465]

    # All eye contour + iris points that MUST NOT move
    LEFT_EYE_PINS = [
        # Upper eyelid
        33, 7, 163, 144, 145, 153, 154, 155, 133,
        # Lower eyelid  
        173, 157, 158, 159, 160, 161, 246,
        # Eye contour
        130, 25, 110, 24, 23, 22, 26, 112, 243,
        # Iris
        468, 469, 470, 471, 472,
    ]
    RIGHT_EYE_PINS = [
        # Upper eyelid
        263, 249, 390, 373, 374, 380, 381, 382, 362,
        # Lower eyelid
        398, 384, 385, 386, 387, 388, 466,
        # Eye contour
        359, 255, 339, 254, 253, 252, 256, 341, 463,
        # Iris
        473, 474, 475, 476, 477,
    ]

    ALL_EYE_PINS = set(LEFT_EYE_PINS + RIGHT_EYE_PINS)

    # ── Step 1: Save the original positions of ALL eye points ──
    saved_eye_positions = {}
    for idx in ALL_EYE_PINS:
        if idx < len(dst_points):
            saved_eye_positions[idx] = [dst_points[idx][0], dst_points[idx][1]]

    # ── Step 2: Apply eyebrow shifts with small, targeted radius ──
    brow_radius = scale * 0.10   # Small radius — only affects brow area
    shift_y = intensity * (scale * 0.04)

    # Graduated intensity: inner brow lifts more, outer brow lifts less
    inner_weight = 1.0
    mid_weight   = 0.85
    outer_weight = 0.6

    # Left brow - upper edge (main lift targets)
    for i, idx in enumerate(LEFT_BROW_UPPER):
        w = inner_weight if i < 3 else (mid_weight if i < 6 else outer_weight)
        if idx < len(src_points):
            apply_gaussian_field(src_points, dst_points, idx, 0, -shift_y * w, brow_radius)

    # Right brow - upper edge (main lift targets)
    for i, idx in enumerate(RIGHT_BROW_UPPER):
        w = inner_weight if i < 3 else (mid_weight if i < 6 else outer_weight)
        if idx < len(src_points):
            apply_gaussian_field(src_points, dst_points, idx, 0, -shift_y * w, brow_radius)

    # Lower brow edges move with reduced intensity (50%) to maintain brow thickness
    lower_factor = 0.5
    for idx in LEFT_BROW_LOWER:
        if idx < len(src_points):
            apply_gaussian_field(src_points, dst_points, idx, 0, -shift_y * lower_factor, brow_radius * 0.7)

    for idx in RIGHT_BROW_LOWER:
        if idx < len(src_points):
            apply_gaussian_field(src_points, dst_points, idx, 0, -shift_y * lower_factor, brow_radius * 0.7)

    # ── Step 3: RESTORE all eye points to their original positions ──
    # This is the critical fix: even if the Gaussian field leaked into
    # the eye area, we force all eye points back to where they were.
    for idx, pos in saved_eye_positions.items():
        if idx < len(dst_points):
            dst_points[idx][0] = pos[0]
            dst_points[idx][1] = pos[1]

def shift_for_lip_widening(src_points: List[List[float]], dst_points: List[List[float]], intensity: float, scale: float):
    shift_x = intensity * (scale * 0.15)
    apply_gaussian_field(src_points, dst_points, 61, -shift_x, 0, scale * 0.35)
    apply_gaussian_field(src_points, dst_points, 291, shift_x, 0, scale * 0.35)


# ── Dense Displacement Field Warping (seamless, no triangle artifacts) ────────

def warp_image(image: np.ndarray, src_points: List[Tuple[float, float]], dst_points: List[Tuple[float, float]]) -> np.ndarray:
    """
    Warp image using a dense displacement field.
    Only points that actually moved are fed to the RBF interpolator,
    plus sparse anchor points to pin non-deformed regions.
    """
    from scipy.interpolate import RBFInterpolator

    h, w = image.shape[:2]
    src_arr = np.array(src_points, dtype=np.float64)
    dst_arr = np.array(dst_points, dtype=np.float64)

    # Displacement at each control point
    all_dx = dst_arr[:, 0] - src_arr[:, 0]
    all_dy = dst_arr[:, 1] - src_arr[:, 1]

    # ── Only keep points that actually moved (threshold > 0.1 px) ──
    disp_mag = np.sqrt(all_dx**2 + all_dy**2)
    moved_mask = disp_mag > 0.1

    moved_src = src_arr[moved_mask]
    moved_dx  = all_dx[moved_mask]
    moved_dy  = all_dy[moved_mask]

    if len(moved_src) == 0:
        return image.copy()

    # ── Add sparse anchor points (zero displacement) ──
    # Image corners + edge midpoints + a grid of interior anchors
    anchors = [
        [0, 0], [w//2, 0], [w-1, 0],
        [0, h//2], [w-1, h//2],
        [0, h-1], [w//2, h-1], [w-1, h-1],
        [w//4, h//4], [3*w//4, h//4],
        [w//4, 3*h//4], [3*w//4, 3*h//4],
        [w//2, h//4], [w//2, 3*h//4],
        [w//4, h//2], [3*w//4, h//2],
    ]
    anchor_arr = np.array(anchors, dtype=np.float64)
    anchor_dx  = np.zeros(len(anchors))
    anchor_dy  = np.zeros(len(anchors))

    # Combine moved points + anchors
    ctrl_pts = np.vstack([moved_src, anchor_arr])
    ctrl_dx  = np.concatenate([moved_dx, anchor_dx])
    ctrl_dy  = np.concatenate([moved_dy, anchor_dy])

    # Build RBF interpolators
    rbf_dx = RBFInterpolator(ctrl_pts, ctrl_dx, kernel='thin_plate_spline', smoothing=2.0)
    rbf_dy = RBFInterpolator(ctrl_pts, ctrl_dy, kernel='thin_plate_spline', smoothing=2.0)

    # Evaluate on a coarse grid then upscale (for speed)
    GRID_STEP = 4
    gy = np.arange(0, h, GRID_STEP)
    gx = np.arange(0, w, GRID_STEP)
    grid_x, grid_y = np.meshgrid(gx, gy)
    query_pts = np.column_stack([grid_x.ravel(), grid_y.ravel()])

    disp_x_coarse = rbf_dx(query_pts).reshape(grid_y.shape).astype(np.float32)
    disp_y_coarse = rbf_dy(query_pts).reshape(grid_y.shape).astype(np.float32)

    # Upscale to full resolution
    disp_x = cv2.resize(disp_x_coarse, (w, h), interpolation=cv2.INTER_CUBIC)
    disp_y = cv2.resize(disp_y_coarse, (w, h), interpolation=cv2.INTER_CUBIC)

    # Inverse remap: for each output pixel, where to sample from source
    map_y, map_x = np.mgrid[0:h, 0:w].astype(np.float32)
    map_x = map_x - disp_x
    map_y = map_y - disp_y

    result = cv2.remap(image, map_x, map_y,
                       interpolation=cv2.INTER_CUBIC,
                       borderMode=cv2.BORDER_REFLECT_101)
    return result


# ── Main Dispatcher ───────────────────────────────────────────────────────────

def apply_expression_transform(
    image: np.ndarray,
    landmarks: list,
    smile_intensity: float = 0.0,
    eyebrow_intensity: float = 0.0,
    lip_intensity: float = 0.0,
    slim_intensity: float = 0.0,
    **kwargs
) -> np.ndarray:
    
    eyebrow_val = eyebrow_intensity if eyebrow_intensity is not None else kwargs.get('eyebrow_height', 0.0)
    lip_val = lip_intensity if lip_intensity is not None else kwargs.get('lip_widening', 0.0)
    slim_val = slim_intensity if slim_intensity is not None else kwargs.get('face_slimming', 0.0)
    
    h, w = image.shape[:2]
    src_points = landmarks_to_points(landmarks, w, h)
    if not src_points: return image
        
    dst_points = copy.deepcopy(src_points)
    scale, nose_tip = get_face_properties(src_points)

    if smile_intensity != 0:
        shift_for_duchenne_smile(src_points, dst_points, smile_intensity, scale)

    if eyebrow_val != 0:
        shift_for_eyebrow(src_points, dst_points, eyebrow_val, scale)

    if lip_val != 0:
        shift_for_lip_widening(src_points, dst_points, lip_val, scale)

    if slim_val != 0:
        shift_for_face_slimming(src_points, dst_points, slim_val, scale, nose_tip)

    return warp_image(image, [tuple(p) for p in src_points], [tuple(p) for p in dst_points])
