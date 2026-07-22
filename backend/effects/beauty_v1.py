from __future__ import annotations

import cv2
import numpy as np

from backend.face_parsing import feather_mask


# =========================================================
# HELPERS
# =========================================================

def _ensure_mask(mask: np.ndarray, shape) -> np.ndarray:
    h, w = shape[:2]

    if mask is None:
        return np.zeros((h, w), dtype=np.uint8)

    if mask.shape[:2] != (h, w):
        mask = cv2.resize(
            mask,
            (w, h),
            interpolation=cv2.INTER_NEAREST,
        )

    return mask


def _soft_alpha(mask: np.ndarray, blur: int = 21) -> np.ndarray:
    mask = (mask > 20).astype(np.uint8) * 255

    if blur > 0:
        if blur % 2 == 0:
            blur += 1

        mask = cv2.GaussianBlur(
            mask,
            (blur, blur),
            0,
        )

    alpha = mask.astype(np.float32) / 255.0

    return np.clip(alpha, 0.0, 1.0)


def _hex_to_bgr(hex_color: str) -> tuple[int, int, int]:
    """
    '#RRGGBB' -> BGR tuple
    """

    hex_color = hex_color.strip().replace("#", "")

    if len(hex_color) != 6:
        raise ValueError("hex_color must be like '#AA3366'")

    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)

    return b, g, r


def _fallback_lip_mask_from_landmarks(
    landmarks: np.ndarray,
    shape: tuple[int, ...],
) -> np.ndarray:
    h, w = shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    if not isinstance(landmarks, np.ndarray) or landmarks.shape[0] <= 409:
        return mask

    upper_outer = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291]
    lower_outer_right_to_left = [291, 375, 321, 405, 314, 17, 84, 181, 91, 146, 61]
    center_pts = landmarks[[61, 291, 13, 14]].astype(np.float32)
    center = np.array(
        [float(np.mean(center_pts[:, 0])), float(np.mean(center_pts[:, 1]))],
        dtype=np.float32,
    )

    def _poly(indices: list[int]) -> np.ndarray:
        pts = landmarks[indices].astype(np.float32)
        # Keep close to the actual visible lip contour. Over-shrinking makes
        # lipstick look like a thin mouth-line on small lips.
        pts[:, 0] = center[0] + (pts[:, 0] - center[0]) * 0.99
        pts[:, 1] = center[1] + (pts[:, 1] - center[1]) * 0.98
        pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
        pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)
        return pts.astype(np.int32)

    outer_lip = _poly(upper_outer + lower_outer_right_to_left)
    cv2.fillPoly(mask, [outer_lip.reshape((-1, 1, 2))], 255, cv2.LINE_AA)

    outer_pts = landmarks[[61, 291, 0, 17, 13, 14]].astype(np.float32)
    lip_w = float(np.max(outer_pts[:, 0]) - np.min(outer_pts[:, 0]))
    lip_h = float(np.max(outer_pts[:, 1]) - np.min(outer_pts[:, 1]))
    if lip_w < 8 or lip_h < 3:
        return mask

    mouth_inner = _mouth_inner_mask_from_landmarks(
        landmarks,
        shape,
        dilate=_lip_inner_cut_dilate(landmarks),
    )
    if int(np.count_nonzero(mouth_inner > 20)) > 0:
        mask = cv2.bitwise_and(mask, cv2.bitwise_not(mouth_inner))

    return mask


def _mouth_inner_mask_from_landmarks(
    landmarks: np.ndarray,
    shape: tuple[int, ...],
    dilate: int = 1,
) -> np.ndarray:
    h, w = shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    if not isinstance(landmarks, np.ndarray) or landmarks.shape[0] <= 415:
        return mask

    inner_idx = [
        78, 191, 80, 81, 82, 13, 312, 311, 310, 415,
        308, 324, 318, 402, 317, 14, 87, 178, 88, 95,
    ]
    pts = landmarks[inner_idx].astype(np.float32)
    pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)
    cv2.fillPoly(mask, [pts.astype(np.int32).reshape((-1, 1, 2))], 255, cv2.LINE_AA)

    if dilate > 0:
        mask = cv2.dilate(mask, np.ones((3, 3), dtype=np.uint8), iterations=dilate)

    return mask


def _lip_inner_cut_dilate(landmarks: np.ndarray | None) -> int:
    if not isinstance(landmarks, np.ndarray) or landmarks.shape[0] <= 17:
        return 1

    try:
        outer_top = float(landmarks[0][1])
        outer_bottom = float(landmarks[17][1])
        inner_top = float(landmarks[13][1])
        inner_bottom = float(landmarks[14][1])
        lip_h = max(1.0, abs(outer_bottom - outer_top))
        inner_gap = abs(inner_bottom - inner_top)
    except Exception:
        return 1

    if inner_gap < max(2.0, lip_h * 0.16):
        return 0
    if inner_gap < lip_h * 0.30:
        return 1
    return 2


def _safe_erode_mask(
    mask: np.ndarray,
    iterations: int = 1,
    min_keep_ratio: float = 0.55,
) -> np.ndarray:
    pixels_before = int(np.count_nonzero(mask > 20))
    if pixels_before == 0:
        return mask

    eroded = cv2.erode(
        mask,
        np.ones((2, 2), dtype=np.uint8),
        iterations=iterations,
    )
    pixels_after = int(np.count_nonzero(eroded > 20))
    if pixels_after >= max(8, int(pixels_before * min_keep_ratio)):
        return eroded

    return mask


def _refine_lip_mask(
    semantic_mask: np.ndarray,
    ctx: dict,
    image_shape: tuple[int, ...],
) -> tuple[np.ndarray, bool]:
    h, w = image_shape[:2]
    semantic_mask = _ensure_mask(semantic_mask, image_shape)
    landmark_mask = _fallback_lip_mask_from_landmarks(ctx.get("landmarks_2d"), image_shape)
    landmark_pixels = int(np.count_nonzero(landmark_mask > 20))

    if landmark_pixels == 0:
        return semantic_mask, False

    # Face parsing can miss the upper lip or include inner-mouth pixels. For
    # lipstick the face-mesh lip contour is usually the tighter, safer source.
    landmarks = ctx.get("landmarks_2d")
    mouth_inner = _mouth_inner_mask_from_landmarks(
        landmarks,
        image_shape,
        dilate=_lip_inner_cut_dilate(landmarks),
    )
    landmark_mask = cv2.bitwise_and(landmark_mask, cv2.bitwise_not(mouth_inner))
    return landmark_mask, True


# =========================================================
# LIPSTICK
# =========================================================

def apply_lipstick(
    image_bgr: np.ndarray,
    ctx: dict,
    params: dict | None = None,
) -> np.ndarray:
    """
    Realistic-ish lipstick effect.

    Expected params:
    {
        "color": "#A02045",
        "intensity": 0.65
    }
    """

    params = params or {}

    color_hex = params.get("color", "#A02045")
    intensity = float(params.get("intensity", 0.65))
    intensity = float(np.clip(intensity, 0.0, 1.0))

    lip_mask, fallback_used = _refine_lip_mask(
        ctx.get("masks", {}).get("lips"),
        ctx,
        image_bgr.shape,
    )

    alpha = _soft_alpha(
        lip_mask,
        blur=3,
    ) * intensity
    landmarks = ctx.get("landmarks_2d")
    mouth_inner_hard = _mouth_inner_mask_from_landmarks(
        landmarks,
        image_bgr.shape,
        dilate=_lip_inner_cut_dilate(landmarks),
    )
    if int(np.count_nonzero(mouth_inner_hard > 20)) > 0:
        alpha[mouth_inner_hard > 20] = 0.0

    mask_pixels = int(np.count_nonzero(alpha > 0.02))
    ctx.setdefault("effect_debug_meta", {})["lipstick"] = {
        "lipstick_debug": {
            "fallback_used": fallback_used,
            "mask_pixels": mask_pixels,
            "reason": None if mask_pixels else "empty_lip_mask",
        }
    }

    if mask_pixels == 0 or intensity <= 0.0:
        return image_bgr.copy()

    target_bgr = _hex_to_bgr(color_hex)
    visible_alpha = np.clip(alpha * 1.12, 0.0, 1.0)

    # LAB keeps texture better than raw RGB/BGR paint
    lab = cv2.cvtColor(
        image_bgr,
        cv2.COLOR_BGR2LAB,
    )

    l, a, b = cv2.split(lab)

    target_img = np.full_like(
        image_bgr,
        target_bgr,
        dtype=np.uint8,
    )

    target_lab = cv2.cvtColor(
        target_img,
        cv2.COLOR_BGR2LAB,
    )

    target_l, target_a, target_b = cv2.split(target_lab)

    # Keep brightness mostly, change chroma
    strength = visible_alpha

    a_new = (
        a.astype(np.float32) * (1.0 - strength)
        + target_a.astype(np.float32) * strength
    )

    b_new = (
        b.astype(np.float32) * (1.0 - strength)
        + target_b.astype(np.float32) * strength
    )

    # Darker lipstick colors need a small luminance pull; otherwise they only
    # alter hue and look invisible on naturally bright lips.
    l_strength = np.clip(visible_alpha * 0.32, 0.0, 0.32)
    l_new = (
        l.astype(np.float32) * (1.0 - l_strength)
        + target_l.astype(np.float32) * l_strength
    )

    result_lab = cv2.merge([
        np.clip(l_new, 0, 255).astype(np.uint8),
        np.clip(a_new, 0, 255).astype(np.uint8),
        np.clip(b_new, 0, 255).astype(np.uint8),
    ])

    result = cv2.cvtColor(
        result_lab,
        cv2.COLOR_LAB2BGR,
    )
    result = np.where(
        visible_alpha[:, :, None] > 0.001,
        result,
        image_bgr,
    )

    # A low-opacity color layer makes the effect visible on low-saturation lips
    # while LAB chroma preserves most natural texture.
    tint_strength = np.clip(visible_alpha * 0.28, 0.0, 0.28)
    tint = np.full_like(result, target_bgr, dtype=np.uint8)
    result = (
        result.astype(np.float32) * (1.0 - tint_strength[:, :, None])
        + tint.astype(np.float32) * tint_strength[:, :, None]
    )

    return np.clip(result, 0, 255).astype(np.uint8)


# =========================================================
# SKIN SMOOTH
# =========================================================

def apply_skin_smooth(
    image_bgr: np.ndarray,
    ctx: dict,
    params: dict | None = None,
) -> np.ndarray:
    """
    Edge-preserving skin smoothing.

    Expected params:
    {
        "intensity": 0.35
    }
    """

    params = params or {}

    intensity = float(params.get("intensity", 0.35))
    intensity = float(np.clip(intensity, 0.0, 1.0))

    skin_mask = ctx["masks"].get("skin_effect")
    skin_mask = _ensure_mask(skin_mask, image_bgr.shape)

    alpha = _soft_alpha(
        skin_mask,
        blur=25,
    ) * intensity

    # Bilateral keeps edges better than Gaussian blur
    smooth = cv2.bilateralFilter(
        image_bgr,
        d=9,
        sigmaColor=45,
        sigmaSpace=45,
    )

    result = (
        image_bgr.astype(np.float32) * (1.0 - alpha[:, :, None])
        + smooth.astype(np.float32) * alpha[:, :, None]
    )

    return np.clip(result, 0, 255).astype(np.uint8)


# =========================================================
# BLUSH
# =========================================================

def apply_blush(
    image_bgr: np.ndarray,
    ctx: dict,
    params: dict | None = None,
) -> np.ndarray:
    """
    Cheek blush using landmarks, not only parsing.

    Expected params:
    {
        "color": "#D96C7C",
        "intensity": 0.35
    }
    """

    params = params or {}

    color_hex = params.get("color", "#D96C7C")
    intensity = float(params.get("intensity", 0.35))
    intensity = float(np.clip(intensity, 0.0, 1.0))

    landmarks = ctx["landmarks_2d"]

    h, w = image_bgr.shape[:2]

    # Approx cheek anchors
    left_cheek = landmarks[205].astype(np.float32)
    right_cheek = landmarks[425].astype(np.float32)

    face_width = ctx["anchors"]["metrics"]["face_width"]

    radius_x = int(face_width * 0.13)
    radius_y = int(face_width * 0.08)

    blush_mask = np.zeros((h, w), dtype=np.uint8)

    for p in [left_cheek, right_cheek]:
        cv2.ellipse(
            blush_mask,
            (int(p[0]), int(p[1])),
            (radius_x, radius_y),
            0,
            0,
            360,
            255,
            -1,
            cv2.LINE_AA,
        )

    # Restrict to skin_effect
    skin_mask = ctx.get("masks", {}).get("skin_effect")
    skin_mask = _ensure_mask(skin_mask, image_bgr.shape)

    if int(np.count_nonzero(skin_mask > 20)) > 0:
        blush_mask = cv2.bitwise_and(
            blush_mask,
            skin_mask,
        )

    alpha = _soft_alpha(
        blush_mask,
        blur=55,
    ) * intensity

    target_bgr = np.array(
        _hex_to_bgr(color_hex),
        dtype=np.float32,
    )

    color_layer = np.full_like(
        image_bgr,
        target_bgr,
        dtype=np.float32,
    )

    # Soft overlay-like blend
    result = (
        image_bgr.astype(np.float32) * (1.0 - alpha[:, :, None])
        + color_layer * alpha[:, :, None]
    )

    return np.clip(result, 0, 255).astype(np.uint8)


# =========================================================
# EYEBROW DARKEN
# =========================================================

def apply_eyebrow_enhance(
    image_bgr: np.ndarray,
    ctx: dict,
    params: dict | None = None,
) -> np.ndarray:
    """
    Slight eyebrow darkening.

    Expected params:
    {
        "intensity": 0.35
    }
    """

    params = params or {}

    intensity = float(params.get("intensity", 0.35))
    intensity = float(np.clip(intensity, 0.0, 1.0))

    brow_mask = ctx["masks"].get("eyebrows")
    brow_mask = _ensure_mask(brow_mask, image_bgr.shape)

    alpha = _soft_alpha(
        brow_mask,
        blur=9,
    ) * intensity

    darker = (image_bgr.astype(np.float32) * 0.55)

    result = (
        image_bgr.astype(np.float32) * (1.0 - alpha[:, :, None])
        + darker * alpha[:, :, None]
    )

    return np.clip(result, 0, 255).astype(np.uint8)


# =========================================================
# BEARD EFFECT
# =========================================================

def apply_beard(
    image_bgr: np.ndarray,
    ctx: dict,
    params: dict | None = None,
) -> np.ndarray:
    """
    Adds a realistic beard/stubble effect.

    Expected params:
    {
        "color": "#1a1108",
        "intensity": 0.6
    }
    """

    params = params or {}

    color_hex = params.get("color", "#1a1108")
    intensity = float(params.get("intensity", 0.6))
    intensity = float(np.clip(intensity, 0.0, 1.0))

    landmarks = ctx.get("landmarks_2d")
    parsing = ctx.get("parsing")

    if landmarks is None or parsing is None or intensity <= 0.0:
        return image_bgr.copy()

    h, w = image_bgr.shape[:2]
    landmarks = np.asarray(landmarks, dtype=np.float32)

    required = [1, 10, 13, 14, 61, 132, 152, 234, 291, 361, 454]
    if len(landmarks) <= max(required):
        return image_bgr.copy()

    face_left = landmarks[234]
    face_right = landmarks[454]
    chin = landmarks[152]
    forehead = landmarks[10]
    nose = landmarks[1]
    left_mouth = landmarks[61]
    right_mouth = landmarks[291]
    mouth_center = (left_mouth + right_mouth) * 0.5
    face_width = float(max(1.0, np.linalg.norm(face_right - face_left)))
    face_height = float(max(1.0, chin[1] - forehead[1]))
    mouth_width = float(max(1.0, np.linalg.norm(right_mouth - left_mouth)))

    roi = np.zeros((h, w), dtype=np.uint8)

    jaw_indices = [132, 58, 172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 397, 288, 361]
    if len(landmarks) > max(jaw_indices):
        top_y = float(mouth_center[1] + face_height * 0.025)
        top_left = np.array([left_mouth[0] - face_width * 0.14, top_y], dtype=np.float32)
        top_mid = np.array([mouth_center[0], top_y + face_height * 0.025], dtype=np.float32)
        top_right = np.array([right_mouth[0] + face_width * 0.14, top_y], dtype=np.float32)
        jaw_pts = landmarks[jaw_indices].copy()
        poly = np.vstack([top_left, top_mid, top_right, jaw_pts[::-1]])
        poly[:, 0] = np.clip(poly[:, 0], 0, w - 1)
        poly[:, 1] = np.clip(poly[:, 1], 0, h - 1)
        cv2.fillPoly(roi, [poly.astype(np.int32)], 255, cv2.LINE_AA)

    moustache = np.zeros((h, w), dtype=np.uint8)
    nose_to_mouth = max(4.0, float(mouth_center[1] - nose[1]))
    moustache_center = (
        int(round(mouth_center[0])),
        int(round(nose[1] + nose_to_mouth * 0.72)),
    )
    cv2.ellipse(
        moustache,
        moustache_center,
        (int(mouth_width * 0.62), int(max(3.0, face_height * 0.032))),
        0,
        0,
        360,
        255,
        -1,
        cv2.LINE_AA,
    )
    roi = cv2.max(roi, moustache)

    skin_area = np.isin(parsing, [1]).astype(np.uint8) * 255
    if int(np.count_nonzero(skin_area)) == 0:
        skin_area = (roi > 0).astype(np.uint8) * 255
    exclude = np.isin(parsing, [2, 3, 4, 5, 10, 11, 12, 13]).astype(np.uint8) * 255
    candidate = cv2.bitwise_and(roi, skin_area)
    candidate = cv2.subtract(candidate, exclude)
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))

    base_alpha = feather_mask(candidate, blur=29).astype(np.float32)

    yy, xx = np.indices((h, w), dtype=np.float32)
    upper_fade_start = float(nose[1] + nose_to_mouth * 0.35)
    upper_fade_end = float(mouth_center[1] + face_height * 0.08)
    vertical_density = np.clip((yy - upper_fade_start) / max(1.0, upper_fade_end - upper_fade_start), 0.0, 1.0)
    horizontal_density = 1.0 - np.clip(np.abs(xx - mouth_center[0]) / (face_width * 0.58), 0.0, 1.0) * 0.35
    density = np.clip(vertical_density * horizontal_density, 0.0, 1.0)

    # Separate anatomical density for "new stubble" mode. A broad lower-face
    # region is acceptable as a search area, but it must not become the paint
    # alpha. New stubble should be densest on the chin, jaw, and moustache,
    # and much sparser on cheeks.
    chin_center = (
        float(mouth_center[0]),
        float(mouth_center[1] + (chin[1] - mouth_center[1]) * 0.62),
    )
    chin_density = np.exp(
        -(
            ((xx - chin_center[0]) / max(1.0, face_width * 0.26)) ** 2
            + ((yy - chin_center[1]) / max(1.0, face_height * 0.16)) ** 2
        )
    )
    moustache_density = np.exp(
        -(
            ((xx - mouth_center[0]) / max(1.0, mouth_width * 0.72)) ** 2
            + ((yy - moustache_center[1]) / max(1.0, face_height * 0.035)) ** 2
        )
    )
    jaw_y = mouth_center[1] + (chin[1] - mouth_center[1]) * 0.72
    jaw_density = np.exp(-((yy - jaw_y) / max(1.0, face_height * 0.13)) ** 2)
    side_distance = np.abs(xx - mouth_center[0]) / max(1.0, face_width * 0.48)
    side_density = np.clip(side_distance - 0.18, 0.0, 1.0) * jaw_density
    cheek_taper = 1.0 - np.clip((yy - mouth_center[1]) / max(1.0, chin[1] - mouth_center[1]), 0.0, 1.0) * 0.28
    stubble_density = np.clip(
        chin_density * 0.95
        + moustache_density * 0.68
        + side_density * 0.42,
        0.0,
        1.0,
    )
    stubble_density *= cheek_taper

    geometry_alpha = np.clip(base_alpha * density, 0.0, 1.0)
    if np.count_nonzero(geometry_alpha > 0.025) == 0:
        return image_bgr.copy()

    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    lab_orig = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    lap = np.abs(cv2.Laplacian(gray, cv2.CV_32F, ksize=3))
    _, sat, val = cv2.split(hsv)
    l_orig, a_orig, b_orig = cv2.split(lab_orig)

    existing = (
        ((val < 120) | (l_orig < 125))
        & (sat > 18)
        & (lap > 4)
        & (geometry_alpha > 0.05)
    ).astype(np.uint8) * 255
    existing = cv2.morphologyEx(existing, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    existing = cv2.morphologyEx(existing, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    existing_alpha = feather_mask(existing, blur=17).astype(np.float32)
    existing_alpha = np.clip(existing_alpha * geometry_alpha, 0.0, 1.0)
    existing_pixels = int(np.count_nonzero(existing_alpha > 0.08))

    target_bgr = _hex_to_bgr(color_hex)
    target_img = np.full_like(image_bgr, target_bgr, dtype=np.uint8)
    target_lab = cv2.cvtColor(target_img, cv2.COLOR_BGR2LAB)
    target_l, target_a, target_b = cv2.split(target_lab)

    target_l_value = float(target_l[0, 0])
    h_idx, w_idx = np.indices((h, w))
    hash_grid = (((h_idx * 127 + w_idx * 311 + (h_idx * w_idx) % 97) % 1000) / 1000.0).astype(np.float32)
    pore_noise = cv2.GaussianBlur(hash_grid, (0, 0), 0.55)
    pore_noise -= cv2.GaussianBlur(hash_grid, (0, 0), 2.0)
    pore_noise = cv2.normalize(pore_noise, None, 0.0, 1.0, cv2.NORM_MINMAX)

    local_texture = cv2.GaussianBlur(lap, (0, 0), 0.8)
    local_texture = cv2.normalize(local_texture, None, 0.0, 1.0, cv2.NORM_MINMAX)

    # If the portrait already has a beard, mostly recolor/enhance that actual
    # hair. If not, add sparse short stubble only in the anatomical beard zone.
    has_existing_beard = existing_pixels > max(80, int(face_width * 0.45))
    sparse_threshold = 0.82 - 0.12 * intensity
    sparse_stubble = (pore_noise > sparse_threshold).astype(np.float32)
    sparse_stubble *= (0.35 + 0.65 * local_texture)
    sparse_stubble = cv2.GaussianBlur(sparse_stubble, (3, 3), 0)
    sparse_stubble *= geometry_alpha * stubble_density

    hairlet_alpha = np.zeros((h, w), dtype=np.float32)
    if not has_existing_beard:
        candidate_strength = geometry_alpha * stubble_density
        seed_mask = (candidate_strength > 0.055) & (pore_noise > (0.88 - 0.13 * intensity))
        ys, xs = np.where(seed_mask)
        if len(xs) > 0:
            strengths = candidate_strength[ys, xs] * (0.45 + 0.55 * pore_noise[ys, xs])
            max_hairs = int(np.clip(face_width * face_height * (0.018 + 0.028 * intensity), 350, 2600))
            if len(xs) > max_hairs:
                order = np.argsort(strengths)[-max_hairs:]
                ys = ys[order]
                xs = xs[order]
                strengths = strengths[order]

            # Draw tiny deterministic fibers instead of a continuous painted mask.
            # Direction follows anatomy: moustache mostly horizontal, chin/jaw mostly vertical.
            for y, x, strength in zip(ys, xs, strengths):
                local = float(np.clip(strength, 0.0, 1.0))
                if local <= 0.0:
                    continue

                rel_y = (float(y) - float(mouth_center[1])) / max(1.0, float(chin[1] - mouth_center[1]))
                rel_x = (float(x) - float(mouth_center[0])) / max(1.0, face_width * 0.5)
                hash_angle = ((int(y) * 37 + int(x) * 17) % 100) / 100.0 - 0.5

                if float(y) < float(mouth_center[1]) + face_height * 0.06:
                    angle = hash_angle * 0.45
                    length = 1.2 + 2.0 * intensity
                else:
                    angle = np.pi * 0.5 + rel_x * 0.28 + hash_angle * 0.55
                    length = 1.0 + 2.8 * intensity * (0.55 + local)

                dx = float(np.cos(angle) * length)
                dy = float(np.sin(angle) * length)
                value = float(np.clip((0.28 + 0.42 * intensity) * local, 0.0, 0.78))
                cv2.line(
                    hairlet_alpha,
                    (int(round(x - dx * 0.5)), int(round(y - dy * 0.5))),
                    (int(round(x + dx * 0.5)), int(round(y + dy * 0.5))),
                    value,
                    1,
                    cv2.LINE_AA,
                )

            hairlet_alpha = cv2.GaussianBlur(hairlet_alpha, (3, 3), 0)
            hairlet_alpha = np.clip(hairlet_alpha * geometry_alpha, 0.0, 1.0)

    if has_existing_beard:
        grown_existing = cv2.dilate((existing_alpha > 0.06).astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1)
        grown_existing = feather_mask(grown_existing * 255, blur=13).astype(np.float32)
        beard_alpha = np.clip(existing_alpha * 0.95 + grown_existing * geometry_alpha * 0.18 + sparse_stubble * 0.08, 0.0, 1.0)
    else:
        broad_shadow = geometry_alpha * stubble_density * (0.006 + 0.018 * intensity)
        beard_alpha = np.clip(hairlet_alpha + sparse_stubble * (0.16 + 0.22 * intensity) + broad_shadow, 0.0, 0.34)

    if np.count_nonzero(beard_alpha > 0.02) == 0:
        return image_bgr.copy()

    l_base = l_orig.astype(np.float32)
    a_base = a_orig.astype(np.float32)
    b_base = b_orig.astype(np.float32)

    color_alpha = np.clip(beard_alpha * (0.12 + 0.30 * intensity), 0.0, 0.42)
    chroma_alpha = color_alpha * (0.66 if has_existing_beard else 0.24)

    # Light target colors should tint/soften existing beard instead of erasing
    # it into flat skin; dark targets should deepen the same hair texture.
    l_target_delta = np.clip((target_l_value - l_base) * color_alpha * 0.34, -20.0, 22.0)
    natural_shadow = (1.0 - np.clip(target_l_value / 235.0, 0.0, 1.0)) * (4.0 + 14.0 * intensity)
    if not has_existing_beard:
        natural_shadow *= 0.42
    texture_shadow = (0.45 * pore_noise + 0.55 * local_texture) * beard_alpha * (5.0 + 10.0 * intensity)
    if not has_existing_beard:
        texture_shadow += hairlet_alpha * (7.0 + 12.0 * intensity)
    l_new = np.clip(l_base + l_target_delta - natural_shadow * beard_alpha - texture_shadow, 0, 255)
    a_new = a_base * (1.0 - chroma_alpha) + target_a.astype(np.float32) * chroma_alpha
    b_new = b_base * (1.0 - chroma_alpha) + target_b.astype(np.float32) * chroma_alpha

    result_lab = cv2.merge([
        np.clip(l_new, 0, 255).astype(np.uint8),
        np.clip(a_new, 0, 255).astype(np.uint8),
        np.clip(b_new, 0, 255).astype(np.uint8)
    ])

    result = cv2.cvtColor(result_lab, cv2.COLOR_LAB2BGR)

    ctx.setdefault("effect_debug_meta", {})["beard"] = {
        "beard_debug": {
            "mode": "recolor_existing" if has_existing_beard else "add_sparse_stubble",
            "mask_pixels": int(np.count_nonzero(beard_alpha > 0.025)),
            "existing_beard_pixels": existing_pixels,
            "cleanup_applied": False,
            "max_alpha": float(np.max(beard_alpha)),
            "intensity": intensity,
            "stubble_density_pixels": int(np.count_nonzero((geometry_alpha * stubble_density) > 0.05)),
            "hairlet_pixels": int(np.count_nonzero(hairlet_alpha > 0.015)),
        }
    }

    return result

