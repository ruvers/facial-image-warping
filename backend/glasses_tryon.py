from __future__ import annotations

import os
import cv2
import numpy as np


# =========================================================
# PATHS
# =========================================================

BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BACKEND_DIR)

ASSET_DIR = os.path.join(ROOT_DIR, "assets", "glasses")
DEFAULT_GLASSES_PATH = os.path.join(ASSET_DIR, "default_black.png")


# =========================================================
# ASSET GENERATION
# =========================================================

def ensure_default_glasses_asset() -> str:
    """
    Create a transparent procedural glasses PNG if no asset exists.
    This is for testing. Later you can replace it with a real PNG.
    """

    os.makedirs(ASSET_DIR, exist_ok=True)

    if os.path.exists(DEFAULT_GLASSES_PATH):
        return DEFAULT_GLASSES_PATH

    w, h = 900, 300
    img = np.zeros((h, w, 4), dtype=np.uint8)

    # Lens positions
    left_center = (280, 150)
    right_center = (620, 150)

    lens_axes = (145, 85)

    # Subtle lens fill
    cv2.ellipse(
        img,
        left_center,
        lens_axes,
        0,
        0,
        360,
        (180, 200, 220, 35),
        -1,
    )

    cv2.ellipse(
        img,
        right_center,
        lens_axes,
        0,
        0,
        360,
        (180, 200, 220, 35),
        -1,
    )

    # Thick black frame
    frame_color = (15, 15, 18, 255)

    cv2.ellipse(
        img,
        left_center,
        lens_axes,
        0,
        0,
        360,
        frame_color,
        18,
    )

    cv2.ellipse(
        img,
        right_center,
        lens_axes,
        0,
        0,
        360,
        frame_color,
        18,
    )

    # Bridge
    cv2.line(
        img,
        (420, 145),
        (480, 145),
        frame_color,
        18,
        cv2.LINE_AA,
    )

    cv2.line(
        img,
        (420, 160),
        (480, 160),
        frame_color,
        10,
        cv2.LINE_AA,
    )

    # Temple arms
    cv2.line(
        img,
        (140, 145),
        (20, 120),
        frame_color,
        14,
        cv2.LINE_AA,
    )

    cv2.line(
        img,
        (760, 145),
        (880, 120),
        frame_color,
        14,
        cv2.LINE_AA,
    )

    # Small highlights on frame
    cv2.ellipse(
        img,
        left_center,
        (125, 65),
        0,
        200,
        320,
        (255, 255, 255, 45),
        4,
        cv2.LINE_AA,
    )

    cv2.ellipse(
        img,
        right_center,
        (125, 65),
        0,
        200,
        320,
        (255, 255, 255, 45),
        4,
        cv2.LINE_AA,
    )

    cv2.imwrite(DEFAULT_GLASSES_PATH, img)

    return DEFAULT_GLASSES_PATH


# =========================================================
# HELPERS
# =========================================================

def load_bgra(path: str) -> np.ndarray:
    asset = cv2.imread(path, cv2.IMREAD_UNCHANGED)

    if asset is None:
        raise FileNotFoundError(path)

    if asset.shape[2] == 3:
        alpha = np.full(asset.shape[:2], 255, dtype=np.uint8)
        asset = np.dstack([asset, alpha])

    return asset


def alpha_blend_bgra(
    base_bgr: np.ndarray,
    overlay_bgra: np.ndarray,
) -> np.ndarray:
    """
    Blend full-size BGRA overlay onto BGR image.
    """

    overlay_bgr = overlay_bgra[:, :, :3].astype(np.float32)
    alpha = overlay_bgra[:, :, 3].astype(np.float32) / 255.0

    alpha_3 = alpha[:, :, None]

    base = base_bgr.astype(np.float32)

    out = (
        base * (1.0 - alpha_3)
        + overlay_bgr * alpha_3
    )

    return np.clip(out, 0, 255).astype(np.uint8)


def add_soft_shadow(
    image_bgr: np.ndarray,
    alpha: np.ndarray,
    dx: int = 4,
    dy: int = 5,
    blur: int = 21,
    opacity: float = 0.25,
) -> np.ndarray:
    """
    Add a soft shadow under the accessory.
    """

    h, w = alpha.shape[:2]

    if blur % 2 == 0:
        blur += 1

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
        0,
        1,
    )

    shadow_3 = shadow[:, :, None]

    out = image_bgr.astype(np.float32)

    out = out * (1.0 - shadow_3)

    return np.clip(out, 0, 255).astype(np.uint8)


def warp_asset_to_face(
    image_bgr: np.ndarray,
    asset_bgra: np.ndarray,
    center: tuple[float, float],
    width: float,
    angle_deg: float,
) -> np.ndarray:
    """
    Resize + rotate glasses asset into full-frame BGRA overlay.
    """

    h_img, w_img = image_bgr.shape[:2]
    h_asset, w_asset = asset_bgra.shape[:2]

    target_w = int(width)
    target_h = int(target_w * (h_asset / w_asset))

    if target_w <= 10 or target_h <= 10:
        return np.zeros((h_img, w_img, 4), dtype=np.uint8)

    resized = cv2.resize(
        asset_bgra,
        (target_w, target_h),
        interpolation=cv2.INTER_AREA,
    )

    cx, cy = center

    M = cv2.getRotationMatrix2D(
        (target_w / 2, target_h / 2),
        angle_deg,
        1.0,
    )

    M[0, 2] += cx - target_w / 2
    M[1, 2] += cy - target_h / 2

    full_overlay = cv2.warpAffine(
        resized,
        M,
        (w_img, h_img),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )

    return full_overlay


# =========================================================
# MAIN GLASSES TRY-ON
# =========================================================

def apply_glasses(
    image_bgr: np.ndarray,
    analysis: dict,
    asset_path: str | None = None,
    scale: float = 2.20,
    y_offset: float = 0.10,
    use_hair_occlusion: bool = True,
) -> np.ndarray:
    """
    Apply glasses using MediaPipe landmarks + segmentation masks.

    image_bgr:
        OpenCV BGR image

    analysis:
        output of analyze_face(image_bgr)

    asset_path:
        transparent PNG glasses asset. If None, default asset is generated.

    scale:
        glasses width relative to eye distance

    y_offset:
        vertical offset relative to eye distance
    """

    if asset_path is None:
        asset_path = ensure_default_glasses_asset()

    landmarks = analysis["landmarks"]
    masks = analysis.get("masks", {})

    # MediaPipe eye outer corners
    left_eye = landmarks[33].astype(np.float32)
    right_eye = landmarks[263].astype(np.float32)

    eye_vec = right_eye - left_eye
    eye_dist = float(np.linalg.norm(eye_vec))

    if eye_dist < 10:
        return image_bgr

    angle = np.degrees(
        np.arctan2(
            eye_vec[1],
            eye_vec[0],
        )
    )

    center = (
        (left_eye + right_eye) / 2.0
    )

    center[1] += eye_dist * y_offset

    target_width = eye_dist * scale

    asset = load_bgra(asset_path)

    overlay = warp_asset_to_face(
        image_bgr=image_bgr,
        asset_bgra=asset,
        center=(float(center[0]), float(center[1])),
        width=target_width,
        angle_deg=float(angle),
    )

    # =====================================================
    # SIMPLE DEPTH / OCCLUSION HACK
    # =====================================================

    if use_hair_occlusion and "hair" in masks:
        hair_mask = masks["hair"]

        if hair_mask.shape[:2] == overlay.shape[:2]:
            alpha = overlay[:, :, 3].astype(np.float32)

            # Only hide side arms slightly under hair.
            h, w = alpha.shape
            side_mask = np.zeros((h, w), dtype=np.uint8)

            x_center = int(center[0])
            side_margin = int(eye_dist * 0.75)

            side_mask[:, :max(0, x_center - side_margin)] = 255
            side_mask[:, min(w, x_center + side_margin):] = 255

            occ = (
                (hair_mask > 20)
                & (side_mask > 0)
                & (alpha > 0)
            )

            alpha[occ] *= 0.35

            overlay[:, :, 3] = np.clip(
                alpha,
                0,
                255,
            ).astype(np.uint8)

    # =====================================================
    # SHADOW + BLEND
    # =====================================================

    result = add_soft_shadow(
        image_bgr,
        overlay[:, :, 3],
        dx=3,
        dy=4,
        blur=19,
        opacity=0.22,
    )

    result = alpha_blend_bgra(
        result,
        overlay,
    )

    return result