from __future__ import annotations

import os

import cv2
import numpy as np

from backend.accessories.base import (
    alpha_blend_rgba,
    clamp_glasses_side_arms,
    load_asset_meta,
    load_rgba_asset,
    warp_asset_affine,
)


def apply_glasses_v2(
    image_bgr: np.ndarray,
    analysis: dict,
    asset_path: str,
    width_scale: float = 1.0,
    y_offset_ratio: float = 0.0,
) -> np.ndarray:
    """
    Improved glasses placement using:
    - detected facial anchors
    - affine alignment
    - optional asset anchor metadata
    - simple arm clamping to reduce fake overlay look
    """

    if image_bgr is None:
        raise ValueError("image_bgr is None")

    if not os.path.exists(asset_path):
        raise FileNotFoundError(f"Glasses asset not found: {asset_path}")

    if "anchors" not in analysis or "glasses" not in analysis["anchors"]:
        raise ValueError("analysis['anchors']['glasses'] missing")

    anchor = analysis["anchors"]["glasses"]

    left_eye = np.array(anchor["left_eye"], dtype=np.float32)
    right_eye = np.array(anchor["right_eye"], dtype=np.float32)
    nose_bridge = np.array(anchor["nose_bridge"], dtype=np.float32)

    # 1. Retrieve temple distance from anchors (computed in build_glasses_anchor)
    temple_distance = float(anchor.get("temple_distance", 0.0))
    if temple_distance < 1.0:
        # Fallback: compute from landmarks
        landmarks = analysis.get("landmarks_2d")
        if landmarks is None:
            landmarks = analysis.get("landmarks")
        if landmarks is not None and len(landmarks) > 356:
            left_temple = np.array(landmarks[127], dtype=np.float32)
            right_temple = np.array(landmarks[356], dtype=np.float32)
            temple_distance = float(np.linalg.norm(right_temple - left_temple))
        else:
            temple_distance = float(np.linalg.norm(right_eye - left_eye)) * 2.0

    # 2. Load asset and retrieve asset anchors
    asset_rgba = load_rgba_asset(asset_path)
    meta = load_asset_meta(asset_path, asset_rgba.shape)

    src_left = np.array(meta["anchors"]["left_eye"], dtype=np.float32)
    src_right = np.array(meta["anchors"]["right_eye"], dtype=np.float32)
    src_bridge = np.array(meta["anchors"]["nose_bridge"], dtype=np.float32)

    # 3. Calculate target eye distance based on asset's eye-distance-to-width ratio
    asset_width = asset_rgba.shape[1]
    asset_eye_dist = np.linalg.norm(src_right - src_left)
    asset_eye_ratio = asset_eye_dist / max(1.0, asset_width)

    # 4. Calibrate the target eye coordinates along the original eye vector
    center = (left_eye + right_eye) / 2.0
    eye_vec = right_eye - left_eye
    eye_dist_current = np.linalg.norm(eye_vec)
    if eye_dist_current > 0.001:
        u = eye_vec / eye_dist_current
    else:
        u = np.array([1.0, 0.0], dtype=np.float32)

    # Use temple distance with the asset's eye-to-width ratio for proper sizing
    target_eye_distance = asset_eye_ratio * temple_distance * 1.05 * width_scale
    left_eye = center - u * (target_eye_distance / 2.0)
    right_eye = center + u * (target_eye_distance / 2.0)

    # 5. Position nose bridge and perform affine warp
    nose_bridge = nose_bridge + np.array([0.0, target_eye_distance * y_offset_ratio], dtype=np.float32)

    src_pts = np.array([src_left, src_right, src_bridge], dtype=np.float32)
    dst_pts = np.array([left_eye, right_eye, nose_bridge], dtype=np.float32)

    h, w = image_bgr.shape[:2]

    warped = warp_asset_affine(
        asset_rgba=asset_rgba,
        src_points=src_pts,
        dst_points=dst_pts,
        output_size=(w, h),
    )

    # 6. Combine skin and ears masks for side arm clamping to avoid premature cutting
    face_mask = None
    if "masks" in analysis:
        skin_mask = analysis["masks"].get("skin")
        ears_mask = analysis["masks"].get("ears")
        if skin_mask is not None and ears_mask is not None:
            face_mask = cv2.bitwise_or(skin_mask, ears_mask)
        elif skin_mask is not None:
            face_mask = skin_mask
        elif ears_mask is not None:
            face_mask = ears_mask

    warped = clamp_glasses_side_arms(
        warped_rgba=warped,
        face_mask=face_mask,
        extra_ratio=0.08,
    )

    result = alpha_blend_rgba(
        background_bgr=image_bgr,
        overlay_rgba=warped,
    )

    return result