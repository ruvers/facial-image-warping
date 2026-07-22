from __future__ import annotations

import cv2
import numpy as np


# =========================================================
# LANDMARK IDS
# =========================================================

LEFT_EYE_OUTER = 33
RIGHT_EYE_OUTER = 263
NOSE_TIP = 1
NOSE_BRIDGE = 168
CHIN = 152
LEFT_FACE_SIDE = 234
RIGHT_FACE_SIDE = 454
LEFT_EAR_APPROX = 234
RIGHT_EAR_APPROX = 454
LEFT_MOUTH = 61
RIGHT_MOUTH = 291


# =========================================================
# CAMERA
# =========================================================

def build_camera(image_shape) -> dict:
    """
    Simple pinhole camera approximation.

    This is not calibrated camera data.
    It is enough for 2.5D / future 3D projection logic.
    """

    h, w = image_shape[:2]

    focal = float(w)

    camera_matrix = np.array(
        [
            [focal, 0.0, w / 2.0],
            [0.0, focal, h / 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )

    return {
        "width": int(w),
        "height": int(h),
        "focal_length": focal,
        "cx": float(w / 2.0),
        "cy": float(h / 2.0),
        "camera_matrix": camera_matrix,
    }


# =========================================================
# DEPTH MAP FROM LANDMARK Z
# =========================================================

def sparse_landmark_depth_map(
    landmarks_3d: np.ndarray,
    image_shape,
    base_depth_map: np.ndarray | None = None,
) -> np.ndarray:
    """
    Build a soft pseudo depth map from MediaPipe landmark z.

    This is NOT true metric depth.
    It provides a smoother 2.5D depth signal for:
        - accessory occlusion
        - z-ordering
        - future 3D debug
    """

    h, w = image_shape[:2]

    if landmarks_3d is None or len(landmarks_3d) == 0:
        if base_depth_map is not None:
            return base_depth_map.astype(np.float32)

        return np.zeros((h, w), dtype=np.float32)

    z = landmarks_3d[:, 2].astype(np.float32)

    z_min = float(np.min(z))
    z_max = float(np.max(z))

    if abs(z_max - z_min) < 1e-6:
        z_norm = np.ones_like(z, dtype=np.float32) * 0.5
    else:
        z_norm = (z - z_min) / (z_max - z_min)

    # MediaPipe z sign can be unintuitive.
    # We normalize only for relative ordering, not physical distance.
    sparse = np.zeros((h, w), dtype=np.float32)
    weights = np.zeros((h, w), dtype=np.float32)

    radius = max(4, int(min(h, w) * 0.018))

    for p, depth_value in zip(landmarks_3d, z_norm):
        x = int(np.clip(p[0], 0, w - 1))
        y = int(np.clip(p[1], 0, h - 1))

        cv2.circle(
            sparse,
            (x, y),
            radius,
            float(depth_value),
            -1,
            cv2.LINE_AA,
        )

        cv2.circle(
            weights,
            (x, y),
            radius,
            1.0,
            -1,
            cv2.LINE_AA,
        )

    blur = max(15, int(min(h, w) * 0.08))
    if blur % 2 == 0:
        blur += 1

    sparse_blur = cv2.GaussianBlur(
        sparse,
        (blur, blur),
        0,
    )

    weights_blur = cv2.GaussianBlur(
        weights,
        (blur, blur),
        0,
    )

    depth = np.zeros_like(sparse_blur, dtype=np.float32)

    valid = weights_blur > 1e-5
    depth[valid] = sparse_blur[valid] / weights_blur[valid]

    if base_depth_map is not None:
        base = base_depth_map.astype(np.float32)

        if base.shape[:2] != depth.shape[:2]:
            base = cv2.resize(
                base,
                (w, h),
                interpolation=cv2.INTER_LINEAR,
            )

        # Combine semantic depth and landmark z depth.
        depth = 0.55 * base + 0.45 * depth

    depth = cv2.GaussianBlur(
        depth,
        (15, 15),
        0,
    )

    return depth.astype(np.float32)


# =========================================================
# 3D ANCHOR POINTS
# =========================================================

def _safe_point_3d(
    landmarks_3d: np.ndarray,
    idx: int,
) -> list[float]:
    if landmarks_3d is None or idx >= len(landmarks_3d):
        return [0.0, 0.0, 0.0]

    p = landmarks_3d[idx]

    return [
        float(p[0]),
        float(p[1]),
        float(p[2]),
    ]


def build_3d_anchor_points(
    landmarks_3d: np.ndarray,
) -> dict:
    """
    3D-ish anatomical anchor points.

    Later DECA/FLAME provider can replace these with true mesh vertices.
    """

    left_eye = np.array(
        _safe_point_3d(landmarks_3d, LEFT_EYE_OUTER),
        dtype=np.float32,
    )

    right_eye = np.array(
        _safe_point_3d(landmarks_3d, RIGHT_EYE_OUTER),
        dtype=np.float32,
    )

    eye_center = ((left_eye + right_eye) / 2.0).tolist()

    return {
        "nose_tip": _safe_point_3d(landmarks_3d, NOSE_TIP),
        "nose_bridge": _safe_point_3d(landmarks_3d, NOSE_BRIDGE),
        "chin": _safe_point_3d(landmarks_3d, CHIN),

        "left_eye_outer": _safe_point_3d(landmarks_3d, LEFT_EYE_OUTER),
        "right_eye_outer": _safe_point_3d(landmarks_3d, RIGHT_EYE_OUTER),
        "eye_center": [
            float(eye_center[0]),
            float(eye_center[1]),
            float(eye_center[2]),
        ],

        "left_face_side": _safe_point_3d(landmarks_3d, LEFT_FACE_SIDE),
        "right_face_side": _safe_point_3d(landmarks_3d, RIGHT_FACE_SIDE),

        "left_ear_approx": _safe_point_3d(landmarks_3d, LEFT_EAR_APPROX),
        "right_ear_approx": _safe_point_3d(landmarks_3d, RIGHT_EAR_APPROX),

        "left_mouth": _safe_point_3d(landmarks_3d, LEFT_MOUTH),
        "right_mouth": _safe_point_3d(landmarks_3d, RIGHT_MOUTH),
    }


# =========================================================
# MESH PLACEHOLDER
# =========================================================

def build_mediapipe_mesh_placeholder(
    landmarks_3d: np.ndarray,
) -> dict:
    """
    Current mesh placeholder.

    vertices:
        MediaPipe landmark vertices in image-space 3D.

    faces:
        Empty for now.
        Later:
            - MediaPipe tessellation faces
            - or FLAME mesh faces
    """

    if landmarks_3d is None:
        vertices = np.zeros((0, 3), dtype=np.float32)
    else:
        vertices = landmarks_3d.astype(np.float32)

    return {
        "type": "mediapipe_landmark_mesh",
        "vertices": vertices,
        "faces": np.zeros((0, 3), dtype=np.int32),
        "is_true_3d": False,
        "note": "This is MediaPipe image-space 3D, not DECA/FLAME metric mesh.",
    }


# =========================================================
# MAIN 3D ENRICHMENT
# =========================================================

def enrich_with_3d_context(
    ctx: dict,
    image_bgr: np.ndarray,
) -> dict:
    """
    Add 3D-ready fields to face context.

    This does not change existing keys.
    It only adds ctx['three_d'].

    Later replacement point:
        ctx['three_d']['provider'] = 'deca_flame'
    """

    if image_bgr is None:
        raise ValueError("image_bgr is None")

    landmarks_3d = ctx.get("landmarks_3d")
    base_depth = ctx.get("depth_map")

    camera = build_camera(
        image_bgr.shape,
    )

    depth_map_3d = sparse_landmark_depth_map(
        landmarks_3d=landmarks_3d,
        image_shape=image_bgr.shape,
        base_depth_map=base_depth,
    )

    anchor_points = build_3d_anchor_points(
        landmarks_3d,
    )

    mesh = build_mediapipe_mesh_placeholder(
        landmarks_3d,
    )

    ctx["three_d"] = {
        "provider": "mediapipe_pseudo_3d",
        "is_true_3d": False,

        "camera": camera,

        "mesh": mesh,

        "vertices": mesh["vertices"],
        "faces": mesh["faces"],

        "depth_map": depth_map_3d,
        "anchor_points": anchor_points,

        "future_provider_slots": {
            "deca": False,
            "flame": False,
            "true_depth": False,
            "metric_mesh": False,
        },
    }

    return ctx