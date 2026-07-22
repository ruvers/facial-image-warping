from __future__ import annotations

import cv2
import numpy as np

from backend.face_parsing import (
    parse_face,
    get_mask,
    subtract_mask,
)
from backend.face_detection import _resolve_face_landmarker_task_path

# =========================================================
# MEDIAPIPE
# =========================================================

_face_landmarker = None
_face_landmarker_error: str | None = None


def get_face_analysis_status() -> dict:
    return {
        "provider": "mediapipe_tasks_face_landmarker",
        "available": _face_landmarker_error is None,
        "lazy_loaded": _face_landmarker is not None,
        "error": _face_landmarker_error,
    }


def _get_face_landmarker():
    global _face_landmarker, _face_landmarker_error

    if _face_landmarker is not None:
        return _face_landmarker

    try:
        import mediapipe as mp

        model_path = _resolve_face_landmarker_task_path()
        model_buffer = model_path.read_bytes()

        base_options = mp.tasks.BaseOptions(
            model_asset_buffer=model_buffer,
        )

        options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=base_options,
            num_faces=1,
            min_face_detection_confidence=0.5,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )

        _face_landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(
            options,
        )
        _face_landmarker_error = None
        return _face_landmarker

    except Exception as exc:
        _face_landmarker_error = str(exc)
        raise RuntimeError(
            f"MediaPipe face analysis unavailable: {_face_landmarker_error}"
        ) from exc


def _iter_landmarks(face_landmarks):
    if face_landmarks is None:
        return []

    if hasattr(face_landmarks, "landmark"):
        return list(face_landmarks.landmark)

    if isinstance(face_landmarks, list):
        if len(face_landmarks) > 0 and hasattr(face_landmarks[0], "landmark"):
            return list(face_landmarks[0].landmark)

        return face_landmarks

    return []


def _landmark_xy(lm) -> tuple[float, float]:
    return float(lm.x), float(lm.y)


# =========================================================
# LANDMARK IDS
# =========================================================

LEFT_EYE = 33
RIGHT_EYE = 263

LEFT_EYE_INNER = 133
RIGHT_EYE_INNER = 362

NOSE_TIP = 1
NOSE_BRIDGE = 168

CHIN = 152

LEFT_MOUTH = 61
RIGHT_MOUTH = 291

LEFT_FACE_SIDE = 234
RIGHT_FACE_SIDE = 454


# =========================================================
# MASK REFINEMENT
# =========================================================

def keep_largest_component(mask: np.ndarray) -> np.ndarray:
    mask = (mask > 127).astype(np.uint8)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask,
        connectivity=8,
    )

    if num_labels <= 1:
        return (mask * 255).astype(np.uint8)

    largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])

    out = np.zeros_like(mask, dtype=np.uint8)
    out[labels == largest] = 255

    return out


def refine_mask(
    mask: np.ndarray,
    open_kernel: int = 3,
    close_kernel: int = 7,
    blur: int = 9,
    keep_largest: bool = False,
) -> np.ndarray:
    mask = (mask > 20).astype(np.uint8) * 255

    if keep_largest:
        mask = keep_largest_component(mask)

    if open_kernel > 0:
        k = np.ones((open_kernel, open_kernel), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)

    if close_kernel > 0:
        k = np.ones((close_kernel, close_kernel), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)

    if blur > 0:
        if blur % 2 == 0:
            blur += 1

        mask = cv2.GaussianBlur(
            mask,
            (blur, blur),
            0,
        )

    return mask


# =========================================================
# BEARD DETECTION
# =========================================================

def detect_beard_region_v2(
    image_bgr: np.ndarray,
    landmarks_2d: np.ndarray,
    skin_mask: np.ndarray,
    parsing: np.ndarray,
) -> np.ndarray:
    """
    Beard / moustache candidate detection.

    This is not a trained beard model.
    It is a hybrid heuristic:
    - landmark ROI
    - dark pixel detection
    - texture detection
    - excludes lips, eyes, eyebrows, nose
    """

    h, w = image_bgr.shape[:2]

    roi = np.zeros((h, w), dtype=np.uint8)

    lower_face_indices = [
        61, 291,
        361, 323, 454, 447, 365, 379, 378, 377,
        152,
        148, 149, 150, 136, 172, 58, 132,
    ]

    if len(landmarks_2d) > max(lower_face_indices):
        poly = np.array(
            [
                [landmarks_2d[i][0], landmarks_2d[i][1]]
                for i in lower_face_indices
            ],
            dtype=np.int32,
        )

        cv2.fillPoly(
            roi,
            [poly],
            255,
        )

    # Moustache zone: between nose and mouth
    if len(landmarks_2d) > 291:
        left_mouth = landmarks_2d[LEFT_MOUTH]
        right_mouth = landmarks_2d[RIGHT_MOUTH]
        nose = landmarks_2d[NOSE_TIP]

        x1 = max(
            0,
            min(left_mouth[0], right_mouth[0]) - 25,
        )

        x2 = min(
            w - 1,
            max(left_mouth[0], right_mouth[0]) + 25,
        )

        y1 = max(
            0,
            nose[1],
        )

        y2 = min(
            h - 1,
            max(left_mouth[1], right_mouth[1]) + 15,
        )

        roi[y1:y2, x1:x2] = 255

    hsv = cv2.cvtColor(
        image_bgr,
        cv2.COLOR_BGR2HSV,
    )

    lab = cv2.cvtColor(
        image_bgr,
        cv2.COLOR_BGR2LAB,
    )

    gray = cv2.cvtColor(
        image_bgr,
        cv2.COLOR_BGR2GRAY,
    )

    _, s, v = cv2.split(hsv)
    l, _, _ = cv2.split(lab)

    # Dark-ish beard candidates
    dark_v = v < 140
    dark_l = l < 155

    # Texture detection
    lap = cv2.Laplacian(
        gray,
        cv2.CV_64F,
    )

    texture = np.abs(lap) > 6

    # Local variance
    gray_f = gray.astype(np.float32)

    mean = cv2.blur(
        gray_f,
        (9, 9),
    )

    mean_sq = cv2.blur(
        gray_f ** 2,
        (9, 9),
    )

    variance = mean_sq - mean ** 2

    textured_var = variance > 35

    candidate = (
        (roi > 0)
        & (skin_mask > 20)
        & (dark_v | dark_l)
        & (texture | textured_var)
    )

    beard = candidate.astype(np.uint8) * 255

    # Exclude semantic regions
    exclude = np.isin(
        parsing,
        [
            2,   # left eyebrow
            3,   # right eyebrow
            4,   # left eye
            5,   # right eye
            10,  # nose
            12,  # upper lip
            13,  # lower lip
        ],
    ).astype(np.uint8) * 255

    beard[exclude > 0] = 0

    kernel_small = np.ones((3, 3), np.uint8)
    kernel_mid = np.ones((7, 7), np.uint8)

    beard = cv2.morphologyEx(
        beard,
        cv2.MORPH_OPEN,
        kernel_small,
    )

    beard = cv2.morphologyEx(
        beard,
        cv2.MORPH_CLOSE,
        kernel_mid,
    )

    beard = cv2.dilate(
        beard,
        kernel_small,
        iterations=1,
    )

    beard = cv2.GaussianBlur(
        beard,
        (7, 7),
        0,
    )

    return beard


# =========================================================
# HEAD POSE
# =========================================================

def rotation_vector_to_euler(rotation_vec: np.ndarray) -> dict:
    """
    Convert rotation vector to approximate pitch/yaw/roll degrees.
    """

    rotation_mat, _ = cv2.Rodrigues(rotation_vec)

    sy = np.sqrt(
        rotation_mat[0, 0] * rotation_mat[0, 0]
        + rotation_mat[1, 0] * rotation_mat[1, 0]
    )

    singular = sy < 1e-6

    if not singular:
        x = np.arctan2(
            rotation_mat[2, 1],
            rotation_mat[2, 2],
        )

        y = np.arctan2(
            -rotation_mat[2, 0],
            sy,
        )

        z = np.arctan2(
            rotation_mat[1, 0],
            rotation_mat[0, 0],
        )
    else:
        x = np.arctan2(
            -rotation_mat[1, 2],
            rotation_mat[1, 1],
        )

        y = np.arctan2(
            -rotation_mat[2, 0],
            sy,
        )

        z = 0

    return {
        "pitch": float(np.degrees(x)),
        "yaw": float(np.degrees(y)),
        "roll": float(np.degrees(z)),
    }


def estimate_head_pose(
    landmarks_2d: np.ndarray,
    image_shape,
) -> dict:
    h, w = image_shape[:2]

    # Sol ve sağ göz merkezleri (dış ve iç kenarların ortalaması)
    left_eye_center = (landmarks_2d[33].astype(np.float64) + landmarks_2d[133].astype(np.float64)) / 2.0
    right_eye_center = (landmarks_2d[263].astype(np.float64) + landmarks_2d[362].astype(np.float64)) / 2.0

    image_points = np.array(
        [
            landmarks_2d[NOSE_TIP],
            landmarks_2d[CHIN],
            left_eye_center,
            right_eye_center,
            landmarks_2d[LEFT_MOUTH],
            landmarks_2d[RIGHT_MOUTH],
        ],
        dtype=np.float64,
    ).reshape(-1, 1, 2)

    model_points = np.array(
        [
            (0.0, 0.0, 0.0),
            (0.0, -63.6, -12.5),
            (-43.3, 32.7, -26.0),
            (43.3, 32.7, -26.0),
            (-28.0, -28.0, -20.0),
            (28.0, -28.0, -20.0),
        ],
        dtype=np.float64,
    ).reshape(-1, 1, 3)

    focal_length = w

    camera_matrix = np.array(
        [
            [focal_length, 0, w / 2],
            [0, focal_length, h / 2],
            [0, 0, 1],
        ],
        dtype=np.float64,
    )

    dist_coeffs = np.zeros(
        (4, 1),
        dtype=np.float64,
    )

    success, rotation_vec, translation_vec = cv2.solvePnP(
        model_points,
        image_points,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )

    euler = {
        "pitch": 0.0,
        "yaw": 0.0,
        "roll": 0.0,
    }

    if success:
        euler = rotation_vector_to_euler(rotation_vec)

    return {
        "success": bool(success),
        "rotation_vector": rotation_vec,
        "translation_vector": translation_vec,
        "euler": euler,
    }


# =========================================================
# DEPTH MAP
# =========================================================

def create_depth_map(parsing: np.ndarray) -> np.ndarray:
    """
    Pseudo depth map.

    This is not true metric depth.
    It is a semantic depth approximation for accessories / occlusion.
    """

    depth = np.zeros_like(
        parsing,
        dtype=np.float32,
    )

    depth[parsing == 1] = 0.70       # skin
    depth[parsing == 10] = 1.00      # nose
    depth[parsing == 17] = 0.35      # hair
    depth[parsing == 14] = 0.20      # neck
    depth[(parsing == 7) | (parsing == 8)] = 0.50
    depth[(parsing == 12) | (parsing == 13)] = 0.75

    depth = cv2.GaussianBlur(
        depth,
        (15, 15),
        0,
    )

    return depth


# =========================================================
# ANCHORS
# =========================================================

def _pt(
    landmarks_2d: np.ndarray,
    idx: int,
) -> np.ndarray:
    return landmarks_2d[idx].astype(np.float32)


def _mask_bbox(mask: np.ndarray):
    if mask is None:
        return None

    ys, xs = np.where(mask > 20)

    if len(xs) == 0 or len(ys) == 0:
        return None

    return {
        "x1": int(xs.min()),
        "y1": int(ys.min()),
        "x2": int(xs.max()),
        "y2": int(ys.max()),
        "cx": int(xs.mean()),
        "cy": int(ys.mean()),
        "w": int(xs.max() - xs.min()),
        "h": int(ys.max() - ys.min()),
    }


def _roll_angle(
    left: np.ndarray,
    right: np.ndarray,
) -> float:
    v = right - left

    return float(
        np.degrees(
            np.arctan2(
                v[1],
                v[0],
            )
        )
    )


def build_glasses_anchor(
    landmarks_2d: np.ndarray,
) -> dict:
    # Outer eye corners (used for roll and distance measurements)
    left_eye_outer = _pt(
        landmarks_2d,
        LEFT_EYE,
    )

    right_eye_outer = _pt(
        landmarks_2d,
        RIGHT_EYE,
    )

    # Inner eye corners
    left_eye_inner = _pt(
        landmarks_2d,
        LEFT_EYE_INNER,
    )

    right_eye_inner = _pt(
        landmarks_2d,
        RIGHT_EYE_INNER,
    )

    # True eye centers: average of inner and outer corners per eye
    left_eye_center = (left_eye_outer + left_eye_inner) / 2.0
    right_eye_center = (right_eye_outer + right_eye_inner) / 2.0

    # Use iris landmarks if available for even more accurate centers
    if len(landmarks_2d) > 473:
        left_iris = _pt(landmarks_2d, 468)
        right_iris = _pt(landmarks_2d, 473)
        # Blend: 70% iris + 30% corner-average for stability
        left_eye_center = left_iris * 0.7 + left_eye_center * 0.3
        right_eye_center = right_iris * 0.7 + right_eye_center * 0.3

    eye_midpoint = (left_eye_center + right_eye_center) / 2.0

    eye_dist = float(
        np.linalg.norm(
            right_eye_outer - left_eye_outer,
        )
    )

    face_left = _pt(
        landmarks_2d,
        LEFT_FACE_SIDE,
    )

    face_right = _pt(
        landmarks_2d,
        RIGHT_FACE_SIDE,
    )

    face_width = float(
        np.linalg.norm(
            face_right - face_left,
        )
    )

    nose_bridge = _pt(
        landmarks_2d,
        NOSE_BRIDGE,
    )

    # Temple points (near the ears) — landmarks 127 (left) and 356 (right)
    left_temple = _pt(landmarks_2d, 127)
    right_temple = _pt(landmarks_2d, 356)
    temple_distance = float(
        np.linalg.norm(
            right_temple - left_temple,
        )
    )

    # Center X: midpoint of eyes
    # Center Y: almost at eye level, with a tiny nose-bridge nudge
    #           so glasses sit right on the eyes, not above them
    center = eye_midpoint.copy()
    center[1] = (
        eye_midpoint[1] * 0.93
        + nose_bridge[1] * 0.07
    )

    # Width: use temple distance so the glasses edges align with the ears.
    # Fallback to face_width or eye_dist if temple distance is too small.
    glasses_width = max(
        temple_distance * 1.05,
        face_width * 0.82,
        eye_dist * 2.20,
    )

    # Roll: computed from true eye centers (not just outer corners)
    # for more accurate face-angle tracking.
    # Negated because cv2.getRotationMatrix2D rotates CCW for positive angles
    # in image coords (y-down), which is opposite to the geometric eye-line angle.
    roll = -_roll_angle(
        left_eye_center,
        right_eye_center,
    )

    return {
        "center": (
            float(center[0]),
            float(center[1]),
        ),
        "eye_distance": eye_dist,
        "face_width": face_width,
        "temple_distance": temple_distance,
        "width": glasses_width,
        "roll_deg": roll,
        "left_eye": (
            float(left_eye_center[0]),
            float(left_eye_center[1]),
        ),
        "right_eye": (
            float(right_eye_center[0]),
            float(right_eye_center[1]),
        ),
        "nose_bridge": (
            float(nose_bridge[0]),
            float(nose_bridge[1]),
        ),
        "left_temple": (
            float(left_temple[0]),
            float(left_temple[1]),
        ),
        "right_temple": (
            float(right_temple[0]),
            float(right_temple[1]),
        ),
    }


def build_earring_anchors(
    landmarks_2d: np.ndarray,
    masks: dict[str, np.ndarray],
) -> dict:
    if len(landmarks_2d) > 323:
        left_earlobe = _pt(
            landmarks_2d,
            93,
        ).astype(np.float64)
        right_earlobe = _pt(
            landmarks_2d,
            323,
        ).astype(np.float64)

        v = right_earlobe - left_earlobe
        dist = np.linalg.norm(v)
        if dist > 0:
            outward_shift = (v / dist) * (dist * 0.035)
            left_earlobe -= outward_shift
            right_earlobe += outward_shift

        return {
            "left": (
                float(left_earlobe[0]),
                float(left_earlobe[1]),
            ),
            "right": (
                float(right_earlobe[0]),
                float(right_earlobe[1]),
            ),
        }

    ears_mask = masks.get("ears")

    left_anchor = None
    right_anchor = None

    if ears_mask is not None:
        h, w = ears_mask.shape[:2]

        left_half = np.zeros_like(ears_mask)
        right_half = np.zeros_like(ears_mask)

        left_half[:, : w // 2] = ears_mask[:, : w // 2]
        right_half[:, w // 2 :] = ears_mask[:, w // 2 :]

        left_box = _mask_bbox(left_half)
        right_box = _mask_bbox(right_half)

        if left_box:
            left_anchor = (
                float(left_box["cx"]),
                float(left_box["y2"] - left_box["h"] * 0.18),
            )

        if right_box:
            right_anchor = (
                float(right_box["cx"]),
                float(right_box["y2"] - right_box["h"] * 0.18),
            )

    if left_anchor is None:
        p = _pt(
            landmarks_2d,
            LEFT_FACE_SIDE,
        )

        left_anchor = (
            float(p[0] - 12),
            float(p[1] + 45),
        )

    if right_anchor is None:
        p = _pt(
            landmarks_2d,
            RIGHT_FACE_SIDE,
        )

        right_anchor = (
            float(p[0] + 12),
            float(p[1] + 45),
        )

    return {
        "left": left_anchor,
        "right": right_anchor,
    }


def build_necklace_anchor(
    landmarks_2d: np.ndarray,
    masks: dict[str, np.ndarray],
) -> dict:
    neck_mask = masks.get("neck")

    neck_box = _mask_bbox(
        neck_mask,
    )

    chin = _pt(
        landmarks_2d,
        CHIN,
    )

    face_left = _pt(
        landmarks_2d,
        LEFT_FACE_SIDE,
    )

    face_right = _pt(
        landmarks_2d,
        RIGHT_FACE_SIDE,
    )

    face_width = float(
        np.linalg.norm(
            face_right - face_left,
        )
    )

    if neck_box:
        center = (
            float(neck_box["cx"]),
            float(neck_box["y1"] + neck_box["h"] * 0.45),
        )

        width = float(
            neck_box["w"] * 1.15,
        )
    else:
        center = (
            float(chin[0]),
            float(chin[1] + face_width * 0.32),
        )

        width = face_width * 0.72

    return {
        "center": center,
        "width": width,
        "face_width": face_width,
        "chin": (
            float(chin[0]),
            float(chin[1]),
        ),
    }


def build_face_metrics(
    landmarks_2d: np.ndarray,
) -> dict:
    left_face = _pt(
        landmarks_2d,
        LEFT_FACE_SIDE,
    )

    right_face = _pt(
        landmarks_2d,
        RIGHT_FACE_SIDE,
    )

    chin = _pt(
        landmarks_2d,
        CHIN,
    )

    nose = _pt(
        landmarks_2d,
        NOSE_TIP,
    )

    face_width = float(
        np.linalg.norm(
            right_face - left_face,
        )
    )

    face_height = float(
        np.linalg.norm(
            chin - nose,
        )
    )

    return {
        "face_width": face_width,
        "face_height": face_height,
        "face_center": (
            float((left_face[0] + right_face[0]) / 2),
            float((nose[1] + chin[1]) / 2),
        ),
    }


def build_hat_anchor(
    landmarks_2d: np.ndarray,
) -> dict:
    left_eye = _pt(
        landmarks_2d,
        LEFT_EYE,
    )
    right_eye = _pt(
        landmarks_2d,
        RIGHT_EYE,
    )
    face_left = _pt(
        landmarks_2d,
        LEFT_FACE_SIDE,
    )
    face_right = _pt(
        landmarks_2d,
        RIGHT_FACE_SIDE,
    )
    nose_bridge = _pt(
        landmarks_2d,
        NOSE_BRIDGE,
    )

    face_width = float(
        np.linalg.norm(
            face_right - face_left,
        )
    )

    if len(landmarks_2d) > 10:
        forehead = _pt(
            landmarks_2d,
            10,
        )
    else:
        forehead = nose_bridge.copy()
        forehead[1] -= face_width * 0.35

    center_x = float((face_left[0] + face_right[0]) / 2.0)
    center_y = float(forehead[1] - face_width * 0.12)

    return {
        "center": (
            center_x,
            center_y,
        ),
        "width": face_width * 1.35,
        "roll_deg": _roll_angle(
            left_eye,
            right_eye,
        ),
    }


def build_hair_clip_anchor(
    landmarks_2d: np.ndarray,
) -> dict:
    left_eye = _pt(
        landmarks_2d,
        LEFT_EYE,
    )
    right_eye = _pt(
        landmarks_2d,
        RIGHT_EYE,
    )
    temple = _pt(
        landmarks_2d,
        162,
    )
    forehead = _pt(
        landmarks_2d,
        10,
    )

    clip_center = temple + (temple - forehead) * 0.2
    face_left = _pt(
        landmarks_2d,
        LEFT_FACE_SIDE,
    )
    face_right = _pt(
        landmarks_2d,
        RIGHT_FACE_SIDE,
    )
    face_width = float(
        np.linalg.norm(
            face_right - face_left,
        )
    )

    return {
        "center": (
            float(clip_center[0]),
            float(clip_center[1]),
        ),
        "width": face_width * 0.5,
        "roll_deg": _roll_angle(
            left_eye,
            right_eye,
        ),
    }


def build_anchors(
    landmarks_2d: np.ndarray,
    landmarks_3d: np.ndarray,
    masks: dict[str, np.ndarray],
    image_shape,
) -> dict:
    return {
        "metrics": build_face_metrics(
            landmarks_2d,
        ),
        "glasses": build_glasses_anchor(
            landmarks_2d,
        ),
        "earrings": build_earring_anchors(
            landmarks_2d,
            masks,
        ),
        "necklace": build_necklace_anchor(
            landmarks_2d,
            masks,
        ),
        "hat": build_hat_anchor(
            landmarks_2d,
        ),
        "hair_clip": build_hair_clip_anchor(
            landmarks_2d,
        ),
    }


# =========================================================
# MAIN PIPELINE
# =========================================================

def analyze_face(image_bgr: np.ndarray) -> dict:
    if image_bgr is None:
        raise ValueError("Input image is None")

    rgb = cv2.cvtColor(
        image_bgr,
        cv2.COLOR_BGR2RGB,
    )

    landmarker = _get_face_landmarker()

    import mediapipe as mp

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb,
    )

    results = landmarker.detect(mp_image)

    if not results.face_landmarks:
        raise RuntimeError("No face detected")

    face_landmarks = results.face_landmarks[0]

    h, w = image_bgr.shape[:2]

    landmarks_2d = np.array(
        [
            [
                int(np.clip(_landmark_xy(lm)[0] * w, 0, w - 1)),
                int(np.clip(_landmark_xy(lm)[1] * h, 0, h - 1)),
            ]
            for lm in _iter_landmarks(face_landmarks)
        ],
        dtype=np.int32,
    )

    landmarks_3d = np.array(
        [
            [
                float(np.clip(_landmark_xy(lm)[0] * w, 0, w - 1)),
                float(np.clip(_landmark_xy(lm)[1] * h, 0, h - 1)),
                float(lm.z * w),
            ]
            for lm in _iter_landmarks(face_landmarks)
        ],
        dtype=np.float32,
    )

    parsing = parse_face(rgb)

    # Raw masks
    raw_skin = get_mask(parsing, [1])
    raw_hair = get_mask(parsing, [17])
    raw_lips = get_mask(parsing, [12, 13])
    raw_eyes = get_mask(parsing, [4, 5])
    raw_eyebrows = get_mask(parsing, [2, 3])
    raw_ears = get_mask(parsing, [7, 8])
    raw_neck = get_mask(parsing, [14])
    raw_nose = get_mask(parsing, [10])

    beard_mask = detect_beard_region_v2(
        image_bgr=image_bgr,
        landmarks_2d=landmarks_2d,
        skin_mask=raw_skin,
        parsing=parsing,
    )

    beard_mask = refine_mask(
        beard_mask,
        open_kernel=2,
        close_kernel=7,
        blur=7,
        keep_largest=False,
    )

    # Remove beard from skin / hair
    skin_clean = subtract_mask(
        raw_skin,
        beard_mask,
    )

    hair_clean = subtract_mask(
        raw_hair,
        beard_mask,
    )

    # Skin effect mask = skin without lips/eyes/eyebrows/beard
    skin_effect = skin_clean.copy()

    for remove in [
        raw_lips,
        raw_eyes,
        raw_eyebrows,
        beard_mask,
    ]:
        skin_effect = subtract_mask(
            skin_effect,
            remove,
        )

    skin_effect = refine_mask(
        skin_effect,
        open_kernel=3,
        close_kernel=9,
        blur=15,
        keep_largest=True,
    )

    masks = {
        "skin": refine_mask(
            skin_clean,
            open_kernel=3,
            close_kernel=9,
            blur=11,
            keep_largest=True,
        ),
        "skin_effect": skin_effect,
        "hair": refine_mask(
            hair_clean,
            open_kernel=3,
            close_kernel=9,
            blur=15,
            keep_largest=True,
        ),
        "lips": refine_mask(
            raw_lips,
            open_kernel=2,
            close_kernel=5,
            blur=7,
            keep_largest=False,
        ),
        "eyes": refine_mask(
            raw_eyes,
            open_kernel=1,
            close_kernel=3,
            blur=5,
            keep_largest=False,
        ),
        "eyebrows": refine_mask(
            raw_eyebrows,
            open_kernel=1,
            close_kernel=5,
            blur=5,
            keep_largest=False,
        ),
        "ears": refine_mask(
            raw_ears,
            open_kernel=2,
            close_kernel=5,
            blur=7,
            keep_largest=False,
        ),
        "neck": refine_mask(
            raw_neck,
            open_kernel=3,
            close_kernel=11,
            blur=11,
            keep_largest=True,
        ),
        "nose": refine_mask(
            raw_nose,
            open_kernel=2,
            close_kernel=5,
            blur=7,
            keep_largest=False,
        ),
        "beard": beard_mask,
    }

    depth_map = create_depth_map(
        parsing,
    )

    pose = estimate_head_pose(
        landmarks_2d,
        image_bgr.shape,
    )

    anchors = build_anchors(
        landmarks_2d=landmarks_2d,
        landmarks_3d=landmarks_3d,
        masks=masks,
        image_shape=image_bgr.shape,
    )

    return {
        # legacy compatibility
        "landmarks": landmarks_2d,

        # new standardized outputs
        "landmarks_2d": landmarks_2d,
        "landmarks_3d": landmarks_3d,

        "parsing": parsing,
        "masks": masks,
        "depth_map": depth_map,
        "pose": pose,
        "anchors": anchors,
    }


def detect_face_landmarks(image_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Lightweight face landmark detector (only runs MediaPipe, skips face parsing).
    Returns (landmarks_2d, landmarks_3d) or None.
    """
    try:
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        landmarker = _get_face_landmarker()
        import mediapipe as mp
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        results = landmarker.detect(mp_image)
        if not results.face_landmarks:
            return None
        face_landmarks = results.face_landmarks[0]
        h, w = image_bgr.shape[:2]
        landmarks_2d = np.array([
            [
                int(np.clip(_landmark_xy(lm)[0] * w, 0, w - 1)),
                int(np.clip(_landmark_xy(lm)[1] * h, 0, h - 1)),
            ]
            for lm in _iter_landmarks(face_landmarks)
        ], dtype=np.int32)
        landmarks_3d = np.array([
            [
                float(np.clip(_landmark_xy(lm)[0] * w, 0, w - 1)),
                float(np.clip(_landmark_xy(lm)[1] * h, 0, h - 1)),
                float(lm.z * w),
            ]
            for lm in _iter_landmarks(face_landmarks)
        ], dtype=np.float32)
        return landmarks_2d, landmarks_3d
    except Exception:
        return None

