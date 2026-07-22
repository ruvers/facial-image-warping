from __future__ import annotations

import cv2
import numpy as np


# MediaPipe landmark ids
LEFT_EYE_OUTER = 33
RIGHT_EYE_OUTER = 263

LEFT_EYE_INNER = 133
RIGHT_EYE_INNER = 362

NOSE_BRIDGE = 168
NOSE_TIP = 1
CHIN = 152

LEFT_FACE_SIDE = 234
RIGHT_FACE_SIDE = 454

LEFT_MOUTH = 61
RIGHT_MOUTH = 291


def _pt(landmarks: np.ndarray, idx: int) -> np.ndarray:
    return landmarks[idx].astype(np.float32)


def _mask_bbox(mask: np.ndarray):
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


def _roll_angle(left: np.ndarray, right: np.ndarray) -> float:
    v = right - left
    return float(np.degrees(np.arctan2(v[1], v[0])))


def build_glasses_anchor(
    landmarks_2d: np.ndarray,
) -> dict:
    left_eye = _pt(landmarks_2d, LEFT_EYE_OUTER)
    right_eye = _pt(landmarks_2d, RIGHT_EYE_OUTER)

    eye_center = (left_eye + right_eye) / 2.0
    eye_dist = float(np.linalg.norm(right_eye - left_eye))

    face_left = _pt(landmarks_2d, LEFT_FACE_SIDE)
    face_right = _pt(landmarks_2d, RIGHT_FACE_SIDE)
    face_width = float(np.linalg.norm(face_right - face_left))

    nose_bridge = _pt(landmarks_2d, NOSE_BRIDGE)

    # Better than raw eye center: bridge-aware vertical placement
    center = eye_center.copy()
    center[1] = eye_center[1] * 0.75 + nose_bridge[1] * 0.25

    return {
        "center": (float(center[0]), float(center[1])),
        "eye_distance": eye_dist,
        "face_width": face_width,
        "width": max(eye_dist * 2.35, face_width * 0.78),
        "roll_deg": _roll_angle(left_eye, right_eye),
        "left_eye": (float(left_eye[0]), float(left_eye[1])),
        "right_eye": (float(right_eye[0]), float(right_eye[1])),
        "nose_bridge": (float(nose_bridge[0]), float(nose_bridge[1])),
    }


def build_earring_anchors(
    landmarks_2d: np.ndarray,
    masks: dict[str, np.ndarray],
) -> dict:
    if len(landmarks_2d) > 323:
        left_earlobe = _pt(landmarks_2d, 93).astype(np.float64)
        right_earlobe = _pt(landmarks_2d, 323).astype(np.float64)

        v = right_earlobe - left_earlobe
        dist = np.linalg.norm(v)
        if dist > 0:
            outward_shift = (v / dist) * (dist * 0.035)
            left_earlobe -= outward_shift
            right_earlobe += outward_shift

        return {
            "left": (float(left_earlobe[0]), float(left_earlobe[1])),
            "right": (float(right_earlobe[0]), float(right_earlobe[1])),
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

    # Fallback approximate points
    if left_anchor is None:
        p = _pt(landmarks_2d, 234)
        left_anchor = (float(p[0] - 12), float(p[1] + 45))

    if right_anchor is None:
        p = _pt(landmarks_2d, 454)
        right_anchor = (float(p[0] + 12), float(p[1] + 45))

    return {
        "left": left_anchor,
        "right": right_anchor,
    }


def build_necklace_anchor(
    landmarks_2d: np.ndarray,
    masks: dict[str, np.ndarray],
) -> dict:
    neck_mask = masks.get("neck")

    neck_box = _mask_bbox(neck_mask) if neck_mask is not None else None

    chin = _pt(landmarks_2d, CHIN)
    face_left = _pt(landmarks_2d, LEFT_FACE_SIDE)
    face_right = _pt(landmarks_2d, RIGHT_FACE_SIDE)

    face_width = float(np.linalg.norm(face_right - face_left))

    if neck_box:
        center = (
            float(neck_box["cx"]),
            float(neck_box["y1"] + neck_box["h"] * 0.45),
        )

        width = float(neck_box["w"] * 1.15)
    else:
        center = (
            float(chin[0]),
            float(chin[1] + face_width * 0.32),
        )

        width = face_width * 0.72

    return {
        "center": center,
        "width": width,
        "chin": (float(chin[0]), float(chin[1])),
    }


def build_face_metrics(
    landmarks_2d: np.ndarray,
) -> dict:
    left_face = _pt(landmarks_2d, LEFT_FACE_SIDE)
    right_face = _pt(landmarks_2d, RIGHT_FACE_SIDE)
    chin = _pt(landmarks_2d, CHIN)
    nose = _pt(landmarks_2d, NOSE_TIP)

    face_width = float(np.linalg.norm(right_face - left_face))
    face_height = float(np.linalg.norm(chin - nose))

    return {
        "face_width": face_width,
        "face_height": face_height,
        "face_center": (
            float((left_face[0] + right_face[0]) / 2),
            float((nose[1] + chin[1]) / 2),
        ),
    }


def build_anchors(
    landmarks_2d: np.ndarray,
    landmarks_3d: np.ndarray,
    masks: dict[str, np.ndarray],
    image_shape,
) -> dict:
    return {
        "metrics": build_face_metrics(landmarks_2d),
        "glasses": build_glasses_anchor(landmarks_2d),
        "earrings": build_earring_anchors(landmarks_2d, masks),
        "necklace": build_necklace_anchor(landmarks_2d, masks),
    }