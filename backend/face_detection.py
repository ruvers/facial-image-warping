"""
FaceWarp Lab — Face detection interface and stub.

Provides the abstract contract (FaceDetector) that Group 2 will implement
with MediaPipe Face Mesh. The StubFaceDetector returns pending_group_2
status with null bbox/landmarks — no fake detections are produced.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict

import mediapipe as mp
import numpy as np


_EXPECTED_SHAPE = (512, 512, 3)


# ── Input validation ─────────────────────────────────────────────────────────


def validate_face_detection_input(image: np.ndarray) -> None:
    """Validate that *image* is a 512x512 3-channel array.

    Raises:
        TypeError: if *image* is not an ndarray.
        ValueError: if *image* is None, empty, or has an unexpected shape.
    """

    if image is None:
        raise ValueError("Input image is None.")

    if not isinstance(image, np.ndarray):
        raise TypeError(
            f"Expected np.ndarray, got {type(image).__name__}."
        )

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(
            f"Expected image shape (H, W, 3), got {image.shape}."
        )

    h, w = image.shape[:2]

    if h != _EXPECTED_SHAPE[0] or w != _EXPECTED_SHAPE[1]:
        raise ValueError(
            f"Expected image shape {_EXPECTED_SHAPE}, got {image.shape}."
        )


def _resolve_face_landmarker_task_path() -> Path:
    """
    Resolve face_landmarker.task safely.

    We return a Path, but MediaPipe receives the file as bytes.
    This avoids Windows path bugs such as:
        site-packages/D:\\...\\face_landmarker.task
    """

    project_root = Path(__file__).resolve().parent.parent

    candidates = [
        project_root / "face_landmarker.task",
        project_root / "models" / "face_landmarker.task",
        project_root / "assets" / "face_landmarker.task",
        project_root / "backend" / "models" / "face_landmarker.task",
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    searched = "\n".join(str(path) for path in candidates)

    raise FileNotFoundError(
        "face_landmarker.task not found. Searched:\n" + searched
    )


# ── Abstract interface ────────────────────────────────────────────────────────


class FaceDetector(ABC):
    @abstractmethod
    def detect(self, image: np.ndarray) -> Dict[str, Any]:
        ...

    @abstractmethod
    def detect_landmarks(self, image: np.ndarray) -> Dict[str, Any]:
        ...


# ── Stub implementation ──────────────────────────────────────────────────────


class StubFaceDetector(FaceDetector):
    """Placeholder detector that returns pending_group_2 status.

    No fake bounding boxes or landmarks are generated. If the input
    image fails validation the response status is set to "failed"
    instead of raising an exception.
    """

    def detect(self, image: np.ndarray) -> Dict[str, Any]:
        """Return a pending stub response with no real detection."""

        try:
            validate_face_detection_input(image)

        except (TypeError, ValueError) as exc:
            return {
                "enabled": True,
                "status": "failed",
                "bbox": None,
                "confidence": None,
                "message": f"Invalid face detection input: {exc}",
            }

        return {
            "enabled": True,
            "status": "pending_group_2",
            "bbox": None,
            "confidence": None,
            "message": (
                "Face detection interface is ready; "
                "implementation will be provided by Group 2."
            ),
        }

    def detect_landmarks(self, image: np.ndarray) -> Dict[str, Any]:
        """Return a pending stub response with no real landmarks."""

        try:
            validate_face_detection_input(image)

        except (TypeError, ValueError) as exc:
            return {
                "enabled": True,
                "status": "failed",
                "count": 0,
                "coordinate_space": "processed_512",
                "points": None,
                "message": f"Invalid landmark detection input: {exc}",
            }

        return {
            "enabled": True,
            "status": "pending_group_2",
            "count": 0,
            "coordinate_space": "processed_512",
            "points": None,
            "message": (
                "Landmark detection interface is ready; "
                "implementation will be provided by Group 2."
            ),
        }


class MediaPipeFaceDetector(FaceDetector):
    def __init__(self):
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

        self.face_mesh = mp.tasks.vision.FaceLandmarker.create_from_options(
            options,
        )

    def detect(self, image: np.ndarray) -> Dict[str, Any]:
        try:
            validate_face_detection_input(image)

        except (TypeError, ValueError) as exc:
            return {
                "enabled": True,
                "status": "failed",
                "bbox": None,
                "confidence": None,
                "message": str(exc),
            }

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=image,
        )

        results = self.face_mesh.detect(mp_image)

        if not results.face_landmarks:
            return {
                "enabled": True,
                "status": "failed",
                "bbox": None,
                "confidence": None,
                "message": "No face detected in the image.",
            }

        landmarks = results.face_landmarks[0]
        h, w = image.shape[:2]

        xs = [lm.x * w for lm in landmarks]
        ys = [lm.y * h for lm in landmarks]

        x_min = float(min(xs))
        y_min = float(min(ys))
        x_max = float(max(xs))
        y_max = float(max(ys))

        visibility = getattr(
            landmarks[0],
            "visibility",
            None,
        )

        return {
            "enabled": True,
            "status": "completed",
            "bbox": {
                "x": x_min,
                "y": y_min,
                "width": x_max - x_min,
                "height": y_max - y_min,
                "coordinate_space": "processed_512",
            },
            "confidence": float(visibility)
            if visibility is not None
            else 1.0,
            "message": "Face detected successfully.",
        }

    def detect_landmarks(self, image: np.ndarray) -> Dict[str, Any]:
        try:
            validate_face_detection_input(image)

        except (TypeError, ValueError) as exc:
            return {
                "enabled": True,
                "status": "failed",
                "count": 0,
                "coordinate_space": "processed_512",
                "points": None,
                "message": str(exc),
            }

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=image,
        )

        results = self.face_mesh.detect(mp_image)

        if not results.face_landmarks:
            return {
                "enabled": True,
                "status": "failed",
                "count": 0,
                "coordinate_space": "processed_512",
                "points": None,
                "message": "No face detected, landmarks could not be extracted.",
            }

        landmarks = results.face_landmarks[0]
        h, w = image.shape[:2]

        points = []

        for idx, lm in enumerate(landmarks):
            visibility = getattr(
                lm,
                "visibility",
                None,
            )

            points.append(
                {
                    "index": idx,
                    "x": float(lm.x * w),
                    "y": float(lm.y * h),
                    "z": float(lm.z),
                    "visibility": float(visibility)
                    if visibility is not None
                    else 1.0,
                }
            )

        return {
            "enabled": True,
            "status": "completed",
            "count": len(points),
            "coordinate_space": "processed_512",
            "points": points,
            "message": f"{len(points)} landmarks detected.",
        }


# ── Factory ───────────────────────────────────────────────────────────────────


def get_face_detector() -> FaceDetector:
    return MediaPipeFaceDetector()