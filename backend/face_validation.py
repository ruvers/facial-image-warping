"""
FaceWarp Lab — Image validation and face-detection gate.

Provides two public entry points used by main.py **before**
``preprocess_pipeline`` is called:

    validate_image_input(image, filename) -> ValidationResult
    check_face_orientation(image)         -> FaceOrientationResult

Both return dataclass instances that carry a boolean ``ok`` flag,
a human-readable ``message`` suitable for surfacing directly in the
API response, and optional detail fields for programmatic use.

Design goals (SRS FR-1.1.2, FR-1.2.1, NFR-2.1; SDD §4.1, §9):
  • Validate format, resolution, and image integrity before any DSP work.
  • Detect whether a face is present and roughly frontal.
  • Return structured error codes so the frontend can localise messages.
  • Never raise unhandled exceptions — all errors are captured into the
    result dataclass so the caller decides whether to abort or warn.

MediaPipe integration
---------------------
Face detection and landmark extraction are delegated entirely to
``face_detection.get_face_detector()`` (MediaPipeFaceDetector).  This
module contains **no direct MediaPipe calls**, which means:
  • A single MediaPipe model instance is shared across the codebase.
  • The Tasks-API / Solutions-API choice lives in one place only.
  • face_detection.py remains the single source of truth for landmarks.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from backend.face_detection import get_face_detector

# ── Constants ────────────────────────────────────────────────────────────────

# SDD §4.1 / §9: hard minimum and soft recommendation
MIN_DIMENSION_HARD: int = 100   # reject below this (SDD §9 test case)
MIN_DIMENSION_SOFT: int = 200   # warn below this (SDD §4.1 input spec)

SUPPORTED_EXTENSIONS: frozenset[str] = frozenset(
    {".jpg", ".jpeg", ".png"}
)

# Yaw / pitch limits for "frontal" classification (degrees).
# Values outside these ranges → "not frontal" warning.
YAW_FRONTAL_MAX_DEG:   float = 30.0   # left/right turn
PITCH_FRONTAL_MAX_DEG: float = 25.0   # up/down tilt

# Face Mesh landmark indices used for orientation estimation.
# These are stable across the 468-point MediaPipe topology and are
# referenced by numeric index into the ``points`` list returned by
# face_detection.MediaPipeFaceDetector.detect_landmarks().
_NOSE_TIP:        int = 1
_NOSE_BASE:       int = 168   # midpoint between eyes, nose bridge
_LEFT_EYE_OUTER:  int = 33
_RIGHT_EYE_OUTER: int = 263
_CHIN:            int = 152
_FOREHEAD:        int = 10


# ── Result dataclasses ────────────────────────────────────────────────────────

@dataclass
class ValidationResult:
    """Outcome of ``validate_image_input``.

    Attributes
    ----------
    ok : bool
        ``True`` only when the image is safe to pass to the preprocessing
        pipeline.  ``False`` means processing must be aborted.
    message : str
        Human-readable explanation shown directly in the API response.
    error_code : str
        Machine-readable tag for frontend localisation, e.g.
        ``"UNSUPPORTED_FORMAT"``, ``"RESOLUTION_TOO_LOW"``,
        ``"IMAGE_CORRUPTED"``, ``"OK"``.
    width : int | None
        Detected image width in pixels (``None`` when image is unreadable).
    height : int | None
        Detected image height in pixels (``None`` when image is unreadable).
    """
    ok:         bool
    message:    str
    error_code: str = "OK"
    width:      Optional[int] = None
    height:     Optional[int] = None


@dataclass
class FaceOrientationResult:
    """Outcome of ``check_face_orientation``.

    Attributes
    ----------
    ok : bool
        ``True`` when a frontal face was detected and the pipeline may
        continue.  ``False`` means the caller should abort (no face) or
        warn (non-frontal face), depending on ``error_code``.
    message : str
        Human-readable explanation.
    error_code : str
        One of:
          ``"OK"``               – frontal face detected, all good.
          ``"NO_FACE_DETECTED"`` – MediaPipe found no face at all.
          ``"FACE_NOT_FRONTAL"`` – face found but rotated beyond thresholds.
          ``"DETECTION_ERROR"``  – unexpected runtime error during detection.
    yaw_deg : float | None
        Estimated horizontal rotation in degrees (positive = turned right).
    pitch_deg : float | None
        Estimated vertical rotation in degrees (positive = tilted down).
    face_count : int
        Number of faces MediaPipe detected (may be > 1).
    """
    ok:         bool
    message:    str
    error_code: str             = "OK"
    yaw_deg:    Optional[float] = None
    pitch_deg:  Optional[float] = None
    face_count: int             = 0


# ── Internal helpers ──────────────────────────────────────────────────────────

def _estimate_yaw_pitch(
    points: List[Dict],
) -> Tuple[float, float]:
    """Estimate yaw and pitch from ``detect_landmarks()`` point dicts.

    Consumes the ``points`` list returned by
    ``face_detection.MediaPipeFaceDetector.detect_landmarks()``.
    Each element is a dict with keys ``"index"``, ``"x"``, ``"y"``, ``"z"``.
    Coordinates are already in pixel space (``coordinate_space: processed_512``).

    Uses a lightweight geometric approach (no PnP solver required):
      - Yaw  ≈ horizontal asymmetry between nose tip and eye midpoint.
      - Pitch ≈ vertical position of nose relative to the forehead–chin axis.

    Returns (yaw_deg, pitch_deg).
      Positive yaw   = face turned right.
      Positive pitch = face tilted downward (chin toward camera).
    """
    # Build an index-keyed lookup so we can access landmarks by their
    # canonical Face Mesh index without iterating the full list each time.
    lm: Dict[int, Dict] = {pt["index"]: pt for pt in points}

    def xy(idx: int) -> Tuple[float, float]:
        pt = lm[idx]
        return pt["x"], pt["y"]

    nose_x,  nose_y     = xy(_NOSE_TIP)
    l_eye_x, _          = xy(_LEFT_EYE_OUTER)
    r_eye_x, _          = xy(_RIGHT_EYE_OUTER)
    _,       chin_y     = xy(_CHIN)
    _,       forehead_y = xy(_FOREHEAD)

    eye_mid_x = (l_eye_x + r_eye_x) / 2.0
    eye_span  = abs(r_eye_x - l_eye_x)

    # Yaw: normalised horizontal offset of nose tip from eye midpoint.
    # Half-eye-span normalisation gives ≈ [-1, 1] for ±45°.
    yaw_norm = (nose_x - eye_mid_x) / (eye_span / 2.0 + 1e-6)
    yaw_deg  = math.degrees(math.atan(yaw_norm)) * 2.0   # empirical scale

    # Pitch: relative nose position in the forehead→chin range.
    # Neutral ≈ 0.55; deviation scaled to degrees.
    face_height = abs(chin_y - forehead_y) + 1e-6
    nose_rel    = (nose_y - forehead_y) / face_height
    pitch_deg   = (nose_rel - 0.55) * 90.0               # empirical scale

    return float(yaw_deg), float(pitch_deg)


# ── Public API ────────────────────────────────────────────────────────────────

def validate_image_input(
    image: np.ndarray,
    filename: str = "",
) -> ValidationResult:
    """Validate format, extension, and resolution of an uploaded image.

    Parameters
    ----------
    image:
        Raw NumPy array as returned by ``cv2.imdecode``.  May be ``None``
        if decoding failed (treated as a corrupted/unsupported file).
    filename:
        Original upload filename including extension (case-insensitive
        check).  Pass an empty string to skip extension validation.

    Returns
    -------
    ValidationResult
        ``ok=True`` when the image passes all checks.  ``ok=False`` with
        a descriptive ``message`` and ``error_code`` when it does not.

    Covered requirements
    --------------------
    FR-1.1.1  – only JPG/PNG accepted.
    FR-1.1.2  – resolution validated before processing.
    NFR-2.1   – clear error message, no crash on bad input.
    SDD §4.1  – minimum 200×200 recommended; 100×100 hard floor.
    SDD §9    – test: <100×100 → reject.
    """
    # ── 1. Extension check ────────────────────────────────────────────────────
    if filename:
        import os
        ext = os.path.splitext(filename.lower())[1]
        if ext not in SUPPORTED_EXTENSIONS:
            return ValidationResult(
                ok=False,
                message=f"Unsupported format '{ext}'. Please upload a JPG or PNG file.",
                error_code="UNSUPPORTED_FORMAT",
            )

    # ── 2. Decode / corruption check ─────────────────────────────────────────
    if image is None or not isinstance(image, np.ndarray):
        return ValidationResult(
            ok=False,
            message="File could not be read. It may be corrupted or in an unsupported format.",
            error_code="IMAGE_CORRUPTED",
        )

    if image.size == 0:
        return ValidationResult(
            ok=False,
            message="The uploaded file is empty (zero pixels).",
            error_code="IMAGE_EMPTY",
        )

    # ── 3. Resolution checks ──────────────────────────────────────────────────
    h, w = image.shape[:2]

    if h < MIN_DIMENSION_HARD or w < MIN_DIMENSION_HARD:
        return ValidationResult(
            ok=False,
            message=(
                f"Image too small ({w}×{h} px). "
                f"Minimum required size is {MIN_DIMENSION_HARD}×{MIN_DIMENSION_HARD} px."
            ),
            error_code="RESOLUTION_TOO_LOW",
            width=w,
            height=h,
        )

    if h < MIN_DIMENSION_SOFT or w < MIN_DIMENSION_SOFT:
        # Soft warning: processing continues but quality may be degraded.
        # The caller may choose to surface this as a non-fatal warning.
        return ValidationResult(
            ok=True,
            message=(
                f"Low resolution ({w}×{h} px). "
                f"For best results, use an image of at least {MIN_DIMENSION_SOFT}×{MIN_DIMENSION_SOFT} px."
            ),
            error_code="RESOLUTION_LOW_WARNING",
            width=w,
            height=h,
        )

    return ValidationResult(
        ok=True,
        message="Image validation passed.",
        error_code="OK",
        width=w,
        height=h,
    )


def check_face_orientation(
    image: np.ndarray,
) -> FaceOrientationResult:
    """Detect faces and assess whether the subject is frontal.

    Delegates detection entirely to ``face_detection.get_face_detector()``
    (``MediaPipeFaceDetector``) so that a single MediaPipe integration
    point exists in the codebase.

    The image must be **512×512 RGB uint8** — i.e. the output of
    ``preprocess_pipeline``'s resize step — because ``MediaPipeFaceDetector``
    enforces that shape via ``validate_face_detection_input``.
    Pass ``processed_image_uint8`` from the pipeline result.

    Parameters
    ----------
    image:
        RGB uint8 NumPy array, shape (512, 512, 3).

    Returns
    -------
    FaceOrientationResult
        ``ok=True``  – frontal face detected; pipeline may continue.
        ``ok=False`` – no face found or face is too rotated; surface
                       ``message`` to the user and abort.

    Covered requirements
    --------------------
    FR-1.2.1  – automatic face detection in the uploaded image.
    NFR-2.1   – no-face and non-frontal cases handled gracefully.
    SDD §9    – "no face detected" test case.
    """
    try:
        detector     = get_face_detector()
        detect_result = detector.detect_landmarks(image)

        # ── Detector reported failure (no face / bad input) ───────────────────
        if detect_result["status"] != "completed":
            # Distinguish between "no face" and a pending stub / other failure.
            status = detect_result["status"]
            raw_msg = detect_result.get("message", "")

            if status == "pending_group_2":
                # Stub detector active — cannot make a real decision.
                # Treat as non-blocking so the pipeline can still run during dev.
                return FaceOrientationResult(
                    ok=True,
                    message="Face detection not yet active (awaiting Group 2). Skipping orientation check in dev mode.",
                    error_code="PENDING_IMPLEMENTATION",
                    face_count=0,
                )

            # Real detector returned failed → no face in image
            return FaceOrientationResult(
                ok=False,
                message=(
                    "No face detected. Please upload a clear, well-lit, "
                    "front-facing photo without heavy obstructions."
                ),
                error_code="NO_FACE_DETECTED",
                face_count=0,
            )

        # ── Landmarks available — estimate orientation ─────────────────────────
        points: list = detect_result["points"]
        face_count   = 1   # MediaPipeFaceDetector always returns one face

        yaw_deg, pitch_deg = _estimate_yaw_pitch(points)

        is_frontal = (
            abs(yaw_deg)   <= YAW_FRONTAL_MAX_DEG and
            abs(pitch_deg) <= PITCH_FRONTAL_MAX_DEG
        )

        if not is_frontal:
            direction_hints: list[str] = []
            if abs(yaw_deg) > YAW_FRONTAL_MAX_DEG:
                side = "right" if yaw_deg > 0 else "left"
                direction_hints.append(f"rotated ~{abs(yaw_deg):.0f}° {side}")
            if abs(pitch_deg) > PITCH_FRONTAL_MAX_DEG:
                side = "down" if pitch_deg > 0 else "up"
                direction_hints.append(f"tilted ~{abs(pitch_deg):.0f}° {side}")

            hint_str = " and ".join(direction_hints)
            return FaceOrientationResult(
                ok=False,
                message=f"Face detected but not frontal ({hint_str}). Please use a front-facing photo.",
                error_code="FACE_NOT_FRONTAL",
                yaw_deg=yaw_deg,
                pitch_deg=pitch_deg,
                face_count=face_count,
            )

        return FaceOrientationResult(
            ok=True,
            message=f"Frontal face detected (yaw={yaw_deg:.1f}°, pitch={pitch_deg:.1f}°).",
            error_code="OK",
            yaw_deg=yaw_deg,
            pitch_deg=pitch_deg,
            face_count=face_count,
        )

    except Exception as exc:  # noqa: BLE001 — never crash the pipeline
        return FaceOrientationResult(
            ok=False,
            message=f"Face detection failed unexpectedly: {exc}",
            error_code="DETECTION_ERROR",
        )


# ── Convenience wrapper ───────────────────────────────────────────────────────

def run_input_gate(
    image: np.ndarray,
    filename: str = "",
) -> Tuple[bool, str, str]:
    """Run full input validation + face orientation check in one call.

    Designed for use in the FastAPI endpoint.  Executes in two stages:

    Stage 1 — ``validate_image_input``:
        Checks extension, corruption, and minimum resolution on the raw
        uploaded array.  Aborts immediately on hard failures.

    Stage 2 — ``check_face_orientation``:
        Runs after the preprocessing pipeline resizes the image to 512×512,
        because ``MediaPipeFaceDetector`` requires that exact shape.
        Internally calls ``get_face_detector().detect_landmarks()``.

    Parameters
    ----------
    image:
        Raw decoded image array (BGR from ``cv2.imdecode``, or RGB).
    filename:
        Original upload filename for extension checking.

    Returns
    -------
    (proceed, message, error_code)
        ``proceed=True``  – all checks passed; safe to call the pipeline.
        ``proceed=False`` – a check failed; surface ``message`` to the user.
        ``error_code``    – machine-readable code for the frontend.

    Example usage in main.py
    ------------------------
    .. code-block:: python

        from preprocessing import preprocess_pipeline
        from face_validation import run_input_gate

        # Stage 1: validate raw upload
        proceed, msg, code = run_input_gate(raw_image, upload.filename)
        if not proceed:
            raise HTTPException(status_code=422, detail={"message": msg, "code": code})

        # Run preprocessing (resize → 512×512, normalise, grayscale)
        pipeline_result = preprocess_pipeline(raw_image)
        processed_uint8 = pipeline_result["processed_image_uint8"]  # (512,512,3) RGB

        # Stage 2: face orientation on the 512×512 image
        from face_validation import check_face_orientation
        orientation = check_face_orientation(processed_uint8)
        if not orientation.ok:
            raise HTTPException(status_code=422, detail={
                "message": orientation.message,
                "code":    orientation.error_code,
            })
    """
    from preprocessing import ensure_rgb  # local import to avoid circular deps

    # Stage 1: format, corruption, resolution
    val = validate_image_input(image, filename)
    if not val.ok:
        return False, val.message, val.error_code

    # Resize to 512×512 so MediaPipeFaceDetector's shape check passes
    try:
        from preprocessing import ensure_rgb, resize_to_target
        rgb = ensure_rgb(image)
        resized, _ = resize_to_target(rgb, size=512)
    except (ValueError, Exception) as exc:
        return False, f"Image preprocessing error: {exc}", "PREPROCESS_ERROR"

    # Stage 2: face presence + orientation
    face = check_face_orientation(resized)
    if not face.ok:
        return False, face.message, face.error_code

    # Surface non-fatal low-resolution warning without blocking
    if val.error_code == "RESOLUTION_LOW_WARNING":
        return True, val.message, val.error_code

    return True, "All validation checks passed.", "OK"