from __future__ import annotations

import cv2
import numpy as np
import mediapipe as mp

from mediapipe.python.solutions import face_mesh

from backend.face_parsing import (
    parse_face,
    get_mask,
)

mp_face_mesh = face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

# =========================================================
# TEMPORAL SMOOTHING
# =========================================================

previous_landmarks = None

SMOOTHING = 0.8

# =========================================================
# HELPERS
# =========================================================

def smooth_landmarks(current, previous):

    if previous is None:
        return current

    return (
        previous * SMOOTHING +
        current * (1.0 - SMOOTHING)
    )

# =========================================================
# HAIR COLOR EFFECT
# =========================================================

def apply_hair_color(frame_bgr, hair_mask):

    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

    # mor saç
    hsv[..., 0][hair_mask > 0] = 135

    # saturation
    hsv[..., 1][hair_mask > 0] = np.clip(
        hsv[..., 1][hair_mask > 0] * 1.3,
        0,
        255,
    )

    result = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # yumuşak blend
    alpha = cv2.GaussianBlur(
        hair_mask,
        (31, 31),
        0,
    ).astype(np.float32) / 255.0

    blended = (
        frame_bgr * (1 - alpha[..., None]) +
        result * alpha[..., None]
    ).astype(np.uint8)

    return blended

# =========================================================
# MAIN LOOP
# =========================================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Camera not found")

print("[+] Webcam started")

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # =====================================================
    # LANDMARKS
    # =====================================================

    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:

        face_landmarks = results.multi_face_landmarks[0]

        h, w = frame.shape[:2]

        pts = []

        for lm in face_landmarks.landmark:

            x = int(lm.x * w)
            y = int(lm.y * h)

            pts.append([x, y])

        pts = np.array(pts, dtype=np.float32)

        pts = smooth_landmarks(
            pts,
            previous_landmarks,
        )

        previous_landmarks = pts

        # =================================================
        # FACE PARSING
        # =================================================

        parsing = parse_face(rgb)

        hair_mask = get_mask(parsing, [17])

        # =================================================
        # APPLY EFFECTS
        # =================================================

        frame = apply_hair_color(
            frame,
            hair_mask,
        )

        # =================================================
        # DRAW LANDMARKS DEBUG
        # =================================================

        for p in pts[::10]:

            cv2.circle(
                frame,
                (int(p[0]), int(p[1])),
                1,
                (0, 255, 0),
                -1,
            )

    # =====================================================
    # SHOW
    # =====================================================

    cv2.imshow(
        "FaceWarp Realtime AR",
        frame,
    )

    key = cv2.waitKey(1)

    if key == 27:
        break

cap.release()

cv2.destroyAllWindows()