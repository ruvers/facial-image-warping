from __future__ import annotations

import cv2
import numpy as np

from backend.face_parsing import (
    parse_face,
    get_mask,
)

# =========================================================
# LOAD IMAGE
# =========================================================

image_bgr = cv2.imread("test.jpg")

if image_bgr is None:
    raise FileNotFoundError("test.jpg not found")

image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

# =========================================================
# PARSE
# =========================================================

print("[+] Face parsing...")

parsing = parse_face(image_rgb)

# =========================================================
# MASKS
# =========================================================

hair_mask = get_mask(parsing, [17])

skin_mask = get_mask(parsing, [1])

neck_mask = get_mask(parsing, [14])

ear_mask = get_mask(parsing, [7, 8])

lip_mask = get_mask(parsing, [12, 13])

eye_mask = get_mask(parsing, [4, 5])

eyebrow_mask = get_mask(parsing, [2, 3])

# =========================================================
# COLOR OVERLAY
# =========================================================

overlay = image_bgr.copy()

overlay[hair_mask > 0] = (255, 0, 255)

overlay[skin_mask > 0] = (0, 255, 255)

overlay[neck_mask > 0] = (255, 255, 0)

overlay[ear_mask > 0] = (0, 255, 0)

overlay[lip_mask > 0] = (0, 0, 255)

overlay[eye_mask > 0] = (255, 0, 0)

overlay[eyebrow_mask > 0] = (50, 50, 50)

# =========================================================
# BLEND
# =========================================================

result = cv2.addWeighted(
    image_bgr,
    0.45,
    overlay,
    0.55,
    0,
)

# =========================================================
# SAVE
# =========================================================

cv2.imwrite(
    "segmentation_debug.jpg",
    result,
)

print("[+] Saved:")
print("    segmentation_debug.jpg")