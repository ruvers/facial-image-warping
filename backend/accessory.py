import numpy as np
import cv2

from backend.face_parsing import get_mask, feather_mask

def apply_necklace_with_depth(image_rgb, necklace_png_rgba, parsing_map):
    # ... (Az önce verdiğim kolye occlusion kodu) ...
    pass

def get_earring_anchor(parsing_map):
    # ... (Az önce verdiğim kulak merkezi kodu) ...
    pass

def landmark_point(
    landmarks,
    idx,
):
    p = landmarks[idx]

    return int(p["x"]), int(p["y"])