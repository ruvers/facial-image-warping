# backend/accessories.py

import cv2
import numpy as np

# Senin face_parsing.py içindeki fonksiyonlarını buraya çağırıyoruz
from backend.face_parsing import get_mask, feather_mask

def apply_necklace_with_depth(image_rgb, necklace_png_rgba, parsing_map):
    # ... (Az önce verdiğim kolye occlusion kodu) ...
    pass

def get_earring_anchor(parsing_map):
    # ... (Az önce verdiğim kulak merkezi kodu) ...
    pass