import cv2

from backend.face_analysis import analyze_face
from backend.hair_color_v2 import apply_hair_color

img_bgr = cv2.imread("test.jpg")

if img_bgr is None:
    raise FileNotFoundError("test.jpg not found")

analysis = analyze_face(img_bgr)

img_rgb = cv2.cvtColor(
    img_bgr,
    cv2.COLOR_BGR2RGB,
)

hair_mask = analysis["masks"]["hair"]

result_rgb = apply_hair_color(
    img_rgb,
    hair_mask,
    target_bgr=(180, 60, 60),
    intensity=0.80,
)

result_bgr = cv2.cvtColor(
    result_rgb,
    cv2.COLOR_RGB2BGR,
)

cv2.imwrite(
    "hair_result.png",
    result_bgr,
)

print("[+] hair_result.png saved")