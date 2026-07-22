import cv2

from backend.face_analysis import analyze_face
from backend.accessories.glasses_v2 import apply_glasses_v2


img = cv2.imread("test.jpg")

if img is None:
    raise FileNotFoundError("test.jpg not found")

ctx = analyze_face(img)

result = apply_glasses_v2(
    image_bgr=img,
    analysis=ctx,
    asset_path="assets/accessories/glasses/thin_black_round.png",
    width_scale=1.0,
    y_offset_ratio=0.03,
)

cv2.imwrite(
    "glasses_v2_result.png",
    result,
)

print("[+] glasses_v2_result.png saved")