import cv2

from backend.face_analysis import analyze_face
from backend.glasses_tryon import apply_glasses


img = cv2.imread("test.jpg")

if img is None:
    raise FileNotFoundError("test.jpg not found")

analysis = analyze_face(img)

result = apply_glasses(
    image_bgr=img,
    analysis=analysis,
    asset_path=None,      # None => otomatik test gözlüğü üretir
    scale=2.20,
    y_offset=0.10,
    use_hair_occlusion=True,
)

cv2.imwrite(
    "glasses_result.png",
    result,
)

print("[+] glasses_result.png saved")