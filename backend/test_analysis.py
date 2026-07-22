import cv2

from backend.face_analysis import analyze_face

img = cv2.imread("test.jpg")

result = analyze_face(img)

for name, mask in result["masks"].items():
    cv2.imwrite(f"{name}_mask.jpg", mask)

print("[+] Done")
print(result["masks"].keys())