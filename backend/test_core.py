import cv2

from backend.face_analysis import analyze_face

img = cv2.imread("test.jpg")

if img is None:
    raise FileNotFoundError("test.jpg not found")

ctx = analyze_face(img)

print("[+] keys:", ctx.keys())
print("[+] landmarks_2d:", ctx["landmarks_2d"].shape)
print("[+] landmarks_3d:", ctx["landmarks_3d"].shape)
print("[+] pose:", ctx["pose"]["euler"])

print("[+] anchors:")
for key, value in ctx["anchors"].items():
    print("   ", key, "=>", value)

vis = img.copy()

gx, gy = ctx["anchors"]["glasses"]["center"]
cv2.circle(vis, (int(gx), int(gy)), 6, (0, 0, 255), -1)
cv2.putText(vis, "glasses", (int(gx) + 8, int(gy)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

lx, ly = ctx["anchors"]["earrings"]["left"]
rx, ry = ctx["anchors"]["earrings"]["right"]

cv2.circle(vis, (int(lx), int(ly)), 6, (255, 0, 0), -1)
cv2.putText(vis, "L earring", (int(lx) + 8, int(ly)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

cv2.circle(vis, (int(rx), int(ry)), 6, (255, 0, 0), -1)
cv2.putText(vis, "R earring", (int(rx) + 8, int(ry)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

nx, ny = ctx["anchors"]["necklace"]["center"]
cv2.circle(vis, (int(nx), int(ny)), 6, (0, 255, 0), -1)
cv2.putText(vis, "necklace", (int(nx) + 8, int(ny)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

cv2.imwrite("core_anchor_debug.jpg", vis)

print("[+] core_anchor_debug.jpg saved")