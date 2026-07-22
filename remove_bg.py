import cv2
import numpy as np

def p(i, o):
    img = cv2.imread(i, cv2.IMREAD_UNCHANGED)
    if img is None:
        print("Failed to load", i)
        return
    if img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    mask = cv2.GaussianBlur(mask, (3, 3), 0)
    b, g, r = cv2.split(img[:, :, :3])
    out = cv2.merge((b, g, r, mask))
    cv2.imwrite(o, out)
    print("Saved", o)

p(r'C:\Users\HP\.gemini\antigravity\brain\6af7c32b-82f1-493d-967e-abb8c34ba202\flat_reading_glasses_1781748057582.png', r'C:\Users\HP\Desktop\signalsonnn\facial-image-warping\assets\accessories\glasses\reading_glasses.png')
p(r'C:\Users\HP\.gemini\antigravity\brain\6af7c32b-82f1-493d-967e-abb8c34ba202\flat_browline_glasses_1781748072218.png', r'C:\Users\HP\Desktop\signalsonnn\facial-image-warping\assets\accessories\glasses\browline_glasses.png')
