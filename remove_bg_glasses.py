import cv2
import numpy as np
import sys

def process(in_path, out_path):
    img = cv2.imread(in_path, cv2.IMREAD_UNCHANGED)
    if img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA@BGR)
        # Wait, if it's already 4 channels, let's just make sure we drop alpha first if it's fake
    else:
        pass

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Threshold: anything that is not very bright white becomes foreground
    _, mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    # Find contours to get the largest object
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        print('No contours found')
        return
    
    # We could just use the mask directly and add an alpha channel
    # But for glasses, we might have clear lenses that are also white.
    # If the lenses are white, thresholding will make them transparent. This is good for clear glasses!
    
    # Optional: smooth the mask
    mask = cv2.GaussianBlur(mask, (3, 3), 0)
    
    # Create alpha channel
    b_channel, g_channel, r_channel = cv2.split(img[:,:,:3])
    alpha_channel = mask
    img_BGRA = cv2.merge((b_channel, g_channel, r_channel, alpha_channel))
    cv2.imwrite(out_path, img_BGRA)

process(r'C:\Users\HP\.gemini\antigravity\brain\6af7c32b-82f1-493d-967e-abb8c34ba202\modern_wayfarer_1781747582870.png', r'C:\Users\HP\Desktop\signalsonnn\facial-image-warping\assets\accessories\glasses\modern_wayfarer.png')
process(r'C:\Users\HP\.gemini\antigravity\brain\6af7c32b-82f1-493d-967e-abb8c34ba202\elegant_round_1781747598136.png', r'C:\Users\HP\Desktop\signalsonnn\facial-image-warping\assets\accessories\glasses\elegant_round.png')
print('Done!')
