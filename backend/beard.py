import cv2
import numpy as np


def detect_beard_region(image_rgb, skin_mask):
    """
    Detect dark textured beard-like regions inside skin.
    """

    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)

    h, s, v = cv2.split(hsv)

    # darker regions
    dark = v < 110

    # textured edges
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

    edges = cv2.Laplacian(
        gray,
        cv2.CV_64F
    )

    edges = np.abs(edges)

    texture = edges > 12

    beard = (
        dark &
        texture &
        (skin_mask > 0)
    )

    beard = beard.astype(np.uint8) * 255

    beard = cv2.medianBlur(beard, 5)

    return beard