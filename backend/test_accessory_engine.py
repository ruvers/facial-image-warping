from __future__ import annotations

import cv2

from backend.face_analysis import analyze_face
from backend.three_d.manager import enrich_with_best_available_3d
from backend.accessory_engine import apply_accessory_pack


def main():
    img = cv2.imread("test.jpg")

    if img is None:
        raise RuntimeError("Could not read test.jpg")

    ctx = analyze_face(img)
    ctx = enrich_with_best_available_3d(ctx, img)

    out = apply_accessory_pack(
        img,
        ctx,
        {
            "enabled": True,
            "items": [
                {
                    "type": "glasses",
                    "debug_placeholder": True,
                    "scale": 1.0,
                },
                {
                    "type": "left_earring",
                    "debug_placeholder": True,
                    "scale": 0.12,
                    "offset_y_ratio": 0.07,
                },
                {
                    "type": "right_earring",
                    "debug_placeholder": True,
                    "scale": 0.12,
                    "offset_y_ratio": 0.07,
                },
                {
                    "type": "necklace",
                    "debug_placeholder": True,
                    "scale": 1.1,
                    "offset_y_ratio": 0.14,
                },
            ],
        },
    )

    cv2.imwrite(
        "accessory_engine_debug.jpg",
        out,
    )

    print("[+] saved accessory_engine_debug.jpg")


if __name__ == "__main__":
    main()