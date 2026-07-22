from __future__ import annotations

import cv2

from backend.face_analysis import analyze_face
from backend.three_d.manager import enrich_with_best_available_3d
from backend.three_d.deca_debug import draw_deca_debug


def main():
    image_path = "test.jpg"
    output_path = "deca_debug.jpg"

    img = cv2.imread(image_path)

    if img is None:
        raise RuntimeError(f"Could not read image: {image_path}")

    ctx = analyze_face(img)

    ctx = enrich_with_best_available_3d(
        ctx,
        img,
    )

    print("[+] provider:", ctx["three_d"]["provider"])
    print("[+] is_true_3d:", ctx["three_d"]["is_true_3d"])
    print("[+] vertices:", ctx["three_d"]["vertices"].shape)
    print("[+] faces:", ctx["three_d"]["faces"].shape)

    if "landmarks2d" in ctx["three_d"]:
        print("[+] deca landmarks2d:", ctx["three_d"]["landmarks2d"].shape)

    saved = draw_deca_debug(
        img,
        ctx,
        output_path,
    )

    print("[+] saved:", saved)


if __name__ == "__main__":
    main()