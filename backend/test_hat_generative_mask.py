from __future__ import annotations

import numpy as np

from backend.accessories.hat_generative.mask import build_hat_placement_mask


FACE_OVAL = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
    397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
    172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109,
]


def _synthetic_ctx() -> dict:
    h, w = 512, 512
    landmarks = np.zeros((478, 2), dtype=np.float32)
    landmarks[:, 0] = w * 0.5
    landmarks[:, 1] = h * 0.5
    cx, cy, rx, ry = 256.0, 238.0, 92.0, 132.0
    for idx, angle in zip(FACE_OVAL, np.linspace(-96, 264, len(FACE_OVAL), endpoint=False)):
        rad = np.deg2rad(angle)
        landmarks[idx] = [cx + np.cos(rad) * rx, cy + np.sin(rad) * ry]
    landmarks[10] = [256, 118]
    landmarks[151] = [256, 150]
    landmarks[152] = [256, 348]
    landmarks[234] = [176, 240]
    landmarks[454] = [336, 240]
    landmarks[33] = [214, 208]
    landmarks[133] = [238, 210]
    landmarks[263] = [298, 208]
    landmarks[362] = [274, 210]
    landmarks[159] = [226, 202]
    landmarks[145] = [226, 218]
    landmarks[386] = [286, 202]
    landmarks[374] = [286, 218]
    for idx, x in zip([46, 53, 52, 65, 55, 70, 63, 105, 66, 107], np.linspace(204, 250, 10)):
        landmarks[idx] = [x, 182]
    for idx, x in zip([276, 283, 282, 295, 285, 300, 293, 334, 296, 336], np.linspace(262, 308, 10)):
        landmarks[idx] = [x, 182]

    return {
        "landmarks_2d": landmarks,
        "anchors": {
            "metrics": {
                "face_width": 160.0,
                "face_height": 210.0,
            },
        },
    }


def test_hat_generative_mask_safe_placement() -> None:
    mask, debug = build_hat_placement_mask(
        (512, 512, 3),
        _synthetic_ctx(),
    )

    brow_y = float(debug["brow_y"])
    bottom_y = float(debug["bottom_y"])

    assert mask.shape == (512, 512)
    assert debug["mask_area"] > 1000
    assert debug["bottom_above_brow"] is True
    assert bottom_y <= brow_y - 20.0
    assert mask[int(brow_y), 256] == 0
    assert mask[230, 256] == 0
    assert mask[256, 256] == 0


if __name__ == "__main__":
    test_hat_generative_mask_safe_placement()
    print("hat generative mask ok")
