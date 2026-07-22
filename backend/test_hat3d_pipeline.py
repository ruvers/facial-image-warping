from __future__ import annotations

import numpy as np

from backend.accessories.hat3d.pipeline import apply_parametric_hat


def _synthetic_ctx() -> dict:
    h, w = 512, 512
    landmarks = np.zeros((478, 2), dtype=np.float32)
    landmarks[:, 0] = w * 0.5
    landmarks[:, 1] = h * 0.5
    landmarks[10] = [256, 120]
    landmarks[151] = [256, 150]
    landmarks[152] = [256, 340]
    landmarks[234] = [176, 240]
    landmarks[454] = [336, 240]
    landmarks[33] = [215, 205]
    landmarks[263] = [297, 205]
    for idx, x in zip([46, 53, 52, 65, 55, 70, 63, 105, 66, 107], np.linspace(205, 250, 10)):
        landmarks[idx] = [x, 182]
    for idx, x in zip([276, 283, 282, 295, 285, 300, 293, 334, 296, 336], np.linspace(262, 307, 10)):
        landmarks[idx] = [x, 182]

    hair = np.zeros((h, w), dtype=np.uint8)
    yy, xx = np.ogrid[:h, :w]
    hair[((xx - 256) ** 2) / (104**2) + ((yy - 150) ** 2) / (82**2) <= 1.0] = 255
    skin = np.zeros((h, w), dtype=np.uint8)
    skin[((xx - 256) ** 2) / (92**2) + ((yy - 245) ** 2) / (130**2) <= 1.0] = 255

    return {
        "landmarks_2d": landmarks,
        "masks": {
            "hair": hair,
            "skin": skin,
        },
        "pose": {
            "yaw": 8.0,
            "pitch": 1.0,
            "roll": 3.0,
        },
        "anchors": {
            "metrics": {
                "face_width": 160.0,
                "face_height": 190.0,
            },
        },
    }


def test_parametric_beanie_pipeline_changes_image() -> None:
    image = np.full((512, 512, 3), 235, dtype=np.uint8)
    ctx = _synthetic_ctx()
    item = {
        "type": "hat",
        "category": "beanie",
        "render_mode": "parametric_3d",
        "asset_id": "beanie_black_wool_procedural",
        "metadata": {
            "color": "#222222",
            "skull_fit": 1.05,
            "fold_height": 0.18,
            "top_sag": 0.08,
            "thickness": 0.08,
        },
    }

    output, meta = apply_parametric_hat(image, ctx, item)
    changed_pixels = int(np.count_nonzero(np.any(output != image, axis=2)))
    hat_debug = meta["debug"]["hat_debug"]
    render_debug = meta["debug"]["render"]
    bottom_y = float(hat_debug["bottom_y"])
    top_y = float(hat_debug["top_y"])
    brow_y = float(hat_debug["brow_y"])
    eye_y = float(hat_debug["eye_y"])
    alpha_bbox = render_debug["alpha_bbox"]

    assert output.shape == image.shape
    assert changed_pixels > 1000
    assert meta["type"] == "hat"
    assert meta["render_mode"] == "parametric_3d"
    assert meta["debug"]["head_proxy"]["confidence"] > 0.5
    assert meta["fallback_used"] is False
    assert bottom_y < brow_y - 15
    assert bottom_y < eye_y - 25
    assert top_y < bottom_y
    assert render_debug["mask_area"] > 1000
    assert render_debug["shape"] == "beanie_dome"
    assert hat_debug["bottom_above_brow"] is True
    assert alpha_bbox is not None
    assert alpha_bbox["y2"] < int(brow_y - 5)

    # The lower face/eye band must not receive dark hat pixels.
    dark_pixels_below_brow = np.count_nonzero(
        np.any(output[int(brow_y - 4):, :] < 80, axis=2)
    )
    assert dark_pixels_below_brow == 0


if __name__ == "__main__":
    test_parametric_beanie_pipeline_changes_image()
    print("hat3d pipeline ok")
