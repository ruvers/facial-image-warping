from __future__ import annotations

import numpy as np

from backend.accessories.necklace3d.pipeline import apply_physics_necklace


def _synthetic_ctx() -> dict:
    h, w = 512, 512
    landmarks = np.zeros((478, 2), dtype=np.float32)
    landmarks[:, 0] = w * 0.5
    landmarks[:, 1] = h * 0.5
    landmarks[152] = [256, 285]
    landmarks[234] = [178, 230]
    landmarks[454] = [334, 230]
    landmarks[33] = [215, 195]
    landmarks[263] = [297, 195]
    landmarks[1] = [256, 225]

    neck = np.zeros((h, w), dtype=np.uint8)
    yy, xx = np.ogrid[:h, :w]
    neck[((xx - 256) ** 2) / (58**2) + ((yy - 330) ** 2) / (78**2) <= 1.0] = 255

    hair = np.zeros((h, w), dtype=np.uint8)
    hair[:190, 170:342] = 255

    return {
        "landmarks_2d": landmarks,
        "masks": {
            "neck": neck,
            "hair": hair,
            "skin": neck.copy(),
        },
        "anchors": {
            "metrics": {
                "face_width": 156.0,
                "face_height": 120.0,
            },
            "necklace": {
                "center": (256.0, 333.0),
                "width": 132.0,
                "chin": (256.0, 285.0),
            },
        },
    }


def test_physics_necklace_pipeline_changes_image() -> None:
    image = np.full((512, 512, 3), 238, dtype=np.uint8)
    ctx = _synthetic_ctx()
    item = {
        "type": "necklace",
        "category": "pendant_necklace",
        "render_mode": "physics_3d",
        "metadata": {
            "chain_length": 1.0,
            "chain_thickness": 3.0,
            "stiffness": 0.75,
            "pendant_enabled": True,
            "pendant_size": 0.12,
            "pendant_weight": 1.0,
            "material": "gold",
            "anchor_mode": "clavicle_drape",
            "node_count": 48,
        },
    }

    output, meta = apply_physics_necklace(image, ctx, item)

    changed_pixels = int(np.count_nonzero(np.any(output != image, axis=2)))
    physics_debug = meta["debug"]["physics"]

    assert output.shape == image.shape
    assert changed_pixels > 1000
    assert meta["applied"] is True
    assert meta["fallback_used"] is False
    assert physics_debug
    assert physics_debug["node_count"] > 16
    assert physics_debug["pendant_position"]
    pendant_y = float(physics_debug["pendant_position"][1])
    max_drape_y = float(physics_debug["max_drape_y"])
    assert pendant_y <= max_drape_y
    assert pendant_y < 390
    assert max_drape_y < 400


if __name__ == "__main__":
    test_physics_necklace_pipeline_changes_image()
    print("necklace3d pipeline ok")
