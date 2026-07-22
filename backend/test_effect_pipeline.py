from __future__ import annotations

import cv2
import numpy as np

from backend.photo_pipeline import apply_photo_pipeline


def _mse(a: np.ndarray, b: np.ndarray) -> float:
    diff = a.astype(np.float32) - b.astype(np.float32)
    return float(np.mean(diff * diff))


def _changed_pixels(a: np.ndarray, b: np.ndarray) -> int:
    return int(np.count_nonzero(np.any(a != b, axis=2)))


def _load_test_image() -> np.ndarray:
    image = cv2.imread("test.jpg", cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError("test.jpg is required for effect pipeline integration tests.")
    return image


def test_hair_color_changes_pixels() -> None:
    image = _load_test_image()
    output, ctx = apply_photo_pipeline(
        image,
        {
            "hair_color": {
                "enabled": True,
                "color": "#7B3FE4",
                "intensity": 0.75,
            },
        },
    )

    assert _mse(image, output) > 0.0
    assert _changed_pixels(image, output) > 1000
    assert any(
        item["effect"] == "hair_color" and item["applied"]
        for item in ctx.get("effects_meta", [])
    )


def test_lipstick_changes_pixels() -> None:
    image = _load_test_image()
    output, ctx = apply_photo_pipeline(
        image,
        {
            "lipstick": {
                "enabled": True,
                "color": "#A02040",
                "intensity": 0.7,
            },
        },
    )

    assert _mse(image, output) > 0.0
    assert _changed_pixels(image, output) > 1000
    assert any(
        item["effect"] == "lipstick" and item["applied"]
        for item in ctx.get("effects_meta", [])
    )
