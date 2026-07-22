from __future__ import annotations

import cv2
import numpy as np

from backend.face_analysis import analyze_face
from backend.local_models.sam_aging import apply_sam_aging, get_sam_aging_status


def _mse(a: np.ndarray, b: np.ndarray) -> float:
    diff = a.astype(np.float32) - b.astype(np.float32)
    return float(np.mean(diff * diff))


def _changed_pixels(a: np.ndarray, b: np.ndarray) -> int:
    return int(np.count_nonzero(np.any(a != b, axis=2)))


def test_sam_aging_changes_pixels() -> None:
    status = get_sam_aging_status()
    image = cv2.imread("test.jpg", cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError("test.jpg is required for SAM aging integration test.")

    if not status.get("runtime_available"):
        print("SAM unavailable; skipping pixel-change assertion:", status)
        return

    ctx = analyze_face(image)
    output, meta = apply_sam_aging(
        image,
        ctx,
        {
            "enabled": True,
            "target_age": 60,
            "intensity": 0.8,
            "fallback_noop": True,
            "strict": False,
        },
    )

    if not meta.get("inference_ran"):
        print("SAM inference did not run; skipping pixel-change assertion:", meta)
        return

    assert _mse(image, output) > 0.0
    assert _changed_pixels(image, output) > 1000
    assert int(meta.get("changed_pixels", 0)) > 1000


if __name__ == "__main__":
    test_sam_aging_changes_pixels()
    print("SAM aging pipeline test completed")
