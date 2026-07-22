from __future__ import annotations

import numpy as np

from backend.aging import apply_frequency_aging_effect
from backend.test_hat_generative_mask import _synthetic_ctx


def _aging_ctx() -> dict:
    ctx = _synthetic_ctx()
    landmarks = ctx["landmarks_2d"].copy()
    for idx, xy in {
        1: (256, 245),
        4: (256, 230),
        5: (256, 210),
        6: (256, 198),
        9: (256, 178),
        50: (205, 255),
        280: (307, 255),
        61: (225, 292),
        84: (246, 306),
        91: (232, 300),
        146: (222, 294),
        181: (238, 303),
        291: (287, 292),
        314: (266, 306),
        321: (280, 300),
        375: (290, 294),
        405: (274, 303),
        205: (220, 250),
        187: (226, 262),
        207: (230, 275),
        216: (236, 288),
        212: (242, 298),
        202: (248, 306),
        425: (292, 250),
        411: (286, 262),
        427: (282, 275),
        436: (276, 288),
        432: (270, 298),
        422: (264, 306),
    }.items():
        landmarks[idx] = xy
    ctx["landmarks_2d"] = landmarks
    ctx["skip_face_parsing"] = True
    return ctx


def test_frequency_aging_adds_face_wrinkle_debug() -> None:
    h, w = 512, 512
    yy = np.linspace(0, 1, h, dtype=np.float32)[:, None]
    xx = np.linspace(0, 1, w, dtype=np.float32)[None, :]
    base = np.zeros((h, w, 3), dtype=np.uint8)
    base[..., 0] = np.clip(178 + xx * 18 - yy * 12, 0, 255).astype(np.uint8)
    base[..., 1] = np.clip(154 + xx * 10 - yy * 10, 0, 255).astype(np.uint8)
    base[..., 2] = np.clip(136 + xx * 8 - yy * 8, 0, 255).astype(np.uint8)
    ctx = _aging_ctx()

    output = apply_frequency_aging_effect(
        base,
        ctx,
        {
            "intensity": 1.55,
            "landmarks": None,
        },
    )
    changed = int(np.count_nonzero(np.any(output != base, axis=2)))
    diff = np.mean(np.abs(output.astype(np.float32) - base.astype(np.float32)), axis=2)
    forehead_delta = float(np.mean(diff[135:190, 190:322]))
    cheek_delta = float(np.mean(diff[225:295, 180:332]))
    jaw_delta = float(np.mean(diff[305:360, 190:322]))

    assert output.shape == base.shape
    assert changed > 1000
    assert forehead_delta > 0.05
    assert cheek_delta > 0.05
    assert jaw_delta > 0.02


if __name__ == "__main__":
    test_frequency_aging_adds_face_wrinkle_debug()
    print("frequency aging enhancement ok")
