from __future__ import annotations

import cv2
import numpy as np

from backend.effects.color_v1 import apply_eye_color
from backend.effects.hair_color_v2 import apply_hair_color_hsl


def test_light_hair_color_preserves_texture() -> None:
    height, width = 128, 128
    image = np.full((height, width, 3), 150, dtype=np.uint8)

    yy, xx = np.mgrid[0:height, 0:width]
    texture = (
        28.0
        + 18.0 * np.sin(xx * 0.31)
        + 12.0 * np.cos(yy * 0.23)
        + 0.20 * (xx - 64)
    )
    texture = np.clip(texture, 8, 82).astype(np.uint8)
    hair_region = (yy >= 18) & (yy <= 104) & (xx >= 18) & (xx <= 110)
    image[hair_region, 0] = texture[hair_region]
    image[hair_region, 1] = np.clip(texture[hair_region] + 4, 0, 255)
    image[hair_region, 2] = np.clip(texture[hair_region] + 9, 0, 255)

    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.rectangle(mask, (18, 18), (110, 104), 255, -1)
    mask = cv2.GaussianBlur(mask, (0, 0), 3.0)

    result = apply_hair_color_hsl(
        image,
        mask,
        target_color_rgb=(232, 224, 200),
        intensity=0.75,
    )

    original_l = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)[:, :, 1].astype(np.float32)
    result_l = cv2.cvtColor(result, cv2.COLOR_BGR2HLS)[:, :, 1].astype(np.float32)
    core = mask > 245

    assert float(np.mean(result_l[core])) > float(np.mean(original_l[core])) + 25.0
    assert float(np.std(result_l[core])) > float(np.std(original_l[core])) * 0.55
    assert float(np.std(result_l[core])) > 8.0
    assert int(np.max(np.abs(result[:8].astype(np.int16) - image[:8].astype(np.int16)))) == 0


def _set_eye_landmarks(
    points: np.ndarray,
    center: tuple[float, float],
    iris_ids: tuple[int, ...],
    contour_ids: tuple[int, ...],
) -> None:
    cx, cy = center
    points[iris_ids[0]] = (cx, cy)
    points[iris_ids[1]] = (cx + 5, cy)
    points[iris_ids[2]] = (cx, cy - 5)
    points[iris_ids[3]] = (cx - 5, cy)
    points[iris_ids[4]] = (cx, cy + 5)

    upper_count = 9
    upper_angles = np.linspace(np.pi, 0.0, upper_count)
    lower_angles = np.linspace(0.0, -np.pi, len(contour_ids) - upper_count + 2)[1:-1]
    contour_angles = np.concatenate([upper_angles, lower_angles])
    for idx, angle in zip(contour_ids, contour_angles):
        points[idx] = (cx + np.cos(angle) * 16.0, cy - np.sin(angle) * 7.0)


def test_eye_color_is_visible_while_pupil_stays_dark() -> None:
    image = np.full((100, 160, 3), (95, 125, 165), dtype=np.uint8)
    centers = ((50, 50), (110, 50))
    for center in centers:
        cv2.ellipse(image, center, (16, 7), 0, 0, 360, (225, 225, 225), -1, cv2.LINE_AA)
        cv2.circle(image, center, 5, (30, 55, 82), -1, cv2.LINE_AA)
        cv2.circle(image, center, 2, (7, 7, 7), -1, cv2.LINE_AA)
        cv2.circle(image, (center[0] - 2, center[1] - 2), 1, (245, 245, 245), -1, cv2.LINE_AA)

    points = np.zeros((478, 2), dtype=np.float32)
    left_contour = (33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7)
    right_contour = (263, 466, 388, 387, 386, 385, 384, 398, 362, 382, 381, 380, 374, 373, 390, 249)
    _set_eye_landmarks(points, centers[0], (468, 469, 470, 471, 472), left_contour)
    _set_eye_landmarks(points, centers[1], (473, 474, 475, 476, 477), right_contour)

    ctx: dict = {"landmarks_2d": points}
    result = apply_eye_color(
        image,
        ctx,
        {"enabled": True, "color": "#3F7FBF", "intensity": 0.45},
    )

    ring_pixels = []
    for cx, cy in centers:
        for y in range(cy - 5, cy + 6):
            for x in range(cx - 5, cx + 6):
                radius = np.hypot(x - cx, y - cy)
                if 3.0 <= radius <= 4.8:
                    ring_pixels.append(result[y, x])
        assert int(np.max(np.abs(result[cy, cx].astype(np.int16) - image[cy, cx].astype(np.int16)))) < 8

    ring = np.asarray(ring_pixels, dtype=np.float32)
    assert float(np.mean(ring[:, 0] - ring[:, 2])) > 18.0
    assert int(np.max(np.abs(result[:12].astype(np.int16) - image[:12].astype(np.int16)))) == 0
    debug = ctx["effect_debug_meta"]["eye_color"]["eye_color_debug"]
    assert debug["mask_pixels"] > 40
    assert debug["max_alpha"] > 0.55
