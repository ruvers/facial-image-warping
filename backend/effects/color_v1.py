from __future__ import annotations

import cv2
import numpy as np


def _hex_to_bgr(hex_color: str) -> tuple[int, int, int]:
    hex_color = str(hex_color or "").strip().replace("#", "")
    if len(hex_color) != 6:
        raise ValueError("hex_color must be like '#3366AA'")

    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return b, g, r


def _landmarks(ctx: dict) -> np.ndarray | None:
    points = ctx.get("landmarks_2d")
    if isinstance(points, np.ndarray) and points.ndim == 2 and points.shape[0] > 0:
        return points
    return None


def _soft_mask(mask: np.ndarray, blur: int) -> np.ndarray:
    if blur % 2 == 0:
        blur += 1
    if blur > 1:
        mask = cv2.GaussianBlur(mask, (blur, blur), 0)
    return np.clip(mask.astype(np.float32) / 255.0, 0.0, 1.0)


def _blend_color(
    image_bgr: np.ndarray,
    mask: np.ndarray,
    color_hex: str,
    intensity: float,
    mode: str = "normal",
) -> np.ndarray:
    intensity = float(np.clip(intensity, 0.0, 1.0))
    if intensity <= 0:
        return image_bgr

    target = np.array(_hex_to_bgr(color_hex), dtype=np.float32)
    alpha = mask[:, :, None] * intensity
    base = image_bgr.astype(np.float32)
    color = np.full_like(base, target)

    if mode == "multiply":
        color = base * (target / 255.0)

    result = base * (1.0 - alpha) + color * alpha
    return np.clip(result, 0, 255).astype(np.uint8)


def _ellipse_mask(
    shape: tuple[int, int],
    center: np.ndarray,
    axes: tuple[int, int],
    angle: float = 0.0,
) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    cv2.ellipse(
        mask,
        (int(center[0]), int(center[1])),
        axes,
        angle,
        0,
        360,
        255,
        -1,
        cv2.LINE_AA,
    )
    return mask


def apply_eye_color(
    image_bgr: np.ndarray,
    ctx: dict,
    params: dict | None = None,
) -> np.ndarray:
    params = params or {}
    if not params.get("enabled", False):
        return image_bgr

    points = _landmarks(ctx)
    if points is None or points.shape[0] < 478:
        return image_bgr

    h, w = image_bgr.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    pupil_protect = np.zeros((h, w), dtype=np.uint8)
    eye_aperture = np.zeros((h, w), dtype=np.uint8)

    eye_specs = (
        (
            (468, 469, 470, 471, 472),
            (33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7),
        ),
        (
            (473, 474, 475, 476, 477),
            (263, 466, 388, 387, 386, 385, 384, 398, 362, 382, 381, 380, 374, 373, 390, 249),
        ),
    )
    for indices, eye_contour in eye_specs:
        iris = points[list(indices)].astype(np.float32)
        center = iris[0]
        boundary = iris[1:]
        radius = max(2, int(round(np.median(np.linalg.norm(boundary - center, axis=1)) * 1.15)))
        cv2.circle(
            mask,
            (int(center[0]), int(center[1])),
            radius,
            255,
            -1,
            cv2.LINE_AA,
        )
        cv2.circle(
            pupil_protect,
            (int(center[0]), int(center[1])),
            max(1, int(radius * 0.30)),
            255,
            -1,
            cv2.LINE_AA,
        )
        contour = points[list(eye_contour)].astype(np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(eye_aperture, [contour], 255, cv2.LINE_AA)

    raw_intensity = float(np.clip(params.get("intensity", 0.45), 0.0, 1.0))
    if raw_intensity <= 0.0:
        return image_bgr

    strength = 0.51 + raw_intensity * 0.36
    iris_alpha = _soft_mask(cv2.bitwise_and(mask, eye_aperture), 3)
    pupil_alpha = _soft_mask(pupil_protect, 3)
    hls = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HLS).astype(np.float32)
    H, L, S = cv2.split(hls)
    highlight_protect = np.clip((L - 185.0) / 70.0, 0.0, 0.88)
    darkness_protect = np.clip((18.0 - L) / 18.0, 0.0, 0.75)
    alpha = np.clip(
        iris_alpha
        * strength
        * (1.0 - pupil_alpha * 0.88)
        * (1.0 - highlight_protect)
        * (1.0 - darkness_protect),
        0.0,
        1.0,
    )
    if float(np.max(alpha)) <= 0.001:
        return image_bgr

    target_bgr = np.uint8([[_hex_to_bgr(params.get("color", "#3F7FBF"))]])
    target_hls = cv2.cvtColor(target_bgr, cv2.COLOR_BGR2HLS)[0, 0].astype(np.float32)
    target_h = float(target_hls[0])
    target_s = float(target_hls[2])
    is_green_target = 35.0 <= target_h <= 85.0
    hue_delta = ((target_h - H + 90.0) % 180.0) - 90.0
    hue_strength = 0.88 if is_green_target else 0.92
    new_H = (H + hue_delta * hue_strength) % 180.0
    effective_target_s = min(255.0, target_s * (1.16 if is_green_target else 1.05))
    new_S = np.clip(S * 0.24 + effective_target_s * 0.76, 0, 255)
    green_visibility_lift = 8.0 if is_green_target else 0.0
    visibility_floor = max(
        float(target_hls[1]) + green_visibility_lift,
        80.0 + raw_intensity * 22.0,
    )
    lift = (
        np.maximum(0.0, visibility_floor - L)
        * raw_intensity
        * 0.40
        * (1.0 - pupil_alpha * 0.92)
    )
    new_L = np.clip(L + lift, 0, 255)
    recolored = cv2.cvtColor(
        np.dstack([new_H, new_L, new_S]).astype(np.uint8),
        cv2.COLOR_HLS2BGR,
    ).astype(np.float32)

    base = image_bgr.astype(np.float32)
    out = base * (1.0 - alpha[..., None]) + recolored * alpha[..., None]
    ctx.setdefault("effect_debug_meta", {})["eye_color"] = {
        "eye_color_debug": {
            "mask_pixels": int(np.count_nonzero(alpha > 0.02)),
            "max_alpha": round(float(np.max(alpha)), 3),
            "intensity": round(raw_intensity, 3),
        }
    }
    return np.clip(out, 0, 255).astype(np.uint8)


def apply_eyeshadow(
    image_bgr: np.ndarray,
    ctx: dict,
    params: dict | None = None,
) -> np.ndarray:
    params = params or {}
    if not params.get("enabled", False):
        return image_bgr

    points = _landmarks(ctx)
    if points is None or points.shape[0] <= 386:
        return image_bgr

    h, w = image_bgr.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    for eye_ids in ((33, 133, 159, 145), (362, 263, 386, 374)):
        eye = points[list(eye_ids)].astype(np.float32)
        center = np.mean(eye, axis=0)
        width = max(8, int(np.linalg.norm(eye[0] - eye[1]) * 0.65))
        height = max(4, int(width * 0.28))
        center[1] -= height * 0.9
        mask = cv2.bitwise_or(
            mask,
            _ellipse_mask((h, w), center, (width, height), 0.0),
        )

    alpha = _soft_mask(mask, 17)
    return _blend_color(
        image_bgr,
        alpha,
        params.get("color", "#8C7A6B"),
        float(params.get("intensity", 0.25)),
        mode="multiply",
    )


def apply_eyeliner(
    image_bgr: np.ndarray,
    ctx: dict,
    params: dict | None = None,
) -> np.ndarray:
    params = params or {}
    if not params.get("enabled", False):
        return image_bgr

    points = _landmarks(ctx)
    if points is None or points.shape[0] <= 386:
        return image_bgr

    h, w = image_bgr.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    intensity = float(np.clip(params.get("intensity", 0.5), 0.0, 1.0))
    if intensity <= 0.0:
        return image_bgr

    face_width = float(np.linalg.norm(points[454].astype(np.float32) - points[234].astype(np.float32)))
    thickness = max(1, int(round(face_width * (0.006 + 0.004 * intensity))))

    eye_specs = (
        {
            "upper": [33, 246, 161, 160, 159, 158, 157, 173, 133],
            "lower": [33, 7, 163, 144, 145, 153, 154, 155, 133],
            "outer": 33,
            "inner": 133,
            "top": 159,
            "wing_dir": -1.0,
        },
        {
            "upper": [362, 398, 384, 385, 386, 387, 388, 466, 263],
            "lower": [362, 382, 381, 380, 374, 373, 390, 249, 263],
            "outer": 263,
            "inner": 362,
            "top": 386,
            "wing_dir": 1.0,
        },
    )

    for spec in eye_specs:
        upper = points[spec["upper"]].astype(np.float32)
        lower = points[spec["lower"]].astype(np.float32)
        outer = points[spec["outer"]].astype(np.float32)
        inner = points[spec["inner"]].astype(np.float32)
        top = points[spec["top"]].astype(np.float32)
        eye_width = max(1.0, float(np.linalg.norm(inner - outer)))

        # Move the line a touch above the lash landmarks so it reads as makeup
        # instead of painting over the iris/eyeball.
        y_lift = max(1.0, eye_width * 0.025)
        upper[:, 1] -= y_lift
        lower[:, 1] += y_lift * 0.35

        cv2.polylines(
            mask,
            [upper.astype(np.int32).reshape((-1, 1, 2))],
            isClosed=False,
            color=255,
            thickness=thickness,
            lineType=cv2.LINE_AA,
        )

        if intensity > 0.78:
            cv2.polylines(
                mask,
                [lower.astype(np.int32).reshape((-1, 1, 2))],
                isClosed=False,
                color=90,
                thickness=max(1, thickness - 1),
                lineType=cv2.LINE_AA,
            )

        wing_len = eye_width * (0.055 + 0.07 * intensity)
        wing_up = max(1.5, eye_width * (0.025 + 0.035 * intensity))
        wing_end = np.array(
            [outer[0] + spec["wing_dir"] * wing_len, top[1] - wing_up],
            dtype=np.float32,
        )
        cv2.line(
            mask,
            tuple(np.round(outer - np.array([0.0, y_lift], dtype=np.float32)).astype(int)),
            tuple(np.round(wing_end).astype(int)),
            255,
            max(1, thickness - 1),
            cv2.LINE_AA,
        )

    alpha = _soft_mask(mask, 1) * (0.52 + intensity * 0.26)
    if float(np.max(alpha)) <= 0.001:
        return image_bgr

    target = np.array(_hex_to_bgr(params.get("color", "#080808")), dtype=np.float32)
    target = np.clip(target * 0.72, 0, 255)
    base = image_bgr.astype(np.float32)
    color_layer = np.full_like(base, target)

    # Normal alpha blending keeps colored eyeliner visible; multiply made
    # non-black colors nearly disappear on skin.
    result = base * (1.0 - alpha[:, :, None]) + color_layer * alpha[:, :, None]

    ctx.setdefault("effect_debug_meta", {})["eyeliner"] = {
        "eyeliner_debug": {
            "mask_pixels": int(np.count_nonzero(alpha > 0.02)),
            "thickness": int(thickness),
        }
    }

    return np.clip(result, 0, 255).astype(np.uint8)
