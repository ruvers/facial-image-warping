import cv2
import numpy as np

from backend.face_parsing import feather_mask


def apply_hair_color(
    image_rgb: np.ndarray,
    hair_mask: np.ndarray,
    target_bgr=(180, 60, 60),
    intensity=0.75,
):
    """
    Realistic hair recoloring.

    Input:
        image_rgb: RGB uint8 image
        hair_mask: uint8 mask, 0-255
        target_bgr: target color in BGR
        intensity: 0.0 - 1.0

    Method:
        LAB color transfer.
        Preserves L channel, changes only chroma channels.
    """

    if image_rgb is None:
        raise ValueError("image_rgb is None")

    if hair_mask is None:
        raise ValueError("hair_mask is None")

    intensity = float(np.clip(intensity, 0.0, 1.0))

    h, w = image_rgb.shape[:2]

    if hair_mask.shape[:2] != (h, w):
        hair_mask = cv2.resize(
            hair_mask,
            (w, h),
            interpolation=cv2.INTER_NEAREST,
        )

    # Hard mask cleanup
    hair_mask = (hair_mask > 20).astype(np.uint8) * 255

    # Soft alpha mask
    alpha = feather_mask(
        hair_mask,
        blur=31,
    )

    alpha = np.clip(alpha, 0.0, 1.0)

    # RGB -> BGR
    img_bgr = cv2.cvtColor(
        image_rgb,
        cv2.COLOR_RGB2BGR,
    )

    # BGR -> LAB
    lab = cv2.cvtColor(
        img_bgr,
        cv2.COLOR_BGR2LAB,
    )

    l, a, b = cv2.split(lab)

    # Target color image
    target_img = np.full_like(
        img_bgr,
        target_bgr,
        dtype=np.uint8,
    )

    target_lab = cv2.cvtColor(
        target_img,
        cv2.COLOR_BGR2LAB,
    )

    _, target_a, target_b = cv2.split(target_lab)

    # Preserve highlights/shadows
    light_factor = l.astype(np.float32) / 255.0

    # Very dark regions should not become flat paint
    light_factor = np.clip(
        light_factor,
        0.30,
        1.0,
    )

    blend_strength = alpha * intensity * light_factor

    a_new = (
        a.astype(np.float32) * (1.0 - blend_strength)
        + target_a.astype(np.float32) * blend_strength
    )

    b_new = (
        b.astype(np.float32) * (1.0 - blend_strength)
        + target_b.astype(np.float32) * blend_strength
    )

    a_new = np.clip(a_new, 0, 255).astype(np.uint8)
    b_new = np.clip(b_new, 0, 255).astype(np.uint8)

    recolored_lab = cv2.merge([
        l,
        a_new,
        b_new,
    ])

    recolored_bgr = cv2.cvtColor(
        recolored_lab,
        cv2.COLOR_LAB2BGR,
    )

    result_rgb = cv2.cvtColor(
        recolored_bgr,
        cv2.COLOR_BGR2RGB,
    )

    return result_rgb