from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_ONNX_PATHS = (
    ROOT_DIR / "models" / "bisenet_resnet18.onnx",
    ROOT_DIR / "models" / "bisenet_hair.onnx",
)
HAIR_LABEL = 17


def parse_target_color_rgb(color: Any, fallback: tuple[int, int, int] = (123, 63, 228)) -> tuple[int, int, int]:
    if isinstance(color, str):
        value = color.strip().lstrip("#")
        if len(value) == 6:
            try:
                return int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16)
            except ValueError:
                return fallback

    if isinstance(color, (list, tuple)) and len(color) >= 3:
        try:
            r, g, b = color[:3]
            return (
                int(np.clip(float(r), 0, 255)),
                int(np.clip(float(g), 0, 255)),
                int(np.clip(float(b), 0, 255)),
            )
        except Exception:
            return fallback

    return fallback


def _resolve_model_path(model_path: str | os.PathLike[str] | None = None) -> Path | None:
    if model_path:
        candidate = Path(model_path)
        if candidate.exists():
            return candidate.resolve()

    env_path = os.getenv("FACEWARP_BISENET_ONNX")
    if env_path:
        candidate = Path(env_path)
        if candidate.exists():
            return candidate.resolve()

    for candidate in DEFAULT_ONNX_PATHS:
        if candidate.exists():
            return candidate.resolve()

    return None


def _clean_hair_mask(mask: np.ndarray, image_shape: tuple[int, int]) -> np.ndarray:
    h, w = image_shape
    if mask.shape[:2] != (h, w):
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)

    mask_u8 = np.clip(mask, 0, 255).astype(np.uint8)
    _, binary = cv2.threshold(mask_u8, 32, 255, cv2.THRESH_BINARY)

    kernel3 = np.ones((3, 3), np.uint8)
    kernel5 = np.ones((5, 5), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel3, iterations=1)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel5, iterations=2)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if num_labels > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]
        keep = np.where(areas >= max(80, int(h * w * 0.001)))[0] + 1
        filtered = np.zeros_like(binary)
        for label in keep:
            filtered[labels == label] = 255
        binary = filtered

    soft = cv2.GaussianBlur(binary, (0, 0), sigmaX=max(2.0, min(h, w) * 0.012))
    return np.clip(soft, 0, 255).astype(np.uint8)


class HairSegmentor:
    """
    BiSeNet face parsing hair segmentor.

    It prefers ONNX Runtime when a model exists in models/bisenet_resnet18.onnx.
    If ONNX is not available, it falls back to the project's existing local
    PyTorch BiSeNet parser instead of guessing a geometric upper-head mask.
    """

    def __init__(self, model_path: str | os.PathLike[str] | None = None) -> None:
        self.model_path = _resolve_model_path(model_path)
        self.session = None
        self.input_name: str | None = None
        self.input_size = (512, 512)
        self.provider = "face_parsing_fallback"
        self.last_error: str | None = None

        if self.model_path is not None:
            self._load_onnx(self.model_path)

    def _load_onnx(self, model_path: Path) -> None:
        try:
            import onnxruntime as ort

            available = set(ort.get_available_providers())
            providers: list[str] = []
            if "CUDAExecutionProvider" in available:
                providers.append("CUDAExecutionProvider")
            providers.append("CPUExecutionProvider")

            self.session = ort.InferenceSession(str(model_path), providers=providers)
            self.input_name = self.session.get_inputs()[0].name

            shape = self.session.get_inputs()[0].shape
            if len(shape) == 4:
                height = shape[2] if isinstance(shape[2], int) else 512
                width = shape[3] if isinstance(shape[3], int) else 512
                self.input_size = (int(width), int(height))

            self.provider = f"onnxruntime:{self.session.get_providers()[0]}"
            self.last_error = None
        except Exception as exc:
            self.session = None
            self.input_name = None
            self.provider = "face_parsing_fallback"
            self.last_error = f"onnx_load_failed: {exc}"

    def _get_hair_mask_onnx(self, frame_bgr: np.ndarray) -> np.ndarray | None:
        if self.session is None or self.input_name is None:
            return None

        h, w = frame_bgr.shape[:2]
        img = cv2.resize(frame_bgr, self.input_size, interpolation=cv2.INTER_LINEAR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img = (img - mean) / std
        img = img.transpose(2, 0, 1)[np.newaxis, ...]

        try:
            output = self.session.run(None, {self.input_name: img})[0]
        except Exception as exc:
            self.last_error = f"onnx_inference_failed: {exc}"
            return None

        output = np.asarray(output)
        if output.ndim == 4:
            if output.shape[1] >= 19:
                parsing = output[0].argmax(axis=0).astype(np.uint8)
            else:
                parsing = output[0, 0].astype(np.uint8)
        elif output.ndim == 3:
            if output.shape[0] >= 19:
                parsing = output.argmax(axis=0).astype(np.uint8)
            else:
                parsing = output[0].astype(np.uint8)
        elif output.ndim == 2:
            parsing = output.astype(np.uint8)
        else:
            self.last_error = f"unexpected_onnx_output_shape: {output.shape}"
            return None

        hair_mask = (parsing == HAIR_LABEL).astype(np.uint8) * 255
        return _clean_hair_mask(hair_mask, (h, w))

    def _get_hair_mask_fallback(self, frame_bgr: np.ndarray) -> np.ndarray | None:
        try:
            from backend.face_parsing import get_mask, parse_face

            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            parsing = parse_face(rgb)
            hair_mask = get_mask(parsing, [HAIR_LABEL])
            if int(np.count_nonzero(hair_mask > 20)) == 0:
                self.last_error = "fallback_no_hair_pixels"
                return None
            self.provider = "face-parsing.PyTorch"
            return _clean_hair_mask(hair_mask, frame_bgr.shape[:2])
        except Exception as exc:
            self.last_error = f"fallback_failed: {exc}"
            return None

    def get_hair_mask(self, frame_bgr: np.ndarray) -> np.ndarray | None:
        if frame_bgr is None or frame_bgr.ndim != 3:
            return None

        mask = self._get_hair_mask_onnx(frame_bgr)
        if mask is None:
            mask = self._get_hair_mask_fallback(frame_bgr)
        return mask


def refine_mask_edges(hair_mask: np.ndarray, frame_bgr: np.ndarray) -> np.ndarray:
    h, w = frame_bgr.shape[:2]
    if hair_mask.shape[:2] != (h, w):
        hair_mask = cv2.resize(hair_mask, (w, h), interpolation=cv2.INTER_LINEAR)

    mask = np.clip(hair_mask, 0, 255).astype(np.uint8)
    guide = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

    try:
        if hasattr(cv2, "ximgproc") and hasattr(cv2.ximgproc, "guidedFilter"):
            refined = cv2.ximgproc.guidedFilter(
                guide=guide.astype(np.float32) / 255.0,
                src=mask.astype(np.float32) / 255.0,
                radius=8,
                eps=1e-3,
            )
            return np.clip(refined * 255.0, 0, 255).astype(np.uint8)
    except Exception:
        pass

    refined = cv2.bilateralFilter(mask, d=7, sigmaColor=55, sigmaSpace=9)
    refined = cv2.GaussianBlur(refined, (0, 0), sigmaX=2.0)
    return np.clip(refined, 0, 255).astype(np.uint8)


def apply_hair_color_hsl(
    frame_bgr: np.ndarray,
    hair_mask: np.ndarray,
    target_color_rgb: tuple[int, int, int],
    intensity: float = 0.85,
) -> np.ndarray:
    intensity = float(np.clip(intensity, 0.0, 1.0))
    if intensity <= 0.001:
        return frame_bgr.copy()

    h, w = frame_bgr.shape[:2]
    if hair_mask.shape[:2] != (h, w):
        hair_mask = cv2.resize(hair_mask, (w, h), interpolation=cv2.INTER_LINEAR)

    mask_f = np.clip(hair_mask.astype(np.float32) / 255.0, 0.0, 1.0)
    edge_alpha = np.power(mask_f, 1.12)

    target_bgr = np.uint8([[[target_color_rgb[2], target_color_rgb[1], target_color_rgb[0]]]])
    target_hls = cv2.cvtColor(target_bgr, cv2.COLOR_BGR2HLS)[0, 0].astype(np.float32)
    target_h = float(target_hls[0])
    target_s = float(target_hls[2])
    target_l = float(target_hls[1])
    low_chroma_light = target_l >= 135.0 and target_s < 112.0

    hls = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HLS).astype(np.float32)
    H = hls[:, :, 0]
    L = hls[:, :, 1]
    S = hls[:, :, 2]

    if target_l >= 135.0:
        if low_chroma_light:
            color_strength = float(np.clip(0.58 + intensity * 0.24, 0.0, 1.0))
            saturation_strength = float(np.clip(0.46 + intensity * 0.24, 0.0, 1.0))
            effective_target_s = target_s * 0.68
        else:
            color_strength = float(np.clip(0.78 + intensity * 0.18, 0.0, 1.0))
            saturation_strength = float(np.clip(0.66 + intensity * 0.28, 0.0, 1.0))
            effective_target_s = target_s * 0.88
    else:
        color_strength = float(np.clip(0.46 + intensity * 0.54, 0.0, 1.0))
        saturation_strength = float(np.clip(0.42 + intensity * 0.58, 0.0, 1.0))
        effective_target_s = target_s
    alpha_h = edge_alpha * color_strength
    alpha_s = edge_alpha * saturation_strength

    hue_delta = ((target_h - H + 90.0) % 180.0) - 90.0
    new_H = (H + hue_delta * alpha_h) % 180.0
    new_S = S * (1.0 - alpha_s) + effective_target_s * alpha_s

    # Light dye on dark hair needs a tone curve, not a flat RGB overlay.
    # Gamma raises the average lightness while retaining local strand contrast.
    selected = edge_alpha > 0.08
    hair_weights = edge_alpha[selected]
    hair_luma = L[selected]
    mean_l = (
        float(np.average(hair_luma, weights=hair_weights))
        if hair_luma.size and float(np.sum(hair_weights)) > 1e-6
        else float(np.mean(L))
    )
    mean_l = float(np.clip(mean_l, 8.0, 247.0))

    if target_l >= 135.0:
        light_score = float(np.clip((target_l - 135.0) / 100.0, 0.0, 1.0))
        tone_base = 0.72 if low_chroma_light else 0.82
        tone_amount = intensity * (tone_base + 0.08 * light_score)
        desired_mean = mean_l + (target_l - mean_l) * tone_amount
        desired_cap = 220.0 if low_chroma_light else 228.0
        desired_mean = float(np.clip(desired_mean, 10.0, desired_cap))
        gamma = float(
            np.clip(
                np.log(desired_mean / 255.0) / np.log(mean_l / 255.0),
                0.28,
                1.0,
            )
        )
        toned_l = 255.0 * np.power(np.clip(L / 255.0, 0.0, 1.0), gamma)
        shadow_detail = 0.70 + 0.30 * np.clip((L - 14.0) / 96.0, 0.0, 1.0)
        luma_base = 0.64 if low_chroma_light else 0.70
        luma_alpha = edge_alpha * (luma_base + intensity * 0.16) * shadow_detail
        new_L = L * (1.0 - luma_alpha) + toned_l * luma_alpha
    elif target_l < 65.0:
        tone_amount = intensity * 0.32
        desired_mean = mean_l + (target_l - mean_l) * tone_amount
        desired_mean = float(np.clip(desired_mean, 6.0, 245.0))
        gamma = float(
            np.clip(
                np.log(desired_mean / 255.0) / np.log(mean_l / 255.0),
                1.0,
                2.2,
            )
        )
        toned_l = 255.0 * np.power(np.clip(L / 255.0, 0.0, 1.0), gamma)
        luma_alpha = edge_alpha * (0.42 + intensity * 0.20)
        new_L = L * (1.0 - luma_alpha) + toned_l * luma_alpha
    else:
        new_L = L

    recolored_hls = np.dstack(
        [new_H, np.clip(new_L, 0, 255), np.clip(new_S, 0, 255)]
    ).astype(np.uint8)
    recolored = cv2.cvtColor(recolored_hls, cv2.COLOR_HLS2BGR).astype(np.float32)

    original = frame_bgr.astype(np.float32)
    is_light_target = target_l >= 135.0
    if low_chroma_light:
        core_alpha = 0.80 + intensity * 0.10
    elif is_light_target:
        core_alpha = 0.84 + intensity * 0.10
    else:
        core_alpha = 0.87 + intensity * 0.10
    alpha = (edge_alpha * float(np.clip(core_alpha, 0.0, 0.98)))[..., None]
    result = original * (1.0 - alpha) + recolored * alpha
    return np.clip(result, 0, 255).astype(np.uint8)


class FaceAppQualityHairEffect:
    def __init__(self, bisenet_model_path: str | os.PathLike[str] | None = None) -> None:
        self.segmentor = HairSegmentor(bisenet_model_path)

    def get_hair_mask(self, frame_bgr: np.ndarray, existing_mask: np.ndarray | None = None) -> np.ndarray | None:
        h, w = frame_bgr.shape[:2]

        if existing_mask is not None and int(np.count_nonzero(existing_mask > 20)) > 0:
            raw_mask = _clean_hair_mask(existing_mask, (h, w))
        else:
            raw_mask = self.segmentor.get_hair_mask(frame_bgr)

        if raw_mask is None or int(np.count_nonzero(raw_mask > 20)) == 0:
            return None

        # Photo requests must be stateless. Realtime temporal smoothing lives in
        # OptimalRealtimeHairProcessor, where masks are scoped to one session.
        return refine_mask_edges(raw_mask, frame_bgr)

    def apply(
        self,
        frame_bgr: np.ndarray,
        target_color_rgb: tuple[int, int, int],
        intensity: float = 0.85,
        hair_mask: np.ndarray | None = None,
    ) -> np.ndarray:
        refined_mask = self.get_hair_mask(frame_bgr, hair_mask)
        if refined_mask is None:
            return frame_bgr.copy()
        return apply_hair_color_hsl(frame_bgr, refined_mask, target_color_rgb, intensity)

    def reset(self) -> None:
        pass


_HAIR_EFFECT: FaceAppQualityHairEffect | None = None


def get_faceapp_hair_effect() -> FaceAppQualityHairEffect:
    global _HAIR_EFFECT
    if _HAIR_EFFECT is None:
        _HAIR_EFFECT = FaceAppQualityHairEffect()
    return _HAIR_EFFECT


def apply_faceapp_hair_color(
    image_bgr: np.ndarray,
    hair_mask: np.ndarray | None = None,
    color: Any = "#7B3FE4",
    intensity: float = 0.85,
) -> np.ndarray:
    effect = get_faceapp_hair_effect()
    target_rgb = parse_target_color_rgb(color)
    return effect.apply(image_bgr, target_rgb, intensity, hair_mask=hair_mask)
