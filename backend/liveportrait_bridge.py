from __future__ import annotations

import importlib.util
import pickle
import sys
import threading
import uuid
from pathlib import Path
from typing import Any

import time
import cv2
import numpy as np


ROOT_DIR = Path(__file__).resolve().parents[1]
LIVEPORTRAIT_REPO_DIR = ROOT_DIR / "local_models" / "LivePortrait"
LIVEPORTRAIT_WEIGHTS_DIR = LIVEPORTRAIT_REPO_DIR / "pretrained_weights"

REQUIRED_WEIGHT_FILES = [
    LIVEPORTRAIT_WEIGHTS_DIR
    / "liveportrait"
    / "base_models"
    / "appearance_feature_extractor.pth",
    LIVEPORTRAIT_WEIGHTS_DIR / "liveportrait" / "base_models" / "motion_extractor.pth",
    LIVEPORTRAIT_WEIGHTS_DIR / "liveportrait" / "base_models" / "spade_generator.pth",
    LIVEPORTRAIT_WEIGHTS_DIR / "liveportrait" / "base_models" / "warping_module.pth",
    LIVEPORTRAIT_WEIGHTS_DIR
    / "liveportrait"
    / "retargeting_models"
    / "stitching_retargeting_module.pth",
    LIVEPORTRAIT_WEIGHTS_DIR / "liveportrait" / "landmark.onnx",
    LIVEPORTRAIT_WEIGHTS_DIR / "insightface" / "models" / "buffalo_l" / "det_10g.onnx",
    LIVEPORTRAIT_WEIGHTS_DIR
    / "insightface"
    / "models"
    / "buffalo_l"
    / "2d106det.onnx",
]

REQUIRED_MODULES = [
    "torch",
    "yaml",
    "onnx",
    "onnxruntime",
    "rich",
    "cv2",
    "numpy",
    "pykalman",
]
SUPPORTED_DIRECT_PRESETS = {"smile", "eyebrow_raise"}
SUPPORTED_TEMPLATE_NAMES = {"laugh", "open_lip", "wink", "shy", "aggrieved"}
TEMPLATE_SCORING_PRESETS = {
    "smile",
    "natural_smile",
    "soft_laugh",
    "laugh",
    "open_lip",
    "wink",
    "surprise",
    "aggrieved",
}
TEMPLATE_TO_SCORING_PRESET = {
    "laugh": "laugh",
    "open_lip": "open_lip",
    "wink": "wink",
    "shy": "smile",
    "aggrieved": "smile",
}
RUNTIME_OUTPUT_DIR = ROOT_DIR / "processed" / "liveportrait_runtime"
CANDIDATE_OUTPUT_DIR = ROOT_DIR / "processed" / "liveportrait_candidates"
OVERRIDE_TOP_NAMES = {
    f"top_{idx:02d}": idx - 1
    for idx in range(1, 6)
}

_RUNTIME = None
_RUNTIME_ERROR: str | None = None
_RUNTIME_LOCK = threading.Lock()


def _module_available(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def _missing_modules() -> list[str]:
    return [name for name in REQUIRED_MODULES if not _module_available(name)]


def _missing_weight_files() -> list[str]:
    return [str(path) for path in REQUIRED_WEIGHT_FILES if not path.exists()]


def _template_paths() -> dict[str, str]:
    driving_dir = LIVEPORTRAIT_REPO_DIR / "assets" / "examples" / "driving"
    if not driving_dir.exists():
        return {}

    return {
        path.stem: str(path)
        for path in sorted(driving_dir.glob("*.pkl"))
    }


def is_liveportrait_runtime_available() -> dict[str, Any]:
    repo_ok = (LIVEPORTRAIT_REPO_DIR / "inference.py").exists()
    missing_modules = _missing_modules()
    missing_weights = _missing_weight_files()
    templates = _template_paths()
    runtime_available = repo_ok and not missing_modules and not missing_weights

    return {
        "provider": "liveportrait",
        "repo_path": str(LIVEPORTRAIT_REPO_DIR),
        "weights_dir": str(LIVEPORTRAIT_WEIGHTS_DIR),
        "repo_ok": repo_ok,
        "weights_ok": not missing_weights,
        "files_available": repo_ok and not missing_weights,
        "runtime_available": runtime_available,
        "inference_bridge_implemented": runtime_available,
        "bridge_prototype_implemented": True,
        "missing_modules": missing_modules,
        "missing_weight_files": missing_weights,
        "supports_direct_single_image_presets": sorted(SUPPORTED_DIRECT_PRESETS),
        "supports_driving_templates": sorted(SUPPORTED_TEMPLATE_NAMES),
        "requires_driving_template_for_other_presets": True,
        "available_driving_templates": templates,
        "last_runtime_error": _RUNTIME_ERROR,
        "notes": [
            "Official CLI entrypoint is local_models/LivePortrait/inference.py.",
            "Official animation path requires source plus driving video or .pkl motion template.",
            "The Gradio image retargeting path supports direct single-image sliders including smile.",
            "This wrapper uses image retargeting for direct smile and the official pipeline for template-driven tests.",
        ],
    }


def _ensure_repo_on_path() -> None:
    repo = str(LIVEPORTRAIT_REPO_DIR)
    if repo not in sys.path:
        sys.path.insert(0, repo)


class _LivePortraitExpressionRuntime:
    def __init__(self) -> None:
        _ensure_repo_on_path()

        import torch
        from src.config.crop_config import CropConfig
        from src.config.inference_config import InferenceConfig
        from src.config.argument_config import ArgumentConfig
        import src.live_portrait_pipeline as live_portrait_pipeline_module
        import src.utils.video as video_module
        from src.live_portrait_pipeline import LivePortraitPipeline
        from src.live_portrait_wrapper import LivePortraitWrapper
        from src.utils.camera import get_rotation_matrix
        from src.utils.crop import paste_back, prepare_paste_back
        from src.utils.cropper import Cropper

        self.torch = torch
        self.ArgumentConfig = ArgumentConfig
        self.LivePortraitPipeline = LivePortraitPipeline
        self.live_portrait_pipeline_module = live_portrait_pipeline_module
        self.video_module = video_module
        self.get_rotation_matrix = get_rotation_matrix
        self.prepare_paste_back = prepare_paste_back
        self.paste_back = paste_back
        self.apply_lock = threading.Lock()

        force_cpu = not torch.cuda.is_available()

        self.inference_cfg = InferenceConfig(
            flag_force_cpu=force_cpu,
            flag_use_half_precision=False,
            flag_do_torch_compile=False,
        )
        self.crop_cfg = CropConfig(
            flag_force_cpu=force_cpu,
            scale=2.3,
            vx_ratio=0.0,
            vy_ratio=-0.125,
            det_thresh=0.1,
        )
        self.wrapper = LivePortraitWrapper(inference_cfg=self.inference_cfg)
        self.live_portrait_wrapper = self.wrapper
        self.cropper = Cropper(crop_cfg=self.crop_cfg, flag_force_cpu=force_cpu)

    @staticmethod
    def _plain_track(iterable: Any, *args: Any, **kwargs: Any) -> Any:
        return iterable

    def _disable_rich_progress_for_windows_console(self) -> None:
        self.live_portrait_pipeline_module.track = self._plain_track
        self.video_module.track = self._plain_track

    @staticmethod
    def _safe_dist(
        landmarks: np.ndarray,
        a: int,
        b: int,
    ) -> float:
        if landmarks.shape[0] <= max(a, b):
            return 0.0
        return float(np.linalg.norm(landmarks[a] - landmarks[b]))

    @staticmethod
    def _safe_y(
        landmarks: np.ndarray,
        idx: int,
    ) -> float:
        if landmarks.shape[0] <= idx:
            return 0.0
        return float(landmarks[idx][1])

    @staticmethod
    def _extract_frame_landmarks(frame_bgr: np.ndarray) -> np.ndarray | None:
        try:
            from backend.face_analysis import _get_face_landmarker
            import mediapipe as mp

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=frame_rgb,
            )
            results = _get_face_landmarker().detect(mp_image)
            if not results.face_landmarks:
                return None

            h, w = frame_bgr.shape[:2]
            return np.array(
                [
                    [
                        float(np.clip(lm.x * w, 0, w - 1)),
                        float(np.clip(lm.y * h, 0, h - 1)),
                    ]
                    for lm in results.face_landmarks[0]
                ],
                dtype=np.float32,
            )
        except Exception:
            return None

    def _expression_metrics(
        self,
        landmarks: np.ndarray,
    ) -> dict[str, float]:
        face_width = max(
            self._safe_dist(landmarks, 234, 454),
            self._safe_dist(landmarks, 127, 356),
            1.0,
        )
        face_height = max(
            self._safe_dist(landmarks, 10, 152),
            self._safe_dist(landmarks, 168, 152),
            face_width,
            1.0,
        )

        mouth_width = self._safe_dist(landmarks, 61, 291) / face_width
        mouth_open = self._safe_dist(landmarks, 13, 14) / face_height
        mouth_center_y = (
            self._safe_y(landmarks, 13)
            + self._safe_y(landmarks, 14)
        ) / 2.0
        mouth_corner_y = (
            self._safe_y(landmarks, 61)
            + self._safe_y(landmarks, 291)
        ) / 2.0
        corner_raise = (mouth_center_y - mouth_corner_y) / face_height
        mouth_asymmetry = abs(
            self._safe_y(landmarks, 61) - self._safe_y(landmarks, 291)
        ) / face_height

        left_eye_open = self._safe_dist(landmarks, 159, 145) / face_height
        right_eye_open = self._safe_dist(landmarks, 386, 374) / face_height
        eye_open = (left_eye_open + right_eye_open) / 2.0
        eye_asymmetry = abs(left_eye_open - right_eye_open)

        brow_inner_y = (
            self._safe_y(landmarks, 105)
            + self._safe_y(landmarks, 334)
        ) / 2.0
        brow_outer_y = (
            self._safe_y(landmarks, 70)
            + self._safe_y(landmarks, 300)
        ) / 2.0
        brow_position = brow_inner_y / face_height
        brow_frown = max(0.0, (brow_inner_y - brow_outer_y) / face_height)
        cheek_y = (
            self._safe_y(landmarks, 116)
            + self._safe_y(landmarks, 345)
        ) / 2.0 / face_height

        return {
            "mouth_width": mouth_width,
            "mouth_open": mouth_open,
            "corner_raise": corner_raise,
            "mouth_asymmetry": mouth_asymmetry,
            "left_eye_open": left_eye_open,
            "right_eye_open": right_eye_open,
            "eye_open": eye_open,
            "eye_asymmetry": eye_asymmetry,
            "cheek_y": cheek_y,
            "brow_position": brow_position,
            "brow_frown": brow_frown,
        }

    @staticmethod
    def _positive_delta(value: float, scale: float) -> float:
        return max(0.0, value) / max(scale, 1e-6)

    @staticmethod
    def _target_band_score(value: float, target: float, tolerance: float) -> float:
        return max(0.0, 1.0 - abs(value - target) / max(tolerance, 1e-6))

    def _expression_quality_scores(
        self,
        metrics: dict[str, float],
        baseline: dict[str, float],
    ) -> dict[str, float]:
        mouth_width_gain = metrics["mouth_width"] - baseline["mouth_width"]
        mouth_open_gain = metrics["mouth_open"] - baseline["mouth_open"]
        lip_corner_raise = metrics["corner_raise"] - baseline["corner_raise"]
        cheek_raise = baseline["cheek_y"] - metrics["cheek_y"]
        eye_squint = baseline["eye_open"] - metrics["eye_open"]
        brow_down = metrics["brow_position"] - baseline["brow_position"]

        grimace_penalty = (
            metrics["mouth_asymmetry"] * 5.0
            + metrics["eye_asymmetry"] * 2.0
            + metrics["brow_frown"] * 2.5
            + self._positive_delta(brow_down, 0.025) * 1.4
        )
        scream_penalty = (
            self._positive_delta(metrics["mouth_open"] - 0.175, 0.045) * 2.4
            + self._positive_delta(mouth_open_gain - 0.120, 0.045) * 2.0
            + self._positive_delta(metrics["eye_open"] - baseline["eye_open"] - 0.020, 0.025)
            + self._positive_delta(-lip_corner_raise, 0.025) * 1.4
        )

        smile_score = (
            self._positive_delta(mouth_width_gain, 0.060) * 1.6
            + self._positive_delta(lip_corner_raise, 0.035) * 3.1
            + self._positive_delta(cheek_raise, 0.024) * 1.4
            + self._target_band_score(eye_squint, 0.010, 0.028) * 0.5
            - self._positive_delta(metrics["mouth_open"] - 0.125, 0.035) * 3.6
            - self._positive_delta(mouth_open_gain - 0.045, 0.035) * 2.8
            - scream_penalty * 0.8
            - grimace_penalty
        )

        laugh_score = (
            self._target_band_score(metrics["mouth_open"], 0.15, 0.08) * 4.0
            + self._positive_delta(lip_corner_raise, 0.03) * 3.5
            + self._positive_delta(cheek_raise, 0.02) * 2.0
            + self._positive_delta(mouth_width_gain, 0.06) * 1.5
            + self._target_band_score(eye_squint, 0.015, 0.015) * 0.8
            - self._positive_delta(0.05 - metrics["mouth_open"], 0.05) * 5.0
            - self._positive_delta(metrics["brow_frown"], 0.02) * 2.0
            - self._positive_delta(metrics["mouth_asymmetry"], 0.03) * 2.0
            - self._positive_delta(eye_squint - 0.04, 0.03) * 1.5
        )
        if lip_corner_raise < 0.0 and metrics["mouth_open"] > 0.08:
            laugh_score -= abs(lip_corner_raise) * 4.0

        return {
            "mouth_open": float(metrics["mouth_open"]),
            "mouth_open_delta": float(mouth_open_gain),
            "lip_corner_raise": float(lip_corner_raise),
            "cheek_raise": float(cheek_raise),
            "eye_squint": float(eye_squint),
            "brow_down": float(brow_down),
            "grimace_penalty": float(grimace_penalty),
            "scream_penalty": float(scream_penalty),
            "smile_score": float(smile_score),
            "laugh_score": float(laugh_score),
        }

    @staticmethod
    def _candidate_rejected(
        preset: str,
        quality: dict[str, float],
    ) -> bool:
        if preset in {"smile", "natural_smile", "broad_smile"}:
            return (
                quality.get("mouth_open", 0.0) > 0.135
                or quality.get("grimace_penalty", 0.0) > 2.6
                or quality.get("scream_penalty", 0.0) > 1.4
                or quality.get("lip_corner_raise", 0.0) < -0.005
            )

        if preset in {"laugh", "soft_laugh"}:
            mouth_open = quality.get("mouth_open", 0.0)
            corner_raise = quality.get("lip_corner_raise", 0.0)
            return (
                mouth_open > 0.28
                or quality.get("grimace_penalty", 0.0) > 3.2
                or (mouth_open > 0.10 and corner_raise <= 0.0)
            )

        return False

    def score_smile_frame(
        self,
        metrics: dict[str, float],
        baseline: dict[str, float],
    ) -> float:
        return self._expression_quality_scores(metrics, baseline)["smile_score"]

    def score_laugh_frame(
        self,
        metrics: dict[str, float],
        baseline: dict[str, float],
    ) -> float:
        mouth_width_gain = metrics["mouth_width"] - baseline["mouth_width"]
        corner_raise = metrics["corner_raise"] - baseline["corner_raise"]
        cheek_lift_gain = baseline["cheek_y"] - metrics["cheek_y"]
        eye_open_decrease = baseline["eye_open"] - metrics["eye_open"]
        mouth_open = metrics["mouth_open"]

        score = (
            self._target_band_score(mouth_open, 0.15, 0.08) * 4.0
            + self._positive_delta(corner_raise, 0.03) * 3.5
            + self._positive_delta(cheek_lift_gain, 0.02) * 2.0
            + self._positive_delta(mouth_width_gain, 0.06) * 1.5
            + self._target_band_score(eye_open_decrease, 0.015, 0.015) * 0.8
            - self._positive_delta(0.05 - mouth_open, 0.05) * 5.0
            - self._positive_delta(metrics["brow_frown"], 0.02) * 2.0
            - self._positive_delta(metrics["mouth_asymmetry"], 0.03) * 2.0
            - self._positive_delta(eye_open_decrease - 0.04, 0.03) * 1.5
        )
        if corner_raise < 0.0 and mouth_open > 0.08:
            score -= abs(corner_raise) * 4.0
        return score

    def score_soft_laugh_frame(
        self,
        metrics: dict[str, float],
        baseline: dict[str, float],
    ) -> float:
        mouth_width_gain = metrics["mouth_width"] - baseline["mouth_width"]
        mouth_open_gain = metrics["mouth_open"] - baseline["mouth_open"]
        corner_lift_gain = metrics["corner_raise"] - baseline["corner_raise"]
        cheek_lift_gain = baseline["cheek_y"] - metrics["cheek_y"]
        eye_softening = baseline["eye_open"] - metrics["eye_open"]
        brow_down = metrics["brow_position"] - baseline["brow_position"]

        moderate_open = self._target_band_score(metrics["mouth_open"], 0.075, 0.040)
        moderate_open_gain = self._target_band_score(mouth_open_gain, 0.030, 0.035)
        mild_eye_softening = self._target_band_score(eye_softening, 0.010, 0.025)

        return (
            self._positive_delta(mouth_width_gain, 0.070) * 1.7
            + self._positive_delta(corner_lift_gain, 0.032) * 2.4
            + self._positive_delta(cheek_lift_gain, 0.024) * 1.2
            + moderate_open * 1.0
            + moderate_open_gain * 0.8
            + mild_eye_softening * 0.5
            - self._positive_delta(metrics["mouth_open"] - 0.115, 0.035) * 5.0
            - self._positive_delta(mouth_open_gain - 0.065, 0.030) * 3.5
            - self._positive_delta(0.030 - metrics["mouth_open"], 0.030) * 1.4
            - self._positive_delta(
                baseline["eye_open"] - metrics["eye_open"] - 0.040,
                0.030,
            ) * 1.4
            - metrics["mouth_asymmetry"] * 5.0
            - metrics["eye_asymmetry"] * 2.0
            - metrics["brow_frown"] * 2.0
            - self._positive_delta(brow_down, 0.020) * 1.3
        )

    def score_wink_frame(
        self,
        metrics: dict[str, float],
        baseline: dict[str, float],
    ) -> float:
        left_close = baseline["left_eye_open"] - metrics["left_eye_open"]
        right_close = baseline["right_eye_open"] - metrics["right_eye_open"]
        one_eye_close = max(left_close, right_close)
        other_eye_close = min(left_close, right_close)

        return (
            self._positive_delta(one_eye_close, 0.06) * 2.8
            + self._positive_delta(metrics["eye_asymmetry"], 0.07) * 2.0
            - self._positive_delta(other_eye_close, 0.03) * 2.0
            - self._positive_delta(
                metrics["mouth_open"] - baseline["mouth_open"],
                0.08,
            ) * 0.7
        )

    def score_open_lip_frame(
        self,
        metrics: dict[str, float],
        baseline: dict[str, float],
    ) -> float:
        mouth_open_gain = metrics["mouth_open"] - baseline["mouth_open"]
        mouth_width_gain = metrics["mouth_width"] - baseline["mouth_width"]
        corner_lift_gain = metrics["corner_raise"] - baseline["corner_raise"]

        return (
            self._positive_delta(mouth_open_gain, 0.075) * 4.0
            + self._positive_delta(metrics["mouth_open"] - 0.055, 0.070) * 1.6
            - self._positive_delta(metrics["mouth_open"] - 0.160, 0.050) * 2.4
            - self._positive_delta(abs(mouth_width_gain), 0.09) * 0.7
            - self._positive_delta(abs(corner_lift_gain), 0.04) * 1.0
            - metrics["mouth_asymmetry"] * 2.0
            - metrics["eye_asymmetry"] * 1.0
        )

    def score_surprise_frame(
        self,
        metrics: dict[str, float],
        baseline: dict[str, float],
    ) -> float:
        mouth_open_gain = metrics["mouth_open"] - baseline["mouth_open"]
        brow_up = baseline["brow_position"] - metrics["brow_position"]
        eye_open_gain = metrics["eye_open"] - baseline["eye_open"]

        return (
            self._positive_delta(mouth_open_gain, 0.070) * 3.4
            + self._positive_delta(metrics["mouth_open"] - 0.060, 0.065) * 1.6
            + self._positive_delta(brow_up, 0.018) * 1.2
            + self._positive_delta(eye_open_gain, 0.015) * 0.8
            - self._positive_delta(metrics["mouth_open"] - 0.170, 0.050) * 2.5
            - self._positive_delta(metrics["corner_raise"] - baseline["corner_raise"], 0.040) * 1.2
            - metrics["mouth_asymmetry"] * 2.0
            - metrics["brow_frown"] * 2.4
        )

    def score_shy_frame(
        self,
        metrics: dict[str, float],
        baseline: dict[str, float],
    ) -> float:
        return self.score_smile_frame(metrics, baseline) * 0.8

    def score_aggrieved_frame(
        self,
        metrics: dict[str, float],
        baseline: dict[str, float],
    ) -> float:
        return self.score_smile_frame(metrics, baseline) * 0.65

    def _score_expression_frame(
        self,
        preset: str,
        metrics: dict[str, float],
        baseline: dict[str, float],
    ) -> float:
        scorers = {
            "smile": self.score_smile_frame,
            "natural_smile": self.score_smile_frame,
            "broad_smile": self.score_smile_frame,
            "soft_laugh": self.score_soft_laugh_frame,
            "laugh": self.score_laugh_frame,
            "wink": self.score_wink_frame,
            "open_lip": self.score_open_lip_frame,
            "surprise": self.score_surprise_frame,
            "shy": self.score_shy_frame,
            "aggrieved": self.score_aggrieved_frame,
        }

        scorer = scorers.get(preset)
        if scorer is not None:
            return scorer(metrics, baseline)

        return self.score_smile_frame(metrics, baseline)

    @staticmethod
    def _export_candidate_frames(
        frames: list[np.ndarray],
        candidate_dir: Path,
        template_name: str,
    ) -> list[str]:
        candidate_dir.mkdir(parents=True, exist_ok=True)
        paths = []

        for idx, frame in enumerate(frames):
            path = candidate_dir / f"{template_name}_frame_{idx:03d}.png"
            cv2.imwrite(str(path), frame)
            paths.append(str(path))

        return paths

    @staticmethod
    def _export_top_frames(
        frames: list[np.ndarray],
        top_scores: list[dict[str, Any]],
        candidate_dir: Path,
    ) -> list[str]:
        paths = []

        for idx, item in enumerate(top_scores[:5], start=1):
            frame_index = int(item["frame_index"])
            if frame_index < 0 or frame_index >= len(frames):
                continue

            path = candidate_dir / f"top_{idx:02d}.png"
            cv2.imwrite(str(path), frames[frame_index])
            paths.append(str(path))

        return paths

    @staticmethod
    def _resolve_candidate_override(
        override: Any,
        top_scores: list[dict[str, Any]],
        frame_count: int,
    ) -> int | None:
        if override in (None, "", "auto"):
            return None

        if isinstance(override, str):
            key = override.strip().lower()
            if key in OVERRIDE_TOP_NAMES:
                top_idx = OVERRIDE_TOP_NAMES[key]
                if top_idx < len(top_scores):
                    return int(top_scores[top_idx]["frame_index"])

            try:
                candidate_idx = int(key)
            except ValueError:
                return None
        else:
            try:
                candidate_idx = int(override)
            except (TypeError, ValueError):
                return None

        if 0 <= candidate_idx < frame_count:
            return candidate_idx

        return None

    def _score_frames_by_preset(
        self,
        frames: list[np.ndarray],
        preset: str,
    ) -> tuple[int, float, list[dict[str, Any]]]:
        baseline_landmarks = self._extract_frame_landmarks(frames[0])
        if baseline_landmarks is None:
            return 0, 0.0, []

        baseline = self._expression_metrics(baseline_landmarks)
        scores: list[dict[str, Any]] = []
        best_idx = 0
        best_score = float("-inf")

        for idx, frame in enumerate(frames):
            landmarks = self._extract_frame_landmarks(frame)
            if landmarks is None:
                continue

            metrics = self._expression_metrics(landmarks)
            quality = self._expression_quality_scores(metrics, baseline)
            score = self._score_expression_frame(preset, metrics, baseline)
            item = {
                "frame_index": idx,
                "score": float(score),
                "scores": quality,
                "rejected": self._candidate_rejected(preset, quality),
                "mouth_width_delta": float(
                    metrics["mouth_width"] - baseline["mouth_width"]
                ),
                "mouth_open_delta": float(
                    metrics["mouth_open"] - baseline["mouth_open"]
                ),
                "corner_lift_delta": float(
                    metrics["corner_raise"] - baseline["corner_raise"]
                ),
                "cheek_lift_delta": float(
                    baseline["cheek_y"] - metrics["cheek_y"]
                ),
                "brow_down_delta": float(
                    metrics["brow_position"] - baseline["brow_position"]
                ),
                "mouth_open_ratio": float(metrics["mouth_open"]),
                "eye_open_delta": float(
                    metrics["eye_open"] - baseline["eye_open"]
                ),
                "mouth_asymmetry": float(metrics["mouth_asymmetry"]),
                "eye_asymmetry": float(metrics["eye_asymmetry"]),
            }
            scores.append(item)

            if score > best_score:
                best_score = score
                best_idx = idx

        if not scores:
            return 0, 0.0, []

        scores.sort(
            key=lambda item: item["score"],
            reverse=True,
        )

        selected = next(
            (item for item in scores if not bool(item.get("rejected", False))),
            scores[0],
        )

        return (
            int(selected["frame_index"]),
            float(selected["score"]),
            scores[:5],
        )

    @staticmethod
    def _apply_natural_smile_delta(delta_new: Any, smile: Any) -> Any:
        # Mirrors LivePortrait's Gradio smile retargeter with moderate values.
        delta_new[0, 20, 1] += smile * -0.01
        delta_new[0, 14, 1] += smile * -0.02
        delta_new[0, 17, 1] += smile * 0.0065
        delta_new[0, 17, 2] += smile * 0.003
        delta_new[0, 13, 1] += smile * -0.00275
        delta_new[0, 16, 1] += smile * -0.00275
        delta_new[0, 3, 1] += smile * -0.0035
        delta_new[0, 7, 1] += smile * -0.0035
        return delta_new

    @staticmethod
    def _apply_subtle_lip_grin_delta(delta_new: Any, grin: Any) -> Any:
        delta_new[0, 20, 2] += grin * -0.001
        delta_new[0, 20, 1] += grin * -0.001
        delta_new[0, 14, 1] += grin * -0.001
        return delta_new

    @staticmethod
    def _add_delta(delta_new: Any, idx: int, axis: int, value: Any) -> None:
        try:
            if delta_new.shape[1] > idx and delta_new.shape[2] > axis:
                delta_new[0, idx, axis] += value
        except Exception:
            return

    def _apply_eyebrow_raise_delta(
        self,
        delta_new: Any,
        raise_value: Any,
        eye_open_value: Any,
    ) -> Any:
        # Upper-face direct retargeting. Stronger brow lift, restrained eye opening.
        for idx in (13, 16):
            self._add_delta(delta_new, idx, 1, raise_value * -0.0090)

        for idx in (11, 15):
            self._add_delta(delta_new, idx, 1, raise_value * -0.0055)

        for idx in (3, 7):
            self._add_delta(delta_new, idx, 1, eye_open_value * -0.0005)

        for idx in (1, 2, 4, 6, 8, 9):
            self._add_delta(delta_new, idx, 1, raise_value * -0.0015)

        return delta_new

    def apply_smile(
        self,
        image_bgr: np.ndarray,
        intensity: float,
        params: dict | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        if image_bgr is None:
            raise ValueError("image_bgr is None")

        params = params or {}
        intensity = float(np.clip(intensity, 0.0, 1.0))
        smile_strength = float(
            np.clip(
                params.get("smile_strength", 1.45 * intensity),
                0.0,
                1.60,
            )
        )
        lip_open_strength = float(
            np.clip(
                params.get("lip_open_strength", 0.045 * intensity),
                0.0,
                0.18,
            )
        )
        eye_soften_strength = float(
            np.clip(
                params.get("eye_soften_strength", 0.36 * intensity),
                0.0,
                0.45,
            )
        )
        preserve_identity_strength = float(
            np.clip(
                params.get("preserve_identity_strength", 0.92),
                0.35,
                1.0,
            )
        )
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        crop_info = self.cropper.crop_source_image(image_rgb, self.crop_cfg)
        if crop_info is None:
            raise RuntimeError("LivePortrait could not detect a face in the source image.")

        I_s = self.wrapper.prepare_source(crop_info["img_crop_256x256"])
        x_s_info = self.wrapper.get_kp_info(I_s)
        R_s = self.get_rotation_matrix(
            x_s_info["pitch"],
            x_s_info["yaw"],
            x_s_info["roll"],
        )
        f_s = self.wrapper.extract_feature_3d(I_s)
        x_s = self.wrapper.transform_keypoint(x_s_info)

        device = self.wrapper.device
        x_c_s = x_s_info["kp"].to(device)
        delta_new = x_s_info["exp"].to(device)
        scale_new = x_s_info["scale"].to(device)
        t_new = x_s_info["t"].to(device)

        # Natural smiles combine lip-corner pull with mild cheek/eye-region lift.
        smile_value = self.torch.tensor(smile_strength, device=device)
        grin_value = self.torch.tensor(
            (1.25 + lip_open_strength * 0.6) * intensity,
            device=device,
        )
        delta_new = self._apply_natural_smile_delta(delta_new, smile_value)
        delta_new = self._apply_subtle_lip_grin_delta(delta_new, grin_value)

        if eye_soften_strength > 0:
            eye_soften = self.torch.tensor(eye_soften_strength, device=device)
            delta_new[0, 13, 1] += eye_soften * 0.0006
            delta_new[0, 16, 1] += eye_soften * 0.0006

        if preserve_identity_strength < 1.0:
            base_delta = x_s_info["exp"].to(device)
            blend = self.torch.tensor(preserve_identity_strength, device=device)
            delta_new = base_delta + (delta_new - base_delta) * blend

        x_d_new = scale_new * (x_c_s @ R_s + delta_new) + t_new
        x_d_new = self.wrapper.stitching(x_s, x_d_new)
        out = self.wrapper.warp_decode(f_s, x_s, x_d_new)
        out_rgb = self.wrapper.parse_output(out["out"])[0]

        mask_ori = self.prepare_paste_back(
            self.inference_cfg.mask_crop,
            crop_info["M_c2o"],
            dsize=(image_rgb.shape[1], image_rgb.shape[0]),
        )
        out_to_ori_rgb = self.paste_back(
            out_rgb,
            crop_info["M_c2o"],
            image_rgb,
            mask_ori,
        )
        out_bgr = cv2.cvtColor(out_to_ori_rgb, cv2.COLOR_RGB2BGR)

        return out_bgr, {
            "mode": "direct_smile",
            "direct_retargeting": True,
            "used_gradio_retargeting_formula": True,
            "smile_slider_value": float(smile_value.detach().cpu().item()),
            "lip_grin_slider_value": float(grin_value.detach().cpu().item()),
            "smile_strength": smile_strength,
            "lip_open_strength": lip_open_strength,
            "eye_soften_strength": eye_soften_strength,
            "preserve_identity_strength": preserve_identity_strength,
        }

    def apply_eyebrow_raise(
        self,
        image_bgr: np.ndarray,
        intensity: float,
        params: dict | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        if image_bgr is None:
            raise ValueError("image_bgr is None")

        params = params or {}
        intensity = float(np.clip(intensity, 0.0, 1.0))
        eyebrow_raise_strength = float(
            np.clip(
                params.get("eyebrow_raise_strength", 1.85 * intensity),
                0.0,
                2.20,
            )
        )
        eye_open_strength = float(
            np.clip(
                params.get("eye_open_strength", 0.18 * intensity),
                0.0,
                0.35,
            )
        )
        preserve_identity_strength = float(
            np.clip(
                params.get("preserve_identity_strength", 0.88),
                0.35,
                1.0,
            )
        )
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        crop_info = self.cropper.crop_source_image(image_rgb, self.crop_cfg)
        if crop_info is None:
            raise RuntimeError("LivePortrait could not detect a face in the source image.")

        I_s = self.wrapper.prepare_source(crop_info["img_crop_256x256"])
        x_s_info = self.wrapper.get_kp_info(I_s)
        R_s = self.get_rotation_matrix(
            x_s_info["pitch"],
            x_s_info["yaw"],
            x_s_info["roll"],
        )
        f_s = self.wrapper.extract_feature_3d(I_s)
        x_s = self.wrapper.transform_keypoint(x_s_info)

        device = self.wrapper.device
        x_c_s = x_s_info["kp"].to(device)
        delta_new = x_s_info["exp"].to(device)
        scale_new = x_s_info["scale"].to(device)
        t_new = x_s_info["t"].to(device)

        raise_value = self.torch.tensor(eyebrow_raise_strength, device=device)
        eye_open_value = self.torch.tensor(eye_open_strength, device=device)
        base_delta = x_s_info["exp"].to(device)
        delta_new = self._apply_eyebrow_raise_delta(
            delta_new,
            raise_value,
            eye_open_value,
        )

        if preserve_identity_strength < 1.0:
            blend = self.torch.tensor(preserve_identity_strength, device=device)
            delta_new = base_delta + (delta_new - base_delta) * blend

        x_d_new = scale_new * (x_c_s @ R_s + delta_new) + t_new
        x_d_new = self.wrapper.stitching(x_s, x_d_new)
        out = self.wrapper.warp_decode(f_s, x_s, x_d_new)
        out_rgb = self.wrapper.parse_output(out["out"])[0]

        mask_ori = self.prepare_paste_back(
            self.inference_cfg.mask_crop,
            crop_info["M_c2o"],
            dsize=(image_rgb.shape[1], image_rgb.shape[0]),
        )
        out_to_ori_rgb = self.paste_back(
            out_rgb,
            crop_info["M_c2o"],
            image_rgb,
            mask_ori,
        )
        out_bgr = cv2.cvtColor(out_to_ori_rgb, cv2.COLOR_RGB2BGR)

        return out_bgr, {
            "mode": "direct_eyebrow_raise",
            "direct_retargeting": True,
            "eyebrow_raise_strength": eyebrow_raise_strength,
            "eye_open_strength": eye_open_strength,
            "preserve_identity_strength": preserve_identity_strength,
        }

    def _representative_frame_from_video(
        self,
        video_path: Path,
        preset: str,
        template_name: str,
        candidate_dir: Path,
        candidate_frame_override: Any = "auto",
    ) -> tuple[np.ndarray, int, int, float, list[dict[str, Any]], str, list[str], list[str]]:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open LivePortrait video output: {video_path}")

        frames: list[np.ndarray] = []
        ok, frame = cap.read()
        while ok:
            frames.append(frame)
            ok, frame = cap.read()
        cap.release()

        if not frames:
            raise RuntimeError(f"LivePortrait video output had no readable frames: {video_path}")

        candidate_paths = self._export_candidate_frames(
            frames,
            candidate_dir,
            template_name,
        )

        if len(frames) == 1:
            return frames[0], 0, 1, 0.0, [], "single_frame", candidate_paths, []

        best_idx, best_score, top_scores = self._score_frames_by_preset(
            frames,
            preset,
        )
        top_paths = self._export_top_frames(
            frames,
            top_scores,
            candidate_dir,
        )

        if top_scores:
            override_idx = self._resolve_candidate_override(
                candidate_frame_override,
                top_scores,
                len(frames),
            )
            if override_idx is not None:
                best_idx = override_idx
                matched = [
                    item
                    for item in top_scores
                    if int(item["frame_index"]) == best_idx
                ]
                best_score = float(matched[0]["score"]) if matched else 0.0
                strategy = f"manual_override:{candidate_frame_override}"
            else:
                strategy = f"preset_landmark_score:{preset}"

            return (
                frames[best_idx],
                best_idx,
                len(frames),
                best_score,
                top_scores,
                strategy,
                candidate_paths,
                top_paths,
            )

        base = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY).astype(np.float32)
        best_idx = 0
        best_score = -1.0

        for idx, candidate in enumerate(frames[1:], start=1):
            gray = cv2.cvtColor(candidate, cv2.COLOR_BGR2GRAY).astype(np.float32)
            score = float(np.mean(np.abs(gray - base)))
            if score > best_score:
                best_score = score
                best_idx = idx

        return (
            frames[best_idx],
            best_idx,
            len(frames),
            float(best_score),
            [],
            "fallback_max_grayscale_difference_from_first_frame",
            candidate_paths,
            top_paths,
        )

    def apply_driving_template(
        self,
        image_bgr: np.ndarray,
        template_name: str,
        intensity: float,
        scoring_preset: str | None = None,
        candidate_frame_override: Any = "auto",
    ) -> tuple[np.ndarray, dict[str, Any]]:
        if image_bgr is None:
            raise ValueError("image_bgr is None")

        template_paths = _template_paths()
        template_path_str = template_paths.get(template_name)
        if not template_path_str:
            raise FileNotFoundError(f"LivePortrait driving template not found: {template_name}")

        run_id = uuid.uuid4().hex
        run_dir = RUNTIME_OUTPUT_DIR / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        candidate_dir = CANDIDATE_OUTPUT_DIR / run_id

        source_path = run_dir / "source.png"
        if not cv2.imwrite(str(source_path), image_bgr):
            raise RuntimeError(f"Could not write LivePortrait source image: {source_path}")

        template_for_run = run_dir / f"{template_name}.pkl"
        with open(template_path_str, "rb") as f:
            template_data = pickle.load(f)

        n_frames = int(template_data.get("n_frames") or len(template_data.get("motion", [])))
        if n_frames <= 0:
            raise RuntimeError(f"LivePortrait template has no frames: {template_path_str}")

        template_data.setdefault(
            "c_eyes_lst",
            [np.zeros((1, 2), dtype=np.float32) for _ in range(n_frames)],
        )
        template_data.setdefault(
            "c_lip_lst",
            [np.zeros((1, 1), dtype=np.float32) for _ in range(n_frames)],
        )

        with open(template_for_run, "wb") as f:
            pickle.dump(template_data, f)

        scoring_key = str(scoring_preset or template_name).lower()
        if str(template_name).lower() == "laugh":
            driving_multiplier = float(np.clip(intensity * 1.3, 0.4, 1.8))
        else:
            driving_multiplier = float(np.clip(intensity, 0.2, 1.5))

        args = self.ArgumentConfig(
            source=str(source_path),
            driving=str(template_for_run),
            output_dir=str(run_dir),
            flag_force_cpu=not self.torch.cuda.is_available(),
            flag_use_half_precision=False,
            flag_normalize_lip=False,
            flag_relative_motion=True,
            flag_do_crop=True,
            flag_pasteback=True,
            flag_stitching=True,
            driving_option="expression-friendly",
            driving_multiplier=driving_multiplier,
            animation_region="exp",
        )

        with self.apply_lock:
            self._disable_rich_progress_for_windows_console()
            output_path_str, _concat_path = self.LivePortraitPipeline.execute(self, args)

        output_path = Path(output_path_str)
        if not output_path.exists():
            raise RuntimeError(f"LivePortrait did not create output: {output_path}")

        if output_path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            frame_bgr = cv2.imread(str(output_path), cv2.IMREAD_COLOR)
            if frame_bgr is None:
                raise RuntimeError(f"Could not read LivePortrait image output: {output_path}")
            frame_index = 0
            frame_count = 1
            selected_score = 0.0
            top_scores: list[dict[str, Any]] = []
            candidate_paths = [str(output_path)]
            top_paths: list[str] = []
            selection_strategy = "single_image_output"
        else:
            (
                frame_bgr,
                frame_index,
                frame_count,
                selected_score,
                top_scores,
                selection_strategy,
                candidate_paths,
                top_paths,
            ) = self._representative_frame_from_video(
                output_path,
                scoring_key,
                template_name,
                candidate_dir,
                candidate_frame_override,
            )

        selected_item = next(
            (
                item
                for item in top_scores
                if int(item.get("frame_index", -1)) == int(frame_index)
            ),
            None,
        )
        selected_scores = (
            selected_item.get("scores", {})
            if isinstance(selected_item, dict)
            else {}
        )

        return frame_bgr, {
            "mode": "driving_template",
            "preset": scoring_preset or template_name,
            "used_driving_template": template_name,
            "driving_template_path": template_path_str,
            "output_path": str(output_path),
            "frame_index": frame_index,
            "selected_frame": frame_index,
            "selected_frame_index": frame_index,
            "frame_count": frame_count,
            "top_frame_indices": [
                int(item["frame_index"])
                for item in top_scores[:5]
            ],
            "selected_expression_score": selected_score,
            "top_expression_scores": top_scores,
            "scores": selected_scores,
            "selected_frame_rejected": bool(
                selected_item.get("rejected", False)
            ) if isinstance(selected_item, dict) else False,
            "candidate_dir": str(candidate_dir),
            "candidate_frame_paths": candidate_paths,
            "top_frame_paths": top_paths,
            "driving_multiplier": args.driving_multiplier,
            "representative_frame_strategy": selection_strategy,
            "scoring_method": selection_strategy,
            "selection_note": (
                "Preset-aware MediaPipe landmark scoring; grayscale difference "
                "is used only if landmark scoring fails."
            ),
        }


def _get_runtime() -> _LivePortraitExpressionRuntime:
    global _RUNTIME, _RUNTIME_ERROR

    with _RUNTIME_LOCK:
        if _RUNTIME is not None:
            return _RUNTIME

        status = is_liveportrait_runtime_available()
        if not status["runtime_available"]:
            _RUNTIME_ERROR = (
                "LivePortrait runtime is not available: "
                f"missing_modules={status['missing_modules']}; "
                f"missing_weight_files={status['missing_weight_files']}"
            )
            raise RuntimeError(_RUNTIME_ERROR)

        try:
            _RUNTIME = _LivePortraitExpressionRuntime()
            _RUNTIME_ERROR = None
            return _RUNTIME
        except Exception as exc:
            _RUNTIME_ERROR = str(exc)
            raise

def is_liveportrait_loaded() -> bool:
    return _RUNTIME is not None


def preload_liveportrait() -> dict[str, Any]:
    """
    Load LivePortrait runtime once at FastAPI startup.

    This loads model weights/wrappers into memory.
    It does NOT run driving-template video generation.
    """

    started = time.perf_counter()

    status = is_liveportrait_runtime_available()

    if _RUNTIME is not None:
        return {
            "ok": True,
            "loaded": True,
            "cache_hit": True,
            "provider": "liveportrait",
            "seconds": round(time.perf_counter() - started, 3),
            "status": {
                **status,
                "loaded": True,
            },
        }

    if not status.get("runtime_available", False):
        return {
            "ok": False,
            "loaded": False,
            "cache_hit": False,
            "provider": "liveportrait",
            "reason": "runtime_unavailable",
            "seconds": round(time.perf_counter() - started, 3),
            "status": {
                **status,
                "loaded": False,
            },
        }

    try:
        runtime = _get_runtime()

        return {
            "ok": True,
            "loaded": runtime is not None,
            "cache_hit": False,
            "provider": "liveportrait",
            "seconds": round(time.perf_counter() - started, 3),
            "status": {
                **is_liveportrait_runtime_available(),
                "loaded": runtime is not None,
            },
        }

    except Exception as exc:
        return {
            "ok": False,
            "loaded": False,
            "cache_hit": False,
            "provider": "liveportrait",
            "reason": "preload_failed",
            "error": repr(exc),
            "seconds": round(time.perf_counter() - started, 3),
            "status": {
                **is_liveportrait_runtime_available(),
                "loaded": False,
            },
        }


def apply_liveportrait_expression(
    image_bgr: np.ndarray,
    preset: str,
    intensity: float,
    params: dict | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    params = params or {}
    preset = str(preset or "smile").lower()
    internal_preset = str(params.get("internal_preset") or preset).lower()
    if internal_preset in {"laugh", "soft_laugh"}:
        intensity = float(np.clip(intensity, 0.0, 1.0))
    elif internal_preset in {"smile", "natural_smile", "broad_smile"}:
        intensity = float(np.clip(intensity, 0.0, 0.62))
    else:
        intensity = float(np.clip(intensity, 0.0, 0.85))
    status = is_liveportrait_runtime_available()
    use_driving_template = bool(params.get("use_driving_template", False))
    driving_template = str(params.get("driving_template", "") or "").lower()
    candidate_frame_override = params.get("candidate_frame_override", "auto")
    meta: dict[str, Any] = {
        "provider": "liveportrait",
        "mode": "direct_smile",
        "runtime_available": status["runtime_available"],
        "inference_bridge_implemented": status["inference_bridge_implemented"],
        "preset": internal_preset,
        "effective_intensity": intensity,
        "requires_driving_template": (
            use_driving_template or preset not in SUPPORTED_DIRECT_PRESETS
        ),
        "used_driving_template": None,
        "frame_index": None,
        "fallback_used": False,
        "error": None,
        "status": status,
    }

    if use_driving_template:
        if driving_template not in SUPPORTED_TEMPLATE_NAMES:
            meta["provider"] = "not_available"
            meta["mode"] = "driving_template"
            meta["used_driving_template"] = driving_template or None
            meta["error"] = (
                "Unsupported LivePortrait driving template. "
                f"Supported templates: {sorted(SUPPORTED_TEMPLATE_NAMES)}"
            )
            return image_bgr.copy(), meta

        if not status["runtime_available"]:
            meta["provider"] = "not_available"
            meta["mode"] = "driving_template"
            meta["used_driving_template"] = driving_template
            meta["error"] = (
                "LivePortrait runtime unavailable. "
                f"Missing modules: {status['missing_modules']}; "
                f"missing weight files: {status['missing_weight_files']}"
            )
            return image_bgr.copy(), meta

        try:
            runtime = _get_runtime()
            scoring_preset = (
                internal_preset
                if internal_preset in TEMPLATE_SCORING_PRESETS
                else TEMPLATE_TO_SCORING_PRESET.get(
                    driving_template,
                    internal_preset,
                )
            )
            output_bgr, details = runtime.apply_driving_template(
                image_bgr,
                driving_template,
                intensity,
                scoring_preset,
                candidate_frame_override,
            )
            meta.update(details)
            meta["runtime_available"] = True
            meta["inference_bridge_implemented"] = True
            meta["fallback_used"] = False
            meta["error"] = None
            return output_bgr, meta
        except Exception as exc:
            meta["provider"] = "not_available"
            meta["mode"] = "driving_template"
            meta["used_driving_template"] = driving_template
            meta["runtime_available"] = False
            meta["inference_bridge_implemented"] = False
            meta["fallback_used"] = False
            meta["error"] = str(exc)
            return image_bgr.copy(), meta

    if preset not in SUPPORTED_DIRECT_PRESETS:
        template_path = params.get("driving_template")
        meta["used_driving_template"] = template_path
        meta["error"] = (
            "Preset requires a LivePortrait driving video or .pkl motion template; "
            "direct single-image retargeting is currently implemented only for smile "
            "and eyebrow_raise."
        )
        return image_bgr.copy(), meta

    if not status["runtime_available"]:
        meta["error"] = (
            "LivePortrait runtime unavailable. "
            f"Missing modules: {status['missing_modules']}; "
            f"missing weight files: {status['missing_weight_files']}"
        )
        return image_bgr.copy(), meta

    try:
        runtime = _get_runtime()
        if preset == "eyebrow_raise":
            output_bgr, details = runtime.apply_eyebrow_raise(
                image_bgr,
                intensity,
                params,
            )
        else:
            output_bgr, details = runtime.apply_smile(
                image_bgr,
                intensity,
                params,
            )
        meta.update(details)
        meta["runtime_available"] = True
        meta["inference_bridge_implemented"] = True
        meta["error"] = None
        return output_bgr, meta
    except Exception as exc:
        meta["runtime_available"] = False
        meta["inference_bridge_implemented"] = False
        meta["error"] = str(exc)
        return image_bgr.copy(), meta
