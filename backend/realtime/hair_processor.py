from __future__ import annotations

import hashlib
import time
from typing import Any

import cv2
import numpy as np

from backend.effects.hair_color_v2 import apply_hair_color_hsl, parse_target_color_rgb
from backend.realtime.bisenet_worker import BiSeNetWorker
from backend.realtime.klt_propagator import KLTMaskPropagator


def _target_bgr_from_param(color: Any) -> tuple[int, int, int]:
    r, g, b = parse_target_color_rgb(color, fallback=(123, 63, 228))
    return b, g, r


def apply_hair_color_hls_fast(
    frame_bgr: np.ndarray,
    mask_float: np.ndarray,
    target_color_bgr: tuple[int, int, int],
    intensity: float = 0.85,
) -> np.ndarray:
    """Use the same texture-preserving transfer as the photo pipeline."""
    intensity = float(np.clip(intensity, 0.0, 1.0))
    if intensity <= 0.001:
        return frame_bgr

    h, w = frame_bgr.shape[:2]
    if mask_float.shape[:2] != (h, w):
        mask_float = cv2.resize(mask_float, (w, h), interpolation=cv2.INTER_LINEAR)

    mask_u8 = np.clip(mask_float.astype(np.float32), 0.0, 1.0)
    mask_u8 = (mask_u8 * 255.0).astype(np.uint8)
    mask_u8 = cv2.bilateralFilter(mask_u8, d=7, sigmaColor=50, sigmaSpace=7)
    target_rgb = (
        int(target_color_bgr[2]),
        int(target_color_bgr[1]),
        int(target_color_bgr[0]),
    )
    return apply_hair_color_hsl(
        frame_bgr,
        mask_u8,
        target_rgb,
        intensity,
    )


class OptimalRealtimeHairProcessor:
    """
    Async BiSeNet + KLT mask propagation + HLS transfer for realtime hair color.

    Enhancements:
    - Temporal mask smoothing: blends the newly received BiSeNet mask with the
      propagated mask to avoid sudden pop-in when the background worker delivers
      a fresh segmentation.
    - Per-processor EMA mask buffer so jitter between BiSeNet refreshes is
      absorbed without extra latency.
    """

    # EMA weight for temporal mask blending (higher = slower transition, less flicker)
    TEMPORAL_ALPHA = 0.40

    def __init__(self, bisenet_model_path: str | None = None) -> None:
        self._worker = BiSeNetWorker(bisenet_model_path)
        self._propagator = KLTMaskPropagator()
        self._frame_count = 0
        self._last_params_hash: int | None = None
        self._last_request_frame = -999
        self._ema_mask: np.ndarray | None = None   # temporal smoothing buffer
        self.last_debug: dict[str, Any] = {}

    def _hash_params(self, params: dict | None) -> int:
        hair = (params or {}).get("hair_color", {})
        if not isinstance(hair, dict):
            hair = {}
        payload = "|".join(
            [
                str(bool(hair.get("enabled", False))),
                str(hair.get("color", "")),
                f"{float(hair.get('intensity', 0.85)):.2f}",
            ]
        )
        return int(hashlib.blake2s(payload.encode("utf-8"), digest_size=8).hexdigest(), 16)

    def _smooth_mask(self, new_mask: np.ndarray) -> np.ndarray:
        """Apply EMA temporal smoothing to the mask to remove flicker."""
        if self._ema_mask is None or self._ema_mask.shape != new_mask.shape:
            self._ema_mask = new_mask.copy()
            return new_mask

        self._ema_mask = (
            self._ema_mask * self.TEMPORAL_ALPHA
            + new_mask * (1.0 - self.TEMPORAL_ALPHA)
        ).astype(np.float32)
        return self._ema_mask

    def process(
        self,
        frame_bgr: np.ndarray,
        landmarks: np.ndarray,
        params: dict,
    ) -> np.ndarray:
        self._frame_count += 1
        hair = params.get("hair_color", {}) if isinstance(params, dict) else {}
        if not isinstance(hair, dict) or not hair.get("enabled", False):
            return frame_bgr

        params_hash = self._hash_params(params)
        frame_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        bisenet_polled = False
        worker_result = self._worker.poll()
        if worker_result is not None:
            mask_float, ref_frame, result_hash = worker_result
            ref_gray = cv2.cvtColor(ref_frame, cv2.COLOR_BGR2GRAY)
            self._propagator.init_from_bisenet(mask_float, ref_gray, landmarks, result_hash)
            bisenet_polled = True

        params_changed = params_hash != self._last_params_hash
        refresh_due = (self._frame_count - self._last_request_frame) >= BiSeNetWorker.REFRESH_INTERVAL_FRAMES
        low_confidence = self._propagator.last_confidence < 0.22
        no_cache = self._propagator.get_params_hash() is None

        bisenet_requested = False
        if params_changed or refresh_due or low_confidence or no_cache:
            self._worker.request(frame_bgr, params_hash)
            self._last_request_frame = self._frame_count
            self._last_params_hash = params_hash
            bisenet_requested = True

        klt_started_at = time.perf_counter()
        mask_float = self._propagator.propagate(frame_gray, landmarks)
        klt_ms = (time.perf_counter() - klt_started_at) * 1000.0

        if mask_float is None:
            self.last_debug = {
                "mode": "waiting_for_bisenet",
                "provider_error": self._worker.last_error,
                "params_hash": params_hash,
                "bisenet_thread_alive": self._worker.is_alive,
                "bisenet_requested": bisenet_requested,
                "bisenet_request_count": self._worker.request_count,
                "bisenet_completed_count": self._worker.completed_count,
                "bisenet_refresh_interval_frames": BiSeNetWorker.REFRESH_INTERVAL_FRAMES,
                "klt_ms": round(klt_ms, 2),
            }
            return frame_bgr

        # Temporal smoothing: blend EMA mask with the freshly propagated mask
        mask_float = self._smooth_mask(mask_float)

        hls_started_at = time.perf_counter()
        result = apply_hair_color_hls_fast(
            frame_bgr,
            mask_float,
            _target_bgr_from_param(hair.get("color", "#7B3FE4")),
            float(hair.get("intensity", 0.85)),
        )
        hls_ms = (time.perf_counter() - hls_started_at) * 1000.0
        self.last_debug = {
            "mode": "klt_hls",
            "confidence": round(float(self._propagator.last_confidence), 3),
            "mask_pixels": int(np.count_nonzero(mask_float > 0.05)),
            "params_hash": params_hash,
            "worker_error": self._worker.last_error,
            "bisenet_thread_alive": self._worker.is_alive,
            "bisenet_polled": bisenet_polled,
            "bisenet_requested": bisenet_requested,
            "bisenet_request_count": self._worker.request_count,
            "bisenet_completed_count": self._worker.completed_count,
            "bisenet_last_inference_ms": (
                round(float(self._worker.last_inference_ms), 2)
                if self._worker.last_inference_ms is not None
                else None
            ),
            "bisenet_refresh_interval_frames": BiSeNetWorker.REFRESH_INTERVAL_FRAMES,
            "klt_ms": round(klt_ms, 2),
            "hls_ms": round(hls_ms, 2),
        }
        return result

    def reset(self) -> None:
        self._propagator.reset()
        self._frame_count = 0
        self._last_params_hash = None
        self._last_request_frame = -999
        self._ema_mask = None
        self.last_debug = {}

    def stop(self) -> None:
        self._worker.stop()
