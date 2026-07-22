from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RealtimeConfig:
    """
    Realtime processing config.

    Full photo pipeline is heavy because it can run:
    - face parsing
    - 3D context
    - masks
    - effects

    In realtime, we should not run all heavy steps on every frame.
    """

    enabled: bool = True

    # Legacy compatibility. Async realtime tracks every frame and refreshes
    # masks in a background worker instead of relying on full-frame cadence.
    full_analysis_every_n_frames: int = 999

    # Resize camera frames before processing for speed.
    processing_width: int = 360

    # Smooth anchors / pose to reduce jitter.
    smoothing_alpha: float = 0.55

    # Predict anchors slightly ahead to compensate capture/network/inference
    # latency. This keeps masks/accessories on the current video frame instead
    # of visibly trailing one frame behind.
    prediction_ms: float = 65.0

    # Use a constant-velocity stabilizer instead of plain EMA for rigid overlay
    # anchors. EMA reduces jitter but causes visible lag during head movement.
    prediction_smoothing_enabled: bool = True

    # If true, return last processed result on skipped frames.
    reuse_last_result: bool = True

    # Hard safety: if processing fails, return original frame.
    fail_safe_original: bool = True

    # Run the heavy face-parsing/mask refresh path off the render thread.
    mask_worker_enabled: bool = True
