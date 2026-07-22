from __future__ import annotations

import queue
import threading
import time

import cv2
import numpy as np

from backend.effects.hair_color_v2 import HairSegmentor


class BiSeNetWorker:
    """
    Async BiSeNet hair-mask worker for realtime sessions.

    The render path only polls this worker; it never waits for segmentation.
    """

    REFRESH_INTERVAL_FRAMES = 60

    def __init__(self, bisenet_model_path: str | None = None) -> None:
        self._segmentor = HairSegmentor(bisenet_model_path)
        self._request_queue: queue.Queue = queue.Queue(maxsize=1)
        self._result_queue: queue.Queue = queue.Queue(maxsize=1)
        self._running = True
        self.last_error: str | None = None
        self.request_count = 0
        self.completed_count = 0
        self.last_inference_ms: float | None = None

        self._thread = threading.Thread(
            target=self._worker_loop,
            daemon=True,
            name="bisenet-worker",
        )
        self._thread.start()

    def _worker_loop(self) -> None:
        while self._running:
            try:
                frame_bgr, params_hash = self._request_queue.get(timeout=0.05)
                started_at = time.perf_counter()
                mask = self._segmentor.get_hair_mask(frame_bgr)
                if mask is None:
                    self.last_error = self._segmentor.last_error or "hair_mask_unavailable"
                    continue

                mask_float = np.clip(mask.astype(np.float32) / 255.0, 0.0, 1.0)
                try:
                    guide = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
                    if hasattr(cv2, "ximgproc") and hasattr(cv2.ximgproc, "guidedFilter"):
                        mask_float = cv2.ximgproc.guidedFilter(
                            guide,
                            mask_float.astype(np.float32),
                            radius=8,
                            eps=1e-3,
                        )
                        mask_float = np.clip(mask_float, 0.0, 1.0)
                except Exception:
                    pass

                result = (mask_float.astype(np.float32), frame_bgr.copy(), int(params_hash))
                try:
                    self._result_queue.get_nowait()
                except queue.Empty:
                    pass
                self._result_queue.put_nowait(result)
                self.completed_count += 1
                self.last_inference_ms = (time.perf_counter() - started_at) * 1000.0
                self.last_error = None
            except queue.Empty:
                continue
            except Exception as exc:
                self.last_error = str(exc)
                continue

    def request(self, frame_bgr: np.ndarray, params_hash: int) -> None:
        try:
            self._request_queue.get_nowait()
        except queue.Empty:
            pass
        try:
            self._request_queue.put_nowait((frame_bgr.copy(), int(params_hash)))
            self.request_count += 1
        except queue.Full:
            pass
        except Exception:
            pass

    def poll(self) -> tuple[np.ndarray, np.ndarray, int] | None:
        try:
            return self._result_queue.get_nowait()
        except queue.Empty:
            return None
        except Exception:
            return None

    def stop(self) -> None:
        self._running = False
        if self._thread.is_alive():
            self._thread.join(timeout=0.5)

    @property
    def is_alive(self) -> bool:
        return self._thread.is_alive()
