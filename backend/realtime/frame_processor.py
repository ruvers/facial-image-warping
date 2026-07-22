from __future__ import annotations

import copy
import queue
import threading
import time

import cv2
import numpy as np

from backend.accessories.earring_motion import mark_earring_motion_missing
from backend.assets_manager import resolve_asset_by_id
from backend.effect_catalog import normalize_params
from backend.effect_engine import run_photo_engine_with_context
from backend.face_analysis import build_anchors, detect_face_landmarks, estimate_head_pose
from backend.photo_pipeline import apply_photo_pipeline
from backend.realtime.config import RealtimeConfig
from backend.realtime.hair_processor import OptimalRealtimeHairProcessor
from backend.realtime.smoothing import RealtimeAnchorStabilizer, smooth_anchors, smooth_pose
from backend.register_default_effects import register_default_effects


def _category_to_accessory_type(category: str) -> str:
    if category == "hair_clips":
        return "hair_clip"
    if category == "hats":
        return "hat"
    if category == "necklaces":
        return "necklace"
    return category


def _resolve_realtime_accessory_items(params: dict | None) -> dict:
    params = dict(params or {})
    accessories = params.get("accessories")
    if not isinstance(accessories, dict) or not isinstance(accessories.get("items"), list):
        return params

    resolved_items = []
    for item in accessories.get("items", []):
        if not isinstance(item, dict):
            continue

        if item.get("asset_path"):
            resolved_items.append(item)
            continue

        category = str(item.get("category") or item.get("type") or "").strip()
        asset_id = str(item.get("asset_id") or "").strip()
        if not category or not asset_id:
            resolved_items.append(item)
            continue

        try:
            asset = resolve_asset_by_id(category, asset_id)
            metadata = {}
            if isinstance(asset.get("default_metadata"), dict):
                metadata.update(asset.get("default_metadata"))
            if isinstance(item.get("metadata"), dict):
                metadata.update(item.get("metadata"))

            default_scale = float(asset.get("default_scale", 1.0))
            raw_scale = item.get("scale")
            if raw_scale is None:
                scale = default_scale
            else:
                requested_scale = float(raw_scale)
                if category in {"earrings", "hair_clips"} and requested_scale >= 0.7:
                    scale = default_scale * requested_scale
                else:
                    scale = requested_scale

            resolved_items.append(
                {
                    **item,
                    "type": str(asset.get("type") or _category_to_accessory_type(category)),
                    "category": str(asset.get("category") or category),
                    "asset_path": str(asset.get("path") or ""),
                    "render_mode": str(item.get("render_mode") or "overlay_2d"),
                    "metadata": metadata,
                    "scale": scale,
                    "offset_x": float(item.get("offset_x", asset.get("default_offset_x", 0.0))),
                    "offset_y": float(item.get("offset_y", asset.get("default_offset_y", 0.0))),
                    "offset_y_ratio": float(item.get("offset_y_ratio", asset.get("default_offset_y_ratio", 0.0))),
                    "alpha": float(item.get("alpha", item.get("opacity", asset.get("default_alpha", 1.0)))),
                }
            )
        except Exception:
            resolved_items.append({**item, "fallback_reason": "realtime_asset_resolve_failed"})

    params["accessories"] = {
        **accessories,
        "enabled": bool(resolved_items),
        "items": resolved_items,
    }
    return params


def _effect_enabled(value) -> bool:
    if not isinstance(value, dict):
        return False
    if isinstance(value.get("items"), list) and value.get("items"):
        return True
    return bool(value.get("enabled", False))


def _has_enabled_realtime_effect(params: dict | None) -> bool:
    if not isinstance(params, dict):
        return False
    if any(_effect_enabled(value) for value in params.values()):
        return True

    makeup = params.get("makeup")
    if isinstance(makeup, dict):
        return any(_effect_enabled(value) for value in makeup.values())

    return False


def _hair_color_enabled(params: dict | None) -> bool:
    if not isinstance(params, dict):
        return False
    hair = params.get("hair_color")
    return isinstance(hair, dict) and bool(hair.get("enabled", False))


def _without_hair_color(params: dict | None) -> dict:
    copied = copy.deepcopy(params or {})
    hair = copied.get("hair_color")
    if isinstance(hair, dict):
        copied["hair_color"] = {**hair, "enabled": False}
    return copied


def _stable_affine_from_landmarks(
    previous_landmarks: np.ndarray | None,
    current_landmarks: np.ndarray | None,
) -> np.ndarray | None:
    stable_indices = [133, 362, 152]
    if previous_landmarks is None or current_landmarks is None:
        return None
    if len(previous_landmarks) <= max(stable_indices) or len(current_landmarks) <= max(stable_indices):
        return None

    pts_src = np.float32([previous_landmarks[idx] for idx in stable_indices])
    pts_dst = np.float32([current_landmarks[idx] for idx in stable_indices])
    return cv2.getAffineTransform(pts_src, pts_dst)


class RealtimeFrameProcessor:
    """
    Realtime processor with a Snap-style async mask pipeline.

    Render thread:
    - tracks landmarks on the raw current frame
    - warps cached masks to the current landmarks
    - renders immediately

    Worker thread:
    - refreshes full face parsing / mask context in the background
    - replaces cache when a newer result is ready
    """

    def __init__(
        self,
        config: RealtimeConfig | None = None,
    ):
        self.config = config or RealtimeConfig()

        self.frame_index = 0
        self.last_ctx: dict | None = None
        self.last_result_bgr: np.ndarray | None = None
        self.last_result_small: np.ndarray | None = None
        self.last_error: str | None = None
        self.last_frame_time: float | None = None
        self.stabilizer = RealtimeAnchorStabilizer()
        self.accessory_motion_state: dict = {"sides": {}}
        self._hair_processor = OptimalRealtimeHairProcessor()

        self._mask_cache: dict | None = None
        self._mask_lock = threading.Lock()
        self._mask_worker_thread: threading.Thread | None = None
        self._mask_request_queue: queue.Queue = queue.Queue(maxsize=1)
        self._mask_result_queue: queue.Queue = queue.Queue(maxsize=1)
        self._mask_worker_running = False
        if bool(self.config.mask_worker_enabled):
            self._start_mask_worker()

    def _start_mask_worker(self) -> None:
        if self._mask_worker_thread is not None and self._mask_worker_thread.is_alive():
            return

        self._mask_worker_running = True
        self._mask_worker_thread = threading.Thread(
            target=self._mask_worker_loop,
            daemon=True,
            name="mask-worker",
        )
        self._mask_worker_thread.start()

    def _mask_worker_loop(self) -> None:
        while self._mask_worker_running:
            try:
                frame, params = self._mask_request_queue.get(timeout=0.1)
                params_for_worker = self._params_for_pipeline(
                    _without_hair_color(params) if _hair_color_enabled(params) else params
                )
                _, ctx = apply_photo_pipeline(frame, params_for_worker)
                if isinstance(ctx, dict):
                    ctx["__analysis_mode"] = "async_mask"
                    try:
                        self._mask_result_queue.get_nowait()
                    except queue.Empty:
                        pass
                    self._mask_result_queue.put_nowait(ctx)
            except queue.Empty:
                continue
            except Exception as exc:
                self.last_error = f"mask_worker: {exc}"
                continue

    def _request_mask_refresh(self, frame: np.ndarray, params: dict | None) -> None:
        if not bool(self.config.mask_worker_enabled) or not self._mask_worker_running:
            return
        try:
            try:
                self._mask_request_queue.get_nowait()
            except queue.Empty:
                pass
            self._mask_request_queue.put_nowait((frame.copy(), copy.deepcopy(params or {})))
        except Exception:
            pass

    def _poll_mask_result(self) -> dict | None:
        try:
            ctx = self._mask_result_queue.get_nowait()
        except queue.Empty:
            return None
        except Exception:
            return None

        if isinstance(ctx, dict):
            with self._mask_lock:
                self._mask_cache = ctx
            return ctx
        return None

    def _clear_queue(self, q: queue.Queue) -> None:
        while True:
            try:
                q.get_nowait()
            except queue.Empty:
                break
            except Exception:
                break

    def _resize_for_processing(
        self,
        frame_bgr: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        h, w = frame_bgr.shape[:2]
        target_w = int(self.config.processing_width)
        if target_w <= 0 or w <= target_w:
            return frame_bgr, 1.0

        scale = target_w / float(w)
        target_h = int(h * scale)
        resized = cv2.resize(frame_bgr, (target_w, target_h), interpolation=cv2.INTER_AREA)
        return resized, scale

    def _resize_back(
        self,
        processed_bgr: np.ndarray,
        original_shape,
    ) -> np.ndarray:
        oh, ow = original_shape[:2]
        if processed_bgr.shape[:2] == (oh, ow):
            return processed_bgr
        return cv2.resize(processed_bgr, (ow, oh), interpolation=cv2.INTER_LINEAR)

    def should_run_full_pipeline(self) -> bool:
        n = max(1, int(self.config.full_analysis_every_n_frames))
        return self.frame_index % n == 0 or self.last_result_bgr is None

    def _frame_dt(self) -> float:
        now = time.perf_counter()
        if self.last_frame_time is None:
            dt = 1.0 / 28.0
        else:
            dt = now - self.last_frame_time
        self.last_frame_time = now
        return float(np.clip(dt, 1.0 / 60.0, 0.12))

    def _lead_time(self) -> float:
        return float(np.clip(float(self.config.prediction_ms) / 1000.0, 0.0, 0.12))

    def _stabilize_context(self, ctx: dict, dt: float) -> dict:
        if not isinstance(ctx, dict):
            return ctx

        if bool(self.config.prediction_smoothing_enabled):
            lead_time = self._lead_time()
            if isinstance(ctx.get("anchors"), dict):
                ctx["raw_anchors"] = ctx.get("anchors")
                ctx["anchors"] = self.stabilizer.stabilize_anchors(
                    ctx.get("anchors", {}),
                    dt=dt,
                    lead_time=lead_time,
                )
            if isinstance(ctx.get("pose"), dict):
                ctx["raw_pose"] = ctx.get("pose")
                ctx["pose"] = self.stabilizer.stabilize_pose(
                    ctx.get("pose", {}),
                    dt=dt,
                    lead_time=lead_time,
                )
            ctx.setdefault("realtime_debug", {})["stabilization"] = {
                "mode": "constant_velocity_prediction",
                "dt_ms": round(dt * 1000.0, 2),
                "lead_ms": round(lead_time * 1000.0, 2),
            }
            return ctx

        if self.last_ctx is not None:
            if "anchors" in ctx:
                ctx["anchors"] = smooth_anchors(
                    self.last_ctx.get("anchors"),
                    ctx.get("anchors", {}),
                    alpha=self.config.smoothing_alpha,
                )
            if "pose" in ctx:
                ctx["pose"] = smooth_pose(
                    self.last_ctx.get("pose"),
                    ctx.get("pose", {}),
                    alpha=self.config.smoothing_alpha,
                )
        return ctx

    def _params_for_pipeline(self, params: dict | None) -> dict:
        return {
            **(copy.deepcopy(params or {})),
            "__accessory_motion_state": self.accessory_motion_state,
            "__realtime": True,
            "__skip_effect_diff": True,
        }

    def _prepare_realtime_params(self, params: dict | None) -> dict:
        normalized_params = normalize_params(params)
        normalized_params["__accessory_motion_state"] = self.accessory_motion_state
        normalized_params["__realtime"] = True
        normalized_params["__skip_effect_diff"] = True

        accessories = normalized_params.get("accessories")
        if isinstance(accessories, dict):
            items = accessories.get("items")
            if isinstance(items, list):
                has_3d_item = any(
                    str(item.get("render_mode", "")).lower()
                    in {"physics_3d", "parametric_3d", "hybrid_3d_refine"}
                    for item in items
                    if isinstance(item, dict)
                )
                if has_3d_item:
                    normalized_params["accessory_3d"] = {
                        "enabled": True,
                        "items": items,
                    }

        return normalized_params

    def _refine_warped_hair_mask(
        self,
        hair_mask: np.ndarray | None,
        frame_bgr: np.ndarray,
        landmarks_2d: np.ndarray,
    ) -> np.ndarray | None:
        if hair_mask is None:
            return None
        if not isinstance(landmarks_2d, np.ndarray) or len(landmarks_2d) <= 454:
            return hair_mask

        h, w = frame_bgr.shape[:2]
        if hair_mask.shape[:2] != (h, w):
            hair_mask = cv2.resize(hair_mask, (w, h), interpolation=cv2.INTER_LINEAR)

        try:
            left = landmarks_2d[234].astype(np.float32)
            right = landmarks_2d[454].astype(np.float32)
            top = landmarks_2d[10].astype(np.float32)
            face_width = max(1.0, float(np.linalg.norm(right - left)))
            cx = float((left[0] + right[0]) * 0.5)
            cy = float(top[1] - face_width * 0.15)

            yy, xx = np.indices((h, w), dtype=np.float32)
            hair_roi = (((xx - cx) / (face_width * 0.58)) ** 2 + ((yy - cy) / (face_width * 0.46)) ** 2) <= 1.0
            face_roi = (
                (((xx - cx) / (face_width * 0.44)) ** 2 + ((yy - (top[1] + face_width * 0.23)) / (face_width * 0.30)) ** 2)
                <= 1.0
            ) & (yy > top[1] - face_width * 0.10)

            hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            sat = hsv[:, :, 1]
            val = hsv[:, :, 2]
            lap = np.abs(cv2.Laplacian(gray, cv2.CV_32F, ksize=3))
            pixel_hair = (((val < 96) & (sat > 10)) | ((val < 132) & (sat > 48)) | ((val < 118) & (lap > 5))).astype(np.uint8)

            refined = ((hair_mask > 20) & hair_roi & (~face_roi) & (pixel_hair > 0)).astype(np.uint8) * 255
            refined = cv2.morphologyEx(refined, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
            refined = cv2.morphologyEx(refined, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats((refined > 0).astype(np.uint8), 8)
            if num_labels > 1:
                min_area = max(25, int(face_width * 0.20))
                keep = np.zeros_like(refined)
                for label in range(1, num_labels):
                    if stats[label, cv2.CC_STAT_AREA] >= min_area:
                        keep[labels == label] = 255
                refined = keep

            return cv2.GaussianBlur(refined, (5, 5), 0)
        except Exception:
            return hair_mask

    def _warp_cached_context(
        self,
        cached_ctx: dict,
        landmarks_2d: np.ndarray,
        landmarks_3d: np.ndarray,
        small_frame: np.ndarray,
        frame_dt: float,
    ) -> dict:
        sh, sw = small_frame.shape[:2]
        previous_landmarks = cached_ctx.get("landmarks_2d")
        affine = _stable_affine_from_landmarks(previous_landmarks, landmarks_2d)

        if affine is not None:
            warped_masks = {}
            for name, mask in (cached_ctx.get("masks") or {}).items():
                if mask is None:
                    continue
                warped = cv2.warpAffine(
                    mask,
                    affine,
                    (sw, sh),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_REFLECT,
                )
                if name == "hair":
                    warped = self._refine_warped_hair_mask(warped, small_frame, landmarks_2d)
                warped_masks[name] = warped

            last_depth = cached_ctx.get("depth_map")
            if last_depth is not None:
                warped_depth = cv2.warpAffine(
                    last_depth,
                    affine,
                    (sw, sh),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_REFLECT,
                )
            else:
                warped_depth = np.zeros((sh, sw), dtype=np.float32)
        else:
            warped_masks = copy.deepcopy(cached_ctx.get("masks") or {})
            warped_depth = cached_ctx.get("depth_map")
            if warped_depth is None:
                warped_depth = np.zeros((sh, sw), dtype=np.float32)

        ctx_curr = dict(cached_ctx)
        ctx_curr["effect_debug_meta"] = {}
        ctx_curr["landmarks_2d"] = landmarks_2d
        ctx_curr["landmarks"] = landmarks_2d
        ctx_curr["landmarks_3d"] = landmarks_3d
        ctx_curr["masks"] = warped_masks
        ctx_curr["depth_map"] = warped_depth
        ctx_curr["pose"] = estimate_head_pose(landmarks_2d, small_frame.shape)
        ctx_curr["anchors"] = build_anchors(
            landmarks_2d=landmarks_2d,
            landmarks_3d=landmarks_3d,
            masks=warped_masks,
            image_shape=small_frame.shape,
        )
        ctx_curr["accessory_motion_state"] = self.accessory_motion_state
        ctx_curr["__analysis_mode"] = "async_tracked"
        return self._stabilize_context(ctx_curr, frame_dt)

    def process_frame(
        self,
        frame_bgr: np.ndarray,
        params: dict | None = None,
    ) -> dict:
        frame_started_at = time.perf_counter()
        if frame_bgr is None:
            return {
                "success": False,
                "frame_index": self.frame_index,
                "used_full_pipeline": False,
                "result_bgr": None,
                "ctx": self.last_ctx,
                "error": "frame_bgr is None",
            }

        original_shape = frame_bgr.shape
        used_full = False
        params = _resolve_realtime_accessory_items(params)
        frame_dt = self._frame_dt()

        if not _has_enabled_realtime_effect(params):
            self.last_result_bgr = frame_bgr.copy()
            self.last_result_small = None
            self.last_error = None
            self.frame_index += 1
            return {
                "success": True,
                "frame_index": self.frame_index - 1,
                "used_full_pipeline": False,
                "result_bgr": self.last_result_bgr,
                "ctx": self.last_ctx,
                "error": None,
            }

        try:
            self._poll_mask_result()
            small_frame, _scale = self._resize_for_processing(frame_bgr)
            landmarks_data = detect_face_landmarks(small_frame)

            if landmarks_data is None:
                mark_earring_motion_missing(self.accessory_motion_state)
                result_bgr = self.last_result_bgr.copy() if self.last_result_bgr is not None else frame_bgr.copy()
                self.last_error = "no_face"
                self.frame_index += 1
                return {
                    "success": True,
                    "frame_index": self.frame_index - 1,
                    "used_full_pipeline": False,
                    "result_bgr": result_bgr,
                    "ctx": self.last_ctx,
                    "error": "no_face",
                }

            landmarks_2d, landmarks_3d = landmarks_data
            render_frame = small_frame
            render_params = params
            if _hair_color_enabled(params):
                render_frame = self._hair_processor.process(
                    small_frame,
                    landmarks_2d.astype(np.float32),
                    params,
                )
                render_params = _without_hair_color(params)

            with self._mask_lock:
                cached_ctx = self._mask_cache

            if cached_ctx is None:
                result_small, ctx = apply_photo_pipeline(
                    render_frame,
                    self._params_for_pipeline(render_params),
                )
                if isinstance(ctx, dict):
                    ctx = self._stabilize_context(ctx, frame_dt)
                    ctx["__analysis_mode"] = "full"
                    ctx["accessory_motion_state"] = self.accessory_motion_state
                    with self._mask_lock:
                        self._mask_cache = ctx
                else:
                    ctx = None

                result_bgr = self._resize_back(result_small, original_shape)
                self.last_result_bgr = result_bgr
                self.last_result_small = result_small.copy()
                self.last_ctx = ctx
                self.last_error = None
                used_full = True
                self._request_mask_refresh(small_frame, params)
            else:
                ctx_curr = self._warp_cached_context(
                    cached_ctx=cached_ctx,
                    landmarks_2d=landmarks_2d,
                    landmarks_3d=landmarks_3d,
                    small_frame=render_frame,
                    frame_dt=frame_dt,
                )
                if _hair_color_enabled(params):
                    ctx_curr.setdefault("effect_debug_meta", {})["hair_color"] = {
                        "hair_realtime_debug": self._hair_processor.last_debug,
                    }

                register_default_effects()
                result_small, _ = run_photo_engine_with_context(
                    render_frame,
                    ctx_curr,
                    self._prepare_realtime_params(render_params),
                )
                result_bgr = self._resize_back(result_small, original_shape)
                self.last_result_bgr = result_bgr
                self.last_result_small = result_small.copy()
                self.last_ctx = ctx_curr
                self.last_error = None
                self._request_mask_refresh(small_frame, params)

            output = {
                "success": True,
                "frame_index": self.frame_index,
                "used_full_pipeline": used_full,
                "result_bgr": self.last_result_bgr,
                "ctx": self.last_ctx,
                "error": self.last_error,
            }
            total_ms = (time.perf_counter() - frame_started_at) * 1000.0
            if isinstance(self.last_ctx, dict):
                realtime_debug = self.last_ctx.setdefault("realtime_debug", {})
                realtime_debug["frame_processor_ms"] = round(total_ms, 2)
                realtime_debug["processing_width"] = int(self.config.processing_width)
                realtime_debug["hair_color"] = self._hair_processor.last_debug
            if self.frame_index % 30 == 0 and _hair_color_enabled(params):
                hair_debug = self._hair_processor.last_debug or {}
                print(
                    "[REALTIME_PERF] "
                    f"frame={self.frame_index} "
                    f"total_ms={total_ms:.2f} "
                    f"klt_ms={hair_debug.get('klt_ms')} "
                    f"hls_ms={hair_debug.get('hls_ms')} "
                    f"bisenet_req={hair_debug.get('bisenet_request_count')} "
                    f"bisenet_done={hair_debug.get('bisenet_completed_count')} "
                    f"refresh={hair_debug.get('bisenet_refresh_interval_frames')}"
                )

        except Exception as exc:
            self.last_error = str(exc)
            if self.config.fail_safe_original:
                result_bgr = frame_bgr.copy()
            elif self.last_result_bgr is not None:
                result_bgr = self.last_result_bgr.copy()
            else:
                result_bgr = frame_bgr.copy()

            output = {
                "success": False,
                "frame_index": self.frame_index,
                "used_full_pipeline": used_full,
                "result_bgr": result_bgr,
                "ctx": self.last_ctx,
                "error": str(exc),
            }

        self.frame_index += 1
        return output

    def reset(self) -> None:
        self._mask_worker_running = False
        if self._mask_worker_thread is not None and self._mask_worker_thread.is_alive():
            self._mask_worker_thread.join(timeout=1.0)
        self._clear_queue(self._mask_request_queue)
        self._clear_queue(self._mask_result_queue)

        with self._mask_lock:
            self._mask_cache = None

        self.frame_index = 0
        self.last_ctx = None
        self.last_result_bgr = None
        self.last_result_small = None
        self.last_error = None
        self.last_frame_time = None
        self.stabilizer.reset()
        self.accessory_motion_state = {"sides": {}}
        self._hair_processor.reset()

        if bool(self.config.mask_worker_enabled):
            self._mask_worker_thread = None
            self._start_mask_worker()

    def stop(self) -> None:
        self._mask_worker_running = False
        if self._mask_worker_thread is not None and self._mask_worker_thread.is_alive():
            self._mask_worker_thread.join(timeout=1.0)
        self._hair_processor.stop()
