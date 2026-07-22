from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class MaskCache:
    mask: np.ndarray
    landmarks: np.ndarray
    klt_points: np.ndarray
    frame_gray: np.ndarray
    params_hash: int


class KLTMaskPropagator:
    """
    Propagate a soft mask between frames with KLT optical flow.

    This is used only in realtime. Photo-mode hair color still uses the
    higher-quality static BiSeNet/HLS path in backend.effects.hair_color_v2.
    """

    LK_PARAMS = dict(
        winSize=(21, 21),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
        flags=cv2.OPTFLOW_LK_GET_MIN_EIGENVALS,
        minEigThreshold=1e-3,
    )

    def __init__(self) -> None:
        self._cache: MaskCache | None = None
        self.last_confidence: float = 0.0

    def init_from_bisenet(
        self,
        mask_float: np.ndarray,
        frame_gray: np.ndarray,
        landmarks: np.ndarray,
        params_hash: int,
    ) -> None:
        mask_float = np.clip(mask_float.astype(np.float32), 0.0, 1.0)
        mask_uint8 = (mask_float * 255.0).astype(np.uint8)

        klt_points = cv2.goodFeaturesToTrack(
            frame_gray,
            maxCorners=80,
            qualityLevel=0.01,
            minDistance=8,
            mask=mask_uint8,
        )

        if klt_points is None or len(klt_points) < 8:
            klt_points = self._landmark_fallback_points(landmarks)

        self._cache = MaskCache(
            mask=mask_float.copy(),
            landmarks=landmarks.astype(np.float32).copy(),
            klt_points=klt_points.astype(np.float32),
            frame_gray=frame_gray.copy(),
            params_hash=int(params_hash),
        )
        self.last_confidence = 1.0

    def propagate(
        self,
        curr_frame_gray: np.ndarray,
        curr_landmarks: np.ndarray,
    ) -> np.ndarray | None:
        if self._cache is None:
            return None

        cache = self._cache
        h, w = cache.mask.shape[:2]

        if cache.klt_points is None or len(cache.klt_points) < 4:
            return self._affine_fallback(cache, curr_landmarks, h, w)

        try:
            next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
                cache.frame_gray,
                curr_frame_gray,
                cache.klt_points,
                None,
                **self.LK_PARAMS,
            )
            if next_pts is None or status is None:
                return self._affine_fallback(cache, curr_landmarks, h, w)

            prev_pts_back, status_back, _ = cv2.calcOpticalFlowPyrLK(
                curr_frame_gray,
                cache.frame_gray,
                next_pts,
                None,
                **self.LK_PARAMS,
            )
            if prev_pts_back is None or status_back is None:
                return self._affine_fallback(cache, curr_landmarks, h, w)

            fb_error = np.abs(cache.klt_points - prev_pts_back).reshape(-1, 2).max(axis=1)
            good = (status.ravel() == 1) & (status_back.ravel() == 1) & (fb_error < 2.0)

            if int(good.sum()) < 4:
                return self._affine_fallback(cache, curr_landmarks, h, w)

            src_pts = cache.klt_points[good].reshape(-1, 2)
            dst_pts = next_pts[good].reshape(-1, 2)

            matrix, inliers = cv2.estimateAffinePartial2D(
                src_pts,
                dst_pts,
                method=cv2.RANSAC,
                ransacReprojThreshold=3.0,
            )
            if matrix is None:
                return self._affine_fallback(cache, curr_landmarks, h, w)

            warped = cv2.warpAffine(
                cache.mask,
                matrix,
                (w, h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
            warped = np.clip(warped.astype(np.float32), 0.0, 1.0)

            updated_klt = next_pts[good].reshape(-1, 1, 2).astype(np.float32)
            if len(updated_klt) < 30:
                new_pts = cv2.goodFeaturesToTrack(
                    curr_frame_gray,
                    maxCorners=max(8, 80 - len(updated_klt)),
                    qualityLevel=0.01,
                    minDistance=8,
                    mask=(warped * 255.0).astype(np.uint8),
                )
                if new_pts is not None:
                    updated_klt = np.concatenate([updated_klt, new_pts.astype(np.float32)], axis=0)

            confidence = float(good.sum() / max(1, len(cache.klt_points)))
            if inliers is not None and len(inliers) > 0:
                confidence *= float(np.count_nonzero(inliers) / max(1, len(inliers)))
            self.last_confidence = float(np.clip(confidence, 0.0, 1.0))

            self._cache = MaskCache(
                mask=warped,
                landmarks=curr_landmarks.astype(np.float32).copy(),
                klt_points=updated_klt,
                frame_gray=curr_frame_gray.copy(),
                params_hash=cache.params_hash,
            )
            return warped
        except Exception:
            return self._affine_fallback(cache, curr_landmarks, h, w)

    def _affine_fallback(
        self,
        cache: MaskCache,
        curr_landmarks: np.ndarray,
        h: int,
        w: int,
    ) -> np.ndarray:
        try:
            idxs = [133, 362, 152]
            src = np.float32([cache.landmarks[i] for i in idxs])
            dst = np.float32([curr_landmarks[i] for i in idxs])
            matrix = cv2.getAffineTransform(src, dst)
            warped = cv2.warpAffine(cache.mask, matrix, (w, h), flags=cv2.INTER_LINEAR)
            self._cache = MaskCache(
                mask=np.clip(warped.astype(np.float32), 0.0, 1.0),
                landmarks=curr_landmarks.astype(np.float32).copy(),
                klt_points=self._landmark_fallback_points(curr_landmarks),
                frame_gray=cache.frame_gray.copy(),
                params_hash=cache.params_hash,
            )
            self.last_confidence = 0.25
            return self._cache.mask
        except Exception:
            self.last_confidence = 0.0
            return cache.mask

    def _landmark_fallback_points(self, landmarks: np.ndarray) -> np.ndarray:
        indices = [
            10, 338, 297, 332, 284, 251, 389, 356,
            454, 323, 361, 288, 397, 365, 379, 378,
            400, 377, 152, 148, 176, 149, 150, 136,
            172, 58, 132, 93, 234, 127, 162, 21,
            54, 103, 67, 109,
        ]
        valid = [idx for idx in indices if idx < len(landmarks)]
        if not valid:
            return np.zeros((0, 1, 2), dtype=np.float32)
        return landmarks[valid].astype(np.float32).reshape(-1, 1, 2)

    def get_params_hash(self) -> int | None:
        return self._cache.params_hash if self._cache is not None else None

    def reset(self) -> None:
        self._cache = None
        self.last_confidence = 0.0
