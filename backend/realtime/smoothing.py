from __future__ import annotations

import copy
import numpy as np


class ConstantVelocityPointFilter:
    """
    Lightweight constant-velocity Kalman-style alpha-beta filter.

    It is intentionally cheaper than cv2.KalmanFilter per landmark and works
    well for realtime overlay anchors:
    - predict position from previous velocity
    - correct with current MediaPipe measurement
    - extrapolate a small lead time to compensate camera/API latency
    """

    def __init__(
        self,
        alpha: float = 0.55,
        beta: float = 0.22,
        velocity_decay: float = 0.85,
    ):
        self.alpha = float(np.clip(alpha, 0.05, 1.0))
        self.beta = float(np.clip(beta, 0.0, 1.0))
        self.velocity_decay = float(np.clip(velocity_decay, 0.0, 1.0))
        self.position: np.ndarray | None = None
        self.velocity = np.zeros(2, dtype=np.float32)

    def reset(self) -> None:
        self.position = None
        self.velocity[:] = 0.0

    def update(
        self,
        measurement,
        dt: float,
        lead_time: float = 0.0,
        max_jump: float | None = None,
    ):
        if measurement is None:
            if self.position is None:
                return None
            lead = max(0.0, float(lead_time))
            return tuple(float(x) for x in (self.position + self.velocity * lead))

        measured = np.array(measurement, dtype=np.float32)
        if measured.size < 2 or not np.all(np.isfinite(measured[:2])):
            return measurement
        measured = measured[:2]

        if self.position is None:
            self.position = measured.copy()
            self.velocity[:] = 0.0
            return tuple(float(x) for x in measured)

        dt = float(np.clip(dt, 1.0 / 120.0, 0.18))
        predicted = self.position + self.velocity * dt
        residual = measured - predicted

        if max_jump is not None and float(np.linalg.norm(residual)) > float(max_jump):
            self.position = measured.copy()
            self.velocity[:] = 0.0
            return tuple(float(x) for x in measured)

        self.position = predicted + residual * self.alpha
        self.velocity = self.velocity * self.velocity_decay + (residual / dt) * self.beta

        lead = float(np.clip(lead_time, 0.0, 0.12))
        out = self.position + self.velocity * lead
        return tuple(float(x) for x in out)


class ConstantVelocityScalarFilter:
    def __init__(
        self,
        alpha: float = 0.55,
        beta: float = 0.18,
        velocity_decay: float = 0.85,
    ):
        self.alpha = float(np.clip(alpha, 0.05, 1.0))
        self.beta = float(np.clip(beta, 0.0, 1.0))
        self.velocity_decay = float(np.clip(velocity_decay, 0.0, 1.0))
        self.value: float | None = None
        self.velocity = 0.0

    def reset(self) -> None:
        self.value = None
        self.velocity = 0.0

    def update(
        self,
        measurement,
        dt: float,
        lead_time: float = 0.0,
        max_jump: float | None = None,
    ):
        if measurement is None:
            if self.value is None:
                return None
            return float(self.value + self.velocity * max(0.0, float(lead_time)))

        measured = float(measurement)
        if not np.isfinite(measured):
            return measurement

        if self.value is None:
            self.value = measured
            self.velocity = 0.0
            return measured

        dt = float(np.clip(dt, 1.0 / 120.0, 0.18))
        predicted = self.value + self.velocity * dt
        residual = measured - predicted

        if max_jump is not None and abs(residual) > float(max_jump):
            self.value = measured
            self.velocity = 0.0
            return measured

        self.value = predicted + residual * self.alpha
        self.velocity = self.velocity * self.velocity_decay + (residual / dt) * self.beta

        lead = float(np.clip(lead_time, 0.0, 0.12))
        return float(self.value + self.velocity * lead)


class RealtimeAnchorStabilizer:
    """
    Stabilizes only low-dimensional render anchors, not all 478 landmarks.

    This keeps CPU cost tiny while fixing the visible jitter of glasses/hair
    clips and reducing the lag that plain temporal averaging introduces.
    """

    def __init__(self):
        self._point_filters: dict[str, ConstantVelocityPointFilter] = {}
        self._scalar_filters: dict[str, ConstantVelocityScalarFilter] = {}

    def reset(self) -> None:
        self._point_filters.clear()
        self._scalar_filters.clear()

    def _point(
        self,
        key: str,
        value,
        dt: float,
        lead_time: float,
        max_jump: float | None = None,
    ):
        filt = self._point_filters.setdefault(key, ConstantVelocityPointFilter())
        return filt.update(value, dt=dt, lead_time=lead_time, max_jump=max_jump)

    def _scalar(
        self,
        key: str,
        value,
        dt: float,
        lead_time: float,
        max_jump: float | None = None,
    ):
        filt = self._scalar_filters.setdefault(key, ConstantVelocityScalarFilter())
        return filt.update(value, dt=dt, lead_time=lead_time, max_jump=max_jump)

    def stabilize_anchors(
        self,
        anchors: dict,
        dt: float,
        lead_time: float,
    ) -> dict:
        if not isinstance(anchors, dict):
            return anchors

        out = copy.deepcopy(anchors)
        metrics = out.get("metrics", {}) if isinstance(out.get("metrics"), dict) else {}
        face_width = float(metrics.get("face_width", 160.0) or 160.0)
        max_point_jump = max(18.0, face_width * 0.35)

        try:
            if isinstance(out.get("metrics"), dict):
                for key in ("face_width", "face_height"):
                    if key in out["metrics"]:
                        out["metrics"][key] = self._scalar(
                            f"metrics.{key}",
                            out["metrics"].get(key),
                            dt,
                            lead_time * 0.35,
                            max_jump=face_width * 0.35,
                        )
        except Exception:
            pass

        try:
            glasses = out.get("glasses")
            if isinstance(glasses, dict):
                if "center" in glasses:
                    glasses["center"] = self._point(
                        "glasses.center",
                        glasses.get("center"),
                        dt,
                        lead_time,
                        max_jump=max_point_jump,
                    )
                for key in ("width", "eye_distance", "temple_distance"):
                    if key in glasses:
                        glasses[key] = self._scalar(
                            f"glasses.{key}",
                            glasses.get(key),
                            dt,
                            lead_time * 0.35,
                            max_jump=face_width * 0.30,
                        )
                if "roll_deg" in glasses:
                    glasses["roll_deg"] = self._scalar(
                        "glasses.roll",
                        glasses.get("roll_deg"),
                        dt,
                        lead_time,
                        max_jump=18.0,
                    )
        except Exception:
            pass

        try:
            earrings = out.get("earrings")
            if isinstance(earrings, dict):
                for side in ("left", "right"):
                    if side in earrings:
                        earrings[side] = self._point(
                            f"earrings.{side}",
                            earrings.get(side),
                            dt,
                            lead_time * 0.70,
                            max_jump=max_point_jump,
                        )
        except Exception:
            pass

        try:
            hair_clip = out.get("hair_clip")
            if isinstance(hair_clip, dict):
                if "center" in hair_clip:
                    hair_clip["center"] = self._point(
                        "hair_clip.center",
                        hair_clip.get("center"),
                        dt,
                        lead_time,
                        max_jump=max_point_jump,
                    )
                if "width" in hair_clip:
                    hair_clip["width"] = self._scalar(
                        "hair_clip.width",
                        hair_clip.get("width"),
                        dt,
                        lead_time * 0.35,
                        max_jump=face_width * 0.30,
                    )
                if "roll_deg" in hair_clip:
                    hair_clip["roll_deg"] = self._scalar(
                        "hair_clip.roll",
                        hair_clip.get("roll_deg"),
                        dt,
                        lead_time,
                        max_jump=18.0,
                    )
        except Exception:
            pass

        try:
            necklace = out.get("necklace")
            if isinstance(necklace, dict):
                if "center" in necklace:
                    necklace["center"] = self._point(
                        "necklace.center",
                        necklace.get("center"),
                        dt,
                        lead_time * 0.50,
                        max_jump=max_point_jump,
                    )
                if "width" in necklace:
                    necklace["width"] = self._scalar(
                        "necklace.width",
                        necklace.get("width"),
                        dt,
                        lead_time * 0.25,
                        max_jump=face_width * 0.35,
                    )
        except Exception:
            pass

        return out

    def stabilize_pose(
        self,
        pose: dict,
        dt: float,
        lead_time: float,
    ) -> dict:
        if not isinstance(pose, dict):
            return pose

        out = copy.deepcopy(pose)
        euler = out.get("euler")
        if not isinstance(euler, dict):
            return out

        for key in ("pitch", "yaw", "roll"):
            if key in euler:
                euler[key] = self._scalar(
                    f"pose.{key}",
                    euler.get(key),
                    dt,
                    lead_time,
                    max_jump=25.0,
                )
        return out


def smooth_point(
    previous,
    current,
    alpha: float,
):
    if previous is None:
        return current

    if current is None:
        return previous

    p = np.array(previous, dtype=np.float32)
    c = np.array(current, dtype=np.float32)

    out = p * alpha + c * (1.0 - alpha)

    return tuple(float(x) for x in out)


def smooth_numeric(
    previous,
    current,
    alpha: float,
):
    if previous is None:
        return current

    if current is None:
        return previous

    return float(previous) * alpha + float(current) * (1.0 - alpha)


def smooth_anchors(
    previous_anchors: dict | None,
    current_anchors: dict,
    alpha: float = 0.65,
) -> dict:
    """
    Smooth anchor values between frames.

    This reduces jitter for:
    - glasses center
    - earrings anchor
    - necklace anchor
    - face metrics
    """

    if previous_anchors is None:
        return copy.deepcopy(current_anchors)

    smoothed = copy.deepcopy(current_anchors)

    # Glasses
    try:
        if "glasses" in current_anchors:
            smoothed["glasses"]["center"] = smooth_point(
                previous_anchors.get("glasses", {}).get("center"),
                current_anchors["glasses"].get("center"),
                alpha,
            )

            smoothed["glasses"]["roll_deg"] = smooth_numeric(
                previous_anchors.get("glasses", {}).get("roll_deg"),
                current_anchors["glasses"].get("roll_deg"),
                alpha,
            )

            smoothed["glasses"]["width"] = smooth_numeric(
                previous_anchors.get("glasses", {}).get("width"),
                current_anchors["glasses"].get("width"),
                alpha,
            )
    except Exception:
        pass

    # Earrings
    try:
        if "earrings" in current_anchors:
            smoothed["earrings"]["left"] = smooth_point(
                previous_anchors.get("earrings", {}).get("left"),
                current_anchors["earrings"].get("left"),
                alpha,
            )

            smoothed["earrings"]["right"] = smooth_point(
                previous_anchors.get("earrings", {}).get("right"),
                current_anchors["earrings"].get("right"),
                alpha,
            )
    except Exception:
        pass

    # Necklace
    try:
        if "necklace" in current_anchors:
            smoothed["necklace"]["center"] = smooth_point(
                previous_anchors.get("necklace", {}).get("center"),
                current_anchors["necklace"].get("center"),
                alpha,
            )

            smoothed["necklace"]["width"] = smooth_numeric(
                previous_anchors.get("necklace", {}).get("width"),
                current_anchors["necklace"].get("width"),
                alpha,
            )
    except Exception:
        pass

    return smoothed


def smooth_pose(
    previous_pose: dict | None,
    current_pose: dict,
    alpha: float = 0.65,
) -> dict:
    if previous_pose is None:
        return copy.deepcopy(current_pose)

    smoothed = copy.deepcopy(current_pose)

    try:
        prev_euler = previous_pose.get("euler", {})
        curr_euler = current_pose.get("euler", {})

        smoothed.setdefault("euler", {})

        for key in ["pitch", "yaw", "roll"]:
            smoothed["euler"][key] = smooth_numeric(
                prev_euler.get(key),
                curr_euler.get(key),
                alpha,
            )

    except Exception:
        pass

    return smoothed
