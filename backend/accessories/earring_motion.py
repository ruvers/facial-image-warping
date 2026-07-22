from __future__ import annotations

from typing import Any

import numpy as np


_PROFILES: dict[str, dict[str, float]] = {
    "stud": {
        "spring": 0.10,
        "damping": 0.72,
        "movement_gain": 0.06,
        "pose_gain": 0.025,
        "max_angle": 2.0,
        "vertical_lag": 0.02,
    },
    "hoop": {
        "spring": 0.16,
        "damping": 0.78,
        "movement_gain": 0.16,
        "pose_gain": 0.055,
        "max_angle": 6.0,
        "vertical_lag": 0.04,
    },
    "dangle": {
        "spring": 0.20,
        "damping": 0.82,
        "movement_gain": 0.24,
        "pose_gain": 0.075,
        "max_angle": 9.0,
        "vertical_lag": 0.07,
    },
    "long_dangle": {
        "spring": 0.22,
        "damping": 0.85,
        "movement_gain": 0.30,
        "pose_gain": 0.095,
        "max_angle": 12.0,
        "vertical_lag": 0.10,
    },
}


def infer_earring_type(params: dict[str, Any] | None = None) -> str:
    params = params or {}
    metadata = params.get("metadata") if isinstance(params.get("metadata"), dict) else {}
    explicit = str(
        metadata.get("earring_type")
        or metadata.get("motion_type")
        or params.get("earring_type")
        or ""
    ).strip().lower()
    if explicit in _PROFILES:
        return explicit

    haystack = " ".join(
        str(value or "").lower()
        for value in (
            params.get("asset_id"),
            params.get("asset_path"),
            params.get("category"),
            metadata.get("name"),
            metadata.get("label"),
        )
    )
    if any(term in haystack for term in ("stud", "dot", "button")):
        return "stud"
    if any(term in haystack for term in ("long", "chain", "drop", "pearl_drop")):
        return "long_dangle"
    if any(term in haystack for term in ("dangle", "drop", "pendant", "chandelier")):
        return "dangle"
    if any(term in haystack for term in ("hoop", "ring", "loop")):
        return "hoop"
    return "hoop"


def motion_profile(earring_type: str, preset: str = "normal") -> dict[str, float]:
    profile = dict(_PROFILES.get(earring_type, _PROFILES["hoop"]))
    preset_scale = {
        "off": 0.0,
        "static": 0.0,
        "subtle": 0.55,
        "auto": 1.0,
        "normal": 1.0,
        "strong": 1.45,
    }.get(str(preset or "normal").lower(), 1.0)
    profile["preset_scale"] = preset_scale
    return profile


def _as_point(value: Any, fallback: tuple[float, float]) -> np.ndarray:
    try:
        return np.array([float(value[0]), float(value[1])], dtype=np.float32)
    except Exception:
        return np.array(fallback, dtype=np.float32)


def _landmarks(ctx: dict[str, Any]) -> np.ndarray | None:
    pts = ctx.get("landmarks_2d")
    if pts is None:
        pts = ctx.get("landmarks")
    if isinstance(pts, np.ndarray) and pts.ndim == 2 and pts.shape[0] > 362:
        return pts[:, :2].astype(np.float32, copy=False)
    return None


def compute_ear_anchor(
    ctx: dict[str, Any],
    side: str,
    fallback: tuple[float, float],
) -> tuple[float, float, float]:
    """Return image-side earlobe pivot and an anchor confidence."""

    pts = _landmarks(ctx)
    anchors = ctx.get("anchors", {}).get("earrings", {})
    if pts is None:
        return tuple(_as_point(anchors.get(side), fallback)), 0.45

    if side == "left":
        candidates = [93, 132, 234]
        face_side = pts[234]
        outward = 1.0  # Left ear is on the right side of screen (high X), so add to X to move outward
    else:
        candidates = [323, 361, 454]
        face_side = pts[454]
        outward = -1.0 # Right ear is on the left side of screen (low X), so subtract from X to move outward

    valid = [pts[idx] for idx in candidates if idx < pts.shape[0]]
    if not valid:
        return tuple(_as_point(anchors.get(side), fallback)), 0.45

    point = np.mean(np.vstack(valid[:2]), axis=0)
    face_width = float(ctx.get("anchors", {}).get("metrics", {}).get("face_width", 140.0))
    point[0] += outward * face_width * 0.08  # Push further outward to the earlobe
    point[1] = max(point[1], face_side[1] + face_width * 0.09)  # Push downward off the cheek

    fallback_point = _as_point(anchors.get(side), tuple(point))
    if np.linalg.norm(point - fallback_point) > face_width * 0.25:
        point = point * 0.65 + fallback_point * 0.35
        confidence = 0.58
    else:
        confidence = 0.82

    return (float(point[0]), float(point[1])), confidence


def compute_visibility(
    ctx: dict[str, Any],
    side: str,
    anchor: tuple[float, float],
    anchor_confidence: float,
) -> tuple[float, dict[str, Any]]:
    pose = ctx.get("pose", {})
    euler = pose.get("euler", {}) if isinstance(pose, dict) else {}
    yaw = float(euler.get("yaw", 0.0))
    side_sign = -1.0 if side == "left" else 1.0

    yaw_hide = np.clip((side_sign * yaw - 8.0) / 42.0, 0.0, 0.70)
    visibility = float(np.clip(anchor_confidence * (1.0 - yaw_hide), 0.12, 1.0))

    hair_mask = ctx.get("masks", {}).get("hair")
    hair_cover = 0.0
    if isinstance(hair_mask, np.ndarray) and hair_mask.ndim == 2:
        h, w = hair_mask.shape[:2]
        x = int(np.clip(round(anchor[0]), 0, max(0, w - 1)))
        y = int(np.clip(round(anchor[1]), 0, max(0, h - 1)))
        radius = max(3, int(ctx.get("anchors", {}).get("metrics", {}).get("face_width", 140.0) * 0.045))
        crop = hair_mask[max(0, y - radius):min(h, y + radius + 1), max(0, x - radius):min(w, x + radius + 1)]
        if crop.size:
            hair_cover = float(np.mean(crop > 20))
            visibility *= float(np.clip(1.0 - hair_cover * 0.55, 0.25, 1.0))

    return visibility, {
        "yaw": yaw,
        "yaw_hide": float(yaw_hide),
        "hair_cover": hair_cover,
        "anchor_confidence": float(anchor_confidence),
    }


def _side_state(state: dict[str, Any], side: str) -> dict[str, Any]:
    sides = state.setdefault("sides", {})
    return sides.setdefault(
        side,
        {
            "swing_angle": 0.0,
            "swing_velocity": 0.0,
            "offset_x": 0.0,
            "offset_y": 0.0,
            "prev_anchor": None,
            "prev_pose": None,
            "visibility": 1.0,
            "missing_frames": 0,
        },
    )


def update_earring_motion(
    motion_state: dict[str, Any] | None,
    side: str,
    anchor: tuple[float, float],
    ctx: dict[str, Any],
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    params = params or {}
    earring_type = infer_earring_type(params)
    metadata = params.get("metadata") if isinstance(params.get("metadata"), dict) else {}
    preset = str(metadata.get("motion_preset") or params.get("motion_preset") or "normal")
    profile = motion_profile(earring_type, preset)
    intensity = float(metadata.get("swing_intensity", params.get("swing_intensity", 1.0)))
    intensity = float(np.clip(intensity, 0.0, 1.8)) * profile["preset_scale"]

    pose = ctx.get("pose", {})
    euler = pose.get("euler", {}) if isinstance(pose, dict) else {}
    yaw = float(euler.get("yaw", 0.0))
    roll = float(euler.get("roll", 0.0))

    if motion_state is None or intensity <= 0.001:
        return {
            "angle_deg": 0.0,
            "offset": (0.0, 0.0),
            "earring_type": earring_type,
            "profile": profile,
            "fallback_static": motion_state is None,
        }

    side_state = _side_state(motion_state, side)
    prev_anchor = side_state.get("prev_anchor")
    prev_pose = side_state.get("prev_pose") or {"yaw": yaw, "roll": roll}

    dx = dy = 0.0
    if prev_anchor is not None:
        dx = float(anchor[0] - prev_anchor[0])
        dy = float(anchor[1] - prev_anchor[1])

    dyaw = float(yaw - float(prev_pose.get("yaw", yaw)))
    droll = float(roll - float(prev_pose.get("roll", roll)))
    side_sign = -1.0 if side == "left" else 1.0
    face_width = float(ctx.get("anchors", {}).get("metrics", {}).get("face_width", 140.0))

    movement_signal = (-dx / max(1.0, face_width)) * profile["movement_gain"] * 180.0
    pose_signal = (droll * 0.35 + dyaw * side_sign * 0.20) * profile["pose_gain"]
    target = (movement_signal + pose_signal) * intensity
    target = float(np.clip(target, -profile["max_angle"], profile["max_angle"]))

    angle = float(side_state.get("swing_angle", 0.0))
    velocity = float(side_state.get("swing_velocity", 0.0))
    velocity = velocity * profile["damping"] + (target - angle) * profile["spring"]
    angle = float(np.clip(angle + velocity, -profile["max_angle"], profile["max_angle"]))

    offset_y = float(np.clip(-dy * profile["vertical_lag"] * intensity, -4.0, 4.0))
    offset_x = float(np.sin(np.deg2rad(angle)) * face_width * 0.006)

    side_state.update(
        {
            "swing_angle": angle,
            "swing_velocity": velocity,
            "offset_x": offset_x,
            "offset_y": offset_y,
            "prev_anchor": (float(anchor[0]), float(anchor[1])),
            "prev_pose": {"yaw": yaw, "roll": roll},
            "missing_frames": 0,
        }
    )

    return {
        "angle_deg": angle,
        "offset": (offset_x, offset_y),
        "earring_type": earring_type,
        "profile": profile,
        "fallback_static": False,
    }


def mark_earring_motion_missing(motion_state: dict[str, Any] | None) -> None:
    if motion_state is None:
        return
    for side_state in motion_state.get("sides", {}).values():
        side_state["swing_velocity"] = float(side_state.get("swing_velocity", 0.0)) * 0.5
        side_state["swing_angle"] = float(side_state.get("swing_angle", 0.0)) * 0.65
        side_state["missing_frames"] = int(side_state.get("missing_frames", 0)) + 1
