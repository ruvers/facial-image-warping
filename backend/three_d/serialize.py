from __future__ import annotations

import os
from typing import Any

import cv2
import numpy as np


def _to_builtin(value: Any) -> Any:
    """
    Convert numpy-heavy objects to JSON-safe Python values.
    """

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, dict):
        return {
            str(k): _to_builtin(v)
            for k, v in value.items()
        }

    if isinstance(value, list):
        return [
            _to_builtin(v)
            for v in value
        ]

    if isinstance(value, tuple):
        return [
            _to_builtin(v)
            for v in value
        ]

    return value


def depth_stats(depth_map: np.ndarray | None) -> dict:
    if depth_map is None:
        return {
            "available": False,
            "min": None,
            "max": None,
            "mean": None,
            "shape": None,
        }

    d = depth_map.astype(np.float32)

    return {
        "available": True,
        "min": float(np.min(d)),
        "max": float(np.max(d)),
        "mean": float(np.mean(d)),
        "shape": list(d.shape),
    }


def save_depth_visualization(
    depth_map: np.ndarray,
    output_path: str,
) -> None:
    """
    Save pseudo/true depth map as a colored debug image.
    """

    os.makedirs(
        os.path.dirname(output_path),
        exist_ok=True,
    )

    d = depth_map.astype(np.float32)

    min_v = float(np.min(d))
    max_v = float(np.max(d))

    if max_v > min_v:
        norm = (
            (d - min_v)
            * (255.0 / (max_v - min_v))
        ).astype(np.uint8)
    else:
        norm = np.zeros_like(
            d,
            dtype=np.uint8,
        )

    colored = cv2.applyColorMap(
        norm,
        cv2.COLORMAP_TURBO,
    )

    ok = cv2.imwrite(
        output_path,
        colored,
    )

    if not ok:
        raise RuntimeError(
            f"Could not save depth visualization: {output_path}"
        )


def summarize_face_context(
    ctx: dict,
) -> dict:
    """
    Return compact JSON-safe summary of the face/3D context.
    """

    three_d = ctx.get("three_d", {})

    landmarks_2d = ctx.get("landmarks_2d")
    landmarks_3d = ctx.get("landmarks_3d")

    vertices = three_d.get("vertices")
    faces = three_d.get("faces")
    depth_map = three_d.get("depth_map")

    summary = {
        "landmarks": {
            "landmarks_2d_shape": list(landmarks_2d.shape)
            if isinstance(landmarks_2d, np.ndarray)
            else None,
            "landmarks_3d_shape": list(landmarks_3d.shape)
            if isinstance(landmarks_3d, np.ndarray)
            else None,
        },

        "pose": ctx.get("pose", {}),

        "anchors": ctx.get("anchors", {}),

        "three_d": {
            "provider": three_d.get("provider"),
            "provider_priority": three_d.get("provider_priority"),
            "is_true_3d": bool(three_d.get("is_true_3d", False)),

            "camera": three_d.get("camera", {}),

            "mesh": {
                "type": three_d.get("mesh", {}).get("type"),
                "is_true_3d": bool(
                    three_d.get("mesh", {}).get("is_true_3d", False)
                ),
                "vertex_count": int(vertices.shape[0])
                if isinstance(vertices, np.ndarray)
                else 0,
                "face_count": int(faces.shape[0])
                if isinstance(faces, np.ndarray)
                else 0,
            },

            "depth": depth_stats(depth_map)
            if isinstance(depth_map, np.ndarray)
            else depth_stats(None),

            "anchor_points": three_d.get("anchor_points", {}),

            "future_provider_slots": three_d.get(
                "future_provider_slots",
                {},
            ),
        },
    }

    return _to_builtin(summary)