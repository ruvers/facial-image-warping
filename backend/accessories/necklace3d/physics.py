from __future__ import annotations

from typing import Any

import numpy as np


def _as_point(value) -> np.ndarray:
    return np.asarray(value, dtype=np.float32).reshape(2)


def _surface_y_for_chest(x: float, proxy: dict[str, Any]) -> float | None:
    chest = proxy.get("upper_chest_proxy", {})
    cx, cy = _as_point(chest.get("center", [0.0, 0.0]))
    rx = float(chest.get("radius_x", 1.0))
    ry = float(chest.get("radius_y", 1.0))
    if rx <= 1.0 or ry <= 1.0:
        return None
    u = (float(x) - float(cx)) / rx
    if abs(u) > 1.0:
        return None
    return float(cy - ry * np.sqrt(max(0.0, 1.0 - u * u)) * 0.20)


def _push_out_of_neck(node: np.ndarray, proxy: dict[str, Any]) -> tuple[np.ndarray, bool]:
    chest = proxy.get("upper_chest_proxy", {})
    neck_base = _as_point(proxy.get("neck_base", [0.0, 0.0]))
    rx = float(chest.get("neck_radius_x", 1.0))
    ry = float(chest.get("neck_radius_y", 1.0))
    if rx <= 1.0 or ry <= 1.0:
        return node, False

    local = node - neck_base
    q = (local[0] / rx) ** 2 + (local[1] / ry) ** 2
    if q >= 1.0:
        return node, False

    scale = 1.0 / (np.sqrt(float(q)) + 1e-6)
    pushed = neck_base + local * scale
    if pushed[1] < neck_base[1]:
        pushed[1] = neck_base[1] + abs(local[1]) * 0.20
    return pushed.astype(np.float32), True


def simulate_necklace_chain(
    proxy: dict[str, Any],
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    metadata = metadata or {}
    node_count = int(max(16, min(80, int(metadata.get("node_count", 48)))))
    iterations = int(max(12, min(80, int(metadata.get("simulation_iterations", 36)))))
    stiffness = float(max(0.05, min(1.0, float(metadata.get("stiffness", 0.75)))))
    chain_length = float(max(0.45, min(1.8, float(metadata.get("chain_length", 1.0)))))
    pendant_weight = float(max(0.0, min(3.0, float(metadata.get("pendant_weight", 1.0)))))

    left = _as_point(proxy["neck_left"])
    right = _as_point(proxy["neck_right"])
    sternum = _as_point(proxy["sternum_center"])
    neck_base = _as_point(proxy["neck_base"])
    image_size = proxy.get("image_size") or proxy.get("debug", {}).get("image_size") or [512, 512]
    image_w = float(image_size[0])
    image_h = float(image_size[1])
    max_drape_y = float(proxy.get("debug", {}).get("max_drape_y", image_h - 3.0))
    max_drape_y = float(np.clip(max_drape_y, neck_base[1] + 12.0, image_h - 3.0))

    span = float(np.linalg.norm(right - left))
    total_length = max(span * (1.03 + 0.14 * chain_length), span + 5.0)
    segment_length = total_length / float(node_count - 1)
    sag = max(10.0, span * (0.14 + 0.075 * chain_length + 0.012 * pendant_weight))
    sag = min(sag, max(14.0, max_drape_y - neck_base[1] - 10.0))

    nodes = np.zeros((node_count, 2), dtype=np.float32)
    for i in range(node_count):
        t = i / float(node_count - 1)
        base = left * (1.0 - t) + right * t
        u = (t - 0.5) * 2.0
        base[1] += sag * (1.0 - u * u)
        nodes[i] = base

    midpoint = node_count // 2
    nodes[midpoint, 1] = max(nodes[midpoint, 1], sternum[1] - sag * 0.10)
    prev = nodes.copy()
    collision_count = 0

    for _ in range(iterations):
        velocity = (nodes - prev) * 0.80
        prev = nodes.copy()
        nodes += velocity
        nodes[1:-1, 1] += 0.035 + pendant_weight * 0.006
        nodes[midpoint, 1] += 0.055 * pendant_weight

        nodes[0] = left
        nodes[-1] = right

        for _constraint_pass in range(2):
            for i in range(node_count - 1):
                p1 = nodes[i]
                p2 = nodes[i + 1]
                delta = p2 - p1
                dist = float(np.linalg.norm(delta)) + 1e-6
                correction = delta * ((dist - segment_length) / dist) * 0.5 * stiffness

                if i != 0:
                    nodes[i] += correction
                if i + 1 != node_count - 1:
                    nodes[i + 1] -= correction

            nodes[0] = left
            nodes[-1] = right

        for i in range(1, node_count - 1):
            pushed, collided = _push_out_of_neck(nodes[i], proxy)
            if collided:
                collision_count += 1
                nodes[i] = pushed

            surface_y = _surface_y_for_chest(float(nodes[i, 0]), proxy)
            if surface_y is not None and nodes[i, 1] < surface_y:
                nodes[i, 1] = surface_y
                collision_count += 1

            nodes[i, 1] = max(nodes[i, 1], neck_base[1] - span * 0.04)
            nodes[i, 0] = float(np.clip(nodes[i, 0], 2.0, image_w - 3.0))
            nodes[i, 1] = float(np.clip(nodes[i, 1], 2.0, max_drape_y))

    pendant_position = nodes[midpoint].copy()
    pendant_position[1] += max(3.0, span * 0.026 * (1.0 + pendant_weight * 0.12))
    pendant_position[0] = float(np.clip(pendant_position[0], 2.0, image_w - 3.0))
    pendant_position[1] = float(np.clip(pendant_position[1], 2.0, max_drape_y))

    clipped = np.count_nonzero(nodes[:, 1] >= max_drape_y - 0.5)
    if clipped > max(2, node_count // 10):
        # If the lightweight solver flattens against the lower clamp, replace the
        # center section with a controlled catenary-like drape. This keeps the
        # necklace on the visible upper chest instead of creating a straight
        # horizontal line at the image/crop boundary.
        max_curve_y = float(min(max_drape_y - 4.0, neck_base[1] + max(18.0, (max_drape_y - neck_base[1]) * 0.58)))
        for i in range(1, node_count - 1):
            t = i / float(node_count - 1)
            p = left * (1.0 - t) + right * t
            u = (t - 0.5) * 2.0
            p[1] += (max_curve_y - p[1]) * (1.0 - u * u)
            nodes[i] = p
        pendant_position = nodes[midpoint].copy()
        pendant_position[1] = float(min(max_drape_y - 2.0, max_curve_y + max(3.0, span * 0.030)))

    return {
        "nodes_2d": nodes.astype(np.float32),
        "pendant_position": pendant_position.astype(np.float32),
        "collision_count": int(collision_count),
        "simulation_iterations": iterations,
        "debug": {
            "node_count": int(node_count),
            "segment_length": float(segment_length),
            "total_length": float(total_length),
            "stiffness": stiffness,
            "pendant_weight": pendant_weight,
            "fixed_nodes": [0, int(node_count - 1)],
            "max_drape_y": max_drape_y,
        },
    }
