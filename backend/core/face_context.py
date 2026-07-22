from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class FaceContext:
    """
    Central face analysis object.

    All effects should use this instead of calling MediaPipe / parsing again.
    """

    image_bgr: np.ndarray

    landmarks_2d: np.ndarray        # (468, 2), pixel coordinates
    landmarks_3d: np.ndarray        # (468, 3), pixel x/y + MediaPipe z

    parsing: np.ndarray             # semantic label map
    masks: dict[str, np.ndarray]    # skin, hair, lips, ears, neck, beard...

    pose: dict[str, Any]            # solvePnP output
    depth_map: np.ndarray           # pseudo-depth map

    anchors: dict[str, Any]         # glasses, earrings, necklace etc.

    def to_dict(self) -> dict[str, Any]:
        return {
            "image_bgr": self.image_bgr,
            "landmarks_2d": self.landmarks_2d,
            "landmarks_3d": self.landmarks_3d,
            "parsing": self.parsing,
            "masks": self.masks,
            "pose": self.pose,
            "depth_map": self.depth_map,
            "anchors": self.anchors,
        }