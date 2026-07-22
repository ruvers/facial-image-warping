from __future__ import annotations

from typing import Any

from backend.accessories.metadata import normalize_beanie_metadata


def build_beanie_template(item: dict[str, Any] | None = None) -> dict[str, Any]:
    metadata = normalize_beanie_metadata(item)
    item = item or {}
    metadata["scale"] = float(item.get("scale", 1.0))
    metadata["offset_x"] = float(item.get("offset_x", 0.0))
    metadata["offset_y"] = float(item.get("offset_y", 0.0))
    return {
        "category": "beanie",
        "render_mode": "parametric_3d",
        "metadata": metadata,
        "supports_refinement": False,
        "refinement_slot": "generative_refiner",
    }
