from __future__ import annotations

from copy import deepcopy
from typing import Any


PHYSICS_RENDER_MODES = {
    "physics_3d",
    "parametric_3d",
    "hybrid_3d_refine",
    "hat_light_inpaint",
    "anydoor_inpaint",
    "sdxl_ip_adapter_inpaint",
    "generative_hat_inpaint",
    "hybrid_reference_inpaint",
}


DEFAULT_NECKLACE_METADATA: dict[str, Any] = {
    "chain_length": 1.0,
    "chain_thickness": 2.0,
    "stiffness": 0.75,
    "pendant_enabled": True,
    "pendant_size": 0.12,
    "pendant_weight": 1.0,
    "material": "gold",
    "anchor_mode": "clavicle_drape",
    "node_count": 48,
}


DEFAULT_BEANIE_METADATA: dict[str, Any] = {
    "skull_fit": 1.05,
    "fold_height": 0.18,
    "top_sag": 0.08,
    "thickness": 0.08,
    "material": "fabric",
    "color": "#222222",
}


def item_render_mode(item: dict[str, Any] | None) -> str:
    if not isinstance(item, dict):
        return "overlay_2d"
    return str(item.get("render_mode") or "overlay_2d").strip().lower()


def item_type(item: dict[str, Any] | None) -> str:
    if not isinstance(item, dict):
        return ""
    return str(item.get("type") or item.get("category") or "").strip().lower()


def item_category(item: dict[str, Any] | None) -> str:
    if not isinstance(item, dict):
        return ""
    return str(item.get("category") or item.get("type") or "").strip().lower()


def is_physics_necklace_item(item: dict[str, Any] | None) -> bool:
    if not isinstance(item, dict):
        return False

    kind = item_type(item)
    category = item_category(item)

    return (
        item_render_mode(item) in PHYSICS_RENDER_MODES
        and kind in {"necklace", "necklaces", "pendant_necklace", "chain_necklace", "choker"}
        and category in {"necklace", "necklaces", "pendant_necklace", "chain_necklace", "choker"}
    )


def is_parametric_hat_item(item: dict[str, Any] | None) -> bool:
    if not isinstance(item, dict):
        return False

    kind = item_type(item)
    category = item_category(item)

    return (
        item_render_mode(item) in {"parametric_3d", "hybrid_3d_refine"}
        and kind in {"hat", "hats", "beanie"}
        and category in {"hat", "hats", "beanie"}
    )


def normalize_necklace_metadata(item: dict[str, Any] | None) -> dict[str, Any]:
    item = item or {}
    incoming = item.get("metadata")
    metadata = deepcopy(DEFAULT_NECKLACE_METADATA)

    if isinstance(incoming, dict):
        metadata.update(incoming)

    if item.get("scale") is not None:
        metadata["chain_length"] = float(metadata.get("chain_length", 1.0)) * float(item.get("scale", 1.0))

    metadata["chain_length"] = float(max(0.45, min(1.8, float(metadata.get("chain_length", 1.0)))))
    metadata["chain_thickness"] = float(max(1.0, min(8.0, float(metadata.get("chain_thickness", 2.0)))))
    metadata["stiffness"] = float(max(0.05, min(1.0, float(metadata.get("stiffness", 0.75)))))
    metadata["pendant_size"] = float(max(0.03, min(0.28, float(metadata.get("pendant_size", 0.12)))))
    metadata["pendant_weight"] = float(max(0.0, min(3.0, float(metadata.get("pendant_weight", 1.0)))))
    metadata["node_count"] = int(max(16, min(80, int(metadata.get("node_count", 48)))))
    metadata["pendant_enabled"] = bool(metadata.get("pendant_enabled", True))
    metadata["material"] = str(metadata.get("material", "gold")).lower()
    metadata["anchor_mode"] = str(metadata.get("anchor_mode", "clavicle_drape")).lower()

    return metadata


def normalize_beanie_metadata(item: dict[str, Any] | None) -> dict[str, Any]:
    item = item or {}
    incoming = item.get("metadata")
    metadata = deepcopy(DEFAULT_BEANIE_METADATA)

    if isinstance(incoming, dict):
        metadata.update(incoming)

    metadata["skull_fit"] = float(max(0.82, min(1.32, float(metadata.get("skull_fit", 1.05)))))
    metadata["fold_height"] = float(max(0.08, min(0.34, float(metadata.get("fold_height", 0.18)))))
    metadata["top_sag"] = float(max(0.0, min(0.22, float(metadata.get("top_sag", 0.08)))))
    metadata["thickness"] = float(max(0.03, min(0.18, float(metadata.get("thickness", 0.08)))))
    metadata["material"] = str(metadata.get("material", "fabric")).lower()
    metadata["color"] = str(metadata.get("color", "#222222"))

    return metadata
