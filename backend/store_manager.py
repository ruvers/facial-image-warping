from __future__ import annotations

import json
import re
import unicodedata
from copy import deepcopy
from pathlib import Path
from typing import Any

from backend.assets_manager import ASSETS_DIR, ROOT_DIR, load_asset_manifest, resolve_asset_by_id
from backend.archive_pairs import archive_exists, list_archive_cloths, get_archive_stats


STORE_DIR = ASSETS_DIR / "store"
STORE_MANIFEST_PATH = STORE_DIR / "manifest.json"
STORE_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}

DEFAULT_SLOTS = {
    "upperbody": {
        "label": "Upper Body",
        "type": "garment",
        "provider": "ootdiffusion",
        "conflicts": ["dress"],
        "required_fit": ["front_product_image", "clean_background", "full_garment_visible"],
    },
    "lowerbody": {
        "label": "Lower Body",
        "type": "garment",
        "provider": "ootdiffusion",
        "conflicts": ["dress"],
        "required_fit": ["front_product_image", "clean_background", "full_garment_visible"],
    },
    "dress": {
        "label": "Dress",
        "type": "garment",
        "provider": "ootdiffusion",
        "conflicts": ["upperbody", "lowerbody"],
        "required_fit": ["front_product_image", "clean_background", "full_garment_visible"],
    },
    "hat": {
        "label": "Hat",
        "type": "accessory",
        "provider": "accessory_overlay",
        "conflicts": [],
        "required_fit": ["transparent_png", "object_tightly_cropped"],
    },
    "glasses": {
        "label": "Glasses",
        "type": "accessory",
        "provider": "accessory_overlay",
        "conflicts": [],
        "required_fit": ["transparent_png", "lens_anchor_metadata_preferred"],
    },
    "earrings": {
        "label": "Earrings",
        "type": "accessory",
        "provider": "accessory_overlay",
        "conflicts": [],
        "required_fit": ["transparent_png", "pivot_anchor_metadata_preferred"],
    },
    "necklace": {
        "label": "Necklace",
        "type": "accessory",
        "provider": "accessory_overlay",
        "conflicts": [],
        "required_fit": ["transparent_png_or_physics_reference"],
    },
    "hair_clip": {
        "label": "Hair Clip",
        "type": "accessory",
        "provider": "accessory_overlay",
        "conflicts": [],
        "required_fit": ["transparent_png", "object_tightly_cropped"],
    },
}


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _slug_id(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    ascii_value = normalized.encode("ascii", "ignore").decode("ascii")
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", ascii_value).strip("_").lower()
    return slug or "store_item"


def _ensure_relative_asset_path(path: str, *, allowed_ext: set[str] | None = None) -> Path:
    if not path or not isinstance(path, str):
        raise ValueError("Store asset path is required.")

    raw_path = Path(path)
    if raw_path.is_absolute():
        raise ValueError("Absolute store asset paths are not allowed.")
    if ".." in raw_path.parts:
        raise ValueError("Path traversal is not allowed in store asset paths.")
    if not raw_path.parts or raw_path.parts[0] != "assets":
        raise ValueError("Store asset path must start with assets/.")

    ext = raw_path.suffix.lower()
    if allowed_ext is not None and ext not in allowed_ext:
        raise ValueError(f"Unsupported store asset extension: {ext}")

    return raw_path


def validate_store_asset_path(path: str, *, required: bool = True) -> str | None:
    if not path:
        if required:
            raise ValueError("Store asset path is required.")
        return None

    rel_path = _ensure_relative_asset_path(path, allowed_ext=STORE_IMAGE_EXTENSIONS)
    abs_path = (ROOT_DIR / rel_path).resolve()
    assets_root = ASSETS_DIR.resolve()

    try:
        abs_path.relative_to(assets_root)
    except ValueError as exc:
        raise ValueError("Store asset must be under assets/.") from exc

    if not abs_path.exists():
        raise FileNotFoundError(f"Store asset file does not exist: {rel_path.as_posix()}")

    return str(abs_path)


def _default_manifest() -> dict[str, Any]:
    return {
        "version": 1,
        "local_only": True,
        "schema": "facewarp_store_manifest_v1",
        "slots": deepcopy(DEFAULT_SLOTS),
        "quality_contract": {
            "garment_images": {
                "accepted_extensions": sorted(STORE_IMAGE_EXTENSIONS),
                "requirements": [
                    "single front-facing garment product image",
                    "no model body in the garment image",
                    "full garment visible without crop",
                    "prefer plain or transparent background",
                ],
            },
            "accessory_images": {
                "accepted_extensions": [".png"],
                "requirements": [
                    "transparent PNG",
                    "tight crop around visible object",
                    "manifest fit_profile includes anchor and scale hints when needed",
                ],
            },
        },
        "items": [],
    }


def _normalize_slot(slot: str) -> str:
    value = str(slot or "").strip().lower().replace("-", "_")
    aliases = {
        "upper_body": "upperbody",
        "upper": "upperbody",
        "top": "upperbody",
        "lower_body": "lowerbody",
        "bottom": "lowerbody",
        "hats": "hat",
        "necklaces": "necklace",
        "hair_clips": "hair_clip",
    }
    return aliases.get(value, value)


def _infer_garment_category(slot: str) -> str:
    if slot in {"upperbody", "lowerbody", "dress"}:
        return slot
    return "upperbody"


def _store_image_item(path: Path, slot: str) -> dict[str, Any]:
    rel = path.relative_to(ROOT_DIR).as_posix()
    item_id = _slug_id(path.stem)
    name = re.sub(r"[_\-]+", " ", path.stem).strip().title() or path.stem
    return {
        "id": item_id,
        "name": name,
        "type": "garment",
        "category": slot,
        "slot": slot,
        "pipeline": "virtual_tryon",
        "provider": "ootdiffusion",
        "enabled": True,
        "thumbnail": rel,
        "tryon_image": rel,
        "model_type": "dc",
        "tryon_category": _infer_garment_category(slot),
        "asset_quality": {
            "source": "auto_discovered",
            "background": "unknown",
            "fit_ready": True,
            "notes": ["Auto-discovered store garment. Add explicit fit_profile for production assets."],
        },
        "fit_profile": {
            "target_region": slot,
            "canonical_view": "front_product",
            "mask_policy": "provider_generated",
            "scale_hint": 1.0,
            "offset_x": 0.0,
            "offset_y": 0.0,
        },
    }


def _merge_auto_store_garments(manifest: dict[str, Any]) -> None:
    items = manifest.setdefault("items", [])
    if not isinstance(items, list):
        manifest["items"] = []
        items = manifest["items"]

    existing_ids = {
        str(item.get("id"))
        for item in items
        if isinstance(item, dict) and item.get("id")
    }
    existing_paths = {
        str(item.get("tryon_image") or item.get("path") or item.get("thumbnail")).replace("\\", "/")
        for item in items
        if isinstance(item, dict)
    }

    for slot in ("upperbody", "lowerbody", "dress"):
        directory = STORE_DIR / "garments" / slot
        if not directory.exists():
            continue
        for path in sorted(directory.iterdir()):
            if not path.is_file() or path.suffix.lower() not in STORE_IMAGE_EXTENSIONS:
                continue
            rel = path.relative_to(ROOT_DIR).as_posix()
            if rel in existing_paths:
                continue
            record = _store_image_item(path, slot)
            base_id = record["id"]
            candidate_id = base_id
            suffix = 2
            while candidate_id in existing_ids:
                candidate_id = f"{base_id}_{suffix}"
                suffix += 1
            record["id"] = candidate_id
            existing_ids.add(candidate_id)
            existing_paths.add(rel)
            items.append(record)


def _merge_archive_cloths(manifest: dict[str, Any]) -> None:
    """Discover curated VITON-HD archive cloth images and add them to the store."""
    if not archive_exists():
        return

    items = manifest.setdefault("items", [])
    if not isinstance(items, list):
        manifest["items"] = []
        items = manifest["items"]

    existing_ids = {
        str(item.get("id"))
        for item in items
        if isinstance(item, dict) and item.get("id")
    }
    existing_paths = {
        str(item.get("tryon_image") or item.get("path") or item.get("thumbnail")).replace("\\", "/")
        for item in items
        if isinstance(item, dict)
    }

    for cloth in list_archive_cloths(split="test", curated_only=True):
        rel = cloth["relative_path"]
        if rel in existing_paths:
            continue

        stem = cloth["stem"]
        item_id = f"archive_{stem}"
        if item_id in existing_ids:
            continue

        name = f"VITON Cloth {stem.replace('_00', '').lstrip('0') or '0'}"
        record = {
            "id": item_id,
            "name": name,
            "type": "garment",
            "category": "upperbody",
            "slot": "upperbody",
            "pipeline": "virtual_tryon",
            "provider": "ootdiffusion",
            "enabled": True,
            "thumbnail": rel,
            "tryon_image": rel,
            "model_type": "dc",
            "tryon_category": "upperbody",
            "source": "viton_hd_archive",
            "archive_stem": stem,
            "asset_quality": {
                "source": "viton_hd_archive",
                "background": "plain_white",
                "fit_ready": True,
                "notes": [
                    "VITON-HD dataset garment with pre-processed data available.",
                ],
            },
            "fit_profile": {
                "target_region": "upperbody",
                "canonical_view": "front_product",
                "mask_policy": "provider_generated",
                "scale_hint": 1.0,
                "offset_x": 0.0,
                "offset_y": 0.0,
            },
        }
        existing_ids.add(item_id)
        existing_paths.add(rel)
        items.append(record)


def _merge_accessory_store_items(manifest: dict[str, Any]) -> None:
    """Expose existing accessory assets in the store without changing accessory logic."""
    items = manifest.setdefault("items", [])
    if not isinstance(items, list):
        manifest["items"] = []
        items = manifest["items"]

    existing_ids = {
        str(item.get("id"))
        for item in items
        if isinstance(item, dict) and item.get("id")
    }
    accessory_categories = load_asset_manifest().get("categories", {})
    category_to_slot = {
        "glasses": "glasses",
        "earrings": "earrings",
        "hats": "hat",
        "necklaces": "necklace",
        "hair_clips": "hair_clip",
    }

    for category, slot in category_to_slot.items():
        assets = accessory_categories.get(category, [])
        if not isinstance(assets, list):
            continue
        for asset in assets:
            if not isinstance(asset, dict):
                continue
            asset_id = str(asset.get("id") or "").strip()
            if not asset_id:
                continue
            store_id = f"{slot}_{asset_id}"
            if store_id in existing_ids:
                continue
            render_modes = asset.get("render_modes") if isinstance(asset.get("render_modes"), list) else []
            path = str(asset.get("path") or "")
            is_overlay_ready = bool(path and "overlay_2d" in render_modes)
            item = {
                "id": store_id,
                "name": asset.get("name") or asset.get("label") or asset_id,
                "type": "accessory",
                "category": category,
                "slot": slot,
                "pipeline": "accessory_overlay",
                "provider": "facewarp_accessory_engine",
                "enabled": is_overlay_ready,
                "thumbnail": path or None,
                "asset_category": category,
                "asset_id": asset_id,
                "render_mode": "overlay_2d" if is_overlay_ready else (render_modes[0] if render_modes else "overlay_2d"),
                "asset_quality": {
                    "source": "accessory_manifest",
                    "fit_ready": is_overlay_ready,
                    "requires_transparent_png": True,
                    "notes": [] if is_overlay_ready else ["Not exposed as a store action because no overlay PNG path is available."],
                },
                "fit_profile": {
                    "target_region": slot,
                    "scale_hint": float(asset.get("default_scale", 1.0)),
                    "offset_x": float(asset.get("default_offset_x", 0.0)),
                    "offset_y": float(asset.get("default_offset_y", 0.0)),
                    "alpha": float(asset.get("default_alpha", 0.96)),
                    "anchor_schema": "facewarp_accessory_anchors_v1",
                },
            }
            existing_ids.add(store_id)
            items.append(item)


def _validate_item(item: dict[str, Any]) -> dict[str, Any]:
    normalized = deepcopy(item)
    normalized["slot"] = _normalize_slot(str(normalized.get("slot") or normalized.get("category") or ""))
    normalized.setdefault("enabled", True)
    normalized.setdefault("pipeline", "virtual_tryon" if normalized.get("type") == "garment" else "accessory_overlay")
    normalized.setdefault("fit_profile", {})
    normalized.setdefault("asset_quality", {})

    pipeline = normalized.get("pipeline")
    if pipeline == "virtual_tryon":
        tryon_path = str(normalized.get("tryon_image") or normalized.get("path") or "")
        try:
            normalized["absolute_tryon_path"] = validate_store_asset_path(tryon_path)
        except Exception as exc:
            normalized["enabled"] = False
            normalized["absolute_tryon_path"] = None
            quality = normalized.setdefault("asset_quality", {})
            quality["fit_ready"] = False
            quality.setdefault("notes", []).append(str(exc))
            normalized["validation_error"] = str(exc)
        if not normalized.get("thumbnail"):
            normalized["thumbnail"] = tryon_path
        normalized.setdefault("model_type", "dc")
        normalized.setdefault("tryon_category", _infer_garment_category(normalized["slot"]))
    elif pipeline == "accessory_overlay":
        asset_category = str(normalized.get("asset_category") or normalized.get("category") or "")
        asset_id = str(normalized.get("asset_id") or "")
        if asset_category and asset_id:
            try:
                resolved = resolve_asset_by_id(asset_category, asset_id)
                normalized["resolved_accessory"] = {
                    "category": asset_category,
                    "asset_id": asset_id,
                    "path": resolved.get("path"),
                    "render_modes": resolved.get("render_modes", []),
                }
                if not normalized.get("thumbnail"):
                    normalized["thumbnail"] = resolved.get("path")
            except Exception as exc:
                normalized["enabled"] = False
                normalized.setdefault("asset_quality", {})["fit_ready"] = False
                normalized.setdefault("asset_quality", {}).setdefault("notes", []).append(str(exc))
        elif normalized.get("thumbnail"):
            validate_store_asset_path(str(normalized.get("thumbnail")), required=False)
    else:
        normalized["enabled"] = False
        normalized.setdefault("asset_quality", {})["fit_ready"] = False
        normalized.setdefault("asset_quality", {}).setdefault("notes", []).append(f"Unknown pipeline: {pipeline}")

    return normalized


def load_store_manifest() -> dict[str, Any]:
    manifest = _default_manifest()
    if STORE_MANIFEST_PATH.exists():
        loaded = _read_json(STORE_MANIFEST_PATH)
        if isinstance(loaded, dict):
            manifest.update(loaded)
            slots = deepcopy(DEFAULT_SLOTS)
            slots.update(loaded.get("slots", {}) if isinstance(loaded.get("slots"), dict) else {})
            manifest["slots"] = slots

    _merge_auto_store_garments(manifest)
    _merge_archive_cloths(manifest)
    _merge_accessory_store_items(manifest)

    items = manifest.get("items", [])
    manifest["items"] = [
        _validate_item(item)
        for item in items
        if isinstance(item, dict) and item.get("id")
    ]
    return deepcopy(manifest)


def list_store_items(slot: str | None = None, item_type: str | None = None) -> list[dict[str, Any]]:
    items = load_store_manifest().get("items", [])
    if slot:
        normalized_slot = _normalize_slot(slot)
        items = [item for item in items if item.get("slot") == normalized_slot]
    if item_type:
        items = [item for item in items if item.get("type") == item_type]
    return deepcopy(items)


def resolve_store_item(item_id: str) -> dict[str, Any]:
    if not item_id:
        raise ValueError("item_id is required.")

    for item in load_store_manifest().get("items", []):
        if item.get("id") == item_id:
            return deepcopy(item)

    raise KeyError(f"Unknown store item id '{item_id}'.")


def resolve_outfit_slots(item_ids: list[str]) -> dict[str, Any]:
    selected: dict[str, dict[str, Any]] = {}
    conflicts: list[dict[str, Any]] = []

    for item_id in item_ids:
        item = resolve_store_item(item_id)
        slot = str(item.get("slot") or "")
        slot_def = DEFAULT_SLOTS.get(slot, {})
        for conflict_slot in slot_def.get("conflicts", []):
            if conflict_slot in selected:
                conflicts.append(
                    {
                        "item_id": item_id,
                        "slot": slot,
                        "conflicts_with_slot": conflict_slot,
                        "conflicts_with_item": selected[conflict_slot].get("id"),
                    }
                )
        selected[slot] = item

    return {
        "success": not conflicts,
        "items": list(selected.values()),
        "slots": selected,
        "conflicts": conflicts,
    }
