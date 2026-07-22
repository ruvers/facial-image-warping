from __future__ import annotations

import json
import re
import unicodedata
from copy import deepcopy
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[1]
ASSETS_DIR = ROOT_DIR / "assets"
MANIFEST_PATH = ASSETS_DIR / "manifest.json"
PALETTES_DIR = ASSETS_DIR / "palettes"

PALETTE_FILES = {
    "hair_colors": "hair_colors.json",
    "eye_colors": "eye_colors.json",
    "makeup_colors": "makeup_colors.json",
    "accessory_colors": "accessory_colors.json",
}

AUTO_ACCESSORY_DIRS = {
    "glasses": [
        ASSETS_DIR / "glasses",
        ASSETS_DIR / "accessories" / "glasses",
    ],
    "earrings": [
        ASSETS_DIR / "earrings",
        ASSETS_DIR / "accessories" / "earrings",
    ],
    "hats": [
        ASSETS_DIR / "hats",
        ASSETS_DIR / "accessories" / "hats",
    ],
    "hair_clips": [
        ASSETS_DIR / "hair_clips",
        ASSETS_DIR / "accessories" / "hair_clips",
    ],
    "necklaces": [
        ASSETS_DIR / "necklaces",
        ASSETS_DIR / "necklace",
        ASSETS_DIR / "accessories" / "necklaces",
        ASSETS_DIR / "accessories" / "necklace",
    ],
}


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _slug_id(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    ascii_value = normalized.encode("ascii", "ignore").decode("ascii")
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", ascii_value).strip("_").lower()
    return slug or "asset"


def _title_from_stem(stem: str) -> str:
    cleaned = re.sub(r"[_\-]+", " ", stem).strip()
    return cleaned or stem


def _auto_asset_record(category: str, path: Path) -> dict[str, Any]:
    rel_path = path.relative_to(ROOT_DIR).as_posix()
    stem = path.stem
    asset_id = _slug_id(stem)

    base = {
        "id": asset_id,
        "label": _title_from_stem(stem),
        "name": _title_from_stem(stem),
        "path": rel_path,
        "license": "user-local",
        "placeholder": False,
        "asset_role": "user_png_overlay",
        "supported_modes": ["overlay_2d"],
        "render_modes": ["overlay_2d"],
    }

    if category == "glasses":
        return {
            **base,
            "type": "glasses",
            "category": "glasses",
            "default_scale": 1.0,
            "default_offset_x": 0.0,
            "default_offset_y": 0.0,
            "default_alpha": 0.96,
        }

    if category == "earrings":
        inferred_type = "hoop"
        stem_lower = stem.lower()
        if any(term in stem_lower for term in ("stud", "dot", "button")):
            inferred_type = "stud"
        elif any(term in stem_lower for term in ("long", "chain", "drop")):
            inferred_type = "long_dangle"
        elif any(term in stem_lower for term in ("dangle", "pendant", "chandelier")):
            inferred_type = "dangle"
        elif any(term in stem_lower for term in ("hoop", "ring", "loop")):
            inferred_type = "hoop"
        return {
            **base,
            "type": "earrings",
            "category": "earrings",
            "default_scale": 0.16,
            "default_offset_x": 0.0,
            "default_offset_y": 0.0,
            "default_offset_y_ratio": 0.04,
            "default_alpha": 0.96,
            "default_metadata": {
                "earring_type": inferred_type,
                "motion_preset": "normal",
                "swing_intensity": 0.8,
                "anchor_x": 0.5,
                "anchor_y": 0.08,
            },
        }

    if category == "hats":
        return {
            **base,
            "type": "hat",
            "category": "hats",
            "default_scale": 0.95,
            "default_offset_x": 0.0,
            "default_offset_y": 0.0,
            "default_alpha": 0.96,
        }

    if category == "hair_clips":
        return {
            **base,
            "type": "hair_clip",
            "category": "hair_clips",
            "default_scale": 0.34,
            "default_offset_x": 0.0,
            "default_offset_y": 0.0,
            "default_rotation": -12.0,
            "default_alpha": 0.96,
        }

    return {
        **base,
        "type": "necklace",
        "category": "necklaces",
        "default_scale": 1.0,
        "default_offset_x": 0.0,
        "default_offset_y": 0.0,
        "default_offset_y_ratio": 0.0,
        "default_alpha": 0.96,
    }


def _merge_auto_accessory_assets(manifest: dict[str, Any]) -> None:
    categories = manifest.setdefault("categories", {})
    if not isinstance(categories, dict):
        manifest["categories"] = {}
        categories = manifest["categories"]

    existing_paths = set()
    existing_ids: dict[str, set[str]] = {}
    for category, assets in categories.items():
        if not isinstance(assets, list):
            continue
        existing_ids[category] = {
            str(asset.get("id"))
            for asset in assets
            if isinstance(asset, dict) and asset.get("id")
        }
        for asset in assets:
            if isinstance(asset, dict) and asset.get("path"):
                existing_paths.add(str(asset.get("path")).replace("\\", "/"))

    for category, dirs in AUTO_ACCESSORY_DIRS.items():
        assets = categories.setdefault(category, [])
        if not isinstance(assets, list):
            assets = []
            categories[category] = assets

        used_ids = existing_ids.setdefault(category, set())
        for directory in dirs:
            if not directory.exists():
                continue
            for path in sorted(directory.glob("*.png")):
                rel_path = path.relative_to(ROOT_DIR).as_posix()
                if rel_path in existing_paths:
                    continue

                record = _auto_asset_record(category, path)
                base_id = record["id"]
                candidate_id = base_id
                suffix = 2
                while candidate_id in used_ids:
                    candidate_id = f"{base_id}_{suffix}"
                    suffix += 1
                record["id"] = candidate_id
                used_ids.add(candidate_id)
                existing_paths.add(rel_path)
                assets.append(record)


def _ensure_relative_asset_path(path: str) -> Path:
    if not path or not isinstance(path, str):
        raise ValueError("Asset path is required.")

    raw_path = Path(path)
    if raw_path.is_absolute():
        raise ValueError("Absolute asset paths are not allowed.")

    if ".." in raw_path.parts:
        raise ValueError("Path traversal is not allowed in asset paths.")

    if raw_path.parts[0] != "assets":
        raise ValueError("Asset path must start with assets/.")

    return raw_path


def load_asset_manifest() -> dict[str, Any]:
    if not MANIFEST_PATH.exists():
        return {
            "version": 1,
            "local_only": True,
            "categories": {},
            "error": f"Manifest not found: {MANIFEST_PATH}",
        }

    manifest = _read_json(MANIFEST_PATH)
    _merge_auto_accessory_assets(manifest)
    return deepcopy(manifest)


def load_palettes() -> dict[str, Any]:
    palettes: dict[str, Any] = {}

    for key, filename in PALETTE_FILES.items():
        path = PALETTES_DIR / filename
        if not path.exists():
            palettes[key] = []
            continue

        palettes[key] = _read_json(path)

    return palettes


def validate_asset_path(path: str) -> str:
    rel_path = _ensure_relative_asset_path(path)
    abs_path = (ROOT_DIR / rel_path).resolve()
    assets_root = ASSETS_DIR.resolve()

    try:
        abs_path.relative_to(assets_root)
    except ValueError as exc:
        raise ValueError("Asset must be under assets/.") from exc

    if abs_path.suffix.lower() != ".png":
        raise ValueError("Only PNG accessory assets are allowed.")

    if not abs_path.exists():
        raise FileNotFoundError(f"Asset file does not exist: {rel_path.as_posix()}")

    return str(abs_path)


def _iter_manifest_assets(
    manifest: dict[str, Any] | None = None,
) -> list[tuple[str, dict[str, Any]]]:
    manifest = manifest or load_asset_manifest()
    categories = manifest.get("categories", {})
    items: list[tuple[str, dict[str, Any]]] = []

    if not isinstance(categories, dict):
        return items

    for category, assets in categories.items():
        if not isinstance(assets, list):
            continue

        for asset in assets:
            if isinstance(asset, dict):
                items.append((category, asset))

    return items


def resolve_asset_by_id(
    category: str,
    asset_id: str,
) -> dict[str, Any]:
    if not category or not asset_id:
        raise ValueError("Both category and asset_id are required.")

    manifest = load_asset_manifest()
    assets = manifest.get("categories", {}).get(category, [])

    for asset in assets:
        if not isinstance(asset, dict):
            continue

        if asset.get("id") != asset_id:
            continue

        resolved = deepcopy(asset)
        asset_path = str(asset.get("path") or "").strip()
        if asset_path:
            resolved["absolute_path"] = validate_asset_path(asset_path)
        else:
            render_modes = resolved.get("render_modes", [])
            is_procedural = (
                resolved.get("asset_role") == "procedural_reference"
                or "physics_3d" in render_modes
                or "parametric_3d" in render_modes
                or "hybrid_3d_refine" in render_modes
            )
            if not is_procedural:
                raise ValueError("Asset path is required for non-procedural assets.")
            resolved["absolute_path"] = None
        return resolved

    raise KeyError(f"Unknown asset id '{asset_id}' in category '{category}'.")


def list_assets(category: str | None = None) -> dict[str, Any]:
    manifest = load_asset_manifest()
    categories = manifest.get("categories", {})

    if category:
        return {
            category: deepcopy(categories.get(category, [])),
        }

    return deepcopy(categories)


def category_counts() -> dict[str, int]:
    categories = load_asset_manifest().get("categories", {})
    if not isinstance(categories, dict):
        return {}

    return {
        category: len(assets) if isinstance(assets, list) else 0
        for category, assets in categories.items()
    }


def asset_path_is_manifest_allowed(path: str) -> bool:
    try:
        absolute = Path(validate_asset_path(path)).resolve()
    except Exception:
        return False

    for _category, asset in _iter_manifest_assets():
        try:
            candidate = Path(validate_asset_path(str(asset.get("path", "")))).resolve()
        except Exception:
            continue

        if candidate == absolute:
            return True

    return False
