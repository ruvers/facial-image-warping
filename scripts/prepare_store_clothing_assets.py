from __future__ import annotations

import argparse
import json
import math
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image


ROOT_DIR = Path(__file__).resolve().parents[1]
ASSETS_DIR = ROOT_DIR / "assets"
STORE_DIR = ASSETS_DIR / "store"
MANIFEST_PATH = STORE_DIR / "manifest.json"

DATASET_URL = "https://github.com/alexeygrigorev/clothing-dataset-small/archive/refs/heads/master.zip"
DATASET_ID = "alexeygrigorev_clothing_dataset_small_cc0"
DATASET_SOURCE = "https://github.com/alexeygrigorev/clothing-dataset-small"
GENERATED_ID_PREFIXES = ("cc0_upperbody_", "cc0_lowerbody_", "cc0_dress_")

SLOT_CLASSES = {
    "upperbody": ["t-shirt", "shirt", "longsleeve", "outwear"],
    "lowerbody": ["pants", "shorts", "skirt"],
    "dress": ["dress"],
}

BAD_SOURCE_FILENAMES = {
    "4c2b720f-6916-4edf-a732-847c55eec881.jpg",
    "923d98e5-ff69-4b81-a16a-3fd3571a9a4a.jpg",
    "24890459-3da7-4976-a837-f289e7af0d48.jpg",
    "facbf570-2a46-43cd-b58b-ea03df3e7919.jpg",
}


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"version": 1, "local_only": True, "schema": "facewarp_store_manifest_v1", "items": []}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")


def _download_dataset(zip_path: Path) -> None:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    if zip_path.exists() and zip_path.stat().st_size > 1024 * 1024:
        return
    print(f"[store-assets] downloading {DATASET_URL}")
    urllib.request.urlretrieve(DATASET_URL, zip_path)


def _iter_images(root: Path, class_name: str) -> list[Path]:
    paths: list[Path] = []
    for split in ("validation", "test", "train"):
        directory = root / split / class_name
        if not directory.exists():
            continue
        for path in sorted(directory.iterdir()):
            if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
                paths.append(path)
    return paths


def _estimate_light_bg_mask(rgb: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    corner_size = max(8, int(min(rgb.shape[:2]) * 0.08))
    corners = np.concatenate(
        [
            rgb[:corner_size, :corner_size].reshape(-1, 3),
            rgb[:corner_size, -corner_size:].reshape(-1, 3),
            rgb[-corner_size:, :corner_size].reshape(-1, 3),
            rgb[-corner_size:, -corner_size:].reshape(-1, 3),
        ],
        axis=0,
    )
    bg_color = np.median(corners, axis=0).astype(np.float32)
    dist = np.linalg.norm(rgb.astype(np.float32) - bg_color, axis=2)
    light_bg = ((hsv[:, :, 1] < 45) & (hsv[:, :, 2] > 205)).astype(np.uint8) * 255
    color_bg = (dist < 35).astype(np.uint8) * 255
    return cv2.bitwise_or(light_bg, color_bg)


def _alpha_from_rgb(rgb: np.ndarray) -> np.ndarray:
    h, w = rgb.shape[:2]
    bg = _estimate_light_bg_mask(rgb)
    bg = cv2.morphologyEx(bg, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))

    mask = np.full((h, w), cv2.GC_PR_BGD, dtype=np.uint8)
    mask[bg > 0] = cv2.GC_BGD

    margin_x = max(6, int(w * 0.05))
    margin_y = max(6, int(h * 0.05))
    rect = (
        margin_x,
        margin_y,
        max(1, w - margin_x * 2),
        max(1, h - margin_y * 2),
    )
    inner = np.zeros((h, w), dtype=bool)
    inner[margin_y : h - margin_y, margin_x : w - margin_x] = True
    mask[inner & (bg == 0)] = cv2.GC_PR_FGD

    sure_fg = cv2.erode((bg == 0).astype(np.uint8), np.ones((5, 5), np.uint8), iterations=1) > 0
    mask[sure_fg & inner] = cv2.GC_FGD

    try:
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)
        cv2.grabCut(rgb, mask, rect, bgd_model, fgd_model, 4, cv2.GC_INIT_WITH_MASK)
        fg = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    except Exception:
        fg = 255 - bg

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
    if num_labels > 1:
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        fg = (labels == largest).astype(np.uint8) * 255

    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return cv2.GaussianBlur(fg, (5, 5), 0)


def _make_rgba(path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    with Image.open(path) as image:
        rgb = np.array(image.convert("RGB"), dtype=np.uint8)

    h, w = rgb.shape[:2]
    alpha = _alpha_from_rgb(rgb)

    ys, xs = np.where(alpha > 12)
    if len(xs) == 0 or len(ys) == 0:
        alpha[:, :] = 255
        ys, xs = np.where(alpha > 12)

    pad = max(10, int(max(w, h) * 0.04))
    x1 = max(0, int(xs.min()) - pad)
    x2 = min(w, int(xs.max()) + pad + 1)
    y1 = max(0, int(ys.min()) - pad)
    y2 = min(h, int(ys.max()) + pad + 1)

    rgba = np.dstack([rgb, alpha])[y1:y2, x1:x2]
    debug = {
        "source_size": [int(w), int(h)],
        "crop_box": [int(x1), int(y1), int(x2), int(y2)],
        "foreground_ratio": float(np.mean(alpha > 20)),
    }
    return rgba, debug


def _quality_score(path: Path, slot: str) -> tuple[float, dict[str, Any]]:
    with Image.open(path) as image:
        rgb = np.array(image.convert("RGB"), dtype=np.uint8)
    src_h, src_w = rgb.shape[:2]
    scale = min(1.0, 256.0 / max(src_w, src_h))
    if scale < 1.0:
        rgb = cv2.resize(
            rgb,
            (max(1, int(src_w * scale)), max(1, int(src_h * scale))),
            interpolation=cv2.INTER_AREA,
        )
    h, w = rgb.shape[:2]
    bg = _estimate_light_bg_mask(rgb)
    alpha = _alpha_from_rgb(rgb)
    fg = alpha > 24
    fg_ratio = float(np.mean(fg))

    ys, xs = np.where(fg)
    if len(xs) == 0 or len(ys) == 0:
        return -999.0, {"reason": "no_foreground"}

    bbox_w = max(1, int(xs.max() - xs.min()))
    bbox_h = max(1, int(ys.max() - ys.min()))
    bbox_area_ratio = (bbox_w * bbox_h) / max(1, w * h)
    solidity = float(np.count_nonzero(fg) / max(1, bbox_w * bbox_h))
    aspect = float(bbox_w / max(1, bbox_h))
    center_x = float((xs.min() + xs.max()) / 2.0 / w)
    center_y = float((ys.min() + ys.max()) / 2.0 / h)
    centered = 1.0 - min(1.0, math.hypot(center_x - 0.5, center_y - 0.52) * 2.0)
    resolution = min(1.0, (w * h) / (224 * 224))
    background_simple = float(np.mean(bg > 0))
    if slot == "upperbody":
        aspect_target = 0.85
        fg_target = 0.34
    elif slot == "lowerbody":
        aspect_target = 0.58
        fg_target = 0.30
    else:
        aspect_target = 0.48
        fg_target = 0.34

    score = (
        resolution * 1.2
        + centered * 1.4
        + background_simple * 0.8
        + min(1.0, solidity * 1.6) * 1.8
        + min(1.0, bbox_area_ratio * 2.2) * 1.0
        - abs(fg_ratio - fg_target) * 2.2
        - abs(math.log(max(0.12, aspect) / aspect_target)) * 0.7
    )
    if fg_ratio < 0.08 or fg_ratio > 0.82:
        score -= 3.0
    if solidity < 0.22:
        score -= 4.0
    if bbox_w < w * 0.18 or bbox_h < h * 0.20:
        score -= 3.0
    return score, {
        "width": int(src_w),
        "height": int(src_h),
        "foreground_ratio": round(fg_ratio, 4),
        "bbox_area_ratio": round(float(bbox_area_ratio), 4),
        "solidity": round(solidity, 4),
        "aspect": round(aspect, 4),
        "background_simple": round(background_simple, 4),
        "centered": round(centered, 4),
        "score": round(float(score), 4),
    }


def _canvas_rgba(rgba: np.ndarray, size: int = 768) -> np.ndarray:
    h, w = rgba.shape[:2]
    scale = min((size * 0.86) / max(1, w), (size * 0.88) / max(1, h))
    nw = max(1, int(w * scale))
    nh = max(1, int(h * scale))
    resized = cv2.resize(rgba, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((size, size, 4), dtype=np.uint8)
    x = (size - nw) // 2
    y = (size - nh) // 2
    canvas[y : y + nh, x : x + nw] = resized
    return canvas


def _prepare_slot(
    extracted_root: Path,
    slot: str,
    count: int,
    force: bool,
    max_per_class: int,
) -> list[dict[str, Any]]:
    candidates: list[tuple[float, Path, str, dict[str, Any]]] = []
    for class_name in SLOT_CLASSES[slot]:
        paths = [path for path in _iter_images(extracted_root, class_name) if path.name not in BAD_SOURCE_FILENAMES]
        for path in paths[:max_per_class]:
            score, debug = _quality_score(path, slot)
            candidates.append((score, path, class_name, debug))

    candidates.sort(key=lambda item: item[0], reverse=True)
    selected = candidates[:count]
    if len(selected) < count:
        raise RuntimeError(f"Not enough candidates for {slot}: {len(selected)} < {count}")

    out_dir = STORE_DIR / "garments" / slot
    out_dir.mkdir(parents=True, exist_ok=True)
    if force:
        for old in out_dir.glob("cc0_*.png"):
            old.unlink()

    items: list[dict[str, Any]] = []
    for idx, (score, path, class_name, quality) in enumerate(selected, start=1):
        asset_id = f"cc0_{slot}_{idx:02d}"
        out_path = out_dir / f"{asset_id}.png"
        rgba, alpha_debug = _make_rgba(path)
        canvas = _canvas_rgba(rgba)
        Image.fromarray(canvas, mode="RGBA").save(out_path)
        rel_path = out_path.relative_to(ROOT_DIR).as_posix()
        category = slot
        items.append(
            {
                "id": asset_id,
                "name": f"CC0 {slot.title()} {idx:02d} {class_name.title()}",
                "type": "garment",
                "category": category,
                "slot": slot,
                "pipeline": "virtual_tryon",
                "provider": "ootdiffusion",
                "enabled": True,
                "thumbnail": rel_path,
                "tryon_image": rel_path,
                "model_type": "dc",
                "tryon_category": category,
                "license": "CC0-1.0",
                "source": DATASET_SOURCE,
                "source_dataset_id": DATASET_ID,
                "source_class": class_name,
                "source_filename": path.name,
                "asset_quality": {
                    "source": DATASET_ID,
                    "license": "CC0-1.0",
                    "background": "auto_removed",
                    "fit_ready": True,
                    "score": round(float(score), 4),
                    "metrics": quality,
                    "notes": [
                        "Auto-selected from CC0 dataset by resolution, centered object, and simple background heuristics.",
                        "Alpha/crop prepared locally for store preview and OOTDiffusion garment input.",
                    ],
                },
                "fit_profile": {
                    "target_region": slot,
                    "canonical_view": "front_product",
                    "mask_policy": "provider_generated",
                    "alpha_policy": "transparent_png_prepared",
                    "canvas_size": [768, 768],
                    "scale_hint": 1.0,
                    "offset_x": 0.0,
                    "offset_y": 0.0,
                    "preprocess": alpha_debug,
                },
            }
        )
    return items


def _update_manifest(new_items: list[dict[str, Any]]) -> None:
    manifest = _read_json(MANIFEST_PATH)
    existing = manifest.get("items", [])
    if not isinstance(existing, list):
        existing = []
    existing = [
        item
        for item in existing
        if not (
            isinstance(item, dict)
            and (
                item.get("source_dataset_id") == DATASET_ID
                or str(item.get("id") or "").startswith(GENERATED_ID_PREFIXES)
            )
        )
    ]
    manifest["items"] = existing + new_items
    _write_json(MANIFEST_PATH, manifest)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=15)
    parser.add_argument("--max-per-class", type=int, default=90)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--keep-source", action="store_true")
    args = parser.parse_args()

    cache_dir = STORE_DIR / "_download_cache"
    zip_path = cache_dir / "clothing-dataset-small-master.zip"
    _download_dataset(zip_path)

    all_items: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory() as tmp:
        extract_dir = Path(tmp)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_dir)
        dataset_root = extract_dir / "clothing-dataset-small-master"
        for slot in ("upperbody", "lowerbody", "dress"):
            items = _prepare_slot(dataset_root, slot, args.count, args.force, args.max_per_class)
            all_items.extend(items)
            print(f"[store-assets] prepared {len(items)} {slot} assets")

    _update_manifest(all_items)
    if not args.keep_source:
        shutil.rmtree(cache_dir, ignore_errors=True)
    print({"ok": True, "items": len(all_items), "manifest": str(MANIFEST_PATH)})


if __name__ == "__main__":
    main()
