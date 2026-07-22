"""VITON-HD archive data manager.

Manages the VITON-HD dataset placed at ``assets/store/garments/archive/``.
Provides curated cloth selection, pair lookups, and pre-processed data paths
for integration with the OOTDiffusion virtual try-on pipeline.
"""
from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

_BACKEND_DIR = Path(__file__).resolve().parent
_ROOT_DIR = _BACKEND_DIR.parent
_DEFAULT_ARCHIVE = _ROOT_DIR / "assets" / "store" / "garments" / "archive"

# Curated cloth IDs — hand-picked for variety and quality.
# Each ID corresponds to a file like ``00006_00.jpg`` in ``test/cloth/``.
_CURATED_CLOTH_IDS: list[str] = [
    "00006_00", "00034_00", "00055_00", "00064_00", "00094_00",
    "00135_00", "00205_00", "00286_00", "00348_00", "00396_00",
    "00470_00", "00568_00", "00620_00", "00713_00", "00832_00",
    "00955_00", "01069_00", "01216_00", "01341_00", "01518_00",
    "01731_00", "01900_00", "02097_00", "02400_00", "02579_00",
]

MAX_STORE_CLOTHS = int(os.getenv("FACEWARP_ARCHIVE_MAX_CLOTHS", "25"))


def get_archive_root() -> Path:
    """Return the resolved archive root directory."""
    env = os.getenv("FACEWARP_OOTDIFFUSION_ARCHIVE", "").strip()
    if env:
        return Path(env).resolve()
    return _DEFAULT_ARCHIVE.resolve()


def archive_exists() -> bool:
    """Return *True* if the archive directory exists with expected structure."""
    root = get_archive_root()
    return (root / "test" / "cloth").is_dir()


def get_archive_split_root(split: str = "test") -> Path:
    """Return the root of a dataset split (``test`` or ``train``)."""
    return get_archive_root() / split


# ── Pair Parsing ──────────────────────────────────────────────────────────────

def _parse_pairs_file(path: Path) -> list[tuple[str, str]]:
    """Parse a VITON-HD ``test_pairs.txt`` file.

    Each line has the format ``person_id.jpg cloth_id.jpg``.
    Returns a list of ``(person_stem, cloth_stem)`` tuples.
    """
    pairs: list[tuple[str, str]] = []
    if not path.is_file():
        return pairs
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return pairs
    for line in text.splitlines():
        parts = line.strip().split()
        if len(parts) >= 2:
            person = Path(parts[0]).stem
            cloth = Path(parts[1]).stem
            pairs.append((person, cloth))
    return pairs


def load_test_pairs() -> list[tuple[str, str]]:
    """Load all test pairs from the archive."""
    return _parse_pairs_file(get_archive_root() / "test_pairs.txt")


def pairs_by_cloth() -> dict[str, list[str]]:
    """Return a mapping of cloth_stem → list of person_stems."""
    mapping: dict[str, list[str]] = {}
    for person, cloth in load_test_pairs():
        mapping.setdefault(cloth, []).append(person)
    return mapping


def pairs_by_person() -> dict[str, list[str]]:
    """Return a mapping of person_stem → list of cloth_stems."""
    mapping: dict[str, list[str]] = {}
    for person, cloth in load_test_pairs():
        mapping.setdefault(person, []).append(cloth)
    return mapping


def find_pair_person(cloth_stem: str) -> str | None:
    """Given a cloth stem (e.g. ``00006_00``), return a matching person stem."""
    for person, cloth in load_test_pairs():
        if cloth == cloth_stem:
            return person
    return None


# ── Cloth Selection ───────────────────────────────────────────────────────────

def list_archive_cloths(
    split: str = "test",
    *,
    limit: int | None = None,
    curated_only: bool = True,
) -> list[dict[str, Any]]:
    """List cloth images from the archive.

    Parameters
    ----------
    split:
        Dataset split — ``test`` or ``train``.
    limit:
        Maximum number of items to return.  Defaults to ``MAX_STORE_CLOTHS``.
    curated_only:
        If *True*, only return images from the curated list.
    """
    cloth_dir = get_archive_split_root(split) / "cloth"
    if not cloth_dir.is_dir():
        return []

    if limit is None:
        limit = MAX_STORE_CLOTHS

    items: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    if curated_only:
        candidates = _CURATED_CLOTH_IDS[:limit]
    else:
        all_files = sorted(cloth_dir.iterdir())
        candidates = [f.stem for f in all_files if f.suffix.lower() in {".jpg", ".jpeg", ".png"}]
        if len(candidates) > limit:
            candidates = candidates[:limit]

    for stem in candidates:
        for ext in (".jpg", ".jpeg", ".png"):
            path = cloth_dir / f"{stem}{ext}"
            if path.is_file():
                if stem in seen_ids:
                    break
                seen_ids.add(stem)
                rel = path.relative_to(_ROOT_DIR).as_posix()
                items.append({
                    "stem": stem,
                    "path": path,
                    "relative_path": rel,
                    "split": split,
                })
                break

    return items


# ── Pre-processed Data Lookup ─────────────────────────────────────────────────

def get_preprocess_paths(
    person_stem: str,
    split: str = "test",
) -> dict[str, Path | None]:
    """Return paths to pre-processed data for an archive person image.

    The VITON-HD archive includes:
      - ``agnostic-v3.2/``  — clothing-agnostic person representation
      - ``image-densepose/`` — DensePose output
      - ``image-parse-v3/``  — human parsing segmentation
      - ``image-parse-agnostic-v3.2/`` — agnostic parsing
      - ``openpose_img/``    — OpenPose skeleton visualisation
      - ``openpose_json/``   — OpenPose keypoints JSON
    """
    split_root = get_archive_split_root(split)
    lookup = {
        "agnostic": split_root / "agnostic-v3.2" / f"{person_stem}.jpg",
        "densepose": split_root / "image-densepose" / f"{person_stem}.jpg",
        "parse_v3": split_root / "image-parse-v3" / f"{person_stem}.png",
        "parse_agnostic": split_root / "image-parse-agnostic-v3.2" / f"{person_stem}.png",
        "openpose_img": split_root / "openpose_img" / f"{person_stem}_rendered.png",
        "openpose_json": split_root / "openpose_json" / f"{person_stem}_keypoints.json",
        "person_image": split_root / "image" / f"{person_stem}.jpg",
        "cloth_mask": split_root / "cloth-mask" / f"{person_stem}.jpg",
    }
    return {key: (path if path.is_file() else None) for key, path in lookup.items()}


def is_archive_cloth(cloth_path: str | Path) -> tuple[bool, str | None]:
    """Check if a cloth path is from the archive; return ``(True, stem)`` or ``(False, None)``."""
    try:
        p = Path(cloth_path).resolve()
        archive = get_archive_root()
        p.relative_to(archive)
        return True, p.stem
    except (ValueError, Exception):
        return False, None


def get_archive_person_image(person_stem: str, split: str = "test") -> Path | None:
    """Return the path to an archive person image if it exists."""
    path = get_archive_split_root(split) / "image" / f"{person_stem}.jpg"
    return path if path.is_file() else None


# ── Statistics ────────────────────────────────────────────────────────────────

def get_archive_stats() -> dict[str, Any]:
    """Return archive statistics for status endpoints."""
    root = get_archive_root()
    if not archive_exists():
        return {
            "available": False,
            "root": str(root),
        }

    test_cloth = root / "test" / "cloth"
    train_cloth = root / "train" / "cloth"
    pairs = load_test_pairs()

    cloth_count_test = len(list(test_cloth.iterdir())) if test_cloth.is_dir() else 0
    cloth_count_train = len(list(train_cloth.iterdir())) if train_cloth.is_dir() else 0

    test_image = root / "test" / "image"
    person_count_test = len(list(test_image.iterdir())) if test_image.is_dir() else 0

    preprocess_dirs = [
        "agnostic-v3.2", "image-densepose", "image-parse-v3",
        "image-parse-agnostic-v3.2", "openpose_img", "openpose_json",
    ]
    preprocess_available = {
        d: (root / "test" / d).is_dir()
        for d in preprocess_dirs
    }

    return {
        "available": True,
        "root": str(root),
        "test_cloths": cloth_count_test,
        "train_cloths": cloth_count_train,
        "test_persons": person_count_test,
        "test_pairs": len(pairs),
        "curated_cloths_in_store": min(MAX_STORE_CLOTHS, len(_CURATED_CLOTH_IDS)),
        "preprocess_available": preprocess_available,
        "cuda_optimized": True,
    }
