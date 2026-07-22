"""
FaceWarp Lab — FastAPI application.

Main responsibilities:
    - Upload image
    - Validate file type / size / dimensions
    - Preprocess image to standard RGB 512x512
    - Run face detection / landmark detection
    - Route processing to:
        1. photo pipeline / effect engine
        2. expression warp
        3. aging simulation
    - Compute FFT + quality metrics
    - Return flat API response compatible with frontend
"""

from __future__ import annotations

from base64 import b64encode
import json
import mimetypes
import os
import time
import uuid
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import cv2
import numpy as np

from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from starlette.requests import Request

from skimage.metrics import (
    mean_squared_error as sk_mse,
    peak_signal_noise_ratio as sk_psnr,
    structural_similarity as sk_ssim,
)

from backend.face_analysis import analyze_face, detect_face_landmarks
from backend.three_d.manager import enrich_with_best_available_3d
from backend.local_models.slots import get_model_slot_status
from backend.three_d.serialize import (
    summarize_face_context,
    save_depth_visualization,
)

from backend.realtime.session_manager import realtime_sessions
from backend.local_models.registry import get_local_model_status
from backend.three_d.deca_provider import get_deca_runtime_status
from backend.three_d.deca_diagnostics import get_deca_diagnostics
from backend.local_models.deca_prepare import prepare_deca_repo_data
from backend.local_models.plugin_validator import validate_all_plugins

from backend.schemas import (
    ErrorDetail,
    ErrorResponse,
    OriginalInfo,
    PathsInfo,
    UploadResponse,
    ProcessRequest,
    PipelineStatus,
    PreprocessInfo,
    MetadataInfo,
    FaceDetectionInfo,
    MetricsInfo,
    LandmarkDetectionInfo,
    ProcessResponse,
)

from backend.preprocessing import (
    preprocess_pipeline,
    image_uint8_rgb_to_bgr,
    build_preprocess_metadata,
)

from backend.face_detection import get_face_detector
from backend.face_parsing import get_face_parsing_status
from backend.face_validation import check_face_orientation
from backend.warping import apply_expression_transform
from backend.ai_expression import (
    apply_ai_expression,
    get_ai_expression_status,
)
from backend.photo_pipeline import apply_photo_pipeline
from backend.effect_catalog import get_effect_catalog
from backend.effects.hair_color_v2 import apply_faceapp_hair_color
from backend.effects.color_v1 import apply_eye_color as apply_eye_color_effect
from backend.local_models.model_requirements import get_model_requirements
from backend.local_models.sam_aging import get_sam_aging_status
from backend.local_models.liveportrait_expression import (
    get_liveportrait_expression_status,
)
from backend.local_models.generative_refiner import get_generative_refiner_status
from backend.virtual_tryon.ootdiffusion import (
    get_virtual_tryon_status,
    run_virtual_tryon,
)
from backend.archive_pairs import (
    get_archive_stats as _archive_stats,
    load_test_pairs as _archive_pairs,
    list_archive_cloths as _archive_cloths,
    archive_exists as _archive_exists,
)
from backend.assets_manager import (
    ASSETS_DIR,
    category_counts,
    list_assets,
    load_asset_manifest,
    load_palettes,
    resolve_asset_by_id,
)
from backend.store_manager import (
    list_store_items,
    load_store_manifest,
    resolve_outfit_slots,
    resolve_store_item,
)


# ── Paths ─────────────────────────────────────────────────────────────────────

BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BACKEND_DIR)
UPLOAD_ROOT = os.path.join(ROOT_DIR, "uploads")
PROCESSED_ROOT = os.path.join(ROOT_DIR, "processed")

STATIC_DIR = os.path.join(ROOT_DIR, "frontend", "static")
TEMPLATES_DIR = os.path.join(ROOT_DIR, "frontend", "templates")

try:
    from dotenv import load_dotenv

    load_dotenv(os.path.join(ROOT_DIR, ".env"))
except Exception as exc:
    print(f"[ENV][WARN] .env could not be loaded: {exc}")


# ── Limits ────────────────────────────────────────────────────────────────────

MAX_FILE_BYTES = 10 * 1024 * 1024
MIN_WIDTH = 200
MIN_HEIGHT = 200


# ── Allowed types ─────────────────────────────────────────────────────────────

ALLOWED_MIME = frozenset(
    {
        "image/jpeg",
        "image/jpg",
        "image/png",
    }
)

ALLOWED_EXTENSIONS = frozenset(
    {
        ".jpg",
        ".jpeg",
        ".png",
    }
)


# ── App ───────────────────────────────────────────────────────────────────────

mimetypes.add_type("application/javascript", ".mjs")
mimetypes.add_type("application/wasm", ".wasm")

app = FastAPI(
    title="FaceWarp Lab API",
    version="2.0.0",
    description="Facial image preprocessing, warping, analysis, and photo effect engine.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def startup_preload_models() -> None:
    try:
        from backend.local_models.preloader import preload_from_env

        result = preload_from_env()
        app.state.model_preload = result

        if result.get("enabled"):
            print("[MODEL_PRELOAD]", result)
        else:
            print("[MODEL_PRELOAD] disabled")

    except Exception as exc:
        app.state.model_preload = {
            "enabled": False,
            "ok": False,
            "error": repr(exc),
        }
        print(f"[MODEL_PRELOAD][WARN] failed: {exc}")

# ── Helpers ───────────────────────────────────────────────────────────────────

def _err(
    code: str,
    message: str,
    status: int = 400,
    details: dict[str, Any] | None = None,
) -> JSONResponse:
    body = ErrorResponse(
        error=ErrorDetail(
            code=code,
            message=message,
            details=details,
        )
    )
    return JSONResponse(
        status_code=status,
        content=body.model_dump(),
    )


def _magic_matches_jpeg(b: bytes) -> bool:
    return (
        len(b) >= 3
        and b[0] == 0xFF
        and b[1] == 0xD8
        and b[2] == 0xFF
    )


def _magic_matches_png(b: bytes) -> bool:
    if len(b) < 8:
        return False

    return b[:8] == bytes(
        [
            0x89,
            0x50,
            0x4E,
            0x47,
            0x0D,
            0x0A,
            0x1A,
            0x0A,
        ]
    )


def _detect_format(data: bytes) -> str | None:
    if _magic_matches_jpeg(data):
        return "JPEG"

    if _magic_matches_png(data):
        return "PNG"

    return None


def _file_extension(filename: str) -> str:
    _, ext = os.path.splitext(filename)
    return ext.lower()


def _decode_raw(data: bytes) -> np.ndarray | None:
    buf = np.frombuffer(
        data,
        dtype=np.uint8,
    )

    return cv2.imdecode(
        buf,
        cv2.IMREAD_UNCHANGED,
    )


def _find_stored_image(
    session_id: str,
    image_id: str,
) -> str | None:
    d = os.path.join(
        UPLOAD_ROOT,
        session_id,
    )

    if not os.path.isdir(d):
        return None

    for ext in ("jpg", "jpeg", "png"):
        p = os.path.join(
            d,
            f"{image_id}.{ext}",
        )

        if os.path.isfile(p):
            return p

    return None


def _safe_imwrite(
    path: str,
    image: np.ndarray,
) -> None:
    ok = cv2.imwrite(
        path,
        image,
    )

    if not ok:
        raise RuntimeError(f"Failed to write image: {path}")


def _build_legacy_photo_params(
    mode: str | None,
    params: dict[str, Any],
) -> dict[str, Any]:
    """
    Compatibility bridge.

    New architecture expects:
        {
            "lipstick": {"enabled": True, ...},
            "hair_color": {"enabled": True, ...},
            ...
        }

    Old frontend still sends:
        mode: "accessory"
        params: {"item_type": "makeup"}

    This keeps old frontend from breaking while allowing the new engine format.
    """

    params = dict(params or {})

    # Already new-format. Keep as-is.
    new_format_keys = {
        "hair_color",
        "lipstick",
        "skin_smooth",
        "blush",
        "eyebrow",
        "accessories",
        "glasses",
        "earrings",
        "necklace",
        "aging_model",
    }

    if any(k in params for k in new_format_keys):
        return params

    # Old accessory → makeup compatibility.
    if mode == "accessory" and params.get("item_type") == "makeup":
        return {
            "skin_smooth": {
                "enabled": True,
                "intensity": 0.20,
            },
            "lipstick": {
                "enabled": True,
                "color": "#A02045",
                "intensity": 0.55,
            },
            "blush": {
                "enabled": True,
                "color": "#D96C7C",
                "intensity": 0.22,
            },
            "eyebrow": {
                "enabled": True,
                "intensity": 0.20,
            },
        }

    # Old accessory glasses/necklace/earring buttons do not carry real asset params yet.
    # Return params unchanged; the engine will simply skip unsupported/missing assets.
    return params


def _merge_process_effect_params(body: ProcessRequest) -> dict[str, Any]:
    params = dict(body.params or {})

    nested_effects = params.pop("effects", None)
    if isinstance(nested_effects, dict):
        params.update(_normalize_editor_effects(nested_effects))

    if body.effects:
        params.update(_normalize_editor_effects(body.effects))

    return params


def _normalize_aging_algorithm(raw: Any) -> str:
    """
    Map frontend / legacy aging labels to the two supported backends.
    """
    s = str(raw or "frequency").strip().lower().replace(" ", "_").replace("-", "_")

    if s in {
        "ai",
        "sam",
        "generative",
        "neural",
        "replicate",
        "ai_based",
        "deep",
        "ml",
    }:
        return "ai"

    if s in {
        "frequency",
        "frequency_based",
        "traditional",
        "legacy",
        "fft",
        "cv",
        "classic",
        "hybrid",
    }:
        return "frequency"

    return "frequency"


def _build_aging_model_params(
    request_params: dict[str, Any],
    body: ProcessRequest,
) -> dict[str, Any]:
    raw_model_params = request_params.get("aging_model")
    model_params: dict[str, Any] = (
        dict(raw_model_params)
        if isinstance(raw_model_params, dict)
        else {}
    )

    intensity = float(
        np.clip(
            float(
                model_params.get(
                    "intensity",
                    request_params.get(
                        "aging_intensity",
                        getattr(body, "aging_intensity", 1.0),
                    ),
                )
            ),
            0.0,
            1.0,
        )
    )

    model_params.setdefault("enabled", intensity > 0.0)
    model_params["intensity"] = intensity

    if "target_age" not in model_params:
        target_age = request_params.get("target_age")
        if target_age is None:
            raw_intensity = request_params.get("aging_intensity", getattr(body, "aging_intensity", 1.0))
            try:
                val = float(raw_intensity)
                if val > 2.0:
                    target_age = val
            except (ValueError, TypeError):
                pass
        
        if target_age is not None:
            model_params["target_age"] = int(np.clip(float(target_age), 7.0, 85.0))
        elif intensity > 0.0:
            model_params["target_age"] = int(round(35 + intensity * 25))

    model_params.setdefault("fallback_to_legacy", True)
    model_params.setdefault("fallback_noop", True)
    model_params.setdefault("strict", False)

    return model_params


def _category_to_accessory_type(category: str) -> str:
    return {
        "glasses": "glasses",
        "earrings": "earrings",
        "necklaces": "necklace",
        "hair_clips": "hair_clip",
        "hats": "hat",
    }.get(category, category)


def _normalize_accessory_items(items: list[Any]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []

    for item in items:
        if not isinstance(item, dict):
            continue

        category = str(item.get("category") or item.get("type") or "").strip()
        accessory_type_raw = str(item.get("type") or category).strip()
        render_mode = str(item.get("render_mode") or "").strip().lower()
        is_necklace_like = accessory_type_raw in {
            "necklace",
            "necklaces",
            "pendant_necklace",
            "chain_necklace",
            "choker",
        } or category in {
            "necklace",
            "necklaces",
            "pendant_necklace",
            "chain_necklace",
            "choker",
        }
        is_hat_like = accessory_type_raw in {
            "hat",
            "hats",
            "beanie",
            "baseball_cap",
            "bucket_hat",
            "fedora",
        } or category in {
            "hat",
            "hats",
            "beanie",
            "baseball_cap",
            "bucket_hat",
            "fedora",
        }
        is_physics_necklace = (
            render_mode in {"physics_3d", "parametric_3d", "hybrid_3d_refine"}
            and is_necklace_like
        )
        is_parametric_hat = (
            render_mode in {"parametric_3d", "hybrid_3d_refine"}
            and is_hat_like
        )

        if is_physics_necklace:
            normalized.append(
                {
                    "type": "necklace",
                    "category": category or "pendant_necklace",
                    "asset_id": str(item.get("asset_id") or ""),
                    "asset_path": str(item.get("asset_path") or ""),
                    "render_mode": render_mode,
                    "metadata": item.get("metadata") if isinstance(item.get("metadata"), dict) else {},
                    "scale": float(item.get("scale", 1.0)),
                    "offset_x": float(item.get("offset_x", 0.0)),
                    "offset_y": float(item.get("offset_y", 0.0)),
                    "offset_y_ratio": float(item.get("offset_y_ratio", 0.0)),
                    "rotation": float(item.get("rotation", 0.0)),
                    "alpha": float(item.get("alpha", item.get("opacity", 1.0))),
                    "debug_placeholder": bool(item.get("debug_placeholder", False)),
                }
            )
            continue

        if is_parametric_hat:
            normalized.append(
                {
                    "type": "hat",
                    "category": "beanie" if category in {"hat", "hats"} else (category or "beanie"),
                    "asset_id": str(item.get("asset_id") or ""),
                    "asset_path": str(item.get("asset_path") or ""),
                    "render_mode": render_mode,
                    "metadata": item.get("metadata") if isinstance(item.get("metadata"), dict) else {},
                    "scale": float(item.get("scale", 1.0)),
                    "offset_x": float(item.get("offset_x", 0.0)),
                    "offset_y": float(item.get("offset_y", 0.0)),
                    "offset_y_ratio": float(item.get("offset_y_ratio", 0.0)),
                    "rotation": float(item.get("rotation", 0.0)),
                    "alpha": float(item.get("alpha", item.get("opacity", 1.0))),
                    "debug_placeholder": bool(item.get("debug_placeholder", False)),
                }
            )
            continue

        asset_id = str(item.get("asset_id") or "").strip()
        resolved_asset: dict[str, Any] | None = None

        if category and asset_id:
            candidate_categories = [category]
            if is_necklace_like:
                candidate_categories.append("necklaces")
            if is_hat_like:
                candidate_categories.append("hats")

            for candidate_category in dict.fromkeys(candidate_categories):
                try:
                    resolved_asset = resolve_asset_by_id(
                        candidate_category,
                        asset_id,
                    )
                    break
                except Exception:
                    resolved_asset = None

        if resolved_asset is None:
            if asset_id and category in {"glasses", "earrings", "hair_clips"}:
                normalized.append(
                    {
                        "type": _category_to_accessory_type(category),
                        "category": category,
                        "asset_id": asset_id,
                        "asset_path": "",
                        "render_mode": render_mode or "overlay_2d",
                        "metadata": item.get("metadata") if isinstance(item.get("metadata"), dict) else {},
                        "scale": float(item.get("scale", 1.0)),
                        "offset_x": float(item.get("offset_x", 0.0)),
                        "offset_y": float(item.get("offset_y", 0.0)),
                        "offset_y_ratio": float(item.get("offset_y_ratio", 0.0)),
                        "rotation": float(item.get("rotation", 0.0)),
                        "alpha": float(item.get("alpha", item.get("opacity", 1.0))),
                        "debug_placeholder": bool(item.get("debug_placeholder", False)),
                        "fallback_reason": "asset_not_found_or_invalid_manifest_path",
                    }
                )
            continue

        accessory_type = str(
            item.get("type")
            or resolved_asset.get("type")
            or _category_to_accessory_type(category)
        )
        resolved_category = str(resolved_asset.get("category") or category)
        requested_metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
        default_metadata = (
            resolved_asset.get("default_metadata")
            if isinstance(resolved_asset.get("default_metadata"), dict)
            else {}
        )
        metadata = {
            **default_metadata,
            **requested_metadata,
        }
        resolved_render_modes = resolved_asset.get("render_modes", [])
        if not isinstance(resolved_render_modes, list):
            resolved_render_modes = []
        resolved_path = str(resolved_asset.get("path") or item.get("asset_path") or "")
        effective_render_mode = str(item.get("render_mode") or "").strip().lower()
        if not effective_render_mode:
            effective_render_mode = "overlay_2d"

        if not resolved_path and resolved_asset.get("asset_role") == "procedural_reference":
            for candidate_mode in (
                "physics_3d",
                "parametric_3d",
                "hybrid_3d_refine",
                "generative_hat_inpaint",
                "hybrid_reference_inpaint",
            ):
                if candidate_mode in resolved_render_modes:
                    effective_render_mode = candidate_mode
                    break

        default_scale = float(resolved_asset.get("default_scale", 1.0))
        raw_scale = item.get("scale")
        if raw_scale is None:
            effective_scale = default_scale
        else:
            requested_scale = float(raw_scale)
            if category in {"earrings", "hair_clips"} and requested_scale >= 0.7:
                # The editor slider is a relative multiplier for tiny overlay
                # assets. Older UI builds send 1.0, which should mean "asset
                # default size", not "one full face width".
                effective_scale = default_scale * requested_scale
            else:
                effective_scale = requested_scale

        base_item = {
            "type": accessory_type,
            "category": resolved_category,
            "asset_id": asset_id,
            "asset_path": resolved_path,
            "render_mode": effective_render_mode,
            "metadata": metadata,
            "scale": effective_scale,
            "offset_x": float(
                item.get(
                    "offset_x",
                    resolved_asset.get("default_offset_x", 0.0),
                )
            ),
            "offset_y": float(
                item.get(
                    "offset_y",
                    resolved_asset.get("default_offset_y", 0.0),
                )
            ),
            "offset_y_ratio": float(
                item.get(
                    "offset_y_ratio",
                    resolved_asset.get("default_offset_y_ratio", 0.0),
                )
            ),
            "rotation": float(item.get("rotation", resolved_asset.get("default_rotation", 0.0))),
            "alpha": float(
                item.get(
                    "alpha",
                    item.get(
                        "opacity",
                        resolved_asset.get("default_alpha", 1.0),
                    ),
                )
            ),
            "debug_placeholder": bool(item.get("debug_placeholder", False)),
        }

        if category == "earrings" and accessory_type in {"earring", "earrings"}:
            left = {
                **base_item,
                "type": "left_earring",
                "asset_id": f"{asset_id}:left",
            }
            right = {
                **base_item,
                "type": "right_earring",
                "asset_id": f"{asset_id}:right",
            }
            normalized.extend([left, right])
        else:
            normalized.append(base_item)

    return normalized


def _normalize_editor_effects(effects: dict[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}

    hair_color = effects.get("hair_color")
    if isinstance(hair_color, dict):
        normalized["hair_color"] = hair_color

    eye_color = effects.get("eye_color")
    if isinstance(eye_color, dict):
        normalized["eye_color"] = eye_color

    makeup = effects.get("makeup")
    if isinstance(makeup, dict):
        for key in ("skin_smooth", "lipstick", "blush", "eyeshadow", "eyeliner", "beard"):
            value = makeup.get(key)
            if isinstance(value, dict):
                normalized[key] = value

    accessories = effects.get("accessories")
    if isinstance(accessories, dict):
        normalized_items = _normalize_accessory_items(
            accessories.get("items", [])
            if isinstance(accessories.get("items"), list)
            else []
        )
        normalized["accessories"] = {
            "enabled": bool(accessories.get("enabled", False)) and bool(normalized_items),
            "items": normalized_items,
        }

    for key, value in effects.items():
        if key not in {"hair_color", "eye_color", "makeup", "accessories"}:
            normalized[key] = value

    return normalized


def _face_detection_status() -> dict[str, Any]:
    try:
        from backend.face_detection import _resolve_face_landmarker_task_path

        task_path = _resolve_face_landmarker_task_path()
        return {
            "provider": "mediapipe",
            "available": True,
            "task_path": str(task_path),
            "primary_anchor_source": True,
        }
    except Exception as e:
        return {
            "provider": "mediapipe",
            "available": False,
            "error": str(e),
            "primary_anchor_source": True,
        }


def compute_fft_info(
    img_rgb: np.ndarray,
) -> dict[str, Any]:
    if img_rgb is None or img_rgb.size == 0:
        raise ValueError("Empty image passed to FFT.")

    if img_rgb.ndim == 2:
        gray = img_rgb.astype(np.float32)

    elif img_rgb.ndim == 3 and img_rgb.shape[2] >= 3:
        gray = cv2.cvtColor(
            img_rgb[:, :, :3],
            cv2.COLOR_RGB2GRAY,
        ).astype(np.float32)

    else:
        raise ValueError(
            f"Unsupported image shape for FFT: {img_rgb.shape}"
        )

    f = np.fft.fft2(gray)
    fsh = np.fft.fftshift(f)
    mag = np.abs(fsh)

    h, w = gray.shape
    cy, cx = h // 2, w // 2

    y, x = np.ogrid[
        -cy : h - cy,
        -cx : w - cx,
    ]

    low_mask = (x**2 + y**2) <= 30**2

    total_energy = float(
        np.sum(mag**2)
    )

    low_energy = float(
        np.sum((mag * low_mask) ** 2)
    )

    high_energy = total_energy - low_energy

    ratio = (
        high_energy / low_energy
        if low_energy > 0
        else 0.0
    )

    log_mag = np.log1p(mag).astype(np.float32)

    min_v = float(np.min(log_mag))
    max_v = float(np.max(log_mag))

    if max_v > min_v:
        log_mag_norm = (
            (log_mag - min_v)
            * (255.0 / (max_v - min_v))
        ).astype(np.uint8)

    else:
        log_mag_norm = np.zeros_like(
            log_mag,
            dtype=np.uint8,
        )

    colored = cv2.applyColorMap(
        log_mag_norm,
        cv2.COLORMAP_PLASMA,
    )

    phase = np.angle(fsh)
    phase_norm = ((phase + np.pi) * (255.0 / (2.0 * np.pi))).astype(np.uint8)

    try:
        phase_colored = cv2.applyColorMap(
            phase_norm,
            cv2.COLORMAP_TWILIGHT,
        )
    except Exception:
        phase_colored = cv2.applyColorMap(
            phase_norm,
            cv2.COLORMAP_HSV,
        )

    return {
        "total_energy": total_energy,
        "low_energy": low_energy,
        "high_energy": high_energy,
        "ratio": ratio,
        "spectrum_img": colored,
        "phase_img": phase_colored,
    }


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/api/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "backend": "fastapi",
        "pipeline": {
            "upload": "ready",
            "preprocessing": "ready",
            "face_detection": "ready",
            "landmarks": "ready",
            "photo_pipeline": "ready",
            "effect_engine": "ready",
            "expression_warp": "ready",
            "aging": "ready",
            "fft_analysis": "ready",
            "metrics": "ready",
        },
    }

@app.get("/api/models/preload/status")
def models_preload_status() -> dict[str, Any]:
    try:
        from backend.local_models.preloader import get_preload_status

        return get_preload_status()

    except Exception as exc:
        return {
            "enabled": False,
            "ok": False,
            "error": repr(exc),
        }
    

@app.get("/api/models/plugins/validate")
def validate_model_plugins() -> dict[str, Any]:
    return validate_all_plugins()

@app.get("/api/models/deca/diagnostics")
def deca_diagnostics() -> dict[str, Any]:
    return get_deca_diagnostics()

@app.post("/api/models/deca/prepare")
def deca_prepare() -> dict[str, Any]:
    return prepare_deca_repo_data()

@app.get("/api/models/deca/status")
def deca_runtime_status() -> dict[str, Any]:
    return get_deca_runtime_status()

@app.get("/api/models/slots")
def models_slots() -> dict[str, Any]:
    return get_model_slot_status()

@app.get("/api/models/requirements")
def models_requirements() -> dict[str, Any]:
    return get_model_requirements()

@app.get("/api/models/status")
def models_status() -> dict[str, Any]:
    return get_local_model_status()



@app.get("/api/system/status")
def system_status() -> dict[str, Any]:
    sam_status = get_sam_aging_status()
    liveportrait_status = get_liveportrait_expression_status()

    return {
        "local_only": True,
        "cloud_or_paid_api_required": False,
        "face_detection": _face_detection_status(),
        "face_parsing": get_face_parsing_status(),
        "deca": get_deca_runtime_status(),
        "sam": sam_status,
        "liveportrait": liveportrait_status,
        "generative_refiner": get_generative_refiner_status(),
        "expression": get_ai_expression_status(),
        "plugin_validator": validate_all_plugins(),
        "notes": {
            "anchors": "MediaPipe is the primary 2D anchor source for accessories.",
            "masks": "BiSeNet face parsing provides semantic masks when repo and weights are available; otherwise mask-dependent effects fall back to empty masks/no-op behavior.",
            "deca": "DECA/FLAME is the true 3D mesh provider when local files and dependencies are available. Renderer remains disabled.",
            "sam": "Available files exist if repo and weight checks pass; inference bridge not implemented yet.",
            "liveportrait": "Available files exist if repo and weight checks pass; inference bridge not implemented yet.",
        },
    }


@app.get("/api/debug/3d/{session_id}/{image_id}")
def debug_3d_context(
    session_id: str,
    image_id: str,
) -> Any:
    """
    Debug endpoint for the new 3D-ready face context.

    This does not modify the image.
    It loads the uploaded image, preprocesses it, analyzes face context,
    enriches it with the best available 3D provider, and returns a compact
    JSON-safe summary.
    """

    path = _find_stored_image(
        session_id,
        image_id,
    )

    if not path:
        return _err(
            "IMAGE_NOT_FOUND",
            "No image found.",
            status=404,
        )

    try:
        with open(path, "rb") as f:
            raw = f.read()

    except OSError as e:
        return _err(
            "PROCESSING_FAILED",
            f"Read error: {e}",
            status=500,
        )

    original_img = _decode_raw(
        raw,
    )

    if original_img is None:
        return _err(
            "PROCESSING_FAILED",
            "Decode failed.",
            status=500,
        )

    try:
        pipeline_out = preprocess_pipeline(
            original_img,
            target_size=512,
        )

        processed_rgb = pipeline_out["processed_image_uint8"]

        processed_bgr = cv2.cvtColor(
            processed_rgb,
            cv2.COLOR_RGB2BGR,
        )

        ctx = analyze_face(
            processed_bgr,
        )

        ctx = enrich_with_best_available_3d(
            ctx,
            processed_bgr,
        )

        proc_dir = os.path.join(
            PROCESSED_ROOT,
            session_id,
        )

        os.makedirs(
            proc_dir,
            exist_ok=True,
        )

        depth_name = f"{image_id}_debug_depth.png"
        depth_path = os.path.join(
            proc_dir,
            depth_name,
        )

        depth_rel = None

        depth_map = ctx.get("three_d", {}).get("depth_map")

        if isinstance(depth_map, np.ndarray):
            save_depth_visualization(
                depth_map,
                depth_path,
            )

            depth_rel = f"processed/{session_id}/{depth_name}"

        summary = summarize_face_context(
            ctx,
        )

        return {
            "success": True,
            "session_id": session_id,
            "image_id": image_id,
            "debug": summary,
            "paths": {
                "depth_debug_path": depth_rel,
            },
        }

    except Exception as e:
        import traceback

        traceback.print_exc()

        return _err(
            "PROCESSING_FAILED",
            f"3D debug failed: {e}",
            status=500,
        )

@app.get("/api/effects/catalog")
def effects_catalog() -> dict[str, Any]:
    return get_effect_catalog()


@app.get("/api/assets/manifest")
def assets_manifest() -> dict[str, Any]:
    return load_asset_manifest()


@app.get("/api/assets/palettes")
def assets_palettes() -> dict[str, Any]:
    return load_palettes()


@app.get("/api/assets/categories")
def assets_categories() -> dict[str, Any]:
    return {
        "categories": list_assets(),
        "counts": category_counts(),
    }


@app.get("/api/store/manifest")
def store_manifest() -> dict[str, Any]:
    return load_store_manifest()


@app.get("/api/store/items")
def store_items(
    slot: str | None = None,
    item_type: str | None = None,
) -> dict[str, Any]:
    return {
        "success": True,
        "items": list_store_items(
            slot=slot,
            item_type=item_type,
        ),
    }


@app.post("/api/store/outfit/resolve")
async def store_outfit_resolve(
    request: Request,
) -> Any:
    try:
        payload = await request.json()
    except Exception:
        payload = {}

    item_ids = payload.get("item_ids", [])
    if not isinstance(item_ids, list):
        return _err(
            "store_invalid_item_ids",
            "item_ids must be a list.",
            status=400,
        )

    try:
        return resolve_outfit_slots([str(item_id) for item_id in item_ids])
    except Exception as exc:
        return _err(
            "store_outfit_resolve_failed",
            str(exc),
            status=400,
        )


@app.get("/api/tryon/status")
def virtual_tryon_status() -> dict[str, Any]:
    return {
        "success": True,
        "tryon": get_virtual_tryon_status(),
    }


@app.get("/api/archive/status")
def archive_status() -> dict[str, Any]:
    """Return VITON-HD archive statistics and sample pairs."""
    stats = _archive_stats()
    sample_pairs = _archive_pairs()[:20] if _archive_exists() else []
    cloths = _archive_cloths(split="test", curated_only=True) if _archive_exists() else []
    return {
        "success": True,
        "archive": stats,
        "sample_pairs": [
            {"person": p, "cloth": c} for p, c in sample_pairs
        ],
        "curated_cloths": [
            {
                "stem": c["stem"],
                "path": c["relative_path"],
            }
            for c in cloths
        ],
    }


@app.post("/api/tryon/process")
async def virtual_tryon_process(
    session_id: str = Form(...),
    image_id: str = Form(...),
    cloth: UploadFile = File(...),
    model_type: str = Form(default="dc"),
    category: str = Form(default="upperbody"),
    sample: int = Form(default=1),
    steps: int = Form(default=20),
    scale: float = Form(default=2.0),
    seed: int = Form(default=-1),
) -> Any:
    person_path = _find_stored_image(session_id, image_id)
    if person_path is None:
        return _err(
            "tryon_person_not_found",
            "Uploaded person image was not found for this session.",
            status=404,
            details={"session_id": session_id, "image_id": image_id},
        )

    data = await cloth.read()
    if not data:
        return _err("tryon_empty_cloth", "Garment image is empty.", status=400)
    if len(data) > MAX_FILE_BYTES:
        return _err(
            "tryon_cloth_too_large",
            "Garment image exceeds the maximum file size.",
            status=413,
            details={"max_bytes": MAX_FILE_BYTES},
        )

    fmt = _detect_format(data)
    if fmt is None:
        return _err(
            "tryon_invalid_cloth",
            "Garment must be a PNG or JPEG image.",
            status=400,
        )

    cloth_ext = ".png" if fmt == "PNG" else ".jpg"
    cloth_dir = os.path.join(UPLOAD_ROOT, session_id, "garments")
    os.makedirs(cloth_dir, exist_ok=True)
    cloth_id = f"cloth_{uuid.uuid4()}"
    cloth_path = os.path.join(cloth_dir, f"{cloth_id}{cloth_ext}")
    with open(cloth_path, "wb") as f:
        f.write(data)

    proc_dir = os.path.join(PROCESSED_ROOT, session_id)
    os.makedirs(proc_dir, exist_ok=True)

    result = run_virtual_tryon(
        person_path=person_path,
        cloth_path=cloth_path,
        output_dir=proc_dir,
        model_type=model_type,
        category=category,
        sample=sample,
        steps=steps,
        scale=scale,
        seed=seed,
    )

    result["session_id"] = session_id
    result["image_id"] = image_id
    result["cloth_id"] = cloth_id
    result["cloth_path"] = f"uploads/{session_id}/garments/{os.path.basename(cloth_path)}"

    output_paths = []
    for output in result.get("outputs", []):
        path = output.get("path")
        if not path:
            continue
        rel = os.path.relpath(path, ROOT_DIR).replace("\\", "/")
        output["relative_path"] = rel
        output_paths.append(rel)

    result["output_paths"] = output_paths
    if result.get("mask_path"):
        result["mask_relative_path"] = os.path.relpath(
            result["mask_path"],
            ROOT_DIR,
        ).replace("\\", "/")

    if output_paths:
        first_path = os.path.join(ROOT_DIR, output_paths[0])
        try:
            with open(first_path, "rb") as f:
                result["result_image"] = b64encode(f.read()).decode("ascii")
        except Exception:
            result["result_image"] = None
    else:
        result["result_image"] = None

    return result


@app.post("/api/store/tryon")
async def store_virtual_tryon_process(
    session_id: str = Form(...),
    image_id: str = Form(...),
    item_id: str = Form(...),
    model_type: str | None = Form(default=None),
    category: str | None = Form(default=None),
    sample: int = Form(default=1),
    steps: int = Form(default=20),
    scale: float = Form(default=2.0),
    seed: int = Form(default=-1),
) -> Any:
    person_path = _find_stored_image(session_id, image_id)
    if person_path is None:
        return _err(
            "tryon_person_not_found",
            "Uploaded person image was not found for this session.",
            status=404,
            details={"session_id": session_id, "image_id": image_id},
        )

    try:
        item = resolve_store_item(item_id)
    except Exception as exc:
        return _err(
            "store_item_not_found",
            str(exc),
            status=404,
            details={"item_id": item_id},
        )

    if item.get("pipeline") != "virtual_tryon":
        return _err(
            "store_item_not_virtual_tryon",
            "Selected store item is not a virtual try-on garment.",
            status=400,
            details={
                "item_id": item_id,
                "pipeline": item.get("pipeline"),
                "slot": item.get("slot"),
            },
        )

    if not item.get("enabled", True):
        return _err(
            "store_item_disabled",
            "Selected store item is disabled or not fit-ready.",
            status=400,
            details={
                "item_id": item_id,
                "asset_quality": item.get("asset_quality", {}),
            },
        )

    cloth_path = item.get("absolute_tryon_path")
    if not cloth_path:
        return _err(
            "store_item_missing_tryon_image",
            "Selected store item has no valid try-on image.",
            status=400,
            details={"item_id": item_id},
        )

    proc_dir = os.path.join(PROCESSED_ROOT, session_id)
    os.makedirs(proc_dir, exist_ok=True)

    result = run_virtual_tryon(
        person_path=person_path,
        cloth_path=cloth_path,
        output_dir=proc_dir,
        model_type=model_type or str(item.get("model_type") or "dc"),
        category=category or str(item.get("tryon_category") or item.get("slot") or "upperbody"),
        sample=sample,
        steps=steps,
        scale=scale,
        seed=seed,
    )

    result["session_id"] = session_id
    result["image_id"] = image_id
    result["store_item"] = {
        "id": item.get("id"),
        "name": item.get("name"),
        "slot": item.get("slot"),
        "type": item.get("type"),
        "thumbnail": item.get("thumbnail"),
        "fit_profile": item.get("fit_profile", {}),
        "asset_quality": item.get("asset_quality", {}),
    }

    output_paths = []
    for output in result.get("outputs", []):
        path = output.get("path")
        if not path:
            continue
        rel = os.path.relpath(path, ROOT_DIR).replace("\\", "/")
        output["relative_path"] = rel
        output_paths.append(rel)

    result["output_paths"] = output_paths
    if result.get("mask_path"):
        result["mask_relative_path"] = os.path.relpath(
            result["mask_path"],
            ROOT_DIR,
        ).replace("\\", "/")

    if output_paths:
        first_path = os.path.join(ROOT_DIR, output_paths[0])
        try:
            with open(first_path, "rb") as f:
                result["result_image"] = b64encode(f.read()).decode("ascii")
        except Exception:
            result["result_image"] = None
    else:
        result["result_image"] = None

    return result

@app.post("/api/realtime/start")
def realtime_start() -> dict[str, Any]:
    """
    Start an in-memory realtime session.

    Frontend should call this once before sending webcam frames.
    """

    return {
        "success": True,
        **realtime_sessions.create_session(),
    }


@app.get("/api/realtime/sessions")
def realtime_list_sessions() -> dict[str, Any]:
    """
    Debug endpoint for active realtime sessions.
    """

    return {
        "success": True,
        "sessions": realtime_sessions.list_sessions(),
    }


@app.post("/api/realtime/reset/{session_id}")
def realtime_reset_session(
    session_id: str,
) -> dict[str, Any]:
    ok = realtime_sessions.reset(
        session_id,
    )

    return {
        "success": ok,
        "session_id": session_id,
    }


@app.delete("/api/realtime/{session_id}")
def realtime_delete_session(
    session_id: str,
) -> dict[str, Any]:
    ok = realtime_sessions.delete(
        session_id,
    )

    return {
        "success": ok,
        "session_id": session_id,
    }


def _hex_to_bgr_tuple(hex_color: str, fallback: tuple[int, int, int] = (128, 64, 128)) -> tuple[int, int, int]:
    value = str(hex_color or "").strip().replace("#", "")
    if len(value) != 6:
        return fallback
    try:
        r = int(value[0:2], 16)
        g = int(value[2:4], 16)
        b = int(value[4:6], 16)
        return b, g, r
    except ValueError:
        return fallback


def _refine_realtime_hair_mask(mask: np.ndarray | None, ctx: dict, image_shape: tuple[int, ...]) -> np.ndarray:
    h, w = image_shape[:2]
    if mask is None:
        return np.zeros((h, w), dtype=np.uint8)

    if mask.shape[:2] != (h, w):
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

    hair = (mask > 20).astype(np.uint8) * 255
    points = ctx.get("landmarks_2d")
    if not isinstance(points, np.ndarray) or points.shape[0] <= 454:
        return hair

    top = points[10].astype(np.float32)
    left = points[234].astype(np.float32)
    right = points[454].astype(np.float32)
    chin = points[152].astype(np.float32)
    face_width = max(1.0, float(np.linalg.norm(right - left)))

    roi = np.zeros((h, w), dtype=np.uint8)
    x1 = int(max(0, min(left[0], right[0]) - face_width * 0.38))
    x2 = int(min(w - 1, max(left[0], right[0]) + face_width * 0.38))
    y1 = int(max(0, top[1] - face_width * 0.65))
    y2 = int(min(h - 1, top[1] + face_width * 0.34))
    roi[y1 : y2 + 1, x1 : x2 + 1] = 255
    hair = cv2.bitwise_and(hair, roi)

    face_cut = np.zeros((h, w), dtype=np.uint8)
    center = (
        int((left[0] + right[0]) * 0.5),
        int(top[1] + face_width * 0.20),
    )
    axes = (
        int(face_width * 0.36),
        int(face_width * 0.20),
    )
    cv2.ellipse(face_cut, center, axes, 0, 0, 360, 255, -1, cv2.LINE_AA)
    hair = cv2.bitwise_and(hair, cv2.bitwise_not(face_cut))

    kernel = np.ones((3, 3), dtype=np.uint8)
    hair = cv2.morphologyEx(hair, cv2.MORPH_CLOSE, kernel, iterations=2)
    return hair


def _realtime_layer_anchor_header(ctx: dict, image_shape: tuple[int, ...]) -> str:
    h, w = image_shape[:2]
    points = ctx.get("landmarks_2d")
    anchor_ids = (33, 263, 152)
    payload: dict[str, Any] = {
        "width": int(w),
        "height": int(h),
        "points": {},
    }
    if isinstance(points, np.ndarray) and points.shape[0] > max(anchor_ids):
        payload["points"] = {
            str(idx): [
                float(points[idx][0]) / max(1.0, float(w - 1)),
                float(points[idx][1]) / max(1.0, float(h - 1)),
            ]
            for idx in anchor_ids
        }
    return json.dumps(payload, separators=(",", ":"))


@app.post("/api/realtime/landmarks")
async def realtime_detect_landmarks(
    file: UploadFile = File(...),
) -> Any:
    """
    Fast webcam tracker endpoint.

    This intentionally returns only MediaPipe landmarks. Camera rendering for
    makeup/accessories happens in the browser so the video can stay near the
    camera's native frame rate.
    """

    try:
        started_at = time.perf_counter()
        raw = await file.read()

        if not raw:
            return _err(
                "INVALID_FILE_TYPE",
                "Empty frame body.",
                status=400,
            )

        frame = _decode_raw(raw)

        if frame is None:
            return _err(
                "CORRUPTED_IMAGE",
                "Frame could not be decoded.",
                status=400,
            )

        if frame.ndim == 2:
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif frame.ndim == 3 and frame.shape[2] == 4:
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        elif frame.ndim == 3 and frame.shape[2] == 3:
            frame_bgr = frame
        else:
            return _err(
                "INVALID_FRAME",
                f"Unsupported frame shape: {frame.shape}",
                status=400,
            )

        h, w = frame_bgr.shape[:2]
        landmarks_data = detect_face_landmarks(frame_bgr)

        if landmarks_data is None:
            return {
                "success": False,
                "landmarks": [],
                "width": w,
                "height": h,
                "processing_ms": int((time.perf_counter() - started_at) * 1000),
                "error": "No face detected",
            }

        landmarks_2d, _landmarks_3d = landmarks_data
        normalized = [
            [
                float(point[0]) / max(1.0, float(w - 1)),
                float(point[1]) / max(1.0, float(h - 1)),
            ]
            for point in landmarks_2d
        ]

        return {
            "success": True,
            "landmarks": normalized,
            "width": w,
            "height": h,
            "processing_ms": int((time.perf_counter() - started_at) * 1000),
        }

    except Exception as e:
        import traceback

        traceback.print_exc()

        return _err(
            "PROCESSING_FAILED",
            f"Realtime landmark detection failed: {e}",
            status=500,
        )


@app.post("/api/realtime/effect-layer")
async def realtime_effect_layer(
    file: UploadFile = File(...),
    params_json: str = Form(default="{}"),
) -> Any:
    """
    Return a transparent PNG layer for effects that need real backend masks.

    Hair color uses BiSeNet hair parsing and eye color uses the existing iris
    effect. The browser composites this layer over live video, keeping camera
    FPS independent from heavy mask refresh rate.
    """

    try:
        started_at = time.perf_counter()
        raw = await file.read()

        if not raw:
            return _err(
                "INVALID_FILE_TYPE",
                "Empty frame body.",
                status=400,
            )

        frame = _decode_raw(raw)

        if frame is None:
            return _err(
                "CORRUPTED_IMAGE",
                "Frame could not be decoded.",
                status=400,
            )

        if frame.ndim == 2:
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif frame.ndim == 3 and frame.shape[2] == 4:
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        elif frame.ndim == 3 and frame.shape[2] == 3:
            frame_bgr = frame
        else:
            return _err(
                "INVALID_FRAME",
                f"Unsupported frame shape: {frame.shape}",
                status=400,
            )

        try:
            params = json.loads(params_json or "{}")
            if not isinstance(params, dict):
                params = {}
        except json.JSONDecodeError:
            return _err(
                "INVALID_PARAMS",
                "params_json must be valid JSON object.",
                status=400,
            )

        layer_params = {
            "hair_color": params.get("hair_color", {"enabled": False}),
            "eye_color": params.get("eye_color", {"enabled": False}),
        }

        if not (
            isinstance(layer_params["hair_color"], dict)
            and layer_params["hair_color"].get("enabled")
        ) and not (
            isinstance(layer_params["eye_color"], dict)
            and layer_params["eye_color"].get("enabled")
        ):
            empty = np.zeros((frame_bgr.shape[0], frame_bgr.shape[1], 4), dtype=np.uint8)
            ok, buf = cv2.imencode(".png", empty)
            if not ok:
                raise RuntimeError("Could not encode empty effect layer.")
            return Response(
                content=buf.tobytes(),
                media_type="image/png",
                headers={
                    "X-Realtime-Processing-Ms": str(int((time.perf_counter() - started_at) * 1000)),
                    "X-Realtime-Layer-Anchors": json.dumps({"width": int(frame_bgr.shape[1]), "height": int(frame_bgr.shape[0]), "points": {}}, separators=(",", ":")),
                },
            )

        h, w = frame_bgr.shape[:2]
        result_bgr = frame_bgr.copy()
        ctx = analyze_face(frame_bgr)
        alpha = np.zeros((h, w), dtype=np.uint8)

        if (
            isinstance(layer_params["hair_color"], dict)
            and layer_params["hair_color"].get("enabled")
        ):
            hair_mask = _refine_realtime_hair_mask(
                (ctx.get("masks") or {}).get("hair"),
                ctx,
                frame_bgr.shape,
            )
            if np.count_nonzero(hair_mask > 20) > 0:
                result_bgr = apply_faceapp_hair_color(
                    result_bgr,
                    hair_mask,
                    color=layer_params["hair_color"].get("color", "#6F3BB8"),
                    intensity=float(layer_params["hair_color"].get("intensity", 0.65)),
                )
                alpha = np.maximum(alpha, (hair_mask > 20).astype(np.uint8) * 255)

        if (
            isinstance(layer_params["eye_color"], dict)
            and layer_params["eye_color"].get("enabled")
        ):
            result_bgr = apply_eye_color_effect(
                result_bgr,
                ctx,
                layer_params["eye_color"],
            )
            points = ctx.get("landmarks_2d")
            if isinstance(points, np.ndarray) and points.shape[0] >= 478:
                eye_mask = np.zeros((h, w), dtype=np.uint8)
                for indices in ((468, 469, 470, 471, 472), (473, 474, 475, 476, 477)):
                    iris = points[list(indices)].astype(np.float32)
                    center = np.mean(iris, axis=0)
                    radius = max(2, int(np.max(np.linalg.norm(iris - center, axis=1)) * 1.15))
                    cv2.circle(
                        eye_mask,
                        (int(center[0]), int(center[1])),
                        radius,
                        255,
                        -1,
                        cv2.LINE_AA,
                    )
                alpha = np.maximum(alpha, eye_mask)

        alpha = cv2.GaussianBlur(alpha, (7, 7), 0)

        rgba = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGBA)
        rgba[:, :, 3] = alpha

        ok, buf = cv2.imencode(".png", rgba)
        if not ok:
            return _err(
                "PROCESSING_FAILED",
                "Could not encode realtime effect layer.",
                status=500,
            )

        return Response(
            content=buf.tobytes(),
            media_type="image/png",
            headers={
                "X-Realtime-Processing-Ms": str(int((time.perf_counter() - started_at) * 1000)),
                "X-Realtime-Layer-Anchors": _realtime_layer_anchor_header(ctx, frame_bgr.shape),
            },
        )

    except Exception as e:
        import traceback

        traceback.print_exc()

        return _err(
            "PROCESSING_FAILED",
            f"Realtime effect layer failed: {e}",
            status=500,
        )


@app.post("/api/realtime/frame")
async def realtime_process_frame(
    file: UploadFile = File(...),
    session_id: str | None = Form(default=None),
    params_json: str = Form(default="{}"),
    response_format: str = Form(default="json"),
) -> Any:
    """
    Process one webcam frame.

    Expected multipart form:
        file: image frame, jpg/png
        session_id: optional realtime session id
        params_json: JSON string with effect params

    Returns:
        result_image: base64 PNG, or binary JPEG when response_format=image/jpeg
        session_id: realtime session id
        frame_index
        used_full_pipeline
    """

    try:
        started_at = time.perf_counter()
        raw = await file.read()

        if not raw:
            return _err(
                "INVALID_FILE_TYPE",
                "Empty frame body.",
            )

        frame = _decode_raw(
            raw,
        )

        if frame is None:
            return _err(
                "CORRUPTED_IMAGE",
                "Frame could not be decoded.",
                status=400,
            )

        # cv2.imdecode returns BGR/BGRA/gray depending on input.
        if frame.ndim == 2:
            frame_bgr = cv2.cvtColor(
                frame,
                cv2.COLOR_GRAY2BGR,
            )

        elif frame.ndim == 3 and frame.shape[2] == 4:
            frame_bgr = cv2.cvtColor(
                frame,
                cv2.COLOR_BGRA2BGR,
            )

        elif frame.ndim == 3 and frame.shape[2] == 3:
            frame_bgr = frame

        else:
            return _err(
                "INVALID_FRAME",
                f"Unsupported frame shape: {frame.shape}",
                status=400,
            )

        try:
            params = json.loads(
                params_json or "{}",
            )

            if not isinstance(params, dict):
                params = {}

        except json.JSONDecodeError:
            return _err(
                "INVALID_PARAMS",
                "params_json must be valid JSON object.",
                status=400,
            )

        rt_session_id, processor = realtime_sessions.get_or_create(
            session_id,
        )

        output = processor.process_frame(
            frame_bgr,
            params,
        )

        result_bgr = output.get("result_bgr")

        if result_bgr is None:
            return _err(
                "PROCESSING_FAILED",
                output.get("error") or "Realtime processing failed.",
                status=500,
            )

        wants_binary = str(response_format or "").lower() in {
            "binary",
            "image",
            "jpg",
            "jpeg",
            "image/jpeg",
        }

        if wants_binary:
            jpeg_quality = int(os.getenv("FACEWARP_REALTIME_JPEG_QUALITY", "75"))
            jpeg_quality = max(55, min(90, jpeg_quality))

            ok, buf = cv2.imencode(
                ".jpg",
                result_bgr,
                [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality],
            )

            if not ok:
                return _err(
                    "PROCESSING_FAILED",
                    "Could not encode realtime output frame.",
                    status=500,
                )

            ctx = output.get("ctx") or {}
            realtime_debug = ctx.get("realtime_debug", {}) if isinstance(ctx, dict) else {}
            hair_debug = realtime_debug.get("hair_color", {}) if isinstance(realtime_debug, dict) else {}
            headers = {
                "X-Realtime-Session-Id": rt_session_id,
                "X-Realtime-Frame-Index": str(output.get("frame_index")),
                "X-Realtime-Used-Full-Pipeline": "1" if output.get("used_full_pipeline") else "0",
                "X-Realtime-Success": "1" if output.get("success") else "0",
                "X-Realtime-Processing-Ms": str(int((time.perf_counter() - started_at) * 1000)),
                "X-Realtime-Output-Bytes": str(int(len(buf))),
            }
            if hair_debug:
                headers["X-Realtime-Hair-Debug"] = json.dumps(
                    {
                        "mode": hair_debug.get("mode"),
                        "klt_ms": hair_debug.get("klt_ms"),
                        "hls_ms": hair_debug.get("hls_ms"),
                        "bisenet_request_count": hair_debug.get("bisenet_request_count"),
                        "bisenet_completed_count": hair_debug.get("bisenet_completed_count"),
                        "bisenet_refresh_interval_frames": hair_debug.get("bisenet_refresh_interval_frames"),
                    },
                    separators=(",", ":"),
                )
            if output.get("error"):
                headers["X-Realtime-Error"] = str(output.get("error"))

            return Response(
                content=buf.tobytes(),
                media_type="image/jpeg",
                headers=headers,
            )

        ok, buf = cv2.imencode(
            ".png",
            result_bgr,
        )

        if not ok:
            return _err(
                "PROCESSING_FAILED",
                "Could not encode realtime output frame.",
                status=500,
            )

        result_b64 = b64encode(
            buf.tobytes(),
        ).decode("utf-8")

        ctx = output.get("ctx") or {}
        three_d = ctx.get("three_d", {}) if isinstance(ctx, dict) else {}

        return {
            "success": bool(output.get("success")),
            "session_id": rt_session_id,
            "frame_index": output.get("frame_index"),
            "used_full_pipeline": output.get("used_full_pipeline"),
            "error": output.get("error"),
            "result_image": result_b64,
            "debug": {
                "has_ctx": bool(ctx),
                "three_d_provider": three_d.get("provider"),
                "is_true_3d": bool(three_d.get("is_true_3d", False)),
            },
        }

    except Exception as e:
        import traceback

        traceback.print_exc()

        return _err(
            "PROCESSING_FAILED",
            f"Realtime frame processing failed: {e}",
            status=500,
        )

@app.post("/api/upload")
async def upload(
    request: Request,
    file: UploadFile = File(...),
) -> Any:
    try:
        if not file.filename:
            return _err(
                "INVALID_FILE_TYPE",
                "Empty filename.",
            )

        ext = _file_extension(
            file.filename,
        )

        if ext not in ALLOWED_EXTENSIONS:
            return _err(
                "INVALID_FILE_TYPE",
                f"File extension '{ext}' is not allowed.",
                details={
                    "allowed_extensions": sorted(ALLOWED_EXTENSIONS),
                },
            )

        content_length = request.headers.get(
            "content-length",
        )

        if content_length is not None:
            try:
                if int(content_length) > MAX_FILE_BYTES:
                    return _err(
                        "FILE_TOO_LARGE",
                        f"File exceeds maximum size of {MAX_FILE_BYTES // (1024 * 1024)} MB.",
                        status=413,
                    )

            except ValueError:
                pass

        data = await file.read()

        if not data:
            return _err(
                "INVALID_FILE_TYPE",
                "Empty file body.",
            )

        if len(data) > MAX_FILE_BYTES:
            return _err(
                "FILE_TOO_LARGE",
                f"File exceeds maximum size of {MAX_FILE_BYTES // (1024 * 1024)} MB.",
                status=413,
            )

        content_type = file.content_type or ""

        if content_type not in ALLOWED_MIME:
            return _err(
                "INVALID_FILE_TYPE",
                "Only image/jpeg and image/png are allowed.",
                details={
                    "received_mime": content_type,
                    "allowed_mime": sorted(ALLOWED_MIME),
                },
            )

        detected_format = _detect_format(
            data,
        )

        if detected_format is None:
            return _err(
                "INVALID_FILE_TYPE",
                "File content does not match any supported image format.",
            )

        mime_to_format = {
            "image/jpeg": "JPEG",
            "image/jpg": "JPEG",
            "image/png": "PNG",
        }

        if mime_to_format.get(content_type) != detected_format:
            return _err(
                "INVALID_FILE_TYPE",
                "Declared MIME type does not match actual file content.",
                details={
                    "declared_mime": content_type,
                    "detected_format": detected_format,
                },
            )

        img = _decode_raw(
            data,
        )

        if img is None:
            return _err(
                "CORRUPTED_IMAGE",
                "Uploaded file could not be decoded as a valid image.",
                details={
                    "allowed_formats": [
                        "image/jpeg",
                        "image/png",
                    ],
                },
            )

        h, w = img.shape[:2]

        if w < MIN_WIDTH or h < MIN_HEIGHT:
            return _err(
                "IMAGE_TOO_SMALL",
                f"Image must be at least {MIN_WIDTH}x{MIN_HEIGHT} px "
                f"(got {w}x{h}).",
                details={
                    "min_width": MIN_WIDTH,
                    "min_height": MIN_HEIGHT,
                    "actual_width": w,
                    "actual_height": h,
                },
            )

        channels = (
            1
            if img.ndim == 2
            else int(img.shape[2])
        )

        session_id = f"ses_{uuid.uuid4()}"
        image_id = f"img_{uuid.uuid4()}"

        save_ext = (
            "jpg"
            if detected_format == "JPEG"
            else "png"
        )

        dest_dir = os.path.join(
            UPLOAD_ROOT,
            session_id,
        )

        os.makedirs(
            dest_dir,
            exist_ok=True,
        )

        dest_name = f"{image_id}.{save_ext}"
        dest_path = os.path.join(
            dest_dir,
            dest_name,
        )

        with open(dest_path, "wb") as f:
            f.write(data)

        relative_path = f"uploads/{session_id}/{dest_name}"

        resp = UploadResponse(
            image_id=image_id,
            session_id=session_id,
            status="uploaded",
            original=OriginalInfo(
                filename=file.filename,
                content_type=content_type,
                width=w,
                height=h,
                channels=channels,
                format=detected_format,
                size_bytes=len(data),
            ),
            paths=PathsInfo(
                original_path=relative_path,
            ),
            message="Image uploaded successfully.",
        )

        return resp.model_dump()

    except Exception as exc:
        import traceback

        traceback.print_exc()

        return _err(
            "UPLOAD_FAILED",
            f"An unexpected error occurred during upload: {exc}",
            status=500,
        )


@app.post("/api/process")
def process_image(
    body: ProcessRequest,
) -> Any:
    try:
        return _process_image_inner(
            body,
        )

    except Exception as exc:
        import traceback

        traceback.print_exc()

        return _err(
            "PROCESSING_FAILED",
            str(exc),
            status=500,
        )


def _process_image_inner(
    body: ProcessRequest,
) -> Any:
    request_params = _merge_process_effect_params(body)
    raw_effects = body.effects or (body.params or {}).get("effects", {})
    print(
        "[DEBUG] /api/process:",
        {
            "mode": body.mode,
            "params_keys": sorted((body.params or {}).keys()),
            "effects_keys": sorted(raw_effects.keys()) if isinstance(raw_effects, dict) else [],
        },
    )
    if raw_effects:
        print("[DEBUG] effects:", raw_effects)

    path = _find_stored_image(
        body.session_id,
        body.image_id,
    )

    if not path:
        return _err(
            "IMAGE_NOT_FOUND",
            "No image found.",
            status=404,
        )

    try:
        with open(path, "rb") as f:
            raw = f.read()

    except OSError as e:
        return _err(
            "PROCESSING_FAILED",
            f"Read error: {e}",
            status=500,
        )

    original_img = _decode_raw(
        raw,
    )

    if original_img is None:
        return _err(
            "PROCESSING_FAILED",
            "Decode failed.",
            status=500,
        )

    target_size = body.options.target_size

    try:
        pipeline_out = preprocess_pipeline(
            original_img,
            target_size=target_size,
        )

    except Exception as e:
        return _err(
            "PROCESSING_FAILED",
            f"{e}",
            status=500,
        )

    processed_uint8 = pipeline_out["processed_image_uint8"]
    grayscale = pipeline_out["grayscale_image"]
    preprocess_info = pipeline_out["preprocess_info"]

    # Face validation gate.
    # If this is too strict for side-face photos, loosen face_validation.py later.
    orientation = check_face_orientation(
        processed_uint8,
    )

    if not orientation.ok:
        return _err(
            orientation.error_code,
            orientation.message,
            status=422,
            details={
                "yaw_deg": orientation.yaw_deg,
                "pitch_deg": orientation.pitch_deg,
            },
        )

    proc_dir = os.path.join(
        PROCESSED_ROOT,
        body.session_id,
    )

    os.makedirs(
        proc_dir,
        exist_ok=True,
    )

    # ── Save preprocessed RGB as PNG ──────────────────────────────────────────

    preprocessed_name = f"{body.image_id}_preprocessed.png"
    preprocessed_path = os.path.join(
        proc_dir,
        preprocessed_name,
    )

    try:
        bgr = image_uint8_rgb_to_bgr(
            processed_uint8,
        )

        ok, buf = cv2.imencode(
            ".png",
            bgr,
        )

        if not ok:
            raise RuntimeError("encode failed")

        with open(preprocessed_path, "wb") as f:
            f.write(buf.tobytes())

    except Exception as e:
        return _err(
            "PROCESSING_FAILED",
            f"Save RGB failed: {e}",
            status=500,
        )

    # ── Save grayscale ────────────────────────────────────────────────────────

    grayscale_name = f"{body.image_id}_grayscale.png"
    grayscale_path = os.path.join(
        proc_dir,
        grayscale_name,
    )

    try:
        if grayscale.ndim == 3:
            gray_2d = grayscale[:, :, 0]
        else:
            gray_2d = grayscale

        ok, buf = cv2.imencode(
            ".png",
            gray_2d,
        )

        if not ok:
            raise RuntimeError("encode grayscale failed")

        with open(grayscale_path, "wb") as f:
            f.write(buf.tobytes())

    except Exception as e:
        return _err(
            "PROCESSING_FAILED",
            f"Save grayscale failed: {e}",
            status=500,
        )

    # ── Metadata ──────────────────────────────────────────────────────────────

    meta_dict = build_preprocess_metadata(
        original_img,
        processed_uint8,
        channels=3,
    )

    metadata_name = f"{body.image_id}_metadata.json"
    metadata_path = os.path.join(
        proc_dir,
        metadata_name,
    )

    with open(
        metadata_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            {
                "image_id": body.image_id,
                "session_id": body.session_id,
                "preprocess_info": preprocess_info,
                "metadata": meta_dict,
            },
            f,
            indent=2,
        )

    # ── Face detection / landmarks for response compatibility ────────────────

    detector = get_face_detector()

    face_result = detector.detect(
        processed_uint8,
    )

    landmark_result = detector.detect_landmarks(
        processed_uint8,
    )

    # ── Processing router ─────────────────────────────────────────────────────
    #
    # processed_uint8 is RGB.
    # photo_pipeline expects BGR.
    # expression/aging legacy functions still expect RGB.
    # ─────────────────────────────────────────────────────────────────────────

    mode = body.mode or "expression"
    warped_uint8 = processed_uint8.copy()
    ai_expression_meta: dict[str, Any] | None = None
    effects_meta: list[dict[str, Any]] | None = None
    aging_debug: dict[str, Any] | None = None

    try:
        if body.effects and body.mode is None:
            mode = "photo"

        if mode == "ai_accessory_makeup":
            mode = "accessory"

        # New architecture:
        # photo / beauty / accessory all go through photo_pipeline/effect_engine.
        if mode in ("photo", "beauty", "accessory"):
            photo_params = _build_legacy_photo_params(
                mode,
                request_params,
            )

            input_bgr = cv2.cvtColor(
                processed_uint8,
                cv2.COLOR_RGB2BGR,
            )

            output_bgr, _face_ctx = apply_photo_pipeline(
                input_bgr,
                photo_params,
            )

            effects_meta = _face_ctx.get("effects_meta", [])

            warped_uint8 = cv2.cvtColor(
                output_bgr,
                cv2.COLOR_BGR2RGB,
            )

        elif mode == "expression":
            landmark_points = (
                landmark_result.get("points")
                if landmark_result.get("status") == "completed"
                else None
            )

            if landmark_points:
                expression_preset = str(
                    request_params.get("expression_preset", "")
                ).lower()
                smile_value = float(
                    np.clip(
                        request_params.get("smile_intensity", 0.0),
                        0.0,
                        0.65,
                    )
                )
                eyebrow_value = float(
                    np.clip(
                        request_params.get("eyebrow_height", 0.0),
                        -0.35,
                        0.35,
                    )
                )
                lip_value = float(
                    np.clip(
                        request_params.get(
                            "lip_intensity",
                            request_params.get("lip_widening", 0.0),
                        ),
                        0.0,
                        1.0,
                    )
                )
                slim_value = float(
                    np.clip(
                        request_params.get("face_slimming", 0.0),
                        0.0,
                        0.55,
                    )
                )

                if expression_preset == "laugh":
                    # Classical warp cannot synthesize teeth or true open-mouth laugh.
                    # Keep this as a restrained smile variant to avoid mouth artifacts.
                    smile_value = min(max(smile_value, 0.35), 0.55)
                    eyebrow_value = max(eyebrow_value, 0.08)
                    lip_value = min(max(lip_value, 0.10), 0.18)

                warped_uint8 = apply_expression_transform(
                    processed_uint8,
                    landmark_points,
                    smile_intensity=smile_value,
                    eyebrow_intensity=eyebrow_value,
                    lip_intensity=lip_value,
                    slim_intensity=slim_value,
                )

        elif mode == "ai_expression":
            input_bgr = cv2.cvtColor(
                processed_uint8,
                cv2.COLOR_RGB2BGR,
            )

            try:
                ai_ctx = analyze_face(
                    input_bgr,
                )

                ai_ctx = enrich_with_best_available_3d(
                    ai_ctx,
                    input_bgr,
                )

            except Exception as e:
                ai_ctx = {}

                landmark_points = (
                    landmark_result.get("points")
                    if landmark_result.get("status") == "completed"
                    else []
                )

                if landmark_points:
                    ai_ctx["landmarks"] = np.asarray(
                        [
                            [
                                point.get("x", 0.0),
                                point.get("y", 0.0),
                                point.get("z", 0.0),
                            ]
                            for point in landmark_points
                        ],
                        dtype=np.float32,
                    )

                ai_ctx["analysis_error"] = str(e)

            output_bgr, ai_expression_meta = apply_ai_expression(
                input_bgr,
                ai_ctx,
                request_params,
            )

            warped_uint8 = cv2.cvtColor(
                output_bgr,
                cv2.COLOR_BGR2RGB,
            )

        elif mode == "aging":
            aging_algorithm_raw = request_params.get("aging_algorithm")
            if aging_algorithm_raw is None:
                aging_algorithm_raw = getattr(body, "aging_algorithm", "frequency")

            aging_algorithm = _normalize_aging_algorithm(aging_algorithm_raw)

            if aging_algorithm == "frequency":
                # Frequency-based aging: completely separate from AI aging target_age parameter.
                target_age = None
                raw_intensity_val = request_params.get(
                    "aging_intensity",
                    getattr(body, "aging_intensity", 1.0),
                )
                try:
                    aging_intensity_raw = float(raw_intensity_val)
                except (ValueError, TypeError):
                    aging_intensity_raw = 1.0
                aging_intensity_raw = float(np.clip(aging_intensity_raw, 0.0, 2.0))
            else:
                # AI-based aging: extract target_age (or map from intensity if it looks like an age)
                target_age = request_params.get("target_age")
                if target_age is None:
                    # Check if aging_intensity seems to be an age (e.g. > 2.0)
                    raw_intensity_val = request_params.get(
                        "aging_intensity",
                        getattr(body, "aging_intensity", 1.0),
                    )
                    try:
                        val = float(raw_intensity_val)
                        if val > 2.0:
                            target_age = val
                    except (ValueError, TypeError):
                        pass

                if target_age is not None:
                    try:
                        target_age = float(target_age)
                    except (ValueError, TypeError):
                        target_age = None

                t_age = target_age if target_age is not None else 60.0
                t_age = float(np.clip(t_age, 7.0, 85.0))
                # For fallback to frequency aging if SAM is not available/fails
                if t_age <= 35.0:
                    aging_intensity_raw = 0.2 + 0.8 * (t_age - 7.0) / 28.0
                else:
                    aging_intensity_raw = 1.0 + 1.0 * (t_age - 35.0) / 50.0

            landmark_points = (
                landmark_result.get("points")
                if landmark_result.get("status") == "completed"
                else None
            )

            input_bgr = cv2.cvtColor(
                processed_uint8,
                cv2.COLOR_RGB2BGR,
            )

            if aging_algorithm == "frequency":
                from backend.aging import apply_aging_simulation

                warped_uint8 = apply_aging_simulation(
                    processed_uint8,
                    aging_intensity_raw,
                    landmarks=landmark_points,
                    target_age=target_age,
                )

                effects_meta = []
                aging_debug = {
                    "aging_backend": "frequency_fft",
                    "aging_algorithm_requested": str(aging_algorithm_raw),
                    "normalized_algorithm": aging_algorithm,
                    "intensity": aging_intensity_raw,
                    "inference_ran": True,
                    "inference_bridge_implemented": True,
                    "fallback_used": False,
                    "plugin": "backend.aging.apply_aging_simulation",
                    "landmark_zone_aging": bool(landmark_points),
                }

            else:
                aging_model_params = _build_aging_model_params(
                    request_params,
                    body,
                )

                output_bgr, _face_ctx = apply_photo_pipeline(
                    input_bgr,
                    {
                        "aging_model": aging_model_params,
                    },
                )

                effects_meta = _face_ctx.get("effects_meta", [])
                aging_debug = (
                    _face_ctx.get("debug_plugins", {}).get("aging_model")
                    if isinstance(_face_ctx.get("debug_plugins"), dict)
                    else None
                )

                if not aging_debug:
                    sam_status = get_sam_aging_status()
                    slot_status = (
                        _face_ctx.get("model_slots", {}).get("aging_model", {})
                        if isinstance(_face_ctx.get("model_slots"), dict)
                        else {}
                    )
                    aging_debug = {
                        "aging_backend": "SAM",
                        "sam_available": bool(sam_status.get("runtime_available", False)),
                        "repo_ok": bool(sam_status.get("repo_ok", False)),
                        "weights_ok": bool(sam_status.get("weights_ok", False)),
                        "runtime_available": bool(sam_status.get("runtime_available", False)),
                        "inference_bridge_implemented": bool(
                            sam_status.get("inference_bridge_implemented", False)
                        ),
                        "inference_ran": False,
                        "target_age": aging_model_params.get("target_age"),
                        "intensity": aging_model_params.get("intensity"),
                        "slot_status": slot_status,
                        "fallback_used": False,
                        "aging_algorithm_requested": str(aging_algorithm_raw),
                        "normalized_algorithm": aging_algorithm,
                        "error": (
                            "SAM unavailable: "
                            f"repo_ok={sam_status.get('repo_ok')}; "
                            f"weights_ok={sam_status.get('weights_ok')}; "
                            f"missing_modules={sam_status.get('missing_modules')}"
                        ),
                    }

                warped_uint8 = cv2.cvtColor(
                    output_bgr,
                    cv2.COLOR_BGR2RGB,
                )

                sam_changed = int(aging_debug.get("changed_pixels") or 0) > 1000
                sam_ran = bool(aging_debug.get("inference_ran", False))

                if (
                    aging_model_params.get("fallback_to_legacy", True)
                    and abs(aging_intensity_raw - 1.0) >= 0.01
                    and (not sam_ran or not sam_changed)
                ):
                    from backend.aging import apply_aging_simulation

                    warped_uint8 = apply_aging_simulation(
                        processed_uint8,
                        aging_intensity_raw,
                        landmarks=landmark_points,
                        target_age=target_age,
                    )
                    aging_debug = {
                        **aging_debug,
                        "fallback_used": True,
                        "fallback": "legacy_aging_simulation",
                        "fallback_reason": (
                            "sam_not_run"
                            if not sam_ran
                            else "sam_output_had_no_visible_change"
                        ),
                        "aging_algorithm_requested": str(aging_algorithm_raw),
                        "normalized_algorithm": aging_algorithm,
                    }

        else:
            warped_uint8 = processed_uint8.copy()

    except Exception as e:
        import traceback

        traceback.print_exc()

        print(
            f"[WARN] Processing branch failed ({e}); "
            "falling back to preprocessed image."
        )

        warped_uint8 = processed_uint8.copy()

    # ── FFT analysis ──────────────────────────────────────────────────────────

    fft_orig_name = f"{body.image_id}_fft_orig.png"
    fft_proc_name = f"{body.image_id}_fft_proc.png"
    fft_phase_orig_name = f"{body.image_id}_fft_phase_orig.png"
    fft_phase_proc_name = f"{body.image_id}_fft_phase_proc.png"

    fft_orig_path = os.path.join(
        proc_dir,
        fft_orig_name,
    )

    fft_proc_path = os.path.join(
        proc_dir,
        fft_proc_name,
    )

    fft_phase_orig_path = os.path.join(
        proc_dir,
        fft_phase_orig_name,
    )

    fft_phase_proc_path = os.path.join(
        proc_dir,
        fft_phase_proc_name,
    )

    fft_orig_rel: str | None = None
    fft_proc_rel: str | None = None
    fft_phase_orig_rel: str | None = None
    fft_phase_proc_rel: str | None = None

    fft_fallback = {
        "total_energy": 0.0,
        "low_energy": 0.0,
        "high_energy": 0.0,
        "ratio": 0.0,
        "spectrum_img": np.zeros(
            (64, 64, 3),
            dtype=np.uint8,
        ),
        "phase_img": np.zeros(
            (64, 64, 3),
            dtype=np.uint8,
        ),
    }

    try:
        fft_orig_info = compute_fft_info(
            processed_uint8,
        )

        if cv2.imwrite(
            fft_orig_path,
            fft_orig_info["spectrum_img"],
        ):
            fft_orig_rel = f"processed/{body.session_id}/{fft_orig_name}"

        if cv2.imwrite(
            fft_phase_orig_path,
            fft_orig_info["phase_img"],
        ):
            fft_phase_orig_rel = f"processed/{body.session_id}/{fft_phase_orig_name}"

    except Exception as e:
        import traceback

        traceback.print_exc()
        print(f"[WARN] Original FFT failed: {e}")

        fft_orig_info = fft_fallback.copy()

    try:
        fft_proc_info = compute_fft_info(
            warped_uint8,
        )

        if cv2.imwrite(
            fft_proc_path,
            fft_proc_info["spectrum_img"],
        ):
            fft_proc_rel = f"processed/{body.session_id}/{fft_proc_name}"

        if cv2.imwrite(
            fft_phase_proc_path,
            fft_proc_info["phase_img"],
        ):
            fft_phase_proc_rel = f"processed/{body.session_id}/{fft_phase_proc_name}"

    except Exception as e:
        import traceback

        traceback.print_exc()
        print(f"[WARN] Processed FFT failed: {e}")

        fft_proc_info = fft_fallback.copy()


    # ── Metrics ───────────────────────────────────────────────────────────────

    try:
        orig_gray = cv2.cvtColor(
            processed_uint8,
            cv2.COLOR_RGB2GRAY,
        ).astype(float)

        warp_gray = cv2.cvtColor(
            warped_uint8,
            cv2.COLOR_RGB2GRAY,
        ).astype(float)

        mse_val = float(
            sk_mse(
                orig_gray,
                warp_gray,
            )
        )

        psnr_val = float(
            sk_psnr(
                orig_gray,
                warp_gray,
                data_range=255,
            )
        )

        ssim_val = float(
            sk_ssim(
                orig_gray,
                warp_gray,
                data_range=255,
            )
        )

    except Exception as e:
        print(f"[WARN] Metrics computation failed: {e}")

        mse_val = None
        psnr_val = None
        ssim_val = None

    changed_pixels = int(
        np.count_nonzero(
            np.any(processed_uint8 != warped_uint8, axis=2)
        )
    )

    mask_stats: dict[str, Any] = {}
    if effects_meta:
        for item in effects_meta:
            stats = item.get("mask_stats")
            if isinstance(stats, dict) and stats.get("mask"):
                mask_stats[str(stats["mask"])] = {
                    "pixels": stats.get("pixels", 0),
                    "coverage": stats.get("coverage", 0.0),
                }

    result_name = f"{body.image_id}_result_{uuid.uuid4().hex[:8]}.png"
    result_path = os.path.join(
        proc_dir,
        result_name,
    )
    result_rel: str | None = None

    try:
        if cv2.imwrite(
            result_path,
            cv2.cvtColor(
                warped_uint8,
                cv2.COLOR_RGB2BGR,
            ),
        ):
            result_rel = f"processed/{body.session_id}/{result_name}"

    except Exception as e:
        print(f"[WARN] Result image save failed: {e}")

    # ── Encode result image ───────────────────────────────────────────────────

    try:
        ok, buf = cv2.imencode(
            ".png",
            cv2.cvtColor(
                warped_uint8,
                cv2.COLOR_RGB2BGR,
            ),
        )

        if not ok:
            raise RuntimeError("cv2.imencode failed")

        warped_b64 = b64encode(
            buf.tobytes(),
        ).decode("utf-8")

    except Exception as e:
        return _err(
            "PROCESSING_FAILED",
            f"Encode warped image failed: {e}",
            status=500,
        )

    # ── Relative paths ────────────────────────────────────────────────────────

    original_rel = f"uploads/{body.session_id}/{os.path.basename(path)}"
    preprocessed_rel = f"processed/{body.session_id}/{preprocessed_name}"
    grayscale_rel = f"processed/{body.session_id}/{grayscale_name}"
    metadata_rel = f"processed/{body.session_id}/{metadata_name}"

    debug_payload: dict[str, Any] = {
        "mode": mode,
        "effects_applied": [
            item.get("effect")
            for item in (effects_meta or [])
            if item.get("applied")
        ],
        "mask_stats": mask_stats,
        "changed_pixels": changed_pixels,
        "mse": mse_val,
        "output_url": result_rel or "inline_result_image",
    }

    if aging_debug:
        aging_debug = {
            **aging_debug,
            "changed_pixels": int(aging_debug.get("changed_pixels") or changed_pixels),
            "mse": float(aging_debug.get("mse") or mse_val or 0.0),
            "output_url": result_rel or "inline_result_image",
        }
        debug_payload.update(
            {
                "aging_backend": aging_debug.get("aging_backend", "SAM"),
                "sam_available": aging_debug.get("sam_available"),
                "repo_ok": aging_debug.get("repo_ok"),
                "weights_ok": aging_debug.get("weights_ok"),
                "inference_ran": aging_debug.get("inference_ran"),
                "aging": aging_debug,
            }
        )

    # ── Response ──────────────────────────────────────────────────────────────

    resp = ProcessResponse(
        image_id=body.image_id,
        session_id=body.session_id,
        status="processed",
        result_image=warped_b64,
        metrics=MetricsInfo(
            mse=mse_val,
            psnr=psnr_val,
            ssim=ssim_val,
            energy_ratio_orig=fft_orig_info["ratio"],
            energy_ratio_proc=fft_proc_info["ratio"],
            total_energy_orig=fft_orig_info["total_energy"],
            total_energy_proc=fft_proc_info["total_energy"],
            high_energy_orig=fft_orig_info["high_energy"],
            high_energy_proc=fft_proc_info["high_energy"],
            low_energy_orig=fft_orig_info["low_energy"],
            low_energy_proc=fft_proc_info["low_energy"],
        ),
        pipeline=PipelineStatus(
            upload="completed",
            decode="completed",
            preprocess="completed",
            face_detection=face_result.get("status"),
            landmark_detection=landmark_result.get("status"),
        ),
        preprocess=PreprocessInfo(**preprocess_info),
        metadata=MetadataInfo(**meta_dict),
        paths=PathsInfo(
            original_path=original_rel,
            preprocessed_path=preprocessed_rel,
            result_path=result_rel,
            metadata_path=metadata_rel,
            grayscale_path=grayscale_rel,
            fft_orig_path=fft_orig_rel,
            fft_proc_path=fft_proc_rel,
            fft_phase_orig_path=fft_phase_orig_rel,
            fft_phase_proc_path=fft_phase_proc_rel,
        ),
        face_detection=FaceDetectionInfo(**face_result),
        landmark_detection=LandmarkDetectionInfo(**landmark_result),
        ai_expression=ai_expression_meta,
        effects_meta=effects_meta,
        debug=debug_payload,
    )

    return resp.model_dump()


@app.get("/api/gallery")
def get_gallery() -> Any:
    items = []

    if not os.path.exists(PROCESSED_ROOT):
        return {
            "success": True,
            "items": [],
        }

    for session_id in os.listdir(PROCESSED_ROOT):
        session_dir = os.path.join(
            PROCESSED_ROOT,
            session_id,
        )

        if not os.path.isdir(session_dir):
            continue

        for file in os.listdir(session_dir):
            if not file.endswith("_metadata.json"):
                continue

            meta_path = os.path.join(
                session_dir,
                file,
            )

            try:
                with open(
                    meta_path,
                    "r",
                    encoding="utf-8",
                ) as f:
                    data = json.load(f)

                image_id = data.get("image_id")

                if not image_id:
                    continue

                preprocessed_path = (
                    f"processed/{session_id}/{image_id}_preprocessed.png"
                )

                original_path = ""

                upload_dir = os.path.join(
                    UPLOAD_ROOT,
                    session_id,
                )

                if os.path.isdir(upload_dir):
                    for uf in os.listdir(upload_dir):
                        if uf.startswith(image_id):
                            original_path = f"uploads/{session_id}/{uf}"
                            break

                items.append(
                    {
                        "session_id": session_id,
                        "image_id": image_id,
                        "original_path": original_path,
                        "preprocessed_path": preprocessed_path,
                        "created_at": os.path.getmtime(meta_path),
                    }
                )

            except Exception:
                pass

    items.sort(
        key=lambda x: x["created_at"],
        reverse=True,
    )

    return {
        "success": True,
        "items": items,
    }


@app.delete("/api/gallery/{session_id}/{image_id}")
def delete_gallery_item(
    session_id: str,
    image_id: str,
) -> Any:
    deleted = []
    errors = []

    proc_dir = os.path.join(
        PROCESSED_ROOT,
        session_id,
    )

    if os.path.isdir(proc_dir):
        for file in os.listdir(proc_dir):
            if file.startswith(image_id):
                try:
                    os.remove(
                        os.path.join(
                            proc_dir,
                            file,
                        )
                    )
                    deleted.append(file)

                except Exception as e:
                    errors.append(str(e))

        if not os.listdir(proc_dir):
            os.rmdir(proc_dir)

    upload_dir = os.path.join(
        UPLOAD_ROOT,
        session_id,
    )

    if os.path.isdir(upload_dir):
        for file in os.listdir(upload_dir):
            if file.startswith(image_id):
                try:
                    os.remove(
                        os.path.join(
                            upload_dir,
                            file,
                        )
                    )
                    deleted.append(file)

                except Exception as e:
                    errors.append(str(e))

        if not os.listdir(upload_dir):
            os.rmdir(upload_dir)

    if not deleted:
        return _err(
            "NOT_FOUND",
            "No files found to delete.",
            status=404,
        )

    return {
        "success": True,
        "deleted": deleted,
        "errors": errors,
    }


# ── Static / frontend ────────────────────────────────────────────────────────

app.mount(
    "/static",
    StaticFiles(directory=STATIC_DIR),
    name="static",
)

app.mount(
    "/uploads",
    StaticFiles(directory=UPLOAD_ROOT),
    name="uploads",
)

app.mount(
    "/processed",
    StaticFiles(directory=PROCESSED_ROOT),
    name="processed",
)

app.mount(
    "/assets",
    StaticFiles(directory=str(ASSETS_DIR)),
    name="assets",
)

MODELS_DIR = os.path.join(ROOT_DIR, "models")
if os.path.isdir(MODELS_DIR):
    app.mount(
        "/models",
        StaticFiles(directory=MODELS_DIR),
        name="models",
    )


@app.get("/")
def serve_index() -> FileResponse:
    return FileResponse(
        os.path.join(
            TEMPLATES_DIR,
            "index.html",
        )
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=False,
    )
