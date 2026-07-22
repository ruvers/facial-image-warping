"""
FaceWarp Lab — Pydantic response / request schemas.

Standard JSON contract for all API endpoints.
No data-wrapper; responses are flat.
"""

from __future__ import annotations
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field


# ── Error ─────────────────────────────────────────────────────────────────────


class ErrorDetail(BaseModel):
    code: str
    message: str
    details: Optional[Dict[str, Any]] = None


class ErrorResponse(BaseModel):
    success: Literal[False] = False
    error: ErrorDetail


# ── Upload ────────────────────────────────────────────────────────────────────


class OriginalInfo(BaseModel):
    filename: str
    content_type: str
    width: int
    height: int
    channels: int
    format: str
    size_bytes: int


class PathsInfo(BaseModel):
    original_path: str
    preprocessed_path: Optional[str] = None
    result_path: Optional[str] = None
    metadata_path: Optional[str] = None
    grayscale_path: Optional[str] = None
    fft_orig_path: Optional[str] = None
    fft_proc_path: Optional[str] = None
    fft_phase_orig_path: Optional[str] = None
    fft_phase_proc_path: Optional[str] = None


class UploadResponse(BaseModel):
    success: Literal[True] = True
    image_id: str
    session_id: str
    status: str
    original: OriginalInfo
    paths: PathsInfo
    message: str


# ── Process Request ───────────────────────────────────────────────────────────


class ProcessOptions(BaseModel):
    target_size: int = 512
    normalize_rgb: bool = True
    grayscale: bool = False
    debug: bool = False


class ProcessRequest(BaseModel):
    session_id: str
    image_id: str
    mode: Optional[str] = None
    params: Dict[str, Any] = Field(default_factory=dict)
    effects: Dict[str, Any] = Field(default_factory=dict)
    options: ProcessOptions = Field(default_factory=ProcessOptions)
    aging_intensity: float = 1.0
    aging_algorithm: str = "frequency"

# ── Process Response ──────────────────────────────────────────────────────────


class PipelineStatus(BaseModel):
    upload: str
    decode: str
    preprocess: str
    face_detection: str
    landmark_detection: str


class PreprocessInfo(BaseModel):
    target_size: int
    resized_width: int
    resized_height: int
    color_space: str
    normalized: bool
    normalization_range: List[float]
    scale: float = 1.0
    pad_top: int = 0
    pad_bottom: int = 0
    pad_left: int = 0
    pad_right: int = 0
    letterbox: bool = False
    grayscale_generated: bool
    grayscale_dtype: Optional[str] = None
    histogram_equalized: Optional[bool] = None


class MetadataInfo(BaseModel):
    original_width: int
    original_height: int
    processed_width: int
    processed_height: int
    channels: int


class FaceDetectionInfo(BaseModel):
    enabled: bool
    status: str
    bbox: Optional[Dict[str, Any]] = None
    confidence: Optional[float] = None
    message: str

class LandmarkPoint(BaseModel):
    index: int
    x: float
    y: float
    z: float
    visibility: float

class LandmarkDetectionInfo(BaseModel):
    enabled: bool
    status: str
    count: int
    coordinate_space: str
    points: Optional[List[LandmarkPoint]] = None
    message: str

class MetricsInfo(BaseModel):
    mse:               Optional[float] = None
    psnr:              Optional[float] = None
    ssim:              Optional[float] = None
    energy_ratio_orig: Optional[float] = None
    energy_ratio_proc: Optional[float] = None
    total_energy_orig: Optional[float] = None
    total_energy_proc: Optional[float] = None
    high_energy_orig:  Optional[float] = None
    high_energy_proc:  Optional[float] = None
    low_energy_orig:   Optional[float] = None
    low_energy_proc:   Optional[float] = None

class ProcessResponse(BaseModel):
    success: Literal[True] = True
    image_id: str
    session_id: str
    status: str
    pipeline: PipelineStatus
    preprocess: PreprocessInfo
    metadata: MetadataInfo
    paths: PathsInfo
    face_detection: FaceDetectionInfo
    landmark_detection: LandmarkDetectionInfo
    result_image: Optional[str] = None
    metrics: Optional[MetricsInfo] = None
    ai_expression: Optional[Dict[str, Any]] = None
    effects_meta: Optional[List[Dict[str, Any]]] = None
    debug: Optional[Dict[str, Any]] = None

class GalleryItemInfo(BaseModel):
    session_id: str
    image_id: str
    original_path: str
    preprocessed_path: str
    created_at: float

class GalleryResponse(BaseModel):
    success: Literal[True] = True
    items: List[GalleryItemInfo]
