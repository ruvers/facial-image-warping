"""
Smoke / integration tests for FaceWarp Lab API.

Run from the repository root:
    python -m backend.test_upload_api
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from typing import Any, Dict, Tuple

import cv2
import numpy as np
from fastapi.testclient import TestClient

from backend.main import app, UPLOAD_ROOT, PROCESSED_ROOT

client = TestClient(app)

# ── Helpers: synthetic image generators ───────────────────────────────────────


def make_png_bytes(width: int, height: int, channels: int = 3) -> bytes:
    """Return a valid PNG as raw bytes."""
    if channels == 1:
        img = np.ones((height, width), dtype=np.uint8) * 180
    else:
        img = np.ones((height, width, channels), dtype=np.uint8) * 180
    ok, buf = cv2.imencode(".png", img)
    assert ok, "cv2.imencode(.png) failed"
    return buf.tobytes()


def make_jpeg_bytes(width: int, height: int) -> bytes:
    """Return a valid JPEG as raw bytes."""
    img = np.ones((height, width, 3), dtype=np.uint8) * 120
    ok, buf = cv2.imencode(".jpg", img)
    assert ok, "cv2.imencode(.jpg) failed"
    return buf.tobytes()


def make_grayscale_png_bytes(width: int, height: int) -> bytes:
    """Return a valid single-channel PNG as raw bytes."""
    return make_png_bytes(width, height, channels=1)


def upload_png(width: int = 300, height: int = 300,
               filename: str = "test.png",
               channels: int = 3) -> Dict[str, Any]:
    """Upload a synthetic PNG and return the parsed JSON response."""
    if channels == 1:
        data = make_grayscale_png_bytes(width, height)
    else:
        data = make_png_bytes(width, height, channels)
    resp = client.post(
        "/api/upload",
        files={"file": (filename, data, "image/png")},
    )
    assert resp.status_code == 200, f"Upload failed: {resp.text}"
    return resp.json()


# ── Cleanup tracker ──────────────────────────────────────────────────────────

_created_sessions: list[str] = []


def _track(j: Dict[str, Any]) -> Dict[str, Any]:
    """Remember session IDs so we can clean up afterwards."""
    sid = j.get("session_id")
    if sid and sid.startswith("ses_"):
        _created_sessions.append(sid)
    return j


def _cleanup() -> None:
    """Remove upload/processed dirs created during tests."""
    for sid in _created_sessions:
        for root in (UPLOAD_ROOT, PROCESSED_ROOT):
            p = os.path.join(root, sid)
            if os.path.isdir(p):
                shutil.rmtree(p, ignore_errors=True)


# ── Test functions ────────────────────────────────────────────────────────────


def test_health_endpoint() -> None:
    r = client.get("/api/health")
    assert r.status_code == 200
    j = r.json()
    assert j["status"] == "ok"
    assert j["backend"] == "fastapi"
    assert j["pipeline"]["upload"] == "ready"
    assert j["pipeline"]["preprocessing"] == "ready"
    assert j["pipeline"]["face_detection"] == "stub_pending_group_2"
    print("  PASS  test_health_endpoint")


def test_upload_valid_png() -> None:
    data = make_png_bytes(300, 300)
    r = client.post("/api/upload",
                     files={"file": ("photo.png", data, "image/png")})
    assert r.status_code == 200
    j = _track(r.json())

    assert j["success"] is True
    assert j["image_id"].startswith("img_")
    assert j["session_id"].startswith("ses_")
    assert j["status"] == "uploaded"

    orig = j["original"]
    assert orig["width"] == 300
    assert orig["height"] == 300
    assert orig["channels"] == 3
    assert orig["format"] == "PNG"
    assert orig["content_type"] == "image/png"
    assert orig["size_bytes"] > 0
    assert orig["filename"] == "photo.png"

    assert j["paths"]["original_path"].startswith("uploads/")
    assert "Image uploaded successfully" in j["message"]
    print("  PASS  test_upload_valid_png")


def test_upload_valid_jpeg() -> None:
    data = make_jpeg_bytes(500, 400)
    r = client.post("/api/upload",
                     files={"file": ("shot.jpg", data, "image/jpeg")})
    assert r.status_code == 200
    j = _track(r.json())

    assert j["success"] is True
    orig = j["original"]
    assert orig["format"] == "JPEG"
    assert orig["content_type"] == "image/jpeg"
    assert orig["width"] == 500
    assert orig["height"] == 400
    assert orig["channels"] == 3
    print("  PASS  test_upload_valid_jpeg")


def test_upload_rejects_invalid_mime() -> None:
    data = make_png_bytes(300, 300)
    r = client.post("/api/upload",
                     files={"file": ("x.png", data, "text/plain")})
    assert r.status_code == 400
    j = r.json()
    assert j["success"] is False
    assert j["error"]["code"] == "INVALID_FILE_TYPE"
    print("  PASS  test_upload_rejects_invalid_mime")


def test_upload_rejects_fake_png_magic_bytes() -> None:
    r = client.post("/api/upload",
                     files={"file": ("fake.png", b"not-an-image", "image/png")})
    assert r.status_code == 400
    j = r.json()
    assert j["success"] is False
    assert j["error"]["code"] == "INVALID_FILE_TYPE"
    print("  PASS  test_upload_rejects_fake_png_magic_bytes")


def test_upload_rejects_corrupted_image() -> None:
    # Valid PNG signature followed by garbage — magic passes but decode fails
    corrupt = bytes([0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A]) + b"\x00" * 200
    r = client.post("/api/upload",
                     files={"file": ("bad.png", corrupt, "image/png")})
    assert r.status_code == 400
    j = r.json()
    assert j["success"] is False
    assert j["error"]["code"] == "CORRUPTED_IMAGE"
    print("  PASS  test_upload_rejects_corrupted_image")


def test_upload_rejects_too_small_image() -> None:
    data = make_png_bytes(100, 100)
    r = client.post("/api/upload",
                     files={"file": ("tiny.png", data, "image/png")})
    assert r.status_code == 400
    j = r.json()
    assert j["success"] is False
    assert j["error"]["code"] == "IMAGE_TOO_SMALL"
    assert j["error"]["details"]["actual_width"] == 100
    assert j["error"]["details"]["actual_height"] == 100
    print("  PASS  test_upload_rejects_too_small_image")


def test_upload_accepts_boundary_200x200() -> None:
    data = make_png_bytes(200, 200)
    r = client.post("/api/upload",
                     files={"file": ("edge.png", data, "image/png")})
    assert r.status_code == 200
    j = _track(r.json())
    assert j["success"] is True
    assert j["original"]["width"] == 200
    assert j["original"]["height"] == 200
    print("  PASS  test_upload_accepts_boundary_200x200")


def test_upload_grayscale_png() -> None:
    data = make_grayscale_png_bytes(250, 250)
    r = client.post("/api/upload",
                     files={"file": ("gray.png", data, "image/png")})
    assert r.status_code == 200
    j = _track(r.json())
    assert j["success"] is True
    assert j["original"]["channels"] == 1
    print("  PASS  test_upload_grayscale_png")


def test_process_valid_uploaded_image() -> None:
    # Upload
    uj = _track(upload_png(300, 300))
    sid, iid = uj["session_id"], uj["image_id"]

    # Process
    r = client.post("/api/process", json={
        "session_id": sid,
        "image_id": iid,
        "mode": None,
        "params": {},
        "options": {
            "target_size": 512,
            "normalize_rgb": True,
            "grayscale": False,
            "debug": True,
        },
    })
    assert r.status_code == 200
    j = r.json()

    assert j["success"] is True
    assert j["status"] == "preprocessed"
    assert j["image_id"] == iid
    assert j["session_id"] == sid

    # Pipeline
    pipe = j["pipeline"]
    assert pipe["upload"] == "completed"
    assert pipe["decode"] == "completed"
    assert pipe["preprocess"] == "completed"

    # Preprocess
    pre = j["preprocess"]
    assert pre["target_size"] == 512
    assert pre["resized_width"] == 512
    assert pre["resized_height"] == 512
    assert pre["color_space"] == "RGB"
    assert pre["normalized"] is True
    assert pre["normalization_range"] == [0.0, 1.0]
    assert pre["grayscale_generated"] is False

    # Metadata
    meta = j["metadata"]
    assert meta["original_width"] == 300
    assert meta["original_height"] == 300
    assert meta["processed_width"] == 512
    assert meta["processed_height"] == 512
    assert meta["channels"] == 3

    # Paths
    paths = j["paths"]
    assert paths["original_path"] is not None
    assert paths["preprocessed_path"] is not None
    assert paths["metadata_path"] is not None

    # Verify files on disk
    prep_file = os.path.join(
        PROCESSED_ROOT, sid, f"{iid}_preprocessed.png",
    )
    meta_file = os.path.join(
        PROCESSED_ROOT, sid, f"{iid}_metadata.json",
    )
    assert os.path.isfile(prep_file), f"Preprocessed file missing: {prep_file}"
    assert os.path.isfile(meta_file), f"Metadata file missing: {meta_file}"

    # Verify preprocessed image is 512x512x3
    raw_data = np.fromfile(prep_file, dtype=np.uint8)
    saved = cv2.imdecode(raw_data, cv2.IMREAD_UNCHANGED)
    assert saved is not None, "Could not decode saved preprocessed image"
    assert saved.shape == (512, 512, 3), f"Wrong shape: {saved.shape}"

    # Verify metadata JSON
    with open(meta_file, "r", encoding="utf-8") as f:
        md = json.load(f)
    assert md["image_id"] == iid
    assert md["session_id"] == sid
    assert md["metadata"]["processed_width"] == 512

    print("  PASS  test_process_valid_uploaded_image")


def test_process_face_detection_stub() -> None:
    uj = _track(upload_png(300, 300))
    r = client.post("/api/process", json={
        "session_id": uj["session_id"],
        "image_id": uj["image_id"],
    })
    assert r.status_code == 200
    j = r.json()

    fd = j["face_detection"]
    assert fd["enabled"] is True
    assert fd["status"] == "pending_group_2"
    assert fd["bbox"] is None
    assert fd["confidence"] is None
    assert "Group 2" in fd["message"]

    assert j["pipeline"]["face_detection"] == "pending_group_2"
    assert j["pipeline"]["landmark_detection"] == "pending_group_2"
    print("  PASS  test_process_face_detection_stub")


def test_process_invalid_image_id() -> None:
    r = client.post("/api/process", json={
        "session_id": "ses_nonexistent",
        "image_id": "img_nonexistent",
    })
    assert r.status_code == 404
    j = r.json()
    assert j["success"] is False
    assert j["error"]["code"] == "IMAGE_NOT_FOUND"
    print("  PASS  test_process_invalid_image_id")


def test_process_grayscale_upload_becomes_rgb() -> None:
    uj = _track(upload_png(250, 250, filename="gray.png", channels=1))
    assert uj["original"]["channels"] == 1

    r = client.post("/api/process", json={
        "session_id": uj["session_id"],
        "image_id": uj["image_id"],
    })
    assert r.status_code == 200
    j = r.json()

    assert j["metadata"]["channels"] == 3
    assert j["preprocess"]["color_space"] == "RGB"
    assert j["metadata"]["processed_width"] == 512
    assert j["metadata"]["processed_height"] == 512
    print("  PASS  test_process_grayscale_upload_becomes_rgb")


# ── Runner ────────────────────────────────────────────────────────────────────

ALL_TESTS = [
    test_health_endpoint,
    test_upload_valid_png,
    test_upload_valid_jpeg,
    test_upload_rejects_invalid_mime,
    test_upload_rejects_fake_png_magic_bytes,
    test_upload_rejects_corrupted_image,
    test_upload_rejects_too_small_image,
    test_upload_accepts_boundary_200x200,
    test_upload_grayscale_png,
    test_process_valid_uploaded_image,
    test_process_face_detection_stub,
    test_process_invalid_image_id,
    test_process_grayscale_upload_becomes_rgb,
]


def main() -> int:
    passed = 0
    failed = 0
    errors: list[str] = []

    for fn in ALL_TESTS:
        try:
            fn()
            passed += 1
        except Exception as exc:
            failed += 1
            errors.append(f"  FAIL  {fn.__name__}: {exc}")
            print(f"  FAIL  {fn.__name__}: {exc}")

    print(f"\n{'=' * 50}")
    print(f"Results: {passed}/{passed + failed} passed")

    if errors:
        print("\nFailed tests:")
        for e in errors:
            print(e)

    _cleanup()

    if failed:
        print(f"\n{failed} test(s) FAILED.")
        return 1

    print("\nAll checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
