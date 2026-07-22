from __future__ import annotations

from io import BytesIO

from fastapi.testclient import TestClient
from PIL import Image, ImageDraw

from backend.main import app


def _person_png_bytes() -> bytes:
    image = Image.new("RGB", (512, 768), "white")
    draw = ImageDraw.Draw(image)
    draw.ellipse((206, 70, 306, 170), fill=(220, 178, 145))
    draw.rectangle((170, 170, 342, 650), fill=(72, 76, 86))
    out = BytesIO()
    image.save(out, format="PNG")
    return out.getvalue()


def main() -> None:
    with TestClient(app) as client:
        manifest = client.get("/api/store/manifest")
        assert manifest.status_code == 200
        garments = [
            item
            for item in manifest.json().get("items", [])
            if item.get("pipeline") == "virtual_tryon" and item.get("slot") == "upperbody"
        ]
        assert garments, "No upperbody store garment found."

        upload = client.post(
            "/api/upload",
            files={"file": ("person.png", _person_png_bytes(), "image/png")},
        )
        assert upload.status_code == 200, upload.text
        upload_json = upload.json()

        tryon = client.post(
            "/api/store/tryon",
            data={
                "session_id": upload_json["session_id"],
                "image_id": upload_json["image_id"],
                "item_id": garments[0]["id"],
                "category": "upperbody",
                "model_type": "dc",
            },
        )
        assert tryon.status_code == 200, tryon.text
        data = tryon.json()
        assert data["success"] is True
        assert data["fallback_used"] is True
        assert data["error"] == "cpu_preview_fallback"
        assert data.get("result_image")
        assert data.get("output_paths")
        print(
            {
                "ok": True,
                "item_id": garments[0]["id"],
                "provider": data.get("provider"),
                "output": data["output_paths"][0],
            }
        )


if __name__ == "__main__":
    main()
