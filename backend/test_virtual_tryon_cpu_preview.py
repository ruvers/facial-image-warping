from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from backend.virtual_tryon.ootdiffusion import run_cpu_preview_tryon


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        person_path = root / "person.png"
        cloth_path = root / "cloth.png"
        output_dir = root / "out"

        person = Image.new("RGB", (512, 768), "white")
        draw = ImageDraw.Draw(person)
        draw.ellipse((206, 70, 306, 170), fill=(220, 175, 145))
        draw.rectangle((165, 170, 347, 640), fill=(70, 70, 80))
        person.save(person_path)

        cloth = Image.new("RGBA", (280, 220), (0, 0, 0, 0))
        draw = ImageDraw.Draw(cloth)
        draw.polygon(
            [(50, 20), (230, 20), (270, 210), (10, 210)],
            fill=(180, 30, 45, 235),
        )
        cloth.save(cloth_path)

        result = run_cpu_preview_tryon(
            person_path=person_path,
            cloth_path=cloth_path,
            output_dir=output_dir,
            category="upperbody",
        )

        assert result["success"] is True
        assert result["fallback_used"] is True
        assert result["error"] == "cpu_preview_fallback"
        outputs = result.get("outputs", [])
        assert outputs and Path(outputs[0]["path"]).exists()

        original = np.array(Image.open(person_path).convert("RGB"))
        preview = np.array(Image.open(outputs[0]["path"]).convert("RGB"))
        changed = int(np.count_nonzero(np.any(original != preview, axis=2)))
        assert changed > 1000, f"Preview changed too few pixels: {changed}"

        print({"ok": True, "changed_pixels": changed, "output": outputs[0]["filename"]})


if __name__ == "__main__":
    main()
