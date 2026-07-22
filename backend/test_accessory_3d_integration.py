from __future__ import annotations

import numpy as np

from backend.photo_pipeline import apply_photo_pipeline
from backend.test_hat3d_pipeline import _synthetic_ctx as _hat_ctx


def test_accessory_3d_photo_pipeline_integration() -> None:
    import backend.effect_engine as effect_engine

    original_analyze = effect_engine.analyze_face
    original_enrich = effect_engine.enrich_with_best_available_3d

    ctx = _hat_ctx()
    ctx["masks"]["neck"] = np.zeros((512, 512), dtype=np.uint8)
    yy, xx = np.ogrid[:512, :512]
    ctx["masks"]["neck"][((xx - 256) ** 2) / (58**2) + ((yy - 330) ** 2) / (78**2) <= 1.0] = 255
    ctx["anchors"]["necklace"] = {
        "center": (256.0, 333.0),
        "width": 132.0,
        "chin": (256.0, 285.0),
    }

    def fake_analyze(_image_bgr):
        return ctx

    def fake_enrich(in_ctx, _image_bgr):
        in_ctx["three_d"] = {"provider": "test_stub", "is_true_3d": False}
        return in_ctx

    try:
        effect_engine.analyze_face = fake_analyze
        effect_engine.enrich_with_best_available_3d = fake_enrich

        image = np.full((512, 512, 3), 235, dtype=np.uint8)
        output, out_ctx = apply_photo_pipeline(
            image,
            {
                "accessories": {
                    "enabled": True,
                    "items": [
                        {
                            "type": "hat",
                            "category": "beanie",
                            "render_mode": "parametric_3d",
                            "asset_id": "beanie_black_wool_procedural",
                            "metadata": {"color": "#222222"},
                        },
                        {
                            "type": "necklace",
                            "category": "pendant_necklace",
                            "render_mode": "physics_3d",
                            "asset_id": "gold_pendant_procedural",
                            "metadata": {
                                "chain_length": 1.0,
                                "chain_thickness": 2.0,
                                "stiffness": 0.75,
                                "pendant_enabled": True,
                                "pendant_size": 0.12,
                                "pendant_weight": 1.0,
                                "material": "gold",
                            },
                        },
                    ],
                },
            },
        )
    finally:
        effect_engine.analyze_face = original_analyze
        effect_engine.enrich_with_best_available_3d = original_enrich

    changed_pixels = int(np.count_nonzero(np.any(output != image, axis=2)))
    accessory_meta = [
        item for item in out_ctx.get("effects_meta", [])
        if item.get("effect") == "accessory_3d"
    ]

    assert output.shape == image.shape
    assert changed_pixels > 2000
    assert accessory_meta
    assert len(accessory_meta[0].get("items", [])) == 2
    hat_meta = next(item for item in accessory_meta[0]["items"] if item.get("type") == "hat")
    necklace_meta = next(item for item in accessory_meta[0]["items"] if item.get("type") == "necklace")
    assert hat_meta.get("applied") is False
    assert hat_meta.get("experimental") is True
    assert hat_meta.get("error") == "parametric_hat_disabled_experimental_only"
    assert necklace_meta.get("applied") is True
    assert necklace_meta.get("fallback_used") is False


if __name__ == "__main__":
    test_accessory_3d_photo_pipeline_integration()
    print("accessory3d integration ok")
