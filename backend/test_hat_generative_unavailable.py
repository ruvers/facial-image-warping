from __future__ import annotations

import numpy as np

from backend.accessories.hat_generative.mask import build_hat_placement_mask
from backend.accessories.hat_generative.pipeline import apply_generative_hat
from backend.effects.accessory_3d_v1 import apply_accessory_3d
from backend.local_models.generative_refiner import apply_generative_hat_inpaint
from backend.test_hat_generative_mask import _synthetic_ctx


def test_unavailable_provider_returns_unchanged_fallback() -> None:
    image = np.full((512, 512, 3), 235, dtype=np.uint8)
    mask, _debug = build_hat_placement_mask(
        image.shape,
        _synthetic_ctx(),
    )

    output, meta = apply_generative_hat_inpaint(
        image,
        None,
        mask,
        {
            "provider": "hat_light_inpaint",
            "render_mode": "hat_light_inpaint",
        },
    )

    assert np.array_equal(output, image)
    assert meta["fallback_used"] is True
    assert meta["applied"] is False
    assert meta["error"] in {
        "generative_hat_requires_cuda",
        "provider_not_installed",
        "hat_reference_missing",
        "hat_light_inpaint_inference_not_implemented",
    }


def test_hat_generative_pipeline_unavailable_fallback() -> None:
    image = np.full((512, 512, 3), 235, dtype=np.uint8)
    output, meta = apply_generative_hat(
        image,
        _synthetic_ctx(),
        {
            "type": "hat",
            "category": "beanie",
            "render_mode": "hat_light_inpaint",
            "asset_id": "hat_ref_missing",
        },
    )

    assert np.array_equal(output, image)
    assert meta["type"] == "hat"
    assert meta["provider"] == "hat_light_inpaint"
    assert meta["fallback_used"] is True
    assert meta["applied"] is False
    assert meta["error"] is not None
    assert meta["debug"]["mask"]["bottom_above_brow"] is True


def test_procedural_hat_does_not_apply_to_output() -> None:
    image = np.full((512, 512, 3), 235, dtype=np.uint8)
    ctx = _synthetic_ctx()
    output = apply_accessory_3d(
        image,
        ctx,
        {
            "enabled": True,
            "items": [
                {
                    "type": "hat",
                    "category": "beanie",
                    "render_mode": "parametric_3d",
                    "asset_id": "beanie_black_wool_procedural",
                }
            ],
        },
    )

    meta = ctx.get("effect_debug_meta", {}).get("accessory_3d", {})
    assert np.array_equal(output, image)
    assert meta.get("items")
    assert meta["items"][0]["experimental"] is True
    assert meta["items"][0]["applied"] is False
    assert meta["items"][0]["fallback_used"] is True
    assert meta["items"][0]["error"] == "parametric_hat_disabled_experimental_only"


if __name__ == "__main__":
    test_unavailable_provider_returns_unchanged_fallback()
    test_hat_generative_pipeline_unavailable_fallback()
    test_procedural_hat_does_not_apply_to_output()
    print("hat generative unavailable ok")
