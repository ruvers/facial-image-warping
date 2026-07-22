from __future__ import annotations

from backend.local_models.generative_refiner import get_generative_refiner_status


def test_generative_refiner_status_does_not_crash() -> None:
    status = get_generative_refiner_status()
    assert isinstance(status, dict)
    assert status.get("primary_provider") == "hat_light_inpaint"
    assert "hat_light_inpaint" in status.get("providers", {})
    assert "anydoor" in status.get("providers", {})
    assert isinstance(status.get("supported_modes"), list)
    assert "hat_light_inpaint" in status["supported_modes"]
    assert "anydoor_inpaint" in status["supported_modes"]


if __name__ == "__main__":
    test_generative_refiner_status_does_not_crash()
    print("generative refiner status ok")
