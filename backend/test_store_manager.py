from __future__ import annotations

from backend.store_manager import (
    DEFAULT_SLOTS,
    list_store_items,
    load_store_manifest,
    resolve_outfit_slots,
)


def main() -> None:
    manifest = load_store_manifest()
    assert manifest["schema"] == "facewarp_store_manifest_v1"
    assert set(DEFAULT_SLOTS).issubset(set(manifest.get("slots", {})))

    items = manifest.get("items", [])
    assert isinstance(items, list)

    accessory_items = [item for item in items if item.get("pipeline") == "accessory_overlay"]
    assert accessory_items, "Expected existing accessory manifest items to be exposed as store items."

    glasses_items = list_store_items(slot="glasses")
    assert glasses_items, "Expected glasses store items from accessory manifest."
    first_glasses = glasses_items[0]
    assert first_glasses.get("asset_category") == "glasses"
    assert first_glasses.get("fit_profile", {}).get("anchor_schema") == "facewarp_accessory_anchors_v1"

    try:
      resolve_outfit_slots(["missing_store_item"])
    except KeyError:
      pass
    else:
      raise AssertionError("Unknown store item should raise KeyError.")

    print(
        {
            "ok": True,
            "slots": len(manifest.get("slots", {})),
            "items": len(items),
            "accessory_items": len(accessory_items),
            "glasses_items": len(glasses_items),
        }
    )


if __name__ == "__main__":
    main()
