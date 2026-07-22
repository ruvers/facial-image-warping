# FaceWarp Store Assets

This directory is for store-style products that users can pick and try on.

## Garments

Place product images in:

- `assets/store/garments/upperbody/`
- `assets/store/garments/lowerbody/`
- `assets/store/garments/dress/`

Accepted extensions: `.png`, `.jpg`, `.jpeg`, `.webp`.

Garment image quality matters for fit. Use a single front-facing clothing product image, full garment visible, clean or transparent background, and at least a 768px long edge when possible. Files in these folders are auto-discovered as virtual try-on store items.

## Accessories

Existing accessory manifest items are automatically exposed in the store as accessory products. For production-grade fit, keep accessory PNGs transparent, tightly cropped, and define scale/offset/default alpha in `assets/manifest.json`.

## Explicit Store Items

Add explicit items to `assets/store/manifest.json` when you need brand, price, thumbnails, or custom fit metadata. Garment items use:

```json
{
  "id": "black_tshirt",
  "name": "Black T-Shirt",
  "type": "garment",
  "category": "upperbody",
  "slot": "upperbody",
  "pipeline": "virtual_tryon",
  "provider": "ootdiffusion",
  "thumbnail": "assets/store/garments/upperbody/black_tshirt.png",
  "tryon_image": "assets/store/garments/upperbody/black_tshirt.png",
  "model_type": "dc",
  "tryon_category": "upperbody",
  "asset_quality": {
    "source": "project-local",
    "background": "plain",
    "fit_ready": true
  },
  "fit_profile": {
    "target_region": "upperbody",
    "canonical_view": "front_product",
    "mask_policy": "provider_generated",
    "scale_hint": 1.0,
    "offset_x": 0.0,
    "offset_y": 0.0
  }
}
```
