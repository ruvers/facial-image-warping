# Generative Hat SD 1.5 Inpaint Plan

## Decision

Production hat insertion should use a lightweight local pipeline first:

- Primary: SD 1.5 Inpainting + IP-Adapter reference conditioning.
- Optional future provider: AnyDoor.
- Procedural/parametric hats remain experimental preview only and must not be applied to production output.

## Runtime Constraints

- Run only on CUDA. CPU inference is intentionally disabled.
- No model download or training happens in the FastAPI request path.
- If models or CUDA are unavailable, return the input image unchanged with explicit fallback metadata.

## Planned Pipeline

1. Build a hat target mask from face landmarks.
2. Keep the mask above brows and outside eye/nose/mouth regions.
3. Load a hat reference image or asset reference.
4. Run SD 1.5 inpainting with IP-Adapter reference conditioning.
5. Composite local result with edge cleanup and hair/forehead-aware blending.
6. Return debug metadata: provider, mask stats, CUDA status, fallback reason, and changed pixels.

## Provider Status Contract

`backend.local_models.generative_refiner.get_generative_refiner_status()` reports:

- `primary_provider: "hat_light_inpaint"`
- CUDA availability.
- SD 1.5/IP-Adapter dependency availability.
- AnyDoor optional provider availability.

## Failure Behavior

The refiner must never crash the server. Expected fallback errors include:

- `generative_hat_requires_cuda`
- `provider_not_installed`
- `hat_reference_missing`
- `hat_mask_missing`
- `hat_light_inpaint_inference_not_implemented`

