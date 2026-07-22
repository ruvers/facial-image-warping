from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Callable

import numpy as np

from backend.face_analysis import analyze_face
from backend.three_d.manager import enrich_with_best_available_3d


EffectFn = Callable[[np.ndarray, dict, dict], np.ndarray]


@dataclass
class RegisteredEffect:
    name: str
    stage: str
    fn: EffectFn


class EffectEngine:
    """
    Central effect engine.

    Responsibilities:
    - run face analysis once
    - enrich context with best available 3D data
    - keep shared FaceContext-like dict
    - run enabled effects in stage order
    - allow other developers to plug in their own effects

    Every effect must follow:

        def apply_xxx(image_bgr, ctx, params) -> image_bgr:
            ...
            return image_bgr
    """

    STAGE_ORDER = [
        "pre_geometry",
        "geometry",
        "beauty",
        "hair",
        "accessory",
        "postprocess",
    ]

    def __init__(self):
        self.effects: dict[str, RegisteredEffect] = {}

    @staticmethod
    def _mask_stats_for_effect(
        ctx: dict,
        name: str,
        image_shape: tuple[int, ...],
    ) -> dict | None:
        mask_name = {
            "hair_color": "hair",
            "lipstick": "lips",
            "blush": "skin_effect",
            "skin_smooth": "skin_effect",
            "eyeshadow": "eyes",
            "eyeliner": "eyes",
        }.get(name)

        if not mask_name:
            return None

        mask = ctx.get("masks", {}).get(mask_name)
        if mask is None:
            return {
                "mask": mask_name,
                "pixels": 0,
                "coverage": 0.0,
                "reason": "mask missing",
            }

        pixels = int(np.count_nonzero(mask > 20))
        total = max(1, int(image_shape[0] * image_shape[1]))
        return {
            "mask": mask_name,
            "pixels": pixels,
            "coverage": float(pixels / total),
        }

    def register(
        self,
        name: str,
        stage: str,
        fn: EffectFn,
    ) -> None:
        if stage not in self.STAGE_ORDER:
            raise ValueError(
                f"Invalid stage '{stage}'. Valid stages: {self.STAGE_ORDER}"
            )

        self.effects[name] = RegisteredEffect(
            name=name,
            stage=stage,
            fn=fn,
        )

    def run(
        self,
        image_bgr: np.ndarray,
        params: dict | None = None,
    ) -> tuple[np.ndarray, dict]:
        if image_bgr is None:
            raise ValueError("image_bgr is None")

        params = params or {}

        # =====================================================
        # STAGE 1 — FACE ANALYSIS
        # =====================================================

        ctx = analyze_face(
            image_bgr,
        )
        motion_state = params.get("__accessory_motion_state")
        if isinstance(motion_state, dict):
            ctx["accessory_motion_state"] = motion_state

        # =====================================================
        # STAGE 1.5 — BEST AVAILABLE 3D CONTEXT
        # =====================================================
        # Current:
        #   MediaPipe pseudo-3D
        #
        # Future:
        #   DECA / FLAME true 3D if available
        #
        # Bu fonksiyon provider seçimini kendi yapar.
        # Frontend ya da effect modülleri provider seçmez.

        return self.run_with_context(image_bgr, ctx, params)

    def run_with_context(
        self,
        image_bgr: np.ndarray,
        ctx: dict,
        params: dict | None = None,
    ) -> tuple[np.ndarray, dict]:
        params = params or {}
        result = image_bgr.copy()
        effects_meta: list[dict] = []
        skip_effect_diff = bool(params.get("__skip_effect_diff", False))
        debug_effects = bool(params.get("__debug_effects", False)) or os.getenv("FACEWARP_EFFECT_DEBUG") == "1"

        # =====================================================
        # EFFECT EXECUTION BY STAGE ORDER
        # =====================================================

        for stage in self.STAGE_ORDER:
            for name, effect in self.effects.items():
                if effect.stage != stage:
                    continue

                effect_params = params.get(
                    name,
                    {},
                )

                enabled = bool(effect_params.get("enabled", False))

                if not enabled:
                    continue

                before = result
                mask_stats = self._mask_stats_for_effect(
                    ctx,
                    name,
                    result.shape,
                )

                try:
                    next_result = effect.fn(
                        result,
                        ctx,
                        effect_params,
                    )

                    if next_result is None:
                        effects_meta.append(
                            {
                                "effect": name,
                                "enabled": True,
                                "applied": False,
                                "reason": "Effect returned None.",
                                "color": effect_params.get("color"),
                                "intensity": effect_params.get("intensity"),
                                "mask_stats": mask_stats,
                                "changed_pixels": 0,
                            }
                        )
                        continue

                    if skip_effect_diff:
                        changed_pixels = -1
                        changed = True
                    else:
                        diff = np.any(before != next_result, axis=2)
                        changed_pixels = int(np.count_nonzero(diff))
                        changed = changed_pixels > 0

                    result = next_result

                    meta = {
                        "effect": name,
                        "enabled": True,
                        "applied": bool(changed),
                        "reason": None if changed else "No visible pixel change.",
                        "color": effect_params.get("color"),
                        "intensity": effect_params.get("intensity"),
                        "mask_stats": mask_stats,
                        "changed_pixels": changed_pixels,
                    }
                    effect_debug = ctx.get("effect_debug_meta", {}).get(name)
                    if isinstance(effect_debug, dict):
                        meta.update(effect_debug)
                    effects_meta.append(meta)
                    if debug_effects:
                        print("[DEBUG] effect_meta:", meta)

                except Exception as exc:
                    meta = {
                        "effect": name,
                        "enabled": True,
                        "applied": False,
                        "reason": str(exc),
                        "color": effect_params.get("color"),
                        "intensity": effect_params.get("intensity"),
                        "mask_stats": mask_stats,
                        "changed_pixels": 0,
                    }
                    effects_meta.append(meta)
                    if debug_effects:
                        print("[WARN] effect_failed:", meta)

                    continue

        ctx["effects_meta"] = effects_meta

        return result, ctx


# =========================================================
# GLOBAL ENGINE INSTANCE
# =========================================================

photo_engine = EffectEngine()


def run_photo_engine(
    image_bgr: np.ndarray,
    params: dict | None = None,
) -> tuple[np.ndarray, dict]:
    return photo_engine.run(
        image_bgr,
        params,
    )


def run_photo_engine_with_context(
    image_bgr: np.ndarray,
    ctx: dict,
    params: dict | None = None,
) -> tuple[np.ndarray, dict]:
    return photo_engine.run_with_context(
        image_bgr,
        ctx,
        params,
    )

