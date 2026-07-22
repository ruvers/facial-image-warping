from __future__ import annotations

from copy import deepcopy
from typing import Any


# =========================================================
# DEFAULT PARAMS
# =========================================================

DEFAULT_PARAMS: dict[str, Any] = {
    # STAGE 4 — geometry
    "face_reshape": {
        "enabled": False,
        "face_slimming": 0.0,
        "lip_intensity": 0.0,
    },

    # STAGE 5 — expression
    "expression": {
        "enabled": False,
        "smile_intensity": 0.0,
        "eyebrow_intensity": 0.0,
        "lip_intensity": 0.0,
    },

    # STAGE 3 — beauty
    "skin_smooth": {
        "enabled": False,
        "intensity": 0.20,
    },

    "blush": {
        "enabled": False,
        "color": "#D96C7C",
        "intensity": 0.22,
    },

    "lipstick": {
        "enabled": False,
        "color": "#A02045",
        "intensity": 0.55,
    },

    "eyebrow": {
        "enabled": False,
        "intensity": 0.20,
    },

    "hair_color": {
        "enabled": False,
        "color": "#6D38D8",
        "intensity": 0.75,
    },

    "eye_color": {
        "enabled": False,
        "color": "#3F7FBF",
        "intensity": 0.35,
    },

    "eyeshadow": {
        "enabled": False,
        "color": "#8C7A6B",
        "intensity": 0.25,
    },

    "eyeliner": {
        "enabled": False,
        "color": "#080808",
        "intensity": 0.45,
    },

    # STAGE 2 — accessories dispatcher
    "accessories": {
        "enabled": False,

        "glasses": {
            "enabled": False,
            "asset_path": "",
            "width_scale": 1.0,
            "y_offset_ratio": 0.0,
        },

        "earrings": {
            "enabled": False,
            "side": "both",
            "style": "diamond",
            "color": "gold",
            "scale": 1.0,
            "opacity": 0.95,
            "hair_occlusion": True,
        },

        "necklace": {
            "enabled": False,
            "style": "diamond",
            "color": "gold",
            "scale": 1.0,
            "opacity": 0.90,
            "sag": 0.22,
        },
    },

    # Direct top-level accessory support
    "glasses": {
        "enabled": False,
        "asset_path": "",
        "width_scale": 1.0,
        "y_offset_ratio": 0.0,
    },

    "accessories": {
        "enabled": False,
        "items": [],
    },

    "accessory_3d": {
        "enabled": False,
        "items": [],
    },

    "earrings": {
        "enabled": False,
        "side": "both",
        "style": "diamond",
        "color": "gold",
        "scale": 1.0,
        "opacity": 0.95,
        "hair_occlusion": True,
    },

    "necklace": {
        "enabled": False,
        "style": "diamond",
        "color": "gold",
        "scale": 1.0,
        "opacity": 0.90,
        "sag": 0.22,
    },

        "ai_enhance": {
        "enabled": False,
        "use_real_ai": False,
        "fallback_cv_polish": True,
        "intensity": 0.25,
    },

        "aging_model": {
        "enabled": False,
        "strict": False,
        "fallback_noop": True,
        "intensity": 0.0,
    },

    "makeup_model": {
        "enabled": False,
        "strict": False,
        "fallback_noop": True,
        "intensity": 0.0,
    },

    "expression_model": {
        "enabled": False,
        "strict": False,
        "fallback_noop": True,
        "intensity": 0.0,
    },

    "accessory_model": {
        "enabled": False,
        "strict": False,
        "fallback_noop": True,
        "intensity": 0.0,
    },

    "face_restore": {
        "enabled": False,
        "strict": False,
        "fallback_noop": True,
        "intensity": 0.0,
    },
}


# =========================================================
# CATALOG FOR FRONTEND / TEAM
# =========================================================

EFFECT_CATALOG: dict[str, Any] = {
    "stages": [
        "pre_geometry",
        "geometry",
        "beauty",
        "hair",
        "accessory",
        "postprocess",
    ],

    "accessories": {
        "stage": "accessory",
        "description": "Generic accessory engine for glasses, earrings and necklace using MediaPipe anchors.",
        "params": {
            "enabled": "bool",
            "items": "list of accessory placement configs",
        },
    },

    "effects": {
        "face_reshape": {
            "stage": "geometry",
            "description": "Landmark-based face shape warping.",
            "params": {
                "enabled": "bool",
                "face_slimming": "float -1.0..1.0",
                "lip_intensity": "float -1.0..1.0",
            },
        },

        "ai_enhance": {
            "stage": "postprocess",
            "description": "Optional final AI enhancement layer. Currently uses safe CV fallback if real AI is unavailable.",
            "params": {
                "enabled": "bool",
                "use_real_ai": "bool",
                "fallback_cv_polish": "bool",
                "intensity": "float 0.0..1.0",
            },
        },

        "expression": {
            "stage": "geometry",
            "description": "Smile, eyebrow raise and lip expression warp.",
            "params": {
                "enabled": "bool",
                "smile_intensity": "float -1.0..1.0",
                "eyebrow_intensity": "float -1.0..1.0",
                "lip_intensity": "float -1.0..1.0",
            },
        },

        "skin_smooth": {
            "stage": "beauty",
            "description": "Edge-preserving smoothing on skin_effect mask.",
            "params": {
                "enabled": "bool",
                "intensity": "float 0.0..1.0",
            },
        },

        "blush": {
            "stage": "beauty",
            "description": "Soft cheek blush using landmarks and skin mask.",
            "params": {
                "enabled": "bool",
                "color": "hex string",
                "intensity": "float 0.0..1.0",
            },
        },

        "lipstick": {
            "stage": "beauty",
            "description": "LAB-based lipstick effect on lips mask.",
            "params": {
                "enabled": "bool",
                "color": "hex string",
                "intensity": "float 0.0..1.0",
            },
        },

        "eyebrow": {
            "stage": "beauty",
            "description": "Eyebrow darkening/enhancement.",
            "params": {
                "enabled": "bool",
                "intensity": "float 0.0..1.0",
            },
        },

        "hair_color": {
            "stage": "hair",
            "description": "LAB-based hair recoloring.",
            "params": {
                "enabled": "bool",
                "color": "hex string",
                "intensity": "float 0.0..1.0",
            },
        },

        "eye_color": {
            "stage": "beauty",
            "description": "Conservative iris recoloring using MediaPipe iris landmarks.",
            "params": {
                "enabled": "bool",
                "color": "hex string",
                "intensity": "float 0.0..1.0",
            },
        },

        "eyeshadow": {
            "stage": "beauty",
            "description": "Soft upper-eyelid color using MediaPipe eye landmarks.",
            "params": {
                "enabled": "bool",
                "color": "hex string",
                "intensity": "float 0.0..1.0",
            },
        },

        "eyeliner": {
            "stage": "beauty",
            "description": "Conservative upper-eye contour darkening using MediaPipe landmarks.",
            "params": {
                "enabled": "bool",
                "color": "hex string",
                "intensity": "float 0.0..1.0",
            },
        },

        "accessories": {
            "stage": "accessory",
            "description": "Accessory dispatcher. Other developers can plug assets here.",
            "params": {
                "enabled": "bool",
                "glasses": "object",
                "earrings": "object",
                "necklace": "object",
            },
        },

        "accessory_3d": {
            "stage": "accessory",
            "description": "Physics/parametric 3D accessory dispatcher. MVP supports pendant_necklace with render_mode physics_3d.",
            "params": {
                "enabled": "bool",
                "items": "list of accessory items with render_mode physics_3d",
            },
        },

                "aging_model": {
            "stage": "postprocess",
            "description": "Local-only aging model plugin slot. Core backend does not implement aging.",
            "params": {
                "enabled": "bool",
                "strict": "bool",
                "fallback_noop": "bool",
                "intensity": "float 0.0..1.0",
            },
        },

        "makeup_model": {
            "stage": "beauty",
            "description": "Local-only AI makeup/refinement plugin slot.",
            "params": {
                "enabled": "bool",
                "strict": "bool",
                "fallback_noop": "bool",
                "intensity": "float 0.0..1.0",
            },
        },

        "expression_model": {
            "stage": "geometry",
            "description": "Local-only expression/geometry plugin slot.",
            "params": {
                "enabled": "bool",
                "strict": "bool",
                "fallback_noop": "bool",
                "intensity": "float 0.0..1.0",
            },
        },

        "accessory_model": {
            "stage": "accessory",
            "description": "Local-only accessory fitting/refinement plugin slot.",
            "params": {
                "enabled": "bool",
                "strict": "bool",
                "fallback_noop": "bool",
                "intensity": "float 0.0..1.0",
            },
        },

        "face_restore": {
            "stage": "postprocess",
            "description": "Local-only face restoration/postprocess plugin slot.",
            "params": {
                "enabled": "bool",
                "strict": "bool",
                "fallback_noop": "bool",
                "intensity": "float 0.0..1.0",
            },
        },
    },

    "default_params": DEFAULT_PARAMS,
}


# =========================================================
# NORMALIZATION
# =========================================================

def _deep_merge(
    base: dict[str, Any],
    incoming: dict[str, Any],
) -> dict[str, Any]:
    out = deepcopy(base)

    for key, value in incoming.items():
        if (
            key in out
            and isinstance(out[key], dict)
            and isinstance(value, dict)
        ):
            out[key] = _deep_merge(
                out[key],
                value,
            )
        else:
            out[key] = value

    return out


def normalize_params(
    params: dict[str, Any] | None,
) -> dict[str, Any]:
    """
    Merge incoming params with defaults.

    Important:
    - Known effects get safe defaults.
    - Unknown future effects are preserved.
    - This prevents missing-key crashes.
    """

    params = params or {}

    normalized = deepcopy(DEFAULT_PARAMS)

    normalized = _deep_merge(
        normalized,
        params,
    )

    # Preserve unknown future keys.
    for key, value in params.items():
        if key not in normalized:
            normalized[key] = value

    return normalized


def get_effect_catalog() -> dict[str, Any]:
    return deepcopy(EFFECT_CATALOG)
