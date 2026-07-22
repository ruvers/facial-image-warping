from __future__ import annotations

import inspect
import sys
import traceback
from typing import Any

import numpy as np


# =========================================================
# COMPATIBILITY PATCH
# =========================================================
# DECA / FLAME / chumpy stack is old.
# Python 3.11 removed inspect.getargspec.
# New NumPy removed aliases such as np.int, np.float, np.unicode.
# Patch before importing DECA / chumpy.

if not hasattr(inspect, "getargspec"):
    inspect.getargspec = inspect.getfullargspec


_numpy_aliases = {
    "bool": bool,
    "int": int,
    "float": float,
    "complex": complex,
    "object": object,
    "str": str,
    "unicode": str,
}

for _name, _value in _numpy_aliases.items():
    if _name not in np.__dict__:
        setattr(np, _name, _value)


from backend.local_models.registry import DECA_DIR, DECA_MODEL_DIR, deca_status
from backend.local_models.runtime import get_torch_device


class DecaRuntime:
    """
    Safe local DECA runtime wrapper.

    Policy:
    - No cloud
    - No paid API
    - No fake true-3D output
    - If DECA files/deps are missing, return unavailable
    - PyTorch3D renderer disabled on Windows/CPU
    - MediaPipe pseudo-3D remains fallback
    """

    def __init__(self):
        self.loaded = False
        self.device: str | None = None
        self.deca = None
        self.cfg = None
        self.last_error: str | None = None
        self.renderer_enabled: bool | None = None

    def repo_available(self) -> bool:
        return (DECA_DIR / "decalib").exists()

    def files_available(self) -> bool:
        status = deca_status()
        return bool(status.get("available", False))

    def deps_importable(self) -> bool:
        """
        Checks whether DECA Python modules can be imported.
        Does not load the model.
        """

        if not self.repo_available():
            self.last_error = "DECA repo not found"
            return False

        deca_path = str(DECA_DIR)

        if deca_path not in sys.path:
            sys.path.insert(0, deca_path)

        try:
            import torch  # noqa: F401
            from decalib.deca import DECA  # noqa: F401
            from decalib.utils.config import cfg as deca_cfg  # noqa: F401

            return True

        except Exception as e:
            self.last_error = f"DECA dependencies/import failed: {e}"
            return False

    def is_available(self) -> bool:
        """
        DECA is usable only when:
        - repo exists
        - model files exist
        - dependencies import
        """

        if not self.repo_available():
            self.last_error = "repo_missing"
            return False

        if not self.files_available():
            self.last_error = "weights_or_flame_files_missing"
            return False

        if not self.deps_importable():
            return False

        return True

    def load(self) -> bool:
        """
        Load DECA once.

        CPU-safe behavior:
        - force torch.load(..., map_location=device)
        - disable PyTorch3D renderer before DECA constructor runs

        We do not need renderer for the first integration step.
        We only need encode/decode mesh data.
        """

        if self.loaded and self.deca is not None:
            return True

        if not self.is_available():
            return False

        try:
            deca_path = str(DECA_DIR)

            if deca_path not in sys.path:
                sys.path.insert(0, deca_path)

            import torch

            from decalib.deca import DECA
            from decalib.utils.config import cfg as deca_cfg

            selected_device = get_torch_device(
                prefer_gpu=True,
            )
            self.device = selected_device

            cfg = deca_cfg

            try:
                cfg.model.use_tex = False
            except Exception:
                pass

            self.cfg = cfg

            original_torch_load = torch.load

            def safe_torch_load(*args, **kwargs):
                if "map_location" not in kwargs:
                    kwargs["map_location"] = torch.device("cpu")

                return original_torch_load(*args, **kwargs)

            original_setup_renderer = DECA._setup_renderer

            def no_renderer_setup(self_deca, model_cfg):
                self_deca.render = None
                self_deca.renderer_disabled = True

            torch.load = safe_torch_load
            DECA._setup_renderer = no_renderer_setup

            try:
                self.deca = DECA(
                    config=cfg,
                    device=self.device,
                )

            finally:
                torch.load = original_torch_load
                DECA._setup_renderer = original_setup_renderer

            self.loaded = True
            self.renderer_enabled = False
            self.last_error = None

            return True

        except Exception as e:
            traceback.print_exc()

            self.loaded = False
            self.deca = None
            self.renderer_enabled = None
            self.last_error = str(e)

            return False

    def status(self) -> dict[str, Any]:
        status = deca_status()

        return {
            "provider": "deca_flame",
            "repo_ok": bool(status.get("repo_ok", False)),
            "model_dir_ok": bool(status.get("model_dir_ok", False)),
            "found_weights": status.get("found_weights", []),
            "available": self.is_available(),
            "loaded": self.loaded,
            "device": self.device,
            "renderer_enabled": self.renderer_enabled,
            "last_error": self.last_error,
            "deca_dir": str(DECA_DIR),
            "deca_model_dir": str(DECA_MODEL_DIR),
        }

    def reconstruct(
        self,
        image_bgr: np.ndarray,
    ) -> dict[str, Any] | None:
        """
        Run DECA reconstruction without renderer.

        Important:
        - DECA repo TestData feeds image tensors in 0..1 range.
        - Do not normalize to -1..1.
        - Do not project landmarks to image pixels here.
        - Projection is handled in deca_provider.py using crop_box.
        """

        if image_bgr is None:
            raise ValueError("image_bgr is None")

        if not self.load():
            return None

        try:
            import cv2
            import torch

            h, w = image_bgr.shape[:2]

            rgb = cv2.cvtColor(
                image_bgr,
                cv2.COLOR_BGR2RGB,
            )

            rgb_224 = cv2.resize(
                rgb,
                (224, 224),
                interpolation=cv2.INTER_AREA,
            )

            # DECA repo uses image / 255.0, not [-1, 1].
            img = rgb_224.astype(np.float32) / 255.0

            tensor = torch.from_numpy(
                img.transpose(2, 0, 1),
            ).float().unsqueeze(0).to(self.device)

            with torch.no_grad():
                codedict = self.deca.encode(
                    tensor,
                )

                decoded = self.deca.decode(
                    codedict,
                    rendering=False,
                    use_detail=False,
                    vis_lmk=False,
                    return_vis=False,
                )

                if isinstance(decoded, tuple):
                    opdict = decoded[0]
                else:
                    opdict = decoded

            def tensor_to_numpy(value):
                if value is None:
                    return None

                if hasattr(value, "detach"):
                    value = value.detach().cpu().numpy()

                return value

            vertices = tensor_to_numpy(
                opdict.get("verts"),
            )

            trans_verts = tensor_to_numpy(
                opdict.get("trans_verts"),
            )

            landmarks2d = tensor_to_numpy(
                opdict.get("landmarks2d"),
            )

            landmarks3d = tensor_to_numpy(
                opdict.get("landmarks3d"),
            )

            try:
                faces = self.deca.flame.faces_tensor.detach().cpu().numpy()
            except Exception:
                try:
                    faces = self.deca.flame.faces
                except Exception:
                    faces = np.zeros((0, 3), dtype=np.int32)

            if vertices is not None and vertices.ndim == 3:
                vertices = vertices[0]

            if trans_verts is not None and trans_verts.ndim == 3:
                trans_verts = trans_verts[0]

            if landmarks2d is not None and landmarks2d.ndim == 3:
                landmarks2d = landmarks2d[0]

            if landmarks3d is not None and landmarks3d.ndim == 3:
                landmarks3d = landmarks3d[0]

            if vertices is None:
                vertices = np.zeros((0, 3), dtype=np.float32)

            if trans_verts is None:
                trans_verts = np.zeros((0, 3), dtype=np.float32)

            if landmarks2d is None:
                landmarks2d = np.zeros((0, 2), dtype=np.float32)

            if landmarks3d is None:
                landmarks3d = np.zeros((0, 3), dtype=np.float32)

            if faces is None:
                faces = np.zeros((0, 3), dtype=np.int32)

            faces = np.asarray(
                faces,
                dtype=np.int32,
            )

            flame_params = {}

            for key in [
                "shape",
                "exp",
                "pose",
                "cam",
                "light",
                "tex",
                "detail",
            ]:
                if key in codedict:
                    val = tensor_to_numpy(
                        codedict[key],
                    )

                    if val is not None:
                        flame_params[key] = val.tolist()

            self.last_error = None

            return {
                "vertices": vertices.astype(np.float32),
                "trans_verts": trans_verts.astype(np.float32),
                "faces": faces,

                # Raw DECA outputs. Projection is not done here.
                "landmarks2d": landmarks2d.astype(np.float32),
                "landmarks3d": landmarks3d.astype(np.float32),

                "depth_map": None,

                "flame_params": flame_params,

                "camera": {
                    "source": "deca_cam_params",
                    "image_width": int(w),
                    "image_height": int(h),
                },

                "renderer_enabled": self.renderer_enabled,
            }

        except Exception as e:
            traceback.print_exc()
            self.last_error = str(e)
            return None


_deca_runtime = DecaRuntime()


def get_deca_runtime() -> DecaRuntime:
    return _deca_runtime
