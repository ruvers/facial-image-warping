from __future__ import annotations

import sys
from typing import Any

from backend.local_models.registry import DECA_DIR
from backend.local_models.deca_files import get_deca_file_status
from backend.three_d.deca_runtime import get_deca_runtime


def check_deca_imports() -> dict[str, Any]:
    """
    Check DECA Python imports without running reconstruction.
    """

    result = {
        "deca_repo_in_sys_path": False,
        "torch_import": False,
        "deca_import": False,
        "config_import": False,
        "errors": [],
    }

    deca_path = str(DECA_DIR)

    if deca_path not in sys.path:
        sys.path.insert(0, deca_path)

    result["deca_repo_in_sys_path"] = deca_path in sys.path

    try:
        import torch  # noqa: F401
        result["torch_import"] = True
    except Exception as e:
        result["errors"].append(f"torch import failed: {e}")

    try:
        from decalib.deca import DECA  # noqa: F401
        result["deca_import"] = True
    except Exception as e:
        result["errors"].append(f"decalib.deca import failed: {e}")

    try:
        from decalib.utils.config import cfg  # noqa: F401
        result["config_import"] = True
    except Exception as e:
        result["errors"].append(f"decalib config import failed: {e}")

    result["ok"] = (
        result["torch_import"]
        and result["deca_import"]
        and result["config_import"]
    )

    return result


def get_deca_diagnostics() -> dict[str, Any]:
    runtime = get_deca_runtime()

    files = get_deca_file_status()
    imports = check_deca_imports()
    runtime_status = runtime.status()

    return {
        "files": files,
        "imports": imports,
        "runtime": runtime_status,
        "can_attempt_true_3d": bool(
            files["ready"]
            and imports["ok"]
            and runtime_status.get("available", False)
        ),
    }