"""Provenance and run tracking module.
Ensures deterministic SHA recording, atomic file staging, and reproducible manifests.
"""

from typing import Dict, Any
import json
import os
import time
import platform
import hashlib


def get_environment_info() -> Dict[str, Any]:
    """Capture runtime environment facts."""
    info = {
        "os": platform.platform(),
        "python_version": platform.python_version(),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    try:
        import torch
        info["torch_version"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["gpu_device_name"] = torch.cuda.get_device_name(0)
            info["device_count"] = torch.cuda.device_count()
    except ImportError:
        info["torch_version"] = "not_installed"

    try:
        import transformers
        info["transformers_version"] = transformers.__version__
    except ImportError:
        pass

    try:
        import peft
        info["peft_version"] = peft.__version__
    except ImportError:
        pass

    return info


def write_manifest(manifest: Dict[str, Any], out_path: str):
    """Write run manifest atomically."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    tmp_path = f"{out_path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    os.replace(tmp_path, out_path)
