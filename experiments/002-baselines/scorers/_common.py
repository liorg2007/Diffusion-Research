"""Shared helpers for the baseline scorers (device choice, MPS->CPU fallback, path-imports of third-party code)."""
import importlib.util
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]          # project root
THIRD_PARTY = ROOT / "third_party"
WEIGHTS = ROOT / "data" / "weights"
os.environ.setdefault("HF_HOME", str(ROOT / "data" / "hf_cache"))   # keep any HF cache inside the project
os.environ.setdefault("HF_HUB_OFFLINE", "1")                          # weights are local; never download at score time


def pick_device(device=None):
    """DD_DEVICE env var > explicit arg > mps (if available) > cpu."""
    device = os.environ.get("DD_DEVICE") or device
    if device:
        return torch.device(device)
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def import_by_path(name, path, package=False):
    """Import a third-party module/package by file path under a private name (no sys.path pollution, no name clashes)."""
    path = Path(path)
    kw = dict(submodule_search_locations=[str(path.parent)]) if package else {}
    spec = importlib.util.spec_from_file_location(name, str(path), **kw)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


class DeviceModel:
    """Holds a model on `device`; if a forward raises on MPS, permanently falls back to CPU (with a warning)."""

    def __init__(self, model, device=None):
        self.model = model.eval()
        self.device = pick_device(device)
        self.model.to(self.device)

    def __call__(self, forward, *tensors):
        """forward(model, *tensors_on_device) -> tensor; returns float32 numpy on CPU."""
        with torch.no_grad():
            try:
                out = forward(self.model, *[t.to(self.device) for t in tensors])
                return out.float().cpu().numpy()
            except RuntimeError as e:
                if self.device.type == "cpu":
                    raise
                warnings.warn(f"{self.device} failed ({str(e)[:120]}...); falling back to CPU permanently")
                self.device = torch.device("cpu")
                self.model.to(self.device)
                out = forward(self.model, *[t.to(self.device) for t in tensors])
                return out.float().cpu().numpy()


def to_rgb(img):
    return img.convert("RGB") if img.mode != "RGB" else img


def batched(n, bs):
    for i in range(0, n, bs):
        yield slice(i, min(i + bs, n))
