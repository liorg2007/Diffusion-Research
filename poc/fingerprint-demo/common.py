"""Shared helpers for the fingerprint demo (paths, markdown-table parsing). Python via env/.venv/bin/python -I."""
import json
import os
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
MODELS = HERE / "models"
SAMPLES = HERE / "samples"

GENERATORS = ["dalle2", "dalle3", "firefly", "midjourney-v5", "stable-diffusion-2", "stable-diffusion-xl", "sd3_large", "flux"]
GEN_LABELS = {"dalle2": "DALL-E 2", "dalle3": "DALL-E 3", "firefly": "Adobe Firefly", "midjourney-v5": "Midjourney v5",
              "stable-diffusion-2": "Stable Diffusion 2", "stable-diffusion-xl": "Stable Diffusion XL", "sd3_large": "Stable Diffusion 3",
              "flux": "FLUX"}
CONDITIONS = ["clean", "jpeg75_resize05", "social_sim", "screenshot_sim"]


def setup_env():
    """cwd = repo root, offline HF cache, lib/004 on sys.path (call before importing torch / features)."""
    os.chdir(ROOT)
    os.environ.setdefault("HF_HOME", str(ROOT / "data" / "hf_cache"))
    os.environ["HF_HUB_OFFLINE"] = "1"
    for p in (ROOT / "experiments" / "lib", ROOT / "experiments" / "004-phase4-features"):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))


def rel(p):
    return str(Path(p).resolve().relative_to(ROOT))


def md_tables(path):
    """Parse every markdown table of a file -> list of dict(heading, header, rows) (cells stripped, ** removed)."""
    out, head, cur = [], "", None
    for line in Path(path).read_text().splitlines():
        if line.startswith("#"):
            head = line.lstrip("# ").strip()
        if line.startswith("|"):
            cells = [c.strip().replace("**", "") for c in line.strip().strip("|").split("|")]
            if cur is None:
                cur = dict(heading=head, header=cells, rows=[])
            elif all(re.fullmatch(r":?-{2,}:?", c) for c in cells if c):
                pass
            else:
                cur["rows"].append(cells)
        else:
            if cur is not None:
                out.append(cur)
                cur = None
    if cur is not None:
        out.append(cur)
    return out


def find_table(tables, heading_sub, header0=None, nth=0):
    c = [t for t in tables if heading_sub in t["heading"] and (header0 is None or t["header"][0] == header0)]
    return c[nth]


def num_ci(s):
    """'0.855 [0.834, 0.876]' -> (0.855, 0.834, 0.876); '0.855' -> (0.855, None, None)."""
    m = re.match(r"\s*(-?\d*\.?\d+)\s*(?:\[\s*(-?\d*\.?\d+)\s*,\s*(-?\d*\.?\d+)\s*\])?", s)
    if not m:
        return None, None, None
    return tuple(float(x) if x is not None else None for x in m.groups())
