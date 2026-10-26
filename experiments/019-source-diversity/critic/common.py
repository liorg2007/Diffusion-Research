"""Critic helpers (post-hoc). Imports E10's run.py read-only; never writes outside critic/."""
import importlib.util, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
E10 = HERE.parent
spec = importlib.util.spec_from_file_location("e10run", E10 / "run.py")
R = importlib.util.module_from_spec(spec); sys.modules["e10run"] = R; spec.loader.exec_module(R)
ORDER, ALLGENS = R.ORDER, R.ALLGENS
JR = R.JR

def load(T, feat="F_b8", var=""):
    return R.file_loader(feat, tuple(T), var)

def auroc(sr, sf):
    sr, sf = np.asarray(sr, float), np.asarray(sf, float)
    M = (sf[None, :] > sr[:, None]).mean() + 0.5 * (sf[None, :] == sr[:, None]).mean()
    return float(M)
