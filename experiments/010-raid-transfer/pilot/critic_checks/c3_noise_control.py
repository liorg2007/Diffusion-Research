"""Critic check 3 (post hoc, NOT pre-registered; CPU): is the eps8 effect targeted transfer or generic perturbation
sensitivity? Primary-set items only (48 unpadded reals + 400 fakes). Two controls built from the real RAID delta
d = eps8 - clean (integer grey levels, |d| <= 9):
  signflip : clean + random_sign * |d|      (same per-pixel magnitude / L1 / L2 / Linf, structure destroyed)
  shuffled : clean + d of the NEXT id of the same class (structured adversarial-looking pattern, wrong image)
Scores every detector with run.py's own scorers; first rescores 8 clean items to check the harness reproduces the saved
scores. Usage: CUDA_VISIBLE_DEVICES="" env/.venv/bin/python -I c3_noise_control.py [detectors...]"""
import importlib.util
import os
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["DD_DEVICE"] = "cpu"
H = Path(__file__).resolve().parent
RUN = H.parents[1] / "run.py"
spec = importlib.util.spec_from_file_location("e3run", RUN)
run = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run)          # module import only: os.chdir(ROOT), paths, no stage executed
import numpy as np                    # noqa: E402
import pandas as pd                   # noqa: E402
import torch                          # noqa: E402
from PIL import Image                 # noqa: E402

torch.set_num_threads(4)
run.DEVICE = "cpu"
items = run.build_items(run.load_manifest())
items = items[~items.padded].reset_index(drop=True)
saved = pd.read_csv(H.parent / "scores_long.csv.gz", dtype={"id": str})
rng = np.random.default_rng(12345)


def arr(img):
    return np.asarray(img, np.int16)


cache = {}
recs = items.to_dict("records")
by_cls = {c: [r for r in recs if r["cls"] == c] for c in run.CLASSES}
nxt = {}
for c, rs in by_cls.items():
    for j, r in enumerate(rs):
        nxt[r["key"]] = rs[(j + 1) % len(rs)]


def delta(r):
    if r["key"] not in cache:
        cache[r["key"]] = arr(run.make_input(r, "eps8")) - arr(run.make_input(r, "clean"))
    return cache[r["key"]]


signs = {r["key"]: rng.choice(np.array([-1, 1], np.int16), size=(256, 256, 3)) for r in recs}


def make(r, ctl):
    c = arr(run.make_input(r, "clean"))
    d = np.abs(delta(r)) * signs[r["key"]] if ctl == "signflip" else delta(nxt[r["key"]])
    return Image.fromarray(np.clip(c + d, 0, 255).astype(np.uint8))


dets = sys.argv[1:] or ["communityforensics", "b0", "clipdet", "rine"]
outp = H / "c3_noise_scores.csv"
prev = pd.read_csv(outp, dtype={"id": str}) if outp.exists() else pd.DataFrame()
rows = []
for det in dets:
    fn = run.get_fn(det)
    chk = [recs[i] for i in (0, 1, 2, 3, 100, 200, 300, 400)]
    sc = np.asarray(fn([run.make_input(r, "clean") for r in chk]), float).reshape(-1)
    ref = [saved[(saved.detector == det) & (saved.input_set == "clean") & (saved.cls == r["cls"]) & (saved.id == r["id"])].score.item() for r in chk]
    print(det, "harness check max|cpu - saved cuda| =", float(np.abs(sc - np.array(ref)).max()), flush=True)
    for ctl in ("signflip", "shuffled"):
        for i in range(0, len(recs), 16):
            ch = recs[i:i + 16]
            s = np.asarray(fn([make(r, ctl) for r in ch]), float).reshape(-1)
            rows += [dict(detector=det, input_set=ctl, cls=r["cls"], id=r["id"], label=r["label"], score=v) for r, v in zip(ch, s)]
        print(det, ctl, "done", flush=True)
    pd.concat([prev, pd.DataFrame(rows)]).to_csv(outp, index=False)
