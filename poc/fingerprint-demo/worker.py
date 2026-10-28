"""Persistent inference worker. One JSON object per stdin line -> one JSON object per stdout line (logs go to stderr).

    env/.venv/bin/python -I poc/fingerprint-demo/worker.py [--device cpu]
Request : {"id": "...", "image_path": "/abs/path" | "image_base64": "...", "condition": "clean|jpeg75_resize05|social_sim|screenshot_sim", "key": optional str}
Response: {"id", "ok", "generators":[{name,label,prob}...sorted], "top", "top_prob", "uncertain", "ai":{...}, "timing", "preview", ...}
The pipeline is the cached one: EXIF-orient + RGB -> fov512 (short side to 512, bicubic, center crop 512; experiments/lib/preprocess.fov_resize)
-> experiments/lib/perturb.apply_condition(img, cond, key, seed=0) -> DINOv2-base (blocks 4/8/12, CLS+mean) on the 224 center crop.
"""
import argparse
import base64
import hashlib
import io
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as C  # noqa: E402

C.setup_env()

import numpy as np  # noqa: E402
from PIL import Image, ImageOps  # noqa: E402

SIZE = 512
SEED = 0
MAX_BYTES = 40 * 1024 * 1024


def err(*a):
    print(*a, file=sys.stderr, flush=True)


def softmax(z):
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def fov512(img):
    """Same operation as preprocess.process_image(mode='fov'): EXIF-oriented RGB, short side -> 512, center crop 512.
    Returns (crop, upscaled_flag). The validated pipeline SKIPS images with short side < 512; the demo upscales them (flagged)."""
    from preprocess import center_crop, fov_resize
    up = min(img.size) < SIZE
    return center_crop(fov_resize(img, SIZE), SIZE), up


class Engine:
    def __init__(self, device=None):
        import torch
        from features import embed_images, get_device, load_backbone
        self.torch, self.embed_images = torch, embed_images
        t0 = time.time()
        self.bb = load_backbone("hf:dinov2:facebook/dinov2-base", input_size=224, layers=(4, 8, 12), use_cls=True, use_mean=True)
        self.device = str(get_device(device))
        self.bb.model.to(self.device)
        a = np.load(C.MODELS / "attribution_model.npz")
        d = np.load(C.MODELS / "detector_model.npz")
        self.a = {k: a[k] for k in a.files}
        self.d = {k: d[k] for k in d.files}
        self.meta = json.loads((C.MODELS / "meta.json").read_text())
        self.load_s = time.time() - t0

    def features(self, imgs):
        return self.embed_images(self.bb, imgs, batch_size=len(imgs), device=self.device)

    def score(self, f):
        """f: [4608] cached-layout feature vector -> (attribution probs [8], ai logit)."""
        za = ((f - self.a["mean"]) / self.a["scale"]) @ self.a["coef"].T + self.a["intercept"]
        s0, s1 = [int(v) for v in self.d["feature_slice"]]
        zd = ((f[s0:s1] - self.d["mean"]) / self.d["scale"]) @ self.d["coef"] + float(self.d["intercept"])
        return softmax(za), float(zd)

    def run(self, req):
        t0 = time.time()
        cond = req.get("condition", "clean")
        if cond not in C.CONDITIONS:
            raise ValueError(f"unknown condition {cond!r}")
        if req.get("image_path"):
            raw = Path(req["image_path"]).read_bytes()
        elif req.get("image_base64"):
            raw = base64.b64decode(req["image_base64"])
        else:
            raise ValueError("no image")
        if len(raw) > MAX_BYTES:
            raise ValueError("image too large")
        key = req.get("key") or hashlib.sha256(raw).hexdigest()
        warnings = []
        im = Image.open(io.BytesIO(raw))
        im.load()
        im = ImageOps.exif_transpose(im).convert("RGB")
        w0, h0 = im.size
        if min(w0, h0) < 64:
            raise ValueError("image is too small (< 64 px)")
        crop, up = fov512(im)
        if up:
            warnings.append(f"Short side {min(w0, h0)} px is below 512: the image was upscaled. The validated pipeline skips such images, so the result is outside the tested regime.")
        t1 = time.time()
        from perturb import apply_condition
        lau = apply_condition(crop, cond, key=key, seed=SEED)
        t2 = time.time()
        f = self.features([lau])[0]
        t3 = time.time()
        pa, zd = self.score(f)
        order = np.argsort(-pa)
        meta_a = self.meta["attribution"]
        tau = float(self.a["msp_threshold"])
        thr = float(self.d["threshold_logit"])
        pv = lau.copy()
        pv.thumbnail((256, 256), Image.BICUBIC)
        buf = io.BytesIO()
        pv.save(buf, "JPEG", quality=88)
        return dict(
            id=req.get("id"), ok=True, condition=cond, original_size=[w0, h0], laundered_size=list(lau.size),
            generators=[dict(name=C.GENERATORS[i], label=C.GEN_LABELS[C.GENERATORS[i]], prob=float(pa[i])) for i in order],
            top=C.GENERATORS[int(order[0])], top_label=C.GEN_LABELS[C.GENERATORS[int(order[0])]], top_prob=float(pa[order[0]]),
            uncertain=bool(pa[order[0]] < tau), msp_threshold=tau,
            ai=dict(logit=zd, prob=float(1 / (1 + np.exp(-zd))), threshold_logit=thr, threshold_prob=float(1 / (1 + np.exp(-thr))),
                    verdict="ai" if zd >= thr else "not_flagged"),
            timing=dict(prep_ms=round((t1 - t0) * 1000, 1), launder_ms=round((t2 - t1) * 1000, 1), forward_ms=round((t3 - t2) * 1000, 1),
                        total_ms=round((time.time() - t0) * 1000, 1)),
            preview="data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode(), device=self.device, warnings=warnings,
            features=[float(x) for x in f] if req.get("return_features") else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default=None)
    ap.add_argument("--selfcheck", action="store_true", help="parity vs cached features on 20 test rows, then exit")
    a = ap.parse_args()
    eng = Engine(a.device)
    err(f"worker ready on {eng.device} (model load {eng.load_s:.1f}s)")
    if a.selfcheck:
        return selfcheck(eng)
    # warm-up so the first request is fast
    eng.features([Image.new("RGB", (224, 224), (127, 127, 127))])
    print(json.dumps(dict(event="ready", device=eng.device, load_s=round(eng.load_s, 1))), flush=True)
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        rid = None
        try:
            req = json.loads(line)
            rid = req.get("id")
            out = eng.run(req)
        except Exception as e:  # noqa: BLE001
            out = dict(id=rid, ok=False, error=f"{type(e).__name__}: {e}")
        print(json.dumps(out), flush=True)


def selfcheck(eng):
    """Live extraction on the fov512 crop PNGs of 20 cached test rows vs the cache-based expected outputs (models/parity_expected.json)."""
    exp = json.loads((C.MODELS / "parity_expected.json").read_text())
    dmax_p, dmax_l, n = 0.0, 0.0, 0
    per = {}
    for row in exp:
        for cond, e in row["conds"].items():
            out = eng.run(dict(image_path=str(C.ROOT / row["path"]), condition=cond, key=row["sha"]))
            pa = np.array([g["prob"] for g in sorted(out["generators"], key=lambda g: C.GENERATORS.index(g["name"]))])
            dp = float(np.abs(pa - np.array(e["attr_probs"])).max())
            dl = abs(out["ai"]["logit"] - e["ai_logit"])
            per[cond] = (max(per.get(cond, (0, 0))[0], dp), max(per.get(cond, (0, 0))[1], dl))
            dmax_p, dmax_l, n = max(dmax_p, dp), max(dmax_l, dl), n + 1
    res = dict(n_checks=n, n_rows=len(exp), device=eng.device, max_abs_diff_attr_prob=dmax_p, max_abs_diff_ai_logit=dmax_l,
               per_condition={k: dict(max_attr_prob=v[0], max_ai_logit=v[1]) for k, v in per.items()})
    (C.MODELS / "parity_result.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
