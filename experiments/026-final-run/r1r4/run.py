"""026-final-run/r1r4: docs/11 (D-020) challenges R1 (generality across backbones) and R4 (scaling with data).

    env/.venv/bin/python -I experiments/026-final-run/r1r4/run.py extract_all      # GPU: per-block CLS+mean features for 5 backbones x 5 conditions, then D-008 (resumable)
    env/.venv/bin/python -I experiments/026-final-run/r1r4/run.py probe_all        # CPU: T-mix probe per block (020 closed_task, imported), parity, permutation gates
    env/.venv/bin/python -I experiments/026-final-run/r1r4/run.py r4               # learning curves
    env/.venv/bin/python -I experiments/026-final-run/r1r4/run.py report           # results.json, tables.md, profiles.png

Sub-stages: extract1 <bb>, guard1 <bb>, cpuref <bb> <tag>, probe1 <bb>, perm1 <bb>, parity.
Frozen constants: config.json. Protocol = experiments/020-attr-foundation/run.py imported by path (same rows, folds, probe, bootstrap).
Caches (git-ignored): r1r4/cache/. No downloads (HF_HUB_OFFLINE=1).
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import argparse
import hashlib
import importlib.util
import json
import pickle
import subprocess
import sys
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "lib"))
sys.path.insert(0, str(ROOT / "experiments" / "002-baselines"))
sys.path.insert(0, str(ROOT / "experiments" / "004-phase4-features"))
os.chdir(ROOT)
os.environ.setdefault("HF_HOME", str(ROOT / "data" / "hf_cache"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np
import pandas as pd
from PIL import Image

warnings.filterwarnings("ignore")
CFG = json.loads((HERE / "config.json").read_text())
SEED, B = CFG["seed"], CFG["n_boot"]
CONDS, TMIX, PRIMARY, UNSEEN = CFG["conditions"], CFG["tmix"], CFG["primary"], CFG["unseen"]
BBS = list(CFG["backbones"])
CACHE, RES = HERE / "cache", HERE / "results"
PLATEAU_B_TOL = 0.05          # CI of b "includes 0" iff its 2.5th percentile <= 0.05 (b is bounded below by 0; n^-0.05 changes <9% over n = 15..96, i.e. indistinguishable from flat)
MOD020 = ROOT / "experiments" / "020-attr-foundation" / "run.py"


def log(*a):
    print(time.strftime("%H:%M:%S"), "R1R4", *a, flush=True)


def safe(c):
    return "".join(ch if ch.isalnum() or ch in "-_.+()" else "_" for ch in c)


# ============================================================================ 020 protocol (imported unchanged)
def load020():
    if "m020" not in sys.modules:
        spec = importlib.util.spec_from_file_location("m020", MOD020)
        m = importlib.util.module_from_spec(spec)
        sys.modules["m020"] = m
        spec.loader.exec_module(m)
    return sys.modules["m020"]


def build_rows():
    """The 1,020 fakes of 020 ha1_main (same selection and order); folds read from the committed folds_ha1_main.json and cross-checked against make_folds."""
    m = load020()
    cfg = m.CFG["analyses"]["ha1_main"]
    df, u = m.load_universe(cfg["universe"])
    sel = np.ones(len(df), bool)
    for col, vals in cfg["select"].items():
        sel &= df[col].isin(vals).to_numpy()
    idx = np.where(sel)[0]
    cls = {c: i for i, c in enumerate(cfg["classes"])}
    y = np.array([cls[c] for c in df[cfg["class_column"]].iloc[idx]])
    groups = df.grp.to_numpy()[idx]
    sub = df.iloc[idx].reset_index(drop=True)
    fj = json.loads((ROOT / CFG["folds_file"]).read_text())
    assert fj["rows"] == sub.sha256.tolist(), "rows differ from the committed folds file"
    fold = np.array(fj["fold"])
    assert (fold == m.make_folds(y, groups, m.CFG["outer_folds"], m.SEED)).all(), "committed folds != make_folds"
    chk = m.check_grouped_folds(fold, groups)
    assert chk["ok"], chk
    assert u["conditions"] == CONDS and u["tmix"] == TMIX and cfg["primary"] == PRIMARY
    assert sub.sha256.is_unique and len(sub) == 1020 and not sub.dup_drop.any()
    return m, cfg, y, groups, fold, sub


def make_block_analysis(m, name, F, y, classes, groups, fold):
    """Analysis whose feature set 'blk<l>' is block l of F = {cond: [L, n, 2D]} (a view; no copy). Registers blk0.. in 020's SL so grid_for() uses the dino C grid."""
    class BlockAnalysis(m.Analysis):
        def getX(self, fs, cond):
            return self.XF[cond][int(fs[3:])]
    L = next(iter(F.values())).shape[0]
    for l in range(max(L, 24)):
        m.SL[f"blk{l}"] = slice(0, None)
    return BlockAnalysis(name, {}, y, classes, groups, fold, list(F), TMIX, PRIMARY, F)


# ============================================================================ images
def load_cond(path, sha, cond):
    from perturb import apply_condition
    with Image.open(path) as im:
        im.load()
        return apply_condition(im, cond, key=sha, seed=SEED)


def guard_subset(sub, per_gen=None, classes=None):
    """Fixed 32 images: 4 per generator, rng(seed 0) over the sorted shas (same rule as 023)."""
    per_gen = per_gen or CFG["d008"]["per_generator"]
    classes = classes or sorted(set(sub.generator), key=lambda g: g)
    rng = np.random.default_rng(SEED)
    out = []
    for g in ["dalle2", "dalle3", "firefly", "midjourney-v5", "stable-diffusion-2", "stable-diffusion-xl", "sd3_large", "flux"]:
        s = sorted(sub.sha256[sub.generator == g])
        out += [s[i] for i in rng.choice(len(s), per_gen, replace=False)]
    return out


# ============================================================================ per-block hooks (generic; tested on tiny models in selftest)
class BlockHook:
    """Forward hooks on a list of transformer blocks. Per block: CLS = token 0, mean = mean of tokens n_prefix.. (registers skipped), float32.
    Handles (N, L, D) and (L, N, D) block outputs by matching the known token count ntok."""

    def __init__(self, blocks, n_prefix, ntok):
        import torch
        self.torch, self.n_prefix, self.ntok = torch, n_prefix, ntok
        self.feats = [None] * len(blocks)
        self.last = None
        self.n = len(blocks)
        self.hs = [b.register_forward_hook(self._mk(i)) for i, b in enumerate(blocks)]

    def _mk(self, i):
        def hook(mod, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            if h.shape[0] == self.ntok and h.shape[1] != self.ntok:
                h = h.transpose(0, 1)
            assert h.shape[1] == self.ntok, (tuple(h.shape), self.ntok)
            self.feats[i] = self.torch.stack([h[:, 0].float(), h[:, self.n_prefix:].float().mean(1)], 1)       # [N, 2, D]
            if i == self.n - 1:
                self.last = h.detach()
        return hook

    def reset(self):
        self.feats = [None] * self.n
        self.last = None

    def stack(self):
        assert all(f is not None for f in self.feats), "a block hook did not fire"
        return self.torch.stack(self.feats, 1)                                                                     # [N, L, 2, D]

    def close(self):
        for h in self.hs:
            h.remove()


class Extractor:
    """One backbone: prep(PIL)->tensor, fwd(tensor batch)->numpy [N, L, 2, D] (CLS | mean-patch of every block), hookcheck(x)."""

    def __init__(self, bb, device=None):
        import torch
        self.torch, self.bb = torch, bb
        sp = CFG["backbones"][bb]
        self.L, self.D, self.npre, self.ntok = sp["L"], sp["D"], sp["n_prefix"], sp["ntok"]
        torch.backends.cuda.matmul.allow_tf32 = False          # D-008: fp32 everywhere
        torch.backends.cudnn.allow_tf32 = False
        dev = device or os.environ.get("DD_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dev = torch.device(dev)
        self.hook = None
        self.last_bs = sp["batch"]
        if bb == "dinov2":
            import p4features as P
            from features import load_backbone
            self.bbn = load_backbone(P.SPEC, input_size=224, layers=tuple(range(1, self.L + 1)), use_cls=True, use_mean=True)
            self.model = self.bbn.model.to(self.dev).eval()
            self.prep = self.bbn.to_tensor
        elif bb in ("clip_openai", "clip_laion"):
            from features import CLIP, Backbone
            tf = Backbone("tf", torch.nn.Identity(), "hf", *CLIP, input_size=224, crop_mode="crop", layers=(-1,), use_cls=True, use_mean=False)
            self.prep = tf.to_tensor
            from scorers._common import THIRD_PARTY, WEIGHTS, import_by_path
            if bb == "clip_openai":
                clipmod = import_by_path("openai_clip_model", THIRD_PARTY / "CLIP" / "clip" / "model.py")
                sd = torch.jit.load(str(WEIGHTS / "openai-clip-vitl14" / "ViT-L-14.pt"), map_location="cpu").eval().state_dict()
                net = clipmod.build_model(sd).float().eval()
                self.visual = net.visual.to(self.dev).eval()
                del net
            else:
                import open_clip
                from safetensors.torch import load_file
                net = open_clip.create_model("ViT-L-14", pretrained=None)
                net.load_state_dict(load_file(str(WEIGHTS / "laion-commonpool-vitl14" / "open_clip_pytorch_model.safetensors")), strict=True)
                self.visual = net.visual.to(self.dev).float().eval()
                del net
            self.run = lambda x: self.visual(x)
            self.hook = BlockHook(list(self.visual.transformer.resblocks), self.npre, self.ntok)
        elif bb == "bfree":
            from preprocess import center_crop
            from scorers import bfree as mod
            m = mod._load()
            self.mod, self.net, self.dev = mod, m.model.eval(), m.device
            self.prep = lambda im: mod._TF(center_crop(im.convert("RGB"), 224))
            self.run = lambda x: mod._fwd(self.net, x)
            self.hook = BlockHook(list(self.net.model.blocks), self.npre, self.ntok)
        elif bb == "commfor":
            from scorers import communityforensics as mod
            m = mod._load()
            self.mod, self.net, self.dev = mod, m.model.eval(), m.device
            self.prep = lambda im: mod._TF(im.convert("RGB"))
            self.run = lambda x: self.net(x)
            self.hook = BlockHook(list(self.net.blocks), self.npre, self.ntok)
        else:
            raise ValueError(bb)

    def _batch(self, x):
        torch = self.torch
        with torch.no_grad():
            x = x.to(self.dev)
            if self.bb == "dinov2":
                f = self.bbn.forward(x).float().reshape(len(x), self.L, 2, self.D)
            else:
                self.hook.reset()
                self.run(x)
                f = self.hook.stack()
        assert f.shape[1:] == (self.L, 2, self.D), tuple(f.shape)
        assert str(f.device).startswith(self.dev.type), "features left the chosen device (silent fallback?)"
        return f.cpu().numpy()

    def fwd(self, X, bs=None):
        """X: [N, 3, H, W] CPU tensor -> [N, L, 2, D]. CUDA OOM: halve the batch and retry (other jobs share the GPU); never falls back to CPU."""
        torch = self.torch
        bs = bs or self.last_bs
        out = np.zeros((len(X), self.L, 2, self.D), np.float32)
        j = 0
        while j < len(X):
            try:
                out[j:j + bs] = self._batch(X[j:j + bs])
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if bs == 1:
                    time.sleep(20)
                else:
                    bs = max(1, bs // 2)
                    log(f"{self.bb}: CUDA OOM -> batch {bs}")
                continue
            j += bs
        self.last_bs = bs
        return out

    def close(self):
        if self.hook is not None:
            self.hook.close()
        if hasattr(self, "mod") and hasattr(self.mod, "_M"):
            self.mod._M = None
        for k in ("visual", "model", "net", "mod", "bbn", "run", "prep"):
            if hasattr(self, k):
                setattr(self, k, None)

    def hookcheck(self, X):
        """Max |difference| between the model's own final output and the same quantity rebuilt from the hooked LAST block (proves the hook sits where we think)."""
        torch = self.torch
        with torch.no_grad():
            x = X.to(self.dev)
            if self.bb == "dinov2":
                o = self.model(pixel_values=x, output_hidden_states=True)
                F = self.bbn.forward(x).float().reshape(len(x), self.L, 2, self.D)
                ref = o.last_hidden_state[:, 0]
                mine = self.model.layernorm(F[:, -1, 0])
                hs_diff = float((o.hidden_states[-1][:, 0] - F[:, -1, 0]).abs().max())
                return dict(final=float((ref - mine).abs().max()), hs_last_vs_feat=hs_diff)
            self.hook.reset()
            if self.bb in ("clip_openai", "clip_laion"):
                ref = self.visual(x)
                cls = self.hook.feats[-1][:, 0]
                v = self.visual
                mine = v.ln_post(cls) @ v.proj
            elif self.bb == "bfree":
                ref = self.run(x)[:, 0]
                last = self.hook.last
                mine = self.net.model.forward_head(self.net.model.norm(last))[:, 0]
            else:
                ref = self.run(x)[:, 0]
                last = self.hook.last
                mine = self.net.forward_head(self.net.norm(last))[:, 0]
            return dict(final=float((ref - mine).abs().max()), ref_scale=float(ref.abs().max()))


# ============================================================================ feature cache
def bb_dir(bb):
    return CACHE / bb


def meta_for(bb):
    import timm
    import torch
    sp = CFG["backbones"][bb]
    return dict(bb=bb, L=sp["L"], D=sp["D"], transform=sp["transform"], pipeline=CFG["image_pipeline"], seed=SEED, torch=torch.__version__, timm=timm.__version__)


def cached_keys(bb, cond):
    out = set()
    for p in (bb_dir(bb) / safe(cond)).glob("part_*.npz"):
        out.update(np.load(p, allow_pickle=False)["keys"].tolist())
    return out


def read_cond(bb, cond, shas):
    store = {}
    for p in sorted((bb_dir(bb) / safe(cond)).glob("part_*.npz")):
        z = np.load(p, allow_pickle=False)
        store.update(zip(z["keys"].tolist(), z["feats"]))
    miss = [s for s in shas if s not in store]
    assert not miss, f"{bb}/{cond}: {len(miss)} rows missing in the cache"
    return np.stack([store[s] for s in shas])                              # [n, L, 2, D]


def load_F(bb, shas):
    """{cond: [L, n, 2D]} contiguous per block, layout [CLS | mean] like the 004 slices."""
    F = {}
    for c in CONDS:
        X = read_cond(bb, c, shas)
        n, L, _, D = X.shape
        F[c] = np.ascontiguousarray(X.transpose(1, 0, 2, 3).reshape(L, n, 2 * D))
    return F


def stage_extract_one(bb, limit=None):
    m, cfg, y, groups, fold, sub = build_rows()
    if limit:
        sub = sub.iloc[:limit]
    d = bb_dir(bb)
    d.mkdir(parents=True, exist_ok=True)
    mp = d / "meta.json"
    meta = meta_for(bb)
    if mp.exists() and json.loads(mp.read_text()) != meta:
        raise SystemExit(f"cache config mismatch for {bb}; delete {d}")
    mp.write_text(json.dumps(meta, indent=1, sort_keys=True))
    ex = Extractor(bb)
    assert ex.dev.type == "cuda" or os.environ.get("DD_DEVICE") == "cpu", f"expected CUDA, got {ex.dev}"
    import torch
    chunk = CFG["extract"]["chunk"]
    tim = {}
    with ThreadPoolExecutor(CFG["extract"]["workers"]) as pool:
        for cond in CONDS:
            have = cached_keys(bb, cond)
            todo = sub[~sub.sha256.isin(have)]
            t0 = time.time()
            for s in range(0, len(todo), chunk):
                ch = todo.iloc[s:s + chunk]
                X = torch.stack(list(pool.map(lambda r: ex.prep(load_cond(r[0], r[1], cond)), zip(ch.path, ch.sha256))))
                Fe = ex.fwd(X)
                assert np.isfinite(Fe).all()
                cd = d / safe(cond)
                cd.mkdir(exist_ok=True)
                k = len(list(cd.glob("part_*.npz")))
                np.savez(cd / f"tmp_part_{k:05d}.npz", keys=np.array(ch.sha256.tolist()), feats=Fe)
                os.replace(cd / f"tmp_part_{k:05d}.npz", cd / f"part_{k:05d}.npz")
                log(f"  {bb}/{cond}: {min(s + chunk, len(todo))}/{len(todo)} new ({len(have)} cached) bs={ex.last_bs}")
            tim[cond] = dict(n_new=len(todo), seconds=round(time.time() - t0, 1))
    (d / "timings.json").write_text(json.dumps(tim, indent=1))
    # hook-placement check on 8 real images (clean) -------------------------------------------------
    sel = []
    for g in ("dalle3", "stable-diffusion-xl", "sd3_large", "firefly"):
        sel += list(sub[sub.generator == g].head(2).itertuples())
    X = torch.stack([ex.prep(load_cond(r.path, r.sha256, "clean")) for r in sel])
    hc = ex.hookcheck(X)
    (d / "hookcheck.json").write_text(json.dumps(hc))
    log(f"{bb}: hookcheck {hc}")


# ============================================================================ D-008
def scaled_diff(A, C):
    """A, C: [c, n, L, 2, D] (A = accelerator, C = CPU reference). max over everything of |A-C| / max(1, max_i |C_ij|) (column scale over the images)."""
    sc = np.maximum(1.0, np.abs(C).max(axis=1, keepdims=True))
    return float((np.abs(A - C) / sc).max())


def d008_decision(gpu_vs_a, gpu_vs_b, spread, floor=1e-3):
    tol = max(floor, 2.0 * spread)
    return dict(tol=tol, ok=bool(gpu_vs_a <= tol and gpu_vs_b <= tol))


def stage_cpuref(bb, tag):
    import torch
    run = CFG["d008"]["cpu_runs"][tag]
    torch.set_num_threads(run["threads"])
    m, cfg, y, groups, fold, sub = build_rows()
    shas = guard_subset(sub)
    dfi = sub.set_index("sha256", drop=False)
    ex = Extractor(bb, device="cpu")
    out = {}
    for ci, cond in enumerate(CONDS):
        t0 = time.time()
        X = torch.stack([ex.prep(load_cond(dfi.path[s], s, cond)) for s in shas])
        out[f"c{ci}"] = ex.fwd(X, bs=run["batch"])
        log(f"cpuref {bb}/{tag}/{cond} {time.time() - t0:.0f}s")
    np.savez(bb_dir(bb) / f"cpuref_{tag}.tmp.npz", keys=np.array(shas), **out)
    os.replace(bb_dir(bb) / f"cpuref_{tag}.tmp.npz", bb_dir(bb) / f"cpuref_{tag}.npz")


def stage_guard_one(bb):
    m, cfg, y, groups, fold, sub = build_rows()
    shas = guard_subset(sub)
    for tag in CFG["d008"]["cpu_runs"]:
        p = bb_dir(bb) / f"cpuref_{tag}.npz"
        if not p.exists():
            log(f"{bb}: CPU reference run {tag}")
            env = dict(os.environ, DD_DEVICE="cpu", CUDA_VISIBLE_DEVICES="")
            r = subprocess.run([sys.executable, "-I", str(HERE / "run.py"), "cpuref", bb, tag], env=env)
            if r.returncode != 0:
                raise SystemExit(f"cpuref {bb} {tag} failed")
    ref = {t: np.load(bb_dir(bb) / f"cpuref_{t}.npz", allow_pickle=False) for t in CFG["d008"]["cpu_runs"]}
    assert ref["A"]["keys"].tolist() == shas
    G = np.stack([read_cond(bb, c, shas) for c in CONDS])
    CA = np.stack([ref["A"][f"c{i}"] for i in range(len(CONDS))])
    CB = np.stack([ref["B"][f"c{i}"] for i in range(len(CONDS))])
    ga, gb, sp = scaled_diff(G, CA), scaled_diff(G, CB), scaled_diff(CA, CB)
    dec = d008_decision(ga, gb, sp, 1e-3)
    per_block = [float(max((np.abs(G[:, :, l] - CA[:, :, l]) / np.maximum(1.0, np.abs(CA[:, :, l]).max(axis=1, keepdims=True))).max(), 0)) for l in range(G.shape[2])]
    res = dict(n=len(shas), gpu_vs_cpuA=ga, gpu_vs_cpuB=gb, cpu_spread_AB=sp, per_block_scaled_gpu_vs_A=per_block, max_abs_gpu_vs_A=float(np.abs(G - CA).max()), **dec)
    gp = RES / "d008_guard.json"
    cur = json.loads(gp.read_text()) if gp.exists() else {}
    cur[bb] = res
    gp.write_text(json.dumps(cur, indent=1))
    log(f"D-008 {bb}: gpu-vs-cpuA {ga:.2e} gpu-vs-cpuB {gb:.2e} cpu spread {sp:.2e} tol {dec['tol']:.2e} -> {'ok' if dec['ok'] else 'FAIL'}")
    (bb_dir(bb) / "d008_ok.json").write_text(json.dumps(res))
    if not dec["ok"]:
        raise SystemExit(f"D-008 ABORT for {bb}")


def sub_run(args, env=None):
    r = subprocess.run([sys.executable, "-I", str(HERE / "run.py")] + args, env=env)
    return r.returncode


def stage_extract_all(bbs):
    RES.mkdir(exist_ok=True)
    for bb in bbs:
        if (bb_dir(bb) / "d008_ok.json").exists() and (bb_dir(bb) / "hookcheck.json").exists():
            log(f"{bb}: done already")
            continue
        for stage in ("extract1", "guard1"):
            for attempt in range(3):
                rc = sub_run([stage, bb])
                if rc == 0:
                    break
                log(f"{bb} {stage} failed (rc {rc}), attempt {attempt + 1}")
                if stage == "guard1" and (bb_dir(bb) / "cpuref_A.npz").exists() and not (bb_dir(bb) / "d008_ok.json").exists() and rc != 0:
                    pass
                time.sleep(10)
            else:
                (bb_dir(bb) / f"{stage}_FAILED").write_text("failed 3x")
                log(f"{bb}: {stage} failed 3 times; stop and ask")
                break


# ============================================================================ helpers on profiles
def mid_blocks(L):
    """0-based indices of blocks with 0.25 <= (idx+1)/L <= 0.75."""
    lo, hi = CFG["mid_range"]
    return [l for l in range(L) if lo - 1e-9 <= (l + 1) / L <= hi + 1e-9]


def pooled_block(res, A, l, f_range=None):
    """{cond: [n, K]} OOF log-probs of block l from {(l, f): task result}."""
    out = {c: np.full((A.n, A.K), np.nan, np.float32) for c in A.conds}
    for f in range(5):
        r = res[(l, f)]
        for c in A.conds:
            out[c][r["te"]] = r["logp"][c]
    for c in A.conds:
        assert not np.isnan(out[c]).any()
    return out


def nested_pool(res, A, blocks):
    """Per outer fold the block of `blocks` with the best inner-CV bacc (train folds only; ties -> earlier block); pooled OOF log-probs + choices."""
    out = {c: np.full((A.n, A.K), np.nan, np.float32) for c in A.conds}
    choice, inner = {}, {}
    for f in range(5):
        best = max(blocks, key=lambda l: (round(res[(l, f)]["inner_best"], 12), -l))
        choice[f], inner[f] = int(best), float(res[(best, f)]["inner_best"])
        for c in A.conds:
            out[c][res[(best, f)]["te"]] = res[(best, f)]["logp"][c]
    for c in A.conds:
        assert not np.isnan(out[c]).any()
    return out, choice, inner


def bacc_boot(m, Wrow, y, logp, K):
    """-> [B+1] array (row 0 = point) of 8-way balanced accuracy."""
    return m.conf_stats(m.boot_conf(Wrow, y, logp.argmax(1), K))["bacc"]


def ci3(m, arr):
    return m.ci(arr)


def r1_verdict(sat, nonref, frozen_pair):
    """sat: {bb: bool}. -> dict(overall, n_satisfied, frozen_pair)."""
    n = sum(bool(sat[b]) for b in nonref)
    overall = "general" if n >= 3 else ("not general" if n <= 1 else "partly general")
    fz = sum(bool(sat[b]) for b in frozen_pair)
    fzv = "general across pretrained backbones" if fz == len(frozen_pair) else ("no" if fz == 0 else f"{fz} of {len(frozen_pair)}")
    return dict(overall=overall, n_satisfied=int(n), n_nonref=len(nonref), frozen_pair=fzv, frozen_pair_n=int(fz))


def gap_sat(d):
    """d = [pt, lo, hi] of (best-mid - last)."""
    return bool(d[0] >= CFG["r1_rule"]["gap_min"] and d[1] > 0)


# ============================================================================ probe stage
def probe_tasks_run(m, an, blocks, workers, cache_file):
    """closed_task (020, unchanged) for every (block, outer fold); resumable pickle."""
    done = pickle.loads(cache_file.read_bytes()) if cache_file.exists() else {}
    tasks = [(an, f"blk{l}", "tmix", f) for l in blocks for f in range(5) if (l, f) not in done]
    step = max(workers * 2, 5)
    for s in range(0, len(tasks), step):
        part = tasks[s:s + step]
        out = m.run_tasks(m.closed_task, part, workers, f"probe {an} {s}/{len(tasks)}")
        for t, r in zip(part, out):
            done[(int(t[1][3:]), t[3])] = r
        cache_file.write_bytes(pickle.dumps(done, protocol=4))
    return done


def stage_probe_one(bb, workers):
    m, cfg, y, groups, fold, sub = build_rows()
    F = load_F(bb, sub.sha256.tolist())
    L = F[CONDS[0]].shape[0]
    A = make_block_analysis(m, bb, F, y, cfg["classes"], groups, fold)
    m._G[bb] = A
    res = probe_tasks_run(m, bb, list(range(L)), workers, CACHE / f"probe_{bb}.pkl")
    assert len(res) == L * 5
    return res


def cached_dino_blocks(m, sub):
    """Cached 004/016/019 DINOv2-base features of blocks 4/8/12: {cond: [3, n, 1536]}."""
    u = m.CFG["universes"]["fov512"]
    F = {}
    for c in CONDS:
        store = m.read_caches(u["caches"], c)
        X = np.stack([store[s] for s in sub.sha256]).astype(np.float32)
        F[c] = np.ascontiguousarray(np.stack([X[:, m.SL[k]] for k in ("b4", "b8", "b12")]))
    return F


def stage_parity(workers):
    """(1) harness: the 020 probe on the CACHED features of blocks 4/8/12 reproduces 020 results.json exactly; (2) fresh GPU DINOv2 features vs cached."""
    m, cfg, y, groups, fold, sub = build_rows()
    Fc = cached_dino_blocks(m, sub)
    A = make_block_analysis(m, "dinocache", Fc, y, cfg["classes"], groups, fold)
    m._G["dinocache"] = A
    res = probe_tasks_run(m, "dinocache", [0, 1, 2], workers, CACHE / "probe_dinocache.pkl")
    r020 = json.loads((ROOT / "experiments/020-attr-foundation/results/results.json").read_text())["analyses"]["ha1_main"]["fs"]
    Wc, inv, ug = m.cluster_weights(A.groups, B, SEED, m.strata_of)
    Wrow = Wc[:, inv]
    out = dict(blocks={})
    for i, k in enumerate(("b4", "b8", "b12")):
        lp = pooled_block(res, A, i)
        mine = {c: float(bacc_boot(m, Wrow, y, lp[c], A.K)[0]) for c in CONDS}
        ref = {c: float(r020[f"tmix|{k}"]["summary"][c]["bacc"][0]) for c in CONDS}
        out["blocks"][k] = dict(mine=mine, r020=ref, max_abs_diff=max(abs(mine[c] - ref[c]) for c in CONDS))
    out["harness_reproduces_020"] = bool(max(v["max_abs_diff"] for v in out["blocks"].values()) < 1e-9)
    # fresh features vs cache
    Ff = load_F("dinov2", sub.sha256.tolist())
    feat = {}
    for c in CONDS:
        d = 0.0
        for i, bl in enumerate(CFG["parity"]["dinov2_blocks"]):
            C = Fc[c][i]
            Gf = Ff[c][bl - 1]
            sc = np.maximum(1.0, np.abs(C).max(0, keepdims=True))
            d = max(d, float((np.abs(Gf - C) / sc).max()))
        feat[c] = d
    out["fresh_vs_cached_scaled_max"] = feat
    out["feature_parity_ok"] = bool(max(feat.values()) <= CFG["parity"]["feature_tol_scaled"])
    (RES / "parity.json").write_text(json.dumps(out, indent=1))
    log(f"parity: harness reproduces 020 = {out['harness_reproduces_020']}; fresh-vs-cache scaled max {max(feat.values()):.2e}")
    return out


def stage_perm_one(bb, workers):
    m, cfg, y, groups, fold, sub = build_rows()
    F = load_F(bb, sub.sha256.tolist())
    L = F[CONDS[0]].shape[0]
    A = make_block_analysis(m, bb, F, y, cfg["classes"], groups, fold)
    m._G[bb] = A
    pf = CACHE / f"perm_{bb}.pkl"
    if pf.exists():
        return pickle.loads(pf.read_bytes())
    P = CFG["perm"]
    fs = f"blk{max(0, round(0.5 * L) - 1)}"
    n = P["n"]
    res = m.run_tasks(m.perm_closed_task, [(bb, s, f, fs, P["C"]) for s in range(n) for f in range(5)], workers, f"perm {bb}")
    accs = []
    for s in range(n):
        yp = np.random.default_rng([SEED, 4242, s]).permutation(A.y)
        pred = np.zeros(A.n, int)
        for (s2, te, pr) in res[s * 5:(s + 1) * 5]:
            assert s2 == s
            pred[te] = pr
        accs.append(m.bacc_simple(yp, pred))
    out = dict(block=fs, perm_bacc=accs)
    pf.write_bytes(pickle.dumps(out))
    return out


def stage_probe_all(workers):
    for bb in BBS:
        while not (bb_dir(bb) / "d008_ok.json").exists():
            if (bb_dir(bb) / "extract1_FAILED").exists() or (bb_dir(bb) / "guard1_FAILED").exists():
                log(f"{bb}: extraction/guard failed; skipping")
                break
            time.sleep(30)
        if not (bb_dir(bb) / "d008_ok.json").exists():
            continue
        stage_probe_one(bb, workers)
        stage_perm_one(bb, workers)
        log(f"{bb}: probe + permutation done")
    stage_parity(workers)


# ============================================================================ R1 analysis
def analyse_r1(m, A, res, Wc, inv, L, bb):
    """Everything R1 needs for one backbone from {(block, fold): task result}. Returns (summary, boots)."""
    Wrow = Wc[:, inv]
    y, K = A.y, A.K
    mids = mid_blocks(L)
    prof = {c: [] for c in CONDS}
    bt = {c: [] for c in CONDS}
    for l in range(L):
        lp = pooled_block(res, A, l)
        for c in CONDS:
            b = bacc_boot(m, Wrow, y, lp[c], K)
            prof[c].append(m.ci(b))
            bt[c].append(b)
    out = dict(L=L, mid_blocks_1based=[l + 1 for l in mids], rel_depth=[(l + 1) / L for l in range(L)], profile={c: prof[c] for c in CONDS},
               chosen_C={str(l): {str(f): res[(l, f)]["C"] for f in range(5)} for l in range(L)},
               inner_cv={str(l): [res[(l, f)]["inner_best"] for f in range(5)] for l in range(L)})
    boots = dict(block=bt)
    for nm, blocks in (("nested_all", list(range(L))), ("nested_mid", mids)):
        lp, choice, inner = nested_pool(res, A, blocks)
        bb_ = {c: bacc_boot(m, Wrow, y, lp[c], K) for c in CONDS}
        boots[nm] = bb_
        out[nm] = dict(bacc={c: m.ci(bb_[c]) for c in CONDS}, choice_1based={str(f): choice[f] + 1 for f in choice}, inner_cv_bacc={str(f): inner[f] for f in inner},
                       mean_inner_cv=float(np.mean(list(inner.values()))))
    gaps = {}
    for c in CONDS:
        last = boots["block"][c][L - 1]
        g_nested = m.ci(boots["nested_mid"][c] - last)
        pts = np.array([boots["block"][c][l][0] for l in mids])
        lbest = mids[int(pts.argmax())]
        g_oracle = m.ci(boots["block"][c][lbest] - last)
        gaps[c] = dict(last_bacc=m.ci(last), nested_mid_minus_last=g_nested, oracle_best_mid_block_1based=lbest + 1, oracle_best_mid_bacc=m.ci(boots["block"][c][lbest]),
                       oracle_minus_last=g_oracle, nested_sat=gap_sat(g_nested), oracle_sat=gap_sat(g_oracle))
    out["gaps"] = gaps
    best_all = int(np.argmax([p[0] for p in prof[PRIMARY]]))
    out["best_block_overall_1based_by_test_point"] = best_all + 1
    return out, boots


def stage_report(workers=1):
    m, cfg, y, groups, fold, sub = build_rows()
    Wc, inv, ug = m.cluster_weights(groups.astype(str), B, SEED, m.strata_of)
    Wrow = Wc[:, inv]
    R = dict(config=CFG, n=len(y), n_clusters=len(ug), classes=cfg["classes"], chance=1 / len(cfg["classes"]), backbones={}, gates={})
    BO = {}
    for bb in BBS:
        pf = CACHE / f"probe_{bb}.pkl"
        if not pf.exists():
            log(f"{bb}: no probe results, skipped")
            continue
        res = pickle.loads(pf.read_bytes())
        L = CFG["backbones"][bb]["L"]
        A = make_block_analysis(m, bb, {c: np.zeros((L, 1, 1), np.float32) for c in CONDS}, y, cfg["classes"], groups, fold)
        out, boots = analyse_r1(m, A, res, Wc, inv, L, bb)
        R["backbones"][bb] = out
        BO[bb] = boots
    # headline vs DINOv2-base --------------------------------------------------------------------
    if "dinov2" in BO:
        for bb in R["backbones"]:
            if bb == "dinov2":
                continue
            R["backbones"][bb]["vs_dinov2"] = {}
            for nm in ("nested_all", "nested_mid"):
                R["backbones"][bb]["vs_dinov2"][nm] = {c: m.ci(BO[bb][nm][c] - BO["dinov2"]["nested_all"][c]) for c in CONDS}
    # verdicts ----------------------------------------------------------------------------------------
    have = [b for b in CFG["nonref"] if b in R["backbones"]]
    if len(have) == len(CFG["nonref"]):
        sat_n = {b: R["backbones"][b]["gaps"][PRIMARY]["nested_sat"] for b in have}
        sat_o = {b: R["backbones"][b]["gaps"][PRIMARY]["oracle_sat"] for b in have}
        R["verdict"] = dict(primary_nested=r1_verdict(sat_n, CFG["nonref"], CFG["frozen_pair"]), sensitivity_oracle=r1_verdict(sat_o, CFG["nonref"], CFG["frozen_pair"]),
                            satisfied_nested=sat_n, satisfied_oracle=sat_o,
                            finetuned_pair=dict(satisfied_nested={b: sat_n[b] for b in CFG["finetuned_pair"]}),
                            dinov2_reference_gap=R["backbones"]["dinov2"]["gaps"][PRIMARY] if "dinov2" in R["backbones"] else None)
    # gates ---------------------------------------------------------------------------------------
    G = R["gates"]
    G["grouped_splits"] = dict(ok=True, folds_sha256=hashlib.sha256(fold.tobytes()).hexdigest(), note="committed folds_ha1_main.json == make_folds == rows; check_grouped_folds ok; assert_disjoint_groups runs inside every closed_task")
    G["sha256_dedupe"] = dict(m.gate_dedupe(sub.crop_sha.tolist(), y.tolist()), raw_sha_unique=bool(sub.sha256.is_unique))
    G["train_conditions"] = m.gate_train_conds(TMIX, UNSEEN)
    d8 = json.loads((RES / "d008_guard.json").read_text()) if (RES / "d008_guard.json").exists() else {}
    G["D-008"] = dict(ok=bool(all(b in d8 and d8[b]["ok"] for b in R["backbones"])), per_backbone={b: dict(gpu_vs_cpuA=d8[b]["gpu_vs_cpuA"], gpu_vs_cpuB=d8[b]["gpu_vs_cpuB"], cpu_spread=d8[b]["cpu_spread_AB"], tol=d8[b]["tol"], ok=d8[b]["ok"]) for b in d8})
    hc = {b: json.loads((bb_dir(b) / "hookcheck.json").read_text()) for b in BBS if (bb_dir(b) / "hookcheck.json").exists()}
    G["hook_placement"] = dict(ok=bool(all(v["final"] <= 1e-3 * max(1.0, v.get("ref_scale", 1.0)) for v in hc.values()) and len(hc) == len(R["backbones"])), per_backbone=hc,
                               note="model's own final output vs the same quantity rebuilt from the hooked last block (|diff| <= 1e-3 x max(1, output scale))")
    for bb in R["backbones"]:
        pf = CACHE / f"perm_{bb}.pkl"
        if pf.exists():
            pr = pickle.loads(pf.read_bytes())
            g = m.gate_permutation(pr["perm_bacc"], R["chance"], CFG["perm"]["mean_tol"], CFG["perm"]["p95_excess"])
            g["block"] = pr["block"]
            G[f"permutation_200.{bb}"] = g
    if (RES / "parity.json").exists():
        par = json.loads((RES / "parity.json").read_text())
        G["dino_parity"] = dict(ok=bool(par["harness_reproduces_020"] and par["feature_parity_ok"]), harness_reproduces_020=par["harness_reproduces_020"], fresh_vs_cached_scaled_max=par["fresh_vs_cached_scaled_max"],
                                probe_fresh_vs_020_note="see backbones.dinov2.profile (blocks 4/8/12) vs 020 b4/b8/b12 in parity.json")
        # fresh-feature probe vs 020 at blocks 4/8/12
        if "dinov2" in R["backbones"]:
            dd = {}
            for k, bl in (("b4", 4), ("b8", 8), ("b12", 12)):
                dd[k] = {c: R["backbones"]["dinov2"]["profile"][c][bl - 1][0] - par["blocks"][k]["r020"][c] for c in CONDS}
            R["dino_fresh_minus_020_bacc"] = dd
    R["gates_ok"] = bool(all(v.get("ok", False) for v in G.values()))
    # R4 ---------------------------------------------------------------------------------------------
    if (RES / "r4.json").exists():
        R["r4"] = json.loads((RES / "r4.json").read_text())
    (RES / "results.json").write_text(json.dumps(R, indent=1, default=float))
    write_tables(R)
    try:
        write_figure(R)
    except Exception as e:  # figure is cosmetic
        log("figure failed:", e)
    log("report written; gates_ok", R["gates_ok"])
    return R


# ============================================================================ tables / figure
def fmt(v, w=3):
    return f"{v[0]:.{w}f} [{v[1]:.{w}f}, {v[2]:.{w}f}]"


def write_tables(R):
    o = []
    o.append("# 026 r1r4: R1 (generality across backbones) and R4 (scaling with data)\n")
    o.append(f"Frozen plan: docs/11 + addendum. Rows: the 1,020 fakes of 020 ha1_main, committed caption-family folds. T-mix probe per block (020 closed_task). 1,000 cluster-bootstrap replicates; point [95% CI]. Chance bacc = {R['chance']:.3f}. All gates ok: {R['gates_ok']}.\n")
    v = R.get("verdict")
    if v:
        o.append("## R1 verdict (docs/11 rule at jpeg75_resize05)\n")
        p = v["primary_nested"]
        o.append(f"- **Overall (inner-CV nested mid block vs last block): {p['overall']}** ({p['n_satisfied']} of {p['n_nonref']} non-reference backbones satisfy gap >= 0.10 with paired CI lo > 0).")
        o.append(f"- **Frozen CLIP pair: {p['frozen_pair']}** ({p['frozen_pair_n']} of 2).")
        o.append(f"- Fine-tuned pair (B-Free, CommFor), reported separately: {v['finetuned_pair']['satisfied_nested']}.")
        s = v["sensitivity_oracle"]
        o.append(f"- Sensitivity (oracle best mid block chosen on the test points, optimistic): overall {s['overall']} ({s['n_satisfied']} of {s['n_nonref']}); frozen pair {s['frozen_pair']}.\n")
        o.append("| backbone | role | L | nested mid bacc | last-block bacc | gap (nested mid - last) | satisfied | oracle best mid block | oracle gap | oracle satisfied |")
        o.append("|---|---|---|---|---|---|---|---|---|---|")
        for bb, d in R["backbones"].items():
            g = d["gaps"][PRIMARY]
            o.append(f"| {bb} | {CFG['backbones'][bb]['role']} | {d['L']} | {fmt(d['nested_mid']['bacc'][PRIMARY])} | {fmt(g['last_bacc'])} | {fmt(g['nested_mid_minus_last'])} | {g['nested_sat']} | "
                     f"b{g['oracle_best_mid_block_1based']} {fmt(g['oracle_best_mid_bacc'])} | {fmt(g['oracle_minus_last'])} | {g['oracle_sat']} |")
        o.append("\nSame gap at the two never-trained launderings (descriptive, nested mid - last):\n")
        o.append("| backbone | screenshot_sim | mavali_medium |")
        o.append("|---|---|---|")
        for bb, d in R["backbones"].items():
            o.append(f"| {bb} | {fmt(d['gaps']['screenshot_sim']['nested_mid_minus_last'])} | {fmt(d['gaps']['mavali_medium']['nested_mid_minus_last'])} |")
    o.append("\n## Headline per backbone (inner-CV block choice; jpeg75_resize05 | screenshot_sim | mavali_medium)\n")
    o.append("| backbone | selection | clean | jpeg75_resize05 | social_sim | screenshot_sim | mavali_medium | blocks chosen per fold (1-based) | vs DINOv2-base nested-all at jpeg75_resize05 |")
    o.append("|---|---|---|---|---|---|---|---|---|")
    for bb, d in R["backbones"].items():
        for nm in ("nested_all", "nested_mid"):
            vs = d.get("vs_dinov2", {}).get(nm, {}).get(PRIMARY)
            o.append(f"| {bb} | {nm} | " + " | ".join(fmt(d[nm]["bacc"][c]) for c in CONDS) + f" | {list(d[nm]['choice_1based'].values())} | {fmt(vs) if vs else '-'} |")
    o.append("\n## Layer profiles (bacc per block, T-mix probe; relative depth = block / L)\n")
    for bb, d in R["backbones"].items():
        o.append(f"### {bb} ({CFG['backbones'][bb]['role']}, L = {d['L']}; mid blocks {d['mid_blocks_1based'][0]}..{d['mid_blocks_1based'][-1]})\n")
        o.append("| block | rel. depth | jpeg75_resize05 | screenshot_sim | mavali_medium | clean | social_sim |")
        o.append("|---|---|---|---|---|---|---|")
        for l in range(d["L"]):
            mark = "*" if (l + 1) in d["mid_blocks_1based"] else ""
            o.append(f"| {l + 1}{mark} | {d['rel_depth'][l]:.2f} | " + " | ".join(fmt(d["profile"][c][l]) for c in (PRIMARY, "screenshot_sim", "mavali_medium", "clean", "social_sim")) + " |")
        o.append("")
    o.append("## Validity gates\n")
    o.append("| gate | ok | detail |")
    o.append("|---|---|---|")
    for k, g in R["gates"].items():
        det = {kk: vv for kk, vv in g.items() if kk != "ok"}
        s = json.dumps(det, default=float)
        o.append(f"| {k} | {'PASS' if g.get('ok') else 'FAIL'} | {s[:600]} |")
    if "dino_fresh_minus_020_bacc" in R:
        o.append("\nDINOv2-base fresh GPU features, probe bacc minus 020's cached-feature result (blocks 4/8/12): " + json.dumps(R["dino_fresh_minus_020_bacc"]))
    if "r4" in R:
        r4 = R["r4"]
        o.append("\n## R4: learning curves (jpeg75_resize05, T-mix; error = 1 - bacc; chance error 0.875)\n")
        o.append("**Extrapolations beyond n ~ 96 are extrapolations: 5 points, 3 parameters; the CI covers sampling noise only, not misspecification of the power law.**\n")
        for nm, d in r4["models"].items():
            o.append(f"### {nm}: {d['backbone']} block {d['block_1based']} ({d['why']})\n")
            o.append("| n per generator (effective) | bacc mean over 5 subsamples | bacc range over subsamples | bootstrap CI of the mean |")
            o.append("|---|---|---|---|")
            for row in d["points"]:
                o.append(f"| {row['label']} ({row['n_eff']:.1f}) | {row['bacc_mean']:.3f} | {row['bacc_min']:.3f}..{row['bacc_max']:.3f} | [{row['ci'][0]:.3f}, {row['ci'][1]:.3f}] |")
            f = d["fit"]
            o.append(f"\nFit error = a n^-b + c: a = {f['a'][0]:.3g} [{f['a'][1]:.3g}, {f['a'][2]:.3g}], **b = {f['b'][0]:.3f} [{f['b'][1]:.3f}, {f['b'][2]:.3f}]**, **c = {f['c'][0]:.3f} [{f['c'][1]:.3f}, {f['c'][2]:.3f}]** "
                     f"({f['n_failed']} of {f['n_boot']} replicate fits failed).")
            o.append(f"- Plateau (CI of b includes 0): **{f['plateau']}**.")
            for N, e in f["extrapolated_bacc"].items():
                o.append(f"- EXTRAPOLATED bacc at n = {N} per generator: {e[0]:.3f} [{e[1]:.3f}, {e[2]:.3f}] (asymptote 1 - c = {1 - f['c'][0]:.3f}).")
            o.append("")
    (RES / "tables.md").write_text("\n".join(o) + "\n")


def write_figure(R):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.2), sharey=True)
    for ax, c in zip(axs, (PRIMARY, "screenshot_sim", "mavali_medium")):
        for bb, d in R["backbones"].items():
            x = d["rel_depth"]
            y = [p[0] for p in d["profile"][c]]
            lo = [p[1] for p in d["profile"][c]]
            hi = [p[2] for p in d["profile"][c]]
            ax.plot(x, y, marker="o", ms=3, label=bb)
            ax.fill_between(x, lo, hi, alpha=0.12)
        ax.axvspan(0.25, 0.75, color="grey", alpha=0.08)
        ax.axhline(R["chance"], color="k", ls=":", lw=1)
        ax.set_title(c)
        ax.set_xlabel("relative depth (block / L)")
    axs[0].set_ylabel("8-way balanced accuracy")
    axs[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(RES / "profiles.png", dpi=130)


# ============================================================================ R4
def lc_curve(ns, err):
    """error = a n^-b + c by scipy curve_fit (bounds a in [0,1e4], b in [0,5], c in [0,1]); NaNs on failure."""
    from scipy.optimize import curve_fit
    ns, err = np.asarray(ns, float), np.asarray(err, float)
    c0 = max(0.0, 0.5 * float(err.min()))
    b0 = 0.5
    a0 = max(1e-3, float(err[0] - c0) * float(ns[0]) ** b0)
    try:
        p, _ = curve_fit(lambda n, a, b, c: a * np.power(n, -b) + c, ns, err, p0=(a0, b0, c0), bounds=([0, 0, 0], [1e4, 5, 1]), maxfev=20000)
        return p
    except Exception:
        return np.full(3, np.nan)


def lc_extrap(p, N):
    return 1.0 - (p[0] * float(N) ** (-p[1]) + p[2])


def lc_bootstrap(ns, bacc, n_boot, seed, extrap=(1000, 10000)):
    """ns: effective n per level (list). bacc: {level index: [S arrays of length B+1]} (row 0 = point, rows 1.. = family-bootstrap replicates, same weights across seeds/levels).
    Replicate r: per level mean over a with-replacement redraw of the S subsamples (for r = 0 all of them). -> dict(a, b, c, extrapolated_bacc, plateau, n_failed)."""
    rng = np.random.default_rng([seed, 5151])
    levels = sorted(bacc)
    pars = np.full((n_boot + 1, 3), np.nan)
    for r in range(n_boot + 1):
        means = []
        for li in levels:
            arrs = bacc[li]
            if r == 0 or len(arrs) == 1:
                idx = range(len(arrs))
            else:
                idx = rng.integers(0, len(arrs), len(arrs))
            means.append(np.mean([arrs[i][r] for i in idx]))
        pars[r] = lc_curve(ns, 1.0 - np.array(means))
    ok = ~np.isnan(pars[1:, 0])

    def ci_(v):
        v = np.asarray(v)
        w = v[1:][~np.isnan(v[1:])]
        return [float(v[0]), float(np.percentile(w, 2.5)), float(np.percentile(w, 97.5))] if len(w) else [float("nan")] * 3
    ex = {}
    for N in extrap:
        v = np.array([lc_extrap(p, N) if not np.isnan(p[0]) else np.nan for p in pars])
        ex[str(N)] = ci_(v)
    cb = ci_(pars[:, 1])
    return dict(a=ci_(pars[:, 0]), b=cb, c=ci_(pars[:, 2]), extrapolated_bacc=ex, plateau=bool(cb[1] <= PLATEAU_B_TOL), n_failed=int((~ok).sum()), n_boot=int(n_boot))


def subsample_rows(y, fold, f, n, s, seed=SEED):
    """Training rows (fold != f), n per generator (all if n is None / larger than available), drawn with rng [seed, 7001, s, n, f]. Same draw for every model."""
    tr = np.where(fold != f)[0]
    if n is None:
        return tr
    rng = np.random.default_rng([seed, 7001, s, n, f])
    out = []
    for k in np.unique(y):
        rows = tr[y[tr] == k]
        out.append(rng.choice(rows, size=min(n, len(rows)), replace=False))
    return np.sort(np.concatenate(out))


def lc_task(args):
    m = load020()
    an, fs, f, n, s = args
    A = m._G[an]
    tr = subsample_rows(A.y, A.fold, f, n, s)
    te = np.where(A.fold == f)[0]
    m.assert_disjoint_groups(A.groups, tr, te, f"lc {an} n={n} s={s} fold {f}")
    assert not (set(tr.tolist()) & set(te.tolist()))
    assert (A.fold[tr] != f).all()
    ctrain, sel = m.regime_conds(A, "tmix")
    table, Cb = m.inner_table(A, fs, tr, ctrain, sel, SEED * 1000 + f + 1, list(range(A.K)))
    mdl = m.fit_model("lr", *m.stack(A, fs, tr, ctrain), Cb)
    lp = {c: m.logp_full(mdl, A.getX(fs, c)[te], A.K) for c in A.conds}
    return dict(an=an, fs=fs, f=f, n=n, s=s, te=te, C=Cb, n_train=len(tr), per_class=np.bincount(A.y[tr], minlength=A.K).tolist(), logp=lp)


def choose_r4_models(R_or_res):
    """R_or_res: {bb: {'res': {(l,f): task}, 'L': L}} -> dict(dinov2=(block0, why), best_nonref=(bb, block0, why)). Train-fold information only."""
    from collections import Counter

    def mode_block(res, blocks):
        ch = []
        inner = {}
        for f in range(5):
            b = max(blocks, key=lambda l: (round(res[(l, f)]["inner_best"], 12), -l))
            ch.append(b)
            inner.setdefault(b, []).append(res[(b, f)]["inner_best"])
        cnt = Counter(ch)
        best = max(cnt, key=lambda b: (cnt[b], float(np.mean(inner[b])), -b))
        return best, dict(cnt), float(np.mean([res[(b, f)]["inner_best"] for f, b in enumerate(ch)]))
    d = R_or_res["dinov2"]
    b0, cnt, _ = mode_block(d["res"], mid_blocks(d["L"]))
    out = dict(dinov2=(b0, f"mode of the outer-fold inner-CV choices among mid blocks {cnt}"))
    cand = {}
    for bb in CFG["nonref"]:
        if bb in R_or_res:
            r = R_or_res[bb]
            b, cnt, mi = mode_block(r["res"], list(range(r["L"])))
            cand[bb] = (mi, b, cnt)
    # selection of the backbone: mean over folds of the best inner-CV bacc over ALL blocks (train folds only)
    sc = {bb: float(np.mean([max(R_or_res[bb]["res"][(l, f)]["inner_best"] for l in range(R_or_res[bb]["L"])) for f in range(5)])) for bb in cand}
    bbest = max(sc, key=sc.get)
    out["best_nonref"] = (bbest, cand[bbest][1], f"backbone with the best mean outer-fold inner-CV bacc {sc}; block = mode of nested-all choices {cand[bbest][2]}")
    return out


def stage_r4(workers):
    m, cfg, y, groups, fold, sub = build_rows()
    probes = {}
    for bb in BBS:
        pf = CACHE / f"probe_{bb}.pkl"
        if pf.exists():
            probes[bb] = dict(res=pickle.loads(pf.read_bytes()), L=CFG["backbones"][bb]["L"])
    assert "dinov2" in probes and all(b in probes for b in CFG["nonref"]), "R1 probes must be finished before R4"
    ch = choose_r4_models(probes)
    models = {"dinov2": ("dinov2", ch["dinov2"][0], ch["dinov2"][1]), "best_nonref": (ch["best_nonref"][0], ch["best_nonref"][1], ch["best_nonref"][2])}
    log("R4 models:", models)
    levels = CFG["r4"]["n_levels"]
    Wc, inv, ug = m.cluster_weights(groups.astype(str), B, SEED, m.strata_of)
    Wrow = Wc[:, inv]
    out = dict(models={}, levels=levels, choice={k: dict(backbone=v[0], block_1based=v[1] + 1, why=v[2]) for k, v in models.items()})
    for key, (bb, blk, why) in models.items():
        F = load_F(bb, sub.sha256.tolist())
        A = make_block_analysis(m, f"r4_{bb}", F, y, cfg["classes"], groups, fold)
        m._G[f"r4_{bb}"] = A
        cf = CACHE / f"r4_{bb}_blk{blk}.pkl"
        done = pickle.loads(cf.read_bytes()) if cf.exists() else {}
        tasks = []
        for li, n in enumerate(levels):
            nn = None if n == "all" else int(n)
            for s in range(1 if nn is None else CFG["r4"]["seeds"]):
                for f in range(5):
                    if (li, s, f) not in done:
                        tasks.append((f"r4_{bb}", f"blk{blk}", f, nn, s, li))
        step = max(workers * 2, 5)
        for st in range(0, len(tasks), step):
            part = tasks[st:st + step]
            r = m.run_tasks(lc_task, [t[:5] for t in part], workers, f"r4 {key} {st}/{len(tasks)}")
            for t, rr in zip(part, r):
                done[(t[5], t[4], t[2])] = rr
            cf.write_bytes(pickle.dumps(done, protocol=4))
        bacc, n_eff, pts = {}, [], []
        for li, n in enumerate(levels):
            nn = None if n == "all" else int(n)
            S = 1 if nn is None else CFG["r4"]["seeds"]
            arrs, neffs = [], []
            for s in range(S):
                lp = np.full((A.n, A.K), np.nan, np.float32)
                for f in range(5):
                    rr = done[(li, s, f)]
                    lp[rr["te"]] = rr["logp"][PRIMARY]
                    neffs += rr["per_class"]
                assert not np.isnan(lp).any()
                arrs.append(bacc_boot(m, Wrow, y, lp, A.K))
            bacc[li] = arrs
            n_eff.append(float(np.mean(neffs)))
            pts_ = [a[0] for a in arrs]
            mean_boot = np.mean([a for a in arrs], 0)
            pts.append(dict(label=str(n), n_eff=n_eff[-1], bacc_mean=float(np.mean(pts_)), bacc_min=float(np.min(pts_)), bacc_max=float(np.max(pts_)),
                            ci=[float(np.percentile(mean_boot[1:], 2.5)), float(np.percentile(mean_boot[1:], 97.5))], per_seed=[float(p) for p in pts_]))
        fit = lc_bootstrap(n_eff, bacc, B, SEED, tuple(CFG["r4"]["extrapolate_n"]))
        out["models"][key] = dict(backbone=bb, block_1based=blk + 1, why=why, points=pts, n_eff=n_eff, fit=fit)
    (RES / "r4.json").write_text(json.dumps(out, indent=1, default=float))
    log("R4 done")


# ============================================================================ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage")
    ap.add_argument("args", nargs="*")
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    RES.mkdir(exist_ok=True)
    CACHE.mkdir(exist_ok=True)
    s, ar = a.stage, a.args
    if s == "extract1":
        stage_extract_one(ar[0], int(ar[1]) if len(ar) > 1 else None)
    elif s == "cpuref":
        stage_cpuref(ar[0], ar[1])
    elif s == "guard1":
        stage_guard_one(ar[0])
    elif s == "extract_all":
        stage_extract_all(ar or BBS)
    elif s == "probe1":
        stage_probe_one(ar[0], a.workers)
    elif s == "perm1":
        stage_perm_one(ar[0], a.workers)
    elif s == "parity":
        stage_parity(a.workers)
    elif s == "probe_all":
        stage_probe_all(a.workers)
    elif s == "r4":
        stage_r4(a.workers)
    elif s == "report":
        stage_report(a.workers)
    else:
        raise SystemExit(f"unknown stage {s}")


if __name__ == "__main__":
    main()
