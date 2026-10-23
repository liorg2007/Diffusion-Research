"""H4 PILOT (docs/04-phase4-plan.md Amendment 5): is a training-free RIGID-style score as format-shortcut-prone as B0?

  env/.venv/bin/python experiments/006-format-shortcut/run.py --dry-run          # ~16 images, tiny bootstrap, own scratch dir
  env/.venv/bin/python experiments/006-format-shortcut/run.py --subdir pilot     # full pilot -> experiments/006-format-shortcut/pilot/

--subdir NAME is both the output directory name and the 004 feature-cache subdir of the `orig` arm (NAME) /
`NAME_fov512` arm. All parameters live in config.json. No criterion is chosen here: see config.json["verdict"].
"""
import argparse
import copy
import hashlib
import importlib.util
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "experiments" / "lib"))
sys.path.insert(0, str(ROOT / "experiments" / "004-phase4-features"))
os.chdir(ROOT)
os.environ.setdefault("HF_HOME", str(ROOT / "data" / "hf_cache"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np
import pandas as pd
import torch
from PIL import Image

import metrics
import probe
import p4features as P
from features import IMAGENET, get_device, load_backbone
from preprocess import center_crop

CFG = json.loads((HERE / "config.json").read_text())
SIGMAS = CFG["rigid"]["sigmas"]
DRAWS = CFG["rigid"]["draws"]
BATCH = CFG["rigid"]["batch"]
FVS = ["post", "pre"]                       # feature variants, index = position in the stored array
PRIMARY_FV = CFG["rigid"]["primary_feature_variant"]
COND = CFG["conditions"]                    # setting -> condition
SETS = ["i", "iii-p", "iv-p"]
ARMS = CFG["arms"]
GENS = CFG["data"]["generators"]
FF_VARIANTS = {"with_firefly": GENS, "without_firefly": [g for g in GENS if g != "firefly"]}
POOLS = [("vision_native", "native"), ("vision_whatsapp", "WA"), ("vision_facebook_hq", "FBH")]
SEED = CFG["aggregation"]["seed"]
FEAT_TOL = CFG["d008"]["feat_tol_scaled"]
SPLITS = {"orig": ROOT / CFG["data"]["splits_orig"], "fov512": ROOT / CFG["data"]["splits_fov512"]}
HIDDEN = 768
T0 = time.time()
LOGF = None


def log(*a):
    s = time.strftime("%H:%M:%S") + " " + " ".join(str(x) for x in a)
    print(s, flush=True)
    if LOGF:
        LOGF.write(s + "\n"); LOGF.flush()


def load_004_run():
    """004's run.py as a module: reuse its load_cond (condition application on a crop file), unchanged."""
    spec = importlib.util.spec_from_file_location("p4run", ROOT / "experiments/004-phase4-features/run.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


# ----------------------------------------------------------------------------- RIGID score
def noise_seed(sha, draw, sigma):
    h = hashlib.sha256(f"{sha}:{draw}:{sigma:.4f}".encode()).digest()
    return int.from_bytes(h[:8], "little") & (2 ** 63 - 1)


def make_noise(sha, draw, sigma, size=224):
    g = torch.Generator(device="cpu")
    g.manual_seed(noise_seed(sha, draw, sigma))
    return torch.randn((3, size, size), generator=g, dtype=torch.float32) * float(sigma)


def to_unit_tensor(img, size=224):
    """Same steps as features.Backbone.to_tensor up to (and excluding) the mean/std normalisation."""
    img = img.convert("RGB")
    if min(img.size) < size:
        raise ValueError(f"image {img.size} smaller than {size}")
    img = center_crop(img, size)
    return torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0).permute(2, 0, 1)


_M, _S = torch.tensor(IMAGENET[0])[:, None, None], torch.tensor(IMAGENET[1])[:, None, None]


def normalise(x):
    return (x - _M.to(x.device)) / _S.to(x.device)


@torch.no_grad()
def cls_feats(model, x_unit, dev, batch=None):
    """x_unit [N,3,224,224] in [0,1] (cpu) -> float32 numpy [N, 2 (post, pre), 768] CLS features."""
    out = []
    batch = batch or BATCH
    for i in range(0, len(x_unit), batch):
        xb = normalise(x_unit[i:i + batch].to(dev))
        o = model(pixel_values=xb, output_hidden_states=True)
        out.append(torch.stack([o.last_hidden_state[:, 0], o.hidden_states[-1][:, 0]], 1).float().cpu())
        del o, xb
    return torch.cat(out).numpy()


def cos_score(f0, fn):
    a, b = f0.astype(np.float64), fn.astype(np.float64)
    cos = (a * b).sum(-1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1))
    return 1.0 - cos                                    # [N, 2]


def noisy(x_unit, shas, draw, sigma):
    n = torch.stack([make_noise(s, draw, sigma) for s in shas])
    return (x_unit + n).clamp_(0.0, 1.0)


def chunk_feats(model, dev, x_unit, shas, sd_pairs):
    """f(x) and f(x + n) for every (sigma_idx, draw) in sd_pairs -> f0 [c,2,768], {pair: [c,2,768]}."""
    f0 = cls_feats(model, x_unit, dev)
    return f0, {(si, d): cls_feats(model, noisy(x_unit, shas, d, SIGMAS[si]), dev) for si, d in sd_pairs}


def score_condition(model, dev, p4, pool, paths, shas, cond):
    """-> scores [n, 2 (feature variant), len(SIGMAS)], the mean over DRAWS draws of 1 - cos(f(x), f(x+n))."""
    res = np.zeros((len(paths), 2, len(SIGMAS)))
    pairs = [(si, d) for si in range(len(SIGMAS)) for d in range(DRAWS)]
    for s in range(0, len(paths), BATCH):
        ps, sh = paths[s:s + BATCH], shas[s:s + BATCH]
        x = torch.stack(list(pool.map(lambda a: to_unit_tensor(p4.load_cond(a[0], a[1], cond)), zip(ps, sh))))
        f0, fn = chunk_feats(model, dev, x, sh, pairs)
        if dev.type == "cuda":
            torch.cuda.empty_cache()
        for (si, d), f in fn.items():
            res[s:s + len(ps), :, si] += cos_score(f0, f) / DRAWS
        if (s // BATCH) % 5 == 0:
            log(f"    {cond}: {min(s + BATCH, len(paths))}/{len(paths)}")
    return res


# ----------------------------------------------------------------------------- D-008 guard + pipeline check
def d008_guard(df, model, dev, p4, out_dir):
    te = df[(df.split == "test")]
    r = sorted(te.sha256[(te.label == 0) & (te.pool == "vision_native")])
    f = sorted(te.sha256[te.label == 1])
    rng = np.random.default_rng(SEED)
    k = CFG["d008"]["n"] // 2
    sub = [r[i] for i in rng.choice(len(r), min(k, len(r)), replace=False)] + \
          [f[i] for i in rng.choice(len(f), min(k, len(f)), replace=False)]
    d = df.set_index("sha256", drop=False).loc[sub]
    x = torch.stack([to_unit_tensor(p4.load_cond(p, s, "clean")) for p, s in zip(d.path, d.sha256)])
    pairs = [(si, 0) for si in range(len(SIGMAS))]
    fa, na = chunk_feats(model, dev, x, sub, pairs)
    cpu_model = copy.deepcopy(model).cpu().eval()
    global BATCH
    ob, BATCH = BATCH, CFG["d008"]["cpu_batch"]
    try:
        fc, nc = chunk_feats(cpu_model, torch.device("cpu"), x, sub, pairs)
    finally:
        BATCH = ob
    del cpu_model
    out = {"n": len(sub), "device": str(dev), "feat_tol": FEAT_TOL, "variants": {}}
    ok = True
    for vi, v in enumerate(FVS):
        A = np.concatenate([fa[:, vi]] + [na[p][:, vi] for p in pairs])
        C = np.concatenate([fc[:, vi]] + [nc[p][:, vi] for p in pairs])
        dd = np.abs(A - C)
        scaled = float((dd / np.maximum(1.0, np.abs(C).max(0, keepdims=True))).max())
        dsc = max(float(np.abs(cos_score(fa, na[p])[:, vi] - cos_score(fc, nc[p])[:, vi]).max()) for p in pairs)
        out["variants"][v] = dict(max_abs_feature_diff=float(dd.max()), max_scaled_feature_diff=scaled,
                                  max_abs_score_diff=dsc, ok=bool(scaled <= FEAT_TOL))
        ok &= scaled <= FEAT_TOL
        log(f"D-008 [{v}]: max|dF|={dd.max():.2e} scaled={scaled:.2e} max|dscore|={dsc:.2e} -> {'ok' if scaled <= FEAT_TOL else 'FAIL'}")
    out["ok"] = bool(ok)
    # pipeline identity vs the 004 feature cache: our unnoised pre-LN CLS must equal primary[:, :768] of the cache
    keys, X = P.load_features("primary", "clean", subdir=args_subdir_for_check["orig"])
    pos = {kk: i for i, kk in enumerate(keys.tolist())}
    cached = np.stack([X[pos[s], :HIDDEN] for s in sub])
    dd = np.abs(fa[:, 1] - cached)
    sc = float((dd / np.maximum(1.0, np.abs(cached).max(0, keepdims=True))).max())
    out["pipeline_vs_004_cache"] = dict(max_abs_diff=float(dd.max()), max_scaled_diff=sc, ok=bool(sc <= FEAT_TOL))
    log(f"pipeline check vs 004 cache (unnoised pre-LN CLS): max|d|={dd.max():.2e} scaled={sc:.2e}")
    out["ok"] = bool(out["ok"] and sc <= FEAT_TOL)
    (out_dir / "d008_guard.json").write_text(json.dumps(out, indent=1))
    if not out["ok"]:
        raise SystemExit(f"D-008 ABORT: {out}")
    return out


# ----------------------------------------------------------------------------- bootstrap plan
class Plan:
    """Cluster bootstrap for reals R (row indices; clusters = device) vs the fakes of `gens` (row indices in Fmask).
    Fakes are resampled by image, stratified per generator. LOGO-macro = mean over generators of AUROC."""

    def __init__(self, df, R, gens, fmask, n_boot, seed=SEED):
        gen = df.generator.to_numpy()
        self.R, self.gens = np.asarray(R), list(gens)
        self.F = {g: np.where(fmask & (gen == g))[0] for g in self.gens}
        strat = np.concatenate([[g] * len(self.F[g]) for g in self.gens])
        dev, sha = df.device.to_numpy(), df.sha256.to_numpy()
        clus = [dev[i] if dev[i] else f"img:{sha[i]}" for i in self.R]
        self.Wr, self.Wf = metrics.bootstrap_weights(clus, strat, n_boot, seed)
        self.cols, o = {}, 0
        for g in self.gens:
            self.cols[g] = slice(o, o + len(self.F[g])); o += len(self.F[g])
        self.allF = np.concatenate([self.F[g] for g in self.gens])
        self.one_r, self.one_f = np.ones((1, len(self.R))), {g: np.ones((1, len(self.F[g]))) for g in self.gens}

    def macro(self, sfun):
        """sfun(g) -> score array over ALL rows for generator g. -> (point, boot[n_boot])."""
        pt, bt = 0.0, 0.0
        for g in self.gens:
            s = np.asarray(sfun(g), float)
            pt += metrics.weighted_auroc(s[self.R], s[self.F[g]], self.one_r, self.one_f[g])[0]
            bt = bt + metrics.weighted_auroc(s[self.R], s[self.F[g]], self.Wr, self.Wf[:, self.cols[g]])
        return float(pt / len(self.gens)), bt / len(self.gens)

    def pooled(self, s):
        s = np.asarray(s, float)
        pt = metrics.weighted_auroc(s[self.R], s[self.allF], self.one_r, np.ones((1, len(self.allF))))[0]
        return float(pt), metrics.weighted_auroc(s[self.R], s[self.allF], self.Wr, self.Wf)


def ci(x):
    return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]


def vc(pt, boot):
    return dict(value=float(pt), ci=ci(boot))


def dc(p1, b1, p2, b2):
    return dict(delta=float(p1 - p2), ci=ci(b1 - b2))


def verdict(auc_i, gaps, b0_gap):
    """Amendment 5 label from point estimates. auc_i, gaps: {sigma: value}; b0_gap: float."""
    best = max(auc_i, key=lambda s: auc_i[s])
    if gaps[best] >= 0.10 and b0_gap < 0.05:
        lab = "SUPPORTED"
    elif all(g < 0.03 for g in gaps.values()):
        lab = "REFUTED"
    else:
        lab = "INCONCLUSIVE"
    return lab, best


def selftest_verdict():
    assert verdict({0.01: .9, 0.02: .95, 0.05: .8}, {0.01: .3, 0.02: .12, 0.05: .3}, 0.01) == ("SUPPORTED", 0.02)
    assert verdict({0.01: .9, 0.02: .95, 0.05: .8}, {0.01: .3, 0.02: .12, 0.05: .3}, 0.06)[0] == "INCONCLUSIVE"
    assert verdict({0.01: .9, 0.02: .95, 0.05: .8}, {0.01: .02, 0.02: .0, 0.05: -.1}, 0.2)[0] == "REFUTED"
    assert verdict({0.01: .9, 0.02: .95, 0.05: .8}, {0.01: .05, 0.02: .0, 0.05: .0}, 0.0)[0] == "INCONCLUSIVE"
    a, b = make_noise("x", 1, 0.02), make_noise("x", 1, 0.02)
    assert torch.equal(a, b) and not torch.equal(a, make_noise("x", 2, 0.02))
    assert abs(float(make_noise("y", 0, 0.05).std()) - 0.05) < 2e-3


# ----------------------------------------------------------------------------- B0 (005 A0, cached features only)
def b0_scores(df, arm, subdir):
    """005's A0: fit on clean primary features of b0_train rows (standardised with the b0_train reals' mean/std),
    grouped CV (group = sha256) + balanced, one logo_<g> model per generator. -> {cond: {g: decision [n_rows]}}"""
    conds = [COND[s] for s in SETS]
    Xc = {c: P.load_features_aligned(df, "primary", c, subdir=subdir) for c in conds}
    lab, gen = df.label.to_numpy(), df.generator.to_numpy()
    b0 = np.where(df.b0_train.to_numpy())[0]
    rtr = b0[lab[b0] == 0]
    mu = Xc["clean"][rtr].mean(0).astype(np.float64)
    sd = np.maximum(Xc["clean"][rtr].std(0).astype(np.float64), 1e-6)
    Z = {c: ((Xc[c] - mu) / sd) for c in conds}
    out = {c: {} for c in conds}
    for g in GENS:
        keep = b0[gen[b0] != g]
        m = probe.fit_probe(Z["clean"][keep], lab[keep], SEED, groups=df.sha256.to_numpy()[keep], class_weight="balanced")
        for c in conds:
            out[c][g] = m.decision_function(Z[c])
        log(f"B0[{arm}] logo_{g}: n_train={len(keep)} C={m[-1].C_[0]:.3g}")
    return out


# ----------------------------------------------------------------------------- analysis
def analyse(dfs, SC, B0, active, n_boot):
    """dfs[arm], SC[arm][cond] -> [n,2,3] (NaN where not scored), B0[arm][cond][g]. Returns results dict."""
    res = dict(cells={}, per_pool={}, secondary={}, operating_point={}, b0_005style={}, plans={})
    plans = {}                                  # (ff, fakes) -> Plan, shared by both arms (rows are identical)
    d0 = dfs["orig"]
    lab, sp, pool, gen = (d0.label.to_numpy(), d0.split.to_numpy(), d0.pool.to_numpy(), d0.generator.to_numpy())
    test, calib = (sp == "test") & active, (sp == "calib") & active
    Rn = np.where(test & (lab == 0) & (pool == "vision_native"))[0]
    for ffn, gl in FF_VARIANTS.items():
        plans[(ffn, "test")] = Plan(d0, Rn, gl, test & (lab == 1), n_boot)
        plans[(ffn, "all")] = Plan(d0, Rn, gl, active & (lab == 1), n_boot)
    res["plans"] = dict(n_real_native_test=int(len(Rn)), n_clusters=len(set(d0.device.to_numpy()[Rn].tolist())),
                        n_fake_test={g: int((test & (lab == 1) & (gen == g)).sum()) for g in GENS},
                        n_boot=n_boot, seed=SEED)
    store = {}      # (fv, arm, ff) -> dict used by the secondary comparison
    for fv in FVS:
        vi = FVS.index(fv)
        res["cells"][fv] = {}
        for arm in ARMS:
            res["cells"][fv][arm] = {}
            for ffn in FF_VARIANTS:
                pl = plans[(ffn, "test")]
                cell = dict(rigid={}, b0={}, b0_005style={})
                A = {}                                   # (sigma, setting) -> (pt, boot)
                for si, sg in enumerate(SIGMAS):
                    for st in SETS:
                        s = SC[arm][COND[st]][:, vi, si]
                        A[(sg, st)] = pl.macro(lambda g, s=s: s)
                B = {}
                for st in SETS:
                    bs = B0[arm][COND[st]]
                    B[st] = pl.macro(lambda g, bs=bs: bs[g])
                store[(fv, arm, ffn)] = dict(A=A, B=B)
                gaps_iii, auc_i = {}, {}
                for si, sg in enumerate(SIGMAS):
                    e = {st: vc(*A[(sg, st)]) for st in SETS}
                    e["gap_iii-p"] = dc(*A[(sg, "i")], *A[(sg, "iii-p")])
                    e["gap_iv-p"] = dc(*A[(sg, "i")], *A[(sg, "iv-p")])
                    pooled = {st: pl.pooled(SC[arm][COND[st]][:, vi, si]) for st in SETS}
                    e["pooled"] = {st: vc(*pooled[st]) for st in SETS}
                    e["pooled"]["gap_iii-p"] = dc(*pooled["i"], *pooled["iii-p"])
                    cell["rigid"][str(sg)] = e
                    gaps_iii[sg], auc_i[sg] = e["gap_iii-p"]["delta"], e["i"]["value"]
                b0e = {st: vc(*B[st]) for st in SETS}
                b0e["gap_iii-p"] = dc(*B["i"], *B["iii-p"])
                b0e["gap_iv-p"] = dc(*B["i"], *B["iv-p"])
                cell["b0"] = b0e
                p2 = plans[(ffn, "all")]
                B2 = {st: p2.macro(lambda g, bs=B0[arm][COND[st]]: bs[g]) for st in SETS}
                cell["b0_005style"] = {st: vc(*B2[st]) for st in SETS}
                cell["b0_005style"]["gap_iii-p"] = dc(*B2["i"], *B2["iii-p"])
                cell["b0_005style"]["gap_iv-p"] = dc(*B2["i"], *B2["iv-p"])
                lab_, best = verdict(auc_i, gaps_iii, b0e["gap_iii-p"]["delta"])
                lab_005, _ = verdict(auc_i, gaps_iii, cell["b0_005style"]["gap_iii-p"]["delta"])
                cell["best_sigma"], cell["verdict"] = best, lab_
                cell["verdict_if_b0_gap_from_005style_fakes"] = lab_005
                cell["gap_iii-p_by_sigma"] = {str(k): v for k, v in gaps_iii.items()}
                res["cells"][fv][arm][ffn] = cell
                log(f"[{fv}/{arm}/{ffn}] best sigma {best}: AUROC(i)={auc_i[best]:.3f} gap(iii-p)={gaps_iii[best]:+.3f} "
                    f"B0 gap={b0e['gap_iii-p']['delta']:+.3f} -> {lab_}")
        # secondary: orig - fov512 in AUROC(i), paired
        res["secondary"][fv] = {}
        for ffn in FF_VARIANTS:
            so, sf = store[(fv, "orig", ffn)], store[(fv, "fov512", ffn)]
            e = {"rigid": {str(sg): dc(*so["A"][(sg, "i")], *sf["A"][(sg, "i")]) for sg in SIGMAS},
                 "b0": dc(*so["B"]["i"], *sf["B"]["i"])}
            e["rigid_at_best_sigma_orig"] = e["rigid"][str(res["cells"][fv]["orig"][ffn]["best_sigma"])]
            res["secondary"][fv][ffn] = e
        # per-pool descriptive (setting i)
        res["per_pool"][fv] = {}
        for arm in ARMS:
            res["per_pool"][fv][arm] = {}
            for ffn, gl in FF_VARIANTS.items():
                o = {}
                for pn, nm in POOLS:
                    R = np.where(test & (lab == 0) & (pool == pn))[0]
                    if not len(R):
                        continue
                    pl = Plan(d0, R, gl, test & (lab == 1), n_boot)
                    o[nm] = dict(n_real=int(len(R)),
                                 b0=vc(*pl.macro(lambda g: B0[arm]["clean"][g])),
                                 rigid={str(sg): vc(*pl.macro(lambda g, s=SC[arm]["clean"][:, vi, si]: s))
                                        for si, sg in enumerate(SIGMAS)})
                res["per_pool"][fv][arm][ffn] = o
        # operating point: 1% FPR from calib native reals (setting i)
        res["operating_point"][fv] = {}
        for arm in ARMS:
            cn = np.where(calib & (lab == 0) & (pool == "vision_native"))[0]
            o = {}
            for si, sg in enumerate(SIGMAS):
                thr = metrics.threshold_for_fpr(SC[arm]["clean"][cn, vi, si], 0.01)
                e = dict(threshold=thr, n_calib=int(len(cn)))
                for st in SETS:
                    s = SC[arm][COND[st]][:, vi, si]
                    e[f"fpr_native_{st}"] = float((s[Rn] >= thr).mean())
                    for ffn, gl in FF_VARIANTS.items():
                        Fm = np.where(test & (lab == 1) & np.isin(gen, gl))[0]
                        e[f"tpr_{ffn}_{st}"] = float((s[Fm] >= thr).mean())
                for pn, nm in POOLS[1:]:
                    R = np.where(test & (lab == 0) & (pool == pn))[0]
                    if len(R):
                        e[f"fpr_{nm}_i"] = float((SC[arm]["clean"][R, vi, si] >= thr).mean())
                o[str(sg)] = e
            res["operating_point"][fv][arm] = o
    return res


# ----------------------------------------------------------------------------- tables
def f3(v):
    return f"{v:.3f}"


def fv_ci(e):
    return f"{e['value']:.3f} [{e['ci'][0]:.3f},{e['ci'][1]:.3f}]"


def fd_ci(e):
    return f"{e['delta']:+.3f} [{e['ci'][0]:+.3f},{e['ci'][1]:+.3f}]"


def write_tables(res, meta, path):
    L = [f"# H4 PILOT (experiment 006): format shortcut, RIGID-style score vs B0 (Amendment 5). Not a Phase 4 verdict.\n",
         f"Run: {meta['stamp']} | n_boot={res['plans']['n_boot']} | primary feature variant = {PRIMARY_FV} (post-final-LayerNorm CLS) | "
         f"reals: {res['plans']['n_real_native_test']} test natives in {res['plans']['n_clusters']} device clusters\n",
         "AUROC = LOGO-macro over generators (each generator's test fakes vs the same native reals). 95% CI = cluster bootstrap "
         "(reals by device, fakes by image per generator); gaps are paired. Higher RIGID score = fake, never flipped.\n"]
    for fv in [PRIMARY_FV] + [v for v in FVS if v != PRIMARY_FV]:
        tag = "PRIMARY" if fv == PRIMARY_FV else "SENSITIVITY (pre-final-LayerNorm CLS)"
        L.append(f"\n## Verdict labels, feature variant `{fv}` ({tag})\n")
        L.append("| arm | Firefly | best sigma | AUROC(i) | gap(i)-(iii-p) | B0 gap (test fakes) | label | label if B0 gap from 005-style fakes |")
        L.append("|---|---|---|---|---|---|---|---|")
        for arm in ARMS:
            for ffn in FF_VARIANTS:
                c = res["cells"][fv][arm][ffn]
                bs = str(c["best_sigma"])
                L.append(f"| {arm} | {ffn} | {bs} | {fv_ci(c['rigid'][bs]['i'])} | {fd_ci(c['rigid'][bs]['gap_iii-p'])} | "
                         f"{fd_ci(c['b0']['gap_iii-p'])} | **{c['verdict']}** | {c['verdict_if_b0_gap_from_005style_fakes']} |")
        for arm in ARMS:
            for ffn in FF_VARIANTS:
                c = res["cells"][fv][arm][ffn]
                L.append(f"\n### {fv} / {arm} / {ffn}: AUROC per setting and gap\n")
                L.append("| score | AUROC (i) | AUROC (iii-p) | AUROC (iv-p) | gap (i)-(iii-p) | gap (i)-(iv-p) |")
                L.append("|---|---|---|---|---|---|")
                for sg in SIGMAS:
                    e = c["rigid"][str(sg)]
                    L.append(f"| RIGID s={sg} | {fv_ci(e['i'])} | {fv_ci(e['iii-p'])} | {fv_ci(e['iv-p'])} | {fd_ci(e['gap_iii-p'])} | {fd_ci(e['gap_iv-p'])} |")
                for sg in SIGMAS:
                    e = c["rigid"][str(sg)]["pooled"]
                    L.append(f"| RIGID s={sg} (pooled AUROC) | {fv_ci(e['i'])} | {fv_ci(e['iii-p'])} | {fv_ci(e['iv-p'])} | {fd_ci(e['gap_iii-p'])} | - |")
                b = c["b0"]
                L.append(f"| B0 (LOGO-macro) | {fv_ci(b['i'])} | {fv_ci(b['iii-p'])} | {fv_ci(b['iv-p'])} | {fd_ci(b['gap_iii-p'])} | {fd_ci(b['gap_iv-p'])} |")
                b = c["b0_005style"]
                L.append(f"| B0 (005-style: all fakes of g) | {fv_ci(b['i'])} | {fv_ci(b['iii-p'])} | {fv_ci(b['iv-p'])} | {fd_ci(b['gap_iii-p'])} | {fd_ci(b['gap_iv-p'])} |")
        L.append(f"\n## Secondary, `{fv}`: AUROC(i) orig - fov512 (paired CI; descriptive, no verdict)\n")
        L.append("| Firefly | score | orig - fov512 |")
        L.append("|---|---|---|")
        for ffn in FF_VARIANTS:
            s = res["secondary"][fv][ffn]
            for sg in SIGMAS:
                L.append(f"| {ffn} | RIGID s={sg} | {fd_ci(s['rigid'][str(sg)])} |")
            L.append(f"| {ffn} | B0 | {fd_ci(s['b0'])} |")
        L.append(f"\n## Descriptive per real pool, setting (i), `{fv}`: LOGO-macro AUROC vs clean fakes\n")
        L.append("| arm | Firefly | pool | n_real | B0 | " + " | ".join(f"RIGID s={s}" for s in SIGMAS) + " |")
        L.append("|---|---|---|---|---|" + "---|" * len(SIGMAS))
        for arm in ARMS:
            for ffn in FF_VARIANTS:
                for nm, e in res["per_pool"][fv][arm][ffn].items():
                    L.append(f"| {arm} | {ffn} | {nm} | {e['n_real']} | {fv_ci(e['b0'])} | " +
                             " | ".join(fv_ci(e['rigid'][str(s)]) for s in SIGMAS) + " |")
        L.append(f"\n## 1%-FPR operating point of the RIGID score, `{fv}` (threshold from calib-split native reals, setting i)\n")
        L.append("| arm | sigma | thr | n_calib | FPR native (i) | FPR WA (i) | FPR FBH (i) | FPR native (iii-p) | FPR native (iv-p) | TPR (i) with FF | TPR (i) w/o FF | TPR (iii-p) with FF | TPR (iv-p) with FF |")
        L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for arm in ARMS:
            for sg in SIGMAS:
                e = res["operating_point"][fv][arm][str(sg)]
                L.append(f"| {arm} | {sg} | {e['threshold']:.4g} | {e['n_calib']} | {f3(e['fpr_native_i'])} | {f3(e.get('fpr_WA_i', float('nan')))} | "
                         f"{f3(e.get('fpr_FBH_i', float('nan')))} | {f3(e['fpr_native_iii-p'])} | {f3(e['fpr_native_iv-p'])} | "
                         f"{f3(e['tpr_with_firefly_i'])} | {f3(e['tpr_without_firefly_i'])} | {f3(e['tpr_with_firefly_iii-p'])} | {f3(e['tpr_with_firefly_iv-p'])} |")
    L.append("\n## D-008 guard and pipeline check\n")
    g = meta["d008"]
    L.append(f"n={g['n']}, device {g['device']}, tol (scaled feature diff) {g['feat_tol']}, overall ok={g['ok']}\n")
    L.append("| variant | max abs feat diff | max scaled feat diff | max abs score diff | ok |")
    L.append("|---|---|---|---|---|")
    for v, e in g["variants"].items():
        L.append(f"| {v} | {e['max_abs_feature_diff']:.2e} | {e['max_scaled_feature_diff']:.2e} | {e['max_abs_score_diff']:.2e} | {e['ok']} |")
    p = g["pipeline_vs_004_cache"]
    L.append(f"\nPipeline vs 004 cache (unnoised pre-LN CLS vs primary[:, :768], same 32 images): max abs {p['max_abs_diff']:.2e}, scaled {p['max_scaled_diff']:.2e}, ok={p['ok']}")
    if meta.get("check_005"):
        c = meta["check_005"]
        L.append(f"\nB0 sanity vs 005 pilot_v2 (orig, clean, all fakes of g, with Firefly): ours {c['ours']:.4f} vs 005 {c['005']:.4f} (|d|={abs(c['ours'] - c['005']):.1e})")
    path.write_text("\n".join(L) + "\n")


# ----------------------------------------------------------------------------- main
args_subdir_for_check = {}


def main():
    global LOGF
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subdir", default="pilot")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    selftest_verdict()
    subdirs = {"orig": a.subdir, "fov512": a.subdir + "_fov512"}
    args_subdir_for_check.update(subdirs)
    out_dir = (HERE / "cache" / "dryrun_out") if a.dry_run else HERE / a.subdir
    cache = HERE / "cache" / ("dryrun" if a.dry_run else a.subdir)
    out_dir.mkdir(parents=True, exist_ok=True); cache.mkdir(parents=True, exist_ok=True)
    LOGF = open(out_dir / "run.log", "a")
    n_boot = 50 if a.dry_run else CFG["aggregation"]["n_boot"]
    log(f"H4 pilot {'DRY-RUN ' if a.dry_run else ''}out={out_dir.relative_to(ROOT)} n_boot={n_boot}")

    dfs = {k: pd.read_csv(v, keep_default_na=False, dtype={"vidx": str}).astype({"in_scoring_set": bool, "b0_train": bool})
           for k, v in SPLITS.items()}
    assert (dfs["orig"].sha256 == dfs["fov512"].sha256).all(), "arms must list the same rows in the same order"
    for k in ARMS:
        miss = [p for p in dfs[k].path if not Path(p).exists()]
        if miss:
            raise SystemExit(f"{len(miss)} crops missing for arm {k}, e.g. {miss[0]}")
    n = len(dfs["orig"])
    d0 = dfs["orig"]
    sp, lab, pool = d0.split.to_numpy(), d0.label.to_numpy(), d0.pool.to_numpy()
    if a.dry_run:     # 10 natives (spread over devices) + 6 fakes (1/generator) from test, 5 native calib reals
        t = d0[(d0.split == "test") & (d0.pool == "vision_native")].sort_values(["device", "sha256"])
        nat = t.groupby("device").head(1).head(10).index.tolist()
        fk = d0[(d0.split == "test") & (d0.label == 1)].groupby("generator").head(1).index.tolist()
        ca = d0[(d0.split == "calib") & (d0.pool == "vision_native")].head(5).index.tolist()
        active = np.zeros(n, bool); active[nat + fk + ca] = True
    else:
        active = np.ones(n, bool)
    # rows to score per condition (from the 004 plan): setting (i): all test rows + native calib; others: native test + fakes
    test = (sp == "test") & active
    rows_i = np.where(test | ((sp == "calib") & (pool == "vision_native") & active))[0]
    rows_fmt = np.where(test & ((pool == "vision_native") | (lab == 1)))[0]
    rows_for = {COND["i"]: rows_i, COND["iii-p"]: rows_fmt, COND["iv-p"]: rows_fmt}

    from features import get_device
    dev = get_device(a.device)
    p4 = load_004_run()
    if not p4.weights_cached():
        raise SystemExit("facebook/dinov2-base not in data/hf_cache")
    bb = load_backbone(P.SPEC, input_size=224, layers=(-1,), use_cls=True, use_mean=False)
    model = bb.model.to(dev).eval()
    log(f"device {dev}; rows to score: (i) {len(rows_i)}, (iii-p)/(iv-p) {len(rows_fmt)} per arm")
    guard = d008_guard(d0, model, dev, p4, out_dir)

    SC = {arm: {c: np.full((n, 2, len(SIGMAS)), np.nan) for c in rows_for} for arm in ARMS}
    timings = {}
    with ThreadPoolExecutor(4) as pool_:
        for arm in ARMS:
            df = dfs[arm]
            for cond, rows in rows_for.items():
                cp = cache / f"{arm}__{P.safe(cond)}.npz"
                shas = df.sha256.to_numpy()[rows]
                if cp.exists():
                    z = np.load(cp, allow_pickle=False)
                    if z["keys"].tolist() == shas.tolist():
                        SC[arm][cond][rows] = z["scores"]
                        log(f"  {arm}/{cond}: loaded from cache")
                        continue
                t = time.time()
                log(f"  {arm}/{cond}: scoring {len(rows)} images x {len(SIGMAS)} sigmas x {DRAWS} draws")
                s = score_condition(model, dev, p4, pool_, df.path.to_numpy()[rows].tolist(), shas.tolist(), cond)
                np.savez(cp, keys=np.array(shas.tolist(), dtype=str), scores=s)
                SC[arm][cond][rows] = s
                timings[f"{arm}/{cond}"] = time.time() - t
    # B0 from cached features (needs the full df, scores every row)
    B0 = {arm: b0_scores(dfs[arm], arm, subdirs[arm]) for arm in ARMS}
    res = analyse(dfs, SC, B0, active, n_boot)

    meta = dict(stamp=time.strftime("%Y-%m-%d %H:%M:%S"), dry_run=a.dry_run, d008=guard, timings=timings, device=str(dev),
                subdirs=subdirs, config="experiments/006-format-shortcut/config.json")
    try:     # sanity: our B0 A0 (orig, clean, all fakes of g, with Firefly) vs 005 pilot_v2
        r5 = json.loads((ROOT / "experiments/005-nuisance-projection/pilot_v2/orig/results.json").read_text())
        meta["check_005"] = {"ours": res["cells"][PRIMARY_FV]["orig"]["with_firefly"]["b0_005style"]["i"]["value"],
                             "005": r5["arms"]["A0"]["clean"]["macro"]["auroc"]}
        log(f"B0 vs 005 pilot_v2 (orig clean, all-g fakes): {meta['check_005']}")
    except Exception as e:      # noqa
        log("005 check skipped:", e)
    res["meta"] = meta
    (out_dir / "results.json").write_text(json.dumps(res, indent=1, default=float))
    write_tables(res, meta, out_dir / "tables.md")
    # long scores
    rows = []
    for arm in ARMS:
        df = dfs[arm]
        for st in SETS:
            cond = COND[st]
            idx = np.where(~np.isnan(SC[arm][cond][:, 0, 0]))[0]
            for fi, fv in enumerate(FVS):
                for si, sg in enumerate(SIGMAS):
                    sub = df.iloc[idx]
                    rows.append(pd.DataFrame(dict(setting=st if st != "i" else "i", arm=arm, feat=fv, sigma=sg,
                                                  sha256=sub.sha256.values, label=sub.label.values, generator=sub.generator.values,
                                                  pool=sub.pool.values, device=sub.device.values, split=sub.split.values,
                                                  score=SC[arm][cond][idx, fi, si])))
    pd.concat(rows).to_csv(out_dir / "scores_long.csv.gz", index=False)
    log(f"done in {time.time() - T0:.0f}s; outputs in {out_dir.relative_to(ROOT)}")
    meta["total_seconds"] = time.time() - T0
    res["meta"] = meta
    (out_dir / "results.json").write_text(json.dumps(res, indent=1, default=float))
    if a.dry_run:
        print("DRY-RUN OK")


if __name__ == "__main__":
    sys.exit(main() or 0)
