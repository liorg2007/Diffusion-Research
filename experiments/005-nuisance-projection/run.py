"""H1 laundering-nuisance projection (experiment 005). Resumable; per-arm score caches in <out-dir>/cache.
Default run (no arguments) = the PILOT of Amendment 2, all outputs in pilot/:
    env/.venv/bin/python -I experiments/005-nuisance-projection/run.py

Options (all additive; the defaults reproduce the original pilot byte for byte, apart from timings / git head):
    --subdir NAME       004 feature cache subdir (default 'pilot'; 'pilot_fov512' / 'fov512' for the fov512 arm)
    --splits CSV        splits file (default 004 pilot/splits_pilot.csv); features are keyed by sha256, so the fov512
                        splits file and the original one are interchangeable here
    --out-dir DIR       output dir (default experiments/005-nuisance-projection/pilot)
    --label TEXT        label written to results.json / tables.md / log (default PILOT)
    --arms A0 A1 ...    only these arms (default: all of ORDER); verdict only if A1 and A2 are both present
    --generators ...    LOGO generators (default: the 6 Synthbuster pilot generators)
    --heldout-generators ...  test-only generators (e.g. flux): the 'full' model is scored on them; the pass criterion
                        then needs the gain on BOTH the LOGO-macro and every held-out generator (min over endpoints)
    --grouped-cv        probe C selected with StratifiedGroupKFold, group = source image (augmented copies share a fold)
    --balanced          class_weight='balanced' logistic probe
    --group-bootstrap   CIs: reals resampled by cluster (device id; RAISE: per image), fakes by image stratified per
                        generator (lib/metrics.bootstrap_weights), same plan for every arm (paired)
    --per-pool          LOGO tables per real pool (native / WA / FBH / RAISE when present), with CIs under --group-bootstrap
    --headroom-rule     Amendment 3: the +0.03 criterion applies only if A1 LOGO-macro AUROC at jpeg75_resize05 on the
                        primary reals is <= 0.96; otherwise verdict = UNINFORMATIVE_NO_HEADROOM
    --twin-disjoint     Amendment 13 sensitivity: drop from TRAINING (all models, A0..A4) every fake whose RAISE caption-twin id
                        (r<hex>t in the file name) equals a calib or test RAISE real. Reals, projector deltas, calib/test untouched.
    --primary-pools P   comma list of pool names forming the primary reals (default vision_native)
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "experiments" / "lib"))
sys.path.insert(0, str(ROOT / "experiments" / "004-phase4-features"))

import numpy as np
import pandas as pd
from scipy.stats import rankdata

import metrics
import probe
from p4features import EVAL_CONDITIONS, GRID_CONDITIONS, load_features_aligned

OUT = HERE / "pilot"
CACHE = OUT / "cache"
SPLITS = ROOT / "experiments/004-phase4-features/pilot/splits_pilot.csv"
SUBDIR = "pilot"
LABEL = "PILOT"
GROUPED_CV = False
BALANCED = False
GROUP_BOOT = False
PER_POOL = False
HEADROOM = False
PRIMARY_POOLS = ("vision_native",)
HELDOUT = []
TWIN_DISJOINT = False
TWIN_DROP = None      # bool mask over df rows (set in main when TWIN_DISJOINT)
ARM_FILTER = None
HEADROOM_AUROC = 0.96
POOLS = [("vision_native", "native"), ("vision_whatsapp", "WA"), ("vision_facebook_hq", "FBH"), ("raise", "RAISE")]
SEED = 0
CONDS = EVAL_CONDITIONS
CI = {c: i for i, c in enumerate(CONDS)}
GENS = ["dalle2", "dalle3", "firefly", "midjourney-v5", "stable-diffusion-2", "stable-diffusion-xl"]
MODELS = ["full"] + [f"logo_{g}" for g in GENS]
N_BOOT = 1000
T0 = time.time()
TIMINGS = {}


def log(*a):
    print(time.strftime("%H:%M:%S"), LABEL, *a, flush=True)


# ----------------------------------------------------------------------------- data
def load_data(feats=("primary", "secondary")):
    df = pd.read_csv(SPLITS, keep_default_na=False, dtype={"vidx": str})
    D = {}
    for feat in feats:
        D[feat] = dict(
            clean=load_features_aligned(df, feat, "clean", subdir=SUBDIR),
            ev={c: load_features_aligned(df, feat, c, subdir=SUBDIR) for c in CONDS},
            grid={c: load_features_aligned(df, feat, c, subdir=SUBDIR) for c in GRID_CONDITIONS})
        for c in CONDS:
            assert not np.isnan(D[feat]["ev"][c]).any(), (feat, c)
    return df, D


def pair_indices(df):
    """Row indices (nat, wa, fb) of train-split device triplets matched by (device, vidx).
    Also checks the b0_train reals: the natives, plus (official run) RAISE train reals which carry no platform pair
    and enter only the grid deltas / standardization (see real_train_indices)."""
    r = df[(df.label == 0) & (df.split == "train")]
    key = lambda pool: {(d, v): i for i, d, v in zip(r.index[r.pool == pool], r.device[r.pool == pool],
                                                      r.vidx[r.pool == pool])}
    nat, wa, fb = key("vision_native"), key("vision_whatsapp"), key("vision_facebook_hq")
    keys = sorted(nat)
    assert set(keys) == set(wa) == set(fb), "incomplete triplets"
    n = np.array([nat[k] for k in keys]); w = np.array([wa[k] for k in keys]); f = np.array([fb[k] for k in keys])
    b0r = set(df.index[df.b0_train & (df.label == 0)])
    assert set(n) <= b0r and all(df.pool[i] == "raise" for i in b0r - set(n)), "unexpected b0_train reals"
    return n, w, f


def real_train_indices(df, nat):
    """The b0_train reals: the paired natives in pair order (== the original `nat`), then any extra b0_train reals
    (RAISE train, official run) in row order. Equals `nat` exactly when there is no RAISE (the pilot)."""
    extra = [i for i in df.index[df.b0_train & (df.label == 0)] if i not in set(nat.tolist())]
    return np.concatenate([nat, np.array(extra, dtype=nat.dtype)]) if extra else nat


# ----------------------------------------------------------------------------- projector
def build_deltas(Z, Zg, nat, wa, fb, grid_reals=None):
    """Platform deltas use the paired natives; grid deltas use `grid_reals` (default: the paired natives)."""
    zc = Z["clean"]
    gr = nat if grid_reals is None else grid_reals
    d = [zc[wa] - zc[nat], zc[fb] - zc[nat]] + [Zg[c][gr] - zc[gr] for c in GRID_CONDITIONS]
    pa = [zc[nat], zc[nat]] + [zc[gr]] * 9
    pb = [zc[wa], zc[fb]] + [Zg[c][gr] for c in GRID_CONDITIONS]
    return np.concatenate(d).astype(np.float64), np.concatenate(pa), np.concatenate(pb)


def eig_second_moment(Dl, centered):
    X = Dl - Dl.mean(0, keepdims=True) if centered else Dl
    M = X.T @ X / len(X)
    w, V = np.linalg.eigh(M)
    o = np.argsort(w)[::-1]
    return np.clip(w[o], 0, None), V[:, o]


def kstar(w, frac=0.8):
    c = np.cumsum(w) / w.sum()
    return int(np.searchsorted(c, frac) + 1), c


def project(Z, U):
    return Z - (Z @ U) @ U.T


# ----------------------------------------------------------------------------- A4
def train_a4(Zpa, Zpb, Zreal, epochs=300, lr=1e-3, bs=256, w_inv=25.0, w_var=25.0, w_cov=1.0):
    import torch
    torch.manual_seed(SEED)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    d = Zpa.shape[1]
    g = torch.nn.Sequential(torch.nn.Linear(d, 512), torch.nn.GELU(), torch.nn.Linear(512, 256)).to(dev)
    opt = torch.optim.Adam(g.parameters(), lr=lr)
    A = torch.tensor(Zpa, dtype=torch.float32, device=dev)
    B = torch.tensor(Zpb, dtype=torch.float32, device=dev)
    R = torch.tensor(Zreal, dtype=torch.float32, device=dev)
    gen = torch.Generator(device="cpu").manual_seed(SEED)
    n = len(A)
    hist = []
    for ep in range(epochs):
        perm = torch.randperm(n, generator=gen).to(dev)
        for s in range(0, n, bs):
            ix = perm[s:s + bs]
            inv = torch.nn.functional.mse_loss(g(A[ix]), g(B[ix]))
            e = g(R)
            e = e - e.mean(0)
            std = torch.sqrt(e.var(0) + 1e-4)
            var = torch.relu(1 - std).mean()
            cov = (e.T @ e) / (len(e) - 1)
            cov = (cov - torch.diag(torch.diag(cov))).pow(2).sum() / e.shape[1]
            loss = w_inv * inv + w_var * var + w_cov * cov
            opt.zero_grad(); loss.backward(); opt.step()
        if ep % 50 == 0 or ep == epochs - 1:
            hist.append(dict(epoch=ep, loss=float(loss), inv=float(inv), var=float(var), cov=float(cov)))
    g.eval()
    return g, hist, dev


def apply_g(g, dev, Z):
    import torch
    with torch.no_grad():
        out = [g(torch.tensor(Z[i:i + 2048], dtype=torch.float32, device=dev)).cpu().numpy()
               for i in range(0, len(Z), 2048)]
    return np.concatenate(out).astype(np.float64)


# ----------------------------------------------------------------------------- arms
def run_arm(name, df, T, Zev, Zgrid, aug):
    """T: transform fn on z-space arrays. Returns scores [model][len(CONDS), n]. Resumable."""
    path = CACHE / f"scores_{name}.npz"
    if path.exists():
        z = np.load(path)
        return {m: z[m] for m in MODELS}
    t0 = time.time()
    Xev = {c: T(Zev[c]) for c in CONDS}
    Xg = {c: T(Zgrid[c][df.b0_train.to_numpy()]) for c in GRID_CONDITIONS} if aug else {}
    lab, gen = df.label.to_numpy(), df.generator.to_numpy()
    b0 = df.b0_train.to_numpy()
    b0idx = np.where(b0)[0]
    out = {}
    for m in MODELS:
        g = None if m == "full" else m[5:]
        keep = b0idx if g is None else b0idx[gen[b0idx] != g]
        if TWIN_DROP is not None:
            keep = keep[~TWIN_DROP[keep]]
        Xtr = [Xev["clean"][keep]]
        ytr = [lab[keep]]
        if aug:
            pos = {i: j for j, i in enumerate(b0idx)}
            sel = np.array([pos[i] for i in keep])
            for c in GRID_CONDITIONS:
                Xtr.append(Xg[c][sel]); ytr.append(lab[keep])
        kw = {}
        if BALANCED:
            kw["class_weight"] = "balanced"
        if GROUPED_CV:      # all copies (clean + 9 grid cells) of one source image stay in one CV fold
            kw["groups"] = np.concatenate([df.sha256.to_numpy()[keep]] * len(Xtr))
        mdl = probe.fit_probe(np.concatenate(Xtr), np.concatenate(ytr), SEED, **kw)
        out[m] = np.stack([mdl.decision_function(Xev[c]) for c in CONDS]).astype(np.float32)
        log(f"{name}/{m}: n_train={sum(len(x) for x in Xtr)} C={mdl[-1].C_[0]:.3g}")
    np.savez(path, **out)
    TIMINGS[f"arm_{name}"] = time.time() - t0
    return out


# ----------------------------------------------------------------------------- evaluation
def auc_fast(sr, sf):
    """AUROC of fakes sf vs reals sr, ties averaged. sr,sf: 1-D."""
    r = rankdata(np.concatenate([sr, sf]))
    return (r[len(sr):].sum() - len(sf) * (len(sf) + 1) / 2) / (len(sr) * len(sf))


def real_index(df, pools=None):
    """Test-split reals of the given pool names (default: the primary pools; original: VISION natives)."""
    lab, sp, pool = df.label.to_numpy(), df.split.to_numpy(), df.pool.to_numpy()
    return np.where((lab == 0) & (sp == "test") & np.isin(pool, list(pools or PRIMARY_POOLS)))[0]


def logo_metrics(df, S, ci, pools=None):
    """Per-g metrics + macro for condition index ci. S: scores dict model-> [conds, n]."""
    gen = df.generator.to_numpy()
    R = real_index(df, pools)
    per = {}
    for g in GENS:
        F = np.where(gen == g)[0]
        s = S[f"logo_{g}"][ci]
        y = np.r_[np.zeros(len(R)), np.ones(len(F))].astype(int)
        mm = metrics.compute_metrics(y, np.r_[s[R], s[F]], threshold=None, calib_real_scores=None, n_boot=0)
        per[g] = dict(auroc=mm["auroc"]["value"], ap=mm["ap"]["value"], tpr_at_1fpr=mm["tpr_at_1fpr"]["value"],
                      n_real=len(R), n_fake=len(F))
    macro = {k: float(np.mean([per[g][k] for g in GENS])) for k in ("auroc", "ap", "tpr_at_1fpr")}
    return dict(per_generator=per, macro=macro)


def heldout_auc(df, S, ci, g, pools=None):
    """AUROC of the 'full' model (never saw g) on held-out generator g vs the given test reals."""
    R, F = real_index(df, pools), np.where(df.generator.to_numpy() == g)[0]
    s = S["full"][ci]
    return float(auc_fast(s[R], s[F]))


def pool_tables(df, ARMS):
    """LOGO-macro / per-generator AUROC per real pool (test reals of that pool only) and condition, per arm.
    Pools without test reals are omitted."""
    pool = df.pool.to_numpy()
    present = [(p, n) for p, n in POOLS if len(real_index(df, [p]))]
    out = {}
    for a, S in ARMS.items():
        out[a] = {}
        for c in CONDS:
            ci = CI[c]
            o = {}
            for p, n in present:
                m = logo_metrics(df, S, ci, [p])
                o[n] = dict(macro_auroc=m["macro"]["auroc"], n_real=m["per_generator"][GENS[0]]["n_real"],
                            per_generator={g: m["per_generator"][g]["auroc"] for g in GENS})
                for g in HELDOUT:
                    o[n].setdefault("heldout", {})[g] = heldout_auc(df, S, ci, g, [p])
            out[a][c] = o
    return out


def clusters_for(df, R):
    """Bootstrap cluster of each real: its device id (VISION natives/WA/FBH share a device); RAISE etc. have none and
    count as one cluster per image."""
    dev, sha = df.device.to_numpy(), df.sha256.to_numpy()
    return np.array([dev[i] if dev[i] else f"img:{sha[i]}" for i in R])


def boot_setup(df):
    lab, gen, sp, pool = df.label.to_numpy(), df.generator.to_numpy(), df.split.to_numpy(), df.pool.to_numpy()
    R = real_index(df)
    rng = np.random.default_rng(SEED)
    ridx = R[rng.integers(0, len(R), (N_BOOT, len(R)))]
    fidx = {}
    for g in GENS:
        F = np.where(gen == g)[0]
        fidx[g] = F[rng.integers(0, len(F), (N_BOOT, len(F)))]
    return ridx, fidx


def boot_macro_auc(S, ci, ridx, fidx):
    """-> [N_BOOT] LOGO-macro AUROC per resample (same resample indices for every arm)."""
    tot = np.zeros(N_BOOT)
    for g in GENS:
        s = S[f"logo_{g}"][ci].astype(np.float64)
        a, b = s[ridx], s[fidx[g]]
        r = rankdata(np.concatenate([a, b], 1), axis=1)
        nr, nf = a.shape[1], b.shape[1]
        tot += (r[:, nr:].sum(1) - nf * (nf + 1) / 2) / (nr * nf)
    return tot / len(GENS)


class GroupPlan:
    """Cluster bootstrap plan for a set of reals R and the fakes of `gens` (--group-bootstrap): reals by cluster,
    fakes by image stratified per generator (metrics.bootstrap_weights). One plan is shared by all arms/conditions."""

    def __init__(self, df, R, gens, macro_gens=None, n_boot=N_BOOT, seed=SEED):
        self.R, self.gens = R, list(gens)
        self.macro_gens = list(macro_gens or gens)
        gen = df.generator.to_numpy()
        self.F = {g: np.where(gen == g)[0] for g in self.gens}
        allF = np.concatenate([self.F[g] for g in self.gens])
        strat = np.concatenate([[g] * len(self.F[g]) for g in self.gens])
        self.Wr, self.Wf = metrics.bootstrap_weights(clusters_for(df, R), strat, n_boot, seed)
        self.cols, o = {}, 0
        for g in self.gens:
            self.cols[g] = slice(o, o + len(self.F[g])); o += len(self.F[g])
        self.n_clusters = len(set(clusters_for(df, R).tolist()))

    def macro_boot(self, S, ci):
        """[n_boot] LOGO-macro AUROC (model logo_g scores reals and g's fakes)."""
        tot = 0.0
        for g in self.macro_gens:
            s = S[f"logo_{g}"][ci].astype(np.float64)
            tot = tot + metrics.weighted_auroc(s[self.R], s[self.F[g]], self.Wr, self.Wf[:, self.cols[g]])
        return tot / len(self.macro_gens)

    def heldout_boot(self, S, ci, g):
        s = S["full"][ci].astype(np.float64)
        return metrics.weighted_auroc(s[self.R], s[self.F[g]], self.Wr, self.Wf[:, self.cols[g]])


def pct_ci(x, a=0.05):
    return float(np.percentile(x, 100 * a / 2)), float(np.percentile(x, 100 * (1 - a / 2)))


def drift(df, s_full_clean, fpr=0.05):
    sp, pool = df.split.to_numpy(), df.pool.to_numpy()
    cal = s_full_clean[(sp == "calib") & (pool == "vision_native")]
    thr = metrics.threshold_for_fpr(cal, fpr)
    f = {}
    for k, p in (("native", "vision_native"), ("WA", "vision_whatsapp"), ("FB", "vision_facebook_hq")):
        s = s_full_clean[(sp == "test") & (pool == p)]
        f[k] = float((s >= thr).mean()); f[f"n_{k}"] = int(len(s))
    f["calib_native_fpr"] = float((cal >= thr).mean())
    f["threshold"] = float(thr)
    f["drift_WA"] = abs(f["WA"] - f["native"]); f["drift_FB"] = abs(f["FB"] - f["native"])
    f["mean_drift"] = 0.5 * (f["drift_WA"] + f["drift_FB"])
    if (pool == "raise").any() and (sp[pool == "raise"] == "test").any():   # official run: RAISE reals at the same threshold
        s = s_full_clean[(sp == "test") & (pool == "raise")]
        f["RAISE"] = float((s >= thr).mean()); f["n_RAISE"] = int(len(s))
    return f


# ----------------------------------------------------------------------------- main
NEWMODE = False       # any of the Amendment-3 options active -> extra keys in results.json / tables.md


def twin_drop_mask(df):
    """Train fakes whose RAISE caption-twin id (r<hex>t in the file name of src_path; fakes are named after the RAISE image
    whose caption prompted them) is the id of a calib or test RAISE real (same regex as 006 critic_checks/c4)."""
    import os, re
    rid = np.array([(re.search(r"(r[0-9a-f]{8,}t)", os.path.basename(p)) or [None, ""])[1] for p in df.src_path])
    ev = set(rid[(df.pool.to_numpy() == "raise") & df.split.isin(["calib", "test"]).to_numpy()]) - {""}
    return (df.label.to_numpy() == 1) & (df.split.to_numpy() == "train") & np.isin(rid, list(ev))


def check_cache_opts(opts):
    """A score cache is only valid for the option set that produced it. A cache without opts.json is the original
    default pilot cache."""
    default = dict(subdir="pilot", splits=str(SPLITS), grouped_cv=False, balanced=False, generators=list(GENS))
    p = CACHE / "opts.json"
    if p.exists():
        old = json.loads(p.read_text())
    elif any(CACHE.glob("scores_*.npz")):
        old = default
    else:
        p.write_text(json.dumps(opts, indent=1))
        return
    if old != opts:
        raise SystemExit(f"score cache {CACHE} was made with {old}, now {opts}: use another --out-dir")


def main():
    OUT.mkdir(parents=True, exist_ok=True); CACHE.mkdir(parents=True, exist_ok=True)
    opts = dict(subdir=SUBDIR, splits=str(SPLITS), grouped_cv=GROUPED_CV, balanced=BALANCED, generators=list(GENS))
    if TWIN_DISJOINT:
        opts["twin_disjoint"] = True
    check_cache_opts(opts)
    want = lambda name: ARM_FILTER is None or name in ARM_FILTER
    need_sec = want("A1_sec") or want("A2_sec")
    df, D = load_data(("primary", "secondary") if need_sec else ("primary",))
    nat, wa, fb = pair_indices(df)
    global TWIN_DROP
    if TWIN_DISJOINT:
        TWIN_DROP = twin_drop_mask(df)
        log("twin-disjoint: dropping", int(TWIN_DROP.sum()), "train fakes",
            {g: int(TWIN_DROP[df.generator.to_numpy() == g].sum()) for g in sorted(set(df.generator[TWIN_DROP]))})
    rtr = real_train_indices(df, nat)     # == nat in the pilot; + RAISE train reals in the official run
    log("rows", len(df), "pairs", len(nat), "b0_train reals", len(rtr))
    ARMS = {}          # name -> scores
    info = {}
    for feat, tag in (("primary", ""), ("secondary", "_sec")):
        if feat not in D:
            continue
        t0 = time.time()
        F = D[feat]
        mu = F["clean"][rtr].mean(0).astype(np.float64)
        sd = np.maximum(F["clean"][rtr].std(0).astype(np.float64), 1e-6)
        zf = lambda X: ((X - mu) / sd).astype(np.float64)
        Zev = {c: zf(F["ev"][c]) for c in CONDS}
        Zg = {c: zf(F["grid"][c]) for c in GRID_CONDITIONS}     # NaN outside split==train rows (never used)
        Z = {"clean": Zev["clean"]}
        Dl, Pa, Pb = build_deltas(Z, Zg, nat, wa, fb, grid_reals=rtr)
        w, V = eig_second_moment(Dl, centered=False)
        wc, Vc = eig_second_moment(Dl, centered=True)
        ks, cum = kstar(w); ksc, cumc = kstar(wc)
        mean_energy = float((Dl.mean(0) ** 2).sum() / (Dl ** 2).sum(0).sum() * len(Dl))
        ii = dict(feature=feat, dim=int(Dl.shape[1]), n_delta=int(len(Dl)), k_star=ks, k_star_centered=ksc,
                  trace_M=float(w.sum()), mean_shift_energy_fraction=float((Dl.mean(0) ** 2).sum() / w.sum()),
                  energy_at_k={str(k): float(cum[k - 1]) for k in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512) if k <= len(cum)},
                  energy_at_k_centered={str(k): float(cumc[k - 1]) for k in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512) if k <= len(cumc)},
                  k_at_energy={str(f): int(np.searchsorted(cum, f) + 1) for f in (0.5, 0.8, 0.9, 0.95, 0.99)},
                  top_eigvals=[float(x) for x in w[:10]])
        info[feat] = ii
        log(f"{feat}: k*={ks} (centered {ksc}); mean-shift energy frac {ii['mean_shift_energy_fraction']:.3f}")
        Uk = lambda k: V[:, :k]
        Ident = lambda X: X
        if feat == "primary":
            Pk = lambda k: (lambda X: project(X, Uk(k)))
            specs = [("A0", Ident, False), ("A1", Ident, True), ("A2", Pk(ks), True), ("A3", Pk(ks), False)]
            specs += [(f"A2_k{k}", Pk(k), True) for k in (16, 64, 256)]
            specs += [("A2c", lambda X: project(X, Vc[:, :ksc]), True)]
            for name, T, aug in specs:
                if want(name):
                    ARMS[name] = run_arm(name, df, T, Zev, Zg, aug=aug)
            # A4: hyperparameters fixed in config.json before any detection metric was looked at
            if want("A4"):
                a4p = CACHE / "a4_hist.json"
                t1 = time.time()
                if not (CACHE / "scores_A4.npz").exists():
                    g, hist, dev = train_a4(Pa, Pb, Zev["clean"][rtr])
                    a4p.write_text(json.dumps(hist, indent=1))
                    info["a4_train_hist"] = hist
                    TIMINGS["a4_train"] = time.time() - t1
                    ARMS["A4"] = run_arm("A4", df, lambda X: apply_g(g, dev, X), Zev, Zg, aug=True)
                else:
                    ARMS["A4"] = run_arm("A4", df, None, Zev, Zg, aug=True)
                    info["a4_train_hist"] = json.loads(a4p.read_text()) if a4p.exists() else None
            info["energy_pairs_check"] = dict(
                platform_vs_grid_trace=[float((Dl[:198] ** 2).sum(1).mean()), float((Dl[198:] ** 2).sum(1).mean())])
        else:
            if want("A1_sec"):
                ARMS["A1_sec"] = run_arm("A1_sec", df, Ident, Zev, Zg, aug=True)
            if want("A2_sec"):
                ARMS["A2_sec"] = run_arm("A2_sec", df, lambda X: project(X, V[:, :ks]), Zev, Zg, aug=True)
        TIMINGS[f"feature_block_{feat}"] = time.time() - t0

    # ---- evaluate
    res = {"label": LABEL, "info": info, "arms": {}}
    if NEWMODE:
        res["options"] = dict(subdir=SUBDIR, splits=str(SPLITS.relative_to(ROOT)) if SPLITS.is_relative_to(ROOT) else str(SPLITS),
                              grouped_cv=GROUPED_CV, balanced=BALANCED, group_bootstrap=GROUP_BOOT, per_pool=PER_POOL,
                              headroom_rule=HEADROOM, primary_pools=list(PRIMARY_POOLS), generators=list(GENS),
                              heldout_generators=list(HELDOUT), twin_disjoint=TWIN_DISJOINT, arms=sorted(ARMS), n_boot=N_BOOT,
                              bootstrap=("cluster: reals by device (RAISE per image), fakes by image stratified per "
                                         "generator" if GROUP_BOOT else "image-level (original pilot bootstrap)"))
    for a, S in ARMS.items():
        res["arms"][a] = {c: logo_metrics(df, S, CI[c]) for c in CONDS}
        res["arms"][a]["drift"] = drift(df, S["full"][CI["clean"]].astype(np.float64))
    if HELDOUT:
        res["heldout"] = {a: {c: {g: heldout_auc(df, S, CI[c], g) for g in HELDOUT} for c in CONDS} for a, S in ARMS.items()}
    # bootstrap
    pairs = []             # (arm, ref)
    for a in ARMS:
        if a == "A1":
            continue
        ref = "A1_sec" if a == "A2_sec" else "A1"
        if a == "A0" and "A1" not in ARMS:
            continue
        if ref in ARMS:
            pairs.append((a, ref))
    plan = None
    if GROUP_BOOT:
        plan = GroupPlan(df, real_index(df), GENS + HELDOUT, macro_gens=GENS)
        res["bootstrap_plan"] = dict(n_real=int(len(plan.R)), n_clusters=int(plan.n_clusters), n_boot=N_BOOT, seed=SEED,
                                     pools=list(PRIMARY_POOLS))
        bm = lambda a, c: plan.macro_boot(ARMS[a], CI[c])
    else:
        ridx, fidx = boot_setup(df)
        bm = lambda a, c: boot_macro_auc(ARMS[a], CI[c], ridx, fidx)
    boot = {}
    for a, ref in pairs:
        for x in (a, ref):
            if x not in boot:
                boot[x] = {c: bm(x, c) for c in CONDS}
    res["bootstrap"] = {}
    for a, ref in pairs:
        res["bootstrap"][f"{a}-{ref}"] = {}
        for c in CONDS:
            d = boot[a][c] - boot[ref][c]
            pt = res["arms"][a][c]["macro"]["auroc"] - res["arms"][ref][c]["macro"]["auroc"]
            res["bootstrap"][f"{a}-{ref}"][c] = dict(
                delta=float(pt), ci_lo=float(np.percentile(d, 2.5)), ci_hi=float(np.percentile(d, 97.5)),
                boot_mean=float(d.mean()), p_boot_le0=float((d <= 0).mean()))
    if GROUP_BOOT:            # per-arm macro CIs (primary pools)
        res["macro_ci"] = {a: {c: dict(zip(("ci_lo", "ci_hi"), pct_ci(bm(a, c)))) for c in CONDS} for a in ARMS}
        if HELDOUT:
            res["bootstrap_heldout"] = {}
            for a, ref in pairs:
                res["bootstrap_heldout"][f"{a}-{ref}"] = {}
                for g in HELDOUT:
                    res["bootstrap_heldout"][f"{a}-{ref}"][g] = {}
                    for c in CONDS:
                        d = plan.heldout_boot(ARMS[a], CI[c], g) - plan.heldout_boot(ARMS[ref], CI[c], g)
                        pt = res["heldout"][a][c][g] - res["heldout"][ref][c][g]
                        lo, hi = pct_ci(d)
                        res["bootstrap_heldout"][f"{a}-{ref}"][g][c] = dict(delta=float(pt), ci_lo=lo, ci_hi=hi)
    if PER_POOL:
        res["per_pool"] = pool_tables(df, ARMS)
        if GROUP_BOOT:
            for p, n in [(p, n) for p, n in POOLS if len(real_index(df, [p]))]:
                pl = GroupPlan(df, real_index(df, [p]), GENS + HELDOUT, macro_gens=GENS)
                for a in ARMS:
                    for c in CONDS:
                        res["per_pool"][a][c][n]["ci"] = dict(zip(("ci_lo", "ci_hi"), pct_ci(pl.macro_boot(ARMS[a], CI[c]))))
                res.setdefault("per_pool_bootstrap", {})[n] = {}
                for a, ref in pairs:
                    res["per_pool_bootstrap"][n][f"{a}-{ref}"] = {}
                    for c in ("clean", "jpeg75_resize05"):
                        d = pl.macro_boot(ARMS[a], CI[c]) - pl.macro_boot(ARMS[ref], CI[c])
                        pt = res["per_pool"][a][c][n]["macro_auroc"] - res["per_pool"][ref][c][n]["macro_auroc"]
                        lo, hi = pct_ci(d)
                        res["per_pool_bootstrap"][n][f"{a}-{ref}"][c] = dict(delta=float(pt), ci_lo=lo, ci_hi=hi)
    # ---- headroom (Amendment 3): A1 LOGO-macro at jpeg75_resize05 on the primary reals
    A = res["arms"]; m = lambda a, c: A[a][c]["macro"]["auroc"]
    if HEADROOM and "A1" in A:
        h = m("A1", "jpeg75_resize05")
        res["headroom"] = dict(A1_logo_macro_auroc_jpeg75_resize05=h, threshold=HEADROOM_AUROC,
                               has_headroom=bool(h <= HEADROOM_AUROC), pools=list(PRIMARY_POOLS),
                               max_possible_gain=float(1 - h))
    # ---- verdict
    if "A1" in A and "A2" in A:
        gain_macro = m("A2", "jpeg75_resize05") - m("A1", "jpeg75_resize05")
        gains = {"logo_macro": gain_macro}
        for g in HELDOUT:
            gains[g] = res["heldout"]["A2"]["jpeg75_resize05"][g] - res["heldout"]["A1"]["jpeg75_resize05"][g]
        gain = min(gains.values())
        clean_d = m("A2", "clean") - m("A1", "clean")
        ss_d = m("A2", "screenshot_sim") - m("A1", "screenshot_sim")
        d1, d2 = A["A1"]["drift"]["mean_drift"], A["A2"]["drift"]["mean_drift"]
        red = (d1 - d2) / d1 if d1 > 0 else float("nan")
        crit = dict(
            gain_ge_0p03=dict(value=gain, threshold=0.03, holds=bool(gain >= 0.03)),
            clean_ge_A1_minus_0p01=dict(value=clean_d, threshold=-0.01, holds=bool(clean_d >= -0.01)),
            screenshot_sim_ge_A1=dict(value=ss_d, threshold=0.0, holds=bool(ss_d >= 0)),
            drift_reduced_ge_50pct=dict(value=red, drift_A1=d1, drift_A2=d2, threshold=0.5,
                                        holds=bool(np.isfinite(red) and red >= 0.5)))
        if HELDOUT:
            crit["gain_ge_0p03"]["gains_by_endpoint"] = gains
        fail = dict(gain_lt_0p01=bool(gain < 0.01), clean_loss_gt_0p02=bool(-clean_d > 0.02))
        verdict = "PASS" if all(c["holds"] for c in crit.values()) else ("FAIL" if any(fail.values()) else "INCONCLUSIVE")
        if HEADROOM and not res["headroom"]["has_headroom"]:
            res["verdict_ignoring_headroom_rule"] = verdict
            verdict = "UNINFORMATIVE_NO_HEADROOM"
        res["PILOT_verdict"] = verdict
        res["criteria"] = crit; res["fail_conditions"] = fail
        res["primary_endpoint"] = dict(name="LOGO-macro AUROC @ jpeg75_resize05, A2 - A1", **res["bootstrap"]["A2-A1"]["jpeg75_resize05"])
    else:
        res["PILOT_verdict"] = "NOT_EVALUATED (needs A1 and A2)"
        verdict, gain = res["PILOT_verdict"], float("nan")
    if "A0" in A:
        res["sanity_A0_logo_macro_clean_auroc"] = m("A0", "clean")
    res["reference_003_B0_logo_clean"] = 0.916
    res["git_head"] = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    res["timings_s"] = dict(TIMINGS, total=time.time() - T0)
    (OUT / "results.json").write_text(json.dumps(res, indent=1))
    write_tables(res)
    write_scores_long(df, ARMS)
    log("verdict", verdict, "| gain", round(gain, 4), "| A0 clean", round(m("A0", "clean"), 4) if "A0" in A else "n/a")


ORDER = ["A0", "A1", "A2", "A3", "A2_k16", "A2_k64", "A2_k256", "A2c", "A4", "A1_sec", "A2_sec"]


def extra_sections(R, ORD):
    """Amendment-3 sections (macro CIs, held-out generators, per-real-pool LOGO tables); empty in the default run."""
    L = []
    if "macro_ci" in R:
        L += ["", "## LOGO-macro AUROC with 95% cluster-bootstrap CI (primary reals)", "",
              "| arm | " + " | ".join(CONDS) + " |", "|---|" + "---|" * len(CONDS)]
        for a in ORD:
            L.append(f"| {a} | " + " | ".join(f"{R['arms'][a][c]['macro']['auroc']:.3f} [{R['macro_ci'][a][c]['ci_lo']:.3f},"
                                              f"{R['macro_ci'][a][c]['ci_hi']:.3f}]" for c in CONDS) + " |")
    if "heldout" in R:
        L += ["", "## Held-out generator AUROC ('full' model never saw it; primary reals)", "",
              "| arm | " + " | ".join(f"{g}:{c}" for g in HELDOUT for c in ("clean", "jpeg75_resize05")) + " |",
              "|---|" + "---|" * (2 * len(HELDOUT))]
        for a in ORD:
            L.append(f"| {a} | " + " | ".join(f"{R['heldout'][a][c][g]:.3f}" for g in HELDOUT for c in ("clean", "jpeg75_resize05")) + " |")
    if "per_pool" in R:
        pools = list(next(iter(R["per_pool"].values()))["clean"])
        for c in ("clean", "jpeg75_resize05"):
            L += ["", f"## LOGO-macro AUROC per real pool, condition {c} (test reals of that pool only; fakes of g)", "",
                  "| arm | " + " | ".join(f"{n} (n={R['per_pool'][ORD[0]][c][n]['n_real']})" for n in pools) + " |", "|---|" + "---|" * len(pools)]
            for a in ORD:
                cell = lambda n: (f"{R['per_pool'][a][c][n]['macro_auroc']:.3f}" +
                                  (f" [{R['per_pool'][a][c][n]['ci']['ci_lo']:.3f},{R['per_pool'][a][c][n]['ci']['ci_hi']:.3f}]"
                                   if "ci" in R["per_pool"][a][c][n] else ""))
                L.append(f"| {a} | " + " | ".join(cell(n) for n in pools) + " |")
        if "per_pool_bootstrap" in R:
            L += ["", "## Paired cluster-bootstrap delta per real pool (LOGO-macro AUROC)", "",
                  "| pool | pair | clean | jpeg75_resize05 |", "|---|---|---|---|"]
            for n, d in R["per_pool_bootstrap"].items():
                for k, v in d.items():
                    L.append(f"| {n} | {k} | " + " | ".join(f"{v[c]['delta']:+.3f} [{v[c]['ci_lo']:+.3f},{v[c]['ci_hi']:+.3f}]"
                                                          for c in ("clean", "jpeg75_resize05")) + " |")
    return L


def write_tables(R):
    ORD = [a for a in ORDER if a in R["arms"]]
    if NEWMODE:
        o = R["options"]
        L = [f"# {LABEL}: experiment 005 (H1 nuisance projection). Not a Phase 4 verdict.", "",
             "Options: " + ", ".join(f"{k}={v}" for k, v in o.items() if k not in ("arms", "bootstrap")), f"Bootstrap: {o['bootstrap']}", "",
             f"## LOGO-macro AUROC ({len(GENS)} generators; test reals of pools {', '.join(PRIMARY_POOLS)}; fakes of g)", "",
             "| arm | " + " | ".join(CONDS) + " |", "|---|" + "---|" * len(CONDS)]
    else:
        L = ["# PILOT: experiment 005 (H1 nuisance projection). Not a Phase 4 verdict.", "",
             "## LOGO-macro AUROC (6 Synthbuster generators; test = 358 VISION test natives vs 120 fakes per g)", "",
         "| arm | " + " | ".join(CONDS) + " |", "|---|" + "---|" * len(CONDS)]
    for a in ORD:
        L.append(f"| {a} | " + " | ".join(f"{R['arms'][a][c]['macro']['auroc']:.3f}" for c in CONDS) + " |")
    L += ["", "## LOGO-macro TPR@1%FPR", "", "| arm | " + " | ".join(CONDS) + " |", "|---|" + "---|" * len(CONDS)]
    for a in ORD:
        L.append(f"| {a} | " + " | ".join(f"{R['arms'][a][c]['macro']['tpr_at_1fpr']:.3f}" for c in CONDS) + " |")
    L += ["", "## LOGO-macro AP", "", "| arm | " + " | ".join(CONDS) + " |", "|---|" + "---|" * len(CONDS)]
    for a in ORD:
        L.append(f"| {a} | " + " | ".join(f"{R['arms'][a][c]['macro']['ap']:.3f}" for c in CONDS) + " |")
    L += ["", "## Paired bootstrap delta vs A1 (A2_sec vs A1_sec), LOGO-macro AUROC, 95% CI (1000 reps, seed 0)", "",
          "| arm | " + " | ".join(CONDS) + " |", "|---|" + "---|" * len(CONDS)]
    for k, v in R["bootstrap"].items():
        L.append(f"| {k} | " + " | ".join(f"{v[c]['delta']:+.3f} [{v[c]['ci_lo']:+.3f},{v[c]['ci_hi']:+.3f}]" for c in CONDS) + " |")
    L += ["", "## Threshold drift (full-T model; threshold = 5% FPR quantile of clean calib natives; test-split clean reals)", "",
          "| arm | FPR native | FPR WA | FPR FB-HQ | drift_WA | drift_FB | mean drift |", "|---|---|---|---|---|---|---|"]
    for a in ORD:
        d = R["arms"][a]["drift"]
        L.append(f"| {a} | {d['native']:.3f} | {d['WA']:.3f} | {d['FB']:.3f} | {d['drift_WA']:.3f} | {d['drift_FB']:.3f} | {d['mean_drift']:.3f} |")
    L += ["", "## Per-generator AUROC at jpeg75_resize05", "", "| arm | " + " | ".join(GENS) + " |", "|---|" + "---|" * len(GENS)]
    for a in ORD:
        L.append(f"| {a} | " + " | ".join(f"{R['arms'][a]['jpeg75_resize05']['per_generator'][g]['auroc']:.3f}" for g in GENS) + " |")
    L += ["", "## Projector", ""]
    for f, i in R["info"].items():
        if isinstance(i, dict) and "k_star" in i:
            L.append(f"- {f} ({i['dim']}-d): k*={i['k_star']} (centered variant k*={i['k_star_centered']}); "
                     f"energy at k=16/64/256: {i['energy_at_k'].get('16'):.3f}/{i['energy_at_k'].get('64'):.3f}/{i['energy_at_k'].get('256'):.3f}; "
                     f"mean-shift share of trace(M) {i['mean_shift_energy_fraction']:.3f}")
    L += extra_sections(R, ORD)
    L += ["", f"## Verdict: PILOT_verdict = **{R['PILOT_verdict']}**", ""]
    for k, v in R.get("criteria", {}).items():
        L.append(f"- {k}: value {v['value']:.4f} (threshold {v['threshold']}) -> {'holds' if v['holds'] else 'FAILS'}")
    if "fail_conditions" in R:
        L.append(f"- fail conditions: {R['fail_conditions']}")
    if "headroom" in R:
        h = R["headroom"]
        L.append(f"- headroom rule: A1 LOGO-macro jpeg75_resize05 = {h['A1_logo_macro_auroc_jpeg75_resize05']:.4f} "
                 f"(needs <= {h['threshold']}) -> {'headroom' if h['has_headroom'] else 'NO HEADROOM (an A2-A1 test is UNINFORMATIVE_NO_HEADROOM)'}"
                 + (f"; verdict without the rule would be {R['verdict_ignoring_headroom_rule']}" if "verdict_ignoring_headroom_rule" in R else ""))
    (OUT / "tables.md").write_text("\n".join(L) + "\n")


def write_scores_long(df, ARMS):
    rows = []
    for a in [x for x in ("A0", "A1", "A2") if x in ARMS]:
        for m in MODELS:
            S = ARMS[a][m]
            for c in CONDS:
                rows.append(pd.DataFrame(dict(arm=a, model=m, condition=c, sha256=df.sha256.to_numpy(),
                                              score=S[CI[c]].astype(np.float32))))
    pd.concat(rows).to_csv(OUT / "scores_long.csv.gz", index=False)


def parse(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subdir", default="pilot")
    ap.add_argument("--splits", default=str(SPLITS))
    ap.add_argument("--out-dir", default=str(OUT))
    ap.add_argument("--label", default="PILOT")
    ap.add_argument("--arms", nargs="*", default=None, choices=ORDER)
    ap.add_argument("--generators", nargs="*", default=None)
    ap.add_argument("--heldout-generators", nargs="*", default=[])
    ap.add_argument("--grouped-cv", action="store_true")
    ap.add_argument("--balanced", action="store_true")
    ap.add_argument("--group-bootstrap", action="store_true")
    ap.add_argument("--per-pool", action="store_true")
    ap.add_argument("--headroom-rule", action="store_true")
    ap.add_argument("--twin-disjoint", action="store_true")
    ap.add_argument("--primary-pools", default="vision_native")
    return ap.parse_args(argv)


def configure(a):
    global OUT, CACHE, SPLITS, SUBDIR, LABEL, GROUPED_CV, BALANCED, GROUP_BOOT, PER_POOL, HEADROOM, PRIMARY_POOLS
    global HELDOUT, ARM_FILTER, GENS, MODELS, NEWMODE, TWIN_DISJOINT
    OUT, SPLITS = Path(a.out_dir).resolve(), Path(a.splits).resolve()
    CACHE = OUT / "cache"
    SUBDIR, LABEL = a.subdir, a.label
    GROUPED_CV, BALANCED, GROUP_BOOT, PER_POOL, HEADROOM = (a.grouped_cv, a.balanced, a.group_bootstrap, a.per_pool,
                                                            a.headroom_rule)
    PRIMARY_POOLS = tuple(x for x in a.primary_pools.split(",") if x)
    HELDOUT = list(a.heldout_generators)
    TWIN_DISJOINT = a.twin_disjoint
    ARM_FILTER = a.arms
    if a.generators:
        GENS = list(a.generators)
        MODELS = ["full"] + [f"logo_{g}" for g in GENS]
    NEWMODE = bool(GROUPED_CV or BALANCED or GROUP_BOOT or PER_POOL or HEADROOM or HELDOUT or ARM_FILTER
                   or a.generators or LABEL != "PILOT" or PRIMARY_POOLS != ("vision_native",) or TWIN_DISJOINT)


if __name__ == "__main__":
    configure(parse())
    main()
