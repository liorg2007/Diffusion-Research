"""PILOT: H1 laundering-nuisance projection (experiment 005). Resumable; all outputs in pilot/.
Run: env/.venv/bin/python -I experiments/005-nuisance-projection/run.py
"""
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
SEED = 0
CONDS = EVAL_CONDITIONS
CI = {c: i for i, c in enumerate(CONDS)}
GENS = ["dalle2", "dalle3", "firefly", "midjourney-v5", "stable-diffusion-2", "stable-diffusion-xl"]
MODELS = ["full"] + [f"logo_{g}" for g in GENS]
N_BOOT = 1000
T0 = time.time()
TIMINGS = {}


def log(*a):
    print(time.strftime("%H:%M:%S"), "PILOT", *a, flush=True)


# ----------------------------------------------------------------------------- data
def load_data():
    df = pd.read_csv(ROOT / "experiments/004-phase4-features/pilot/splits_pilot.csv", keep_default_na=False,
                     dtype={"vidx": str})
    D = {}
    for feat in ("primary", "secondary"):
        D[feat] = dict(
            clean=load_features_aligned(df, feat, "clean", subdir="pilot"),
            ev={c: load_features_aligned(df, feat, c, subdir="pilot") for c in CONDS},
            grid={c: load_features_aligned(df, feat, c, subdir="pilot") for c in GRID_CONDITIONS})
        for c in CONDS:
            assert not np.isnan(D[feat]["ev"][c]).any(), (feat, c)
    return df, D


def pair_indices(df):
    """Row indices (nat, wa, fb) of train-split device triplets matched by (device, vidx)."""
    r = df[(df.label == 0) & (df.split == "train")]
    key = lambda pool: {(d, v): i for i, d, v in zip(r.index[r.pool == pool], r.device[r.pool == pool],
                                                      r.vidx[r.pool == pool])}
    nat, wa, fb = key("vision_native"), key("vision_whatsapp"), key("vision_facebook_hq")
    keys = sorted(nat)
    assert set(keys) == set(wa) == set(fb), "incomplete triplets"
    assert len(keys) == 99
    n = np.array([nat[k] for k in keys]); w = np.array([wa[k] for k in keys]); f = np.array([fb[k] for k in keys])
    assert set(n) == set(df.index[df.b0_train & (df.label == 0)])
    return n, w, f


# ----------------------------------------------------------------------------- projector
def build_deltas(Z, Zg, nat, wa, fb):
    zc = Z["clean"]
    d = [zc[wa] - zc[nat], zc[fb] - zc[nat]] + [Zg[c][nat] - zc[nat] for c in GRID_CONDITIONS]
    pa = [zc[nat], zc[nat]] + [zc[nat]] * 9
    pb = [zc[wa], zc[fb]] + [Zg[c][nat] for c in GRID_CONDITIONS]
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
        Xtr = [Xev["clean"][keep]]
        ytr = [lab[keep]]
        if aug:
            pos = {i: j for j, i in enumerate(b0idx)}
            sel = np.array([pos[i] for i in keep])
            for c in GRID_CONDITIONS:
                Xtr.append(Xg[c][sel]); ytr.append(lab[keep])
        mdl = probe.fit_probe(np.concatenate(Xtr), np.concatenate(ytr), SEED)
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


def logo_metrics(df, S, ci):
    """Per-g metrics + macro for condition index ci. S: scores dict model-> [conds, n]."""
    lab, gen, sp, pool = df.label.to_numpy(), df.generator.to_numpy(), df.split.to_numpy(), df.pool.to_numpy()
    R = np.where((lab == 0) & (sp == "test") & (pool == "vision_native"))[0]
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


def boot_setup(df):
    lab, gen, sp, pool = df.label.to_numpy(), df.generator.to_numpy(), df.split.to_numpy(), df.pool.to_numpy()
    R = np.where((lab == 0) & (sp == "test") & (pool == "vision_native"))[0]
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
    return f


# ----------------------------------------------------------------------------- main
def main():
    OUT.mkdir(exist_ok=True); CACHE.mkdir(exist_ok=True)
    df, D = load_data()
    nat, wa, fb = pair_indices(df)
    log("rows", len(df), "pairs", len(nat))
    ARMS = {}          # name -> scores
    info = {}
    for feat, tag in (("primary", ""), ("secondary", "_sec")):
        t0 = time.time()
        F = D[feat]
        mu = F["clean"][nat].mean(0).astype(np.float64)
        sd = np.maximum(F["clean"][nat].std(0).astype(np.float64), 1e-6)
        zf = lambda X: ((X - mu) / sd).astype(np.float64)
        Zev = {c: zf(F["ev"][c]) for c in CONDS}
        Zg = {c: zf(F["grid"][c]) for c in GRID_CONDITIONS}     # NaN outside split==train rows (never used)
        Z = {"clean": Zev["clean"]}
        Dl, Pa, Pb = build_deltas(Z, Zg, nat, wa, fb)
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
            ARMS["A0"] = run_arm("A0", df, Ident, Zev, Zg, aug=False)
            ARMS["A1"] = run_arm("A1", df, Ident, Zev, Zg, aug=True)
            ARMS["A2"] = run_arm("A2", df, Pk(ks), Zev, Zg, aug=True)
            ARMS["A3"] = run_arm("A3", df, Pk(ks), Zev, Zg, aug=False)
            for k in (16, 64, 256):
                ARMS[f"A2_k{k}"] = run_arm(f"A2_k{k}", df, Pk(k), Zev, Zg, aug=True)
            ARMS["A2c"] = run_arm("A2c", df, lambda X: project(X, Vc[:, :ksc]), Zev, Zg, aug=True)
            # A4: hyperparameters fixed in config.json before any detection metric was looked at
            a4p = CACHE / "a4_hist.json"
            t1 = time.time()
            if not (CACHE / "scores_A4.npz").exists():
                g, hist, dev = train_a4(Pa, Pb, Zev["clean"][nat])
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
            ARMS["A1_sec"] = run_arm("A1_sec", df, Ident, Zev, Zg, aug=True)
            ARMS["A2_sec"] = run_arm("A2_sec", df, lambda X: project(X, V[:, :ks]), Zev, Zg, aug=True)
        TIMINGS[f"feature_block_{feat}"] = time.time() - t0

    # ---- evaluate
    res = {"label": "PILOT", "info": info, "arms": {}}
    for a, S in ARMS.items():
        res["arms"][a] = {c: logo_metrics(df, S, CI[c]) for c in CONDS}
        res["arms"][a]["drift"] = drift(df, S["full"][CI["clean"]].astype(np.float64))
    # bootstrap
    ridx, fidx = boot_setup(df)
    boot = {a: {c: boot_macro_auc(ARMS[a], CI[c], ridx, fidx) for c in CONDS} for a in ARMS if a != "A1"}
    boot["A1"] = {c: boot_macro_auc(ARMS["A1"], CI[c], ridx, fidx) for c in CONDS}
    res["bootstrap"] = {}
    for a in ARMS:
        if a == "A1":
            continue
        ref = "A1_sec" if a == "A2_sec" else "A1"
        if ref not in boot:
            boot[ref] = {c: boot_macro_auc(ARMS[ref], CI[c], ridx, fidx) for c in CONDS}
        res["bootstrap"][f"{a}-{ref}"] = {}
        for c in CONDS:
            d = boot[a][c] - boot[ref][c]
            pt = res["arms"][a][c]["macro"]["auroc"] - res["arms"][ref][c]["macro"]["auroc"]
            res["bootstrap"][f"{a}-{ref}"][c] = dict(
                delta=float(pt), ci_lo=float(np.percentile(d, 2.5)), ci_hi=float(np.percentile(d, 97.5)),
                boot_mean=float(d.mean()), p_boot_le0=float((d <= 0).mean()))
    # ---- verdict
    A = res["arms"]; m = lambda a, c: A[a][c]["macro"]["auroc"]
    gain = m("A2", "jpeg75_resize05") - m("A1", "jpeg75_resize05")
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
    fail = dict(gain_lt_0p01=bool(gain < 0.01), clean_loss_gt_0p02=bool(-clean_d > 0.02))
    verdict = "PASS" if all(c["holds"] for c in crit.values()) else ("FAIL" if any(fail.values()) else "INCONCLUSIVE")
    res["PILOT_verdict"] = verdict
    res["criteria"] = crit; res["fail_conditions"] = fail
    res["primary_endpoint"] = dict(name="LOGO-macro AUROC @ jpeg75_resize05, A2 - A1", **res["bootstrap"]["A2-A1"]["jpeg75_resize05"])
    res["sanity_A0_logo_macro_clean_auroc"] = m("A0", "clean")
    res["reference_003_B0_logo_clean"] = 0.916
    res["git_head"] = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    res["timings_s"] = dict(TIMINGS, total=time.time() - T0)
    (OUT / "results.json").write_text(json.dumps(res, indent=1))
    write_tables(res)
    write_scores_long(df, ARMS)
    log("verdict", verdict, "| gain", round(gain, 4), "| A0 clean", round(m("A0", "clean"), 4))


ORDER = ["A0", "A1", "A2", "A3", "A2_k16", "A2_k64", "A2_k256", "A2c", "A4", "A1_sec", "A2_sec"]


def write_tables(R):
    L = ["# PILOT: experiment 005 (H1 nuisance projection). Not a Phase 4 verdict.", "",
         "## LOGO-macro AUROC (6 Synthbuster generators; test = 358 VISION test natives vs 120 fakes per g)", "",
         "| arm | " + " | ".join(CONDS) + " |", "|---|" + "---|" * len(CONDS)]
    for a in ORDER:
        L.append(f"| {a} | " + " | ".join(f"{R['arms'][a][c]['macro']['auroc']:.3f}" for c in CONDS) + " |")
    L += ["", "## LOGO-macro TPR@1%FPR", "", "| arm | " + " | ".join(CONDS) + " |", "|---|" + "---|" * len(CONDS)]
    for a in ORDER:
        L.append(f"| {a} | " + " | ".join(f"{R['arms'][a][c]['macro']['tpr_at_1fpr']:.3f}" for c in CONDS) + " |")
    L += ["", "## LOGO-macro AP", "", "| arm | " + " | ".join(CONDS) + " |", "|---|" + "---|" * len(CONDS)]
    for a in ORDER:
        L.append(f"| {a} | " + " | ".join(f"{R['arms'][a][c]['macro']['ap']:.3f}" for c in CONDS) + " |")
    L += ["", "## Paired bootstrap delta vs A1 (A2_sec vs A1_sec), LOGO-macro AUROC, 95% CI (1000 reps, seed 0)", "",
          "| arm | " + " | ".join(CONDS) + " |", "|---|" + "---|" * len(CONDS)]
    for k, v in R["bootstrap"].items():
        L.append(f"| {k} | " + " | ".join(f"{v[c]['delta']:+.3f} [{v[c]['ci_lo']:+.3f},{v[c]['ci_hi']:+.3f}]" for c in CONDS) + " |")
    L += ["", "## Threshold drift (full-T model; threshold = 5% FPR quantile of clean calib natives; test-split clean reals)", "",
          "| arm | FPR native | FPR WA | FPR FB-HQ | drift_WA | drift_FB | mean drift |", "|---|---|---|---|---|---|---|"]
    for a in ORDER:
        d = R["arms"][a]["drift"]
        L.append(f"| {a} | {d['native']:.3f} | {d['WA']:.3f} | {d['FB']:.3f} | {d['drift_WA']:.3f} | {d['drift_FB']:.3f} | {d['mean_drift']:.3f} |")
    L += ["", "## Per-generator AUROC at jpeg75_resize05", "", "| arm | " + " | ".join(GENS) + " |", "|---|" + "---|" * len(GENS)]
    for a in ORDER:
        L.append(f"| {a} | " + " | ".join(f"{R['arms'][a]['jpeg75_resize05']['per_generator'][g]['auroc']:.3f}" for g in GENS) + " |")
    L += ["", "## Projector", ""]
    for f, i in R["info"].items():
        if isinstance(i, dict) and "k_star" in i:
            L.append(f"- {f} ({i['dim']}-d): k*={i['k_star']} (centered variant k*={i['k_star_centered']}); "
                     f"energy at k=16/64/256: {i['energy_at_k'].get('16'):.3f}/{i['energy_at_k'].get('64'):.3f}/{i['energy_at_k'].get('256'):.3f}; "
                     f"mean-shift share of trace(M) {i['mean_shift_energy_fraction']:.3f}")
    L += ["", f"## Verdict: PILOT_verdict = **{R['PILOT_verdict']}**", ""]
    for k, v in R["criteria"].items():
        L.append(f"- {k}: value {v['value']:.4f} (threshold {v['threshold']}) -> {'holds' if v['holds'] else 'FAILS'}")
    L.append(f"- fail conditions: {R['fail_conditions']}")
    (OUT / "tables.md").write_text("\n".join(L) + "\n")


def write_scores_long(df, ARMS):
    rows = []
    for a in ("A0", "A1", "A2"):
        for m in MODELS:
            S = ARMS[a][m]
            for c in CONDS:
                rows.append(pd.DataFrame(dict(arm=a, model=m, condition=c, sha256=df.sha256.to_numpy(),
                                              score=S[CI[c]].astype(np.float32))))
    pd.concat(rows).to_csv(OUT / "scores_long.csv.gz", index=False)


if __name__ == "__main__":
    main()
