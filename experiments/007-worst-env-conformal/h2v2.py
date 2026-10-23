"""H2-v2 (Amendment 3, item 4): worst-group vs pooled split-conformal thresholds, LODO x LOPO, device bootstrap.

Called from analyze.py --v2 (kept separate so the original pilot analysis stays untouched). No import of analyze.py.

Design (all choices documented in notes.md, section "H2-v2"):
* Groups. platform (native / WA / FBH) x phone brand, brand parsed from the VISION directory in `src_path`
  (data/raw/vision/D01_Samsung_GalaxyS3Mini/...). If the brand cannot be parsed for every VISION real, the run falls
  back to platform-only groups and says so. Grouping rule (`derive_brand_labels`, per platform, on the test reals):
    1. a brand with >= MIN_GROUP reals stands alone;
    2. all other brands are merged into "other";
    3. if "other" has < MIN_GROUP reals it is absorbed by the smallest standing brand (label "<brand>+other");
       if no brand stands alone the platform is one group ("all").
  A second, coarser grouping "platform" (358+ reals each) is always computed (the critic's primary grouping).
* Cells. LODO x LOPO: a cell (d, P) = held-out device d x held-out platform P. For P in native/WA/FBH the test reals are
  d's reals on P and the calibration reals are the reals of all OTHER devices on all OTHER platforms. For the synthetic
  platforms (jpeg75_resize05, screenshot_sim) the test reals are d's natives under that transform and the calibration
  reals are all reals of the other devices (clean). A held-out device never contributes to its own threshold.
* Rules. pooled@a   : pooled split-conformal threshold at level a (the ceil((n+1)(1-a))-th smallest calibration score);
                       M2 = pooled@1%.
         wg@a       : worst-group = max over calibration groups of the PER-GROUP split-conformal threshold at level a.
         M1         : 99th percentile of the clean scores of the 10 calibration devices (different phones).
  A group (or pool) with too few reals for level a (ceil((n+1)(1-a)) > n) has no finite conformal threshold; it is
  CLIPPED to its maximum score and counted (`clipped`), so the table can say how often the guarantee was unreachable.
* Flag fake iff score > threshold. Models: logo_<g> scores reals and generator g's fakes (never saw g); a single
  pretrained model is scored on all generators. FPR counts are averaged over models, TPR over (model, generator).
* Statistics. FPR pooled over all cells, per held-out platform, per held-out group; worst-group FPR = max over held-out
  groups. TPR is the n-weighted mean over cells. 95% CIs resample the 25 DEVICES (all their cells) with replacement.
* Decisive comparison ("matched TPR"): the pooled rule sweeps a fine alpha grid, giving a (TPR, worst-group FPR) frontier;
  for each alpha in {0.5,1,2,5}% the worst-group rule's operating point is compared with the pooled frontier
  interpolated at the SAME TPR. diff = FPR(worst-group) - FPR(pooled at matched TPR); diff < 0 means worst-group buys
  something a smaller pooled alpha does not.
* Verdict (PROPOSED by the code author, not pre-registered): SUPPORTED if the diff CI upper bound is < 0 at >= 3 of 4
  alphas (clean TPR); NOT_SUPPORTED if the diff point estimate is >= 0 at >= 3 of 4 alphas; else INCONCLUSIVE.
  "Problem is real" = M1 or M2 FPR > 2% on some held-out platform or held-out group.
"""
import json
import math
import re

import numpy as np
import pandas as pd

MIN_GROUP = 100
ALPHAS = (0.005, 0.01, 0.02, 0.05)
PLAT_ORDER = ["native", "WA", "FBH", "jpeg75_resize05", "screenshot_sim"]
REAL_PLATS = ("native", "WA", "FBH")
GRID = sorted(set(round(float(a), 6) for a in np.geomspace(0.0005, 0.2, 25)) | set(ALPHAS))
TPR_CONDS = ("clean", "jpeg75_resize05")
BRAND_RE = re.compile(r"/D\d+_([A-Za-z0-9]+)_")
N_BOOT = 2000
SEED = 0


# ----------------------------------------------------------------------------- thresholds
def conformal_threshold(sorted_scores, alpha):
    """Split-conformal threshold from a SORTED calibration array. Returns (threshold, clipped).
    k = ceil((n+1)(1-alpha)); if k > n there is no finite threshold with the guarantee -> (max score, True)."""
    n = len(sorted_scores)
    if n == 0:
        return -np.inf, True
    k = math.ceil((n + 1) * (1 - alpha) - 1e-12)
    return (float(sorted_scores[-1]), True) if k > n else (float(sorted_scores[k - 1]), False)


def rule_names():
    return [f"pooled@{a:g}" for a in GRID] + [f"wg@{a:g}" for a in ALPHAS] + ["M1"]


# ----------------------------------------------------------------------------- groups
def parse_brand(src_path):
    m = BRAND_RE.search(str(src_path).replace("\\", "/"))
    return m.group(1) if m else None


def add_brand(D):
    """Adds `brand` to D.test / D.calib; returns True if every VISION real has a parseable brand."""
    ok = True
    for name in ("test", "calib"):
        df = getattr(D, name)
        df["brand"] = [parse_brand(p) for p in df.src_path]
        ok = ok and bool(df.brand.notna().all())
    return ok


def derive_brand_labels(T, min_n=MIN_GROUP):
    """{(platform, brand): label} by the rule in the module docstring (counts on the test reals T)."""
    out = {}
    for plat, g in T.groupby("plat"):
        cnt = g.groupby("brand").size()
        big = cnt[cnt >= min_n]
        small = cnt[cnt < min_n]
        if len(big) == 0:
            for b in cnt.index:
                out[(plat, b)] = "all"
            continue
        absorb = None
        if 0 < small.sum() < min_n:
            absorb = big.idxmin()
        for b in cnt.index:
            if b in big.index:
                out[(plat, b)] = f"{b}+other" if (absorb == b) else b
            else:
                out[(plat, b)] = f"{absorb}+other" if absorb is not None else "other"
    return out


def make_grouping(D, kind, min_n=MIN_GROUP):
    """-> dict(kind, brand_labels, cal_group (array over D.test rows), dev_brand_label (device -> label per platform fn),
    groups (table))."""
    T = D.test.reset_index(drop=True)
    if kind == "platform":
        labels = {(p, b): "all" for p, b in zip(T.plat, T.brand)}
        cal_group = T.plat.to_numpy().astype(object)
        egroup = lambda P, dev: P
    else:
        labels = derive_brand_labels(T, min_n)
        cal_group = np.array([f"{p}/{labels[(p, b)]}" for p, b in zip(T.plat, T.brand)], dtype=object)
        dev_brand = T[T.plat == "native"].groupby("device").brand.first().to_dict()
        egroup = lambda P, dev: f"{P}/{labels[('native' if P not in REAL_PLATS else P, dev_brand[dev])]}"
    tab = (pd.DataFrame(dict(group=cal_group, device=T.device.to_numpy()))
           .groupby("group").agg(n_reals=("device", "size"), n_devices=("device", "nunique")).reset_index())
    members = {}
    for p, b in zip(T.plat, T.brand):
        members.setdefault(f"{p}/{labels[(p, b)]}" if kind != "platform" else p, set()).add(b)
    tab["brands"] = [",".join(sorted(members[g])) for g in tab.group]
    return dict(kind=kind, min_group=min_n, cal_group=cal_group, egroup=egroup, table=tab.to_dict("records"),
                n_groups=len(tab), min_group_size=int(tab.n_reals.min()))


# ----------------------------------------------------------------------------- cells
def build_cells(D, G, plats=PLAT_ORDER):
    """All (held-out device, held-out platform) cells with index arrays into T = D.test (reset_index)."""
    T = D.test.reset_index(drop=True)
    dev, plat = T.device.to_numpy(), T.plat.to_numpy()
    cells = []
    for P in plats:
        for d in D.devices:
            if P in REAL_PLATS:
                te = np.where((dev == d) & (plat == P))[0]
                cal = np.where((dev != d) & (plat != P))[0]
            else:
                te = np.where((dev == d) & (plat == "native"))[0]
                cal = np.where(dev != d)[0]
            cells.append(dict(device=d, platform=P, te=te, cal=cal, egroup=G["egroup"](P, d)))
    return cells


def model_cell_arrays(D, S, G, cells, gens, fakes_by_gen, m1, clip):
    """One model. S = {condition: Series by sha}. Returns K [ncell, nrule] (flag counts on the cell's test reals) and
    TPR_sum [ncell, ncond, nrule] summed over the model's generators (+ the number of generators)."""
    T = D.test.reset_index(drop=True)
    sha = T.sha256.to_numpy()
    clean = S["clean"].loc[sha].to_numpy(float)
    names = rule_names()
    R = len(names)
    K = np.zeros((len(cells), R))
    TPR = np.zeros((len(cells), len(TPR_CONDS), R))
    fk = {c: {g: S[c].loc[fakes_by_gen[g]].to_numpy(float) for g in gens} for c in TPR_CONDS if c in S}
    cal_group = G["cal_group"]
    for ci, cell in enumerate(cells):
        if cell["platform"] in REAL_PLATS:
            ts = clean[cell["te"]]
        else:
            ts = S[cell["platform"]].loc[sha[cell["te"]]].to_numpy(float)
        cal = np.sort(clean[cell["cal"]])
        thr = np.empty(R)
        for j, a in enumerate(GRID):
            t, c = conformal_threshold(cal, a)
            thr[j] = t
            clip["pooled"][a] = clip["pooled"].get(a, 0) + int(c)
        cg = cal_group[cell["cal"]]
        sc = clean[cell["cal"]]
        groups = {g: np.sort(sc[cg == g]) for g in np.unique(cg)}
        for j, a in enumerate(ALPHAS):
            ts_g = [conformal_threshold(v, a) for v in groups.values()]
            thr[len(GRID) + j] = max(t for t, _ in ts_g)
            clip["wg_groups"][a] = clip["wg_groups"].get(a, 0) + sum(int(c) for _, c in ts_g)
            clip["wg_total"][a] = clip["wg_total"].get(a, 0) + len(ts_g)
        thr[-1] = m1
        K[ci] = (ts[:, None] > thr[None, :]).sum(0)
        for k, c in enumerate(TPR_CONDS):
            if c in fk:
                for g in gens:
                    TPR[ci, k] += (fk[c][g][:, None] > thr[None, :]).mean(0)
    return K, TPR


def collect_arm(D, models, G, cells):
    """models: {model_name: {condition: Series}}. Returns dict(K, N, TPR, cells...) averaged across models."""
    T = D.test.reset_index(drop=True)
    logos = {m[len("logo_"):]: S for m, S in models.items() if m.startswith("logo_")}
    fakes_by_gen = {g: D.fakes.sha256[D.fakes.generator == g].tolist() for g in sorted(D.fakes.generator.unique())}
    if logos:
        units = [(S, [g]) for g, S in sorted(logos.items())]
        kind = "LOGO: logo_<g> scores reals and generator g's fakes"
    else:
        name = "pretrained" if "pretrained" in models else ("full" if "full" in models else sorted(models)[0])
        units = [(models[name], sorted(fakes_by_gen))]
        kind = f"single model '{name}' on all generators"
    clip = dict(pooled={}, wg_groups={}, wg_total={})
    Ksum, TPRsum, nK, nT = 0.0, 0.0, 0, 0
    for S, gens in units:
        m1 = float(np.quantile(S["clean"].loc[D.calib.sha256].to_numpy(float), 0.99, method="higher"))
        K, TPR = model_cell_arrays(D, S, G, cells, gens, fakes_by_gen, m1, clip)
        Ksum = Ksum + K
        TPRsum = TPRsum + TPR
        nK += 1
        nT += len(gens)
    N = np.array([len(c["te"]) for c in cells], float)
    return dict(K=Ksum / nK, N=N, TPR=TPRsum / nT, kind=kind, n_models=nK, n_assessments=nT,
                clip=dict(pooled_rate={f"{a:g}": clip["pooled"][a] / (len(cells) * nK) for a in GRID},
                          wg_group_rate={f"{a:g}": clip["wg_groups"][a] / clip["wg_total"][a] for a in ALPHAS}))


# ----------------------------------------------------------------------------- statistics
class Stats:
    """Device-weighted statistics for cells (arrays from collect_arm)."""

    def __init__(self, cells, arr):
        self.cells, self.arr = cells, arr
        self.devs = sorted({c["device"] for c in cells})
        di = {d: i for i, d in enumerate(self.devs)}
        self.dev_idx = np.array([di[c["device"]] for c in cells])
        self.groups = sorted({c["egroup"] for c in cells})
        self.plats = [p for p in PLAT_ORDER if any(c["platform"] == p for c in cells)]
        self.Mg = np.array([[c["egroup"] == g for g in self.groups] for c in cells], float)
        self.Mp = np.array([[c["platform"] == p for p in self.plats] for c in cells], float)

    def compute(self, W):
        """W [B, n_devices] multiplicities -> dict of arrays (B leading)."""
        K, N, TPR = self.arr["K"], self.arr["N"], self.arr["TPR"]
        Wc = W[:, self.dev_idx]
        den = Wc @ N
        out = dict(pooled=(Wc @ K) / den[:, None])
        dg = np.einsum("bc,cg,c->bg", Wc, self.Mg, N)
        ng = np.einsum("bc,cg,cr->bgr", Wc, self.Mg, K)
        with np.errstate(invalid="ignore", divide="ignore"):
            out["group"] = np.where(dg[:, :, None] > 0, ng / dg[:, :, None], np.nan)
            dp = np.einsum("bc,cp,c->bp", Wc, self.Mp, N)
            npl = np.einsum("bc,cp,cr->bpr", Wc, self.Mp, K)
            out["plat"] = np.where(dp[:, :, None] > 0, npl / dp[:, :, None], np.nan)
        out["worst_group"] = np.nanmax(out["group"], axis=1)
        out["tpr"] = np.einsum("bc,c,ctr->btr", Wc, N, TPR) / den[:, None, None]
        return out


def device_weights(n_dev, n_boot=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, n_dev, (n_boot, n_dev))
    return np.stack([np.bincount(d, minlength=n_dev) for d in draws]).astype(float)


def matched_diff(st, ridx_pooled, ridx_wg, cond_i, metric):
    """For every replicate: wg operating point minus the pooled frontier interpolated at the same TPR.
    st: Stats.compute output (B leading). metric: 'worst_group' or 'pooled'. Returns (diff [B], clamped [B] bool)."""
    x = st["tpr"][:, cond_i, :][:, ridx_pooled]                     # [B, grid] pooled TPR (non-decreasing in alpha)
    y = st[metric][:, ridx_pooled]
    xq = st["tpr"][:, cond_i, ridx_wg]
    yq = st[metric][:, ridx_wg]
    diff = np.empty(len(x))
    clamped = np.zeros(len(x), bool)
    for b in range(len(x)):
        xm = np.maximum.accumulate(x[b])
        clamped[b] = bool(xq[b] < xm[0] or xq[b] > xm[-1])
        diff[b] = yq[b] - np.interp(xq[b], xm, y[b])
    return diff, clamped


# ----------------------------------------------------------------------------- one arm
def pct(v, q):
    return float(np.nanpercentile(v, q))


def analyze_arm_v2(D, models, G, n_boot=N_BOOT):
    avail = [p for p in PLAT_ORDER if p in REAL_PLATS or all(p in S for S in models.values())]
    cells = build_cells(D, G, avail)
    arr = collect_arm(D, models, G, cells)
    st = Stats(cells, arr)
    names = rule_names()
    R = {n: i for i, n in enumerate(names)}
    one = np.ones((1, len(st.devs)))
    pt = st.compute(one)
    W = device_weights(len(st.devs), n_boot)
    bs = st.compute(W)
    ci = lambda a: [pct(a, 2.5), pct(a, 97.5)]
    pg = [R[f"pooled@{a:g}"] for a in GRID]
    out = dict(kind=arr["kind"], n_cells=len(cells), n_devices=len(st.devs), n_models=arr["n_models"], clipped=arr["clip"],
               eval_groups=st.groups, platforms=st.plats)
    # rule table
    rules = {}
    for name in names:
        j = R[name]
        if name.startswith("pooled@") and float(name.split("@")[1]) not in set(ALPHAS):
            continue
        rules[name] = dict(
            fpr_pooled=float(pt["pooled"][0, j]), fpr_pooled_ci=ci(bs["pooled"][:, j]),
            fpr_worst_group=float(pt["worst_group"][0, j]), fpr_worst_group_ci=ci(bs["worst_group"][:, j]),
            tpr={c: dict(value=float(pt["tpr"][0, k, j]), ci=ci(bs["tpr"][:, k, j])) for k, c in enumerate(TPR_CONDS)},
            fpr_by_platform={p: float(pt["plat"][0, i, j]) for i, p in enumerate(st.plats)},
            fpr_by_group={g: float(pt["group"][0, i, j]) for i, g in enumerate(st.groups)})
    out["rules"] = rules
    # matched-TPR comparison
    mt = {}
    for a in ALPHAS:
        j = R[f"wg@{a:g}"]
        mt[f"{a:g}"] = {}
        for k, c in enumerate(TPR_CONDS):
            row = {}
            for metric in ("worst_group", "pooled"):
                d_pt, cl_pt = matched_diff(pt, pg, j, k, metric)
                d_b, cl_b = matched_diff(bs, pg, j, k, metric)
                row[metric] = dict(diff=float(d_pt[0]), ci=ci(d_b), p_le0=float((d_b <= 0).mean()),
                                   clamped_point=bool(cl_pt[0]), clamped_boot_share=float(cl_b.mean()))
            mt[f"{a:g}"][c] = row
    out["matched_tpr"] = mt
    # verdict + problem is real
    wgd = [mt[f"{a:g}"]["clean"]["worst_group"] for a in ALPHAS]
    sup = sum(r["ci"][1] < 0 for r in wgd)
    nsup = sum(r["diff"] >= 0 for r in wgd)
    verdict = "SUPPORTED" if sup >= 3 else ("NOT_SUPPORTED" if nsup >= 3 else "INCONCLUSIVE")
    exceed = []
    for name in ("M1", "pooled@0.01"):
        j = R[name]
        exceed += [f"{name}/platform/{p}" for i, p in enumerate(st.plats) if pt["plat"][0, i, j] > 0.02]
        exceed += [f"{name}/group/{g}" for i, g in enumerate(st.groups) if pt["group"][0, i, j] > 0.02]
    out["verdict"] = dict(verdict=verdict, n_alphas_ci_upper_lt0=int(sup), n_alphas_diff_ge0=int(nsup),
                          criterion="worst-group FPR at matched clean TPR vs pooled frontier; SUPPORTED if CI upper<0 at >=3/4 alphas, "
                                    "NOT_SUPPORTED if diff>=0 at >=3/4 alphas (author-proposed, not pre-registered)",
                          problem_real_M1_or_M2_gt_2pct=bool(exceed), exceeding=exceed)
    return out


# ----------------------------------------------------------------------------- driver pieces
def pc(x, d=2):
    return "nan" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{100 * x:.{d}f}%"


def tables_v2(R, stem):
    L = [f"# H2-v2 tables ({stem})", "",
         "Amendment 3 item 4. LODO x LOPO cells (device x platform held out), per-group split-conformal thresholds, "
         "95% CIs by resampling the 25 test devices. Flag fake iff score > threshold. Rule definitions and the grouping rule: "
         "h2v2.py docstring / notes.md.", ""]
    if R.get("PILOT"):
        L += ["PILOT data (VISION + 6 Synthbuster): not a Phase 4 verdict.", ""]
    L += [f"Brand derivable from src_path: **{R['brand_derivable']}**. Groupings: {', '.join(R['groupings'])}.", ""]
    for arm, per in R["arms"].items():
        for gk, r in per.items():
            gi = R["grouping_info"][gk]
            L += [f"## {arm} / grouping {gk}  [{r['kind']}]", "",
                  f"Held-out groups ({len(r['eval_groups'])}, min size {gi['min_group_size']} reals): "
                  + "; ".join(f"{t['group']} (n={t['n_reals']}, {t['n_devices']} dev)" for t in gi["table"]), ""]
            L += ["| rule | FPR pooled [CI] | worst-group FPR [CI] | TPR clean [CI] | TPR jpeg75_resize05 [CI] |", "|---|---|---|---|---|"]
            for name, x in r["rules"].items():
                L.append(f"| {name} | {pc(x['fpr_pooled'])} [{pc(x['fpr_pooled_ci'][0])}, {pc(x['fpr_pooled_ci'][1])}] | "
                         f"{pc(x['fpr_worst_group'])} [{pc(x['fpr_worst_group_ci'][0])}, {pc(x['fpr_worst_group_ci'][1])}] | "
                         + " | ".join(f"{pc(x['tpr'][c]['value'], 1)} [{pc(x['tpr'][c]['ci'][0], 1)}, {pc(x['tpr'][c]['ci'][1], 1)}]"
                                      for c in TPR_CONDS) + " |")
            L += ["", "Share of calibration groups / pools with no finite conformal threshold (clipped to the max): "
                  f"worst-group {json.dumps({k: round(v, 3) for k, v in r['clipped']['wg_group_rate'].items()})}; "
                  f"pooled {json.dumps({k: round(v, 3) for k, v in r['clipped']['pooled_rate'].items() if k in {f'{a:g}' for a in ALPHAS}})}", ""]
            L += ["### Held-out platform FPR (LOPO view, pooled over devices)", "",
                  "| rule | " + " | ".join(r["platforms"]) + " |", "|---|" + "---|" * len(r["platforms"])]
            for name in ("M1", "pooled@0.01", "wg@0.01"):
                x = r["rules"][name]["fpr_by_platform"]
                L.append(f"| {name} | " + " | ".join(pc(x[p]) for p in r["platforms"]) + " |")
            L += ["", "### Worst-group vs pooled at MATCHED clean TPR (diff = FPR(wg) - FPR(pooled interpolated at the same TPR); <0 favours worst-group)", "",
                  "| alpha | wg TPR | worst-group-FPR diff [CI] | pooled-FPR diff [CI] | clamped (boot share) |", "|---|---|---|---|---|"]
            for a in ALPHAS:
                m = r["matched_tpr"][f"{a:g}"]["clean"]
                w, p = m["worst_group"], m["pooled"]
                L.append(f"| {100 * a:g}% | {pc(r['rules'][f'wg@{a:g}']['tpr']['clean']['value'], 1)} | "
                         f"{pc(w['diff'])} [{pc(w['ci'][0])}, {pc(w['ci'][1])}] | {pc(p['diff'])} [{pc(p['ci'][0])}, {pc(p['ci'][1])}] | "
                         f"{w['clamped_boot_share']:.2f} |")
            v = r["verdict"]
            L += ["", f"**H2-v2 verdict (proposed rule): {v['verdict']}**; problem real (M1 or M2 > 2% on a held-out platform/group): "
                  f"{v['problem_real_M1_or_M2_gt_2pct']} ({', '.join(v['exceeding']) or 'none'})", ""]
    return "\n".join(L)



def run_v2(D, long, groupings=("platform_brand", "platform"), min_group=MIN_GROUP, n_boot=N_BOOT, stem="", pilot=True):
    """long: DataFrame(arm, model, condition, sha256, score). Returns the results dict."""
    brand_ok = add_brand(D)
    kinds = [g for g in groupings if brand_ok or g == "platform"]
    if not brand_ok:
        print("brand NOT derivable from src_path for all VISION reals: platform-only groups")
        D.test["brand"] = D.test["brand"].fillna("unknown")
        D.calib["brand"] = D.calib["brand"].fillna("unknown")
    G = {k: make_grouping(D, k, min_group) for k in kinds}
    R = dict(PILOT=pilot, scores_stem=stem, brand_derivable=bool(brand_ok), groupings=kinds, min_group=min_group,
             n_boot=n_boot, seed=SEED, alphas=list(ALPHAS), grid=list(GRID),
             grouping_info={k: {kk: vv for kk, vv in g.items() if kk in ("table", "n_groups", "min_group_size", "kind", "min_group")}
                            for k, g in G.items()}, arms={})
    for arm, g in long.groupby("arm", sort=True):
        models = {}
        for model, gm in g.groupby("model"):
            S = {}
            for cond, gc in gm.groupby("condition"):
                s = gc.set_index("sha256").score
                assert s.index.is_unique, f"duplicate rows {arm}/{model}/{cond}"
                S[cond] = s
            need = set(D.test.sha256) | set(D.calib.sha256)
            if "clean" not in S or not need <= set(S["clean"].index):
                print(f"skip {arm}/{model}: clean reals incomplete")
                continue
            models[model] = S
        if not models:
            continue
        R["arms"][str(arm)] = {k: analyze_arm_v2(D, models, G[k], n_boot) for k in kinds}
        print(arm, {k: R["arms"][str(arm)][k]["verdict"]["verdict"] for k in kinds})
    return R
