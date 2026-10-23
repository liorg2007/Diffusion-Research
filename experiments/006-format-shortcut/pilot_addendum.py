"""Pilot addendum (descriptive, post-hoc to the pilot, pre-registered for the official run in Amendment 6.5/6.6): CPU only.
  (a) cross-scale B0: probe trained on arm A (orig / fov512) features, tested on arm B, vs within-arm B0;
  (b) Laplacian texture baseline AUROC for setting (i), orig and fov512 (direction from calib natives vs train fakes).
Data: the pilot test split (358 natives in 25 devices, 6 Synthbuster generators x 60). Reads cached 004 features and the crops.
  env/.venv/bin/python -I experiments/006-format-shortcut/pilot_addendum.py
Writes pilot_addendum/{results.json,tables.md}. Does not touch pilot/."""
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np
import pandas as pd
from PIL import Image

import run_official as RO
import h4

H, OC = RO.H, RO.OC
PR = OC["profiles"]["pilot"]
OUT = HERE / "pilot_addendum"
N_BOOT = OC["aggregation"]["n_boot"]
ARMS = ["orig", "fov512"]
CONDS = {"i": "clean", "iii-p": "jpeg(90)", "iv-p": "jpeg75_resize05"}


def main():
    t0 = time.time()
    dfs = {a: RO.load_df(RO.ROOT / PR["splits"][a]) for a in ARMS}
    d0 = dfs["orig"]
    assert (dfs["fov512"].sha256 == d0.sha256).all()
    lab, gen, pool, sp = d0.label.to_numpy(), d0.generator.to_numpy(), d0.pool.to_numpy(), d0.split.to_numpy()
    gens = [g for g in RO.MAIN if (gen == g).any()]
    FFs = {"with_firefly": gens, "without_firefly": [g for g in gens if g != "firefly"]}
    test = sp == "test"
    FS = {k: h4.FakeSet(gen, v, test & (lab == 1), N_BOOT, RO.SEED) for k, v in FFs.items()}
    nat = np.where(test & (lab == 0) & (pool == "vision_native"))[0]
    RS = h4.RealSet(nat, d0.device.to_numpy(), d0.sha256.to_numpy(), N_BOOT, RO.SEED)
    # ---- B0 within / cross
    FIT, XF, B0, B0X = {}, {}, {}, {}
    for a in ARMS:
        XF[a] = {c: RO.load_aligned_multi(dfs[a], c, [PR["feature_subdirs"][a]]) for c in CONDS.values()}
        FIT[a] = RO.fit_b0(dfs[a], XF[a]["clean"], gens)
        B0[a] = {c: RO.apply_b0(FIT[a], X, gens) for c, X in XF[a].items()}
    for A in ARMS:
        for B in ARMS:
            if A != B:
                B0X[(A, B)] = {c: RO.apply_b0(FIT[A], XF[B][c], gens) for c in CONDS.values()}
    res = dict(label="DESCRIPTIVE, post-hoc to the pilot, pre-registered for the official run in Amendment 6 (items 5-6). PILOT data, not a Phase 4 verdict.",
               n_real_test=int(len(nat)), n_clusters=RS.n_clusters, generators=gens, n_boot=N_BOOT, cross_scale={}, texture={})
    for ffn, fs in FS.items():
        res["cross_scale"][ffn] = {}
        for st, c in CONDS.items():
            f = lambda d: (lambda g: d[c][g])
            within = {a: h4.macro_auroc(f(B0[a]), RS, fs) for a in ARMS}
            e = {f"within_{a}": h4.vc(*within[a]) for a in ARMS}
            for A, B in B0X:
                x = h4.macro_auroc(f(B0X[(A, B)]), RS, fs)
                e[f"{A}->{B}"] = h4.vc(*x)
                e[f"{A}->{B} minus within_{B}"] = h4.dc(*x, *within[B])
                e[f"{A}->{B} minus within_{A}"] = h4.dc(*x, *within[A])
            e["within_orig minus within_fov512"] = h4.dc(*within["orig"], *within["fov512"])
            res["cross_scale"][ffn][st] = e
    # ---- texture baseline, setting (i)
    lap = {}
    for a in ARMS:
        L = np.full(len(d0), np.nan)
        idx = np.where((test & ((pool == "vision_native") | (lab == 1))) | ((sp == "calib") & (pool == "vision_native"))
                       | ((sp == "train") & (lab == 1)))[0]
        for i in idx:
            with Image.open(dfs[a].path.iloc[i]) as im:
                L[i] = h4.laplacian_texture(H.to_unit_tensor(im).permute(1, 2, 0).numpy())
        lap[a] = L
    for ffn, fs in FS.items():
        res["texture"][ffn] = {}
        for a in ARMS:
            L = lap[a]
            r = L[np.where((sp == "calib") & (pool == "vision_native"))[0]]
            fk = L[np.where((sp == "train") & (lab == 1) & np.isin(gen, FFs[ffn]))[0]]
            d = h4.texture_direction(r, fk)
            pt, bt = h4.macro_auroc(d * L, RS, fs)
            pr, br = h4.macro_auroc(L, RS, fs)
            res["texture"][ffn][a] = dict(direction=d, auroc=h4.vc(pt, bt), auroc_plus_laplacian_fixed=h4.vc(pr, br),
                                          median_lap_native_test=float(np.median(L[nat])),
                                          median_lap_fakes_test=float(np.median(L[np.where(test & (lab == 1))[0]])),
                                          n_calib_reals=int(len(r)), n_train_fakes=int(len(fk)), _boot=(pt, bt))
        res["texture"][ffn]["orig_minus_fov512"] = h4.dc(*res["texture"][ffn]["orig"].pop("_boot"), *res["texture"][ffn]["fov512"].pop("_boot"))
    res["seconds"] = time.time() - t0
    (OUT / "results.json").write_text(json.dumps(RO.sanitize(res), indent=1))
    fc, fd = RO.fv_ci, RO.fd_ci
    L = ["# Pilot addendum: cross-scale B0 and Laplacian texture baseline (experiment 006)\n",
         "**DESCRIPTIVE, post-hoc to the pilot, pre-registered for the official run in Amendment 6 (items 5 and 6).** PILOT data (VISION + 6 Synthbuster); not a Phase 4 verdict. "
         "Code: `pilot_addendum.py` (CPU only; cached 004 features + the crops). Test split: "
         f"{res['n_real_test']} native reals in {res['n_clusters']} device clusters vs 60 test fakes per generator. AUROC = LOGO-macro over generators, "
         f"cluster-bootstrap 95% CI ({N_BOOT} resamples; differences are paired: same reals, same fakes).\n",
         "## Cross-scale B0 (probe trained on arm A, features of arm B), setting (i) = clean, (iii-p) = jpeg(90) both classes, (iv-p) = jpeg75_resize05 both classes\n",
         "| Firefly | setting | within orig | within fov512 | orig->fov512 | fov512->orig | orig->fov512 - within fov512 | fov512->orig - within orig |", "|---|---|---|---|---|---|---|---|"]
    for ffn, o in res["cross_scale"].items():
        for st, e in o.items():
            L.append(f"| {ffn} | ({st}) | {fc(e['within_orig'])} | {fc(e['within_fov512'])} | {fc(e['orig->fov512'])} | {fc(e['fov512->orig'])} | "
                     f"{fd(e['orig->fov512 minus within_fov512'])} | {fd(e['fov512->orig minus within_orig'])} |")
    L += ["\n## Laplacian texture baseline, setting (i) (mean |Laplacian| of the grey 224 centre crop; direction from calib natives vs train fakes)\n",
          "| Firefly | arm | direction (+1 = more texture is fake) | AUROC | AUROC of +Laplacian (fixed direction) | median Lap native test | median Lap fakes test | n calib reals / train fakes |", "|---|---|---|---|---|---|---|---|"]
    for ffn, o in res["texture"].items():
        for a in ARMS:
            e = o[a]
            L.append(f"| {ffn} | {a} | {e['direction']:+.0f} | {fc(e['auroc'])} | {fc(e['auroc_plus_laplacian_fixed'])} | {e['median_lap_native_test']:.4f} | "
                     f"{e['median_lap_fakes_test']:.4f} | {e['n_calib_reals']} / {e['n_train_fakes']} |")
        L.append(f"| {ffn} | orig - fov512 | | {fd(o['orig_minus_fov512'])} | | | | |")
    (OUT / "tables.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
