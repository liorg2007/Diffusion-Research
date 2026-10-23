"""005 run.py Amendment-3 options: bootstrap plan, per-pool tables, headroom logic helpers, CLI flags."""
import _path  # noqa
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import metrics

F = Path(__file__).resolve().parents[2] / "005-nuisance-projection" / "run.py"
sys.path.insert(0, str(F.parents[1] / "004-phase4-features"))
spec = importlib.util.spec_from_file_location("h1run_under_test", F)
H1 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(H1)


def _df_arms(seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for d in range(6):
        for p in range(10):
            for pool in ("vision_native", "vision_whatsapp", "vision_facebook_hq"):
                rows.append(dict(label=0, split="test", pool=pool, device=f"D{d}", generator="real", sha256=f"r{d}{p}{pool}"))
    for i in range(12):
        rows.append(dict(label=0, split="test", pool="raise", device="", generator="real", sha256=f"raise{i}"))
    for g in ("a", "b", "x"):
        for i in range(30):
            rows.append(dict(label=1, split="test", pool="fake", device="", generator=g, sha256=f"f{g}{i}"))
    df = pd.DataFrame(rows)
    n = len(df)
    S = {}
    for m in ["full", "logo_a", "logo_b"]:
        s = rng.normal(0, 1, n) + 2.5 * df.label.to_numpy()
        S[m] = np.stack([s + 0.1 * c for c in range(len(H1.CONDS))]).astype(np.float32)
    return df, S


def test_group_plan_and_pool_tables_and_cluster_arrays():
    old = H1.GENS
    H1.GENS = ["a", "b"]
    H1.HELDOUT = ["x"]
    try:
        df, S = _df_arms()
        R = H1.real_index(df)
        assert len(R) == 60 and set(df.pool.to_numpy()[R]) == {"vision_native"}
        cl = H1.clusters_for(df, H1.real_index(df, ["vision_native", "raise"]))
        assert len(set(cl.tolist())) == 6 + 12 and sum(c.startswith("img:") for c in cl) == 12
        plan = H1.GroupPlan(df, R, ["a", "b", "x"], macro_gens=["a", "b"], n_boot=50)
        b = plan.macro_boot(S, 0)
        ref = np.mean([metrics.weighted_auroc(S[f"logo_{g}"][0][R].astype(float), S[f"logo_{g}"][0][plan.F[g]].astype(float),
                                              plan.Wr, plan.Wf[:, plan.cols[g]]) for g in ("a", "b")], 0)
        assert np.allclose(b, ref) and plan.n_clusters == 6
        point = H1.logo_metrics(df, S, 0)["macro"]["auroc"]
        assert abs(point - np.mean([metrics.weighted_auroc(S[f"logo_{g}"][0][R].astype(float), S[f"logo_{g}"][0][np.where(df.generator.to_numpy() == g)[0]].astype(float),
                                                          np.ones((1, len(R))), np.ones((1, 30)))[0] for g in ("a", "b")])) < 1e-12
        pt = H1.pool_tables(df, {"A0": S})["A0"]["clean"]
        assert set(pt) == {"native", "WA", "FBH", "RAISE"} and pt["RAISE"]["n_real"] == 12 and pt["native"]["n_real"] == 60
        assert "x" in pt["WA"]["heldout"] and abs(pt["native"]["macro_auroc"] - point) < 1e-12
    finally:
        H1.GENS = old
        H1.HELDOUT = []


def test_build_deltas_default_unchanged_and_real_train_indices():
    rng = np.random.default_rng(0)
    nat, wa, fb = np.array([0, 1, 2]), np.array([3, 4, 5]), np.array([6, 7, 8])
    zc = rng.normal(size=(12, 4))
    Zg = {c: rng.normal(size=(12, 4)) for c in H1.GRID_CONDITIONS}
    a = H1.build_deltas({"clean": zc}, Zg, nat, wa, fb)
    b = H1.build_deltas({"clean": zc}, Zg, nat, wa, fb, grid_reals=nat)
    assert all(np.array_equal(x, y) for x, y in zip(a, b))
    c = H1.build_deltas({"clean": zc}, Zg, nat, wa, fb, grid_reals=np.array([0, 1, 2, 10, 11]))
    assert len(c[0]) == 2 * 3 + 9 * 5
    df = pd.DataFrame(dict(b0_train=[True] * 3 + [False] * 6 + [True] * 3, label=[0] * 12))
    assert np.array_equal(H1.real_train_indices(df.assign(pool="vision_native"), nat), [0, 1, 2, 9, 10, 11])
    assert np.array_equal(H1.real_train_indices(df.iloc[:6].assign(pool="x"), nat), nat)


def test_cli_flags_set_modes_and_defaults_stay_off():
    keep = {k: getattr(H1, k) for k in ("OUT", "CACHE", "SPLITS", "SUBDIR", "LABEL", "GROUPED_CV", "BALANCED", "GROUP_BOOT",
                                        "PER_POOL", "HEADROOM", "PRIMARY_POOLS", "HELDOUT", "ARM_FILTER", "GENS", "MODELS", "NEWMODE")}
    try:
        H1.configure(H1.parse([]))
        assert not H1.NEWMODE and H1.OUT == keep["OUT"] and H1.SUBDIR == "pilot" and not (H1.GROUPED_CV or H1.BALANCED)
        H1.configure(H1.parse(["--subdir", "pilot_fov512", "--out-dir", "/tmp/x_h1", "--arms", "A0", "A1", "--grouped-cv",
                               "--balanced", "--group-bootstrap", "--per-pool", "--headroom-rule"]))
        assert H1.NEWMODE and H1.GROUPED_CV and H1.BALANCED and H1.GROUP_BOOT and H1.PER_POOL and H1.HEADROOM
        assert H1.SUBDIR == "pilot_fov512" and H1.ARM_FILTER == ["A0", "A1"] and str(H1.CACHE) == "/tmp/x_h1/cache"
    finally:
        for k, v in keep.items():
            setattr(H1, k, v)
