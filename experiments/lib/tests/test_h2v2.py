"""007 H2-v2 (analyze.py --v2 / h2v2.py): thresholds, grouping rule, cells, matched-TPR, synthetic end to end."""
import _path  # noqa
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

H007 = Path(__file__).resolve().parents[2] / "007-worst-env-conformal"
if str(H007) not in sys.path:
    sys.path.insert(0, str(H007))
import h2v2  # noqa: E402


def test_conformal_threshold_index_and_clipping():
    s = np.arange(1.0, 100.0)                       # n = 99
    t, clipped = h2v2.conformal_threshold(s, 0.01)  # ceil(100 * 0.99) = 99 -> 99th smallest = max
    assert (t, clipped) == (99.0, False)
    t, clipped = h2v2.conformal_threshold(s[:98], 0.01)   # n = 98: ceil(99*.99)=99 > 98 -> no finite threshold
    assert clipped and t == 98.0
    t, c = h2v2.conformal_threshold(np.arange(1.0, 21.0), 0.1)   # n=20: ceil(21*.9)=19 -> 19
    assert (t, c) == (19.0, False)
    # coverage: P(new real > t) <= alpha on exchangeable data
    rng = np.random.default_rng(0)
    hits = []
    for _ in range(300):
        cal = np.sort(rng.normal(size=199))
        t, _ = h2v2.conformal_threshold(cal, 0.05)
        hits.append(rng.normal(size=200) > t)
    assert np.mean(hits) <= 0.05 + 0.01


def _test_frame(sizes):
    rows = []
    for brand, n in sizes.items():
        for i in range(n):
            rows.append(dict(plat="native", brand=brand, device=f"{brand}{i % 3}"))
    return pd.DataFrame(rows)


def test_brand_grouping_rule():
    lab = h2v2.derive_brand_labels(_test_frame({"A": 120, "B": 60, "C": 30, "D": 20}), 100)
    assert lab[("native", "A")] == "A" and lab[("native", "B")] == "other" and lab[("native", "C")] == "other"
    # "other" (< min) is absorbed by the smallest standing brand
    lab = h2v2.derive_brand_labels(_test_frame({"A": 300, "B": 130, "C": 30, "D": 20}), 100)
    assert lab[("native", "A")] == "A" and lab[("native", "B")] == "B+other" and lab[("native", "C")] == "B+other"
    # nobody stands alone -> one group
    lab = h2v2.derive_brand_labels(_test_frame({"A": 99, "B": 60}), 100)
    assert set(lab.values()) == {"all"}
    assert h2v2.parse_brand("data/raw/vision/D01_Samsung_GalaxyS3Mini/nat/D01_I_nat_0025.jpg") == "Samsung"
    assert h2v2.parse_brand("data/raw/other/x.png") is None


def _synth(seed=0, n_photos=14):
    rng = np.random.default_rng(seed)
    brands = ["Apple"] * 8 + ["Samsung"] * 7 + ["Huawei"] * 5 + ["LG"] * 5            # 25 test devices
    rows, k = [], 0

    def add(dev, brand, split, photo):
        nonlocal k
        for plat in ("native", "WA", "FBH"):
            k += 1
            rows.append(dict(sha256=f"s{k}", device=dev, plat=plat, brand=brand, split=split, photo=f"{dev}_{photo}",
                             src_path=f"data/raw/vision/{dev}_{brand}_Model/x_{photo}.jpg", label=0,
                             pool={"native": "vision_native", "WA": "vision_whatsapp", "FBH": "vision_facebook_hq"}[plat],
                             vidx=str(photo)))
    for i, b in enumerate(brands):
        for p in range(n_photos):
            add(f"D{i:02d}", b, "test", p)
    for i in range(10):
        for p in range(8):
            add(f"D{30 + i}", "Asus", "calib", p)
    r = pd.DataFrame(rows)
    test, calib = r[r.split == "test"].copy(), r[r.split == "calib"].copy()
    fakes = pd.DataFrame([dict(sha256=f"f{g}{i}", generator=g, label=1) for g in ("g1", "g2") for i in range(40)])
    D = SimpleNamespace(test=test, calib=calib, fakes=fakes, devices=sorted(test.device.unique()))
    # scores: reals ~ N(0,1) + platform shift (WA higher), fakes ~ N(3,1); laundering shifts reals
    base = {"native": 0.0, "WA": 0.8, "FBH": 0.3}
    long = []
    for g in ("g1", "g2"):
        for cond, rs, fs in (("clean", 0.0, 0.0), ("jpeg75_resize05", 0.5, -0.3), ("screenshot_sim", 0.4, -0.2)):
            sr = rng.normal(0, 1, len(r)) + np.array([base[p] for p in r.plat]) + (rs if cond != "clean" else 0)
            long.append(pd.DataFrame(dict(arm="A0", model=f"logo_{g}", condition=cond, sha256=r.sha256, score=sr)))
            sf = rng.normal(3 + fs, 1, 40)
            long.append(pd.DataFrame(dict(arm="A0", model=f"logo_{g}", condition=cond,
                                          sha256=fakes.sha256[fakes.generator == g], score=sf)))
    return D, pd.concat(long, ignore_index=True)


def test_h2v2_end_to_end_synthetic_and_invariants():
    D, long = _synth()
    R = h2v2.run_v2(D, long, ("platform_brand", "platform"), min_group=100, n_boot=60, stem="synth")
    assert R["brand_derivable"] and set(R["arms"]["A0"]) == {"platform_brand", "platform"}
    info = R["grouping_info"]["platform_brand"]
    assert info["min_group_size"] >= 100 and info["n_groups"] == 3 * 2       # Apple(112)+Samsung(98 -> absorbed) ...
    for gk, r in R["arms"]["A0"].items():
        assert r["n_cells"] == 25 * 5 and r["n_devices"] == 25 and r["n_models"] == 2
        rules = r["rules"]
        # monotone in alpha: smaller alpha -> higher threshold -> no more FPR and no more TPR
        f = [rules[f"pooled@{a:g}"]["fpr_pooled"] for a in h2v2.ALPHAS]
        t = [rules[f"pooled@{a:g}"]["tpr"]["clean"]["value"] for a in h2v2.ALPHAS]
        assert f == sorted(f) and t == sorted(t)
        # worst-group threshold >= pooled threshold at the same alpha (max over groups >= pooled in expectation); FPR lower
        for a in h2v2.ALPHAS:
            assert rules[f"wg@{a:g}"]["fpr_pooled"] <= rules[f"pooled@{a:g}"]["fpr_pooled"] + 0.01
            assert rules[f"wg@{a:g}"]["tpr"]["clean"]["value"] <= rules[f"pooled@{a:g}"]["tpr"]["clean"]["value"] + 0.01
        for x in rules.values():
            assert x["fpr_pooled_ci"][0] <= x["fpr_pooled_ci"][1] and len(x["tpr"]["clean"]["ci"]) == 2
        assert r["verdict"]["verdict"] in ("SUPPORTED", "NOT_SUPPORTED", "INCONCLUSIVE")
        assert set(r["matched_tpr"]) == {f"{a:g}" for a in h2v2.ALPHAS}
    # the WA platform is shifted up: calibrating on native+FBH only must produce a larger WA FPR than native FPR
    fp = R["arms"]["A0"]["platform"]["rules"]["pooled@0.01"]["fpr_by_platform"]
    assert fp["WA"] > fp["native"]
    txt = h2v2.tables_v2(R, "synth")
    assert "MATCHED" in txt and "platform_brand" in txt


def test_h2v2_held_out_device_never_in_its_own_calibration():
    D, long = _synth(seed=3)
    h2v2.add_brand(D)
    G = h2v2.make_grouping(D, "platform")
    cells = h2v2.build_cells(D, G)
    T = D.test.reset_index(drop=True)
    for c in cells:
        assert not (set(T.device.to_numpy()[c["cal"]]) & {c["device"]}), "held-out device leaked into calibration"
        if c["platform"] in h2v2.REAL_PLATS:
            assert c["platform"] not in set(T.plat.to_numpy()[c["cal"]])
            assert set(T.plat.to_numpy()[c["te"]]) == {c["platform"]}
        else:
            assert set(T.plat.to_numpy()[c["te"]]) == {"native"}
        assert len(c["te"]) == 14


def test_h2v2_matched_diff_interpolates_and_flags_clamping():
    B = 3
    grid_tpr = np.array([0.5, 0.7, 0.9])
    st = dict(tpr=np.zeros((B, 1, 4)), worst_group=np.zeros((B, 4)), pooled=np.zeros((B, 4)))
    st["tpr"][:, 0, :3] = grid_tpr
    st["worst_group"][:, :3] = np.array([0.01, 0.03, 0.09])
    st["tpr"][:, 0, 3] = [0.6, 0.95, 0.4]            # wg operating points: inside, above, below the frontier
    st["worst_group"][:, 3] = [0.01, 0.05, 0.02]
    d, cl = h2v2.matched_diff(st, [0, 1, 2], 3, 0, "worst_group")
    assert abs(d[0] - (0.01 - 0.02)) < 1e-12 and not cl[0]            # frontier at TPR .6 = 0.02
    assert cl[1] and cl[2]
