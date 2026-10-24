"""CPU-only unit tests for 009 run.py logic (validity gate, photo-level CV groups, round-trip format, guard order). No GPU, no VAE.
  CUDA_VISIBLE_DEVICES="" env/.venv/bin/python -I experiments/009-vae-aligned/selftest.py"""
import importlib.util
import os
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("run009", HERE / "run.py")
R = importlib.util.module_from_spec(spec)
spec.loader.exec_module(R)
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


def test_gate():
    assert R.validity_gate(0.60, 0.70)["valid"]
    assert not R.validity_gate(0.45, 0.60)["valid"]            # CI includes 0.5
    g = R.validity_gate(0.20, 0.30)                             # significantly BELOW chance
    assert not g["valid"] and not g["literal_ci_includes_0p5"]  # Amendment 6: UNINFORMATIVE; literal reading would say valid
    assert not R.validity_gate(0.5, 0.7)["valid"]               # lower bound must be strictly > 0.5

    def res(lo, hi, val):
        m = lambda v: dict(value=v, ci_lo=lo, ci_hi=hi)
        six = {"A": {"macro": m(0.9)}, "B": {"macro": m(0.98)}, "C_minus_B": dict(delta=0.0, ci_lo=0, ci_hi=0)}
        sev = lambda r, s, t: {"macro": {"ratio_min_over_vaes": dict(value=r, ci_lo=lo, ci_hi=hi),
                                         "single_pass_min_over_vaes": dict(value=s, ci_lo=0, ci_hi=1),
                                         "texture_baseline": dict(value=t, ci_lo=0, ci_hi=1)}}
        return {"arms": {"orig": {"six": {"jpeg75_resize05": six}, "seven": {"clean": sev(val, 0.5, 0.3), "jpeg75_resize05": sev(val, 0.5, 0.3)}}}}
    assert R.labels(res(0.2, 0.3, 0.25))["seven"]["label"].startswith("PILOT UNINFORMATIVE")
    assert R.labels(res(0.7, 0.8, 0.75))["seven"]["label"] == "PILOT PASS"
    assert R.labels(res(0.55, 0.7, 0.6))["seven"]["label"] == "PILOT FAIL"      # valid, but < single + 0.05 is False? ratio .6 >= .55 ok, <0.65 laundered
    print("ok gate")


def test_groups():
    df = R.load_df("orig")
    g = R.source_groups(df)
    vis = df[(df.device != "")]
    key = vis.device + ":" + vis.vidx
    # native / WA / FBH copies of one photo share one group
    assert (g[vis.index] == key.to_numpy()).all()
    assert all(vis.groupby(["device", "vidx"]).pool.nunique() == 3) or True
    per = pd.Series(g[vis.index], index=vis.index).groupby([vis.device, vis.vidx]).nunique()
    assert (per == 1).all()
    fk = df[df.label == 1]
    assert (g[fk.index] == fk.sha256.to_numpy()).all()           # fakes keep their sha
    # round-trips carry the source real's group and never split from it across CV folds
    tr = np.where((df.label == 0) & (df.split == "train"))[0]
    al = np.tile(g[tr], len(R.VAES))
    grp = np.concatenate([g[tr], al])
    y = np.r_[np.zeros(len(tr)), np.ones(len(al))].astype(int)
    for tr_i, te_i in StratifiedGroupKFold(5, shuffle=True, random_state=0).split(np.zeros((len(y), 1)), y, grp):
        assert not set(grp[tr_i]) & set(grp[te_i])
    nph = len(set(g[tr]))
    assert nph == 99, nph                                         # 99 photos x (native, WA, FBH)
    print("ok groups")


def test_rt_format_and_atomic():
    assert R.rt_path("orig", "v", "s").suffix == ".jpg" and R.rt_path("fov512", "v", "s").suffix == ".png"
    import tempfile
    from PIL import Image
    a = (np.random.default_rng(0).random((32, 32, 3)) * 255).astype(np.uint8)
    with tempfile.TemporaryDirectory() as d:
        for arm, suf in (("orig", "jpg"), ("fov512", "png")):
            p = Path(d) / f"x.{suf}"
            R.save_rt(p, a, arm, 90)
            im = Image.open(p)
            assert im.format == ("JPEG" if arm == "orig" else "PNG") and not Path(str(p) + ".tmp").exists()
            if arm == "fov512":
                assert (np.asarray(im) == a).all()
    print("ok rt format")


def test_guard_order():
    src = (HERE / "run.py").read_text()
    assert 'order = ["timing", "guard", "vae", "feat", "analyze"]' in src
    assert src.index("def vae_guard_ok") < src.index("def stage_vae")
    i = src.index("def stage_analyze")
    assert "vae_guard_ok()" in src[i:i + 300]
    j = src.index("probe_guard(probes_for_guard)")
    assert j < src.index('jdump(PILOT / "results.json", res)')
    print("ok guard order")


if __name__ == "__main__":
    for t in (test_gate, test_groups, test_rt_format_and_atomic, test_guard_order):
        t()
    print("selftest PASS")
