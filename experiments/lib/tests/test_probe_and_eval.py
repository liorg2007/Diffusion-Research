import _path  # noqa
import json
import subprocess
import sys

import numpy as np
import pandas as pd

import format_bias_probe
import probe
import run_eval
import synth
from manifest import assign_splits, write_manifest
from preprocess import preprocess_manifest
from _path import TESTS as TESTS_DIR


def _feature_frame(seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for c in range(4):
        rows += [dict(label=0, generator="real", camera_or_platform=f"c{c}") for _ in range(40)]
    for g in "ABC":
        rows += [dict(label=1, generator=g, camera_or_platform="n/a") for _ in range(40)]
    df = pd.DataFrame(rows)
    df["path"], df["source_dataset"], df["license"], df["sha256"] = [f"p{i}" for i in range(len(df))], "s", "l", ""
    df["split"] = ""
    df = assign_splits(df, "C", seed=seed)
    X = rng.normal(size=(len(df), 8))
    X[:, 1:] += 0  # col 0 = row id (spy), col 1 = signal
    X[:, 1] += 3.0 * df.label.to_numpy()
    X[:, 0] = np.arange(len(df))
    return df, X


def test_fit_probe_separable():
    df, X = _feature_frame()
    m = probe.fit_probe(X[:, 1:], df.label.to_numpy())
    s = probe.predict_scores(m, X[:, 1:])
    assert ((s > .5) == df.label.to_numpy()).mean() > 0.9


def test_logo_never_trains_on_heldout_or_test_rows():
    df, X = _feature_frame()
    perm = np.random.default_rng(1).permutation(len(df))  # col 0: random unique id (noise feature)
    X[:, 0] = perm
    inv = np.argsort(perm)
    calls, orig = [], probe.fit_probe

    def spy(Xtr, y, seed=0, **kw):
        calls.append(inv[Xtr[:, 0].astype(int)])
        return orig(Xtr, y, seed, **kw)

    probe.fit_probe = spy
    try:
        res = probe.logo_eval(df, X, X, n_boot=20)
    finally:
        probe.fit_probe = orig
    gens = sorted(res["folds"])
    assert gens == ["A", "B", "C"] and len(calls) == 3
    for g, ids in zip(gens, calls):
        sub = df.iloc[ids]
        assert (sub.split == "train").all(), "only train-split rows may be used for fitting"
        assert g not in set(sub.generator) and "C" not in set(sub.generator)
        assert set(sub.generator) >= {"real"}
    for g, f in res["folds"].items():
        assert f["auroc"]["value"] > 0.9 and f["n_pos"] == (df.generator == g).sum()
        assert f["n_neg"] == ((df.label == 0) & (df.split == "test")).sum()
    assert res["macro"]["auroc"] > 0.9


def _prepared(tmp_path, leaky=True, size=64):
    src = tmp_path / "src"; src.mkdir()
    raw = synth.make_dataset(src, leaky=leaky, size=96)
    pm, side, rep = preprocess_manifest(raw, tmp_path / "proc", size=size)
    assert rep["n_skipped_small"] == 2
    pm = assign_splits(pm, "genC", seed=0)
    write_manifest(pm, tmp_path / "proc" / "manifest_processed.csv")
    return pm, side


def test_format_bias_detects_leakage_and_clean_control(tmp_path):
    (tmp_path / "leaky").mkdir()
    pm, side = _prepared(tmp_path / "leaky", leaky=True)
    r = format_bias_probe.format_bias_report(pm, side, n_boot=20)
    assert r["auroc"] > 0.9 and r["leakage_flag"] and r["used_sidecar"]
    (tmp_path / "clean").mkdir()
    pm2, side2 = _prepared(tmp_path / "clean", leaky=False)
    r2 = format_bias_probe.format_bias_report(pm2, side2, n_boot=20)
    assert r2["auroc"] < 0.8 and not r2["leakage_flag"], r2


def test_run_eval_scorer_end_to_end(tmp_path):
    pm, _ = _prepared(tmp_path)
    out = tmp_path / "res"
    args = run_eval.parse(["--manifest", str(tmp_path / "proc" / "manifest_processed.csv"), "--out", str(out),
                           "--conditions", "clean", "jpeg75_resize05", "social_sim", "--scorer",
                           "dummy_scorer:score", "--path", str(TESTS_DIR), "--n-boot", "30", "--seed", "5"])
    run_eval.run(args)
    r = json.load(open(out / "results.json"))
    assert r["seed"] == 5 and r["git"]["commit"] and "started_utc" in r and "finished_utc" in r
    assert r["config"]["scorer"] == "dummy_scorer:score" and r["n_counts"]["total"] == len(pm)
    assert r["format_bias"]["auroc"] > 0.9  # synthetic set is deliberately leaky
    assert set(r["conditions"]) == {"clean", "jpeg75_resize05", "social_sim"}
    c = r["conditions"]["clean"]
    assert set(c["per_generator"]) == {"genA", "genB", "genC"}
    assert c["overall"]["auroc"]["value"] > 0.9
    assert c["per_generator"]["genC"]["n_pos"] == 16
    for k in ("ap", "bal_acc", "bal_acc_fpr5", "fpr_at_95tpr", "tpr_at_1fpr"):
        assert k in c["overall"]
    # determinism: same seed -> identical numbers
    out2 = tmp_path / "res2"
    args.out = str(out2)
    run_eval.run(args)
    r2 = json.load(open(out2 / "results.json"))
    assert r["conditions"] == r2["conditions"]


def test_run_eval_backbone_probe_logo_and_cli(tmp_path):
    pm, _ = _prepared(tmp_path)
    out = tmp_path / "resb"
    cmd = [sys.executable, str(run_eval.HERE + "/run_eval.py"), "--manifest",
           str(tmp_path / "proc" / "manifest_processed.csv"), "--out", str(out), "--conditions", "clean",
           "jpeg75_resize05", "--backbone", "py:tiny_backbones:tiny_cli_backbone", "--path", str(TESTS_DIR),
           "--logo", "--n-boot", "20", "--exp-dir", str(tmp_path / "exp"), "--batch-size", "8"]
    p = subprocess.run(cmd, capture_output=True, text=True)
    assert p.returncode == 0, p.stderr[-2000:]
    r = json.load(open(out / "results.json"))
    c = r["conditions"]["clean"]
    assert set(c["logo"]["folds"]) == {"genA", "genB", "genC"} and "macro" in c["logo"]
    assert c["overall"]["auroc"]["value"] is not None
    assert (tmp_path / "exp" / "cache" / "tiny-dinov3_clean.npz").exists()
    assert (tmp_path / "exp" / "cache" / "tiny-dinov3_jpeg75_resize05.npz").exists()
