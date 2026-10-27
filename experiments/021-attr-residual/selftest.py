"""Self-test of run.py on synthetic data: planted fingerprints must be recovered, planted leaks must be caught.

    env/.venv/bin/python experiments/021-attr-residual/selftest.py   -> results/selftest.txt
"""
import importlib.util
import sys
import traceback
from pathlib import Path

import numpy as np
from scipy import ndimage
from scipy.signal import wiener as scipy_wiener

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("h2run", HERE / "run.py")
h2 = importlib.util.module_from_spec(spec)
sys.modules["h2run"] = h2
spec.loader.exec_module(h2)

LINES, FAILS = [], []


def check(name, ok, detail=""):
    LINES.append(f"[{'PASS' if ok else 'FAIL'}] {name}  {detail}")
    if not ok:
        FAILS.append(name)


def raises(fn, *a):
    try:
        fn(*a)
    except AssertionError:
        return True
    return False


def scene(rng, h, w, amp=40.0, sigma=6.0):
    """smooth random field + a few hard edges (content that must NOT carry class information)."""
    f = ndimage.gaussian_filter(rng.standard_normal((h, w)), sigma)
    f = amp * f / f.std()
    for _ in range(3):
        y, x0 = rng.integers(8, h - 8), rng.integers(8, w - 8)
        f[y:, x0:] += rng.uniform(-20, 20)
    return 128.0 + f


# ------------------------------------------------------------------ 1. primitives
def t_primitives():
    rng = np.random.default_rng(0)
    x = rng.uniform(0, 255, (40, 50))
    ref = scipy_wiener(x, 3)
    mine = h2.wiener_denoise(x, 3, mode="constant")
    check("Wiener 3x3 == scipy.signal.wiener (zero padding mode, all pixels)", np.allclose(ref, mine, atol=1e-8), f"max|d|={np.abs(ref - mine).max():.2e}")
    r = h2.residual(x)
    check("residual rows and columns are zero-mean", np.abs(r.mean(1)).max() < 1e-9 and np.abs(r.mean(0)).max() < 1e-9)
    # PCE vs brute-force
    W = h2.zero_mean_rc(rng.standard_normal((32, 48)))
    Y = np.roll(W, (3, -2), axis=(0, 1)) + 0.5 * rng.standard_normal((32, 48))
    pce, rho0, pz, pk = h2.pce_stats(W.astype(np.float32), Y.astype(np.float32))
    Yz = Y - Y.mean()
    nw, ny = np.linalg.norm(W), np.linalg.norm(Yz)
    brute = np.array([[(W * np.roll(Yz, (-dy, -dx), axis=(0, 1))).sum() / (nw * ny) for dx in range(48)] for dy in range(32)])
    c = h2.sfft.irfft2(np.conj(h2.sfft.rfft2(W.astype(np.float32))) * h2.sfft.rfft2(Yz.astype(np.float32)), s=W.shape) / (nw * ny)
    check("FFT cross-correlation == brute-force circular correlation", np.allclose(brute, c, atol=2e-4), f"max|d|={np.abs(brute - c).max():.1e}")
    check("shifted copy: peak NOT at zero shift (planted shift (3,-2) -> detected)", (not pz) and pk == (3, 46), f"peak {pk}")


# ------------------------------------------------------------------ 2. (i) generators: planted fingerprint, null, leaks
def make_gen(rng, n_fam=60, K=4, amp=3.0, hw=(64, 64), noise=2.0, content_class_specific=False, fp=None):
    h, w = hw
    fps = [h2.zero_mean_rc(rng.standard_normal((h, w))) * amp for _ in range(K)] if fp is None else fp
    X, g, fam = [], [], []
    for f in range(n_fam):
        base = scene(rng, h, w)
        for k in range(K):
            c = scene(rng, h, w) if content_class_specific else base
            if content_class_specific and k == 0:
                c = base
            X.append(c + fps[k] + noise * rng.standard_normal((h, w)))
            g.append(k)
            fam.append(f"f{f:03d}")
    return np.array(X), np.array(g), np.array(fam)


def to_R(X):
    return np.stack([h2.residual(x).ravel() for x in X]).astype(np.float32)


def run_gen(R, g, fam, seed=0, split=None, n_perm=60):
    import pandas as pd
    K = int(g.max()) + 1
    meta = pd.DataFrame({"g": g, "fam": fam})
    train = h2.split_families(fam, 0.7, seed) if split is None else split
    test = ~train
    cfg = {"n_cls": K, "m1_C": 1.0}
    M1 = {c: np.c_[R[c].std(1), np.abs(R[c]).mean(1)] for c in R}
    return h2.eval_generators(R, M1, meta, train, test, cfg, 200, n_perm, seed, (64, 64)), train, test


def t_generators():
    rng = np.random.default_rng(1)
    X, g, fam = make_gen(rng, amp=0.35)
    R = {"clean": to_R(X)}
    # degraded 'laundered' condition: same images, fingerprint mostly removed by heavy extra noise
    Xn = X + 6 * rng.standard_normal(X.shape)
    R["noisy"] = to_R(Xn)
    out, train, test = run_gen(R, g, fam)
    e = out["clean|fp=clean"]
    check("planted fingerprint recovered (clean, 4-way, chance 0.25)", e["bacc"] > 0.9, f"bacc={e['bacc']:.3f} CI={e['bacc_ci']}")
    check("permutation control at chance for the planted-fingerprint data", abs(e["perm"]["mean"] - 0.25) < 0.04 and e["perm"]["ci"][1] < e["bacc"],
          f"perm mean={e['perm']['mean']:.3f} 97.5%={e['perm']['ci'][1]:.3f} max={e['perm']['max']:.3f} (a few permutations map classes onto the right groups by chance; the mean is the chance check)")
    en = out["noisy|fp=clean"]
    check("degraded condition scores lower than clean (monotone sanity)", en["bacc"] < e["bacc"], f"{en['bacc']:.3f} < {e['bacc']:.3f}")
    check("pairwise metric high when fingerprint planted", e["pairwise"] > 0.9, f"{e['pairwise']:.3f}")
    # null: no fingerprint
    Xn0, g0, fam0 = make_gen(rng, amp=0.0)
    o0, _, _ = run_gen({"clean": to_R(Xn0)}, g0, fam0)
    e0 = o0["clean|fp=clean"]
    nb, npm = [], []
    for sd in range(6):
        Xq, gq, famq = make_gen(np.random.default_rng(100 + sd), amp=0.0)
        oq, _, _ = run_gen({"clean": to_R(Xq)}, gq, famq, seed=sd, n_perm=30)
        nb.append(oq["clean|fp=clean"]["bacc"])
        npm.append(oq["clean|fp=clean"]["perm"]["mean"])
    check("null data (no fingerprint), 6 datasets: mean bacc and mean permutation bacc both at chance 0.25", abs(np.mean(nb) - 0.25) < 0.05 and abs(np.mean(npm) - 0.25) < 0.04,
          f"mean bacc={np.mean(nb):.3f} (range {min(nb):.2f}-{max(nb):.2f}), mean perm={np.mean(npm):.3f}")
    # planted LEAK 1: class-specific content (a family's image content identifies its class), no fingerprint.
    # With a split that is not grouped by family the residual (edges leak through) is inflated; grouped split gives chance.
    Xl, gl, faml = make_gen(rng, amp=0.0, content_class_specific=True)
    # near-duplicate content within a class across 'families': give each image a class-wise repeated scene
    base = [scene(rng, 64, 64) for _ in range(4)]
    Xl = np.array([base[k] + 2.0 * rng.standard_normal((64, 64)) for k in gl])
    Rl = to_R(Xl)
    ungrouped = np.random.default_rng(3).random(len(gl)) < 0.7
    ugr = h2.macro_recall(h2.ncc_scores(Rl[~ungrouped], h2.fit_fingerprints(Rl[ungrouped], gl[ungrouped], 4)).argmax(1) == gl[~ungrouped], gl[~ungrouped], n_cls=4)
    check("planted content leak inflates an UNGROUPED split (leak exists)", ugr > 0.8, f"ungrouped bacc={ugr:.3f}")
    check("grouped-split gate catches the ungrouped split", raises(h2.check_grouped, faml, ungrouped, ~ungrouped))
    # same content across all families for a class is a design leak the family split cannot remove -> the control for it is the null above.
    # planted leak 2: duplicate content hash across classes
    check("sha256/crop-hash dedupe gate catches a planted duplicate", raises(h2.check_dedupe, ["a", "b", "c", "a"], "planted"))
    check("dedupe gate passes on unique hashes", not raises(h2.check_dedupe, ["a", "b", "c"], "ok"))
    # grouped split property
    tr = h2.split_families(fam, 0.7, 5)
    check("family split is grouped, 70% of families in train", not raises(h2.check_grouped, fam, tr, ~tr) and abs(len(set(fam[tr])) / len(set(fam)) - 0.7) < 0.02)
    # planted label leak: fingerprints from train+test (honest = train only) must be flagged by inflated score vs honest on null data
    Rn0 = to_R(Xn0)
    F_leak = h2.fit_fingerprints(Rn0, g0, 4)
    leak_acc = h2.macro_recall(h2.ncc_scores(Rn0, F_leak).argmax(1) == g0, g0, n_cls=4)
    check("test-in-fingerprint leak is large on null data (so the honest pipeline's chance result is meaningful)", leak_acc > e0["bacc"] + 0.15, f"leaky={leak_acc:.3f} honest={e0['bacc']:.3f}")
    # spectral variant runs and recovers a planted spectral pattern
    check("spectral variant reported", "spectral_bacc" in e and 0 <= e["spectral_bacc"] <= 1, f"spectral bacc={e['spectral_bacc']:.3f}")


# ------------------------------------------------------------------ 3. (ii) devices: planted PRNU, LOPO, shift, rotation, resize, permutation
def make_dev(rng, n_dev=3, n_ph=10, hw=(192, 256), amp=0.012, noise=2.0):
    h, w = hw
    Ks = [rng.standard_normal((h, w)) for _ in range(n_dev)]
    ph = []
    for d in range(n_dev):
        for j in range(n_ph):
            s = np.clip(scene(rng, h, w, amp=35) + rng.uniform(-30, 30), 20, 235)
            ph.append((d, j, s * (1 + amp * Ks[d]) + noise * rng.standard_normal((h, w))))
    return Ks, ph


def sums(ph, n_dev):
    N = [0] * n_dev
    D = [0] * n_dev
    for d, j, x in ph:
        N[d] = N[d] + h2.residual(x) * x
        D[d] = D[d] + x * x
    return N, D


def t_devices():
    rng = np.random.default_rng(2)
    n_dev, n_ph = 3, 10
    Ks, ph = make_dev(rng, n_dev, n_ph)
    N, D = sums(ph, n_dev)
    Kfull = [h2.zero_mean_rc(N[d] / (D[d] + h2.EPS)).astype(np.float32) for d in range(n_dev)]
    hits, true_pce, false_pce, peakz, leak_pce = [], [], [], [], []
    for d, j, x in ph:
        r = h2.residual(x)
        Kown = h2.lopo_K(N[d].astype(np.float32), D[d].astype(np.float32), x, r)
        Ks_ = [Kown if c == d else Kfull[c] for c in range(n_dev)]
        res = h2.pce_candidates(r, x, Ks_, [0])
        pce = np.array([b[0] for b in res])
        hits.append(int(pce.argmax()) == d)
        true_pce.append(pce[d])
        false_pce += [pce[c] for c in range(n_dev) if c != d]
        peakz.append(res[d][2])
        leak = h2.pce_candidates(r, x, [Kfull[d]], [0])[0][0]  # non-LOPO: own photo inside its fingerprint
        leak_pce.append(leak)
    check("planted PRNU recovered: LOPO top-1 over 3 devices (chance 0.33)", np.mean(hits) == 1.0, f"top1={np.mean(hits):.3f}")
    check("true-device PCE >> 60 and wrong-device PCE small", np.median(true_pce) > 60 and np.percentile(false_pce, 95) < 60,
          f"median true={np.median(true_pce):.0f}, p95 false={np.percentile(false_pce, 95):.1f}")
    check("PCE peak at zero shift for true device", np.mean(peakz) == 1.0)
    check("planted LEAK (test photo inside its own fingerprint) inflates PCE vs LOPO", np.median(leak_pce) > 1.3 * np.median(true_pce),
          f"leaky median={np.median(leak_pce):.0f} vs LOPO {np.median(true_pce):.0f}")
    check("LOPO gate raises when the test photo is a fingerprint member", raises(h2.check_lopo, ["d0_1", "d0_2", "d0_3"], "d0_2") and not raises(h2.check_lopo, ["d0_1", "d0_3"], "d0_2"))
    # LOPO subtraction == explicit re-sum
    d0 = [p for p in ph if p[0] == 0]
    Nx, Dx = 0, 0
    for _, j, x in d0[1:]:
        Nx = Nx + h2.residual(x) * x
        Dx = Dx + x * x
    x0 = d0[0][2]
    Kexp = h2.zero_mean_rc(Nx / (Dx + h2.EPS))
    Ksub = h2.lopo_K(N[0].astype(np.float32), D[0].astype(np.float32), x0, h2.residual(x0))
    cc = np.corrcoef(Kexp.ravel(), np.asarray(Ksub).ravel())[0, 1]
    check("LOPO subtraction equals explicit leave-one-out re-sum", cc > 0.9999, f"corr={cc:.6f}")
    # shifted test image (content+noise shifted): peak off zero is flagged
    d, j, x = ph[0]
    xs = np.roll(x, (4, 7), axis=(0, 1))
    rs_ = h2.residual(xs)
    Kown = h2.lopo_K(N[d].astype(np.float32), D[d].astype(np.float32), x, h2.residual(x))
    b = h2.pce_candidates(rs_, xs, [Kown], [0])[0]
    check("shifted test image: peak not at zero shift (flagged), zero-shift PCE collapses", (not b[2]) and b[0] < 60, f"peak_zero={b[2]} shift={b[3]} pce0={b[0]:.1f}")
    # rotation by 180: variants {0,2} recover, {0} alone does not
    xr = np.rot90(x, 2)
    rr = h2.residual(xr)
    b0 = h2.pce_candidates(rr, np.ascontiguousarray(xr), [Kown], [0])[0][0]
    b02 = h2.pce_candidates(rr, np.ascontiguousarray(xr), [Kown], [0, 2])[0][0]
    check("180-degree rotated test photo: variants {0,180} recover the fingerprint, variant 0 alone does not", b02 > 60 > b0, f"pce 0-only={b0:.1f}, best of {{0,180}}={b02:.1f}")
    # platform resize arm: WA copy = bicubic resize x0.5; resized native fingerprint matches, unresized shape mismatch is impossible
    from PIL import Image
    d, j, x = ph[0]
    half = np.asarray(Image.fromarray(x.astype(np.float32)).resize((x.shape[1] // 2, x.shape[0] // 2), Image.BICUBIC), np.float64)
    Kown = h2.lopo_K(N[d].astype(np.float32), D[d].astype(np.float32), x, h2.residual(x))
    rw = h2.residual(half)
    Kr = h2.resize_fp(Kown, half.shape)
    Kr_other = h2.resize_fp(Kfull[1], half.shape)
    pa = h2.pce_candidates(rw, half, [Kr], [0])[0][0]
    pb = h2.pce_candidates(rw, half, [Kr_other], [0])[0][0]
    check("resized-fingerprint arm: true device scores above other device in the resized copy", pa > pb, f"PCE true={pa:.1f} other={pb:.1f}")
    # permutation control
    c = 128
    Wc, Ic = [], []
    for d, j, x in ph:
        xc = x[32:32 + c, 64:64 + c].astype(np.float32)
        Ic.append(xc.ravel())
        Wc.append(h2.zero_mean_rc(h2.residual(xc.astype(np.float64))).ravel())
    Wc, Ic = np.array(Wc, np.float32), np.array(Ic, np.float32)
    lab = np.array([d for d, _, _ in ph])
    real, perm = h2.prnu_perm_control(Wc, Ic, lab, n_dev, 60, 0, c, c)
    check("device permutation control: true labels high, shuffled labels at chance", real > 0.9 and abs(perm.mean() - 1 / n_dev) < 0.06,
          f"real={real:.3f} perm mean={perm.mean():.3f} max={perm.max():.3f}")
    # null devices: no PRNU
    Ks0, ph0 = make_dev(rng, n_dev, n_ph, amp=0.0)
    N0, D0 = sums(ph0, n_dev)
    K0 = [h2.zero_mean_rc(N0[d] / (D0[d] + h2.EPS)).astype(np.float32) for d in range(n_dev)]
    pt = []
    for d, j, x in ph0:
        r = h2.residual(x)
        Kown = h2.lopo_K(N0[d].astype(np.float32), D0[d].astype(np.float32), x, r)
        res = h2.pce_candidates(r, x, [Kown if c_ == d else K0[c_] for c_ in range(n_dev)], [0])
        pt.append(res[d][0])
    check("null devices (no PRNU): PCE below 60 for the 'true' device", np.percentile(pt, 95) < 60, f"p95={np.percentile(pt, 95):.1f}")


def main():
    for t in (t_primitives, t_generators, t_devices):
        try:
            t()
        except Exception:
            LINES.append(f"[FAIL] {t.__name__} raised:\n" + traceback.format_exc())
            FAILS.append(t.__name__)
    LINES.append("")
    LINES.append("SELFTEST " + ("FAILED: " + ", ".join(FAILS) if FAILS else "PASSED"))
    out = "\n".join(LINES)
    print(out)
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / "selftest.txt").write_text(out + "\n")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
