"""Pure-numpy helpers for the OFFICIAL H4 run (docs/04-phase4-plan.md, H4 section + Amendments 5-6).
No torch / sklearn imports here (only lib/metrics) so everything is unit-testable (lib/tests/test_h4.py).

  laplacian_texture    mean |Laplacian| of the grey 224 input (the texture baseline of Amendment 6.1)
  texture_direction    +1/-1, set ONCE on calib reals vs fakes (never on test)
  residualize_on_log_texture   Amendment 6.5: OLS of the score on log-Laplacian, fitted on calib reals only
  RealSet / FakeSet / macro_auroc   cluster-bootstrap plans with the fakes' resampling SHARED across real pools,
                       so contrasts between settings are paired on the fakes (and fully paired on the same reals)
  validity_gate / official_label    Amendment 6.1 / 6.2 label logic on point estimates (+ CI for the gate)
"""
import zlib

import numpy as np

import metrics

SUPPORTED_GAP, REFUTED_GAP, B0_GAP_MAX = 0.10, 0.03, 0.05     # H4 thresholds, unchanged by Amendment 6
EPS_TEXTURE = 1e-6


# ----------------------------------------------------------------------------- texture baseline
def laplacian_texture(x):
    """Mean |4 g(i,j) - g(i-1,j) - g(i+1,j) - g(i,j-1) - g(i,j+1)| over the interior, g = channel mean of x.
    x: float array [H, W] or [H, W, 3] in [0, 1]."""
    x = np.asarray(x, np.float32)
    g = x.mean(-1) if x.ndim == 3 else x
    return float(np.abs(4 * g[1:-1, 1:-1] - g[:-2, 1:-1] - g[2:, 1:-1] - g[1:-1, :-2] - g[1:-1, 2:]).mean())


def auroc_point(real, fake):
    """Fakes above reals, ties 0.5 (NaNs must be removed by the caller)."""
    r, f = np.asarray(real, float), np.asarray(fake, float)
    return float(metrics.weighted_auroc(r, f, np.ones((1, len(r))), np.ones((1, len(f))))[0])


def texture_direction(calib_real_lap, fake_lap):
    """+1 if higher texture means 'fake' (AUROC >= 0.5 on the calibration material), else -1. Set once, then fixed."""
    r, f = np.asarray(calib_real_lap, float), np.asarray(fake_lap, float)
    r, f = r[np.isfinite(r)], f[np.isfinite(f)]
    if not len(r) or not len(f):
        raise ValueError("texture_direction needs calibration reals and fakes")
    return 1.0 if auroc_point(r, f) >= 0.5 else -1.0


def residualize_on_log_texture(fit_score, fit_lap, score, lap):
    """OLS  score ~ a + b log(lap)  fitted on (fit_score, fit_lap) (calib-split REALS only, caller's responsibility),
    returns (residual = score - (a + b log lap) for every given row, dict(a, b, n_fit, r2))."""
    fs, fl = np.asarray(fit_score, float), np.log(np.maximum(np.asarray(fit_lap, float), EPS_TEXTURE))
    ok = np.isfinite(fs) & np.isfinite(fl)
    if ok.sum() < 3:
        raise ValueError("need >= 3 calibration reals to residualise")
    A = np.c_[np.ones(ok.sum()), fl[ok]]
    coef, *_ = np.linalg.lstsq(A, fs[ok], rcond=None)
    pred = A @ coef
    ss_tot = float(((fs[ok] - fs[ok].mean()) ** 2).sum())
    r2 = 1.0 - float(((fs[ok] - pred) ** 2).sum()) / ss_tot if ss_tot > 0 else float("nan")
    s, l = np.asarray(score, float), np.log(np.maximum(np.asarray(lap, float), EPS_TEXTURE))
    return s - (coef[0] + coef[1] * l), dict(a=float(coef[0]), b=float(coef[1]), n_fit=int(ok.sum()), r2=r2)


# ----------------------------------------------------------------------------- bootstrap plans
class FakeSet:
    """Fakes of `gens` (row indices of the rows where `fmask` is True), resampled BY IMAGE per generator. One FakeSet
    (one Wf) is shared by every RealSet, so a contrast between two real pools is paired on the fakes."""

    def __init__(self, generator, gens, fmask, n_boot, seed=0):
        gen = np.asarray(generator)
        self.gens = list(gens)
        self.F = {g: np.where(np.asarray(fmask) & (gen == g))[0] for g in self.gens}
        self.Wf = {}
        for g in self.gens:
            n = len(self.F[g])
            # one stream per generator: a generator's resampling does not depend on which other generators are listed
            rng = np.random.default_rng([seed + 7919, zlib.crc32(g.encode())])
            W = np.zeros((n_boot, n))
            if n:
                d = rng.integers(0, n, (n_boot, n))
                for b in range(n_boot):
                    W[b] = np.bincount(d[b], minlength=n)
            self.Wf[g] = W
        self.ones = {g: np.ones((1, len(self.F[g]))) for g in self.gens}
        self.n_boot = n_boot


class RealSet:
    """Reals R (row indices), resampled by cluster (device id; an empty id = its own cluster, e.g. a RAISE image)."""

    def __init__(self, R, device, sha, n_boot, seed=0):
        self.R = np.asarray(R, int)
        dev, sha = np.asarray(device), np.asarray(sha)
        clus = [dev[i] if dev[i] else f"img:{sha[i]}" for i in self.R]
        self.n_clusters = len(set(clus))
        self.Wr = metrics.bootstrap_weights(clus, None, n_boot, seed)[0] if len(self.R) else np.zeros((n_boot, 0))
        self.one = np.ones((1, len(self.R)))


def _pair_auc(s, rs, fs, g):
    sr, sf = s[rs.R], s[fs.F[g]]
    return (metrics.weighted_auroc(sr, sf, rs.one, fs.ones[g])[0],
            metrics.weighted_auroc(sr, sf, rs.Wr, fs.Wf[g]))


def macro_auroc(s, rs, fs, gens=None):
    """LOGO-macro: mean over generators of AUROC(reals vs that generator's fakes). `s`: score per row, higher = fake,
    or a callable g -> per-row scores (B0's logo model). -> (point, boot[n_boot]). NaN scores give NaN."""
    gens = list(gens if gens is not None else fs.gens)
    pt, bt = 0.0, 0.0
    for g in gens:
        sg = np.asarray(s(g) if callable(s) else s, float)
        if not len(rs.R) or not len(fs.F[g]) or np.isnan(sg[rs.R]).any() or np.isnan(sg[fs.F[g]]).any():
            return float("nan"), np.full(fs.n_boot, np.nan)
        p, b = _pair_auc(sg, rs, fs, g)
        pt += p
        bt = bt + b
    return float(pt / len(gens)), bt / len(gens)


def ci(x, lo=2.5, hi=97.5):
    return [float(np.percentile(x, lo)), float(np.percentile(x, hi))]


def vc(pt, boot):
    return dict(value=float(pt), ci=ci(boot) if np.isfinite(pt) else [float("nan")] * 2)


def dc(p1, b1, p2, b2):
    d = float(p1 - p2)
    return dict(delta=d, ci=ci(b1 - b2) if np.isfinite(d) else [float("nan")] * 2)


# ----------------------------------------------------------------------------- label logic (Amendment 6)
def validity_gate(ci_lower_ii, rigid_auc_ii, texture_auc_ii):
    """Amendment 6.1 at the best sigma: CI lower bound of AUROC(ii) > 0.5 AND RIGID AUROC(ii) > texture AUROC(ii)."""
    vals = (ci_lower_ii, rigid_auc_ii, texture_auc_ii)
    if any(v is None or not np.isfinite(v) for v in vals):
        return dict(ok=False, evaluable=False, ci_lower_gt_half=None, beats_texture=None,
                    reason="AUROC(ii) or the texture baseline is not available (no RAISE reals?)")
    a, b = ci_lower_ii > 0.5, rigid_auc_ii > texture_auc_ii
    why = [] if (a and b) else ([] if a else ["CI lower bound of AUROC(ii) <= 0.5"]) + ([] if b else ["RIGID AUROC(ii) <= texture baseline"])
    return dict(ok=bool(a and b), evaluable=True, ci_lower_gt_half=bool(a), beats_texture=bool(b), reason="; ".join(why))


def best_sigma(auc_ii):
    """Amendment 6.1: the sigma with the highest AUROC(ii) (NaN-safe; None if nothing is available)."""
    ok = {s: v for s, v in auc_ii.items() if np.isfinite(v)}
    return max(ok, key=lambda s: ok[s]) if ok else None


def official_label(auc_ii, ci_lower_ii, gaps_m_ii, b0_gap, texture_auc_ii):
    """auc_ii, ci_lower_ii, gaps_m_ii: {sigma: value}; b0_gap, texture_auc_ii: float.
    -> dict(best_sigma, gate, label). Thresholds as pre-registered (0.10 / 0.03 / B0 < 0.05); point estimates."""
    best = best_sigma(auc_ii)
    if best is None or not np.isfinite(gaps_m_ii.get(best, np.nan)):
        return dict(best_sigma=best, label="NOT_EVALUABLE",
                    gate=validity_gate(None, None, None), note="settings (ii)/(M) missing")
    gate = validity_gate(ci_lower_ii[best], auc_ii[best], texture_auc_ii)
    if not gate["ok"]:
        return dict(best_sigma=best, label="UNINFORMATIVE", gate=gate)
    if not np.isfinite(b0_gap):
        return dict(best_sigma=best, label="NOT_EVALUABLE", gate=gate, note="B0 gap missing")
    if gaps_m_ii[best] >= SUPPORTED_GAP and b0_gap < B0_GAP_MAX:
        lab = "SUPPORTED"
    elif all(g < REFUTED_GAP for g in gaps_m_ii.values() if np.isfinite(g)):
        lab = "REFUTED"
    else:
        lab = "INCONCLUSIVE"
    return dict(best_sigma=best, label=lab, gate=gate)


# ----------------------------------------------------------------------------- manipulation check
def mean_change(a, b):
    """Mean of a minus mean of b over finite entries, and the ratio of means (NaN if empty)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if not len(a) or not len(b):
        return dict(delta=float("nan"), ratio=float("nan"))
    return dict(delta=float(a.mean() - b.mean()), ratio=float(a.mean() / b.mean()) if b.mean() else float("nan"))
