"""c6 (post-hoc, from results.json only): k=5 vs the BEST and WORST k=2 pair per held-out source (oracle selection, descriptive), and the
k=2 mean without pairs containing UNS; plus effect of adding each source to the other 4 (k=4 -> k=5) on each held-out source's AUROC."""
import json, itertools
from pathlib import Path
import numpy as np
from scipy.stats import fisher_exact
E10 = Path(__file__).resolve().parents[1]
H = json.load(open(E10 / "results" / "results.json"))["result"]["heldout"]
ORDER = ["VISION", "RAISE", "OI", "UNS", "WMC", "HDRP"]
def a(S, T): e = H[S]["+".join(T)]["F_b8"]["conds"]["jpeg75_resize05"]; return e["macro"][0], e["fpr"]["A"]["full"]["fpr"]
out = {}
print(f"{'S':6s} {'k5':>12s} {'mean k2':>8s} {'best k2 (pair)':>28s} {'worst k2':>8s} {'k2 mean w/o UNS':>15s}")
for S in ORDER:
    o = [s for s in ORDER if s != S]
    k5 = a(S, tuple(o))
    p = {T: a(S, T) for T in itertools.combinations(o, 2)}
    best = max(p, key=lambda T: p[T][0]); worst = min(p, key=lambda T: p[T][0])
    nou = [v[0] for T, v in p.items() if "UNS" not in T]
    out[S] = dict(k5=k5, k2_mean=float(np.mean([v[0] for v in p.values()])), best=("+".join(best), p[best]), worst=("+".join(worst), p[worst]), k2_mean_noUNS=float(np.mean(nou)))
    print(f"{S:6s} {k5[0]:.3f}/{k5[1]:.3f} {out[S]['k2_mean']:8.3f} {'+'.join(best):>14s} {p[best][0]:.3f}/{p[best][1]:.3f} {p[worst][0]:8.3f} {np.mean(nou):15.3f}")
print("\nAUROC change when source X is added to the other 4 (k=4 -> k=5), per held-out S:")
add = {}
for S in ORDER:
    o = [s for s in ORDER if s != S]
    k5 = a(S, tuple(o))[0]
    add[S] = {X: round(k5 - a(S, tuple(s for s in o if s != X))[0], 3) for X in o}
    print(S, add[S])
out["add_one"] = add
t = fisher_exact([[20, 85 - 20], [14, 115 - 14]])
out["uns_portrait_fisher_p"] = float(t.pvalue); print("UNS portrait vs landscape FP Fisher p =", round(t.pvalue, 3))
json.dump(out, open(Path(__file__).resolve().parent / "c6_subsets.json", "w"), indent=1)
