"""c7 (post-hoc): 2x2 decomposition of the AUROC loss, F_b8 linear probe, jpeg75_resize05, macro over 8 generators:
   real source {in-distribution (k=6, own train reals included) | held out (k=5)} x generator {seen ('full' model scores every generator; FLUX
   is unseen in both) | unseen (LOGO, E10 convention)}. k=6 'full' refit here; LOGO k=6 and the k=5 cells come from c3 / cache."""
import sys, json
from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
from common import R, ORDER, ALLGENS, JR, load, auroc
D = R.Data(load_feats=True)
ci = D.conds.index(JR)
Ff = {g: D.idx("test", None, gens=[g], label=1) for g in ALLGENS}
s6 = R.fit_scores(D, "F_b8", tuple(ORDER), "full")[0][ci]
c3 = json.load(open(Path(__file__).resolve().parent / "c3_capacity.json"))
out = {}
for S in ORDER:
    rr = R.src_eval_rows(D, S)
    out[S] = dict(in_seen=float(np.mean([auroc(s6[rr], s6[Ff[g]]) for g in ALLGENS])), in_unseen=c3["in_dist"][S]["auroc_in_dist_k6"],
                  out_seen=c3["mlp"][S]["linear_full_macro"], out_unseen=c3["in_dist"][S]["auroc_heldout_k5"])
out["MEAN"] = {k: float(np.mean([out[S][k] for S in ORDER])) for k in out[ORDER[0]]}
json.dump(out, open(Path(__file__).resolve().parent / "c7_ceiling.json", "w"), indent=1)
print(f"{'S':6s} {'in/seen':>8s} {'in/LOGO':>8s} {'out/seen':>8s} {'out/LOGO':>8s}")
for S in ORDER + ["MEAN"]:
    o = out[S]; print(f"{S:6s} {o['in_seen']:8.3f} {o['in_unseen']:8.3f} {o['out_seen']:8.3f} {o['out_unseen']:8.3f}")
