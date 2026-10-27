"""Critic scratch: RAID 3-way (gen_1..gen_3, no unique-size gen_0) from committed OOF log-probs."""
import numpy as np
from pathlib import Path
H = Path(__file__).resolve().parents[1]
for reg in ["tmix", "clean"]:
    z = np.load(H/f"results/oof_ha1_raid_{reg}_nested_jpeg75.npz"); lp, y = z["logp"], z["y"]
    m = y > 0; k = [1, 2, 3]; p = np.array(k)[lp[m][:, k].argmax(1)]
    print(reg, "RAID 3-way gen1-3 restricted bacc", np.mean([np.mean(p[y[m] == c] == c) for c in k]).round(3), "(chance 0.333); 4-way recall", [round(np.mean(lp.argmax(1)[y == c] == c), 3) for c in range(4)])
