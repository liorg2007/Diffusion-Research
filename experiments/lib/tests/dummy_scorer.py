import numpy as np


def score(images):
    """Mean |x - shifted x|: a crude high-frequency energy score in [0,1]-ish range."""
    out = []
    for im in images:
        a = np.asarray(im.convert("L"), np.float32)
        out.append(np.abs(np.diff(a, axis=1)).mean() / 50.0)
    return np.array(out)
