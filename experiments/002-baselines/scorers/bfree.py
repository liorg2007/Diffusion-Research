"""B-Free (Guillaro et al., CVPR 2025) DINOv2-reg4 ViT -- official GRIP-UNINA weights. NON-COMMERCIAL licence (research use only).

score(images) -> np.ndarray of LOGITS (higher = more fake; official threshold 0). Single-logit output, as in
third_party/B-Free/code/main_bfree.py.

Test-time preprocessing, reproduced from main_bfree.py: NO resize and NO pre-crop. The full-resolution RGB image ->
ToTensor -> Normalize(norm_type from config.yaml) is fed to networks.Wrapper5crops, which patch-embeds the WHOLE image
(patch 14, valid conv => floor(side/14) tokens) and then takes five 36x36-token windows (centre + 4 corners, i.e.
504x504 px crops in token space) and averages the five logits. Images <504 px are tiled (replicate-wrapped) up to 504.
For a 512x512 input the grid is 36x36, so all five windows coincide (only the last 8 px row/col are dropped).
Official code only supports batch size 1; here equal-size images are batched (mathematically identical per image).
Speed-up (result-identical): when the token grid is <=36x36 in both dims (any image <=~511 px per side after the valid-conv
floor, incl. 512x512) the five windows are the same tensor, so only one is computed (5x cheaper). Larger images use the
official Wrapper5crops forward unchanged. Verified equal to the official forward in smoke_results.json (bfree_dedupe_check).
Model/transform code is imported by path from the official repo (third_party/B-Free/code), unmodified.
"""
import sys

import numpy as np
import torch
import yaml

from ._common import THIRD_PARTY, WEIGHTS, DeviceModel, import_by_path, to_rgb

CODE = THIRD_PARTY / "B-Free" / "code"
WDIR = WEIGHTS / "b-free" / "BFREE_dino2reg4"
_M = None
_TF = None


def _load():
    global _M, _TF, _WRAP
    if _M is None:
        nets = import_by_path("bfree_networks", CODE / "networks" / "__init__.py", package=True)
        norm = import_by_path("bfree_normalization", CODE / "utils" / "normalization.py")
        cfg = yaml.safe_load(open(WDIR / "config.yaml"))
        net = nets.load_weights(nets.get_network(cfg["arch"]), str(WDIR / cfg["weights_file"]))
        from torchvision.transforms import Compose
        _TF = Compose(norm.get_list_norm(cfg["norm_type"]))
        _WRAP = sys.modules["bfree_networks.wrapper5crops"]
        _M = DeviceModel(net)
    return _M


def _fwd(net, x):
    gh, gw = net.patch_embed.grid_size
    if x.shape[-2] // 14 <= gh and x.shape[-1] // 14 <= gw:        # all 5 windows identical -> compute one
        emb = net.patch_embed.proj(x)
        if emb.shape[-2] < gh or emb.shape[-1] < gw:
            emb = _WRAP.replicate_wrap(emb, (gh, gw))
        emb = emb[:, :, :gh, :gw]
        emb = emb.flatten(2).transpose(1, 2) if net.patch_embed.flatten else emb
        return net.model(net.patch_embed.norm(emb))
    return net(x)                                                  # official 5-window forward


_WRAP = None


def score(images, batch_size=4, dedupe=True):
    m = _load()
    imgs = [to_rgb(im) for im in images]
    out = np.zeros(len(imgs), np.float64)
    groups = {}
    for i, im in enumerate(imgs):
        groups.setdefault(im.size, []).append(i)
    for idxs in groups.values():
        for j in range(0, len(idxs), batch_size):
            ii = idxs[j:j + batch_size]
            x = torch.stack([_TF(imgs[i]) for i in ii])
            y = m(_fwd if dedupe else (lambda net, t: net(t)), x)   # (B, 1) logit
            out[ii] = y[:, 0] if y.shape[1] == 1 else y[:, 1] - y[:, 0]
    return out
