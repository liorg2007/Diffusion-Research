"""RINE (Koutlis & Papadopoulos, ECCV 2024) on frozen OpenAI CLIP ViT-L/14 -- official mever-team/rine checkpoints (Apache-2.0).

Variants (all return LOGITS, higher = more fake; sigmoid(logit) is the official probability):
    score                 -> LDM-trained checkpoint (model_ldm_trainable.pth), single 224 centre crop   [default]
    score_ldm_tencrop     -> LDM-trained, TenCrop(224), mean of the 10 logits (paper's protocol for Synthbuster/commercial tools)
    score_4class          -> ProGAN 4-class checkpoint, single 224 centre crop (paper's main-benchmark protocol, repo demo)
    score_4class_tencrop  -> ProGAN 4-class, TenCrop

MPS caveat (observed on this AMD GPU): a ViT-L forward over 40 images (4 images x 10 crops) returned silently WRONG logits
(no exception; |diff| up to 0.4 vs CPU), while <=30 images matched CPU to 1e-5. The default batch sizes therefore keep
every forward <= 20 images (8 images, or 2 images x 10 crops). Do not raise batch_size without re-running smoke.py.

Test-time preprocessing, reproduced from third_party/rine/src/utils.py::get_transforms():
    NO resize. CenterCrop(224) (zero-pads if the image is smaller) -> ToTensor -> Normalize(CLIP mean/std);
    ten-crop variant: TenCrop(224) (4 corners + centre + their horizontal flips), same normalisation, logits averaged.
The model class (src/models.py) and the OpenAI CLIP architecture (third_party/CLIP/clip/model.py) are imported by path,
unmodified. The CLIP weights (ViT-L-14.pt, OpenAI) are built via clip.model.build_model exactly as clip.load(jit=False) does,
but kept in float32 (clip.load keeps fp16 on GPU; the stored weights are fp16 so this is a pure upcast).
`clip.load` is replaced by a shim so that RINE's Model class gets this local backbone instead of downloading one.
"""
import sys
import types

import numpy as np
import torch
from torchvision import transforms as T

from ._common import THIRD_PARTY, WEIGHTS, DeviceModel, batched, import_by_path, pick_device, to_rgb

CLIP_PT = WEIGHTS / "openai-clip-vitl14" / "ViT-L-14.pt"
CKPT = {"ldm": WEIGHTS / "rine" / "model_ldm_trainable.pth", "4class": WEIGHTS / "rine" / "model_4class_trainable.pth",
        "1class": WEIGHTS / "rine" / "model_1class_trainable.pth", "2class": WEIGHTS / "rine" / "model_2class_trainable.pth"}
HP = {"ldm": (4, 1024), "4class": (2, 1024), "1class": (4, 1024), "2class": (4, 128)}   # (nproj, proj_dim), src/utils.py::get_our_trained_model
MEAN, STD = (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)
_TF_CENTER = T.Compose([T.CenterCrop(224), T.ToTensor(), T.Normalize(MEAN, STD)])
_NORM = T.Normalize(MEAN, STD)
_BACKBONE = None
_MODELS = {}


def _tencrop(img):
    return torch.stack([T.PILToTensor()(c) for c in T.TenCrop(224)(img)]).float() / 255


def _backbone():
    global _BACKBONE
    if _BACKBONE is None:
        clipmod = import_by_path("openai_clip_model", THIRD_PARTY / "CLIP" / "clip" / "model.py")
        sd = torch.jit.load(str(CLIP_PT), map_location="cpu").eval().state_dict()
        net = clipmod.build_model(sd).float().eval()
        _BACKBONE = net
    return _BACKBONE


def _load(which):
    if which not in _MODELS:
        shim = types.ModuleType("clip")
        shim.load = lambda name, device="cpu": (_backbone(), None)
        sys.modules["clip"] = shim
        rm = import_by_path("rine_src_models", THIRD_PARTY / "rine" / "src" / "models.py")
        nproj, dim = HP[which]
        model = rm.Model(backbone=("ViT-L/14", 1024), nproj=nproj, proj_dim=dim, device="cpu")
        sd = torch.load(str(CKPT[which]), map_location="cpu", weights_only=True)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        assert not unexpected and all(k.startswith("clip.") for k in missing), (missing, unexpected)
        # DeviceModel moves the whole module (incl. the shared CLIP backbone) to the device
        _MODELS[which] = DeviceModel(model)
    return _MODELS[which]


def _fwd_single(net, x):            # x: (B,3,224,224); RINE's forward only supports the CLS token of the batch -> per-image outputs
    return net(x)[0][:, 0]


def _score(images, which, tencrop, batch_size=8):
    m = _load(which)
    out = []
    for sl in batched(len(images), batch_size):
        ims = [to_rgb(im) for im in images[sl]]
        if tencrop:
            x = torch.stack([_NORM(_tencrop(im)) for im in ims])           # (B,10,3,224,224)
            B = x.shape[0]
            y = m(lambda net, t: _fwd_single(net, t.view(-1, 3, 224, 224)), x).reshape(B, 10).mean(1)
        else:
            x = torch.stack([_TF_CENTER(im) for im in ims])
            y = m(_fwd_single, x)
        out.append(y)
    return np.concatenate(out).astype(np.float64) if out else np.zeros(0)


def score(images, batch_size=8):
    return _score(images, "ldm", False, batch_size)


def score_ldm_tencrop(images, batch_size=2):
    return _score(images, "ldm", True, batch_size)


def score_4class(images, batch_size=8):
    return _score(images, "4class", False, batch_size)


def score_4class_tencrop(images, batch_size=2):
    return _score(images, "4class", True, batch_size)
