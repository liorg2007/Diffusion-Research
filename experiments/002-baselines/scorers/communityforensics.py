"""Community Forensics (Park & Owens, CVPR 2025) ViT-S/16 @384 -- official OwensLab/commfor-model-384 weights (MIT).

score(images) -> np.ndarray of LOGITS (higher = more fake); sigmoid(logit) is the official probability.

Test-time preprocessing, reproduced from third_party/Community-Forensics/dataloader.py::get_transform(mode="test")
(the HF `commfor-data-preprocessor` runs exactly this code):
    RGB -> Resize(440) [shorter side -> 440, keep aspect, bilinear, PIL/antialias]
        -> CenterCrop(384) -> ToTensor -> Normalize(ImageNet mean/std)
NOTE: this RESIZES (a 512x512 input is down-scaled to 440, then center-cropped to 384).
Architecture: timm vit_small_patch16_384 + Linear(384,1) head (models.ViTClassifier); built here with pretrained=False
(the repo's constructor would fetch ImageNet init weights that the checkpoint overwrites anyway).
"""
import numpy as np
import torch
from safetensors.torch import load_file
from torchvision import transforms as T

from ._common import WEIGHTS, DeviceModel, batched, to_rgb

CKPT = WEIGHTS / "community-forensics" / "commfor-model-384" / "model.safetensors"
RESIZE, CROP = 440, 384
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
_TF = T.Compose([T.Resize(RESIZE), T.CenterCrop(CROP), T.ToTensor(), T.Normalize(MEAN, STD)])
_M = None


def _load():
    global _M
    if _M is None:
        import timm
        vit = timm.create_model("vit_small_patch16_384.augreg_in21k_ft_in1k", pretrained=False)
        vit.head = torch.nn.Linear(384, 1)
        sd = load_file(str(CKPT))
        assert all(k.startswith("vit.") for k in sd), "unexpected checkpoint keys"
        vit.load_state_dict({k[4:]: v for k, v in sd.items()}, strict=True)
        _M = DeviceModel(vit)
    return _M


def score(images, batch_size=8):
    m = _load()
    out = []
    for sl in batched(len(images), batch_size):
        x = torch.stack([_TF(to_rgb(im)) for im in images[sl]])
        out.append(m(lambda net, t: net(t)[:, 0], x))
    return np.concatenate(out).astype(np.float64) if out else np.zeros(0)
