"""ClipDet / "Raising the Bar of AI-generated Image Detection with CLIP" (Cozzolino et al., CVPRW 2024) -- official
GRIP-UNINA linear heads (Apache-2.0) on the open_clip ViT-L/14 CommonPool.XL backbone (laion, MIT).

score(images) -> np.ndarray of LLR LOGITS (higher = more fake; official threshold 0 = "synthetic"), model clipdet_latent10k_plus
(the repo's default: 10k pairs of Latent Diffusion fakes + resized/compressed augmented reference set).
score_10k -> clipdet_latent10k (same architecture, trained without the resized/compressed images).

Test-time preprocessing, reproduced from third_party/ClipBased-SyntheticImageDetection/main.py (patch_size 'Clip224', norm 'clip'):
    RGB -> Resize(224, BICUBIC) [shorter side -> 224, keep aspect, PIL antialiased] -> CenterCrop(224)
        -> ToTensor -> Normalize(CLIP mean/std)
NOTE: this RESIZES (512x512 -> 224x224, a 2.3x down-scale); that is how the released detector was evaluated.
Features: next-to-last layer (visual.proj=None, i.e. 1024-d pooled+ln_post CLS before the CLIP projection), L2-normalised,
then a single linear layer (ChannelLinear) -> 1 logit. Model code (networks/openclipnet.py) is imported by path, unmodified.
Deviation: the backbone is read from the local open_clip_pytorch_model.safetensors (same repo/file content as the .bin the code
would fetch) via two temporary monkey-patches (hf_hub_download -> local path; open_clip's checkpoint loader accepts .safetensors).
"""
import numpy as np
import torch
from safetensors.torch import load_file
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode

from ._common import THIRD_PARTY, WEIGHTS, DeviceModel, batched, import_by_path, to_rgb

CODE = THIRD_PARTY / "ClipBased-SyntheticImageDetection"
BB = WEIGHTS / "laion-commonpool-vitl14" / "open_clip_pytorch_model.safetensors"
HEADS = {"latent10k_plus": WEIGHTS / "clipdet" / "clipdet_latent10k_plus" / "weights.pth",
         "latent10k": WEIGHTS / "clipdet" / "clipdet_latent10k" / "weights.pth"}
MEAN, STD = (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)
_TF = T.Compose([T.Resize(224, interpolation=InterpolationMode.BICUBIC), T.CenterCrop((224, 224)),
                 T.ToTensor(), T.Normalize(MEAN, STD)])
_MODELS = {}


def _load(which):
    if which not in _MODELS:
        import huggingface_hub
        import open_clip.factory as ocf
        pkg = import_by_path("clipdet_networks", CODE / "networks" / "__init__.py", package=True)
        orig_dl, orig_ld = huggingface_hub.hf_hub_download, ocf.load_state_dict

        def ld(path, map_location="cpu"):
            return load_file(str(path)) if str(path).endswith(".safetensors") else orig_ld(path, map_location)
        huggingface_hub.hf_hub_download, ocf.load_state_dict = (lambda *a, **k: str(BB)), ld
        try:
            net = pkg.create_architecture("opencliplinearnext_clipL14commonpool")
        finally:
            huggingface_hub.hf_hub_download, ocf.load_state_dict = orig_dl, orig_ld
        net = pkg.load_weights(net, str(HEADS[which]))
        _MODELS[which] = DeviceModel(net)       # OpenClipLinear.to() also moves the (list-hidden) backbone
    return _MODELS[which]


def _score(images, which, batch_size=8):
    m = _load(which)
    out = []
    for sl in batched(len(images), batch_size):
        x = torch.stack([_TF(to_rgb(im)) for im in images[sl]])
        y = m(lambda net, t: net(t), x)                      # (B,1)
        out.append(y.reshape(len(x), -1)[:, 0])
    return np.concatenate(out).astype(np.float64) if out else np.zeros(0)


def score(images, batch_size=8):
    return _score(images, "latent10k_plus", batch_size)


def score_10k(images, batch_size=8):
    return _score(images, "latent10k", batch_size)
