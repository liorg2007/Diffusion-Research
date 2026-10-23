"""Frozen-backbone feature extraction with an on-disk cache keyed by sha256.

Cache file: <exp_dir>/cache/<backbone.name>_<condition>.npz  (keys, feats, config).
Images are CENTER-CROPPED (never resized) to backbone.input_size unless crop_mode='resize'.
"""
import json
import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")  # must precede torch import

import numpy as np
import torch
from PIL import Image

from manifest import sha256_file
from perturb import apply_condition
from preprocess import center_crop

IMAGENET = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
CLIP = ((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))


def get_device(prefer=None):
    if prefer:
        return torch.device(prefer)
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


class Backbone:
    """Wraps a model into `forward(x[B,3,H,W]) -> [B, D]`.

    kind: 'hf' (transformers vision model; ViT-style tokens), 'timm', 'tv' (torchvision, head removed).
    For 'hf': `layers` indexes output hidden_states (0 = embeddings, -1 = last block, pre final-LN);
    for each layer we concatenate CLS (use_cls) and/or mean of patch tokens (use_mean; register tokens skipped).
    """

    def __init__(self, name, model, kind="hf", mean=IMAGENET[0], std=IMAGENET[1], input_size=224,
                 crop_mode="crop", layers=(-1,), use_cls=True, use_mean=False):
        assert kind in ("hf", "timm", "tv") and crop_mode in ("crop", "resize")
        assert use_cls or use_mean
        if kind != "hf" and tuple(layers) != (-1,):
            raise ValueError("intermediate layers are only supported for kind='hf'")
        self.name, self.model, self.kind = name, model.eval(), kind
        self.mean, self.std, self.input_size, self.crop_mode = mean, std, input_size, crop_mode
        self.layers, self.use_cls, self.use_mean = tuple(layers), use_cls, use_mean
        self.n_forward = 0  # number of images embedded (used by cache tests)

    def config(self):
        return dict(name=self.name, kind=self.kind, input_size=self.input_size, crop_mode=self.crop_mode,
                    layers=list(self.layers), use_cls=self.use_cls, use_mean=self.use_mean)

    def to_tensor(self, img):
        img = img.convert("RGB")
        n = self.input_size
        if n:
            if self.crop_mode == "resize":
                img = img.resize((n, n), Image.BICUBIC)
            else:
                if min(img.size) < n:
                    raise ValueError(f"image {img.size} smaller than input_size {n}")
                img = center_crop(img, n)
        x = torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0).permute(2, 0, 1)
        m, s = torch.tensor(self.mean)[:, None, None], torch.tensor(self.std)[:, None, None]
        return (x - m) / s

    @torch.no_grad()
    def forward(self, x):
        if self.kind == "hf":
            hs = self.model(pixel_values=x, output_hidden_states=True).hidden_states
            n_prefix = 1 + int(getattr(self.model.config, "num_register_tokens", 0) or 0)
            parts = []
            for l in self.layers:
                h = hs[l]
                if self.use_cls:
                    parts.append(h[:, 0])
                if self.use_mean:
                    parts.append(h[:, n_prefix:].mean(1))
            return torch.cat(parts, 1)
        if self.kind == "timm":
            f = self.model.forward_features(x)
            if f.ndim == 4:
                return f.mean((2, 3))
            parts = ([f[:, 0]] if self.use_cls else []) + ([f[:, 1:].mean(1)] if self.use_mean else [])
            return torch.cat(parts, 1)
        return self.model(x)


def load_backbone(spec, input_size=224, layers=(-1,), use_cls=True, use_mean=False, crop_mode="crop"):
    """spec: 'hf:clip:<repo-or-path>' | 'hf:dinov2:<..>' | 'hf:dinov3:<..>' | 'timm:<name>' | 'tv:<name>'.
    NOTE: this downloads weights (if not cached). Never call it from tests."""
    kind, _, rest = spec.partition(":")
    kw = dict(input_size=input_size, crop_mode=crop_mode)
    if kind == "hf":
        fam, _, ident = rest.partition(":")
        from transformers import AutoModel, CLIPVisionModel
        if fam == "clip":
            model, norm = CLIPVisionModel.from_pretrained(ident), CLIP
        elif fam == "dinov3":
            from transformers import DINOv3ViTModel
            model, norm = DINOv3ViTModel.from_pretrained(ident), IMAGENET
        elif fam == "dinov2":
            model, norm = AutoModel.from_pretrained(ident), IMAGENET
        else:
            raise ValueError(f"unknown hf family {fam!r}")
        return Backbone(spec.replace("/", "_").replace(":", "-"), model, "hf", *norm,
                        layers=layers, use_cls=use_cls, use_mean=use_mean, **kw)
    if kind == "timm":
        import timm
        m = timm.create_model(rest, pretrained=True)
        cfg = m.pretrained_cfg
        return Backbone(f"timm-{rest}", m, "timm", cfg.get("mean", IMAGENET[0]), cfg.get("std", IMAGENET[1]),
                        use_cls=use_cls, use_mean=use_mean, **kw)
    if kind == "tv":
        import torchvision
        m = getattr(torchvision.models, rest)(weights="DEFAULT")
        for head in ("fc", "classifier", "head"):
            if hasattr(m, head):
                setattr(m, head, torch.nn.Identity())
                break
        return Backbone(f"tv-{rest}", m, "tv", *IMAGENET, use_cls=use_cls, use_mean=use_mean, **kw)
    raise ValueError(f"unknown backbone spec {spec!r}")


def embed_images(backbone, images, batch_size=16, device=None):
    """PIL images -> float32 array [N, D]. Falls back to CPU if the accelerator raises.
    The fallback is sticky for the backbone's lifetime (D-010): bouncing a model between MPS and CPU after a
    failed MPS forward produced silently wrong DINOv2 features in 003-baselines-eval."""
    dev = torch.device("cpu") if getattr(backbone, "force_cpu", False) else get_device(device)
    out = []
    for i in range(0, len(images), batch_size):
        x = torch.stack([backbone.to_tensor(im) for im in images[i:i + batch_size]])
        try:
            backbone.model.to(dev)
            f = backbone.forward(x.to(dev))
        except (RuntimeError, NotImplementedError) as e:
            if dev.type == "cpu":
                raise
            print(f"[features] {dev} failed ({type(e).__name__}); falling back to cpu")
            dev = torch.device("cpu")
            backbone.force_cpu = True
            backbone.model.to(dev)
            f = backbone.forward(x.to(dev))
        out.append(f.float().cpu().numpy())
        backbone.n_forward += len(x)
    return np.concatenate(out) if out else np.zeros((0, 0), np.float32)


def cache_path(exp_dir, backbone, condition):
    safe = lambda s: "".join(c if c.isalnum() or c in "-_.+()" else "_" for c in s)
    return os.path.join(exp_dir, "cache", f"{safe(backbone.name)}_{safe(condition)}.npz")


def extract_features(backbone, paths, condition="clean", exp_dir=".", batch_size=16, keys=None, seed=0,
                     device=None):
    """Embeddings for `paths` under `condition`, cached by key (default sha256 of file bytes).
    Only keys missing from the cache are computed; a config mismatch invalidates the cache.
    Returns float32 [len(paths), D] in input order."""
    paths = [str(p) for p in paths]
    keys = list(keys) if keys is not None else [sha256_file(p) for p in paths]
    cfg = json.dumps(backbone.config(), sort_keys=True)
    cp = cache_path(exp_dir, backbone, condition)
    cache = {}
    if os.path.exists(cp):
        z = np.load(cp, allow_pickle=False)
        if str(z["config"]) == cfg:
            cache = dict(zip(z["keys"].tolist(), z["feats"]))
        else:
            print(f"[features] cache config mismatch for {cp}; recomputing")
    todo = [i for i, k in enumerate(keys) if k not in cache]
    done = set()
    for s in range(0, len(todo), 256):  # chunk to bound memory
        chunk = [i for i in todo[s:s + 256] if keys[i] not in done]
        imgs = [apply_condition(Image.open(paths[i]), condition, key=keys[i], seed=seed) for i in chunk]
        feats = embed_images(backbone, imgs, batch_size, device)
        for i, f in zip(chunk, feats):
            cache[keys[i]] = f
            done.add(keys[i])
    if todo:
        os.makedirs(os.path.dirname(cp), exist_ok=True)
        ks = sorted(cache)
        tmp = cp + ".tmp.npz"
        np.savez(tmp, keys=np.array(ks), feats=np.stack([cache[k] for k in ks]), config=np.array(cfg))
        os.replace(tmp, cp)
    return np.stack([cache[k] for k in keys]).astype(np.float32)
