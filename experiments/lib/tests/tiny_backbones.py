"""Randomly initialised tiny models (no downloads)."""
import torch

from features import CLIP, IMAGENET, Backbone


def tiny_clip(**kw):
    from transformers import CLIPVisionConfig, CLIPVisionModel
    torch.manual_seed(0)
    cfg = CLIPVisionConfig(hidden_size=32, intermediate_size=64, num_hidden_layers=3, num_attention_heads=2,
                           image_size=32, patch_size=16)
    return Backbone("tiny-clip", CLIPVisionModel(cfg), "hf", *CLIP, input_size=32, **kw)


def tiny_dinov3(**kw):
    from transformers import DINOv3ViTConfig, DINOv3ViTModel
    torch.manual_seed(0)
    cfg = DINOv3ViTConfig(hidden_size=32, num_attention_heads=2, num_hidden_layers=3, intermediate_size=64,
                          patch_size=16, num_register_tokens=2)
    return Backbone("tiny-dinov3", DINOv3ViTModel(cfg), "hf", *IMAGENET, input_size=32, **kw)


def tiny_cli_backbone():
    return tiny_dinov3(layers=(-1,), use_cls=True, use_mean=True)
