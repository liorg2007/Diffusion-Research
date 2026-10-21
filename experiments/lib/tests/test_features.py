import _path  # noqa
import numpy as np
import torch
from PIL import Image

import synth
import tiny_backbones as tb
from features import Backbone, IMAGENET, cache_path, embed_images, extract_features, get_device


def _paths(tmp_path, n=6, size=48):
    rng = np.random.default_rng(0)
    ps = []
    for i in range(n):
        p = tmp_path / f"im{i}.png"
        Image.fromarray(synth.gradient(size, rng, 10 + 5 * i)).save(p)
        ps.append(p)
    return ps


def test_cache_roundtrip_clip(tmp_path):
    ps = _paths(tmp_path, size=80)  # jpeg75_resize05 halves it; must stay >= input_size 32
    bb = tb.tiny_clip(layers=(-1, 1), use_cls=True, use_mean=True)
    f1 = extract_features(bb, ps[:4], "clean", tmp_path, batch_size=2, device="cpu")
    assert f1.shape == (4, 4 * 32) and bb.n_forward == 4
    cp = cache_path(tmp_path, bb, "clean")
    assert cp.startswith(str(tmp_path / "cache")) and cp.endswith("tiny-clip_clean.npz")
    z = np.load(cp)
    assert len(z["keys"]) == 4 and z["feats"].shape == (4, 128)
    f2 = extract_features(bb, ps[:4], "clean", tmp_path, device="cpu")
    assert bb.n_forward == 4 and np.array_equal(f1, f2)  # fully served from cache
    f3 = extract_features(bb, ps[2:6][::-1], "clean", tmp_path, device="cpu")  # reorder + 2 new
    assert bb.n_forward == 6 and np.allclose(f3[::-1][:2], f1[2:4])
    # different condition -> separate file, different features
    fj = extract_features(bb, ps[:4], "jpeg75_resize05", tmp_path, device="cpu")
    assert cache_path(tmp_path, bb, "jpeg75_resize05") != cp and not np.allclose(fj, f1)
    # config change invalidates cache
    bb2 = tb.tiny_clip(layers=(-1, 1), use_cls=True, use_mean=False)
    f4 = extract_features(bb2, ps[:2], "clean", tmp_path, device="cpu")
    assert bb2.n_forward == 2 and f4.shape == (2, 64)


def test_dinov3_registers_and_layers(tmp_path):
    ps = _paths(tmp_path, 3)
    bb = tb.tiny_dinov3(layers=(-1,), use_cls=True, use_mean=True)
    f = extract_features(bb, ps, "clean", tmp_path, device="cpu")
    assert f.shape == (3, 64) and np.isfinite(f).all()
    # mean part must exclude cls + 2 register tokens
    x = torch.stack([bb.to_tensor(Image.open(p)) for p in ps])
    hs = bb.model(pixel_values=x, output_hidden_states=True).hidden_states[-1]
    assert hs.shape[1] == 1 + 2 + 4
    assert np.allclose(f[:, 32:], hs[:, 3:].mean(1).detach().numpy(), atol=1e-5)
    assert np.allclose(f[:, :32], hs[:, 0].detach().numpy(), atol=1e-5)


def test_tv_and_timm_adapters(tmp_path):
    import timm, torchvision
    ps = _paths(tmp_path, 2, 48)
    torch.manual_seed(0)
    m = torchvision.models.resnet18(weights=None); m.fc = torch.nn.Identity()
    f = extract_features(Backbone("tv-r18", m, "tv", input_size=32), ps, "clean", tmp_path, device="cpu")
    assert f.shape == (2, 512)
    vit = timm.create_model("vit_tiny_patch16_224", pretrained=False, img_size=32, num_classes=0)
    f = extract_features(Backbone("timm-vit", vit, "timm", input_size=32, use_mean=True), ps, "clean",
                         tmp_path, device="cpu")
    assert f.shape == (2, 2 * 192)
    cnn = timm.create_model("resnet18", pretrained=False, num_classes=0)
    f = extract_features(Backbone("timm-r18", cnn, "timm", input_size=32), ps, "clean", tmp_path, device="cpu")
    assert f.shape == (2, 512)


def test_crop_not_resize_and_small_image_error(tmp_path):
    bb = tb.tiny_clip()
    arr = np.arange(48 * 48 * 3, dtype=np.uint8).reshape(48, 48, 3)
    t = bb.to_tensor(Image.fromarray(arr))
    assert t.shape == (3, 32, 32)
    m, s = torch.tensor(bb.mean)[:, None, None], torch.tensor(bb.std)[:, None, None]
    ref = (torch.from_numpy(arr[8:40, 8:40].astype(np.float32) / 255).permute(2, 0, 1) - m) / s
    assert torch.allclose(t, ref, atol=1e-6)
    try:
        bb.to_tensor(Image.fromarray(arr[:20, :20])); raise RuntimeError("should fail")
    except ValueError:
        pass


def test_default_device_smoke(tmp_path):
    ps = _paths(tmp_path, 2)
    bb = tb.tiny_dinov3()
    imgs = [Image.open(p) for p in ps]
    a = embed_images(bb, imgs)  # mps if available (cpu fallback on error)
    b = embed_images(bb, imgs, device="cpu")
    assert a.shape == b.shape and np.allclose(a, b, atol=1e-2), get_device()
