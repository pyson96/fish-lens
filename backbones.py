"""
Backbone registry: every model is used frozen (eval mode, no gradients) as an image -> embedding function.

Add a backbone by adding one entry to BACKBONES (torch.hub DINOv2 or any timm model name).

    from backbones import load_backbone
    bb = load_backbone("siglip2_so400m", device)
    emb = bb.model(bb.transforms["crop"](img).unsqueeze(0).to(device))
"""

from dataclasses import dataclass, field

import torch
from PIL import Image
from torchvision import transforms as T

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

DEFAULT_BACKBONE = "dinov2_vits14"

BACKBONES = {
    "dinov2_vits14": dict(
        label="DINOv2 ViT-S/14",
        hub=("facebookresearch/dinov2", "dinov2_vits14"),
        source="Meta official weights via torch.hub (self-supervised, LVD-142M)",
        amp=False,
    ),
    "eva02_large": dict(
        label="EVA-02 Large/14 (CLIP)",
        timm="eva02_large_patch14_clip_224.merged2b",
        source="timm eva02_large_patch14_clip_224.merged2b (EVA-02-CLIP-L, Merged-2B image-text pairs)",
        amp=True,
    ),
    "siglip2_so400m": dict(
        label="SigLIP 2 So400m/14",
        timm="vit_so400m_patch14_siglip_224.v2_webli",
        source="timm vit_so400m_patch14_siglip_224.v2_webli (Google SigLIP 2, WebLI)",
        amp=True,
    ),
    "siglip2_giant": dict(
        label="SigLIP 2 Giant-opt/16",
        timm="vit_giantopt_patch16_siglip_256.v2_webli",
        source="timm vit_giantopt_patch16_siglip_256.v2_webli (Google SigLIP 2 g-opt, WebLI, 256 px)",
        amp=True, batch=16,
    ),
    "pe_core_g14": dict(
        label="PE-Core G/14 (448)",
        timm="vit_pe_core_gigantic_patch14_448.fb",
        source="timm vit_pe_core_gigantic_patch14_448.fb (Meta Perception Encoder Core G/14, 448 px)",
        amp=True, batch=8,
    ),
    # Optional, not part of the default comparison: ImageNet-22k/1k fine-tuned EVA-02 L at 448 px (4x tokens)
    "eva02_large_ft448": dict(
        label="EVA-02 Large/14 (IN-22k ft, 448)",
        timm="eva02_large_patch14_448.mim_m38m_ft_in22k_in1k",
        source="timm eva02_large_patch14_448.mim_m38m_ft_in22k_in1k",
        amp=True,
    ),
}


@dataclass
class Backbone:
    key: str
    label: str
    source: str
    model: torch.nn.Module
    dim: int
    transforms: dict = field(default_factory=dict)  # "crop" (model's eval transform) / "pad" (no crop)
    transform_desc: dict = field(default_factory=dict)
    amp: bool = False
    batch: int = 64


def pad_to_square(img: Image.Image, fill_mean) -> Image.Image:
    """Letterbox to a square with the model's mean color, so nothing is cropped away."""
    w, h = img.size
    side = max(w, h)
    canvas = Image.new("RGB", (side, side), tuple(int(m * 255) for m in fill_mean))
    canvas.paste(img, ((side - w) // 2, (side - h) // 2))
    return canvas


def _interp(name: str):
    return {"bicubic": T.InterpolationMode.BICUBIC, "bilinear": T.InterpolationMode.BILINEAR}.get(
        name, T.InterpolationMode.BICUBIC)


def _make_transforms(size, mean, std, interpolation, crop_pct):
    resize_to = int(round(size / crop_pct))
    crop = T.Compose([
        T.Resize(resize_to, interpolation=_interp(interpolation)),
        T.CenterCrop(size),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])
    pad = T.Compose([
        T.Lambda(lambda im: pad_to_square(im, mean)),
        T.Resize((size, size), interpolation=_interp(interpolation)),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])
    desc = {
        "crop": f"RGB -> Resize({resize_to}, {interpolation}) -> CenterCrop({size}) -> Normalize(mean={mean}, std={std})",
        "pad": f"RGB -> pad to square (mean color) -> Resize({size}x{size}, {interpolation}) -> Normalize",
    }
    return {"crop": crop, "pad": pad}, desc


def load_backbone(key: str, device: torch.device) -> Backbone:
    if key not in BACKBONES:
        raise ValueError(f"unknown backbone '{key}'. choices: {list(BACKBONES)}")
    spec = BACKBONES[key]
    if "hub" in spec:
        model = torch.hub.load(*spec["hub"], trust_repo=True)
        dim = model.embed_dim
        tfms, desc = _make_transforms(224, IMAGENET_MEAN, IMAGENET_STD, "bicubic", 224 / 256)
    else:
        import timm
        model = timm.create_model(spec["timm"], pretrained=True, num_classes=0)  # num_classes=0 -> pooled embedding
        dim = model.num_features
        cfg = timm.data.resolve_model_data_config(model)
        size = cfg["input_size"][-1]
        tfms, desc = _make_transforms(size, tuple(cfg["mean"]), tuple(cfg["std"]), cfg["interpolation"],
                                      cfg.get("crop_pct", 1.0))
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    model.to(device)
    return Backbone(key=key, label=spec["label"], source=spec["source"], model=model, dim=dim,
                    transforms=tfms, transform_desc=desc, amp=spec["amp"] and device.type == "cuda",
                    batch=spec.get("batch", 64))


def output_root(root, key: str):
    """Default backbone keeps the original output/ layout; others go under output/backbones/<key>/."""
    return root / "output" if key == DEFAULT_BACKBONE else root / "output" / "backbones" / key
