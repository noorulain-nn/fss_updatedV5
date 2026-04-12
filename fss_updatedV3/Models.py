"""
Models.py  —  Backbone loader for Few-Shot Segmentation
========================================================
KEY DIFFERENCE from the original APM repo:
  Original: backbone ends with  AdaptiveAvgPool2d(1,1) + Flatten
            → output shape [B, C]   (one vector per image)

  This file: backbone ends BEFORE the avgpool
            → output shape [B, C, h, w]  (spatial feature map)

  We need spatial maps so we can compare EACH PIXEL's feature vector
  against the class prototypes stored in the memory module.
  This is what makes segmentation (per-pixel prediction) possible.

Freezing strategy (same as original):
  All layers frozen EXCEPT the last major block.
  For ResNet-34 that is layer4.
  This is "fine-tuning the last block" — identical philosophy to the
  original APM repo, just without the pooling step at the end.
"""

import torch
import torch.nn as nn
from torchvision import models


def load_backbone(backbone_name: str):
    """
    Load a pretrained backbone and return it as a SPATIAL feature extractor.

    Returns
    -------
    backbone : nn.Sequential  — call backbone(x) where x=[B,3,H,W]
                                returns [B, C, H/32, W/32]
    feature_dim : int         — number of channels C in the output map

    Supported backbones:  resnet18, resnet34, resnet50, resnet101
    (Same names as original repo — just no pooling at the end)
    """

    name = backbone_name.lower().strip()

    def _load_pretrained(ctor):
        try:
            return ctor(weights="IMAGENET1K_V1")
        except Exception:
            return ctor(pretrained=True)

    # ── ResNet family ──────────────────────────────────────────────────────
    if name in {"resnet18", "resnet34", "resnet50", "resnet101"}:
        ctor_map = {
            "resnet18":  models.resnet18,
            "resnet34":  models.resnet34,
            "resnet50":  models.resnet50,
            "resnet101": models.resnet101,
        }
        m = _load_pretrained(ctor_map[name])

        # Freeze everything except layer4
        for param_name, param in m.named_parameters():
            if not param_name.startswith("layer4."):
                param.requires_grad = False
            # layer4 parameters keep requires_grad=True (default)

        feature_dim = m.fc.in_features   # e.g. 512 for resnet34, 2048 for resnet50

        # Build the backbone WITHOUT avgpool and fc.
        # The output will be [B, feature_dim, H/32, W/32].
        # For a 224×224 input that means [B, 512, 7, 7] with ResNet-34.
        backbone = nn.Sequential(
            m.conv1,    # stride 2 → H/2
            m.bn1,
            m.relu,
            m.maxpool,  # stride 2 → H/4
            m.layer1,   # stride 1 → H/4
            m.layer2,   # stride 2 → H/8
            m.layer3,   # stride 2 → H/16
            m.layer4,   # stride 2 → H/32
            # ← STOP HERE.  No avgpool. No flatten.
        )

        # Count how many parameters are actually trainable (sanity check)
        trainable = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
        total     = sum(p.numel() for p in backbone.parameters())
        print(f"[Models] {backbone_name}: "
              f"trainable params = {trainable:,} / {total:,} "
              f"({100*trainable/total:.1f}%)")
        print(f"[Models] Feature map channels = {feature_dim}")
        print(f"[Models] For 224×224 input → spatial map = "
              f"{feature_dim} × {224//32} × {224//32}  "
              f"(i.e. {feature_dim} × 7 × 7)")

        return backbone, feature_dim

    raise ValueError(
        f"Unsupported backbone: '{backbone_name}'. "
        f"Choose one of: resnet18, resnet34, resnet50, resnet101"
    )