"""
APM.py  —  Adaptive Prototype Memory for Proper Few-Shot Segmentation
======================================================================
Three-phase design:

  Phase 1 — LEARNING (base classes)
    Backbone last block fine-tunes on 15 base classes.
    Memory builds prototypes for these 15 classes.
    Exactly like original APM training, just with spatial features.

  Phase 2 — ADAPTATION (novel classes, K-shot, NO weight updates)
    Backbone and memory are COMPLETELY FROZEN.
    We show K support images of a novel class.
    We extract their features, do masked avg-pool → new prototype.
    Write it into a temporary memory slot.
    Zero gradient updates. Pure feature extraction.

  Phase 3 — TEST (novel class query images)
    Everything still frozen.
    Query image → backbone → spatial features.
    Compare every pixel against the novel class prototype.
    Upsample → segmentation mask.
    Compute mIoU.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MemoryModule(nn.Module):
    """
    Stores class prototypes as a matrix.
    Phase 1: slots filled for base classes (with gradient-free EMA updates).
    Phase 2: extra slot added for each novel class (pure feature extraction).
    """

    def __init__(self, num_base_classes, feature_dim):
        super().__init__()
        self.num_base_classes = num_base_classes
        self.feature_dim      = feature_dim

        # Slots: 0=background, 1..num_base_classes = one per base class
        # During Phase 2 we add novel class slots dynamically (not in nn.Parameter
        # — we store them as plain tensors so they never interact with the optimizer)
        self.num_base_slots = num_base_classes + 1   # +1 for background

        self.memory = nn.Parameter(
            torch.randn(self.num_base_slots, feature_dim),
            requires_grad=False
        )
        nn.init.normal_(self.memory, mean=0.0, std=0.01)

        self.slot_ready = [False] * self.num_base_slots

        # Novel class prototypes stored separately (plain dict, not nn.Parameter)
        # key = novel_class_id, value = prototype tensor [feature_dim]
        self.novel_prototypes = {}

        print(f"[APM] Base memory: {self.num_base_slots} slots × {feature_dim} dims")

    # ─────────────────────────────────────────────────────────────
    # Forward (used in all 3 phases)
    # ─────────────────────────────────────────────────────────────
    def forward(self, feature_map, novel_cls_id=None):
        """
        Compute cosine similarity between every spatial location and
        either base class prototypes (Phase 1) or a specific novel class
        prototype (Phase 2 & 3).

        Parameters
        ----------
        feature_map : [B, D, h, w]
        novel_cls_id: int or None
            None   → use full base memory (Phase 1 training)
            int    → use only [background, novel_class] (Phase 2 & 3)

        Returns
        -------
        logits : [B, num_slots, h, w]
        """
        B, D, h, w = feature_map.shape
        feat_norm  = F.normalize(feature_map, p=2, dim=1)   # [B, D, h, w]

        if novel_cls_id is None:
            # Phase 1: compare against all base class slots
            mem = F.normalize(self.memory, p=2, dim=1)      # [S, D]
        else:
            # Phase 2 & 3: compare against [background, novel_class] only
            bg_proto    = F.normalize(self.memory[0], p=2, dim=0)  # [D]
            novel_proto = self.novel_prototypes[novel_cls_id]       # [D]
            mem = torch.stack([bg_proto, novel_proto], dim=0)       # [2, D]

        # Spatial dot-product: every pixel vs every prototype
        S = mem.shape[0]
        feat_flat = feat_norm.view(B, D, h * w)             # [B, D, h*w]
        mem_T     = mem.t()                                  # [D, S]
        sim       = torch.einsum('bdi,ds->bsi',
                                 feat_flat,
                                 mem_T.unsqueeze(0).expand(B, -1, -1))
        # einsum gives [B, S, h*w] → reshape to [B, S, h, w]
        logits = sim.view(B, S, h, w)
        return logits

    # ─────────────────────────────────────────────────────────────
    # Phase 1 update — called after each training batch
    # ─────────────────────────────────────────────────────────────
    def update_from_batch(self, feature_map, binary_masks, class_labels):
        """
        Update base class prototypes using masked average pooling + EMA.
        Identical logic to original APM update_memory().
        """
        B = feature_map.shape[0]
        for i in range(B):
            feat_i = feature_map[i].unsqueeze(0)   # [1, D, h, w]
            mask_i = binary_masks[i].unsqueeze(0)  # [1, H, W]
            cls    = class_labels[i]
            fg_slot = cls + 1                       # slot 0 = background

            fg_mask = (mask_i == 1).long()
            bg_mask = (mask_i == 0).long()

            self._update_slot(feat_i, fg_mask, fg_slot)
            self._update_slot(feat_i, bg_mask, 0)

    def _update_slot(self, feature_map, mask, slot_idx):
        """Masked average pooling → adaptive EMA update of one slot."""
        D, h, w = feature_map.shape[1:]

        mask_down = F.interpolate(
            mask.float().unsqueeze(1), size=(h, w), mode="nearest"
        )
        valid     = (mask_down != 255).float()
        mask_down = mask_down * valid

        denom     = mask_down.sum(dim=[0, 2, 3]).clamp(min=1e-6)
        proto_new = (feature_map * mask_down).sum(dim=[0, 2, 3]) / denom
        proto_new = F.normalize(proto_new, p=2, dim=0)

        if not self.slot_ready[slot_idx]:
            self.memory.data[slot_idx] = proto_new
            self.slot_ready[slot_idx]  = True
        else:
            proto_old  = F.normalize(self.memory.data[slot_idx], p=2, dim=0)
            sim        = F.cosine_similarity(
                proto_new.unsqueeze(0), proto_old.unsqueeze(0)
            ).item()
            alpha      = max(0.0, min(1.0 - sim, 1.0))
            self.memory.data[slot_idx] = (
                (1 - alpha) * self.memory.data[slot_idx] + alpha * proto_new
            )

    # ─────────────────────────────────────────────────────────────
    # Phase 2 — build novel class prototype from K support images
    # ─────────────────────────────────────────────────────────────
    @torch.no_grad()
    def build_novel_prototype(self, support_features, support_masks, novel_cls_id):
        """
        This is the FEW-SHOT ADAPTATION step.
        We see K support images and build ONE prototype for the novel class.
        No weight updates anywhere. Pure feature extraction.

        Parameters
        ----------
        support_features : list of [1, D, h, w] tensors  (length = K)
        support_masks    : list of [1, H, W]  tensors    (length = K)
        novel_cls_id     : int  — used as dict key
        """
        accumulated = None
        count       = 0

        for feat_i, mask_i in zip(support_features, support_masks):
            D, h, w = feat_i.shape[1:]
            mask_down = F.interpolate(
                mask_i.float().unsqueeze(1), size=(h, w), mode="nearest"
            )
            valid     = (mask_down != 255).float()
            mask_down = mask_down * valid

            denom = mask_down.sum(dim=[0, 2, 3]).clamp(min=1e-6)
            proto = (feat_i * mask_down).sum(dim=[0, 2, 3]) / denom  # [D]

            if accumulated is None:
                accumulated = proto
            else:
                # Average across all K support images
                accumulated = accumulated + proto
            count += 1

        # Final prototype = average of K support prototypes, then normalise
        novel_proto = F.normalize(accumulated / count, p=2, dim=0)
        self.novel_prototypes[novel_cls_id] = novel_proto

        print(f"[APM] Novel prototype built for class {novel_cls_id} "
              f"from {count} support image(s).")


# ─────────────────────────────────────────────────────────────────
# Full Model
# ─────────────────────────────────────────────────────────────────
class SegAPM(nn.Module):
    def __init__(self, backbone, num_base_classes, feature_dim):
        super().__init__()
        self.backbone      = backbone
        self.memory_module = MemoryModule(num_base_classes, feature_dim)

    def forward(self, x, novel_cls_id=None):
        """
        novel_cls_id = None  → Phase 1 (all base class slots)
        novel_cls_id = int   → Phase 2 & 3 (binary: bg vs novel class)
        """
        feat   = self.backbone(x)
        logits = self.memory_module(feat, novel_cls_id)
        return logits, feat

    def freeze_everything(self):
        """Call before Phase 2 & 3. Freezes backbone AND memory."""
        for param in self.parameters():
            param.requires_grad = False
        print("[APM] All weights frozen for Phase 2 & 3.")