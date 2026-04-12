"""
main.py  —  Proper Few-Shot Segmentation (3-Phase Pipeline)
============================================================

PHASE 1 — Learning on BASE classes (15 classes, normal batches)
  • Fine-tune backbone last block (layer4)
  • Build memory prototypes via adaptive EMA
  • No episodic sampling — just like original APM repo
  • 10 epochs, Adam, StepLR

PHASE 2 — Adaptation to NOVEL classes (K-SHOT, NO weight updates)
  • Freeze EVERYTHING (backbone + memory)
  • For each novel class, take K support images
  • Extract features → masked avg-pool → write novel prototype
  • This is the "few-shot" moment: only K examples used

PHASE 3 — Test on NOVEL classes (query images)
  • Still frozen
  • Run query images through model
  • Compare pixels against [background, novel_class] prototypes
  • Compute mIoU for each novel class
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

import os

import Data_Loader
import Models
import APM
import Metrics

# ─────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────
VOC_ROOT       = "/data/VOCdevkit/VOC2012"   # ← CHANGE THIS
FOLD           = 0         # which 5 classes are novel (0,1,2,3)
K_SHOT         = 1         # 1-shot or 5-shot
BACKBONE_NAME  = "resnet50"
BATCH_SIZE     = 8
NUM_EPOCHS     = 10
LEARNING_RATE  = 0.001
IMG_SIZE       = 224

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}  |  Backbone: {BACKBONE_NAME}  |  {K_SHOT}-shot  |  Fold {FOLD}")

# ─────────────────────────────────────────────────────────────────
# Load data
# ─────────────────────────────────────────────────────────────────
train_loader, val_loader, NUM_BASE = Data_Loader.prepare_base_loaders(
    voc_root   = VOC_ROOT,
    fold       = FOLD,
    batch_size = BATCH_SIZE,
)

novel_dataset, novel_classes = Data_Loader.prepare_novel_dataset(
    voc_root = VOC_ROOT,
    fold     = FOLD,
)

# ─────────────────────────────────────────────────────────────────
# Build model
# ─────────────────────────────────────────────────────────────────
backbone, FEATURE_DIM = Models.load_backbone(BACKBONE_NAME)
model = APM.SegAPM(backbone, NUM_BASE, FEATURE_DIM).to(device)

criterion = nn.CrossEntropyLoss(ignore_index=255)
optimizer = optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LEARNING_RATE
)
scheduler = StepLR(optimizer, step_size=1, gamma=0.30)


# ─────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────
def compute_batch_loss(model, images, masks, class_labels, novel_cls_id=None):
    """
    Forward pass + per-sample binary CE loss.
    Works for both Phase 1 (novel_cls_id=None) and Phase 3 (novel_cls_id=int).
    """
    logits, feat = model(images, novel_cls_id)   # [B, S, h, w]

    logits_full = F.interpolate(
        logits, size=(IMG_SIZE, IMG_SIZE),
        mode="bilinear", align_corners=False
    )   # [B, S, H, W]

    B    = images.shape[0]
    loss = torch.tensor(0.0, device=device)
    preds = []

    for i in range(B):
        if novel_cls_id is None:
            # Phase 1: logits has num_base_slots channels
            cls_idx  = class_labels[i].item()
            fg_slot  = cls_idx + 1
            logits_i = torch.stack(
                [logits_full[i, 0], logits_full[i, fg_slot]], dim=0
            ).unsqueeze(0)
        else:
            # Phase 3: logits already has only 2 channels [bg, fg]
            logits_i = logits_full[i].unsqueeze(0)

        mask_i = masks[i].unsqueeze(0)
        loss  += criterion(logits_i, mask_i)
        preds.append(logits_i.argmax(dim=1).squeeze(0))

    return loss / B, preds, feat


# ─────────────────────────────────────────────────────────────────
# PHASE 1 — Train on base classes
# ─────────────────────────────────────────────────────────────────
def phase1_train():
    print("\n" + "="*60)
    print("  PHASE 1 — Learning on BASE classes")
    print("="*60)

    train_losses, val_mious = [], []
    best_val_miou = 0.0

    for epoch in range(NUM_EPOCHS):
        model.train()
        metrics   = Metrics.SegMetrics(num_classes=2)
        epoch_loss = 0.0

        for batch_idx, (images, masks, labels) in enumerate(train_loader):
            images = images.to(device)
            masks  = masks.to(device)

            optimizer.zero_grad()
            loss, preds, feat = compute_batch_loss(model, images, masks, labels)
            loss.backward()
            optimizer.step()

            # Memory update (no gradients)
            with torch.no_grad():
                model.memory_module.update_from_batch(
                    feat.detach(), masks, labels.tolist()
                )

            for i in range(images.shape[0]):
                metrics.update(preds[i].unsqueeze(0), masks[i].unsqueeze(0))

            epoch_loss += loss.item()

            if batch_idx % 30 == 0:
                print(f"  Epoch {epoch+1}/{NUM_EPOCHS} | "
                      f"Batch {batch_idx}/{len(train_loader)} | "
                      f"Loss {loss.item():.4f}")

        # Validation
        _, val_miou, val_acc = phase1_validate()
        _, train_miou, train_acc = metrics.compute()

        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)
        val_mious.append(val_miou)

        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS} | LR={optimizer.param_groups[0]['lr']:.5f}")
        print(f"  Train mIoU={train_miou*100:.2f}%  PixAcc={train_acc*100:.2f}%")
        print(f"  Val   mIoU={val_miou*100:.2f}%    PixAcc={val_acc*100:.2f}%")

        if val_miou > best_val_miou:
            best_val_miou = val_miou
            torch.save(model.state_dict(), "phase1_best.pth")
            print(f"  ★ Saved best model (val mIoU={best_val_miou*100:.2f}%)")

        scheduler.step()

    print(f"\n[Phase 1 done] Best val mIoU on BASE classes = {best_val_miou*100:.2f}%")
    return best_val_miou


def phase1_validate():
    model.eval()
    metrics = Metrics.SegMetrics(num_classes=2)
    total_loss = 0.0

    with torch.no_grad():
        for images, masks, labels in val_loader:
            images = images.to(device)
            masks  = masks.to(device)
            loss, preds, _ = compute_batch_loss(model, images, masks, labels)
            total_loss += loss.item()
            for i in range(images.shape[0]):
                metrics.update(preds[i].unsqueeze(0), masks[i].unsqueeze(0))

    return total_loss / len(val_loader), *metrics.compute()[1:]


# ─────────────────────────────────────────────────────────────────
# PHASE 2 — Adapt to novel classes using K support images
# ─────────────────────────────────────────────────────────────────
def phase2_adapt(novel_dataset, novel_classes, k_shot):
    print("\n" + "="*60)
    print(f"  PHASE 2 — {k_shot}-shot adaptation to NOVEL classes")
    print("="*60)
    print("Loading best Phase 1 weights...")
    model.load_state_dict(torch.load("phase1_best.pth", map_location=device))

    # Freeze EVERYTHING — no weight updates from here on
    model.freeze_everything()

    model.eval()
    support_data = {}   # cls_id → (support_images, support_masks)

    for cls_id in novel_classes:
        cls_name = Data_Loader.VOC_CLASS_NAMES[cls_id]
        print(f"\n  Adapting to: {cls_name} (class {cls_id}) | {k_shot} support image(s)")

        # Get K support images and all query images for this class
        support, queries = novel_dataset.get_support_and_queries(
            cls_id, k_shot=k_shot, seed=42
        )
        support_data[cls_id] = queries   # save queries for Phase 3

        # Extract features for each support image
        support_feats = []
        support_masks_list = []

        with torch.no_grad():
            for img, msk in support:
                img_t = img.unsqueeze(0).to(device)   # [1, 3, H, W]
                feat, _ = model(img_t)                 # just need backbone output

                # Get raw backbone feature (not logits)
                feat_raw = model.backbone(img_t)       # [1, D, h, w]
                support_feats.append(feat_raw)
                support_masks_list.append(msk.unsqueeze(0).to(device))

        # Build novel prototype from K support features
        # This is the FEW-SHOT ADAPTATION — only K examples used
        model.memory_module.build_novel_prototype(
            support_feats, support_masks_list, cls_id
        )

    print("\n[Phase 2 done] Novel prototypes built.")
    return support_data


# ─────────────────────────────────────────────────────────────────
# PHASE 3 — Test on novel class query images
# ─────────────────────────────────────────────────────────────────
def phase3_test(novel_classes, query_data):
    print("\n" + "="*60)
    print("  PHASE 3 — Testing on NOVEL class query images")
    print("="*60)

    model.eval()
    all_mious = []

    with torch.no_grad():
        for cls_id in novel_classes:
            cls_name = Data_Loader.VOC_CLASS_NAMES[cls_id]
            queries  = query_data[cls_id]
            metrics  = Metrics.SegMetrics(num_classes=2)

            for q_img, q_mask in queries:
                img_t  = q_img.unsqueeze(0).to(device)    # [1, 3, H, W]
                mask_t = q_mask.unsqueeze(0).to(device)   # [1, H, W]

                # Forward: binary prediction — bg vs this novel class
                logits, _ = model(img_t, novel_cls_id=cls_id)
                logits_full = F.interpolate(
                    logits, size=(IMG_SIZE, IMG_SIZE),
                    mode="bilinear", align_corners=False
                )

                pred = logits_full.argmax(dim=1)   # [1, H, W] values: 0 or 1
                metrics.update(pred, mask_t)

            _, cls_miou, cls_acc = metrics.compute()
            all_mious.append(cls_miou)
            print(f"  {cls_name:15s} (class {cls_id:2d}) | "
                  f"mIoU={cls_miou*100:.2f}%  PixAcc={cls_acc*100:.2f}%  "
                  f"({len(queries)} query images)")

    mean_novel_miou = sum(all_mious) / len(all_mious)
    print(f"\n[Phase 3 done] Mean mIoU over {len(novel_classes)} NOVEL classes = "
          f"{mean_novel_miou*100:.2f}%")
    print(f"  (This is the final FSS performance number)")

    Metrics.plot_iou_histogram(
        val_miou=0,
        test_miou=mean_novel_miou,
        save_path="novel_miou.png"
    )
    return mean_novel_miou


# ─────────────────────────────────────────────────────────────────
# RUN ALL 3 PHASES
# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # ── Phase 1 ──
    phase1_val_miou = phase1_train()

    # ── Phase 2 ──
    query_data = phase2_adapt(novel_dataset, novel_classes, K_SHOT)

    # ── Phase 3 ──
    novel_miou = phase3_test(novel_classes, query_data)

    print("\n" + "="*60)
    print("  FINAL RESULTS")
    print("="*60)
    print(f"  Phase 1 best val mIoU  (base classes) = {phase1_val_miou*100:.2f}%")
    print(f"  Phase 3 mean mIoU    (novel classes)  = {novel_miou*100:.2f}%")
    print(f"  Setting: Fold={FOLD} | {K_SHOT}-shot | {BACKBONE_NAME}")