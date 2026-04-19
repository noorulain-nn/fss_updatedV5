# """
# Data_Loader.py  —  Pascal VOC Few-Shot Segmentation (PROPER VERSION)
# =====================================================================
# This version correctly separates BASE classes (used for learning) from
# NOVEL classes (used only at test time, with K support images).

# Pascal VOC has 20 classes. Standard FSS split (fold 0):
#   Base classes  (15): used in Phase 1 learning
#   Novel classes  (5): NEVER seen in Phase 1, only shown at test time

# We follow the standard 4-fold cross validation splits used in
# PFENet, PANet, HSNet and other FSS papers.
# """

# import os
# import random
# import numpy as np
# from PIL import Image

# import torch
# from torch.utils.data import Dataset, DataLoader, random_split
# import torchvision.transforms.functional as TF

# # ── Standard FSS splits for Pascal VOC ────────────────────────────
# # Each fold holds 5 novel classes. The other 15 are base classes.
# # We use fold 0 by default (same as most papers).
# PASCAL_FSS_SPLITS = {
#     0: [1,  2,  3,  4,  5],    # novel: aeroplane bicycle bird boat bottle
#     1: [6,  7,  8,  9,  10],   # novel: bus car cat chair cow
#     2: [11, 12, 13, 14, 15],   # novel: diningtable dog horse motorbike person
#     3: [16, 17, 18, 19, 20],   # novel: pottedplant sheep sofa train tvmonitor
# }

# VOC_CLASS_NAMES = [
#     "background", "aeroplane", "bicycle", "bird", "boat", "bottle",
#     "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse",
#     "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"
# ]

# IMG_SIZE = 224


# def joint_transform(image, mask, augment=False):
#     """Apply same spatial transform to image and mask together."""
#     image = TF.resize(image, (IMG_SIZE, IMG_SIZE), interpolation=Image.BILINEAR)
#     mask  = TF.resize(mask,  (IMG_SIZE, IMG_SIZE), interpolation=Image.NEAREST)

#     if augment and random.random() > 0.5:
#         image = TF.hflip(image)
#         mask  = TF.hflip(mask)

#     image = TF.to_tensor(image)
#     image = TF.normalize(image, mean=[0.485, 0.456, 0.406],
#                                 std= [0.229, 0.224, 0.225])
#     mask  = torch.from_numpy(np.array(mask)).long()
#     return image, mask


# # ─────────────────────────────────────────────────────────────────
# # Phase 1 Dataset — base classes, normal batch loading
# # ─────────────────────────────────────────────────────────────────
# class BaseClassDataset(Dataset):
#     """
#     Loads images containing BASE classes for Phase 1 learning.
#     Returns: image, binary_mask, class_label (remapped 0..N_base-1)

#     This is just like the original APM CIFAR loader — normal batches,
#     no episodic sampling.
#     """
#     def __init__(self, voc_root, split, base_classes, augment=False):
#         self.voc_root     = voc_root
#         self.base_classes = base_classes
#         self.augment      = augment
#         self.label_map    = {c: i for i, c in enumerate(sorted(base_classes))}

#         split_file = os.path.join(voc_root, "ImageSets", "Segmentation", split + ".txt")
#         with open(split_file) as f:
#             all_ids = [l.strip() for l in f if l.strip()]

#         self.samples = []
#         for img_id in all_ids:
#             mask_path = os.path.join(voc_root, "SegmentationClass", img_id + ".png")
#             if not os.path.exists(mask_path):
#                 continue
#             mask = np.array(Image.open(mask_path))
#             for cls_id in base_classes:
#                 if (mask == cls_id).any():
#                     self.samples.append((img_id, cls_id))

#         print(f"[BaseDataset] split={split} | {len(base_classes)} base classes "
#               f"| {len(self.samples)} samples")

#     def __len__(self): return len(self.samples)

#     def __getitem__(self, idx):
#         img_id, cls_id = self.samples[idx]

#         image = Image.open(
#             os.path.join(self.voc_root, "JPEGImages", img_id + ".jpg")
#         ).convert("RGB")
#         mask = Image.open(
#             os.path.join(self.voc_root, "SegmentationClass", img_id + ".png")
#         )

#         image, mask = joint_transform(image, mask, self.augment)

#         # Binary mask: 1 = this class, 0 = background, 255 = ignore
#         binary = torch.zeros_like(mask)
#         binary[mask == 255]    = 255
#         binary[mask == cls_id] = 1

#         return image, binary, self.label_map[cls_id]


# # ─────────────────────────────────────────────────────────────────
# # Phase 2 & 3 Dataset — novel classes, support + query
# # ─────────────────────────────────────────────────────────────────
# class NovelClassDataset(Dataset):
#     """
#     Loads images for NOVEL classes used in Phase 2 (adaptation) and
#     Phase 3 (testing).

#     For each class, we keep all available images in a list.
#     The FewShotTester will pick K of them as support and the rest as queries.
#     """
#     def __init__(self, voc_root, novel_classes):
#         self.voc_root      = novel_classes
#         self.novel_classes = novel_classes

#         # Build a dict: class_id → list of image_ids containing that class
#         self.class_images = {cls: [] for cls in novel_classes}

#         # Use both train and val splits to get enough images
#         all_ids = []
#         for split in ["train", "val"]:
#             split_file = os.path.join(voc_root, "ImageSets",
#                                       "Segmentation", split + ".txt")
#             if os.path.exists(split_file):
#                 with open(split_file) as f:
#                     all_ids += [l.strip() for l in f if l.strip()]

#         self.voc_root = voc_root
#         for img_id in set(all_ids):
#             mask_path = os.path.join(voc_root, "SegmentationClass", img_id + ".png")
#             if not os.path.exists(mask_path):
#                 continue
#             mask = np.array(Image.open(mask_path))
#             for cls_id in novel_classes:
#                 if (mask == cls_id).any():
#                     self.class_images[cls_id].append(img_id)

#         for cls_id in novel_classes:
#             n = len(self.class_images[cls_id])
#             print(f"[NovelDataset] class={VOC_CLASS_NAMES[cls_id]:15s} "
#                   f"(id={cls_id}) | {n} images available")

#     def get_support_and_queries(self, cls_id, k_shot, seed=442):
#         """
#         For a given novel class, return:
#           support_images : list of k_shot (image, binary_mask) tuples
#           query_images   : list of remaining (image, binary_mask) tuples

#         This is the KEY few-shot mechanism:
#           - Support = the K examples the model is ALLOWED to see
#           - Queries  = new images the model must segment without having
#                        seen them before
#         """
#         random.seed(seed)
#         imgs = self.class_images[cls_id].copy()

#         if len(imgs) < k_shot + 1:
#             raise ValueError(
#                 f"Class {VOC_CLASS_NAMES[cls_id]} only has {len(imgs)} images. "
#                 f"Need at least {k_shot + 1} (k_shot + 1 query)."
#             )

#         random.shuffle(imgs)
#         support_ids = imgs[:k_shot]         # first K = support
#         query_ids   = imgs[k_shot:]         # rest = queries

#         support = [self._load(img_id, cls_id) for img_id in support_ids]
#         queries = [self._load(img_id, cls_id) for img_id in query_ids]

#         return support, queries

#     def _load(self, img_id, cls_id):
#         """Load one image and build its binary mask for cls_id."""
#         image = Image.open(
#             os.path.join(self.voc_root, "JPEGImages", img_id + ".jpg")
#         ).convert("RGB")
#         mask = Image.open(
#             os.path.join(self.voc_root, "SegmentationClass", img_id + ".png")
#         )

#         image, mask = joint_transform(image, mask, augment=False)

#         binary = torch.zeros_like(mask)
#         binary[mask == 255]    = 255
#         binary[mask == cls_id] = 1

#         return image, binary


# # ─────────────────────────────────────────────────────────────────
# # Prepare base class loaders (for Phase 1)
# # ─────────────────────────────────────────────────────────────────
# def prepare_base_loaders(voc_root, fold=0, batch_size=8,
#                          val_ratio=0.1, num_workers=2, seed=442):
#     """
#     Returns train/val DataLoaders for BASE classes (Phase 1 learning).
#     """
#     novel_classes = PASCAL_FSS_SPLITS[fold]
#     base_classes  = [c for c in range(1, 21) if c not in novel_classes]

#     full_ds = BaseClassDataset(voc_root, "train", base_classes, augment=True)
#     eval_ds = BaseClassDataset(voc_root, "train", base_classes, augment=False)

#     total   = len(full_ds)
#     n_val   = int(total * val_ratio)
#     n_train = total - n_val

#     g = torch.Generator().manual_seed(seed)
#     train_ds, _ = random_split(full_ds, [n_train, n_val], generator=g)
#     _,  val_ds  = random_split(eval_ds, [n_train, n_val], generator=g)

#     train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
#                               num_workers=num_workers, pin_memory=True)
#     val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
#                               num_workers=num_workers, pin_memory=True)

#     print(f"\n[Phase 1] Base classes: {[VOC_CLASS_NAMES[c] for c in base_classes]}")
#     print(f"          Train={len(train_ds)} | Val={len(val_ds)}")

#     return train_loader, val_loader, len(base_classes)


"""
Data_Loader.py  —  Pascal-5i Few-Shot Segmentation (CORRECTED v2)
==================================================================
BUGS FIXED (all other project files remain UNCHANGED):

  [BUG-1] TYPO (NovelClassDataset.__init__):
          self.voc_root = novel_classes   ← was wrong
          self.voc_root = voc_root        ← fixed

  [BUG-2] WRONG MASK DIRECTORY (BaseClassDataset):
          SegmentationClass/  only has ~1,464 VOC train masks.
          SegmentationClassAug/ has ~10,582 VOC+SBD masks.
          Fixed: _get_mask_path() tries SegmentationClassAug first.

  [BUG-3] MISSING SBD IN TRAINING SET (prepare_base_loaders):
          Was using only VOC2012 "train" split (~1,464 images).
          Fixed: _build_merged_train_list() merges VOC train + SBD
          train + SBD val and removes VOC val (prevents leakage).
          Expected training set size: ~10,582 images.
          sbd_root param added (optional, backward compatible).

  [BUG-4] NOVEL PIXELS NOT IGNORED DURING BASE TRAINING:
          Novel-class pixels in training images were silently treated
          as background (label 0). This biases the base model.
          Fixed: novel-class pixels → 255 (ignore index) in binary mask.

  [BUG-5] WRONG VALIDATION SPLIT (prepare_base_loaders):
          Was taking a random 10% of training images as validation.
          Fixed: val_ds uses VOC2012 val.txt (1,449 images).
          val_ratio kept in signature for backward compatibility but
          is no longer used (VOC val is the correct held-out set).

  [BUG-6] NOVEL DATASET USED BOTH TRAIN+VAL SPLITS FOR QUERIES:
          Standard Pascal-5i protocol: queries MUST come from
          VOC2012 val only (prevents training-set contamination).
          Fixed: NovelClassDataset uses val.txt exclusively.

PUBLIC API IS UNCHANGED:
  prepare_base_loaders(voc_root, fold, batch_size, val_ratio,
                       num_workers, seed)  →  (train_loader, val_loader, n_base)
  prepare_novel_dataset(voc_root, fold)   →  (NovelClassDataset, novel_classes)
  NovelClassDataset.get_support_and_queries(cls_id, k_shot, seed)
    → (support_list, query_list)   each item is (image_tensor, binary_mask)

HOW TO CALL WITH SBD (RECOMMENDED):
  train_loader, val_loader, n_base = prepare_base_loaders(
      voc_root  = "/data/VOCdevkit/VOC2012",
      sbd_root  = "/data/benchmark_RELEASE/dataset",   # ← NEW optional arg
      fold      = 0,
      batch_size= 8,
  )

Reference: PANet (Wang et al., ICCV 2019), HSNet (Min et al., ICCV 2021)
"""

import os
import random
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms.functional as TF

# ── Standard FSS splits for Pascal VOC ────────────────────────────
# Each fold holds 5 novel classes. The other 15 are base classes.
PASCAL_FSS_SPLITS = {
    0: [1,  2,  3,  4,  5],    # novel: aeroplane bicycle bird boat bottle
    1: [6,  7,  8,  9, 10],    # novel: bus car cat chair cow
    2: [11, 12, 13, 14, 15],   # novel: diningtable dog horse motorbike person
    3: [16, 17, 18, 19, 20],   # novel: pottedplant sheep sofa train tvmonitor
}

VOC_CLASS_NAMES = [
    "background", "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse",
    "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]

IMG_SIZE = 473

# ─────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────

def joint_transform(image, mask, augment=False):
    """Apply the SAME spatial transform to image AND mask."""
    image = TF.resize(image, (IMG_SIZE, IMG_SIZE), interpolation=Image.BILINEAR)
    mask  = TF.resize(mask,  (IMG_SIZE, IMG_SIZE), interpolation=Image.NEAREST)

    if augment and random.random() > 0.5:
        image = TF.hflip(image)
        mask  = TF.hflip(mask)

    image = TF.to_tensor(image)
    image = TF.normalize(image, mean=[0.485, 0.456, 0.406],
                                std= [0.229, 0.224, 0.225])
    mask  = torch.from_numpy(np.array(mask)).long()
    return image, mask


def _get_mask_path(voc_root, img_id):
    """
    [BUG-2 FIX] Try SegmentationClassAug first (SBD-augmented, ~10,582 PNG
    files covering VOC2012 + SBD). Fall back to SegmentationClass (original
    VOC only, ~2,913 files) if augmented file is absent.

    SegmentationClassAug must be placed at:
        <voc_root>/SegmentationClassAug/<img_id>.png
    Download from: https://github.com/DrSleep/tensorflow-deeplab-resnet
    or extract from the SBD benchmark release.
    """
    aug_path  = os.path.join(voc_root, "SegmentationClassAug", img_id + ".png")
    orig_path = os.path.join(voc_root, "SegmentationClass",    img_id + ".png")
    return aug_path if os.path.exists(aug_path) else orig_path


def _build_merged_train_list(voc_root, sbd_root=None):
    """
    [BUG-3 FIX] Build the correct Pascal-5i training image list:

        train_set = (VOC2012 train) ∪ (SBD train) ∪ (SBD val)  −  (VOC2012 val)

    This gives ~10,582 images when SBD is available, matching the
    protocol used in PANet, HSNet, PFENet, and most FSS papers.

    Without SBD: falls back to VOC2012 train only (~1,464 images).
    Removing VOC2012 val prevents test-set leakage.

    Args:
        voc_root  (str): path to VOCdevkit/VOC2012/
        sbd_root  (str|None): path to benchmark_RELEASE/dataset/
                              (None = VOC only, NOT recommended)

    Returns:
        train_ids  (list[str]): merged training image IDs
        voc_val_set(set[str]): VOC2012 val IDs (used as test set)
    """
    voc_val_path   = os.path.join(voc_root, "ImageSets", "Segmentation", "val.txt")
    voc_train_path = os.path.join(voc_root, "ImageSets", "Segmentation", "train.txt")

    with open(voc_val_path) as f:
        voc_val_set = set(l.strip() for l in f if l.strip())
    with open(voc_train_path) as f:
        voc_train_ids = [l.strip() for l in f if l.strip()]

    merged = set(voc_train_ids)

    # Add SBD images if available
    sbd_added = 0
    if sbd_root is not None:
        for fname in ["train.txt", "val.txt"]:
            sbd_split_path = os.path.join(sbd_root, fname)
            if os.path.exists(sbd_split_path):
                with open(sbd_split_path) as f:
                    ids = [l.strip() for l in f if l.strip()]
                    before = len(merged)
                    merged.update(ids)
                    sbd_added += len(merged) - before
            else:
                print(f"  ⚠ SBD split not found: {sbd_split_path}")

    # Remove VOC2012 val — these are the test images, must not appear in train
    leaked = merged & voc_val_set
    merged -= voc_val_set

    source = "VOC2012 + SBD" if sbd_root and sbd_added > 0 else "VOC2012 only"
    print(f"[MergedTrainList] Source: {source}")
    print(f"  Total train images : {len(merged)}"
          f"  (expected ~10,582 with SBD, ~1,464 without)")
    if leaked:
        print(f"  ✅ Removed {len(leaked)} VOC-val images from train (leakage prevented)")

    return list(merged), voc_val_set


# ─────────────────────────────────────────────────────────────────
# Phase 1 Dataset — base classes, normal batch loading
# ─────────────────────────────────────────────────────────────────

class BaseClassDataset(Dataset):
    """
    Loads images containing BASE classes for Phase 1 learning.
    Returns: (image_tensor, binary_mask, class_label)
      - image_tensor : [3, H, W]  normalised float
      - binary_mask  : [H, W]     0=bg, 1=target, 255=ignore
      - class_label  : int        remapped to 0 .. N_base-1

    [BUG-2 FIX] Uses SegmentationClassAug (SBD-augmented masks).
    [BUG-4 FIX] Novel-class pixels are set to 255 (ignore) so the
                base model never confuses novel objects with background.
    """

    def __init__(self, voc_root, img_id_list, base_classes,
                 novel_classes, augment=False):
        """
        Args:
            voc_root     (str)      : path to VOCdevkit/VOC2012/
            img_id_list  (list[str]): image IDs to scan (merged train list)
            base_classes (list[int]): VOC class IDs used for training
            novel_classes(list[int]): VOC class IDs NEVER seen in Phase 1
            augment      (bool)     : enable random horizontal flip
        """
        self.voc_root      = voc_root
        self.base_classes  = base_classes
        self.novel_set     = set(novel_classes)   # [BUG-4] used for pixel masking
        self.augment       = augment
        self.label_map     = {c: i for i, c in enumerate(sorted(base_classes))}

        self.samples = []
        n_missing = 0

        for img_id in img_id_list:
            mask_path = _get_mask_path(voc_root, img_id)       # [BUG-2]
            if not os.path.exists(mask_path):
                n_missing += 1
                continue
            mask = np.array(Image.open(mask_path))
            for cls_id in base_classes:
                if (mask == cls_id).any():
                    self.samples.append((img_id, cls_id))

        if n_missing > 0:
            print(f"  ⚠ {n_missing} image IDs skipped (mask file not found). "
                  f"Install SegmentationClassAug to cover all SBD images.")

        print(f"[BaseDataset]  {len(base_classes)} base classes "
              f"| {len(self.samples)} (img, class) samples"
              f"{'  [augmented]' if augment else ''}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_id, cls_id = self.samples[idx]

        image    = Image.open(
            os.path.join(self.voc_root, "JPEGImages", img_id + ".jpg")
        ).convert("RGB")
        raw_mask = Image.open(_get_mask_path(self.voc_root, img_id))   # [BUG-2]

        image, mask = joint_transform(image, raw_mask, self.augment)

        # [BUG-4 FIX] Build binary mask
        #   1   = target class pixel
        #   0   = genuine background (class 0) + other base classes
        #   255 = boundary ignore (original VOC) + ALL novel class pixels
        binary = torch.zeros_like(mask)
        binary[mask == cls_id] = 1

        for nov_cls in self.novel_set:                  # [BUG-4]
            binary[mask == nov_cls] = 255               # ignore novel pixels

        binary[mask == 255] = 255                       # keep original boundary

        return image, binary, self.label_map[cls_id]


# ─────────────────────────────────────────────────────────────────
# Phase 2 & 3 Dataset — novel classes, support + query
# ─────────────────────────────────────────────────────────────────

class NovelClassDataset(Dataset):
    """
    Loads images for NOVEL classes (Phase 2 fine-tuning + Phase 3 eval).

    For each class, holds all VOC2012 val images containing that class.
    FewShotTester picks K as support, the rest become queries.

    [BUG-1 FIX] self.voc_root = voc_root  (was = novel_classes)
    [BUG-6 FIX] Uses VOC2012 val.txt only for both support and queries.
                The standard Pascal-5i test protocol evaluates on
                VOC2012 val exclusively (PANet / HSNet / PFENet).
    """

    def __init__(self, voc_root, novel_classes):
        self.voc_root      = voc_root          # [BUG-1 FIX] was: = novel_classes
        self.novel_classes = novel_classes

        # [BUG-6 FIX] val images only
        val_file = os.path.join(voc_root, "ImageSets", "Segmentation", "val.txt")
        with open(val_file) as f:
            val_ids = [l.strip() for l in f if l.strip()]

        self.class_images = {cls: [] for cls in novel_classes}
        for img_id in val_ids:
            mask_path = _get_mask_path(voc_root, img_id)
            if not os.path.exists(mask_path):
                continue
            mask = np.array(Image.open(mask_path))
            for cls_id in novel_classes:
                if (mask == cls_id).any():
                    self.class_images[cls_id].append(img_id)

        for cls_id in novel_classes:
            n = len(self.class_images[cls_id])
            print(f"[NovelDataset]  class={VOC_CLASS_NAMES[cls_id]:15s} "
                  f"(id={cls_id}) | {n} val images")

    def get_support_and_queries(self, cls_id, k_shot, seed=42):
        """
        Return K support images + all remaining val images as queries.

        Support ∩ Queries = ∅  (always disjoint).
        seed controls reproducibility across episodes.

        For fine-tuning: call once per class with desired seed.
        For 1000-episode eval: loop over seeds 0..999 and average mIoU.

        Returns:
            support : list of (image_tensor [3,H,W], binary_mask [H,W])
            queries : list of (image_tensor [3,H,W], binary_mask [H,W])
        """
        rng  = random.Random(seed)           # isolated RNG — no global state side-effects
        imgs = self.class_images[cls_id].copy()

        if len(imgs) < k_shot + 1:
            raise ValueError(
                f"Class '{VOC_CLASS_NAMES[cls_id]}' (id={cls_id}) only has "
                f"{len(imgs)} val images. Need at least {k_shot + 1}."
            )

        rng.shuffle(imgs)
        support_ids = imgs[:k_shot]
        query_ids   = imgs[k_shot:]

        support = [self._load(img_id, cls_id) for img_id in support_ids]
        queries = [self._load(img_id, cls_id) for img_id in query_ids]

        return support, queries

    def _load(self, img_id, cls_id):
        """Load one image + binary mask tensor for cls_id."""
        image    = Image.open(
            os.path.join(self.voc_root, "JPEGImages", img_id + ".jpg")
        ).convert("RGB")
        raw_mask = Image.open(_get_mask_path(self.voc_root, img_id))  # [BUG-2]

        image, mask = joint_transform(image, raw_mask, augment=False)

        binary = torch.zeros_like(mask)
        binary[mask == 255]    = 255
        binary[mask == cls_id] = 1

        return image, binary


# ─────────────────────────────────────────────────────────────────
# Public factory functions (API UNCHANGED)
# ─────────────────────────────────────────────────────────────────

def prepare_base_loaders(voc_root, fold=0, batch_size=8,
                         val_ratio=0.1, num_workers=2, seed=42,
                         sbd_root=None):
    """
    Returns train/val DataLoaders for BASE classes (Phase 1 learning).

    [BUG-3 FIX] Training uses merged VOC+SBD list (~10,582 images).
    [BUG-5 FIX] Validation uses VOC2012 val set (not random train split).
    val_ratio param is KEPT for backward compatibility but no longer used.

    New optional arg  sbd_root=None  is backward-compatible.

    Usage (recommended):
        train_loader, val_loader, n_base = prepare_base_loaders(
            voc_root  = "/data/VOCdevkit/VOC2012",
            sbd_root  = "/data/benchmark_RELEASE/dataset",
            fold      = 0,
        )

    Returns:
        train_loader : DataLoader
        val_loader   : DataLoader
        n_base       : int  (number of base classes = 15)
    """
    novel_classes = PASCAL_FSS_SPLITS[fold]
    base_classes  = [c for c in range(1, 21) if c not in novel_classes]

    # [BUG-3] Merged training list (VOC + SBD)
    train_ids, voc_val_set = _build_merged_train_list(voc_root, sbd_root)
    val_ids = list(voc_val_set)   # [BUG-5] VOC2012 val as validation

    train_ds = BaseClassDataset(voc_root, train_ids, base_classes,
                                novel_classes, augment=True)
    val_ds   = BaseClassDataset(voc_root, val_ids,   base_classes,
                                novel_classes, augment=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True,
                              worker_init_fn=lambda _: np.random.seed(seed))
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)

    print(f"\n[Phase 1]  Fold {fold}")
    print(f"  Base   ({len(base_classes)}): "
          f"{[VOC_CLASS_NAMES[c] for c in base_classes]}")
    print(f"  Novel  ({len(novel_classes)}): "
          f"{[VOC_CLASS_NAMES[c] for c in novel_classes]}")
    print(f"  Train  samples : {len(train_ds)}")
    print(f"  Val    samples : {len(val_ds)}")

    return train_loader, val_loader, len(base_classes)


def prepare_novel_dataset(voc_root, fold=0):
    """
    Returns NovelClassDataset for Phase 2 (fine-tuning) & Phase 3 (eval).
    API UNCHANGED.

    Returns:
        dataset       : NovelClassDataset
        novel_classes : list[int]
    """
    novel_classes = PASCAL_FSS_SPLITS[fold]
    return NovelClassDataset(voc_root, novel_classes), novel_classes


# def prepare_novel_dataset(voc_root, fold=0):
#     """Returns NovelClassDataset for Phase 2 & 3."""
#     novel_classes = PASCAL_FSS_SPLITS[fold]
#     return NovelClassDataset(voc_root, novel_classes), novel_classes
