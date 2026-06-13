# **1. no RL, fixed preset**


```python
import os
import sys
import time
import math
import copy
import hashlib
import random
import subprocess
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    log_loss, confusion_matrix, roc_auc_score,
    roc_curve, precision_recall_curve, average_precision_score,
    matthews_corrcoef, cohen_kappa_score, balanced_accuracy_score,
    jaccard_score
)

# ============================================================
# ARCF-Net ABLATION 1
# NO RL / FIXED PRESET / FULL PARTICIPATION / NO PLOTS
# ------------------------------------------------------------
# - Uses BOTH datasets
# - Kaggle-ready
# - True FL with FedAvg + FedProx + prototype sharing
# - NO RL-UCB for client count or preprocessing selection
# - Fixed client count per dataset
# - Fixed preprocessing preset per dataset
# - Full participation every round
# - No plots generated
# ============================================================

# -------------------------
# Helpers
# -------------------------
def pip_install(pkg):
    subprocess.check_call([sys.executable, "-m", "pip", "-q", "install", pkg])

def running_on_kaggle():
    return os.path.exists("/kaggle/input")

IS_KAGGLE = running_on_kaggle()

try:
    import kagglehub
except Exception:
    try:
        pip_install("kagglehub")
        import kagglehub
    except Exception:
        kagglehub = None

try:
    from torchvision import transforms
    from torchvision.models import resnet50, ResNet50_Weights
except Exception:
    pip_install("torchvision")
    from torchvision import transforms
    from torchvision.models import resnet50, ResNet50_Weights

# -------------------------
# Reproducibility + Device
# -------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if DEVICE.type == "cuda":
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

print("=" * 118)
print("ARCF-Net ABLATION 1: NO RL + FIXED PRESET + FULL PARTICIPATION + NO PLOTS")
print("=" * 118)
print(f"ENV: {'KAGGLE' if IS_KAGGLE else 'NON-KAGGLE'} | DEVICE: {DEVICE} | torch={torch.__version__}")
print("=" * 118)

# -------------------------
# Configuration
# -------------------------
CFG = {
    "rounds": 15,

    # fixed clients (no RL planning)
    "fixed_clients_ds1": 3,
    "fixed_clients_ds2": 3,

    # local training
    "local_epochs": 2,
    "lr_head": 1e-3,
    "lr_backbone": 2e-4,
    "weight_decay": 5e-4,
    "warmup_epochs": 1,
    "label_smoothing": 0.02,
    "focal_gamma": 1.35,
    "grad_clip": 1.0,
    "fedprox_mu": 0.01,
    "proto_lambda": 0.12,

    # image
    "img_size": 224 if torch.cuda.is_available() else 160,
    "batch_size": 16 if torch.cuda.is_available() else 8,
    "num_workers": 2 if torch.cuda.is_available() else 2,

    # global split
    "global_val_frac": 0.15,
    "test_frac": 0.15,

    # client split
    "client_val_frac": 0.12,
    "client_tune_frac": 0.12,
    "min_per_class_per_client": 5,

    # non-iid
    "dirichlet_alpha": 0.35,

    # preprocessing / augmentation
    "use_preprocessing": True,
    "use_augmentation": True,

    # transfer learning
    "freeze_backbone_rounds": 2,
    "unfreeze_last_blocks": 2,

    # final inference
    "final_use_tta": True,

    # reward
    "reward_f1_weight": 0.75,
    "reward_acc_weight": 0.25,

    # best-round selection: equal dataset importance
    "best_round_mass_ds1": 0.50,
    "best_round_mass_ds2": 0.50,
    "best_round_min_bonus": 0.15,

    # FedAvg tempering
    "fedavg_temper": 0.50,

    # misc / sanity
    "quick_hash_subset_per_split": 300,
    "preproc_val_sample_n": 400,

    # no plots
    "make_plots": False,
    "calibration_bins": 12,
}

OUTDIR = "/kaggle/working/outputs" if IS_KAGGLE else "/content/outputs"
os.makedirs(OUTDIR, exist_ok=True)
MODEL_PATH = os.path.join(OUTDIR, "ARCFNet_Ablation_NoRL_FixedPreset_checkpoint.pth")
CSV_PATH = os.path.join(OUTDIR, "ARCFNet_Ablation_NoRL_FixedPreset_outputs.csv")

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], device=DEVICE).view(1, 3, 1, 1)
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225], device=DEVICE).view(1, 3, 1, 1)

labels = ["glioma", "meningioma", "notumor", "pituitary"]
label2id = {l: i for i, l in enumerate(labels)}
id2label = {i: l for l, i in label2id.items()}
NUM_CLASSES = len(labels)

METHOD_INFO = {
    "acronym": "ARCF-Net",
    "full_form": "Adaptive RACE-FELCM with CRAF Fusion Network",
    "preprocessing_full_form": "Robust Adaptive Context-Enhanced Fuzzy Edge Local Contrast Mapping",
    "fusion_full_form": "Cross-Residual Adaptive Fusion",
    "backbone_full_form": "Residual Network-50",
    "ablation_name": "No RL, Fixed Preset, Full Participation",
}

# ============================================================
# Fixed presets (NO RL selection)
# ============================================================
THETA_FULLFORMS = {
    "gamma": "Robust power exponent",
    "alpha": "Local contrast enhancement weight",
    "beta": "Contrast saturation sharpness",
    "tau": "Robust clipping threshold",
    "k": "Local context blur kernel",
    "edge_gain": "Gradient structure gain",
    "blend": "Structure-preserving residual blend",
}

PRESET_BANK_DS1 = [
    ("race_balanced",  (1.00, 0.24, 4.6, 2.4, 5, 0.10, 0.78)),
    ("race_sharp",     (1.04, 0.30, 5.2, 2.5, 5, 0.12, 0.82)),
    ("race_texture",   (0.92, 0.34, 5.8, 2.3, 7, 0.10, 0.80)),
    ("race_robust",    (1.08, 0.22, 4.5, 2.8, 5, 0.11, 0.76)),
    ("race_edge_plus", (1.02, 0.32, 6.0, 2.6, 3, 0.15, 0.84)),
    ("race_focus",     (1.10, 0.36, 6.4, 2.7, 3, 0.17, 0.86)),
]

PRESET_BANK_DS2 = [
    ("race_soft",      (0.95, 0.18, 3.8, 2.2, 3, 0.08, 0.72)),
    ("race_balanced",  (1.00, 0.24, 4.6, 2.4, 5, 0.10, 0.78)),
    ("race_robust",    (1.08, 0.22, 4.5, 2.8, 5, 0.11, 0.76)),
    ("race_smoothmix", (0.90, 0.20, 4.0, 2.1, 7, 0.07, 0.70)),
]

FIXED_THETA_NAME_DS1 = "race_balanced"
FIXED_THETA_NAME_DS2 = "race_balanced"

def get_preset_by_name(bank, target_name):
    for name, theta in bank:
        if name == target_name:
            return name, theta
    raise ValueError(f"Preset {target_name} not found.")

FIXED_THETA_NAME_DS1, FIXED_THETA_DS1 = get_preset_by_name(PRESET_BANK_DS1, FIXED_THETA_NAME_DS1)
FIXED_THETA_NAME_DS2, FIXED_THETA_DS2 = get_preset_by_name(PRESET_BANK_DS2, FIXED_THETA_NAME_DS2)

# ============================================================
# CSV collector
# ============================================================
ALL_ROWS = []

def add_table_to_csv(df, table_name):
    if df is None or len(df) == 0:
        return
    df2 = df.copy()
    df2.insert(0, "table_name", table_name)
    for _, row in df2.iterrows():
        ALL_ROWS.append(row.to_dict())

def print_table(df, title, max_rows=12):
    print("\n" + "-" * 118)
    print(title)
    print("-" * 118)
    if df is None or len(df) == 0:
        print("[empty]")
    else:
        print(df.head(max_rows).to_string(index=False))
        if len(df) > max_rows:
            print(f"... showing first {max_rows} of {len(df)} rows")

def safe_float(x, default=np.nan):
    try:
        return float(x)
    except Exception:
        return default

def normalize_imagenet(x):
    return (x - IMAGENET_MEAN) / IMAGENET_STD

def score_metric(m):
    return (
        CFG["reward_f1_weight"] * safe_float(m.get("f1_macro"), 0.0)
        + CFG["reward_acc_weight"] * safe_float(m.get("acc"), 0.0)
    )

def frame_to_manifest(df_):
    return df_[["path", "label", "source", "filename", "y"]].to_dict(orient="list")

# ============================================================
# STEP 0: ACCESS DATASETS
# ============================================================
print("\n" + "=" * 118)
print("STEP 0: ACCESS DATASETS")
print("=" * 118)

REQ1 = {"512Glioma", "512Meningioma", "512Normal", "512Pituitary"}
REQ2 = {"glioma", "meningioma", "notumor", "pituitary"}

def norm_label(name: str):
    s = str(name).strip().lower()
    if "glioma" in s:
        return "glioma"
    if "meningioma" in s:
        return "meningioma"
    if "pituitary" in s:
        return "pituitary"
    if "normal" in s or "no_tumor" in s or "no tumor" in s or "notumor" in s:
        return "notumor"
    return None

def find_root_with_required_class_dirs(base_dir, required_set, prefer_raw=True):
    candidates = []
    for root, dirs, _ in os.walk(base_dir):
        if required_set.issubset(set(dirs)):
            candidates.append(root)
    if not candidates:
        return None

    def score(p):
        pl = p.lower()
        sc = 0
        if prefer_raw:
            if "raw data" in pl:
                sc += 7
            if os.path.basename(p).lower() == "raw":
                sc += 7
            if "/raw/" in pl or "\\raw\\" in pl:
                sc += 3
            if "augmented" in pl:
                sc -= 20
        sc -= 0.0001 * len(p)
        return sc

    return max(candidates, key=score)

def search_kaggle_input_for_root(required_set, prefer_raw=True):
    if not os.path.exists("/kaggle/input"):
        return None
    return find_root_with_required_class_dirs("/kaggle/input", required_set, prefer_raw=prefer_raw)

def try_kagglehub_download(slug, required_set, prefer_raw=True):
    if kagglehub is None:
        return None, None
    try:
        base = kagglehub.dataset_download(slug)
        root = find_root_with_required_class_dirs(base, required_set, prefer_raw=prefer_raw)
        return base, root
    except Exception:
        return None, None

DS1_BASE = None
DS2_BASE = None

DS1_ROOT = search_kaggle_input_for_root(REQ1, prefer_raw=True)
DS2_ROOT = search_kaggle_input_for_root(REQ2, prefer_raw=False)

if DS1_ROOT is None:
    DS1_BASE, DS1_ROOT = try_kagglehub_download(
        "orvile/pmram-bangladeshi-brain-cancer-mri-dataset",
        REQ1,
        prefer_raw=True,
    )

if DS2_ROOT is None:
    DS2_BASE, DS2_ROOT = try_kagglehub_download(
        "yassinebazgour/preprocessed-brain-mri-scans-for-tumors-detection",
        REQ2,
        prefer_raw=False,
    )

if DS1_ROOT is None:
    raise RuntimeError(
        "Could not locate DS1 under /kaggle/input. Add "
        "'orvile/pmram-bangladeshi-brain-cancer-mri-dataset'."
    )
if DS2_ROOT is None:
    raise RuntimeError(
        "Could not locate DS2 under /kaggle/input. Add "
        "'yassinebazgour/preprocessed-brain-mri-scans-for-tumors-detection'."
    )

print(f"Dataset-1 RAW root detected:\n  {DS1_ROOT}")
print(f"Dataset-2 root detected:\n  {DS2_ROOT}")

# ============================================================
# STEP 1: BUILD MANIFESTS
# ============================================================
print("\n" + "=" * 118)
print("STEP 1: BUILD DATA MANIFESTS")
print("=" * 118)

def list_images_under_class_root(class_root, class_dir_name):
    class_dir = os.path.join(class_root, class_dir_name)
    out = []
    for r, _, files in os.walk(class_dir):
        for fn in files:
            if fn.lower().endswith(IMG_EXTS):
                out.append(os.path.join(r, fn))
    return out

def build_df_from_root(ds_root, class_dirs, source_name):
    rows = []
    for c in class_dirs:
        lab = norm_label(c)
        imgs = list_images_under_class_root(ds_root, c)
        print(f"{source_name}: {c} -> {lab} | {len(imgs)} images")
        for p in imgs:
            rows.append({"path": p, "label": lab, "source": source_name})
    dfm = pd.DataFrame(rows).dropna().reset_index(drop=True)
    dfm["path"] = dfm["path"].astype(str)
    dfm["label"] = dfm["label"].astype(str)
    dfm["source"] = dfm["source"].astype(str)
    dfm = dfm.drop_duplicates(subset=["path"]).reset_index(drop=True)
    dfm["filename"] = dfm["path"].apply(os.path.basename)
    return dfm

df1 = build_df_from_root(DS1_ROOT, ["512Glioma", "512Meningioma", "512Normal", "512Pituitary"], "ds1_raw")
df2 = build_df_from_root(DS2_ROOT, ["glioma", "meningioma", "notumor", "pituitary"], "ds2")

def enforce_labels(df_):
    df_ = df_.copy()
    df_["label"] = df_["label"].astype(str).str.strip().str.lower()
    df_ = df_[df_["label"].isin(set(labels))].reset_index(drop=True)
    df_["y"] = df_["label"].map(label2id).astype(int)
    return df_

df1 = enforce_labels(df1)
df2 = enforce_labels(df2)

print("\nDataset-1 images:", len(df1))
print(df1["label"].value_counts().reindex(labels, fill_value=0).to_string())
print("\nDataset-2 images:", len(df2))
print(df2["label"].value_counts().reindex(labels, fill_value=0).to_string())

# ============================================================
# STEP 2: TRAIN / VAL / TEST SPLIT
# ============================================================
print("\n" + "=" * 118)
print("STEP 2: TRAIN / VAL / TEST SPLIT")
print("=" * 118)

def split_dataset(df_):
    train_df, temp_df = train_test_split(
        df_,
        test_size=(CFG["global_val_frac"] + CFG["test_frac"]),
        stratify=df_["y"],
        random_state=SEED,
    )
    val_rel = CFG["global_val_frac"] / (CFG["global_val_frac"] + CFG["test_frac"])
    val_df, test_df = train_test_split(
        temp_df,
        test_size=(1 - val_rel),
        stratify=temp_df["y"],
        random_state=SEED,
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)

train1, val1, test1 = split_dataset(df1)
train2, val2, test2 = split_dataset(df2)

print(f"DS1 TRAIN: {len(train1)} | VAL: {len(val1)} | TEST: {len(test1)}")
print(f"DS2 TRAIN: {len(train2)} | VAL: {len(val2)} | TEST: {len(test2)}")

# ============================================================
# STEP 2.5: SANITY / LEAKAGE CHECKS
# ============================================================
print("\n" + "=" * 118)
print("STEP 2.5: SANITY / LEAKAGE CHECKS")
print("=" * 118)

def split_overlap_checks(train_df, val_df, test_df):
    tr = set(train_df["path"].tolist())
    va = set(val_df["path"].tolist())
    te = set(test_df["path"].tolist())
    checks = {
        "path_overlap_train_val": len(tr.intersection(va)),
        "path_overlap_train_test": len(tr.intersection(te)),
        "path_overlap_val_test": len(va.intersection(te)),
        "unique_paths_train": len(tr),
        "unique_paths_val": len(va),
        "unique_paths_test": len(te),
    }
    trf = set(train_df["filename"].tolist())
    vaf = set(val_df["filename"].tolist())
    tef = set(test_df["filename"].tolist())
    checks.update({
        "filename_overlap_train_val": len(trf.intersection(vaf)),
        "filename_overlap_train_test": len(trf.intersection(tef)),
        "filename_overlap_val_test": len(vaf.intersection(tef)),
    })
    return checks

def md5_file(path, max_bytes=2_000_000):
    h = hashlib.md5()
    try:
        with open(path, "rb") as f:
            h.update(f.read(max_bytes))
        return h.hexdigest()
    except Exception:
        return None

def quick_hash_subset(frame, n=300):
    n = min(n, len(frame))
    if n <= 0:
        return set()
    idx = np.random.choice(len(frame), size=n, replace=False)
    hashes = []
    for i in idx:
        hv = md5_file(frame.iloc[i]["path"])
        if hv is not None:
            hashes.append(hv)
    return set(hashes)

def leakage_report(name, tr, va, te):
    over = split_overlap_checks(tr, va, te)
    leak_df = pd.DataFrame([over])

    n_hash = int(CFG["quick_hash_subset_per_split"])
    trh = quick_hash_subset(tr, n_hash)
    vah = quick_hash_subset(va, n_hash)
    teh = quick_hash_subset(te, n_hash)

    hash_over = {
        "subset_hash_train_val": len(trh.intersection(vah)),
        "subset_hash_train_test": len(trh.intersection(teh)),
        "subset_hash_val_test": len(vah.intersection(teh)),
        "subset_hash_n_train": len(trh),
        "subset_hash_n_val": len(vah),
        "subset_hash_n_test": len(teh),
    }
    leak_df = pd.concat([leak_df, pd.DataFrame([hash_over])], axis=1)
    print_table(leak_df, f"Leakage / Sanity Summary — {name}", max_rows=5)
    add_table_to_csv(leak_df, f"leakage_sanity_{name}")

leakage_report("ds1", train1, val1, test1)
leakage_report("ds2", train2, val2, test2)

# ============================================================
# STEP 3: NON-IID CLIENT PARTITIONING (FIXED)
# ============================================================
print("\n" + "=" * 118)
print("STEP 3: FIXED NON-IID CLIENT PARTITIONING")
print("=" * 118)

def make_clients_non_iid(train_df, n_clients, num_classes, min_per_class=5, alpha=0.35):
    y = train_df["y"].values
    idx_by_class = {c: np.where(y == c)[0].tolist() for c in range(num_classes)}
    for c in idx_by_class:
        random.shuffle(idx_by_class[c])

    client_indices = [[] for _ in range(n_clients)]

    for c in range(num_classes):
        idxs = idx_by_class[c]
        feasible = min(min_per_class, max(1, len(idxs) // n_clients))
        for k in range(n_clients):
            take = idxs[:feasible]
            idxs = idxs[feasible:]
            client_indices[k].extend(take)
        idx_by_class[c] = idxs

    for c in range(num_classes):
        idxs = idx_by_class[c]
        if len(idxs) == 0:
            continue
        props = np.random.dirichlet([alpha] * n_clients)
        counts = (props * len(idxs)).astype(int)
        diff = len(idxs) - counts.sum()
        counts[np.argmax(props)] += diff

        start = 0
        for k in range(n_clients):
            client_indices[k].extend(idxs[start:start + counts[k]])
            start += counts[k]

    for k in range(n_clients):
        random.shuffle(client_indices[k])

    return client_indices

def robust_client_splits(train_df, indices, val_frac, tune_frac):
    idxs = np.array(indices, dtype=int)
    if len(idxs) < 3:
        return idxs.tolist(), idxs.tolist(), idxs.tolist()

    yk = train_df.loc[idxs, "y"].values

    if len(np.unique(yk)) < 2 or len(idxs) < 20:
        n_tune = max(1, int(round(len(idxs) * tune_frac)))
        n_tune = min(n_tune, max(1, len(idxs) - 2))
        tune_idx = idxs[:n_tune]
        rem_idx = idxs[n_tune:]
    else:
        rem_idx, tune_idx = train_test_split(
            idxs,
            test_size=tune_frac,
            stratify=yk,
            random_state=SEED,
        )

    if len(rem_idx) < 2:
        return rem_idx.tolist(), tune_idx.tolist(), rem_idx.tolist()

    yk2 = train_df.loc[rem_idx, "y"].values
    if len(np.unique(yk2)) < 2 or len(rem_idx) < 12:
        n_val = max(1, int(round(len(rem_idx) * val_frac)))
        n_val = min(n_val, max(1, len(rem_idx) - 1))
        val_idx = rem_idx[:n_val]
        train_idx = rem_idx[n_val:]
    else:
        train_idx, val_idx = train_test_split(
            rem_idx,
            test_size=val_frac,
            stratify=yk2,
            random_state=SEED,
        )

    if len(train_idx) == 0:
        train_idx = val_idx[:]
    if len(val_idx) == 0:
        val_idx = train_idx[:1]
    return train_idx.tolist(), tune_idx.tolist(), val_idx.tolist()

n_clients_ds1 = int(CFG["fixed_clients_ds1"])
n_clients_ds2 = int(CFG["fixed_clients_ds2"])

client_indices_ds1 = make_clients_non_iid(
    train1,
    n_clients=n_clients_ds1,
    num_classes=NUM_CLASSES,
    min_per_class=CFG["min_per_class_per_client"],
    alpha=CFG["dirichlet_alpha"],
)
client_indices_ds2 = make_clients_non_iid(
    train2,
    n_clients=n_clients_ds2,
    num_classes=NUM_CLASSES,
    min_per_class=CFG["min_per_class_per_client"],
    alpha=CFG["dirichlet_alpha"],
)

print(f"Fixed clients for DS1: {n_clients_ds1}")
print(f"Fixed clients for DS2: {n_clients_ds2}")

# ============================================================
# STEP 4: DATA LOADERS
# ============================================================
print("\n" + "=" * 118)
print("STEP 4: DATA LOADERS")
print("=" * 118)

def load_rgb(path):
    try:
        return Image.open(path).convert("RGB")
    except Exception:
        return Image.new("RGB", (CFG["img_size"], CFG["img_size"]), (128, 128, 128))

EVAL_TFMS = transforms.Compose([
    transforms.Resize((CFG["img_size"], CFG["img_size"])),
    transforms.ToTensor(),
])

TRAIN_TFMS = transforms.Compose([
    transforms.Resize((CFG["img_size"], CFG["img_size"])),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=12),
    transforms.RandomAffine(degrees=0, translate=(0.04, 0.04), scale=(0.96, 1.04)),
    transforms.ColorJitter(brightness=0.10, contrast=0.10),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.6)),
    transforms.ToTensor(),
])

class MRIDataset(Dataset):
    def __init__(self, frame, indices=None, tfms=None, source_id=0):
        self.df = frame
        self.indices = indices if indices is not None else list(range(len(frame)))
        self.tfms = tfms
        self.source_id = int(source_id)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        j = self.indices[i]
        row = self.df.iloc[j]
        img = load_rgb(row["path"])
        x = self.tfms(img) if self.tfms is not None else transforms.ToTensor()(img)
        y = int(row["y"])
        return x, y, row["path"], self.source_id

def make_weighted_sampler(frame, indices, num_classes):
    if len(indices) == 0:
        return None
    ys = frame.loc[indices, "y"].values
    class_counts = np.bincount(ys, minlength=num_classes)
    class_weights = 1.0 / np.sqrt(np.clip(class_counts, 1, None))
    sample_weights = class_weights[ys]
    return WeightedRandomSampler(
        weights=torch.DoubleTensor(sample_weights),
        num_samples=len(sample_weights),
        replacement=True,
    )

def make_loader(frame, indices, bs, tfms, shuffle=False, sampler=None, source_id=0):
    ds = MRIDataset(frame, indices=indices, tfms=tfms, source_id=source_id)
    return DataLoader(
        ds,
        batch_size=bs,
        shuffle=(shuffle and sampler is None),
        sampler=sampler,
        num_workers=CFG["num_workers"],
        pin_memory=(DEVICE.type == "cuda"),
        drop_last=False,
        persistent_workers=(CFG["num_workers"] > 0),
    )

clients = []
gid = 0

for ds_name, df_src, client_indices, source_id in [
    ("ds1", train1, client_indices_ds1, 0),
    ("ds2", train2, client_indices_ds2, 1),
]:
    for local_id, idxs in enumerate(client_indices):
        tr_idx, tune_idx, val_idx = robust_client_splits(df_src, idxs, CFG["client_val_frac"], CFG["client_tune_frac"])
        sampler = make_weighted_sampler(df_src, tr_idx, NUM_CLASSES)

        train_loader = make_loader(
            df_src, tr_idx, CFG["batch_size"], TRAIN_TFMS,
            shuffle=(sampler is None), sampler=sampler, source_id=source_id
        )
        tune_loader = make_loader(
            df_src,
            tune_idx if len(tune_idx) else tr_idx[:max(1, min(len(tr_idx), CFG["batch_size"]))],
            CFG["batch_size"], EVAL_TFMS, shuffle=False, sampler=None, source_id=source_id
        )
        val_loader = make_loader(
            df_src,
            val_idx if len(val_idx) else tr_idx[:max(1, min(len(tr_idx), CFG["batch_size"]))],
            CFG["batch_size"], EVAL_TFMS, shuffle=False, sampler=None, source_id=source_id
        )
        proto_loader = make_loader(
            df_src, tr_idx, CFG["batch_size"], EVAL_TFMS,
            shuffle=False, sampler=None, source_id=source_id
        )

        counts = df_src.loc[tr_idx, "label"].value_counts().reindex(labels, fill_value=0)

        if ds_name == "ds1":
            fixed_theta_name, fixed_theta = FIXED_THETA_NAME_DS1, FIXED_THETA_DS1
        else:
            fixed_theta_name, fixed_theta = FIXED_THETA_NAME_DS2, FIXED_THETA_DS2

        clients.append({
            "gid": gid,
            "local_id": local_id,
            "dataset": ds_name,
            "source_id": source_id,
            "train_loader": train_loader,
            "tune_loader": tune_loader,
            "val_loader": val_loader,
            "proto_loader": proto_loader,
            "n_train": len(tr_idx),
            "n_tune": len(tune_idx),
            "n_val": len(val_idx),
            "train_indices": tr_idx,
            "tune_indices": tune_idx,
            "val_indices": val_idx,
            "class_counts": counts.to_dict(),
            "fixed_theta_name": fixed_theta_name,
            "fixed_theta": fixed_theta,
        })

        print(f"{ds_name} | client_{gid} | train={len(tr_idx)} | tune={len(tune_idx)} | val={len(val_idx)}")
        gid += 1

CLIENTS_TOTAL = len(clients)

dist_rows = []
for c in clients:
    row = {
        "client": f"client_{c['gid']}",
        "dataset": c["dataset"],
        "total_train": c["n_train"],
        "total_tune": c["n_tune"],
        "total_val": c["n_val"],
        "fixed_theta_name": c["fixed_theta_name"],
    }
    row.update({lab: int(c["class_counts"].get(lab, 0)) for lab in labels})
    dist_rows.append(row)

dist_df = pd.DataFrame(dist_rows)
print_table(dist_df, "Fixed client class distribution", max_rows=20)
add_table_to_csv(dist_df, "fixed_client_distribution")

val_loader_ds1 = make_loader(val1, list(range(len(val1))), CFG["batch_size"], EVAL_TFMS, shuffle=False, source_id=0)
val_loader_ds2 = make_loader(val2, list(range(len(val2))), CFG["batch_size"], EVAL_TFMS, shuffle=False, source_id=1)
test_loader_ds1 = make_loader(test1, list(range(len(test1))), CFG["batch_size"], EVAL_TFMS, shuffle=False, source_id=0)
test_loader_ds2 = make_loader(test2, list(range(len(test2))), CFG["batch_size"], EVAL_TFMS, shuffle=False, source_id=1)

print(f"Augmentation: {'ON' if CFG['use_augmentation'] else 'OFF'}")
print(f"Preprocessing: {'ON' if CFG['use_preprocessing'] else 'OFF'}")
print(f"Total clients: {CLIENTS_TOTAL}")
print(f"Fixed preprocessing preset DS1: {FIXED_THETA_NAME_DS1} {FIXED_THETA_DS1}")
print(f"Fixed preprocessing preset DS2: {FIXED_THETA_NAME_DS2} {FIXED_THETA_DS2}")

# ============================================================
# STEP 5: PREPROCESSING — RACE-FELCM
# ============================================================
print("\n" + "=" * 118)
print("STEP 5: FIXED PREPROCESSING — RACE-FELCM")
print("=" * 118)

class RACEFELCM(nn.Module):
    def __init__(self, gamma=1.0, alpha=0.24, beta=4.5, tau=2.4, blur_k=5, edge_gain=0.10, blend=0.78):
        super().__init__()
        self.gamma = float(gamma)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.tau = float(tau)
        self.blur_k = int(blur_k)
        self.edge_gain = float(edge_gain)
        self.blend = float(blend)

        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        lap = torch.tensor([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=torch.float32)

        self.register_buffer("sobel_x", sobel_x.view(1, 1, 3, 3))
        self.register_buffer("sobel_y", sobel_y.view(1, 1, 3, 3))
        self.register_buffer("lap", lap.view(1, 1, 3, 3))

    def forward(self, x):
        eps = 1e-6

        gray0 = x.mean(dim=1, keepdim=True)
        mu0 = F.avg_pool2d(F.pad(gray0, (1, 1, 1, 1), mode="reflect"), 3, stride=1)
        var0 = F.avg_pool2d(F.pad((gray0 - mu0).pow(2), (1, 1, 1, 1), mode="reflect"), 3, stride=1)
        tex = torch.sqrt(var0 + eps)
        tex_norm = tex / (tex.amax(dim=(2, 3), keepdim=True) + eps)
        flat_gate = 1.0 - tex_norm

        x_smooth = F.avg_pool2d(F.pad(x, (1, 1, 1, 1), mode="reflect"), 3, stride=1)
        x_dn = x * (1.0 - 0.18 * flat_gate) + x_smooth * (0.18 * flat_gate)

        mu = x_dn.mean(dim=(2, 3), keepdim=True)
        sd = x_dn.std(dim=(2, 3), keepdim=True).clamp_min(eps)
        z = ((x_dn - mu) / sd).clamp(-self.tau, self.tau)
        z = torch.sign(z) * torch.pow(z.abs().clamp_min(eps), self.gamma)

        gray = z.mean(dim=1, keepdim=True)
        k = self.blur_k if self.blur_k % 2 == 1 else self.blur_k + 1
        pad = k // 2

        local_mean = F.avg_pool2d(F.pad(gray, (pad, pad, pad, pad), mode="reflect"), k, stride=1)
        local_var = F.avg_pool2d(F.pad((gray - local_mean).pow(2), (pad, pad, pad, pad), mode="reflect"), k, stride=1)
        local_std = torch.sqrt(local_var + eps)
        local_norm = (gray - local_mean) / (local_std + eps)
        contrast_field = torch.tanh(self.beta * local_norm)

        gx = F.conv2d(F.pad(gray, (1, 1, 1, 1), mode="reflect"), self.sobel_x)
        gy = F.conv2d(F.pad(gray, (1, 1, 1, 1), mode="reflect"), self.sobel_y)
        edge = torch.sqrt(gx * gx + gy * gy + eps)
        edge = edge / (edge.amax(dim=(2, 3), keepdim=True) + eps)

        lap = F.conv2d(F.pad(gray, (1, 1, 1, 1), mode="reflect"), self.lap).abs()
        lap = lap / (lap.amax(dim=(2, 3), keepdim=True) + eps)
        coherence = torch.sigmoid(4.0 * (edge - 0.5 * lap))

        enhanced = z + self.alpha * contrast_field + self.edge_gain * edge * coherence

        mn = enhanced.amin(dim=(2, 3), keepdim=True)
        mx = enhanced.amax(dim=(2, 3), keepdim=True)
        enhanced = (enhanced - mn) / (mx - mn + eps)

        out = self.blend * enhanced + (1.0 - self.blend) * x
        return out.clamp(0, 1)

def theta_to_module(theta):
    return RACEFELCM(*theta)

def theta_to_vec(theta, batch_size):
    gamma, alpha, beta, tau, blur_k, edge_gain, blend = theta
    theta = torch.tensor(
        [gamma, alpha, beta / 8.0, tau / 4.0, blur_k / 7.0, edge_gain, blend],
        device=DEVICE,
        dtype=torch.float32
    )
    return theta.unsqueeze(0).repeat(batch_size, 1)

def theta_str(theta):
    if theta is None:
        return "None"
    g, a, b, t, k, eg, m = theta
    return f"(g={g:.2f}, a={a:.2f}, b={b:.2f}, t={t:.2f}, k={k}, eg={eg:.2f}, mix={m:.2f})"

# ============================================================
# STEP 6: MODEL — ResNet-50 + CRAF Fusion
# ============================================================
print("\n" + "=" * 118)
print("STEP 6: MODEL — ResNet-50 + CRAF Fusion")
print("=" * 118)

class CompactProjector(nn.Module):
    def __init__(self, in_dim, out_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            nn.Dropout(0.10),
        )

    def forward(self, x):
        return self.net(x)

class CRAFFusion(nn.Module):
    def __init__(self, in_dim, fuse_dim=256, cond_dim=64):
        super().__init__()
        self.proj_raw = CompactProjector(in_dim, fuse_dim)
        self.proj_enh = CompactProjector(in_dim, fuse_dim)
        self.proj_res = CompactProjector(in_dim, fuse_dim)

        self.router = nn.Sequential(
            nn.Linear(fuse_dim * 6 + cond_dim, fuse_dim),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(fuse_dim, 3),
        )

        self.residual_refine = nn.Sequential(
            nn.Linear(fuse_dim * 2 + cond_dim, fuse_dim),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(fuse_dim, fuse_dim),
        )

        self.out_norm = nn.LayerNorm(fuse_dim)

    def forward(self, f_raw, f_enh, f_res, cond):
        fr = self.proj_raw(f_raw)
        fe = self.proj_enh(f_enh)
        fs = self.proj_res(f_res)

        d_re = torch.abs(fr - fe)
        d_rs = torch.abs(fr - fs)
        d_es = torch.abs(fe - fs)

        router_in = torch.cat([fr, fe, fs, d_re, d_rs, d_es, cond], dim=1)
        gates = torch.softmax(self.router(router_in), dim=1)

        routed = gates[:, 0:1] * fr + gates[:, 1:2] * fe + gates[:, 2:3] * fs
        disagreement = (d_re + d_rs + d_es) / 3.0
        refine = self.residual_refine(torch.cat([routed, disagreement, cond], dim=1))
        fused = self.out_norm(routed + refine)

        return fused, gates

class ResNet50CRAF(nn.Module):
    def __init__(self, num_classes, cond_dim=64, fuse_dim=256, embed_dim=256, pretrained=True):
        super().__init__()
        try:
            weights = ResNet50_Weights.DEFAULT if pretrained else None
        except Exception:
            weights = None

        net = resnet50(weights=weights)
        self.backbone = nn.Sequential(
            net.conv1,
            net.bn1,
            net.relu,
            net.maxpool,
            net.layer1,
            net.layer2,
            net.layer3,
            net.layer4,
        )
        self.backbone_dim = net.fc.in_features
        self.pool = nn.AdaptiveAvgPool2d(1)

        self.theta_mlp = nn.Sequential(
            nn.Linear(7, cond_dim),
            nn.GELU(),
            nn.Linear(cond_dim, cond_dim),
        )
        self.source_emb = nn.Embedding(2, cond_dim)
        self.cond_norm = nn.LayerNorm(cond_dim)

        self.fusion = CRAFFusion(self.backbone_dim, fuse_dim=fuse_dim, cond_dim=cond_dim)

        self.neck = nn.Sequential(
            nn.LayerNorm(fuse_dim),
            nn.Linear(fuse_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(0.20),
        )
        self.classifier = nn.Linear(embed_dim, num_classes)
        self.embed_dim = embed_dim

    def _encode(self, x):
        f = self.backbone(x)
        f = self.pool(f).flatten(1)
        return f

    def forward(self, x_raw_n, x_enh_n, x_res_n, theta_vec, source_id, return_extra=False):
        cond = self.theta_mlp(theta_vec) + self.source_emb(source_id)
        cond = self.cond_norm(cond)

        f_raw = self._encode(x_raw_n)
        f_enh = self._encode(x_enh_n)
        f_res = self._encode(x_res_n)

        fused, gates = self.fusion(f_raw, f_enh, f_res, cond)
        embed = self.neck(fused)
        logits = self.classifier(embed)

        if return_extra:
            return logits, embed, gates
        return logits

def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

def set_trainable_for_round(model, rnd):
    for p in model.backbone.parameters():
        p.requires_grad = False

    for n, p in model.named_parameters():
        if not n.startswith("backbone."):
            p.requires_grad = True

    if rnd > CFG["freeze_backbone_rounds"]:
        blocks = list(model.backbone.children())
        tail_blocks = blocks[-CFG["unfreeze_last_blocks"]:]
        for blk in tail_blocks:
            for p in blk.parameters():
                p.requires_grad = True

def freeze_backbone_bn_stats(model):
    for m in model.backbone.modules():
        if isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm)):
            m.eval()
            for p in m.parameters():
                p.requires_grad = False

def make_optimizer(model):
    head_params, bb_params = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if n.startswith("backbone."):
            bb_params.append(p)
        else:
            head_params.append(p)

    groups = []
    if head_params:
        groups.append({"params": head_params, "lr": CFG["lr_head"]})
    if bb_params:
        groups.append({"params": bb_params, "lr": CFG["lr_backbone"]})

    return torch.optim.AdamW(groups, weight_decay=CFG["weight_decay"])

def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
    def lr_lambda(step):
        if step < num_warmup_steps:
            return float(step) / float(max(1, num_warmup_steps))
        progress = float(step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

global_model = ResNet50CRAF(
    num_classes=NUM_CLASSES,
    cond_dim=64,
    fuse_dim=256,
    embed_dim=256,
    pretrained=True,
).to(DEVICE)

set_trainable_for_round(global_model, rnd=1)
total_params, trainable_params = count_params(global_model)

print("Backbone: ResNet-50 | pretrained_loaded=True")
print(f"Total params: {total_params:,}")
print(f"Trainable params: {trainable_params:,} ({(100.0 * trainable_params / total_params):.2f}%)")

# ============================================================
# STEP 7: LOSSES + PROTOTYPE SHARING
# ============================================================
print("\n" + "=" * 118)
print("STEP 7: LOSSES + PROTOTYPE SHARING")
print("=" * 118)

counts1 = train1["y"].value_counts().sort_index().reindex(range(NUM_CLASSES), fill_value=0).values
counts2 = train2["y"].value_counts().sort_index().reindex(range(NUM_CLASSES), fill_value=0).values
counts = counts1 + counts2
w = (counts.sum() / np.clip(counts, 1, None)).astype(np.float32)
w = w / max(1e-6, w.mean())
class_w = torch.tensor(w, device=DEVICE)

class ClassBalancedFocalLoss(nn.Module):
    def __init__(self, class_weights, gamma=1.5, label_smoothing=0.0):
        super().__init__()
        self.class_weights = class_weights
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, logits, target):
        ce = F.cross_entropy(
            logits,
            target,
            weight=self.class_weights,
            reduction="none",
            label_smoothing=self.label_smoothing,
        )
        pt = torch.exp(-ce)
        focal = (1.0 - pt).pow(self.gamma) * ce
        return focal.mean()

criterion = ClassBalancedFocalLoss(
    class_weights=class_w,
    gamma=CFG["focal_gamma"],
    label_smoothing=CFG["label_smoothing"],
)

scaler = torch.amp.GradScaler("cuda", enabled=(DEVICE.type == "cuda")) if DEVICE.type == "cuda" else None

def fedprox_term(local_model, global_model):
    loss = 0.0
    for p_local, p_global in zip(local_model.parameters(), global_model.parameters()):
        loss = loss + ((p_local - p_global.detach()) ** 2).sum()
    return loss

def prototype_alignment_loss(embed, y, global_prototypes):
    if global_prototypes is None:
        return torch.tensor(0.0, device=embed.device)

    proto = global_prototypes["proto"]
    mask = global_prototypes["mask"]
    valid = mask[y]
    if not valid.any():
        return torch.tensor(0.0, device=embed.device)

    emb_n = F.normalize(embed[valid], dim=1)
    ref_n = F.normalize(proto[y[valid]], dim=1)
    return (1.0 - (emb_n * ref_n).sum(dim=1)).mean()

@torch.no_grad()
def compute_prototype_payload(model, loader, preproc, theta):
    model.eval()
    freeze_backbone_bn_stats(model)
    preproc.eval()

    sums = torch.zeros(NUM_CLASSES, model.embed_dim, device=DEVICE)
    counts = torch.zeros(NUM_CLASSES, device=DEVICE)

    for x, y, _, source_id in loader:
        x = x.to(DEVICE, non_blocking=True)
        y = y.to(DEVICE, non_blocking=True)
        source_id = source_id.to(DEVICE, non_blocking=True)

        x_enh = preproc(x)
        x_res = torch.abs(x_enh - x)

        x_raw_n = normalize_imagenet(x)
        x_enh_n = normalize_imagenet(x_enh)
        x_res_n = normalize_imagenet(x_res)
        theta_vec = theta_to_vec(theta, x.size(0))

        _, embed, _ = model(x_raw_n, x_enh_n, x_res_n, theta_vec, source_id, return_extra=True)

        for c in range(NUM_CLASSES):
            m = (y == c)
            if m.any():
                sums[c] += embed[m].sum(dim=0)
                counts[c] += m.sum()

    return sums.detach().cpu(), counts.detach().cpu()

def aggregate_prototypes(payloads, embed_dim):
    if len(payloads) == 0:
        return None
    sum_all = torch.zeros(NUM_CLASSES, embed_dim)
    cnt_all = torch.zeros(NUM_CLASSES)
    for s, c in payloads:
        sum_all += s
        cnt_all += c
    proto = sum_all / cnt_all.unsqueeze(1).clamp_min(1.0)
    mask = cnt_all > 0
    return {
        "proto": proto.to(DEVICE),
        "mask": mask.to(DEVICE),
        "counts": cnt_all.to(DEVICE),
    }

def gate_entropy(g):
    eps = 1e-6
    p = g.clamp(eps, 1 - eps)
    return -(p * torch.log2(p)).sum(dim=1)

@torch.no_grad()
def _auc_metrics(y_true, p_pred, num_classes):
    out = {}
    try:
        out["auc_roc_macro_ovr"] = float(roc_auc_score(y_true, p_pred, multi_class="ovr", average="macro"))
    except Exception:
        out["auc_roc_macro_ovr"] = np.nan

    for c in range(num_classes):
        try:
            yc = (y_true == c).astype(int)
            if yc.sum() > 0 and yc.sum() < len(yc):
                out[f"auc_class_{c}"] = float(roc_auc_score(yc, p_pred[:, c]))
            else:
                out[f"auc_class_{c}"] = np.nan
        except Exception:
            out[f"auc_class_{c}"] = np.nan
    return out

@torch.no_grad()
def predict_probs_single_theta(model, x, source_id, preproc, theta, use_tta=False):
    model.eval()
    freeze_backbone_bn_stats(model)
    preproc.eval()

    probs_acc = None
    versions = [x]
    if use_tta:
        versions.append(torch.flip(x, dims=[3]))

    for xv in versions:
        x_enh = preproc(xv)
        x_res = torch.abs(x_enh - xv)

        x_raw_n = normalize_imagenet(xv)
        x_enh_n = normalize_imagenet(x_enh)
        x_res_n = normalize_imagenet(x_res)
        theta_vec = theta_to_vec(theta, xv.size(0))

        logits = model(x_raw_n, x_enh_n, x_res_n, theta_vec, source_id)
        probs = torch.softmax(logits, dim=1)
        probs_acc = probs if probs_acc is None else (probs_acc + probs)

    probs_acc = probs_acc / len(versions)
    return probs_acc

@torch.no_grad()
def evaluate_full(model, loader, preproc, theta, return_gates=False, use_tta=False):
    t0 = time.time()
    model.eval()
    freeze_backbone_bn_stats(model)
    preproc.eval()

    all_y, all_p, all_loss = [], [], []
    gate_stats = []

    for x, y, _, source_id in loader:
        x = x.to(DEVICE, non_blocking=True)
        y = y.to(DEVICE, non_blocking=True)
        source_id = source_id.to(DEVICE, non_blocking=True)

        x_enh = preproc(x)
        x_res = torch.abs(x_enh - x)

        x_raw_n = normalize_imagenet(x)
        x_enh_n = normalize_imagenet(x_enh)
        x_res_n = normalize_imagenet(x_res)
        theta_vec = theta_to_vec(theta, x.size(0))

        if return_gates and (not use_tta):
            logits, _, gates = model(x_raw_n, x_enh_n, x_res_n, theta_vec, source_id, return_extra=True)
            gate_stats.append(gates.detach().cpu())
            probs = torch.softmax(logits, dim=1)
        else:
            probs = predict_probs_single_theta(model, x, source_id, preproc, theta, use_tta=use_tta)
            logits = torch.log(probs.clamp_min(1e-8))

        loss = F.cross_entropy(logits, y)

        all_loss.append(float(loss.item()))
        all_y.append(y.detach().cpu().numpy())
        all_p.append(probs.detach().cpu().numpy())

    if len(all_y) == 0:
        return {"acc": np.nan}, np.array([]), np.array([])

    y_true = np.concatenate(all_y)
    p_pred = np.concatenate(all_p)
    y_hat = np.argmax(p_pred, axis=1)

    met = {
        "loss_ce": float(np.mean(all_loss)),
        "acc": float(accuracy_score(y_true, y_hat)),
        "precision_macro": float(precision_score(y_true, y_hat, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_hat, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_hat, average="macro", zero_division=0)),
        "precision_weighted": float(precision_score(y_true, y_hat, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(y_true, y_hat, average="weighted", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_hat, average="weighted", zero_division=0)),
        "log_loss": float(log_loss(y_true, p_pred, labels=list(range(NUM_CLASSES)))),
        "eval_time_s": float(time.time() - t0),
    }
    met.update(_auc_metrics(y_true, p_pred, NUM_CLASSES))

    if return_gates and len(gate_stats) > 0:
        g = torch.cat(gate_stats, dim=0)
        met["fusion_gate_mean_raw"] = float(g[:, 0].mean().item())
        met["fusion_gate_mean_enh"] = float(g[:, 1].mean().item())
        met["fusion_gate_mean_res"] = float(g[:, 2].mean().item())
        met["fusion_gate_entropy"] = float(gate_entropy(g).mean().item())

    return met, y_true, p_pred

def train_one_epoch(model, loader, optimizer, preproc, theta, global_model=None, global_prototypes=None, scheduler=None):
    model.train()
    freeze_backbone_bn_stats(model)
    preproc.eval()

    losses, ce_losses, proto_losses, correct, total = [], [], [], 0, 0

    for x, y, _, source_id in loader:
        x = x.to(DEVICE, non_blocking=True)
        y = y.to(DEVICE, non_blocking=True)
        source_id = source_id.to(DEVICE, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        amp_enabled = (DEVICE.type == "cuda")

        with torch.amp.autocast(device_type=DEVICE.type, enabled=amp_enabled):
            x_enh = preproc(x)
            x_res = torch.abs(x_enh - x)

            x_raw_n = normalize_imagenet(x)
            x_enh_n = normalize_imagenet(x_enh)
            x_res_n = normalize_imagenet(x_res)
            theta_vec = theta_to_vec(theta, x.size(0))

            logits, embed, _ = model(x_raw_n, x_enh_n, x_res_n, theta_vec, source_id, return_extra=True)

            ce = criterion(logits, y)
            proto = prototype_alignment_loss(embed, y, global_prototypes)
            loss = ce + CFG["proto_lambda"] * proto

            if global_model is not None and CFG["fedprox_mu"] > 0:
                loss = loss + 0.5 * CFG["fedprox_mu"] * fedprox_term(model, global_model)

        if scaler is not None and amp_enabled:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if CFG["grad_clip"] > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), CFG["grad_clip"])
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if CFG["grad_clip"] > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), CFG["grad_clip"])
            optimizer.step()

        if scheduler is not None:
            scheduler.step()

        losses.append(float(loss.item()))
        ce_losses.append(float(ce.item()))
        proto_losses.append(float(proto.item()))

        preds = logits.argmax(dim=1)
        correct += int((preds == y).sum().item())
        total += int(y.size(0))

    return {
        "loss": float(np.mean(losses)) if losses else np.nan,
        "ce_loss": float(np.mean(ce_losses)) if ce_losses else np.nan,
        "proto_loss": float(np.mean(proto_losses)) if proto_losses else np.nan,
        "acc": float(correct / max(1, total)),
    }

def build_tempered_fedavg_weights(clients_subset):
    sizes = np.array([c["n_train"] for c in clients_subset], dtype=np.float64)
    sizes = np.power(np.clip(sizes, 1.0, None), CFG["fedavg_temper"])
    sizes = sizes / max(1e-12, sizes.sum())
    return sizes.tolist()

def fedavg_update(global_model, local_models, weights):
    global_sd = global_model.state_dict()
    new_sd = {}

    for key in global_sd.keys():
        ref = global_sd[key]
        if not torch.is_floating_point(ref):
            new_sd[key] = local_models[0].state_dict()[key].detach().clone()
        else:
            acc = None
            for m, w in zip(local_models, weights):
                t = m.state_dict()[key].detach().float().cpu()
                acc = t * w if acc is None else acc + t * w
            new_sd[key] = acc.to(ref.dtype)

    global_model.load_state_dict(new_sd, strict=True)

def weighted_mean(rows, key, weight_key):
    vals, ws = [], []
    for r in rows:
        v = r.get(key, np.nan)
        w = r.get(weight_key, 0)
        if np.isfinite(v) and w > 0:
            vals.append(v)
            ws.append(w)
    if len(vals) == 0:
        return np.nan
    return float(np.average(vals, weights=ws))

def aggregate_by_dataset(round_local_rows, dataset_name):
    rows = [r for r in round_local_rows if r["dataset"] == dataset_name]
    return {
        "acc": weighted_mean(rows, "val_acc", "val_size"),
        "f1_macro": weighted_mean(rows, "val_f1_macro", "val_size"),
        "precision_macro": weighted_mean(rows, "val_precision_macro", "val_size"),
        "recall_macro": weighted_mean(rows, "val_recall_macro", "val_size"),
        "log_loss": weighted_mean(rows, "val_log_loss", "val_size"),
        "loss_ce": weighted_mean(rows, "val_loss_ce", "val_size"),
        "eval_time_s": weighted_mean(rows, "val_eval_time_s", "val_size"),
    }

# ============================================================
# STEP 8: TRUE FEDERATED TRAINING (FULL PARTICIPATION)
# ============================================================
print("\n" + "=" * 118)
print("STEP 8: TRUE FEDERATED TRAINING — FULL PARTICIPATION")
print("=" * 118)

history_global = []
history_local = []

best_reward = -1.0
best_round_saved = None
best_model_state = None
best_global_prototypes = None
global_prototypes = None

t_global_start = time.time()

print(f"Clients => DS1={n_clients_ds1}, DS2={n_clients_ds2}, TOTAL={CLIENTS_TOTAL}")
print(f"Rounds: {CFG['rounds']} | Local epochs: {CFG['local_epochs']}")
print(f"FedProx μ={CFG['fedprox_mu']} | Proto λ={CFG['proto_lambda']}")
print(f"Tempered FedAvg exponent = {CFG['fedavg_temper']:.2f}")
print(f"Fixed preset DS1: {FIXED_THETA_NAME_DS1} {theta_str(FIXED_THETA_DS1)}")
print(f"Fixed preset DS2: {FIXED_THETA_NAME_DS2} {theta_str(FIXED_THETA_DS2)}")

for rnd in range(1, CFG["rounds"] + 1):
    round_t0 = time.time()
    selected_ids = list(range(len(clients)))  # full participation every round

    print("\n" + "=" * 118)
    print(f"ROUND {rnd}/{CFG['rounds']} | selected={selected_ids}")
    print("=" * 118)

    local_models = []
    proto_payloads = []
    round_local_rows = []
    selected_clients_meta = []

    for cid in selected_ids:
        client = clients[cid]
        theta_name = client["fixed_theta_name"]
        theta = client["fixed_theta"]
        preproc = theta_to_module(theta).to(DEVICE)

        local_model = ResNet50CRAF(
            num_classes=NUM_CLASSES,
            cond_dim=64,
            fuse_dim=256,
            embed_dim=256,
            pretrained=False,
        ).to(DEVICE)
        local_model.load_state_dict(global_model.state_dict(), strict=True)

        set_trainable_for_round(local_model, rnd)
        opt = make_optimizer(local_model)

        total_steps = max(1, len(client["train_loader"]) * CFG["local_epochs"])
        warmup_steps = max(1, len(client["train_loader"]) * CFG["warmup_epochs"])
        scheduler = get_cosine_schedule_with_warmup(opt, warmup_steps, total_steps)

        train_logs = []
        for _ in range(CFG["local_epochs"]):
            log_ep = train_one_epoch(
                local_model,
                client["train_loader"],
                opt,
                preproc,
                theta,
                global_model=global_model,
                global_prototypes=global_prototypes,
                scheduler=scheduler,
            )
            train_logs.append(log_ep)

        met_loc, _, _ = evaluate_full(local_model, client["val_loader"], preproc, theta, return_gates=True, use_tta=False)
        proto_payload = compute_prototype_payload(local_model, client["proto_loader"], preproc, theta)

        reward = score_metric(met_loc)

        local_models.append(local_model)
        selected_clients_meta.append(client)
        proto_payloads.append(proto_payload)

        g, a, b, t, kk, eg, mix = theta
        row = {
            "round": rnd,
            "client": f"client_{cid}",
            "dataset": client["dataset"],
            "selected": 1,
            "theta_name": theta_name,
            "theta_str": theta_str(theta),
            "gamma_power": g,
            "alpha_contrast_weight": a,
            "beta_contrast_sharpness": b,
            "tau_clip": t,
            "k_blur_kernel_size": kk,
            "edge_gain": eg,
            "blend_mix": mix,
            "train_loss": float(np.mean([x["loss"] for x in train_logs])),
            "train_ce_loss": float(np.mean([x["ce_loss"] for x in train_logs])),
            "train_proto_loss": float(np.mean([x["proto_loss"] for x in train_logs])),
            "train_acc": float(np.mean([x["acc"] for x in train_logs])),
            "val_size": client["n_val"],
            **{f"val_{k}": v for k, v in met_loc.items()},
            "reward": reward,
        }
        round_local_rows.append(row)

        auc_val = row.get("val_auc_roc_macro_ovr", np.nan)
        print(
            f"Client {cid} ({client['dataset']}) | "
            f"train_acc={row['train_acc']:.4f} | "
            f"val_acc={row['val_acc']:.4f} | "
            f"val_f1={row['val_f1_macro']:.4f} | "
            f"val_auc={auc_val:.4f} | "
            f"reward={reward:.4f} | "
            f"theta={theta_name} {theta_str(theta)}"
        )

    agg_weights = build_tempered_fedavg_weights(selected_clients_meta)
    fedavg_update(global_model, local_models, agg_weights)
    global_prototypes = aggregate_prototypes(proto_payloads, global_model.embed_dim)

    ds1_metrics = aggregate_by_dataset(round_local_rows, "ds1")
    ds2_metrics = aggregate_by_dataset(round_local_rows, "ds2")

    global_metrics = {
        "acc": weighted_mean(round_local_rows, "val_acc", "val_size"),
        "f1_macro": weighted_mean(round_local_rows, "val_f1_macro", "val_size"),
        "precision_macro": weighted_mean(round_local_rows, "val_precision_macro", "val_size"),
        "recall_macro": weighted_mean(round_local_rows, "val_recall_macro", "val_size"),
        "log_loss": weighted_mean(round_local_rows, "val_log_loss", "val_size"),
        "loss_ce": weighted_mean(round_local_rows, "val_loss_ce", "val_size"),
        "eval_time_s": weighted_mean(round_local_rows, "val_eval_time_s", "val_size"),
    }

    ds1_score = score_metric(ds1_metrics)
    ds2_score = score_metric(ds2_metrics)
    round_reward = (
        CFG["best_round_mass_ds1"] * ds1_score +
        CFG["best_round_mass_ds2"] * ds2_score +
        CFG["best_round_min_bonus"] * min(ds1_score, ds2_score)
    )

    history_local.extend(round_local_rows)
    history_global.append({
        "round": rnd,
        "round_time_s": float(time.time() - round_t0),
        "n_selected_clients": len(selected_ids),
        "active_fraction": 1.0,
        "global_reward": round_reward,
        "global_acc": global_metrics["acc"],
        "global_f1_macro": global_metrics["f1_macro"],
        "global_precision_macro": global_metrics["precision_macro"],
        "global_recall_macro": global_metrics["recall_macro"],
        "global_log_loss": global_metrics["log_loss"],
        "global_loss_ce": global_metrics["loss_ce"],
        "global_eval_time_s": global_metrics["eval_time_s"],
        "ds1_acc": ds1_metrics["acc"],
        "ds1_f1_macro": ds1_metrics["f1_macro"],
        "ds1_log_loss": ds1_metrics["log_loss"],
        "ds2_acc": ds2_metrics["acc"],
        "ds2_f1_macro": ds2_metrics["f1_macro"],
        "ds2_log_loss": ds2_metrics["log_loss"],
    })

    print(
        f"GLOBAL VAL (Round {rnd}) | "
        f"global_acc={global_metrics['acc']:.4f} | "
        f"global_f1={global_metrics['f1_macro']:.4f} | "
        f"ds1_acc={ds1_metrics['acc']:.4f} | "
        f"ds1_f1={ds1_metrics['f1_macro']:.4f} | "
        f"ds2_acc={ds2_metrics['acc']:.4f} | "
        f"ds2_f1={ds2_metrics['f1_macro']:.4f} | "
        f"reward={round_reward:.4f} | "
        f"round_time={history_global[-1]['round_time_s']:.1f}s"
    )

    if np.isfinite(round_reward) and round_reward > best_reward:
        best_reward = float(round_reward)
        best_round_saved = rnd
        best_model_state = {k: v.detach().cpu().clone() for k, v in global_model.state_dict().items()}
        best_global_prototypes = None if global_prototypes is None else {
            "proto": global_prototypes["proto"].detach().cpu().clone(),
            "mask": global_prototypes["mask"].detach().cpu().clone(),
            "counts": global_prototypes["counts"].detach().cpu().clone(),
        }

if best_model_state is not None:
    global_model.load_state_dict({k: v.to(DEVICE) for k, v in best_model_state.items()})

if best_global_prototypes is not None:
    global_prototypes = {
        "proto": best_global_prototypes["proto"].to(DEVICE),
        "mask": best_global_prototypes["mask"].to(DEVICE),
        "counts": best_global_prototypes["counts"].to(DEVICE),
    }

t_total = float(time.time() - t_global_start)
print("\n" + "=" * 118)
print(f"TRAINING COMPLETE | total_time={t_total:.1f}s | best_round={best_round_saved} | best_reward={best_reward:.4f}")
print("=" * 118)

glob_df = pd.DataFrame(history_global)
loc_df = pd.DataFrame(history_local)

print_table(glob_df, "GLOBAL per-round metrics", max_rows=20)
print_table(loc_df, "LOCAL per-client per-round metrics", max_rows=20)
add_table_to_csv(glob_df, "global_round_metrics_full")
add_table_to_csv(loc_df, "client_round_metrics_full")

# ============================================================
# STEP 9: FINAL EVALUATION (FIXED PRESETS)
# ============================================================
print("\n" + "=" * 118)
print("STEP 9: FINAL EVALUATION (FIXED PRESETS)")
print("=" * 118)

@torch.no_grad()
def evaluate_with_single_theta(model, loader, theta, use_tta=False):
    pre = theta_to_module(theta).to(DEVICE)
    return evaluate_full(model, loader, pre, theta, return_gates=False, use_tta=use_tta)

def equal_merge_metrics(m1, m2):
    out = {}
    keys = sorted(set(m1.keys()).union(set(m2.keys())))
    for k in keys:
        a, b = m1.get(k, np.nan), m2.get(k, np.nan)
        if np.isfinite(a) and np.isfinite(b):
            out[k] = float(np.mean([a, b]))
        elif np.isfinite(a):
            out[k] = float(a)
        elif np.isfinite(b):
            out[k] = float(b)
        else:
            out[k] = np.nan
    return out

val_ds1, _, _ = evaluate_with_single_theta(global_model, val_loader_ds1, FIXED_THETA_DS1, use_tta=CFG["final_use_tta"])
test_ds1, y_ds1, p_ds1 = evaluate_with_single_theta(global_model, test_loader_ds1, FIXED_THETA_DS1, use_tta=CFG["final_use_tta"])

val_ds2, _, _ = evaluate_with_single_theta(global_model, val_loader_ds2, FIXED_THETA_DS2, use_tta=CFG["final_use_tta"])
test_ds2, y_ds2, p_ds2 = evaluate_with_single_theta(global_model, test_loader_ds2, FIXED_THETA_DS2, use_tta=CFG["final_use_tta"])

val_global = equal_merge_metrics(val_ds1, val_ds2)
test_global = equal_merge_metrics(test_ds1, test_ds2)

choice_df = pd.DataFrame([
    {
        "dataset": "ds1",
        "strategy": "fixed_single",
        "theta_names": str([FIXED_THETA_NAME_DS1]),
        "score": score_metric(val_ds1),
        "val_acc": safe_float(val_ds1.get("acc")),
        "val_f1": safe_float(val_ds1.get("f1_macro")),
    },
    {
        "dataset": "ds2",
        "strategy": "fixed_single",
        "theta_names": str([FIXED_THETA_NAME_DS2]),
        "score": score_metric(val_ds2),
        "val_acc": safe_float(val_ds2.get("acc")),
        "val_f1": safe_float(val_ds2.get("f1_macro")),
    },
])
print_table(choice_df, "Fixed final preprocessing strategy", max_rows=5)
add_table_to_csv(choice_df, "final_theta_strategy_choice_fixed")

# ============================================================
# STEP 10: EXTENDED METRICS + CALIBRATION + ERROR ANALYSIS
# ============================================================
print("\n" + "=" * 118)
print("STEP 10: EXTENDED METRICS + CALIBRATION + ERROR ANALYSIS")
print("=" * 118)

def top_label_calibration(y_true, p_pred, n_bins=12):
    conf = np.max(p_pred, axis=1)
    pred = np.argmax(p_pred, axis=1)
    correct = (pred == y_true).astype(np.float32)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(conf, bins) - 1
    bin_ids = np.clip(bin_ids, 0, n_bins - 1)

    rows = []
    ece = 0.0
    mce = 0.0
    N = len(y_true)

    for b in range(n_bins):
        m = (bin_ids == b)
        if m.sum() == 0:
            rows.append({
                "bin_id": b,
                "bin_left": bins[b],
                "bin_right": bins[b + 1],
                "bin_confidence": np.nan,
                "bin_accuracy": np.nan,
                "bin_gap": np.nan,
                "bin_count": 0,
            })
            continue

        bc = float(conf[m].mean())
        ba = float(correct[m].mean())
        gap = abs(ba - bc)
        ece += (m.sum() / N) * gap
        mce = max(mce, gap)
        rows.append({
            "bin_id": b,
            "bin_left": bins[b],
            "bin_right": bins[b + 1],
            "bin_confidence": bc,
            "bin_accuracy": ba,
            "bin_gap": gap,
            "bin_count": int(m.sum()),
        })

    return pd.DataFrame(rows), float(ece), float(mce)

def multiclass_brier_score(y_true, p_pred, num_classes):
    y_oh = np.eye(num_classes)[y_true]
    return float(np.mean(np.sum((p_pred - y_oh) ** 2, axis=1)))

def classwise_confusion_metrics(y_true, p_pred):
    y_hat = np.argmax(p_pred, axis=1)
    cm = confusion_matrix(y_true, y_hat, labels=list(range(NUM_CLASSES)))
    total = cm.sum()

    rows = []
    for c, lab in enumerate(labels):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        tn = total - tp - fp - fn

        ppv = tp / max(1, (tp + fp))
        npv = tn / max(1, (tn + fn))
        sensitivity = tp / max(1, (tp + fn))
        specificity = tn / max(1, (tn + fp))
        fpr = fp / max(1, (fp + tn))
        fnr = fn / max(1, (fn + tp))
        jacc = tp / max(1, (tp + fp + fn))
        support = tp + fn
        prevalence = support / max(1, total)
        bal = 0.5 * (sensitivity + specificity)

        rows.append({
            "class_id": c,
            "class_name": lab,
            "support": int(support),
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
            "tn": int(tn),
            "prevalence": float(prevalence),
            "ppv": float(ppv),
            "npv": float(npv),
            "recall": float(sensitivity),
            "specificity": float(specificity),
            "fpr": float(fpr),
            "fnr": float(fnr),
            "jaccard": float(jacc),
            "balanced_acc": float(bal),
        })

    return cm, pd.DataFrame(rows)

def top_confusion_pairs(y_true, p_pred):
    y_hat = np.argmax(p_pred, axis=1)
    cm = confusion_matrix(y_true, y_hat, labels=list(range(NUM_CLASSES)))
    rows = []
    for i, lab_i in enumerate(labels):
        for j, lab_j in enumerate(labels):
            if i == j:
                continue
            rows.append({
                "true_class": lab_i,
                "pred_class": lab_j,
                "count": int(cm[i, j]),
            })
    return pd.DataFrame(rows).sort_values("count", ascending=False).reset_index(drop=True)

def wrong_right_confidence_table(y_true, p_pred):
    pred = np.argmax(p_pred, axis=1)
    conf = np.max(p_pred, axis=1)
    ok = pred == y_true
    return pd.DataFrame({
        "confidence": conf,
        "correct": ok.astype(int),
        "status": np.where(ok, "correct", "wrong"),
        "true_label": [id2label[int(x)] for x in y_true],
        "pred_label": [id2label[int(x)] for x in pred],
    })

def compute_extended_bundle(y_true, p_pred, dataset_name):
    y_hat = np.argmax(p_pred, axis=1)
    cm, class_df = classwise_confusion_metrics(y_true, p_pred)
    cal_df, ece, mce = top_label_calibration(y_true, p_pred, n_bins=CFG["calibration_bins"])
    conf_df = top_confusion_pairs(y_true, p_pred)
    conf_table = wrong_right_confidence_table(y_true, p_pred)

    scalar = {
        "dataset": dataset_name,
        "acc": float(accuracy_score(y_true, y_hat)),
        "balanced_acc": float(balanced_accuracy_score(y_true, y_hat)),
        "precision_macro": float(precision_score(y_true, y_hat, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_hat, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_hat, average="macro", zero_division=0)),
        "precision_weighted": float(precision_score(y_true, y_hat, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(y_true, y_hat, average="weighted", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_hat, average="weighted", zero_division=0)),
        "log_loss": float(log_loss(y_true, p_pred, labels=list(range(NUM_CLASSES)))),
        "mcc": float(matthews_corrcoef(y_true, y_hat)),
        "kappa": float(cohen_kappa_score(y_true, y_hat)),
        "jaccard_macro": float(jaccard_score(y_true, y_hat, average="macro")),
        "ppv_macro": float(class_df["ppv"].mean()),
        "npv_macro": float(class_df["npv"].mean()),
        "specificity_macro": float(class_df["specificity"].mean()),
        "fpr_macro": float(class_df["fpr"].mean()),
        "fnr_macro": float(class_df["fnr"].mean()),
        "ece": float(ece),
        "mce": float(mce),
        "brier_multi": float(multiclass_brier_score(y_true, p_pred, NUM_CLASSES)),
    }
    scalar.update(_auc_metrics(y_true, p_pred, NUM_CLASSES))
    return scalar, class_df, cal_df, conf_df, conf_table, cm

ext_ds1, class_df1, cal_df1, conf_pairs1, conf_table1, cm1 = compute_extended_bundle(y_ds1, p_ds1, "ds1_test")
ext_ds2, class_df2, cal_df2, conf_pairs2, conf_table2, cm2 = compute_extended_bundle(y_ds2, p_ds2, "ds2_test")

ext_df = pd.DataFrame([ext_ds1, ext_ds2])

print_table(ext_df, "Extended TEST metrics (DS1 vs DS2)", max_rows=5)
print_table(class_df1, "Classwise metrics — DS1 TEST", max_rows=10)
print_table(class_df2, "Classwise metrics — DS2 TEST", max_rows=10)
print_table(conf_pairs1.head(10), "Top confusion pairs — DS1 TEST", max_rows=10)
print_table(conf_pairs2.head(10), "Top confusion pairs — DS2 TEST", max_rows=10)
print_table(cal_df1, "Calibration bins — DS1 TEST", max_rows=15)
print_table(cal_df2, "Calibration bins — DS2 TEST", max_rows=15)

add_table_to_csv(ext_df, "extended_test_metrics_ds1_ds2")
add_table_to_csv(class_df1, "classwise_metrics_ds1_test")
add_table_to_csv(class_df2, "classwise_metrics_ds2_test")
add_table_to_csv(cal_df1, "calibration_bins_ds1_test")
add_table_to_csv(cal_df2, "calibration_bins_ds2_test")
add_table_to_csv(conf_pairs1, "top_confusions_ds1_test")
add_table_to_csv(conf_pairs2, "top_confusions_ds2_test")
add_table_to_csv(conf_table1, "confidence_correctness_ds1_test")
add_table_to_csv(conf_table2, "confidence_correctness_ds2_test")

test_ds1.update(ext_ds1)
test_ds2.update(ext_ds2)

def compact_metrics(m):
    keep = [
        "acc", "balanced_acc", "precision_macro", "recall_macro", "f1_macro",
        "precision_weighted", "recall_weighted", "f1_weighted", "log_loss",
        "auc_roc_macro_ovr", "mcc", "kappa", "ppv_macro", "npv_macro",
        "specificity_macro", "ece", "mce", "brier_multi", "loss_ce", "eval_time_s"
    ]
    out = {}
    for k in keep:
        if k in m and np.isfinite(m[k]):
            out[k] = float(m[k])
    return out

paper_df = pd.DataFrame([
    {"setting": "ARCF-Net NoRL FixedPreset", "split": "VAL",  "dataset": "ds1",          **compact_metrics(val_ds1)},
    {"setting": "ARCF-Net NoRL FixedPreset", "split": "VAL",  "dataset": "ds2",          **compact_metrics(val_ds2)},
    {"setting": "ARCF-Net NoRL FixedPreset", "split": "VAL",  "dataset": "global_equal", **compact_metrics(val_global)},
    {"setting": "ARCF-Net NoRL FixedPreset", "split": "TEST", "dataset": "ds1",          **compact_metrics(test_ds1)},
    {"setting": "ARCF-Net NoRL FixedPreset", "split": "TEST", "dataset": "ds2",          **compact_metrics(test_ds2)},
    {"setting": "ARCF-Net NoRL FixedPreset", "split": "TEST", "dataset": "global_equal", **compact_metrics(test_global)},
])

print_table(paper_df, "VAL + TEST tables", max_rows=10)
add_table_to_csv(paper_df, "paper_ready_metrics")

print("\nSelection summary:")
print(f"- Best round: {best_round_saved} | best_reward={best_reward:.4f}")
print(f"- DS1 fixed strategy: fixed_single | names={[FIXED_THETA_NAME_DS1]}")
print(f"- DS2 fixed strategy: fixed_single | names={[FIXED_THETA_NAME_DS2]}")

# ============================================================
# STEP 11: PREPROCESSING VALIDATION
# ============================================================
print("\n" + "=" * 118)
print("STEP 11: PREPROCESSING VALIDATION")
print("=" * 118)

@torch.no_grad()
def entropy_per_image(x01):
    gray = x01.mean(dim=1)
    B = gray.shape[0]
    ent = []
    for i in range(B):
        g = (gray[i].detach().cpu().numpy() * 255).astype(np.uint8)
        hist = np.bincount(g.flatten(), minlength=256).astype(np.float32)
        p = hist / np.clip(hist.sum(), 1, None)
        p = p[p > 0]
        ent.append(float(-(p * np.log2(p)).sum()))
    return np.array(ent)

@torch.no_grad()
def edge_energy(x01, kernel):
    kernel = kernel.to(device=x01.device, dtype=x01.dtype)
    gray = x01.mean(dim=1, keepdim=True)
    lap = F.conv2d(F.pad(gray, (1, 1, 1, 1), mode="reflect"), kernel).abs()
    return lap.mean(dim=(1, 2, 3)).detach().cpu().numpy()

@torch.no_grad()
def contrast_proxy(x01):
    gray = x01.mean(dim=1)
    return gray.std(dim=(1, 2)).detach().cpu().numpy()

@torch.no_grad()
def run_preproc_validation(frame, preproc, sample_n=400):
    n = min(sample_n, len(frame))
    if n <= 0:
        return pd.DataFrame(), pd.DataFrame()

    idx = np.random.choice(len(frame), size=n, replace=False)
    ds = MRIDataset(frame, indices=idx.tolist(), tfms=EVAL_TFMS, source_id=0)

    xs = []
    for i in range(len(ds)):
        x, _, _, _ = ds[i]
        xs.append(x)
    x = torch.stack(xs).to(DEVICE)

    x_after = preproc(x).clamp(0, 1)
    lap_kernel = preproc.lap if hasattr(preproc, "lap") else RACEFELCM().to(DEVICE).lap

    ee_before = edge_energy(x, lap_kernel)
    ee_after  = edge_energy(x_after, lap_kernel)
    ent_before = entropy_per_image(x)
    ent_after  = entropy_per_image(x_after)
    con_before = contrast_proxy(x)
    con_after  = contrast_proxy(x_after)

    dfm = pd.DataFrame({
        "edge_energy_before": ee_before,
        "edge_energy_after": ee_after,
        "entropy_before": ent_before,
        "entropy_after": ent_after,
        "contrast_before": con_before,
        "contrast_after": con_after,
        "edge_gain_ratio": (ee_after / np.clip(ee_before, 1e-9, None)),
        "entropy_delta": (ent_after - ent_before),
        "contrast_delta": (con_after - con_before),
    })
    summary = dfm.agg(["mean", "std", "min", "max"]).T.reset_index().rename(columns={"index": "metric"})
    return dfm, summary

preproc_ds1 = theta_to_module(FIXED_THETA_DS1).to(DEVICE)
preproc_ds2 = theta_to_module(FIXED_THETA_DS2).to(DEVICE)

preproc_df1, preproc_summary_df1 = run_preproc_validation(val1, preproc_ds1, CFG["preproc_val_sample_n"])
preproc_df2, preproc_summary_df2 = run_preproc_validation(val2, preproc_ds2, CFG["preproc_val_sample_n"])

print_table(preproc_summary_df1, "Preprocessing validation summary (DS1 VAL sample)", max_rows=15)
print_table(preproc_summary_df2, "Preprocessing validation summary (DS2 VAL sample)", max_rows=15)
add_table_to_csv(preproc_summary_df1, "preprocessing_validation_summary_ds1")
add_table_to_csv(preproc_summary_df2, "preprocessing_validation_summary_ds2")

# ============================================================
# STEP 12: SAVE CHECKPOINT + CSV
# ============================================================
print("\n" + "=" * 118)
print("STEP 12: SAVING CHECKPOINT + CSV")
print("=" * 118)

client_process_manifest = []
for c in clients:
    df_src = train1 if c["dataset"] == "ds1" else train2
    client_process_manifest.append({
        "gid": c["gid"],
        "local_id": c["local_id"],
        "dataset": c["dataset"],
        "source_id": c["source_id"],
        "n_train": c["n_train"],
        "n_tune": c["n_tune"],
        "n_val": c["n_val"],
        "train_indices": c["train_indices"],
        "tune_indices": c["tune_indices"],
        "val_indices": c["val_indices"],
        "train_paths": df_src.loc[c["train_indices"], "path"].tolist(),
        "tune_paths": df_src.loc[c["tune_indices"], "path"].tolist(),
        "val_paths": df_src.loc[c["val_indices"], "path"].tolist(),
        "class_counts": c["class_counts"],
        "fixed_theta_name": c["fixed_theta_name"],
        "fixed_theta": c["fixed_theta"],
    })

all_df = pd.DataFrame(ALL_ROWS)

checkpoint = {
    "method_info": METHOD_INFO,

    "state_dict": {k: v.detach().cpu() for k, v in global_model.state_dict().items()},
    "best_model_state_dict": {k: v.detach().cpu() for k, v in global_model.state_dict().items()},

    "config": CFG,
    "seed": SEED,
    "device_used": str(DEVICE),
    "backbone_name": "resnet50",
    "fusion_name": "CRAF",
    "preprocessing_name": "RACE-FELCM",

    "dataset1_raw_root": DS1_ROOT,
    "dataset2_root": DS2_ROOT,
    "labels": labels,
    "label2id": label2id,
    "id2label": id2label,
    "num_classes": NUM_CLASSES,
    "theta_fullforms": THETA_FULLFORMS,

    "full_dataset_df1": frame_to_manifest(df1),
    "full_dataset_df2": frame_to_manifest(df2),

    "split_train1": frame_to_manifest(train1),
    "split_val1": frame_to_manifest(val1),
    "split_test1": frame_to_manifest(test1),
    "split_train2": frame_to_manifest(train2),
    "split_val2": frame_to_manifest(val2),
    "split_test2": frame_to_manifest(test2),

    "fixed_clients_ds1": n_clients_ds1,
    "fixed_clients_ds2": n_clients_ds2,
    "clients_total": CLIENTS_TOTAL,
    "client_indices_ds1": client_indices_ds1,
    "client_indices_ds2": client_indices_ds2,
    "client_process_manifest": client_process_manifest,

    "preset_bank_ds1": PRESET_BANK_DS1,
    "preset_bank_ds2": PRESET_BANK_DS2,
    "fixed_theta_name_ds1": FIXED_THETA_NAME_DS1,
    "fixed_theta_name_ds2": FIXED_THETA_NAME_DS2,
    "fixed_theta_ds1": FIXED_THETA_DS1,
    "fixed_theta_ds2": FIXED_THETA_DS2,

    "best_round_saved": best_round_saved,
    "best_reward": best_reward,
    "history_global": glob_df.to_dict(orient="list"),
    "history_local": loc_df.to_dict(orient="list"),
    "total_training_time_s": t_total,

    "final_choice_ds1": {
        "strategy": "fixed_single",
        "theta_names": [FIXED_THETA_NAME_DS1],
        "theta_list": [FIXED_THETA_DS1],
        "score": score_metric(val_ds1),
        "val_acc": safe_float(val_ds1.get("acc")),
        "val_f1": safe_float(val_ds1.get("f1_macro")),
    },
    "final_choice_ds2": {
        "strategy": "fixed_single",
        "theta_names": [FIXED_THETA_NAME_DS2],
        "theta_list": [FIXED_THETA_DS2],
        "score": score_metric(val_ds2),
        "val_acc": safe_float(val_ds2.get("acc")),
        "val_f1": safe_float(val_ds2.get("f1_macro")),
    },

    "best_global_prototypes": None if best_global_prototypes is None else {
        "proto": best_global_prototypes["proto"],
        "mask": best_global_prototypes["mask"],
        "counts": best_global_prototypes["counts"],
    },

    "final_val_ds1": val_ds1,
    "final_val_ds2": val_ds2,
    "final_val_global": val_global,
    "final_test_ds1": test_ds1,
    "final_test_ds2": test_ds2,
    "final_test_global_equal": test_global,
    "paper_ready_metrics": paper_df.to_dict(orient="list"),
    "extended_test_metrics_ds1_ds2": ext_df.to_dict(orient="list"),
    "classwise_metrics_ds1_test": class_df1.to_dict(orient="list"),
    "classwise_metrics_ds2_test": class_df2.to_dict(orient="list"),
    "calibration_bins_ds1_test": cal_df1.to_dict(orient="list"),
    "calibration_bins_ds2_test": cal_df2.to_dict(orient="list"),
    "top_confusions_ds1_test": conf_pairs1.to_dict(orient="list"),
    "top_confusions_ds2_test": conf_pairs2.to_dict(orient="list"),
    "confidence_correctness_ds1_test": conf_table1.to_dict(orient="list"),
    "confidence_correctness_ds2_test": conf_table2.to_dict(orient="list"),
    "fixed_client_distribution_table": dist_df.to_dict(orient="list"),
    "global_round_metrics_table": glob_df.to_dict(orient="list"),
    "local_round_metrics_table": loc_df.to_dict(orient="list"),
    "all_tables_long_format": all_df.to_dict(orient="list"),

    "ds1_test_y_true": y_ds1,
    "ds1_test_p_pred": p_ds1,
    "ds1_test_y_hat": np.argmax(p_ds1, axis=1),
    "ds2_test_y_true": y_ds2,
    "ds2_test_p_pred": p_ds2,
    "ds2_test_y_hat": np.argmax(p_ds2, axis=1),

    "preprocessing_validation_ds1_raw": preproc_df1.to_dict(orient="list") if len(preproc_df1) else {},
    "preprocessing_validation_ds2_raw": preproc_df2.to_dict(orient="list") if len(preproc_df2) else {},
    "preprocessing_validation_summary_ds1": preproc_summary_df1.to_dict(orient="list") if len(preproc_summary_df1) else {},
    "preprocessing_validation_summary_ds2": preproc_summary_df2.to_dict(orient="list") if len(preproc_summary_df2) else {},
}

torch.save(checkpoint, MODEL_PATH)
print(f"Saved checkpoint: {MODEL_PATH}")

all_df.to_csv(CSV_PATH, index=False)
print(f"Saved CSV: {CSV_PATH}")

print("\nDONE")
print(f"Method: {METHOD_INFO['acronym']} = {METHOD_INFO['full_form']}")
print(f"Ablation: {METHOD_INFO['ablation_name']}")
print(f"Backbone: {METHOD_INFO['backbone_full_form']}")
print(f"Best round: {best_round_saved}")
print(f"Fixed clients => DS1={n_clients_ds1}, DS2={n_clients_ds2}, TOTAL={CLIENTS_TOTAL}")
print(f"Rounds completed: {CFG['rounds']}")
print(f"Global TEST acc: {safe_float(test_global.get('acc')):.4f}")
print(f"Global TEST f1_macro: {safe_float(test_global.get('f1_macro')):.4f}")
print(f"DS1 TEST acc: {safe_float(test_ds1.get('acc')):.4f}")
print(f"DS2 TEST acc: {safe_float(test_ds2.get('acc')):.4f}")
print(f"DS1 fixed strategy: fixed_single | names={[FIXED_THETA_NAME_DS1]}")
print(f"DS2 fixed strategy: fixed_single | names={[FIXED_THETA_NAME_DS2]}")
```

    ======================================================================================================================
    ARCF-Net ABLATION 1: NO RL + FIXED PRESET + FULL PARTICIPATION + NO PLOTS
    ======================================================================================================================
    ENV: KAGGLE | DEVICE: cuda | torch=2.10.0+cu128
    ======================================================================================================================
    
    ======================================================================================================================
    STEP 0: ACCESS DATASETS
    ======================================================================================================================
    Warning: Looks like you're using an outdated `kagglehub` version (installed: 0.3.13), please consider upgrading to the latest version (1.0.0).
    Downloading from https://www.kaggle.com/api/v1/datasets/download/orvile/pmram-bangladeshi-brain-cancer-mri-dataset?dataset_version_number=2...


    100%|██████████| 161M/161M [00:09<00:00, 18.0MB/s]

    Extracting files...


    


    Warning: Looks like you're using an outdated `kagglehub` version (installed: 0.3.13), please consider upgrading to the latest version (1.0.0).
    Downloading from https://www.kaggle.com/api/v1/datasets/download/yassinebazgour/preprocessed-brain-mri-scans-for-tumors-detection?dataset_version_number=1...


    100%|██████████| 130M/130M [00:07<00:00, 17.3MB/s]

    Extracting files...


    


    Dataset-1 RAW root detected:
      /root/.cache/kagglehub/datasets/orvile/pmram-bangladeshi-brain-cancer-mri-dataset/versions/2/PMRAM Bangladeshi Brain Cancer - MRI Dataset/PMRAM Bangladeshi Brain Cancer - MRI Dataset/Raw Data/Raw
    Dataset-2 root detected:
      /root/.cache/kagglehub/datasets/yassinebazgour/preprocessed-brain-mri-scans-for-tumors-detection/versions/1/preprocessed_brain_mri_dataset
    
    ======================================================================================================================
    STEP 1: BUILD DATA MANIFESTS
    ======================================================================================================================
    ds1_raw: 512Glioma -> glioma | 373 images
    ds1_raw: 512Meningioma -> meningioma | 363 images
    ds1_raw: 512Normal -> notumor | 396 images
    ds1_raw: 512Pituitary -> pituitary | 373 images
    ds2: glioma -> glioma | 1621 images
    ds2: meningioma -> meningioma | 1646 images
    ds2: notumor -> notumor | 2000 images
    ds2: pituitary -> pituitary | 1764 images
    
    Dataset-1 images: 1505
    label
    glioma        373
    meningioma    363
    notumor       396
    pituitary     373
    
    Dataset-2 images: 7031
    label
    glioma        1621
    meningioma    1646
    notumor       2000
    pituitary     1764
    
    ======================================================================================================================
    STEP 2: TRAIN / VAL / TEST SPLIT
    ======================================================================================================================
    DS1 TRAIN: 1053 | VAL: 226 | TEST: 226
    DS2 TRAIN: 4921 | VAL: 1055 | TEST: 1055
    
    ======================================================================================================================
    STEP 2.5: SANITY / LEAKAGE CHECKS
    ======================================================================================================================
    
    ----------------------------------------------------------------------------------------------------------------------
    Leakage / Sanity Summary — ds1
    ----------------------------------------------------------------------------------------------------------------------
     path_overlap_train_val  path_overlap_train_test  path_overlap_val_test  unique_paths_train  unique_paths_val  unique_paths_test  filename_overlap_train_val  filename_overlap_train_test  filename_overlap_val_test  subset_hash_train_val  subset_hash_train_test  subset_hash_val_test  subset_hash_n_train  subset_hash_n_val  subset_hash_n_test
                          0                        0                      0                1053               226                226                           0                            0                          0                      9                       4                     7                  297                224                 225
    
    ----------------------------------------------------------------------------------------------------------------------
    Leakage / Sanity Summary — ds2
    ----------------------------------------------------------------------------------------------------------------------
     path_overlap_train_val  path_overlap_train_test  path_overlap_val_test  unique_paths_train  unique_paths_val  unique_paths_test  filename_overlap_train_val  filename_overlap_train_test  filename_overlap_val_test  subset_hash_train_val  subset_hash_train_test  subset_hash_val_test  subset_hash_n_train  subset_hash_n_val  subset_hash_n_test
                          0                        0                      0                4921              1055               1055                           0                            0                          0                      2                       2                     2                  300                297                 298
    
    ======================================================================================================================
    STEP 3: FIXED NON-IID CLIENT PARTITIONING
    ======================================================================================================================
    Fixed clients for DS1: 3
    Fixed clients for DS2: 3
    
    ======================================================================================================================
    STEP 4: DATA LOADERS
    ======================================================================================================================
    ds1 | client_0 | train=490 | tune=77 | val=67
    ds1 | client_1 | train=125 | tune=20 | val=18
    ds1 | client_2 | train=198 | tune=31 | val=27
    ds2 | client_3 | train=629 | tune=98 | val=86
    ds2 | client_4 | train=527 | tune=82 | val=72
    ds2 | client_5 | train=2653 | tune=412 | val=362
    
    ----------------------------------------------------------------------------------------------------------------------
    Fixed client class distribution
    ----------------------------------------------------------------------------------------------------------------------
      client dataset  total_train  total_tune  total_val fixed_theta_name  glioma  meningioma  notumor  pituitary
    client_0     ds1          490          77         67    race_balanced     111          46      176        157
    client_1     ds1          125          20         18    race_balanced      75           4        8         38
    client_2     ds1          198          31         27    race_balanced      16         147       30          5
    client_3     ds2          629          98         86    race_balanced      12         197      416          4
    client_4     ds2          527          82         72    race_balanced     202           4      284         37
    client_5     ds2         2653         412        362    race_balanced     665         691      383        914
    Augmentation: ON
    Preprocessing: ON
    Total clients: 6
    Fixed preprocessing preset DS1: race_balanced (1.0, 0.24, 4.6, 2.4, 5, 0.1, 0.78)
    Fixed preprocessing preset DS2: race_balanced (1.0, 0.24, 4.6, 2.4, 5, 0.1, 0.78)
    
    ======================================================================================================================
    STEP 5: FIXED PREPROCESSING — RACE-FELCM
    ======================================================================================================================
    
    ======================================================================================================================
    STEP 6: MODEL — ResNet-50 + CRAF Fusion
    ======================================================================================================================
    Downloading: "https://download.pytorch.org/models/resnet50-11ad3fa6.pth" to /root/.cache/torch/hub/checkpoints/resnet50-11ad3fa6.pth


    100%|██████████| 97.8M/97.8M [00:00<00:00, 129MB/s]


    Backbone: ResNet-50 | pretrained_loaded=True
    Total params: 25,790,855
    Trainable params: 2,282,823 (8.85%)
    
    ======================================================================================================================
    STEP 7: LOSSES + PROTOTYPE SHARING
    ======================================================================================================================
    
    ======================================================================================================================
    STEP 8: TRUE FEDERATED TRAINING — FULL PARTICIPATION
    ======================================================================================================================
    Clients => DS1=3, DS2=3, TOTAL=6
    Rounds: 15 | Local epochs: 2
    FedProx μ=0.01 | Proto λ=0.12
    Tempered FedAvg exponent = 0.50
    Fixed preset DS1: race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    Fixed preset DS2: race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    
    ======================================================================================================================
    ROUND 1/15 | selected=[0, 1, 2, 3, 4, 5]
    ======================================================================================================================
    Client 0 (ds1) | train_acc=0.7245 | val_acc=0.8806 | val_f1=0.8030 | val_auc=0.9671 | reward=0.8224 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    Client 1 (ds1) | train_acc=0.6480 | val_acc=0.6111 | val_f1=0.5797 | val_auc=nan | reward=0.5876 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    Client 2 (ds1) | train_acc=0.6338 | val_acc=0.8148 | val_f1=0.6404 | val_auc=0.7895 | reward=0.6840 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    Client 3 (ds2) | train_acc=0.8068 | val_acc=0.8837 | val_f1=0.5417 | val_auc=0.9809 | reward=0.6272 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    Client 4 (ds2) | train_acc=0.7979 | val_acc=0.9444 | val_f1=0.6866 | val_auc=nan | reward=0.7510 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    Client 5 (ds2) | train_acc=0.7671 | val_acc=0.8564 | val_f1=0.8468 | val_auc=0.9810 | reward=0.8492 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    GLOBAL VAL (Round 1) | global_acc=0.8639 | global_f1=0.7659 | ds1_acc=0.8214 | ds1_f1=0.7279 | ds2_acc=0.8731 | ds2_f1=0.7741 | reward=0.8878 | round_time=166.2s
    
    ======================================================================================================================
    ROUND 2/15 | selected=[0, 1, 2, 3, 4, 5]
    ======================================================================================================================
    Client 0 (ds1) | train_acc=0.9071 | val_acc=0.8806 | val_f1=0.7584 | val_auc=0.9748 | reward=0.7890 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    Client 1 (ds1) | train_acc=0.8840 | val_acc=0.8333 | val_f1=0.7886 | val_auc=nan | reward=0.7998 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    Client 2 (ds1) | train_acc=0.8333 | val_acc=0.8519 | val_f1=0.6626 | val_auc=0.7734 | reward=0.7099 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    Client 3 (ds2) | train_acc=0.9078 | val_acc=0.9651 | val_f1=0.6508 | val_auc=0.9941 | reward=0.7294 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    Client 4 (ds2) | train_acc=0.9374 | val_acc=0.9306 | val_f1=0.6655 | val_auc=nan | reward=0.7317 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    Client 5 (ds2) | train_acc=0.8377 | val_acc=0.8923 | val_f1=0.8898 | val_auc=0.9801 | reward=0.8904 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    GLOBAL VAL (Round 2) | global_acc=0.9019 | global_f1=0.8052 | ds1_acc=0.8661 | ds1_f1=0.7402 | ds2_acc=0.9096 | ds2_f1=0.8192 | reward=0.9225 | round_time=150.9s
    
    ======================================================================================================================
    ROUND 3/15 | selected=[0, 1, 2, 3, 4, 5]
    ======================================================================================================================
    Client 0 (ds1) | train_acc=0.8031 | val_acc=0.9104 | val_f1=0.9064 | val_auc=0.9919 | reward=0.9074 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    Client 1 (ds1) | train_acc=0.7720 | val_acc=0.8889 | val_f1=0.8000 | val_auc=nan | reward=0.8222 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    Client 2 (ds1) | train_acc=0.8258 | val_acc=0.8519 | val_f1=0.6637 | val_auc=0.8685 | reward=0.7107 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    Client 3 (ds2) | train_acc=0.8816 | val_acc=0.9070 | val_f1=0.7001 | val_auc=0.9966 | reward=0.7518 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    Client 4 (ds2) | train_acc=0.9118 | val_acc=0.9444 | val_f1=0.8941 | val_auc=nan | reward=0.9067 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    Client 5 (ds2) | train_acc=0.8656 | val_acc=0.9503 | val_f1=0.9546 | val_auc=0.9953 | reward=0.9535 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    GLOBAL VAL (Round 3) | global_acc=0.9335 | global_f1=0.8911 | ds1_acc=0.8929 | ds1_f1=0.8308 | ds2_acc=0.9423 | ds2_f1=0.9041 | reward=1.0069 | round_time=288.2s
    
    ======================================================================================================================
    ROUND 4/15 | selected=[0, 1, 2, 3, 4, 5]
    ======================================================================================================================
    Client 0 (ds1) | train_acc=0.9347 | val_acc=0.9104 | val_f1=0.8870 | val_auc=0.9958 | reward=0.8929 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    Client 1 (ds1) | train_acc=0.9560 | val_acc=0.9444 | val_f1=0.9585 | val_auc=nan | reward=0.9550 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    Client 2 (ds1) | train_acc=0.9495 | val_acc=0.9259 | val_f1=0.7161 | val_auc=0.9561 | reward=0.7686 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    Client 3 (ds2) | train_acc=0.9571 | val_acc=0.9651 | val_f1=0.9793 | val_auc=0.9988 | reward=0.9758 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    Client 4 (ds2) | train_acc=0.9668 | val_acc=0.9722 | val_f1=0.9321 | val_auc=nan | reward=0.9421 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    Client 5 (ds2) | train_acc=0.9133 | val_acc=0.9724 | val_f1=0.9730 | val_auc=0.9994 | reward=0.9728 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    GLOBAL VAL (Round 4) | global_acc=0.9620 | global_f1=0.9487 | ds1_acc=0.9196 | ds1_f1=0.8573 | ds2_acc=0.9712 | ds2_f1=0.9684 | reward=1.0519 | round_time=191.6s
    
    ======================================================================================================================
    ROUND 5/15 | selected=[0, 1, 2, 3, 4, 5]
    ======================================================================================================================
    Client 0 (ds1) | train_acc=0.9327 | val_acc=0.9403 | val_f1=0.9464 | val_auc=0.9979 | reward=0.9449 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    Client 1 (ds1) | train_acc=0.9320 | val_acc=0.8889 | val_f1=0.8000 | val_auc=nan | reward=0.8222 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    Client 2 (ds1) | train_acc=0.9722 | val_acc=0.9259 | val_f1=0.6606 | val_auc=0.9829 | reward=0.7269 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    Client 3 (ds2) | train_acc=0.9714 | val_acc=0.9535 | val_f1=0.8097 | val_auc=0.9960 | reward=0.8457 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    Client 4 (ds2) | train_acc=0.9668 | val_acc=1.0000 | val_f1=1.0000 | val_auc=nan | reward=1.0000 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    Client 5 (ds2) | train_acc=0.9395 | val_acc=0.9586 | val_f1=0.9586 | val_auc=0.9976 | reward=0.9586 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    GLOBAL VAL (Round 5) | global_acc=0.9573 | global_f1=0.9245 | ds1_acc=0.9286 | ds1_f1=0.8540 | ds2_acc=0.9635 | ds2_f1=0.9397 | reward=1.0400 | round_time=168.7s
    
    ======================================================================================================================
    ROUND 6/15 | selected=[0, 1, 2, 3, 4, 5]
    ======================================================================================================================
    Client 0 (ds1) | train_acc=0.9561 | val_acc=0.9701 | val_f1=0.9720 | val_auc=0.9989 | reward=0.9715 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    Client 1 (ds1) | train_acc=0.9160 | val_acc=1.0000 | val_f1=1.0000 | val_auc=nan | reward=1.0000 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    Client 2 (ds1) | train_acc=0.9495 | val_acc=0.9259 | val_f1=0.6606 | val_auc=0.9364 | reward=0.7269 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    Client 3 (ds2) | train_acc=0.9825 | val_acc=0.9651 | val_f1=0.8977 | val_auc=0.9958 | reward=0.9145 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    Client 4 (ds2) | train_acc=0.9791 | val_acc=0.9861 | val_f1=0.7455 | val_auc=nan | reward=0.8056 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    Client 5 (ds2) | train_acc=0.9352 | val_acc=0.9586 | val_f1=0.9583 | val_auc=0.9973 | reward=0.9584 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    GLOBAL VAL (Round 6) | global_acc=0.9636 | global_f1=0.9157 | ds1_acc=0.9643 | ds1_f1=0.9014 | ds2_acc=0.9635 | ds2_f1=0.9188 | reward=1.0611 | round_time=168.9s
    
    ======================================================================================================================
    ROUND 7/15 | selected=[0, 1, 2, 3, 4, 5]
    ======================================================================================================================
    Client 0 (ds1) | train_acc=0.9745 | val_acc=0.9701 | val_f1=0.9617 | val_auc=0.9959 | reward=0.9638 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    Client 1 (ds1) | train_acc=0.9760 | val_acc=0.9444 | val_f1=0.6522 | val_auc=nan | reward=0.7252 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    Client 2 (ds1) | train_acc=0.9697 | val_acc=0.8519 | val_f1=0.5659 | val_auc=0.9236 | reward=0.6374 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    Client 3 (ds2) | train_acc=0.9777 | val_acc=0.9651 | val_f1=0.8981 | val_auc=0.9986 | reward=0.9149 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    Client 4 (ds2) | train_acc=0.9820 | val_acc=1.0000 | val_f1=1.0000 | val_auc=nan | reward=1.0000 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    Client 5 (ds2) | train_acc=0.9444 | val_acc=0.9807 | val_f1=0.9794 | val_auc=0.9991 | reward=0.9797 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    GLOBAL VAL (Round 7) | global_acc=0.9731 | global_f1=0.9418 | ds1_acc=0.9375 | ds1_f1=0.8166 | ds2_acc=0.9808 | ds2_f1=0.9688 | reward=1.0363 | round_time=169.1s
    
    ======================================================================================================================
    ROUND 8/15 | selected=[0, 1, 2, 3, 4, 5]
    ======================================================================================================================
    Client 0 (ds1) | train_acc=0.9806 | val_acc=0.9851 | val_f1=0.9861 | val_auc=1.0000 | reward=0.9859 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    Client 1 (ds1) | train_acc=0.9600 | val_acc=0.9444 | val_f1=0.9552 | val_auc=nan | reward=0.9525 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    Client 2 (ds1) | train_acc=0.9697 | val_acc=0.8519 | val_f1=0.5859 | val_auc=0.9735 | reward=0.6524 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    Client 3 (ds2) | train_acc=0.9738 | val_acc=0.9535 | val_f1=0.6431 | val_auc=0.9986 | reward=0.7207 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    Client 4 (ds2) | train_acc=0.9877 | val_acc=0.9861 | val_f1=0.7455 | val_auc=nan | reward=0.8056 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    Client 5 (ds2) | train_acc=0.9534 | val_acc=0.9807 | val_f1=0.9796 | val_auc=0.9996 | reward=0.9799 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    GLOBAL VAL (Round 8) | global_acc=0.9715 | global_f1=0.8903 | ds1_acc=0.9464 | ds1_f1=0.8847 | ds2_acc=0.9769 | ds2_f1=0.8915 | reward=1.0415 | round_time=170.3s
    
    ======================================================================================================================
    ROUND 9/15 | selected=[0, 1, 2, 3, 4, 5]
    ======================================================================================================================
    Client 0 (ds1) | train_acc=0.9694 | val_acc=0.9701 | val_f1=0.9616 | val_auc=0.9993 | reward=0.9637 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    Client 1 (ds1) | train_acc=0.9720 | val_acc=0.9444 | val_f1=0.9585 | val_auc=nan | reward=0.9550 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    Client 2 (ds1) | train_acc=0.9621 | val_acc=0.9259 | val_f1=0.8764 | val_auc=0.9882 | reward=0.8888 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    Client 3 (ds2) | train_acc=0.9706 | val_acc=0.9767 | val_f1=0.8654 | val_auc=0.9891 | reward=0.8932 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    Client 4 (ds2) | train_acc=0.9791 | val_acc=0.9861 | val_f1=0.7455 | val_auc=nan | reward=0.8056 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    Client 5 (ds2) | train_acc=0.9521 | val_acc=0.9779 | val_f1=0.9787 | val_auc=0.9993 | reward=0.9785 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    GLOBAL VAL (Round 9) | global_acc=0.9747 | global_f1=0.9300 | ds1_acc=0.9554 | ds1_f1=0.9405 | ds2_acc=0.9788 | ds2_f1=0.9277 | reward=1.0834 | round_time=167.2s
    
    ======================================================================================================================
    ROUND 10/15 | selected=[0, 1, 2, 3, 4, 5]
    ======================================================================================================================
    Client 0 (ds1) | train_acc=0.9765 | val_acc=0.9701 | val_f1=0.9725 | val_auc=0.9976 | reward=0.9719 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    Client 1 (ds1) | train_acc=0.9640 | val_acc=0.9444 | val_f1=0.9585 | val_auc=nan | reward=0.9550 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    Client 2 (ds1) | train_acc=0.9621 | val_acc=0.9259 | val_f1=0.7381 | val_auc=0.9625 | reward=0.7851 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    Client 3 (ds2) | train_acc=0.9825 | val_acc=0.9884 | val_f1=0.9119 | val_auc=0.9997 | reward=0.9311 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    Client 4 (ds2) | train_acc=0.9839 | val_acc=0.9861 | val_f1=0.9636 | val_auc=nan | reward=0.9693 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    Client 5 (ds2) | train_acc=0.9612 | val_acc=0.9751 | val_f1=0.9741 | val_auc=0.9981 | reward=0.9744 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    GLOBAL VAL (Round 10) | global_acc=0.9747 | global_f1=0.9538 | ds1_acc=0.9554 | ds1_f1=0.9137 | ds2_acc=0.9788 | ds2_f1=0.9624 | reward=1.0839 | round_time=168.0s
    
    ======================================================================================================================
    ROUND 11/15 | selected=[0, 1, 2, 3, 4, 5]
    ======================================================================================================================
    Client 0 (ds1) | train_acc=0.9745 | val_acc=0.9552 | val_f1=0.9442 | val_auc=0.9992 | reward=0.9470 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    Client 1 (ds1) | train_acc=0.9680 | val_acc=0.8333 | val_f1=0.6080 | val_auc=nan | reward=0.6643 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    Client 2 (ds1) | train_acc=0.9697 | val_acc=0.8889 | val_f1=0.6190 | val_auc=0.9679 | reward=0.6865 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    Client 3 (ds2) | train_acc=0.9905 | val_acc=0.9767 | val_f1=0.9049 | val_auc=0.9966 | reward=0.9228 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    Client 4 (ds2) | train_acc=0.9924 | val_acc=1.0000 | val_f1=1.0000 | val_auc=nan | reward=1.0000 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    Client 5 (ds2) | train_acc=0.9715 | val_acc=0.9834 | val_f1=0.9849 | val_auc=0.9995 | reward=0.9845 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    GLOBAL VAL (Round 11) | global_acc=0.9731 | global_f1=0.9450 | ds1_acc=0.9196 | ds1_f1=0.8118 | ds2_acc=0.9846 | ds2_f1=0.9737 | reward=1.0334 | round_time=168.4s
    
    ======================================================================================================================
    ROUND 12/15 | selected=[0, 1, 2, 3, 4, 5]
    ======================================================================================================================
    Client 0 (ds1) | train_acc=0.9755 | val_acc=0.9701 | val_f1=0.9578 | val_auc=0.9988 | reward=0.9609 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    Client 1 (ds1) | train_acc=0.9600 | val_acc=1.0000 | val_f1=1.0000 | val_auc=nan | reward=1.0000 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    Client 2 (ds1) | train_acc=0.9773 | val_acc=0.8889 | val_f1=0.6968 | val_auc=0.9946 | reward=0.7449 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    Client 3 (ds2) | train_acc=0.9722 | val_acc=0.9884 | val_f1=0.9119 | val_auc=1.0000 | reward=0.9311 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    Client 4 (ds2) | train_acc=0.9848 | val_acc=0.9722 | val_f1=0.7378 | val_auc=nan | reward=0.7964 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    Client 5 (ds2) | train_acc=0.9695 | val_acc=0.9917 | val_f1=0.9922 | val_auc=0.9997 | reward=0.9921 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    GLOBAL VAL (Round 12) | global_acc=0.9826 | global_f1=0.9363 | ds1_acc=0.9554 | ds1_f1=0.9017 | ds2_acc=0.9885 | ds2_f1=0.9437 | reward=1.0723 | round_time=168.3s
    
    ======================================================================================================================
    ROUND 13/15 | selected=[0, 1, 2, 3, 4, 5]
    ======================================================================================================================
    Client 0 (ds1) | train_acc=0.9735 | val_acc=0.9701 | val_f1=0.9725 | val_auc=0.9906 | reward=0.9719 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    Client 1 (ds1) | train_acc=0.9880 | val_acc=0.9444 | val_f1=0.7381 | val_auc=nan | reward=0.7897 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    Client 2 (ds1) | train_acc=0.9798 | val_acc=0.9259 | val_f1=0.7381 | val_auc=0.9932 | reward=0.7851 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    Client 3 (ds2) | train_acc=0.9873 | val_acc=0.9767 | val_f1=0.9052 | val_auc=0.9997 | reward=0.9231 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    Client 4 (ds2) | train_acc=0.9801 | val_acc=0.9861 | val_f1=0.7455 | val_auc=nan | reward=0.8056 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    Client 5 (ds2) | train_acc=0.9680 | val_acc=0.9890 | val_f1=0.9899 | val_auc=0.9997 | reward=0.9896 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    GLOBAL VAL (Round 13) | global_acc=0.9810 | global_f1=0.9307 | ds1_acc=0.9554 | ds1_f1=0.8783 | ds2_acc=0.9865 | ds2_f1=0.9420 | reward=1.0600 | round_time=168.1s
    
    ======================================================================================================================
    ROUND 14/15 | selected=[0, 1, 2, 3, 4, 5]
    ======================================================================================================================
    Client 0 (ds1) | train_acc=0.9755 | val_acc=0.9851 | val_f1=0.9717 | val_auc=1.0000 | reward=0.9751 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    Client 1 (ds1) | train_acc=0.9720 | val_acc=0.9444 | val_f1=0.6410 | val_auc=nan | reward=0.7169 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    Client 2 (ds1) | train_acc=0.9848 | val_acc=0.9259 | val_f1=0.7161 | val_auc=0.8882 | reward=0.7686 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    Client 3 (ds2) | train_acc=0.9905 | val_acc=0.9884 | val_f1=0.9119 | val_auc=0.9911 | reward=0.9311 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    Client 4 (ds2) | train_acc=0.9877 | val_acc=0.9861 | val_f1=0.9898 | val_auc=nan | reward=0.9889 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    Client 5 (ds2) | train_acc=0.9751 | val_acc=0.9917 | val_f1=0.9922 | val_auc=0.9993 | reward=0.9921 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    GLOBAL VAL (Round 14) | global_acc=0.9858 | global_f1=0.9570 | ds1_acc=0.9643 | ds1_f1=0.8570 | ds2_acc=0.9904 | ds2_f1=0.9786 | reward=1.0652 | round_time=167.3s
    
    ======================================================================================================================
    ROUND 15/15 | selected=[0, 1, 2, 3, 4, 5]
    ======================================================================================================================
    Client 0 (ds1) | train_acc=0.9837 | val_acc=0.9254 | val_f1=0.9340 | val_auc=0.9967 | reward=0.9318 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    Client 1 (ds1) | train_acc=0.9760 | val_acc=0.9444 | val_f1=0.7381 | val_auc=nan | reward=0.7897 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    Client 2 (ds1) | train_acc=0.9823 | val_acc=0.9630 | val_f1=0.9106 | val_auc=0.9811 | reward=0.9237 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    Client 3 (ds2) | train_acc=0.9857 | val_acc=0.9535 | val_f1=0.6453 | val_auc=0.9957 | reward=0.7223 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    Client 4 (ds2) | train_acc=0.9886 | val_acc=0.9861 | val_f1=0.7455 | val_auc=nan | reward=0.8056 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    Client 5 (ds2) | train_acc=0.9791 | val_acc=0.9890 | val_f1=0.9892 | val_auc=0.9997 | reward=0.9891 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    GLOBAL VAL (Round 15) | global_acc=0.9747 | global_f1=0.8983 | ds1_acc=0.9375 | ds1_f1=0.8968 | ds2_acc=0.9827 | ds2_f1=0.8986 | reward=1.0494 | round_time=167.7s
    
    ======================================================================================================================
    TRAINING COMPLETE | total_time=2649.6s | best_round=10 | best_reward=1.0839
    ======================================================================================================================
    
    ----------------------------------------------------------------------------------------------------------------------
    GLOBAL per-round metrics
    ----------------------------------------------------------------------------------------------------------------------
     round  round_time_s  n_selected_clients  active_fraction  global_reward  global_acc  global_f1_macro  global_precision_macro  global_recall_macro  global_log_loss  global_loss_ce  global_eval_time_s  ds1_acc  ds1_f1_macro  ds1_log_loss  ds2_acc  ds2_f1_macro  ds2_log_loss
         1    166.248067                   6              1.0       0.887771    0.863924         0.765939                0.780114             0.801591         0.347819        0.348143            2.978949 0.821429      0.727908      0.551751 0.873077      0.774131      0.303896
         2    150.865474                   6              1.0       0.922469    0.901899         0.805196                0.811304             0.829208         0.316019        0.307048            2.372459 0.866071      0.740163      0.444448 0.909615      0.819203      0.288357
         3    288.233470                   6              1.0       1.006945    0.933544         0.891129                0.884197             0.933345         0.204036        0.202803            2.481749 0.892857      0.830816      0.326720 0.942308      0.904120      0.177612
         4    191.647808                   6              1.0       1.051899    0.962025         0.948674                0.949499             0.952988         0.122068        0.119416            2.372016 0.919643      0.857291      0.249291 0.971154      0.968357      0.094666
         5    168.734538                   6              1.0       1.040024    0.957278         0.924498                0.913737             0.952374         0.123385        0.121803            2.351543 0.928571      0.853983      0.188579 0.963462      0.939686      0.109344
         6    168.895574                   6              1.0       1.061121    0.963608         0.915733                0.908779             0.930627         0.136920        0.132327            2.357472 0.964286      0.901405      0.141356 0.963462      0.918819      0.135965
         7    169.118992                   6              1.0       1.036315    0.973101         0.941829                0.937848             0.955701         0.111866        0.106314            2.348202 0.937500      0.816556      0.230942 0.980769      0.968811      0.086219
         8    170.321199                   6              1.0       1.041505    0.971519         0.890307                0.885851             0.902807         0.100548        0.102219            2.359793 0.946429      0.884664      0.189267 0.976923      0.891522      0.081439
         9    167.239279                   6              1.0       1.083437    0.974684         0.929973                0.927047             0.946926         0.102940        0.094295            2.363233 0.955357      0.940541      0.172562 0.978846      0.927697      0.087944
        10    168.035895                   6              1.0       1.083942    0.974684         0.953777                0.947416             0.967059         0.094809        0.091763            2.353128 0.955357      0.913722      0.197432 0.978846      0.962405      0.072706
        11    168.358597                   6              1.0       1.033417    0.973101         0.945045                0.940195             0.960184         0.089613        0.082992            2.356244 0.919643      0.811780      0.226467 0.984615      0.973748      0.060136
        12    168.314303                   6              1.0       1.072272    0.982595         0.936263                0.934307             0.944869         0.078158        0.076781            2.361562 0.955357      0.901700      0.169400 0.988462      0.943707      0.058506
        13    168.101901                   6              1.0       1.059992    0.981013         0.930728                0.925992             0.941644         0.073084        0.068038            2.356920 0.955357      0.878302      0.142025 0.986538      0.942019      0.058236
        14    167.304343                   6              1.0       1.065234    0.985759         0.957042                0.952103             0.968505         0.075040        0.072494            2.391140 0.964286      0.856954      0.142645 0.990385      0.978599      0.060479
        15    167.668415                   6              1.0       1.049353    0.974684         0.898256                0.898095             0.907478         0.100865        0.129417            2.365908 0.937500      0.896847      0.243316 0.982692      0.898559      0.070184
    
    ----------------------------------------------------------------------------------------------------------------------
    LOCAL per-client per-round metrics
    ----------------------------------------------------------------------------------------------------------------------
     round   client dataset  selected    theta_name                                                theta_str  gamma_power  alpha_contrast_weight  beta_contrast_sharpness  tau_clip  k_blur_kernel_size  edge_gain  blend_mix  train_loss  train_ce_loss  train_proto_loss  train_acc  val_size  val_loss_ce  val_acc  val_precision_macro  val_recall_macro  val_f1_macro  val_precision_weighted  val_recall_weighted  val_f1_weighted  val_log_loss  val_eval_time_s  val_auc_roc_macro_ovr  val_auc_class_0  val_auc_class_1  val_auc_class_2  val_auc_class_3  val_fusion_gate_mean_raw  val_fusion_gate_mean_enh  val_fusion_gate_mean_res  val_fusion_gate_entropy   reward
         1 client_0     ds1         1 race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)          1.0                   0.24                      4.6       2.4                   5        0.1       0.78    0.454737       0.393591          0.000000   0.724490        67     0.431130 0.880597             0.911239          0.783523      0.802999                0.896526             0.880597         0.870271      0.380659         2.363425               0.967058         0.934615         0.959016         0.979651         0.994949                  0.413173                  0.317641                  0.269186                 1.512410 0.822399
         1 client_1     ds1         1 race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)          1.0                   0.24                      4.6       2.4                   5        0.1       0.78    0.534149       0.510898          0.000000   0.648000        18     0.707444 0.611111             0.626263          0.787879      0.579739                0.811448             0.611111         0.588998      0.894574         0.822940                    NaN         0.922078              NaN         1.000000         0.888889                  0.395573                  0.360594                  0.243833                 1.493129 0.587582
         1 client_2     ds1         1 race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)          1.0                   0.24                      4.6       2.4                   5        0.1       0.78    0.575322       0.540095          0.000000   0.633838        27     0.758031 0.814815             0.611111          0.712500      0.640351                0.810700             0.814815         0.798571      0.747761         1.523519               0.789550         0.320000         0.914286         0.923913         1.000000                  0.383915                  0.369588                  0.246497                 1.551428 0.683967
         1 client_3     ds2         1 race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)          1.0                   0.24                      4.6       2.4                   5        0.1       0.78    0.355141       0.296799          0.000000   0.806836        86     0.285178 0.883721             0.506059          0.691033      0.541721                0.902190             0.883721         0.889180      0.267388         0.930342               0.980852         1.000000         0.954802         0.992136         0.976471                  0.465086                  0.496764                  0.038150                 1.136528 0.627221
         1 client_4     ds2         1 race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)          1.0                   0.24                      4.6       2.4                   5        0.1       0.78    0.371067       0.308798          0.000000   0.797913        72     0.186594 0.944444             0.668956          0.716804      0.686568                0.965201             0.944444         0.952585      0.199169         1.484894                    NaN         0.985390              NaN         0.998446         0.997015                  0.273137                  0.459998                  0.266865                 1.433646 0.751037
         1 client_5     ds2         1 race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)          1.0                   0.24                      4.6       2.4                   5        0.1       0.78    0.456689       0.352562          0.000000   0.767056       362     0.331436 0.856354             0.863317          0.855391      0.846760                0.864545             0.856354         0.847401      0.333398         4.092477               0.980992         0.980049         0.951493         0.997457         0.994970                  0.564599                  0.352577                  0.082823                 1.275153 0.849158
         2 client_0     ds1         1 race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)          1.0                   0.24                      4.6       2.4                   5        0.1       0.78    0.220292       0.152218          0.269994   0.907143        67     0.336328 0.880597             0.917026          0.753220      0.758402                0.897903             0.880597         0.859421      0.369626         0.796569               0.974824         0.964103         0.950820         0.987403         0.996970                  0.515118                  0.331081                  0.153801                 1.417370 0.788951
         2 client_1     ds1         1 race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)          1.0                   0.24                      4.6       2.4                   5        0.1       0.78    0.223370       0.166005          0.330406   0.884000        18     0.247214 0.833333             0.750000          0.909091      0.788638                0.888889             0.833333         0.837371      0.385953         0.289529                    NaN         1.000000              NaN         1.000000         0.986111                  0.457295                  0.321492                  0.221212                 1.514007 0.799812
         2 client_2     ds1         1 race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)          1.0                   0.24                      4.6       2.4                   5        0.1       0.78    0.307833       0.237997          0.356882   0.833333        27     0.680009 0.851852             0.629699          0.725000      0.662587                0.823448             0.851852         0.828542      0.669113         0.383322               0.773416         0.180000         0.957143         0.956522         1.000000                  0.371219                  0.360779                  0.268002                 1.562437 0.709904
         2 client_3     ds2         1 race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)          1.0                   0.24                      4.6       2.4                   5        0.1       0.78    0.215811       0.146047          0.246685   0.907790        86     0.143652 0.965116             0.607143          0.736355      0.650818                0.960133             0.965116         0.961506      0.138407         0.902314               0.994106         1.000000         0.991212         0.996975         0.988235                  0.604194                  0.331807                  0.063998                 1.192353 0.729393
         2 client_4     ds2         1 race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)          1.0                   0.24                      4.6       2.4                   5        0.1       0.78    0.167795       0.104470          0.245595   0.937381        72     0.175444 0.930556             0.646250          0.707875      0.665476                0.958403             0.930556         0.940252      0.191208         0.789386                    NaN         0.987825              NaN         1.000000         0.997015                  0.431171                  0.375538                  0.193291                 1.503780 0.731746
         2 client_5     ds2         1 race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)          1.0                   0.24                      4.6       2.4                   5        0.1       0.78    0.344170       0.242579          0.276068   0.837731       362     0.341780 0.892265             0.889660          0.893264      0.889781                0.894248             0.892265         0.891531      0.343303         3.580188               0.980056         0.985159         0.947483         0.992680         0.994903                  0.494812                  0.445970                  0.059218                 1.253848 0.890402
         3 client_0     ds1         1 race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)          1.0                   0.24                      4.6       2.4                   5        0.1       0.78    0.465154       0.356540          0.322513   0.803061        67     0.285136 0.910448             0.928571          0.903409      0.906444                0.936034             0.910448         0.914945      0.269796         0.911332               0.991887         0.994872         0.972678         1.000000         1.000000                  0.729181                  0.161304                  0.109515                 1.083339 0.907445
         3 client_1     ds1         1 race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)          1.0                   0.24                      4.6       2.4                   5        0.1       0.78    0.455491       0.396222          0.308163   0.772000        18     0.362842 0.888889             0.777778          0.939394      0.800000                0.962963             0.888889         0.911111      0.412891         0.274035                    NaN         0.961039              NaN         1.000000         1.000000                  0.455135                  0.396171                  0.148694                 1.453705 0.822222
         3 client_2     ds1         1 race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)          1.0                   0.24                      4.6       2.4                   5        0.1       0.78    0.327441       0.260031          0.272490   0.825758        27     0.401999 0.851852             0.653409          0.675000      0.663690                0.787879             0.851852         0.818342      0.410527         0.475097               0.868494         0.660000         0.835714         0.978261         1.000000                  0.496874                  0.227704                  0.275422                 1.473582 0.710731
         3 client_3     ds2         1 race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)          1.0                   0.24                      4.6       2.4                   5        0.1       0.78    0.282352       0.192176          0.234889   0.881558        86     0.245728 0.906977             0.662500          0.925926      0.700111                0.951744             0.906977         0.916777      0.232501         0.907100               0.996582         1.000000         0.989956         0.996370         1.000000                  0.852472                  0.111449                  0.036079                 0.712288 0.751828
         3 client_4     ds2         1 race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)          1.0                   0.24                      4.6       2.4                   5        0.1       0.78    0.259786       0.179563          0.228143   0.911765        72     0.149908 0.944444             0.862179          0.955739      0.894057                0.959001             0.944444         0.948133      0.164970         0.846987                    NaN         0.995130              NaN         1.000000         0.988060                  0.746580                  0.205786                  0.047634                 0.990651 0.906654
         3 client_5     ds2         1 race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)          1.0                   0.24                      4.6       2.4                   5        0.1       0.78    0.378518       0.231466          0.239594   0.865624       362     0.165073 0.950276             0.955537          0.955162      0.954587                0.951976             0.950276         0.950346      0.167087         3.731084               0.995295         0.995742         0.989759         1.000000         0.995679                  0.739285                  0.231357                  0.029358                 0.948604 0.953509
         4 client_0     ds1         1 race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)          1.0                   0.24                      4.6       2.4                   5        0.1       0.78    0.194789       0.132459          0.194332   0.934694        67     0.244208 0.910448             0.924641          0.873106      0.886991                0.923016             0.910448         0.910326      0.263498         0.909052               0.995830         0.996154         0.997268         1.000000         0.989899                  0.545739                  0.337393                  0.116868                 1.351546 0.892855
         4 client_1     ds1         1 race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)          1.0                   0.24                      4.6       2.4                   5        0.1       0.78    0.146019       0.103350          0.210151   0.956000        18     0.089032 0.944444             0.952381          0.969697      0.958486                0.952381             0.944444         0.945258      0.115536         0.427773                    NaN         1.000000              NaN         1.000000         1.000000                  0.672358                  0.273107                  0.054534                 1.124347 0.954976
    ... showing first 20 of 90 rows
    
    ======================================================================================================================
    STEP 9: FINAL EVALUATION (FIXED PRESETS)
    ======================================================================================================================
    
    ----------------------------------------------------------------------------------------------------------------------
    Fixed final preprocessing strategy
    ----------------------------------------------------------------------------------------------------------------------
    dataset     strategy       theta_names    score  val_acc   val_f1
        ds1 fixed_single ['race_balanced'] 0.964259 0.964602 0.964144
        ds2 fixed_single ['race_balanced'] 0.976431 0.977251 0.976158
    
    ======================================================================================================================
    STEP 10: EXTENDED METRICS + CALIBRATION + ERROR ANALYSIS
    ======================================================================================================================
    
    ----------------------------------------------------------------------------------------------------------------------
    Extended TEST metrics (DS1 vs DS2)
    ----------------------------------------------------------------------------------------------------------------------
     dataset      acc  balanced_acc  precision_macro  recall_macro  f1_macro  precision_weighted  recall_weighted  f1_weighted  log_loss      mcc    kappa  jaccard_macro  ppv_macro  npv_macro  specificity_macro  fpr_macro  fnr_macro      ece      mce  brier_multi  auc_roc_macro_ovr  auc_class_0  auc_class_1  auc_class_2  auc_class_3
    ds1_test 0.982301      0.981899         0.982748      0.981899  0.982040            0.982680         0.982301     0.982211  0.059833 0.976571 0.976392       0.964804   0.982748   0.994219           0.994091   0.005909   0.018101 0.019583 0.301542     0.026459           0.999790     0.999580     0.999787     0.999899     0.999895
    ds2_test 0.973460      0.971968         0.973139      0.971968  0.972009            0.974001         0.973460     0.973214  0.078518 0.964838 0.964522       0.946064   0.973139   0.991442           0.991234   0.008766   0.028032 0.021434 0.314444     0.036582           0.999412     0.999353     0.998873     0.999792     0.999631
    
    ----------------------------------------------------------------------------------------------------------------------
    Classwise metrics — DS1 TEST
    ----------------------------------------------------------------------------------------------------------------------
     class_id class_name  support  tp  fp  fn  tn  prevalence      ppv      npv   recall  specificity      fpr      fnr  jaccard  balanced_acc
            0     glioma       56  55   1   1 169    0.247788 0.982143 0.994118 0.982143     0.994118 0.005882 0.017857 0.964912      0.988130
            1 meningioma       55  52   0   3 171    0.243363 1.000000 0.982759 0.945455     1.000000 0.000000 0.054545 0.945455      0.972727
            2    notumor       59  59   1   0 166    0.261062 0.983333 1.000000 1.000000     0.994012 0.005988 0.000000 0.983333      0.997006
            3  pituitary       56  56   2   0 168    0.247788 0.965517 1.000000 1.000000     0.988235 0.011765 0.000000 0.965517      0.994118
    
    ----------------------------------------------------------------------------------------------------------------------
    Classwise metrics — DS2 TEST
    ----------------------------------------------------------------------------------------------------------------------
     class_id class_name  support  tp  fp  fn  tn  prevalence      ppv      npv   recall  specificity      fpr      fnr  jaccard  balanced_acc
            0     glioma      244 240  12   4 799    0.231280 0.952381 0.995019 0.983607     0.985203 0.014797 0.016393 0.937500      0.984405
            1 meningioma      247 225   3  22 805    0.234123 0.986842 0.973398 0.910931     0.996287 0.003713 0.089069 0.900000      0.953609
            2    notumor      300 298   2   2 753    0.284360 0.993333 0.997351 0.993333     0.997351 0.002649 0.006667 0.986755      0.995342
            3  pituitary      264 264  11   0 780    0.250237 0.960000 1.000000 1.000000     0.986094 0.013906 0.000000 0.960000      0.993047
    
    ----------------------------------------------------------------------------------------------------------------------
    Top confusion pairs — DS1 TEST
    ----------------------------------------------------------------------------------------------------------------------
    true_class pred_class  count
    meningioma  pituitary      2
        glioma    notumor      1
    meningioma     glioma      1
        glioma meningioma      0
        glioma  pituitary      0
    meningioma    notumor      0
       notumor     glioma      0
       notumor meningioma      0
       notumor  pituitary      0
     pituitary     glioma      0
    
    ----------------------------------------------------------------------------------------------------------------------
    Top confusion pairs — DS2 TEST
    ----------------------------------------------------------------------------------------------------------------------
    true_class pred_class  count
    meningioma     glioma     12
    meningioma  pituitary      8
        glioma meningioma      3
    meningioma    notumor      2
       notumor  pituitary      2
        glioma  pituitary      1
        glioma    notumor      0
       notumor     glioma      0
       notumor meningioma      0
     pituitary     glioma      0
    
    ----------------------------------------------------------------------------------------------------------------------
    Calibration bins — DS1 TEST
    ----------------------------------------------------------------------------------------------------------------------
     bin_id  bin_left  bin_right  bin_confidence  bin_accuracy  bin_gap  bin_count
          0  0.000000   0.083333             NaN           NaN      NaN          0
          1  0.083333   0.166667             NaN           NaN      NaN          0
          2  0.166667   0.250000             NaN           NaN      NaN          0
          3  0.250000   0.333333             NaN           NaN      NaN          0
          4  0.333333   0.416667             NaN           NaN      NaN          0
          5  0.416667   0.500000             NaN           NaN      NaN          0
          6  0.500000   0.583333             NaN           NaN      NaN          0
          7  0.583333   0.666667        0.598813      0.500000 0.098813          2
          8  0.666667   0.750000        0.698458      1.000000 0.301542          1
          9  0.750000   0.833333        0.789172      0.666667 0.122505          3
         10  0.833333   0.916667        0.888445      0.750000 0.138445          4
         11  0.916667   1.000000        0.981457      0.995370 0.013913        216
    
    ----------------------------------------------------------------------------------------------------------------------
    Calibration bins — DS2 TEST
    ----------------------------------------------------------------------------------------------------------------------
     bin_id  bin_left  bin_right  bin_confidence  bin_accuracy  bin_gap  bin_count
          0  0.000000   0.083333             NaN           NaN      NaN          0
          1  0.083333   0.166667             NaN           NaN      NaN          0
          2  0.166667   0.250000             NaN           NaN      NaN          0
          3  0.250000   0.333333             NaN           NaN      NaN          0
          4  0.333333   0.416667             NaN           NaN      NaN          0
          5  0.416667   0.500000        0.480794      0.500000 0.019206          2
          6  0.500000   0.583333        0.545139      0.375000 0.170139          8
          7  0.583333   0.666667        0.623828      0.545455 0.078373         11
          8  0.666667   0.750000        0.714444      0.400000 0.314444         10
          9  0.750000   0.833333        0.803286      0.600000 0.203286          5
         10  0.833333   0.916667        0.900675      0.727273 0.173402         11
         11  0.916667   1.000000        0.979878      0.994048 0.014170       1008
    
    ----------------------------------------------------------------------------------------------------------------------
    VAL + TEST tables
    ----------------------------------------------------------------------------------------------------------------------
                      setting split      dataset      acc  precision_macro  recall_macro  f1_macro  precision_weighted  recall_weighted  f1_weighted  log_loss  auc_roc_macro_ovr  loss_ce  eval_time_s  balanced_acc      mcc    kappa  ppv_macro  npv_macro  specificity_macro      ece      mce  brier_multi
    ARCF-Net NoRL FixedPreset   VAL          ds1 0.964602         0.964879      0.963790  0.964144            0.964789         0.964602     0.964506  0.110216           0.998448 0.215577     4.705351           NaN      NaN      NaN        NaN        NaN                NaN      NaN      NaN          NaN
    ARCF-Net NoRL FixedPreset   VAL          ds2 0.977251         0.976750      0.976146  0.976158            0.977460         0.977251     0.977084  0.087171           0.998338 0.087135    21.027271           NaN      NaN      NaN        NaN        NaN                NaN      NaN      NaN          NaN
    ARCF-Net NoRL FixedPreset   VAL global_equal 0.970926         0.970814      0.969968  0.970151            0.971124         0.970926     0.970795  0.098693           0.998393 0.151356    12.866311           NaN      NaN      NaN        NaN        NaN                NaN      NaN      NaN          NaN
    ARCF-Net NoRL FixedPreset  TEST          ds1 0.982301         0.982748      0.981899  0.982040            0.982680         0.982301     0.982211  0.059833           0.999790 0.057353     4.692175      0.981899 0.976571 0.976392   0.982748   0.994219           0.994091 0.019583 0.301542     0.026459
    ARCF-Net NoRL FixedPreset  TEST          ds2 0.973460         0.973139      0.971968  0.972009            0.974001         0.973460     0.973214  0.078518           0.999412 0.078468    20.924420      0.971968 0.964838 0.964522   0.973139   0.991442           0.991234 0.021434 0.314444     0.036582
    ARCF-Net NoRL FixedPreset  TEST global_equal 0.977880         0.977944      0.976934  0.977024            0.978340         0.977880     0.977713  0.069176           0.999601 0.067911    12.808298           NaN      NaN      NaN        NaN        NaN                NaN      NaN      NaN          NaN
    
    Selection summary:
    - Best round: 10 | best_reward=1.0839
    - DS1 fixed strategy: fixed_single | names=['race_balanced']
    - DS2 fixed strategy: fixed_single | names=['race_balanced']
    
    ======================================================================================================================
    STEP 11: PREPROCESSING VALIDATION
    ======================================================================================================================
    
    ----------------------------------------------------------------------------------------------------------------------
    Preprocessing validation summary (DS1 VAL sample)
    ----------------------------------------------------------------------------------------------------------------------
                metric     mean      std       min      max
    edge_energy_before 0.045663 0.032394  0.017934 0.259006
     edge_energy_after 0.114549 0.038270  0.064359 0.315860
        entropy_before 5.858106 0.690578  3.533567 7.745560
         entropy_after 6.620221 0.646442  3.917019 7.828447
       contrast_before 0.192887 0.054095  0.098088 0.355714
        contrast_after 0.248830 0.022302  0.202713 0.334953
       edge_gain_ratio 2.828335 0.705322  1.219509 4.862013
         entropy_delta 0.762115 0.261006  0.082888 1.505201
        contrast_delta 0.055943 0.034738 -0.020792 0.128680
    
    ----------------------------------------------------------------------------------------------------------------------
    Preprocessing validation summary (DS2 VAL sample)
    ----------------------------------------------------------------------------------------------------------------------
                metric     mean      std       min      max
    edge_energy_before 0.070850 0.038502  0.018920 0.325266
     edge_energy_after 0.140618 0.037632  0.079806 0.370389
        entropy_before 6.827144 0.491417  5.110378 7.763547
         entropy_after 7.356943 0.304290  5.691029 7.814556
       contrast_before 0.235443 0.032823  0.148270 0.350341
        contrast_after 0.260808 0.017986  0.216137 0.332543
       edge_gain_ratio 2.174032 0.500358  1.107402 5.065071
         entropy_delta 0.529799 0.234489  0.042579 1.252218
        contrast_delta 0.025365 0.018302 -0.018649 0.091019
    
    ======================================================================================================================
    STEP 12: SAVING CHECKPOINT + CSV
    ======================================================================================================================
    Saved checkpoint: /kaggle/working/outputs/ARCFNet_Ablation_NoRL_FixedPreset_checkpoint.pth
    Saved CSV: /kaggle/working/outputs/ARCFNet_Ablation_NoRL_FixedPreset_outputs.csv
    
    DONE
    Method: ARCF-Net = Adaptive RACE-FELCM with CRAF Fusion Network
    Ablation: No RL, Fixed Preset, Full Participation
    Backbone: Residual Network-50
    Best round: 10
    Fixed clients => DS1=3, DS2=3, TOTAL=6
    Rounds completed: 15
    Global TEST acc: 0.9779
    Global TEST f1_macro: 0.9770
    DS1 TEST acc: 0.9823
    DS2 TEST acc: 0.9735
    DS1 fixed strategy: fixed_single | names=['race_balanced']
    DS2 fixed strategy: fixed_single | names=['race_balanced']


# **2. Random preset per round**


```python
import os
import sys
import time
import math
import copy
import hashlib
import random
import subprocess
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    log_loss, confusion_matrix, roc_auc_score,
    roc_curve, precision_recall_curve, average_precision_score,
    matthews_corrcoef, cohen_kappa_score, balanced_accuracy_score,
    jaccard_score
)

# ============================================================
# ARCF-Net ABLATION 2
# RANDOM PRESET PER ROUND / FULL PARTICIPATION / NO PLOTS
# ------------------------------------------------------------
# - Uses BOTH datasets
# - Kaggle-ready
# - True FL with FedAvg + FedProx + prototype sharing
# - NO RL-UCB for client count or preprocessing selection
# - Fixed client count per dataset
# - Random preprocessing preset per client per round
# - Full participation every round
# - No plots generated
# ============================================================

# -------------------------
# Helpers
# -------------------------
def pip_install(pkg):
    subprocess.check_call([sys.executable, "-m", "pip", "-q", "install", pkg])

def running_on_kaggle():
    return os.path.exists("/kaggle/input")

IS_KAGGLE = running_on_kaggle()

try:
    import kagglehub
except Exception:
    try:
        pip_install("kagglehub")
        import kagglehub
    except Exception:
        kagglehub = None

try:
    from torchvision import transforms
    from torchvision.models import resnet50, ResNet50_Weights
except Exception:
    pip_install("torchvision")
    from torchvision import transforms
    from torchvision.models import resnet50, ResNet50_Weights

# -------------------------
# Reproducibility + Device
# -------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if DEVICE.type == "cuda":
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

print("=" * 118)
print("ARCF-Net ABLATION 2: RANDOM PRESET PER ROUND + FULL PARTICIPATION + NO PLOTS")
print("=" * 118)
print(f"ENV: {'KAGGLE' if IS_KAGGLE else 'NON-KAGGLE'} | DEVICE: {DEVICE} | torch={torch.__version__}")
print("=" * 118)

# -------------------------
# Configuration
# -------------------------
CFG = {
    "rounds": 15,

    # fixed clients (no RL planning)
    "fixed_clients_ds1": 3,
    "fixed_clients_ds2": 3,

    # local training
    "local_epochs": 2,
    "lr_head": 1e-3,
    "lr_backbone": 2e-4,
    "weight_decay": 5e-4,
    "warmup_epochs": 1,
    "label_smoothing": 0.02,
    "focal_gamma": 1.35,
    "grad_clip": 1.0,
    "fedprox_mu": 0.01,
    "proto_lambda": 0.12,

    # image
    "img_size": 224 if torch.cuda.is_available() else 160,
    "batch_size": 16 if torch.cuda.is_available() else 8,
    "num_workers": 2 if torch.cuda.is_available() else 2,

    # global split
    "global_val_frac": 0.15,
    "test_frac": 0.15,

    # client split
    "client_val_frac": 0.12,
    "client_tune_frac": 0.12,
    "min_per_class_per_client": 5,

    # non-iid
    "dirichlet_alpha": 0.35,

    # preprocessing / augmentation
    "use_preprocessing": True,
    "use_augmentation": True,

    # transfer learning
    "freeze_backbone_rounds": 2,
    "unfreeze_last_blocks": 2,

    # final inference
    "final_use_tta": True,

    # reward
    "reward_f1_weight": 0.75,
    "reward_acc_weight": 0.25,

    # best-round selection: equal dataset importance
    "best_round_mass_ds1": 0.50,
    "best_round_mass_ds2": 0.50,
    "best_round_min_bonus": 0.15,

    # FedAvg tempering
    "fedavg_temper": 0.50,

    # misc / sanity
    "quick_hash_subset_per_split": 300,
    "preproc_val_sample_n": 400,

    # no plots
    "make_plots": False,
    "calibration_bins": 12,
}

OUTDIR = "/kaggle/working/outputs" if IS_KAGGLE else "/content/outputs"
os.makedirs(OUTDIR, exist_ok=True)
MODEL_PATH = os.path.join(OUTDIR, "ARCFNet_Ablation_RandomPresetPerRound_checkpoint.pth")
CSV_PATH = os.path.join(OUTDIR, "ARCFNet_Ablation_RandomPresetPerRound_outputs.csv")

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], device=DEVICE).view(1, 3, 1, 1)
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225], device=DEVICE).view(1, 3, 1, 1)

labels = ["glioma", "meningioma", "notumor", "pituitary"]
label2id = {l: i for i, l in enumerate(labels)}
id2label = {i: l for l, i in label2id.items()}
NUM_CLASSES = len(labels)

METHOD_INFO = {
    "acronym": "ARCF-Net",
    "full_form": "Adaptive RACE-FELCM with CRAF Fusion Network",
    "preprocessing_full_form": "Robust Adaptive Context-Enhanced Fuzzy Edge Local Contrast Mapping",
    "fusion_full_form": "Cross-Residual Adaptive Fusion",
    "backbone_full_form": "Residual Network-50",
    "ablation_name": "Random Preset Per Round, Full Participation",
}

# ============================================================
# Preset banks (random preset per round during training)
# ============================================================
THETA_FULLFORMS = {
    "gamma": "Robust power exponent",
    "alpha": "Local contrast enhancement weight",
    "beta": "Contrast saturation sharpness",
    "tau": "Robust clipping threshold",
    "k": "Local context blur kernel",
    "edge_gain": "Gradient structure gain",
    "blend": "Structure-preserving residual blend",
}

PRESET_BANK_DS1 = [
    ("race_balanced",  (1.00, 0.24, 4.6, 2.4, 5, 0.10, 0.78)),
    ("race_sharp",     (1.04, 0.30, 5.2, 2.5, 5, 0.12, 0.82)),
    ("race_texture",   (0.92, 0.34, 5.8, 2.3, 7, 0.10, 0.80)),
    ("race_robust",    (1.08, 0.22, 4.5, 2.8, 5, 0.11, 0.76)),
    ("race_edge_plus", (1.02, 0.32, 6.0, 2.6, 3, 0.15, 0.84)),
    ("race_focus",     (1.10, 0.36, 6.4, 2.7, 3, 0.17, 0.86)),
]

PRESET_BANK_DS2 = [
    ("race_soft",      (0.95, 0.18, 3.8, 2.2, 3, 0.08, 0.72)),
    ("race_balanced",  (1.00, 0.24, 4.6, 2.4, 5, 0.10, 0.78)),
    ("race_robust",    (1.08, 0.22, 4.5, 2.8, 5, 0.11, 0.76)),
    ("race_smoothmix", (0.90, 0.20, 4.0, 2.1, 7, 0.07, 0.70)),
]

def get_preset_by_name(bank, target_name):
    for name, theta in bank:
        if name == target_name:
            return name, theta
    raise ValueError(f"Preset {target_name} not found.")

def choose_random_preset(bank):
    idx = random.randrange(len(bank))
    name, theta = bank[idx]
    return idx, name, theta

# ============================================================
# CSV collector
# ============================================================
ALL_ROWS = []

def add_table_to_csv(df, table_name):
    if df is None or len(df) == 0:
        return
    df2 = df.copy()
    df2.insert(0, "table_name", table_name)
    for _, row in df2.iterrows():
        ALL_ROWS.append(row.to_dict())

def print_table(df, title, max_rows=12):
    print("\n" + "-" * 118)
    print(title)
    print("-" * 118)
    if df is None or len(df) == 0:
        print("[empty]")
    else:
        print(df.head(max_rows).to_string(index=False))
        if len(df) > max_rows:
            print(f"... showing first {max_rows} of {len(df)} rows")

def safe_float(x, default=np.nan):
    try:
        return float(x)
    except Exception:
        return default

def normalize_imagenet(x):
    return (x - IMAGENET_MEAN) / IMAGENET_STD

def score_metric(m):
    return (
        CFG["reward_f1_weight"] * safe_float(m.get("f1_macro"), 0.0)
        + CFG["reward_acc_weight"] * safe_float(m.get("acc"), 0.0)
    )

def frame_to_manifest(df_):
    return df_[["path", "label", "source", "filename", "y"]].to_dict(orient="list")

# ============================================================
# STEP 0: ACCESS DATASETS
# ============================================================
print("\n" + "=" * 118)
print("STEP 0: ACCESS DATASETS")
print("=" * 118)

REQ1 = {"512Glioma", "512Meningioma", "512Normal", "512Pituitary"}
REQ2 = {"glioma", "meningioma", "notumor", "pituitary"}

def norm_label(name: str):
    s = str(name).strip().lower()
    if "glioma" in s:
        return "glioma"
    if "meningioma" in s:
        return "meningioma"
    if "pituitary" in s:
        return "pituitary"
    if "normal" in s or "no_tumor" in s or "no tumor" in s or "notumor" in s:
        return "notumor"
    return None

def find_root_with_required_class_dirs(base_dir, required_set, prefer_raw=True):
    candidates = []
    for root, dirs, _ in os.walk(base_dir):
        if required_set.issubset(set(dirs)):
            candidates.append(root)
    if not candidates:
        return None

    def score(p):
        pl = p.lower()
        sc = 0
        if prefer_raw:
            if "raw data" in pl:
                sc += 7
            if os.path.basename(p).lower() == "raw":
                sc += 7
            if "/raw/" in pl or "\\raw\\" in pl:
                sc += 3
            if "augmented" in pl:
                sc -= 20
        sc -= 0.0001 * len(p)
        return sc

    return max(candidates, key=score)

def search_kaggle_input_for_root(required_set, prefer_raw=True):
    if not os.path.exists("/kaggle/input"):
        return None
    return find_root_with_required_class_dirs("/kaggle/input", required_set, prefer_raw=prefer_raw)

def try_kagglehub_download(slug, required_set, prefer_raw=True):
    if kagglehub is None:
        return None, None
    try:
        base = kagglehub.dataset_download(slug)
        root = find_root_with_required_class_dirs(base, required_set, prefer_raw=prefer_raw)
        return base, root
    except Exception:
        return None, None

DS1_BASE = None
DS2_BASE = None

DS1_ROOT = search_kaggle_input_for_root(REQ1, prefer_raw=True)
DS2_ROOT = search_kaggle_input_for_root(REQ2, prefer_raw=False)

if DS1_ROOT is None:
    DS1_BASE, DS1_ROOT = try_kagglehub_download(
        "orvile/pmram-bangladeshi-brain-cancer-mri-dataset",
        REQ1,
        prefer_raw=True,
    )

if DS2_ROOT is None:
    DS2_BASE, DS2_ROOT = try_kagglehub_download(
        "yassinebazgour/preprocessed-brain-mri-scans-for-tumors-detection",
        REQ2,
        prefer_raw=False,
    )

if DS1_ROOT is None:
    raise RuntimeError(
        "Could not locate DS1 under /kaggle/input. Add "
        "'orvile/pmram-bangladeshi-brain-cancer-mri-dataset'."
    )
if DS2_ROOT is None:
    raise RuntimeError(
        "Could not locate DS2 under /kaggle/input. Add "
        "'yassinebazgour/preprocessed-brain-mri-scans-for-tumors-detection'."
    )

print(f"Dataset-1 RAW root detected:\n  {DS1_ROOT}")
print(f"Dataset-2 root detected:\n  {DS2_ROOT}")

# ============================================================
# STEP 1: BUILD MANIFESTS
# ============================================================
print("\n" + "=" * 118)
print("STEP 1: BUILD DATA MANIFESTS")
print("=" * 118)

def list_images_under_class_root(class_root, class_dir_name):
    class_dir = os.path.join(class_root, class_dir_name)
    out = []
    for r, _, files in os.walk(class_dir):
        for fn in files:
            if fn.lower().endswith(IMG_EXTS):
                out.append(os.path.join(r, fn))
    return out

def build_df_from_root(ds_root, class_dirs, source_name):
    rows = []
    for c in class_dirs:
        lab = norm_label(c)
        imgs = list_images_under_class_root(ds_root, c)
        print(f"{source_name}: {c} -> {lab} | {len(imgs)} images")
        for p in imgs:
            rows.append({"path": p, "label": lab, "source": source_name})
    dfm = pd.DataFrame(rows).dropna().reset_index(drop=True)
    dfm["path"] = dfm["path"].astype(str)
    dfm["label"] = dfm["label"].astype(str)
    dfm["source"] = dfm["source"].astype(str)
    dfm = dfm.drop_duplicates(subset=["path"]).reset_index(drop=True)
    dfm["filename"] = dfm["path"].apply(os.path.basename)
    return dfm

df1 = build_df_from_root(DS1_ROOT, ["512Glioma", "512Meningioma", "512Normal", "512Pituitary"], "ds1_raw")
df2 = build_df_from_root(DS2_ROOT, ["glioma", "meningioma", "notumor", "pituitary"], "ds2")

def enforce_labels(df_):
    df_ = df_.copy()
    df_["label"] = df_["label"].astype(str).str.strip().str.lower()
    df_ = df_[df_["label"].isin(set(labels))].reset_index(drop=True)
    df_["y"] = df_["label"].map(label2id).astype(int)
    return df_

df1 = enforce_labels(df1)
df2 = enforce_labels(df2)

print("\nDataset-1 images:", len(df1))
print(df1["label"].value_counts().reindex(labels, fill_value=0).to_string())
print("\nDataset-2 images:", len(df2))
print(df2["label"].value_counts().reindex(labels, fill_value=0).to_string())

# ============================================================
# STEP 2: TRAIN / VAL / TEST SPLIT
# ============================================================
print("\n" + "=" * 118)
print("STEP 2: TRAIN / VAL / TEST SPLIT")
print("=" * 118)

def split_dataset(df_):
    train_df, temp_df = train_test_split(
        df_,
        test_size=(CFG["global_val_frac"] + CFG["test_frac"]),
        stratify=df_["y"],
        random_state=SEED,
    )
    val_rel = CFG["global_val_frac"] / (CFG["global_val_frac"] + CFG["test_frac"])
    val_df, test_df = train_test_split(
        temp_df,
        test_size=(1 - val_rel),
        stratify=temp_df["y"],
        random_state=SEED,
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)

train1, val1, test1 = split_dataset(df1)
train2, val2, test2 = split_dataset(df2)

print(f"DS1 TRAIN: {len(train1)} | VAL: {len(val1)} | TEST: {len(test1)}")
print(f"DS2 TRAIN: {len(train2)} | VAL: {len(val2)} | TEST: {len(test2)}")

# ============================================================
# STEP 2.5: SANITY / LEAKAGE CHECKS
# ============================================================
print("\n" + "=" * 118)
print("STEP 2.5: SANITY / LEAKAGE CHECKS")
print("=" * 118)

def split_overlap_checks(train_df, val_df, test_df):
    tr = set(train_df["path"].tolist())
    va = set(val_df["path"].tolist())
    te = set(test_df["path"].tolist())
    checks = {
        "path_overlap_train_val": len(tr.intersection(va)),
        "path_overlap_train_test": len(tr.intersection(te)),
        "path_overlap_val_test": len(va.intersection(te)),
        "unique_paths_train": len(tr),
        "unique_paths_val": len(va),
        "unique_paths_test": len(te),
    }
    trf = set(train_df["filename"].tolist())
    vaf = set(val_df["filename"].tolist())
    tef = set(test_df["filename"].tolist())
    checks.update({
        "filename_overlap_train_val": len(trf.intersection(vaf)),
        "filename_overlap_train_test": len(trf.intersection(tef)),
        "filename_overlap_val_test": len(vaf.intersection(tef)),
    })
    return checks

def md5_file(path, max_bytes=2_000_000):
    h = hashlib.md5()
    try:
        with open(path, "rb") as f:
            h.update(f.read(max_bytes))
        return h.hexdigest()
    except Exception:
        return None

def quick_hash_subset(frame, n=300):
    n = min(n, len(frame))
    if n <= 0:
        return set()
    idx = np.random.choice(len(frame), size=n, replace=False)
    hashes = []
    for i in idx:
        hv = md5_file(frame.iloc[i]["path"])
        if hv is not None:
            hashes.append(hv)
    return set(hashes)

def leakage_report(name, tr, va, te):
    over = split_overlap_checks(tr, va, te)
    leak_df = pd.DataFrame([over])

    n_hash = int(CFG["quick_hash_subset_per_split"])
    trh = quick_hash_subset(tr, n_hash)
    vah = quick_hash_subset(va, n_hash)
    teh = quick_hash_subset(te, n_hash)

    hash_over = {
        "subset_hash_train_val": len(trh.intersection(vah)),
        "subset_hash_train_test": len(trh.intersection(teh)),
        "subset_hash_val_test": len(vah.intersection(teh)),
        "subset_hash_n_train": len(trh),
        "subset_hash_n_val": len(vah),
        "subset_hash_n_test": len(teh),
    }
    leak_df = pd.concat([leak_df, pd.DataFrame([hash_over])], axis=1)
    print_table(leak_df, f"Leakage / Sanity Summary — {name}", max_rows=5)
    add_table_to_csv(leak_df, f"leakage_sanity_{name}")

leakage_report("ds1", train1, val1, test1)
leakage_report("ds2", train2, val2, test2)

# ============================================================
# STEP 3: NON-IID CLIENT PARTITIONING (FIXED)
# ============================================================
print("\n" + "=" * 118)
print("STEP 3: FIXED NON-IID CLIENT PARTITIONING")
print("=" * 118)

def make_clients_non_iid(train_df, n_clients, num_classes, min_per_class=5, alpha=0.35):
    y = train_df["y"].values
    idx_by_class = {c: np.where(y == c)[0].tolist() for c in range(num_classes)}
    for c in idx_by_class:
        random.shuffle(idx_by_class[c])

    client_indices = [[] for _ in range(n_clients)]

    for c in range(num_classes):
        idxs = idx_by_class[c]
        feasible = min(min_per_class, max(1, len(idxs) // n_clients))
        for k in range(n_clients):
            take = idxs[:feasible]
            idxs = idxs[feasible:]
            client_indices[k].extend(take)
        idx_by_class[c] = idxs

    for c in range(num_classes):
        idxs = idx_by_class[c]
        if len(idxs) == 0:
            continue
        props = np.random.dirichlet([alpha] * n_clients)
        counts = (props * len(idxs)).astype(int)
        diff = len(idxs) - counts.sum()
        counts[np.argmax(props)] += diff

        start = 0
        for k in range(n_clients):
            client_indices[k].extend(idxs[start:start + counts[k]])
            start += counts[k]

    for k in range(n_clients):
        random.shuffle(client_indices[k])

    return client_indices

def robust_client_splits(train_df, indices, val_frac, tune_frac):
    idxs = np.array(indices, dtype=int)
    if len(idxs) < 3:
        return idxs.tolist(), idxs.tolist(), idxs.tolist()

    yk = train_df.loc[idxs, "y"].values

    if len(np.unique(yk)) < 2 or len(idxs) < 20:
        n_tune = max(1, int(round(len(idxs) * tune_frac)))
        n_tune = min(n_tune, max(1, len(idxs) - 2))
        tune_idx = idxs[:n_tune]
        rem_idx = idxs[n_tune:]
    else:
        rem_idx, tune_idx = train_test_split(
            idxs,
            test_size=tune_frac,
            stratify=yk,
            random_state=SEED,
        )

    if len(rem_idx) < 2:
        return rem_idx.tolist(), tune_idx.tolist(), rem_idx.tolist()

    yk2 = train_df.loc[rem_idx, "y"].values
    if len(np.unique(yk2)) < 2 or len(rem_idx) < 12:
        n_val = max(1, int(round(len(rem_idx) * val_frac)))
        n_val = min(n_val, max(1, len(rem_idx) - 1))
        val_idx = rem_idx[:n_val]
        train_idx = rem_idx[n_val:]
    else:
        train_idx, val_idx = train_test_split(
            rem_idx,
            test_size=val_frac,
            stratify=yk2,
            random_state=SEED,
        )

    if len(train_idx) == 0:
        train_idx = val_idx[:]
    if len(val_idx) == 0:
        val_idx = train_idx[:1]
    return train_idx.tolist(), tune_idx.tolist(), val_idx.tolist()

n_clients_ds1 = int(CFG["fixed_clients_ds1"])
n_clients_ds2 = int(CFG["fixed_clients_ds2"])

client_indices_ds1 = make_clients_non_iid(
    train1,
    n_clients=n_clients_ds1,
    num_classes=NUM_CLASSES,
    min_per_class=CFG["min_per_class_per_client"],
    alpha=CFG["dirichlet_alpha"],
)
client_indices_ds2 = make_clients_non_iid(
    train2,
    n_clients=n_clients_ds2,
    num_classes=NUM_CLASSES,
    min_per_class=CFG["min_per_class_per_client"],
    alpha=CFG["dirichlet_alpha"],
)

print(f"Fixed clients for DS1: {n_clients_ds1}")
print(f"Fixed clients for DS2: {n_clients_ds2}")

# ============================================================
# STEP 4: DATA LOADERS
# ============================================================
print("\n" + "=" * 118)
print("STEP 4: DATA LOADERS")
print("=" * 118)

def load_rgb(path):
    try:
        return Image.open(path).convert("RGB")
    except Exception:
        return Image.new("RGB", (CFG["img_size"], CFG["img_size"]), (128, 128, 128))

EVAL_TFMS = transforms.Compose([
    transforms.Resize((CFG["img_size"], CFG["img_size"])),
    transforms.ToTensor(),
])

TRAIN_TFMS = transforms.Compose([
    transforms.Resize((CFG["img_size"], CFG["img_size"])),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=12),
    transforms.RandomAffine(degrees=0, translate=(0.04, 0.04), scale=(0.96, 1.04)),
    transforms.ColorJitter(brightness=0.10, contrast=0.10),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.6)),
    transforms.ToTensor(),
])

class MRIDataset(Dataset):
    def __init__(self, frame, indices=None, tfms=None, source_id=0):
        self.df = frame
        self.indices = indices if indices is not None else list(range(len(frame)))
        self.tfms = tfms
        self.source_id = int(source_id)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        j = self.indices[i]
        row = self.df.iloc[j]
        img = load_rgb(row["path"])
        x = self.tfms(img) if self.tfms is not None else transforms.ToTensor()(img)
        y = int(row["y"])
        return x, y, row["path"], self.source_id

def make_weighted_sampler(frame, indices, num_classes):
    if len(indices) == 0:
        return None
    ys = frame.loc[indices, "y"].values
    class_counts = np.bincount(ys, minlength=num_classes)
    class_weights = 1.0 / np.sqrt(np.clip(class_counts, 1, None))
    sample_weights = class_weights[ys]
    return WeightedRandomSampler(
        weights=torch.DoubleTensor(sample_weights),
        num_samples=len(sample_weights),
        replacement=True,
    )

def make_loader(frame, indices, bs, tfms, shuffle=False, sampler=None, source_id=0):
    ds = MRIDataset(frame, indices=indices, tfms=tfms, source_id=source_id)
    return DataLoader(
        ds,
        batch_size=bs,
        shuffle=(shuffle and sampler is None),
        sampler=sampler,
        num_workers=CFG["num_workers"],
        pin_memory=(DEVICE.type == "cuda"),
        drop_last=False,
        persistent_workers=(CFG["num_workers"] > 0),
    )

clients = []
gid = 0

for ds_name, df_src, client_indices, source_id in [
    ("ds1", train1, client_indices_ds1, 0),
    ("ds2", train2, client_indices_ds2, 1),
]:
    for local_id, idxs in enumerate(client_indices):
        tr_idx, tune_idx, val_idx = robust_client_splits(df_src, idxs, CFG["client_val_frac"], CFG["client_tune_frac"])
        sampler = make_weighted_sampler(df_src, tr_idx, NUM_CLASSES)

        train_loader = make_loader(
            df_src, tr_idx, CFG["batch_size"], TRAIN_TFMS,
            shuffle=(sampler is None), sampler=sampler, source_id=source_id
        )
        tune_loader = make_loader(
            df_src,
            tune_idx if len(tune_idx) else tr_idx[:max(1, min(len(tr_idx), CFG["batch_size"]))],
            CFG["batch_size"], EVAL_TFMS, shuffle=False, sampler=None, source_id=source_id
        )
        val_loader = make_loader(
            df_src,
            val_idx if len(val_idx) else tr_idx[:max(1, min(len(tr_idx), CFG["batch_size"]))],
            CFG["batch_size"], EVAL_TFMS, shuffle=False, sampler=None, source_id=source_id
        )
        proto_loader = make_loader(
            df_src, tr_idx, CFG["batch_size"], EVAL_TFMS,
            shuffle=False, sampler=None, source_id=source_id
        )

        counts = df_src.loc[tr_idx, "label"].value_counts().reindex(labels, fill_value=0)

        clients.append({
            "gid": gid,
            "local_id": local_id,
            "dataset": ds_name,
            "source_id": source_id,
            "train_loader": train_loader,
            "tune_loader": tune_loader,
            "val_loader": val_loader,
            "proto_loader": proto_loader,
            "n_train": len(tr_idx),
            "n_tune": len(tune_idx),
            "n_val": len(val_idx),
            "train_indices": tr_idx,
            "tune_indices": tune_idx,
            "val_indices": val_idx,
            "class_counts": counts.to_dict(),
            "preset_bank": PRESET_BANK_DS1 if ds_name == "ds1" else PRESET_BANK_DS2,
        })

        print(f"{ds_name} | client_{gid} | train={len(tr_idx)} | tune={len(tune_idx)} | val={len(val_idx)}")
        gid += 1

CLIENTS_TOTAL = len(clients)

dist_rows = []
for c in clients:
    row = {
        "client": f"client_{c['gid']}",
        "dataset": c["dataset"],
        "total_train": c["n_train"],
        "total_tune": c["n_tune"],
        "total_val": c["n_val"],
        "preset_bank_names": str([x[0] for x in c["preset_bank"]]),
    }
    row.update({lab: int(c["class_counts"].get(lab, 0)) for lab in labels})
    dist_rows.append(row)

dist_df = pd.DataFrame(dist_rows)
print_table(dist_df, "Random-preset client class distribution", max_rows=20)
add_table_to_csv(dist_df, "random_preset_client_distribution")

val_loader_ds1 = make_loader(val1, list(range(len(val1))), CFG["batch_size"], EVAL_TFMS, shuffle=False, source_id=0)
val_loader_ds2 = make_loader(val2, list(range(len(val2))), CFG["batch_size"], EVAL_TFMS, shuffle=False, source_id=1)
test_loader_ds1 = make_loader(test1, list(range(len(test1))), CFG["batch_size"], EVAL_TFMS, shuffle=False, source_id=0)
test_loader_ds2 = make_loader(test2, list(range(len(test2))), CFG["batch_size"], EVAL_TFMS, shuffle=False, source_id=1)

print(f"Augmentation: {'ON' if CFG['use_augmentation'] else 'OFF'}")
print(f"Preprocessing: {'ON' if CFG['use_preprocessing'] else 'OFF'}")
print(f"Total clients: {CLIENTS_TOTAL}")
print(f"Training preset policy DS1: random uniform from {[x[0] for x in PRESET_BANK_DS1]}")
print(f"Training preset policy DS2: random uniform from {[x[0] for x in PRESET_BANK_DS2]}")

# ============================================================
# STEP 5: PREPROCESSING — RACE-FELCM
# ============================================================
print("\n" + "=" * 118)
print("STEP 5: RANDOM-PRESET PREPROCESSING — RACE-FELCM")
print("=" * 118)

class RACEFELCM(nn.Module):
    def __init__(self, gamma=1.0, alpha=0.24, beta=4.5, tau=2.4, blur_k=5, edge_gain=0.10, blend=0.78):
        super().__init__()
        self.gamma = float(gamma)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.tau = float(tau)
        self.blur_k = int(blur_k)
        self.edge_gain = float(edge_gain)
        self.blend = float(blend)

        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        lap = torch.tensor([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=torch.float32)

        self.register_buffer("sobel_x", sobel_x.view(1, 1, 3, 3))
        self.register_buffer("sobel_y", sobel_y.view(1, 1, 3, 3))
        self.register_buffer("lap", lap.view(1, 1, 3, 3))

    def forward(self, x):
        eps = 1e-6

        gray0 = x.mean(dim=1, keepdim=True)
        mu0 = F.avg_pool2d(F.pad(gray0, (1, 1, 1, 1), mode="reflect"), 3, stride=1)
        var0 = F.avg_pool2d(F.pad((gray0 - mu0).pow(2), (1, 1, 1, 1), mode="reflect"), 3, stride=1)
        tex = torch.sqrt(var0 + eps)
        tex_norm = tex / (tex.amax(dim=(2, 3), keepdim=True) + eps)
        flat_gate = 1.0 - tex_norm

        x_smooth = F.avg_pool2d(F.pad(x, (1, 1, 1, 1), mode="reflect"), 3, stride=1)
        x_dn = x * (1.0 - 0.18 * flat_gate) + x_smooth * (0.18 * flat_gate)

        mu = x_dn.mean(dim=(2, 3), keepdim=True)
        sd = x_dn.std(dim=(2, 3), keepdim=True).clamp_min(eps)
        z = ((x_dn - mu) / sd).clamp(-self.tau, self.tau)
        z = torch.sign(z) * torch.pow(z.abs().clamp_min(eps), self.gamma)

        gray = z.mean(dim=1, keepdim=True)
        k = self.blur_k if self.blur_k % 2 == 1 else self.blur_k + 1
        pad = k // 2

        local_mean = F.avg_pool2d(F.pad(gray, (pad, pad, pad, pad), mode="reflect"), k, stride=1)
        local_var = F.avg_pool2d(F.pad((gray - local_mean).pow(2), (pad, pad, pad, pad), mode="reflect"), k, stride=1)
        local_std = torch.sqrt(local_var + eps)
        local_norm = (gray - local_mean) / (local_std + eps)
        contrast_field = torch.tanh(self.beta * local_norm)

        gx = F.conv2d(F.pad(gray, (1, 1, 1, 1), mode="reflect"), self.sobel_x)
        gy = F.conv2d(F.pad(gray, (1, 1, 1, 1), mode="reflect"), self.sobel_y)
        edge = torch.sqrt(gx * gx + gy * gy + eps)
        edge = edge / (edge.amax(dim=(2, 3), keepdim=True) + eps)

        lap = F.conv2d(F.pad(gray, (1, 1, 1, 1), mode="reflect"), self.lap).abs()
        lap = lap / (lap.amax(dim=(2, 3), keepdim=True) + eps)
        coherence = torch.sigmoid(4.0 * (edge - 0.5 * lap))

        enhanced = z + self.alpha * contrast_field + self.edge_gain * edge * coherence

        mn = enhanced.amin(dim=(2, 3), keepdim=True)
        mx = enhanced.amax(dim=(2, 3), keepdim=True)
        enhanced = (enhanced - mn) / (mx - mn + eps)

        out = self.blend * enhanced + (1.0 - self.blend) * x
        return out.clamp(0, 1)

def theta_to_module(theta):
    return RACEFELCM(*theta)

def theta_to_vec(theta, batch_size):
    gamma, alpha, beta, tau, blur_k, edge_gain, blend = theta
    theta = torch.tensor(
        [gamma, alpha, beta / 8.0, tau / 4.0, blur_k / 7.0, edge_gain, blend],
        device=DEVICE,
        dtype=torch.float32
    )
    return theta.unsqueeze(0).repeat(batch_size, 1)

def theta_str(theta):
    if theta is None:
        return "None"
    g, a, b, t, k, eg, m = theta
    return f"(g={g:.2f}, a={a:.2f}, b={b:.2f}, t={t:.2f}, k={k}, eg={eg:.2f}, mix={m:.2f})"

# ============================================================
# STEP 6: MODEL — ResNet-50 + CRAF Fusion
# ============================================================
print("\n" + "=" * 118)
print("STEP 6: MODEL — ResNet-50 + CRAF Fusion")
print("=" * 118)

class CompactProjector(nn.Module):
    def __init__(self, in_dim, out_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            nn.Dropout(0.10),
        )

    def forward(self, x):
        return self.net(x)

class CRAFFusion(nn.Module):
    def __init__(self, in_dim, fuse_dim=256, cond_dim=64):
        super().__init__()
        self.proj_raw = CompactProjector(in_dim, fuse_dim)
        self.proj_enh = CompactProjector(in_dim, fuse_dim)
        self.proj_res = CompactProjector(in_dim, fuse_dim)

        self.router = nn.Sequential(
            nn.Linear(fuse_dim * 6 + cond_dim, fuse_dim),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(fuse_dim, 3),
        )

        self.residual_refine = nn.Sequential(
            nn.Linear(fuse_dim * 2 + cond_dim, fuse_dim),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(fuse_dim, fuse_dim),
        )

        self.out_norm = nn.LayerNorm(fuse_dim)

    def forward(self, f_raw, f_enh, f_res, cond):
        fr = self.proj_raw(f_raw)
        fe = self.proj_enh(f_enh)
        fs = self.proj_res(f_res)

        d_re = torch.abs(fr - fe)
        d_rs = torch.abs(fr - fs)
        d_es = torch.abs(fe - fs)

        router_in = torch.cat([fr, fe, fs, d_re, d_rs, d_es, cond], dim=1)
        gates = torch.softmax(self.router(router_in), dim=1)

        routed = gates[:, 0:1] * fr + gates[:, 1:2] * fe + gates[:, 2:3] * fs
        disagreement = (d_re + d_rs + d_es) / 3.0
        refine = self.residual_refine(torch.cat([routed, disagreement, cond], dim=1))
        fused = self.out_norm(routed + refine)

        return fused, gates

class ResNet50CRAF(nn.Module):
    def __init__(self, num_classes, cond_dim=64, fuse_dim=256, embed_dim=256, pretrained=True):
        super().__init__()
        try:
            weights = ResNet50_Weights.DEFAULT if pretrained else None
        except Exception:
            weights = None

        net = resnet50(weights=weights)
        self.backbone = nn.Sequential(
            net.conv1,
            net.bn1,
            net.relu,
            net.maxpool,
            net.layer1,
            net.layer2,
            net.layer3,
            net.layer4,
        )
        self.backbone_dim = net.fc.in_features
        self.pool = nn.AdaptiveAvgPool2d(1)

        self.theta_mlp = nn.Sequential(
            nn.Linear(7, cond_dim),
            nn.GELU(),
            nn.Linear(cond_dim, cond_dim),
        )
        self.source_emb = nn.Embedding(2, cond_dim)
        self.cond_norm = nn.LayerNorm(cond_dim)

        self.fusion = CRAFFusion(self.backbone_dim, fuse_dim=fuse_dim, cond_dim=cond_dim)

        self.neck = nn.Sequential(
            nn.LayerNorm(fuse_dim),
            nn.Linear(fuse_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(0.20),
        )
        self.classifier = nn.Linear(embed_dim, num_classes)
        self.embed_dim = embed_dim

    def _encode(self, x):
        f = self.backbone(x)
        f = self.pool(f).flatten(1)
        return f

    def forward(self, x_raw_n, x_enh_n, x_res_n, theta_vec, source_id, return_extra=False):
        cond = self.theta_mlp(theta_vec) + self.source_emb(source_id)
        cond = self.cond_norm(cond)

        f_raw = self._encode(x_raw_n)
        f_enh = self._encode(x_enh_n)
        f_res = self._encode(x_res_n)

        fused, gates = self.fusion(f_raw, f_enh, f_res, cond)
        embed = self.neck(fused)
        logits = self.classifier(embed)

        if return_extra:
            return logits, embed, gates
        return logits

def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

def set_trainable_for_round(model, rnd):
    for p in model.backbone.parameters():
        p.requires_grad = False

    for n, p in model.named_parameters():
        if not n.startswith("backbone."):
            p.requires_grad = True

    if rnd > CFG["freeze_backbone_rounds"]:
        blocks = list(model.backbone.children())
        tail_blocks = blocks[-CFG["unfreeze_last_blocks"]:]
        for blk in tail_blocks:
            for p in blk.parameters():
                p.requires_grad = True

def freeze_backbone_bn_stats(model):
    for m in model.backbone.modules():
        if isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm)):
            m.eval()
            for p in m.parameters():
                p.requires_grad = False

def make_optimizer(model):
    head_params, bb_params = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if n.startswith("backbone."):
            bb_params.append(p)
        else:
            head_params.append(p)

    groups = []
    if head_params:
        groups.append({"params": head_params, "lr": CFG["lr_head"]})
    if bb_params:
        groups.append({"params": bb_params, "lr": CFG["lr_backbone"]})

    return torch.optim.AdamW(groups, weight_decay=CFG["weight_decay"])

def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
    def lr_lambda(step):
        if step < num_warmup_steps:
            return float(step) / float(max(1, num_warmup_steps))
        progress = float(step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

global_model = ResNet50CRAF(
    num_classes=NUM_CLASSES,
    cond_dim=64,
    fuse_dim=256,
    embed_dim=256,
    pretrained=True,
).to(DEVICE)

set_trainable_for_round(global_model, rnd=1)
total_params, trainable_params = count_params(global_model)

print("Backbone: ResNet-50 | pretrained_loaded=True")
print(f"Total params: {total_params:,}")
print(f"Trainable params: {trainable_params:,} ({(100.0 * trainable_params / total_params):.2f}%)")

# ============================================================
# STEP 7: LOSSES + PROTOTYPE SHARING
# ============================================================
print("\n" + "=" * 118)
print("STEP 7: LOSSES + PROTOTYPE SHARING")
print("=" * 118)

counts1 = train1["y"].value_counts().sort_index().reindex(range(NUM_CLASSES), fill_value=0).values
counts2 = train2["y"].value_counts().sort_index().reindex(range(NUM_CLASSES), fill_value=0).values
counts = counts1 + counts2
w = (counts.sum() / np.clip(counts, 1, None)).astype(np.float32)
w = w / max(1e-6, w.mean())
class_w = torch.tensor(w, device=DEVICE)

class ClassBalancedFocalLoss(nn.Module):
    def __init__(self, class_weights, gamma=1.5, label_smoothing=0.0):
        super().__init__()
        self.class_weights = class_weights
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, logits, target):
        ce = F.cross_entropy(
            logits,
            target,
            weight=self.class_weights,
            reduction="none",
            label_smoothing=self.label_smoothing,
        )
        pt = torch.exp(-ce)
        focal = (1.0 - pt).pow(self.gamma) * ce
        return focal.mean()

criterion = ClassBalancedFocalLoss(
    class_weights=class_w,
    gamma=CFG["focal_gamma"],
    label_smoothing=CFG["label_smoothing"],
)

scaler = torch.amp.GradScaler("cuda", enabled=(DEVICE.type == "cuda")) if DEVICE.type == "cuda" else None

def fedprox_term(local_model, global_model):
    loss = 0.0
    for p_local, p_global in zip(local_model.parameters(), global_model.parameters()):
        loss = loss + ((p_local - p_global.detach()) ** 2).sum()
    return loss

def prototype_alignment_loss(embed, y, global_prototypes):
    if global_prototypes is None:
        return torch.tensor(0.0, device=embed.device)

    proto = global_prototypes["proto"]
    mask = global_prototypes["mask"]
    valid = mask[y]
    if not valid.any():
        return torch.tensor(0.0, device=embed.device)

    emb_n = F.normalize(embed[valid], dim=1)
    ref_n = F.normalize(proto[y[valid]], dim=1)
    return (1.0 - (emb_n * ref_n).sum(dim=1)).mean()

@torch.no_grad()
def compute_prototype_payload(model, loader, preproc, theta):
    model.eval()
    freeze_backbone_bn_stats(model)
    preproc.eval()

    sums = torch.zeros(NUM_CLASSES, model.embed_dim, device=DEVICE)
    counts = torch.zeros(NUM_CLASSES, device=DEVICE)

    for x, y, _, source_id in loader:
        x = x.to(DEVICE, non_blocking=True)
        y = y.to(DEVICE, non_blocking=True)
        source_id = source_id.to(DEVICE, non_blocking=True)

        x_enh = preproc(x)
        x_res = torch.abs(x_enh - x)

        x_raw_n = normalize_imagenet(x)
        x_enh_n = normalize_imagenet(x_enh)
        x_res_n = normalize_imagenet(x_res)
        theta_vec = theta_to_vec(theta, x.size(0))

        _, embed, _ = model(x_raw_n, x_enh_n, x_res_n, theta_vec, source_id, return_extra=True)

        for c in range(NUM_CLASSES):
            m = (y == c)
            if m.any():
                sums[c] += embed[m].sum(dim=0)
                counts[c] += m.sum()

    return sums.detach().cpu(), counts.detach().cpu()

def aggregate_prototypes(payloads, embed_dim):
    if len(payloads) == 0:
        return None
    sum_all = torch.zeros(NUM_CLASSES, embed_dim)
    cnt_all = torch.zeros(NUM_CLASSES)
    for s, c in payloads:
        sum_all += s
        cnt_all += c
    proto = sum_all / cnt_all.unsqueeze(1).clamp_min(1.0)
    mask = cnt_all > 0
    return {
        "proto": proto.to(DEVICE),
        "mask": mask.to(DEVICE),
        "counts": cnt_all.to(DEVICE),
    }

def gate_entropy(g):
    eps = 1e-6
    p = g.clamp(eps, 1 - eps)
    return -(p * torch.log2(p)).sum(dim=1)

@torch.no_grad()
def _auc_metrics(y_true, p_pred, num_classes):
    out = {}
    try:
        out["auc_roc_macro_ovr"] = float(roc_auc_score(y_true, p_pred, multi_class="ovr", average="macro"))
    except Exception:
        out["auc_roc_macro_ovr"] = np.nan

    for c in range(num_classes):
        try:
            yc = (y_true == c).astype(int)
            if yc.sum() > 0 and yc.sum() < len(yc):
                out[f"auc_class_{c}"] = float(roc_auc_score(yc, p_pred[:, c]))
            else:
                out[f"auc_class_{c}"] = np.nan
        except Exception:
            out[f"auc_class_{c}"] = np.nan
    return out

@torch.no_grad()
def predict_probs_single_theta(model, x, source_id, preproc, theta, use_tta=False):
    model.eval()
    freeze_backbone_bn_stats(model)
    preproc.eval()

    probs_acc = None
    versions = [x]
    if use_tta:
        versions.append(torch.flip(x, dims=[3]))

    for xv in versions:
        x_enh = preproc(xv)
        x_res = torch.abs(x_enh - xv)

        x_raw_n = normalize_imagenet(xv)
        x_enh_n = normalize_imagenet(x_enh)
        x_res_n = normalize_imagenet(x_res)
        theta_vec = theta_to_vec(theta, xv.size(0))

        logits = model(x_raw_n, x_enh_n, x_res_n, theta_vec, source_id)
        probs = torch.softmax(logits, dim=1)
        probs_acc = probs if probs_acc is None else (probs_acc + probs)

    probs_acc = probs_acc / len(versions)
    return probs_acc

@torch.no_grad()
def evaluate_full(model, loader, preproc, theta, return_gates=False, use_tta=False):
    t0 = time.time()
    model.eval()
    freeze_backbone_bn_stats(model)
    preproc.eval()

    all_y, all_p, all_loss = [], [], []
    gate_stats = []

    for x, y, _, source_id in loader:
        x = x.to(DEVICE, non_blocking=True)
        y = y.to(DEVICE, non_blocking=True)
        source_id = source_id.to(DEVICE, non_blocking=True)

        x_enh = preproc(x)
        x_res = torch.abs(x_enh - x)

        x_raw_n = normalize_imagenet(x)
        x_enh_n = normalize_imagenet(x_enh)
        x_res_n = normalize_imagenet(x_res)
        theta_vec = theta_to_vec(theta, x.size(0))

        if return_gates and (not use_tta):
            logits, _, gates = model(x_raw_n, x_enh_n, x_res_n, theta_vec, source_id, return_extra=True)
            gate_stats.append(gates.detach().cpu())
            probs = torch.softmax(logits, dim=1)
        else:
            probs = predict_probs_single_theta(model, x, source_id, preproc, theta, use_tta=use_tta)
            logits = torch.log(probs.clamp_min(1e-8))

        loss = F.cross_entropy(logits, y)

        all_loss.append(float(loss.item()))
        all_y.append(y.detach().cpu().numpy())
        all_p.append(probs.detach().cpu().numpy())

    if len(all_y) == 0:
        return {"acc": np.nan}, np.array([]), np.array([])

    y_true = np.concatenate(all_y)
    p_pred = np.concatenate(all_p)
    y_hat = np.argmax(p_pred, axis=1)

    met = {
        "loss_ce": float(np.mean(all_loss)),
        "acc": float(accuracy_score(y_true, y_hat)),
        "precision_macro": float(precision_score(y_true, y_hat, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_hat, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_hat, average="macro", zero_division=0)),
        "precision_weighted": float(precision_score(y_true, y_hat, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(y_true, y_hat, average="weighted", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_hat, average="weighted", zero_division=0)),
        "log_loss": float(log_loss(y_true, p_pred, labels=list(range(NUM_CLASSES)))),
        "eval_time_s": float(time.time() - t0),
    }
    met.update(_auc_metrics(y_true, p_pred, NUM_CLASSES))

    if return_gates and len(gate_stats) > 0:
        g = torch.cat(gate_stats, dim=0)
        met["fusion_gate_mean_raw"] = float(g[:, 0].mean().item())
        met["fusion_gate_mean_enh"] = float(g[:, 1].mean().item())
        met["fusion_gate_mean_res"] = float(g[:, 2].mean().item())
        met["fusion_gate_entropy"] = float(gate_entropy(g).mean().item())

    return met, y_true, p_pred

def train_one_epoch(model, loader, optimizer, preproc, theta, global_model=None, global_prototypes=None, scheduler=None):
    model.train()
    freeze_backbone_bn_stats(model)
    preproc.eval()

    losses, ce_losses, proto_losses, correct, total = [], [], [], 0, 0

    for x, y, _, source_id in loader:
        x = x.to(DEVICE, non_blocking=True)
        y = y.to(DEVICE, non_blocking=True)
        source_id = source_id.to(DEVICE, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        amp_enabled = (DEVICE.type == "cuda")

        with torch.amp.autocast(device_type=DEVICE.type, enabled=amp_enabled):
            x_enh = preproc(x)
            x_res = torch.abs(x_enh - x)

            x_raw_n = normalize_imagenet(x)
            x_enh_n = normalize_imagenet(x_enh)
            x_res_n = normalize_imagenet(x_res)
            theta_vec = theta_to_vec(theta, x.size(0))

            logits, embed, _ = model(x_raw_n, x_enh_n, x_res_n, theta_vec, source_id, return_extra=True)

            ce = criterion(logits, y)
            proto = prototype_alignment_loss(embed, y, global_prototypes)
            loss = ce + CFG["proto_lambda"] * proto

            if global_model is not None and CFG["fedprox_mu"] > 0:
                loss = loss + 0.5 * CFG["fedprox_mu"] * fedprox_term(model, global_model)

        if scaler is not None and amp_enabled:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if CFG["grad_clip"] > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), CFG["grad_clip"])
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if CFG["grad_clip"] > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), CFG["grad_clip"])
            optimizer.step()

        if scheduler is not None:
            scheduler.step()

        losses.append(float(loss.item()))
        ce_losses.append(float(ce.item()))
        proto_losses.append(float(proto.item()))

        preds = logits.argmax(dim=1)
        correct += int((preds == y).sum().item())
        total += int(y.size(0))

    return {
        "loss": float(np.mean(losses)) if losses else np.nan,
        "ce_loss": float(np.mean(ce_losses)) if ce_losses else np.nan,
        "proto_loss": float(np.mean(proto_losses)) if proto_losses else np.nan,
        "acc": float(correct / max(1, total)),
    }

def build_tempered_fedavg_weights(clients_subset):
    sizes = np.array([c["n_train"] for c in clients_subset], dtype=np.float64)
    sizes = np.power(np.clip(sizes, 1.0, None), CFG["fedavg_temper"])
    sizes = sizes / max(1e-12, sizes.sum())
    return sizes.tolist()

def fedavg_update(global_model, local_models, weights):
    global_sd = global_model.state_dict()
    new_sd = {}

    for key in global_sd.keys():
        ref = global_sd[key]
        if not torch.is_floating_point(ref):
            new_sd[key] = local_models[0].state_dict()[key].detach().clone()
        else:
            acc = None
            for m, w in zip(local_models, weights):
                t = m.state_dict()[key].detach().float().cpu()
                acc = t * w if acc is None else acc + t * w
            new_sd[key] = acc.to(ref.dtype)

    global_model.load_state_dict(new_sd, strict=True)

def weighted_mean(rows, key, weight_key):
    vals, ws = [], []
    for r in rows:
        v = r.get(key, np.nan)
        w = r.get(weight_key, 0)
        if np.isfinite(v) and w > 0:
            vals.append(v)
            ws.append(w)
    if len(vals) == 0:
        return np.nan
    return float(np.average(vals, weights=ws))

def aggregate_by_dataset(round_local_rows, dataset_name):
    rows = [r for r in round_local_rows if r["dataset"] == dataset_name]
    return {
        "acc": weighted_mean(rows, "val_acc", "val_size"),
        "f1_macro": weighted_mean(rows, "val_f1_macro", "val_size"),
        "precision_macro": weighted_mean(rows, "val_precision_macro", "val_size"),
        "recall_macro": weighted_mean(rows, "val_recall_macro", "val_size"),
        "log_loss": weighted_mean(rows, "val_log_loss", "val_size"),
        "loss_ce": weighted_mean(rows, "val_loss_ce", "val_size"),
        "eval_time_s": weighted_mean(rows, "val_eval_time_s", "val_size"),
    }

# ============================================================
# STEP 8: TRUE FEDERATED TRAINING (FULL PARTICIPATION)
# ============================================================
print("\n" + "=" * 118)
print("STEP 8: TRUE FEDERATED TRAINING — FULL PARTICIPATION")
print("=" * 118)

history_global = []
history_local = []

best_reward = -1.0
best_round_saved = None
best_model_state = None
best_global_prototypes = None
global_prototypes = None

t_global_start = time.time()

print(f"Clients => DS1={n_clients_ds1}, DS2={n_clients_ds2}, TOTAL={CLIENTS_TOTAL}")
print(f"Rounds: {CFG['rounds']} | Local epochs: {CFG['local_epochs']}")
print(f"FedProx μ={CFG['fedprox_mu']} | Proto λ={CFG['proto_lambda']}")
print(f"Tempered FedAvg exponent = {CFG['fedavg_temper']:.2f}")
print(f"Training preset policy DS1: random uniform from {[x[0] for x in PRESET_BANK_DS1]}")
print(f"Training preset policy DS2: random uniform from {[x[0] for x in PRESET_BANK_DS2]}")

for rnd in range(1, CFG["rounds"] + 1):
    round_t0 = time.time()
    selected_ids = list(range(len(clients)))  # full participation every round

    print("\n" + "=" * 118)
    print(f"ROUND {rnd}/{CFG['rounds']} | selected={selected_ids}")
    print("=" * 118)

    local_models = []
    proto_payloads = []
    round_local_rows = []
    selected_clients_meta = []

    for cid in selected_ids:
        client = clients[cid]
        theta_arm, theta_name, theta = choose_random_preset(client["preset_bank"])
        preproc = theta_to_module(theta).to(DEVICE)

        local_model = ResNet50CRAF(
            num_classes=NUM_CLASSES,
            cond_dim=64,
            fuse_dim=256,
            embed_dim=256,
            pretrained=False,
        ).to(DEVICE)
        local_model.load_state_dict(global_model.state_dict(), strict=True)

        set_trainable_for_round(local_model, rnd)
        opt = make_optimizer(local_model)

        total_steps = max(1, len(client["train_loader"]) * CFG["local_epochs"])
        warmup_steps = max(1, len(client["train_loader"]) * CFG["warmup_epochs"])
        scheduler = get_cosine_schedule_with_warmup(opt, warmup_steps, total_steps)

        train_logs = []
        for _ in range(CFG["local_epochs"]):
            log_ep = train_one_epoch(
                local_model,
                client["train_loader"],
                opt,
                preproc,
                theta,
                global_model=global_model,
                global_prototypes=global_prototypes,
                scheduler=scheduler,
            )
            train_logs.append(log_ep)

        met_loc, _, _ = evaluate_full(local_model, client["val_loader"], preproc, theta, return_gates=True, use_tta=False)
        proto_payload = compute_prototype_payload(local_model, client["proto_loader"], preproc, theta)

        reward = score_metric(met_loc)

        local_models.append(local_model)
        selected_clients_meta.append(client)
        proto_payloads.append(proto_payload)

        g, a, b, t, kk, eg, mix = theta
        row = {
            "round": rnd,
            "client": f"client_{cid}",
            "dataset": client["dataset"],
            "selected": 1,
            "theta_arm": theta_arm,
            "theta_name": theta_name,
            "theta_str": theta_str(theta),
            "gamma_power": g,
            "alpha_contrast_weight": a,
            "beta_contrast_sharpness": b,
            "tau_clip": t,
            "k_blur_kernel_size": kk,
            "edge_gain": eg,
            "blend_mix": mix,
            "train_loss": float(np.mean([x["loss"] for x in train_logs])),
            "train_ce_loss": float(np.mean([x["ce_loss"] for x in train_logs])),
            "train_proto_loss": float(np.mean([x["proto_loss"] for x in train_logs])),
            "train_acc": float(np.mean([x["acc"] for x in train_logs])),
            "val_size": client["n_val"],
            **{f"val_{k}": v for k, v in met_loc.items()},
            "reward": reward,
        }
        round_local_rows.append(row)

        auc_val = row.get("val_auc_roc_macro_ovr", np.nan)
        print(
            f"Client {cid} ({client['dataset']}) | "
            f"train_acc={row['train_acc']:.4f} | "
            f"val_acc={row['val_acc']:.4f} | "
            f"val_f1={row['val_f1_macro']:.4f} | "
            f"val_auc={auc_val:.4f} | "
            f"reward={reward:.4f} | "
            f"theta={theta_name} {theta_str(theta)}"
        )

    agg_weights = build_tempered_fedavg_weights(selected_clients_meta)
    fedavg_update(global_model, local_models, agg_weights)
    global_prototypes = aggregate_prototypes(proto_payloads, global_model.embed_dim)

    ds1_metrics = aggregate_by_dataset(round_local_rows, "ds1")
    ds2_metrics = aggregate_by_dataset(round_local_rows, "ds2")

    global_metrics = {
        "acc": weighted_mean(round_local_rows, "val_acc", "val_size"),
        "f1_macro": weighted_mean(round_local_rows, "val_f1_macro", "val_size"),
        "precision_macro": weighted_mean(round_local_rows, "val_precision_macro", "val_size"),
        "recall_macro": weighted_mean(round_local_rows, "val_recall_macro", "val_size"),
        "log_loss": weighted_mean(round_local_rows, "val_log_loss", "val_size"),
        "loss_ce": weighted_mean(round_local_rows, "val_loss_ce", "val_size"),
        "eval_time_s": weighted_mean(round_local_rows, "val_eval_time_s", "val_size"),
    }

    ds1_score = score_metric(ds1_metrics)
    ds2_score = score_metric(ds2_metrics)
    round_reward = (
        CFG["best_round_mass_ds1"] * ds1_score +
        CFG["best_round_mass_ds2"] * ds2_score +
        CFG["best_round_min_bonus"] * min(ds1_score, ds2_score)
    )

    history_local.extend(round_local_rows)
    history_global.append({
        "round": rnd,
        "round_time_s": float(time.time() - round_t0),
        "n_selected_clients": len(selected_ids),
        "active_fraction": 1.0,
        "global_reward": round_reward,
        "global_acc": global_metrics["acc"],
        "global_f1_macro": global_metrics["f1_macro"],
        "global_precision_macro": global_metrics["precision_macro"],
        "global_recall_macro": global_metrics["recall_macro"],
        "global_log_loss": global_metrics["log_loss"],
        "global_loss_ce": global_metrics["loss_ce"],
        "global_eval_time_s": global_metrics["eval_time_s"],
        "ds1_acc": ds1_metrics["acc"],
        "ds1_f1_macro": ds1_metrics["f1_macro"],
        "ds1_log_loss": ds1_metrics["log_loss"],
        "ds2_acc": ds2_metrics["acc"],
        "ds2_f1_macro": ds2_metrics["f1_macro"],
        "ds2_log_loss": ds2_metrics["log_loss"],
    })

    print(
        f"GLOBAL VAL (Round {rnd}) | "
        f"global_acc={global_metrics['acc']:.4f} | "
        f"global_f1={global_metrics['f1_macro']:.4f} | "
        f"ds1_acc={ds1_metrics['acc']:.4f} | "
        f"ds1_f1={ds1_metrics['f1_macro']:.4f} | "
        f"ds2_acc={ds2_metrics['acc']:.4f} | "
        f"ds2_f1={ds2_metrics['f1_macro']:.4f} | "
        f"reward={round_reward:.4f} | "
        f"round_time={history_global[-1]['round_time_s']:.1f}s"
    )

    if np.isfinite(round_reward) and round_reward > best_reward:
        best_reward = float(round_reward)
        best_round_saved = rnd
        best_model_state = {k: v.detach().cpu().clone() for k, v in global_model.state_dict().items()}
        best_global_prototypes = None if global_prototypes is None else {
            "proto": global_prototypes["proto"].detach().cpu().clone(),
            "mask": global_prototypes["mask"].detach().cpu().clone(),
            "counts": global_prototypes["counts"].detach().cpu().clone(),
        }

if best_model_state is not None:
    global_model.load_state_dict({k: v.to(DEVICE) for k, v in best_model_state.items()})

if best_global_prototypes is not None:
    global_prototypes = {
        "proto": best_global_prototypes["proto"].to(DEVICE),
        "mask": best_global_prototypes["mask"].to(DEVICE),
        "counts": best_global_prototypes["counts"].to(DEVICE),
    }

t_total = float(time.time() - t_global_start)
print("\n" + "=" * 118)
print(f"TRAINING COMPLETE | total_time={t_total:.1f}s | best_round={best_round_saved} | best_reward={best_reward:.4f}")
print("=" * 118)

glob_df = pd.DataFrame(history_global)
loc_df = pd.DataFrame(history_local)

print_table(glob_df, "GLOBAL per-round metrics", max_rows=20)
print_table(loc_df, "LOCAL per-client per-round metrics", max_rows=20)
add_table_to_csv(glob_df, "global_round_metrics_full")
add_table_to_csv(loc_df, "client_round_metrics_full")

# ============================================================
# STEP 9: FINAL EVALUATION (VALIDATION-SELECTED AFTER RANDOM TRAINING)
# ============================================================
print("\n" + "=" * 118)
print("STEP 9: FINAL EVALUATION (VALIDATION-SELECTED AFTER RANDOM TRAINING)")
print("=" * 118)

@torch.no_grad()
def evaluate_with_single_theta(model, loader, theta, use_tta=False):
    pre = theta_to_module(theta).to(DEVICE)
    return evaluate_full(model, loader, pre, theta, return_gates=False, use_tta=use_tta)

def equal_merge_metrics(m1, m2):
    out = {}
    keys = sorted(set(m1.keys()).union(set(m2.keys())))
    for k in keys:
        a, b = m1.get(k, np.nan), m2.get(k, np.nan)
        if np.isfinite(a) and np.isfinite(b):
            out[k] = float(np.mean([a, b]))
        elif np.isfinite(a):
            out[k] = float(a)
        elif np.isfinite(b):
            out[k] = float(b)
        else:
            out[k] = np.nan
    return out

def rank_static_presets(model, loader, preset_bank):
    rows = []
    for name, theta in preset_bank:
        met, _, _ = evaluate_with_single_theta(model, loader, theta, use_tta=CFG["final_use_tta"])
        rows.append({
            "theta_name": name,
            "theta": theta,
            "score": score_metric(met),
            "val_acc": safe_float(met.get("acc")),
            "val_f1": safe_float(met.get("f1_macro")),
        })
    rows = sorted(rows, key=lambda x: x["score"], reverse=True)
    return rows

ranked_ds1 = rank_static_presets(global_model, val_loader_ds1, PRESET_BANK_DS1)
ranked_ds2 = rank_static_presets(global_model, val_loader_ds2, PRESET_BANK_DS2)

choice_ds1 = ranked_ds1[0]
choice_ds2 = ranked_ds2[0]

val_ds1, _, _ = evaluate_with_single_theta(global_model, val_loader_ds1, choice_ds1["theta"], use_tta=CFG["final_use_tta"])
test_ds1, y_ds1, p_ds1 = evaluate_with_single_theta(global_model, test_loader_ds1, choice_ds1["theta"], use_tta=CFG["final_use_tta"])

val_ds2, _, _ = evaluate_with_single_theta(global_model, val_loader_ds2, choice_ds2["theta"], use_tta=CFG["final_use_tta"])
test_ds2, y_ds2, p_ds2 = evaluate_with_single_theta(global_model, test_loader_ds2, choice_ds2["theta"], use_tta=CFG["final_use_tta"])

val_global = equal_merge_metrics(val_ds1, val_ds2)
test_global = equal_merge_metrics(test_ds1, test_ds2)

choice_df = pd.DataFrame([
    {
        "dataset": "ds1",
        "strategy": "validation_selected_single_after_random_training",
        "theta_names": str([choice_ds1["theta_name"]]),
        "score": choice_ds1["score"],
        "val_acc": safe_float(val_ds1.get("acc")),
        "val_f1": safe_float(val_ds1.get("f1_macro")),
    },
    {
        "dataset": "ds2",
        "strategy": "validation_selected_single_after_random_training",
        "theta_names": str([choice_ds2["theta_name"]]),
        "score": choice_ds2["score"],
        "val_acc": safe_float(val_ds2.get("acc")),
        "val_f1": safe_float(val_ds2.get("f1_macro")),
    },
])
ranked_ds1_df = pd.DataFrame([{k: v for k, v in r.items() if k != "theta"} for r in ranked_ds1])
ranked_ds2_df = pd.DataFrame([{k: v for k, v in r.items() if k != "theta"} for r in ranked_ds2])
print_table(ranked_ds1_df, "Validation ranking of static presets after random training — DS1", max_rows=10)
print_table(ranked_ds2_df, "Validation ranking of static presets after random training — DS2", max_rows=10)
print_table(choice_df, "Final preprocessing strategy after random training", max_rows=5)
add_table_to_csv(ranked_ds1_df, "validation_ranking_static_presets_after_random_training_ds1")
add_table_to_csv(ranked_ds2_df, "validation_ranking_static_presets_after_random_training_ds2")
add_table_to_csv(choice_df, "final_theta_strategy_choice_after_random_training")

# ============================================================
# STEP 10: EXTENDED METRICS + CALIBRATION + ERROR ANALYSIS
# ============================================================
print("\n" + "=" * 118)
print("STEP 10: EXTENDED METRICS + CALIBRATION + ERROR ANALYSIS")
print("=" * 118)

def top_label_calibration(y_true, p_pred, n_bins=12):
    conf = np.max(p_pred, axis=1)
    pred = np.argmax(p_pred, axis=1)
    correct = (pred == y_true).astype(np.float32)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(conf, bins) - 1
    bin_ids = np.clip(bin_ids, 0, n_bins - 1)

    rows = []
    ece = 0.0
    mce = 0.0
    N = len(y_true)

    for b in range(n_bins):
        m = (bin_ids == b)
        if m.sum() == 0:
            rows.append({
                "bin_id": b,
                "bin_left": bins[b],
                "bin_right": bins[b + 1],
                "bin_confidence": np.nan,
                "bin_accuracy": np.nan,
                "bin_gap": np.nan,
                "bin_count": 0,
            })
            continue

        bc = float(conf[m].mean())
        ba = float(correct[m].mean())
        gap = abs(ba - bc)
        ece += (m.sum() / N) * gap
        mce = max(mce, gap)
        rows.append({
            "bin_id": b,
            "bin_left": bins[b],
            "bin_right": bins[b + 1],
            "bin_confidence": bc,
            "bin_accuracy": ba,
            "bin_gap": gap,
            "bin_count": int(m.sum()),
        })

    return pd.DataFrame(rows), float(ece), float(mce)

def multiclass_brier_score(y_true, p_pred, num_classes):
    y_oh = np.eye(num_classes)[y_true]
    return float(np.mean(np.sum((p_pred - y_oh) ** 2, axis=1)))

def classwise_confusion_metrics(y_true, p_pred):
    y_hat = np.argmax(p_pred, axis=1)
    cm = confusion_matrix(y_true, y_hat, labels=list(range(NUM_CLASSES)))
    total = cm.sum()

    rows = []
    for c, lab in enumerate(labels):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        tn = total - tp - fp - fn

        ppv = tp / max(1, (tp + fp))
        npv = tn / max(1, (tn + fn))
        sensitivity = tp / max(1, (tp + fn))
        specificity = tn / max(1, (tn + fp))
        fpr = fp / max(1, (fp + tn))
        fnr = fn / max(1, (fn + tp))
        jacc = tp / max(1, (tp + fp + fn))
        support = tp + fn
        prevalence = support / max(1, total)
        bal = 0.5 * (sensitivity + specificity)

        rows.append({
            "class_id": c,
            "class_name": lab,
            "support": int(support),
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
            "tn": int(tn),
            "prevalence": float(prevalence),
            "ppv": float(ppv),
            "npv": float(npv),
            "recall": float(sensitivity),
            "specificity": float(specificity),
            "fpr": float(fpr),
            "fnr": float(fnr),
            "jaccard": float(jacc),
            "balanced_acc": float(bal),
        })

    return cm, pd.DataFrame(rows)

def top_confusion_pairs(y_true, p_pred):
    y_hat = np.argmax(p_pred, axis=1)
    cm = confusion_matrix(y_true, y_hat, labels=list(range(NUM_CLASSES)))
    rows = []
    for i, lab_i in enumerate(labels):
        for j, lab_j in enumerate(labels):
            if i == j:
                continue
            rows.append({
                "true_class": lab_i,
                "pred_class": lab_j,
                "count": int(cm[i, j]),
            })
    return pd.DataFrame(rows).sort_values("count", ascending=False).reset_index(drop=True)

def wrong_right_confidence_table(y_true, p_pred):
    pred = np.argmax(p_pred, axis=1)
    conf = np.max(p_pred, axis=1)
    ok = pred == y_true
    return pd.DataFrame({
        "confidence": conf,
        "correct": ok.astype(int),
        "status": np.where(ok, "correct", "wrong"),
        "true_label": [id2label[int(x)] for x in y_true],
        "pred_label": [id2label[int(x)] for x in pred],
    })

def compute_extended_bundle(y_true, p_pred, dataset_name):
    y_hat = np.argmax(p_pred, axis=1)
    cm, class_df = classwise_confusion_metrics(y_true, p_pred)
    cal_df, ece, mce = top_label_calibration(y_true, p_pred, n_bins=CFG["calibration_bins"])
    conf_df = top_confusion_pairs(y_true, p_pred)
    conf_table = wrong_right_confidence_table(y_true, p_pred)

    scalar = {
        "dataset": dataset_name,
        "acc": float(accuracy_score(y_true, y_hat)),
        "balanced_acc": float(balanced_accuracy_score(y_true, y_hat)),
        "precision_macro": float(precision_score(y_true, y_hat, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_hat, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_hat, average="macro", zero_division=0)),
        "precision_weighted": float(precision_score(y_true, y_hat, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(y_true, y_hat, average="weighted", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_hat, average="weighted", zero_division=0)),
        "log_loss": float(log_loss(y_true, p_pred, labels=list(range(NUM_CLASSES)))),
        "mcc": float(matthews_corrcoef(y_true, y_hat)),
        "kappa": float(cohen_kappa_score(y_true, y_hat)),
        "jaccard_macro": float(jaccard_score(y_true, y_hat, average="macro")),
        "ppv_macro": float(class_df["ppv"].mean()),
        "npv_macro": float(class_df["npv"].mean()),
        "specificity_macro": float(class_df["specificity"].mean()),
        "fpr_macro": float(class_df["fpr"].mean()),
        "fnr_macro": float(class_df["fnr"].mean()),
        "ece": float(ece),
        "mce": float(mce),
        "brier_multi": float(multiclass_brier_score(y_true, p_pred, NUM_CLASSES)),
    }
    scalar.update(_auc_metrics(y_true, p_pred, NUM_CLASSES))
    return scalar, class_df, cal_df, conf_df, conf_table, cm

ext_ds1, class_df1, cal_df1, conf_pairs1, conf_table1, cm1 = compute_extended_bundle(y_ds1, p_ds1, "ds1_test")
ext_ds2, class_df2, cal_df2, conf_pairs2, conf_table2, cm2 = compute_extended_bundle(y_ds2, p_ds2, "ds2_test")

ext_df = pd.DataFrame([ext_ds1, ext_ds2])

print_table(ext_df, "Extended TEST metrics (DS1 vs DS2)", max_rows=5)
print_table(class_df1, "Classwise metrics — DS1 TEST", max_rows=10)
print_table(class_df2, "Classwise metrics — DS2 TEST", max_rows=10)
print_table(conf_pairs1.head(10), "Top confusion pairs — DS1 TEST", max_rows=10)
print_table(conf_pairs2.head(10), "Top confusion pairs — DS2 TEST", max_rows=10)
print_table(cal_df1, "Calibration bins — DS1 TEST", max_rows=15)
print_table(cal_df2, "Calibration bins — DS2 TEST", max_rows=15)

add_table_to_csv(ext_df, "extended_test_metrics_ds1_ds2")
add_table_to_csv(class_df1, "classwise_metrics_ds1_test")
add_table_to_csv(class_df2, "classwise_metrics_ds2_test")
add_table_to_csv(cal_df1, "calibration_bins_ds1_test")
add_table_to_csv(cal_df2, "calibration_bins_ds2_test")
add_table_to_csv(conf_pairs1, "top_confusions_ds1_test")
add_table_to_csv(conf_pairs2, "top_confusions_ds2_test")
add_table_to_csv(conf_table1, "confidence_correctness_ds1_test")
add_table_to_csv(conf_table2, "confidence_correctness_ds2_test")

test_ds1.update(ext_ds1)
test_ds2.update(ext_ds2)

def compact_metrics(m):
    keep = [
        "acc", "balanced_acc", "precision_macro", "recall_macro", "f1_macro",
        "precision_weighted", "recall_weighted", "f1_weighted", "log_loss",
        "auc_roc_macro_ovr", "mcc", "kappa", "ppv_macro", "npv_macro",
        "specificity_macro", "ece", "mce", "brier_multi", "loss_ce", "eval_time_s"
    ]
    out = {}
    for k in keep:
        if k in m and np.isfinite(m[k]):
            out[k] = float(m[k])
    return out

paper_df = pd.DataFrame([
    {"setting": "ARCF-Net RandomPresetPerRound", "split": "VAL",  "dataset": "ds1",          **compact_metrics(val_ds1)},
    {"setting": "ARCF-Net RandomPresetPerRound", "split": "VAL",  "dataset": "ds2",          **compact_metrics(val_ds2)},
    {"setting": "ARCF-Net RandomPresetPerRound", "split": "VAL",  "dataset": "global_equal", **compact_metrics(val_global)},
    {"setting": "ARCF-Net RandomPresetPerRound", "split": "TEST", "dataset": "ds1",          **compact_metrics(test_ds1)},
    {"setting": "ARCF-Net RandomPresetPerRound", "split": "TEST", "dataset": "ds2",          **compact_metrics(test_ds2)},
    {"setting": "ARCF-Net RandomPresetPerRound", "split": "TEST", "dataset": "global_equal", **compact_metrics(test_global)},
])

print_table(paper_df, "VAL + TEST tables", max_rows=10)
add_table_to_csv(paper_df, "paper_ready_metrics")

print("\nSelection summary:")
print(f"- Best round: {best_round_saved} | best_reward={best_reward:.4f}")
print(f"- DS1 final strategy: validation_selected_single_after_random_training | names={[choice_ds1['theta_name']]}")
print(f"- DS2 final strategy: validation_selected_single_after_random_training | names={[choice_ds2['theta_name']]}")

# ============================================================
# STEP 11: PREPROCESSING VALIDATION
# ============================================================
print("\n" + "=" * 118)
print("STEP 11: PREPROCESSING VALIDATION")
print("=" * 118)

@torch.no_grad()
def entropy_per_image(x01):
    gray = x01.mean(dim=1)
    B = gray.shape[0]
    ent = []
    for i in range(B):
        g = (gray[i].detach().cpu().numpy() * 255).astype(np.uint8)
        hist = np.bincount(g.flatten(), minlength=256).astype(np.float32)
        p = hist / np.clip(hist.sum(), 1, None)
        p = p[p > 0]
        ent.append(float(-(p * np.log2(p)).sum()))
    return np.array(ent)

@torch.no_grad()
def edge_energy(x01, kernel):
    kernel = kernel.to(device=x01.device, dtype=x01.dtype)
    gray = x01.mean(dim=1, keepdim=True)
    lap = F.conv2d(F.pad(gray, (1, 1, 1, 1), mode="reflect"), kernel).abs()
    return lap.mean(dim=(1, 2, 3)).detach().cpu().numpy()

@torch.no_grad()
def contrast_proxy(x01):
    gray = x01.mean(dim=1)
    return gray.std(dim=(1, 2)).detach().cpu().numpy()

@torch.no_grad()
def run_preproc_validation(frame, preproc, sample_n=400):
    n = min(sample_n, len(frame))
    if n <= 0:
        return pd.DataFrame(), pd.DataFrame()

    idx = np.random.choice(len(frame), size=n, replace=False)
    ds = MRIDataset(frame, indices=idx.tolist(), tfms=EVAL_TFMS, source_id=0)

    xs = []
    for i in range(len(ds)):
        x, _, _, _ = ds[i]
        xs.append(x)
    x = torch.stack(xs).to(DEVICE)

    x_after = preproc(x).clamp(0, 1)
    lap_kernel = preproc.lap if hasattr(preproc, "lap") else RACEFELCM().to(DEVICE).lap

    ee_before = edge_energy(x, lap_kernel)
    ee_after  = edge_energy(x_after, lap_kernel)
    ent_before = entropy_per_image(x)
    ent_after  = entropy_per_image(x_after)
    con_before = contrast_proxy(x)
    con_after  = contrast_proxy(x_after)

    dfm = pd.DataFrame({
        "edge_energy_before": ee_before,
        "edge_energy_after": ee_after,
        "entropy_before": ent_before,
        "entropy_after": ent_after,
        "contrast_before": con_before,
        "contrast_after": con_after,
        "edge_gain_ratio": (ee_after / np.clip(ee_before, 1e-9, None)),
        "entropy_delta": (ent_after - ent_before),
        "contrast_delta": (con_after - con_before),
    })
    summary = dfm.agg(["mean", "std", "min", "max"]).T.reset_index().rename(columns={"index": "metric"})
    return dfm, summary

preproc_ds1 = theta_to_module(choice_ds1["theta"]).to(DEVICE)
preproc_ds2 = theta_to_module(choice_ds2["theta"]).to(DEVICE)

preproc_df1, preproc_summary_df1 = run_preproc_validation(val1, preproc_ds1, CFG["preproc_val_sample_n"])
preproc_df2, preproc_summary_df2 = run_preproc_validation(val2, preproc_ds2, CFG["preproc_val_sample_n"])

print_table(preproc_summary_df1, "Preprocessing validation summary (DS1 VAL sample)", max_rows=15)
print_table(preproc_summary_df2, "Preprocessing validation summary (DS2 VAL sample)", max_rows=15)
add_table_to_csv(preproc_summary_df1, "preprocessing_validation_summary_ds1")
add_table_to_csv(preproc_summary_df2, "preprocessing_validation_summary_ds2")

# ============================================================
# STEP 12: SAVE CHECKPOINT + CSV
# ============================================================
print("\n" + "=" * 118)
print("STEP 12: SAVING CHECKPOINT + CSV")
print("=" * 118)

client_process_manifest = []
for c in clients:
    df_src = train1 if c["dataset"] == "ds1" else train2
    client_process_manifest.append({
        "gid": c["gid"],
        "local_id": c["local_id"],
        "dataset": c["dataset"],
        "source_id": c["source_id"],
        "n_train": c["n_train"],
        "n_tune": c["n_tune"],
        "n_val": c["n_val"],
        "train_indices": c["train_indices"],
        "tune_indices": c["tune_indices"],
        "val_indices": c["val_indices"],
        "train_paths": df_src.loc[c["train_indices"], "path"].tolist(),
        "tune_paths": df_src.loc[c["tune_indices"], "path"].tolist(),
        "val_paths": df_src.loc[c["val_indices"], "path"].tolist(),
        "class_counts": c["class_counts"],
        "preset_bank": c["preset_bank"],
    })

all_df = pd.DataFrame(ALL_ROWS)

checkpoint = {
    "method_info": METHOD_INFO,

    "state_dict": {k: v.detach().cpu() for k, v in global_model.state_dict().items()},
    "best_model_state_dict": {k: v.detach().cpu() for k, v in global_model.state_dict().items()},

    "config": CFG,
    "seed": SEED,
    "device_used": str(DEVICE),
    "backbone_name": "resnet50",
    "fusion_name": "CRAF",
    "preprocessing_name": "RACE-FELCM",

    "dataset1_raw_root": DS1_ROOT,
    "dataset2_root": DS2_ROOT,
    "labels": labels,
    "label2id": label2id,
    "id2label": id2label,
    "num_classes": NUM_CLASSES,
    "theta_fullforms": THETA_FULLFORMS,

    "full_dataset_df1": frame_to_manifest(df1),
    "full_dataset_df2": frame_to_manifest(df2),

    "split_train1": frame_to_manifest(train1),
    "split_val1": frame_to_manifest(val1),
    "split_test1": frame_to_manifest(test1),
    "split_train2": frame_to_manifest(train2),
    "split_val2": frame_to_manifest(val2),
    "split_test2": frame_to_manifest(test2),

    "fixed_clients_ds1": n_clients_ds1,
    "fixed_clients_ds2": n_clients_ds2,
    "clients_total": CLIENTS_TOTAL,
    "client_indices_ds1": client_indices_ds1,
    "client_indices_ds2": client_indices_ds2,
    "client_process_manifest": client_process_manifest,

    "preset_bank_ds1": PRESET_BANK_DS1,
    "preset_bank_ds2": PRESET_BANK_DS2,
    "training_preset_policy": "random_uniform_per_client_per_round",

    "best_round_saved": best_round_saved,
    "best_reward": best_reward,
    "history_global": glob_df.to_dict(orient="list"),
    "history_local": loc_df.to_dict(orient="list"),
    "total_training_time_s": t_total,

    "final_choice_ds1": {
        "strategy": "validation_selected_single_after_random_training",
        "theta_names": [choice_ds1["theta_name"]],
        "theta_list": [choice_ds1["theta"]],
        "score": choice_ds1["score"],
        "val_acc": safe_float(val_ds1.get("acc")),
        "val_f1": safe_float(val_ds1.get("f1_macro")),
    },
    "final_choice_ds2": {
        "strategy": "validation_selected_single_after_random_training",
        "theta_names": [choice_ds2["theta_name"]],
        "theta_list": [choice_ds2["theta"]],
        "score": choice_ds2["score"],
        "val_acc": safe_float(val_ds2.get("acc")),
        "val_f1": safe_float(val_ds2.get("f1_macro")),
    },

    "best_global_prototypes": None if best_global_prototypes is None else {
        "proto": best_global_prototypes["proto"],
        "mask": best_global_prototypes["mask"],
        "counts": best_global_prototypes["counts"],
    },

    "final_val_ds1": val_ds1,
    "final_val_ds2": val_ds2,
    "final_val_global": val_global,
    "final_test_ds1": test_ds1,
    "final_test_ds2": test_ds2,
    "final_test_global_equal": test_global,
    "paper_ready_metrics": paper_df.to_dict(orient="list"),
    "extended_test_metrics_ds1_ds2": ext_df.to_dict(orient="list"),
    "classwise_metrics_ds1_test": class_df1.to_dict(orient="list"),
    "classwise_metrics_ds2_test": class_df2.to_dict(orient="list"),
    "calibration_bins_ds1_test": cal_df1.to_dict(orient="list"),
    "calibration_bins_ds2_test": cal_df2.to_dict(orient="list"),
    "top_confusions_ds1_test": conf_pairs1.to_dict(orient="list"),
    "top_confusions_ds2_test": conf_pairs2.to_dict(orient="list"),
    "confidence_correctness_ds1_test": conf_table1.to_dict(orient="list"),
    "confidence_correctness_ds2_test": conf_table2.to_dict(orient="list"),
    "random_preset_client_distribution_table": dist_df.to_dict(orient="list"),
    "global_round_metrics_table": glob_df.to_dict(orient="list"),
    "local_round_metrics_table": loc_df.to_dict(orient="list"),
    "all_tables_long_format": all_df.to_dict(orient="list"),

    "ds1_test_y_true": y_ds1,
    "ds1_test_p_pred": p_ds1,
    "ds1_test_y_hat": np.argmax(p_ds1, axis=1),
    "ds2_test_y_true": y_ds2,
    "ds2_test_p_pred": p_ds2,
    "ds2_test_y_hat": np.argmax(p_ds2, axis=1),

    "preprocessing_validation_ds1_raw": preproc_df1.to_dict(orient="list") if len(preproc_df1) else {},
    "preprocessing_validation_ds2_raw": preproc_df2.to_dict(orient="list") if len(preproc_df2) else {},
    "preprocessing_validation_summary_ds1": preproc_summary_df1.to_dict(orient="list") if len(preproc_summary_df1) else {},
    "preprocessing_validation_summary_ds2": preproc_summary_df2.to_dict(orient="list") if len(preproc_summary_df2) else {},
}

torch.save(checkpoint, MODEL_PATH)
print(f"Saved checkpoint: {MODEL_PATH}")

all_df.to_csv(CSV_PATH, index=False)
print(f"Saved CSV: {CSV_PATH}")

print("\nDONE")
print(f"Method: {METHOD_INFO['acronym']} = {METHOD_INFO['full_form']}")
print(f"Ablation: {METHOD_INFO['ablation_name']}")
print(f"Backbone: {METHOD_INFO['backbone_full_form']}")
print(f"Best round: {best_round_saved}")
print(f"Fixed clients => DS1={n_clients_ds1}, DS2={n_clients_ds2}, TOTAL={CLIENTS_TOTAL}")
print(f"Rounds completed: {CFG['rounds']}")
print(f"Global TEST acc: {safe_float(test_global.get('acc')):.4f}")
print(f"Global TEST f1_macro: {safe_float(test_global.get('f1_macro')):.4f}")
print(f"DS1 TEST acc: {safe_float(test_ds1.get('acc')):.4f}")
print(f"DS2 TEST acc: {safe_float(test_ds2.get('acc')):.4f}")
print(f"DS1 final strategy: validation_selected_single_after_random_training | names={[choice_ds1['theta_name']]}")
print(f"DS2 final strategy: validation_selected_single_after_random_training | names={[choice_ds2['theta_name']]}")
```

    ======================================================================================================================
    ARCF-Net ABLATION 2: RANDOM PRESET PER ROUND + FULL PARTICIPATION + NO PLOTS
    ======================================================================================================================
    ENV: KAGGLE | DEVICE: cuda | torch=2.10.0+cu128
    ======================================================================================================================
    
    ======================================================================================================================
    STEP 0: ACCESS DATASETS
    ======================================================================================================================
    Using Colab cache for faster access to the 'pmram-bangladeshi-brain-cancer-mri-dataset' dataset.
    Using Colab cache for faster access to the 'preprocessed-brain-mri-scans-for-tumors-detection' dataset.
    Dataset-1 RAW root detected:
      /kaggle/input/pmram-bangladeshi-brain-cancer-mri-dataset/PMRAM Bangladeshi Brain Cancer - MRI Dataset/PMRAM Bangladeshi Brain Cancer - MRI Dataset/Raw Data/Raw
    Dataset-2 root detected:
      /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset
    
    ======================================================================================================================
    STEP 1: BUILD DATA MANIFESTS
    ======================================================================================================================
    ds1_raw: 512Glioma -> glioma | 373 images
    ds1_raw: 512Meningioma -> meningioma | 363 images
    ds1_raw: 512Normal -> notumor | 396 images
    ds1_raw: 512Pituitary -> pituitary | 373 images
    ds2: glioma -> glioma | 1621 images
    ds2: meningioma -> meningioma | 1646 images
    ds2: notumor -> notumor | 2000 images
    ds2: pituitary -> pituitary | 1764 images
    
    Dataset-1 images: 1505
    label
    glioma        373
    meningioma    363
    notumor       396
    pituitary     373
    
    Dataset-2 images: 7031
    label
    glioma        1621
    meningioma    1646
    notumor       2000
    pituitary     1764
    
    ======================================================================================================================
    STEP 2: TRAIN / VAL / TEST SPLIT
    ======================================================================================================================
    DS1 TRAIN: 1053 | VAL: 226 | TEST: 226
    DS2 TRAIN: 4921 | VAL: 1055 | TEST: 1055
    
    ======================================================================================================================
    STEP 2.5: SANITY / LEAKAGE CHECKS
    ======================================================================================================================
    
    ----------------------------------------------------------------------------------------------------------------------
    Leakage / Sanity Summary — ds1
    ----------------------------------------------------------------------------------------------------------------------
     path_overlap_train_val  path_overlap_train_test  path_overlap_val_test  unique_paths_train  unique_paths_val  unique_paths_test  filename_overlap_train_val  filename_overlap_train_test  filename_overlap_val_test  subset_hash_train_val  subset_hash_train_test  subset_hash_val_test  subset_hash_n_train  subset_hash_n_val  subset_hash_n_test
                          0                        0                      0                1053               226                226                           0                            0                          0                      5                       5                     6                  298                222                 224
    
    ----------------------------------------------------------------------------------------------------------------------
    Leakage / Sanity Summary — ds2
    ----------------------------------------------------------------------------------------------------------------------
     path_overlap_train_val  path_overlap_train_test  path_overlap_val_test  unique_paths_train  unique_paths_val  unique_paths_test  filename_overlap_train_val  filename_overlap_train_test  filename_overlap_val_test  subset_hash_train_val  subset_hash_train_test  subset_hash_val_test  subset_hash_n_train  subset_hash_n_val  subset_hash_n_test
                          0                        0                      0                4921              1055               1055                           0                            0                          0                      0                       3                     4                  299                298                 299
    
    ======================================================================================================================
    STEP 3: FIXED NON-IID CLIENT PARTITIONING
    ======================================================================================================================
    Fixed clients for DS1: 3
    Fixed clients for DS2: 3
    
    ======================================================================================================================
    STEP 4: DATA LOADERS
    ======================================================================================================================
    ds1 | client_0 | train=490 | tune=77 | val=67
    ds1 | client_1 | train=125 | tune=20 | val=18
    ds1 | client_2 | train=198 | tune=31 | val=27
    ds2 | client_3 | train=629 | tune=98 | val=86
    ds2 | client_4 | train=527 | tune=82 | val=72
    ds2 | client_5 | train=2653 | tune=412 | val=362
    
    ----------------------------------------------------------------------------------------------------------------------
    Random-preset client class distribution
    ----------------------------------------------------------------------------------------------------------------------
      client dataset  total_train  total_tune  total_val                                                                              preset_bank_names  glioma  meningioma  notumor  pituitary
    client_0     ds1          490          77         67 ['race_balanced', 'race_sharp', 'race_texture', 'race_robust', 'race_edge_plus', 'race_focus']     111          46      176        157
    client_1     ds1          125          20         18 ['race_balanced', 'race_sharp', 'race_texture', 'race_robust', 'race_edge_plus', 'race_focus']      75           4        8         38
    client_2     ds1          198          31         27 ['race_balanced', 'race_sharp', 'race_texture', 'race_robust', 'race_edge_plus', 'race_focus']      16         147       30          5
    client_3     ds2          629          98         86                                ['race_soft', 'race_balanced', 'race_robust', 'race_smoothmix']      12         197      416          4
    client_4     ds2          527          82         72                                ['race_soft', 'race_balanced', 'race_robust', 'race_smoothmix']     202           4      284         37
    client_5     ds2         2653         412        362                                ['race_soft', 'race_balanced', 'race_robust', 'race_smoothmix']     665         691      383        914
    Augmentation: ON
    Preprocessing: ON
    Total clients: 6
    Training preset policy DS1: random uniform from ['race_balanced', 'race_sharp', 'race_texture', 'race_robust', 'race_edge_plus', 'race_focus']
    Training preset policy DS2: random uniform from ['race_soft', 'race_balanced', 'race_robust', 'race_smoothmix']
    
    ======================================================================================================================
    STEP 5: RANDOM-PRESET PREPROCESSING — RACE-FELCM
    ======================================================================================================================
    
    ======================================================================================================================
    STEP 6: MODEL — ResNet-50 + CRAF Fusion
    ======================================================================================================================
    Backbone: ResNet-50 | pretrained_loaded=True
    Total params: 25,790,855
    Trainable params: 2,282,823 (8.85%)
    
    ======================================================================================================================
    STEP 7: LOSSES + PROTOTYPE SHARING
    ======================================================================================================================
    
    ======================================================================================================================
    STEP 8: TRUE FEDERATED TRAINING — FULL PARTICIPATION
    ======================================================================================================================
    Clients => DS1=3, DS2=3, TOTAL=6
    Rounds: 15 | Local epochs: 2
    FedProx μ=0.01 | Proto λ=0.12
    Tempered FedAvg exponent = 0.50
    Training preset policy DS1: random uniform from ['race_balanced', 'race_sharp', 'race_texture', 'race_robust', 'race_edge_plus', 'race_focus']
    Training preset policy DS2: random uniform from ['race_soft', 'race_balanced', 'race_robust', 'race_smoothmix']
    
    ======================================================================================================================
    ROUND 1/15 | selected=[0, 1, 2, 3, 4, 5]
    ======================================================================================================================
    Client 0 (ds1) | train_acc=0.7153 | val_acc=0.8955 | val_f1=0.8306 | val_auc=0.9868 | reward=0.8468 | theta=race_texture (g=0.92, a=0.34, b=5.80, t=2.30, k=7, eg=0.10, mix=0.80)
    Client 1 (ds1) | train_acc=0.6440 | val_acc=0.9444 | val_f1=0.8730 | val_auc=nan | reward=0.8909 | theta=race_focus (g=1.10, a=0.36, b=6.40, t=2.70, k=3, eg=0.17, mix=0.86)
    Client 2 (ds1) | train_acc=0.6465 | val_acc=0.5556 | val_f1=0.5491 | val_auc=0.9089 | reward=0.5507 | theta=race_texture (g=0.92, a=0.34, b=5.80, t=2.30, k=7, eg=0.10, mix=0.80)
    Client 3 (ds2) | train_acc=0.8148 | val_acc=0.9302 | val_f1=0.4636 | val_auc=0.9076 | reward=0.5803 | theta=race_soft (g=0.95, a=0.18, b=3.80, t=2.20, k=3, eg=0.08, mix=0.72)
    Client 4 (ds2) | train_acc=0.8083 | val_acc=0.9444 | val_f1=0.6676 | val_auc=nan | reward=0.7368 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    Client 5 (ds2) | train_acc=0.7799 | val_acc=0.8398 | val_f1=0.8378 | val_auc=0.9684 | reward=0.8383 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    GLOBAL VAL (Round 1) | global_acc=0.8608 | global_f1=0.7554 | ds1_acc=0.8214 | ds1_f1=0.7696 | ds2_acc=0.8692 | ds2_f1=0.7524 | reward=0.8993 | round_time=155.7s
    
    ======================================================================================================================
    ROUND 2/15 | selected=[0, 1, 2, 3, 4, 5]
    ======================================================================================================================
    Client 0 (ds1) | train_acc=0.8908 | val_acc=0.8955 | val_f1=0.8453 | val_auc=0.9710 | reward=0.8579 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    Client 1 (ds1) | train_acc=0.8760 | val_acc=0.9444 | val_f1=0.7381 | val_auc=nan | reward=0.7897 | theta=race_edge_plus (g=1.02, a=0.32, b=6.00, t=2.60, k=3, eg=0.15, mix=0.84)
    Client 2 (ds1) | train_acc=0.7980 | val_acc=0.9259 | val_f1=0.7381 | val_auc=0.9329 | reward=0.7851 | theta=race_sharp (g=1.04, a=0.30, b=5.20, t=2.50, k=5, eg=0.12, mix=0.82)
    Client 3 (ds2) | train_acc=0.8919 | val_acc=0.9186 | val_f1=0.7059 | val_auc=0.9514 | reward=0.7591 | theta=race_robust (g=1.08, a=0.22, b=4.50, t=2.80, k=5, eg=0.11, mix=0.76)
    Client 4 (ds2) | train_acc=0.9402 | val_acc=1.0000 | val_f1=1.0000 | val_auc=nan | reward=1.0000 | theta=race_soft (g=0.95, a=0.18, b=3.80, t=2.20, k=3, eg=0.08, mix=0.72)
    Client 5 (ds2) | train_acc=0.8349 | val_acc=0.8343 | val_f1=0.8275 | val_auc=0.9745 | reward=0.8292 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    GLOBAL VAL (Round 2) | global_acc=0.8782 | global_f1=0.8262 | ds1_acc=0.9107 | ds1_f1=0.8022 | ds2_acc=0.8712 | ds2_f1=0.8313 | reward=0.9597 | round_time=151.2s
    
    ======================================================================================================================
    ROUND 3/15 | selected=[0, 1, 2, 3, 4, 5]
    ======================================================================================================================
    Client 0 (ds1) | train_acc=0.8827 | val_acc=0.9254 | val_f1=0.8772 | val_auc=0.9937 | reward=0.8892 | theta=race_texture (g=0.92, a=0.34, b=5.80, t=2.30, k=7, eg=0.10, mix=0.80)
    Client 1 (ds1) | train_acc=0.7640 | val_acc=0.9444 | val_f1=0.8730 | val_auc=nan | reward=0.8909 | theta=race_sharp (g=1.04, a=0.30, b=5.20, t=2.50, k=5, eg=0.12, mix=0.82)
    Client 2 (ds1) | train_acc=0.7727 | val_acc=0.9630 | val_f1=0.9106 | val_auc=0.9643 | reward=0.9237 | theta=race_texture (g=0.92, a=0.34, b=5.80, t=2.30, k=7, eg=0.10, mix=0.80)
    Client 3 (ds2) | train_acc=0.8824 | val_acc=0.9651 | val_f1=0.9797 | val_auc=0.9950 | reward=0.9761 | theta=race_soft (g=0.95, a=0.18, b=3.80, t=2.20, k=3, eg=0.08, mix=0.72)
    Client 4 (ds2) | train_acc=0.9156 | val_acc=0.9861 | val_f1=0.9654 | val_auc=nan | reward=0.9706 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    Client 5 (ds2) | train_acc=0.8811 | val_acc=0.9613 | val_f1=0.9621 | val_auc=0.9958 | reward=0.9619 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    GLOBAL VAL (Round 3) | global_acc=0.9604 | global_f1=0.9511 | ds1_acc=0.9375 | ds1_f1=0.8846 | ds2_acc=0.9654 | ds2_f1=0.9655 | reward=1.0663 | round_time=168.5s
    
    ======================================================================================================================
    ROUND 4/15 | selected=[0, 1, 2, 3, 4, 5]
    ======================================================================================================================
    Client 0 (ds1) | train_acc=0.9194 | val_acc=0.9552 | val_f1=0.9164 | val_auc=0.9950 | reward=0.9261 | theta=race_edge_plus (g=1.02, a=0.32, b=6.00, t=2.60, k=3, eg=0.15, mix=0.84)
    Client 1 (ds1) | train_acc=0.8840 | val_acc=1.0000 | val_f1=1.0000 | val_auc=nan | reward=1.0000 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    Client 2 (ds1) | train_acc=0.9571 | val_acc=0.9259 | val_f1=0.7381 | val_auc=0.9814 | reward=0.7851 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    Client 3 (ds2) | train_acc=0.9452 | val_acc=0.9651 | val_f1=0.4842 | val_auc=0.9967 | reward=0.6044 | theta=race_robust (g=1.08, a=0.22, b=4.50, t=2.80, k=5, eg=0.11, mix=0.76)
    Client 4 (ds2) | train_acc=0.9326 | val_acc=0.9861 | val_f1=0.9571 | val_auc=nan | reward=0.9644 | theta=race_robust (g=1.08, a=0.22, b=4.50, t=2.80, k=5, eg=0.11, mix=0.76)
    Client 5 (ds2) | train_acc=0.9144 | val_acc=0.9586 | val_f1=0.9569 | val_auc=0.9932 | reward=0.9573 | theta=race_robust (g=1.08, a=0.22, b=4.50, t=2.80, k=5, eg=0.11, mix=0.76)
    GLOBAL VAL (Round 4) | global_acc=0.9620 | global_f1=0.8802 | ds1_acc=0.9554 | ds1_f1=0.8869 | ds2_acc=0.9635 | ds2_f1=0.8788 | reward=1.0370 | round_time=170.5s
    
    ======================================================================================================================
    ROUND 5/15 | selected=[0, 1, 2, 3, 4, 5]
    ======================================================================================================================
    Client 0 (ds1) | train_acc=0.9694 | val_acc=0.9851 | val_f1=0.9863 | val_auc=0.9994 | reward=0.9860 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    Client 1 (ds1) | train_acc=0.8880 | val_acc=1.0000 | val_f1=1.0000 | val_auc=nan | reward=1.0000 | theta=race_focus (g=1.10, a=0.36, b=6.40, t=2.70, k=3, eg=0.17, mix=0.86)
    Client 2 (ds1) | train_acc=0.9520 | val_acc=0.8889 | val_f1=0.6968 | val_auc=0.9761 | reward=0.7449 | theta=race_edge_plus (g=1.02, a=0.32, b=6.00, t=2.60, k=3, eg=0.15, mix=0.84)
    Client 3 (ds2) | train_acc=0.9618 | val_acc=0.9767 | val_f1=0.7389 | val_auc=1.0000 | reward=0.7983 | theta=race_robust (g=1.08, a=0.22, b=4.50, t=2.80, k=5, eg=0.11, mix=0.76)
    Client 4 (ds2) | train_acc=0.9649 | val_acc=1.0000 | val_f1=1.0000 | val_auc=nan | reward=1.0000 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    Client 5 (ds2) | train_acc=0.9386 | val_acc=0.9613 | val_f1=0.9607 | val_auc=0.9948 | reward=0.9608 | theta=race_smoothmix (g=0.90, a=0.20, b=4.00, t=2.10, k=7, eg=0.07, mix=0.70)
    GLOBAL VAL (Round 5) | global_acc=0.9684 | global_f1=0.9275 | ds1_acc=0.9643 | ds1_f1=0.9187 | ds2_acc=0.9692 | ds2_f1=0.9294 | reward=1.0743 | round_time=169.1s
    
    ======================================================================================================================
    ROUND 6/15 | selected=[0, 1, 2, 3, 4, 5]
    ======================================================================================================================
    Client 0 (ds1) | train_acc=0.9724 | val_acc=0.9552 | val_f1=0.9467 | val_auc=0.9981 | reward=0.9488 | theta=race_edge_plus (g=1.02, a=0.32, b=6.00, t=2.60, k=3, eg=0.15, mix=0.84)
    Client 1 (ds1) | train_acc=0.9440 | val_acc=0.9444 | val_f1=0.7273 | val_auc=nan | reward=0.7816 | theta=race_texture (g=0.92, a=0.34, b=5.80, t=2.30, k=7, eg=0.10, mix=0.80)
    Client 2 (ds1) | train_acc=0.9722 | val_acc=0.9259 | val_f1=0.7381 | val_auc=0.9729 | reward=0.7851 | theta=race_texture (g=0.92, a=0.34, b=5.80, t=2.30, k=7, eg=0.10, mix=0.80)
    Client 3 (ds2) | train_acc=0.9785 | val_acc=0.9884 | val_f1=0.7478 | val_auc=0.9997 | reward=0.8080 | theta=race_soft (g=0.95, a=0.18, b=3.80, t=2.20, k=3, eg=0.08, mix=0.72)
    Client 4 (ds2) | train_acc=0.9763 | val_acc=0.9722 | val_f1=0.9321 | val_auc=nan | reward=0.9421 | theta=race_soft (g=0.95, a=0.18, b=3.80, t=2.20, k=3, eg=0.08, mix=0.72)
    Client 5 (ds2) | train_acc=0.9504 | val_acc=0.9558 | val_f1=0.9544 | val_auc=0.9961 | reward=0.9548 | theta=race_robust (g=1.08, a=0.22, b=4.50, t=2.80, k=5, eg=0.11, mix=0.76)
    GLOBAL VAL (Round 6) | global_acc=0.9604 | global_f1=0.9072 | ds1_acc=0.9464 | ds1_f1=0.8612 | ds2_acc=0.9635 | ds2_f1=0.9172 | reward=1.0380 | round_time=169.9s
    
    ======================================================================================================================
    ROUND 7/15 | selected=[0, 1, 2, 3, 4, 5]
    ======================================================================================================================
    Client 0 (ds1) | train_acc=0.9592 | val_acc=0.9403 | val_f1=0.9239 | val_auc=0.9931 | reward=0.9280 | theta=race_sharp (g=1.04, a=0.30, b=5.20, t=2.50, k=5, eg=0.12, mix=0.82)
    Client 1 (ds1) | train_acc=0.9600 | val_acc=1.0000 | val_f1=1.0000 | val_auc=nan | reward=1.0000 | theta=race_focus (g=1.10, a=0.36, b=6.40, t=2.70, k=3, eg=0.17, mix=0.86)
    Client 2 (ds1) | train_acc=0.9798 | val_acc=0.9259 | val_f1=0.8690 | val_auc=0.9861 | reward=0.8833 | theta=race_texture (g=0.92, a=0.34, b=5.80, t=2.30, k=7, eg=0.10, mix=0.80)
    Client 3 (ds2) | train_acc=0.9722 | val_acc=0.9767 | val_f1=0.9861 | val_auc=1.0000 | reward=0.9837 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    Client 4 (ds2) | train_acc=0.9753 | val_acc=0.9861 | val_f1=0.9636 | val_auc=nan | reward=0.9693 | theta=race_robust (g=1.08, a=0.22, b=4.50, t=2.80, k=5, eg=0.11, mix=0.76)
    Client 5 (ds2) | train_acc=0.9452 | val_acc=0.9724 | val_f1=0.9723 | val_auc=0.9983 | reward=0.9723 | theta=race_soft (g=0.95, a=0.18, b=3.80, t=2.20, k=3, eg=0.08, mix=0.72)
    GLOBAL VAL (Round 7) | global_acc=0.9699 | global_f1=0.9644 | ds1_acc=0.9464 | ds1_f1=0.9229 | ds2_acc=0.9750 | ds2_f1=0.9734 | reward=1.0906 | round_time=170.2s
    
    ======================================================================================================================
    ROUND 8/15 | selected=[0, 1, 2, 3, 4, 5]
    ======================================================================================================================
    Client 0 (ds1) | train_acc=0.9694 | val_acc=0.9552 | val_f1=0.9479 | val_auc=0.9994 | reward=0.9497 | theta=race_edge_plus (g=1.02, a=0.32, b=6.00, t=2.60, k=3, eg=0.15, mix=0.84)
    Client 1 (ds1) | train_acc=0.9400 | val_acc=0.9444 | val_f1=0.7273 | val_auc=nan | reward=0.7816 | theta=race_focus (g=1.10, a=0.36, b=6.40, t=2.70, k=3, eg=0.17, mix=0.86)
    Client 2 (ds1) | train_acc=0.9697 | val_acc=0.9259 | val_f1=0.8606 | val_auc=1.0000 | reward=0.8769 | theta=race_edge_plus (g=1.02, a=0.32, b=6.00, t=2.60, k=3, eg=0.15, mix=0.84)
    Client 3 (ds2) | train_acc=0.9769 | val_acc=0.9884 | val_f1=0.9145 | val_auc=0.9928 | reward=0.9329 | theta=race_soft (g=0.95, a=0.18, b=3.80, t=2.20, k=3, eg=0.08, mix=0.72)
    Client 4 (ds2) | train_acc=0.9753 | val_acc=0.9861 | val_f1=0.7455 | val_auc=nan | reward=0.8056 | theta=race_smoothmix (g=0.90, a=0.20, b=4.00, t=2.10, k=7, eg=0.07, mix=0.70)
    Client 5 (ds2) | train_acc=0.9583 | val_acc=0.9724 | val_f1=0.9744 | val_auc=0.9969 | reward=0.9739 | theta=race_robust (g=1.08, a=0.22, b=4.50, t=2.80, k=5, eg=0.11, mix=0.76)
    GLOBAL VAL (Round 8) | global_acc=0.9715 | global_f1=0.9255 | ds1_acc=0.9464 | ds1_f1=0.8914 | ds2_acc=0.9769 | ds2_f1=0.9328 | reward=1.0603 | round_time=169.0s
    
    ======================================================================================================================
    ROUND 9/15 | selected=[0, 1, 2, 3, 4, 5]
    ======================================================================================================================
    Client 0 (ds1) | train_acc=0.9694 | val_acc=0.9552 | val_f1=0.9303 | val_auc=0.9943 | reward=0.9365 | theta=race_focus (g=1.10, a=0.36, b=6.40, t=2.70, k=3, eg=0.17, mix=0.86)
    Client 1 (ds1) | train_acc=0.9480 | val_acc=1.0000 | val_f1=1.0000 | val_auc=nan | reward=1.0000 | theta=race_focus (g=1.10, a=0.36, b=6.40, t=2.70, k=3, eg=0.17, mix=0.86)
    Client 2 (ds1) | train_acc=0.9747 | val_acc=0.8519 | val_f1=0.6750 | val_auc=0.9798 | reward=0.7192 | theta=race_edge_plus (g=1.02, a=0.32, b=6.00, t=2.60, k=3, eg=0.15, mix=0.84)
    Client 3 (ds2) | train_acc=0.9714 | val_acc=0.9884 | val_f1=0.7478 | val_auc=1.0000 | reward=0.8080 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    Client 4 (ds2) | train_acc=0.9867 | val_acc=0.9722 | val_f1=0.9321 | val_auc=nan | reward=0.9421 | theta=race_smoothmix (g=0.90, a=0.20, b=4.00, t=2.10, k=7, eg=0.07, mix=0.70)
    Client 5 (ds2) | train_acc=0.9487 | val_acc=0.9724 | val_f1=0.9733 | val_auc=0.9978 | reward=0.9731 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    GLOBAL VAL (Round 9) | global_acc=0.9684 | global_f1=0.9214 | ds1_acc=0.9375 | ds1_f1=0.8800 | ds2_acc=0.9750 | ds2_f1=0.9303 | reward=1.0521 | round_time=168.1s
    
    ======================================================================================================================
    ROUND 10/15 | selected=[0, 1, 2, 3, 4, 5]
    ======================================================================================================================
    Client 0 (ds1) | train_acc=0.9653 | val_acc=0.9552 | val_f1=0.9474 | val_auc=0.9964 | reward=0.9494 | theta=race_texture (g=0.92, a=0.34, b=5.80, t=2.30, k=7, eg=0.10, mix=0.80)
    Client 1 (ds1) | train_acc=0.9560 | val_acc=1.0000 | val_f1=1.0000 | val_auc=nan | reward=1.0000 | theta=race_texture (g=0.92, a=0.34, b=5.80, t=2.30, k=7, eg=0.10, mix=0.80)
    Client 2 (ds1) | train_acc=0.9722 | val_acc=0.9630 | val_f1=0.9106 | val_auc=1.0000 | reward=0.9237 | theta=race_sharp (g=1.04, a=0.30, b=5.20, t=2.50, k=5, eg=0.12, mix=0.82)
    Client 3 (ds2) | train_acc=0.9857 | val_acc=0.9884 | val_f1=0.7455 | val_auc=1.0000 | reward=0.8062 | theta=race_smoothmix (g=0.90, a=0.20, b=4.00, t=2.10, k=7, eg=0.07, mix=0.70)
    Client 4 (ds2) | train_acc=0.9725 | val_acc=1.0000 | val_f1=1.0000 | val_auc=nan | reward=1.0000 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    Client 5 (ds2) | train_acc=0.9661 | val_acc=0.9669 | val_f1=0.9691 | val_auc=0.9958 | reward=0.9685 | theta=race_smoothmix (g=0.90, a=0.20, b=4.00, t=2.10, k=7, eg=0.07, mix=0.70)
    GLOBAL VAL (Round 10) | global_acc=0.9731 | global_f1=0.9382 | ds1_acc=0.9643 | ds1_f1=0.9470 | ds2_acc=0.9750 | ds2_f1=0.9364 | reward=1.0906 | round_time=169.3s
    
    ======================================================================================================================
    ROUND 11/15 | selected=[0, 1, 2, 3, 4, 5]
    ======================================================================================================================
    Client 0 (ds1) | train_acc=0.9592 | val_acc=0.9403 | val_f1=0.9198 | val_auc=0.9934 | reward=0.9250 | theta=race_sharp (g=1.04, a=0.30, b=5.20, t=2.50, k=5, eg=0.12, mix=0.82)
    Client 1 (ds1) | train_acc=0.9400 | val_acc=1.0000 | val_f1=1.0000 | val_auc=nan | reward=1.0000 | theta=race_focus (g=1.10, a=0.36, b=6.40, t=2.70, k=3, eg=0.17, mix=0.86)
    Client 2 (ds1) | train_acc=0.9874 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | reward=1.0000 | theta=race_focus (g=1.10, a=0.36, b=6.40, t=2.70, k=3, eg=0.17, mix=0.86)
    Client 3 (ds2) | train_acc=0.9825 | val_acc=0.9884 | val_f1=0.7478 | val_auc=1.0000 | reward=0.8080 | theta=race_smoothmix (g=0.90, a=0.20, b=4.00, t=2.10, k=7, eg=0.07, mix=0.70)
    Client 4 (ds2) | train_acc=0.9877 | val_acc=1.0000 | val_f1=1.0000 | val_auc=nan | reward=1.0000 | theta=race_robust (g=1.08, a=0.22, b=4.50, t=2.80, k=5, eg=0.11, mix=0.76)
    Client 5 (ds2) | train_acc=0.9646 | val_acc=0.9807 | val_f1=0.9804 | val_auc=0.9986 | reward=0.9805 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    GLOBAL VAL (Round 11) | global_acc=0.9810 | global_f1=0.9460 | ds1_acc=0.9643 | ds1_f1=0.9520 | ds2_acc=0.9846 | ds2_f1=0.9446 | reward=1.0981 | round_time=169.6s
    
    ======================================================================================================================
    ROUND 12/15 | selected=[0, 1, 2, 3, 4, 5]
    ======================================================================================================================
    Client 0 (ds1) | train_acc=0.9755 | val_acc=0.9403 | val_f1=0.9316 | val_auc=0.9995 | reward=0.9338 | theta=race_robust (g=1.08, a=0.22, b=4.50, t=2.80, k=5, eg=0.11, mix=0.76)
    Client 1 (ds1) | train_acc=0.9640 | val_acc=1.0000 | val_f1=1.0000 | val_auc=nan | reward=1.0000 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    Client 2 (ds1) | train_acc=0.9722 | val_acc=0.9259 | val_f1=0.9091 | val_auc=0.9932 | reward=0.9133 | theta=race_robust (g=1.08, a=0.22, b=4.50, t=2.80, k=5, eg=0.11, mix=0.76)
    Client 3 (ds2) | train_acc=0.9833 | val_acc=0.9884 | val_f1=0.9145 | val_auc=0.9998 | reward=0.9329 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    Client 4 (ds2) | train_acc=0.9877 | val_acc=1.0000 | val_f1=1.0000 | val_auc=nan | reward=1.0000 | theta=race_smoothmix (g=0.90, a=0.20, b=4.00, t=2.10, k=7, eg=0.07, mix=0.70)
    Client 5 (ds2) | train_acc=0.9685 | val_acc=0.9641 | val_f1=0.9641 | val_auc=0.9988 | reward=0.9641 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    GLOBAL VAL (Round 12) | global_acc=0.9684 | global_f1=0.9567 | ds1_acc=0.9464 | ds1_f1=0.9372 | ds2_acc=0.9731 | ds2_f1=0.9609 | reward=1.0926 | round_time=168.3s
    
    ======================================================================================================================
    ROUND 13/15 | selected=[0, 1, 2, 3, 4, 5]
    ======================================================================================================================
    Client 0 (ds1) | train_acc=0.9755 | val_acc=0.9403 | val_f1=0.9189 | val_auc=0.9946 | reward=0.9243 | theta=race_sharp (g=1.04, a=0.30, b=5.20, t=2.50, k=5, eg=0.12, mix=0.82)
    Client 1 (ds1) | train_acc=0.9760 | val_acc=1.0000 | val_f1=1.0000 | val_auc=nan | reward=1.0000 | theta=race_sharp (g=1.04, a=0.30, b=5.20, t=2.50, k=5, eg=0.12, mix=0.82)
    Client 2 (ds1) | train_acc=0.9672 | val_acc=0.9630 | val_f1=0.9143 | val_auc=1.0000 | reward=0.9265 | theta=race_edge_plus (g=1.02, a=0.32, b=6.00, t=2.60, k=3, eg=0.15, mix=0.84)
    Client 3 (ds2) | train_acc=0.9865 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | reward=1.0000 | theta=race_smoothmix (g=0.90, a=0.20, b=4.00, t=2.10, k=7, eg=0.07, mix=0.70)
    Client 4 (ds2) | train_acc=0.9839 | val_acc=0.9861 | val_f1=0.7455 | val_auc=nan | reward=0.8056 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    Client 5 (ds2) | train_acc=0.9663 | val_acc=0.9807 | val_f1=0.9811 | val_auc=0.9992 | reward=0.9810 | theta=race_robust (g=1.08, a=0.22, b=4.50, t=2.80, k=5, eg=0.11, mix=0.76)
    GLOBAL VAL (Round 13) | global_acc=0.9794 | global_f1=0.9479 | ds1_acc=0.9554 | ds1_f1=0.9308 | ds2_acc=0.9846 | ds2_f1=0.9516 | reward=1.0890 | round_time=169.4s
    
    ======================================================================================================================
    ROUND 14/15 | selected=[0, 1, 2, 3, 4, 5]
    ======================================================================================================================
    Client 0 (ds1) | train_acc=0.9745 | val_acc=0.9701 | val_f1=0.9616 | val_auc=1.0000 | reward=0.9637 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    Client 1 (ds1) | train_acc=0.9760 | val_acc=1.0000 | val_f1=1.0000 | val_auc=nan | reward=1.0000 | theta=race_texture (g=0.92, a=0.34, b=5.80, t=2.30, k=7, eg=0.10, mix=0.80)
    Client 2 (ds1) | train_acc=0.9975 | val_acc=0.9630 | val_f1=0.9106 | val_auc=1.0000 | reward=0.9237 | theta=race_sharp (g=1.04, a=0.30, b=5.20, t=2.50, k=5, eg=0.12, mix=0.82)
    Client 3 (ds2) | train_acc=0.9897 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | reward=1.0000 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    Client 4 (ds2) | train_acc=0.9905 | val_acc=0.9861 | val_f1=0.7455 | val_auc=nan | reward=0.8056 | theta=race_smoothmix (g=0.90, a=0.20, b=4.00, t=2.10, k=7, eg=0.07, mix=0.70)
    Client 5 (ds2) | train_acc=0.9755 | val_acc=0.9862 | val_f1=0.9855 | val_auc=0.9992 | reward=0.9856 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    GLOBAL VAL (Round 14) | global_acc=0.9858 | global_f1=0.9548 | ds1_acc=0.9732 | ds1_f1=0.9555 | ds2_acc=0.9885 | ds2_f1=0.9546 | reward=1.1055 | round_time=170.0s
    
    ======================================================================================================================
    ROUND 15/15 | selected=[0, 1, 2, 3, 4, 5]
    ======================================================================================================================
    Client 0 (ds1) | train_acc=0.9837 | val_acc=0.9851 | val_f1=0.9750 | val_auc=0.9960 | reward=0.9775 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    Client 1 (ds1) | train_acc=0.9560 | val_acc=1.0000 | val_f1=1.0000 | val_auc=nan | reward=1.0000 | theta=race_sharp (g=1.04, a=0.30, b=5.20, t=2.50, k=5, eg=0.12, mix=0.82)
    Client 2 (ds1) | train_acc=0.9874 | val_acc=0.9630 | val_f1=0.9106 | val_auc=1.0000 | reward=0.9237 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    Client 3 (ds2) | train_acc=0.9960 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | reward=1.0000 | theta=race_smoothmix (g=0.90, a=0.20, b=4.00, t=2.10, k=7, eg=0.07, mix=0.70)
    Client 4 (ds2) | train_acc=0.9886 | val_acc=0.9861 | val_f1=0.7455 | val_auc=nan | reward=0.8056 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    Client 5 (ds2) | train_acc=0.9764 | val_acc=0.9779 | val_f1=0.9766 | val_auc=0.9990 | reward=0.9769 | theta=race_smoothmix (g=0.90, a=0.20, b=4.00, t=2.10, k=7, eg=0.07, mix=0.70)
    GLOBAL VAL (Round 15) | global_acc=0.9826 | global_f1=0.9511 | ds1_acc=0.9821 | ds1_f1=0.9635 | ds2_acc=0.9827 | ds2_f1=0.9485 | reward=1.1061 | round_time=172.2s
    
    ======================================================================================================================
    TRAINING COMPLETE | total_time=2511.6s | best_round=15 | best_reward=1.1061
    ======================================================================================================================
    
    ----------------------------------------------------------------------------------------------------------------------
    GLOBAL per-round metrics
    ----------------------------------------------------------------------------------------------------------------------
     round  round_time_s  n_selected_clients  active_fraction  global_reward  global_acc  global_f1_macro  global_precision_macro  global_recall_macro  global_log_loss  global_loss_ce  global_eval_time_s  ds1_acc  ds1_f1_macro  ds1_log_loss  ds2_acc  ds2_f1_macro  ds2_log_loss
         1    155.709075                   6              1.0       0.899298    0.860759         0.755426                0.763836             0.757754         0.355193        0.351998            2.525898 0.821429      0.769556      0.467248 0.869231      0.752382      0.331058
         2    151.169605                   6              1.0       0.959719    0.878165         0.826157                0.825882             0.829101         0.294767        0.296437            2.383452 0.910714      0.802242      0.335110 0.871154      0.831308      0.286078
         3    168.549433                   6              1.0       1.066290    0.960443         0.951132                0.950036             0.958216         0.168791        0.174716            2.392656 0.937500      0.884556      0.254842 0.965385      0.965472      0.150257
         4    170.536360                   6              1.0       1.036956    0.962025         0.880197                0.882813             0.878967         0.136305        0.134732            2.369421 0.955357      0.886879      0.143671 0.963462      0.878758      0.134719
         5    169.135407                   6              1.0       1.074261    0.968354         0.927538                0.928876             0.927134         0.107264        0.108099            2.368638 0.964286      0.918709      0.110376 0.969231      0.929439      0.106594
         6    169.856469                   6              1.0       1.037977    0.960443         0.907234                0.904047             0.913263         0.137518        0.148407            2.371366 0.946429      0.861160      0.195074 0.963462      0.917157      0.125122
         7    170.236475                   6              1.0       1.090603    0.969937         0.964441                0.966064             0.967167         0.096397        0.091655            2.365591 0.946429      0.922898      0.175487 0.975000      0.973389      0.079362
         8    168.990061                   6              1.0       1.060253    0.971519         0.925456                0.922549             0.937179         0.111785        0.128992            2.380906 0.946429      0.891372      0.163885 0.976923      0.932797      0.100563
         9    168.051224                   6              1.0       1.052067    0.968354         0.921393                0.918062             0.927470         0.124902        0.122363            2.369338 0.937500      0.879958      0.235801 0.975000      0.930317      0.101016
        10    169.342760                   6              1.0       1.090568    0.973101         0.938242                0.940260             0.938465         0.088014        0.086122            2.363954 0.964286      0.946987      0.110334 0.975000      0.936358      0.083207
        11    169.591285                   6              1.0       1.098064    0.981013         0.945950                0.945275             0.948337         0.088010        0.084626            2.358405 0.964286      0.952048      0.134147 0.984615      0.944637      0.078073
        12    168.339305                   6              1.0       1.092625    0.968354         0.956661                0.952005             0.969368         0.103756        0.101318            2.358047 0.946429      0.937177      0.170979 0.973077      0.960858      0.089278
        13    169.400824                   6              1.0       1.088966    0.979430         0.947946                0.948780             0.948813         0.082615        0.078946            2.363948 0.955357      0.930835      0.134932 0.984615      0.951631      0.071346
        14    169.967492                   6              1.0       1.105476    0.985759         0.954779                0.957428             0.954383         0.063333        0.060216            2.387740 0.973214      0.955452      0.098131 0.988462      0.954634      0.055839
        15    172.249638                   6              1.0       1.106130    0.982595         0.951123                0.955338             0.949159         0.065765        0.063478            2.393196 0.982143      0.963459      0.090920 0.982692      0.948466      0.060347
    
    ----------------------------------------------------------------------------------------------------------------------
    LOCAL per-client per-round metrics
    ----------------------------------------------------------------------------------------------------------------------
     round   client dataset  selected  theta_arm     theta_name                                                theta_str  gamma_power  alpha_contrast_weight  beta_contrast_sharpness  tau_clip  k_blur_kernel_size  edge_gain  blend_mix  train_loss  train_ce_loss  train_proto_loss  train_acc  val_size  val_loss_ce  val_acc  val_precision_macro  val_recall_macro  val_f1_macro  val_precision_weighted  val_recall_weighted  val_f1_weighted  val_log_loss  val_eval_time_s  val_auc_roc_macro_ovr  val_auc_class_0  val_auc_class_1  val_auc_class_2  val_auc_class_3  val_fusion_gate_mean_raw  val_fusion_gate_mean_enh  val_fusion_gate_mean_res  val_fusion_gate_entropy   reward
         1 client_0     ds1         1          2   race_texture (g=0.92, a=0.34, b=5.80, t=2.30, k=7, eg=0.10, mix=0.80)         0.92                   0.34                      5.8       2.3                   7       0.10       0.80    0.495665       0.434024          0.000000   0.715306        67     0.310189 0.895522             0.840030          0.833333      0.830598                0.899437             0.895522         0.892267      0.334955         0.918740               0.986753         0.980769         0.967213         0.999031         1.000000                  0.420704                  0.176548                  0.402748                 1.300078 0.846829
         1 client_1     ds1         1          5     race_focus (g=1.10, a=0.36, b=6.40, t=2.70, k=3, eg=0.17, mix=0.86)         1.10                   0.36                      6.4       2.7                   3       0.17       0.86    0.598768       0.575341          0.000000   0.644000        18     0.533254 0.944444             0.833333          0.969697      0.873016                0.972222             0.944444         0.952381      0.461687         0.402608                    NaN         1.000000              NaN         1.000000         1.000000                  0.049905                  0.690908                  0.259186                 1.048744 0.890873
         1 client_2     ds1         1          2   race_texture (g=0.92, a=0.34, b=5.80, t=2.30, k=7, eg=0.10, mix=0.80)         0.92                   0.34                      5.8       2.3                   7       0.10       0.80    0.573470       0.540322          0.000000   0.646465        27     0.786300 0.555556             0.558333          0.625000      0.549107                0.713580             0.555556         0.584656      0.799236         0.502400               0.908851         0.800000         0.857143         0.978261         1.000000                  0.206930                  0.347068                  0.446003                 1.516273 0.550719
         1 client_3     ds2         1          0      race_soft (g=0.95, a=0.18, b=3.80, t=2.20, k=3, eg=0.08, mix=0.72)         0.95                   0.18                      3.8       2.2                   3       0.08       0.72    0.351630       0.291320          0.000000   0.814785        86     0.276558 0.930233             0.459510          0.467836      0.463602                0.908159             0.930233         0.919006      0.280700         1.074374               0.907598         0.694118         0.961080         0.975197         1.000000                  0.405093                  0.552869                  0.042037                 1.186296 0.580259
         1 client_4     ds2         1          1  race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)         1.00                   0.24                      4.6       2.4                   5       0.10       0.78    0.383922       0.318058          0.000000   0.808349        72     0.144656 0.944444             0.728546          0.632143      0.667614                0.959174             0.944444         0.947885      0.152075         0.914244                    NaN         0.992695              NaN         1.000000         0.967164                  0.724676                  0.086218                  0.189106                 0.960379 0.736821
         1 client_5     ds2         1          1  race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)         1.00                   0.24                      4.6       2.4                   5       0.10       0.78    0.427276       0.329200          0.000000   0.779872       362     0.377492 0.839779             0.840923          0.836987      0.837848                0.838369             0.839779         0.837968      0.378620         3.745245               0.968400         0.963546         0.923627         0.993548         0.992878                  0.371130                  0.374745                  0.254125                 1.555955 0.838330
         2 client_0     ds1         1          0  race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)         1.00                   0.24                      4.6       2.4                   5       0.10       0.78    0.270231       0.191745          0.297548   0.890816        67     0.303268 0.895522             0.860119          0.838636      0.845326                0.900320             0.895522         0.893499      0.343675         0.801231               0.971018         0.970513         0.923497         0.996124         0.993939                  0.389083                  0.539330                  0.071587                 1.273872 0.857875
         2 client_1     ds1         1          4 race_edge_plus (g=1.02, a=0.32, b=6.00, t=2.60, k=3, eg=0.15, mix=0.84)         1.02                   0.32                      6.0       2.6                   3       0.15       0.84    0.308347       0.252612          0.334170   0.876000        18     0.577748 0.944444             0.750000          0.727273      0.738095                1.000000             0.944444         0.970899      0.257061         0.291094                    NaN         1.000000              NaN         1.000000         1.000000                  0.420461                  0.367143                  0.212396                 1.528342 0.789683
         2 client_2     ds1         1          1     race_sharp (g=1.04, a=0.30, b=5.20, t=2.50, k=5, eg=0.12, mix=0.82)         1.04                   0.30                      5.2       2.5                   5       0.12       0.82    0.331166       0.266220          0.326174   0.797980        27     0.381245 0.925926             0.727273          0.750000      0.738095                0.858586             0.925926         0.890653      0.365888         0.431618               0.932857         0.860000         0.871429         1.000000         1.000000                  0.424875                  0.350565                  0.224560                 1.526137 0.785053
         2 client_3     ds2         1          2    race_robust (g=1.08, a=0.22, b=4.50, t=2.80, k=5, eg=0.11, mix=0.76)         1.08                   0.22                      4.5       2.8                   5       0.11       0.76    0.237365       0.167554          0.262783   0.891892        86     0.231799 0.918605             0.699242          0.713938      0.705905                0.911945             0.918605         0.914310      0.243381         0.926399               0.951380         0.858824         0.964846         0.981851         1.000000                  0.800973                  0.143878                  0.055149                 0.811583 0.759080
         2 client_4     ds2         1          0      race_soft (g=0.95, a=0.18, b=3.80, t=2.20, k=3, eg=0.08, mix=0.72)         0.95                   0.18                      3.8       2.2                   3       0.08       0.72    0.176655       0.116972          0.249294   0.940228        72     0.054447 1.000000             1.000000          1.000000      1.000000                1.000000             1.000000         1.000000      0.059518         0.778819                    NaN         1.000000              NaN         1.000000         1.000000                  0.561961                  0.272683                  0.165356                 1.402517 1.000000
         2 client_5     ds2         1          1  race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)         1.00                   0.24                      4.6       2.4                   5       0.10       0.78    0.345645       0.244054          0.269008   0.834904       362     0.338346 0.834254             0.826127          0.831668      0.827548                0.830328             0.834254         0.830916      0.341284         3.591216               0.974539         0.969871         0.938552         0.994727         0.995004                  0.523657                  0.419547                  0.056797                 1.245312 0.829225
         3 client_0     ds1         1          2   race_texture (g=0.92, a=0.34, b=5.80, t=2.30, k=7, eg=0.10, mix=0.80)         0.92                   0.34                      5.8       2.3                   7       0.10       0.80    0.306880       0.215727          0.246711   0.882653        67     0.168489 0.925373             0.875417          0.891667      0.877173                0.933433             0.925373         0.925644      0.191508         0.805030               0.993742         0.985897         0.989071         1.000000         1.000000                  0.528846                  0.337485                  0.133670                 1.375196 0.889223
         3 client_1     ds1         1          1     race_sharp (g=1.04, a=0.30, b=5.20, t=2.50, k=5, eg=0.12, mix=0.82)         1.04                   0.30                      5.2       2.5                   5       0.12       0.82    0.527434       0.465696          0.315484   0.764000        18     0.714433 0.944444             0.833333          0.969697      0.873016                0.972222             0.944444         0.952381      0.357907         0.332201                    NaN         0.987013              NaN         1.000000         1.000000                  0.398142                  0.277466                  0.324391                 1.567941 0.890873
         3 client_2     ds1         1          2   race_texture (g=0.92, a=0.34, b=5.80, t=2.30, k=7, eg=0.10, mix=0.80)         0.92                   0.34                      5.8       2.3                   7       0.10       0.80    0.392054       0.321961          0.292840   0.772727        27     0.345524 0.962963             0.988095          0.875000      0.910569                0.964727             0.962963         0.957242      0.343294         0.388586               0.964286         0.900000         0.957143         1.000000         1.000000                  0.406086                  0.346936                  0.246978                 1.552889 0.923668
         3 client_3     ds2         1          0      race_soft (g=0.95, a=0.18, b=3.80, t=2.20, k=3, eg=0.08, mix=0.72)         0.95                   0.18                      3.8       2.2                   3       0.08       0.72    0.305651       0.195709          0.245370   0.882353        86     0.151879 0.965116             0.977679          0.981969      0.979726                0.965739             0.965116         0.965279      0.152426         0.914888               0.995047         0.988235         0.994978         0.996975         1.000000                  0.003621                  0.995331                  0.001047                 0.044520 0.976074
         3 client_4     ds2         1          1  race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)         1.00                   0.24                      4.6       2.4                   5       0.10       0.78    0.215446       0.148535          0.204009   0.915560        72     0.067659 0.986111             0.944444          0.991453      0.965368                0.988426             0.986111         0.986652      0.073510         0.797199                    NaN         1.000000              NaN         1.000000         0.997015                  0.599808                  0.317850                  0.082343                 1.247457 0.970554
         3 client_5     ds2         1          1  race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)         1.00                   0.24                      4.6       2.4                   5       0.10       0.78    0.344729       0.202675          0.221619   0.881078       362     0.163011 0.961326             0.961356          0.963916      0.962106                0.961990             0.961326         0.961125      0.165006         3.606827               0.995772         0.997364         0.992180         0.995906         0.997637                  0.616811                  0.319966                  0.063223                 1.196541 0.961911
         4 client_0     ds1         1          4 race_edge_plus (g=1.02, a=0.32, b=6.00, t=2.60, k=3, eg=0.15, mix=0.84)         1.02                   0.32                      6.0       2.6                   3       0.15       0.84    0.238195       0.159620          0.208810   0.919388        67     0.117636 0.955224             0.910714          0.925000      0.916446                0.958422             0.955224         0.956174      0.134864         0.800258               0.994982         0.993590         0.986339         1.000000         1.000000                  0.685927                  0.226635                  0.087438                 1.087010 0.926140
         4 client_1     ds1         1          0  race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)         1.00                   0.24                      4.6       2.4                   5       0.10       0.78    0.231820       0.188292          0.206036   0.884000        18     0.083317 1.000000             1.000000          1.000000      1.000000                1.000000             1.000000         1.000000      0.080814         0.295050                    NaN         1.000000              NaN         1.000000         1.000000                  0.418547                  0.300024                  0.281428                 1.527567 1.000000
    ... showing first 20 of 90 rows
    
    ======================================================================================================================
    STEP 9: FINAL EVALUATION (VALIDATION-SELECTED AFTER RANDOM TRAINING)
    ======================================================================================================================
    
    ----------------------------------------------------------------------------------------------------------------------
    Validation ranking of static presets after random training — DS1
    ----------------------------------------------------------------------------------------------------------------------
        theta_name    score  val_acc   val_f1
     race_balanced 0.977582 0.977876 0.977484
        race_sharp 0.977582 0.977876 0.977484
      race_texture 0.977582 0.977876 0.977484
       race_robust 0.977582 0.977876 0.977484
    race_edge_plus 0.977582 0.977876 0.977484
        race_focus 0.977582 0.977876 0.977484
    
    ----------------------------------------------------------------------------------------------------------------------
    Validation ranking of static presets after random training — DS2
    ----------------------------------------------------------------------------------------------------------------------
        theta_name    score  val_acc   val_f1
    race_smoothmix 0.979624 0.980095 0.979467
         race_soft 0.978646 0.979147 0.978479
     race_balanced 0.978646 0.979147 0.978479
       race_robust 0.978646 0.979147 0.978479
    
    ----------------------------------------------------------------------------------------------------------------------
    Final preprocessing strategy after random training
    ----------------------------------------------------------------------------------------------------------------------
    dataset                                         strategy        theta_names    score  val_acc   val_f1
        ds1 validation_selected_single_after_random_training  ['race_balanced'] 0.977582 0.977876 0.977484
        ds2 validation_selected_single_after_random_training ['race_smoothmix'] 0.979624 0.980095 0.979467
    
    ======================================================================================================================
    STEP 10: EXTENDED METRICS + CALIBRATION + ERROR ANALYSIS
    ======================================================================================================================
    
    ----------------------------------------------------------------------------------------------------------------------
    Extended TEST metrics (DS1 vs DS2)
    ----------------------------------------------------------------------------------------------------------------------
     dataset      acc  balanced_acc  precision_macro  recall_macro  f1_macro  precision_weighted  recall_weighted  f1_weighted  log_loss      mcc    kappa  jaccard_macro  ppv_macro  npv_macro  specificity_macro  fpr_macro  fnr_macro      ece      mce  brier_multi  auc_roc_macro_ovr  auc_class_0  auc_class_1  auc_class_2  auc_class_3
    ds1_test 0.977876      0.977824         0.978059      0.977824  0.977903            0.978109         0.977876     0.977954  0.081685 0.970520 0.970495       0.957087   0.978059   0.992620           0.992629   0.007371   0.022176 0.012069 0.724456     0.034311           0.999425     0.998950     0.999787     0.999594     0.999370
    ds2_test 0.985782      0.985447         0.985232      0.985447  0.985318            0.985844         0.985782     0.985791  0.058470 0.981015 0.981000       0.971174   0.985232   0.995287           0.995322   0.004678   0.014553 0.017825 0.274213     0.023542           0.999470     0.999778     0.998752     0.999991     0.999358
    
    ----------------------------------------------------------------------------------------------------------------------
    Classwise metrics — DS1 TEST
    ----------------------------------------------------------------------------------------------------------------------
     class_id class_name  support  tp  fp  fn  tn  prevalence      ppv      npv   recall  specificity      fpr      fnr  jaccard  balanced_acc
            0     glioma       56  54   3   2 167    0.247788 0.947368 0.988166 0.964286     0.982353 0.017647 0.035714 0.915254      0.973319
            1 meningioma       55  54   1   1 170    0.243363 0.981818 0.994152 0.981818     0.994152 0.005848 0.018182 0.964286      0.987985
            2    notumor       59  58   1   1 166    0.261062 0.983051 0.994012 0.983051     0.994012 0.005988 0.016949 0.966667      0.988531
            3  pituitary       56  55   0   1 170    0.247788 1.000000 0.994152 0.982143     1.000000 0.000000 0.017857 0.982143      0.991071
    
    ----------------------------------------------------------------------------------------------------------------------
    Classwise metrics — DS2 TEST
    ----------------------------------------------------------------------------------------------------------------------
     class_id class_name  support  tp  fp  fn  tn  prevalence      ppv      npv   recall  specificity      fpr      fnr  jaccard  balanced_acc
            0     glioma      244 241   3   3 808    0.231280 0.987705 0.996301 0.987705     0.996301 0.003699 0.012295 0.975709      0.992003
            1 meningioma      247 240   6   7 802    0.234123 0.975610 0.991347 0.971660     0.992574 0.007426 0.028340 0.948617      0.982117
            2    notumor      300 297   0   3 755    0.284360 1.000000 0.996042 0.990000     1.000000 0.000000 0.010000 0.990000      0.995000
            3  pituitary      264 262   6   2 785    0.250237 0.977612 0.997459 0.992424     0.992415 0.007585 0.007576 0.970370      0.992419
    
    ----------------------------------------------------------------------------------------------------------------------
    Top confusion pairs — DS1 TEST
    ----------------------------------------------------------------------------------------------------------------------
    true_class pred_class  count
        glioma meningioma      1
        glioma    notumor      1
    meningioma     glioma      1
       notumor     glioma      1
     pituitary     glioma      1
        glioma  pituitary      0
    meningioma  pituitary      0
    meningioma    notumor      0
       notumor meningioma      0
       notumor  pituitary      0
    
    ----------------------------------------------------------------------------------------------------------------------
    Top confusion pairs — DS2 TEST
    ----------------------------------------------------------------------------------------------------------------------
    true_class pred_class  count
    meningioma  pituitary      4
    meningioma     glioma      3
       notumor meningioma      2
        glioma meningioma      2
     pituitary meningioma      2
        glioma  pituitary      1
       notumor  pituitary      1
        glioma    notumor      0
       notumor     glioma      0
    meningioma    notumor      0
    
    ----------------------------------------------------------------------------------------------------------------------
    Calibration bins — DS1 TEST
    ----------------------------------------------------------------------------------------------------------------------
     bin_id  bin_left  bin_right  bin_confidence  bin_accuracy  bin_gap  bin_count
          0  0.000000   0.083333             NaN           NaN      NaN          0
          1  0.083333   0.166667             NaN           NaN      NaN          0
          2  0.166667   0.250000             NaN           NaN      NaN          0
          3  0.250000   0.333333             NaN           NaN      NaN          0
          4  0.333333   0.416667             NaN           NaN      NaN          0
          5  0.416667   0.500000             NaN           NaN      NaN          0
          6  0.500000   0.583333        0.536065      0.500000 0.036065          2
          7  0.583333   0.666667        0.615413      0.500000 0.115413          2
          8  0.666667   0.750000        0.724456      0.000000 0.724456          1
          9  0.750000   0.833333             NaN           NaN      NaN          0
         10  0.833333   0.916667        0.894881      1.000000 0.105119          3
         11  0.916667   1.000000        0.984473      0.990826 0.006353        218
    
    ----------------------------------------------------------------------------------------------------------------------
    Calibration bins — DS2 TEST
    ----------------------------------------------------------------------------------------------------------------------
     bin_id  bin_left  bin_right  bin_confidence  bin_accuracy  bin_gap  bin_count
          0  0.000000   0.083333             NaN           NaN      NaN          0
          1  0.083333   0.166667             NaN           NaN      NaN          0
          2  0.166667   0.250000             NaN           NaN      NaN          0
          3  0.250000   0.333333             NaN           NaN      NaN          0
          4  0.333333   0.416667             NaN           NaN      NaN          0
          5  0.416667   0.500000        0.461783      0.500000 0.038217          2
          6  0.500000   0.583333        0.550601      0.500000 0.050601          6
          7  0.583333   0.666667        0.625646      0.833333 0.207687          6
          8  0.666667   0.750000        0.713154      0.750000 0.036846          4
          9  0.750000   0.833333        0.790706      0.800000 0.009294          5
         10  0.833333   0.916667        0.889597      0.615385 0.274213         13
         11  0.916667   1.000000        0.983886      0.997056 0.013170       1019
    
    ----------------------------------------------------------------------------------------------------------------------
    VAL + TEST tables
    ----------------------------------------------------------------------------------------------------------------------
                          setting split      dataset      acc  precision_macro  recall_macro  f1_macro  precision_weighted  recall_weighted  f1_weighted  log_loss  auc_roc_macro_ovr  loss_ce  eval_time_s  balanced_acc      mcc    kappa  ppv_macro  npv_macro  specificity_macro      ece      mce  brier_multi
    ARCF-Net RandomPresetPerRound   VAL          ds1 0.977876         0.978736      0.977315  0.977484            0.978629         0.977876     0.977728  0.072627           0.999738 0.069359     4.576053           NaN      NaN      NaN        NaN        NaN                NaN      NaN      NaN          NaN
    ARCF-Net RandomPresetPerRound   VAL          ds2 0.980095         0.979379      0.979558  0.979467            0.980108         0.980095     0.980100  0.069162           0.999430 0.069113    20.722351           NaN      NaN      NaN        NaN        NaN                NaN      NaN      NaN          NaN
    ARCF-Net RandomPresetPerRound   VAL global_equal 0.978985         0.979057      0.978437  0.978475            0.979368         0.978985     0.978914  0.070894           0.999584 0.069236    12.649202           NaN      NaN      NaN        NaN        NaN                NaN      NaN      NaN          NaN
    ARCF-Net RandomPresetPerRound  TEST          ds1 0.977876         0.978059      0.977824  0.977903            0.978109         0.977876     0.977954  0.081685           0.999425 0.077651     4.999419      0.977824 0.970520 0.970495   0.978059   0.992620           0.992629 0.012069 0.724456     0.034311
    ARCF-Net RandomPresetPerRound  TEST          ds2 0.985782         0.985232      0.985447  0.985318            0.985844         0.985782     0.985791  0.058470           0.999470 0.058432    20.923018      0.985447 0.981015 0.981000   0.985232   0.995287           0.995322 0.017825 0.274213     0.023542
    ARCF-Net RandomPresetPerRound  TEST global_equal 0.981829         0.981646      0.981636  0.981611            0.981976         0.981829     0.981872  0.070078           0.999448 0.068041    12.961218           NaN      NaN      NaN        NaN        NaN                NaN      NaN      NaN          NaN
    
    Selection summary:
    - Best round: 15 | best_reward=1.1061
    - DS1 final strategy: validation_selected_single_after_random_training | names=['race_balanced']
    - DS2 final strategy: validation_selected_single_after_random_training | names=['race_smoothmix']
    
    ======================================================================================================================
    STEP 11: PREPROCESSING VALIDATION
    ======================================================================================================================
    
    ----------------------------------------------------------------------------------------------------------------------
    Preprocessing validation summary (DS1 VAL sample)
    ----------------------------------------------------------------------------------------------------------------------
                metric     mean      std       min      max
    edge_energy_before 0.041596 0.021720  0.014947 0.165471
     edge_energy_after 0.110492 0.027505  0.066604 0.242659
        entropy_before 5.820408 0.628255  3.547314 7.239670
         entropy_after 6.604196 0.583946  3.940457 7.583500
       contrast_before 0.187640 0.052597  0.101468 0.363759
        contrast_after 0.247487 0.022271  0.209219 0.344404
       edge_gain_ratio 2.891605 0.663157  1.247970 5.213966
         entropy_delta 0.783788 0.252616  0.282053 1.541243
        contrast_delta 0.059847 0.033572 -0.020711 0.124809
    
    ----------------------------------------------------------------------------------------------------------------------
    Preprocessing validation summary (DS2 VAL sample)
    ----------------------------------------------------------------------------------------------------------------------
                metric     mean      std       min      max
    edge_energy_before 0.069022 0.036905  0.016726 0.375572
     edge_energy_after 0.117599 0.036942  0.066190 0.412431
        entropy_before 6.866192 0.472092  5.135309 7.798752
         entropy_after 7.395452 0.282689  5.967234 7.873421
       contrast_before 0.233743 0.033322  0.144569 0.352019
        contrast_after 0.275728 0.018187  0.207640 0.345063
       edge_gain_ratio 1.830210 0.371183  1.086497 4.150893
         entropy_delta 0.529260 0.229139  0.057912 1.248828
        contrast_delta 0.041985 0.018363 -0.008797 0.098341
    
    ======================================================================================================================
    STEP 12: SAVING CHECKPOINT + CSV
    ======================================================================================================================
    Saved checkpoint: /kaggle/working/outputs/ARCFNet_Ablation_RandomPresetPerRound_checkpoint.pth
    Saved CSV: /kaggle/working/outputs/ARCFNet_Ablation_RandomPresetPerRound_outputs.csv
    
    DONE
    Method: ARCF-Net = Adaptive RACE-FELCM with CRAF Fusion Network
    Ablation: Random Preset Per Round, Full Participation
    Backbone: Residual Network-50
    Best round: 15
    Fixed clients => DS1=3, DS2=3, TOTAL=6
    Rounds completed: 15
    Global TEST acc: 0.9818
    Global TEST f1_macro: 0.9816
    DS1 TEST acc: 0.9779
    DS2 TEST acc: 0.9858
    DS1 final strategy: validation_selected_single_after_random_training | names=['race_balanced']
    DS2 final strategy: validation_selected_single_after_random_training | names=['race_smoothmix']


# **3. best static preset**







```python
import os
import sys
import time
import math
import copy
import hashlib
import random
import subprocess
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    log_loss, confusion_matrix, roc_auc_score,
    precision_recall_curve, average_precision_score,
    matthews_corrcoef, cohen_kappa_score, balanced_accuracy_score,
    jaccard_score
)

# ============================================================
# ARCF-Net ABLATION 3
# BEST STATIC PRESET / FULL PARTICIPATION / NO PLOTS
# ------------------------------------------------------------
# - Uses BOTH datasets
# - Kaggle-ready
# - True FL with FedAvg + FedProx + prototype sharing
# - NO RL-UCB for client count or preprocessing selection
# - Fixed client count per dataset
# - Best static preset chosen ONCE per dataset before FL
# - Preset then stays fixed for all clients and all rounds
# - Full participation every round
# - No plots generated
# ============================================================

# -------------------------
# Helpers
# -------------------------
def pip_install(pkg):
    subprocess.check_call([sys.executable, "-m", "pip", "-q", "install", pkg])

def running_on_kaggle():
    return os.path.exists("/kaggle/input")

IS_KAGGLE = running_on_kaggle()

try:
    import kagglehub
except Exception:
    try:
        pip_install("kagglehub")
        import kagglehub
    except Exception:
        kagglehub = None

try:
    from torchvision import transforms
    from torchvision.models import resnet50, ResNet50_Weights
except Exception:
    pip_install("torchvision")
    from torchvision import transforms
    from torchvision.models import resnet50, ResNet50_Weights

# -------------------------
# Reproducibility + Device
# -------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if DEVICE.type == "cuda":
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

print("=" * 118)
print("ARCF-Net ABLATION 3: BEST STATIC PRESET + FULL PARTICIPATION + NO PLOTS")
print("=" * 118)
print(f"ENV: {'KAGGLE' if IS_KAGGLE else 'NON-KAGGLE'} | DEVICE: {DEVICE} | torch={torch.__version__}")
print("=" * 118)

# -------------------------
# Configuration
# -------------------------
CFG = {
    "rounds": 15,

    # fixed clients (no RL planning)
    "fixed_clients_ds1": 3,
    "fixed_clients_ds2": 3,

    # local training
    "local_epochs": 2,
    "lr_head": 1e-3,
    "lr_backbone": 2e-4,
    "weight_decay": 5e-4,
    "warmup_epochs": 1,
    "label_smoothing": 0.02,
    "focal_gamma": 1.35,
    "grad_clip": 1.0,
    "fedprox_mu": 0.01,
    "proto_lambda": 0.12,

    # image
    "img_size": 224 if torch.cuda.is_available() else 160,
    "batch_size": 16 if torch.cuda.is_available() else 8,
    "num_workers": 2 if torch.cuda.is_available() else 2,

    # global split
    "global_val_frac": 0.15,
    "test_frac": 0.15,

    # client split
    "client_val_frac": 0.12,
    "client_tune_frac": 0.12,
    "min_per_class_per_client": 5,

    # non-iid
    "dirichlet_alpha": 0.35,

    # preprocessing / augmentation
    "use_preprocessing": True,
    "use_augmentation": True,

    # transfer learning
    "freeze_backbone_rounds": 2,
    "unfreeze_last_blocks": 2,

    # final inference
    "final_use_tta": True,

    # reward
    "reward_f1_weight": 0.75,
    "reward_acc_weight": 0.25,

    # best-round selection: equal dataset importance
    "best_round_mass_ds1": 0.50,
    "best_round_mass_ds2": 0.50,
    "best_round_min_bonus": 0.15,

    # FedAvg tempering
    "fedavg_temper": 0.50,

    # static preset search (before FL)
    "static_search_epochs": 1,
    "static_search_train_samples": 512,
    "static_search_val_samples": 256,

    # misc / sanity
    "quick_hash_subset_per_split": 300,
    "preproc_val_sample_n": 400,

    # no plots
    "make_plots": False,
    "calibration_bins": 12,
}

OUTDIR = "/kaggle/working/outputs" if IS_KAGGLE else "/content/outputs"
os.makedirs(OUTDIR, exist_ok=True)
MODEL_PATH = os.path.join(OUTDIR, "ARCFNet_Ablation_BestStaticPreset_checkpoint.pth")
CSV_PATH = os.path.join(OUTDIR, "ARCFNet_Ablation_BestStaticPreset_outputs.csv")

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], device=DEVICE).view(1, 3, 1, 1)
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225], device=DEVICE).view(1, 3, 1, 1)

labels = ["glioma", "meningioma", "notumor", "pituitary"]
label2id = {l: i for i, l in enumerate(labels)}
id2label = {i: l for l, i in label2id.items()}
NUM_CLASSES = len(labels)

METHOD_INFO = {
    "acronym": "ARCF-Net",
    "full_form": "Adaptive RACE-FELCM with CRAF Fusion Network",
    "preprocessing_full_form": "Robust Adaptive Context-Enhanced Fuzzy Edge Local Contrast Mapping",
    "fusion_full_form": "Cross-Residual Adaptive Fusion",
    "backbone_full_form": "Residual Network-50",
    "ablation_name": "Best Static Preset, Full Participation",
}

# ============================================================
# Preset banks
# ============================================================
THETA_FULLFORMS = {
    "gamma": "Robust power exponent",
    "alpha": "Local contrast enhancement weight",
    "beta": "Contrast saturation sharpness",
    "tau": "Robust clipping threshold",
    "k": "Local context blur kernel",
    "edge_gain": "Gradient structure gain",
    "blend": "Structure-preserving residual blend",
}

PRESET_BANK_DS1 = [
    ("race_balanced",  (1.00, 0.24, 4.6, 2.4, 5, 0.10, 0.78)),
    ("race_sharp",     (1.04, 0.30, 5.2, 2.5, 5, 0.12, 0.82)),
    ("race_texture",   (0.92, 0.34, 5.8, 2.3, 7, 0.10, 0.80)),
    ("race_robust",    (1.08, 0.22, 4.5, 2.8, 5, 0.11, 0.76)),
    ("race_edge_plus", (1.02, 0.32, 6.0, 2.6, 3, 0.15, 0.84)),
    ("race_focus",     (1.10, 0.36, 6.4, 2.7, 3, 0.17, 0.86)),
]

PRESET_BANK_DS2 = [
    ("race_soft",      (0.95, 0.18, 3.8, 2.2, 3, 0.08, 0.72)),
    ("race_balanced",  (1.00, 0.24, 4.6, 2.4, 5, 0.10, 0.78)),
    ("race_robust",    (1.08, 0.22, 4.5, 2.8, 5, 0.11, 0.76)),
    ("race_smoothmix", (0.90, 0.20, 4.0, 2.1, 7, 0.07, 0.70)),
]

def get_preset_by_name(bank, target_name):
    for name, theta in bank:
        if name == target_name:
            return name, theta
    raise ValueError(f"Preset {target_name} not found.")

# ============================================================
# CSV collector
# ============================================================
ALL_ROWS = []

def add_table_to_csv(df, table_name):
    if df is None or len(df) == 0:
        return
    df2 = df.copy()
    df2.insert(0, "table_name", table_name)
    for _, row in df2.iterrows():
        ALL_ROWS.append(row.to_dict())

def print_table(df, title, max_rows=12):
    print("\n" + "-" * 118)
    print(title)
    print("-" * 118)
    if df is None or len(df) == 0:
        print("[empty]")
    else:
        print(df.head(max_rows).to_string(index=False))
        if len(df) > max_rows:
            print(f"... showing first {max_rows} of {len(df)} rows")

def safe_float(x, default=np.nan):
    try:
        return float(x)
    except Exception:
        return default

def normalize_imagenet(x):
    return (x - IMAGENET_MEAN) / IMAGENET_STD

def score_metric(m):
    return (
        CFG["reward_f1_weight"] * safe_float(m.get("f1_macro"), 0.0)
        + CFG["reward_acc_weight"] * safe_float(m.get("acc"), 0.0)
    )

def frame_to_manifest(df_):
    return df_[["path", "label", "source", "filename", "y"]].to_dict(orient="list")

# ============================================================
# STEP 0: ACCESS DATASETS
# ============================================================
print("\n" + "=" * 118)
print("STEP 0: ACCESS DATASETS")
print("=" * 118)

REQ1 = {"512Glioma", "512Meningioma", "512Normal", "512Pituitary"}
REQ2 = {"glioma", "meningioma", "notumor", "pituitary"}

def norm_label(name: str):
    s = str(name).strip().lower()
    if "glioma" in s:
        return "glioma"
    if "meningioma" in s:
        return "meningioma"
    if "pituitary" in s:
        return "pituitary"
    if "normal" in s or "no_tumor" in s or "no tumor" in s or "notumor" in s:
        return "notumor"
    return None

def find_root_with_required_class_dirs(base_dir, required_set, prefer_raw=True):
    candidates = []
    for root, dirs, _ in os.walk(base_dir):
        if required_set.issubset(set(dirs)):
            candidates.append(root)
    if not candidates:
        return None

    def score(p):
        pl = p.lower()
        sc = 0
        if prefer_raw:
            if "raw data" in pl:
                sc += 7
            if os.path.basename(p).lower() == "raw":
                sc += 7
            if "/raw/" in pl or "\\raw\\" in pl:
                sc += 3
            if "augmented" in pl:
                sc -= 20
        sc -= 0.0001 * len(p)
        return sc

    return max(candidates, key=score)

def search_kaggle_input_for_root(required_set, prefer_raw=True):
    if not os.path.exists("/kaggle/input"):
        return None
    return find_root_with_required_class_dirs("/kaggle/input", required_set, prefer_raw=prefer_raw)

def try_kagglehub_download(slug, required_set, prefer_raw=True):
    if kagglehub is None:
        return None, None
    try:
        base = kagglehub.dataset_download(slug)
        root = find_root_with_required_class_dirs(base, required_set, prefer_raw=prefer_raw)
        return base, root
    except Exception:
        return None, None

DS1_BASE = None
DS2_BASE = None

DS1_ROOT = search_kaggle_input_for_root(REQ1, prefer_raw=True)
DS2_ROOT = search_kaggle_input_for_root(REQ2, prefer_raw=False)

if DS1_ROOT is None:
    DS1_BASE, DS1_ROOT = try_kagglehub_download(
        "orvile/pmram-bangladeshi-brain-cancer-mri-dataset",
        REQ1,
        prefer_raw=True,
    )

if DS2_ROOT is None:
    DS2_BASE, DS2_ROOT = try_kagglehub_download(
        "yassinebazgour/preprocessed-brain-mri-scans-for-tumors-detection",
        REQ2,
        prefer_raw=False,
    )

if DS1_ROOT is None:
    raise RuntimeError(
        "Could not locate DS1 under /kaggle/input. Add "
        "'orvile/pmram-bangladeshi-brain-cancer-mri-dataset'."
    )
if DS2_ROOT is None:
    raise RuntimeError(
        "Could not locate DS2 under /kaggle/input. Add "
        "'yassinebazgour/preprocessed-brain-mri-scans-for-tumors-detection'."
    )

print(f"Dataset-1 RAW root detected:\n  {DS1_ROOT}")
print(f"Dataset-2 root detected:\n  {DS2_ROOT}")

# ============================================================
# STEP 1: BUILD MANIFESTS
# ============================================================
print("\n" + "=" * 118)
print("STEP 1: BUILD DATA MANIFESTS")
print("=" * 118)

def list_images_under_class_root(class_root, class_dir_name):
    class_dir = os.path.join(class_root, class_dir_name)
    out = []
    for r, _, files in os.walk(class_dir):
        for fn in files:
            if fn.lower().endswith(IMG_EXTS):
                out.append(os.path.join(r, fn))
    return out

def build_df_from_root(ds_root, class_dirs, source_name):
    rows = []
    for c in class_dirs:
        lab = norm_label(c)
        imgs = list_images_under_class_root(ds_root, c)
        print(f"{source_name}: {c} -> {lab} | {len(imgs)} images")
        for p in imgs:
            rows.append({"path": p, "label": lab, "source": source_name})
    dfm = pd.DataFrame(rows).dropna().reset_index(drop=True)
    dfm["path"] = dfm["path"].astype(str)
    dfm["label"] = dfm["label"].astype(str)
    dfm["source"] = dfm["source"].astype(str)
    dfm = dfm.drop_duplicates(subset=["path"]).reset_index(drop=True)
    dfm["filename"] = dfm["path"].apply(os.path.basename)
    return dfm

df1 = build_df_from_root(DS1_ROOT, ["512Glioma", "512Meningioma", "512Normal", "512Pituitary"], "ds1_raw")
df2 = build_df_from_root(DS2_ROOT, ["glioma", "meningioma", "notumor", "pituitary"], "ds2")

def enforce_labels(df_):
    df_ = df_.copy()
    df_["label"] = df_["label"].astype(str).str.strip().str.lower()
    df_ = df_[df_["label"].isin(set(labels))].reset_index(drop=True)
    df_["y"] = df_["label"].map(label2id).astype(int)
    return df_

df1 = enforce_labels(df1)
df2 = enforce_labels(df2)

print("\nDataset-1 images:", len(df1))
print(df1["label"].value_counts().reindex(labels, fill_value=0).to_string())
print("\nDataset-2 images:", len(df2))
print(df2["label"].value_counts().reindex(labels, fill_value=0).to_string())

# ============================================================
# STEP 2: TRAIN / VAL / TEST SPLIT
# ============================================================
print("\n" + "=" * 118)
print("STEP 2: TRAIN / VAL / TEST SPLIT")
print("=" * 118)

def split_dataset(df_):
    train_df, temp_df = train_test_split(
        df_,
        test_size=(CFG["global_val_frac"] + CFG["test_frac"]),
        stratify=df_["y"],
        random_state=SEED,
    )
    val_rel = CFG["global_val_frac"] / (CFG["global_val_frac"] + CFG["test_frac"])
    val_df, test_df = train_test_split(
        temp_df,
        test_size=(1 - val_rel),
        stratify=temp_df["y"],
        random_state=SEED,
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)

train1, val1, test1 = split_dataset(df1)
train2, val2, test2 = split_dataset(df2)

print(f"DS1 TRAIN: {len(train1)} | VAL: {len(val1)} | TEST: {len(test1)}")
print(f"DS2 TRAIN: {len(train2)} | VAL: {len(val2)} | TEST: {len(test2)}")

# ============================================================
# STEP 2.5: SANITY / LEAKAGE CHECKS
# ============================================================
print("\n" + "=" * 118)
print("STEP 2.5: SANITY / LEAKAGE CHECKS")
print("=" * 118)

def split_overlap_checks(train_df, val_df, test_df):
    tr = set(train_df["path"].tolist())
    va = set(val_df["path"].tolist())
    te = set(test_df["path"].tolist())
    checks = {
        "path_overlap_train_val": len(tr.intersection(va)),
        "path_overlap_train_test": len(tr.intersection(te)),
        "path_overlap_val_test": len(va.intersection(te)),
        "unique_paths_train": len(tr),
        "unique_paths_val": len(va),
        "unique_paths_test": len(te),
    }
    trf = set(train_df["filename"].tolist())
    vaf = set(val_df["filename"].tolist())
    tef = set(test_df["filename"].tolist())
    checks.update({
        "filename_overlap_train_val": len(trf.intersection(vaf)),
        "filename_overlap_train_test": len(trf.intersection(tef)),
        "filename_overlap_val_test": len(vaf.intersection(tef)),
    })
    return checks

def md5_file(path, max_bytes=2_000_000):
    h = hashlib.md5()
    try:
        with open(path, "rb") as f:
            h.update(f.read(max_bytes))
        return h.hexdigest()
    except Exception:
        return None

def quick_hash_subset(frame, n=300):
    n = min(n, len(frame))
    if n <= 0:
        return set()
    idx = np.random.choice(len(frame), size=n, replace=False)
    hashes = []
    for i in idx:
        hv = md5_file(frame.iloc[i]["path"])
        if hv is not None:
            hashes.append(hv)
    return set(hashes)

def leakage_report(name, tr, va, te):
    over = split_overlap_checks(tr, va, te)
    leak_df = pd.DataFrame([over])

    n_hash = int(CFG["quick_hash_subset_per_split"])
    trh = quick_hash_subset(tr, n_hash)
    vah = quick_hash_subset(va, n_hash)
    teh = quick_hash_subset(te, n_hash)

    hash_over = {
        "subset_hash_train_val": len(trh.intersection(vah)),
        "subset_hash_train_test": len(trh.intersection(teh)),
        "subset_hash_val_test": len(vah.intersection(teh)),
        "subset_hash_n_train": len(trh),
        "subset_hash_n_val": len(vah),
        "subset_hash_n_test": len(teh),
    }
    leak_df = pd.concat([leak_df, pd.DataFrame([hash_over])], axis=1)
    print_table(leak_df, f"Leakage / Sanity Summary — {name}", max_rows=5)
    add_table_to_csv(leak_df, f"leakage_sanity_{name}")

leakage_report("ds1", train1, val1, test1)
leakage_report("ds2", train2, val2, test2)

# ============================================================
# STEP 3: NON-IID CLIENT PARTITIONING (FIXED)
# ============================================================
print("\n" + "=" * 118)
print("STEP 3: FIXED NON-IID CLIENT PARTITIONING")
print("=" * 118)

def make_clients_non_iid(train_df, n_clients, num_classes, min_per_class=5, alpha=0.35):
    y = train_df["y"].values
    idx_by_class = {c: np.where(y == c)[0].tolist() for c in range(num_classes)}
    for c in idx_by_class:
        random.shuffle(idx_by_class[c])

    client_indices = [[] for _ in range(n_clients)]

    for c in range(num_classes):
        idxs = idx_by_class[c]
        feasible = min(min_per_class, max(1, len(idxs) // n_clients))
        for k in range(n_clients):
            take = idxs[:feasible]
            idxs = idxs[feasible:]
            client_indices[k].extend(take)
        idx_by_class[c] = idxs

    for c in range(num_classes):
        idxs = idx_by_class[c]
        if len(idxs) == 0:
            continue
        props = np.random.dirichlet([alpha] * n_clients)
        counts = (props * len(idxs)).astype(int)
        diff = len(idxs) - counts.sum()
        counts[np.argmax(props)] += diff

        start = 0
        for k in range(n_clients):
            client_indices[k].extend(idxs[start:start + counts[k]])
            start += counts[k]

    for k in range(n_clients):
        random.shuffle(client_indices[k])

    return client_indices

def robust_client_splits(train_df, indices, val_frac, tune_frac):
    idxs = np.array(indices, dtype=int)
    if len(idxs) < 3:
        return idxs.tolist(), idxs.tolist(), idxs.tolist()

    yk = train_df.loc[idxs, "y"].values

    if len(np.unique(yk)) < 2 or len(idxs) < 20:
        n_tune = max(1, int(round(len(idxs) * tune_frac)))
        n_tune = min(n_tune, max(1, len(idxs) - 2))
        tune_idx = idxs[:n_tune]
        rem_idx = idxs[n_tune:]
    else:
        rem_idx, tune_idx = train_test_split(
            idxs,
            test_size=tune_frac,
            stratify=yk,
            random_state=SEED,
        )

    if len(rem_idx) < 2:
        return rem_idx.tolist(), tune_idx.tolist(), rem_idx.tolist()

    yk2 = train_df.loc[rem_idx, "y"].values
    if len(np.unique(yk2)) < 2 or len(rem_idx) < 12:
        n_val = max(1, int(round(len(rem_idx) * val_frac)))
        n_val = min(n_val, max(1, len(rem_idx) - 1))
        val_idx = rem_idx[:n_val]
        train_idx = rem_idx[n_val:]
    else:
        train_idx, val_idx = train_test_split(
            rem_idx,
            test_size=val_frac,
            stratify=yk2,
            random_state=SEED,
        )

    if len(train_idx) == 0:
        train_idx = val_idx[:]
    if len(val_idx) == 0:
        val_idx = train_idx[:1]
    return train_idx.tolist(), tune_idx.tolist(), val_idx.tolist()

n_clients_ds1 = int(CFG["fixed_clients_ds1"])
n_clients_ds2 = int(CFG["fixed_clients_ds2"])

client_indices_ds1 = make_clients_non_iid(
    train1,
    n_clients=n_clients_ds1,
    num_classes=NUM_CLASSES,
    min_per_class=CFG["min_per_class_per_client"],
    alpha=CFG["dirichlet_alpha"],
)
client_indices_ds2 = make_clients_non_iid(
    train2,
    n_clients=n_clients_ds2,
    num_classes=NUM_CLASSES,
    min_per_class=CFG["min_per_class_per_client"],
    alpha=CFG["dirichlet_alpha"],
)

print(f"Fixed clients for DS1: {n_clients_ds1}")
print(f"Fixed clients for DS2: {n_clients_ds2}")

# ============================================================
# STEP 4: DATA LOADERS
# ============================================================
print("\n" + "=" * 118)
print("STEP 4: DATA LOADERS")
print("=" * 118)

def load_rgb(path):
    try:
        return Image.open(path).convert("RGB")
    except Exception:
        return Image.new("RGB", (CFG["img_size"], CFG["img_size"]), (128, 128, 128))

EVAL_TFMS = transforms.Compose([
    transforms.Resize((CFG["img_size"], CFG["img_size"])),
    transforms.ToTensor(),
])

TRAIN_TFMS = transforms.Compose([
    transforms.Resize((CFG["img_size"], CFG["img_size"])),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=12),
    transforms.RandomAffine(degrees=0, translate=(0.04, 0.04), scale=(0.96, 1.04)),
    transforms.ColorJitter(brightness=0.10, contrast=0.10),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.6)),
    transforms.ToTensor(),
])

class MRIDataset(Dataset):
    def __init__(self, frame, indices=None, tfms=None, source_id=0):
        self.df = frame
        self.indices = indices if indices is not None else list(range(len(frame)))
        self.tfms = tfms
        self.source_id = int(source_id)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        j = self.indices[i]
        row = self.df.iloc[j]
        img = load_rgb(row["path"])
        x = self.tfms(img) if self.tfms is not None else transforms.ToTensor()(img)
        y = int(row["y"])
        return x, y, row["path"], self.source_id

def make_weighted_sampler(frame, indices, num_classes):
    if len(indices) == 0:
        return None
    ys = frame.loc[indices, "y"].values
    class_counts = np.bincount(ys, minlength=num_classes)
    class_weights = 1.0 / np.sqrt(np.clip(class_counts, 1, None))
    sample_weights = class_weights[ys]
    return WeightedRandomSampler(
        weights=torch.DoubleTensor(sample_weights),
        num_samples=len(sample_weights),
        replacement=True,
    )

def make_loader(frame, indices, bs, tfms, shuffle=False, sampler=None, source_id=0):
    ds = MRIDataset(frame, indices=indices, tfms=tfms, source_id=source_id)
    return DataLoader(
        ds,
        batch_size=bs,
        shuffle=(shuffle and sampler is None),
        sampler=sampler,
        num_workers=CFG["num_workers"],
        pin_memory=(DEVICE.type == "cuda"),
        drop_last=False,
        persistent_workers=(CFG["num_workers"] > 0),
    )

def sample_subset_indices(frame, max_samples, seed):
    idx = np.arange(len(frame))
    if len(idx) <= max_samples:
        return idx.tolist()
    sampled, _ = train_test_split(
        idx,
        train_size=max_samples,
        stratify=frame["y"].values,
        random_state=seed,
    )
    return sampled.tolist()

clients = []
gid = 0

for ds_name, df_src, client_indices, source_id in [
    ("ds1", train1, client_indices_ds1, 0),
    ("ds2", train2, client_indices_ds2, 1),
]:
    for local_id, idxs in enumerate(client_indices):
        tr_idx, tune_idx, val_idx = robust_client_splits(df_src, idxs, CFG["client_val_frac"], CFG["client_tune_frac"])
        sampler = make_weighted_sampler(df_src, tr_idx, NUM_CLASSES)

        train_loader = make_loader(
            df_src, tr_idx, CFG["batch_size"], TRAIN_TFMS,
            shuffle=(sampler is None), sampler=sampler, source_id=source_id
        )
        tune_loader = make_loader(
            df_src,
            tune_idx if len(tune_idx) else tr_idx[:max(1, min(len(tr_idx), CFG["batch_size"]))],
            CFG["batch_size"], EVAL_TFMS, shuffle=False, sampler=None, source_id=source_id
        )
        val_loader = make_loader(
            df_src,
            val_idx if len(val_idx) else tr_idx[:max(1, min(len(tr_idx), CFG["batch_size"]))],
            CFG["batch_size"], EVAL_TFMS, shuffle=False, sampler=None, source_id=source_id
        )
        proto_loader = make_loader(
            df_src, tr_idx, CFG["batch_size"], EVAL_TFMS,
            shuffle=False, sampler=None, source_id=source_id
        )

        counts = df_src.loc[tr_idx, "label"].value_counts().reindex(labels, fill_value=0)

        clients.append({
            "gid": gid,
            "local_id": local_id,
            "dataset": ds_name,
            "source_id": source_id,
            "train_loader": train_loader,
            "tune_loader": tune_loader,
            "val_loader": val_loader,
            "proto_loader": proto_loader,
            "n_train": len(tr_idx),
            "n_tune": len(tune_idx),
            "n_val": len(val_idx),
            "train_indices": tr_idx,
            "tune_indices": tune_idx,
            "val_indices": val_idx,
            "class_counts": counts.to_dict(),
        })

        print(f"{ds_name} | client_{gid} | train={len(tr_idx)} | tune={len(tune_idx)} | val={len(val_idx)}")
        gid += 1

CLIENTS_TOTAL = len(clients)

dist_rows = []
for c in clients:
    row = {
        "client": f"client_{c['gid']}",
        "dataset": c["dataset"],
        "total_train": c["n_train"],
        "total_tune": c["n_tune"],
        "total_val": c["n_val"],
    }
    row.update({lab: int(c["class_counts"].get(lab, 0)) for lab in labels})
    dist_rows.append(row)

dist_df = pd.DataFrame(dist_rows)
print_table(dist_df, "Static-preset client class distribution", max_rows=20)
add_table_to_csv(dist_df, "static_preset_client_distribution")

val_loader_ds1 = make_loader(val1, list(range(len(val1))), CFG["batch_size"], EVAL_TFMS, shuffle=False, source_id=0)
val_loader_ds2 = make_loader(val2, list(range(len(val2))), CFG["batch_size"], EVAL_TFMS, shuffle=False, source_id=1)
test_loader_ds1 = make_loader(test1, list(range(len(test1))), CFG["batch_size"], EVAL_TFMS, shuffle=False, source_id=0)
test_loader_ds2 = make_loader(test2, list(range(len(test2))), CFG["batch_size"], EVAL_TFMS, shuffle=False, source_id=1)

print(f"Augmentation: {'ON' if CFG['use_augmentation'] else 'OFF'}")
print(f"Preprocessing: {'ON' if CFG['use_preprocessing'] else 'OFF'}")
print(f"Total clients: {CLIENTS_TOTAL}")

# ============================================================
# STEP 5: PREPROCESSING — RACE-FELCM
# ============================================================
print("\n" + "=" * 118)
print("STEP 5: PREPROCESSING — RACE-FELCM")
print("=" * 118)

class RACEFELCM(nn.Module):
    def __init__(self, gamma=1.0, alpha=0.24, beta=4.5, tau=2.4, blur_k=5, edge_gain=0.10, blend=0.78):
        super().__init__()
        self.gamma = float(gamma)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.tau = float(tau)
        self.blur_k = int(blur_k)
        self.edge_gain = float(edge_gain)
        self.blend = float(blend)

        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        lap = torch.tensor([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=torch.float32)

        self.register_buffer("sobel_x", sobel_x.view(1, 1, 3, 3))
        self.register_buffer("sobel_y", sobel_y.view(1, 1, 3, 3))
        self.register_buffer("lap", lap.view(1, 1, 3, 3))

    def forward(self, x):
        eps = 1e-6

        gray0 = x.mean(dim=1, keepdim=True)
        mu0 = F.avg_pool2d(F.pad(gray0, (1, 1, 1, 1), mode="reflect"), 3, stride=1)
        var0 = F.avg_pool2d(F.pad((gray0 - mu0).pow(2), (1, 1, 1, 1), mode="reflect"), 3, stride=1)
        tex = torch.sqrt(var0 + eps)
        tex_norm = tex / (tex.amax(dim=(2, 3), keepdim=True) + eps)
        flat_gate = 1.0 - tex_norm

        x_smooth = F.avg_pool2d(F.pad(x, (1, 1, 1, 1), mode="reflect"), 3, stride=1)
        x_dn = x * (1.0 - 0.18 * flat_gate) + x_smooth * (0.18 * flat_gate)

        mu = x_dn.mean(dim=(2, 3), keepdim=True)
        sd = x_dn.std(dim=(2, 3), keepdim=True).clamp_min(eps)
        z = ((x_dn - mu) / sd).clamp(-self.tau, self.tau)
        z = torch.sign(z) * torch.pow(z.abs().clamp_min(eps), self.gamma)

        gray = z.mean(dim=1, keepdim=True)
        k = self.blur_k if self.blur_k % 2 == 1 else self.blur_k + 1
        pad = k // 2

        local_mean = F.avg_pool2d(F.pad(gray, (pad, pad, pad, pad), mode="reflect"), k, stride=1)
        local_var = F.avg_pool2d(F.pad((gray - local_mean).pow(2), (pad, pad, pad, pad), mode="reflect"), k, stride=1)
        local_std = torch.sqrt(local_var + eps)
        local_norm = (gray - local_mean) / (local_std + eps)
        contrast_field = torch.tanh(self.beta * local_norm)

        gx = F.conv2d(F.pad(gray, (1, 1, 1, 1), mode="reflect"), self.sobel_x)
        gy = F.conv2d(F.pad(gray, (1, 1, 1, 1), mode="reflect"), self.sobel_y)
        edge = torch.sqrt(gx * gx + gy * gy + eps)
        edge = edge / (edge.amax(dim=(2, 3), keepdim=True) + eps)

        lap = F.conv2d(F.pad(gray, (1, 1, 1, 1), mode="reflect"), self.lap).abs()
        lap = lap / (lap.amax(dim=(2, 3), keepdim=True) + eps)
        coherence = torch.sigmoid(4.0 * (edge - 0.5 * lap))

        enhanced = z + self.alpha * contrast_field + self.edge_gain * edge * coherence

        mn = enhanced.amin(dim=(2, 3), keepdim=True)
        mx = enhanced.amax(dim=(2, 3), keepdim=True)
        enhanced = (enhanced - mn) / (mx - mn + eps)

        out = self.blend * enhanced + (1.0 - self.blend) * x
        return out.clamp(0, 1)

def theta_to_module(theta):
    return RACEFELCM(*theta)

def theta_to_vec(theta, batch_size):
    gamma, alpha, beta, tau, blur_k, edge_gain, blend = theta
    theta = torch.tensor(
        [gamma, alpha, beta / 8.0, tau / 4.0, blur_k / 7.0, edge_gain, blend],
        device=DEVICE,
        dtype=torch.float32
    )
    return theta.unsqueeze(0).repeat(batch_size, 1)

def theta_str(theta):
    if theta is None:
        return "None"
    g, a, b, t, k, eg, m = theta
    return f"(g={g:.2f}, a={a:.2f}, b={b:.2f}, t={t:.2f}, k={k}, eg={eg:.2f}, mix={m:.2f})"

# ============================================================
# STEP 6: MODEL — ResNet-50 + CRAF Fusion
# ============================================================
print("\n" + "=" * 118)
print("STEP 6: MODEL — ResNet-50 + CRAF Fusion")
print("=" * 118)

class CompactProjector(nn.Module):
    def __init__(self, in_dim, out_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            nn.Dropout(0.10),
        )

    def forward(self, x):
        return self.net(x)

class CRAFFusion(nn.Module):
    def __init__(self, in_dim, fuse_dim=256, cond_dim=64):
        super().__init__()
        self.proj_raw = CompactProjector(in_dim, fuse_dim)
        self.proj_enh = CompactProjector(in_dim, fuse_dim)
        self.proj_res = CompactProjector(in_dim, fuse_dim)

        self.router = nn.Sequential(
            nn.Linear(fuse_dim * 6 + cond_dim, fuse_dim),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(fuse_dim, 3),
        )

        self.residual_refine = nn.Sequential(
            nn.Linear(fuse_dim * 2 + cond_dim, fuse_dim),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(fuse_dim, fuse_dim),
        )

        self.out_norm = nn.LayerNorm(fuse_dim)

    def forward(self, f_raw, f_enh, f_res, cond):
        fr = self.proj_raw(f_raw)
        fe = self.proj_enh(f_enh)
        fs = self.proj_res(f_res)

        d_re = torch.abs(fr - fe)
        d_rs = torch.abs(fr - fs)
        d_es = torch.abs(fe - fs)

        router_in = torch.cat([fr, fe, fs, d_re, d_rs, d_es, cond], dim=1)
        gates = torch.softmax(self.router(router_in), dim=1)

        routed = gates[:, 0:1] * fr + gates[:, 1:2] * fe + gates[:, 2:3] * fs
        disagreement = (d_re + d_rs + d_es) / 3.0
        refine = self.residual_refine(torch.cat([routed, disagreement, cond], dim=1))
        fused = self.out_norm(routed + refine)

        return fused, gates

class ResNet50CRAF(nn.Module):
    def __init__(self, num_classes, cond_dim=64, fuse_dim=256, embed_dim=256, pretrained=True):
        super().__init__()
        try:
            weights = ResNet50_Weights.DEFAULT if pretrained else None
        except Exception:
            weights = None

        net = resnet50(weights=weights)
        self.backbone = nn.Sequential(
            net.conv1,
            net.bn1,
            net.relu,
            net.maxpool,
            net.layer1,
            net.layer2,
            net.layer3,
            net.layer4,
        )
        self.backbone_dim = net.fc.in_features
        self.pool = nn.AdaptiveAvgPool2d(1)

        self.theta_mlp = nn.Sequential(
            nn.Linear(7, cond_dim),
            nn.GELU(),
            nn.Linear(cond_dim, cond_dim),
        )
        self.source_emb = nn.Embedding(2, cond_dim)
        self.cond_norm = nn.LayerNorm(cond_dim)

        self.fusion = CRAFFusion(self.backbone_dim, fuse_dim=fuse_dim, cond_dim=cond_dim)

        self.neck = nn.Sequential(
            nn.LayerNorm(fuse_dim),
            nn.Linear(fuse_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(0.20),
        )
        self.classifier = nn.Linear(embed_dim, num_classes)
        self.embed_dim = embed_dim

    def _encode(self, x):
        f = self.backbone(x)
        f = self.pool(f).flatten(1)
        return f

    def forward(self, x_raw_n, x_enh_n, x_res_n, theta_vec, source_id, return_extra=False):
        cond = self.theta_mlp(theta_vec) + self.source_emb(source_id)
        cond = self.cond_norm(cond)

        f_raw = self._encode(x_raw_n)
        f_enh = self._encode(x_enh_n)
        f_res = self._encode(x_res_n)

        fused, gates = self.fusion(f_raw, f_enh, f_res, cond)
        embed = self.neck(fused)
        logits = self.classifier(embed)

        if return_extra:
            return logits, embed, gates
        return logits

def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

def set_trainable_for_round(model, rnd):
    for p in model.backbone.parameters():
        p.requires_grad = False

    for n, p in model.named_parameters():
        if not n.startswith("backbone."):
            p.requires_grad = True

    if rnd > CFG["freeze_backbone_rounds"]:
        blocks = list(model.backbone.children())
        tail_blocks = blocks[-CFG["unfreeze_last_blocks"]:]
        for blk in tail_blocks:
            for p in blk.parameters():
                p.requires_grad = True

def freeze_backbone_bn_stats(model):
    for m in model.backbone.modules():
        if isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm)):
            m.eval()
            for p in m.parameters():
                p.requires_grad = False

def make_optimizer(model):
    head_params, bb_params = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if n.startswith("backbone."):
            bb_params.append(p)
        else:
            head_params.append(p)

    groups = []
    if head_params:
        groups.append({"params": head_params, "lr": CFG["lr_head"]})
    if bb_params:
        groups.append({"params": bb_params, "lr": CFG["lr_backbone"]})

    return torch.optim.AdamW(groups, weight_decay=CFG["weight_decay"])

def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
    def lr_lambda(step):
        if step < num_warmup_steps:
            return float(step) / float(max(1, num_warmup_steps))
        progress = float(step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

global_model = ResNet50CRAF(
    num_classes=NUM_CLASSES,
    cond_dim=64,
    fuse_dim=256,
    embed_dim=256,
    pretrained=True,
).to(DEVICE)

set_trainable_for_round(global_model, rnd=1)
total_params, trainable_params = count_params(global_model)
INITIAL_GLOBAL_STATE = copy.deepcopy(global_model.state_dict())

print("Backbone: ResNet-50 | pretrained_loaded=True")
print(f"Total params: {total_params:,}")
print(f"Trainable params: {trainable_params:,} ({(100.0 * trainable_params / total_params):.2f}%)")

# ============================================================
# STEP 7: LOSSES + PROTOTYPE SHARING + EVAL HELPERS
# ============================================================
print("\n" + "=" * 118)
print("STEP 7: LOSSES + PROTOTYPE SHARING + EVAL HELPERS")
print("=" * 118)

counts1 = train1["y"].value_counts().sort_index().reindex(range(NUM_CLASSES), fill_value=0).values
counts2 = train2["y"].value_counts().sort_index().reindex(range(NUM_CLASSES), fill_value=0).values
counts = counts1 + counts2
w = (counts.sum() / np.clip(counts, 1, None)).astype(np.float32)
w = w / max(1e-6, w.mean())
class_w = torch.tensor(w, device=DEVICE)

class ClassBalancedFocalLoss(nn.Module):
    def __init__(self, class_weights, gamma=1.5, label_smoothing=0.0):
        super().__init__()
        self.class_weights = class_weights
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, logits, target):
        ce = F.cross_entropy(
            logits,
            target,
            weight=self.class_weights,
            reduction="none",
            label_smoothing=self.label_smoothing,
        )
        pt = torch.exp(-ce)
        focal = (1.0 - pt).pow(self.gamma) * ce
        return focal.mean()

criterion = ClassBalancedFocalLoss(
    class_weights=class_w,
    gamma=CFG["focal_gamma"],
    label_smoothing=CFG["label_smoothing"],
)

scaler = torch.amp.GradScaler("cuda", enabled=(DEVICE.type == "cuda")) if DEVICE.type == "cuda" else None

def fedprox_term(local_model, global_model):
    loss = 0.0
    for p_local, p_global in zip(local_model.parameters(), global_model.parameters()):
        loss = loss + ((p_local - p_global.detach()) ** 2).sum()
    return loss

def prototype_alignment_loss(embed, y, global_prototypes):
    if global_prototypes is None:
        return torch.tensor(0.0, device=embed.device)

    proto = global_prototypes["proto"]
    mask = global_prototypes["mask"]
    valid = mask[y]
    if not valid.any():
        return torch.tensor(0.0, device=embed.device)

    emb_n = F.normalize(embed[valid], dim=1)
    ref_n = F.normalize(proto[y[valid]], dim=1)
    return (1.0 - (emb_n * ref_n).sum(dim=1)).mean()

@torch.no_grad()
def compute_prototype_payload(model, loader, preproc, theta):
    model.eval()
    freeze_backbone_bn_stats(model)
    preproc.eval()

    sums = torch.zeros(NUM_CLASSES, model.embed_dim, device=DEVICE)
    counts = torch.zeros(NUM_CLASSES, device=DEVICE)

    for x, y, _, source_id in loader:
        x = x.to(DEVICE, non_blocking=True)
        y = y.to(DEVICE, non_blocking=True)
        source_id = source_id.to(DEVICE, non_blocking=True)

        x_enh = preproc(x)
        x_res = torch.abs(x_enh - x)

        x_raw_n = normalize_imagenet(x)
        x_enh_n = normalize_imagenet(x_enh)
        x_res_n = normalize_imagenet(x_res)
        theta_vec = theta_to_vec(theta, x.size(0))

        _, embed, _ = model(x_raw_n, x_enh_n, x_res_n, theta_vec, source_id, return_extra=True)

        for c in range(NUM_CLASSES):
            m = (y == c)
            if m.any():
                sums[c] += embed[m].sum(dim=0)
                counts[c] += m.sum()

    return sums.detach().cpu(), counts.detach().cpu()

def aggregate_prototypes(payloads, embed_dim):
    if len(payloads) == 0:
        return None
    sum_all = torch.zeros(NUM_CLASSES, embed_dim)
    cnt_all = torch.zeros(NUM_CLASSES)
    for s, c in payloads:
        sum_all += s
        cnt_all += c
    proto = sum_all / cnt_all.unsqueeze(1).clamp_min(1.0)
    mask = cnt_all > 0
    return {
        "proto": proto.to(DEVICE),
        "mask": mask.to(DEVICE),
        "counts": cnt_all.to(DEVICE),
    }

def gate_entropy(g):
    eps = 1e-6
    p = g.clamp(eps, 1 - eps)
    return -(p * torch.log2(p)).sum(dim=1)

@torch.no_grad()
def _auc_metrics(y_true, p_pred, num_classes):
    out = {}
    try:
        out["auc_roc_macro_ovr"] = float(roc_auc_score(y_true, p_pred, multi_class="ovr", average="macro"))
    except Exception:
        out["auc_roc_macro_ovr"] = np.nan

    for c in range(num_classes):
        try:
            yc = (y_true == c).astype(int)
            if yc.sum() > 0 and yc.sum() < len(yc):
                out[f"auc_class_{c}"] = float(roc_auc_score(yc, p_pred[:, c]))
            else:
                out[f"auc_class_{c}"] = np.nan
        except Exception:
            out[f"auc_class_{c}"] = np.nan
    return out

@torch.no_grad()
def predict_probs_single_theta(model, x, source_id, preproc, theta, use_tta=False):
    model.eval()
    freeze_backbone_bn_stats(model)
    preproc.eval()

    probs_acc = None
    versions = [x]
    if use_tta:
        versions.append(torch.flip(x, dims=[3]))

    for xv in versions:
        x_enh = preproc(xv)
        x_res = torch.abs(x_enh - xv)

        x_raw_n = normalize_imagenet(xv)
        x_enh_n = normalize_imagenet(x_enh)
        x_res_n = normalize_imagenet(x_res)
        theta_vec = theta_to_vec(theta, xv.size(0))

        logits = model(x_raw_n, x_enh_n, x_res_n, theta_vec, source_id)
        probs = torch.softmax(logits, dim=1)
        probs_acc = probs if probs_acc is None else (probs_acc + probs)

    probs_acc = probs_acc / len(versions)
    return probs_acc

@torch.no_grad()
def evaluate_full(model, loader, preproc, theta, return_gates=False, use_tta=False):
    t0 = time.time()
    model.eval()
    freeze_backbone_bn_stats(model)
    preproc.eval()

    all_y, all_p, all_loss = [], [], []
    gate_stats = []

    for x, y, _, source_id in loader:
        x = x.to(DEVICE, non_blocking=True)
        y = y.to(DEVICE, non_blocking=True)
        source_id = source_id.to(DEVICE, non_blocking=True)

        x_enh = preproc(x)
        x_res = torch.abs(x_enh - x)

        x_raw_n = normalize_imagenet(x)
        x_enh_n = normalize_imagenet(x_enh)
        x_res_n = normalize_imagenet(x_res)
        theta_vec = theta_to_vec(theta, x.size(0))

        if return_gates and (not use_tta):
            logits, _, gates = model(x_raw_n, x_enh_n, x_res_n, theta_vec, source_id, return_extra=True)
            gate_stats.append(gates.detach().cpu())
            probs = torch.softmax(logits, dim=1)
        else:
            probs = predict_probs_single_theta(model, x, source_id, preproc, theta, use_tta=use_tta)
            logits = torch.log(probs.clamp_min(1e-8))

        loss = F.cross_entropy(logits, y)

        all_loss.append(float(loss.item()))
        all_y.append(y.detach().cpu().numpy())
        all_p.append(probs.detach().cpu().numpy())

    if len(all_y) == 0:
        return {"acc": np.nan}, np.array([]), np.array([])

    y_true = np.concatenate(all_y)
    p_pred = np.concatenate(all_p)
    y_hat = np.argmax(p_pred, axis=1)

    met = {
        "loss_ce": float(np.mean(all_loss)),
        "acc": float(accuracy_score(y_true, y_hat)),
        "precision_macro": float(precision_score(y_true, y_hat, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_hat, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_hat, average="macro", zero_division=0)),
        "precision_weighted": float(precision_score(y_true, y_hat, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(y_true, y_hat, average="weighted", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_hat, average="weighted", zero_division=0)),
        "log_loss": float(log_loss(y_true, p_pred, labels=list(range(NUM_CLASSES)))),
        "eval_time_s": float(time.time() - t0),
    }
    met.update(_auc_metrics(y_true, p_pred, NUM_CLASSES))

    if return_gates and len(gate_stats) > 0:
        g = torch.cat(gate_stats, dim=0)
        met["fusion_gate_mean_raw"] = float(g[:, 0].mean().item())
        met["fusion_gate_mean_enh"] = float(g[:, 1].mean().item())
        met["fusion_gate_mean_res"] = float(g[:, 2].mean().item())
        met["fusion_gate_entropy"] = float(gate_entropy(g).mean().item())

    return met, y_true, p_pred

def train_one_epoch(model, loader, optimizer, preproc, theta, global_model=None, global_prototypes=None, scheduler=None):
    model.train()
    freeze_backbone_bn_stats(model)
    preproc.eval()

    losses, ce_losses, proto_losses, correct, total = [], [], [], 0, 0

    for x, y, _, source_id in loader:
        x = x.to(DEVICE, non_blocking=True)
        y = y.to(DEVICE, non_blocking=True)
        source_id = source_id.to(DEVICE, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        amp_enabled = (DEVICE.type == "cuda")

        with torch.amp.autocast(device_type=DEVICE.type, enabled=amp_enabled):
            x_enh = preproc(x)
            x_res = torch.abs(x_enh - x)

            x_raw_n = normalize_imagenet(x)
            x_enh_n = normalize_imagenet(x_enh)
            x_res_n = normalize_imagenet(x_res)
            theta_vec = theta_to_vec(theta, x.size(0))

            logits, embed, _ = model(x_raw_n, x_enh_n, x_res_n, theta_vec, source_id, return_extra=True)

            ce = criterion(logits, y)
            proto = prototype_alignment_loss(embed, y, global_prototypes)
            loss = ce + CFG["proto_lambda"] * proto

            if global_model is not None and CFG["fedprox_mu"] > 0:
                loss = loss + 0.5 * CFG["fedprox_mu"] * fedprox_term(model, global_model)

        if scaler is not None and amp_enabled:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if CFG["grad_clip"] > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), CFG["grad_clip"])
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if CFG["grad_clip"] > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), CFG["grad_clip"])
            optimizer.step()

        if scheduler is not None:
            scheduler.step()

        losses.append(float(loss.item()))
        ce_losses.append(float(ce.item()))
        proto_losses.append(float(proto.item()))

        preds = logits.argmax(dim=1)
        correct += int((preds == y).sum().item())
        total += int(y.size(0))

    return {
        "loss": float(np.mean(losses)) if losses else np.nan,
        "ce_loss": float(np.mean(ce_losses)) if ce_losses else np.nan,
        "proto_loss": float(np.mean(proto_losses)) if proto_losses else np.nan,
        "acc": float(correct / max(1, total)),
    }

def build_tempered_fedavg_weights(clients_subset):
    sizes = np.array([c["n_train"] for c in clients_subset], dtype=np.float64)
    sizes = np.power(np.clip(sizes, 1.0, None), CFG["fedavg_temper"])
    sizes = sizes / max(1e-12, sizes.sum())
    return sizes.tolist()

def fedavg_update(global_model, local_models, weights):
    global_sd = global_model.state_dict()
    new_sd = {}

    for key in global_sd.keys():
        ref = global_sd[key]
        if not torch.is_floating_point(ref):
            new_sd[key] = local_models[0].state_dict()[key].detach().clone()
        else:
            acc = None
            for m, w in zip(local_models, weights):
                t = m.state_dict()[key].detach().float().cpu()
                acc = t * w if acc is None else acc + t * w
            new_sd[key] = acc.to(ref.dtype)

    global_model.load_state_dict(new_sd, strict=True)

def weighted_mean(rows, key, weight_key):
    vals, ws = [], []
    for r in rows:
        v = r.get(key, np.nan)
        w = r.get(weight_key, 0)
        if np.isfinite(v) and w > 0:
            vals.append(v)
            ws.append(w)
    if len(vals) == 0:
        return np.nan
    return float(np.average(vals, weights=ws))

def aggregate_by_dataset(round_local_rows, dataset_name):
    rows = [r for r in round_local_rows if r["dataset"] == dataset_name]
    return {
        "acc": weighted_mean(rows, "val_acc", "val_size"),
        "f1_macro": weighted_mean(rows, "val_f1_macro", "val_size"),
        "precision_macro": weighted_mean(rows, "val_precision_macro", "val_size"),
        "recall_macro": weighted_mean(rows, "val_recall_macro", "val_size"),
        "log_loss": weighted_mean(rows, "val_log_loss", "val_size"),
        "loss_ce": weighted_mean(rows, "val_loss_ce", "val_size"),
        "eval_time_s": weighted_mean(rows, "val_eval_time_s", "val_size"),
    }

# ============================================================
# STEP 8: BEST STATIC PRESET SEARCH (ONCE BEFORE FL)
# ============================================================
print("\n" + "=" * 118)
print("STEP 8: BEST STATIC PRESET SEARCH (ONCE BEFORE FL)")
print("=" * 118)

def run_best_static_preset_search(dataset_name, source_id, frame_train, frame_val, preset_bank):
    proxy_train_idx = sample_subset_indices(frame_train, CFG["static_search_train_samples"], seed=SEED + source_id)
    proxy_val_idx = sample_subset_indices(frame_val, CFG["static_search_val_samples"], seed=SEED + 100 + source_id)

    proxy_sampler = make_weighted_sampler(frame_train, proxy_train_idx, NUM_CLASSES)
    proxy_train_loader = make_loader(
        frame_train, proxy_train_idx, CFG["batch_size"], TRAIN_TFMS,
        shuffle=(proxy_sampler is None), sampler=proxy_sampler, source_id=source_id
    )
    proxy_val_loader = make_loader(
        frame_val, proxy_val_idx, CFG["batch_size"], EVAL_TFMS,
        shuffle=False, sampler=None, source_id=source_id
    )

    rows = []
    for preset_id, (theta_name, theta) in enumerate(preset_bank):
        model = ResNet50CRAF(
            num_classes=NUM_CLASSES,
            cond_dim=64,
            fuse_dim=256,
            embed_dim=256,
            pretrained=False,
        ).to(DEVICE)
        model.load_state_dict(INITIAL_GLOBAL_STATE, strict=True)
        set_trainable_for_round(model, rnd=1)

        optimizer = make_optimizer(model)
        total_steps = max(1, len(proxy_train_loader) * CFG["static_search_epochs"])
        warmup_steps = max(1, len(proxy_train_loader) * CFG["warmup_epochs"])
        scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

        preproc = theta_to_module(theta).to(DEVICE)

        train_logs = []
        for _ in range(CFG["static_search_epochs"]):
            train_logs.append(
                train_one_epoch(
                    model,
                    proxy_train_loader,
                    optimizer,
                    preproc,
                    theta,
                    global_model=None,
                    global_prototypes=None,
                    scheduler=scheduler,
                )
            )

        met, _, _ = evaluate_full(model, proxy_val_loader, preproc, theta, return_gates=False, use_tta=False)

        rows.append({
            "dataset": dataset_name,
            "preset_id": preset_id,
            "theta_name": theta_name,
            "theta_str": theta_str(theta),
            "score": score_metric(met),
            "proxy_train_loss": float(np.mean([x["loss"] for x in train_logs])),
            "proxy_train_acc": float(np.mean([x["acc"] for x in train_logs])),
            "proxy_val_acc": safe_float(met.get("acc")),
            "proxy_val_f1_macro": safe_float(met.get("f1_macro")),
            "proxy_val_precision_macro": safe_float(met.get("precision_macro")),
            "proxy_val_recall_macro": safe_float(met.get("recall_macro")),
            "proxy_val_log_loss": safe_float(met.get("log_loss")),
        })

        del model
        del preproc
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    rank_df = pd.DataFrame(rows).sort_values(
        by=["score", "proxy_val_f1_macro", "proxy_val_acc"],
        ascending=False
    ).reset_index(drop=True)

    best_name = str(rank_df.iloc[0]["theta_name"])
    best_name, best_theta = get_preset_by_name(preset_bank, best_name)
    return best_name, best_theta, rank_df

STATIC_THETA_NAME_DS1, STATIC_THETA_DS1, static_rank_df1 = run_best_static_preset_search(
    "ds1", 0, train1, val1, PRESET_BANK_DS1
)
STATIC_THETA_NAME_DS2, STATIC_THETA_DS2, static_rank_df2 = run_best_static_preset_search(
    "ds2", 1, train2, val2, PRESET_BANK_DS2
)

print_table(static_rank_df1, "Best static preset search ranking — DS1", max_rows=10)
print_table(static_rank_df2, "Best static preset search ranking — DS2", max_rows=10)
add_table_to_csv(static_rank_df1, "best_static_preset_search_ds1")
add_table_to_csv(static_rank_df2, "best_static_preset_search_ds2")

print(f"Chosen static preset DS1: {STATIC_THETA_NAME_DS1} {theta_str(STATIC_THETA_DS1)}")
print(f"Chosen static preset DS2: {STATIC_THETA_NAME_DS2} {theta_str(STATIC_THETA_DS2)}")

# ============================================================
# STEP 9: TRUE FEDERATED TRAINING (FULL PARTICIPATION)
# ============================================================
print("\n" + "=" * 118)
print("STEP 9: TRUE FEDERATED TRAINING — FULL PARTICIPATION")
print("=" * 118)

history_global = []
history_local = []

best_reward = -1.0
best_round_saved = None
best_model_state = None
best_global_prototypes = None
global_prototypes = None

t_global_start = time.time()

print(f"Clients => DS1={n_clients_ds1}, DS2={n_clients_ds2}, TOTAL={CLIENTS_TOTAL}")
print(f"Rounds: {CFG['rounds']} | Local epochs: {CFG['local_epochs']}")
print(f"FedProx μ={CFG['fedprox_mu']} | Proto λ={CFG['proto_lambda']}")
print(f"Tempered FedAvg exponent = {CFG['fedavg_temper']:.2f}")
print(f"Static preset DS1: {STATIC_THETA_NAME_DS1} {theta_str(STATIC_THETA_DS1)}")
print(f"Static preset DS2: {STATIC_THETA_NAME_DS2} {theta_str(STATIC_THETA_DS2)}")

for rnd in range(1, CFG["rounds"] + 1):
    round_t0 = time.time()
    selected_ids = list(range(len(clients)))  # full participation every round

    print("\n" + "=" * 118)
    print(f"ROUND {rnd}/{CFG['rounds']} | selected={selected_ids}")
    print("=" * 118)

    local_models = []
    proto_payloads = []
    round_local_rows = []
    selected_clients_meta = []

    for cid in selected_ids:
        client = clients[cid]

        if client["dataset"] == "ds1":
            theta_name = STATIC_THETA_NAME_DS1
            theta = STATIC_THETA_DS1
        else:
            theta_name = STATIC_THETA_NAME_DS2
            theta = STATIC_THETA_DS2

        preproc = theta_to_module(theta).to(DEVICE)

        local_model = ResNet50CRAF(
            num_classes=NUM_CLASSES,
            cond_dim=64,
            fuse_dim=256,
            embed_dim=256,
            pretrained=False,
        ).to(DEVICE)
        local_model.load_state_dict(global_model.state_dict(), strict=True)

        set_trainable_for_round(local_model, rnd)
        opt = make_optimizer(local_model)

        total_steps = max(1, len(client["train_loader"]) * CFG["local_epochs"])
        warmup_steps = max(1, len(client["train_loader"]) * CFG["warmup_epochs"])
        scheduler = get_cosine_schedule_with_warmup(opt, warmup_steps, total_steps)

        train_logs = []
        for _ in range(CFG["local_epochs"]):
            log_ep = train_one_epoch(
                local_model,
                client["train_loader"],
                opt,
                preproc,
                theta,
                global_model=global_model,
                global_prototypes=global_prototypes,
                scheduler=scheduler,
            )
            train_logs.append(log_ep)

        met_loc, _, _ = evaluate_full(local_model, client["val_loader"], preproc, theta, return_gates=True, use_tta=False)
        proto_payload = compute_prototype_payload(local_model, client["proto_loader"], preproc, theta)

        reward = score_metric(met_loc)

        local_models.append(local_model)
        selected_clients_meta.append(client)
        proto_payloads.append(proto_payload)

        g, a, b, t, kk, eg, mix = theta
        row = {
            "round": rnd,
            "client": f"client_{cid}",
            "dataset": client["dataset"],
            "selected": 1,
            "theta_name": theta_name,
            "theta_str": theta_str(theta),
            "gamma_power": g,
            "alpha_contrast_weight": a,
            "beta_contrast_sharpness": b,
            "tau_clip": t,
            "k_blur_kernel_size": kk,
            "edge_gain": eg,
            "blend_mix": mix,
            "train_loss": float(np.mean([x["loss"] for x in train_logs])),
            "train_ce_loss": float(np.mean([x["ce_loss"] for x in train_logs])),
            "train_proto_loss": float(np.mean([x["proto_loss"] for x in train_logs])),
            "train_acc": float(np.mean([x["acc"] for x in train_logs])),
            "val_size": client["n_val"],
            **{f"val_{k}": v for k, v in met_loc.items()},
            "reward": reward,
        }
        round_local_rows.append(row)

        auc_val = row.get("val_auc_roc_macro_ovr", np.nan)
        print(
            f"Client {cid} ({client['dataset']}) | "
            f"train_acc={row['train_acc']:.4f} | "
            f"val_acc={row['val_acc']:.4f} | "
            f"val_f1={row['val_f1_macro']:.4f} | "
            f"val_auc={auc_val:.4f} | "
            f"reward={reward:.4f} | "
            f"theta={theta_name} {theta_str(theta)}"
        )

    agg_weights = build_tempered_fedavg_weights(selected_clients_meta)
    fedavg_update(global_model, local_models, agg_weights)
    global_prototypes = aggregate_prototypes(proto_payloads, global_model.embed_dim)

    ds1_metrics = aggregate_by_dataset(round_local_rows, "ds1")
    ds2_metrics = aggregate_by_dataset(round_local_rows, "ds2")

    global_metrics = {
        "acc": weighted_mean(round_local_rows, "val_acc", "val_size"),
        "f1_macro": weighted_mean(round_local_rows, "val_f1_macro", "val_size"),
        "precision_macro": weighted_mean(round_local_rows, "val_precision_macro", "val_size"),
        "recall_macro": weighted_mean(round_local_rows, "val_recall_macro", "val_size"),
        "log_loss": weighted_mean(round_local_rows, "val_log_loss", "val_size"),
        "loss_ce": weighted_mean(round_local_rows, "val_loss_ce", "val_size"),
        "eval_time_s": weighted_mean(round_local_rows, "val_eval_time_s", "val_size"),
    }

    ds1_score = score_metric(ds1_metrics)
    ds2_score = score_metric(ds2_metrics)
    round_reward = (
        CFG["best_round_mass_ds1"] * ds1_score +
        CFG["best_round_mass_ds2"] * ds2_score +
        CFG["best_round_min_bonus"] * min(ds1_score, ds2_score)
    )

    history_local.extend(round_local_rows)
    history_global.append({
        "round": rnd,
        "round_time_s": float(time.time() - round_t0),
        "n_selected_clients": len(selected_ids),
        "active_fraction": 1.0,
        "global_reward": round_reward,
        "global_acc": global_metrics["acc"],
        "global_f1_macro": global_metrics["f1_macro"],
        "global_precision_macro": global_metrics["precision_macro"],
        "global_recall_macro": global_metrics["recall_macro"],
        "global_log_loss": global_metrics["log_loss"],
        "global_loss_ce": global_metrics["loss_ce"],
        "global_eval_time_s": global_metrics["eval_time_s"],
        "ds1_acc": ds1_metrics["acc"],
        "ds1_f1_macro": ds1_metrics["f1_macro"],
        "ds1_log_loss": ds1_metrics["log_loss"],
        "ds2_acc": ds2_metrics["acc"],
        "ds2_f1_macro": ds2_metrics["f1_macro"],
        "ds2_log_loss": ds2_metrics["log_loss"],
    })

    print(
        f"GLOBAL VAL (Round {rnd}) | "
        f"global_acc={global_metrics['acc']:.4f} | "
        f"global_f1={global_metrics['f1_macro']:.4f} | "
        f"ds1_acc={ds1_metrics['acc']:.4f} | "
        f"ds1_f1={ds1_metrics['f1_macro']:.4f} | "
        f"ds2_acc={ds2_metrics['acc']:.4f} | "
        f"ds2_f1={ds2_metrics['f1_macro']:.4f} | "
        f"reward={round_reward:.4f} | "
        f"round_time={history_global[-1]['round_time_s']:.1f}s"
    )

    if np.isfinite(round_reward) and round_reward > best_reward:
        best_reward = float(round_reward)
        best_round_saved = rnd
        best_model_state = {k: v.detach().cpu().clone() for k, v in global_model.state_dict().items()}
        best_global_prototypes = None if global_prototypes is None else {
            "proto": global_prototypes["proto"].detach().cpu().clone(),
            "mask": global_prototypes["mask"].detach().cpu().clone(),
            "counts": global_prototypes["counts"].detach().cpu().clone(),
        }

if best_model_state is not None:
    global_model.load_state_dict({k: v.to(DEVICE) for k, v in best_model_state.items()})

if best_global_prototypes is not None:
    global_prototypes = {
        "proto": best_global_prototypes["proto"].to(DEVICE),
        "mask": best_global_prototypes["mask"].to(DEVICE),
        "counts": best_global_prototypes["counts"].to(DEVICE),
    }

t_total = float(time.time() - t_global_start)
print("\n" + "=" * 118)
print(f"TRAINING COMPLETE | total_time={t_total:.1f}s | best_round={best_round_saved} | best_reward={best_reward:.4f}")
print("=" * 118)

glob_df = pd.DataFrame(history_global)
loc_df = pd.DataFrame(history_local)

print_table(glob_df, "GLOBAL per-round metrics", max_rows=20)
print_table(loc_df, "LOCAL per-client per-round metrics", max_rows=20)
add_table_to_csv(glob_df, "global_round_metrics_full")
add_table_to_csv(loc_df, "client_round_metrics_full")

# ============================================================
# STEP 10: FINAL EVALUATION (USE SAME STATIC PRESETS)
# ============================================================
print("\n" + "=" * 118)
print("STEP 10: FINAL EVALUATION (USE SAME STATIC PRESETS)")
print("=" * 118)

@torch.no_grad()
def evaluate_with_single_theta(model, loader, theta, use_tta=False):
    pre = theta_to_module(theta).to(DEVICE)
    return evaluate_full(model, loader, pre, theta, return_gates=False, use_tta=use_tta)

def equal_merge_metrics(m1, m2):
    out = {}
    keys = sorted(set(m1.keys()).union(set(m2.keys())))
    for k in keys:
        a, b = m1.get(k, np.nan), m2.get(k, np.nan)
        if np.isfinite(a) and np.isfinite(b):
            out[k] = float(np.mean([a, b]))
        elif np.isfinite(a):
            out[k] = float(a)
        elif np.isfinite(b):
            out[k] = float(b)
        else:
            out[k] = np.nan
    return out

val_ds1, _, _ = evaluate_with_single_theta(global_model, val_loader_ds1, STATIC_THETA_DS1, use_tta=CFG["final_use_tta"])
test_ds1, y_ds1, p_ds1 = evaluate_with_single_theta(global_model, test_loader_ds1, STATIC_THETA_DS1, use_tta=CFG["final_use_tta"])

val_ds2, _, _ = evaluate_with_single_theta(global_model, val_loader_ds2, STATIC_THETA_DS2, use_tta=CFG["final_use_tta"])
test_ds2, y_ds2, p_ds2 = evaluate_with_single_theta(global_model, test_loader_ds2, STATIC_THETA_DS2, use_tta=CFG["final_use_tta"])

val_global = equal_merge_metrics(val_ds1, val_ds2)
test_global = equal_merge_metrics(test_ds1, test_ds2)

choice_df = pd.DataFrame([
    {
        "dataset": "ds1",
        "strategy": "best_static_preset_fixed_all_rounds",
        "theta_names": str([STATIC_THETA_NAME_DS1]),
        "score": score_metric(val_ds1),
        "val_acc": safe_float(val_ds1.get("acc")),
        "val_f1": safe_float(val_ds1.get("f1_macro")),
    },
    {
        "dataset": "ds2",
        "strategy": "best_static_preset_fixed_all_rounds",
        "theta_names": str([STATIC_THETA_NAME_DS2]),
        "score": score_metric(val_ds2),
        "val_acc": safe_float(val_ds2.get("acc")),
        "val_f1": safe_float(val_ds2.get("f1_macro")),
    },
])
print_table(choice_df, "Final static preprocessing strategy", max_rows=5)
add_table_to_csv(choice_df, "final_theta_strategy_choice_static")

# ============================================================
# STEP 11: EXTENDED METRICS + CALIBRATION + ERROR ANALYSIS
# ============================================================
print("\n" + "=" * 118)
print("STEP 11: EXTENDED METRICS + CALIBRATION + ERROR ANALYSIS")
print("=" * 118)

def top_label_calibration(y_true, p_pred, n_bins=12):
    conf = np.max(p_pred, axis=1)
    pred = np.argmax(p_pred, axis=1)
    correct = (pred == y_true).astype(np.float32)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(conf, bins) - 1
    bin_ids = np.clip(bin_ids, 0, n_bins - 1)

    rows = []
    ece = 0.0
    mce = 0.0
    N = len(y_true)

    for b in range(n_bins):
        m = (bin_ids == b)
        if m.sum() == 0:
            rows.append({
                "bin_id": b,
                "bin_left": bins[b],
                "bin_right": bins[b + 1],
                "bin_confidence": np.nan,
                "bin_accuracy": np.nan,
                "bin_gap": np.nan,
                "bin_count": 0,
            })
            continue

        bc = float(conf[m].mean())
        ba = float(correct[m].mean())
        gap = abs(ba - bc)
        ece += (m.sum() / N) * gap
        mce = max(mce, gap)
        rows.append({
            "bin_id": b,
            "bin_left": bins[b],
            "bin_right": bins[b + 1],
            "bin_confidence": bc,
            "bin_accuracy": ba,
            "bin_gap": gap,
            "bin_count": int(m.sum()),
        })

    return pd.DataFrame(rows), float(ece), float(mce)

def multiclass_brier_score(y_true, p_pred, num_classes):
    y_oh = np.eye(num_classes)[y_true]
    return float(np.mean(np.sum((p_pred - y_oh) ** 2, axis=1)))

def classwise_confusion_metrics(y_true, p_pred):
    y_hat = np.argmax(p_pred, axis=1)
    cm = confusion_matrix(y_true, y_hat, labels=list(range(NUM_CLASSES)))
    total = cm.sum()

    rows = []
    for c, lab in enumerate(labels):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        tn = total - tp - fp - fn

        ppv = tp / max(1, (tp + fp))
        npv = tn / max(1, (tn + fn))
        sensitivity = tp / max(1, (tp + fn))
        specificity = tn / max(1, (tn + fp))
        fpr = fp / max(1, (fp + tn))
        fnr = fn / max(1, (fn + tp))
        jacc = tp / max(1, (tp + fp + fn))
        support = tp + fn
        prevalence = support / max(1, total)
        bal = 0.5 * (sensitivity + specificity)

        rows.append({
            "class_id": c,
            "class_name": lab,
            "support": int(support),
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
            "tn": int(tn),
            "prevalence": float(prevalence),
            "ppv": float(ppv),
            "npv": float(npv),
            "recall": float(sensitivity),
            "specificity": float(specificity),
            "fpr": float(fpr),
            "fnr": float(fnr),
            "jaccard": float(jacc),
            "balanced_acc": float(bal),
        })

    return cm, pd.DataFrame(rows)

def top_confusion_pairs(y_true, p_pred):
    y_hat = np.argmax(p_pred, axis=1)
    cm = confusion_matrix(y_true, y_hat, labels=list(range(NUM_CLASSES)))
    rows = []
    for i, lab_i in enumerate(labels):
        for j, lab_j in enumerate(labels):
            if i == j:
                continue
            rows.append({
                "true_class": lab_i,
                "pred_class": lab_j,
                "count": int(cm[i, j]),
            })
    return pd.DataFrame(rows).sort_values("count", ascending=False).reset_index(drop=True)

def wrong_right_confidence_table(y_true, p_pred):
    pred = np.argmax(p_pred, axis=1)
    conf = np.max(p_pred, axis=1)
    ok = pred == y_true
    return pd.DataFrame({
        "confidence": conf,
        "correct": ok.astype(int),
        "status": np.where(ok, "correct", "wrong"),
        "true_label": [id2label[int(x)] for x in y_true],
        "pred_label": [id2label[int(x)] for x in pred],
    })

def compute_extended_bundle(y_true, p_pred, dataset_name):
    y_hat = np.argmax(p_pred, axis=1)
    cm, class_df = classwise_confusion_metrics(y_true, p_pred)
    cal_df, ece, mce = top_label_calibration(y_true, p_pred, n_bins=CFG["calibration_bins"])
    conf_df = top_confusion_pairs(y_true, p_pred)
    conf_table = wrong_right_confidence_table(y_true, p_pred)

    scalar = {
        "dataset": dataset_name,
        "acc": float(accuracy_score(y_true, y_hat)),
        "balanced_acc": float(balanced_accuracy_score(y_true, y_hat)),
        "precision_macro": float(precision_score(y_true, y_hat, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_hat, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_hat, average="macro", zero_division=0)),
        "precision_weighted": float(precision_score(y_true, y_hat, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(y_true, y_hat, average="weighted", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_hat, average="weighted", zero_division=0)),
        "log_loss": float(log_loss(y_true, p_pred, labels=list(range(NUM_CLASSES)))),
        "mcc": float(matthews_corrcoef(y_true, y_hat)),
        "kappa": float(cohen_kappa_score(y_true, y_hat)),
        "jaccard_macro": float(jaccard_score(y_true, y_hat, average="macro")),
        "ppv_macro": float(class_df["ppv"].mean()),
        "npv_macro": float(class_df["npv"].mean()),
        "specificity_macro": float(class_df["specificity"].mean()),
        "fpr_macro": float(class_df["fpr"].mean()),
        "fnr_macro": float(class_df["fnr"].mean()),
        "ece": float(ece),
        "mce": float(mce),
        "brier_multi": float(multiclass_brier_score(y_true, p_pred, NUM_CLASSES)),
    }
    scalar.update(_auc_metrics(y_true, p_pred, NUM_CLASSES))
    return scalar, class_df, cal_df, conf_df, conf_table, cm

ext_ds1, class_df1, cal_df1, conf_pairs1, conf_table1, cm1 = compute_extended_bundle(y_ds1, p_ds1, "ds1_test")
ext_ds2, class_df2, cal_df2, conf_pairs2, conf_table2, cm2 = compute_extended_bundle(y_ds2, p_ds2, "ds2_test")

ext_df = pd.DataFrame([ext_ds1, ext_ds2])

print_table(ext_df, "Extended TEST metrics (DS1 vs DS2)", max_rows=5)
print_table(class_df1, "Classwise metrics — DS1 TEST", max_rows=10)
print_table(class_df2, "Classwise metrics — DS2 TEST", max_rows=10)
print_table(conf_pairs1.head(10), "Top confusion pairs — DS1 TEST", max_rows=10)
print_table(conf_pairs2.head(10), "Top confusion pairs — DS2 TEST", max_rows=10)
print_table(cal_df1, "Calibration bins — DS1 TEST", max_rows=15)
print_table(cal_df2, "Calibration bins — DS2 TEST", max_rows=15)

add_table_to_csv(ext_df, "extended_test_metrics_ds1_ds2")
add_table_to_csv(class_df1, "classwise_metrics_ds1_test")
add_table_to_csv(class_df2, "classwise_metrics_ds2_test")
add_table_to_csv(cal_df1, "calibration_bins_ds1_test")
add_table_to_csv(cal_df2, "calibration_bins_ds2_test")
add_table_to_csv(conf_pairs1, "top_confusions_ds1_test")
add_table_to_csv(conf_pairs2, "top_confusions_ds2_test")
add_table_to_csv(conf_table1, "confidence_correctness_ds1_test")
add_table_to_csv(conf_table2, "confidence_correctness_ds2_test")

test_ds1.update(ext_ds1)
test_ds2.update(ext_ds2)

def compact_metrics(m):
    keep = [
        "acc", "balanced_acc", "precision_macro", "recall_macro", "f1_macro",
        "precision_weighted", "recall_weighted", "f1_weighted", "log_loss",
        "auc_roc_macro_ovr", "mcc", "kappa", "ppv_macro", "npv_macro",
        "specificity_macro", "ece", "mce", "brier_multi", "loss_ce", "eval_time_s"
    ]
    out = {}
    for k in keep:
        if k in m and np.isfinite(m[k]):
            out[k] = float(m[k])
    return out

paper_df = pd.DataFrame([
    {"setting": "ARCF-Net BestStaticPreset", "split": "VAL",  "dataset": "ds1",          **compact_metrics(val_ds1)},
    {"setting": "ARCF-Net BestStaticPreset", "split": "VAL",  "dataset": "ds2",          **compact_metrics(val_ds2)},
    {"setting": "ARCF-Net BestStaticPreset", "split": "VAL",  "dataset": "global_equal", **compact_metrics(val_global)},
    {"setting": "ARCF-Net BestStaticPreset", "split": "TEST", "dataset": "ds1",          **compact_metrics(test_ds1)},
    {"setting": "ARCF-Net BestStaticPreset", "split": "TEST", "dataset": "ds2",          **compact_metrics(test_ds2)},
    {"setting": "ARCF-Net BestStaticPreset", "split": "TEST", "dataset": "global_equal", **compact_metrics(test_global)},
])

print_table(paper_df, "VAL + TEST tables", max_rows=10)
add_table_to_csv(paper_df, "paper_ready_metrics")

print("\nSelection summary:")
print(f"- Best round: {best_round_saved} | best_reward={best_reward:.4f}")
print(f"- DS1 static strategy: best_static_preset_fixed_all_rounds | names={[STATIC_THETA_NAME_DS1]}")
print(f"- DS2 static strategy: best_static_preset_fixed_all_rounds | names={[STATIC_THETA_NAME_DS2]}")

# ============================================================
# STEP 12: PREPROCESSING VALIDATION
# ============================================================
print("\n" + "=" * 118)
print("STEP 12: PREPROCESSING VALIDATION")
print("=" * 118)

@torch.no_grad()
def entropy_per_image(x01):
    gray = x01.mean(dim=1)
    B = gray.shape[0]
    ent = []
    for i in range(B):
        g = (gray[i].detach().cpu().numpy() * 255).astype(np.uint8)
        hist = np.bincount(g.flatten(), minlength=256).astype(np.float32)
        p = hist / np.clip(hist.sum(), 1, None)
        p = p[p > 0]
        ent.append(float(-(p * np.log2(p)).sum()))
    return np.array(ent)

@torch.no_grad()
def edge_energy(x01, kernel):
    kernel = kernel.to(device=x01.device, dtype=x01.dtype)
    gray = x01.mean(dim=1, keepdim=True)
    lap = F.conv2d(F.pad(gray, (1, 1, 1, 1), mode="reflect"), kernel).abs()
    return lap.mean(dim=(1, 2, 3)).detach().cpu().numpy()

@torch.no_grad()
def contrast_proxy(x01):
    gray = x01.mean(dim=1)
    return gray.std(dim=(1, 2)).detach().cpu().numpy()

@torch.no_grad()
def run_preproc_validation(frame, preproc, sample_n=400):
    n = min(sample_n, len(frame))
    if n <= 0:
        return pd.DataFrame(), pd.DataFrame()

    idx = np.random.choice(len(frame), size=n, replace=False)
    ds = MRIDataset(frame, indices=idx.tolist(), tfms=EVAL_TFMS, source_id=0)

    xs = []
    for i in range(len(ds)):
        x, _, _, _ = ds[i]
        xs.append(x)
    x = torch.stack(xs).to(DEVICE)

    x_after = preproc(x).clamp(0, 1)
    lap_kernel = preproc.lap if hasattr(preproc, "lap") else RACEFELCM().to(DEVICE).lap

    ee_before = edge_energy(x, lap_kernel)
    ee_after  = edge_energy(x_after, lap_kernel)
    ent_before = entropy_per_image(x)
    ent_after  = entropy_per_image(x_after)
    con_before = contrast_proxy(x)
    con_after  = contrast_proxy(x_after)

    dfm = pd.DataFrame({
        "edge_energy_before": ee_before,
        "edge_energy_after": ee_after,
        "entropy_before": ent_before,
        "entropy_after": ent_after,
        "contrast_before": con_before,
        "contrast_after": con_after,
        "edge_gain_ratio": (ee_after / np.clip(ee_before, 1e-9, None)),
        "entropy_delta": (ent_after - ent_before),
        "contrast_delta": (con_after - con_before),
    })
    summary = dfm.agg(["mean", "std", "min", "max"]).T.reset_index().rename(columns={"index": "metric"})
    return dfm, summary

preproc_ds1 = theta_to_module(STATIC_THETA_DS1).to(DEVICE)
preproc_ds2 = theta_to_module(STATIC_THETA_DS2).to(DEVICE)

preproc_df1, preproc_summary_df1 = run_preproc_validation(val1, preproc_ds1, CFG["preproc_val_sample_n"])
preproc_df2, preproc_summary_df2 = run_preproc_validation(val2, preproc_ds2, CFG["preproc_val_sample_n"])

print_table(preproc_summary_df1, "Preprocessing validation summary (DS1 VAL sample)", max_rows=15)
print_table(preproc_summary_df2, "Preprocessing validation summary (DS2 VAL sample)", max_rows=15)
add_table_to_csv(preproc_summary_df1, "preprocessing_validation_summary_ds1")
add_table_to_csv(preproc_summary_df2, "preprocessing_validation_summary_ds2")

# ============================================================
# STEP 13: SAVE CHECKPOINT + CSV
# ============================================================
print("\n" + "=" * 118)
print("STEP 13: SAVING CHECKPOINT + CSV")
print("=" * 118)

client_process_manifest = []
for c in clients:
    df_src = train1 if c["dataset"] == "ds1" else train2
    client_process_manifest.append({
        "gid": c["gid"],
        "local_id": c["local_id"],
        "dataset": c["dataset"],
        "source_id": c["source_id"],
        "n_train": c["n_train"],
        "n_tune": c["n_tune"],
        "n_val": c["n_val"],
        "train_indices": c["train_indices"],
        "tune_indices": c["tune_indices"],
        "val_indices": c["val_indices"],
        "train_paths": df_src.loc[c["train_indices"], "path"].tolist(),
        "tune_paths": df_src.loc[c["tune_indices"], "path"].tolist(),
        "val_paths": df_src.loc[c["val_indices"], "path"].tolist(),
        "class_counts": c["class_counts"],
    })

all_df = pd.DataFrame(ALL_ROWS)

checkpoint = {
    "method_info": METHOD_INFO,

    "state_dict": {k: v.detach().cpu() for k, v in global_model.state_dict().items()},
    "best_model_state_dict": {k: v.detach().cpu() for k, v in global_model.state_dict().items()},

    "config": CFG,
    "seed": SEED,
    "device_used": str(DEVICE),
    "backbone_name": "resnet50",
    "fusion_name": "CRAF",
    "preprocessing_name": "RACE-FELCM",

    "dataset1_raw_root": DS1_ROOT,
    "dataset2_root": DS2_ROOT,
    "labels": labels,
    "label2id": label2id,
    "id2label": id2label,
    "num_classes": NUM_CLASSES,
    "theta_fullforms": THETA_FULLFORMS,

    "full_dataset_df1": frame_to_manifest(df1),
    "full_dataset_df2": frame_to_manifest(df2),

    "split_train1": frame_to_manifest(train1),
    "split_val1": frame_to_manifest(val1),
    "split_test1": frame_to_manifest(test1),
    "split_train2": frame_to_manifest(train2),
    "split_val2": frame_to_manifest(val2),
    "split_test2": frame_to_manifest(test2),

    "fixed_clients_ds1": n_clients_ds1,
    "fixed_clients_ds2": n_clients_ds2,
    "clients_total": CLIENTS_TOTAL,
    "client_indices_ds1": client_indices_ds1,
    "client_indices_ds2": client_indices_ds2,
    "client_process_manifest": client_process_manifest,

    "preset_bank_ds1": PRESET_BANK_DS1,
    "preset_bank_ds2": PRESET_BANK_DS2,

    "static_preset_search_strategy": "short_proxy_supervised_search_once_before_fl",
    "static_rank_ds1": static_rank_df1.to_dict(orient="list"),
    "static_rank_ds2": static_rank_df2.to_dict(orient="list"),
    "static_theta_name_ds1": STATIC_THETA_NAME_DS1,
    "static_theta_name_ds2": STATIC_THETA_NAME_DS2,
    "static_theta_ds1": STATIC_THETA_DS1,
    "static_theta_ds2": STATIC_THETA_DS2,

    "best_round_saved": best_round_saved,
    "best_reward": best_reward,
    "history_global": glob_df.to_dict(orient="list"),
    "history_local": loc_df.to_dict(orient="list"),
    "total_training_time_s": t_total,

    "final_choice_ds1": {
        "strategy": "best_static_preset_fixed_all_rounds",
        "theta_names": [STATIC_THETA_NAME_DS1],
        "theta_list": [STATIC_THETA_DS1],
        "score": score_metric(val_ds1),
        "val_acc": safe_float(val_ds1.get("acc")),
        "val_f1": safe_float(val_ds1.get("f1_macro")),
    },
    "final_choice_ds2": {
        "strategy": "best_static_preset_fixed_all_rounds",
        "theta_names": [STATIC_THETA_NAME_DS2],
        "theta_list": [STATIC_THETA_DS2],
        "score": score_metric(val_ds2),
        "val_acc": safe_float(val_ds2.get("acc")),
        "val_f1": safe_float(val_ds2.get("f1_macro")),
    },

    "best_global_prototypes": None if best_global_prototypes is None else {
        "proto": best_global_prototypes["proto"],
        "mask": best_global_prototypes["mask"],
        "counts": best_global_prototypes["counts"],
    },

    "final_val_ds1": val_ds1,
    "final_val_ds2": val_ds2,
    "final_val_global": val_global,
    "final_test_ds1": test_ds1,
    "final_test_ds2": test_ds2,
    "final_test_global_equal": test_global,
    "paper_ready_metrics": paper_df.to_dict(orient="list"),
    "extended_test_metrics_ds1_ds2": ext_df.to_dict(orient="list"),
    "classwise_metrics_ds1_test": class_df1.to_dict(orient="list"),
    "classwise_metrics_ds2_test": class_df2.to_dict(orient="list"),
    "calibration_bins_ds1_test": cal_df1.to_dict(orient="list"),
    "calibration_bins_ds2_test": cal_df2.to_dict(orient="list"),
    "top_confusions_ds1_test": conf_pairs1.to_dict(orient="list"),
    "top_confusions_ds2_test": conf_pairs2.to_dict(orient="list"),
    "confidence_correctness_ds1_test": conf_table1.to_dict(orient="list"),
    "confidence_correctness_ds2_test": conf_table2.to_dict(orient="list"),
    "static_preset_client_distribution_table": dist_df.to_dict(orient="list"),
    "global_round_metrics_table": glob_df.to_dict(orient="list"),
    "local_round_metrics_table": loc_df.to_dict(orient="list"),
    "all_tables_long_format": all_df.to_dict(orient="list"),

    "ds1_test_y_true": y_ds1,
    "ds1_test_p_pred": p_ds1,
    "ds1_test_y_hat": np.argmax(p_ds1, axis=1),
    "ds2_test_y_true": y_ds2,
    "ds2_test_p_pred": p_ds2,
    "ds2_test_y_hat": np.argmax(p_ds2, axis=1),

    "preprocessing_validation_ds1_raw": preproc_df1.to_dict(orient="list") if len(preproc_df1) else {},
    "preprocessing_validation_ds2_raw": preproc_df2.to_dict(orient="list") if len(preproc_df2) else {},
    "preprocessing_validation_summary_ds1": preproc_summary_df1.to_dict(orient="list") if len(preproc_summary_df1) else {},
    "preprocessing_validation_summary_ds2": preproc_summary_df2.to_dict(orient="list") if len(preproc_summary_df2) else {},
}

torch.save(checkpoint, MODEL_PATH)
print(f"Saved checkpoint: {MODEL_PATH}")

all_df.to_csv(CSV_PATH, index=False)
print(f"Saved CSV: {CSV_PATH}")

print("\nDONE")
print(f"Method: {METHOD_INFO['acronym']} = {METHOD_INFO['full_form']}")
print(f"Ablation: {METHOD_INFO['ablation_name']}")
print(f"Backbone: {METHOD_INFO['backbone_full_form']}")
print(f"Best round: {best_round_saved}")
print(f"Fixed clients => DS1={n_clients_ds1}, DS2={n_clients_ds2}, TOTAL={CLIENTS_TOTAL}")
print(f"Rounds completed: {CFG['rounds']}")
print(f"Global TEST acc: {safe_float(test_global.get('acc')):.4f}")
print(f"Global TEST f1_macro: {safe_float(test_global.get('f1_macro')):.4f}")
print(f"DS1 TEST acc: {safe_float(test_ds1.get('acc')):.4f}")
print(f"DS2 TEST acc: {safe_float(test_ds2.get('acc')):.4f}")
print(f"DS1 static strategy: best_static_preset_fixed_all_rounds | names={[STATIC_THETA_NAME_DS1]}")
print(f"DS2 static strategy: best_static_preset_fixed_all_rounds | names={[STATIC_THETA_NAME_DS2]}")
```

    ======================================================================================================================
    ARCF-Net ABLATION 3: BEST STATIC PRESET + FULL PARTICIPATION + NO PLOTS
    ======================================================================================================================
    ENV: KAGGLE | DEVICE: cuda | torch=2.10.0+cu128
    ======================================================================================================================
    
    ======================================================================================================================
    STEP 0: ACCESS DATASETS
    ======================================================================================================================
    Dataset-1 RAW root detected:
      /kaggle/input/pmram-bangladeshi-brain-cancer-mri-dataset/PMRAM Bangladeshi Brain Cancer - MRI Dataset/PMRAM Bangladeshi Brain Cancer - MRI Dataset/Raw Data/Raw
    Dataset-2 root detected:
      /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset
    
    ======================================================================================================================
    STEP 1: BUILD DATA MANIFESTS
    ======================================================================================================================
    ds1_raw: 512Glioma -> glioma | 373 images
    ds1_raw: 512Meningioma -> meningioma | 363 images
    ds1_raw: 512Normal -> notumor | 396 images
    ds1_raw: 512Pituitary -> pituitary | 373 images
    ds2: glioma -> glioma | 1621 images
    ds2: meningioma -> meningioma | 1646 images
    ds2: notumor -> notumor | 2000 images
    ds2: pituitary -> pituitary | 1764 images
    
    Dataset-1 images: 1505
    label
    glioma        373
    meningioma    363
    notumor       396
    pituitary     373
    
    Dataset-2 images: 7031
    label
    glioma        1621
    meningioma    1646
    notumor       2000
    pituitary     1764
    
    ======================================================================================================================
    STEP 2: TRAIN / VAL / TEST SPLIT
    ======================================================================================================================
    DS1 TRAIN: 1053 | VAL: 226 | TEST: 226
    DS2 TRAIN: 4921 | VAL: 1055 | TEST: 1055
    
    ======================================================================================================================
    STEP 2.5: SANITY / LEAKAGE CHECKS
    ======================================================================================================================
    
    ----------------------------------------------------------------------------------------------------------------------
    Leakage / Sanity Summary — ds1
    ----------------------------------------------------------------------------------------------------------------------
     path_overlap_train_val  path_overlap_train_test  path_overlap_val_test  unique_paths_train  unique_paths_val  unique_paths_test  filename_overlap_train_val  filename_overlap_train_test  filename_overlap_val_test  subset_hash_train_val  subset_hash_train_test  subset_hash_val_test  subset_hash_n_train  subset_hash_n_val  subset_hash_n_test
                          0                        0                      0                1053               226                226                           0                            0                          0                      5                       5                     6                  298                222                 224
    
    ----------------------------------------------------------------------------------------------------------------------
    Leakage / Sanity Summary — ds2
    ----------------------------------------------------------------------------------------------------------------------
     path_overlap_train_val  path_overlap_train_test  path_overlap_val_test  unique_paths_train  unique_paths_val  unique_paths_test  filename_overlap_train_val  filename_overlap_train_test  filename_overlap_val_test  subset_hash_train_val  subset_hash_train_test  subset_hash_val_test  subset_hash_n_train  subset_hash_n_val  subset_hash_n_test
                          0                        0                      0                4921              1055               1055                           0                            0                          0                      0                       3                     4                  299                298                 299
    
    ======================================================================================================================
    STEP 3: FIXED NON-IID CLIENT PARTITIONING
    ======================================================================================================================
    Fixed clients for DS1: 3
    Fixed clients for DS2: 3
    
    ======================================================================================================================
    STEP 4: DATA LOADERS
    ======================================================================================================================
    ds1 | client_0 | train=490 | tune=77 | val=67
    ds1 | client_1 | train=125 | tune=20 | val=18
    ds1 | client_2 | train=198 | tune=31 | val=27
    ds2 | client_3 | train=629 | tune=98 | val=86
    ds2 | client_4 | train=527 | tune=82 | val=72
    ds2 | client_5 | train=2653 | tune=412 | val=362
    
    ----------------------------------------------------------------------------------------------------------------------
    Static-preset client class distribution
    ----------------------------------------------------------------------------------------------------------------------
      client dataset  total_train  total_tune  total_val  glioma  meningioma  notumor  pituitary
    client_0     ds1          490          77         67     111          46      176        157
    client_1     ds1          125          20         18      75           4        8         38
    client_2     ds1          198          31         27      16         147       30          5
    client_3     ds2          629          98         86      12         197      416          4
    client_4     ds2          527          82         72     202           4      284         37
    client_5     ds2         2653         412        362     665         691      383        914
    Augmentation: ON
    Preprocessing: ON
    Total clients: 6
    
    ======================================================================================================================
    STEP 5: PREPROCESSING — RACE-FELCM
    ======================================================================================================================
    
    ======================================================================================================================
    STEP 6: MODEL — ResNet-50 + CRAF Fusion
    ======================================================================================================================
    Backbone: ResNet-50 | pretrained_loaded=True
    Total params: 25,790,855
    Trainable params: 2,282,823 (8.85%)
    
    ======================================================================================================================
    STEP 7: LOSSES + PROTOTYPE SHARING + EVAL HELPERS
    ======================================================================================================================
    
    ======================================================================================================================
    STEP 8: BEST STATIC PRESET SEARCH (ONCE BEFORE FL)
    ======================================================================================================================
    
    ----------------------------------------------------------------------------------------------------------------------
    Best static preset search ranking — DS1
    ----------------------------------------------------------------------------------------------------------------------
    dataset  preset_id     theta_name                                                theta_str    score  proxy_train_loss  proxy_train_acc  proxy_val_acc  proxy_val_f1_macro  proxy_val_precision_macro  proxy_val_recall_macro  proxy_val_log_loss
        ds1          4 race_edge_plus (g=1.02, a=0.32, b=6.00, t=2.60, k=3, eg=0.15, mix=0.84) 0.824184          0.576733         0.587891       0.831858            0.821626                   0.854487                0.830853            0.510769
        ds1          2   race_texture (g=0.92, a=0.34, b=5.80, t=2.30, k=7, eg=0.10, mix=0.80) 0.822392          0.573553         0.593750       0.823009            0.822186                   0.859710                0.825298            0.557029
        ds1          1     race_sharp (g=1.04, a=0.30, b=5.20, t=2.50, k=5, eg=0.12, mix=0.82) 0.810200          0.567562         0.595703       0.818584            0.807406                   0.853779                0.819444            0.471595
        ds1          3    race_robust (g=1.08, a=0.22, b=4.50, t=2.80, k=5, eg=0.11, mix=0.76) 0.744094          0.545664         0.632812       0.743363            0.744337                   0.791867                0.739649            0.577636
        ds1          5     race_focus (g=1.10, a=0.36, b=6.40, t=2.70, k=3, eg=0.17, mix=0.86) 0.694610          0.554556         0.615234       0.721239            0.685733                   0.781184                0.716204            0.645305
        ds1          0  race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78) 0.663448          0.621649         0.552734       0.699115            0.651558                   0.806443                0.689418            0.716332
    
    ----------------------------------------------------------------------------------------------------------------------
    Best static preset search ranking — DS2
    ----------------------------------------------------------------------------------------------------------------------
    dataset  preset_id     theta_name                                                theta_str    score  proxy_train_loss  proxy_train_acc  proxy_val_acc  proxy_val_f1_macro  proxy_val_precision_macro  proxy_val_recall_macro  proxy_val_log_loss
        ds2          1  race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78) 0.758740          0.620948         0.566406       0.769531            0.755142                   0.821016                0.761932            0.613245
        ds2          2    race_robust (g=1.08, a=0.22, b=4.50, t=2.80, k=5, eg=0.11, mix=0.76) 0.758351          0.606860         0.582031       0.769531            0.754624                   0.772587                0.758045            0.635586
        ds2          0      race_soft (g=0.95, a=0.18, b=3.80, t=2.20, k=3, eg=0.08, mix=0.72) 0.694381          0.642351         0.525391       0.703125            0.691466                   0.784375                0.691905            0.708701
        ds2          3 race_smoothmix (g=0.90, a=0.20, b=4.00, t=2.10, k=7, eg=0.07, mix=0.70) 0.650276          0.604998         0.601562       0.714844            0.628754                   0.649384                0.702750            0.661355
    Chosen static preset DS1: race_edge_plus (g=1.02, a=0.32, b=6.00, t=2.60, k=3, eg=0.15, mix=0.84)
    Chosen static preset DS2: race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    
    ======================================================================================================================
    STEP 9: TRUE FEDERATED TRAINING — FULL PARTICIPATION
    ======================================================================================================================
    Clients => DS1=3, DS2=3, TOTAL=6
    Rounds: 15 | Local epochs: 2
    FedProx μ=0.01 | Proto λ=0.12
    Tempered FedAvg exponent = 0.50
    Static preset DS1: race_edge_plus (g=1.02, a=0.32, b=6.00, t=2.60, k=3, eg=0.15, mix=0.84)
    Static preset DS2: race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    
    ======================================================================================================================
    ROUND 1/15 | selected=[0, 1, 2, 3, 4, 5]
    ======================================================================================================================
    Client 0 (ds1) | train_acc=0.7306 | val_acc=0.8657 | val_f1=0.7867 | val_auc=0.9730 | reward=0.8065 | theta=race_edge_plus (g=1.02, a=0.32, b=6.00, t=2.60, k=3, eg=0.15, mix=0.84)
    Client 1 (ds1) | train_acc=0.4800 | val_acc=0.8333 | val_f1=0.8807 | val_auc=nan | reward=0.8689 | theta=race_edge_plus (g=1.02, a=0.32, b=6.00, t=2.60, k=3, eg=0.15, mix=0.84)
    Client 2 (ds1) | train_acc=0.7298 | val_acc=0.7778 | val_f1=0.6272 | val_auc=0.9183 | reward=0.6648 | theta=race_edge_plus (g=1.02, a=0.32, b=6.00, t=2.60, k=3, eg=0.15, mix=0.84)
    Client 3 (ds2) | train_acc=0.7893 | val_acc=0.9070 | val_f1=0.6998 | val_auc=0.8930 | reward=0.7516 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    Client 4 (ds2) | train_acc=0.8008 | val_acc=0.9306 | val_f1=0.8237 | val_auc=nan | reward=0.8504 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    Client 5 (ds2) | train_acc=0.7557 | val_acc=0.8287 | val_f1=0.8213 | val_auc=0.9730 | reward=0.8232 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    GLOBAL VAL (Round 1) | global_acc=0.8528 | global_f1=0.7948 | ds1_acc=0.8393 | ds1_f1=0.7634 | ds2_acc=0.8558 | ds2_f1=0.8015 | reward=0.9161 | round_time=157.6s
    
    ======================================================================================================================
    ROUND 2/15 | selected=[0, 1, 2, 3, 4, 5]
    ======================================================================================================================
    Client 0 (ds1) | train_acc=0.8765 | val_acc=0.9104 | val_f1=0.8596 | val_auc=0.9780 | reward=0.8723 | theta=race_edge_plus (g=1.02, a=0.32, b=6.00, t=2.60, k=3, eg=0.15, mix=0.84)
    Client 1 (ds1) | train_acc=0.8640 | val_acc=0.7778 | val_f1=0.7481 | val_auc=nan | reward=0.7556 | theta=race_edge_plus (g=1.02, a=0.32, b=6.00, t=2.60, k=3, eg=0.15, mix=0.84)
    Client 2 (ds1) | train_acc=0.7854 | val_acc=0.8889 | val_f1=0.6968 | val_auc=0.8589 | reward=0.7449 | theta=race_edge_plus (g=1.02, a=0.32, b=6.00, t=2.60, k=3, eg=0.15, mix=0.84)
    Client 3 (ds2) | train_acc=0.8887 | val_acc=0.9070 | val_f1=0.6953 | val_auc=0.9027 | reward=0.7482 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    Client 4 (ds2) | train_acc=0.9194 | val_acc=0.9583 | val_f1=0.7327 | val_auc=nan | reward=0.7891 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    Client 5 (ds2) | train_acc=0.8372 | val_acc=0.8564 | val_f1=0.8541 | val_auc=0.9740 | reward=0.8547 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    GLOBAL VAL (Round 2) | global_acc=0.8797 | global_f1=0.8095 | ds1_acc=0.8839 | ds1_f1=0.8024 | ds2_acc=0.8788 | ds2_f1=0.8111 | reward=0.9488 | round_time=152.5s
    
    ======================================================================================================================
    ROUND 3/15 | selected=[0, 1, 2, 3, 4, 5]
    ======================================================================================================================
    Client 0 (ds1) | train_acc=0.8531 | val_acc=0.8806 | val_f1=0.7958 | val_auc=0.9820 | reward=0.8170 | theta=race_edge_plus (g=1.02, a=0.32, b=6.00, t=2.60, k=3, eg=0.15, mix=0.84)
    Client 1 (ds1) | train_acc=0.7520 | val_acc=0.9444 | val_f1=0.6522 | val_auc=nan | reward=0.7252 | theta=race_edge_plus (g=1.02, a=0.32, b=6.00, t=2.60, k=3, eg=0.15, mix=0.84)
    Client 2 (ds1) | train_acc=0.8131 | val_acc=0.8519 | val_f1=0.6750 | val_auc=0.9893 | reward=0.7192 | theta=race_edge_plus (g=1.02, a=0.32, b=6.00, t=2.60, k=3, eg=0.15, mix=0.84)
    Client 3 (ds2) | train_acc=0.8895 | val_acc=0.9651 | val_f1=0.7315 | val_auc=0.9252 | reward=0.7899 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    Client 4 (ds2) | train_acc=0.8947 | val_acc=0.9306 | val_f1=0.7255 | val_auc=nan | reward=0.7768 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    Client 5 (ds2) | train_acc=0.8619 | val_acc=0.9475 | val_f1=0.9484 | val_auc=0.9956 | reward=0.9482 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    GLOBAL VAL (Round 3) | global_acc=0.9367 | global_f1=0.8572 | ds1_acc=0.8839 | ds1_f1=0.7436 | ds2_acc=0.9481 | ds2_f1=0.8817 | reward=0.9553 | round_time=205.3s
    
    ======================================================================================================================
    ROUND 4/15 | selected=[0, 1, 2, 3, 4, 5]
    ======================================================================================================================
    Client 0 (ds1) | train_acc=0.9276 | val_acc=0.9254 | val_f1=0.8743 | val_auc=0.9920 | reward=0.8871 | theta=race_edge_plus (g=1.02, a=0.32, b=6.00, t=2.60, k=3, eg=0.15, mix=0.84)
    Client 1 (ds1) | train_acc=0.8920 | val_acc=0.9444 | val_f1=0.6522 | val_auc=nan | reward=0.7252 | theta=race_edge_plus (g=1.02, a=0.32, b=6.00, t=2.60, k=3, eg=0.15, mix=0.84)
    Client 2 (ds1) | train_acc=0.8990 | val_acc=0.8889 | val_f1=0.7024 | val_auc=0.9493 | reward=0.7490 | theta=race_edge_plus (g=1.02, a=0.32, b=6.00, t=2.60, k=3, eg=0.15, mix=0.84)
    Client 3 (ds2) | train_acc=0.9507 | val_acc=0.9419 | val_f1=0.4721 | val_auc=0.9896 | reward=0.5895 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    Client 4 (ds2) | train_acc=0.9677 | val_acc=1.0000 | val_f1=1.0000 | val_auc=nan | reward=1.0000 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    Client 5 (ds2) | train_acc=0.9229 | val_acc=0.9558 | val_f1=0.9542 | val_auc=0.9915 | reward=0.9546 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    GLOBAL VAL (Round 4) | global_acc=0.9525 | global_f1=0.8660 | ds1_acc=0.9196 | ds1_f1=0.7971 | ds2_acc=0.9596 | ds2_f1=0.8808 | reward=0.9883 | round_time=170.1s
    
    ======================================================================================================================
    ROUND 5/15 | selected=[0, 1, 2, 3, 4, 5]
    ======================================================================================================================
    Client 0 (ds1) | train_acc=0.9286 | val_acc=0.8955 | val_f1=0.8465 | val_auc=0.9838 | reward=0.8587 | theta=race_edge_plus (g=1.02, a=0.32, b=6.00, t=2.60, k=3, eg=0.15, mix=0.84)
    Client 1 (ds1) | train_acc=0.9720 | val_acc=1.0000 | val_f1=1.0000 | val_auc=nan | reward=1.0000 | theta=race_edge_plus (g=1.02, a=0.32, b=6.00, t=2.60, k=3, eg=0.15, mix=0.84)
    Client 2 (ds1) | train_acc=0.9646 | val_acc=0.9259 | val_f1=0.8690 | val_auc=0.9950 | reward=0.8833 | theta=race_edge_plus (g=1.02, a=0.32, b=6.00, t=2.60, k=3, eg=0.15, mix=0.84)
    Client 3 (ds2) | train_acc=0.9571 | val_acc=0.9651 | val_f1=0.4867 | val_auc=0.9998 | reward=0.6063 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    Client 4 (ds2) | train_acc=0.9639 | val_acc=1.0000 | val_f1=1.0000 | val_auc=nan | reward=1.0000 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    Client 5 (ds2) | train_acc=0.9291 | val_acc=0.9669 | val_f1=0.9671 | val_auc=0.9946 | reward=0.9671 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    GLOBAL VAL (Round 5) | global_acc=0.9620 | global_f1=0.8895 | ds1_acc=0.9196 | ds1_f1=0.8766 | ds2_acc=0.9712 | ds2_f1=0.8922 | reward=1.0328 | round_time=168.2s
    
    ======================================================================================================================
    ROUND 6/15 | selected=[0, 1, 2, 3, 4, 5]
    ======================================================================================================================
    Client 0 (ds1) | train_acc=0.9571 | val_acc=0.9403 | val_f1=0.9177 | val_auc=0.9981 | reward=0.9233 | theta=race_edge_plus (g=1.02, a=0.32, b=6.00, t=2.60, k=3, eg=0.15, mix=0.84)
    Client 1 (ds1) | train_acc=0.9040 | val_acc=0.9444 | val_f1=0.9585 | val_auc=nan | reward=0.9550 | theta=race_edge_plus (g=1.02, a=0.32, b=6.00, t=2.60, k=3, eg=0.15, mix=0.84)
    Client 2 (ds1) | train_acc=0.9520 | val_acc=0.8889 | val_f1=0.8159 | val_auc=0.9950 | reward=0.8341 | theta=race_edge_plus (g=1.02, a=0.32, b=6.00, t=2.60, k=3, eg=0.15, mix=0.84)
    Client 3 (ds2) | train_acc=0.9666 | val_acc=0.9651 | val_f1=0.9005 | val_auc=0.9995 | reward=0.9167 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    Client 4 (ds2) | train_acc=0.9782 | val_acc=0.9861 | val_f1=0.9636 | val_auc=nan | reward=0.9693 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    Client 5 (ds2) | train_acc=0.9508 | val_acc=0.9669 | val_f1=0.9666 | val_auc=0.9961 | reward=0.9666 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    GLOBAL VAL (Round 6) | global_acc=0.9620 | global_f1=0.9454 | ds1_acc=0.9286 | ds1_f1=0.8997 | ds2_acc=0.9692 | ds2_f1=0.9552 | reward=1.0689 | round_time=169.3s
    
    ======================================================================================================================
    ROUND 7/15 | selected=[0, 1, 2, 3, 4, 5]
    ======================================================================================================================
    Client 0 (ds1) | train_acc=0.9439 | val_acc=0.9552 | val_f1=0.9198 | val_auc=0.9947 | reward=0.9287 | theta=race_edge_plus (g=1.02, a=0.32, b=6.00, t=2.60, k=3, eg=0.15, mix=0.84)
    Client 1 (ds1) | train_acc=0.9280 | val_acc=1.0000 | val_f1=1.0000 | val_auc=nan | reward=1.0000 | theta=race_edge_plus (g=1.02, a=0.32, b=6.00, t=2.60, k=3, eg=0.15, mix=0.84)
    Client 2 (ds1) | train_acc=0.9899 | val_acc=0.9259 | val_f1=0.8690 | val_auc=0.9661 | reward=0.8833 | theta=race_edge_plus (g=1.02, a=0.32, b=6.00, t=2.60, k=3, eg=0.15, mix=0.84)
    Client 3 (ds2) | train_acc=0.9610 | val_acc=0.9767 | val_f1=0.7386 | val_auc=0.9994 | reward=0.7981 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    Client 4 (ds2) | train_acc=0.9782 | val_acc=0.9861 | val_f1=0.7455 | val_auc=nan | reward=0.8056 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    Client 5 (ds2) | train_acc=0.9489 | val_acc=0.9669 | val_f1=0.9661 | val_auc=0.9978 | reward=0.9663 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    GLOBAL VAL (Round 7) | global_acc=0.9684 | global_f1=0.9019 | ds1_acc=0.9554 | ds1_f1=0.9205 | ds2_acc=0.9712 | ds2_f1=0.8979 | reward=1.0602 | round_time=168.9s
    
    ======================================================================================================================
    ROUND 8/15 | selected=[0, 1, 2, 3, 4, 5]
    ======================================================================================================================
    Client 0 (ds1) | train_acc=0.9459 | val_acc=0.9701 | val_f1=0.9612 | val_auc=0.9963 | reward=0.9635 | theta=race_edge_plus (g=1.02, a=0.32, b=6.00, t=2.60, k=3, eg=0.15, mix=0.84)
    Client 1 (ds1) | train_acc=0.9800 | val_acc=1.0000 | val_f1=1.0000 | val_auc=nan | reward=1.0000 | theta=race_edge_plus (g=1.02, a=0.32, b=6.00, t=2.60, k=3, eg=0.15, mix=0.84)
    Client 2 (ds1) | train_acc=0.9823 | val_acc=0.8889 | val_f1=0.8210 | val_auc=0.9864 | reward=0.8380 | theta=race_edge_plus (g=1.02, a=0.32, b=6.00, t=2.60, k=3, eg=0.15, mix=0.84)
    Client 3 (ds2) | train_acc=0.9762 | val_acc=0.9884 | val_f1=0.7478 | val_auc=0.9995 | reward=0.8080 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    Client 4 (ds2) | train_acc=0.9896 | val_acc=0.9722 | val_f1=0.9321 | val_auc=nan | reward=0.9421 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    Client 5 (ds2) | train_acc=0.9534 | val_acc=0.9724 | val_f1=0.9719 | val_auc=0.9956 | reward=0.9720 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    GLOBAL VAL (Round 8) | global_acc=0.9715 | global_f1=0.9301 | ds1_acc=0.9554 | ds1_f1=0.9337 | ds2_acc=0.9750 | ds2_f1=0.9293 | reward=1.0808 | round_time=169.7s
    
    ======================================================================================================================
    ROUND 9/15 | selected=[0, 1, 2, 3, 4, 5]
    ======================================================================================================================
    Client 0 (ds1) | train_acc=0.9622 | val_acc=0.9552 | val_f1=0.9479 | val_auc=0.9927 | reward=0.9497 | theta=race_edge_plus (g=1.02, a=0.32, b=6.00, t=2.60, k=3, eg=0.15, mix=0.84)
    Client 1 (ds1) | train_acc=0.9560 | val_acc=1.0000 | val_f1=1.0000 | val_auc=nan | reward=1.0000 | theta=race_edge_plus (g=1.02, a=0.32, b=6.00, t=2.60, k=3, eg=0.15, mix=0.84)
    Client 2 (ds1) | train_acc=0.9747 | val_acc=0.9259 | val_f1=0.8764 | val_auc=0.9846 | reward=0.8888 | theta=race_edge_plus (g=1.02, a=0.32, b=6.00, t=2.60, k=3, eg=0.15, mix=0.84)
    Client 3 (ds2) | train_acc=0.9722 | val_acc=0.9884 | val_f1=0.9931 | val_auc=0.9991 | reward=0.9919 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    Client 4 (ds2) | train_acc=0.9829 | val_acc=1.0000 | val_f1=1.0000 | val_auc=nan | reward=1.0000 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    Client 5 (ds2) | train_acc=0.9582 | val_acc=0.9696 | val_f1=0.9703 | val_auc=0.9987 | reward=0.9701 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    GLOBAL VAL (Round 9) | global_acc=0.9731 | global_f1=0.9712 | ds1_acc=0.9554 | ds1_f1=0.9390 | ds2_acc=0.9769 | ds2_f1=0.9782 | reward=1.1019 | round_time=171.7s
    
    ======================================================================================================================
    ROUND 10/15 | selected=[0, 1, 2, 3, 4, 5]
    ======================================================================================================================
    Client 0 (ds1) | train_acc=0.9612 | val_acc=0.9403 | val_f1=0.9316 | val_auc=0.9957 | reward=0.9338 | theta=race_edge_plus (g=1.02, a=0.32, b=6.00, t=2.60, k=3, eg=0.15, mix=0.84)
    Client 1 (ds1) | train_acc=0.9640 | val_acc=1.0000 | val_f1=1.0000 | val_auc=nan | reward=1.0000 | theta=race_edge_plus (g=1.02, a=0.32, b=6.00, t=2.60, k=3, eg=0.15, mix=0.84)
    Client 2 (ds1) | train_acc=0.9798 | val_acc=0.8889 | val_f1=0.8440 | val_auc=0.9892 | reward=0.8552 | theta=race_edge_plus (g=1.02, a=0.32, b=6.00, t=2.60, k=3, eg=0.15, mix=0.84)
    Client 3 (ds2) | train_acc=0.9754 | val_acc=0.9884 | val_f1=0.9145 | val_auc=1.0000 | reward=0.9329 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    Client 4 (ds2) | train_acc=0.9839 | val_acc=0.9861 | val_f1=0.7455 | val_auc=nan | reward=0.8056 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    Client 5 (ds2) | train_acc=0.9621 | val_acc=0.9724 | val_f1=0.9715 | val_auc=0.9973 | reward=0.9717 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    GLOBAL VAL (Round 10) | global_acc=0.9699 | global_f1=0.9291 | ds1_acc=0.9375 | ds1_f1=0.9215 | ds2_acc=0.9769 | ds2_f1=0.9308 | reward=1.0727 | round_time=167.7s
    
    ======================================================================================================================
    ROUND 11/15 | selected=[0, 1, 2, 3, 4, 5]
    ======================================================================================================================
    Client 0 (ds1) | train_acc=0.9694 | val_acc=0.9552 | val_f1=0.9464 | val_auc=0.9997 | reward=0.9486 | theta=race_edge_plus (g=1.02, a=0.32, b=6.00, t=2.60, k=3, eg=0.15, mix=0.84)
    Client 1 (ds1) | train_acc=0.9800 | val_acc=1.0000 | val_f1=1.0000 | val_auc=nan | reward=1.0000 | theta=race_edge_plus (g=1.02, a=0.32, b=6.00, t=2.60, k=3, eg=0.15, mix=0.84)
    Client 2 (ds1) | train_acc=0.9848 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | reward=1.0000 | theta=race_edge_plus (g=1.02, a=0.32, b=6.00, t=2.60, k=3, eg=0.15, mix=0.84)
    Client 3 (ds2) | train_acc=0.9865 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | reward=1.0000 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    Client 4 (ds2) | train_acc=0.9962 | val_acc=0.9861 | val_f1=0.7455 | val_auc=nan | reward=0.8056 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    Client 5 (ds2) | train_acc=0.9636 | val_acc=0.9669 | val_f1=0.9646 | val_auc=0.9978 | reward=0.9651 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    GLOBAL VAL (Round 11) | global_acc=0.9747 | global_f1=0.9450 | ds1_acc=0.9732 | ds1_f1=0.9680 | ds2_acc=0.9750 | ds2_f1=0.9401 | reward=1.1014 | round_time=168.8s
    
    ======================================================================================================================
    ROUND 12/15 | selected=[0, 1, 2, 3, 4, 5]
    ======================================================================================================================
    Client 0 (ds1) | train_acc=0.9684 | val_acc=0.9254 | val_f1=0.8951 | val_auc=0.9995 | reward=0.9026 | theta=race_edge_plus (g=1.02, a=0.32, b=6.00, t=2.60, k=3, eg=0.15, mix=0.84)
    Client 1 (ds1) | train_acc=0.9800 | val_acc=1.0000 | val_f1=1.0000 | val_auc=nan | reward=1.0000 | theta=race_edge_plus (g=1.02, a=0.32, b=6.00, t=2.60, k=3, eg=0.15, mix=0.84)
    Client 2 (ds1) | train_acc=0.9545 | val_acc=0.9259 | val_f1=0.8690 | val_auc=0.9914 | reward=0.8833 | theta=race_edge_plus (g=1.02, a=0.32, b=6.00, t=2.60, k=3, eg=0.15, mix=0.84)
    Client 3 (ds2) | train_acc=0.9936 | val_acc=0.9884 | val_f1=0.7455 | val_auc=1.0000 | reward=0.8062 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    Client 4 (ds2) | train_acc=0.9810 | val_acc=1.0000 | val_f1=1.0000 | val_auc=nan | reward=1.0000 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    Client 5 (ds2) | train_acc=0.9666 | val_acc=0.9696 | val_f1=0.9699 | val_auc=0.9988 | reward=0.9698 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    GLOBAL VAL (Round 12) | global_acc=0.9699 | global_f1=0.9314 | ds1_acc=0.9375 | ds1_f1=0.9057 | ds2_acc=0.9769 | ds2_f1=0.9369 | reward=1.0673 | round_time=169.2s
    
    ======================================================================================================================
    ROUND 13/15 | selected=[0, 1, 2, 3, 4, 5]
    ======================================================================================================================
    Client 0 (ds1) | train_acc=0.9684 | val_acc=0.9701 | val_f1=0.9612 | val_auc=0.9987 | reward=0.9635 | theta=race_edge_plus (g=1.02, a=0.32, b=6.00, t=2.60, k=3, eg=0.15, mix=0.84)
    Client 1 (ds1) | train_acc=0.9760 | val_acc=1.0000 | val_f1=1.0000 | val_auc=nan | reward=1.0000 | theta=race_edge_plus (g=1.02, a=0.32, b=6.00, t=2.60, k=3, eg=0.15, mix=0.84)
    Client 2 (ds1) | train_acc=0.9899 | val_acc=0.9630 | val_f1=0.9143 | val_auc=1.0000 | reward=0.9265 | theta=race_edge_plus (g=1.02, a=0.32, b=6.00, t=2.60, k=3, eg=0.15, mix=0.84)
    Client 3 (ds2) | train_acc=0.9873 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | reward=1.0000 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    Client 4 (ds2) | train_acc=0.9905 | val_acc=1.0000 | val_f1=1.0000 | val_auc=nan | reward=1.0000 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    Client 5 (ds2) | train_acc=0.9657 | val_acc=0.9779 | val_f1=0.9780 | val_auc=0.9981 | reward=0.9780 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    GLOBAL VAL (Round 13) | global_acc=0.9826 | global_f1=0.9796 | ds1_acc=0.9732 | ds1_f1=0.9561 | ds2_acc=0.9846 | ds2_f1=0.9847 | reward=1.1166 | round_time=168.3s
    
    ======================================================================================================================
    ROUND 14/15 | selected=[0, 1, 2, 3, 4, 5]
    ======================================================================================================================
    Client 0 (ds1) | train_acc=0.9847 | val_acc=0.9701 | val_f1=0.9616 | val_auc=0.9946 | reward=0.9637 | theta=race_edge_plus (g=1.02, a=0.32, b=6.00, t=2.60, k=3, eg=0.15, mix=0.84)
    Client 1 (ds1) | train_acc=0.9680 | val_acc=1.0000 | val_f1=1.0000 | val_auc=nan | reward=1.0000 | theta=race_edge_plus (g=1.02, a=0.32, b=6.00, t=2.60, k=3, eg=0.15, mix=0.84)
    Client 2 (ds1) | train_acc=0.9874 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | reward=1.0000 | theta=race_edge_plus (g=1.02, a=0.32, b=6.00, t=2.60, k=3, eg=0.15, mix=0.84)
    Client 3 (ds2) | train_acc=0.9817 | val_acc=0.9651 | val_f1=0.6533 | val_auc=0.9968 | reward=0.7312 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    Client 4 (ds2) | train_acc=0.9981 | val_acc=0.9861 | val_f1=0.7455 | val_auc=nan | reward=0.8056 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    Client 5 (ds2) | train_acc=0.9689 | val_acc=0.9834 | val_f1=0.9827 | val_auc=0.9990 | reward=0.9829 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    GLOBAL VAL (Round 14) | global_acc=0.9810 | global_f1=0.9099 | ds1_acc=0.9821 | ds1_f1=0.9770 | ds2_acc=0.9808 | ds2_f1=0.8954 | reward=1.0850 | round_time=167.2s
    
    ======================================================================================================================
    ROUND 15/15 | selected=[0, 1, 2, 3, 4, 5]
    ======================================================================================================================
    Client 0 (ds1) | train_acc=0.9745 | val_acc=0.9851 | val_f1=0.9750 | val_auc=0.9963 | reward=0.9775 | theta=race_edge_plus (g=1.02, a=0.32, b=6.00, t=2.60, k=3, eg=0.15, mix=0.84)
    Client 1 (ds1) | train_acc=0.9720 | val_acc=1.0000 | val_f1=1.0000 | val_auc=nan | reward=1.0000 | theta=race_edge_plus (g=1.02, a=0.32, b=6.00, t=2.60, k=3, eg=0.15, mix=0.84)
    Client 2 (ds1) | train_acc=0.9899 | val_acc=0.9630 | val_f1=0.9106 | val_auc=1.0000 | reward=0.9237 | theta=race_edge_plus (g=1.02, a=0.32, b=6.00, t=2.60, k=3, eg=0.15, mix=0.84)
    Client 3 (ds2) | train_acc=0.9952 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | reward=1.0000 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    Client 4 (ds2) | train_acc=0.9934 | val_acc=1.0000 | val_f1=1.0000 | val_auc=nan | reward=1.0000 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    Client 5 (ds2) | train_acc=0.9717 | val_acc=0.9779 | val_f1=0.9784 | val_auc=0.9981 | reward=0.9783 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    GLOBAL VAL (Round 15) | global_acc=0.9842 | global_f1=0.9812 | ds1_acc=0.9821 | ds1_f1=0.9635 | ds2_acc=0.9846 | ds2_f1=0.9850 | reward=1.1217 | round_time=169.2s
    
    ======================================================================================================================
    TRAINING COMPLETE | total_time=2544.2s | best_round=15 | best_reward=1.1217
    ======================================================================================================================
    
    ----------------------------------------------------------------------------------------------------------------------
    GLOBAL per-round metrics
    ----------------------------------------------------------------------------------------------------------------------
     round  round_time_s  n_selected_clients  active_fraction  global_reward  global_acc  global_f1_macro  global_precision_macro  global_recall_macro  global_log_loss  global_loss_ce  global_eval_time_s  ds1_acc  ds1_f1_macro  ds1_log_loss  ds2_acc  ds2_f1_macro  ds2_log_loss
         1    157.574346                   6              1.0       0.916070    0.852848         0.794770                0.803982             0.803376         0.386365        0.375466            2.546025 0.839286      0.763363      0.508925 0.855769      0.801535      0.359967
         2    152.497802                   6              1.0       0.948832    0.879747         0.809530                0.812648             0.812455         0.318605        0.321644            2.351438 0.883929      0.802446      0.348800 0.878846      0.811056      0.312101
         3    205.295835                   6              1.0       0.955282    0.936709         0.857190                0.860910             0.855907         0.192313        0.187756            2.328449 0.883929      0.743620      0.270452 0.948077      0.881651      0.175483
         4    170.073389                   6              1.0       0.988294    0.952532         0.865957                0.865965             0.867385         0.147350        0.146312            2.322730 0.919643      0.797147      0.175976 0.959615      0.880777      0.141185
         5    168.223094                   6              1.0       1.032759    0.962025         0.889452                0.895618             0.886022         0.127330        0.126473            2.324539 0.919643      0.876593      0.197092 0.971154      0.892222      0.112304
         6    169.274167                   6              1.0       1.068871    0.962025         0.945402                0.944076             0.958248         0.117015        0.113860            2.323225 0.928571      0.899710      0.191260 0.969231      0.955243      0.101023
         7    168.940587                   6              1.0       1.060155    0.968354         0.901933                0.908142             0.898558         0.128164        0.126063            2.328752 0.955357      0.920471      0.185936 0.971154      0.897941      0.115721
         8    169.675938                   6              1.0       1.080780    0.971519         0.930104                0.927857             0.935192         0.112115        0.109145            2.321285 0.955357      0.933655      0.176145 0.975000      0.929339      0.098324
         9    171.730935                   6              1.0       1.101942    0.973101         0.971223                0.973667             0.971936         0.106798        0.104101            2.328506 0.955357      0.939017      0.156390 0.976923      0.978160      0.096117
        10    167.686864                   6              1.0       1.072720    0.969937         0.929118                0.925643             0.942723         0.113131        0.109936            2.315437 0.937500      0.921488      0.204546 0.976923      0.930762      0.093442
        11    168.757226                   6              1.0       1.101366    0.974684         0.945023                0.947232             0.943591         0.091395        0.089102            2.310979 0.973214      0.967963      0.087367 0.975000      0.940082      0.092263
        12    169.206425                   6              1.0       1.067321    0.969937         0.931396                0.936090             0.931888         0.091724        0.092768            2.315310 0.937500      0.905662      0.177204 0.976923      0.936938      0.073313
        13    168.282810                   6              1.0       1.116611    0.982595         0.979647                0.979204             0.981697         0.076024        0.076170            2.318499 0.973214      0.956145      0.111769 0.984615      0.984709      0.068325
        14    167.162611                   6              1.0       1.085031    0.981013         0.909866                0.902667             0.923365         0.074309        0.070125            2.319318 0.982143      0.977012      0.094917 0.980769      0.895403      0.069871
        15    169.152574                   6              1.0       1.121730    0.984177         0.981169                0.983821             0.980693         0.068865        0.066403            2.312506 0.982143      0.963459      0.098020 0.984615      0.984984      0.062585
    
    ----------------------------------------------------------------------------------------------------------------------
    LOCAL per-client per-round metrics
    ----------------------------------------------------------------------------------------------------------------------
     round   client dataset  selected     theta_name                                                theta_str  gamma_power  alpha_contrast_weight  beta_contrast_sharpness  tau_clip  k_blur_kernel_size  edge_gain  blend_mix  train_loss  train_ce_loss  train_proto_loss  train_acc  val_size  val_loss_ce  val_acc  val_precision_macro  val_recall_macro  val_f1_macro  val_precision_weighted  val_recall_weighted  val_f1_weighted  val_log_loss  val_eval_time_s  val_auc_roc_macro_ovr  val_auc_class_0  val_auc_class_1  val_auc_class_2  val_auc_class_3  val_fusion_gate_mean_raw  val_fusion_gate_mean_enh  val_fusion_gate_mean_res  val_fusion_gate_entropy   reward
         1 client_0     ds1         1 race_edge_plus (g=1.02, a=0.32, b=6.00, t=2.60, k=3, eg=0.15, mix=0.84)         1.02                   0.32                      6.0       2.6                   3       0.15       0.84    0.480454       0.411765          0.000000   0.730612        67     0.344065 0.865672             0.805195          0.780303      0.786713                0.868773             0.865672         0.861079      0.398804         0.882552               0.973049         0.952564         0.942623         0.999031         0.997980                  0.290371                  0.421479                  0.288150                 1.452588 0.806453
         1 client_1     ds1         1 race_edge_plus (g=1.02, a=0.32, b=6.00, t=2.60, k=3, eg=0.15, mix=0.84)         1.02                   0.32                      6.0       2.6                   3       0.15       0.84    0.740023       0.713592          0.000000   0.480000        18     0.617496 0.833333             0.888889          0.909091      0.880702                0.888889             0.833333         0.836842      0.656346         0.377947                    NaN         0.987013              NaN         1.000000         1.000000                  0.000271                  0.070496                  0.929233                 0.365765 0.868860
         1 client_2     ds1         1 race_edge_plus (g=1.02, a=0.32, b=6.00, t=2.60, k=3, eg=0.15, mix=0.84)         1.02                   0.32                      6.0       2.6                   3       0.15       0.84    0.475086       0.439001          0.000000   0.729798        27     0.693128 0.777778             0.597222          0.700000      0.627193                0.769547             0.777778         0.759584      0.683905         0.490952               0.918276         0.820000         0.885714         0.967391         1.000000                  0.413989                  0.348002                  0.238009                 1.467947 0.664839
         1 client_3     ds2         1  race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)         1.00                   0.24                      4.6       2.4                   5       0.10       0.78    0.381210       0.324318          0.000000   0.789348        86     0.378648 0.906977             0.692354          0.709552      0.699751                0.903059             0.906977         0.903271      0.389088         1.014402               0.893035         0.682353         0.930320         0.959468         1.000000                  0.982748                  0.013878                  0.003374                 0.132114 0.751558
         1 client_4     ds2         1  race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)         1.00                   0.24                      4.6       2.4                   5       0.10       0.78    0.383017       0.322623          0.000000   0.800759        72     0.223689 0.930556             0.811111          0.841026      0.823657                0.939352             0.930556         0.933358      0.233015         1.584398                    NaN         1.000000              NaN         0.996115         0.970149                  0.084038                  0.884944                  0.031018                 0.569179 0.850382
         1 client_5     ds2         1  race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)         1.00                   0.24                      4.6       2.4                   5       0.10       0.78    0.467456       0.362379          0.000000   0.755748       362     0.374982 0.828729             0.840059          0.824902      0.821315                0.838884             0.828729         0.822447      0.378299         3.670117               0.973044         0.971372         0.930653         0.996836         0.993316                  0.601588                  0.232851                  0.165561                 1.318121 0.823169
         2 client_0     ds1         1 race_edge_plus (g=1.02, a=0.32, b=6.00, t=2.60, k=3, eg=0.15, mix=0.84)         1.02                   0.32                      6.0       2.6                   3       0.15       0.84    0.277705       0.196474          0.314897   0.876531        67     0.268698 0.910448             0.866841          0.855303      0.859589                0.910448             0.910448         0.908913      0.293121         0.914335               0.977968         0.982051         0.942623         0.992248         0.994949                  0.421772                  0.329331                  0.248897                 1.541043 0.872303
         2 client_1     ds1         1 race_edge_plus (g=1.02, a=0.32, b=6.00, t=2.60, k=3, eg=0.15, mix=0.84)         1.02                   0.32                      6.0       2.6                   3       0.15       0.84    0.275912       0.221083          0.325062   0.864000        18     0.648643 0.777778             0.722222          0.878788      0.748148                0.861111             0.777778         0.779012      0.369303         0.296272                    NaN         1.000000              NaN         1.000000         1.000000                  0.496870                  0.242681                  0.260449                 1.501029 0.755556
         2 client_2     ds1         1 race_edge_plus (g=1.02, a=0.32, b=6.00, t=2.60, k=3, eg=0.15, mix=0.84)         1.02                   0.32                      6.0       2.6                   3       0.15       0.84    0.372096       0.302927          0.343035   0.785354        27     0.507199 0.888889             0.717391          0.687500      0.696844                0.829308             0.888889         0.853082      0.473299         0.379507               0.858913         0.740000         0.750000         0.945652         1.000000                  0.987907                  0.010649                  0.001444                 0.099449 0.744855
         2 client_3     ds2         1  race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)         1.00                   0.24                      4.6       2.4                   5       0.10       0.78    0.237241       0.166294          0.268172   0.888712        86     0.289284 0.906977             0.704067          0.690058      0.695294                0.896931             0.906977         0.899152      0.307308         0.916609               0.902651         0.682353         0.956685         0.971567         1.000000                  0.476022                  0.460268                  0.063709                 1.275129 0.748215
         2 client_4     ds2         1  race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)         1.00                   0.24                      4.6       2.4                   5       0.10       0.78    0.186081       0.122508          0.248274   0.919355        72     0.112903 0.958333             0.743750          0.723214      0.732684                0.986458             0.958333         0.971131      0.116220         0.761620                    NaN         0.999188              NaN         1.000000         1.000000                  0.815309                  0.081732                  0.102959                 0.865153 0.789097
         2 client_5     ds2         1  race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)         1.00                   0.24                      4.6       2.4                   5       0.10       0.78    0.349733       0.245681          0.276703   0.837165       362     0.350549 0.856354             0.853718          0.857374      0.854146                0.859489             0.856354         0.856639      0.352200         3.523767               0.974020         0.968898         0.935734         0.996712         0.994734                  0.721506                  0.216655                  0.061839                 1.060949 0.854698
         3 client_0     ds1         1 race_edge_plus (g=1.02, a=0.32, b=6.00, t=2.60, k=3, eg=0.15, mix=0.84)         1.02                   0.32                      6.0       2.6                   3       0.15       0.84    0.354626       0.258075          0.269776   0.853061        67     0.202000 0.880597             0.797395          0.797917      0.795841                0.885182             0.880597         0.881583      0.230059         0.928846               0.981964         0.984615         0.948087         0.995155         1.000000                  0.791752                  0.166479                  0.041769                 0.855668 0.817030
         3 client_1     ds1         1 race_edge_plus (g=1.02, a=0.32, b=6.00, t=2.60, k=3, eg=0.15, mix=0.84)         1.02                   0.32                      6.0       2.6                   3       0.15       0.84    0.572017       0.504963          0.346636   0.752000        18     0.266237 0.944444             0.638889          0.666667      0.652174                0.893519             0.944444         0.917874      0.258986         0.342761                    NaN         0.974026              NaN         1.000000         1.000000                  0.595130                  0.219767                  0.185103                 1.375056 0.725242
         3 client_2     ds1         1 race_edge_plus (g=1.02, a=0.32, b=6.00, t=2.60, k=3, eg=0.15, mix=0.84)         1.02                   0.32                      6.0       2.6                   3       0.15       0.84    0.351896       0.286098          0.266406   0.813131        27     0.386814 0.851852             0.641667          0.725000      0.675000                0.802469             0.851852         0.822222      0.378330         0.477237               0.989286         1.000000         0.957143         1.000000         1.000000                  0.698098                  0.271864                  0.030038                 1.013526 0.719213
         3 client_3     ds2         1  race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)         1.00                   0.24                      4.6       2.4                   5       0.10       0.78    0.273643       0.186904          0.232664   0.889507        86     0.179023 0.965116             0.731910          0.731481      0.731539                0.953829             0.965116         0.959174      0.190783         0.893163               0.925237         0.717647         0.989956         0.993345         1.000000                  0.932248                  0.064067                  0.003685                 0.371268 0.789933
         3 client_4     ds2         1  race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)         1.00                   0.24                      4.6       2.4                   5       0.10       0.78    0.279982       0.207235          0.225455   0.894687        72     0.136698 0.930556             0.750000          0.705357      0.725490                1.000000             0.930556         0.961874      0.131523         0.740931                    NaN         0.999188              NaN         1.000000         1.000000                  0.855499                  0.120945                  0.023556                 0.682456 0.776757
         3 client_5     ds2         1  race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)         1.00                   0.24                      4.6       2.4                   5       0.10       0.78    0.356606       0.224395          0.238939   0.861855       362     0.178599 0.947514             0.952764          0.945317      0.948373                0.948813             0.947514         0.947644      0.180592         3.481030               0.995647         0.996553         0.990314         0.998759         0.996962                  0.696275                  0.219109                  0.084615                 1.119675 0.948158
         4 client_0     ds1         1 race_edge_plus (g=1.02, a=0.32, b=6.00, t=2.60, k=3, eg=0.15, mix=0.84)         1.02                   0.32                      6.0       2.6                   3       0.15       0.84    0.216558       0.149001          0.195813   0.927551        67     0.140462 0.925373             0.872549          0.878220      0.874284                0.930641             0.925373         0.926906      0.159984         0.785673               0.992039         0.991026         0.978142         1.000000         0.998990                  0.916393                  0.065783                  0.017824                 0.459977 0.887056
         4 client_1     ds1         1 race_edge_plus (g=1.02, a=0.32, b=6.00, t=2.60, k=3, eg=0.15, mix=0.84)         1.02                   0.32                      6.0       2.6                   3       0.15       0.84    0.213710       0.170856          0.209533   0.892000        18     0.063639 0.944444             0.638889          0.666667      0.652174                0.893519             0.944444         0.917874      0.069040         0.282307                    NaN         1.000000              NaN         1.000000         1.000000                  0.495945                  0.260764                  0.243291                 1.493146 0.725242
    ... showing first 20 of 90 rows
    
    ======================================================================================================================
    STEP 10: FINAL EVALUATION (USE SAME STATIC PRESETS)
    ======================================================================================================================
    
    ----------------------------------------------------------------------------------------------------------------------
    Final static preprocessing strategy
    ----------------------------------------------------------------------------------------------------------------------
    dataset                            strategy        theta_names    score  val_acc   val_f1
        ds1 best_static_preset_fixed_all_rounds ['race_edge_plus'] 0.968529 0.969027 0.968363
        ds2 best_static_preset_fixed_all_rounds  ['race_balanced'] 0.978746 0.979147 0.978613
    
    ======================================================================================================================
    STEP 11: EXTENDED METRICS + CALIBRATION + ERROR ANALYSIS
    ======================================================================================================================
    
    ----------------------------------------------------------------------------------------------------------------------
    Extended TEST metrics (DS1 vs DS2)
    ----------------------------------------------------------------------------------------------------------------------
     dataset      acc  balanced_acc  precision_macro  recall_macro  f1_macro  precision_weighted  recall_weighted  f1_weighted  log_loss      mcc    kappa  jaccard_macro  ppv_macro  npv_macro  specificity_macro  fpr_macro  fnr_macro      ece      mce  brier_multi  auc_roc_macro_ovr  auc_class_0  auc_class_1  auc_class_2  auc_class_3
    ds1_test 0.977876      0.977743         0.978736      0.977743  0.978044            0.978481         0.977876     0.977985  0.099567 0.970617 0.970491       0.957201   0.978736   0.992654           0.992594   0.007406   0.022257 0.016852 0.610775     0.042018           0.999032     0.997899     0.999787     0.999493     0.998950
    ds2_test 0.980095      0.979664         0.979078      0.979664  0.979297            0.980260         0.980095     0.980106  0.061380 0.973449 0.973403       0.959651   0.979078   0.993416           0.993486   0.006514   0.020336 0.020796 0.695346     0.026542           0.999225     0.999773     0.998111     1.000000     0.999018
    
    ----------------------------------------------------------------------------------------------------------------------
    Classwise metrics — DS1 TEST
    ----------------------------------------------------------------------------------------------------------------------
     class_id class_name  support  tp  fp  fn  tn  prevalence      ppv      npv   recall  specificity      fpr      fnr  jaccard  balanced_acc
            0     glioma       56  55   3   1 167    0.247788 0.948276 0.994048 0.982143     0.982353 0.017647 0.017857 0.932203      0.982248
            1 meningioma       55  53   0   2 171    0.243363 1.000000 0.988439 0.963636     1.000000 0.000000 0.036364 0.963636      0.981818
            2    notumor       59  58   2   1 165    0.261062 0.966667 0.993976 0.983051     0.988024 0.011976 0.016949 0.950820      0.985537
            3  pituitary       56  55   0   1 170    0.247788 1.000000 0.994152 0.982143     1.000000 0.000000 0.017857 0.982143      0.991071
    
    ----------------------------------------------------------------------------------------------------------------------
    Classwise metrics — DS2 TEST
    ----------------------------------------------------------------------------------------------------------------------
     class_id class_name  support  tp  fp  fn  tn  prevalence      ppv      npv   recall  specificity      fpr      fnr  jaccard  balanced_acc
            0     glioma      244 242   9   2 802    0.231280 0.964143 0.997512 0.991803     0.988903 0.011097 0.008197 0.956522      0.990353
            1 meningioma      247 239   8   8 800    0.234123 0.967611 0.990099 0.967611     0.990099 0.009901 0.032389 0.937255      0.978855
            2    notumor      300 298   0   2 755    0.284360 1.000000 0.997358 0.993333     1.000000 0.000000 0.006667 0.993333      0.996667
            3  pituitary      264 255   4   9 787    0.250237 0.984556 0.988693 0.965909     0.994943 0.005057 0.034091 0.951493      0.980426
    
    ----------------------------------------------------------------------------------------------------------------------
    Top confusion pairs — DS1 TEST
    ----------------------------------------------------------------------------------------------------------------------
    true_class pred_class  count
        glioma    notumor      1
    meningioma     glioma      1
       notumor     glioma      1
    meningioma    notumor      1
     pituitary     glioma      1
        glioma meningioma      0
        glioma  pituitary      0
    meningioma  pituitary      0
       notumor meningioma      0
       notumor  pituitary      0
    
    ----------------------------------------------------------------------------------------------------------------------
    Top confusion pairs — DS2 TEST
    ----------------------------------------------------------------------------------------------------------------------
    true_class pred_class  count
     pituitary     glioma      5
    meningioma     glioma      4
    meningioma  pituitary      4
     pituitary meningioma      4
        glioma meningioma      2
       notumor meningioma      2
        glioma  pituitary      0
        glioma    notumor      0
       notumor     glioma      0
    meningioma    notumor      0
    
    ----------------------------------------------------------------------------------------------------------------------
    Calibration bins — DS1 TEST
    ----------------------------------------------------------------------------------------------------------------------
     bin_id  bin_left  bin_right  bin_confidence  bin_accuracy  bin_gap  bin_count
          0  0.000000   0.083333             NaN           NaN      NaN          0
          1  0.083333   0.166667             NaN           NaN      NaN          0
          2  0.166667   0.250000             NaN           NaN      NaN          0
          3  0.250000   0.333333             NaN           NaN      NaN          0
          4  0.333333   0.416667             NaN           NaN      NaN          0
          5  0.416667   0.500000        0.482873      1.000000 0.517127          1
          6  0.500000   0.583333        0.508165      1.000000 0.491835          1
          7  0.583333   0.666667        0.610775      0.000000 0.610775          1
          8  0.666667   0.750000        0.726943      1.000000 0.273057          1
          9  0.750000   0.833333        0.777954      1.000000 0.222046          2
         10  0.833333   0.916667        0.866231      0.500000 0.366231          2
         11  0.916667   1.000000        0.982848      0.986239 0.003391        218
    
    ----------------------------------------------------------------------------------------------------------------------
    Calibration bins — DS2 TEST
    ----------------------------------------------------------------------------------------------------------------------
     bin_id  bin_left  bin_right  bin_confidence  bin_accuracy  bin_gap  bin_count
          0  0.000000   0.083333             NaN           NaN      NaN          0
          1  0.083333   0.166667             NaN           NaN      NaN          0
          2  0.166667   0.250000             NaN           NaN      NaN          0
          3  0.250000   0.333333             NaN           NaN      NaN          0
          4  0.333333   0.416667             NaN           NaN      NaN          0
          5  0.416667   0.500000        0.478583      0.000000 0.478583          1
          6  0.500000   0.583333        0.547916      0.333333 0.214583          3
          7  0.583333   0.666667        0.623728      0.400000 0.223728          5
          8  0.666667   0.750000        0.695346      0.000000 0.695346          6
          9  0.750000   0.833333        0.793720      0.800000 0.006280         10
         10  0.833333   0.916667        0.868482      0.750000 0.118482         12
         11  0.916667   1.000000        0.982277      0.996071 0.013794       1018
    
    ----------------------------------------------------------------------------------------------------------------------
    VAL + TEST tables
    ----------------------------------------------------------------------------------------------------------------------
                      setting split      dataset      acc  precision_macro  recall_macro  f1_macro  precision_weighted  recall_weighted  f1_weighted  log_loss  auc_roc_macro_ovr  loss_ce  eval_time_s  balanced_acc      mcc    kappa  ppv_macro  npv_macro  specificity_macro      ece      mce  brier_multi
    ARCF-Net BestStaticPreset   VAL          ds1 0.969027         0.970805      0.968056  0.968363            0.970207         0.969027     0.968584  0.126638           0.995220 0.120071     4.670681           NaN      NaN      NaN        NaN        NaN                NaN      NaN      NaN          NaN
    ARCF-Net BestStaticPreset   VAL          ds2 0.979147         0.978589      0.978700  0.978613            0.979193         0.979147     0.979139  0.069300           0.998166 0.069255    20.483412           NaN      NaN      NaN        NaN        NaN                NaN      NaN      NaN          NaN
    ARCF-Net BestStaticPreset   VAL global_equal 0.974087         0.974697      0.973378  0.973488            0.974700         0.974087     0.973861  0.097969           0.996693 0.094663    12.577047           NaN      NaN      NaN        NaN        NaN                NaN      NaN      NaN          NaN
    ARCF-Net BestStaticPreset  TEST          ds1 0.977876         0.978736      0.977743  0.978044            0.978481         0.977876     0.977985  0.099567           0.999032 0.094590     4.920918      0.977743 0.970617 0.970491   0.978736   0.992654           0.992594 0.016852 0.610775     0.042018
    ARCF-Net BestStaticPreset  TEST          ds2 0.980095         0.979078      0.979664  0.979297            0.980260         0.980095     0.980106  0.061380           0.999225 0.061338    20.537440      0.979664 0.973449 0.973403   0.979078   0.993416           0.993486 0.020796 0.695346     0.026542
    ARCF-Net BestStaticPreset  TEST global_equal 0.978985         0.978907      0.978704  0.978670            0.979370         0.978985     0.979046  0.080474           0.999129 0.077964    12.729179           NaN      NaN      NaN        NaN        NaN                NaN      NaN      NaN          NaN
    
    Selection summary:
    - Best round: 15 | best_reward=1.1217
    - DS1 static strategy: best_static_preset_fixed_all_rounds | names=['race_edge_plus']
    - DS2 static strategy: best_static_preset_fixed_all_rounds | names=['race_balanced']
    
    ======================================================================================================================
    STEP 12: PREPROCESSING VALIDATION
    ======================================================================================================================
    
    ----------------------------------------------------------------------------------------------------------------------
    Preprocessing validation summary (DS1 VAL sample)
    ----------------------------------------------------------------------------------------------------------------------
                metric     mean      std       min      max
    edge_energy_before 0.041596 0.021720  0.014947 0.165471
     edge_energy_after 0.159103 0.033185  0.093722 0.285806
        entropy_before 5.820408 0.628255  3.547314 7.239670
         entropy_after 6.559339 0.597441  3.875062 7.564398
       contrast_before 0.187640 0.052597  0.101468 0.363759
        contrast_after 0.238172 0.021439  0.204715 0.334330
       edge_gain_ratio 4.217191 1.039354  1.458521 7.844795
         entropy_delta 0.738932 0.257089  0.257075 1.537737
        contrast_delta 0.050532 0.034325 -0.032019 0.117839
    
    ----------------------------------------------------------------------------------------------------------------------
    Preprocessing validation summary (DS2 VAL sample)
    ----------------------------------------------------------------------------------------------------------------------
                metric     mean      std       min      max
    edge_energy_before 0.069022 0.036905  0.016726 0.375572
     edge_energy_after 0.139198 0.035390  0.085889 0.419808
        entropy_before 6.866192 0.472092  5.135309 7.798752
         entropy_after 7.384848 0.273545  5.927784 7.898935
       contrast_before 0.233743 0.033322  0.144569 0.352019
        contrast_after 0.259174 0.018040  0.197374 0.337414
       edge_gain_ratio 2.204572 0.536764  1.117782 5.584722
         entropy_delta 0.518656 0.234768  0.068102 1.280064
        contrast_delta 0.025431 0.018836 -0.019016 0.088201
    
    ======================================================================================================================
    STEP 13: SAVING CHECKPOINT + CSV
    ======================================================================================================================
    Saved checkpoint: /kaggle/working/outputs/ARCFNet_Ablation_BestStaticPreset_checkpoint.pth
    Saved CSV: /kaggle/working/outputs/ARCFNet_Ablation_BestStaticPreset_outputs.csv
    
    DONE
    Method: ARCF-Net = Adaptive RACE-FELCM with CRAF Fusion Network
    Ablation: Best Static Preset, Full Participation
    Backbone: Residual Network-50
    Best round: 15
    Fixed clients => DS1=3, DS2=3, TOTAL=6
    Rounds completed: 15
    Global TEST acc: 0.9790
    Global TEST f1_macro: 0.9787
    DS1 TEST acc: 0.9779
    DS2 TEST acc: 0.9801
    DS1 static strategy: best_static_preset_fixed_all_rounds | names=['race_edge_plus']
    DS2 static strategy: best_static_preset_fixed_all_rounds | names=['race_balanced']


# **4. RL-UCB preset selection**


```python
import os
import sys
import time
import math
import copy
import hashlib
import random
import subprocess
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    log_loss, confusion_matrix, roc_auc_score,
    matthews_corrcoef, cohen_kappa_score, balanced_accuracy_score,
    jaccard_score
)

# ============================================================
# ARCF-Net ABLATION 4
# RL-UCB PRESET SELECTION / FIXED CLIENTS / FULL PARTICIPATION / NO PLOTS
# ------------------------------------------------------------
# - Uses BOTH datasets
# - Kaggle-ready
# - True FL with FedAvg + FedProx + prototype sharing
# - RL-UCB ONLY for preprocessing preset selection
# - Fixed client count per dataset
# - Full participation every round
# - No plots generated
# ============================================================

# -------------------------
# Helpers
# -------------------------
def pip_install(pkg):
    subprocess.check_call([sys.executable, "-m", "pip", "-q", "install", pkg])

def running_on_kaggle():
    return os.path.exists("/kaggle/input")

IS_KAGGLE = running_on_kaggle()

try:
    import kagglehub
except Exception:
    try:
        pip_install("kagglehub")
        import kagglehub
    except Exception:
        kagglehub = None

try:
    from torchvision import transforms
    from torchvision.models import resnet50, ResNet50_Weights
except Exception:
    pip_install("torchvision")
    from torchvision import transforms
    from torchvision.models import resnet50, ResNet50_Weights

# -------------------------
# Reproducibility + Device
# -------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if DEVICE.type == "cuda":
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

print("=" * 118)
print("ARCF-Net ABLATION 4: RL-UCB PRESET SELECTION + FIXED CLIENTS + FULL PARTICIPATION + NO PLOTS")
print("=" * 118)
print(f"ENV: {'KAGGLE' if IS_KAGGLE else 'NON-KAGGLE'} | DEVICE: {DEVICE} | torch={torch.__version__}")
print("=" * 118)

# -------------------------
# Configuration
# -------------------------
CFG = {
    "rounds": 15,

    # fixed clients (no RL participation / no RL client-count planning)
    "fixed_clients_ds1": 3,
    "fixed_clients_ds2": 3,

    # local training
    "local_epochs": 2,
    "lr_head": 1e-3,
    "lr_backbone": 2e-4,
    "weight_decay": 5e-4,
    "warmup_epochs": 1,
    "label_smoothing": 0.02,
    "focal_gamma": 1.35,
    "grad_clip": 1.0,
    "fedprox_mu": 0.01,
    "proto_lambda": 0.12,

    # image
    "img_size": 224 if torch.cuda.is_available() else 160,
    "batch_size": 16 if torch.cuda.is_available() else 8,
    "num_workers": 2 if torch.cuda.is_available() else 2,

    # global split
    "global_val_frac": 0.15,
    "test_frac": 0.15,

    # client split
    "client_val_frac": 0.12,
    "client_tune_frac": 0.12,
    "min_per_class_per_client": 5,

    # non-iid
    "dirichlet_alpha": 0.35,

    # preprocessing / augmentation
    "use_preprocessing": True,
    "use_augmentation": True,

    # transfer learning
    "freeze_backbone_rounds": 2,
    "unfreeze_last_blocks": 2,

    # RL-UCB preset selection
    "ucb_c": 1.35,
    "theta_probe_topk": 3,
    "theta_probe_batches": 2,

    # final inference
    "final_use_tta": True,

    # reward
    "reward_f1_weight": 0.75,
    "reward_acc_weight": 0.25,

    # best-round selection: equal dataset importance
    "best_round_mass_ds1": 0.50,
    "best_round_mass_ds2": 0.50,
    "best_round_min_bonus": 0.15,

    # FedAvg tempering
    "fedavg_temper": 0.50,

    # misc / sanity
    "quick_hash_subset_per_split": 300,
    "preproc_val_sample_n": 400,

    # no plots
    "make_plots": False,
    "calibration_bins": 12,
}

OUTDIR = "/kaggle/working/outputs" if IS_KAGGLE else "/content/outputs"
os.makedirs(OUTDIR, exist_ok=True)
MODEL_PATH = os.path.join(OUTDIR, "ARCFNet_Ablation_RLUCB_PresetSelection_checkpoint.pth")
CSV_PATH = os.path.join(OUTDIR, "ARCFNet_Ablation_RLUCB_PresetSelection_outputs.csv")

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], device=DEVICE).view(1, 3, 1, 1)
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225], device=DEVICE).view(1, 3, 1, 1)

labels = ["glioma", "meningioma", "notumor", "pituitary"]
label2id = {l: i for i, l in enumerate(labels)}
id2label = {i: l for l, i in label2id.items()}
NUM_CLASSES = len(labels)

METHOD_INFO = {
    "acronym": "ARCF-Net",
    "full_form": "Adaptive RACE-FELCM with CRAF Fusion Network",
    "preprocessing_full_form": "Robust Adaptive Context-Enhanced Fuzzy Edge Local Contrast Mapping",
    "fusion_full_form": "Cross-Residual Adaptive Fusion",
    "backbone_full_form": "Residual Network-50",
    "ablation_name": "RL-UCB Preset Selection, Fixed Clients, Full Participation",
}

# ============================================================
# Preset banks
# ============================================================
THETA_FULLFORMS = {
    "gamma": "Robust power exponent",
    "alpha": "Local contrast enhancement weight",
    "beta": "Contrast saturation sharpness",
    "tau": "Robust clipping threshold",
    "k": "Local context blur kernel",
    "edge_gain": "Gradient structure gain",
    "blend": "Structure-preserving residual blend",
}

PRESET_BANK_DS1 = [
    ("race_balanced",  (1.00, 0.24, 4.6, 2.4, 5, 0.10, 0.78)),
    ("race_sharp",     (1.04, 0.30, 5.2, 2.5, 5, 0.12, 0.82)),
    ("race_texture",   (0.92, 0.34, 5.8, 2.3, 7, 0.10, 0.80)),
    ("race_robust",    (1.08, 0.22, 4.5, 2.8, 5, 0.11, 0.76)),
    ("race_edge_plus", (1.02, 0.32, 6.0, 2.6, 3, 0.15, 0.84)),
    ("race_focus",     (1.10, 0.36, 6.4, 2.7, 3, 0.17, 0.86)),
]

PRESET_BANK_DS2 = [
    ("race_soft",      (0.95, 0.18, 3.8, 2.2, 3, 0.08, 0.72)),
    ("race_balanced",  (1.00, 0.24, 4.6, 2.4, 5, 0.10, 0.78)),
    ("race_robust",    (1.08, 0.22, 4.5, 2.8, 5, 0.11, 0.76)),
    ("race_smoothmix", (0.90, 0.20, 4.0, 2.1, 7, 0.07, 0.70)),
]

def get_preset_by_name(bank, target_name):
    for name, theta in bank:
        if name == target_name:
            return name, theta
    raise ValueError(f"Preset {target_name} not found.")

# ============================================================
# CSV collector
# ============================================================
ALL_ROWS = []

def add_table_to_csv(df, table_name):
    if df is None or len(df) == 0:
        return
    df2 = df.copy()
    df2.insert(0, "table_name", table_name)
    for _, row in df2.iterrows():
        ALL_ROWS.append(row.to_dict())

def print_table(df, title, max_rows=12):
    print("\n" + "-" * 118)
    print(title)
    print("-" * 118)
    if df is None or len(df) == 0:
        print("[empty]")
    else:
        print(df.head(max_rows).to_string(index=False))
        if len(df) > max_rows:
            print(f"... showing first {max_rows} of {len(df)} rows")

def safe_float(x, default=np.nan):
    try:
        return float(x)
    except Exception:
        return default

def normalize_imagenet(x):
    return (x - IMAGENET_MEAN) / IMAGENET_STD

def score_metric(m):
    return (
        CFG["reward_f1_weight"] * safe_float(m.get("f1_macro"), 0.0)
        + CFG["reward_acc_weight"] * safe_float(m.get("acc"), 0.0)
    )

def frame_to_manifest(df_):
    return df_[["path", "label", "source", "filename", "y"]].to_dict(orient="list")

# ============================================================
# STEP 0: ACCESS DATASETS
# ============================================================
print("\n" + "=" * 118)
print("STEP 0: ACCESS DATASETS")
print("=" * 118)

REQ1 = {"512Glioma", "512Meningioma", "512Normal", "512Pituitary"}
REQ2 = {"glioma", "meningioma", "notumor", "pituitary"}

def norm_label(name: str):
    s = str(name).strip().lower()
    if "glioma" in s:
        return "glioma"
    if "meningioma" in s:
        return "meningioma"
    if "pituitary" in s:
        return "pituitary"
    if "normal" in s or "no_tumor" in s or "no tumor" in s or "notumor" in s:
        return "notumor"
    return None

def find_root_with_required_class_dirs(base_dir, required_set, prefer_raw=True):
    candidates = []
    for root, dirs, _ in os.walk(base_dir):
        if required_set.issubset(set(dirs)):
            candidates.append(root)
    if not candidates:
        return None

    def score(p):
        pl = p.lower()
        sc = 0
        if prefer_raw:
            if "raw data" in pl:
                sc += 7
            if os.path.basename(p).lower() == "raw":
                sc += 7
            if "/raw/" in pl or "\\raw\\" in pl:
                sc += 3
            if "augmented" in pl:
                sc -= 20
        sc -= 0.0001 * len(p)
        return sc

    return max(candidates, key=score)

def search_kaggle_input_for_root(required_set, prefer_raw=True):
    if not os.path.exists("/kaggle/input"):
        return None
    return find_root_with_required_class_dirs("/kaggle/input", required_set, prefer_raw=prefer_raw)

def try_kagglehub_download(slug, required_set, prefer_raw=True):
    if kagglehub is None:
        return None, None
    try:
        base = kagglehub.dataset_download(slug)
        root = find_root_with_required_class_dirs(base, required_set, prefer_raw=prefer_raw)
        return base, root
    except Exception:
        return None, None

DS1_BASE = None
DS2_BASE = None

DS1_ROOT = search_kaggle_input_for_root(REQ1, prefer_raw=True)
DS2_ROOT = search_kaggle_input_for_root(REQ2, prefer_raw=False)

if DS1_ROOT is None:
    DS1_BASE, DS1_ROOT = try_kagglehub_download(
        "orvile/pmram-bangladeshi-brain-cancer-mri-dataset",
        REQ1,
        prefer_raw=True,
    )

if DS2_ROOT is None:
    DS2_BASE, DS2_ROOT = try_kagglehub_download(
        "yassinebazgour/preprocessed-brain-mri-scans-for-tumors-detection",
        REQ2,
        prefer_raw=False,
    )

if DS1_ROOT is None:
    raise RuntimeError(
        "Could not locate DS1 under /kaggle/input. Add "
        "'orvile/pmram-bangladeshi-brain-cancer-mri-dataset'."
    )
if DS2_ROOT is None:
    raise RuntimeError(
        "Could not locate DS2 under /kaggle/input. Add "
        "'yassinebazgour/preprocessed-brain-mri-scans-for-tumors-detection'."
    )

print(f"Dataset-1 RAW root detected:\n  {DS1_ROOT}")
print(f"Dataset-2 root detected:\n  {DS2_ROOT}")

# ============================================================
# STEP 1: BUILD MANIFESTS
# ============================================================
print("\n" + "=" * 118)
print("STEP 1: BUILD DATA MANIFESTS")
print("=" * 118)

def list_images_under_class_root(class_root, class_dir_name):
    class_dir = os.path.join(class_root, class_dir_name)
    out = []
    for r, _, files in os.walk(class_dir):
        for fn in files:
            if fn.lower().endswith(IMG_EXTS):
                out.append(os.path.join(r, fn))
    return out

def build_df_from_root(ds_root, class_dirs, source_name):
    rows = []
    for c in class_dirs:
        lab = norm_label(c)
        imgs = list_images_under_class_root(ds_root, c)
        print(f"{source_name}: {c} -> {lab} | {len(imgs)} images")
        for p in imgs:
            rows.append({"path": p, "label": lab, "source": source_name})
    dfm = pd.DataFrame(rows).dropna().reset_index(drop=True)
    dfm["path"] = dfm["path"].astype(str)
    dfm["label"] = dfm["label"].astype(str)
    dfm["source"] = dfm["source"].astype(str)
    dfm = dfm.drop_duplicates(subset=["path"]).reset_index(drop=True)
    dfm["filename"] = dfm["path"].apply(os.path.basename)
    return dfm

df1 = build_df_from_root(DS1_ROOT, ["512Glioma", "512Meningioma", "512Normal", "512Pituitary"], "ds1_raw")
df2 = build_df_from_root(DS2_ROOT, ["glioma", "meningioma", "notumor", "pituitary"], "ds2")

def enforce_labels(df_):
    df_ = df_.copy()
    df_["label"] = df_["label"].astype(str).str.strip().str.lower()
    df_ = df_[df_["label"].isin(set(labels))].reset_index(drop=True)
    df_["y"] = df_["label"].map(label2id).astype(int)
    return df_

df1 = enforce_labels(df1)
df2 = enforce_labels(df2)

print("\nDataset-1 images:", len(df1))
print(df1["label"].value_counts().reindex(labels, fill_value=0).to_string())
print("\nDataset-2 images:", len(df2))
print(df2["label"].value_counts().reindex(labels, fill_value=0).to_string())

# ============================================================
# STEP 2: TRAIN / VAL / TEST SPLIT
# ============================================================
print("\n" + "=" * 118)
print("STEP 2: TRAIN / VAL / TEST SPLIT")
print("=" * 118)

def split_dataset(df_):
    train_df, temp_df = train_test_split(
        df_,
        test_size=(CFG["global_val_frac"] + CFG["test_frac"]),
        stratify=df_["y"],
        random_state=SEED,
    )
    val_rel = CFG["global_val_frac"] / (CFG["global_val_frac"] + CFG["test_frac"])
    val_df, test_df = train_test_split(
        temp_df,
        test_size=(1 - val_rel),
        stratify=temp_df["y"],
        random_state=SEED,
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)

train1, val1, test1 = split_dataset(df1)
train2, val2, test2 = split_dataset(df2)

print(f"DS1 TRAIN: {len(train1)} | VAL: {len(val1)} | TEST: {len(test1)}")
print(f"DS2 TRAIN: {len(train2)} | VAL: {len(val2)} | TEST: {len(test2)}")

# ============================================================
# STEP 2.5: SANITY / LEAKAGE CHECKS
# ============================================================
print("\n" + "=" * 118)
print("STEP 2.5: SANITY / LEAKAGE CHECKS")
print("=" * 118)

def split_overlap_checks(train_df, val_df, test_df):
    tr = set(train_df["path"].tolist())
    va = set(val_df["path"].tolist())
    te = set(test_df["path"].tolist())
    checks = {
        "path_overlap_train_val": len(tr.intersection(va)),
        "path_overlap_train_test": len(tr.intersection(te)),
        "path_overlap_val_test": len(va.intersection(te)),
        "unique_paths_train": len(tr),
        "unique_paths_val": len(va),
        "unique_paths_test": len(te),
    }
    trf = set(train_df["filename"].tolist())
    vaf = set(val_df["filename"].tolist())
    tef = set(test_df["filename"].tolist())
    checks.update({
        "filename_overlap_train_val": len(trf.intersection(vaf)),
        "filename_overlap_train_test": len(trf.intersection(tef)),
        "filename_overlap_val_test": len(vaf.intersection(tef)),
    })
    return checks

def md5_file(path, max_bytes=2_000_000):
    h = hashlib.md5()
    try:
        with open(path, "rb") as f:
            h.update(f.read(max_bytes))
        return h.hexdigest()
    except Exception:
        return None

def quick_hash_subset(frame, n=300):
    n = min(n, len(frame))
    if n <= 0:
        return set()
    idx = np.random.choice(len(frame), size=n, replace=False)
    hashes = []
    for i in idx:
        hv = md5_file(frame.iloc[i]["path"])
        if hv is not None:
            hashes.append(hv)
    return set(hashes)

def leakage_report(name, tr, va, te):
    over = split_overlap_checks(tr, va, te)
    leak_df = pd.DataFrame([over])

    n_hash = int(CFG["quick_hash_subset_per_split"])
    trh = quick_hash_subset(tr, n_hash)
    vah = quick_hash_subset(va, n_hash)
    teh = quick_hash_subset(te, n_hash)

    hash_over = {
        "subset_hash_train_val": len(trh.intersection(vah)),
        "subset_hash_train_test": len(trh.intersection(teh)),
        "subset_hash_val_test": len(vah.intersection(teh)),
        "subset_hash_n_train": len(trh),
        "subset_hash_n_val": len(vah),
        "subset_hash_n_test": len(teh),
    }
    leak_df = pd.concat([leak_df, pd.DataFrame([hash_over])], axis=1)
    print_table(leak_df, f"Leakage / Sanity Summary — {name}", max_rows=5)
    add_table_to_csv(leak_df, f"leakage_sanity_{name}")

leakage_report("ds1", train1, val1, test1)
leakage_report("ds2", train2, val2, test2)

# ============================================================
# STEP 3: RL-UCB BANDIT
# ============================================================
print("\n" + "=" * 118)
print("STEP 3: RL-UCB BANDIT")
print("=" * 118)

class UCBBandit:
    def __init__(self, n_arms, c=1.35):
        self.n_arms = int(n_arms)
        self.c = float(c)
        self.counts = np.zeros(self.n_arms, dtype=np.int64)
        self.values = np.zeros(self.n_arms, dtype=np.float64)

    def ucb(self, arm):
        if self.counts[arm] == 0:
            return float("inf")
        t = max(1, self.counts.sum())
        return float(self.values[arm] + self.c * math.sqrt(math.log(t + 1) / self.counts[arm]))

    def select(self):
        scores = [self.ucb(i) for i in range(self.n_arms)]
        return int(np.argmax(scores))

    def update(self, arm, reward):
        arm = int(arm)
        reward = float(reward)
        self.counts[arm] += 1
        n = self.counts[arm]
        self.values[arm] += (reward - self.values[arm]) / n

    def best_arm(self):
        if self.counts.sum() == 0:
            return 0
        return int(np.argmax(self.values))

    def state_dict(self):
        return {
            "n_arms": self.n_arms,
            "c": self.c,
            "counts": self.counts.copy(),
            "values": self.values.copy(),
        }

    def load_state_dict(self, sd):
        self.n_arms = int(sd["n_arms"])
        self.c = float(sd["c"])
        self.counts = sd["counts"].copy()
        self.values = sd["values"].copy()

# ============================================================
# STEP 4: FIXED NON-IID CLIENT PARTITIONING
# ============================================================
print("\n" + "=" * 118)
print("STEP 4: FIXED NON-IID CLIENT PARTITIONING")
print("=" * 118)

def make_clients_non_iid(train_df, n_clients, num_classes, min_per_class=5, alpha=0.35):
    y = train_df["y"].values
    idx_by_class = {c: np.where(y == c)[0].tolist() for c in range(num_classes)}
    for c in idx_by_class:
        random.shuffle(idx_by_class[c])

    client_indices = [[] for _ in range(n_clients)]

    for c in range(num_classes):
        idxs = idx_by_class[c]
        feasible = min(min_per_class, max(1, len(idxs) // n_clients))
        for k in range(n_clients):
            take = idxs[:feasible]
            idxs = idxs[feasible:]
            client_indices[k].extend(take)
        idx_by_class[c] = idxs

    for c in range(num_classes):
        idxs = idx_by_class[c]
        if len(idxs) == 0:
            continue
        props = np.random.dirichlet([alpha] * n_clients)
        counts = (props * len(idxs)).astype(int)
        diff = len(idxs) - counts.sum()
        counts[np.argmax(props)] += diff

        start = 0
        for k in range(n_clients):
            client_indices[k].extend(idxs[start:start + counts[k]])
            start += counts[k]

    for k in range(n_clients):
        random.shuffle(client_indices[k])

    return client_indices

def robust_client_splits(train_df, indices, val_frac, tune_frac):
    idxs = np.array(indices, dtype=int)
    if len(idxs) < 3:
        return idxs.tolist(), idxs.tolist(), idxs.tolist()

    yk = train_df.loc[idxs, "y"].values

    if len(np.unique(yk)) < 2 or len(idxs) < 20:
        n_tune = max(1, int(round(len(idxs) * tune_frac)))
        n_tune = min(n_tune, max(1, len(idxs) - 2))
        tune_idx = idxs[:n_tune]
        rem_idx = idxs[n_tune:]
    else:
        rem_idx, tune_idx = train_test_split(
            idxs,
            test_size=tune_frac,
            stratify=yk,
            random_state=SEED,
        )

    if len(rem_idx) < 2:
        return rem_idx.tolist(), tune_idx.tolist(), rem_idx.tolist()

    yk2 = train_df.loc[rem_idx, "y"].values
    if len(np.unique(yk2)) < 2 or len(rem_idx) < 12:
        n_val = max(1, int(round(len(rem_idx) * val_frac)))
        n_val = min(n_val, max(1, len(rem_idx) - 1))
        val_idx = rem_idx[:n_val]
        train_idx = rem_idx[n_val:]
    else:
        train_idx, val_idx = train_test_split(
            rem_idx,
            test_size=val_frac,
            stratify=yk2,
            random_state=SEED,
        )

    if len(train_idx) == 0:
        train_idx = val_idx[:]
    if len(val_idx) == 0:
        val_idx = train_idx[:1]
    return train_idx.tolist(), tune_idx.tolist(), val_idx.tolist()

n_clients_ds1 = int(CFG["fixed_clients_ds1"])
n_clients_ds2 = int(CFG["fixed_clients_ds2"])

client_indices_ds1 = make_clients_non_iid(
    train1,
    n_clients=n_clients_ds1,
    num_classes=NUM_CLASSES,
    min_per_class=CFG["min_per_class_per_client"],
    alpha=CFG["dirichlet_alpha"],
)
client_indices_ds2 = make_clients_non_iid(
    train2,
    n_clients=n_clients_ds2,
    num_classes=NUM_CLASSES,
    min_per_class=CFG["min_per_class_per_client"],
    alpha=CFG["dirichlet_alpha"],
)

print(f"Fixed clients for DS1: {n_clients_ds1}")
print(f"Fixed clients for DS2: {n_clients_ds2}")

# ============================================================
# STEP 5: DATA LOADERS
# ============================================================
print("\n" + "=" * 118)
print("STEP 5: DATA LOADERS")
print("=" * 118)

def load_rgb(path):
    try:
        return Image.open(path).convert("RGB")
    except Exception:
        return Image.new("RGB", (CFG["img_size"], CFG["img_size"]), (128, 128, 128))

EVAL_TFMS = transforms.Compose([
    transforms.Resize((CFG["img_size"], CFG["img_size"])),
    transforms.ToTensor(),
])

TRAIN_TFMS = transforms.Compose([
    transforms.Resize((CFG["img_size"], CFG["img_size"])),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=12),
    transforms.RandomAffine(degrees=0, translate=(0.04, 0.04), scale=(0.96, 1.04)),
    transforms.ColorJitter(brightness=0.10, contrast=0.10),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.6)),
    transforms.ToTensor(),
])

class MRIDataset(Dataset):
    def __init__(self, frame, indices=None, tfms=None, source_id=0):
        self.df = frame
        self.indices = indices if indices is not None else list(range(len(frame)))
        self.tfms = tfms
        self.source_id = int(source_id)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        j = self.indices[i]
        row = self.df.iloc[j]
        img = load_rgb(row["path"])
        x = self.tfms(img) if self.tfms is not None else transforms.ToTensor()(img)
        y = int(row["y"])
        return x, y, row["path"], self.source_id

def make_weighted_sampler(frame, indices, num_classes):
    if len(indices) == 0:
        return None
    ys = frame.loc[indices, "y"].values
    class_counts = np.bincount(ys, minlength=num_classes)
    class_weights = 1.0 / np.sqrt(np.clip(class_counts, 1, None))
    sample_weights = class_weights[ys]
    return WeightedRandomSampler(
        weights=torch.DoubleTensor(sample_weights),
        num_samples=len(sample_weights),
        replacement=True,
    )

def make_loader(frame, indices, bs, tfms, shuffle=False, sampler=None, source_id=0):
    ds = MRIDataset(frame, indices=indices, tfms=tfms, source_id=source_id)
    return DataLoader(
        ds,
        batch_size=bs,
        shuffle=(shuffle and sampler is None),
        sampler=sampler,
        num_workers=CFG["num_workers"],
        pin_memory=(DEVICE.type == "cuda"),
        drop_last=False,
        persistent_workers=(CFG["num_workers"] > 0),
    )

clients = []
gid = 0

for ds_name, df_src, client_indices, source_id in [
    ("ds1", train1, client_indices_ds1, 0),
    ("ds2", train2, client_indices_ds2, 1),
]:
    for local_id, idxs in enumerate(client_indices):
        tr_idx, tune_idx, val_idx = robust_client_splits(df_src, idxs, CFG["client_val_frac"], CFG["client_tune_frac"])
        sampler = make_weighted_sampler(df_src, tr_idx, NUM_CLASSES)

        train_loader = make_loader(
            df_src, tr_idx, CFG["batch_size"], TRAIN_TFMS,
            shuffle=(sampler is None), sampler=sampler, source_id=source_id
        )
        tune_loader = make_loader(
            df_src,
            tune_idx if len(tune_idx) else tr_idx[:max(1, min(len(tr_idx), CFG["batch_size"]))],
            CFG["batch_size"], EVAL_TFMS, shuffle=False, sampler=None, source_id=source_id
        )
        val_loader = make_loader(
            df_src,
            val_idx if len(val_idx) else tr_idx[:max(1, min(len(tr_idx), CFG["batch_size"]))],
            CFG["batch_size"], EVAL_TFMS, shuffle=False, sampler=None, source_id=source_id
        )
        proto_loader = make_loader(
            df_src, tr_idx, CFG["batch_size"], EVAL_TFMS,
            shuffle=False, sampler=None, source_id=source_id
        )

        counts = df_src.loc[tr_idx, "label"].value_counts().reindex(labels, fill_value=0)

        preset_bank = PRESET_BANK_DS1 if ds_name == "ds1" else PRESET_BANK_DS2

        clients.append({
            "gid": gid,
            "local_id": local_id,
            "dataset": ds_name,
            "source_id": source_id,
            "train_loader": train_loader,
            "tune_loader": tune_loader,
            "val_loader": val_loader,
            "proto_loader": proto_loader,
            "n_train": len(tr_idx),
            "n_tune": len(tune_idx),
            "n_val": len(val_idx),
            "train_indices": tr_idx,
            "tune_indices": tune_idx,
            "val_indices": val_idx,
            "class_counts": counts.to_dict(),
            "preset_bank": preset_bank,
            "theta_bandit": UCBBandit(len(preset_bank), c=CFG["ucb_c"]),
        })

        print(f"{ds_name} | client_{gid} | train={len(tr_idx)} | tune={len(tune_idx)} | val={len(val_idx)}")
        gid += 1

CLIENTS_TOTAL = len(clients)

dist_rows = []
for c in clients:
    row = {
        "client": f"client_{c['gid']}",
        "dataset": c["dataset"],
        "total_train": c["n_train"],
        "total_tune": c["n_tune"],
        "total_val": c["n_val"],
        "preset_bank_names": str([x[0] for x in c["preset_bank"]]),
    }
    row.update({lab: int(c["class_counts"].get(lab, 0)) for lab in labels})
    dist_rows.append(row)

dist_df = pd.DataFrame(dist_rows)
print_table(dist_df, "Client class distribution", max_rows=20)
add_table_to_csv(dist_df, "client_distribution")

val_loader_ds1 = make_loader(val1, list(range(len(val1))), CFG["batch_size"], EVAL_TFMS, shuffle=False, source_id=0)
val_loader_ds2 = make_loader(val2, list(range(len(val2))), CFG["batch_size"], EVAL_TFMS, shuffle=False, source_id=1)
test_loader_ds1 = make_loader(test1, list(range(len(test1))), CFG["batch_size"], EVAL_TFMS, shuffle=False, source_id=0)
test_loader_ds2 = make_loader(test2, list(range(len(test2))), CFG["batch_size"], EVAL_TFMS, shuffle=False, source_id=1)

print(f"Augmentation: {'ON' if CFG['use_augmentation'] else 'OFF'}")
print(f"Preprocessing: {'ON' if CFG['use_preprocessing'] else 'OFF'}")
print(f"Total clients: {CLIENTS_TOTAL}")

# ============================================================
# STEP 6: PREPROCESSING — RACE-FELCM
# ============================================================
print("\n" + "=" * 118)
print("STEP 6: PREPROCESSING — RACE-FELCM")
print("=" * 118)

class RACEFELCM(nn.Module):
    def __init__(self, gamma=1.0, alpha=0.24, beta=4.5, tau=2.4, blur_k=5, edge_gain=0.10, blend=0.78):
        super().__init__()
        self.gamma = float(gamma)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.tau = float(tau)
        self.blur_k = int(blur_k)
        self.edge_gain = float(edge_gain)
        self.blend = float(blend)

        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        lap = torch.tensor([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=torch.float32)

        self.register_buffer("sobel_x", sobel_x.view(1, 1, 3, 3))
        self.register_buffer("sobel_y", sobel_y.view(1, 1, 3, 3))
        self.register_buffer("lap", lap.view(1, 1, 3, 3))

    def forward(self, x):
        eps = 1e-6

        gray0 = x.mean(dim=1, keepdim=True)
        mu0 = F.avg_pool2d(F.pad(gray0, (1, 1, 1, 1), mode="reflect"), 3, stride=1)
        var0 = F.avg_pool2d(F.pad((gray0 - mu0).pow(2), (1, 1, 1, 1), mode="reflect"), 3, stride=1)
        tex = torch.sqrt(var0 + eps)
        tex_norm = tex / (tex.amax(dim=(2, 3), keepdim=True) + eps)
        flat_gate = 1.0 - tex_norm

        x_smooth = F.avg_pool2d(F.pad(x, (1, 1, 1, 1), mode="reflect"), 3, stride=1)
        x_dn = x * (1.0 - 0.18 * flat_gate) + x_smooth * (0.18 * flat_gate)

        mu = x_dn.mean(dim=(2, 3), keepdim=True)
        sd = x_dn.std(dim=(2, 3), keepdim=True).clamp_min(eps)
        z = ((x_dn - mu) / sd).clamp(-self.tau, self.tau)
        z = torch.sign(z) * torch.pow(z.abs().clamp_min(eps), self.gamma)

        gray = z.mean(dim=1, keepdim=True)
        k = self.blur_k if self.blur_k % 2 == 1 else self.blur_k + 1
        pad = k // 2

        local_mean = F.avg_pool2d(F.pad(gray, (pad, pad, pad, pad), mode="reflect"), k, stride=1)
        local_var = F.avg_pool2d(F.pad((gray - local_mean).pow(2), (pad, pad, pad, pad), mode="reflect"), k, stride=1)
        local_std = torch.sqrt(local_var + eps)
        local_norm = (gray - local_mean) / (local_std + eps)
        contrast_field = torch.tanh(self.beta * local_norm)

        gx = F.conv2d(F.pad(gray, (1, 1, 1, 1), mode="reflect"), self.sobel_x)
        gy = F.conv2d(F.pad(gray, (1, 1, 1, 1), mode="reflect"), self.sobel_y)
        edge = torch.sqrt(gx * gx + gy * gy + eps)
        edge = edge / (edge.amax(dim=(2, 3), keepdim=True) + eps)

        lap = F.conv2d(F.pad(gray, (1, 1, 1, 1), mode="reflect"), self.lap).abs()
        lap = lap / (lap.amax(dim=(2, 3), keepdim=True) + eps)
        coherence = torch.sigmoid(4.0 * (edge - 0.5 * lap))

        enhanced = z + self.alpha * contrast_field + self.edge_gain * edge * coherence

        mn = enhanced.amin(dim=(2, 3), keepdim=True)
        mx = enhanced.amax(dim=(2, 3), keepdim=True)
        enhanced = (enhanced - mn) / (mx - mn + eps)

        out = self.blend * enhanced + (1.0 - self.blend) * x
        return out.clamp(0, 1)

def theta_to_module(theta):
    return RACEFELCM(*theta)

def theta_to_vec(theta, batch_size):
    gamma, alpha, beta, tau, blur_k, edge_gain, blend = theta
    theta = torch.tensor(
        [gamma, alpha, beta / 8.0, tau / 4.0, blur_k / 7.0, edge_gain, blend],
        device=DEVICE,
        dtype=torch.float32
    )
    return theta.unsqueeze(0).repeat(batch_size, 1)

def theta_str(theta):
    if theta is None:
        return "None"
    g, a, b, t, k, eg, m = theta
    return f"(g={g:.2f}, a={a:.2f}, b={b:.2f}, t={t:.2f}, k={k}, eg={eg:.2f}, mix={m:.2f})"

# ============================================================
# STEP 7: MODEL — ResNet-50 + CRAF Fusion
# ============================================================
print("\n" + "=" * 118)
print("STEP 7: MODEL — ResNet-50 + CRAF Fusion")
print("=" * 118)

class CompactProjector(nn.Module):
    def __init__(self, in_dim, out_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            nn.Dropout(0.10),
        )

    def forward(self, x):
        return self.net(x)

class CRAFFusion(nn.Module):
    def __init__(self, in_dim, fuse_dim=256, cond_dim=64):
        super().__init__()
        self.proj_raw = CompactProjector(in_dim, fuse_dim)
        self.proj_enh = CompactProjector(in_dim, fuse_dim)
        self.proj_res = CompactProjector(in_dim, fuse_dim)

        self.router = nn.Sequential(
            nn.Linear(fuse_dim * 6 + cond_dim, fuse_dim),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(fuse_dim, 3),
        )

        self.residual_refine = nn.Sequential(
            nn.Linear(fuse_dim * 2 + cond_dim, fuse_dim),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(fuse_dim, fuse_dim),
        )

        self.out_norm = nn.LayerNorm(fuse_dim)

    def forward(self, f_raw, f_enh, f_res, cond):
        fr = self.proj_raw(f_raw)
        fe = self.proj_enh(f_enh)
        fs = self.proj_res(f_res)

        d_re = torch.abs(fr - fe)
        d_rs = torch.abs(fr - fs)
        d_es = torch.abs(fe - fs)

        router_in = torch.cat([fr, fe, fs, d_re, d_rs, d_es, cond], dim=1)
        gates = torch.softmax(self.router(router_in), dim=1)

        routed = gates[:, 0:1] * fr + gates[:, 1:2] * fe + gates[:, 2:3] * fs
        disagreement = (d_re + d_rs + d_es) / 3.0
        refine = self.residual_refine(torch.cat([routed, disagreement, cond], dim=1))
        fused = self.out_norm(routed + refine)

        return fused, gates

class ResNet50CRAF(nn.Module):
    def __init__(self, num_classes, cond_dim=64, fuse_dim=256, embed_dim=256, pretrained=True):
        super().__init__()
        try:
            weights = ResNet50_Weights.DEFAULT if pretrained else None
        except Exception:
            weights = None

        net = resnet50(weights=weights)
        self.backbone = nn.Sequential(
            net.conv1,
            net.bn1,
            net.relu,
            net.maxpool,
            net.layer1,
            net.layer2,
            net.layer3,
            net.layer4,
        )
        self.backbone_dim = net.fc.in_features
        self.pool = nn.AdaptiveAvgPool2d(1)

        self.theta_mlp = nn.Sequential(
            nn.Linear(7, cond_dim),
            nn.GELU(),
            nn.Linear(cond_dim, cond_dim),
        )
        self.source_emb = nn.Embedding(2, cond_dim)
        self.cond_norm = nn.LayerNorm(cond_dim)

        self.fusion = CRAFFusion(self.backbone_dim, fuse_dim=fuse_dim, cond_dim=cond_dim)

        self.neck = nn.Sequential(
            nn.LayerNorm(fuse_dim),
            nn.Linear(fuse_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(0.20),
        )
        self.classifier = nn.Linear(embed_dim, num_classes)
        self.embed_dim = embed_dim

    def _encode(self, x):
        f = self.backbone(x)
        f = self.pool(f).flatten(1)
        return f

    def forward(self, x_raw_n, x_enh_n, x_res_n, theta_vec, source_id, return_extra=False):
        cond = self.theta_mlp(theta_vec) + self.source_emb(source_id)
        cond = self.cond_norm(cond)

        f_raw = self._encode(x_raw_n)
        f_enh = self._encode(x_enh_n)
        f_res = self._encode(x_res_n)

        fused, gates = self.fusion(f_raw, f_enh, f_res, cond)
        embed = self.neck(fused)
        logits = self.classifier(embed)

        if return_extra:
            return logits, embed, gates
        return logits

def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

def set_trainable_for_round(model, rnd):
    for p in model.backbone.parameters():
        p.requires_grad = False

    for n, p in model.named_parameters():
        if not n.startswith("backbone."):
            p.requires_grad = True

    if rnd > CFG["freeze_backbone_rounds"]:
        blocks = list(model.backbone.children())
        tail_blocks = blocks[-CFG["unfreeze_last_blocks"]:]
        for blk in tail_blocks:
            for p in blk.parameters():
                p.requires_grad = True

def freeze_backbone_bn_stats(model):
    for m in model.backbone.modules():
        if isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm)):
            m.eval()
            for p in m.parameters():
                p.requires_grad = False

def make_optimizer(model):
    head_params, bb_params = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if n.startswith("backbone."):
            bb_params.append(p)
        else:
            head_params.append(p)

    groups = []
    if head_params:
        groups.append({"params": head_params, "lr": CFG["lr_head"]})
    if bb_params:
        groups.append({"params": bb_params, "lr": CFG["lr_backbone"]})

    return torch.optim.AdamW(groups, weight_decay=CFG["weight_decay"])

def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
    def lr_lambda(step):
        if step < num_warmup_steps:
            return float(step) / float(max(1, num_warmup_steps))
        progress = float(step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

global_model = ResNet50CRAF(
    num_classes=NUM_CLASSES,
    cond_dim=64,
    fuse_dim=256,
    embed_dim=256,
    pretrained=True,
).to(DEVICE)

set_trainable_for_round(global_model, rnd=1)
total_params, trainable_params = count_params(global_model)

print("Backbone: ResNet-50 | pretrained_loaded=True")
print(f"Total params: {total_params:,}")
print(f"Trainable params: {trainable_params:,} ({(100.0 * trainable_params / total_params):.2f}%)")

# ============================================================
# STEP 8: LOSSES + PROTOTYPE SHARING + EVAL HELPERS
# ============================================================
print("\n" + "=" * 118)
print("STEP 8: LOSSES + PROTOTYPE SHARING + EVAL HELPERS")
print("=" * 118)

counts1 = train1["y"].value_counts().sort_index().reindex(range(NUM_CLASSES), fill_value=0).values
counts2 = train2["y"].value_counts().sort_index().reindex(range(NUM_CLASSES), fill_value=0).values
counts = counts1 + counts2
w = (counts.sum() / np.clip(counts, 1, None)).astype(np.float32)
w = w / max(1e-6, w.mean())
class_w = torch.tensor(w, device=DEVICE)

class ClassBalancedFocalLoss(nn.Module):
    def __init__(self, class_weights, gamma=1.5, label_smoothing=0.0):
        super().__init__()
        self.class_weights = class_weights
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, logits, target):
        ce = F.cross_entropy(
            logits,
            target,
            weight=self.class_weights,
            reduction="none",
            label_smoothing=self.label_smoothing,
        )
        pt = torch.exp(-ce)
        focal = (1.0 - pt).pow(self.gamma) * ce
        return focal.mean()

criterion = ClassBalancedFocalLoss(
    class_weights=class_w,
    gamma=CFG["focal_gamma"],
    label_smoothing=CFG["label_smoothing"],
)

scaler = torch.amp.GradScaler("cuda", enabled=(DEVICE.type == "cuda")) if DEVICE.type == "cuda" else None

def fedprox_term(local_model, global_model):
    loss = 0.0
    for p_local, p_global in zip(local_model.parameters(), global_model.parameters()):
        loss = loss + ((p_local - p_global.detach()) ** 2).sum()
    return loss

def prototype_alignment_loss(embed, y, global_prototypes):
    if global_prototypes is None:
        return torch.tensor(0.0, device=embed.device)

    proto = global_prototypes["proto"]
    mask = global_prototypes["mask"]
    valid = mask[y]
    if not valid.any():
        return torch.tensor(0.0, device=embed.device)

    emb_n = F.normalize(embed[valid], dim=1)
    ref_n = F.normalize(proto[y[valid]], dim=1)
    return (1.0 - (emb_n * ref_n).sum(dim=1)).mean()

@torch.no_grad()
def compute_prototype_payload(model, loader, preproc, theta):
    model.eval()
    freeze_backbone_bn_stats(model)
    preproc.eval()

    sums = torch.zeros(NUM_CLASSES, model.embed_dim, device=DEVICE)
    counts = torch.zeros(NUM_CLASSES, device=DEVICE)

    for x, y, _, source_id in loader:
        x = x.to(DEVICE, non_blocking=True)
        y = y.to(DEVICE, non_blocking=True)
        source_id = source_id.to(DEVICE, non_blocking=True)

        x_enh = preproc(x)
        x_res = torch.abs(x_enh - x)

        x_raw_n = normalize_imagenet(x)
        x_enh_n = normalize_imagenet(x_enh)
        x_res_n = normalize_imagenet(x_res)
        theta_vec = theta_to_vec(theta, x.size(0))

        _, embed, _ = model(x_raw_n, x_enh_n, x_res_n, theta_vec, source_id, return_extra=True)

        for c in range(NUM_CLASSES):
            m = (y == c)
            if m.any():
                sums[c] += embed[m].sum(dim=0)
                counts[c] += m.sum()

    return sums.detach().cpu(), counts.detach().cpu()

def aggregate_prototypes(payloads, embed_dim):
    if len(payloads) == 0:
        return None
    sum_all = torch.zeros(NUM_CLASSES, embed_dim)
    cnt_all = torch.zeros(NUM_CLASSES)
    for s, c in payloads:
        sum_all += s
        cnt_all += c
    proto = sum_all / cnt_all.unsqueeze(1).clamp_min(1.0)
    mask = cnt_all > 0
    return {
        "proto": proto.to(DEVICE),
        "mask": mask.to(DEVICE),
        "counts": cnt_all.to(DEVICE),
    }

def gate_entropy(g):
    eps = 1e-6
    p = g.clamp(eps, 1 - eps)
    return -(p * torch.log2(p)).sum(dim=1)

@torch.no_grad()
def _auc_metrics(y_true, p_pred, num_classes):
    out = {}
    try:
        out["auc_roc_macro_ovr"] = float(roc_auc_score(y_true, p_pred, multi_class="ovr", average="macro"))
    except Exception:
        out["auc_roc_macro_ovr"] = np.nan

    for c in range(num_classes):
        try:
            yc = (y_true == c).astype(int)
            if yc.sum() > 0 and yc.sum() < len(yc):
                out[f"auc_class_{c}"] = float(roc_auc_score(yc, p_pred[:, c]))
            else:
                out[f"auc_class_{c}"] = np.nan
        except Exception:
            out[f"auc_class_{c}"] = np.nan
    return out

@torch.no_grad()
def predict_probs_single_theta(model, x, source_id, preproc, theta, use_tta=False):
    model.eval()
    freeze_backbone_bn_stats(model)
    preproc.eval()

    probs_acc = None
    versions = [x]
    if use_tta:
        versions.append(torch.flip(x, dims=[3]))

    for xv in versions:
        x_enh = preproc(xv)
        x_res = torch.abs(x_enh - xv)

        x_raw_n = normalize_imagenet(xv)
        x_enh_n = normalize_imagenet(x_enh)
        x_res_n = normalize_imagenet(x_res)
        theta_vec = theta_to_vec(theta, xv.size(0))

        logits = model(x_raw_n, x_enh_n, x_res_n, theta_vec, source_id)
        probs = torch.softmax(logits, dim=1)
        probs_acc = probs if probs_acc is None else (probs_acc + probs)

    probs_acc = probs_acc / len(versions)
    return probs_acc

@torch.no_grad()
def evaluate_full(model, loader, preproc, theta, return_gates=False, use_tta=False):
    t0 = time.time()
    model.eval()
    freeze_backbone_bn_stats(model)
    preproc.eval()

    all_y, all_p, all_loss = [], [], []
    gate_stats = []

    for x, y, _, source_id in loader:
        x = x.to(DEVICE, non_blocking=True)
        y = y.to(DEVICE, non_blocking=True)
        source_id = source_id.to(DEVICE, non_blocking=True)

        x_enh = preproc(x)
        x_res = torch.abs(x_enh - x)

        x_raw_n = normalize_imagenet(x)
        x_enh_n = normalize_imagenet(x_enh)
        x_res_n = normalize_imagenet(x_res)
        theta_vec = theta_to_vec(theta, x.size(0))

        if return_gates and (not use_tta):
            logits, _, gates = model(x_raw_n, x_enh_n, x_res_n, theta_vec, source_id, return_extra=True)
            gate_stats.append(gates.detach().cpu())
            probs = torch.softmax(logits, dim=1)
        else:
            probs = predict_probs_single_theta(model, x, source_id, preproc, theta, use_tta=use_tta)
            logits = torch.log(probs.clamp_min(1e-8))

        loss = F.cross_entropy(logits, y)

        all_loss.append(float(loss.item()))
        all_y.append(y.detach().cpu().numpy())
        all_p.append(probs.detach().cpu().numpy())

    if len(all_y) == 0:
        return {"acc": np.nan}, np.array([]), np.array([])

    y_true = np.concatenate(all_y)
    p_pred = np.concatenate(all_p)
    y_hat = np.argmax(p_pred, axis=1)

    met = {
        "loss_ce": float(np.mean(all_loss)),
        "acc": float(accuracy_score(y_true, y_hat)),
        "precision_macro": float(precision_score(y_true, y_hat, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_hat, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_hat, average="macro", zero_division=0)),
        "precision_weighted": float(precision_score(y_true, y_hat, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(y_true, y_hat, average="weighted", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_hat, average="weighted", zero_division=0)),
        "log_loss": float(log_loss(y_true, p_pred, labels=list(range(NUM_CLASSES)))),
        "eval_time_s": float(time.time() - t0),
    }
    met.update(_auc_metrics(y_true, p_pred, NUM_CLASSES))

    if return_gates and len(gate_stats) > 0:
        g = torch.cat(gate_stats, dim=0)
        met["fusion_gate_mean_raw"] = float(g[:, 0].mean().item())
        met["fusion_gate_mean_enh"] = float(g[:, 1].mean().item())
        met["fusion_gate_mean_res"] = float(g[:, 2].mean().item())
        met["fusion_gate_entropy"] = float(gate_entropy(g).mean().item())

    return met, y_true, p_pred

def train_one_epoch(model, loader, optimizer, preproc, theta, global_model=None, global_prototypes=None, scheduler=None):
    model.train()
    freeze_backbone_bn_stats(model)
    preproc.eval()

    losses, ce_losses, proto_losses, correct, total = [], [], [], 0, 0

    for x, y, _, source_id in loader:
        x = x.to(DEVICE, non_blocking=True)
        y = y.to(DEVICE, non_blocking=True)
        source_id = source_id.to(DEVICE, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        amp_enabled = (DEVICE.type == "cuda")

        with torch.amp.autocast(device_type=DEVICE.type, enabled=amp_enabled):
            x_enh = preproc(x)
            x_res = torch.abs(x_enh - x)

            x_raw_n = normalize_imagenet(x)
            x_enh_n = normalize_imagenet(x_enh)
            x_res_n = normalize_imagenet(x_res)
            theta_vec = theta_to_vec(theta, x.size(0))

            logits, embed, _ = model(x_raw_n, x_enh_n, x_res_n, theta_vec, source_id, return_extra=True)

            ce = criterion(logits, y)
            proto = prototype_alignment_loss(embed, y, global_prototypes)
            loss = ce + CFG["proto_lambda"] * proto

            if global_model is not None and CFG["fedprox_mu"] > 0:
                loss = loss + 0.5 * CFG["fedprox_mu"] * fedprox_term(model, global_model)

        if scaler is not None and amp_enabled:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if CFG["grad_clip"] > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), CFG["grad_clip"])
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if CFG["grad_clip"] > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), CFG["grad_clip"])
            optimizer.step()

        if scheduler is not None:
            scheduler.step()

        losses.append(float(loss.item()))
        ce_losses.append(float(ce.item()))
        proto_losses.append(float(proto.item()))

        preds = logits.argmax(dim=1)
        correct += int((preds == y).sum().item())
        total += int(y.size(0))

    return {
        "loss": float(np.mean(losses)) if losses else np.nan,
        "ce_loss": float(np.mean(ce_losses)) if ce_losses else np.nan,
        "proto_loss": float(np.mean(proto_losses)) if proto_losses else np.nan,
        "acc": float(correct / max(1, total)),
    }

def build_tempered_fedavg_weights(clients_subset):
    sizes = np.array([c["n_train"] for c in clients_subset], dtype=np.float64)
    sizes = np.power(np.clip(sizes, 1.0, None), CFG["fedavg_temper"])
    sizes = sizes / max(1e-12, sizes.sum())
    return sizes.tolist()

def fedavg_update(global_model, local_models, weights):
    global_sd = global_model.state_dict()
    new_sd = {}

    for key in global_sd.keys():
        ref = global_sd[key]
        if not torch.is_floating_point(ref):
            new_sd[key] = local_models[0].state_dict()[key].detach().clone()
        else:
            acc = None
            for m, w in zip(local_models, weights):
                t = m.state_dict()[key].detach().float().cpu()
                acc = t * w if acc is None else acc + t * w
            new_sd[key] = acc.to(ref.dtype)

    global_model.load_state_dict(new_sd, strict=True)

def weighted_mean(rows, key, weight_key):
    vals, ws = [], []
    for r in rows:
        v = r.get(key, np.nan)
        w = r.get(weight_key, 0)
        if np.isfinite(v) and w > 0:
            vals.append(v)
            ws.append(w)
    if len(vals) == 0:
        return np.nan
    return float(np.average(vals, weights=ws))

def aggregate_by_dataset(round_local_rows, dataset_name):
    rows = [r for r in round_local_rows if r["dataset"] == dataset_name]
    return {
        "acc": weighted_mean(rows, "val_acc", "val_size"),
        "f1_macro": weighted_mean(rows, "val_f1_macro", "val_size"),
        "precision_macro": weighted_mean(rows, "val_precision_macro", "val_size"),
        "recall_macro": weighted_mean(rows, "val_recall_macro", "val_size"),
        "log_loss": weighted_mean(rows, "val_log_loss", "val_size"),
        "loss_ce": weighted_mean(rows, "val_loss_ce", "val_size"),
        "eval_time_s": weighted_mean(rows, "val_eval_time_s", "val_size"),
    }

# ============================================================
# STEP 9: TUNE-AWARE RL-UCB PRESET SELECTION
# ============================================================
print("\n" + "=" * 118)
print("STEP 9: TUNE-AWARE RL-UCB PRESET SELECTION")
print("=" * 118)

@torch.no_grad()
def probe_theta_on_tune_loader(model, tune_loader, preset_bank, arm_candidates, n_batches=2):
    model.eval()
    freeze_backbone_bn_stats(model)

    cached_batches = []
    try:
        it = iter(tune_loader)
        for _ in range(n_batches):
            cached_batches.append(next(it))
    except Exception:
        pass

    if len(cached_batches) == 0:
        return int(arm_candidates[0])

    best_arm = int(arm_candidates[0])
    best_score = -1.0

    for arm in arm_candidates:
        arm = int(arm)
        _, theta = preset_bank[arm]
        pre = theta_to_module(theta).to(DEVICE).eval()

        batch_scores = []
        for x, y, _, source_id in cached_batches:
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            source_id = source_id.to(DEVICE)

            probs = predict_probs_single_theta(model, x, source_id, pre, theta, use_tta=False)
            pred = probs.argmax(dim=1)
            batch_scores.append((pred == y).float().mean().item())

        sc = float(np.mean(batch_scores)) if len(batch_scores) else 0.0
        if sc > best_score:
            best_score = sc
            best_arm = arm

    return int(best_arm)

def select_theta_arm_with_probe(client, model):
    bandit = client["theta_bandit"]
    ucb_scores = [bandit.ucb(i) for i in range(bandit.n_arms)]
    topk = min(CFG["theta_probe_topk"], bandit.n_arms)
    candidates = list(np.argsort(ucb_scores)[::-1][:topk])
    return probe_theta_on_tune_loader(
        model,
        client["tune_loader"],
        client["preset_bank"],
        candidates,
        n_batches=CFG["theta_probe_batches"],
    )

# ============================================================
# STEP 10: TRUE FEDERATED TRAINING
# ============================================================
print("\n" + "=" * 118)
print("STEP 10: TRUE FEDERATED TRAINING")
print("=" * 118)

history_global = []
history_local = []

best_reward = -1.0
best_round_saved = None
best_model_state = None
best_global_prototypes = None
best_theta_bandit_states = None
global_prototypes = None

t_global_start = time.time()

print(f"Clients => DS1={n_clients_ds1}, DS2={n_clients_ds2}, TOTAL={CLIENTS_TOTAL}")
print(f"Rounds: {CFG['rounds']} | Local epochs: {CFG['local_epochs']}")
print(f"Augmentation ON: {CFG['use_augmentation']}")
print("Transfer backbone: ResNet-50")
print("Preprocessing: RACE-FELCM")
print("Fusion: CRAF")
print(f"FedProx μ={CFG['fedprox_mu']} | Proto λ={CFG['proto_lambda']}")
print(f"Tempered FedAvg exponent = {CFG['fedavg_temper']:.2f}")
print(f"Best-round masses => DS1={CFG['best_round_mass_ds1']:.2f}, DS2={CFG['best_round_mass_ds2']:.2f}, min-bonus={CFG['best_round_min_bonus']:.2f}")

for rnd in range(1, CFG["rounds"] + 1):
    round_t0 = time.time()
    selected_ids = list(range(len(clients)))  # full participation every round

    print("\n" + "=" * 118)
    print(f"ROUND {rnd}/{CFG['rounds']} | selected={selected_ids}")
    print("=" * 118)

    local_models = []
    proto_payloads = []
    round_local_rows = []
    selected_clients_meta = []

    for cid in selected_ids:
        client = clients[cid]

        theta_arm = select_theta_arm_with_probe(client, global_model)
        theta_name, theta = client["preset_bank"][theta_arm]
        preproc = theta_to_module(theta).to(DEVICE)

        local_model = ResNet50CRAF(
            num_classes=NUM_CLASSES,
            cond_dim=64,
            fuse_dim=256,
            embed_dim=256,
            pretrained=False,
        ).to(DEVICE)
        local_model.load_state_dict(global_model.state_dict(), strict=True)

        set_trainable_for_round(local_model, rnd)
        opt = make_optimizer(local_model)

        total_steps = max(1, len(client["train_loader"]) * CFG["local_epochs"])
        warmup_steps = max(1, len(client["train_loader"]) * CFG["warmup_epochs"])
        scheduler = get_cosine_schedule_with_warmup(opt, warmup_steps, total_steps)

        train_logs = []
        for _ in range(CFG["local_epochs"]):
            log_ep = train_one_epoch(
                local_model,
                client["train_loader"],
                opt,
                preproc,
                theta,
                global_model=global_model,
                global_prototypes=global_prototypes,
                scheduler=scheduler,
            )
            train_logs.append(log_ep)

        met_loc, _, _ = evaluate_full(local_model, client["val_loader"], preproc, theta, return_gates=True, use_tta=False)
        proto_payload = compute_prototype_payload(local_model, client["proto_loader"], preproc, theta)

        reward = score_metric(met_loc)
        client["theta_bandit"].update(theta_arm, reward)

        local_models.append(local_model)
        selected_clients_meta.append(client)
        proto_payloads.append(proto_payload)

        g, a, b, t, kk, eg, mix = theta
        row = {
            "round": rnd,
            "client": f"client_{cid}",
            "dataset": client["dataset"],
            "selected": 1,
            "theta_arm": int(theta_arm),
            "theta_name": theta_name,
            "theta_str": theta_str(theta),
            "gamma_power": g,
            "alpha_contrast_weight": a,
            "beta_contrast_sharpness": b,
            "tau_clip": t,
            "k_blur_kernel_size": kk,
            "edge_gain": eg,
            "blend_mix": mix,
            "train_loss": float(np.mean([x["loss"] for x in train_logs])),
            "train_ce_loss": float(np.mean([x["ce_loss"] for x in train_logs])),
            "train_proto_loss": float(np.mean([x["proto_loss"] for x in train_logs])),
            "train_acc": float(np.mean([x["acc"] for x in train_logs])),
            "train_time_s": np.nan,
            "val_size": client["n_val"],
            **{f"val_{k}": v for k, v in met_loc.items()},
            "reward": reward,
            "bandit_counts_total": int(client["theta_bandit"].counts.sum()),
            "bandit_value_selected": float(client["theta_bandit"].values[theta_arm]),
        }
        round_local_rows.append(row)

        auc_val = row.get("val_auc_roc_macro_ovr", np.nan)
        print(
            f"Client {cid} ({client['dataset']}) | "
            f"train_acc={row['train_acc']:.4f} | "
            f"val_acc={row['val_acc']:.4f} | "
            f"val_f1={row['val_f1_macro']:.4f} | "
            f"val_auc={auc_val:.4f} | "
            f"reward={reward:.4f} | "
            f"theta={theta_name} {theta_str(theta)}"
        )

    agg_weights = build_tempered_fedavg_weights(selected_clients_meta)
    fedavg_update(global_model, local_models, agg_weights)
    global_prototypes = aggregate_prototypes(proto_payloads, global_model.embed_dim)

    ds1_metrics = aggregate_by_dataset(round_local_rows, "ds1")
    ds2_metrics = aggregate_by_dataset(round_local_rows, "ds2")

    global_metrics = {
        "acc": weighted_mean(round_local_rows, "val_acc", "val_size"),
        "f1_macro": weighted_mean(round_local_rows, "val_f1_macro", "val_size"),
        "precision_macro": weighted_mean(round_local_rows, "val_precision_macro", "val_size"),
        "recall_macro": weighted_mean(round_local_rows, "val_recall_macro", "val_size"),
        "log_loss": weighted_mean(round_local_rows, "val_log_loss", "val_size"),
        "loss_ce": weighted_mean(round_local_rows, "val_loss_ce", "val_size"),
        "eval_time_s": weighted_mean(round_local_rows, "val_eval_time_s", "val_size"),
    }

    ds1_score = score_metric(ds1_metrics)
    ds2_score = score_metric(ds2_metrics)
    round_reward = (
        CFG["best_round_mass_ds1"] * ds1_score +
        CFG["best_round_mass_ds2"] * ds2_score +
        CFG["best_round_min_bonus"] * min(ds1_score, ds2_score)
    )

    history_local.extend(round_local_rows)
    history_global.append({
        "round": rnd,
        "round_time_s": float(time.time() - round_t0),
        "n_selected_clients": len(selected_ids),
        "active_fraction": 1.0,
        "global_reward": round_reward,
        "global_acc": global_metrics["acc"],
        "global_f1_macro": global_metrics["f1_macro"],
        "global_precision_macro": global_metrics["precision_macro"],
        "global_recall_macro": global_metrics["recall_macro"],
        "global_log_loss": global_metrics["log_loss"],
        "global_loss_ce": global_metrics["loss_ce"],
        "global_eval_time_s": global_metrics["eval_time_s"],
        "ds1_acc": ds1_metrics["acc"],
        "ds1_f1_macro": ds1_metrics["f1_macro"],
        "ds1_log_loss": ds1_metrics["log_loss"],
        "ds2_acc": ds2_metrics["acc"],
        "ds2_f1_macro": ds2_metrics["f1_macro"],
        "ds2_log_loss": ds2_metrics["log_loss"],
    })

    print(
        f"GLOBAL VAL (Round {rnd}) | "
        f"global_acc={global_metrics['acc']:.4f} | "
        f"global_f1={global_metrics['f1_macro']:.4f} | "
        f"ds1_acc={ds1_metrics['acc']:.4f} | "
        f"ds1_f1={ds1_metrics['f1_macro']:.4f} | "
        f"ds2_acc={ds2_metrics['acc']:.4f} | "
        f"ds2_f1={ds2_metrics['f1_macro']:.4f} | "
        f"reward={round_reward:.4f} | "
        f"round_time={history_global[-1]['round_time_s']:.1f}s"
    )

    if np.isfinite(round_reward) and round_reward > best_reward:
        best_reward = float(round_reward)
        best_round_saved = rnd
        best_model_state = {k: v.detach().cpu().clone() for k, v in global_model.state_dict().items()}
        best_global_prototypes = None if global_prototypes is None else {
            "proto": global_prototypes["proto"].detach().cpu().clone(),
            "mask": global_prototypes["mask"].detach().cpu().clone(),
            "counts": global_prototypes["counts"].detach().cpu().clone(),
        }
        best_theta_bandit_states = [copy.deepcopy(c["theta_bandit"].state_dict()) for c in clients]

if best_model_state is not None:
    global_model.load_state_dict({k: v.to(DEVICE) for k, v in best_model_state.items()})

if best_theta_bandit_states is not None:
    for c, sd in zip(clients, best_theta_bandit_states):
        c["theta_bandit"].load_state_dict(sd)

if best_global_prototypes is not None:
    global_prototypes = {
        "proto": best_global_prototypes["proto"].to(DEVICE),
        "mask": best_global_prototypes["mask"].to(DEVICE),
        "counts": best_global_prototypes["counts"].to(DEVICE),
    }

t_total = float(time.time() - t_global_start)
print("\n" + "=" * 118)
print(f"TRAINING COMPLETE | total_time={t_total:.1f}s | best_round={best_round_saved} | best_reward={best_reward:.4f}")
print("=" * 118)

glob_df = pd.DataFrame(history_global)
loc_df = pd.DataFrame(history_local)

print_table(glob_df, "GLOBAL per-round metrics", max_rows=20)
print_table(loc_df, "LOCAL per-client per-round metrics", max_rows=20)
add_table_to_csv(glob_df, "global_round_metrics_full")
add_table_to_csv(loc_df, "client_round_metrics_full")

# ============================================================
# STEP 11: FINAL EVALUATION
# ============================================================
print("\n" + "=" * 118)
print("STEP 11: FINAL EVALUATION")
print("=" * 118)

def best_theta_for_client(client):
    arm = client["theta_bandit"].best_arm()
    name, theta = client["preset_bank"][arm]
    return arm, name, theta, client["theta_bandit"].values[arm]

def unique_theta_candidates(csubset):
    out = []
    seen = set()
    for c in csubset:
        arm, name, theta, val = best_theta_for_client(c)
        key = tuple([round(x, 6) if isinstance(x, float) else x for x in theta])
        if key not in seen:
            out.append((name, theta, float(val), c["gid"]))
            seen.add(key)
    return out

@torch.no_grad()
def evaluate_with_single_theta(model, loader, theta, use_tta=False):
    pre = theta_to_module(theta).to(DEVICE)
    return evaluate_full(model, loader, pre, theta, return_gates=False, use_tta=use_tta)

def rank_rlucb_candidates(model, val_loader, candidates):
    ranked = []
    for name, theta, est, gid in candidates:
        met, _, _ = evaluate_with_single_theta(model, val_loader, theta, use_tta=CFG["final_use_tta"])
        ranked.append({
            "theta_name": name,
            "theta": theta,
            "score": score_metric(met),
            "val_acc": safe_float(met.get("acc")),
            "val_f1": safe_float(met.get("f1_macro")),
            "estimated_bandit_value": float(est),
            "from_client_gid": int(gid),
        })
    ranked = sorted(ranked, key=lambda x: x["score"], reverse=True)
    return ranked

def equal_merge_metrics(m1, m2):
    out = {}
    keys = sorted(set(m1.keys()).union(set(m2.keys())))
    for k in keys:
        a, b = m1.get(k, np.nan), m2.get(k, np.nan)
        if np.isfinite(a) and np.isfinite(b):
            out[k] = float(np.mean([a, b]))
        elif np.isfinite(a):
            out[k] = float(a)
        elif np.isfinite(b):
            out[k] = float(b)
        else:
            out[k] = np.nan
    return out

client_theta_rows = []
for c in clients:
    arm, name, theta, val = best_theta_for_client(c)
    client_theta_rows.append({
        "client": f"client_{c['gid']}",
        "dataset": c["dataset"],
        "best_theta_arm": int(arm),
        "best_theta_name": name,
        "best_theta_str": theta_str(theta),
        "estimated_value": float(val),
        "pulls": int(c["theta_bandit"].counts.sum()),
    })
client_theta_df = pd.DataFrame(client_theta_rows)
print_table(client_theta_df, "Best RL-UCB-selected preprocessing preset per client", max_rows=20)
add_table_to_csv(client_theta_df, "best_rlucb_selected_preprocessing_per_client")

ds1_clients = [c for c in clients if c["dataset"] == "ds1"]
ds2_clients = [c for c in clients if c["dataset"] == "ds2"]

cand_ds1 = unique_theta_candidates(ds1_clients)
cand_ds2 = unique_theta_candidates(ds2_clients)

ranked_ds1 = rank_rlucb_candidates(global_model, val_loader_ds1, cand_ds1)
ranked_ds2 = rank_rlucb_candidates(global_model, val_loader_ds2, cand_ds2)

choice_ds1 = ranked_ds1[0]
choice_ds2 = ranked_ds2[0]

ranked_ds1_df = pd.DataFrame([{k: v for k, v in r.items() if k != "theta"} for r in ranked_ds1])
ranked_ds2_df = pd.DataFrame([{k: v for k, v in r.items() if k != "theta"} for r in ranked_ds2])

print_table(ranked_ds1_df, "Validation ranking of RL-UCB candidates — DS1", max_rows=10)
print_table(ranked_ds2_df, "Validation ranking of RL-UCB candidates — DS2", max_rows=10)
add_table_to_csv(ranked_ds1_df, "validation_ranking_rlucb_candidates_ds1")
add_table_to_csv(ranked_ds2_df, "validation_ranking_rlucb_candidates_ds2")

val_ds1, _, _ = evaluate_with_single_theta(global_model, val_loader_ds1, choice_ds1["theta"], use_tta=CFG["final_use_tta"])
test_ds1, y_ds1, p_ds1 = evaluate_with_single_theta(global_model, test_loader_ds1, choice_ds1["theta"], use_tta=CFG["final_use_tta"])

val_ds2, _, _ = evaluate_with_single_theta(global_model, val_loader_ds2, choice_ds2["theta"], use_tta=CFG["final_use_tta"])
test_ds2, y_ds2, p_ds2 = evaluate_with_single_theta(global_model, test_loader_ds2, choice_ds2["theta"], use_tta=CFG["final_use_tta"])

val_global = equal_merge_metrics(val_ds1, val_ds2)
test_global = equal_merge_metrics(test_ds1, test_ds2)

choice_df = pd.DataFrame([
    {
        "dataset": "ds1",
        "strategy": "validation_selected_single_after_rlucb_training",
        "theta_names": str([choice_ds1["theta_name"]]),
        "score": choice_ds1["score"],
        "val_acc": choice_ds1["val_acc"],
        "val_f1": choice_ds1["val_f1"],
    },
    {
        "dataset": "ds2",
        "strategy": "validation_selected_single_after_rlucb_training",
        "theta_names": str([choice_ds2["theta_name"]]),
        "score": choice_ds2["score"],
        "val_acc": choice_ds2["val_acc"],
        "val_f1": choice_ds2["val_f1"],
    },
])
print_table(choice_df, "Final preset strategy after RL-UCB training", max_rows=5)
add_table_to_csv(choice_df, "final_theta_strategy_choice_rlucb")

# ============================================================
# STEP 12: EXTENDED METRICS + CALIBRATION + ERROR ANALYSIS
# ============================================================
print("\n" + "=" * 118)
print("STEP 12: EXTENDED METRICS + CALIBRATION + ERROR ANALYSIS")
print("=" * 118)

def top_label_calibration(y_true, p_pred, n_bins=12):
    conf = np.max(p_pred, axis=1)
    pred = np.argmax(p_pred, axis=1)
    correct = (pred == y_true).astype(np.float32)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(conf, bins) - 1
    bin_ids = np.clip(bin_ids, 0, n_bins - 1)

    rows = []
    ece = 0.0
    mce = 0.0
    N = len(y_true)

    for b in range(n_bins):
        m = (bin_ids == b)
        if m.sum() == 0:
            rows.append({
                "bin_id": b,
                "bin_left": bins[b],
                "bin_right": bins[b + 1],
                "bin_confidence": np.nan,
                "bin_accuracy": np.nan,
                "bin_gap": np.nan,
                "bin_count": 0,
            })
            continue

        bc = float(conf[m].mean())
        ba = float(correct[m].mean())
        gap = abs(ba - bc)
        ece += (m.sum() / N) * gap
        mce = max(mce, gap)
        rows.append({
            "bin_id": b,
            "bin_left": bins[b],
            "bin_right": bins[b + 1],
            "bin_confidence": bc,
            "bin_accuracy": ba,
            "bin_gap": gap,
            "bin_count": int(m.sum()),
        })

    return pd.DataFrame(rows), float(ece), float(mce)

def multiclass_brier_score(y_true, p_pred, num_classes):
    y_oh = np.eye(num_classes)[y_true]
    return float(np.mean(np.sum((p_pred - y_oh) ** 2, axis=1)))

def classwise_confusion_metrics(y_true, p_pred):
    y_hat = np.argmax(p_pred, axis=1)
    cm = confusion_matrix(y_true, y_hat, labels=list(range(NUM_CLASSES)))
    total = cm.sum()

    rows = []
    for c, lab in enumerate(labels):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        tn = total - tp - fp - fn

        ppv = tp / max(1, (tp + fp))
        npv = tn / max(1, (tn + fn))
        sensitivity = tp / max(1, (tp + fn))
        specificity = tn / max(1, (tn + fp))
        fpr = fp / max(1, (fp + tn))
        fnr = fn / max(1, (fn + tp))
        jacc = tp / max(1, (tp + fp + fn))
        support = tp + fn
        prevalence = support / max(1, total)
        bal = 0.5 * (sensitivity + specificity)

        rows.append({
            "class_id": c,
            "class_name": lab,
            "support": int(support),
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
            "tn": int(tn),
            "prevalence": float(prevalence),
            "ppv": float(ppv),
            "npv": float(npv),
            "recall": float(sensitivity),
            "specificity": float(specificity),
            "fpr": float(fpr),
            "fnr": float(fnr),
            "jaccard": float(jacc),
            "balanced_acc": float(bal),
        })

    return cm, pd.DataFrame(rows)

def top_confusion_pairs(y_true, p_pred):
    y_hat = np.argmax(p_pred, axis=1)
    cm = confusion_matrix(y_true, y_hat, labels=list(range(NUM_CLASSES)))
    rows = []
    for i, lab_i in enumerate(labels):
        for j, lab_j in enumerate(labels):
            if i == j:
                continue
            rows.append({
                "true_class": lab_i,
                "pred_class": lab_j,
                "count": int(cm[i, j]),
            })
    return pd.DataFrame(rows).sort_values("count", ascending=False).reset_index(drop=True)

def wrong_right_confidence_table(y_true, p_pred):
    pred = np.argmax(p_pred, axis=1)
    conf = np.max(p_pred, axis=1)
    ok = pred == y_true
    return pd.DataFrame({
        "confidence": conf,
        "correct": ok.astype(int),
        "status": np.where(ok, "correct", "wrong"),
        "true_label": [id2label[int(x)] for x in y_true],
        "pred_label": [id2label[int(x)] for x in pred],
    })

def compute_extended_bundle(y_true, p_pred, dataset_name):
    y_hat = np.argmax(p_pred, axis=1)
    cm, class_df = classwise_confusion_metrics(y_true, p_pred)
    cal_df, ece, mce = top_label_calibration(y_true, p_pred, n_bins=CFG["calibration_bins"])
    conf_df = top_confusion_pairs(y_true, p_pred)
    conf_table = wrong_right_confidence_table(y_true, p_pred)

    scalar = {
        "dataset": dataset_name,
        "acc": float(accuracy_score(y_true, y_hat)),
        "balanced_acc": float(balanced_accuracy_score(y_true, y_hat)),
        "precision_macro": float(precision_score(y_true, y_hat, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_hat, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_hat, average="macro", zero_division=0)),
        "precision_weighted": float(precision_score(y_true, y_hat, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(y_true, y_hat, average="weighted", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_hat, average="weighted", zero_division=0)),
        "log_loss": float(log_loss(y_true, p_pred, labels=list(range(NUM_CLASSES)))),
        "mcc": float(matthews_corrcoef(y_true, y_hat)),
        "kappa": float(cohen_kappa_score(y_true, y_hat)),
        "jaccard_macro": float(jaccard_score(y_true, y_hat, average="macro")),
        "ppv_macro": float(class_df["ppv"].mean()),
        "npv_macro": float(class_df["npv"].mean()),
        "specificity_macro": float(class_df["specificity"].mean()),
        "fpr_macro": float(class_df["fpr"].mean()),
        "fnr_macro": float(class_df["fnr"].mean()),
        "ece": float(ece),
        "mce": float(mce),
        "brier_multi": float(multiclass_brier_score(y_true, p_pred, NUM_CLASSES)),
    }
    scalar.update(_auc_metrics(y_true, p_pred, NUM_CLASSES))
    return scalar, class_df, cal_df, conf_df, conf_table, cm

ext_ds1, class_df1, cal_df1, conf_pairs1, conf_table1, cm1 = compute_extended_bundle(y_ds1, p_ds1, "ds1_test")
ext_ds2, class_df2, cal_df2, conf_pairs2, conf_table2, cm2 = compute_extended_bundle(y_ds2, p_ds2, "ds2_test")

ext_df = pd.DataFrame([ext_ds1, ext_ds2])

print_table(ext_df, "Extended TEST metrics (DS1 vs DS2)", max_rows=5)
print_table(class_df1, "Classwise metrics — DS1 TEST", max_rows=10)
print_table(class_df2, "Classwise metrics — DS2 TEST", max_rows=10)
print_table(conf_pairs1.head(10), "Top confusion pairs — DS1 TEST", max_rows=10)
print_table(conf_pairs2.head(10), "Top confusion pairs — DS2 TEST", max_rows=10)
print_table(cal_df1, "Calibration bins — DS1 TEST", max_rows=15)
print_table(cal_df2, "Calibration bins — DS2 TEST", max_rows=15)

add_table_to_csv(ext_df, "extended_test_metrics_ds1_ds2")
add_table_to_csv(class_df1, "classwise_metrics_ds1_test")
add_table_to_csv(class_df2, "classwise_metrics_ds2_test")
add_table_to_csv(cal_df1, "calibration_bins_ds1_test")
add_table_to_csv(cal_df2, "calibration_bins_ds2_test")
add_table_to_csv(conf_pairs1, "top_confusions_ds1_test")
add_table_to_csv(conf_pairs2, "top_confusions_ds2_test")
add_table_to_csv(conf_table1, "confidence_correctness_ds1_test")
add_table_to_csv(conf_table2, "confidence_correctness_ds2_test")

test_ds1.update(ext_ds1)
test_ds2.update(ext_ds2)

def compact_metrics(m):
    keep = [
        "acc", "balanced_acc", "precision_macro", "recall_macro", "f1_macro",
        "precision_weighted", "recall_weighted", "f1_weighted", "log_loss",
        "auc_roc_macro_ovr", "mcc", "kappa", "ppv_macro", "npv_macro",
        "specificity_macro", "ece", "mce", "brier_multi", "loss_ce", "eval_time_s"
    ]
    out = {}
    for k in keep:
        if k in m and np.isfinite(m[k]):
            out[k] = float(m[k])
    return out

paper_df = pd.DataFrame([
    {"setting": "ARCF-Net RL-UCB Preset Selection", "split": "VAL",  "dataset": "ds1",          **compact_metrics(val_ds1)},
    {"setting": "ARCF-Net RL-UCB Preset Selection", "split": "VAL",  "dataset": "ds2",          **compact_metrics(val_ds2)},
    {"setting": "ARCF-Net RL-UCB Preset Selection", "split": "VAL",  "dataset": "global_equal", **compact_metrics(val_global)},
    {"setting": "ARCF-Net RL-UCB Preset Selection", "split": "TEST", "dataset": "ds1",          **compact_metrics(test_ds1)},
    {"setting": "ARCF-Net RL-UCB Preset Selection", "split": "TEST", "dataset": "ds2",          **compact_metrics(test_ds2)},
    {"setting": "ARCF-Net RL-UCB Preset Selection", "split": "TEST", "dataset": "global_equal", **compact_metrics(test_global)},
])

print_table(paper_df, "VAL + TEST tables", max_rows=10)
add_table_to_csv(paper_df, "paper_ready_metrics")

print("\nSelection summary:")
print(f"- Best round: {best_round_saved} | best_reward={best_reward:.4f}")
print(f"- DS1 final strategy: validation_selected_single_after_rlucb_training | names={[choice_ds1['theta_name']]}")
print(f"- DS2 final strategy: validation_selected_single_after_rlucb_training | names={[choice_ds2['theta_name']]}")

# ============================================================
# STEP 13: PREPROCESSING VALIDATION
# ============================================================
print("\n" + "=" * 118)
print("STEP 13: PREPROCESSING VALIDATION")
print("=" * 118)

@torch.no_grad()
def entropy_per_image(x01):
    gray = x01.mean(dim=1)
    B = gray.shape[0]
    ent = []
    for i in range(B):
        g = (gray[i].detach().cpu().numpy() * 255).astype(np.uint8)
        hist = np.bincount(g.flatten(), minlength=256).astype(np.float32)
        p = hist / np.clip(hist.sum(), 1, None)
        p = p[p > 0]
        ent.append(float(-(p * np.log2(p)).sum()))
    return np.array(ent)

@torch.no_grad()
def edge_energy(x01, kernel):
    kernel = kernel.to(device=x01.device, dtype=x01.dtype)
    gray = x01.mean(dim=1, keepdim=True)
    lap = F.conv2d(F.pad(gray, (1, 1, 1, 1), mode="reflect"), kernel).abs()
    return lap.mean(dim=(1, 2, 3)).detach().cpu().numpy()

@torch.no_grad()
def contrast_proxy(x01):
    gray = x01.mean(dim=1)
    return gray.std(dim=(1, 2)).detach().cpu().numpy()

@torch.no_grad()
def run_preproc_validation(frame, preproc, sample_n=400):
    n = min(sample_n, len(frame))
    if n <= 0:
        return pd.DataFrame(), pd.DataFrame()

    idx = np.random.choice(len(frame), size=n, replace=False)
    ds = MRIDataset(frame, indices=idx.tolist(), tfms=EVAL_TFMS, source_id=0)

    xs = []
    for i in range(len(ds)):
        x, _, _, _ = ds[i]
        xs.append(x)
    x = torch.stack(xs).to(DEVICE)

    x_after = preproc(x).clamp(0, 1)
    lap_kernel = preproc.lap if hasattr(preproc, "lap") else RACEFELCM().to(DEVICE).lap

    ee_before = edge_energy(x, lap_kernel)
    ee_after  = edge_energy(x_after, lap_kernel)
    ent_before = entropy_per_image(x)
    ent_after  = entropy_per_image(x_after)
    con_before = contrast_proxy(x)
    con_after  = contrast_proxy(x_after)

    dfm = pd.DataFrame({
        "edge_energy_before": ee_before,
        "edge_energy_after": ee_after,
        "entropy_before": ent_before,
        "entropy_after": ent_after,
        "contrast_before": con_before,
        "contrast_after": con_after,
        "edge_gain_ratio": (ee_after / np.clip(ee_before, 1e-9, None)),
        "entropy_delta": (ent_after - ent_before),
        "contrast_delta": (con_after - con_before),
    })
    summary = dfm.agg(["mean", "std", "min", "max"]).T.reset_index().rename(columns={"index": "metric"})
    return dfm, summary

preproc_ds1 = theta_to_module(choice_ds1["theta"]).to(DEVICE)
preproc_ds2 = theta_to_module(choice_ds2["theta"]).to(DEVICE)

preproc_df1, preproc_summary_df1 = run_preproc_validation(val1, preproc_ds1, CFG["preproc_val_sample_n"])
preproc_df2, preproc_summary_df2 = run_preproc_validation(val2, preproc_ds2, CFG["preproc_val_sample_n"])

print_table(preproc_summary_df1, "Preprocessing validation summary (DS1 VAL sample)", max_rows=15)
print_table(preproc_summary_df2, "Preprocessing validation summary (DS2 VAL sample)", max_rows=15)
add_table_to_csv(preproc_summary_df1, "preprocessing_validation_summary_ds1")
add_table_to_csv(preproc_summary_df2, "preprocessing_validation_summary_ds2")

# ============================================================
# STEP 14: SAVE CHECKPOINT + CSV
# ============================================================
print("\n" + "=" * 118)
print("STEP 14: SAVING CHECKPOINT + CSV")
print("=" * 118)

client_process_manifest = []
for c in clients:
    df_src = train1 if c["dataset"] == "ds1" else train2
    client_process_manifest.append({
        "gid": c["gid"],
        "local_id": c["local_id"],
        "dataset": c["dataset"],
        "source_id": c["source_id"],
        "n_train": c["n_train"],
        "n_tune": c["n_tune"],
        "n_val": c["n_val"],
        "train_indices": c["train_indices"],
        "tune_indices": c["tune_indices"],
        "val_indices": c["val_indices"],
        "train_paths": df_src.loc[c["train_indices"], "path"].tolist(),
        "tune_paths": df_src.loc[c["tune_indices"], "path"].tolist(),
        "val_paths": df_src.loc[c["val_indices"], "path"].tolist(),
        "class_counts": c["class_counts"],
        "theta_bandit_state": c["theta_bandit"].state_dict(),
        "preset_bank": c["preset_bank"],
    })

all_df = pd.DataFrame(ALL_ROWS)

checkpoint = {
    "method_info": METHOD_INFO,

    "state_dict": {k: v.detach().cpu() for k, v in global_model.state_dict().items()},
    "best_model_state_dict": {k: v.detach().cpu() for k, v in global_model.state_dict().items()},

    "config": CFG,
    "seed": SEED,
    "device_used": str(DEVICE),
    "backbone_name": "resnet50",
    "fusion_name": "CRAF",
    "preprocessing_name": "RACE-FELCM",

    "dataset1_raw_root": DS1_ROOT,
    "dataset2_root": DS2_ROOT,
    "labels": labels,
    "label2id": label2id,
    "id2label": id2label,
    "num_classes": NUM_CLASSES,
    "theta_fullforms": THETA_FULLFORMS,

    "full_dataset_df1": frame_to_manifest(df1),
    "full_dataset_df2": frame_to_manifest(df2),

    "split_train1": frame_to_manifest(train1),
    "split_val1": frame_to_manifest(val1),
    "split_test1": frame_to_manifest(test1),
    "split_train2": frame_to_manifest(train2),
    "split_val2": frame_to_manifest(val2),
    "split_test2": frame_to_manifest(test2),

    "fixed_clients_ds1": n_clients_ds1,
    "fixed_clients_ds2": n_clients_ds2,
    "clients_total": CLIENTS_TOTAL,
    "client_indices_ds1": client_indices_ds1,
    "client_indices_ds2": client_indices_ds2,
    "client_process_manifest": client_process_manifest,

    "preset_bank_ds1": PRESET_BANK_DS1,
    "preset_bank_ds2": PRESET_BANK_DS2,

    "best_round_saved": best_round_saved,
    "best_reward": best_reward,
    "history_global": glob_df.to_dict(orient="list"),
    "history_local": loc_df.to_dict(orient="list"),
    "total_training_time_s": t_total,

    "best_theta_bandit_states": best_theta_bandit_states,
    "best_rlucb_selected_per_client": client_theta_df.to_dict(orient="list"),

    "final_choice_ds1": {
        "strategy": "validation_selected_single_after_rlucb_training",
        "theta_names": [choice_ds1["theta_name"]],
        "theta_list": [choice_ds1["theta"]],
        "score": choice_ds1["score"],
        "val_acc": choice_ds1["val_acc"],
        "val_f1": choice_ds1["val_f1"],
    },
    "final_choice_ds2": {
        "strategy": "validation_selected_single_after_rlucb_training",
        "theta_names": [choice_ds2["theta_name"]],
        "theta_list": [choice_ds2["theta"]],
        "score": choice_ds2["score"],
        "val_acc": choice_ds2["val_acc"],
        "val_f1": choice_ds2["val_f1"],
    },

    "best_global_prototypes": None if best_global_prototypes is None else {
        "proto": best_global_prototypes["proto"],
        "mask": best_global_prototypes["mask"],
        "counts": best_global_prototypes["counts"],
    },

    "final_val_ds1": val_ds1,
    "final_val_ds2": val_ds2,
    "final_val_global": val_global,
    "final_test_ds1": test_ds1,
    "final_test_ds2": test_ds2,
    "final_test_global_equal": test_global,
    "paper_ready_metrics": paper_df.to_dict(orient="list"),
    "extended_test_metrics_ds1_ds2": ext_df.to_dict(orient="list"),
    "classwise_metrics_ds1_test": class_df1.to_dict(orient="list"),
    "classwise_metrics_ds2_test": class_df2.to_dict(orient="list"),
    "calibration_bins_ds1_test": cal_df1.to_dict(orient="list"),
    "calibration_bins_ds2_test": cal_df2.to_dict(orient="list"),
    "top_confusions_ds1_test": conf_pairs1.to_dict(orient="list"),
    "top_confusions_ds2_test": conf_pairs2.to_dict(orient="list"),
    "confidence_correctness_ds1_test": conf_table1.to_dict(orient="list"),
    "confidence_correctness_ds2_test": conf_table2.to_dict(orient="list"),
    "client_distribution_table": dist_df.to_dict(orient="list"),
    "global_round_metrics_table": glob_df.to_dict(orient="list"),
    "local_round_metrics_table": loc_df.to_dict(orient="list"),
    "all_tables_long_format": all_df.to_dict(orient="list"),

    "ds1_test_y_true": y_ds1,
    "ds1_test_p_pred": p_ds1,
    "ds1_test_y_hat": np.argmax(p_ds1, axis=1),
    "ds2_test_y_true": y_ds2,
    "ds2_test_p_pred": p_ds2,
    "ds2_test_y_hat": np.argmax(p_ds2, axis=1),

    "preprocessing_validation_ds1_raw": preproc_df1.to_dict(orient="list") if len(preproc_df1) else {},
    "preprocessing_validation_ds2_raw": preproc_df2.to_dict(orient="list") if len(preproc_df2) else {},
    "preprocessing_validation_summary_ds1": preproc_summary_df1.to_dict(orient="list") if len(preproc_summary_df1) else {},
    "preprocessing_validation_summary_ds2": preproc_summary_df2.to_dict(orient="list") if len(preproc_summary_df2) else {},
}

torch.save(checkpoint, MODEL_PATH)
print(f"Saved checkpoint: {MODEL_PATH}")

all_df.to_csv(CSV_PATH, index=False)
print(f"Saved CSV: {CSV_PATH}")

print("\nDONE")
print(f"Method: {METHOD_INFO['acronym']} = {METHOD_INFO['full_form']}")
print(f"Ablation: {METHOD_INFO['ablation_name']}")
print(f"Backbone: {METHOD_INFO['backbone_full_form']}")
print(f"Best round: {best_round_saved}")
print(f"Fixed clients => DS1={n_clients_ds1}, DS2={n_clients_ds2}, TOTAL={CLIENTS_TOTAL}")
print(f"Rounds completed: {CFG['rounds']}")
print(f"Global TEST acc: {safe_float(test_global.get('acc')):.4f}")
print(f"Global TEST f1_macro: {safe_float(test_global.get('f1_macro')):.4f}")
print(f"DS1 TEST acc: {safe_float(test_ds1.get('acc')):.4f}")
print(f"DS2 TEST acc: {safe_float(test_ds2.get('acc')):.4f}")
print(f"DS1 final strategy: validation_selected_single_after_rlucb_training | names={[choice_ds1['theta_name']]}")
print(f"DS2 final strategy: validation_selected_single_after_rlucb_training | names={[choice_ds2['theta_name']]}")
```

    ======================================================================================================================
    ARCF-Net ABLATION 4: RL-UCB PRESET SELECTION + FIXED CLIENTS + FULL PARTICIPATION + NO PLOTS
    ======================================================================================================================
    ENV: KAGGLE | DEVICE: cuda | torch=2.10.0+cu128
    ======================================================================================================================
    
    ======================================================================================================================
    STEP 0: ACCESS DATASETS
    ======================================================================================================================
    Using Colab cache for faster access to the 'pmram-bangladeshi-brain-cancer-mri-dataset' dataset.
    Using Colab cache for faster access to the 'preprocessed-brain-mri-scans-for-tumors-detection' dataset.
    Dataset-1 RAW root detected:
      /kaggle/input/pmram-bangladeshi-brain-cancer-mri-dataset/PMRAM Bangladeshi Brain Cancer - MRI Dataset/PMRAM Bangladeshi Brain Cancer - MRI Dataset/Raw Data/Raw
    Dataset-2 root detected:
      /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset
    
    ======================================================================================================================
    STEP 1: BUILD DATA MANIFESTS
    ======================================================================================================================
    ds1_raw: 512Glioma -> glioma | 373 images
    ds1_raw: 512Meningioma -> meningioma | 363 images
    ds1_raw: 512Normal -> notumor | 396 images
    ds1_raw: 512Pituitary -> pituitary | 373 images
    ds2: glioma -> glioma | 1621 images
    ds2: meningioma -> meningioma | 1646 images
    ds2: notumor -> notumor | 2000 images
    ds2: pituitary -> pituitary | 1764 images
    
    Dataset-1 images: 1505
    label
    glioma        373
    meningioma    363
    notumor       396
    pituitary     373
    
    Dataset-2 images: 7031
    label
    glioma        1621
    meningioma    1646
    notumor       2000
    pituitary     1764
    
    ======================================================================================================================
    STEP 2: TRAIN / VAL / TEST SPLIT
    ======================================================================================================================
    DS1 TRAIN: 1053 | VAL: 226 | TEST: 226
    DS2 TRAIN: 4921 | VAL: 1055 | TEST: 1055
    
    ======================================================================================================================
    STEP 2.5: SANITY / LEAKAGE CHECKS
    ======================================================================================================================
    
    ----------------------------------------------------------------------------------------------------------------------
    Leakage / Sanity Summary — ds1
    ----------------------------------------------------------------------------------------------------------------------
     path_overlap_train_val  path_overlap_train_test  path_overlap_val_test  unique_paths_train  unique_paths_val  unique_paths_test  filename_overlap_train_val  filename_overlap_train_test  filename_overlap_val_test  subset_hash_train_val  subset_hash_train_test  subset_hash_val_test  subset_hash_n_train  subset_hash_n_val  subset_hash_n_test
                          0                        0                      0                1053               226                226                           0                            0                          0                      5                       5                     6                  298                222                 224
    
    ----------------------------------------------------------------------------------------------------------------------
    Leakage / Sanity Summary — ds2
    ----------------------------------------------------------------------------------------------------------------------
     path_overlap_train_val  path_overlap_train_test  path_overlap_val_test  unique_paths_train  unique_paths_val  unique_paths_test  filename_overlap_train_val  filename_overlap_train_test  filename_overlap_val_test  subset_hash_train_val  subset_hash_train_test  subset_hash_val_test  subset_hash_n_train  subset_hash_n_val  subset_hash_n_test
                          0                        0                      0                4921              1055               1055                           0                            0                          0                      0                       3                     4                  299                298                 299
    
    ======================================================================================================================
    STEP 3: RL-UCB BANDIT
    ======================================================================================================================
    
    ======================================================================================================================
    STEP 4: FIXED NON-IID CLIENT PARTITIONING
    ======================================================================================================================
    Fixed clients for DS1: 3
    Fixed clients for DS2: 3
    
    ======================================================================================================================
    STEP 5: DATA LOADERS
    ======================================================================================================================
    ds1 | client_0 | train=490 | tune=77 | val=67
    ds1 | client_1 | train=125 | tune=20 | val=18
    ds1 | client_2 | train=198 | tune=31 | val=27
    ds2 | client_3 | train=629 | tune=98 | val=86
    ds2 | client_4 | train=527 | tune=82 | val=72
    ds2 | client_5 | train=2653 | tune=412 | val=362
    
    ----------------------------------------------------------------------------------------------------------------------
    Client class distribution
    ----------------------------------------------------------------------------------------------------------------------
      client dataset  total_train  total_tune  total_val                                                                              preset_bank_names  glioma  meningioma  notumor  pituitary
    client_0     ds1          490          77         67 ['race_balanced', 'race_sharp', 'race_texture', 'race_robust', 'race_edge_plus', 'race_focus']     111          46      176        157
    client_1     ds1          125          20         18 ['race_balanced', 'race_sharp', 'race_texture', 'race_robust', 'race_edge_plus', 'race_focus']      75           4        8         38
    client_2     ds1          198          31         27 ['race_balanced', 'race_sharp', 'race_texture', 'race_robust', 'race_edge_plus', 'race_focus']      16         147       30          5
    client_3     ds2          629          98         86                                ['race_soft', 'race_balanced', 'race_robust', 'race_smoothmix']      12         197      416          4
    client_4     ds2          527          82         72                                ['race_soft', 'race_balanced', 'race_robust', 'race_smoothmix']     202           4      284         37
    client_5     ds2         2653         412        362                                ['race_soft', 'race_balanced', 'race_robust', 'race_smoothmix']     665         691      383        914
    Augmentation: ON
    Preprocessing: ON
    Total clients: 6
    
    ======================================================================================================================
    STEP 6: PREPROCESSING — RACE-FELCM
    ======================================================================================================================
    
    ======================================================================================================================
    STEP 7: MODEL — ResNet-50 + CRAF Fusion
    ======================================================================================================================
    Downloading: "https://download.pytorch.org/models/resnet50-11ad3fa6.pth" to /root/.cache/torch/hub/checkpoints/resnet50-11ad3fa6.pth


    100%|██████████| 97.8M/97.8M [00:00<00:00, 210MB/s]


    Backbone: ResNet-50 | pretrained_loaded=True
    Total params: 25,790,855
    Trainable params: 2,282,823 (8.85%)
    
    ======================================================================================================================
    STEP 8: LOSSES + PROTOTYPE SHARING + EVAL HELPERS
    ======================================================================================================================
    
    ======================================================================================================================
    STEP 9: TUNE-AWARE RL-UCB PRESET SELECTION
    ======================================================================================================================
    
    ======================================================================================================================
    STEP 10: TRUE FEDERATED TRAINING
    ======================================================================================================================
    Clients => DS1=3, DS2=3, TOTAL=6
    Rounds: 15 | Local epochs: 2
    Augmentation ON: True
    Transfer backbone: ResNet-50
    Preprocessing: RACE-FELCM
    Fusion: CRAF
    FedProx μ=0.01 | Proto λ=0.12
    Tempered FedAvg exponent = 0.50
    Best-round masses => DS1=0.50, DS2=0.50, min-bonus=0.15
    
    ======================================================================================================================
    ROUND 1/15 | selected=[0, 1, 2, 3, 4, 5]
    ======================================================================================================================
    Client 0 (ds1) | train_acc=0.7265 | val_acc=0.9104 | val_f1=0.8236 | val_auc=0.9837 | reward=0.8453 | theta=race_focus (g=1.10, a=0.36, b=6.40, t=2.70, k=3, eg=0.17, mix=0.86)
    Client 1 (ds1) | train_acc=0.6400 | val_acc=0.9444 | val_f1=0.6522 | val_auc=nan | reward=0.7252 | theta=race_focus (g=1.10, a=0.36, b=6.40, t=2.70, k=3, eg=0.17, mix=0.86)
    Client 2 (ds1) | train_acc=0.6212 | val_acc=0.7037 | val_f1=0.6111 | val_auc=0.8876 | reward=0.6343 | theta=race_focus (g=1.10, a=0.36, b=6.40, t=2.70, k=3, eg=0.17, mix=0.86)
    Client 3 (ds2) | train_acc=0.8116 | val_acc=0.9186 | val_f1=0.7059 | val_auc=0.8573 | reward=0.7591 | theta=race_smoothmix (g=0.90, a=0.20, b=4.00, t=2.10, k=7, eg=0.07, mix=0.70)
    Client 4 (ds2) | train_acc=0.8226 | val_acc=0.9722 | val_f1=0.7376 | val_auc=nan | reward=0.7962 | theta=race_smoothmix (g=0.90, a=0.20, b=4.00, t=2.10, k=7, eg=0.07, mix=0.70)
    Client 5 (ds2) | train_acc=0.7597 | val_acc=0.8260 | val_f1=0.8210 | val_auc=0.9658 | reward=0.8223 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    GLOBAL VAL (Round 1) | global_acc=0.8623 | global_f1=0.7823 | ds1_acc=0.8661 | ds1_f1=0.7448 | ds2_acc=0.8615 | ds2_f1=0.7904 | reward=0.9079 | round_time=179.0s
    
    ======================================================================================================================
    ROUND 2/15 | selected=[0, 1, 2, 3, 4, 5]
    ======================================================================================================================
    Client 0 (ds1) | train_acc=0.8633 | val_acc=0.8955 | val_f1=0.8381 | val_auc=0.9755 | reward=0.8525 | theta=race_edge_plus (g=1.02, a=0.32, b=6.00, t=2.60, k=3, eg=0.15, mix=0.84)
    Client 1 (ds1) | train_acc=0.8760 | val_acc=0.9444 | val_f1=0.7381 | val_auc=nan | reward=0.7897 | theta=race_edge_plus (g=1.02, a=0.32, b=6.00, t=2.60, k=3, eg=0.15, mix=0.84)
    Client 2 (ds1) | train_acc=0.8131 | val_acc=0.8519 | val_f1=0.6439 | val_auc=0.9282 | reward=0.6959 | theta=race_robust (g=1.08, a=0.22, b=4.50, t=2.80, k=5, eg=0.11, mix=0.76)
    Client 3 (ds2) | train_acc=0.8943 | val_acc=0.9186 | val_f1=0.6265 | val_auc=0.9405 | reward=0.6995 | theta=race_robust (g=1.08, a=0.22, b=4.50, t=2.80, k=5, eg=0.11, mix=0.76)
    Client 4 (ds2) | train_acc=0.9326 | val_acc=0.9861 | val_f1=0.7455 | val_auc=nan | reward=0.8056 | theta=race_robust (g=1.08, a=0.22, b=4.50, t=2.80, k=5, eg=0.11, mix=0.76)
    Client 5 (ds2) | train_acc=0.8345 | val_acc=0.8729 | val_f1=0.8736 | val_auc=0.9735 | reward=0.8734 | theta=race_smoothmix (g=0.90, a=0.20, b=4.00, t=2.10, k=7, eg=0.07, mix=0.70)
    GLOBAL VAL (Round 2) | global_acc=0.8956 | global_f1=0.8079 | ds1_acc=0.8929 | ds1_f1=0.7752 | ds2_acc=0.8962 | ds2_f1=0.8150 | reward=0.9407 | round_time=157.7s
    
    ======================================================================================================================
    ROUND 3/15 | selected=[0, 1, 2, 3, 4, 5]
    ======================================================================================================================
    Client 0 (ds1) | train_acc=0.8633 | val_acc=0.9254 | val_f1=0.8772 | val_auc=0.9900 | reward=0.8892 | theta=race_texture (g=0.92, a=0.34, b=5.80, t=2.30, k=7, eg=0.10, mix=0.80)
    Client 1 (ds1) | train_acc=0.8440 | val_acc=1.0000 | val_f1=1.0000 | val_auc=nan | reward=1.0000 | theta=race_texture (g=0.92, a=0.34, b=5.80, t=2.30, k=7, eg=0.10, mix=0.80)
    Client 2 (ds1) | train_acc=0.8258 | val_acc=0.9259 | val_f1=0.8208 | val_auc=0.9964 | reward=0.8471 | theta=race_texture (g=0.92, a=0.34, b=5.80, t=2.30, k=7, eg=0.10, mix=0.80)
    Client 3 (ds2) | train_acc=0.8514 | val_acc=0.9302 | val_f1=0.6335 | val_auc=0.9379 | reward=0.7077 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    Client 4 (ds2) | train_acc=0.9165 | val_acc=0.9722 | val_f1=0.7376 | val_auc=nan | reward=0.7962 | theta=race_soft (g=0.95, a=0.18, b=3.80, t=2.20, k=3, eg=0.08, mix=0.72)
    Client 5 (ds2) | train_acc=0.8673 | val_acc=0.9448 | val_f1=0.9435 | val_auc=0.9931 | reward=0.9438 | theta=race_robust (g=1.08, a=0.22, b=4.50, t=2.80, k=5, eg=0.11, mix=0.76)
    GLOBAL VAL (Round 3) | global_acc=0.9446 | global_f1=0.8672 | ds1_acc=0.9375 | ds1_f1=0.8833 | ds2_acc=0.9462 | ds2_f1=0.8637 | reward=1.0233 | round_time=213.2s
    
    ======================================================================================================================
    ROUND 4/15 | selected=[0, 1, 2, 3, 4, 5]
    ======================================================================================================================
    Client 0 (ds1) | train_acc=0.9296 | val_acc=0.9403 | val_f1=0.9010 | val_auc=0.9984 | reward=0.9109 | theta=race_sharp (g=1.04, a=0.30, b=5.20, t=2.50, k=5, eg=0.12, mix=0.82)
    Client 1 (ds1) | train_acc=0.8360 | val_acc=0.8889 | val_f1=0.9190 | val_auc=nan | reward=0.9115 | theta=race_sharp (g=1.04, a=0.30, b=5.20, t=2.50, k=5, eg=0.12, mix=0.82)
    Client 2 (ds1) | train_acc=0.9444 | val_acc=0.9259 | val_f1=0.8625 | val_auc=0.9421 | reward=0.8784 | theta=race_edge_plus (g=1.02, a=0.32, b=6.00, t=2.60, k=3, eg=0.15, mix=0.84)
    Client 3 (ds2) | train_acc=0.9420 | val_acc=0.9419 | val_f1=0.6021 | val_auc=0.9983 | reward=0.6870 | theta=race_soft (g=0.95, a=0.18, b=3.80, t=2.20, k=3, eg=0.08, mix=0.72)
    Client 4 (ds2) | train_acc=0.9554 | val_acc=0.9861 | val_f1=0.9571 | val_auc=nan | reward=0.9644 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    Client 5 (ds2) | train_acc=0.9103 | val_acc=0.9641 | val_f1=0.9640 | val_auc=0.9942 | reward=0.9640 | theta=race_soft (g=0.95, a=0.18, b=3.80, t=2.20, k=3, eg=0.08, mix=0.72)
    GLOBAL VAL (Round 4) | global_acc=0.9573 | global_f1=0.9017 | ds1_acc=0.9286 | ds1_f1=0.8946 | ds2_acc=0.9635 | ds2_f1=0.9032 | reward=1.0462 | round_time=176.8s
    
    ======================================================================================================================
    ROUND 5/15 | selected=[0, 1, 2, 3, 4, 5]
    ======================================================================================================================
    Client 0 (ds1) | train_acc=0.9541 | val_acc=0.9254 | val_f1=0.8966 | val_auc=0.9863 | reward=0.9038 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    Client 1 (ds1) | train_acc=0.9240 | val_acc=0.9444 | val_f1=0.7273 | val_auc=nan | reward=0.7816 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    Client 2 (ds1) | train_acc=0.9697 | val_acc=0.8889 | val_f1=0.6968 | val_auc=0.9611 | reward=0.7449 | theta=race_sharp (g=1.04, a=0.30, b=5.20, t=2.50, k=5, eg=0.12, mix=0.82)
    Client 3 (ds2) | train_acc=0.9483 | val_acc=0.9767 | val_f1=0.6599 | val_auc=0.9991 | reward=0.7391 | theta=race_smoothmix (g=0.90, a=0.20, b=4.00, t=2.10, k=7, eg=0.07, mix=0.70)
    Client 4 (ds2) | train_acc=0.9516 | val_acc=0.9861 | val_f1=0.9654 | val_auc=nan | reward=0.9706 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    Client 5 (ds2) | train_acc=0.9378 | val_acc=0.9669 | val_f1=0.9657 | val_auc=0.9950 | reward=0.9660 | theta=race_soft (g=0.95, a=0.18, b=3.80, t=2.20, k=3, eg=0.08, mix=0.72)
    GLOBAL VAL (Round 5) | global_acc=0.9620 | global_f1=0.8985 | ds1_acc=0.9196 | ds1_f1=0.8213 | ds2_acc=0.9712 | ds2_f1=0.9151 | reward=1.0144 | round_time=175.7s
    
    ======================================================================================================================
    ROUND 6/15 | selected=[0, 1, 2, 3, 4, 5]
    ======================================================================================================================
    Client 0 (ds1) | train_acc=0.9510 | val_acc=0.9403 | val_f1=0.9117 | val_auc=0.9947 | reward=0.9189 | theta=race_robust (g=1.08, a=0.22, b=4.50, t=2.80, k=5, eg=0.11, mix=0.76)
    Client 1 (ds1) | train_acc=0.9640 | val_acc=1.0000 | val_f1=1.0000 | val_auc=nan | reward=1.0000 | theta=race_robust (g=1.08, a=0.22, b=4.50, t=2.80, k=5, eg=0.11, mix=0.76)
    Client 2 (ds1) | train_acc=0.9722 | val_acc=0.9259 | val_f1=0.7381 | val_auc=0.9207 | reward=0.7851 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    Client 3 (ds2) | train_acc=0.9706 | val_acc=0.9767 | val_f1=0.7410 | val_auc=0.9793 | reward=0.7999 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    Client 4 (ds2) | train_acc=0.9820 | val_acc=0.9861 | val_f1=0.7455 | val_auc=nan | reward=0.8056 | theta=race_robust (g=1.08, a=0.22, b=4.50, t=2.80, k=5, eg=0.11, mix=0.76)
    Client 5 (ds2) | train_acc=0.9433 | val_acc=0.9475 | val_f1=0.9459 | val_auc=0.9973 | reward=0.9463 | theta=race_robust (g=1.08, a=0.22, b=4.50, t=2.80, k=5, eg=0.11, mix=0.76)
    GLOBAL VAL (Round 6) | global_acc=0.9557 | global_f1=0.8842 | ds1_acc=0.9464 | ds1_f1=0.8841 | ds2_acc=0.9577 | ds2_f1=0.8842 | reward=1.0361 | round_time=175.8s
    
    ======================================================================================================================
    ROUND 7/15 | selected=[0, 1, 2, 3, 4, 5]
    ======================================================================================================================
    Client 0 (ds1) | train_acc=0.9561 | val_acc=0.9552 | val_f1=0.9308 | val_auc=0.9950 | reward=0.9369 | theta=race_robust (g=1.08, a=0.22, b=4.50, t=2.80, k=5, eg=0.11, mix=0.76)
    Client 1 (ds1) | train_acc=0.9280 | val_acc=0.9444 | val_f1=0.8730 | val_auc=nan | reward=0.8909 | theta=race_texture (g=0.92, a=0.34, b=5.80, t=2.30, k=7, eg=0.10, mix=0.80)
    Client 2 (ds1) | train_acc=0.9697 | val_acc=0.9259 | val_f1=0.8690 | val_auc=0.9111 | reward=0.8833 | theta=race_edge_plus (g=1.02, a=0.32, b=6.00, t=2.60, k=3, eg=0.15, mix=0.84)
    Client 3 (ds2) | train_acc=0.9738 | val_acc=0.9651 | val_f1=0.8986 | val_auc=0.9960 | reward=0.9152 | theta=race_robust (g=1.08, a=0.22, b=4.50, t=2.80, k=5, eg=0.11, mix=0.76)
    Client 4 (ds2) | train_acc=0.9801 | val_acc=0.9861 | val_f1=0.9636 | val_auc=nan | reward=0.9693 | theta=race_smoothmix (g=0.90, a=0.20, b=4.00, t=2.10, k=7, eg=0.07, mix=0.70)
    Client 5 (ds2) | train_acc=0.9425 | val_acc=0.9641 | val_f1=0.9625 | val_auc=0.9953 | reward=0.9629 | theta=race_smoothmix (g=0.90, a=0.20, b=4.00, t=2.10, k=7, eg=0.07, mix=0.70)
    GLOBAL VAL (Round 7) | global_acc=0.9636 | global_f1=0.9440 | ds1_acc=0.9464 | ds1_f1=0.9066 | ds2_acc=0.9673 | ds2_f1=0.9520 | reward=1.0737 | round_time=175.5s
    
    ======================================================================================================================
    ROUND 8/15 | selected=[0, 1, 2, 3, 4, 5]
    ======================================================================================================================
    Client 0 (ds1) | train_acc=0.9786 | val_acc=0.9851 | val_f1=0.9858 | val_auc=1.0000 | reward=0.9856 | theta=race_sharp (g=1.04, a=0.30, b=5.20, t=2.50, k=5, eg=0.12, mix=0.82)
    Client 1 (ds1) | train_acc=0.9320 | val_acc=0.9444 | val_f1=0.8730 | val_auc=nan | reward=0.8909 | theta=race_robust (g=1.08, a=0.22, b=4.50, t=2.80, k=5, eg=0.11, mix=0.76)
    Client 2 (ds1) | train_acc=0.9697 | val_acc=0.9630 | val_f1=0.9106 | val_auc=0.9729 | reward=0.9237 | theta=race_texture (g=0.92, a=0.34, b=5.80, t=2.30, k=7, eg=0.10, mix=0.80)
    Client 3 (ds2) | train_acc=0.9865 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | reward=1.0000 | theta=race_soft (g=0.95, a=0.18, b=3.80, t=2.20, k=3, eg=0.08, mix=0.72)
    Client 4 (ds2) | train_acc=0.9791 | val_acc=0.9861 | val_f1=0.7455 | val_auc=nan | reward=0.8056 | theta=race_soft (g=0.95, a=0.18, b=3.80, t=2.20, k=3, eg=0.08, mix=0.72)
    Client 5 (ds2) | train_acc=0.9570 | val_acc=0.9641 | val_f1=0.9638 | val_auc=0.9935 | reward=0.9638 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    GLOBAL VAL (Round 8) | global_acc=0.9731 | global_f1=0.9413 | ds1_acc=0.9732 | ds1_f1=0.9496 | ds2_acc=0.9731 | ds2_f1=0.9395 | reward=1.0939 | round_time=173.9s
    
    ======================================================================================================================
    ROUND 9/15 | selected=[0, 1, 2, 3, 4, 5]
    ======================================================================================================================
    Client 0 (ds1) | train_acc=0.9612 | val_acc=0.9851 | val_f1=0.9750 | val_auc=0.9991 | reward=0.9775 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    Client 1 (ds1) | train_acc=0.9600 | val_acc=0.9444 | val_f1=0.9585 | val_auc=nan | reward=0.9550 | theta=race_sharp (g=1.04, a=0.30, b=5.20, t=2.50, k=5, eg=0.12, mix=0.82)
    Client 2 (ds1) | train_acc=0.9848 | val_acc=0.9630 | val_f1=0.9582 | val_auc=0.9964 | reward=0.9594 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    Client 3 (ds2) | train_acc=0.9817 | val_acc=0.9767 | val_f1=0.7389 | val_auc=1.0000 | reward=0.7983 | theta=race_soft (g=0.95, a=0.18, b=3.80, t=2.20, k=3, eg=0.08, mix=0.72)
    Client 4 (ds2) | train_acc=0.9801 | val_acc=1.0000 | val_f1=1.0000 | val_auc=nan | reward=1.0000 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    Client 5 (ds2) | train_acc=0.9563 | val_acc=0.9696 | val_f1=0.9685 | val_auc=0.9945 | reward=0.9688 | theta=race_soft (g=0.95, a=0.18, b=3.80, t=2.20, k=3, eg=0.08, mix=0.72)
    GLOBAL VAL (Round 9) | global_acc=0.9747 | global_f1=0.9408 | ds1_acc=0.9732 | ds1_f1=0.9683 | ds2_acc=0.9750 | ds2_f1=0.9349 | reward=1.0990 | round_time=174.0s
    
    ======================================================================================================================
    ROUND 10/15 | selected=[0, 1, 2, 3, 4, 5]
    ======================================================================================================================
    Client 0 (ds1) | train_acc=0.9704 | val_acc=0.9701 | val_f1=0.9616 | val_auc=0.9987 | reward=0.9637 | theta=race_texture (g=0.92, a=0.34, b=5.80, t=2.30, k=7, eg=0.10, mix=0.80)
    Client 1 (ds1) | train_acc=0.9560 | val_acc=1.0000 | val_f1=1.0000 | val_auc=nan | reward=1.0000 | theta=race_edge_plus (g=1.02, a=0.32, b=6.00, t=2.60, k=3, eg=0.15, mix=0.84)
    Client 2 (ds1) | train_acc=0.9848 | val_acc=0.9630 | val_f1=0.9436 | val_auc=1.0000 | reward=0.9484 | theta=race_sharp (g=1.04, a=0.30, b=5.20, t=2.50, k=5, eg=0.12, mix=0.82)
    Client 3 (ds2) | train_acc=0.9825 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | reward=1.0000 | theta=race_robust (g=1.08, a=0.22, b=4.50, t=2.80, k=5, eg=0.11, mix=0.76)
    Client 4 (ds2) | train_acc=0.9801 | val_acc=0.9861 | val_f1=0.7455 | val_auc=nan | reward=0.8056 | theta=race_smoothmix (g=0.90, a=0.20, b=4.00, t=2.10, k=7, eg=0.07, mix=0.70)
    Client 5 (ds2) | train_acc=0.9578 | val_acc=0.9779 | val_f1=0.9783 | val_auc=0.9989 | reward=0.9782 | theta=race_robust (g=1.08, a=0.22, b=4.50, t=2.80, k=5, eg=0.11, mix=0.76)
    GLOBAL VAL (Round 10) | global_acc=0.9810 | global_f1=0.9521 | ds1_acc=0.9732 | ds1_f1=0.9634 | ds2_acc=0.9827 | ds2_f1=0.9497 | reward=1.1056 | round_time=174.3s
    
    ======================================================================================================================
    ROUND 11/15 | selected=[0, 1, 2, 3, 4, 5]
    ======================================================================================================================
    Client 0 (ds1) | train_acc=0.9643 | val_acc=0.9701 | val_f1=0.9532 | val_auc=1.0000 | reward=0.9574 | theta=race_edge_plus (g=1.02, a=0.32, b=6.00, t=2.60, k=3, eg=0.15, mix=0.84)
    Client 1 (ds1) | train_acc=0.9680 | val_acc=0.9444 | val_f1=0.9552 | val_auc=nan | reward=0.9525 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    Client 2 (ds1) | train_acc=0.9924 | val_acc=0.9630 | val_f1=0.9106 | val_auc=1.0000 | reward=0.9237 | theta=race_robust (g=1.08, a=0.22, b=4.50, t=2.80, k=5, eg=0.11, mix=0.76)
    Client 3 (ds2) | train_acc=0.9833 | val_acc=0.9651 | val_f1=0.7324 | val_auc=1.0000 | reward=0.7906 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    Client 4 (ds2) | train_acc=0.9905 | val_acc=0.9861 | val_f1=0.7455 | val_auc=nan | reward=0.8056 | theta=race_robust (g=1.08, a=0.22, b=4.50, t=2.80, k=5, eg=0.11, mix=0.76)
    Client 5 (ds2) | train_acc=0.9676 | val_acc=0.9696 | val_f1=0.9696 | val_auc=0.9991 | reward=0.9696 | theta=race_smoothmix (g=0.90, a=0.20, b=4.00, t=2.10, k=7, eg=0.07, mix=0.70)
    GLOBAL VAL (Round 11) | global_acc=0.9699 | global_f1=0.9071 | ds1_acc=0.9643 | ds1_f1=0.9432 | ds2_acc=0.9712 | ds2_f1=0.8993 | reward=1.0705 | round_time=174.7s
    
    ======================================================================================================================
    ROUND 12/15 | selected=[0, 1, 2, 3, 4, 5]
    ======================================================================================================================
    Client 0 (ds1) | train_acc=0.9765 | val_acc=0.9701 | val_f1=0.9608 | val_auc=0.9985 | reward=0.9631 | theta=race_focus (g=1.10, a=0.36, b=6.40, t=2.70, k=3, eg=0.17, mix=0.86)
    Client 1 (ds1) | train_acc=0.9720 | val_acc=1.0000 | val_f1=1.0000 | val_auc=nan | reward=1.0000 | theta=race_focus (g=1.10, a=0.36, b=6.40, t=2.70, k=3, eg=0.17, mix=0.86)
    Client 2 (ds1) | train_acc=0.9823 | val_acc=0.9259 | val_f1=0.8690 | val_auc=0.9832 | reward=0.8833 | theta=race_focus (g=1.10, a=0.36, b=6.40, t=2.70, k=3, eg=0.17, mix=0.86)
    Client 3 (ds2) | train_acc=0.9881 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | reward=1.0000 | theta=race_smoothmix (g=0.90, a=0.20, b=4.00, t=2.10, k=7, eg=0.07, mix=0.70)
    Client 4 (ds2) | train_acc=0.9867 | val_acc=1.0000 | val_f1=1.0000 | val_auc=nan | reward=1.0000 | theta=race_soft (g=0.95, a=0.18, b=3.80, t=2.20, k=3, eg=0.08, mix=0.72)
    Client 5 (ds2) | train_acc=0.9698 | val_acc=0.9779 | val_f1=0.9805 | val_auc=0.9993 | reward=0.9799 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    GLOBAL VAL (Round 12) | global_acc=0.9810 | global_f1=0.9791 | ds1_acc=0.9643 | ds1_f1=0.9450 | ds2_acc=0.9846 | ds2_f1=0.9864 | reward=1.1104 | round_time=176.6s
    
    ======================================================================================================================
    ROUND 13/15 | selected=[0, 1, 2, 3, 4, 5]
    ======================================================================================================================
    Client 0 (ds1) | train_acc=0.9776 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | reward=1.0000 | theta=race_sharp (g=1.04, a=0.30, b=5.20, t=2.50, k=5, eg=0.12, mix=0.82)
    Client 1 (ds1) | train_acc=0.9560 | val_acc=1.0000 | val_f1=1.0000 | val_auc=nan | reward=1.0000 | theta=race_texture (g=0.92, a=0.34, b=5.80, t=2.30, k=7, eg=0.10, mix=0.80)
    Client 2 (ds1) | train_acc=0.9848 | val_acc=0.9259 | val_f1=0.8690 | val_auc=0.9887 | reward=0.8833 | theta=race_texture (g=0.92, a=0.34, b=5.80, t=2.30, k=7, eg=0.10, mix=0.80)
    Client 3 (ds2) | train_acc=0.9905 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | reward=1.0000 | theta=race_robust (g=1.08, a=0.22, b=4.50, t=2.80, k=5, eg=0.11, mix=0.76)
    Client 4 (ds2) | train_acc=0.9886 | val_acc=1.0000 | val_f1=1.0000 | val_auc=nan | reward=1.0000 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    Client 5 (ds2) | train_acc=0.9702 | val_acc=0.9641 | val_f1=0.9657 | val_auc=0.9990 | reward=0.9653 | theta=race_soft (g=0.95, a=0.18, b=3.80, t=2.20, k=3, eg=0.08, mix=0.72)
    GLOBAL VAL (Round 13) | global_acc=0.9763 | global_f1=0.9748 | ds1_acc=0.9821 | ds1_f1=0.9684 | ds2_acc=0.9750 | ds2_f1=0.9761 | reward=1.1196 | round_time=179.4s
    
    ======================================================================================================================
    ROUND 14/15 | selected=[0, 1, 2, 3, 4, 5]
    ======================================================================================================================
    Client 0 (ds1) | train_acc=0.9786 | val_acc=0.9552 | val_f1=0.9464 | val_auc=0.9972 | reward=0.9486 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    Client 1 (ds1) | train_acc=0.9600 | val_acc=0.9444 | val_f1=0.7273 | val_auc=nan | reward=0.7816 | theta=race_robust (g=1.08, a=0.22, b=4.50, t=2.80, k=5, eg=0.11, mix=0.76)
    Client 2 (ds1) | train_acc=0.9899 | val_acc=0.9630 | val_f1=0.9436 | val_auc=1.0000 | reward=0.9484 | theta=race_edge_plus (g=1.02, a=0.32, b=6.00, t=2.60, k=3, eg=0.15, mix=0.84)
    Client 3 (ds2) | train_acc=0.9865 | val_acc=0.9884 | val_f1=0.9932 | val_auc=1.0000 | reward=0.9920 | theta=race_smoothmix (g=0.90, a=0.20, b=4.00, t=2.10, k=7, eg=0.07, mix=0.70)
    Client 4 (ds2) | train_acc=0.9915 | val_acc=0.9861 | val_f1=0.9636 | val_auc=nan | reward=0.9693 | theta=race_soft (g=0.95, a=0.18, b=3.80, t=2.20, k=3, eg=0.08, mix=0.72)
    Client 5 (ds2) | train_acc=0.9768 | val_acc=0.9779 | val_f1=0.9776 | val_auc=0.9993 | reward=0.9777 | theta=race_robust (g=1.08, a=0.22, b=4.50, t=2.80, k=5, eg=0.11, mix=0.76)
    GLOBAL VAL (Round 14) | global_acc=0.9763 | global_f1=0.9663 | ds1_acc=0.9554 | ds1_f1=0.9105 | ds2_acc=0.9808 | ds2_f1=0.9783 | reward=1.0886 | round_time=176.9s
    
    ======================================================================================================================
    ROUND 15/15 | selected=[0, 1, 2, 3, 4, 5]
    ======================================================================================================================
    Client 0 (ds1) | train_acc=0.9776 | val_acc=0.9552 | val_f1=0.9317 | val_auc=0.9916 | reward=0.9376 | theta=race_robust (g=1.08, a=0.22, b=4.50, t=2.80, k=5, eg=0.11, mix=0.76)
    Client 1 (ds1) | train_acc=0.9680 | val_acc=0.9444 | val_f1=0.9585 | val_auc=nan | reward=0.9550 | theta=race_sharp (g=1.04, a=0.30, b=5.20, t=2.50, k=5, eg=0.12, mix=0.82)
    Client 2 (ds1) | train_acc=0.9874 | val_acc=0.9259 | val_f1=0.8764 | val_auc=0.9982 | reward=0.8888 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    Client 3 (ds2) | train_acc=0.9944 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | reward=1.0000 | theta=race_soft (g=0.95, a=0.18, b=3.80, t=2.20, k=3, eg=0.08, mix=0.72)
    Client 4 (ds2) | train_acc=0.9953 | val_acc=1.0000 | val_f1=1.0000 | val_auc=nan | reward=1.0000 | theta=race_smoothmix (g=0.90, a=0.20, b=4.00, t=2.10, k=7, eg=0.07, mix=0.70)
    Client 5 (ds2) | train_acc=0.9712 | val_acc=0.9751 | val_f1=0.9754 | val_auc=0.9992 | reward=0.9753 | theta=race_smoothmix (g=0.90, a=0.20, b=4.00, t=2.10, k=7, eg=0.07, mix=0.70)
    GLOBAL VAL (Round 15) | global_acc=0.9763 | global_f1=0.9722 | ds1_acc=0.9464 | ds1_f1=0.9227 | ds2_acc=0.9827 | ds2_f1=0.9829 | reward=1.0950 | round_time=176.2s
    
    ======================================================================================================================
    TRAINING COMPLETE | total_time=2660.5s | best_round=13 | best_reward=1.1196
    ======================================================================================================================
    
    ----------------------------------------------------------------------------------------------------------------------
    GLOBAL per-round metrics
    ----------------------------------------------------------------------------------------------------------------------
     round  round_time_s  n_selected_clients  active_fraction  global_reward  global_acc  global_f1_macro  global_precision_macro  global_recall_macro  global_log_loss  global_loss_ce  global_eval_time_s  ds1_acc  ds1_f1_macro  ds1_log_loss  ds2_acc  ds2_f1_macro  ds2_log_loss
         1    179.047244                   6              1.0       0.907933    0.862342         0.782349                0.794299             0.784508         0.348026        0.342152            2.680001 0.866071      0.744802      0.411654 0.861538      0.790436      0.334321
         2    157.692854                   6              1.0       0.940651    0.895570         0.807942                0.811233             0.814499         0.305183        0.310685            2.355686 0.892857      0.775224      0.310638 0.896154      0.814989      0.304008
         3    213.164238                   6              1.0       1.023256    0.944620         0.867206                0.866363             0.879093         0.170790        0.170086            2.371932 0.937500      0.883331      0.211574 0.946154      0.863732      0.162006
         4    176.843779                   6              1.0       1.046161    0.957278         0.901679                0.902576             0.914565         0.124119        0.120032            2.340060 0.928571      0.894642      0.179464 0.963462      0.903194      0.112198
         5    175.688340                   6              1.0       1.014366    0.962025         0.898477                0.899284             0.907352         0.141693        0.149677            2.342014 0.919643      0.821259      0.262822 0.971154      0.915109      0.115604
         6    175.778476                   6              1.0       1.036081    0.955696         0.884212                0.888394             0.881313         0.144905        0.142201            2.337411 0.946429      0.884070      0.187315 0.957692      0.884243      0.135771
         7    175.539768                   6              1.0       1.073693    0.963608         0.943995                0.938871             0.959549         0.136559        0.143506            2.333230 0.946429      0.906597      0.206643 0.967308      0.952050      0.121463
         8    173.886006                   6              1.0       1.093879    0.973101         0.941305                0.944067             0.942581         0.102269        0.112363            2.362002 0.973214      0.949552      0.102228 0.973077      0.939528      0.102277
         9    174.033121                   6              1.0       1.098960    0.974684         0.940825                0.940995             0.941656         0.106375        0.102859            2.322028 0.973214      0.968266      0.152372 0.975000      0.934915      0.096468
        10    174.329325                   6              1.0       1.105588    0.981013         0.952114                0.951494             0.954144         0.077827        0.076195            2.335211 0.973214      0.963413      0.097560 0.982692      0.949680      0.073577
        11    174.712938                   6              1.0       1.070482    0.969937         0.907114                0.908447             0.909581         0.100677        0.116395            2.341725 0.964286      0.943217      0.129249 0.971154      0.899338      0.094523
        12    176.577248                   6              1.0       1.110359    0.981013         0.979083                0.982518             0.978242         0.076673        0.074769            2.326649 0.964286      0.944972      0.144483 0.984615      0.986430      0.062068
        13    179.441495                   6              1.0       1.119638    0.976266         0.974780                0.979467             0.972316         0.083140        0.082905            2.345623 0.982143      0.968431      0.108790 0.975000      0.976147      0.077616
        14    176.874052                   6              1.0       1.088580    0.976266         0.966270                0.964610             0.970147         0.079962        0.082374            2.325395 0.955357      0.910533      0.113030 0.980769      0.978275      0.072839
        15    176.177664                   6              1.0       1.095009    0.976266         0.972197                0.974181             0.974561         0.087097        0.084433            2.355678 0.946429      0.922675      0.158739 0.982692      0.982863      0.071666
    
    ----------------------------------------------------------------------------------------------------------------------
    LOCAL per-client per-round metrics
    ----------------------------------------------------------------------------------------------------------------------
     round   client dataset  selected  theta_arm     theta_name                                                theta_str  gamma_power  alpha_contrast_weight  beta_contrast_sharpness  tau_clip  k_blur_kernel_size  edge_gain  blend_mix  train_loss  train_ce_loss  train_proto_loss  train_acc  train_time_s  val_size  val_loss_ce  val_acc  val_precision_macro  val_recall_macro  val_f1_macro  val_precision_weighted  val_recall_weighted  val_f1_weighted  val_log_loss  val_eval_time_s  val_auc_roc_macro_ovr  val_auc_class_0  val_auc_class_1  val_auc_class_2  val_auc_class_3  val_fusion_gate_mean_raw  val_fusion_gate_mean_enh  val_fusion_gate_mean_res  val_fusion_gate_entropy   reward  bandit_counts_total  bandit_value_selected
         1 client_0     ds1         1          5     race_focus (g=1.10, a=0.36, b=6.40, t=2.70, k=3, eg=0.17, mix=0.86)         1.10                   0.36                      6.4       2.7                   3       0.17       0.86    0.478585       0.414977          0.000000   0.726531           NaN        67     0.255571 0.910448             0.933532          0.800000      0.823562                0.916844             0.910448         0.896716      0.292966         1.397245               0.983748         0.979487         0.967213         0.990310         0.997980                  0.309523                  0.195793                  0.494684                 1.427033 0.845284                    1               0.845284
         1 client_1     ds1         1          5     race_focus (g=1.10, a=0.36, b=6.40, t=2.70, k=3, eg=0.17, mix=0.86)         1.10                   0.36                      6.4       2.7                   3       0.17       0.86    0.595695       0.571383          0.000000   0.640000           NaN        18     0.471904 0.944444             0.638889          0.666667      0.652174                0.893519             0.944444         0.917874      0.412128         0.893791                    NaN         1.000000              NaN         1.000000         1.000000                  0.009313                  0.601007                  0.389680                 0.958886 0.725242                    1               0.725242
         1 client_2     ds1         1          5     race_focus (g=1.10, a=0.36, b=6.40, t=2.70, k=3, eg=0.17, mix=0.86)         1.10                   0.36                      6.4       2.7                   3       0.17       0.86    0.580066       0.545921          0.000000   0.621212           NaN        27     0.703976 0.703704             0.593750          0.675000      0.611111                0.759259             0.703704         0.711934      0.705858         1.301343               0.887640         0.740000         0.821429         0.989130         1.000000                  0.484866                  0.085450                  0.429684                 1.323847 0.634259                    1               0.634259
         1 client_3     ds2         1          3 race_smoothmix (g=0.90, a=0.20, b=4.00, t=2.10, k=7, eg=0.07, mix=0.70)         0.90                   0.20                      4.0       2.1                   7       0.07       0.70    0.360878       0.299053          0.000000   0.811606           NaN        86     0.285990 0.918605             0.699242          0.713938      0.705905                0.911945             0.918605         0.914310      0.301020         1.061959               0.857317         0.529412         0.940992         0.958863         1.000000                  0.699475                  0.198565                  0.101960                 1.139780 0.759080                    1               0.759080
         1 client_4     ds2         1          3 race_smoothmix (g=0.90, a=0.20, b=4.00, t=2.10, k=7, eg=0.07, mix=0.70)         0.90                   0.20                      4.0       2.1                   7       0.07       0.70    0.365872       0.300157          0.000000   0.822581           NaN        72     0.067625 0.972222             0.743750          0.732143      0.737576                0.986458             0.972222         0.978740      0.073051         1.542242                    NaN         1.000000              NaN         1.000000         1.000000                  0.645152                  0.155418                  0.199430                 0.914343 0.796238                    1               0.796238
         1 client_5     ds2         1          1  race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)         1.00                   0.24                      4.6       2.4                   5       0.10       0.78    0.459611       0.355564          0.000000   0.759706           NaN       362     0.392682 0.825967             0.823852          0.822848      0.821031                0.818662             0.825967         0.819681      0.394198         3.719754               0.965829         0.960788         0.916759         0.993362         0.992405                  0.603115                  0.218503                  0.178382                 1.337239 0.822265                    1               0.822265
         2 client_0     ds1         1          4 race_edge_plus (g=1.02, a=0.32, b=6.00, t=2.60, k=3, eg=0.15, mix=0.84)         1.02                   0.32                      6.0       2.6                   3       0.15       0.84    0.319295       0.234637          0.327988   0.863265           NaN        67     0.293216 0.895522             0.853875          0.833333      0.838105                0.894879             0.895522         0.889880      0.317153         0.902452               0.975492         0.961538         0.945355         0.997093         0.997980                  0.893852                  0.021595                  0.084553                 0.498673 0.852459                    2               0.852459
         2 client_1     ds1         1          4 race_edge_plus (g=1.02, a=0.32, b=6.00, t=2.60, k=3, eg=0.15, mix=0.84)         1.02                   0.32                      6.0       2.6                   3       0.15       0.84    0.271948       0.217495          0.319985   0.876000           NaN        18     0.484939 0.944444             0.750000          0.727273      0.738095                1.000000             0.944444         0.970899      0.203237         0.274975                    NaN         1.000000              NaN         1.000000         1.000000                  0.434503                  0.279355                  0.286142                 1.539767 0.789683                    2               0.789683
         2 client_2     ds1         1          3    race_robust (g=1.08, a=0.22, b=4.50, t=2.80, k=5, eg=0.11, mix=0.76)         1.08                   0.22                      4.5       2.8                   5       0.11       0.76    0.324125       0.257898          0.329188   0.813131           NaN        27     0.401614 0.851852             0.708333          0.625000      0.643939                0.802469             0.851852         0.809203      0.366073         0.392421               0.928214         0.820000         0.892857         1.000000         1.000000                  0.773776                  0.114719                  0.111505                 0.966518 0.695918                    2               0.695918
         2 client_3     ds2         1          2    race_robust (g=1.08, a=0.22, b=4.50, t=2.80, k=5, eg=0.11, mix=0.76)         1.08                   0.22                      4.5       2.8                   5       0.11       0.76    0.238975       0.164370          0.276907   0.894277           NaN        86     0.260523 0.918605             0.581426          0.713938      0.626488                0.915153             0.918605         0.915352      0.261776         0.902562               0.940534         0.811765         0.966102         0.984271         1.000000                  0.972932                  0.022426                  0.004642                 0.195216 0.699517                    2               0.699517
         2 client_4     ds2         1          2    race_robust (g=1.08, a=0.22, b=4.50, t=2.80, k=5, eg=0.11, mix=0.76)         1.08                   0.22                      4.5       2.8                   5       0.11       0.76    0.163864       0.102468          0.241656   0.932638           NaN        72     0.093628 0.986111             0.750000          0.741071      0.745455                1.000000             0.986111         0.992929      0.094614         0.770128                    NaN         0.999188              NaN         1.000000         1.000000                  0.943324                  0.041468                  0.015207                 0.354854 0.805619                    2               0.805619
         2 client_5     ds2         1          3 race_smoothmix (g=0.90, a=0.20, b=4.00, t=2.10, k=7, eg=0.07, mix=0.70)         0.90                   0.20                      4.0       2.1                   7       0.07       0.70    0.332419       0.241497          0.277493   0.834527           NaN       362     0.353560 0.872928             0.880834          0.867979      0.873602                0.875849             0.872928         0.873728      0.355688         3.535124               0.973484         0.970318         0.933034         0.995409         0.995173                  0.722920                  0.160941                  0.116138                 1.113528 0.873433                    2               0.873433
         3 client_0     ds1         1          2   race_texture (g=0.92, a=0.34, b=5.80, t=2.30, k=7, eg=0.10, mix=0.80)         0.92                   0.34                      5.8       2.3                   7       0.10       0.80    0.358695       0.270261          0.276556   0.863265           NaN        67     0.167457 0.925373             0.875417          0.891667      0.877173                0.933433             0.925373         0.925644      0.188471         0.888061               0.989964         0.987179         0.972678         1.000000         1.000000                  0.643085                  0.196654                  0.160262                 0.839505 0.889223                    3               0.889223
         3 client_1     ds1         1          2   race_texture (g=0.92, a=0.34, b=5.80, t=2.30, k=7, eg=0.10, mix=0.80)         0.92                   0.34                      5.8       2.3                   7       0.10       0.80    0.390733       0.336019          0.283287   0.844000           NaN        18     0.226686 1.000000             1.000000          1.000000      1.000000                1.000000             1.000000         1.000000      0.116792         0.278521                    NaN         1.000000              NaN         1.000000         1.000000                  0.543388                  0.093365                  0.363247                 1.321135 1.000000                    3               1.000000
         3 client_2     ds1         1          2   race_texture (g=0.92, a=0.34, b=5.80, t=2.30, k=7, eg=0.10, mix=0.80)         0.92                   0.34                      5.8       2.3                   7       0.10       0.80    0.364482       0.290888          0.294983   0.825758           NaN        27     0.353147 0.925926             0.862500          0.862500      0.820833                0.944444             0.925926         0.925926      0.332094         0.384205               0.996429         1.000000         0.985714         1.000000         1.000000                  0.685874                  0.146619                  0.167507                 1.202846 0.847106                    3               0.847106
         3 client_3     ds2         1          1  race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)         1.00                   0.24                      4.6       2.4                   5       0.10       0.78    0.327611       0.236654          0.251325   0.851351           NaN        86     0.240610 0.930233             0.601871          0.708577      0.633536                0.935776             0.930233         0.930862      0.246660         0.963319               0.937932         0.764706         0.992467         0.994555         1.000000                  0.860364                  0.110219                  0.029417                 0.611450 0.707710                    3               0.707710
         3 client_4     ds2         1          0      race_soft (g=0.95, a=0.18, b=3.80, t=2.20, k=3, eg=0.08, mix=0.72)         0.95                   0.18                      3.8       2.2                   3       0.08       0.72    0.254302       0.173042          0.242780   0.916509           NaN        72     0.059186 0.972222             0.743750          0.732143      0.737576                0.986458             0.972222         0.978740      0.064594         0.786561                    NaN         1.000000              NaN         1.000000         1.000000                  0.830100                  0.141101                  0.028799                 0.726870 0.796238                    3               0.796238
         3 client_5     ds2         1          2    race_robust (g=1.08, a=0.22, b=4.50, t=2.80, k=5, eg=0.11, mix=0.76)         1.08                   0.22                      4.5       2.8                   5       0.11       0.76    0.345208       0.215450          0.229018   0.867320           NaN       362     0.159408 0.944751             0.945552          0.941729      0.943512                0.945002             0.944751         0.944750      0.161269         3.548885               0.993056         0.996391         0.992418         0.985236         0.998177                  0.868717                  0.094684                  0.036599                 0.662537 0.943822                    3               0.943822
         4 client_0     ds1         1          1     race_sharp (g=1.04, a=0.30, b=5.20, t=2.50, k=5, eg=0.12, mix=0.82)         1.04                   0.30                      5.2       2.5                   5       0.12       0.82    0.208628       0.144384          0.198015   0.929592           NaN        67     0.095438 0.940299             0.898471          0.908333      0.901038                0.942916             0.940299         0.940054      0.108590         0.817765               0.998355         0.996154         0.997268         1.000000         1.000000                  0.974406                  0.015892                  0.009702                 0.190217 0.910853                    4               0.910853
         4 client_1     ds1         1          1     race_sharp (g=1.04, a=0.30, b=5.20, t=2.50, k=5, eg=0.12, mix=0.82)         1.04                   0.30                      5.2       2.5                   5       0.12       0.82    0.444250       0.387778          0.284386   0.836000           NaN        18     0.292362 0.888889             0.916667          0.939394      0.919048                0.916667             0.888889         0.891270      0.295433         0.342298                    NaN         1.000000              NaN         1.000000         1.000000                  0.520217                  0.244686                  0.235097                 1.453833 0.911508                    4               0.911508
    ... showing first 20 of 90 rows
    
    ======================================================================================================================
    STEP 11: FINAL EVALUATION
    ======================================================================================================================
    
    ----------------------------------------------------------------------------------------------------------------------
    Best RL-UCB-selected preprocessing preset per client
    ----------------------------------------------------------------------------------------------------------------------
      client dataset  best_theta_arm best_theta_name                                           best_theta_str  estimated_value  pulls
    client_0     ds1               1      race_sharp (g=1.04, a=0.30, b=5.20, t=2.50, k=5, eg=0.12, mix=0.82)         0.965496     13
    client_1     ds1               2    race_texture (g=0.92, a=0.34, b=5.80, t=2.30, k=7, eg=0.10, mix=0.80)         0.963624     13
    client_2     ds1               2    race_texture (g=0.92, a=0.34, b=5.80, t=2.30, k=7, eg=0.10, mix=0.80)         0.884680     13
    client_3     ds2               2     race_robust (g=1.08, a=0.22, b=4.50, t=2.80, k=5, eg=0.11, mix=0.76)         0.903680     13
    client_4     ds2               1   race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)         0.983729     13
    client_5     ds2               0       race_soft (g=0.95, a=0.18, b=3.80, t=2.20, k=3, eg=0.08, mix=0.72)         0.966045     13
    
    ----------------------------------------------------------------------------------------------------------------------
    Validation ranking of RL-UCB candidates — DS1
    ----------------------------------------------------------------------------------------------------------------------
      theta_name    score  val_acc   val_f1  estimated_bandit_value  from_client_gid
      race_sharp 0.968657 0.969027 0.968533                0.965496                0
    race_texture 0.968657 0.969027 0.968533                0.963624                1
    
    ----------------------------------------------------------------------------------------------------------------------
    Validation ranking of RL-UCB candidates — DS2
    ----------------------------------------------------------------------------------------------------------------------
       theta_name    score  val_acc   val_f1  estimated_bandit_value  from_client_gid
      race_robust 0.978617 0.979147 0.978441                0.903680                3
    race_balanced 0.978617 0.979147 0.978441                0.983729                4
        race_soft 0.978617 0.979147 0.978441                0.966045                5
    
    ----------------------------------------------------------------------------------------------------------------------
    Final preset strategy after RL-UCB training
    ----------------------------------------------------------------------------------------------------------------------
    dataset                                        strategy     theta_names    score  val_acc   val_f1
        ds1 validation_selected_single_after_rlucb_training  ['race_sharp'] 0.968657 0.969027 0.968533
        ds2 validation_selected_single_after_rlucb_training ['race_robust'] 0.978617 0.979147 0.978441
    
    ======================================================================================================================
    STEP 12: EXTENDED METRICS + CALIBRATION + ERROR ANALYSIS
    ======================================================================================================================
    
    ----------------------------------------------------------------------------------------------------------------------
    Extended TEST metrics (DS1 vs DS2)
    ----------------------------------------------------------------------------------------------------------------------
     dataset      acc  balanced_acc  precision_macro  recall_macro  f1_macro  precision_weighted  recall_weighted  f1_weighted  log_loss      mcc    kappa  jaccard_macro  ppv_macro  npv_macro  specificity_macro  fpr_macro  fnr_macro      ece      mce  brier_multi  auc_roc_macro_ovr  auc_class_0  auc_class_1  auc_class_2  auc_class_3
    ds1_test 0.969027      0.968815         0.969917      0.968815  0.969179            0.969613         0.969027     0.969131  0.110866 0.958810 0.958685       0.940363   0.969917   0.989712           0.989635   0.010365   0.031185 0.011921 0.866455     0.047424           0.997645     0.992437     0.999575     0.998884     0.999685
    ds2_test 0.982938      0.982392         0.982265      0.982392  0.982319            0.982951         0.982938     0.982936  0.061077 0.977204 0.977198       0.965498   0.982265   0.994367           0.994382   0.005618   0.017608 0.016176 0.468147     0.025040           0.999165     0.999853     0.997870     1.000000     0.998937
    
    ----------------------------------------------------------------------------------------------------------------------
    Classwise metrics — DS1 TEST
    ----------------------------------------------------------------------------------------------------------------------
     class_id class_name  support  tp  fp  fn  tn  prevalence      ppv      npv   recall  specificity      fpr      fnr  jaccard  balanced_acc
            0     glioma       56  54   3   2 167    0.247788 0.947368 0.988166 0.964286     0.982353 0.017647 0.035714 0.915254      0.973319
            1 meningioma       55  53   1   2 170    0.243363 0.981481 0.988372 0.963636     0.994152 0.005848 0.036364 0.946429      0.978894
            2    notumor       59  58   3   1 164    0.261062 0.950820 0.993939 0.983051     0.982036 0.017964 0.016949 0.935484      0.982543
            3  pituitary       56  54   0   2 170    0.247788 1.000000 0.988372 0.964286     1.000000 0.000000 0.035714 0.964286      0.982143
    
    ----------------------------------------------------------------------------------------------------------------------
    Classwise metrics — DS2 TEST
    ----------------------------------------------------------------------------------------------------------------------
     class_id class_name  support  tp  fp  fn  tn  prevalence      ppv      npv   recall  specificity      fpr      fnr  jaccard  balanced_acc
            0     glioma      244 242   4   2 807    0.231280 0.983740 0.997528 0.991803     0.995068 0.004932 0.008197 0.975806      0.993436
            1 meningioma      247 239   6   8 802    0.234123 0.975510 0.990123 0.967611     0.992574 0.007426 0.032389 0.944664      0.980093
            2    notumor      300 299   0   1 755    0.284360 1.000000 0.998677 0.996667     1.000000 0.000000 0.003333 0.996667      0.998333
            3  pituitary      264 257   8   7 783    0.250237 0.969811 0.991139 0.973485     0.989886 0.010114 0.026515 0.944853      0.981686
    
    ----------------------------------------------------------------------------------------------------------------------
    Top confusion pairs — DS1 TEST
    ----------------------------------------------------------------------------------------------------------------------
    true_class pred_class  count
     pituitary     glioma      2
    meningioma    notumor      2
        glioma meningioma      1
        glioma    notumor      1
       notumor     glioma      1
        glioma  pituitary      0
    meningioma     glioma      0
    meningioma  pituitary      0
       notumor meningioma      0
       notumor  pituitary      0
    
    ----------------------------------------------------------------------------------------------------------------------
    Top confusion pairs — DS2 TEST
    ----------------------------------------------------------------------------------------------------------------------
    true_class pred_class  count
    meningioma  pituitary      7
     pituitary meningioma      4
     pituitary     glioma      3
        glioma meningioma      2
    meningioma     glioma      1
       notumor  pituitary      1
        glioma  pituitary      0
        glioma    notumor      0
       notumor meningioma      0
       notumor     glioma      0
    
    ----------------------------------------------------------------------------------------------------------------------
    Calibration bins — DS1 TEST
    ----------------------------------------------------------------------------------------------------------------------
     bin_id  bin_left  bin_right  bin_confidence  bin_accuracy  bin_gap  bin_count
          0  0.000000   0.083333             NaN           NaN      NaN          0
          1  0.083333   0.166667             NaN           NaN      NaN          0
          2  0.166667   0.250000             NaN           NaN      NaN          0
          3  0.250000   0.333333             NaN           NaN      NaN          0
          4  0.333333   0.416667             NaN           NaN      NaN          0
          5  0.416667   0.500000             NaN           NaN      NaN          0
          6  0.500000   0.583333        0.506884      0.000000 0.506884          1
          7  0.583333   0.666667        0.622016      0.666667 0.044650          3
          8  0.666667   0.750000             NaN           NaN      NaN          0
          9  0.750000   0.833333        0.809279      0.666667 0.142612          3
         10  0.833333   0.916667        0.866455      0.000000 0.866455          1
         11  0.916667   1.000000        0.982757      0.986239 0.003482        218
    
    ----------------------------------------------------------------------------------------------------------------------
    Calibration bins — DS2 TEST
    ----------------------------------------------------------------------------------------------------------------------
     bin_id  bin_left  bin_right  bin_confidence  bin_accuracy  bin_gap  bin_count
          0  0.000000   0.083333             NaN           NaN      NaN          0
          1  0.083333   0.166667             NaN           NaN      NaN          0
          2  0.166667   0.250000             NaN           NaN      NaN          0
          3  0.250000   0.333333             NaN           NaN      NaN          0
          4  0.333333   0.416667             NaN           NaN      NaN          0
          5  0.416667   0.500000        0.468147      0.000000 0.468147          1
          6  0.500000   0.583333        0.533401      0.500000 0.033401          8
          7  0.583333   0.666667        0.620055      0.500000 0.120055          4
          8  0.666667   0.750000        0.748265      1.000000 0.251735          1
          9  0.750000   0.833333        0.800564      0.692308 0.108256         13
         10  0.833333   0.916667        0.866615      0.850000 0.016615         20
         11  0.916667   1.000000        0.982283      0.996032 0.013749       1008
    
    ----------------------------------------------------------------------------------------------------------------------
    VAL + TEST tables
    ----------------------------------------------------------------------------------------------------------------------
                             setting split      dataset      acc  precision_macro  recall_macro  f1_macro  precision_weighted  recall_weighted  f1_weighted  log_loss  auc_roc_macro_ovr  loss_ce  eval_time_s  balanced_acc      mcc    kappa  ppv_macro  npv_macro  specificity_macro      ece      mce  brier_multi
    ARCF-Net RL-UCB Preset Selection   VAL          ds1 0.969027         0.971120      0.968056  0.968533            0.970252         0.969027     0.968613  0.097815           0.999195 0.092989     4.500246           NaN      NaN      NaN        NaN        NaN                NaN      NaN      NaN          NaN
    ARCF-Net RL-UCB Preset Selection   VAL          ds2 0.979147         0.978546      0.978351  0.978441            0.979173         0.979147     0.979153  0.081764           0.999075 0.081712    20.527280           NaN      NaN      NaN        NaN        NaN                NaN      NaN      NaN          NaN
    ARCF-Net RL-UCB Preset Selection   VAL global_equal 0.974087         0.974833      0.973203  0.973487            0.974713         0.974087     0.973883  0.089790           0.999135 0.087350    12.513763           NaN      NaN      NaN        NaN        NaN                NaN      NaN      NaN          NaN
    ARCF-Net RL-UCB Preset Selection  TEST          ds1 0.969027         0.969917      0.968815  0.969179            0.969613         0.969027     0.969131  0.110866           0.997645 0.105258     4.661927      0.968815 0.958810 0.958685   0.969917   0.989712           0.989635 0.011921 0.866455     0.047424
    ARCF-Net RL-UCB Preset Selection  TEST          ds2 0.982938         0.982265      0.982392  0.982319            0.982951         0.982938     0.982936  0.061077           0.999165 0.061037    20.623206      0.982392 0.977204 0.977198   0.982265   0.994367           0.994382 0.016176 0.468147     0.025040
    ARCF-Net RL-UCB Preset Selection  TEST global_equal 0.975982         0.976091      0.975603  0.975749            0.976282         0.975982     0.976033  0.085972           0.998405 0.083147    12.642567           NaN      NaN      NaN        NaN        NaN                NaN      NaN      NaN          NaN
    
    Selection summary:
    - Best round: 13 | best_reward=1.1196
    - DS1 final strategy: validation_selected_single_after_rlucb_training | names=['race_sharp']
    - DS2 final strategy: validation_selected_single_after_rlucb_training | names=['race_robust']
    
    ======================================================================================================================
    STEP 13: PREPROCESSING VALIDATION
    ======================================================================================================================
    
    ----------------------------------------------------------------------------------------------------------------------
    Preprocessing validation summary (DS1 VAL sample)
    ----------------------------------------------------------------------------------------------------------------------
                metric     mean      std       min      max
    edge_energy_before 0.041596 0.021720  0.014947 0.165471
     edge_energy_after 0.124185 0.029159  0.073520 0.255910
        entropy_before 5.820408 0.628255  3.547314 7.239670
         entropy_after 6.622424 0.577684  3.945864 7.578726
       contrast_before 0.187640 0.052597  0.101468 0.363759
        contrast_after 0.242398 0.021419  0.208139 0.339002
       edge_gain_ratio 3.265984 0.779384  1.304814 6.057923
         entropy_delta 0.802017 0.260156  0.155579 1.559621
        contrast_delta 0.054758 0.034391 -0.026487 0.121056
    
    ----------------------------------------------------------------------------------------------------------------------
    Preprocessing validation summary (DS2 VAL sample)
    ----------------------------------------------------------------------------------------------------------------------
                metric     mean      std       min      max
    edge_energy_before 0.069022 0.036905  0.016726 0.375572
     edge_energy_after 0.120571 0.034194  0.071821 0.402352
        entropy_before 6.866192 0.472092  5.135309 7.798752
         entropy_after 7.288234 0.279061  5.830363 7.890225
       contrast_before 0.233743 0.033322  0.144569 0.352019
        contrast_after 0.239958 0.022307  0.180111 0.335932
       edge_gain_ratio 1.891872 0.415752  1.071303 4.545427
         entropy_delta 0.422042 0.217386  0.043292 1.127475
        contrast_delta 0.006215 0.015060 -0.022230 0.064666
    
    ======================================================================================================================
    STEP 14: SAVING CHECKPOINT + CSV
    ======================================================================================================================
    Saved checkpoint: /kaggle/working/outputs/ARCFNet_Ablation_RLUCB_PresetSelection_checkpoint.pth
    Saved CSV: /kaggle/working/outputs/ARCFNet_Ablation_RLUCB_PresetSelection_outputs.csv
    
    DONE
    Method: ARCF-Net = Adaptive RACE-FELCM with CRAF Fusion Network
    Ablation: RL-UCB Preset Selection, Fixed Clients, Full Participation
    Backbone: Residual Network-50
    Best round: 13
    Fixed clients => DS1=3, DS2=3, TOTAL=6
    Rounds completed: 15
    Global TEST acc: 0.9760
    Global TEST f1_macro: 0.9757
    DS1 TEST acc: 0.9690
    DS2 TEST acc: 0.9829
    DS1 final strategy: validation_selected_single_after_rlucb_training | names=['race_sharp']
    DS2 final strategy: validation_selected_single_after_rlucb_training | names=['race_robust']


# **5. fixed client participation**


```python
import os
import sys
import time
import math
import copy
import hashlib
import random
import subprocess
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    log_loss, confusion_matrix, roc_auc_score,
    matthews_corrcoef, cohen_kappa_score, balanced_accuracy_score,
    jaccard_score
)

# ============================================================
# ARCF-Net ABLATION 4
# RL-UCB PRESET SELECTION / FIXED CLIENTS / FULL PARTICIPATION / NO PLOTS
# ------------------------------------------------------------
# - Uses BOTH datasets
# - Kaggle-ready
# - True FL with FedAvg + FedProx + prototype sharing
# - RL-UCB ONLY for preprocessing preset selection
# - Fixed client count per dataset
# - Full participation every round
# - No plots generated
# ============================================================

# -------------------------
# Helpers
# -------------------------
def pip_install(pkg):
    subprocess.check_call([sys.executable, "-m", "pip", "-q", "install", pkg])

def running_on_kaggle():
    return os.path.exists("/kaggle/input")

IS_KAGGLE = running_on_kaggle()

try:
    import kagglehub
except Exception:
    try:
        pip_install("kagglehub")
        import kagglehub
    except Exception:
        kagglehub = None

try:
    from torchvision import transforms
    from torchvision.models import resnet50, ResNet50_Weights
except Exception:
    pip_install("torchvision")
    from torchvision import transforms
    from torchvision.models import resnet50, ResNet50_Weights

# -------------------------
# Reproducibility + Device
# -------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if DEVICE.type == "cuda":
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

print("=" * 118)
print("ARCF-Net ABLATION 4: RL-UCB PRESET SELECTION + FIXED CLIENTS + FULL PARTICIPATION + NO PLOTS")
print("=" * 118)
print(f"ENV: {'KAGGLE' if IS_KAGGLE else 'NON-KAGGLE'} | DEVICE: {DEVICE} | torch={torch.__version__}")
print("=" * 118)

# -------------------------
# Configuration
# -------------------------
CFG = {
    "rounds": 15,

    # fixed clients (no RL participation / no RL client-count planning)
    "fixed_clients_ds1": 3,
    "fixed_clients_ds2": 3,

    # local training
    "local_epochs": 2,
    "lr_head": 1e-3,
    "lr_backbone": 2e-4,
    "weight_decay": 5e-4,
    "warmup_epochs": 1,
    "label_smoothing": 0.02,
    "focal_gamma": 1.35,
    "grad_clip": 1.0,
    "fedprox_mu": 0.01,
    "proto_lambda": 0.12,

    # image
    "img_size": 224 if torch.cuda.is_available() else 160,
    "batch_size": 16 if torch.cuda.is_available() else 8,
    "num_workers": 2 if torch.cuda.is_available() else 2,

    # global split
    "global_val_frac": 0.15,
    "test_frac": 0.15,

    # client split
    "client_val_frac": 0.12,
    "client_tune_frac": 0.12,
    "min_per_class_per_client": 5,

    # non-iid
    "dirichlet_alpha": 0.35,

    # preprocessing / augmentation
    "use_preprocessing": True,
    "use_augmentation": True,

    # transfer learning
    "freeze_backbone_rounds": 2,
    "unfreeze_last_blocks": 2,

    # RL-UCB preset selection
    "ucb_c": 1.35,
    "theta_probe_topk": 3,
    "theta_probe_batches": 2,

    # final inference
    "final_use_tta": True,

    # reward
    "reward_f1_weight": 0.75,
    "reward_acc_weight": 0.25,

    # best-round selection: equal dataset importance
    "best_round_mass_ds1": 0.50,
    "best_round_mass_ds2": 0.50,
    "best_round_min_bonus": 0.15,

    # FedAvg tempering
    "fedavg_temper": 0.50,

    # misc / sanity
    "quick_hash_subset_per_split": 300,
    "preproc_val_sample_n": 400,

    # no plots
    "make_plots": False,
    "calibration_bins": 12,
}

OUTDIR = "/kaggle/working/outputs" if IS_KAGGLE else "/content/outputs"
os.makedirs(OUTDIR, exist_ok=True)
MODEL_PATH = os.path.join(OUTDIR, "ARCFNet_Ablation_RLUCB_PresetSelection_checkpoint.pth")
CSV_PATH = os.path.join(OUTDIR, "ARCFNet_Ablation_RLUCB_PresetSelection_outputs.csv")

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], device=DEVICE).view(1, 3, 1, 1)
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225], device=DEVICE).view(1, 3, 1, 1)

labels = ["glioma", "meningioma", "notumor", "pituitary"]
label2id = {l: i for i, l in enumerate(labels)}
id2label = {i: l for l, i in label2id.items()}
NUM_CLASSES = len(labels)

METHOD_INFO = {
    "acronym": "ARCF-Net",
    "full_form": "Adaptive RACE-FELCM with CRAF Fusion Network",
    "preprocessing_full_form": "Robust Adaptive Context-Enhanced Fuzzy Edge Local Contrast Mapping",
    "fusion_full_form": "Cross-Residual Adaptive Fusion",
    "backbone_full_form": "Residual Network-50",
    "ablation_name": "RL-UCB Preset Selection, Fixed Clients, Full Participation",
}

# ============================================================
# Preset banks
# ============================================================
THETA_FULLFORMS = {
    "gamma": "Robust power exponent",
    "alpha": "Local contrast enhancement weight",
    "beta": "Contrast saturation sharpness",
    "tau": "Robust clipping threshold",
    "k": "Local context blur kernel",
    "edge_gain": "Gradient structure gain",
    "blend": "Structure-preserving residual blend",
}

PRESET_BANK_DS1 = [
    ("race_balanced",  (1.00, 0.24, 4.6, 2.4, 5, 0.10, 0.78)),
    ("race_sharp",     (1.04, 0.30, 5.2, 2.5, 5, 0.12, 0.82)),
    ("race_texture",   (0.92, 0.34, 5.8, 2.3, 7, 0.10, 0.80)),
    ("race_robust",    (1.08, 0.22, 4.5, 2.8, 5, 0.11, 0.76)),
    ("race_edge_plus", (1.02, 0.32, 6.0, 2.6, 3, 0.15, 0.84)),
    ("race_focus",     (1.10, 0.36, 6.4, 2.7, 3, 0.17, 0.86)),
]

PRESET_BANK_DS2 = [
    ("race_soft",      (0.95, 0.18, 3.8, 2.2, 3, 0.08, 0.72)),
    ("race_balanced",  (1.00, 0.24, 4.6, 2.4, 5, 0.10, 0.78)),
    ("race_robust",    (1.08, 0.22, 4.5, 2.8, 5, 0.11, 0.76)),
    ("race_smoothmix", (0.90, 0.20, 4.0, 2.1, 7, 0.07, 0.70)),
]

def get_preset_by_name(bank, target_name):
    for name, theta in bank:
        if name == target_name:
            return name, theta
    raise ValueError(f"Preset {target_name} not found.")

# ============================================================
# CSV collector
# ============================================================
ALL_ROWS = []

def add_table_to_csv(df, table_name):
    if df is None or len(df) == 0:
        return
    df2 = df.copy()
    df2.insert(0, "table_name", table_name)
    for _, row in df2.iterrows():
        ALL_ROWS.append(row.to_dict())

def print_table(df, title, max_rows=12):
    print("\n" + "-" * 118)
    print(title)
    print("-" * 118)
    if df is None or len(df) == 0:
        print("[empty]")
    else:
        print(df.head(max_rows).to_string(index=False))
        if len(df) > max_rows:
            print(f"... showing first {max_rows} of {len(df)} rows")

def safe_float(x, default=np.nan):
    try:
        return float(x)
    except Exception:
        return default

def normalize_imagenet(x):
    return (x - IMAGENET_MEAN) / IMAGENET_STD

def score_metric(m):
    return (
        CFG["reward_f1_weight"] * safe_float(m.get("f1_macro"), 0.0)
        + CFG["reward_acc_weight"] * safe_float(m.get("acc"), 0.0)
    )

def frame_to_manifest(df_):
    return df_[["path", "label", "source", "filename", "y"]].to_dict(orient="list")

# ============================================================
# STEP 0: ACCESS DATASETS
# ============================================================
print("\n" + "=" * 118)
print("STEP 0: ACCESS DATASETS")
print("=" * 118)

REQ1 = {"512Glioma", "512Meningioma", "512Normal", "512Pituitary"}
REQ2 = {"glioma", "meningioma", "notumor", "pituitary"}

def norm_label(name: str):
    s = str(name).strip().lower()
    if "glioma" in s:
        return "glioma"
    if "meningioma" in s:
        return "meningioma"
    if "pituitary" in s:
        return "pituitary"
    if "normal" in s or "no_tumor" in s or "no tumor" in s or "notumor" in s:
        return "notumor"
    return None

def find_root_with_required_class_dirs(base_dir, required_set, prefer_raw=True):
    candidates = []
    for root, dirs, _ in os.walk(base_dir):
        if required_set.issubset(set(dirs)):
            candidates.append(root)
    if not candidates:
        return None

    def score(p):
        pl = p.lower()
        sc = 0
        if prefer_raw:
            if "raw data" in pl:
                sc += 7
            if os.path.basename(p).lower() == "raw":
                sc += 7
            if "/raw/" in pl or "\\raw\\" in pl:
                sc += 3
            if "augmented" in pl:
                sc -= 20
        sc -= 0.0001 * len(p)
        return sc

    return max(candidates, key=score)

def search_kaggle_input_for_root(required_set, prefer_raw=True):
    if not os.path.exists("/kaggle/input"):
        return None
    return find_root_with_required_class_dirs("/kaggle/input", required_set, prefer_raw=prefer_raw)

def try_kagglehub_download(slug, required_set, prefer_raw=True):
    if kagglehub is None:
        return None, None
    try:
        base = kagglehub.dataset_download(slug)
        root = find_root_with_required_class_dirs(base, required_set, prefer_raw=prefer_raw)
        return base, root
    except Exception:
        return None, None

DS1_BASE = None
DS2_BASE = None

DS1_ROOT = search_kaggle_input_for_root(REQ1, prefer_raw=True)
DS2_ROOT = search_kaggle_input_for_root(REQ2, prefer_raw=False)

if DS1_ROOT is None:
    DS1_BASE, DS1_ROOT = try_kagglehub_download(
        "orvile/pmram-bangladeshi-brain-cancer-mri-dataset",
        REQ1,
        prefer_raw=True,
    )

if DS2_ROOT is None:
    DS2_BASE, DS2_ROOT = try_kagglehub_download(
        "yassinebazgour/preprocessed-brain-mri-scans-for-tumors-detection",
        REQ2,
        prefer_raw=False,
    )

if DS1_ROOT is None:
    raise RuntimeError(
        "Could not locate DS1 under /kaggle/input. Add "
        "'orvile/pmram-bangladeshi-brain-cancer-mri-dataset'."
    )
if DS2_ROOT is None:
    raise RuntimeError(
        "Could not locate DS2 under /kaggle/input. Add "
        "'yassinebazgour/preprocessed-brain-mri-scans-for-tumors-detection'."
    )

print(f"Dataset-1 RAW root detected:\n  {DS1_ROOT}")
print(f"Dataset-2 root detected:\n  {DS2_ROOT}")

# ============================================================
# STEP 1: BUILD MANIFESTS
# ============================================================
print("\n" + "=" * 118)
print("STEP 1: BUILD DATA MANIFESTS")
print("=" * 118)

def list_images_under_class_root(class_root, class_dir_name):
    class_dir = os.path.join(class_root, class_dir_name)
    out = []
    for r, _, files in os.walk(class_dir):
        for fn in files:
            if fn.lower().endswith(IMG_EXTS):
                out.append(os.path.join(r, fn))
    return out

def build_df_from_root(ds_root, class_dirs, source_name):
    rows = []
    for c in class_dirs:
        lab = norm_label(c)
        imgs = list_images_under_class_root(ds_root, c)
        print(f"{source_name}: {c} -> {lab} | {len(imgs)} images")
        for p in imgs:
            rows.append({"path": p, "label": lab, "source": source_name})
    dfm = pd.DataFrame(rows).dropna().reset_index(drop=True)
    dfm["path"] = dfm["path"].astype(str)
    dfm["label"] = dfm["label"].astype(str)
    dfm["source"] = dfm["source"].astype(str)
    dfm = dfm.drop_duplicates(subset=["path"]).reset_index(drop=True)
    dfm["filename"] = dfm["path"].apply(os.path.basename)
    return dfm

df1 = build_df_from_root(DS1_ROOT, ["512Glioma", "512Meningioma", "512Normal", "512Pituitary"], "ds1_raw")
df2 = build_df_from_root(DS2_ROOT, ["glioma", "meningioma", "notumor", "pituitary"], "ds2")

def enforce_labels(df_):
    df_ = df_.copy()
    df_["label"] = df_["label"].astype(str).str.strip().str.lower()
    df_ = df_[df_["label"].isin(set(labels))].reset_index(drop=True)
    df_["y"] = df_["label"].map(label2id).astype(int)
    return df_

df1 = enforce_labels(df1)
df2 = enforce_labels(df2)

print("\nDataset-1 images:", len(df1))
print(df1["label"].value_counts().reindex(labels, fill_value=0).to_string())
print("\nDataset-2 images:", len(df2))
print(df2["label"].value_counts().reindex(labels, fill_value=0).to_string())

# ============================================================
# STEP 2: TRAIN / VAL / TEST SPLIT
# ============================================================
print("\n" + "=" * 118)
print("STEP 2: TRAIN / VAL / TEST SPLIT")
print("=" * 118)

def split_dataset(df_):
    train_df, temp_df = train_test_split(
        df_,
        test_size=(CFG["global_val_frac"] + CFG["test_frac"]),
        stratify=df_["y"],
        random_state=SEED,
    )
    val_rel = CFG["global_val_frac"] / (CFG["global_val_frac"] + CFG["test_frac"])
    val_df, test_df = train_test_split(
        temp_df,
        test_size=(1 - val_rel),
        stratify=temp_df["y"],
        random_state=SEED,
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)

train1, val1, test1 = split_dataset(df1)
train2, val2, test2 = split_dataset(df2)

print(f"DS1 TRAIN: {len(train1)} | VAL: {len(val1)} | TEST: {len(test1)}")
print(f"DS2 TRAIN: {len(train2)} | VAL: {len(val2)} | TEST: {len(test2)}")

# ============================================================
# STEP 2.5: SANITY / LEAKAGE CHECKS
# ============================================================
print("\n" + "=" * 118)
print("STEP 2.5: SANITY / LEAKAGE CHECKS")
print("=" * 118)

def split_overlap_checks(train_df, val_df, test_df):
    tr = set(train_df["path"].tolist())
    va = set(val_df["path"].tolist())
    te = set(test_df["path"].tolist())
    checks = {
        "path_overlap_train_val": len(tr.intersection(va)),
        "path_overlap_train_test": len(tr.intersection(te)),
        "path_overlap_val_test": len(va.intersection(te)),
        "unique_paths_train": len(tr),
        "unique_paths_val": len(va),
        "unique_paths_test": len(te),
    }
    trf = set(train_df["filename"].tolist())
    vaf = set(val_df["filename"].tolist())
    tef = set(test_df["filename"].tolist())
    checks.update({
        "filename_overlap_train_val": len(trf.intersection(vaf)),
        "filename_overlap_train_test": len(trf.intersection(tef)),
        "filename_overlap_val_test": len(vaf.intersection(tef)),
    })
    return checks

def md5_file(path, max_bytes=2_000_000):
    h = hashlib.md5()
    try:
        with open(path, "rb") as f:
            h.update(f.read(max_bytes))
        return h.hexdigest()
    except Exception:
        return None

def quick_hash_subset(frame, n=300):
    n = min(n, len(frame))
    if n <= 0:
        return set()
    idx = np.random.choice(len(frame), size=n, replace=False)
    hashes = []
    for i in idx:
        hv = md5_file(frame.iloc[i]["path"])
        if hv is not None:
            hashes.append(hv)
    return set(hashes)

def leakage_report(name, tr, va, te):
    over = split_overlap_checks(tr, va, te)
    leak_df = pd.DataFrame([over])

    n_hash = int(CFG["quick_hash_subset_per_split"])
    trh = quick_hash_subset(tr, n_hash)
    vah = quick_hash_subset(va, n_hash)
    teh = quick_hash_subset(te, n_hash)

    hash_over = {
        "subset_hash_train_val": len(trh.intersection(vah)),
        "subset_hash_train_test": len(trh.intersection(teh)),
        "subset_hash_val_test": len(vah.intersection(teh)),
        "subset_hash_n_train": len(trh),
        "subset_hash_n_val": len(vah),
        "subset_hash_n_test": len(teh),
    }
    leak_df = pd.concat([leak_df, pd.DataFrame([hash_over])], axis=1)
    print_table(leak_df, f"Leakage / Sanity Summary — {name}", max_rows=5)
    add_table_to_csv(leak_df, f"leakage_sanity_{name}")

leakage_report("ds1", train1, val1, test1)
leakage_report("ds2", train2, val2, test2)

# ============================================================
# STEP 3: RL-UCB BANDIT
# ============================================================
print("\n" + "=" * 118)
print("STEP 3: RL-UCB BANDIT")
print("=" * 118)

class UCBBandit:
    def __init__(self, n_arms, c=1.35):
        self.n_arms = int(n_arms)
        self.c = float(c)
        self.counts = np.zeros(self.n_arms, dtype=np.int64)
        self.values = np.zeros(self.n_arms, dtype=np.float64)

    def ucb(self, arm):
        if self.counts[arm] == 0:
            return float("inf")
        t = max(1, self.counts.sum())
        return float(self.values[arm] + self.c * math.sqrt(math.log(t + 1) / self.counts[arm]))

    def select(self):
        scores = [self.ucb(i) for i in range(self.n_arms)]
        return int(np.argmax(scores))

    def update(self, arm, reward):
        arm = int(arm)
        reward = float(reward)
        self.counts[arm] += 1
        n = self.counts[arm]
        self.values[arm] += (reward - self.values[arm]) / n

    def best_arm(self):
        if self.counts.sum() == 0:
            return 0
        return int(np.argmax(self.values))

    def state_dict(self):
        return {
            "n_arms": self.n_arms,
            "c": self.c,
            "counts": self.counts.copy(),
            "values": self.values.copy(),
        }

    def load_state_dict(self, sd):
        self.n_arms = int(sd["n_arms"])
        self.c = float(sd["c"])
        self.counts = sd["counts"].copy()
        self.values = sd["values"].copy()

# ============================================================
# STEP 4: FIXED NON-IID CLIENT PARTITIONING
# ============================================================
print("\n" + "=" * 118)
print("STEP 4: FIXED NON-IID CLIENT PARTITIONING")
print("=" * 118)

def make_clients_non_iid(train_df, n_clients, num_classes, min_per_class=5, alpha=0.35):
    y = train_df["y"].values
    idx_by_class = {c: np.where(y == c)[0].tolist() for c in range(num_classes)}
    for c in idx_by_class:
        random.shuffle(idx_by_class[c])

    client_indices = [[] for _ in range(n_clients)]

    for c in range(num_classes):
        idxs = idx_by_class[c]
        feasible = min(min_per_class, max(1, len(idxs) // n_clients))
        for k in range(n_clients):
            take = idxs[:feasible]
            idxs = idxs[feasible:]
            client_indices[k].extend(take)
        idx_by_class[c] = idxs

    for c in range(num_classes):
        idxs = idx_by_class[c]
        if len(idxs) == 0:
            continue
        props = np.random.dirichlet([alpha] * n_clients)
        counts = (props * len(idxs)).astype(int)
        diff = len(idxs) - counts.sum()
        counts[np.argmax(props)] += diff

        start = 0
        for k in range(n_clients):
            client_indices[k].extend(idxs[start:start + counts[k]])
            start += counts[k]

    for k in range(n_clients):
        random.shuffle(client_indices[k])

    return client_indices

def robust_client_splits(train_df, indices, val_frac, tune_frac):
    idxs = np.array(indices, dtype=int)
    if len(idxs) < 3:
        return idxs.tolist(), idxs.tolist(), idxs.tolist()

    yk = train_df.loc[idxs, "y"].values

    if len(np.unique(yk)) < 2 or len(idxs) < 20:
        n_tune = max(1, int(round(len(idxs) * tune_frac)))
        n_tune = min(n_tune, max(1, len(idxs) - 2))
        tune_idx = idxs[:n_tune]
        rem_idx = idxs[n_tune:]
    else:
        rem_idx, tune_idx = train_test_split(
            idxs,
            test_size=tune_frac,
            stratify=yk,
            random_state=SEED,
        )

    if len(rem_idx) < 2:
        return rem_idx.tolist(), tune_idx.tolist(), rem_idx.tolist()

    yk2 = train_df.loc[rem_idx, "y"].values
    if len(np.unique(yk2)) < 2 or len(rem_idx) < 12:
        n_val = max(1, int(round(len(rem_idx) * val_frac)))
        n_val = min(n_val, max(1, len(rem_idx) - 1))
        val_idx = rem_idx[:n_val]
        train_idx = rem_idx[n_val:]
    else:
        train_idx, val_idx = train_test_split(
            rem_idx,
            test_size=val_frac,
            stratify=yk2,
            random_state=SEED,
        )

    if len(train_idx) == 0:
        train_idx = val_idx[:]
    if len(val_idx) == 0:
        val_idx = train_idx[:1]
    return train_idx.tolist(), tune_idx.tolist(), val_idx.tolist()

n_clients_ds1 = int(CFG["fixed_clients_ds1"])
n_clients_ds2 = int(CFG["fixed_clients_ds2"])

client_indices_ds1 = make_clients_non_iid(
    train1,
    n_clients=n_clients_ds1,
    num_classes=NUM_CLASSES,
    min_per_class=CFG["min_per_class_per_client"],
    alpha=CFG["dirichlet_alpha"],
)
client_indices_ds2 = make_clients_non_iid(
    train2,
    n_clients=n_clients_ds2,
    num_classes=NUM_CLASSES,
    min_per_class=CFG["min_per_class_per_client"],
    alpha=CFG["dirichlet_alpha"],
)

print(f"Fixed clients for DS1: {n_clients_ds1}")
print(f"Fixed clients for DS2: {n_clients_ds2}")

# ============================================================
# STEP 5: DATA LOADERS
# ============================================================
print("\n" + "=" * 118)
print("STEP 5: DATA LOADERS")
print("=" * 118)

def load_rgb(path):
    try:
        return Image.open(path).convert("RGB")
    except Exception:
        return Image.new("RGB", (CFG["img_size"], CFG["img_size"]), (128, 128, 128))

EVAL_TFMS = transforms.Compose([
    transforms.Resize((CFG["img_size"], CFG["img_size"])),
    transforms.ToTensor(),
])

TRAIN_TFMS = transforms.Compose([
    transforms.Resize((CFG["img_size"], CFG["img_size"])),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=12),
    transforms.RandomAffine(degrees=0, translate=(0.04, 0.04), scale=(0.96, 1.04)),
    transforms.ColorJitter(brightness=0.10, contrast=0.10),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.6)),
    transforms.ToTensor(),
])

class MRIDataset(Dataset):
    def __init__(self, frame, indices=None, tfms=None, source_id=0):
        self.df = frame
        self.indices = indices if indices is not None else list(range(len(frame)))
        self.tfms = tfms
        self.source_id = int(source_id)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        j = self.indices[i]
        row = self.df.iloc[j]
        img = load_rgb(row["path"])
        x = self.tfms(img) if self.tfms is not None else transforms.ToTensor()(img)
        y = int(row["y"])
        return x, y, row["path"], self.source_id

def make_weighted_sampler(frame, indices, num_classes):
    if len(indices) == 0:
        return None
    ys = frame.loc[indices, "y"].values
    class_counts = np.bincount(ys, minlength=num_classes)
    class_weights = 1.0 / np.sqrt(np.clip(class_counts, 1, None))
    sample_weights = class_weights[ys]
    return WeightedRandomSampler(
        weights=torch.DoubleTensor(sample_weights),
        num_samples=len(sample_weights),
        replacement=True,
    )

def make_loader(frame, indices, bs, tfms, shuffle=False, sampler=None, source_id=0):
    ds = MRIDataset(frame, indices=indices, tfms=tfms, source_id=source_id)
    return DataLoader(
        ds,
        batch_size=bs,
        shuffle=(shuffle and sampler is None),
        sampler=sampler,
        num_workers=CFG["num_workers"],
        pin_memory=(DEVICE.type == "cuda"),
        drop_last=False,
        persistent_workers=(CFG["num_workers"] > 0),
    )

clients = []
gid = 0

for ds_name, df_src, client_indices, source_id in [
    ("ds1", train1, client_indices_ds1, 0),
    ("ds2", train2, client_indices_ds2, 1),
]:
    for local_id, idxs in enumerate(client_indices):
        tr_idx, tune_idx, val_idx = robust_client_splits(df_src, idxs, CFG["client_val_frac"], CFG["client_tune_frac"])
        sampler = make_weighted_sampler(df_src, tr_idx, NUM_CLASSES)

        train_loader = make_loader(
            df_src, tr_idx, CFG["batch_size"], TRAIN_TFMS,
            shuffle=(sampler is None), sampler=sampler, source_id=source_id
        )
        tune_loader = make_loader(
            df_src,
            tune_idx if len(tune_idx) else tr_idx[:max(1, min(len(tr_idx), CFG["batch_size"]))],
            CFG["batch_size"], EVAL_TFMS, shuffle=False, sampler=None, source_id=source_id
        )
        val_loader = make_loader(
            df_src,
            val_idx if len(val_idx) else tr_idx[:max(1, min(len(tr_idx), CFG["batch_size"]))],
            CFG["batch_size"], EVAL_TFMS, shuffle=False, sampler=None, source_id=source_id
        )
        proto_loader = make_loader(
            df_src, tr_idx, CFG["batch_size"], EVAL_TFMS,
            shuffle=False, sampler=None, source_id=source_id
        )

        counts = df_src.loc[tr_idx, "label"].value_counts().reindex(labels, fill_value=0)

        preset_bank = PRESET_BANK_DS1 if ds_name == "ds1" else PRESET_BANK_DS2

        clients.append({
            "gid": gid,
            "local_id": local_id,
            "dataset": ds_name,
            "source_id": source_id,
            "train_loader": train_loader,
            "tune_loader": tune_loader,
            "val_loader": val_loader,
            "proto_loader": proto_loader,
            "n_train": len(tr_idx),
            "n_tune": len(tune_idx),
            "n_val": len(val_idx),
            "train_indices": tr_idx,
            "tune_indices": tune_idx,
            "val_indices": val_idx,
            "class_counts": counts.to_dict(),
            "preset_bank": preset_bank,
            "theta_bandit": UCBBandit(len(preset_bank), c=CFG["ucb_c"]),
        })

        print(f"{ds_name} | client_{gid} | train={len(tr_idx)} | tune={len(tune_idx)} | val={len(val_idx)}")
        gid += 1

CLIENTS_TOTAL = len(clients)

dist_rows = []
for c in clients:
    row = {
        "client": f"client_{c['gid']}",
        "dataset": c["dataset"],
        "total_train": c["n_train"],
        "total_tune": c["n_tune"],
        "total_val": c["n_val"],
        "preset_bank_names": str([x[0] for x in c["preset_bank"]]),
    }
    row.update({lab: int(c["class_counts"].get(lab, 0)) for lab in labels})
    dist_rows.append(row)

dist_df = pd.DataFrame(dist_rows)
print_table(dist_df, "Client class distribution", max_rows=20)
add_table_to_csv(dist_df, "client_distribution")

val_loader_ds1 = make_loader(val1, list(range(len(val1))), CFG["batch_size"], EVAL_TFMS, shuffle=False, source_id=0)
val_loader_ds2 = make_loader(val2, list(range(len(val2))), CFG["batch_size"], EVAL_TFMS, shuffle=False, source_id=1)
test_loader_ds1 = make_loader(test1, list(range(len(test1))), CFG["batch_size"], EVAL_TFMS, shuffle=False, source_id=0)
test_loader_ds2 = make_loader(test2, list(range(len(test2))), CFG["batch_size"], EVAL_TFMS, shuffle=False, source_id=1)

print(f"Augmentation: {'ON' if CFG['use_augmentation'] else 'OFF'}")
print(f"Preprocessing: {'ON' if CFG['use_preprocessing'] else 'OFF'}")
print(f"Total clients: {CLIENTS_TOTAL}")

# ============================================================
# STEP 6: PREPROCESSING — RACE-FELCM
# ============================================================
print("\n" + "=" * 118)
print("STEP 6: PREPROCESSING — RACE-FELCM")
print("=" * 118)

class RACEFELCM(nn.Module):
    def __init__(self, gamma=1.0, alpha=0.24, beta=4.5, tau=2.4, blur_k=5, edge_gain=0.10, blend=0.78):
        super().__init__()
        self.gamma = float(gamma)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.tau = float(tau)
        self.blur_k = int(blur_k)
        self.edge_gain = float(edge_gain)
        self.blend = float(blend)

        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        lap = torch.tensor([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=torch.float32)

        self.register_buffer("sobel_x", sobel_x.view(1, 1, 3, 3))
        self.register_buffer("sobel_y", sobel_y.view(1, 1, 3, 3))
        self.register_buffer("lap", lap.view(1, 1, 3, 3))

    def forward(self, x):
        eps = 1e-6

        gray0 = x.mean(dim=1, keepdim=True)
        mu0 = F.avg_pool2d(F.pad(gray0, (1, 1, 1, 1), mode="reflect"), 3, stride=1)
        var0 = F.avg_pool2d(F.pad((gray0 - mu0).pow(2), (1, 1, 1, 1), mode="reflect"), 3, stride=1)
        tex = torch.sqrt(var0 + eps)
        tex_norm = tex / (tex.amax(dim=(2, 3), keepdim=True) + eps)
        flat_gate = 1.0 - tex_norm

        x_smooth = F.avg_pool2d(F.pad(x, (1, 1, 1, 1), mode="reflect"), 3, stride=1)
        x_dn = x * (1.0 - 0.18 * flat_gate) + x_smooth * (0.18 * flat_gate)

        mu = x_dn.mean(dim=(2, 3), keepdim=True)
        sd = x_dn.std(dim=(2, 3), keepdim=True).clamp_min(eps)
        z = ((x_dn - mu) / sd).clamp(-self.tau, self.tau)
        z = torch.sign(z) * torch.pow(z.abs().clamp_min(eps), self.gamma)

        gray = z.mean(dim=1, keepdim=True)
        k = self.blur_k if self.blur_k % 2 == 1 else self.blur_k + 1
        pad = k // 2

        local_mean = F.avg_pool2d(F.pad(gray, (pad, pad, pad, pad), mode="reflect"), k, stride=1)
        local_var = F.avg_pool2d(F.pad((gray - local_mean).pow(2), (pad, pad, pad, pad), mode="reflect"), k, stride=1)
        local_std = torch.sqrt(local_var + eps)
        local_norm = (gray - local_mean) / (local_std + eps)
        contrast_field = torch.tanh(self.beta * local_norm)

        gx = F.conv2d(F.pad(gray, (1, 1, 1, 1), mode="reflect"), self.sobel_x)
        gy = F.conv2d(F.pad(gray, (1, 1, 1, 1), mode="reflect"), self.sobel_y)
        edge = torch.sqrt(gx * gx + gy * gy + eps)
        edge = edge / (edge.amax(dim=(2, 3), keepdim=True) + eps)

        lap = F.conv2d(F.pad(gray, (1, 1, 1, 1), mode="reflect"), self.lap).abs()
        lap = lap / (lap.amax(dim=(2, 3), keepdim=True) + eps)
        coherence = torch.sigmoid(4.0 * (edge - 0.5 * lap))

        enhanced = z + self.alpha * contrast_field + self.edge_gain * edge * coherence

        mn = enhanced.amin(dim=(2, 3), keepdim=True)
        mx = enhanced.amax(dim=(2, 3), keepdim=True)
        enhanced = (enhanced - mn) / (mx - mn + eps)

        out = self.blend * enhanced + (1.0 - self.blend) * x
        return out.clamp(0, 1)

def theta_to_module(theta):
    return RACEFELCM(*theta)

def theta_to_vec(theta, batch_size):
    gamma, alpha, beta, tau, blur_k, edge_gain, blend = theta
    theta = torch.tensor(
        [gamma, alpha, beta / 8.0, tau / 4.0, blur_k / 7.0, edge_gain, blend],
        device=DEVICE,
        dtype=torch.float32
    )
    return theta.unsqueeze(0).repeat(batch_size, 1)

def theta_str(theta):
    if theta is None:
        return "None"
    g, a, b, t, k, eg, m = theta
    return f"(g={g:.2f}, a={a:.2f}, b={b:.2f}, t={t:.2f}, k={k}, eg={eg:.2f}, mix={m:.2f})"

# ============================================================
# STEP 7: MODEL — ResNet-50 + CRAF Fusion
# ============================================================
print("\n" + "=" * 118)
print("STEP 7: MODEL — ResNet-50 + CRAF Fusion")
print("=" * 118)

class CompactProjector(nn.Module):
    def __init__(self, in_dim, out_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            nn.Dropout(0.10),
        )

    def forward(self, x):
        return self.net(x)

class CRAFFusion(nn.Module):
    def __init__(self, in_dim, fuse_dim=256, cond_dim=64):
        super().__init__()
        self.proj_raw = CompactProjector(in_dim, fuse_dim)
        self.proj_enh = CompactProjector(in_dim, fuse_dim)
        self.proj_res = CompactProjector(in_dim, fuse_dim)

        self.router = nn.Sequential(
            nn.Linear(fuse_dim * 6 + cond_dim, fuse_dim),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(fuse_dim, 3),
        )

        self.residual_refine = nn.Sequential(
            nn.Linear(fuse_dim * 2 + cond_dim, fuse_dim),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(fuse_dim, fuse_dim),
        )

        self.out_norm = nn.LayerNorm(fuse_dim)

    def forward(self, f_raw, f_enh, f_res, cond):
        fr = self.proj_raw(f_raw)
        fe = self.proj_enh(f_enh)
        fs = self.proj_res(f_res)

        d_re = torch.abs(fr - fe)
        d_rs = torch.abs(fr - fs)
        d_es = torch.abs(fe - fs)

        router_in = torch.cat([fr, fe, fs, d_re, d_rs, d_es, cond], dim=1)
        gates = torch.softmax(self.router(router_in), dim=1)

        routed = gates[:, 0:1] * fr + gates[:, 1:2] * fe + gates[:, 2:3] * fs
        disagreement = (d_re + d_rs + d_es) / 3.0
        refine = self.residual_refine(torch.cat([routed, disagreement, cond], dim=1))
        fused = self.out_norm(routed + refine)

        return fused, gates

class ResNet50CRAF(nn.Module):
    def __init__(self, num_classes, cond_dim=64, fuse_dim=256, embed_dim=256, pretrained=True):
        super().__init__()
        try:
            weights = ResNet50_Weights.DEFAULT if pretrained else None
        except Exception:
            weights = None

        net = resnet50(weights=weights)
        self.backbone = nn.Sequential(
            net.conv1,
            net.bn1,
            net.relu,
            net.maxpool,
            net.layer1,
            net.layer2,
            net.layer3,
            net.layer4,
        )
        self.backbone_dim = net.fc.in_features
        self.pool = nn.AdaptiveAvgPool2d(1)

        self.theta_mlp = nn.Sequential(
            nn.Linear(7, cond_dim),
            nn.GELU(),
            nn.Linear(cond_dim, cond_dim),
        )
        self.source_emb = nn.Embedding(2, cond_dim)
        self.cond_norm = nn.LayerNorm(cond_dim)

        self.fusion = CRAFFusion(self.backbone_dim, fuse_dim=fuse_dim, cond_dim=cond_dim)

        self.neck = nn.Sequential(
            nn.LayerNorm(fuse_dim),
            nn.Linear(fuse_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(0.20),
        )
        self.classifier = nn.Linear(embed_dim, num_classes)
        self.embed_dim = embed_dim

    def _encode(self, x):
        f = self.backbone(x)
        f = self.pool(f).flatten(1)
        return f

    def forward(self, x_raw_n, x_enh_n, x_res_n, theta_vec, source_id, return_extra=False):
        cond = self.theta_mlp(theta_vec) + self.source_emb(source_id)
        cond = self.cond_norm(cond)

        f_raw = self._encode(x_raw_n)
        f_enh = self._encode(x_enh_n)
        f_res = self._encode(x_res_n)

        fused, gates = self.fusion(f_raw, f_enh, f_res, cond)
        embed = self.neck(fused)
        logits = self.classifier(embed)

        if return_extra:
            return logits, embed, gates
        return logits

def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

def set_trainable_for_round(model, rnd):
    for p in model.backbone.parameters():
        p.requires_grad = False

    for n, p in model.named_parameters():
        if not n.startswith("backbone."):
            p.requires_grad = True

    if rnd > CFG["freeze_backbone_rounds"]:
        blocks = list(model.backbone.children())
        tail_blocks = blocks[-CFG["unfreeze_last_blocks"]:]
        for blk in tail_blocks:
            for p in blk.parameters():
                p.requires_grad = True

def freeze_backbone_bn_stats(model):
    for m in model.backbone.modules():
        if isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm)):
            m.eval()
            for p in m.parameters():
                p.requires_grad = False

def make_optimizer(model):
    head_params, bb_params = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if n.startswith("backbone."):
            bb_params.append(p)
        else:
            head_params.append(p)

    groups = []
    if head_params:
        groups.append({"params": head_params, "lr": CFG["lr_head"]})
    if bb_params:
        groups.append({"params": bb_params, "lr": CFG["lr_backbone"]})

    return torch.optim.AdamW(groups, weight_decay=CFG["weight_decay"])

def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
    def lr_lambda(step):
        if step < num_warmup_steps:
            return float(step) / float(max(1, num_warmup_steps))
        progress = float(step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

global_model = ResNet50CRAF(
    num_classes=NUM_CLASSES,
    cond_dim=64,
    fuse_dim=256,
    embed_dim=256,
    pretrained=True,
).to(DEVICE)

set_trainable_for_round(global_model, rnd=1)
total_params, trainable_params = count_params(global_model)

print("Backbone: ResNet-50 | pretrained_loaded=True")
print(f"Total params: {total_params:,}")
print(f"Trainable params: {trainable_params:,} ({(100.0 * trainable_params / total_params):.2f}%)")

# ============================================================
# STEP 8: LOSSES + PROTOTYPE SHARING + EVAL HELPERS
# ============================================================
print("\n" + "=" * 118)
print("STEP 8: LOSSES + PROTOTYPE SHARING + EVAL HELPERS")
print("=" * 118)

counts1 = train1["y"].value_counts().sort_index().reindex(range(NUM_CLASSES), fill_value=0).values
counts2 = train2["y"].value_counts().sort_index().reindex(range(NUM_CLASSES), fill_value=0).values
counts = counts1 + counts2
w = (counts.sum() / np.clip(counts, 1, None)).astype(np.float32)
w = w / max(1e-6, w.mean())
class_w = torch.tensor(w, device=DEVICE)

class ClassBalancedFocalLoss(nn.Module):
    def __init__(self, class_weights, gamma=1.5, label_smoothing=0.0):
        super().__init__()
        self.class_weights = class_weights
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, logits, target):
        ce = F.cross_entropy(
            logits,
            target,
            weight=self.class_weights,
            reduction="none",
            label_smoothing=self.label_smoothing,
        )
        pt = torch.exp(-ce)
        focal = (1.0 - pt).pow(self.gamma) * ce
        return focal.mean()

criterion = ClassBalancedFocalLoss(
    class_weights=class_w,
    gamma=CFG["focal_gamma"],
    label_smoothing=CFG["label_smoothing"],
)

scaler = torch.amp.GradScaler("cuda", enabled=(DEVICE.type == "cuda")) if DEVICE.type == "cuda" else None

def fedprox_term(local_model, global_model):
    loss = 0.0
    for p_local, p_global in zip(local_model.parameters(), global_model.parameters()):
        loss = loss + ((p_local - p_global.detach()) ** 2).sum()
    return loss

def prototype_alignment_loss(embed, y, global_prototypes):
    if global_prototypes is None:
        return torch.tensor(0.0, device=embed.device)

    proto = global_prototypes["proto"]
    mask = global_prototypes["mask"]
    valid = mask[y]
    if not valid.any():
        return torch.tensor(0.0, device=embed.device)

    emb_n = F.normalize(embed[valid], dim=1)
    ref_n = F.normalize(proto[y[valid]], dim=1)
    return (1.0 - (emb_n * ref_n).sum(dim=1)).mean()

@torch.no_grad()
def compute_prototype_payload(model, loader, preproc, theta):
    model.eval()
    freeze_backbone_bn_stats(model)
    preproc.eval()

    sums = torch.zeros(NUM_CLASSES, model.embed_dim, device=DEVICE)
    counts = torch.zeros(NUM_CLASSES, device=DEVICE)

    for x, y, _, source_id in loader:
        x = x.to(DEVICE, non_blocking=True)
        y = y.to(DEVICE, non_blocking=True)
        source_id = source_id.to(DEVICE, non_blocking=True)

        x_enh = preproc(x)
        x_res = torch.abs(x_enh - x)

        x_raw_n = normalize_imagenet(x)
        x_enh_n = normalize_imagenet(x_enh)
        x_res_n = normalize_imagenet(x_res)
        theta_vec = theta_to_vec(theta, x.size(0))

        _, embed, _ = model(x_raw_n, x_enh_n, x_res_n, theta_vec, source_id, return_extra=True)

        for c in range(NUM_CLASSES):
            m = (y == c)
            if m.any():
                sums[c] += embed[m].sum(dim=0)
                counts[c] += m.sum()

    return sums.detach().cpu(), counts.detach().cpu()

def aggregate_prototypes(payloads, embed_dim):
    if len(payloads) == 0:
        return None
    sum_all = torch.zeros(NUM_CLASSES, embed_dim)
    cnt_all = torch.zeros(NUM_CLASSES)
    for s, c in payloads:
        sum_all += s
        cnt_all += c
    proto = sum_all / cnt_all.unsqueeze(1).clamp_min(1.0)
    mask = cnt_all > 0
    return {
        "proto": proto.to(DEVICE),
        "mask": mask.to(DEVICE),
        "counts": cnt_all.to(DEVICE),
    }

def gate_entropy(g):
    eps = 1e-6
    p = g.clamp(eps, 1 - eps)
    return -(p * torch.log2(p)).sum(dim=1)

@torch.no_grad()
def _auc_metrics(y_true, p_pred, num_classes):
    out = {}
    try:
        out["auc_roc_macro_ovr"] = float(roc_auc_score(y_true, p_pred, multi_class="ovr", average="macro"))
    except Exception:
        out["auc_roc_macro_ovr"] = np.nan

    for c in range(num_classes):
        try:
            yc = (y_true == c).astype(int)
            if yc.sum() > 0 and yc.sum() < len(yc):
                out[f"auc_class_{c}"] = float(roc_auc_score(yc, p_pred[:, c]))
            else:
                out[f"auc_class_{c}"] = np.nan
        except Exception:
            out[f"auc_class_{c}"] = np.nan
    return out

@torch.no_grad()
def predict_probs_single_theta(model, x, source_id, preproc, theta, use_tta=False):
    model.eval()
    freeze_backbone_bn_stats(model)
    preproc.eval()

    probs_acc = None
    versions = [x]
    if use_tta:
        versions.append(torch.flip(x, dims=[3]))

    for xv in versions:
        x_enh = preproc(xv)
        x_res = torch.abs(x_enh - xv)

        x_raw_n = normalize_imagenet(xv)
        x_enh_n = normalize_imagenet(x_enh)
        x_res_n = normalize_imagenet(x_res)
        theta_vec = theta_to_vec(theta, xv.size(0))

        logits = model(x_raw_n, x_enh_n, x_res_n, theta_vec, source_id)
        probs = torch.softmax(logits, dim=1)
        probs_acc = probs if probs_acc is None else (probs_acc + probs)

    probs_acc = probs_acc / len(versions)
    return probs_acc

@torch.no_grad()
def evaluate_full(model, loader, preproc, theta, return_gates=False, use_tta=False):
    t0 = time.time()
    model.eval()
    freeze_backbone_bn_stats(model)
    preproc.eval()

    all_y, all_p, all_loss = [], [], []
    gate_stats = []

    for x, y, _, source_id in loader:
        x = x.to(DEVICE, non_blocking=True)
        y = y.to(DEVICE, non_blocking=True)
        source_id = source_id.to(DEVICE, non_blocking=True)

        x_enh = preproc(x)
        x_res = torch.abs(x_enh - x)

        x_raw_n = normalize_imagenet(x)
        x_enh_n = normalize_imagenet(x_enh)
        x_res_n = normalize_imagenet(x_res)
        theta_vec = theta_to_vec(theta, x.size(0))

        if return_gates and (not use_tta):
            logits, _, gates = model(x_raw_n, x_enh_n, x_res_n, theta_vec, source_id, return_extra=True)
            gate_stats.append(gates.detach().cpu())
            probs = torch.softmax(logits, dim=1)
        else:
            probs = predict_probs_single_theta(model, x, source_id, preproc, theta, use_tta=use_tta)
            logits = torch.log(probs.clamp_min(1e-8))

        loss = F.cross_entropy(logits, y)

        all_loss.append(float(loss.item()))
        all_y.append(y.detach().cpu().numpy())
        all_p.append(probs.detach().cpu().numpy())

    if len(all_y) == 0:
        return {"acc": np.nan}, np.array([]), np.array([])

    y_true = np.concatenate(all_y)
    p_pred = np.concatenate(all_p)
    y_hat = np.argmax(p_pred, axis=1)

    met = {
        "loss_ce": float(np.mean(all_loss)),
        "acc": float(accuracy_score(y_true, y_hat)),
        "precision_macro": float(precision_score(y_true, y_hat, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_hat, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_hat, average="macro", zero_division=0)),
        "precision_weighted": float(precision_score(y_true, y_hat, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(y_true, y_hat, average="weighted", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_hat, average="weighted", zero_division=0)),
        "log_loss": float(log_loss(y_true, p_pred, labels=list(range(NUM_CLASSES)))),
        "eval_time_s": float(time.time() - t0),
    }
    met.update(_auc_metrics(y_true, p_pred, NUM_CLASSES))

    if return_gates and len(gate_stats) > 0:
        g = torch.cat(gate_stats, dim=0)
        met["fusion_gate_mean_raw"] = float(g[:, 0].mean().item())
        met["fusion_gate_mean_enh"] = float(g[:, 1].mean().item())
        met["fusion_gate_mean_res"] = float(g[:, 2].mean().item())
        met["fusion_gate_entropy"] = float(gate_entropy(g).mean().item())

    return met, y_true, p_pred

def train_one_epoch(model, loader, optimizer, preproc, theta, global_model=None, global_prototypes=None, scheduler=None):
    model.train()
    freeze_backbone_bn_stats(model)
    preproc.eval()

    losses, ce_losses, proto_losses, correct, total = [], [], [], 0, 0

    for x, y, _, source_id in loader:
        x = x.to(DEVICE, non_blocking=True)
        y = y.to(DEVICE, non_blocking=True)
        source_id = source_id.to(DEVICE, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        amp_enabled = (DEVICE.type == "cuda")

        with torch.amp.autocast(device_type=DEVICE.type, enabled=amp_enabled):
            x_enh = preproc(x)
            x_res = torch.abs(x_enh - x)

            x_raw_n = normalize_imagenet(x)
            x_enh_n = normalize_imagenet(x_enh)
            x_res_n = normalize_imagenet(x_res)
            theta_vec = theta_to_vec(theta, x.size(0))

            logits, embed, _ = model(x_raw_n, x_enh_n, x_res_n, theta_vec, source_id, return_extra=True)

            ce = criterion(logits, y)
            proto = prototype_alignment_loss(embed, y, global_prototypes)
            loss = ce + CFG["proto_lambda"] * proto

            if global_model is not None and CFG["fedprox_mu"] > 0:
                loss = loss + 0.5 * CFG["fedprox_mu"] * fedprox_term(model, global_model)

        if scaler is not None and amp_enabled:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if CFG["grad_clip"] > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), CFG["grad_clip"])
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if CFG["grad_clip"] > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), CFG["grad_clip"])
            optimizer.step()

        if scheduler is not None:
            scheduler.step()

        losses.append(float(loss.item()))
        ce_losses.append(float(ce.item()))
        proto_losses.append(float(proto.item()))

        preds = logits.argmax(dim=1)
        correct += int((preds == y).sum().item())
        total += int(y.size(0))

    return {
        "loss": float(np.mean(losses)) if losses else np.nan,
        "ce_loss": float(np.mean(ce_losses)) if ce_losses else np.nan,
        "proto_loss": float(np.mean(proto_losses)) if proto_losses else np.nan,
        "acc": float(correct / max(1, total)),
    }

def build_tempered_fedavg_weights(clients_subset):
    sizes = np.array([c["n_train"] for c in clients_subset], dtype=np.float64)
    sizes = np.power(np.clip(sizes, 1.0, None), CFG["fedavg_temper"])
    sizes = sizes / max(1e-12, sizes.sum())
    return sizes.tolist()

def fedavg_update(global_model, local_models, weights):
    global_sd = global_model.state_dict()
    new_sd = {}

    for key in global_sd.keys():
        ref = global_sd[key]
        if not torch.is_floating_point(ref):
            new_sd[key] = local_models[0].state_dict()[key].detach().clone()
        else:
            acc = None
            for m, w in zip(local_models, weights):
                t = m.state_dict()[key].detach().float().cpu()
                acc = t * w if acc is None else acc + t * w
            new_sd[key] = acc.to(ref.dtype)

    global_model.load_state_dict(new_sd, strict=True)

def weighted_mean(rows, key, weight_key):
    vals, ws = [], []
    for r in rows:
        v = r.get(key, np.nan)
        w = r.get(weight_key, 0)
        if np.isfinite(v) and w > 0:
            vals.append(v)
            ws.append(w)
    if len(vals) == 0:
        return np.nan
    return float(np.average(vals, weights=ws))

def aggregate_by_dataset(round_local_rows, dataset_name):
    rows = [r for r in round_local_rows if r["dataset"] == dataset_name]
    return {
        "acc": weighted_mean(rows, "val_acc", "val_size"),
        "f1_macro": weighted_mean(rows, "val_f1_macro", "val_size"),
        "precision_macro": weighted_mean(rows, "val_precision_macro", "val_size"),
        "recall_macro": weighted_mean(rows, "val_recall_macro", "val_size"),
        "log_loss": weighted_mean(rows, "val_log_loss", "val_size"),
        "loss_ce": weighted_mean(rows, "val_loss_ce", "val_size"),
        "eval_time_s": weighted_mean(rows, "val_eval_time_s", "val_size"),
    }

# ============================================================
# STEP 9: TUNE-AWARE RL-UCB PRESET SELECTION
# ============================================================
print("\n" + "=" * 118)
print("STEP 9: TUNE-AWARE RL-UCB PRESET SELECTION")
print("=" * 118)

@torch.no_grad()
def probe_theta_on_tune_loader(model, tune_loader, preset_bank, arm_candidates, n_batches=2):
    model.eval()
    freeze_backbone_bn_stats(model)

    cached_batches = []
    try:
        it = iter(tune_loader)
        for _ in range(n_batches):
            cached_batches.append(next(it))
    except Exception:
        pass

    if len(cached_batches) == 0:
        return int(arm_candidates[0])

    best_arm = int(arm_candidates[0])
    best_score = -1.0

    for arm in arm_candidates:
        arm = int(arm)
        _, theta = preset_bank[arm]
        pre = theta_to_module(theta).to(DEVICE).eval()

        batch_scores = []
        for x, y, _, source_id in cached_batches:
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            source_id = source_id.to(DEVICE)

            probs = predict_probs_single_theta(model, x, source_id, pre, theta, use_tta=False)
            pred = probs.argmax(dim=1)
            batch_scores.append((pred == y).float().mean().item())

        sc = float(np.mean(batch_scores)) if len(batch_scores) else 0.0
        if sc > best_score:
            best_score = sc
            best_arm = arm

    return int(best_arm)

def select_theta_arm_with_probe(client, model):
    bandit = client["theta_bandit"]
    ucb_scores = [bandit.ucb(i) for i in range(bandit.n_arms)]
    topk = min(CFG["theta_probe_topk"], bandit.n_arms)
    candidates = list(np.argsort(ucb_scores)[::-1][:topk])
    return probe_theta_on_tune_loader(
        model,
        client["tune_loader"],
        client["preset_bank"],
        candidates,
        n_batches=CFG["theta_probe_batches"],
    )

# ============================================================
# STEP 10: TRUE FEDERATED TRAINING
# ============================================================
print("\n" + "=" * 118)
print("STEP 10: TRUE FEDERATED TRAINING")
print("=" * 118)

history_global = []
history_local = []

best_reward = -1.0
best_round_saved = None
best_model_state = None
best_global_prototypes = None
best_theta_bandit_states = None
global_prototypes = None

t_global_start = time.time()

print(f"Clients => DS1={n_clients_ds1}, DS2={n_clients_ds2}, TOTAL={CLIENTS_TOTAL}")
print(f"Rounds: {CFG['rounds']} | Local epochs: {CFG['local_epochs']}")
print(f"Augmentation ON: {CFG['use_augmentation']}")
print("Transfer backbone: ResNet-50")
print("Preprocessing: RACE-FELCM")
print("Fusion: CRAF")
print(f"FedProx μ={CFG['fedprox_mu']} | Proto λ={CFG['proto_lambda']}")
print(f"Tempered FedAvg exponent = {CFG['fedavg_temper']:.2f}")
print(f"Best-round masses => DS1={CFG['best_round_mass_ds1']:.2f}, DS2={CFG['best_round_mass_ds2']:.2f}, min-bonus={CFG['best_round_min_bonus']:.2f}")

for rnd in range(1, CFG["rounds"] + 1):
    round_t0 = time.time()
    selected_ids = list(range(len(clients)))  # full participation every round

    print("\n" + "=" * 118)
    print(f"ROUND {rnd}/{CFG['rounds']} | selected={selected_ids}")
    print("=" * 118)

    local_models = []
    proto_payloads = []
    round_local_rows = []
    selected_clients_meta = []

    for cid in selected_ids:
        client = clients[cid]

        theta_arm = select_theta_arm_with_probe(client, global_model)
        theta_name, theta = client["preset_bank"][theta_arm]
        preproc = theta_to_module(theta).to(DEVICE)

        local_model = ResNet50CRAF(
            num_classes=NUM_CLASSES,
            cond_dim=64,
            fuse_dim=256,
            embed_dim=256,
            pretrained=False,
        ).to(DEVICE)
        local_model.load_state_dict(global_model.state_dict(), strict=True)

        set_trainable_for_round(local_model, rnd)
        opt = make_optimizer(local_model)

        total_steps = max(1, len(client["train_loader"]) * CFG["local_epochs"])
        warmup_steps = max(1, len(client["train_loader"]) * CFG["warmup_epochs"])
        scheduler = get_cosine_schedule_with_warmup(opt, warmup_steps, total_steps)

        train_logs = []
        for _ in range(CFG["local_epochs"]):
            log_ep = train_one_epoch(
                local_model,
                client["train_loader"],
                opt,
                preproc,
                theta,
                global_model=global_model,
                global_prototypes=global_prototypes,
                scheduler=scheduler,
            )
            train_logs.append(log_ep)

        met_loc, _, _ = evaluate_full(local_model, client["val_loader"], preproc, theta, return_gates=True, use_tta=False)
        proto_payload = compute_prototype_payload(local_model, client["proto_loader"], preproc, theta)

        reward = score_metric(met_loc)
        client["theta_bandit"].update(theta_arm, reward)

        local_models.append(local_model)
        selected_clients_meta.append(client)
        proto_payloads.append(proto_payload)

        g, a, b, t, kk, eg, mix = theta
        row = {
            "round": rnd,
            "client": f"client_{cid}",
            "dataset": client["dataset"],
            "selected": 1,
            "theta_arm": int(theta_arm),
            "theta_name": theta_name,
            "theta_str": theta_str(theta),
            "gamma_power": g,
            "alpha_contrast_weight": a,
            "beta_contrast_sharpness": b,
            "tau_clip": t,
            "k_blur_kernel_size": kk,
            "edge_gain": eg,
            "blend_mix": mix,
            "train_loss": float(np.mean([x["loss"] for x in train_logs])),
            "train_ce_loss": float(np.mean([x["ce_loss"] for x in train_logs])),
            "train_proto_loss": float(np.mean([x["proto_loss"] for x in train_logs])),
            "train_acc": float(np.mean([x["acc"] for x in train_logs])),
            "train_time_s": np.nan,
            "val_size": client["n_val"],
            **{f"val_{k}": v for k, v in met_loc.items()},
            "reward": reward,
            "bandit_counts_total": int(client["theta_bandit"].counts.sum()),
            "bandit_value_selected": float(client["theta_bandit"].values[theta_arm]),
        }
        round_local_rows.append(row)

        auc_val = row.get("val_auc_roc_macro_ovr", np.nan)
        print(
            f"Client {cid} ({client['dataset']}) | "
            f"train_acc={row['train_acc']:.4f} | "
            f"val_acc={row['val_acc']:.4f} | "
            f"val_f1={row['val_f1_macro']:.4f} | "
            f"val_auc={auc_val:.4f} | "
            f"reward={reward:.4f} | "
            f"theta={theta_name} {theta_str(theta)}"
        )

    agg_weights = build_tempered_fedavg_weights(selected_clients_meta)
    fedavg_update(global_model, local_models, agg_weights)
    global_prototypes = aggregate_prototypes(proto_payloads, global_model.embed_dim)

    ds1_metrics = aggregate_by_dataset(round_local_rows, "ds1")
    ds2_metrics = aggregate_by_dataset(round_local_rows, "ds2")

    global_metrics = {
        "acc": weighted_mean(round_local_rows, "val_acc", "val_size"),
        "f1_macro": weighted_mean(round_local_rows, "val_f1_macro", "val_size"),
        "precision_macro": weighted_mean(round_local_rows, "val_precision_macro", "val_size"),
        "recall_macro": weighted_mean(round_local_rows, "val_recall_macro", "val_size"),
        "log_loss": weighted_mean(round_local_rows, "val_log_loss", "val_size"),
        "loss_ce": weighted_mean(round_local_rows, "val_loss_ce", "val_size"),
        "eval_time_s": weighted_mean(round_local_rows, "val_eval_time_s", "val_size"),
    }

    ds1_score = score_metric(ds1_metrics)
    ds2_score = score_metric(ds2_metrics)
    round_reward = (
        CFG["best_round_mass_ds1"] * ds1_score +
        CFG["best_round_mass_ds2"] * ds2_score +
        CFG["best_round_min_bonus"] * min(ds1_score, ds2_score)
    )

    history_local.extend(round_local_rows)
    history_global.append({
        "round": rnd,
        "round_time_s": float(time.time() - round_t0),
        "n_selected_clients": len(selected_ids),
        "active_fraction": 1.0,
        "global_reward": round_reward,
        "global_acc": global_metrics["acc"],
        "global_f1_macro": global_metrics["f1_macro"],
        "global_precision_macro": global_metrics["precision_macro"],
        "global_recall_macro": global_metrics["recall_macro"],
        "global_log_loss": global_metrics["log_loss"],
        "global_loss_ce": global_metrics["loss_ce"],
        "global_eval_time_s": global_metrics["eval_time_s"],
        "ds1_acc": ds1_metrics["acc"],
        "ds1_f1_macro": ds1_metrics["f1_macro"],
        "ds1_log_loss": ds1_metrics["log_loss"],
        "ds2_acc": ds2_metrics["acc"],
        "ds2_f1_macro": ds2_metrics["f1_macro"],
        "ds2_log_loss": ds2_metrics["log_loss"],
    })

    print(
        f"GLOBAL VAL (Round {rnd}) | "
        f"global_acc={global_metrics['acc']:.4f} | "
        f"global_f1={global_metrics['f1_macro']:.4f} | "
        f"ds1_acc={ds1_metrics['acc']:.4f} | "
        f"ds1_f1={ds1_metrics['f1_macro']:.4f} | "
        f"ds2_acc={ds2_metrics['acc']:.4f} | "
        f"ds2_f1={ds2_metrics['f1_macro']:.4f} | "
        f"reward={round_reward:.4f} | "
        f"round_time={history_global[-1]['round_time_s']:.1f}s"
    )

    if np.isfinite(round_reward) and round_reward > best_reward:
        best_reward = float(round_reward)
        best_round_saved = rnd
        best_model_state = {k: v.detach().cpu().clone() for k, v in global_model.state_dict().items()}
        best_global_prototypes = None if global_prototypes is None else {
            "proto": global_prototypes["proto"].detach().cpu().clone(),
            "mask": global_prototypes["mask"].detach().cpu().clone(),
            "counts": global_prototypes["counts"].detach().cpu().clone(),
        }
        best_theta_bandit_states = [copy.deepcopy(c["theta_bandit"].state_dict()) for c in clients]

if best_model_state is not None:
    global_model.load_state_dict({k: v.to(DEVICE) for k, v in best_model_state.items()})

if best_theta_bandit_states is not None:
    for c, sd in zip(clients, best_theta_bandit_states):
        c["theta_bandit"].load_state_dict(sd)

if best_global_prototypes is not None:
    global_prototypes = {
        "proto": best_global_prototypes["proto"].to(DEVICE),
        "mask": best_global_prototypes["mask"].to(DEVICE),
        "counts": best_global_prototypes["counts"].to(DEVICE),
    }

t_total = float(time.time() - t_global_start)
print("\n" + "=" * 118)
print(f"TRAINING COMPLETE | total_time={t_total:.1f}s | best_round={best_round_saved} | best_reward={best_reward:.4f}")
print("=" * 118)

glob_df = pd.DataFrame(history_global)
loc_df = pd.DataFrame(history_local)

print_table(glob_df, "GLOBAL per-round metrics", max_rows=20)
print_table(loc_df, "LOCAL per-client per-round metrics", max_rows=20)
add_table_to_csv(glob_df, "global_round_metrics_full")
add_table_to_csv(loc_df, "client_round_metrics_full")

# ============================================================
# STEP 11: FINAL EVALUATION
# ============================================================
print("\n" + "=" * 118)
print("STEP 11: FINAL EVALUATION")
print("=" * 118)

def best_theta_for_client(client):
    arm = client["theta_bandit"].best_arm()
    name, theta = client["preset_bank"][arm]
    return arm, name, theta, client["theta_bandit"].values[arm]

def unique_theta_candidates(csubset):
    out = []
    seen = set()
    for c in csubset:
        arm, name, theta, val = best_theta_for_client(c)
        key = tuple([round(x, 6) if isinstance(x, float) else x for x in theta])
        if key not in seen:
            out.append((name, theta, float(val), c["gid"]))
            seen.add(key)
    return out

@torch.no_grad()
def evaluate_with_single_theta(model, loader, theta, use_tta=False):
    pre = theta_to_module(theta).to(DEVICE)
    return evaluate_full(model, loader, pre, theta, return_gates=False, use_tta=use_tta)

def rank_rlucb_candidates(model, val_loader, candidates):
    ranked = []
    for name, theta, est, gid in candidates:
        met, _, _ = evaluate_with_single_theta(model, val_loader, theta, use_tta=CFG["final_use_tta"])
        ranked.append({
            "theta_name": name,
            "theta": theta,
            "score": score_metric(met),
            "val_acc": safe_float(met.get("acc")),
            "val_f1": safe_float(met.get("f1_macro")),
            "estimated_bandit_value": float(est),
            "from_client_gid": int(gid),
        })
    ranked = sorted(ranked, key=lambda x: x["score"], reverse=True)
    return ranked

def equal_merge_metrics(m1, m2):
    out = {}
    keys = sorted(set(m1.keys()).union(set(m2.keys())))
    for k in keys:
        a, b = m1.get(k, np.nan), m2.get(k, np.nan)
        if np.isfinite(a) and np.isfinite(b):
            out[k] = float(np.mean([a, b]))
        elif np.isfinite(a):
            out[k] = float(a)
        elif np.isfinite(b):
            out[k] = float(b)
        else:
            out[k] = np.nan
    return out

client_theta_rows = []
for c in clients:
    arm, name, theta, val = best_theta_for_client(c)
    client_theta_rows.append({
        "client": f"client_{c['gid']}",
        "dataset": c["dataset"],
        "best_theta_arm": int(arm),
        "best_theta_name": name,
        "best_theta_str": theta_str(theta),
        "estimated_value": float(val),
        "pulls": int(c["theta_bandit"].counts.sum()),
    })
client_theta_df = pd.DataFrame(client_theta_rows)
print_table(client_theta_df, "Best RL-UCB-selected preprocessing preset per client", max_rows=20)
add_table_to_csv(client_theta_df, "best_rlucb_selected_preprocessing_per_client")

ds1_clients = [c for c in clients if c["dataset"] == "ds1"]
ds2_clients = [c for c in clients if c["dataset"] == "ds2"]

cand_ds1 = unique_theta_candidates(ds1_clients)
cand_ds2 = unique_theta_candidates(ds2_clients)

ranked_ds1 = rank_rlucb_candidates(global_model, val_loader_ds1, cand_ds1)
ranked_ds2 = rank_rlucb_candidates(global_model, val_loader_ds2, cand_ds2)

choice_ds1 = ranked_ds1[0]
choice_ds2 = ranked_ds2[0]

ranked_ds1_df = pd.DataFrame([{k: v for k, v in r.items() if k != "theta"} for r in ranked_ds1])
ranked_ds2_df = pd.DataFrame([{k: v for k, v in r.items() if k != "theta"} for r in ranked_ds2])

print_table(ranked_ds1_df, "Validation ranking of RL-UCB candidates — DS1", max_rows=10)
print_table(ranked_ds2_df, "Validation ranking of RL-UCB candidates — DS2", max_rows=10)
add_table_to_csv(ranked_ds1_df, "validation_ranking_rlucb_candidates_ds1")
add_table_to_csv(ranked_ds2_df, "validation_ranking_rlucb_candidates_ds2")

val_ds1, _, _ = evaluate_with_single_theta(global_model, val_loader_ds1, choice_ds1["theta"], use_tta=CFG["final_use_tta"])
test_ds1, y_ds1, p_ds1 = evaluate_with_single_theta(global_model, test_loader_ds1, choice_ds1["theta"], use_tta=CFG["final_use_tta"])

val_ds2, _, _ = evaluate_with_single_theta(global_model, val_loader_ds2, choice_ds2["theta"], use_tta=CFG["final_use_tta"])
test_ds2, y_ds2, p_ds2 = evaluate_with_single_theta(global_model, test_loader_ds2, choice_ds2["theta"], use_tta=CFG["final_use_tta"])

val_global = equal_merge_metrics(val_ds1, val_ds2)
test_global = equal_merge_metrics(test_ds1, test_ds2)

choice_df = pd.DataFrame([
    {
        "dataset": "ds1",
        "strategy": "validation_selected_single_after_rlucb_training",
        "theta_names": str([choice_ds1["theta_name"]]),
        "score": choice_ds1["score"],
        "val_acc": choice_ds1["val_acc"],
        "val_f1": choice_ds1["val_f1"],
    },
    {
        "dataset": "ds2",
        "strategy": "validation_selected_single_after_rlucb_training",
        "theta_names": str([choice_ds2["theta_name"]]),
        "score": choice_ds2["score"],
        "val_acc": choice_ds2["val_acc"],
        "val_f1": choice_ds2["val_f1"],
    },
])
print_table(choice_df, "Final preset strategy after RL-UCB training", max_rows=5)
add_table_to_csv(choice_df, "final_theta_strategy_choice_rlucb")

# ============================================================
# STEP 12: EXTENDED METRICS + CALIBRATION + ERROR ANALYSIS
# ============================================================
print("\n" + "=" * 118)
print("STEP 12: EXTENDED METRICS + CALIBRATION + ERROR ANALYSIS")
print("=" * 118)

def top_label_calibration(y_true, p_pred, n_bins=12):
    conf = np.max(p_pred, axis=1)
    pred = np.argmax(p_pred, axis=1)
    correct = (pred == y_true).astype(np.float32)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(conf, bins) - 1
    bin_ids = np.clip(bin_ids, 0, n_bins - 1)

    rows = []
    ece = 0.0
    mce = 0.0
    N = len(y_true)

    for b in range(n_bins):
        m = (bin_ids == b)
        if m.sum() == 0:
            rows.append({
                "bin_id": b,
                "bin_left": bins[b],
                "bin_right": bins[b + 1],
                "bin_confidence": np.nan,
                "bin_accuracy": np.nan,
                "bin_gap": np.nan,
                "bin_count": 0,
            })
            continue

        bc = float(conf[m].mean())
        ba = float(correct[m].mean())
        gap = abs(ba - bc)
        ece += (m.sum() / N) * gap
        mce = max(mce, gap)
        rows.append({
            "bin_id": b,
            "bin_left": bins[b],
            "bin_right": bins[b + 1],
            "bin_confidence": bc,
            "bin_accuracy": ba,
            "bin_gap": gap,
            "bin_count": int(m.sum()),
        })

    return pd.DataFrame(rows), float(ece), float(mce)

def multiclass_brier_score(y_true, p_pred, num_classes):
    y_oh = np.eye(num_classes)[y_true]
    return float(np.mean(np.sum((p_pred - y_oh) ** 2, axis=1)))

def classwise_confusion_metrics(y_true, p_pred):
    y_hat = np.argmax(p_pred, axis=1)
    cm = confusion_matrix(y_true, y_hat, labels=list(range(NUM_CLASSES)))
    total = cm.sum()

    rows = []
    for c, lab in enumerate(labels):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        tn = total - tp - fp - fn

        ppv = tp / max(1, (tp + fp))
        npv = tn / max(1, (tn + fn))
        sensitivity = tp / max(1, (tp + fn))
        specificity = tn / max(1, (tn + fp))
        fpr = fp / max(1, (fp + tn))
        fnr = fn / max(1, (fn + tp))
        jacc = tp / max(1, (tp + fp + fn))
        support = tp + fn
        prevalence = support / max(1, total)
        bal = 0.5 * (sensitivity + specificity)

        rows.append({
            "class_id": c,
            "class_name": lab,
            "support": int(support),
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
            "tn": int(tn),
            "prevalence": float(prevalence),
            "ppv": float(ppv),
            "npv": float(npv),
            "recall": float(sensitivity),
            "specificity": float(specificity),
            "fpr": float(fpr),
            "fnr": float(fnr),
            "jaccard": float(jacc),
            "balanced_acc": float(bal),
        })

    return cm, pd.DataFrame(rows)

def top_confusion_pairs(y_true, p_pred):
    y_hat = np.argmax(p_pred, axis=1)
    cm = confusion_matrix(y_true, y_hat, labels=list(range(NUM_CLASSES)))
    rows = []
    for i, lab_i in enumerate(labels):
        for j, lab_j in enumerate(labels):
            if i == j:
                continue
            rows.append({
                "true_class": lab_i,
                "pred_class": lab_j,
                "count": int(cm[i, j]),
            })
    return pd.DataFrame(rows).sort_values("count", ascending=False).reset_index(drop=True)

def wrong_right_confidence_table(y_true, p_pred):
    pred = np.argmax(p_pred, axis=1)
    conf = np.max(p_pred, axis=1)
    ok = pred == y_true
    return pd.DataFrame({
        "confidence": conf,
        "correct": ok.astype(int),
        "status": np.where(ok, "correct", "wrong"),
        "true_label": [id2label[int(x)] for x in y_true],
        "pred_label": [id2label[int(x)] for x in pred],
    })

def compute_extended_bundle(y_true, p_pred, dataset_name):
    y_hat = np.argmax(p_pred, axis=1)
    cm, class_df = classwise_confusion_metrics(y_true, p_pred)
    cal_df, ece, mce = top_label_calibration(y_true, p_pred, n_bins=CFG["calibration_bins"])
    conf_df = top_confusion_pairs(y_true, p_pred)
    conf_table = wrong_right_confidence_table(y_true, p_pred)

    scalar = {
        "dataset": dataset_name,
        "acc": float(accuracy_score(y_true, y_hat)),
        "balanced_acc": float(balanced_accuracy_score(y_true, y_hat)),
        "precision_macro": float(precision_score(y_true, y_hat, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_hat, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_hat, average="macro", zero_division=0)),
        "precision_weighted": float(precision_score(y_true, y_hat, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(y_true, y_hat, average="weighted", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_hat, average="weighted", zero_division=0)),
        "log_loss": float(log_loss(y_true, p_pred, labels=list(range(NUM_CLASSES)))),
        "mcc": float(matthews_corrcoef(y_true, y_hat)),
        "kappa": float(cohen_kappa_score(y_true, y_hat)),
        "jaccard_macro": float(jaccard_score(y_true, y_hat, average="macro")),
        "ppv_macro": float(class_df["ppv"].mean()),
        "npv_macro": float(class_df["npv"].mean()),
        "specificity_macro": float(class_df["specificity"].mean()),
        "fpr_macro": float(class_df["fpr"].mean()),
        "fnr_macro": float(class_df["fnr"].mean()),
        "ece": float(ece),
        "mce": float(mce),
        "brier_multi": float(multiclass_brier_score(y_true, p_pred, NUM_CLASSES)),
    }
    scalar.update(_auc_metrics(y_true, p_pred, NUM_CLASSES))
    return scalar, class_df, cal_df, conf_df, conf_table, cm

ext_ds1, class_df1, cal_df1, conf_pairs1, conf_table1, cm1 = compute_extended_bundle(y_ds1, p_ds1, "ds1_test")
ext_ds2, class_df2, cal_df2, conf_pairs2, conf_table2, cm2 = compute_extended_bundle(y_ds2, p_ds2, "ds2_test")

ext_df = pd.DataFrame([ext_ds1, ext_ds2])

print_table(ext_df, "Extended TEST metrics (DS1 vs DS2)", max_rows=5)
print_table(class_df1, "Classwise metrics — DS1 TEST", max_rows=10)
print_table(class_df2, "Classwise metrics — DS2 TEST", max_rows=10)
print_table(conf_pairs1.head(10), "Top confusion pairs — DS1 TEST", max_rows=10)
print_table(conf_pairs2.head(10), "Top confusion pairs — DS2 TEST", max_rows=10)
print_table(cal_df1, "Calibration bins — DS1 TEST", max_rows=15)
print_table(cal_df2, "Calibration bins — DS2 TEST", max_rows=15)

add_table_to_csv(ext_df, "extended_test_metrics_ds1_ds2")
add_table_to_csv(class_df1, "classwise_metrics_ds1_test")
add_table_to_csv(class_df2, "classwise_metrics_ds2_test")
add_table_to_csv(cal_df1, "calibration_bins_ds1_test")
add_table_to_csv(cal_df2, "calibration_bins_ds2_test")
add_table_to_csv(conf_pairs1, "top_confusions_ds1_test")
add_table_to_csv(conf_pairs2, "top_confusions_ds2_test")
add_table_to_csv(conf_table1, "confidence_correctness_ds1_test")
add_table_to_csv(conf_table2, "confidence_correctness_ds2_test")

test_ds1.update(ext_ds1)
test_ds2.update(ext_ds2)

def compact_metrics(m):
    keep = [
        "acc", "balanced_acc", "precision_macro", "recall_macro", "f1_macro",
        "precision_weighted", "recall_weighted", "f1_weighted", "log_loss",
        "auc_roc_macro_ovr", "mcc", "kappa", "ppv_macro", "npv_macro",
        "specificity_macro", "ece", "mce", "brier_multi", "loss_ce", "eval_time_s"
    ]
    out = {}
    for k in keep:
        if k in m and np.isfinite(m[k]):
            out[k] = float(m[k])
    return out

paper_df = pd.DataFrame([
    {"setting": "ARCF-Net RL-UCB Preset Selection", "split": "VAL",  "dataset": "ds1",          **compact_metrics(val_ds1)},
    {"setting": "ARCF-Net RL-UCB Preset Selection", "split": "VAL",  "dataset": "ds2",          **compact_metrics(val_ds2)},
    {"setting": "ARCF-Net RL-UCB Preset Selection", "split": "VAL",  "dataset": "global_equal", **compact_metrics(val_global)},
    {"setting": "ARCF-Net RL-UCB Preset Selection", "split": "TEST", "dataset": "ds1",          **compact_metrics(test_ds1)},
    {"setting": "ARCF-Net RL-UCB Preset Selection", "split": "TEST", "dataset": "ds2",          **compact_metrics(test_ds2)},
    {"setting": "ARCF-Net RL-UCB Preset Selection", "split": "TEST", "dataset": "global_equal", **compact_metrics(test_global)},
])

print_table(paper_df, "VAL + TEST tables", max_rows=10)
add_table_to_csv(paper_df, "paper_ready_metrics")

print("\nSelection summary:")
print(f"- Best round: {best_round_saved} | best_reward={best_reward:.4f}")
print(f"- DS1 final strategy: validation_selected_single_after_rlucb_training | names={[choice_ds1['theta_name']]}")
print(f"- DS2 final strategy: validation_selected_single_after_rlucb_training | names={[choice_ds2['theta_name']]}")

# ============================================================
# STEP 13: PREPROCESSING VALIDATION
# ============================================================
print("\n" + "=" * 118)
print("STEP 13: PREPROCESSING VALIDATION")
print("=" * 118)

@torch.no_grad()
def entropy_per_image(x01):
    gray = x01.mean(dim=1)
    B = gray.shape[0]
    ent = []
    for i in range(B):
        g = (gray[i].detach().cpu().numpy() * 255).astype(np.uint8)
        hist = np.bincount(g.flatten(), minlength=256).astype(np.float32)
        p = hist / np.clip(hist.sum(), 1, None)
        p = p[p > 0]
        ent.append(float(-(p * np.log2(p)).sum()))
    return np.array(ent)

@torch.no_grad()
def edge_energy(x01, kernel):
    kernel = kernel.to(device=x01.device, dtype=x01.dtype)
    gray = x01.mean(dim=1, keepdim=True)
    lap = F.conv2d(F.pad(gray, (1, 1, 1, 1), mode="reflect"), kernel).abs()
    return lap.mean(dim=(1, 2, 3)).detach().cpu().numpy()

@torch.no_grad()
def contrast_proxy(x01):
    gray = x01.mean(dim=1)
    return gray.std(dim=(1, 2)).detach().cpu().numpy()

@torch.no_grad()
def run_preproc_validation(frame, preproc, sample_n=400):
    n = min(sample_n, len(frame))
    if n <= 0:
        return pd.DataFrame(), pd.DataFrame()

    idx = np.random.choice(len(frame), size=n, replace=False)
    ds = MRIDataset(frame, indices=idx.tolist(), tfms=EVAL_TFMS, source_id=0)

    xs = []
    for i in range(len(ds)):
        x, _, _, _ = ds[i]
        xs.append(x)
    x = torch.stack(xs).to(DEVICE)

    x_after = preproc(x).clamp(0, 1)
    lap_kernel = preproc.lap if hasattr(preproc, "lap") else RACEFELCM().to(DEVICE).lap

    ee_before = edge_energy(x, lap_kernel)
    ee_after  = edge_energy(x_after, lap_kernel)
    ent_before = entropy_per_image(x)
    ent_after  = entropy_per_image(x_after)
    con_before = contrast_proxy(x)
    con_after  = contrast_proxy(x_after)

    dfm = pd.DataFrame({
        "edge_energy_before": ee_before,
        "edge_energy_after": ee_after,
        "entropy_before": ent_before,
        "entropy_after": ent_after,
        "contrast_before": con_before,
        "contrast_after": con_after,
        "edge_gain_ratio": (ee_after / np.clip(ee_before, 1e-9, None)),
        "entropy_delta": (ent_after - ent_before),
        "contrast_delta": (con_after - con_before),
    })
    summary = dfm.agg(["mean", "std", "min", "max"]).T.reset_index().rename(columns={"index": "metric"})
    return dfm, summary

preproc_ds1 = theta_to_module(choice_ds1["theta"]).to(DEVICE)
preproc_ds2 = theta_to_module(choice_ds2["theta"]).to(DEVICE)

preproc_df1, preproc_summary_df1 = run_preproc_validation(val1, preproc_ds1, CFG["preproc_val_sample_n"])
preproc_df2, preproc_summary_df2 = run_preproc_validation(val2, preproc_ds2, CFG["preproc_val_sample_n"])

print_table(preproc_summary_df1, "Preprocessing validation summary (DS1 VAL sample)", max_rows=15)
print_table(preproc_summary_df2, "Preprocessing validation summary (DS2 VAL sample)", max_rows=15)
add_table_to_csv(preproc_summary_df1, "preprocessing_validation_summary_ds1")
add_table_to_csv(preproc_summary_df2, "preprocessing_validation_summary_ds2")

# ============================================================
# STEP 14: SAVE CHECKPOINT + CSV
# ============================================================
print("\n" + "=" * 118)
print("STEP 14: SAVING CHECKPOINT + CSV")
print("=" * 118)

client_process_manifest = []
for c in clients:
    df_src = train1 if c["dataset"] == "ds1" else train2
    client_process_manifest.append({
        "gid": c["gid"],
        "local_id": c["local_id"],
        "dataset": c["dataset"],
        "source_id": c["source_id"],
        "n_train": c["n_train"],
        "n_tune": c["n_tune"],
        "n_val": c["n_val"],
        "train_indices": c["train_indices"],
        "tune_indices": c["tune_indices"],
        "val_indices": c["val_indices"],
        "train_paths": df_src.loc[c["train_indices"], "path"].tolist(),
        "tune_paths": df_src.loc[c["tune_indices"], "path"].tolist(),
        "val_paths": df_src.loc[c["val_indices"], "path"].tolist(),
        "class_counts": c["class_counts"],
        "theta_bandit_state": c["theta_bandit"].state_dict(),
        "preset_bank": c["preset_bank"],
    })

all_df = pd.DataFrame(ALL_ROWS)

checkpoint = {
    "method_info": METHOD_INFO,

    "state_dict": {k: v.detach().cpu() for k, v in global_model.state_dict().items()},
    "best_model_state_dict": {k: v.detach().cpu() for k, v in global_model.state_dict().items()},

    "config": CFG,
    "seed": SEED,
    "device_used": str(DEVICE),
    "backbone_name": "resnet50",
    "fusion_name": "CRAF",
    "preprocessing_name": "RACE-FELCM",

    "dataset1_raw_root": DS1_ROOT,
    "dataset2_root": DS2_ROOT,
    "labels": labels,
    "label2id": label2id,
    "id2label": id2label,
    "num_classes": NUM_CLASSES,
    "theta_fullforms": THETA_FULLFORMS,

    "full_dataset_df1": frame_to_manifest(df1),
    "full_dataset_df2": frame_to_manifest(df2),

    "split_train1": frame_to_manifest(train1),
    "split_val1": frame_to_manifest(val1),
    "split_test1": frame_to_manifest(test1),
    "split_train2": frame_to_manifest(train2),
    "split_val2": frame_to_manifest(val2),
    "split_test2": frame_to_manifest(test2),

    "fixed_clients_ds1": n_clients_ds1,
    "fixed_clients_ds2": n_clients_ds2,
    "clients_total": CLIENTS_TOTAL,
    "client_indices_ds1": client_indices_ds1,
    "client_indices_ds2": client_indices_ds2,
    "client_process_manifest": client_process_manifest,

    "preset_bank_ds1": PRESET_BANK_DS1,
    "preset_bank_ds2": PRESET_BANK_DS2,

    "best_round_saved": best_round_saved,
    "best_reward": best_reward,
    "history_global": glob_df.to_dict(orient="list"),
    "history_local": loc_df.to_dict(orient="list"),
    "total_training_time_s": t_total,

    "best_theta_bandit_states": best_theta_bandit_states,
    "best_rlucb_selected_per_client": client_theta_df.to_dict(orient="list"),

    "final_choice_ds1": {
        "strategy": "validation_selected_single_after_rlucb_training",
        "theta_names": [choice_ds1["theta_name"]],
        "theta_list": [choice_ds1["theta"]],
        "score": choice_ds1["score"],
        "val_acc": choice_ds1["val_acc"],
        "val_f1": choice_ds1["val_f1"],
    },
    "final_choice_ds2": {
        "strategy": "validation_selected_single_after_rlucb_training",
        "theta_names": [choice_ds2["theta_name"]],
        "theta_list": [choice_ds2["theta"]],
        "score": choice_ds2["score"],
        "val_acc": choice_ds2["val_acc"],
        "val_f1": choice_ds2["val_f1"],
    },

    "best_global_prototypes": None if best_global_prototypes is None else {
        "proto": best_global_prototypes["proto"],
        "mask": best_global_prototypes["mask"],
        "counts": best_global_prototypes["counts"],
    },

    "final_val_ds1": val_ds1,
    "final_val_ds2": val_ds2,
    "final_val_global": val_global,
    "final_test_ds1": test_ds1,
    "final_test_ds2": test_ds2,
    "final_test_global_equal": test_global,
    "paper_ready_metrics": paper_df.to_dict(orient="list"),
    "extended_test_metrics_ds1_ds2": ext_df.to_dict(orient="list"),
    "classwise_metrics_ds1_test": class_df1.to_dict(orient="list"),
    "classwise_metrics_ds2_test": class_df2.to_dict(orient="list"),
    "calibration_bins_ds1_test": cal_df1.to_dict(orient="list"),
    "calibration_bins_ds2_test": cal_df2.to_dict(orient="list"),
    "top_confusions_ds1_test": conf_pairs1.to_dict(orient="list"),
    "top_confusions_ds2_test": conf_pairs2.to_dict(orient="list"),
    "confidence_correctness_ds1_test": conf_table1.to_dict(orient="list"),
    "confidence_correctness_ds2_test": conf_table2.to_dict(orient="list"),
    "client_distribution_table": dist_df.to_dict(orient="list"),
    "global_round_metrics_table": glob_df.to_dict(orient="list"),
    "local_round_metrics_table": loc_df.to_dict(orient="list"),
    "all_tables_long_format": all_df.to_dict(orient="list"),

    "ds1_test_y_true": y_ds1,
    "ds1_test_p_pred": p_ds1,
    "ds1_test_y_hat": np.argmax(p_ds1, axis=1),
    "ds2_test_y_true": y_ds2,
    "ds2_test_p_pred": p_ds2,
    "ds2_test_y_hat": np.argmax(p_ds2, axis=1),

    "preprocessing_validation_ds1_raw": preproc_df1.to_dict(orient="list") if len(preproc_df1) else {},
    "preprocessing_validation_ds2_raw": preproc_df2.to_dict(orient="list") if len(preproc_df2) else {},
    "preprocessing_validation_summary_ds1": preproc_summary_df1.to_dict(orient="list") if len(preproc_summary_df1) else {},
    "preprocessing_validation_summary_ds2": preproc_summary_df2.to_dict(orient="list") if len(preproc_summary_df2) else {},
}

torch.save(checkpoint, MODEL_PATH)
print(f"Saved checkpoint: {MODEL_PATH}")

all_df.to_csv(CSV_PATH, index=False)
print(f"Saved CSV: {CSV_PATH}")

print("\nDONE")
print(f"Method: {METHOD_INFO['acronym']} = {METHOD_INFO['full_form']}")
print(f"Ablation: {METHOD_INFO['ablation_name']}")
print(f"Backbone: {METHOD_INFO['backbone_full_form']}")
print(f"Best round: {best_round_saved}")
print(f"Fixed clients => DS1={n_clients_ds1}, DS2={n_clients_ds2}, TOTAL={CLIENTS_TOTAL}")
print(f"Rounds completed: {CFG['rounds']}")
print(f"Global TEST acc: {safe_float(test_global.get('acc')):.4f}")
print(f"Global TEST f1_macro: {safe_float(test_global.get('f1_macro')):.4f}")
print(f"DS1 TEST acc: {safe_float(test_ds1.get('acc')):.4f}")
print(f"DS2 TEST acc: {safe_float(test_ds2.get('acc')):.4f}")
print(f"DS1 final strategy: validation_selected_single_after_rlucb_training | names={[choice_ds1['theta_name']]}")
print(f"DS2 final strategy: validation_selected_single_after_rlucb_training | names={[choice_ds2['theta_name']]}")
```

    ======================================================================================================================
    ARCF-Net ABLATION 4: RL-UCB PRESET SELECTION + FIXED CLIENTS + FULL PARTICIPATION + NO PLOTS
    ======================================================================================================================
    ENV: KAGGLE | DEVICE: cuda | torch=2.10.0+cu128
    ======================================================================================================================
    
    ======================================================================================================================
    STEP 0: ACCESS DATASETS
    ======================================================================================================================
    Dataset-1 RAW root detected:
      /kaggle/input/pmram-bangladeshi-brain-cancer-mri-dataset/PMRAM Bangladeshi Brain Cancer - MRI Dataset/PMRAM Bangladeshi Brain Cancer - MRI Dataset/Raw Data/Raw
    Dataset-2 root detected:
      /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset
    
    ======================================================================================================================
    STEP 1: BUILD DATA MANIFESTS
    ======================================================================================================================
    ds1_raw: 512Glioma -> glioma | 373 images
    ds1_raw: 512Meningioma -> meningioma | 363 images
    ds1_raw: 512Normal -> notumor | 396 images
    ds1_raw: 512Pituitary -> pituitary | 373 images
    ds2: glioma -> glioma | 1621 images
    ds2: meningioma -> meningioma | 1646 images
    ds2: notumor -> notumor | 2000 images
    ds2: pituitary -> pituitary | 1764 images
    
    Dataset-1 images: 1505
    label
    glioma        373
    meningioma    363
    notumor       396
    pituitary     373
    
    Dataset-2 images: 7031
    label
    glioma        1621
    meningioma    1646
    notumor       2000
    pituitary     1764
    
    ======================================================================================================================
    STEP 2: TRAIN / VAL / TEST SPLIT
    ======================================================================================================================
    DS1 TRAIN: 1053 | VAL: 226 | TEST: 226
    DS2 TRAIN: 4921 | VAL: 1055 | TEST: 1055
    
    ======================================================================================================================
    STEP 2.5: SANITY / LEAKAGE CHECKS
    ======================================================================================================================
    
    ----------------------------------------------------------------------------------------------------------------------
    Leakage / Sanity Summary — ds1
    ----------------------------------------------------------------------------------------------------------------------
     path_overlap_train_val  path_overlap_train_test  path_overlap_val_test  unique_paths_train  unique_paths_val  unique_paths_test  filename_overlap_train_val  filename_overlap_train_test  filename_overlap_val_test  subset_hash_train_val  subset_hash_train_test  subset_hash_val_test  subset_hash_n_train  subset_hash_n_val  subset_hash_n_test
                          0                        0                      0                1053               226                226                           0                            0                          0                      5                       5                     6                  298                222                 224
    
    ----------------------------------------------------------------------------------------------------------------------
    Leakage / Sanity Summary — ds2
    ----------------------------------------------------------------------------------------------------------------------
     path_overlap_train_val  path_overlap_train_test  path_overlap_val_test  unique_paths_train  unique_paths_val  unique_paths_test  filename_overlap_train_val  filename_overlap_train_test  filename_overlap_val_test  subset_hash_train_val  subset_hash_train_test  subset_hash_val_test  subset_hash_n_train  subset_hash_n_val  subset_hash_n_test
                          0                        0                      0                4921              1055               1055                           0                            0                          0                      0                       3                     4                  299                298                 299
    
    ======================================================================================================================
    STEP 3: RL-UCB BANDIT
    ======================================================================================================================
    
    ======================================================================================================================
    STEP 4: FIXED NON-IID CLIENT PARTITIONING
    ======================================================================================================================
    Fixed clients for DS1: 3
    Fixed clients for DS2: 3
    
    ======================================================================================================================
    STEP 5: DATA LOADERS
    ======================================================================================================================
    ds1 | client_0 | train=490 | tune=77 | val=67
    ds1 | client_1 | train=125 | tune=20 | val=18
    ds1 | client_2 | train=198 | tune=31 | val=27
    ds2 | client_3 | train=629 | tune=98 | val=86
    ds2 | client_4 | train=527 | tune=82 | val=72
    ds2 | client_5 | train=2653 | tune=412 | val=362
    
    ----------------------------------------------------------------------------------------------------------------------
    Client class distribution
    ----------------------------------------------------------------------------------------------------------------------
      client dataset  total_train  total_tune  total_val                                                                              preset_bank_names  glioma  meningioma  notumor  pituitary
    client_0     ds1          490          77         67 ['race_balanced', 'race_sharp', 'race_texture', 'race_robust', 'race_edge_plus', 'race_focus']     111          46      176        157
    client_1     ds1          125          20         18 ['race_balanced', 'race_sharp', 'race_texture', 'race_robust', 'race_edge_plus', 'race_focus']      75           4        8         38
    client_2     ds1          198          31         27 ['race_balanced', 'race_sharp', 'race_texture', 'race_robust', 'race_edge_plus', 'race_focus']      16         147       30          5
    client_3     ds2          629          98         86                                ['race_soft', 'race_balanced', 'race_robust', 'race_smoothmix']      12         197      416          4
    client_4     ds2          527          82         72                                ['race_soft', 'race_balanced', 'race_robust', 'race_smoothmix']     202           4      284         37
    client_5     ds2         2653         412        362                                ['race_soft', 'race_balanced', 'race_robust', 'race_smoothmix']     665         691      383        914
    Augmentation: ON
    Preprocessing: ON
    Total clients: 6
    
    ======================================================================================================================
    STEP 6: PREPROCESSING — RACE-FELCM
    ======================================================================================================================
    
    ======================================================================================================================
    STEP 7: MODEL — ResNet-50 + CRAF Fusion
    ======================================================================================================================
    Backbone: ResNet-50 | pretrained_loaded=True
    Total params: 25,790,855
    Trainable params: 2,282,823 (8.85%)
    
    ======================================================================================================================
    STEP 8: LOSSES + PROTOTYPE SHARING + EVAL HELPERS
    ======================================================================================================================
    
    ======================================================================================================================
    STEP 9: TUNE-AWARE RL-UCB PRESET SELECTION
    ======================================================================================================================
    
    ======================================================================================================================
    STEP 10: TRUE FEDERATED TRAINING
    ======================================================================================================================
    Clients => DS1=3, DS2=3, TOTAL=6
    Rounds: 15 | Local epochs: 2
    Augmentation ON: True
    Transfer backbone: ResNet-50
    Preprocessing: RACE-FELCM
    Fusion: CRAF
    FedProx μ=0.01 | Proto λ=0.12
    Tempered FedAvg exponent = 0.50
    Best-round masses => DS1=0.50, DS2=0.50, min-bonus=0.15
    
    ======================================================================================================================
    ROUND 1/15 | selected=[0, 1, 2, 3, 4, 5]
    ======================================================================================================================
    Client 0 (ds1) | train_acc=0.7265 | val_acc=0.9104 | val_f1=0.8236 | val_auc=0.9844 | reward=0.8453 | theta=race_focus (g=1.10, a=0.36, b=6.40, t=2.70, k=3, eg=0.17, mix=0.86)
    Client 1 (ds1) | train_acc=0.6400 | val_acc=0.9444 | val_f1=0.6522 | val_auc=nan | reward=0.7252 | theta=race_focus (g=1.10, a=0.36, b=6.40, t=2.70, k=3, eg=0.17, mix=0.86)
    Client 2 (ds1) | train_acc=0.6212 | val_acc=0.7037 | val_f1=0.6111 | val_auc=0.8926 | reward=0.6343 | theta=race_focus (g=1.10, a=0.36, b=6.40, t=2.70, k=3, eg=0.17, mix=0.86)
    Client 3 (ds2) | train_acc=0.8116 | val_acc=0.9186 | val_f1=0.7059 | val_auc=0.8572 | reward=0.7591 | theta=race_smoothmix (g=0.90, a=0.20, b=4.00, t=2.10, k=7, eg=0.07, mix=0.70)
    Client 4 (ds2) | train_acc=0.8216 | val_acc=0.9722 | val_f1=0.7376 | val_auc=nan | reward=0.7962 | theta=race_smoothmix (g=0.90, a=0.20, b=4.00, t=2.10, k=7, eg=0.07, mix=0.70)
    Client 5 (ds2) | train_acc=0.7712 | val_acc=0.8315 | val_f1=0.8271 | val_auc=0.9658 | reward=0.8282 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    GLOBAL VAL (Round 1) | global_acc=0.8655 | global_f1=0.7858 | ds1_acc=0.8661 | ds1_f1=0.7448 | ds2_acc=0.8654 | ds2_f1=0.7947 | reward=0.9100 | round_time=179.4s
    
    ======================================================================================================================
    ROUND 2/15 | selected=[0, 1, 2, 3, 4, 5]
    ======================================================================================================================
    Client 0 (ds1) | train_acc=0.8571 | val_acc=0.8657 | val_f1=0.8095 | val_auc=0.9750 | reward=0.8235 | theta=race_edge_plus (g=1.02, a=0.32, b=6.00, t=2.60, k=3, eg=0.15, mix=0.84)
    Client 1 (ds1) | train_acc=0.8600 | val_acc=0.8889 | val_f1=0.7058 | val_auc=nan | reward=0.7515 | theta=race_sharp (g=1.04, a=0.30, b=5.20, t=2.50, k=5, eg=0.12, mix=0.82)
    Client 2 (ds1) | train_acc=0.8232 | val_acc=0.9259 | val_f1=0.7381 | val_auc=0.9061 | reward=0.7851 | theta=race_edge_plus (g=1.02, a=0.32, b=6.00, t=2.60, k=3, eg=0.15, mix=0.84)
    Client 3 (ds2) | train_acc=0.9046 | val_acc=0.9070 | val_f1=0.4521 | val_auc=0.9224 | reward=0.5658 | theta=race_robust (g=1.08, a=0.22, b=4.50, t=2.80, k=5, eg=0.11, mix=0.76)
    Client 4 (ds2) | train_acc=0.9317 | val_acc=0.9861 | val_f1=0.9636 | val_auc=nan | reward=0.9693 | theta=race_robust (g=1.08, a=0.22, b=4.50, t=2.80, k=5, eg=0.11, mix=0.76)
    Client 5 (ds2) | train_acc=0.8326 | val_acc=0.8619 | val_f1=0.8624 | val_auc=0.9731 | reward=0.8623 | theta=race_smoothmix (g=0.90, a=0.20, b=4.00, t=2.10, k=7, eg=0.07, mix=0.70)
    GLOBAL VAL (Round 2) | global_acc=0.8861 | global_f1=0.8027 | ds1_acc=0.8839 | ds1_f1=0.7756 | ds2_acc=0.8865 | ds2_f1=0.8086 | reward=0.9358 | round_time=159.1s
    
    ======================================================================================================================
    ROUND 3/15 | selected=[0, 1, 2, 3, 4, 5]
    ======================================================================================================================
    Client 0 (ds1) | train_acc=0.8255 | val_acc=0.8806 | val_f1=0.8254 | val_auc=0.9892 | reward=0.8392 | theta=race_texture (g=0.92, a=0.34, b=5.80, t=2.30, k=7, eg=0.10, mix=0.80)
    Client 1 (ds1) | train_acc=0.7120 | val_acc=1.0000 | val_f1=1.0000 | val_auc=nan | reward=1.0000 | theta=race_edge_plus (g=1.02, a=0.32, b=6.00, t=2.60, k=3, eg=0.15, mix=0.84)
    Client 2 (ds1) | train_acc=0.8359 | val_acc=0.8148 | val_f1=0.7876 | val_auc=0.9711 | reward=0.7944 | theta=race_texture (g=0.92, a=0.34, b=5.80, t=2.30, k=7, eg=0.10, mix=0.80)
    Client 3 (ds2) | train_acc=0.8768 | val_acc=0.9535 | val_f1=0.7250 | val_auc=0.9700 | reward=0.7821 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    Client 4 (ds2) | train_acc=0.9108 | val_acc=1.0000 | val_f1=1.0000 | val_auc=nan | reward=1.0000 | theta=race_soft (g=0.95, a=0.18, b=3.80, t=2.20, k=3, eg=0.08, mix=0.72)
    Client 5 (ds2) | train_acc=0.8632 | val_acc=0.9475 | val_f1=0.9478 | val_auc=0.9918 | reward=0.9477 | theta=race_soft (g=0.95, a=0.18, b=3.80, t=2.20, k=3, eg=0.08, mix=0.72)
    GLOBAL VAL (Round 3) | global_acc=0.9430 | global_f1=0.9051 | ds1_acc=0.8839 | ds1_f1=0.8444 | ds2_acc=0.9558 | ds2_f1=0.9181 | reward=1.0190 | round_time=208.3s
    
    ======================================================================================================================
    ROUND 4/15 | selected=[0, 1, 2, 3, 4, 5]
    ======================================================================================================================
    Client 0 (ds1) | train_acc=0.9041 | val_acc=0.8806 | val_f1=0.8423 | val_auc=0.9864 | reward=0.8518 | theta=race_sharp (g=1.04, a=0.30, b=5.20, t=2.50, k=5, eg=0.12, mix=0.82)
    Client 1 (ds1) | train_acc=0.8640 | val_acc=1.0000 | val_f1=1.0000 | val_auc=nan | reward=1.0000 | theta=race_texture (g=0.92, a=0.34, b=5.80, t=2.30, k=7, eg=0.10, mix=0.80)
    Client 2 (ds1) | train_acc=0.9217 | val_acc=0.8889 | val_f1=0.6968 | val_auc=0.9882 | reward=0.7449 | theta=race_sharp (g=1.04, a=0.30, b=5.20, t=2.50, k=5, eg=0.12, mix=0.82)
    Client 3 (ds2) | train_acc=0.9428 | val_acc=0.9651 | val_f1=0.4842 | val_auc=0.9978 | reward=0.6044 | theta=race_soft (g=0.95, a=0.18, b=3.80, t=2.20, k=3, eg=0.08, mix=0.72)
    Client 4 (ds2) | train_acc=0.9412 | val_acc=1.0000 | val_f1=1.0000 | val_auc=nan | reward=1.0000 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    Client 5 (ds2) | train_acc=0.9133 | val_acc=0.9586 | val_f1=0.9586 | val_auc=0.9927 | reward=0.9586 | theta=race_robust (g=1.08, a=0.22, b=4.50, t=2.80, k=5, eg=0.11, mix=0.76)
    GLOBAL VAL (Round 4) | global_acc=0.9541 | global_f1=0.8764 | ds1_acc=0.9018 | ds1_f1=0.8326 | ds2_acc=0.9654 | ds2_f1=0.8858 | reward=1.0053 | round_time=175.1s
    
    ======================================================================================================================
    ROUND 5/15 | selected=[0, 1, 2, 3, 4, 5]
    ======================================================================================================================
    Client 0 (ds1) | train_acc=0.9561 | val_acc=0.9254 | val_f1=0.9180 | val_auc=0.9975 | reward=0.9198 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    Client 1 (ds1) | train_acc=0.9440 | val_acc=0.9444 | val_f1=0.8730 | val_auc=nan | reward=0.8909 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    Client 2 (ds1) | train_acc=0.9697 | val_acc=0.8889 | val_f1=0.6968 | val_auc=0.9289 | reward=0.7449 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    Client 3 (ds2) | train_acc=0.9650 | val_acc=0.9884 | val_f1=0.7455 | val_auc=1.0000 | reward=0.8062 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    Client 4 (ds2) | train_acc=0.9592 | val_acc=0.9861 | val_f1=0.7455 | val_auc=nan | reward=0.8056 | theta=race_soft (g=0.95, a=0.18, b=3.80, t=2.20, k=3, eg=0.08, mix=0.72)
    Client 5 (ds2) | train_acc=0.9329 | val_acc=0.9724 | val_f1=0.9723 | val_auc=0.9974 | reward=0.9723 | theta=race_robust (g=1.08, a=0.22, b=4.50, t=2.80, k=5, eg=0.11, mix=0.76)
    GLOBAL VAL (Round 5) | global_acc=0.9668 | global_f1=0.8953 | ds1_acc=0.9196 | ds1_f1=0.8574 | ds2_acc=0.9769 | ds2_f1=0.9034 | reward=1.0283 | round_time=176.4s
    
    ======================================================================================================================
    ROUND 6/15 | selected=[0, 1, 2, 3, 4, 5]
    ======================================================================================================================
    Client 0 (ds1) | train_acc=0.9316 | val_acc=0.9403 | val_f1=0.9058 | val_auc=0.9975 | reward=0.9144 | theta=race_robust (g=1.08, a=0.22, b=4.50, t=2.80, k=5, eg=0.11, mix=0.76)
    Client 1 (ds1) | train_acc=0.9520 | val_acc=0.9444 | val_f1=0.8730 | val_auc=nan | reward=0.8909 | theta=race_robust (g=1.08, a=0.22, b=4.50, t=2.80, k=5, eg=0.11, mix=0.76)
    Client 2 (ds1) | train_acc=0.9773 | val_acc=0.9259 | val_f1=0.7381 | val_auc=0.9261 | reward=0.7851 | theta=race_robust (g=1.08, a=0.22, b=4.50, t=2.80, k=5, eg=0.11, mix=0.76)
    Client 3 (ds2) | train_acc=0.9746 | val_acc=0.9767 | val_f1=0.4933 | val_auc=1.0000 | reward=0.6141 | theta=race_smoothmix (g=0.90, a=0.20, b=4.00, t=2.10, k=7, eg=0.07, mix=0.70)
    Client 4 (ds2) | train_acc=0.9801 | val_acc=1.0000 | val_f1=1.0000 | val_auc=nan | reward=1.0000 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    Client 5 (ds2) | train_acc=0.9406 | val_acc=0.9613 | val_f1=0.9617 | val_auc=0.9947 | reward=0.9616 | theta=race_soft (g=0.95, a=0.18, b=3.80, t=2.20, k=3, eg=0.08, mix=0.72)
    GLOBAL VAL (Round 6) | global_acc=0.9636 | global_f1=0.8843 | ds1_acc=0.9375 | ds1_f1=0.8601 | ds2_acc=0.9692 | ds2_f1=0.8895 | reward=1.0264 | round_time=176.4s
    
    ======================================================================================================================
    ROUND 7/15 | selected=[0, 1, 2, 3, 4, 5]
    ======================================================================================================================
    Client 0 (ds1) | train_acc=0.9582 | val_acc=0.9254 | val_f1=0.8927 | val_auc=0.9931 | reward=0.9009 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    Client 1 (ds1) | train_acc=0.9400 | val_acc=1.0000 | val_f1=1.0000 | val_auc=nan | reward=1.0000 | theta=race_edge_plus (g=1.02, a=0.32, b=6.00, t=2.60, k=3, eg=0.15, mix=0.84)
    Client 2 (ds1) | train_acc=0.9747 | val_acc=0.9630 | val_f1=0.9106 | val_auc=0.9561 | reward=0.9237 | theta=race_texture (g=0.92, a=0.34, b=5.80, t=2.30, k=7, eg=0.10, mix=0.80)
    Client 3 (ds2) | train_acc=0.9809 | val_acc=0.9767 | val_f1=0.9077 | val_auc=0.9961 | reward=0.9249 | theta=race_soft (g=0.95, a=0.18, b=3.80, t=2.20, k=3, eg=0.08, mix=0.72)
    Client 4 (ds2) | train_acc=0.9915 | val_acc=0.9861 | val_f1=0.7455 | val_auc=nan | reward=0.8056 | theta=race_robust (g=1.08, a=0.22, b=4.50, t=2.80, k=5, eg=0.11, mix=0.76)
    Client 5 (ds2) | train_acc=0.9516 | val_acc=0.9724 | val_f1=0.9722 | val_auc=0.9980 | reward=0.9722 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    GLOBAL VAL (Round 7) | global_acc=0.9699 | global_f1=0.9273 | ds1_acc=0.9464 | ds1_f1=0.9143 | ds2_acc=0.9750 | ds2_f1=0.9301 | reward=1.0702 | round_time=174.5s
    
    ======================================================================================================================
    ROUND 8/15 | selected=[0, 1, 2, 3, 4, 5]
    ======================================================================================================================
    Client 0 (ds1) | train_acc=0.9765 | val_acc=0.9701 | val_f1=0.9499 | val_auc=0.9980 | reward=0.9549 | theta=race_robust (g=1.08, a=0.22, b=4.50, t=2.80, k=5, eg=0.11, mix=0.76)
    Client 1 (ds1) | train_acc=0.9360 | val_acc=0.9444 | val_f1=0.7273 | val_auc=nan | reward=0.7816 | theta=race_texture (g=0.92, a=0.34, b=5.80, t=2.30, k=7, eg=0.10, mix=0.80)
    Client 2 (ds1) | train_acc=0.9672 | val_acc=0.9630 | val_f1=0.9106 | val_auc=0.9846 | reward=0.9237 | theta=race_edge_plus (g=1.02, a=0.32, b=6.00, t=2.60, k=3, eg=0.15, mix=0.84)
    Client 3 (ds2) | train_acc=0.9873 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | reward=1.0000 | theta=race_robust (g=1.08, a=0.22, b=4.50, t=2.80, k=5, eg=0.11, mix=0.76)
    Client 4 (ds2) | train_acc=0.9649 | val_acc=0.9722 | val_f1=0.9321 | val_auc=nan | reward=0.9421 | theta=race_smoothmix (g=0.90, a=0.20, b=4.00, t=2.10, k=7, eg=0.07, mix=0.70)
    Client 5 (ds2) | train_acc=0.9574 | val_acc=0.9834 | val_f1=0.9838 | val_auc=0.9983 | reward=0.9837 | theta=race_smoothmix (g=0.90, a=0.20, b=4.00, t=2.10, k=7, eg=0.07, mix=0.70)
    GLOBAL VAL (Round 8) | global_acc=0.9810 | global_f1=0.9661 | ds1_acc=0.9643 | ds1_f1=0.9046 | ds2_acc=0.9846 | ds2_f1=0.9793 | reward=1.0880 | round_time=175.7s
    
    ======================================================================================================================
    ROUND 9/15 | selected=[0, 1, 2, 3, 4, 5]
    ======================================================================================================================
    Client 0 (ds1) | train_acc=0.9643 | val_acc=0.9552 | val_f1=0.9462 | val_auc=0.9968 | reward=0.9485 | theta=race_sharp (g=1.04, a=0.30, b=5.20, t=2.50, k=5, eg=0.12, mix=0.82)
    Client 1 (ds1) | train_acc=0.9560 | val_acc=0.8889 | val_f1=0.9190 | val_auc=nan | reward=0.9115 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    Client 2 (ds1) | train_acc=0.9874 | val_acc=0.9259 | val_f1=0.8690 | val_auc=0.9837 | reward=0.8833 | theta=race_robust (g=1.08, a=0.22, b=4.50, t=2.80, k=5, eg=0.11, mix=0.76)
    Client 3 (ds2) | train_acc=0.9801 | val_acc=0.9419 | val_f1=0.6021 | val_auc=0.9745 | reward=0.6870 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    Client 4 (ds2) | train_acc=0.9858 | val_acc=0.9861 | val_f1=0.9636 | val_auc=nan | reward=0.9693 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    Client 5 (ds2) | train_acc=0.9570 | val_acc=0.9641 | val_f1=0.9625 | val_auc=0.9946 | reward=0.9629 | theta=race_robust (g=1.08, a=0.22, b=4.50, t=2.80, k=5, eg=0.11, mix=0.76)
    GLOBAL VAL (Round 9) | global_acc=0.9589 | global_f1=0.9067 | ds1_acc=0.9375 | ds1_f1=0.9233 | ds2_acc=0.9635 | ds2_f1=0.9031 | reward=1.0602 | round_time=174.4s
    
    ======================================================================================================================
    ROUND 10/15 | selected=[0, 1, 2, 3, 4, 5]
    ======================================================================================================================
    Client 0 (ds1) | train_acc=0.9694 | val_acc=0.9552 | val_f1=0.9316 | val_auc=0.9980 | reward=0.9375 | theta=race_focus (g=1.10, a=0.36, b=6.40, t=2.70, k=3, eg=0.17, mix=0.86)
    Client 1 (ds1) | train_acc=0.9640 | val_acc=1.0000 | val_f1=1.0000 | val_auc=nan | reward=1.0000 | theta=race_robust (g=1.08, a=0.22, b=4.50, t=2.80, k=5, eg=0.11, mix=0.76)
    Client 2 (ds1) | train_acc=0.9798 | val_acc=0.9259 | val_f1=0.8764 | val_auc=0.9864 | reward=0.8888 | theta=race_sharp (g=1.04, a=0.30, b=5.20, t=2.50, k=5, eg=0.12, mix=0.82)
    Client 3 (ds2) | train_acc=0.9897 | val_acc=0.9884 | val_f1=0.9932 | val_auc=1.0000 | reward=0.9920 | theta=race_robust (g=1.08, a=0.22, b=4.50, t=2.80, k=5, eg=0.11, mix=0.76)
    Client 4 (ds2) | train_acc=0.9896 | val_acc=0.9861 | val_f1=0.7455 | val_auc=nan | reward=0.8056 | theta=race_soft (g=0.95, a=0.18, b=3.80, t=2.20, k=3, eg=0.08, mix=0.72)
    Client 5 (ds2) | train_acc=0.9666 | val_acc=0.9751 | val_f1=0.9757 | val_auc=0.9971 | reward=0.9755 | theta=race_soft (g=0.95, a=0.18, b=3.80, t=2.20, k=3, eg=0.08, mix=0.72)
    GLOBAL VAL (Round 10) | global_acc=0.9747 | global_f1=0.9436 | ds1_acc=0.9554 | ds1_f1=0.9293 | ds2_acc=0.9788 | ds2_f1=0.9467 | reward=1.0856 | round_time=174.6s
    
    ======================================================================================================================
    ROUND 11/15 | selected=[0, 1, 2, 3, 4, 5]
    ======================================================================================================================
    Client 0 (ds1) | train_acc=0.9724 | val_acc=0.9851 | val_f1=0.9722 | val_auc=1.0000 | reward=0.9754 | theta=race_texture (g=0.92, a=0.34, b=5.80, t=2.30, k=7, eg=0.10, mix=0.80)
    Client 1 (ds1) | train_acc=0.9600 | val_acc=0.9444 | val_f1=0.7273 | val_auc=nan | reward=0.7816 | theta=race_sharp (g=1.04, a=0.30, b=5.20, t=2.50, k=5, eg=0.12, mix=0.82)
    Client 2 (ds1) | train_acc=0.9874 | val_acc=0.9630 | val_f1=0.9106 | val_auc=0.9864 | reward=0.9237 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    Client 3 (ds2) | train_acc=0.9809 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | reward=1.0000 | theta=race_soft (g=0.95, a=0.18, b=3.80, t=2.20, k=3, eg=0.08, mix=0.72)
    Client 4 (ds2) | train_acc=0.9896 | val_acc=0.9861 | val_f1=0.7455 | val_auc=nan | reward=0.8056 | theta=race_robust (g=1.08, a=0.22, b=4.50, t=2.80, k=5, eg=0.11, mix=0.76)
    Client 5 (ds2) | train_acc=0.9608 | val_acc=0.9834 | val_f1=0.9841 | val_auc=0.9945 | reward=0.9840 | theta=race_smoothmix (g=0.90, a=0.20, b=4.00, t=2.10, k=7, eg=0.07, mix=0.70)
    GLOBAL VAL (Round 11) | global_acc=0.9842 | global_f1=0.9474 | ds1_acc=0.9732 | ds1_f1=0.9180 | ds2_acc=0.9865 | ds2_f1=0.9537 | reward=1.0866 | round_time=177.1s
    
    ======================================================================================================================
    ROUND 12/15 | selected=[0, 1, 2, 3, 4, 5]
    ======================================================================================================================
    Client 0 (ds1) | train_acc=0.9673 | val_acc=0.9552 | val_f1=0.9464 | val_auc=0.9934 | reward=0.9486 | theta=race_edge_plus (g=1.02, a=0.32, b=6.00, t=2.60, k=3, eg=0.15, mix=0.84)
    Client 1 (ds1) | train_acc=0.9560 | val_acc=1.0000 | val_f1=1.0000 | val_auc=nan | reward=1.0000 | theta=race_focus (g=1.10, a=0.36, b=6.40, t=2.70, k=3, eg=0.17, mix=0.86)
    Client 2 (ds1) | train_acc=0.9823 | val_acc=0.9630 | val_f1=0.9658 | val_auc=1.0000 | reward=0.9651 | theta=race_focus (g=1.10, a=0.36, b=6.40, t=2.70, k=3, eg=0.17, mix=0.86)
    Client 3 (ds2) | train_acc=0.9849 | val_acc=0.9767 | val_f1=0.9077 | val_auc=1.0000 | reward=0.9249 | theta=race_smoothmix (g=0.90, a=0.20, b=4.00, t=2.10, k=7, eg=0.07, mix=0.70)
    Client 4 (ds2) | train_acc=0.9772 | val_acc=1.0000 | val_f1=1.0000 | val_auc=nan | reward=1.0000 | theta=race_smoothmix (g=0.90, a=0.20, b=4.00, t=2.10, k=7, eg=0.07, mix=0.70)
    Client 5 (ds2) | train_acc=0.9659 | val_acc=0.9724 | val_f1=0.9727 | val_auc=0.9986 | reward=0.9726 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    GLOBAL VAL (Round 12) | global_acc=0.9747 | global_f1=0.9646 | ds1_acc=0.9643 | ds1_f1=0.9597 | ds2_acc=0.9769 | ds2_f1=0.9657 | reward=1.1088 | round_time=177.7s
    
    ======================================================================================================================
    ROUND 13/15 | selected=[0, 1, 2, 3, 4, 5]
    ======================================================================================================================
    Client 0 (ds1) | train_acc=0.9622 | val_acc=0.9552 | val_f1=0.9316 | val_auc=0.9945 | reward=0.9375 | theta=race_robust (g=1.08, a=0.22, b=4.50, t=2.80, k=5, eg=0.11, mix=0.76)
    Client 1 (ds1) | train_acc=0.9720 | val_acc=1.0000 | val_f1=1.0000 | val_auc=nan | reward=1.0000 | theta=race_edge_plus (g=1.02, a=0.32, b=6.00, t=2.60, k=3, eg=0.15, mix=0.84)
    Client 2 (ds1) | train_acc=0.9874 | val_acc=0.9630 | val_f1=0.9436 | val_auc=0.9914 | reward=0.9484 | theta=race_texture (g=0.92, a=0.34, b=5.80, t=2.30, k=7, eg=0.10, mix=0.80)
    Client 3 (ds2) | train_acc=0.9905 | val_acc=0.9651 | val_f1=0.7320 | val_auc=0.9995 | reward=0.7903 | theta=race_robust (g=1.08, a=0.22, b=4.50, t=2.80, k=5, eg=0.11, mix=0.76)
    Client 4 (ds2) | train_acc=0.9839 | val_acc=1.0000 | val_f1=1.0000 | val_auc=nan | reward=1.0000 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    Client 5 (ds2) | train_acc=0.9700 | val_acc=0.9862 | val_f1=0.9865 | val_auc=0.9996 | reward=0.9864 | theta=race_robust (g=1.08, a=0.22, b=4.50, t=2.80, k=5, eg=0.11, mix=0.76)
    GLOBAL VAL (Round 13) | global_acc=0.9810 | global_f1=0.9461 | ds1_acc=0.9643 | ds1_f1=0.9455 | ds2_acc=0.9846 | ds2_f1=0.9463 | reward=1.0955 | round_time=177.0s
    
    ======================================================================================================================
    ROUND 14/15 | selected=[0, 1, 2, 3, 4, 5]
    ======================================================================================================================
    Client 0 (ds1) | train_acc=0.9837 | val_acc=0.9701 | val_f1=0.9616 | val_auc=0.9894 | reward=0.9637 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    Client 1 (ds1) | train_acc=0.9640 | val_acc=1.0000 | val_f1=1.0000 | val_auc=nan | reward=1.0000 | theta=race_robust (g=1.08, a=0.22, b=4.50, t=2.80, k=5, eg=0.11, mix=0.76)
    Client 2 (ds1) | train_acc=0.9848 | val_acc=0.9630 | val_f1=0.9436 | val_auc=1.0000 | reward=0.9484 | theta=race_edge_plus (g=1.02, a=0.32, b=6.00, t=2.60, k=3, eg=0.15, mix=0.84)
    Client 3 (ds2) | train_acc=0.9873 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | reward=1.0000 | theta=race_soft (g=0.95, a=0.18, b=3.80, t=2.20, k=3, eg=0.08, mix=0.72)
    Client 4 (ds2) | train_acc=0.9896 | val_acc=0.9861 | val_f1=0.9636 | val_auc=nan | reward=0.9693 | theta=race_smoothmix (g=0.90, a=0.20, b=4.00, t=2.10, k=7, eg=0.07, mix=0.70)
    Client 5 (ds2) | train_acc=0.9791 | val_acc=0.9696 | val_f1=0.9691 | val_auc=0.9991 | reward=0.9692 | theta=race_soft (g=0.95, a=0.18, b=3.80, t=2.20, k=3, eg=0.08, mix=0.72)
    GLOBAL VAL (Round 14) | global_acc=0.9763 | global_f1=0.9717 | ds1_acc=0.9732 | ds1_f1=0.9634 | ds2_acc=0.9769 | ds2_f1=0.9735 | reward=1.1150 | round_time=174.3s
    
    ======================================================================================================================
    ROUND 15/15 | selected=[0, 1, 2, 3, 4, 5]
    ======================================================================================================================
    Client 0 (ds1) | train_acc=0.9724 | val_acc=0.9851 | val_f1=0.9750 | val_auc=0.9967 | reward=0.9775 | theta=race_texture (g=0.92, a=0.34, b=5.80, t=2.30, k=7, eg=0.10, mix=0.80)
    Client 1 (ds1) | train_acc=0.9800 | val_acc=1.0000 | val_f1=1.0000 | val_auc=nan | reward=1.0000 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    Client 2 (ds1) | train_acc=0.9848 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | reward=1.0000 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    Client 3 (ds2) | train_acc=0.9944 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | reward=1.0000 | theta=race_smoothmix (g=0.90, a=0.20, b=4.00, t=2.10, k=7, eg=0.07, mix=0.70)
    Client 4 (ds2) | train_acc=0.9858 | val_acc=1.0000 | val_f1=1.0000 | val_auc=nan | reward=1.0000 | theta=race_soft (g=0.95, a=0.18, b=3.80, t=2.20, k=3, eg=0.08, mix=0.72)
    Client 5 (ds2) | train_acc=0.9715 | val_acc=0.9724 | val_f1=0.9728 | val_auc=0.9994 | reward=0.9727 | theta=race_smoothmix (g=0.90, a=0.20, b=4.00, t=2.10, k=7, eg=0.07, mix=0.70)
    GLOBAL VAL (Round 15) | global_acc=0.9826 | global_f1=0.9818 | ds1_acc=0.9911 | ds1_f1=0.9850 | ds2_acc=0.9808 | ds2_f1=0.9810 | reward=1.1309 | round_time=173.7s
    
    ======================================================================================================================
    TRAINING COMPLETE | total_time=2654.4s | best_round=15 | best_reward=1.1309
    ======================================================================================================================
    
    ----------------------------------------------------------------------------------------------------------------------
    GLOBAL per-round metrics
    ----------------------------------------------------------------------------------------------------------------------
     round  round_time_s  n_selected_clients  active_fraction  global_reward  global_acc  global_f1_macro  global_precision_macro  global_recall_macro  global_log_loss  global_loss_ce  global_eval_time_s  ds1_acc  ds1_f1_macro  ds1_log_loss  ds2_acc  ds2_f1_macro  ds2_log_loss
         1    179.391993                   6              1.0       0.909998    0.865506         0.785824                0.798570             0.788461         0.349386        0.343712            2.627037 0.866071      0.744802      0.410524 0.865385      0.794659      0.336218
         2    159.064313                   6              1.0       0.935771    0.886076         0.802716                0.802221             0.806043         0.309533        0.309122            2.344108 0.883929      0.775604      0.343227 0.886538      0.808555      0.302276
         3    208.323706                   6              1.0       1.019036    0.943038         0.905067                0.907610             0.907786         0.196232        0.195132            2.345280 0.883929      0.844353      0.334492 0.955769      0.918144      0.166454
         4    175.143461                   6              1.0       1.005273    0.954114         0.876399                0.878899             0.876517         0.152407        0.152511            2.336093 0.901786      0.832551      0.233416 0.965385      0.885844      0.134959
         5    176.389421                   6              1.0       1.028333    0.966772         0.895254                0.894931             0.898373         0.115242        0.116372            2.364873 0.919643      0.857439      0.244940 0.976923      0.903399      0.087307
         6    176.359953                   6              1.0       1.026368    0.963608         0.884309                0.888996             0.883537         0.128676        0.134786            2.340988 0.937500      0.860101      0.179506 0.969231      0.889523      0.117728
         7    174.528328                   6              1.0       1.070168    0.969937         0.927311                0.927389             0.938416         0.120026        0.112826            2.332615 0.946429      0.914262      0.206225 0.975000      0.930122      0.101460
         8    175.667086                   6              1.0       1.088004    0.981013         0.966054                0.966853             0.970248         0.092236        0.094745            2.332556 0.964286      0.904610      0.124394 0.984615      0.979288      0.085310
         9    174.375984                   6              1.0       1.060223    0.958861         0.906652                0.905786             0.923219         0.150644        0.140925            2.379077 0.937500      0.923264      0.250124 0.963462      0.903074      0.129217
        10    174.628890                   6              1.0       1.085629    0.974684         0.943604                0.947520             0.942267         0.090921        0.086874            2.340040 0.955357      0.929268      0.121748 0.978846      0.946691      0.084281
        11    177.068458                   6              1.0       1.086611    0.984177         0.947373                0.954778             0.942394         0.077949        0.081378            2.331727 0.973214      0.917962      0.070499 0.986538      0.953707      0.079553
        12    177.737629                   6              1.0       1.108812    0.974684         0.964637                0.957747             0.978103         0.097477        0.092103            2.341083 0.964286      0.959722      0.148525 0.976923      0.965696      0.086482
        13    176.987143                   6              1.0       1.095541    0.981013         0.946129                0.944576             0.948722         0.081567        0.076977            2.357725 0.964286      0.945468      0.147136 0.984615      0.946271      0.067445
        14    174.294980                   6              1.0       1.114979    0.976266         0.971690                0.969834             0.975857         0.095232        0.090242            2.328169 0.973214      0.963413      0.136126 0.976923      0.973473      0.086424
        15    173.736511                   6              1.0       1.130901    0.982595         0.981751                0.980159             0.983742         0.070695        0.068820            2.333648 0.991071      0.985018      0.069781 0.980769      0.981047      0.070892
    
    ----------------------------------------------------------------------------------------------------------------------
    LOCAL per-client per-round metrics
    ----------------------------------------------------------------------------------------------------------------------
     round   client dataset  selected  theta_arm     theta_name                                                theta_str  gamma_power  alpha_contrast_weight  beta_contrast_sharpness  tau_clip  k_blur_kernel_size  edge_gain  blend_mix  train_loss  train_ce_loss  train_proto_loss  train_acc  train_time_s  val_size  val_loss_ce  val_acc  val_precision_macro  val_recall_macro  val_f1_macro  val_precision_weighted  val_recall_weighted  val_f1_weighted  val_log_loss  val_eval_time_s  val_auc_roc_macro_ovr  val_auc_class_0  val_auc_class_1  val_auc_class_2  val_auc_class_3  val_fusion_gate_mean_raw  val_fusion_gate_mean_enh  val_fusion_gate_mean_res  val_fusion_gate_entropy   reward  bandit_counts_total  bandit_value_selected
         1 client_0     ds1         1          5     race_focus (g=1.10, a=0.36, b=6.40, t=2.70, k=3, eg=0.17, mix=0.86)         1.10                   0.36                      6.4       2.7                   3       0.17       0.86    0.478106       0.414504          0.000000   0.726531           NaN        67     0.253246 0.910448             0.933532          0.800000      0.823562                0.916844             0.910448         0.896716      0.290163         1.344369               0.984431         0.979487         0.969945         0.990310         0.997980                  0.310161                  0.210281                  0.479558                 1.448798 0.845284                    1               0.845284
         1 client_1     ds1         1          5     race_focus (g=1.10, a=0.36, b=6.40, t=2.70, k=3, eg=0.17, mix=0.86)         1.10                   0.36                      6.4       2.7                   3       0.17       0.86    0.595631       0.571316          0.000000   0.640000           NaN        18     0.472973 0.944444             0.638889          0.666667      0.652174                0.893519             0.944444         0.917874      0.413080         0.860378                    NaN         1.000000              NaN         1.000000         1.000000                  0.009126                  0.601282                  0.389593                 0.957915 0.725242                    1               0.725242
         1 client_2     ds1         1          5     race_focus (g=1.10, a=0.36, b=6.40, t=2.70, k=3, eg=0.17, mix=0.86)         1.10                   0.36                      6.4       2.7                   3       0.17       0.86    0.580098       0.545946          0.000000   0.621212           NaN        27     0.705500 0.703704             0.593750          0.675000      0.611111                0.759259             0.703704         0.711934      0.707494         1.279062               0.892640         0.760000         0.821429         0.989130         1.000000                  0.483722                  0.084594                  0.431684                 1.322149 0.634259                    1               0.634259
         1 client_3     ds2         1          3 race_smoothmix (g=0.90, a=0.20, b=4.00, t=2.10, k=7, eg=0.07, mix=0.70)         0.90                   0.20                      4.0       2.1                   7       0.07       0.70    0.360297       0.298384          0.000000   0.811606           NaN        86     0.287305 0.918605             0.699242          0.713938      0.705905                0.911945             0.918605         0.914310      0.301321         1.012314               0.857160         0.529412         0.940364         0.958863         1.000000                  0.721652                  0.181896                  0.096452                 1.096372 0.759080                    1               0.759080
         1 client_4     ds2         1          3 race_smoothmix (g=0.90, a=0.20, b=4.00, t=2.10, k=7, eg=0.07, mix=0.70)         0.90                   0.20                      4.0       2.1                   7       0.07       0.70    0.365838       0.299979          0.000000   0.821632           NaN        72     0.069151 0.972222             0.743750          0.732143      0.737576                0.986458             0.972222         0.978740      0.074648         1.563979                    NaN         1.000000              NaN         1.000000         1.000000                  0.653077                  0.149203                  0.197720                 0.891839 0.796238                    1               0.796238
         1 client_5     ds2         1          1  race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)         1.00                   0.24                      4.6       2.4                   5       0.10       0.78    0.449403       0.346745          0.000000   0.771202           NaN       362     0.395054 0.831492             0.831307          0.829749      0.827098                0.829532             0.831492         0.826916      0.396533         3.647867               0.965796         0.958720         0.921999         0.990261         0.992203                  0.414751                  0.294056                  0.291193                 1.555069 0.828196                    1               0.828196
         2 client_0     ds1         1          4 race_edge_plus (g=1.02, a=0.32, b=6.00, t=2.60, k=3, eg=0.15, mix=0.84)         1.02                   0.32                      6.0       2.6                   3       0.15       0.84    0.325139       0.238526          0.329364   0.857143           NaN        67     0.339701 0.865672             0.822511          0.810606      0.809481                0.874459             0.865672         0.863429      0.374925         0.792023               0.975007         0.961538         0.945355         0.995155         0.997980                  0.540813                  0.367488                  0.091699                 1.233937 0.823529                    2               0.823529
         2 client_1     ds1         1          1     race_sharp (g=1.04, a=0.30, b=5.20, t=2.50, k=5, eg=0.12, mix=0.82)         1.04                   0.30                      5.2       2.5                   5       0.12       0.82    0.273926       0.220470          0.308047   0.860000           NaN        18     0.406084 0.888889             0.714286          0.704545      0.705769                0.952381             0.888889         0.913248      0.272579         0.284007                    NaN         1.000000              NaN         1.000000         1.000000                  0.383616                  0.287413                  0.328971                 1.562503 0.751549                    2               0.751549
         2 client_2     ds1         1          4 race_edge_plus (g=1.02, a=0.32, b=6.00, t=2.60, k=3, eg=0.15, mix=0.84)         1.02                   0.32                      6.0       2.6                   3       0.15       0.84    0.293149       0.231564          0.301504   0.823232           NaN        27     0.335938 0.925926             0.727273          0.750000      0.738095                0.858586             0.925926         0.890653      0.311666         0.397830               0.906071         0.760000         0.864286         1.000000         1.000000                  0.355587                  0.348436                  0.295976                 1.562137 0.785053                    2               0.785053
         2 client_3     ds2         1          2    race_robust (g=1.08, a=0.22, b=4.50, t=2.80, k=5, eg=0.11, mix=0.76)         1.08                   0.22                      4.5       2.8                   5       0.11       0.76    0.224366       0.153399          0.256254   0.904610           NaN        86     0.256976 0.906977             0.442522          0.463938      0.452124                0.891878             0.906977         0.897934      0.255787         0.893762               0.922445         0.741176         0.967357         0.981246         1.000000                  0.582300                  0.218808                  0.198893                 1.384496 0.565837                    2               0.565837
         2 client_4     ds2         1          2    race_robust (g=1.08, a=0.22, b=4.50, t=2.80, k=5, eg=0.11, mix=0.76)         1.08                   0.22                      4.5       2.8                   5       0.11       0.76    0.167190       0.103775          0.245299   0.931689           NaN        72     0.089137 0.986111             0.944444          0.988095      0.963636                0.988426             0.986111         0.986616      0.091821         0.766956                    NaN         0.999188              NaN         1.000000         1.000000                  0.771409                  0.142595                  0.085997                 0.968852 0.969255                    2               0.969255
         2 client_5     ds2         1          3 race_smoothmix (g=0.90, a=0.20, b=4.00, t=2.10, k=7, eg=0.07, mix=0.70)         0.90                   0.20                      4.0       2.1                   7       0.07       0.70    0.342983       0.247583          0.276244   0.832642           NaN       362     0.352783 0.861878             0.865595          0.859489      0.862388                0.861844             0.861878         0.861732      0.355178         3.537218               0.973147         0.971696         0.930216         0.995099         0.995578                  0.704676                  0.171446                  0.123878                 1.139320 0.862260                    2               0.862260
         3 client_0     ds1         1          2   race_texture (g=0.92, a=0.34, b=5.80, t=2.30, k=7, eg=0.10, mix=0.80)         0.92                   0.34                      5.8       2.3                   7       0.10       0.80    0.397297       0.296802          0.286686   0.825510           NaN        67     0.291211 0.880597             0.833889          0.847917      0.825390                0.898408             0.880597         0.878882      0.319103         0.923385               0.989213         0.970513         0.986339         1.000000         1.000000                  0.578505                  0.237025                  0.184470                 1.056292 0.839192                    3               0.839192
         3 client_1     ds1         1          4 race_edge_plus (g=1.02, a=0.32, b=6.00, t=2.60, k=3, eg=0.15, mix=0.84)         1.02                   0.32                      6.0       2.6                   3       0.15       0.84    0.723908       0.656946          0.377248   0.712000           NaN        18     0.429073 1.000000             1.000000          1.000000      1.000000                1.000000             1.000000         1.000000      0.348580         0.338008                    NaN         1.000000              NaN         1.000000         1.000000                  0.414622                  0.299353                  0.286025                 1.561226 1.000000                    3               1.000000
         3 client_2     ds1         1          2   race_texture (g=0.92, a=0.34, b=5.80, t=2.30, k=7, eg=0.10, mix=0.80)         0.92                   0.34                      5.8       2.3                   7       0.10       0.80    0.355412       0.287465          0.273860   0.835859           NaN        27     0.377372 0.814815             0.785294          0.825000      0.787645                0.897168             0.814815         0.846990      0.363286         0.467984               0.971071         0.920000         0.964286         1.000000         1.000000                  0.582299                  0.211268                  0.206432                 1.395020 0.794437                    3               0.794437
         3 client_3     ds2         1          1  race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)         1.00                   0.24                      4.6       2.4                   5       0.10       0.78    0.298910       0.204862          0.240829   0.876789           NaN        86     0.191482 0.953488             0.722861          0.727096      0.724960                0.942261             0.953488         0.947826      0.189869         0.903929               0.969971         0.894118         0.991212         0.994555         1.000000                  0.821872                  0.154447                  0.023681                 0.767881 0.782092                    3               0.782092
         3 client_4     ds2         1          0      race_soft (g=0.95, a=0.18, b=3.80, t=2.20, k=3, eg=0.08, mix=0.72)         0.95                   0.18                      3.8       2.2                   3       0.08       0.72    0.228418       0.154902          0.224604   0.910816           NaN        72     0.035223 1.000000             1.000000          1.000000      1.000000                1.000000             1.000000         1.000000      0.036253         0.793967                    NaN         1.000000              NaN         1.000000         1.000000                  0.788286                  0.165349                  0.046365                 0.853315 1.000000                    3               1.000000
         3 client_5     ds2         1          0      race_soft (g=0.95, a=0.18, b=3.80, t=2.20, k=3, eg=0.08, mix=0.72)         0.95                   0.18                      3.8       2.2                   3       0.08       0.72    0.358275       0.225776          0.235428   0.863174           NaN       362     0.184798 0.947514             0.951299          0.945042      0.947758                0.948030             0.947514         0.947331      0.186787         3.499245               0.991816         0.995702         0.987814         0.986042         0.997705                  0.746308                  0.160206                  0.093486                 1.032733 0.947697                    3               0.947697
         4 client_0     ds1         1          1     race_sharp (g=1.04, a=0.30, b=5.20, t=2.50, k=5, eg=0.12, mix=0.82)         1.04                   0.30                      5.2       2.5                   5       0.12       0.82    0.269680       0.197858          0.220093   0.904082           NaN        67     0.219978 0.880597             0.835884          0.858523      0.842252                0.887682             0.880597         0.881599      0.250885         0.787400               0.986395         0.983333         0.967213         0.998062         0.996970                  0.884548                  0.066264                  0.049188                 0.582799 0.851838                    4               0.851838
         4 client_1     ds1         1          2   race_texture (g=0.92, a=0.34, b=5.80, t=2.30, k=7, eg=0.10, mix=0.80)         0.92                   0.34                      5.8       2.3                   7       0.10       0.80    0.358828       0.299467          0.289613   0.864000           NaN        18     0.277146 1.000000             1.000000          1.000000      1.000000                1.000000             1.000000         1.000000      0.155663         0.281325                    NaN         1.000000              NaN         1.000000         1.000000                  0.321351                  0.262010                  0.416639                 1.552579 1.000000                    4               1.000000
    ... showing first 20 of 90 rows
    
    ======================================================================================================================
    STEP 11: FINAL EVALUATION
    ======================================================================================================================
    
    ----------------------------------------------------------------------------------------------------------------------
    Best RL-UCB-selected preprocessing preset per client
    ----------------------------------------------------------------------------------------------------------------------
      client dataset  best_theta_arm best_theta_name                                           best_theta_str  estimated_value  pulls
    client_0     ds1               3     race_robust (g=1.08, a=0.22, b=4.50, t=2.80, k=5, eg=0.11, mix=0.76)         0.935612     15
    client_1     ds1               4  race_edge_plus (g=1.02, a=0.32, b=6.00, t=2.60, k=3, eg=0.15, mix=0.84)         1.000000     15
    client_2     ds1               0   race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)         0.889508     15
    client_3     ds2               0       race_soft (g=0.95, a=0.18, b=3.80, t=2.20, k=3, eg=0.08, mix=0.72)         0.882338     15
    client_4     ds2               1   race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)         0.992314     15
    client_5     ds2               2     race_robust (g=1.08, a=0.22, b=4.50, t=2.80, k=5, eg=0.11, mix=0.76)         0.970062     15
    
    ----------------------------------------------------------------------------------------------------------------------
    Validation ranking of RL-UCB candidates — DS1
    ----------------------------------------------------------------------------------------------------------------------
        theta_name    score  val_acc   val_f1  estimated_bandit_value  from_client_gid
       race_robust 0.968856 0.969027 0.968799                0.935612                0
    race_edge_plus 0.968856 0.969027 0.968799                1.000000                1
     race_balanced 0.968856 0.969027 0.968799                0.889508                2
    
    ----------------------------------------------------------------------------------------------------------------------
    Validation ranking of RL-UCB candidates — DS2
    ----------------------------------------------------------------------------------------------------------------------
       theta_name    score  val_acc   val_f1  estimated_bandit_value  from_client_gid
        race_soft 0.982631 0.982938 0.982529                0.882338                3
    race_balanced 0.982631 0.982938 0.982529                0.992314                4
      race_robust 0.982631 0.982938 0.982529                0.970062                5
    
    ----------------------------------------------------------------------------------------------------------------------
    Final preset strategy after RL-UCB training
    ----------------------------------------------------------------------------------------------------------------------
    dataset                                        strategy     theta_names    score  val_acc   val_f1
        ds1 validation_selected_single_after_rlucb_training ['race_robust'] 0.968856 0.969027 0.968799
        ds2 validation_selected_single_after_rlucb_training   ['race_soft'] 0.982631 0.982938 0.982529
    
    ======================================================================================================================
    STEP 12: EXTENDED METRICS + CALIBRATION + ERROR ANALYSIS
    ======================================================================================================================
    
    ----------------------------------------------------------------------------------------------------------------------
    Extended TEST metrics (DS1 vs DS2)
    ----------------------------------------------------------------------------------------------------------------------
     dataset      acc  balanced_acc  precision_macro  recall_macro  f1_macro  precision_weighted  recall_weighted  f1_weighted  log_loss      mcc    kappa  jaccard_macro  ppv_macro  npv_macro  specificity_macro  fpr_macro  fnr_macro      ece      mce  brier_multi  auc_roc_macro_ovr  auc_class_0  auc_class_1  auc_class_2  auc_class_3
    ds1_test 0.973451      0.973587         0.973903      0.973587  0.973634            0.973985         0.973451     0.973608  0.103819 0.964672 0.964596       0.949107   0.973903   0.991124           0.991159   0.008841   0.026413 0.011479 0.369121     0.044098           0.998772     0.997374     0.999681     0.999188     0.998845
    ds2_test 0.983886      0.983619         0.983127      0.983619  0.983362            0.983952         0.983886     0.983907  0.062602 0.978476 0.978468       0.967379   0.983127   0.994644           0.994713   0.005287   0.016381 0.018890 0.531241     0.024534           0.999395     0.999859     0.998577     0.999996     0.999148
    
    ----------------------------------------------------------------------------------------------------------------------
    Classwise metrics — DS1 TEST
    ----------------------------------------------------------------------------------------------------------------------
     class_id class_name  support  tp  fp  fn  tn  prevalence      ppv      npv   recall  specificity      fpr      fnr  jaccard  balanced_acc
            0     glioma       56  54   4   2 166    0.247788 0.931034 0.988095 0.964286     0.976471 0.023529 0.035714 0.900000      0.970378
            1 meningioma       55  54   1   1 170    0.243363 0.981818 0.994152 0.981818     0.994152 0.005848 0.018182 0.964286      0.987985
            2    notumor       59  57   1   2 166    0.261062 0.982759 0.988095 0.966102     0.994012 0.005988 0.033898 0.950000      0.980057
            3  pituitary       56  55   0   1 170    0.247788 1.000000 0.994152 0.982143     1.000000 0.000000 0.017857 0.982143      0.991071
    
    ----------------------------------------------------------------------------------------------------------------------
    Classwise metrics — DS2 TEST
    ----------------------------------------------------------------------------------------------------------------------
     class_id class_name  support  tp  fp  fn  tn  prevalence      ppv      npv   recall  specificity      fpr      fnr  jaccard  balanced_acc
            0     glioma      244 241   5   3 806    0.231280 0.979675 0.996292 0.987705     0.993835 0.006165 0.012295 0.967871      0.990770
            1 meningioma      247 241   7   6 801    0.234123 0.971774 0.992565 0.975709     0.991337 0.008663 0.024291 0.948819      0.983523
            2    notumor      300 297   0   3 755    0.284360 1.000000 0.996042 0.990000     1.000000 0.000000 0.010000 0.990000      0.995000
            3  pituitary      264 259   5   5 786    0.250237 0.981061 0.993679 0.981061     0.993679 0.006321 0.018939 0.962825      0.987370
    
    ----------------------------------------------------------------------------------------------------------------------
    Top confusion pairs — DS1 TEST
    ----------------------------------------------------------------------------------------------------------------------
    true_class pred_class  count
       notumor     glioma      2
        glioma meningioma      1
    meningioma     glioma      1
        glioma    notumor      1
     pituitary     glioma      1
        glioma  pituitary      0
    meningioma  pituitary      0
    meningioma    notumor      0
       notumor meningioma      0
       notumor  pituitary      0
    
    ----------------------------------------------------------------------------------------------------------------------
    Top confusion pairs — DS2 TEST
    ----------------------------------------------------------------------------------------------------------------------
    true_class pred_class  count
        glioma meningioma      3
    meningioma     glioma      3
     pituitary meningioma      3
    meningioma  pituitary      3
       notumor  pituitary      2
     pituitary     glioma      2
       notumor meningioma      1
        glioma    notumor      0
        glioma  pituitary      0
    meningioma    notumor      0
    
    ----------------------------------------------------------------------------------------------------------------------
    Calibration bins — DS1 TEST
    ----------------------------------------------------------------------------------------------------------------------
     bin_id  bin_left  bin_right  bin_confidence  bin_accuracy  bin_gap  bin_count
          0  0.000000   0.083333             NaN           NaN      NaN          0
          1  0.083333   0.166667             NaN           NaN      NaN          0
          2  0.166667   0.250000             NaN           NaN      NaN          0
          3  0.250000   0.333333             NaN           NaN      NaN          0
          4  0.333333   0.416667             NaN           NaN      NaN          0
          5  0.416667   0.500000             NaN           NaN      NaN          0
          6  0.500000   0.583333        0.535363      0.500000 0.035363          2
          7  0.583333   0.666667        0.637644      0.666667 0.029023          3
          8  0.666667   0.750000             NaN           NaN      NaN          0
          9  0.750000   0.833333        0.827506      1.000000 0.172494          1
         10  0.833333   0.916667        0.869121      0.500000 0.369121          2
         11  0.916667   1.000000        0.979239      0.986239 0.006999        218
    
    ----------------------------------------------------------------------------------------------------------------------
    Calibration bins — DS2 TEST
    ----------------------------------------------------------------------------------------------------------------------
     bin_id  bin_left  bin_right  bin_confidence  bin_accuracy  bin_gap  bin_count
          0  0.000000   0.083333             NaN           NaN      NaN          0
          1  0.083333   0.166667             NaN           NaN      NaN          0
          2  0.166667   0.250000             NaN           NaN      NaN          0
          3  0.250000   0.333333             NaN           NaN      NaN          0
          4  0.333333   0.416667             NaN           NaN      NaN          0
          5  0.416667   0.500000        0.468759      1.000000 0.531241          2
          6  0.500000   0.583333        0.533760      0.333333 0.200426          6
          7  0.583333   0.666667        0.613477      0.666667 0.053190          3
          8  0.666667   0.750000        0.699701      0.571429 0.128272          7
          9  0.750000   0.833333        0.796630      0.777778 0.018852          9
         10  0.833333   0.916667        0.878255      0.846154 0.032101         13
         11  0.916667   1.000000        0.979291      0.995074 0.015783       1015
    
    ----------------------------------------------------------------------------------------------------------------------
    VAL + TEST tables
    ----------------------------------------------------------------------------------------------------------------------
                             setting split      dataset      acc  precision_macro  recall_macro  f1_macro  precision_weighted  recall_weighted  f1_weighted  log_loss  auc_roc_macro_ovr  loss_ce  eval_time_s  balanced_acc      mcc    kappa  ppv_macro  npv_macro  specificity_macro      ece      mce  brier_multi
    ARCF-Net RL-UCB Preset Selection   VAL          ds1 0.969027         0.970607      0.968519  0.968799            0.969997         0.969027     0.968774  0.098224           0.998998 0.093806     4.529644           NaN      NaN      NaN        NaN        NaN                NaN      NaN      NaN          NaN
    ARCF-Net RL-UCB Preset Selection   VAL          ds2 0.982938         0.982379      0.982697  0.982529            0.982965         0.982938     0.982942  0.074307           0.998759 0.074259    20.473753           NaN      NaN      NaN        NaN        NaN                NaN      NaN      NaN          NaN
    ARCF-Net RL-UCB Preset Selection   VAL global_equal 0.975982         0.976493      0.975608  0.975664            0.976481         0.975982     0.975858  0.086265           0.998879 0.084033    12.501699           NaN      NaN      NaN        NaN        NaN                NaN      NaN      NaN          NaN
    ARCF-Net RL-UCB Preset Selection  TEST          ds1 0.973451         0.973903      0.973587  0.973634            0.973985         0.973451     0.973608  0.103819           0.998772 0.098826     4.676671      0.973587 0.964672 0.964596   0.973903   0.991124           0.991159 0.011479 0.369121     0.044098
    ARCF-Net RL-UCB Preset Selection  TEST          ds2 0.983886         0.983127      0.983619  0.983362            0.983952         0.983886     0.983907  0.062602           0.999395 0.062564    20.518183      0.983619 0.978476 0.978468   0.983127   0.994644           0.994713 0.018890 0.531241     0.024534
    ARCF-Net RL-UCB Preset Selection  TEST global_equal 0.978669         0.978515      0.978603  0.978498            0.978968         0.978669     0.978757  0.083211           0.999083 0.080695    12.597427           NaN      NaN      NaN        NaN        NaN                NaN      NaN      NaN          NaN
    
    Selection summary:
    - Best round: 15 | best_reward=1.1309
    - DS1 final strategy: validation_selected_single_after_rlucb_training | names=['race_robust']
    - DS2 final strategy: validation_selected_single_after_rlucb_training | names=['race_soft']
    
    ======================================================================================================================
    STEP 13: PREPROCESSING VALIDATION
    ======================================================================================================================
    
    ----------------------------------------------------------------------------------------------------------------------
    Preprocessing validation summary (DS1 VAL sample)
    ----------------------------------------------------------------------------------------------------------------------
                metric     mean      std       min      max
    edge_energy_before 0.041596 0.021720  0.014947 0.165471
     edge_energy_after 0.094055 0.024375  0.057067 0.214411
        entropy_before 5.820408 0.628255  3.547314 7.239670
         entropy_after 6.492028 0.566759  3.888517 7.470318
       contrast_before 0.187640 0.052597  0.101468 0.363759
        contrast_after 0.226862 0.026633  0.190911 0.342708
       edge_gain_ratio 2.445604 0.518596  1.176741 4.282521
         entropy_delta 0.671621 0.235525  0.230648 1.413039
        contrast_delta 0.039222 0.029151 -0.022064 0.100209
    
    ----------------------------------------------------------------------------------------------------------------------
    Preprocessing validation summary (DS2 VAL sample)
    ----------------------------------------------------------------------------------------------------------------------
                metric     mean      std       min      max
    edge_energy_before 0.069022 0.036905  0.016726 0.375572
     edge_energy_after 0.145462 0.035837  0.089937 0.426276
        entropy_before 6.866192 0.472092  5.135309 7.798752
         entropy_after 7.350881 0.299229  5.785421 7.895350
       contrast_before 0.233743 0.033322  0.144569 0.352019
        contrast_after 0.266833 0.018246  0.198879 0.339242
       edge_gain_ratio 2.304945 0.556772  1.135004 6.019547
         entropy_delta 0.484689 0.213825  0.061857 1.168468
        contrast_delta 0.033091 0.018741 -0.015189 0.091739
    
    ======================================================================================================================
    STEP 14: SAVING CHECKPOINT + CSV
    ======================================================================================================================
    Saved checkpoint: /kaggle/working/outputs/ARCFNet_Ablation_RLUCB_PresetSelection_checkpoint.pth
    Saved CSV: /kaggle/working/outputs/ARCFNet_Ablation_RLUCB_PresetSelection_outputs.csv
    
    DONE
    Method: ARCF-Net = Adaptive RACE-FELCM with CRAF Fusion Network
    Ablation: RL-UCB Preset Selection, Fixed Clients, Full Participation
    Backbone: Residual Network-50
    Best round: 15
    Fixed clients => DS1=3, DS2=3, TOTAL=6
    Rounds completed: 15
    Global TEST acc: 0.9787
    Global TEST f1_macro: 0.9785
    DS1 TEST acc: 0.9735
    DS2 TEST acc: 0.9839
    DS1 final strategy: validation_selected_single_after_rlucb_training | names=['race_robust']
    DS2 final strategy: validation_selected_single_after_rlucb_training | names=['race_soft']


# **6. RL-based participation**


```python
import os
import sys
import time
import math
import copy
import hashlib
import random
import subprocess
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    log_loss, confusion_matrix, roc_auc_score,
    matthews_corrcoef, cohen_kappa_score, balanced_accuracy_score,
    jaccard_score
)

# ============================================================
# ARCF-Net ABLATION 6
# RL-BASED PARTICIPATION + RL-UCB PRESET SELECTION + NO PLOTS
# ------------------------------------------------------------
# - Uses BOTH datasets
# - Kaggle-ready
# - True FL with FedAvg + FedProx + prototype sharing
# - RL/UCB decides WHICH clients participate each round
# - RL-UCB also decides preprocessing preset per selected client
# - Fixed client pool: 3 DS1 + 3 DS2
# - Balanced participation each round: 2 DS1 + 2 DS2
# - No plots generated
# ============================================================

# -------------------------
# Helpers
# -------------------------
def pip_install(pkg):
    subprocess.check_call([sys.executable, "-m", "pip", "-q", "install", pkg])

def running_on_kaggle():
    return os.path.exists("/kaggle/input")

IS_KAGGLE = running_on_kaggle()

try:
    import kagglehub
except Exception:
    try:
        pip_install("kagglehub")
        import kagglehub
    except Exception:
        kagglehub = None

try:
    from torchvision import transforms
    from torchvision.models import resnet50, ResNet50_Weights
except Exception:
    pip_install("torchvision")
    from torchvision import transforms
    from torchvision.models import resnet50, ResNet50_Weights

# -------------------------
# Reproducibility + Device
# -------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if DEVICE.type == "cuda":
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

print("=" * 118)
print("ARCF-Net ABLATION 6: RL-BASED PARTICIPATION + RL-UCB PRESET SELECTION + NO PLOTS")
print("=" * 118)
print(f"ENV: {'KAGGLE' if IS_KAGGLE else 'NON-KAGGLE'} | DEVICE: {DEVICE} | torch={torch.__version__}")
print("=" * 118)

# -------------------------
# Configuration
# -------------------------
CFG = {
    "rounds": 15,

    # fixed client pool
    "fixed_clients_ds1": 3,
    "fixed_clients_ds2": 3,

    # RL-based participation target each round
    "active_clients_per_round_ds1": 2,
    "active_clients_per_round_ds2": 2,

    # local training
    "local_epochs": 2,
    "lr_head": 1e-3,
    "lr_backbone": 2e-4,
    "weight_decay": 5e-4,
    "warmup_epochs": 1,
    "label_smoothing": 0.02,
    "focal_gamma": 1.35,
    "grad_clip": 1.0,
    "fedprox_mu": 0.01,
    "proto_lambda": 0.12,

    # image
    "img_size": 224 if torch.cuda.is_available() else 160,
    "batch_size": 16 if torch.cuda.is_available() else 8,
    "num_workers": 2 if torch.cuda.is_available() else 2,

    # global split
    "global_val_frac": 0.15,
    "test_frac": 0.15,

    # client split
    "client_val_frac": 0.12,
    "client_tune_frac": 0.12,
    "min_per_class_per_client": 5,

    # non-iid
    "dirichlet_alpha": 0.35,

    # preprocessing / augmentation
    "use_preprocessing": True,
    "use_augmentation": True,

    # transfer learning
    "freeze_backbone_rounds": 2,
    "unfreeze_last_blocks": 2,

    # RL-UCB
    "ucb_c": 1.35,
    "theta_probe_topk": 3,
    "theta_probe_batches": 2,

    # final inference
    "final_use_tta": True,

    # reward
    "reward_f1_weight": 0.75,
    "reward_acc_weight": 0.25,

    # best-round selection
    "best_round_mass_ds1": 0.50,
    "best_round_mass_ds2": 0.50,
    "best_round_min_bonus": 0.15,

    # FedAvg tempering
    "fedavg_temper": 0.50,

    # misc / sanity
    "quick_hash_subset_per_split": 300,
    "preproc_val_sample_n": 400,

    # no plots
    "make_plots": False,
    "calibration_bins": 12,
}

OUTDIR = "/kaggle/working/outputs" if IS_KAGGLE else "/content/outputs"
os.makedirs(OUTDIR, exist_ok=True)
MODEL_PATH = os.path.join(OUTDIR, "ARCFNet_Ablation_RLParticipation_checkpoint.pth")
CSV_PATH = os.path.join(OUTDIR, "ARCFNet_Ablation_RLParticipation_outputs.csv")

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], device=DEVICE).view(1, 3, 1, 1)
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225], device=DEVICE).view(1, 3, 1, 1)

labels = ["glioma", "meningioma", "notumor", "pituitary"]
label2id = {l: i for i, l in enumerate(labels)}
id2label = {i: l for l, i in label2id.items()}
NUM_CLASSES = len(labels)

METHOD_INFO = {
    "acronym": "ARCF-Net",
    "full_form": "Adaptive RACE-FELCM with CRAF Fusion Network",
    "preprocessing_full_form": "Robust Adaptive Context-Enhanced Fuzzy Edge Local Contrast Mapping",
    "fusion_full_form": "Cross-Residual Adaptive Fusion",
    "backbone_full_form": "Residual Network-50",
    "ablation_name": "RL-Based Participation with RL-UCB Preset Selection",
}

# ============================================================
# Preset banks
# ============================================================
THETA_FULLFORMS = {
    "gamma": "Robust power exponent",
    "alpha": "Local contrast enhancement weight",
    "beta": "Contrast saturation sharpness",
    "tau": "Robust clipping threshold",
    "k": "Local context blur kernel",
    "edge_gain": "Gradient structure gain",
    "blend": "Structure-preserving residual blend",
}

PRESET_BANK_DS1 = [
    ("race_balanced",  (1.00, 0.24, 4.6, 2.4, 5, 0.10, 0.78)),
    ("race_sharp",     (1.04, 0.30, 5.2, 2.5, 5, 0.12, 0.82)),
    ("race_texture",   (0.92, 0.34, 5.8, 2.3, 7, 0.10, 0.80)),
    ("race_robust",    (1.08, 0.22, 4.5, 2.8, 5, 0.11, 0.76)),
    ("race_edge_plus", (1.02, 0.32, 6.0, 2.6, 3, 0.15, 0.84)),
    ("race_focus",     (1.10, 0.36, 6.4, 2.7, 3, 0.17, 0.86)),
]

PRESET_BANK_DS2 = [
    ("race_soft",      (0.95, 0.18, 3.8, 2.2, 3, 0.08, 0.72)),
    ("race_balanced",  (1.00, 0.24, 4.6, 2.4, 5, 0.10, 0.78)),
    ("race_robust",    (1.08, 0.22, 4.5, 2.8, 5, 0.11, 0.76)),
    ("race_smoothmix", (0.90, 0.20, 4.0, 2.1, 7, 0.07, 0.70)),
]

# ============================================================
# CSV collector
# ============================================================
ALL_ROWS = []

def add_table_to_csv(df, table_name):
    if df is None or len(df) == 0:
        return
    df2 = df.copy()
    df2.insert(0, "table_name", table_name)
    for _, row in df2.iterrows():
        ALL_ROWS.append(row.to_dict())

def print_table(df, title, max_rows=12):
    print("\n" + "-" * 118)
    print(title)
    print("-" * 118)
    if df is None or len(df) == 0:
        print("[empty]")
    else:
        print(df.head(max_rows).to_string(index=False))
        if len(df) > max_rows:
            print(f"... showing first {max_rows} of {len(df)} rows")

def safe_float(x, default=np.nan):
    try:
        return float(x)
    except Exception:
        return default

def normalize_imagenet(x):
    return (x - IMAGENET_MEAN) / IMAGENET_STD

def score_metric(m):
    return (
        CFG["reward_f1_weight"] * safe_float(m.get("f1_macro"), 0.0)
        + CFG["reward_acc_weight"] * safe_float(m.get("acc"), 0.0)
    )

def frame_to_manifest(df_):
    return df_[["path", "label", "source", "filename", "y"]].to_dict(orient="list")

# ============================================================
# STEP 0: ACCESS DATASETS
# ============================================================
print("\n" + "=" * 118)
print("STEP 0: ACCESS DATASETS")
print("=" * 118)

REQ1 = {"512Glioma", "512Meningioma", "512Normal", "512Pituitary"}
REQ2 = {"glioma", "meningioma", "notumor", "pituitary"}

def norm_label(name: str):
    s = str(name).strip().lower()
    if "glioma" in s:
        return "glioma"
    if "meningioma" in s:
        return "meningioma"
    if "pituitary" in s:
        return "pituitary"
    if "normal" in s or "no_tumor" in s or "no tumor" in s or "notumor" in s:
        return "notumor"
    return None

def find_root_with_required_class_dirs(base_dir, required_set, prefer_raw=True):
    candidates = []
    for root, dirs, _ in os.walk(base_dir):
        if required_set.issubset(set(dirs)):
            candidates.append(root)
    if not candidates:
        return None

    def score(p):
        pl = p.lower()
        sc = 0
        if prefer_raw:
            if "raw data" in pl:
                sc += 7
            if os.path.basename(p).lower() == "raw":
                sc += 7
            if "/raw/" in pl or "\\raw\\" in pl:
                sc += 3
            if "augmented" in pl:
                sc -= 20
        sc -= 0.0001 * len(p)
        return sc

    return max(candidates, key=score)

def search_kaggle_input_for_root(required_set, prefer_raw=True):
    if not os.path.exists("/kaggle/input"):
        return None
    return find_root_with_required_class_dirs("/kaggle/input", required_set, prefer_raw=prefer_raw)

def try_kagglehub_download(slug, required_set, prefer_raw=True):
    if kagglehub is None:
        return None, None
    try:
        base = kagglehub.dataset_download(slug)
        root = find_root_with_required_class_dirs(base, required_set, prefer_raw=prefer_raw)
        return base, root
    except Exception:
        return None, None

DS1_BASE = None
DS2_BASE = None

DS1_ROOT = search_kaggle_input_for_root(REQ1, prefer_raw=True)
DS2_ROOT = search_kaggle_input_for_root(REQ2, prefer_raw=False)

if DS1_ROOT is None:
    DS1_BASE, DS1_ROOT = try_kagglehub_download(
        "orvile/pmram-bangladeshi-brain-cancer-mri-dataset",
        REQ1,
        prefer_raw=True,
    )

if DS2_ROOT is None:
    DS2_BASE, DS2_ROOT = try_kagglehub_download(
        "yassinebazgour/preprocessed-brain-mri-scans-for-tumors-detection",
        REQ2,
        prefer_raw=False,
    )

if DS1_ROOT is None:
    raise RuntimeError(
        "Could not locate DS1 under /kaggle/input. Add "
        "'orvile/pmram-bangladeshi-brain-cancer-mri-dataset'."
    )
if DS2_ROOT is None:
    raise RuntimeError(
        "Could not locate DS2 under /kaggle/input. Add "
        "'yassinebazgour/preprocessed-brain-mri-scans-for-tumors-detection'."
    )

print(f"Dataset-1 RAW root detected:\n  {DS1_ROOT}")
print(f"Dataset-2 root detected:\n  {DS2_ROOT}")

# ============================================================
# STEP 1: BUILD MANIFESTS
# ============================================================
print("\n" + "=" * 118)
print("STEP 1: BUILD DATA MANIFESTS")
print("=" * 118)

def list_images_under_class_root(class_root, class_dir_name):
    class_dir = os.path.join(class_root, class_dir_name)
    out = []
    for r, _, files in os.walk(class_dir):
        for fn in files:
            if fn.lower().endswith(IMG_EXTS):
                out.append(os.path.join(r, fn))
    return out

def build_df_from_root(ds_root, class_dirs, source_name):
    rows = []
    for c in class_dirs:
        lab = norm_label(c)
        imgs = list_images_under_class_root(ds_root, c)
        print(f"{source_name}: {c} -> {lab} | {len(imgs)} images")
        for p in imgs:
            rows.append({"path": p, "label": lab, "source": source_name})
    dfm = pd.DataFrame(rows).dropna().reset_index(drop=True)
    dfm["path"] = dfm["path"].astype(str)
    dfm["label"] = dfm["label"].astype(str)
    dfm["source"] = dfm["source"].astype(str)
    dfm = dfm.drop_duplicates(subset=["path"]).reset_index(drop=True)
    dfm["filename"] = dfm["path"].apply(os.path.basename)
    return dfm

df1 = build_df_from_root(DS1_ROOT, ["512Glioma", "512Meningioma", "512Normal", "512Pituitary"], "ds1_raw")
df2 = build_df_from_root(DS2_ROOT, ["glioma", "meningioma", "notumor", "pituitary"], "ds2")

def enforce_labels(df_):
    df_ = df_.copy()
    df_["label"] = df_["label"].astype(str).str.strip().str.lower()
    df_ = df_[df_["label"].isin(set(labels))].reset_index(drop=True)
    df_["y"] = df_["label"].map(label2id).astype(int)
    return df_

df1 = enforce_labels(df1)
df2 = enforce_labels(df2)

print("\nDataset-1 images:", len(df1))
print(df1["label"].value_counts().reindex(labels, fill_value=0).to_string())
print("\nDataset-2 images:", len(df2))
print(df2["label"].value_counts().reindex(labels, fill_value=0).to_string())

# ============================================================
# STEP 2: TRAIN / VAL / TEST SPLIT
# ============================================================
print("\n" + "=" * 118)
print("STEP 2: TRAIN / VAL / TEST SPLIT")
print("=" * 118)

def split_dataset(df_):
    train_df, temp_df = train_test_split(
        df_,
        test_size=(CFG["global_val_frac"] + CFG["test_frac"]),
        stratify=df_["y"],
        random_state=SEED,
    )
    val_rel = CFG["global_val_frac"] / (CFG["global_val_frac"] + CFG["test_frac"])
    val_df, test_df = train_test_split(
        temp_df,
        test_size=(1 - val_rel),
        stratify=temp_df["y"],
        random_state=SEED,
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)

train1, val1, test1 = split_dataset(df1)
train2, val2, test2 = split_dataset(df2)

print(f"DS1 TRAIN: {len(train1)} | VAL: {len(val1)} | TEST: {len(test1)}")
print(f"DS2 TRAIN: {len(train2)} | VAL: {len(val2)} | TEST: {len(test2)}")

# ============================================================
# STEP 2.5: SANITY / LEAKAGE CHECKS
# ============================================================
print("\n" + "=" * 118)
print("STEP 2.5: SANITY / LEAKAGE CHECKS")
print("=" * 118)

def split_overlap_checks(train_df, val_df, test_df):
    tr = set(train_df["path"].tolist())
    va = set(val_df["path"].tolist())
    te = set(test_df["path"].tolist())
    checks = {
        "path_overlap_train_val": len(tr.intersection(va)),
        "path_overlap_train_test": len(tr.intersection(te)),
        "path_overlap_val_test": len(va.intersection(te)),
        "unique_paths_train": len(tr),
        "unique_paths_val": len(va),
        "unique_paths_test": len(te),
    }
    trf = set(train_df["filename"].tolist())
    vaf = set(val_df["filename"].tolist())
    tef = set(test_df["filename"].tolist())
    checks.update({
        "filename_overlap_train_val": len(trf.intersection(vaf)),
        "filename_overlap_train_test": len(trf.intersection(tef)),
        "filename_overlap_val_test": len(vaf.intersection(tef)),
    })
    return checks

def md5_file(path, max_bytes=2_000_000):
    h = hashlib.md5()
    try:
        with open(path, "rb") as f:
            h.update(f.read(max_bytes))
        return h.hexdigest()
    except Exception:
        return None

def quick_hash_subset(frame, n=300):
    n = min(n, len(frame))
    if n <= 0:
        return set()
    idx = np.random.choice(len(frame), size=n, replace=False)
    hashes = []
    for i in idx:
        hv = md5_file(frame.iloc[i]["path"])
        if hv is not None:
            hashes.append(hv)
    return set(hashes)

def leakage_report(name, tr, va, te):
    over = split_overlap_checks(tr, va, te)
    leak_df = pd.DataFrame([over])

    n_hash = int(CFG["quick_hash_subset_per_split"])
    trh = quick_hash_subset(tr, n_hash)
    vah = quick_hash_subset(va, n_hash)
    teh = quick_hash_subset(te, n_hash)

    hash_over = {
        "subset_hash_train_val": len(trh.intersection(vah)),
        "subset_hash_train_test": len(trh.intersection(teh)),
        "subset_hash_val_test": len(vah.intersection(teh)),
        "subset_hash_n_train": len(trh),
        "subset_hash_n_val": len(vah),
        "subset_hash_n_test": len(teh),
    }
    leak_df = pd.concat([leak_df, pd.DataFrame([hash_over])], axis=1)
    print_table(leak_df, f"Leakage / Sanity Summary — {name}", max_rows=5)
    add_table_to_csv(leak_df, f"leakage_sanity_{name}")

leakage_report("ds1", train1, val1, test1)
leakage_report("ds2", train2, val2, test2)

# ============================================================
# STEP 3: RL-UCB BANDIT
# ============================================================
print("\n" + "=" * 118)
print("STEP 3: RL-UCB BANDIT")
print("=" * 118)

class UCBBandit:
    def __init__(self, n_arms, c=1.35):
        self.n_arms = int(n_arms)
        self.c = float(c)
        self.counts = np.zeros(self.n_arms, dtype=np.int64)
        self.values = np.zeros(self.n_arms, dtype=np.float64)

    def ucb(self, arm):
        if self.counts[arm] == 0:
            return float("inf")
        t = max(1, self.counts.sum())
        return float(self.values[arm] + self.c * math.sqrt(math.log(t + 1) / self.counts[arm]))

    def select(self):
        scores = [self.ucb(i) for i in range(self.n_arms)]
        return int(np.argmax(scores))

    def select_topk(self, k):
        scores = np.array([self.ucb(i) for i in range(self.n_arms)], dtype=np.float64)
        order = np.argsort(scores)[::-1]
        return [int(x) for x in order[:k]], scores

    def update(self, arm, reward):
        arm = int(arm)
        reward = float(reward)
        self.counts[arm] += 1
        n = self.counts[arm]
        self.values[arm] += (reward - self.values[arm]) / n

    def best_arm(self):
        if self.counts.sum() == 0:
            return 0
        return int(np.argmax(self.values))

    def state_dict(self):
        return {
            "n_arms": self.n_arms,
            "c": self.c,
            "counts": self.counts.copy(),
            "values": self.values.copy(),
        }

    def load_state_dict(self, sd):
        self.n_arms = int(sd["n_arms"])
        self.c = float(sd["c"])
        self.counts = sd["counts"].copy()
        self.values = sd["values"].copy()

# ============================================================
# STEP 4: FIXED NON-IID CLIENT POOL
# ============================================================
print("\n" + "=" * 118)
print("STEP 4: FIXED NON-IID CLIENT POOL")
print("=" * 118)

def make_clients_non_iid(train_df, n_clients, num_classes, min_per_class=5, alpha=0.35):
    y = train_df["y"].values
    idx_by_class = {c: np.where(y == c)[0].tolist() for c in range(num_classes)}
    for c in idx_by_class:
        random.shuffle(idx_by_class[c])

    client_indices = [[] for _ in range(n_clients)]

    for c in range(num_classes):
        idxs = idx_by_class[c]
        feasible = min(min_per_class, max(1, len(idxs) // n_clients))
        for k in range(n_clients):
            take = idxs[:feasible]
            idxs = idxs[feasible:]
            client_indices[k].extend(take)
        idx_by_class[c] = idxs

    for c in range(num_classes):
        idxs = idx_by_class[c]
        if len(idxs) == 0:
            continue
        props = np.random.dirichlet([alpha] * n_clients)
        counts = (props * len(idxs)).astype(int)
        diff = len(idxs) - counts.sum()
        counts[np.argmax(props)] += diff

        start = 0
        for k in range(n_clients):
            client_indices[k].extend(idxs[start:start + counts[k]])
            start += counts[k]

    for k in range(n_clients):
        random.shuffle(client_indices[k])

    return client_indices

def robust_client_splits(train_df, indices, val_frac, tune_frac):
    idxs = np.array(indices, dtype=int)
    if len(idxs) < 3:
        return idxs.tolist(), idxs.tolist(), idxs.tolist()

    yk = train_df.loc[idxs, "y"].values

    if len(np.unique(yk)) < 2 or len(idxs) < 20:
        n_tune = max(1, int(round(len(idxs) * tune_frac)))
        n_tune = min(n_tune, max(1, len(idxs) - 2))
        tune_idx = idxs[:n_tune]
        rem_idx = idxs[n_tune:]
    else:
        rem_idx, tune_idx = train_test_split(
            idxs,
            test_size=tune_frac,
            stratify=yk,
            random_state=SEED,
        )

    if len(rem_idx) < 2:
        return rem_idx.tolist(), tune_idx.tolist(), rem_idx.tolist()

    yk2 = train_df.loc[rem_idx, "y"].values
    if len(np.unique(yk2)) < 2 or len(rem_idx) < 12:
        n_val = max(1, int(round(len(rem_idx) * val_frac)))
        n_val = min(n_val, max(1, len(rem_idx) - 1))
        val_idx = rem_idx[:n_val]
        train_idx = rem_idx[n_val:]
    else:
        train_idx, val_idx = train_test_split(
            rem_idx,
            test_size=val_frac,
            stratify=yk2,
            random_state=SEED,
        )

    if len(train_idx) == 0:
        train_idx = val_idx[:]
    if len(val_idx) == 0:
        val_idx = train_idx[:1]
    return train_idx.tolist(), tune_idx.tolist(), val_idx.tolist()

n_clients_ds1 = int(CFG["fixed_clients_ds1"])
n_clients_ds2 = int(CFG["fixed_clients_ds2"])

client_indices_ds1 = make_clients_non_iid(
    train1,
    n_clients=n_clients_ds1,
    num_classes=NUM_CLASSES,
    min_per_class=CFG["min_per_class_per_client"],
    alpha=CFG["dirichlet_alpha"],
)
client_indices_ds2 = make_clients_non_iid(
    train2,
    n_clients=n_clients_ds2,
    num_classes=NUM_CLASSES,
    min_per_class=CFG["min_per_class_per_client"],
    alpha=CFG["dirichlet_alpha"],
)

print(f"Client pool DS1: {n_clients_ds1}")
print(f"Client pool DS2: {n_clients_ds2}")

# ============================================================
# STEP 5: DATA LOADERS
# ============================================================
print("\n" + "=" * 118)
print("STEP 5: DATA LOADERS")
print("=" * 118)

def load_rgb(path):
    try:
        return Image.open(path).convert("RGB")
    except Exception:
        return Image.new("RGB", (CFG["img_size"], CFG["img_size"]), (128, 128, 128))

EVAL_TFMS = transforms.Compose([
    transforms.Resize((CFG["img_size"], CFG["img_size"])),
    transforms.ToTensor(),
])

TRAIN_TFMS = transforms.Compose([
    transforms.Resize((CFG["img_size"], CFG["img_size"])),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=12),
    transforms.RandomAffine(degrees=0, translate=(0.04, 0.04), scale=(0.96, 1.04)),
    transforms.ColorJitter(brightness=0.10, contrast=0.10),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.6)),
    transforms.ToTensor(),
])

class MRIDataset(Dataset):
    def __init__(self, frame, indices=None, tfms=None, source_id=0):
        self.df = frame
        self.indices = indices if indices is not None else list(range(len(frame)))
        self.tfms = tfms
        self.source_id = int(source_id)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        j = self.indices[i]
        row = self.df.iloc[j]
        img = load_rgb(row["path"])
        x = self.tfms(img) if self.tfms is not None else transforms.ToTensor()(img)
        y = int(row["y"])
        return x, y, row["path"], self.source_id

def make_weighted_sampler(frame, indices, num_classes):
    if len(indices) == 0:
        return None
    ys = frame.loc[indices, "y"].values
    class_counts = np.bincount(ys, minlength=num_classes)
    class_weights = 1.0 / np.sqrt(np.clip(class_counts, 1, None))
    sample_weights = class_weights[ys]
    return WeightedRandomSampler(
        weights=torch.DoubleTensor(sample_weights),
        num_samples=len(sample_weights),
        replacement=True,
    )

def make_loader(frame, indices, bs, tfms, shuffle=False, sampler=None, source_id=0):
    ds = MRIDataset(frame, indices=indices, tfms=tfms, source_id=source_id)
    return DataLoader(
        ds,
        batch_size=bs,
        shuffle=(shuffle and sampler is None),
        sampler=sampler,
        num_workers=CFG["num_workers"],
        pin_memory=(DEVICE.type == "cuda"),
        drop_last=False,
        persistent_workers=(CFG["num_workers"] > 0),
    )

clients = []
gid = 0

for ds_name, df_src, client_indices, source_id in [
    ("ds1", train1, client_indices_ds1, 0),
    ("ds2", train2, client_indices_ds2, 1),
]:
    for local_id, idxs in enumerate(client_indices):
        tr_idx, tune_idx, val_idx = robust_client_splits(df_src, idxs, CFG["client_val_frac"], CFG["client_tune_frac"])
        sampler = make_weighted_sampler(df_src, tr_idx, NUM_CLASSES)

        train_loader = make_loader(
            df_src, tr_idx, CFG["batch_size"], TRAIN_TFMS,
            shuffle=(sampler is None), sampler=sampler, source_id=source_id
        )
        tune_loader = make_loader(
            df_src,
            tune_idx if len(tune_idx) else tr_idx[:max(1, min(len(tr_idx), CFG["batch_size"]))],
            CFG["batch_size"], EVAL_TFMS, shuffle=False, sampler=None, source_id=source_id
        )
        val_loader = make_loader(
            df_src,
            val_idx if len(val_idx) else tr_idx[:max(1, min(len(tr_idx), CFG["batch_size"]))],
            CFG["batch_size"], EVAL_TFMS, shuffle=False, sampler=None, source_id=source_id
        )
        proto_loader = make_loader(
            df_src, tr_idx, CFG["batch_size"], EVAL_TFMS,
            shuffle=False, sampler=None, source_id=source_id
        )

        counts = df_src.loc[tr_idx, "label"].value_counts().reindex(labels, fill_value=0)
        preset_bank = PRESET_BANK_DS1 if ds_name == "ds1" else PRESET_BANK_DS2

        clients.append({
            "gid": gid,
            "local_id": local_id,
            "dataset": ds_name,
            "source_id": source_id,
            "train_loader": train_loader,
            "tune_loader": tune_loader,
            "val_loader": val_loader,
            "proto_loader": proto_loader,
            "n_train": len(tr_idx),
            "n_tune": len(tune_idx),
            "n_val": len(val_idx),
            "train_indices": tr_idx,
            "tune_indices": tune_idx,
            "val_indices": val_idx,
            "class_counts": counts.to_dict(),
            "preset_bank": preset_bank,
            "theta_bandit": UCBBandit(len(preset_bank), c=CFG["ucb_c"]),
            "participation_rounds": 0,
        })

        print(f"{ds_name} | client_{gid} | train={len(tr_idx)} | tune={len(tune_idx)} | val={len(val_idx)}")
        gid += 1

CLIENTS_TOTAL = len(clients)

dist_rows = []
for c in clients:
    row = {
        "client": f"client_{c['gid']}",
        "dataset": c["dataset"],
        "total_train": c["n_train"],
        "total_tune": c["n_tune"],
        "total_val": c["n_val"],
        "preset_bank_names": str([x[0] for x in c["preset_bank"]]),
    }
    row.update({lab: int(c["class_counts"].get(lab, 0)) for lab in labels})
    dist_rows.append(row)

dist_df = pd.DataFrame(dist_rows)
print_table(dist_df, "Client class distribution", max_rows=20)
add_table_to_csv(dist_df, "client_distribution")

val_loader_ds1 = make_loader(val1, list(range(len(val1))), CFG["batch_size"], EVAL_TFMS, shuffle=False, source_id=0)
val_loader_ds2 = make_loader(val2, list(range(len(val2))), CFG["batch_size"], EVAL_TFMS, shuffle=False, source_id=1)
test_loader_ds1 = make_loader(test1, list(range(len(test1))), CFG["batch_size"], EVAL_TFMS, shuffle=False, source_id=0)
test_loader_ds2 = make_loader(test2, list(range(len(test2))), CFG["batch_size"], EVAL_TFMS, shuffle=False, source_id=1)

print(f"Augmentation: {'ON' if CFG['use_augmentation'] else 'OFF'}")
print(f"Preprocessing: {'ON' if CFG['use_preprocessing'] else 'OFF'}")
print(f"Total client pool: {CLIENTS_TOTAL}")

# ============================================================
# STEP 5.5: RL-BASED PARTICIPATION POLICY
# ============================================================
print("\n" + "=" * 118)
print("STEP 5.5: RL-BASED PARTICIPATION POLICY")
print("=" * 118)

ds1_global_ids = [c["gid"] for c in clients if c["dataset"] == "ds1"]
ds2_global_ids = [c["gid"] for c in clients if c["dataset"] == "ds2"]

ds1_gid_to_arm = {gid_: i for i, gid_ in enumerate(ds1_global_ids)}
ds2_gid_to_arm = {gid_: i for i, gid_ in enumerate(ds2_global_ids)}

participation_bandit_ds1 = UCBBandit(len(ds1_global_ids), c=CFG["ucb_c"])
participation_bandit_ds2 = UCBBandit(len(ds2_global_ids), c=CFG["ucb_c"])

def select_participating_clients(global_ids, bandit, k):
    selected_arms, scores = bandit.select_topk(k)
    selected_global_ids = [global_ids[a] for a in selected_arms]
    rows = []
    for arm_idx, gid_ in enumerate(global_ids):
        rows.append({
            "gid": gid_,
            "arm_index": arm_idx,
            "ucb_score": float(scores[arm_idx]) if np.isfinite(scores[arm_idx]) else np.inf,
            "count": int(bandit.counts[arm_idx]),
            "value": float(bandit.values[arm_idx]),
            "selected": int(arm_idx in selected_arms),
        })
    return selected_global_ids, selected_arms, pd.DataFrame(rows)

# ============================================================
# STEP 6: PREPROCESSING — RACE-FELCM
# ============================================================
print("\n" + "=" * 118)
print("STEP 6: PREPROCESSING — RACE-FELCM")
print("=" * 118)

class RACEFELCM(nn.Module):
    def __init__(self, gamma=1.0, alpha=0.24, beta=4.5, tau=2.4, blur_k=5, edge_gain=0.10, blend=0.78):
        super().__init__()
        self.gamma = float(gamma)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.tau = float(tau)
        self.blur_k = int(blur_k)
        self.edge_gain = float(edge_gain)
        self.blend = float(blend)

        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        lap = torch.tensor([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=torch.float32)

        self.register_buffer("sobel_x", sobel_x.view(1, 1, 3, 3))
        self.register_buffer("sobel_y", sobel_y.view(1, 1, 3, 3))
        self.register_buffer("lap", lap.view(1, 1, 3, 3))

    def forward(self, x):
        eps = 1e-6

        gray0 = x.mean(dim=1, keepdim=True)
        mu0 = F.avg_pool2d(F.pad(gray0, (1, 1, 1, 1), mode="reflect"), 3, stride=1)
        var0 = F.avg_pool2d(F.pad((gray0 - mu0).pow(2), (1, 1, 1, 1), mode="reflect"), 3, stride=1)
        tex = torch.sqrt(var0 + eps)
        tex_norm = tex / (tex.amax(dim=(2, 3), keepdim=True) + eps)
        flat_gate = 1.0 - tex_norm

        x_smooth = F.avg_pool2d(F.pad(x, (1, 1, 1, 1), mode="reflect"), 3, stride=1)
        x_dn = x * (1.0 - 0.18 * flat_gate) + x_smooth * (0.18 * flat_gate)

        mu = x_dn.mean(dim=(2, 3), keepdim=True)
        sd = x_dn.std(dim=(2, 3), keepdim=True).clamp_min(eps)
        z = ((x_dn - mu) / sd).clamp(-self.tau, self.tau)
        z = torch.sign(z) * torch.pow(z.abs().clamp_min(eps), self.gamma)

        gray = z.mean(dim=1, keepdim=True)
        k = self.blur_k if self.blur_k % 2 == 1 else self.blur_k + 1
        pad = k // 2

        local_mean = F.avg_pool2d(F.pad(gray, (pad, pad, pad, pad), mode="reflect"), k, stride=1)
        local_var = F.avg_pool2d(F.pad((gray - local_mean).pow(2), (pad, pad, pad, pad), mode="reflect"), k, stride=1)
        local_std = torch.sqrt(local_var + eps)
        local_norm = (gray - local_mean) / (local_std + eps)
        contrast_field = torch.tanh(self.beta * local_norm)

        gx = F.conv2d(F.pad(gray, (1, 1, 1, 1), mode="reflect"), self.sobel_x)
        gy = F.conv2d(F.pad(gray, (1, 1, 1, 1), mode="reflect"), self.sobel_y)
        edge = torch.sqrt(gx * gx + gy * gy + eps)
        edge = edge / (edge.amax(dim=(2, 3), keepdim=True) + eps)

        lap = F.conv2d(F.pad(gray, (1, 1, 1, 1), mode="reflect"), self.lap).abs()
        lap = lap / (lap.amax(dim=(2, 3), keepdim=True) + eps)
        coherence = torch.sigmoid(4.0 * (edge - 0.5 * lap))

        enhanced = z + self.alpha * contrast_field + self.edge_gain * edge * coherence

        mn = enhanced.amin(dim=(2, 3), keepdim=True)
        mx = enhanced.amax(dim=(2, 3), keepdim=True)
        enhanced = (enhanced - mn) / (mx - mn + eps)

        out = self.blend * enhanced + (1.0 - self.blend) * x
        return out.clamp(0, 1)

def theta_to_module(theta):
    return RACEFELCM(*theta)

def theta_to_vec(theta, batch_size):
    gamma, alpha, beta, tau, blur_k, edge_gain, blend = theta
    theta = torch.tensor(
        [gamma, alpha, beta / 8.0, tau / 4.0, blur_k / 7.0, edge_gain, blend],
        device=DEVICE,
        dtype=torch.float32
    )
    return theta.unsqueeze(0).repeat(batch_size, 1)

def theta_str(theta):
    if theta is None:
        return "None"
    g, a, b, t, k, eg, m = theta
    return f"(g={g:.2f}, a={a:.2f}, b={b:.2f}, t={t:.2f}, k={k}, eg={eg:.2f}, mix={m:.2f})"

# ============================================================
# STEP 7: MODEL — ResNet-50 + CRAF Fusion
# ============================================================
print("\n" + "=" * 118)
print("STEP 7: MODEL — ResNet-50 + CRAF Fusion")
print("=" * 118)

class CompactProjector(nn.Module):
    def __init__(self, in_dim, out_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            nn.Dropout(0.10),
        )

    def forward(self, x):
        return self.net(x)

class CRAFFusion(nn.Module):
    def __init__(self, in_dim, fuse_dim=256, cond_dim=64):
        super().__init__()
        self.proj_raw = CompactProjector(in_dim, fuse_dim)
        self.proj_enh = CompactProjector(in_dim, fuse_dim)
        self.proj_res = CompactProjector(in_dim, fuse_dim)

        self.router = nn.Sequential(
            nn.Linear(fuse_dim * 6 + cond_dim, fuse_dim),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(fuse_dim, 3),
        )

        self.residual_refine = nn.Sequential(
            nn.Linear(fuse_dim * 2 + cond_dim, fuse_dim),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(fuse_dim, fuse_dim),
        )

        self.out_norm = nn.LayerNorm(fuse_dim)

    def forward(self, f_raw, f_enh, f_res, cond):
        fr = self.proj_raw(f_raw)
        fe = self.proj_enh(f_enh)
        fs = self.proj_res(f_res)

        d_re = torch.abs(fr - fe)
        d_rs = torch.abs(fr - fs)
        d_es = torch.abs(fe - fs)

        router_in = torch.cat([fr, fe, fs, d_re, d_rs, d_es, cond], dim=1)
        gates = torch.softmax(self.router(router_in), dim=1)

        routed = gates[:, 0:1] * fr + gates[:, 1:2] * fe + gates[:, 2:3] * fs
        disagreement = (d_re + d_rs + d_es) / 3.0
        refine = self.residual_refine(torch.cat([routed, disagreement, cond], dim=1))
        fused = self.out_norm(routed + refine)

        return fused, gates

class ResNet50CRAF(nn.Module):
    def __init__(self, num_classes, cond_dim=64, fuse_dim=256, embed_dim=256, pretrained=True):
        super().__init__()
        try:
            weights = ResNet50_Weights.DEFAULT if pretrained else None
        except Exception:
            weights = None

        net = resnet50(weights=weights)
        self.backbone = nn.Sequential(
            net.conv1,
            net.bn1,
            net.relu,
            net.maxpool,
            net.layer1,
            net.layer2,
            net.layer3,
            net.layer4,
        )
        self.backbone_dim = net.fc.in_features
        self.pool = nn.AdaptiveAvgPool2d(1)

        self.theta_mlp = nn.Sequential(
            nn.Linear(7, cond_dim),
            nn.GELU(),
            nn.Linear(cond_dim, cond_dim),
        )
        self.source_emb = nn.Embedding(2, cond_dim)
        self.cond_norm = nn.LayerNorm(cond_dim)

        self.fusion = CRAFFusion(self.backbone_dim, fuse_dim=fuse_dim, cond_dim=cond_dim)

        self.neck = nn.Sequential(
            nn.LayerNorm(fuse_dim),
            nn.Linear(fuse_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(0.20),
        )
        self.classifier = nn.Linear(embed_dim, num_classes)
        self.embed_dim = embed_dim

    def _encode(self, x):
        f = self.backbone(x)
        f = self.pool(f).flatten(1)
        return f

    def forward(self, x_raw_n, x_enh_n, x_res_n, theta_vec, source_id, return_extra=False):
        cond = self.theta_mlp(theta_vec) + self.source_emb(source_id)
        cond = self.cond_norm(cond)

        f_raw = self._encode(x_raw_n)
        f_enh = self._encode(x_enh_n)
        f_res = self._encode(x_res_n)

        fused, gates = self.fusion(f_raw, f_enh, f_res, cond)
        embed = self.neck(fused)
        logits = self.classifier(embed)

        if return_extra:
            return logits, embed, gates
        return logits

def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

def set_trainable_for_round(model, rnd):
    for p in model.backbone.parameters():
        p.requires_grad = False

    for n, p in model.named_parameters():
        if not n.startswith("backbone."):
            p.requires_grad = True

    if rnd > CFG["freeze_backbone_rounds"]:
        blocks = list(model.backbone.children())
        tail_blocks = blocks[-CFG["unfreeze_last_blocks"]:]
        for blk in tail_blocks:
            for p in blk.parameters():
                p.requires_grad = True

def freeze_backbone_bn_stats(model):
    for m in model.backbone.modules():
        if isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm)):
            m.eval()
            for p in m.parameters():
                p.requires_grad = False

def make_optimizer(model):
    head_params, bb_params = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if n.startswith("backbone."):
            bb_params.append(p)
        else:
            head_params.append(p)

    groups = []
    if head_params:
        groups.append({"params": head_params, "lr": CFG["lr_head"]})
    if bb_params:
        groups.append({"params": bb_params, "lr": CFG["lr_backbone"]})

    return torch.optim.AdamW(groups, weight_decay=CFG["weight_decay"])

def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
    def lr_lambda(step):
        if step < num_warmup_steps:
            return float(step) / float(max(1, num_warmup_steps))
        progress = float(step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

global_model = ResNet50CRAF(
    num_classes=NUM_CLASSES,
    cond_dim=64,
    fuse_dim=256,
    embed_dim=256,
    pretrained=True,
).to(DEVICE)

set_trainable_for_round(global_model, rnd=1)
total_params, trainable_params = count_params(global_model)

print("Backbone: ResNet-50 | pretrained_loaded=True")
print(f"Total params: {total_params:,}")
print(f"Trainable params: {trainable_params:,} ({(100.0 * trainable_params / total_params):.2f}%)")

# ============================================================
# STEP 8: LOSSES + PROTOTYPE SHARING + EVAL HELPERS
# ============================================================
print("\n" + "=" * 118)
print("STEP 8: LOSSES + PROTOTYPE SHARING + EVAL HELPERS")
print("=" * 118)

counts1 = train1["y"].value_counts().sort_index().reindex(range(NUM_CLASSES), fill_value=0).values
counts2 = train2["y"].value_counts().sort_index().reindex(range(NUM_CLASSES), fill_value=0).values
counts = counts1 + counts2
w = (counts.sum() / np.clip(counts, 1, None)).astype(np.float32)
w = w / max(1e-6, w.mean())
class_w = torch.tensor(w, device=DEVICE)

class ClassBalancedFocalLoss(nn.Module):
    def __init__(self, class_weights, gamma=1.5, label_smoothing=0.0):
        super().__init__()
        self.class_weights = class_weights
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, logits, target):
        ce = F.cross_entropy(
            logits,
            target,
            weight=self.class_weights,
            reduction="none",
            label_smoothing=self.label_smoothing,
        )
        pt = torch.exp(-ce)
        focal = (1.0 - pt).pow(self.gamma) * ce
        return focal.mean()

criterion = ClassBalancedFocalLoss(
    class_weights=class_w,
    gamma=CFG["focal_gamma"],
    label_smoothing=CFG["label_smoothing"],
)

scaler = torch.amp.GradScaler("cuda", enabled=(DEVICE.type == "cuda")) if DEVICE.type == "cuda" else None

def fedprox_term(local_model, global_model):
    loss = 0.0
    for p_local, p_global in zip(local_model.parameters(), global_model.parameters()):
        loss = loss + ((p_local - p_global.detach()) ** 2).sum()
    return loss

def prototype_alignment_loss(embed, y, global_prototypes):
    if global_prototypes is None:
        return torch.tensor(0.0, device=embed.device)

    proto = global_prototypes["proto"]
    mask = global_prototypes["mask"]
    valid = mask[y]
    if not valid.any():
        return torch.tensor(0.0, device=embed.device)

    emb_n = F.normalize(embed[valid], dim=1)
    ref_n = F.normalize(proto[y[valid]], dim=1)
    return (1.0 - (emb_n * ref_n).sum(dim=1)).mean()

@torch.no_grad()
def compute_prototype_payload(model, loader, preproc, theta):
    model.eval()
    freeze_backbone_bn_stats(model)
    preproc.eval()

    sums = torch.zeros(NUM_CLASSES, model.embed_dim, device=DEVICE)
    counts = torch.zeros(NUM_CLASSES, device=DEVICE)

    for x, y, _, source_id in loader:
        x = x.to(DEVICE, non_blocking=True)
        y = y.to(DEVICE, non_blocking=True)
        source_id = source_id.to(DEVICE, non_blocking=True)

        x_enh = preproc(x)
        x_res = torch.abs(x_enh - x)

        x_raw_n = normalize_imagenet(x)
        x_enh_n = normalize_imagenet(x_enh)
        x_res_n = normalize_imagenet(x_res)
        theta_vec = theta_to_vec(theta, x.size(0))

        _, embed, _ = model(x_raw_n, x_enh_n, x_res_n, theta_vec, source_id, return_extra=True)

        for c in range(NUM_CLASSES):
            m = (y == c)
            if m.any():
                sums[c] += embed[m].sum(dim=0)
                counts[c] += m.sum()

    return sums.detach().cpu(), counts.detach().cpu()

def aggregate_prototypes(payloads, embed_dim):
    if len(payloads) == 0:
        return None
    sum_all = torch.zeros(NUM_CLASSES, embed_dim)
    cnt_all = torch.zeros(NUM_CLASSES)
    for s, c in payloads:
        sum_all += s
        cnt_all += c
    proto = sum_all / cnt_all.unsqueeze(1).clamp_min(1.0)
    mask = cnt_all > 0
    return {
        "proto": proto.to(DEVICE),
        "mask": mask.to(DEVICE),
        "counts": cnt_all.to(DEVICE),
    }

def gate_entropy(g):
    eps = 1e-6
    p = g.clamp(eps, 1 - eps)
    return -(p * torch.log2(p)).sum(dim=1)

@torch.no_grad()
def _auc_metrics(y_true, p_pred, num_classes):
    out = {}
    try:
        out["auc_roc_macro_ovr"] = float(roc_auc_score(y_true, p_pred, multi_class="ovr", average="macro"))
    except Exception:
        out["auc_roc_macro_ovr"] = np.nan

    for c in range(num_classes):
        try:
            yc = (y_true == c).astype(int)
            if yc.sum() > 0 and yc.sum() < len(yc):
                out[f"auc_class_{c}"] = float(roc_auc_score(yc, p_pred[:, c]))
            else:
                out[f"auc_class_{c}"] = np.nan
        except Exception:
            out[f"auc_class_{c}"] = np.nan
    return out

@torch.no_grad()
def predict_probs_single_theta(model, x, source_id, preproc, theta, use_tta=False):
    model.eval()
    freeze_backbone_bn_stats(model)
    preproc.eval()

    probs_acc = None
    versions = [x]
    if use_tta:
        versions.append(torch.flip(x, dims=[3]))

    for xv in versions:
        x_enh = preproc(xv)
        x_res = torch.abs(x_enh - xv)

        x_raw_n = normalize_imagenet(xv)
        x_enh_n = normalize_imagenet(x_enh)
        x_res_n = normalize_imagenet(x_res)
        theta_vec = theta_to_vec(theta, xv.size(0))

        logits = model(x_raw_n, x_enh_n, x_res_n, theta_vec, source_id)
        probs = torch.softmax(logits, dim=1)
        probs_acc = probs if probs_acc is None else (probs_acc + probs)

    probs_acc = probs_acc / len(versions)
    return probs_acc

@torch.no_grad()
def evaluate_full(model, loader, preproc, theta, return_gates=False, use_tta=False):
    t0 = time.time()
    model.eval()
    freeze_backbone_bn_stats(model)
    preproc.eval()

    all_y, all_p, all_loss = [], [], []
    gate_stats = []

    for x, y, _, source_id in loader:
        x = x.to(DEVICE, non_blocking=True)
        y = y.to(DEVICE, non_blocking=True)
        source_id = source_id.to(DEVICE, non_blocking=True)

        x_enh = preproc(x)
        x_res = torch.abs(x_enh - x)

        x_raw_n = normalize_imagenet(x)
        x_enh_n = normalize_imagenet(x_enh)
        x_res_n = normalize_imagenet(x_res)
        theta_vec = theta_to_vec(theta, x.size(0))

        if return_gates and (not use_tta):
            logits, _, gates = model(x_raw_n, x_enh_n, x_res_n, theta_vec, source_id, return_extra=True)
            gate_stats.append(gates.detach().cpu())
            probs = torch.softmax(logits, dim=1)
        else:
            probs = predict_probs_single_theta(model, x, source_id, preproc, theta, use_tta=use_tta)
            logits = torch.log(probs.clamp_min(1e-8))

        loss = F.cross_entropy(logits, y)

        all_loss.append(float(loss.item()))
        all_y.append(y.detach().cpu().numpy())
        all_p.append(probs.detach().cpu().numpy())

    if len(all_y) == 0:
        return {"acc": np.nan}, np.array([]), np.array([])

    y_true = np.concatenate(all_y)
    p_pred = np.concatenate(all_p)
    y_hat = np.argmax(p_pred, axis=1)

    met = {
        "loss_ce": float(np.mean(all_loss)),
        "acc": float(accuracy_score(y_true, y_hat)),
        "precision_macro": float(precision_score(y_true, y_hat, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_hat, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_hat, average="macro", zero_division=0)),
        "precision_weighted": float(precision_score(y_true, y_hat, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(y_true, y_hat, average="weighted", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_hat, average="weighted", zero_division=0)),
        "log_loss": float(log_loss(y_true, p_pred, labels=list(range(NUM_CLASSES)))),
        "eval_time_s": float(time.time() - t0),
    }
    met.update(_auc_metrics(y_true, p_pred, NUM_CLASSES))

    if return_gates and len(gate_stats) > 0:
        g = torch.cat(gate_stats, dim=0)
        met["fusion_gate_mean_raw"] = float(g[:, 0].mean().item())
        met["fusion_gate_mean_enh"] = float(g[:, 1].mean().item())
        met["fusion_gate_mean_res"] = float(g[:, 2].mean().item())
        met["fusion_gate_entropy"] = float(gate_entropy(g).mean().item())

    return met, y_true, p_pred

def train_one_epoch(model, loader, optimizer, preproc, theta, global_model=None, global_prototypes=None, scheduler=None):
    model.train()
    freeze_backbone_bn_stats(model)
    preproc.eval()

    losses, ce_losses, proto_losses, correct, total = [], [], [], 0, 0

    for x, y, _, source_id in loader:
        x = x.to(DEVICE, non_blocking=True)
        y = y.to(DEVICE, non_blocking=True)
        source_id = source_id.to(DEVICE, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        amp_enabled = (DEVICE.type == "cuda")

        with torch.amp.autocast(device_type=DEVICE.type, enabled=amp_enabled):
            x_enh = preproc(x)
            x_res = torch.abs(x_enh - x)

            x_raw_n = normalize_imagenet(x)
            x_enh_n = normalize_imagenet(x_enh)
            x_res_n = normalize_imagenet(x_res)
            theta_vec = theta_to_vec(theta, x.size(0))

            logits, embed, _ = model(x_raw_n, x_enh_n, x_res_n, theta_vec, source_id, return_extra=True)

            ce = criterion(logits, y)
            proto = prototype_alignment_loss(embed, y, global_prototypes)
            loss = ce + CFG["proto_lambda"] * proto

            if global_model is not None and CFG["fedprox_mu"] > 0:
                loss = loss + 0.5 * CFG["fedprox_mu"] * fedprox_term(model, global_model)

        if scaler is not None and amp_enabled:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if CFG["grad_clip"] > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), CFG["grad_clip"])
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if CFG["grad_clip"] > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), CFG["grad_clip"])
            optimizer.step()

        if scheduler is not None:
            scheduler.step()

        losses.append(float(loss.item()))
        ce_losses.append(float(ce.item()))
        proto_losses.append(float(proto.item()))

        preds = logits.argmax(dim=1)
        correct += int((preds == y).sum().item())
        total += int(y.size(0))

    return {
        "loss": float(np.mean(losses)) if losses else np.nan,
        "ce_loss": float(np.mean(ce_losses)) if ce_losses else np.nan,
        "proto_loss": float(np.mean(proto_losses)) if proto_losses else np.nan,
        "acc": float(correct / max(1, total)),
    }

def build_tempered_fedavg_weights(clients_subset):
    sizes = np.array([c["n_train"] for c in clients_subset], dtype=np.float64)
    sizes = np.power(np.clip(sizes, 1.0, None), CFG["fedavg_temper"])
    sizes = sizes / max(1e-12, sizes.sum())
    return sizes.tolist()

def fedavg_update(global_model, local_models, weights):
    global_sd = global_model.state_dict()
    new_sd = {}

    for key in global_sd.keys():
        ref = global_sd[key]
        if not torch.is_floating_point(ref):
            new_sd[key] = local_models[0].state_dict()[key].detach().clone()
        else:
            acc = None
            for m, w in zip(local_models, weights):
                t = m.state_dict()[key].detach().float().cpu()
                acc = t * w if acc is None else acc + t * w
            new_sd[key] = acc.to(ref.dtype)

    global_model.load_state_dict(new_sd, strict=True)

def weighted_mean(rows, key, weight_key):
    vals, ws = [], []
    for r in rows:
        v = r.get(key, np.nan)
        w = r.get(weight_key, 0)
        if np.isfinite(v) and w > 0:
            vals.append(v)
            ws.append(w)
    if len(vals) == 0:
        return np.nan
    return float(np.average(vals, weights=ws))

def aggregate_by_dataset(round_local_rows, dataset_name):
    rows = [r for r in round_local_rows if r["dataset"] == dataset_name]
    return {
        "acc": weighted_mean(rows, "val_acc", "val_size"),
        "f1_macro": weighted_mean(rows, "val_f1_macro", "val_size"),
        "precision_macro": weighted_mean(rows, "val_precision_macro", "val_size"),
        "recall_macro": weighted_mean(rows, "val_recall_macro", "val_size"),
        "log_loss": weighted_mean(rows, "val_log_loss", "val_size"),
        "loss_ce": weighted_mean(rows, "val_loss_ce", "val_size"),
        "eval_time_s": weighted_mean(rows, "val_eval_time_s", "val_size"),
    }

# ============================================================
# STEP 9: TUNE-AWARE RL-UCB PRESET SELECTION
# ============================================================
print("\n" + "=" * 118)
print("STEP 9: TUNE-AWARE RL-UCB PRESET SELECTION")
print("=" * 118)

@torch.no_grad()
def probe_theta_on_tune_loader(model, tune_loader, preset_bank, arm_candidates, n_batches=2):
    model.eval()
    freeze_backbone_bn_stats(model)

    cached_batches = []
    try:
        it = iter(tune_loader)
        for _ in range(n_batches):
            cached_batches.append(next(it))
    except Exception:
        pass

    if len(cached_batches) == 0:
        return int(arm_candidates[0])

    best_arm = int(arm_candidates[0])
    best_score = -1.0

    for arm in arm_candidates:
        arm = int(arm)
        _, theta = preset_bank[arm]
        pre = theta_to_module(theta).to(DEVICE).eval()

        batch_scores = []
        for x, y, _, source_id in cached_batches:
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            source_id = source_id.to(DEVICE)

            probs = predict_probs_single_theta(model, x, source_id, pre, theta, use_tta=False)
            pred = probs.argmax(dim=1)
            batch_scores.append((pred == y).float().mean().item())

        sc = float(np.mean(batch_scores)) if len(batch_scores) else 0.0
        if sc > best_score:
            best_score = sc
            best_arm = arm

    return int(best_arm)

def select_theta_arm_with_probe(client, model):
    bandit = client["theta_bandit"]
    ucb_scores = [bandit.ucb(i) for i in range(bandit.n_arms)]
    topk = min(CFG["theta_probe_topk"], bandit.n_arms)
    candidates = list(np.argsort(ucb_scores)[::-1][:topk])
    return probe_theta_on_tune_loader(
        model,
        client["tune_loader"],
        client["preset_bank"],
        candidates,
        n_batches=CFG["theta_probe_batches"],
    )

# ============================================================
# STEP 10: TRUE FEDERATED TRAINING
# ============================================================
print("\n" + "=" * 118)
print("STEP 10: TRUE FEDERATED TRAINING")
print("=" * 118)

history_global = []
history_local = []
participation_history = []

best_reward = -1.0
best_round_saved = None
best_model_state = None
best_global_prototypes = None
best_theta_bandit_states = None
best_participation_bandit_ds1_state = None
best_participation_bandit_ds2_state = None
global_prototypes = None

t_global_start = time.time()

print(f"Client pool => DS1={n_clients_ds1}, DS2={n_clients_ds2}, TOTAL={CLIENTS_TOTAL}")
print(f"RL-selected active clients each round => DS1={CFG['active_clients_per_round_ds1']}, DS2={CFG['active_clients_per_round_ds2']}")
print(f"Rounds: {CFG['rounds']} | Local epochs: {CFG['local_epochs']}")
print(f"FedProx μ={CFG['fedprox_mu']} | Proto λ={CFG['proto_lambda']}")
print(f"Tempered FedAvg exponent = {CFG['fedavg_temper']:.2f}")

for rnd in range(1, CFG["rounds"] + 1):
    round_t0 = time.time()

    selected_ds1, selected_arms_ds1, part_df_ds1 = select_participating_clients(
        ds1_global_ids, participation_bandit_ds1, CFG["active_clients_per_round_ds1"]
    )
    selected_ds2, selected_arms_ds2, part_df_ds2 = select_participating_clients(
        ds2_global_ids, participation_bandit_ds2, CFG["active_clients_per_round_ds2"]
    )
    selected_ids = selected_ds1 + selected_ds2

    part_df_ds1["round"] = rnd
    part_df_ds1["dataset"] = "ds1"
    part_df_ds2["round"] = rnd
    part_df_ds2["dataset"] = "ds2"
    participation_history.extend(part_df_ds1.to_dict(orient="records"))
    participation_history.extend(part_df_ds2.to_dict(orient="records"))

    print("\n" + "=" * 118)
    print(f"ROUND {rnd}/{CFG['rounds']} | selected_ds1={selected_ds1} | selected_ds2={selected_ds2}")
    print("=" * 118)

    local_models = []
    proto_payloads = []
    round_local_rows = []
    selected_clients_meta = []

    for cid in selected_ids:
        client = clients[cid]
        client["participation_rounds"] += 1

        theta_arm = select_theta_arm_with_probe(client, global_model)
        theta_name, theta = client["preset_bank"][theta_arm]
        preproc = theta_to_module(theta).to(DEVICE)

        local_model = ResNet50CRAF(
            num_classes=NUM_CLASSES,
            cond_dim=64,
            fuse_dim=256,
            embed_dim=256,
            pretrained=False,
        ).to(DEVICE)
        local_model.load_state_dict(global_model.state_dict(), strict=True)

        set_trainable_for_round(local_model, rnd)
        opt = make_optimizer(local_model)

        total_steps = max(1, len(client["train_loader"]) * CFG["local_epochs"])
        warmup_steps = max(1, len(client["train_loader"]) * CFG["warmup_epochs"])
        scheduler = get_cosine_schedule_with_warmup(opt, warmup_steps, total_steps)

        train_logs = []
        for _ in range(CFG["local_epochs"]):
            log_ep = train_one_epoch(
                local_model,
                client["train_loader"],
                opt,
                preproc,
                theta,
                global_model=global_model,
                global_prototypes=global_prototypes,
                scheduler=scheduler,
            )
            train_logs.append(log_ep)

        met_loc, _, _ = evaluate_full(local_model, client["val_loader"], preproc, theta, return_gates=True, use_tta=False)
        proto_payload = compute_prototype_payload(local_model, client["proto_loader"], preproc, theta)

        reward = score_metric(met_loc)
        client["theta_bandit"].update(theta_arm, reward)

        if client["dataset"] == "ds1":
            participation_bandit_ds1.update(ds1_gid_to_arm[cid], reward)
        else:
            participation_bandit_ds2.update(ds2_gid_to_arm[cid], reward)

        local_models.append(local_model)
        selected_clients_meta.append(client)
        proto_payloads.append(proto_payload)

        g, a, b, t, kk, eg, mix = theta
        row = {
            "round": rnd,
            "client": f"client_{cid}",
            "dataset": client["dataset"],
            "selected": 1,
            "theta_arm": int(theta_arm),
            "theta_name": theta_name,
            "theta_str": theta_str(theta),
            "gamma_power": g,
            "alpha_contrast_weight": a,
            "beta_contrast_sharpness": b,
            "tau_clip": t,
            "k_blur_kernel_size": kk,
            "edge_gain": eg,
            "blend_mix": mix,
            "train_loss": float(np.mean([x["loss"] for x in train_logs])),
            "train_ce_loss": float(np.mean([x["ce_loss"] for x in train_logs])),
            "train_proto_loss": float(np.mean([x["proto_loss"] for x in train_logs])),
            "train_acc": float(np.mean([x["acc"] for x in train_logs])),
            "val_size": client["n_val"],
            **{f"val_{k}": v for k, v in met_loc.items()},
            "reward": reward,
            "bandit_counts_total": int(client["theta_bandit"].counts.sum()),
            "bandit_value_selected": float(client["theta_bandit"].values[theta_arm]),
            "participation_rounds_so_far": int(client["participation_rounds"]),
        }
        round_local_rows.append(row)

        auc_val = row.get("val_auc_roc_macro_ovr", np.nan)
        print(
            f"Client {cid} ({client['dataset']}) | "
            f"train_acc={row['train_acc']:.4f} | "
            f"val_acc={row['val_acc']:.4f} | "
            f"val_f1={row['val_f1_macro']:.4f} | "
            f"val_auc={auc_val:.4f} | "
            f"reward={reward:.4f} | "
            f"theta={theta_name} {theta_str(theta)}"
        )

    # add non-selected rows for traceability
    for cid in range(CLIENTS_TOTAL):
        if cid in selected_ids:
            continue
        client = clients[cid]
        round_local_rows.append({
            "round": rnd,
            "client": f"client_{cid}",
            "dataset": client["dataset"],
            "selected": 0,
            "theta_arm": np.nan,
            "theta_name": np.nan,
            "theta_str": np.nan,
            "gamma_power": np.nan,
            "alpha_contrast_weight": np.nan,
            "beta_contrast_sharpness": np.nan,
            "tau_clip": np.nan,
            "k_blur_kernel_size": np.nan,
            "edge_gain": np.nan,
            "blend_mix": np.nan,
            "train_loss": np.nan,
            "train_ce_loss": np.nan,
            "train_proto_loss": np.nan,
            "train_acc": np.nan,
            "val_size": 0,
            "val_loss_ce": np.nan,
            "val_acc": np.nan,
            "val_precision_macro": np.nan,
            "val_recall_macro": np.nan,
            "val_f1_macro": np.nan,
            "val_precision_weighted": np.nan,
            "val_recall_weighted": np.nan,
            "val_f1_weighted": np.nan,
            "val_log_loss": np.nan,
            "val_eval_time_s": np.nan,
            "val_auc_roc_macro_ovr": np.nan,
            "reward": np.nan,
            "bandit_counts_total": np.nan,
            "bandit_value_selected": np.nan,
            "participation_rounds_so_far": int(client["participation_rounds"]),
        })

    agg_weights = build_tempered_fedavg_weights(selected_clients_meta)
    fedavg_update(global_model, local_models, agg_weights)
    global_prototypes = aggregate_prototypes(proto_payloads, global_model.embed_dim)

    ds1_metrics = aggregate_by_dataset(round_local_rows, "ds1")
    ds2_metrics = aggregate_by_dataset(round_local_rows, "ds2")

    global_metrics = {
        "acc": weighted_mean(round_local_rows, "val_acc", "val_size"),
        "f1_macro": weighted_mean(round_local_rows, "val_f1_macro", "val_size"),
        "precision_macro": weighted_mean(round_local_rows, "val_precision_macro", "val_size"),
        "recall_macro": weighted_mean(round_local_rows, "val_recall_macro", "val_size"),
        "log_loss": weighted_mean(round_local_rows, "val_log_loss", "val_size"),
        "loss_ce": weighted_mean(round_local_rows, "val_loss_ce", "val_size"),
        "eval_time_s": weighted_mean(round_local_rows, "val_eval_time_s", "val_size"),
    }

    ds1_score = score_metric(ds1_metrics)
    ds2_score = score_metric(ds2_metrics)
    round_reward = (
        CFG["best_round_mass_ds1"] * ds1_score +
        CFG["best_round_mass_ds2"] * ds2_score +
        CFG["best_round_min_bonus"] * min(ds1_score, ds2_score)
    )

    history_local.extend(round_local_rows)
    history_global.append({
        "round": rnd,
        "round_time_s": float(time.time() - round_t0),
        "n_selected_clients": len(selected_ids),
        "active_fraction": float(len(selected_ids) / max(1, CLIENTS_TOTAL)),
        "global_reward": round_reward,
        "global_acc": global_metrics["acc"],
        "global_f1_macro": global_metrics["f1_macro"],
        "global_precision_macro": global_metrics["precision_macro"],
        "global_recall_macro": global_metrics["recall_macro"],
        "global_log_loss": global_metrics["log_loss"],
        "global_loss_ce": global_metrics["loss_ce"],
        "global_eval_time_s": global_metrics["eval_time_s"],
        "ds1_acc": ds1_metrics["acc"],
        "ds1_f1_macro": ds1_metrics["f1_macro"],
        "ds1_log_loss": ds1_metrics["log_loss"],
        "ds2_acc": ds2_metrics["acc"],
        "ds2_f1_macro": ds2_metrics["f1_macro"],
        "ds2_log_loss": ds2_metrics["log_loss"],
        "selected_ds1": str(selected_ds1),
        "selected_ds2": str(selected_ds2),
    })

    print(
        f"GLOBAL VAL (Round {rnd}) | "
        f"global_acc={global_metrics['acc']:.4f} | "
        f"global_f1={global_metrics['f1_macro']:.4f} | "
        f"ds1_acc={ds1_metrics['acc']:.4f} | "
        f"ds1_f1={ds1_metrics['f1_macro']:.4f} | "
        f"ds2_acc={ds2_metrics['acc']:.4f} | "
        f"ds2_f1={ds2_metrics['f1_macro']:.4f} | "
        f"reward={round_reward:.4f} | "
        f"round_time={history_global[-1]['round_time_s']:.1f}s"
    )

    if np.isfinite(round_reward) and round_reward > best_reward:
        best_reward = float(round_reward)
        best_round_saved = rnd
        best_model_state = {k: v.detach().cpu().clone() for k, v in global_model.state_dict().items()}
        best_global_prototypes = None if global_prototypes is None else {
            "proto": global_prototypes["proto"].detach().cpu().clone(),
            "mask": global_prototypes["mask"].detach().cpu().clone(),
            "counts": global_prototypes["counts"].detach().cpu().clone(),
        }
        best_theta_bandit_states = [copy.deepcopy(c["theta_bandit"].state_dict()) for c in clients]
        best_participation_bandit_ds1_state = copy.deepcopy(participation_bandit_ds1.state_dict())
        best_participation_bandit_ds2_state = copy.deepcopy(participation_bandit_ds2.state_dict())

if best_model_state is not None:
    global_model.load_state_dict({k: v.to(DEVICE) for k, v in best_model_state.items()})

if best_theta_bandit_states is not None:
    for c, sd in zip(clients, best_theta_bandit_states):
        c["theta_bandit"].load_state_dict(sd)

if best_participation_bandit_ds1_state is not None:
    participation_bandit_ds1.load_state_dict(best_participation_bandit_ds1_state)
if best_participation_bandit_ds2_state is not None:
    participation_bandit_ds2.load_state_dict(best_participation_bandit_ds2_state)

if best_global_prototypes is not None:
    global_prototypes = {
        "proto": best_global_prototypes["proto"].to(DEVICE),
        "mask": best_global_prototypes["mask"].to(DEVICE),
        "counts": best_global_prototypes["counts"].to(DEVICE),
    }

t_total = float(time.time() - t_global_start)
print("\n" + "=" * 118)
print(f"TRAINING COMPLETE | total_time={t_total:.1f}s | best_round={best_round_saved} | best_reward={best_reward:.4f}")
print("=" * 118)

glob_df = pd.DataFrame(history_global)
loc_df = pd.DataFrame(history_local)
participation_df = pd.DataFrame(participation_history)

print_table(glob_df, "GLOBAL per-round metrics", max_rows=20)
print_table(loc_df, "LOCAL per-client per-round metrics", max_rows=20)
print_table(participation_df, "Participation UCB history", max_rows=20)

add_table_to_csv(glob_df, "global_round_metrics_full")
add_table_to_csv(loc_df, "client_round_metrics_full")
add_table_to_csv(participation_df, "participation_ucb_history")

# ============================================================
# STEP 11: FINAL EVALUATION
# ============================================================
print("\n" + "=" * 118)
print("STEP 11: FINAL EVALUATION")
print("=" * 118)

def best_theta_for_client(client):
    arm = client["theta_bandit"].best_arm()
    name, theta = client["preset_bank"][arm]
    return arm, name, theta, client["theta_bandit"].values[arm]

def unique_theta_candidates(csubset):
    out = []
    seen = set()
    for c in csubset:
        if c["participation_rounds"] <= 0:
            continue
        arm, name, theta, val = best_theta_for_client(c)
        key = tuple([round(x, 6) if isinstance(x, float) else x for x in theta])
        if key not in seen:
            out.append((name, theta, float(val), c["gid"]))
            seen.add(key)
    return out

@torch.no_grad()
def evaluate_with_single_theta(model, loader, theta, use_tta=False):
    pre = theta_to_module(theta).to(DEVICE)
    return evaluate_full(model, loader, pre, theta, return_gates=False, use_tta=use_tta)

def rank_rlucb_candidates(model, val_loader, candidates):
    ranked = []
    for name, theta, est, gid in candidates:
        met, _, _ = evaluate_with_single_theta(model, val_loader, theta, use_tta=CFG["final_use_tta"])
        ranked.append({
            "theta_name": name,
            "theta": theta,
            "score": score_metric(met),
            "val_acc": safe_float(met.get("acc")),
            "val_f1": safe_float(met.get("f1_macro")),
            "estimated_bandit_value": float(est),
            "from_client_gid": int(gid),
        })
    ranked = sorted(ranked, key=lambda x: x["score"], reverse=True)
    return ranked

def equal_merge_metrics(m1, m2):
    out = {}
    keys = sorted(set(m1.keys()).union(set(m2.keys())))
    for k in keys:
        a, b = m1.get(k, np.nan), m2.get(k, np.nan)
        if np.isfinite(a) and np.isfinite(b):
            out[k] = float(np.mean([a, b]))
        elif np.isfinite(a):
            out[k] = float(a)
        elif np.isfinite(b):
            out[k] = float(b)
        else:
            out[k] = np.nan
    return out

client_summary_rows = []
for c in clients:
    arm, name, theta, val = best_theta_for_client(c)
    client_summary_rows.append({
        "client": f"client_{c['gid']}",
        "dataset": c["dataset"],
        "participation_rounds": int(c["participation_rounds"]),
        "best_theta_arm": int(arm),
        "best_theta_name": name,
        "best_theta_str": theta_str(theta),
        "estimated_theta_value": float(val),
        "theta_pulls": int(c["theta_bandit"].counts.sum()),
    })
client_theta_df = pd.DataFrame(client_summary_rows)
print_table(client_theta_df, "Client summary with RL-based participation and preset selection", max_rows=20)
add_table_to_csv(client_theta_df, "client_summary_rl_participation_rlucb_preset")

active_clients_ds1 = [c for c in clients if c["dataset"] == "ds1" and c["participation_rounds"] > 0]
active_clients_ds2 = [c for c in clients if c["dataset"] == "ds2" and c["participation_rounds"] > 0]

cand_ds1 = unique_theta_candidates(active_clients_ds1)
cand_ds2 = unique_theta_candidates(active_clients_ds2)

ranked_ds1 = rank_rlucb_candidates(global_model, val_loader_ds1, cand_ds1)
ranked_ds2 = rank_rlucb_candidates(global_model, val_loader_ds2, cand_ds2)

choice_ds1 = ranked_ds1[0]
choice_ds2 = ranked_ds2[0]

ranked_ds1_df = pd.DataFrame([{k: v for k, v in r.items() if k != "theta"} for r in ranked_ds1])
ranked_ds2_df = pd.DataFrame([{k: v for k, v in r.items() if k != "theta"} for r in ranked_ds2])

print_table(ranked_ds1_df, "Validation ranking of RL-UCB candidates — DS1", max_rows=10)
print_table(ranked_ds2_df, "Validation ranking of RL-UCB candidates — DS2", max_rows=10)
add_table_to_csv(ranked_ds1_df, "validation_ranking_rlucb_candidates_ds1")
add_table_to_csv(ranked_ds2_df, "validation_ranking_rlucb_candidates_ds2")

val_ds1, _, _ = evaluate_with_single_theta(global_model, val_loader_ds1, choice_ds1["theta"], use_tta=CFG["final_use_tta"])
test_ds1, y_ds1, p_ds1 = evaluate_with_single_theta(global_model, test_loader_ds1, choice_ds1["theta"], use_tta=CFG["final_use_tta"])

val_ds2, _, _ = evaluate_with_single_theta(global_model, val_loader_ds2, choice_ds2["theta"], use_tta=CFG["final_use_tta"])
test_ds2, y_ds2, p_ds2 = evaluate_with_single_theta(global_model, test_loader_ds2, choice_ds2["theta"], use_tta=CFG["final_use_tta"])

val_global = equal_merge_metrics(val_ds1, val_ds2)
test_global = equal_merge_metrics(test_ds1, test_ds2)

choice_df = pd.DataFrame([
    {
        "dataset": "ds1",
        "strategy": "rl_based_client_participation_plus_rlucb_preset_selection",
        "theta_names": str([choice_ds1["theta_name"]]),
        "score": choice_ds1["score"],
        "val_acc": choice_ds1["val_acc"],
        "val_f1": choice_ds1["val_f1"],
    },
    {
        "dataset": "ds2",
        "strategy": "rl_based_client_participation_plus_rlucb_preset_selection",
        "theta_names": str([choice_ds2["theta_name"]]),
        "score": choice_ds2["score"],
        "val_acc": choice_ds2["val_acc"],
        "val_f1": choice_ds2["val_f1"],
    },
])
print_table(choice_df, "Final strategy after RL-based participation training", max_rows=5)
add_table_to_csv(choice_df, "final_theta_strategy_choice_rl_participation")

# ============================================================
# STEP 12: EXTENDED METRICS + CALIBRATION + ERROR ANALYSIS
# ============================================================
print("\n" + "=" * 118)
print("STEP 12: EXTENDED METRICS + CALIBRATION + ERROR ANALYSIS")
print("=" * 118)

def top_label_calibration(y_true, p_pred, n_bins=12):
    conf = np.max(p_pred, axis=1)
    pred = np.argmax(p_pred, axis=1)
    correct = (pred == y_true).astype(np.float32)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(conf, bins) - 1
    bin_ids = np.clip(bin_ids, 0, n_bins - 1)

    rows = []
    ece = 0.0
    mce = 0.0
    N = len(y_true)

    for b in range(n_bins):
        m = (bin_ids == b)
        if m.sum() == 0:
            rows.append({
                "bin_id": b,
                "bin_left": bins[b],
                "bin_right": bins[b + 1],
                "bin_confidence": np.nan,
                "bin_accuracy": np.nan,
                "bin_gap": np.nan,
                "bin_count": 0,
            })
            continue

        bc = float(conf[m].mean())
        ba = float(correct[m].mean())
        gap = abs(ba - bc)
        ece += (m.sum() / N) * gap
        mce = max(mce, gap)
        rows.append({
            "bin_id": b,
            "bin_left": bins[b],
            "bin_right": bins[b + 1],
            "bin_confidence": bc,
            "bin_accuracy": ba,
            "bin_gap": gap,
            "bin_count": int(m.sum()),
        })

    return pd.DataFrame(rows), float(ece), float(mce)

def multiclass_brier_score(y_true, p_pred, num_classes):
    y_oh = np.eye(num_classes)[y_true]
    return float(np.mean(np.sum((p_pred - y_oh) ** 2, axis=1)))

def classwise_confusion_metrics(y_true, p_pred):
    y_hat = np.argmax(p_pred, axis=1)
    cm = confusion_matrix(y_true, y_hat, labels=list(range(NUM_CLASSES)))
    total = cm.sum()

    rows = []
    for c, lab in enumerate(labels):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        tn = total - tp - fp - fn

        ppv = tp / max(1, (tp + fp))
        npv = tn / max(1, (tn + fn))
        sensitivity = tp / max(1, (tp + fn))
        specificity = tn / max(1, (tn + fp))
        fpr = fp / max(1, (fp + tn))
        fnr = fn / max(1, (fn + tp))
        jacc = tp / max(1, (tp + fp + fn))
        support = tp + fn
        prevalence = support / max(1, total)
        bal = 0.5 * (sensitivity + specificity)

        rows.append({
            "class_id": c,
            "class_name": lab,
            "support": int(support),
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
            "tn": int(tn),
            "prevalence": float(prevalence),
            "ppv": float(ppv),
            "npv": float(npv),
            "recall": float(sensitivity),
            "specificity": float(specificity),
            "fpr": float(fpr),
            "fnr": float(fnr),
            "jaccard": float(jacc),
            "balanced_acc": float(bal),
        })

    return cm, pd.DataFrame(rows)

def top_confusion_pairs(y_true, p_pred):
    y_hat = np.argmax(p_pred, axis=1)
    cm = confusion_matrix(y_true, y_hat, labels=list(range(NUM_CLASSES)))
    rows = []
    for i, lab_i in enumerate(labels):
        for j, lab_j in enumerate(labels):
            if i == j:
                continue
            rows.append({
                "true_class": lab_i,
                "pred_class": lab_j,
                "count": int(cm[i, j]),
            })
    return pd.DataFrame(rows).sort_values("count", ascending=False).reset_index(drop=True)

def wrong_right_confidence_table(y_true, p_pred):
    pred = np.argmax(p_pred, axis=1)
    conf = np.max(p_pred, axis=1)
    ok = pred == y_true
    return pd.DataFrame({
        "confidence": conf,
        "correct": ok.astype(int),
        "status": np.where(ok, "correct", "wrong"),
        "true_label": [id2label[int(x)] for x in y_true],
        "pred_label": [id2label[int(x)] for x in pred],
    })

def compute_extended_bundle(y_true, p_pred, dataset_name):
    y_hat = np.argmax(p_pred, axis=1)
    cm, class_df = classwise_confusion_metrics(y_true, p_pred)
    cal_df, ece, mce = top_label_calibration(y_true, p_pred, n_bins=CFG["calibration_bins"])
    conf_df = top_confusion_pairs(y_true, p_pred)
    conf_table = wrong_right_confidence_table(y_true, p_pred)

    scalar = {
        "dataset": dataset_name,
        "acc": float(accuracy_score(y_true, y_hat)),
        "balanced_acc": float(balanced_accuracy_score(y_true, y_hat)),
        "precision_macro": float(precision_score(y_true, y_hat, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_hat, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_hat, average="macro", zero_division=0)),
        "precision_weighted": float(precision_score(y_true, y_hat, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(y_true, y_hat, average="weighted", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_hat, average="weighted", zero_division=0)),
        "log_loss": float(log_loss(y_true, p_pred, labels=list(range(NUM_CLASSES)))),
        "mcc": float(matthews_corrcoef(y_true, y_hat)),
        "kappa": float(cohen_kappa_score(y_true, y_hat)),
        "jaccard_macro": float(jaccard_score(y_true, y_hat, average="macro")),
        "ppv_macro": float(class_df["ppv"].mean()),
        "npv_macro": float(class_df["npv"].mean()),
        "specificity_macro": float(class_df["specificity"].mean()),
        "fpr_macro": float(class_df["fpr"].mean()),
        "fnr_macro": float(class_df["fnr"].mean()),
        "ece": float(ece),
        "mce": float(mce),
        "brier_multi": float(multiclass_brier_score(y_true, p_pred, NUM_CLASSES)),
    }
    scalar.update(_auc_metrics(y_true, p_pred, NUM_CLASSES))
    return scalar, class_df, cal_df, conf_df, conf_table, cm

ext_ds1, class_df1, cal_df1, conf_pairs1, conf_table1, cm1 = compute_extended_bundle(y_ds1, p_ds1, "ds1_test")
ext_ds2, class_df2, cal_df2, conf_pairs2, conf_table2, cm2 = compute_extended_bundle(y_ds2, p_ds2, "ds2_test")

ext_df = pd.DataFrame([ext_ds1, ext_ds2])

print_table(ext_df, "Extended TEST metrics (DS1 vs DS2)", max_rows=5)
print_table(class_df1, "Classwise metrics — DS1 TEST", max_rows=10)
print_table(class_df2, "Classwise metrics — DS2 TEST", max_rows=10)
print_table(conf_pairs1.head(10), "Top confusion pairs — DS1 TEST", max_rows=10)
print_table(conf_pairs2.head(10), "Top confusion pairs — DS2 TEST", max_rows=10)
print_table(cal_df1, "Calibration bins — DS1 TEST", max_rows=15)
print_table(cal_df2, "Calibration bins — DS2 TEST", max_rows=15)

add_table_to_csv(ext_df, "extended_test_metrics_ds1_ds2")
add_table_to_csv(class_df1, "classwise_metrics_ds1_test")
add_table_to_csv(class_df2, "classwise_metrics_ds2_test")
add_table_to_csv(cal_df1, "calibration_bins_ds1_test")
add_table_to_csv(cal_df2, "calibration_bins_ds2_test")
add_table_to_csv(conf_pairs1, "top_confusions_ds1_test")
add_table_to_csv(conf_pairs2, "top_confusions_ds2_test")
add_table_to_csv(conf_table1, "confidence_correctness_ds1_test")
add_table_to_csv(conf_table2, "confidence_correctness_ds2_test")

test_ds1.update(ext_ds1)
test_ds2.update(ext_ds2)

def compact_metrics(m):
    keep = [
        "acc", "balanced_acc", "precision_macro", "recall_macro", "f1_macro",
        "precision_weighted", "recall_weighted", "f1_weighted", "log_loss",
        "auc_roc_macro_ovr", "mcc", "kappa", "ppv_macro", "npv_macro",
        "specificity_macro", "ece", "mce", "brier_multi", "loss_ce", "eval_time_s"
    ]
    out = {}
    for k in keep:
        if k in m and np.isfinite(m[k]):
            out[k] = float(m[k])
    return out

paper_df = pd.DataFrame([
    {"setting": "ARCF-Net RL-Based Participation", "split": "VAL",  "dataset": "ds1",          **compact_metrics(val_ds1)},
    {"setting": "ARCF-Net RL-Based Participation", "split": "VAL",  "dataset": "ds2",          **compact_metrics(val_ds2)},
    {"setting": "ARCF-Net RL-Based Participation", "split": "VAL",  "dataset": "global_equal", **compact_metrics(val_global)},
    {"setting": "ARCF-Net RL-Based Participation", "split": "TEST", "dataset": "ds1",          **compact_metrics(test_ds1)},
    {"setting": "ARCF-Net RL-Based Participation", "split": "TEST", "dataset": "ds2",          **compact_metrics(test_ds2)},
    {"setting": "ARCF-Net RL-Based Participation", "split": "TEST", "dataset": "global_equal", **compact_metrics(test_global)},
])

print_table(paper_df, "VAL + TEST tables", max_rows=10)
add_table_to_csv(paper_df, "paper_ready_metrics")

print("\nSelection summary:")
print(f"- Best round: {best_round_saved} | best_reward={best_reward:.4f}")
print(f"- DS1 final strategy: rl_based_client_participation_plus_rlucb_preset_selection | names={[choice_ds1['theta_name']]}")
print(f"- DS2 final strategy: rl_based_client_participation_plus_rlucb_preset_selection | names={[choice_ds2['theta_name']]}")

# ============================================================
# STEP 13: PREPROCESSING VALIDATION
# ============================================================
print("\n" + "=" * 118)
print("STEP 13: PREPROCESSING VALIDATION")
print("=" * 118)

@torch.no_grad()
def entropy_per_image(x01):
    gray = x01.mean(dim=1)
    B = gray.shape[0]
    ent = []
    for i in range(B):
        g = (gray[i].detach().cpu().numpy() * 255).astype(np.uint8)
        hist = np.bincount(g.flatten(), minlength=256).astype(np.float32)
        p = hist / np.clip(hist.sum(), 1, None)
        p = p[p > 0]
        ent.append(float(-(p * np.log2(p)).sum()))
    return np.array(ent)

@torch.no_grad()
def edge_energy(x01, kernel):
    kernel = kernel.to(device=x01.device, dtype=x01.dtype)
    gray = x01.mean(dim=1, keepdim=True)
    lap = F.conv2d(F.pad(gray, (1, 1, 1, 1), mode="reflect"), kernel).abs()
    return lap.mean(dim=(1, 2, 3)).detach().cpu().numpy()

@torch.no_grad()
def contrast_proxy(x01):
    gray = x01.mean(dim=1)
    return gray.std(dim=(1, 2)).detach().cpu().numpy()

@torch.no_grad()
def run_preproc_validation(frame, preproc, sample_n=400):
    n = min(sample_n, len(frame))
    if n <= 0:
        return pd.DataFrame(), pd.DataFrame()

    idx = np.random.choice(len(frame), size=n, replace=False)
    ds = MRIDataset(frame, indices=idx.tolist(), tfms=EVAL_TFMS, source_id=0)

    xs = []
    for i in range(len(ds)):
        x, _, _, _ = ds[i]
        xs.append(x)
    x = torch.stack(xs).to(DEVICE)

    x_after = preproc(x).clamp(0, 1)
    lap_kernel = preproc.lap if hasattr(preproc, "lap") else RACEFELCM().to(DEVICE).lap

    ee_before = edge_energy(x, lap_kernel)
    ee_after  = edge_energy(x_after, lap_kernel)
    ent_before = entropy_per_image(x)
    ent_after  = entropy_per_image(x_after)
    con_before = contrast_proxy(x)
    con_after  = contrast_proxy(x_after)

    dfm = pd.DataFrame({
        "edge_energy_before": ee_before,
        "edge_energy_after": ee_after,
        "entropy_before": ent_before,
        "entropy_after": ent_after,
        "contrast_before": con_before,
        "contrast_after": con_after,
        "edge_gain_ratio": (ee_after / np.clip(ee_before, 1e-9, None)),
        "entropy_delta": (ent_after - ent_before),
        "contrast_delta": (con_after - con_before),
    })
    summary = dfm.agg(["mean", "std", "min", "max"]).T.reset_index().rename(columns={"index": "metric"})
    return dfm, summary

preproc_ds1 = theta_to_module(choice_ds1["theta"]).to(DEVICE)
preproc_ds2 = theta_to_module(choice_ds2["theta"]).to(DEVICE)

preproc_df1, preproc_summary_df1 = run_preproc_validation(val1, preproc_ds1, CFG["preproc_val_sample_n"])
preproc_df2, preproc_summary_df2 = run_preproc_validation(val2, preproc_ds2, CFG["preproc_val_sample_n"])

print_table(preproc_summary_df1, "Preprocessing validation summary (DS1 VAL sample)", max_rows=15)
print_table(preproc_summary_df2, "Preprocessing validation summary (DS2 VAL sample)", max_rows=15)
add_table_to_csv(preproc_summary_df1, "preprocessing_validation_summary_ds1")
add_table_to_csv(preproc_summary_df2, "preprocessing_validation_summary_ds2")

# ============================================================
# STEP 14: PARTICIPATION SUMMARY
# ============================================================
print("\n" + "=" * 118)
print("STEP 14: PARTICIPATION SUMMARY")
print("=" * 118)

participation_summary_rows = []
for gid_ in ds1_global_ids:
    arm = ds1_gid_to_arm[gid_]
    participation_summary_rows.append({
        "client": f"client_{gid_}",
        "dataset": "ds1",
        "participation_rounds": int(clients[gid_]["participation_rounds"]),
        "participation_bandit_count": int(participation_bandit_ds1.counts[arm]),
        "participation_bandit_value": float(participation_bandit_ds1.values[arm]),
    })
for gid_ in ds2_global_ids:
    arm = ds2_gid_to_arm[gid_]
    participation_summary_rows.append({
        "client": f"client_{gid_}",
        "dataset": "ds2",
        "participation_rounds": int(clients[gid_]["participation_rounds"]),
        "participation_bandit_count": int(participation_bandit_ds2.counts[arm]),
        "participation_bandit_value": float(participation_bandit_ds2.values[arm]),
    })

participation_summary_df = pd.DataFrame(participation_summary_rows)
print_table(participation_summary_df, "Participation summary per client", max_rows=20)
add_table_to_csv(participation_summary_df, "participation_summary_per_client")

# ============================================================
# STEP 15: SAVE CHECKPOINT + CSV
# ============================================================
print("\n" + "=" * 118)
print("STEP 15: SAVING CHECKPOINT + CSV")
print("=" * 118)

client_process_manifest = []
for c in clients:
    df_src = train1 if c["dataset"] == "ds1" else train2
    client_process_manifest.append({
        "gid": c["gid"],
        "local_id": c["local_id"],
        "dataset": c["dataset"],
        "source_id": c["source_id"],
        "n_train": c["n_train"],
        "n_tune": c["n_tune"],
        "n_val": c["n_val"],
        "train_indices": c["train_indices"],
        "tune_indices": c["tune_indices"],
        "val_indices": c["val_indices"],
        "train_paths": df_src.loc[c["train_indices"], "path"].tolist(),
        "tune_paths": df_src.loc[c["tune_indices"], "path"].tolist(),
        "val_paths": df_src.loc[c["val_indices"], "path"].tolist(),
        "class_counts": c["class_counts"],
        "theta_bandit_state": c["theta_bandit"].state_dict(),
        "preset_bank": c["preset_bank"],
        "participation_rounds": int(c["participation_rounds"]),
    })

all_df = pd.DataFrame(ALL_ROWS)

checkpoint = {
    "method_info": METHOD_INFO,

    "state_dict": {k: v.detach().cpu() for k, v in global_model.state_dict().items()},
    "best_model_state_dict": {k: v.detach().cpu() for k, v in global_model.state_dict().items()},

    "config": CFG,
    "seed": SEED,
    "device_used": str(DEVICE),
    "backbone_name": "resnet50",
    "fusion_name": "CRAF",
    "preprocessing_name": "RACE-FELCM",

    "dataset1_raw_root": DS1_ROOT,
    "dataset2_root": DS2_ROOT,
    "labels": labels,
    "label2id": label2id,
    "id2label": id2label,
    "num_classes": NUM_CLASSES,
    "theta_fullforms": THETA_FULLFORMS,

    "full_dataset_df1": frame_to_manifest(df1),
    "full_dataset_df2": frame_to_manifest(df2),

    "split_train1": frame_to_manifest(train1),
    "split_val1": frame_to_manifest(val1),
    "split_test1": frame_to_manifest(test1),
    "split_train2": frame_to_manifest(train2),
    "split_val2": frame_to_manifest(val2),
    "split_test2": frame_to_manifest(test2),

    "fixed_clients_ds1": n_clients_ds1,
    "fixed_clients_ds2": n_clients_ds2,
    "clients_total": CLIENTS_TOTAL,
    "active_clients_per_round_ds1": CFG["active_clients_per_round_ds1"],
    "active_clients_per_round_ds2": CFG["active_clients_per_round_ds2"],

    "ds1_global_ids": ds1_global_ids,
    "ds2_global_ids": ds2_global_ids,
    "participation_bandit_ds1_state": participation_bandit_ds1.state_dict(),
    "participation_bandit_ds2_state": participation_bandit_ds2.state_dict(),
    "participation_history": participation_df.to_dict(orient="list"),
    "participation_summary": participation_summary_df.to_dict(orient="list"),

    "client_indices_ds1": client_indices_ds1,
    "client_indices_ds2": client_indices_ds2,
    "client_process_manifest": client_process_manifest,

    "preset_bank_ds1": PRESET_BANK_DS1,
    "preset_bank_ds2": PRESET_BANK_DS2,

    "best_round_saved": best_round_saved,
    "best_reward": best_reward,
    "history_global": glob_df.to_dict(orient="list"),
    "history_local": loc_df.to_dict(orient="list"),
    "total_training_time_s": t_total,

    "best_theta_bandit_states": best_theta_bandit_states,
    "best_participation_bandit_ds1_state": best_participation_bandit_ds1_state,
    "best_participation_bandit_ds2_state": best_participation_bandit_ds2_state,
    "client_summary_rl_participation_rlucb_preset": client_theta_df.to_dict(orient="list"),

    "final_choice_ds1": {
        "strategy": "rl_based_client_participation_plus_rlucb_preset_selection",
        "theta_names": [choice_ds1["theta_name"]],
        "theta_list": [choice_ds1["theta"]],
        "score": choice_ds1["score"],
        "val_acc": choice_ds1["val_acc"],
        "val_f1": choice_ds1["val_f1"],
    },
    "final_choice_ds2": {
        "strategy": "rl_based_client_participation_plus_rlucb_preset_selection",
        "theta_names": [choice_ds2["theta_name"]],
        "theta_list": [choice_ds2["theta"]],
        "score": choice_ds2["score"],
        "val_acc": choice_ds2["val_acc"],
        "val_f1": choice_ds2["val_f1"],
    },

    "best_global_prototypes": None if best_global_prototypes is None else {
        "proto": best_global_prototypes["proto"],
        "mask": best_global_prototypes["mask"],
        "counts": best_global_prototypes["counts"],
    },

    "final_val_ds1": val_ds1,
    "final_val_ds2": val_ds2,
    "final_val_global": val_global,
    "final_test_ds1": test_ds1,
    "final_test_ds2": test_ds2,
    "final_test_global_equal": test_global,
    "paper_ready_metrics": paper_df.to_dict(orient="list"),
    "extended_test_metrics_ds1_ds2": ext_df.to_dict(orient="list"),
    "classwise_metrics_ds1_test": class_df1.to_dict(orient="list"),
    "classwise_metrics_ds2_test": class_df2.to_dict(orient="list"),
    "calibration_bins_ds1_test": cal_df1.to_dict(orient="list"),
    "calibration_bins_ds2_test": cal_df2.to_dict(orient="list"),
    "top_confusions_ds1_test": conf_pairs1.to_dict(orient="list"),
    "top_confusions_ds2_test": conf_pairs2.to_dict(orient="list"),
    "confidence_correctness_ds1_test": conf_table1.to_dict(orient="list"),
    "confidence_correctness_ds2_test": conf_table2.to_dict(orient="list"),
    "client_distribution_table": dist_df.to_dict(orient="list"),
    "global_round_metrics_table": glob_df.to_dict(orient="list"),
    "local_round_metrics_table": loc_df.to_dict(orient="list"),
    "participation_ucb_history_table": participation_df.to_dict(orient="list"),
    "all_tables_long_format": all_df.to_dict(orient="list"),

    "ds1_test_y_true": y_ds1,
    "ds1_test_p_pred": p_ds1,
    "ds1_test_y_hat": np.argmax(p_ds1, axis=1),
    "ds2_test_y_true": y_ds2,
    "ds2_test_p_pred": p_ds2,
    "ds2_test_y_hat": np.argmax(p_ds2, axis=1),

    "preprocessing_validation_ds1_raw": preproc_df1.to_dict(orient="list") if len(preproc_df1) else {},
    "preprocessing_validation_ds2_raw": preproc_df2.to_dict(orient="list") if len(preproc_df2) else {},
    "preprocessing_validation_summary_ds1": preproc_summary_df1.to_dict(orient="list") if len(preproc_summary_df1) else {},
    "preprocessing_validation_summary_ds2": preproc_summary_df2.to_dict(orient="list") if len(preproc_summary_df2) else {},
}

torch.save(checkpoint, MODEL_PATH)
print(f"Saved checkpoint: {MODEL_PATH}")

all_df.to_csv(CSV_PATH, index=False)
print(f"Saved CSV: {CSV_PATH}")

print("\nDONE")
print(f"Method: {METHOD_INFO['acronym']} = {METHOD_INFO['full_form']}")
print(f"Ablation: {METHOD_INFO['ablation_name']}")
print(f"Backbone: {METHOD_INFO['backbone_full_form']}")
print(f"Best round: {best_round_saved}")
print(f"Client pool => DS1={n_clients_ds1}, DS2={n_clients_ds2}, TOTAL={CLIENTS_TOTAL}")
print(f"RL-selected active per round => DS1={CFG['active_clients_per_round_ds1']}, DS2={CFG['active_clients_per_round_ds2']}")
print(f"Rounds completed: {CFG['rounds']}")
print(f"Global TEST acc: {safe_float(test_global.get('acc')):.4f}")
print(f"Global TEST f1_macro: {safe_float(test_global.get('f1_macro')):.4f}")
print(f"DS1 TEST acc: {safe_float(test_ds1.get('acc')):.4f}")
print(f"DS2 TEST acc: {safe_float(test_ds2.get('acc')):.4f}")
print(f"DS1 final strategy: rl_based_client_participation_plus_rlucb_preset_selection | names={[choice_ds1['theta_name']]}")
print(f"DS2 final strategy: rl_based_client_participation_plus_rlucb_preset_selection | names={[choice_ds2['theta_name']]}")
```

    ======================================================================================================================
    ARCF-Net ABLATION 6: RL-BASED PARTICIPATION + RL-UCB PRESET SELECTION + NO PLOTS
    ======================================================================================================================
    ENV: KAGGLE | DEVICE: cuda | torch=2.10.0+cu128
    ======================================================================================================================
    
    ======================================================================================================================
    STEP 0: ACCESS DATASETS
    ======================================================================================================================
    Dataset-1 RAW root detected:
      /kaggle/input/pmram-bangladeshi-brain-cancer-mri-dataset/PMRAM Bangladeshi Brain Cancer - MRI Dataset/PMRAM Bangladeshi Brain Cancer - MRI Dataset/Raw Data/Raw
    Dataset-2 root detected:
      /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset
    
    ======================================================================================================================
    STEP 1: BUILD DATA MANIFESTS
    ======================================================================================================================
    ds1_raw: 512Glioma -> glioma | 373 images
    ds1_raw: 512Meningioma -> meningioma | 363 images
    ds1_raw: 512Normal -> notumor | 396 images
    ds1_raw: 512Pituitary -> pituitary | 373 images
    ds2: glioma -> glioma | 1621 images
    ds2: meningioma -> meningioma | 1646 images
    ds2: notumor -> notumor | 2000 images
    ds2: pituitary -> pituitary | 1764 images
    
    Dataset-1 images: 1505
    label
    glioma        373
    meningioma    363
    notumor       396
    pituitary     373
    
    Dataset-2 images: 7031
    label
    glioma        1621
    meningioma    1646
    notumor       2000
    pituitary     1764
    
    ======================================================================================================================
    STEP 2: TRAIN / VAL / TEST SPLIT
    ======================================================================================================================
    DS1 TRAIN: 1053 | VAL: 226 | TEST: 226
    DS2 TRAIN: 4921 | VAL: 1055 | TEST: 1055
    
    ======================================================================================================================
    STEP 2.5: SANITY / LEAKAGE CHECKS
    ======================================================================================================================
    
    ----------------------------------------------------------------------------------------------------------------------
    Leakage / Sanity Summary — ds1
    ----------------------------------------------------------------------------------------------------------------------
     path_overlap_train_val  path_overlap_train_test  path_overlap_val_test  unique_paths_train  unique_paths_val  unique_paths_test  filename_overlap_train_val  filename_overlap_train_test  filename_overlap_val_test  subset_hash_train_val  subset_hash_train_test  subset_hash_val_test  subset_hash_n_train  subset_hash_n_val  subset_hash_n_test
                          0                        0                      0                1053               226                226                           0                            0                          0                      5                       5                     6                  298                222                 224
    
    ----------------------------------------------------------------------------------------------------------------------
    Leakage / Sanity Summary — ds2
    ----------------------------------------------------------------------------------------------------------------------
     path_overlap_train_val  path_overlap_train_test  path_overlap_val_test  unique_paths_train  unique_paths_val  unique_paths_test  filename_overlap_train_val  filename_overlap_train_test  filename_overlap_val_test  subset_hash_train_val  subset_hash_train_test  subset_hash_val_test  subset_hash_n_train  subset_hash_n_val  subset_hash_n_test
                          0                        0                      0                4921              1055               1055                           0                            0                          0                      0                       3                     4                  299                298                 299
    
    ======================================================================================================================
    STEP 3: RL-UCB BANDIT
    ======================================================================================================================
    
    ======================================================================================================================
    STEP 4: FIXED NON-IID CLIENT POOL
    ======================================================================================================================
    Client pool DS1: 3
    Client pool DS2: 3
    
    ======================================================================================================================
    STEP 5: DATA LOADERS
    ======================================================================================================================
    ds1 | client_0 | train=490 | tune=77 | val=67
    ds1 | client_1 | train=125 | tune=20 | val=18
    ds1 | client_2 | train=198 | tune=31 | val=27
    ds2 | client_3 | train=629 | tune=98 | val=86
    ds2 | client_4 | train=527 | tune=82 | val=72
    ds2 | client_5 | train=2653 | tune=412 | val=362
    
    ----------------------------------------------------------------------------------------------------------------------
    Client class distribution
    ----------------------------------------------------------------------------------------------------------------------
      client dataset  total_train  total_tune  total_val                                                                              preset_bank_names  glioma  meningioma  notumor  pituitary
    client_0     ds1          490          77         67 ['race_balanced', 'race_sharp', 'race_texture', 'race_robust', 'race_edge_plus', 'race_focus']     111          46      176        157
    client_1     ds1          125          20         18 ['race_balanced', 'race_sharp', 'race_texture', 'race_robust', 'race_edge_plus', 'race_focus']      75           4        8         38
    client_2     ds1          198          31         27 ['race_balanced', 'race_sharp', 'race_texture', 'race_robust', 'race_edge_plus', 'race_focus']      16         147       30          5
    client_3     ds2          629          98         86                                ['race_soft', 'race_balanced', 'race_robust', 'race_smoothmix']      12         197      416          4
    client_4     ds2          527          82         72                                ['race_soft', 'race_balanced', 'race_robust', 'race_smoothmix']     202           4      284         37
    client_5     ds2         2653         412        362                                ['race_soft', 'race_balanced', 'race_robust', 'race_smoothmix']     665         691      383        914
    Augmentation: ON
    Preprocessing: ON
    Total client pool: 6
    
    ======================================================================================================================
    STEP 5.5: RL-BASED PARTICIPATION POLICY
    ======================================================================================================================
    
    ======================================================================================================================
    STEP 6: PREPROCESSING — RACE-FELCM
    ======================================================================================================================
    
    ======================================================================================================================
    STEP 7: MODEL — ResNet-50 + CRAF Fusion
    ======================================================================================================================
    Backbone: ResNet-50 | pretrained_loaded=True
    Total params: 25,790,855
    Trainable params: 2,282,823 (8.85%)
    
    ======================================================================================================================
    STEP 8: LOSSES + PROTOTYPE SHARING + EVAL HELPERS
    ======================================================================================================================
    
    ======================================================================================================================
    STEP 9: TUNE-AWARE RL-UCB PRESET SELECTION
    ======================================================================================================================
    
    ======================================================================================================================
    STEP 10: TRUE FEDERATED TRAINING
    ======================================================================================================================
    Client pool => DS1=3, DS2=3, TOTAL=6
    RL-selected active clients each round => DS1=2, DS2=2
    Rounds: 15 | Local epochs: 2
    FedProx μ=0.01 | Proto λ=0.12
    Tempered FedAvg exponent = 0.50
    
    ======================================================================================================================
    ROUND 1/15 | selected_ds1=[2, 1] | selected_ds2=[5, 4]
    ======================================================================================================================
    Client 2 (ds1) | train_acc=0.6919 | val_acc=0.7778 | val_f1=0.6272 | val_auc=0.9041 | reward=0.6648 | theta=race_focus (g=1.10, a=0.36, b=6.40, t=2.70, k=3, eg=0.17, mix=0.86)
    Client 1 (ds1) | train_acc=0.5840 | val_acc=0.9444 | val_f1=0.8730 | val_auc=nan | reward=0.8909 | theta=race_focus (g=1.10, a=0.36, b=6.40, t=2.70, k=3, eg=0.17, mix=0.86)
    Client 5 (ds2) | train_acc=0.7680 | val_acc=0.8564 | val_f1=0.8526 | val_auc=0.9677 | reward=0.8535 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    Client 4 (ds2) | train_acc=0.7732 | val_acc=0.9722 | val_f1=0.9232 | val_auc=nan | reward=0.9354 | theta=race_smoothmix (g=0.90, a=0.20, b=4.00, t=2.10, k=7, eg=0.07, mix=0.70)
    GLOBAL VAL (Round 1) | global_acc=0.8727 | global_f1=0.8512 | ds1_acc=0.8444 | ds1_f1=0.7255 | ds2_acc=0.8756 | ds2_f1=0.8643 | reward=0.9245 | round_time=124.2s
    
    ======================================================================================================================
    ROUND 2/15 | selected_ds1=[0, 1] | selected_ds2=[3, 4]
    ======================================================================================================================
    Client 0 (ds1) | train_acc=0.8388 | val_acc=0.8806 | val_f1=0.8252 | val_auc=0.9838 | reward=0.8390 | theta=race_focus (g=1.10, a=0.36, b=6.40, t=2.70, k=3, eg=0.17, mix=0.86)
    Client 1 (ds1) | train_acc=0.8360 | val_acc=0.9444 | val_f1=0.9585 | val_auc=nan | reward=0.9550 | theta=race_robust (g=1.08, a=0.22, b=4.50, t=2.80, k=5, eg=0.11, mix=0.76)
    Client 3 (ds2) | train_acc=0.8768 | val_acc=0.9535 | val_f1=0.7244 | val_auc=0.9232 | reward=0.7816 | theta=race_smoothmix (g=0.90, a=0.20, b=4.00, t=2.10, k=7, eg=0.07, mix=0.70)
    Client 4 (ds2) | train_acc=0.9535 | val_acc=0.9722 | val_f1=0.7145 | val_auc=nan | reward=0.7789 | theta=race_robust (g=1.08, a=0.22, b=4.50, t=2.80, k=5, eg=0.11, mix=0.76)
    GLOBAL VAL (Round 2) | global_acc=0.9383 | global_f1=0.7666 | ds1_acc=0.8941 | ds1_f1=0.8534 | ds2_acc=0.9620 | ds2_f1=0.7199 | reward=0.9391 | round_time=69.3s
    
    ======================================================================================================================
    ROUND 3/15 | selected_ds1=[0, 2] | selected_ds2=[5, 3]
    ======================================================================================================================
    Client 0 (ds1) | train_acc=0.8449 | val_acc=0.8955 | val_f1=0.8306 | val_auc=0.9875 | reward=0.8468 | theta=race_edge_plus (g=1.02, a=0.32, b=6.00, t=2.60, k=3, eg=0.15, mix=0.84)
    Client 2 (ds1) | train_acc=0.8081 | val_acc=0.8148 | val_f1=0.4472 | val_auc=0.9489 | reward=0.5391 | theta=race_robust (g=1.08, a=0.22, b=4.50, t=2.80, k=5, eg=0.11, mix=0.76)
    Client 5 (ds2) | train_acc=0.8615 | val_acc=0.9503 | val_f1=0.9509 | val_auc=0.9919 | reward=0.9508 | theta=race_robust (g=1.08, a=0.22, b=4.50, t=2.80, k=5, eg=0.11, mix=0.76)
    Client 3 (ds2) | train_acc=0.8903 | val_acc=0.9535 | val_f1=0.7250 | val_auc=0.9751 | reward=0.7821 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    GLOBAL VAL (Round 3) | global_acc=0.9373 | global_f1=0.8751 | ds1_acc=0.8723 | ds1_f1=0.7205 | ds2_acc=0.9509 | ds2_f1=0.9075 | reward=0.9522 | round_time=155.9s
    
    ======================================================================================================================
    ROUND 4/15 | selected_ds1=[1, 0] | selected_ds2=[5, 4]
    ======================================================================================================================
    Client 1 (ds1) | train_acc=0.8480 | val_acc=1.0000 | val_f1=1.0000 | val_auc=nan | reward=1.0000 | theta=race_edge_plus (g=1.02, a=0.32, b=6.00, t=2.60, k=3, eg=0.15, mix=0.84)
    Client 0 (ds1) | train_acc=0.9378 | val_acc=0.9403 | val_f1=0.8967 | val_auc=0.9929 | reward=0.9076 | theta=race_texture (g=0.92, a=0.34, b=5.80, t=2.30, k=7, eg=0.10, mix=0.80)
    Client 5 (ds2) | train_acc=0.9252 | val_acc=0.9558 | val_f1=0.9550 | val_auc=0.9977 | reward=0.9552 | theta=race_smoothmix (g=0.90, a=0.20, b=4.00, t=2.10, k=7, eg=0.07, mix=0.70)
    Client 4 (ds2) | train_acc=0.9440 | val_acc=1.0000 | val_f1=1.0000 | val_auc=nan | reward=1.0000 | theta=race_soft (g=0.95, a=0.18, b=3.80, t=2.20, k=3, eg=0.08, mix=0.72)
    GLOBAL VAL (Round 4) | global_acc=0.9615 | global_f1=0.9553 | ds1_acc=0.9529 | ds1_f1=0.9186 | ds2_acc=0.9631 | ds2_f1=0.9625 | reward=1.0840 | round_time=148.7s
    
    ======================================================================================================================
    ROUND 5/15 | selected_ds1=[1, 0] | selected_ds2=[3, 5]
    ======================================================================================================================
    Client 1 (ds1) | train_acc=0.9200 | val_acc=1.0000 | val_f1=1.0000 | val_auc=nan | reward=1.0000 | theta=race_sharp (g=1.04, a=0.30, b=5.20, t=2.50, k=5, eg=0.12, mix=0.82)
    Client 0 (ds1) | train_acc=0.9480 | val_acc=0.9701 | val_f1=0.9344 | val_auc=0.9970 | reward=0.9433 | theta=race_sharp (g=1.04, a=0.30, b=5.20, t=2.50, k=5, eg=0.12, mix=0.82)
    Client 3 (ds2) | train_acc=0.9587 | val_acc=0.9884 | val_f1=0.7455 | val_auc=0.9998 | reward=0.8062 | theta=race_robust (g=1.08, a=0.22, b=4.50, t=2.80, k=5, eg=0.11, mix=0.76)
    Client 5 (ds2) | train_acc=0.9331 | val_acc=0.9669 | val_f1=0.9654 | val_auc=0.9974 | reward=0.9658 | theta=race_soft (g=0.95, a=0.18, b=3.80, t=2.20, k=3, eg=0.08, mix=0.72)
    GLOBAL VAL (Round 5) | global_acc=0.9719 | global_f1=0.9272 | ds1_acc=0.9765 | ds1_f1=0.9483 | ds2_acc=0.9710 | ds2_f1=0.9232 | reward=1.0855 | round_time=151.9s
    
    ======================================================================================================================
    ROUND 6/15 | selected_ds1=[2, 1] | selected_ds2=[4, 3]
    ======================================================================================================================
    Client 2 (ds1) | train_acc=0.9672 | val_acc=0.8889 | val_f1=0.7039 | val_auc=0.9911 | reward=0.7502 | theta=race_edge_plus (g=1.02, a=0.32, b=6.00, t=2.60, k=3, eg=0.15, mix=0.84)
    Client 1 (ds1) | train_acc=0.9520 | val_acc=1.0000 | val_f1=1.0000 | val_auc=nan | reward=1.0000 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    Client 4 (ds2) | train_acc=0.9763 | val_acc=0.9861 | val_f1=0.7455 | val_auc=nan | reward=0.8056 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    Client 3 (ds2) | train_acc=0.9603 | val_acc=0.9767 | val_f1=0.4933 | val_auc=0.9198 | reward=0.6141 | theta=race_soft (g=0.95, a=0.18, b=3.80, t=2.20, k=3, eg=0.08, mix=0.72)
    GLOBAL VAL (Round 6) | global_acc=0.9704 | global_f1=0.6557 | ds1_acc=0.9333 | ds1_f1=0.8224 | ds2_acc=0.9810 | ds2_f1=0.6082 | reward=0.8810 | round_time=64.4s
    
    ======================================================================================================================
    ROUND 7/15 | selected_ds1=[0, 1] | selected_ds2=[5, 4]
    ======================================================================================================================
    Client 0 (ds1) | train_acc=0.9561 | val_acc=0.9701 | val_f1=0.9499 | val_auc=0.9997 | reward=0.9549 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    Client 1 (ds1) | train_acc=0.9640 | val_acc=1.0000 | val_f1=1.0000 | val_auc=nan | reward=1.0000 | theta=race_texture (g=0.92, a=0.34, b=5.80, t=2.30, k=7, eg=0.10, mix=0.80)
    Client 5 (ds2) | train_acc=0.9472 | val_acc=0.9586 | val_f1=0.9600 | val_auc=0.9979 | reward=0.9596 | theta=race_soft (g=0.95, a=0.18, b=3.80, t=2.20, k=3, eg=0.08, mix=0.72)
    Client 4 (ds2) | train_acc=0.9677 | val_acc=0.9861 | val_f1=0.9636 | val_auc=nan | reward=0.9693 | theta=race_soft (g=0.95, a=0.18, b=3.80, t=2.20, k=3, eg=0.08, mix=0.72)
    GLOBAL VAL (Round 7) | global_acc=0.9653 | global_f1=0.9606 | ds1_acc=0.9765 | ds1_f1=0.9605 | ds2_acc=0.9631 | ds2_f1=0.9606 | reward=1.1070 | round_time=146.5s
    
    ======================================================================================================================
    ROUND 8/15 | selected_ds1=[2, 0] | selected_ds2=[5, 4]
    ======================================================================================================================
    Client 2 (ds1) | train_acc=0.9621 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | reward=1.0000 | theta=race_sharp (g=1.04, a=0.30, b=5.20, t=2.50, k=5, eg=0.12, mix=0.82)
    Client 0 (ds1) | train_acc=0.9561 | val_acc=0.9701 | val_f1=0.9608 | val_auc=0.9992 | reward=0.9631 | theta=race_robust (g=1.08, a=0.22, b=4.50, t=2.80, k=5, eg=0.11, mix=0.76)
    Client 5 (ds2) | train_acc=0.9600 | val_acc=0.9751 | val_f1=0.9753 | val_auc=0.9962 | reward=0.9753 | theta=race_smoothmix (g=0.90, a=0.20, b=4.00, t=2.10, k=7, eg=0.07, mix=0.70)
    Client 4 (ds2) | train_acc=0.9820 | val_acc=0.9861 | val_f1=0.7455 | val_auc=nan | reward=0.8056 | theta=race_smoothmix (g=0.90, a=0.20, b=4.00, t=2.10, k=7, eg=0.07, mix=0.70)
    GLOBAL VAL (Round 8) | global_acc=0.9773 | global_f1=0.9434 | ds1_acc=0.9787 | ds1_f1=0.9720 | ds2_acc=0.9770 | ds2_f1=0.9372 | reward=1.1025 | round_time=148.5s
    
    ======================================================================================================================
    ROUND 9/15 | selected_ds1=[1, 2] | selected_ds2=[3, 5]
    ======================================================================================================================
    Client 1 (ds1) | train_acc=0.9680 | val_acc=1.0000 | val_f1=1.0000 | val_auc=nan | reward=1.0000 | theta=race_edge_plus (g=1.02, a=0.32, b=6.00, t=2.60, k=3, eg=0.15, mix=0.84)
    Client 2 (ds1) | train_acc=0.9823 | val_acc=0.9630 | val_f1=0.9106 | val_auc=0.9579 | reward=0.9237 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    Client 3 (ds2) | train_acc=0.9809 | val_acc=0.9884 | val_f1=0.7478 | val_auc=1.0000 | reward=0.8080 | theta=race_robust (g=1.08, a=0.22, b=4.50, t=2.80, k=5, eg=0.11, mix=0.76)
    Client 5 (ds2) | train_acc=0.9578 | val_acc=0.9751 | val_f1=0.9746 | val_auc=0.9986 | reward=0.9747 | theta=race_smoothmix (g=0.90, a=0.20, b=4.00, t=2.10, k=7, eg=0.07, mix=0.70)
    GLOBAL VAL (Round 9) | global_acc=0.9777 | global_f1=0.9324 | ds1_acc=0.9778 | ds1_f1=0.9463 | ds2_acc=0.9777 | ds2_f1=0.9310 | reward=1.0899 | round_time=136.4s
    
    ======================================================================================================================
    ROUND 10/15 | selected_ds1=[0, 1] | selected_ds2=[4, 5]
    ======================================================================================================================
    Client 0 (ds1) | train_acc=0.9724 | val_acc=0.9851 | val_f1=0.9750 | val_auc=0.9991 | reward=0.9775 | theta=race_robust (g=1.08, a=0.22, b=4.50, t=2.80, k=5, eg=0.11, mix=0.76)
    Client 1 (ds1) | train_acc=0.9400 | val_acc=1.0000 | val_f1=1.0000 | val_auc=nan | reward=1.0000 | theta=race_sharp (g=1.04, a=0.30, b=5.20, t=2.50, k=5, eg=0.12, mix=0.82)
    Client 4 (ds2) | train_acc=0.9915 | val_acc=1.0000 | val_f1=1.0000 | val_auc=nan | reward=1.0000 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    Client 5 (ds2) | train_acc=0.9646 | val_acc=0.9779 | val_f1=0.9770 | val_auc=0.9994 | reward=0.9773 | theta=race_robust (g=1.08, a=0.22, b=4.50, t=2.80, k=5, eg=0.11, mix=0.76)
    GLOBAL VAL (Round 10) | global_acc=0.9827 | global_f1=0.9808 | ds1_acc=0.9882 | ds1_f1=0.9803 | ds2_acc=0.9816 | ds2_f1=0.9809 | reward=1.1288 | round_time=147.5s
    
    ======================================================================================================================
    ROUND 11/15 | selected_ds1=[2, 1] | selected_ds2=[3, 4]
    ======================================================================================================================
    Client 2 (ds1) | train_acc=0.9773 | val_acc=0.9259 | val_f1=0.9035 | val_auc=0.9929 | reward=0.9091 | theta=race_texture (g=0.92, a=0.34, b=5.80, t=2.30, k=7, eg=0.10, mix=0.80)
    Client 1 (ds1) | train_acc=0.9400 | val_acc=1.0000 | val_f1=1.0000 | val_auc=nan | reward=1.0000 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    Client 3 (ds2) | train_acc=0.9793 | val_acc=0.9884 | val_f1=0.9931 | val_auc=0.9985 | reward=0.9919 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    Client 4 (ds2) | train_acc=0.9886 | val_acc=0.9861 | val_f1=0.9636 | val_auc=nan | reward=0.9693 | theta=race_robust (g=1.08, a=0.22, b=4.50, t=2.80, k=5, eg=0.11, mix=0.76)
    GLOBAL VAL (Round 11) | global_acc=0.9803 | global_f1=0.9713 | ds1_acc=0.9556 | ds1_f1=0.9421 | ds2_acc=0.9873 | ds2_f1=0.9797 | reward=1.1054 | round_time=64.8s
    
    ======================================================================================================================
    ROUND 12/15 | selected_ds1=[0, 1] | selected_ds2=[5, 3]
    ======================================================================================================================
    Client 0 (ds1) | train_acc=0.9684 | val_acc=0.9552 | val_f1=0.9351 | val_auc=0.9955 | reward=0.9401 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    Client 1 (ds1) | train_acc=0.9720 | val_acc=0.9444 | val_f1=0.9585 | val_auc=nan | reward=0.9550 | theta=race_texture (g=0.92, a=0.34, b=5.80, t=2.30, k=7, eg=0.10, mix=0.80)
    Client 5 (ds2) | train_acc=0.9689 | val_acc=0.9724 | val_f1=0.9708 | val_auc=0.9987 | reward=0.9712 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    Client 3 (ds2) | train_acc=0.9825 | val_acc=0.9651 | val_f1=0.7342 | val_auc=0.9989 | reward=0.7919 | theta=race_smoothmix (g=0.90, a=0.20, b=4.00, t=2.10, k=7, eg=0.07, mix=0.70)
    GLOBAL VAL (Round 12) | global_acc=0.9681 | global_f1=0.9277 | ds1_acc=0.9529 | ds1_f1=0.9400 | ds2_acc=0.9710 | ds2_f1=0.9254 | reward=1.0805 | round_time=147.3s
    
    ======================================================================================================================
    ROUND 13/15 | selected_ds1=[2, 0] | selected_ds2=[4, 5]
    ======================================================================================================================
    Client 2 (ds1) | train_acc=0.9848 | val_acc=0.8889 | val_f1=0.8440 | val_auc=0.9946 | reward=0.8552 | theta=race_sharp (g=1.04, a=0.30, b=5.20, t=2.50, k=5, eg=0.12, mix=0.82)
    Client 0 (ds1) | train_acc=0.9786 | val_acc=0.9851 | val_f1=0.9750 | val_auc=0.9932 | reward=0.9775 | theta=race_sharp (g=1.04, a=0.30, b=5.20, t=2.50, k=5, eg=0.12, mix=0.82)
    Client 4 (ds2) | train_acc=0.9915 | val_acc=0.9861 | val_f1=0.7455 | val_auc=nan | reward=0.8056 | theta=race_soft (g=0.95, a=0.18, b=3.80, t=2.20, k=3, eg=0.08, mix=0.72)
    Client 5 (ds2) | train_acc=0.9697 | val_acc=0.9641 | val_f1=0.9629 | val_auc=0.9976 | reward=0.9632 | theta=race_robust (g=1.08, a=0.22, b=4.50, t=2.80, k=5, eg=0.11, mix=0.76)
    GLOBAL VAL (Round 13) | global_acc=0.9659 | global_f1=0.9287 | ds1_acc=0.9574 | ds1_f1=0.9373 | ds2_acc=0.9677 | ds2_f1=0.9268 | reward=1.0803 | round_time=150.9s
    
    ======================================================================================================================
    ROUND 14/15 | selected_ds1=[1, 0] | selected_ds2=[5, 3]
    ======================================================================================================================
    Client 1 (ds1) | train_acc=0.9800 | val_acc=1.0000 | val_f1=1.0000 | val_auc=nan | reward=1.0000 | theta=race_robust (g=1.08, a=0.22, b=4.50, t=2.80, k=5, eg=0.11, mix=0.76)
    Client 0 (ds1) | train_acc=0.9857 | val_acc=0.9701 | val_f1=0.9616 | val_auc=0.9954 | reward=0.9637 | theta=race_texture (g=0.92, a=0.34, b=5.80, t=2.30, k=7, eg=0.10, mix=0.80)
    Client 5 (ds2) | train_acc=0.9730 | val_acc=0.9807 | val_f1=0.9801 | val_auc=0.9985 | reward=0.9802 | theta=race_soft (g=0.95, a=0.18, b=3.80, t=2.20, k=3, eg=0.08, mix=0.72)
    Client 3 (ds2) | train_acc=0.9873 | val_acc=0.9767 | val_f1=0.7386 | val_auc=0.9988 | reward=0.7981 | theta=race_soft (g=0.95, a=0.18, b=3.80, t=2.20, k=3, eg=0.08, mix=0.72)
    GLOBAL VAL (Round 14) | global_acc=0.9794 | global_f1=0.9395 | ds1_acc=0.9765 | ds1_f1=0.9697 | ds2_acc=0.9799 | ds2_f1=0.9337 | reward=1.1001 | round_time=151.8s
    
    ======================================================================================================================
    ROUND 15/15 | selected_ds1=[2, 1] | selected_ds2=[4, 5]
    ======================================================================================================================
    Client 2 (ds1) | train_acc=0.9823 | val_acc=0.9259 | val_f1=0.8208 | val_auc=0.9982 | reward=0.8471 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    Client 1 (ds1) | train_acc=0.9840 | val_acc=1.0000 | val_f1=1.0000 | val_auc=nan | reward=1.0000 | theta=race_focus (g=1.10, a=0.36, b=6.40, t=2.70, k=3, eg=0.17, mix=0.86)
    Client 4 (ds2) | train_acc=0.9972 | val_acc=1.0000 | val_f1=1.0000 | val_auc=nan | reward=1.0000 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    Client 5 (ds2) | train_acc=0.9708 | val_acc=0.9807 | val_f1=0.9797 | val_auc=0.9986 | reward=0.9800 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    GLOBAL VAL (Round 15) | global_acc=0.9812 | global_f1=0.9746 | ds1_acc=0.9556 | ds1_f1=0.8925 | ds2_acc=0.9839 | ds2_f1=0.9831 | reward=1.0820 | round_time=133.8s
    
    ======================================================================================================================
    TRAINING COMPLETE | total_time=1942.3s | best_round=10 | best_reward=1.1288
    ======================================================================================================================
    
    ----------------------------------------------------------------------------------------------------------------------
    GLOBAL per-round metrics
    ----------------------------------------------------------------------------------------------------------------------
     round  round_time_s  n_selected_clients  active_fraction  global_reward  global_acc  global_f1_macro  global_precision_macro  global_recall_macro  global_log_loss  global_loss_ce  global_eval_time_s  ds1_acc  ds1_f1_macro  ds1_log_loss  ds2_acc  ds2_f1_macro  ds2_log_loss selected_ds1 selected_ds2
         1    124.183754                   4         0.666667       0.924468    0.872651         0.851248                0.845392             0.862690         0.351975        0.356235            2.965191 0.844444      0.725522      0.524633 0.875576      0.864284      0.334073       [2, 1]       [5, 4]
         2     69.252890                   4         0.666667       0.939052    0.938272         0.766575                0.781206             0.767181         0.218214        0.207229            0.975416 0.894118      0.853389      0.317084 0.962025      0.719872      0.165025       [0, 1]       [3, 4]
         3    155.929619                   4         0.666667       0.952181    0.937269         0.875104                0.877704             0.875368         0.209908        0.201314            2.601815 0.872340      0.720480      0.326549 0.950893      0.907547      0.185434       [0, 2]       [5, 3]
         4    148.657110                   4         0.666667       1.083982    0.961464         0.955287                0.957159             0.954147         0.127163        0.125271            2.718181 0.952941      0.918582      0.159734 0.963134      0.962476      0.120784       [1, 0]       [5, 4]
         5    151.859857                   4         0.666667       1.085497    0.971857         0.927184                0.931260             0.925937         0.109399        0.107456            2.661232 0.976471      0.948272      0.095454 0.970982      0.923182      0.112045       [1, 0]       [3, 5]
         6     64.406459                   4         0.666667       0.880961    0.970443         0.655670                0.650831             0.661427         0.110777        0.109052            0.762503 0.933333      0.822358      0.220981 0.981013      0.608195      0.079390       [2, 1]       [4, 3]
         7    146.543820                   4         0.666667       1.107033    0.965318         0.960571                0.957814             0.965503         0.107269        0.104493            2.661522 0.976471      0.960471      0.079078 0.963134      0.960591      0.112790       [0, 1]       [5, 4]
         8    148.461548                   4         0.666667       1.102503    0.977273         0.943410                0.942515             0.944880         0.096067        0.093043            2.647260 0.978723      0.972049      0.088973 0.976959      0.937207      0.097603       [2, 0]       [5, 4]
         9    136.406578                   4         0.666667       1.089860    0.977688         0.932445                0.936651             0.930810         0.079923        0.077824            2.771544 0.977778      0.946341      0.144977 0.977679      0.931050      0.073389       [1, 2]       [3, 5]
        10    147.470287                   4         0.666667       1.128797    0.982659         0.980755                0.978143             0.984035         0.062325        0.060408            2.687354 0.988235      0.980259      0.075790 0.981567      0.980852      0.059688       [0, 1]       [4, 5]
        11     64.770419                   4         0.666667       1.105351    0.980296         0.971349                0.961844             0.988530         0.103653        0.106434            0.747531 0.955556      0.942105      0.205621 0.987342      0.979678      0.074612       [2, 1]       [3, 4]
        12    147.316113                   4         0.666667       1.080529    0.968105         0.927707                0.926127             0.931118         0.094566        0.090301            2.690128 0.952941      0.940025      0.169613 0.970982      0.925370      0.080327       [0, 1]       [5, 3]
        13    150.893582                   4         0.666667       1.080277    0.965909         0.928713                0.926573             0.935226         0.103108        0.098481            2.693000 0.957447      0.937336      0.155952 0.967742      0.926845      0.091662       [2, 0]       [4, 5]
        14    151.818733                   4         0.666667       1.100123    0.979362         0.939458                0.937475             0.942045         0.083502        0.084766            2.641054 0.976471      0.969709      0.086397 0.979911      0.933718      0.082953       [1, 0]       [5, 3]
        15    133.753450                   4         0.666667       1.082012    0.981211         0.974574                0.976922             0.976922         0.071828        0.069536            2.794920 0.955556      0.892500      0.123928 0.983871      0.983084      0.066425       [2, 1]       [4, 5]
    
    ----------------------------------------------------------------------------------------------------------------------
    LOCAL per-client per-round metrics
    ----------------------------------------------------------------------------------------------------------------------
     round   client dataset  selected  theta_arm     theta_name                                                theta_str  gamma_power  alpha_contrast_weight  beta_contrast_sharpness  tau_clip  k_blur_kernel_size  edge_gain  blend_mix  train_loss  train_ce_loss  train_proto_loss  train_acc  val_size  val_loss_ce  val_acc  val_precision_macro  val_recall_macro  val_f1_macro  val_precision_weighted  val_recall_weighted  val_f1_weighted  val_log_loss  val_eval_time_s  val_auc_roc_macro_ovr  val_auc_class_0  val_auc_class_1  val_auc_class_2  val_auc_class_3  val_fusion_gate_mean_raw  val_fusion_gate_mean_enh  val_fusion_gate_mean_res  val_fusion_gate_entropy   reward  bandit_counts_total  bandit_value_selected  participation_rounds_so_far
         1 client_2     ds1         1        5.0     race_focus (g=1.10, a=0.36, b=6.40, t=2.70, k=3, eg=0.17, mix=0.86)         1.10                   0.36                      6.4       2.7                 3.0       0.17       0.86    0.515316       0.474507          0.000000   0.691919        27     0.539682 0.777778             0.597222          0.700000      0.627193                0.769547             0.777778         0.759584      0.534621         0.538035               0.904068         0.820000         0.807143         0.989130         1.000000                  0.454722                  0.228220                  0.317058                 1.518242 0.664839                  1.0               0.664839                            1
         1 client_1     ds1         1        5.0     race_focus (g=1.10, a=0.36, b=6.40, t=2.70, k=3, eg=0.17, mix=0.86)         1.10                   0.36                      6.4       2.7                 3.0       0.17       0.86    0.652675       0.626139          0.000000   0.584000        18     0.653292 0.944444             0.833333          0.969697      0.873016                0.972222             0.944444         0.952381      0.509651         0.399147                    NaN         0.987013              NaN         0.941176         1.000000                  0.003132                  0.545741                  0.451126                 0.898839 0.890873                  1.0               0.890873                            1
         1 client_5     ds2         1        1.0  race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)         1.00                   0.24                      4.6       2.4                 5.0       0.10       0.78    0.453538       0.347147          0.000000   0.767998       362     0.377292 0.856354             0.849297          0.857153      0.852575                0.853378             0.856354         0.854316      0.378092         3.690717               0.967722         0.965127         0.922634         0.994975         0.988152                  0.385675                  0.441133                  0.173192                 1.456642 0.853520                  1.0               0.853520                            1
         1 client_4     ds2         1        3.0 race_smoothmix (g=0.90, a=0.20, b=4.00, t=2.10, k=7, eg=0.07, mix=0.70)         0.90                   0.20                      4.0       2.1                 7.0       0.07       0.70    0.387991       0.323298          0.000000   0.773245        72     0.107308 0.972222             0.921839          0.924786      0.923156                0.972701             0.972222         0.972254      0.112752         0.869101                    NaN         0.996753              NaN         1.000000         0.991045                  0.542953                  0.275234                  0.181813                 1.419552 0.935423                  1.0               0.935423                            1
         1 client_0     ds1         0        NaN            NaN                                                      NaN          NaN                    NaN                      NaN       NaN                 NaN        NaN        NaN         NaN            NaN               NaN        NaN         0          NaN      NaN                  NaN               NaN           NaN                     NaN                  NaN              NaN           NaN              NaN                    NaN              NaN              NaN              NaN              NaN                       NaN                       NaN                       NaN                      NaN      NaN                  NaN                    NaN                            0
         1 client_3     ds2         0        NaN            NaN                                                      NaN          NaN                    NaN                      NaN       NaN                 NaN        NaN        NaN         NaN            NaN               NaN        NaN         0          NaN      NaN                  NaN               NaN           NaN                     NaN                  NaN              NaN           NaN              NaN                    NaN              NaN              NaN              NaN              NaN                       NaN                       NaN                       NaN                      NaN      NaN                  NaN                    NaN                            0
         2 client_0     ds1         1        5.0     race_focus (g=1.10, a=0.36, b=6.40, t=2.70, k=3, eg=0.17, mix=0.86)         1.10                   0.36                      6.4       2.7                 3.0       0.17       0.86    0.315726       0.237462          0.303508   0.838776        67     0.274898 0.880597             0.844406          0.852273      0.825154                0.923599             0.880597         0.888385      0.306141         1.243262               0.983842         0.975641         0.961749         1.000000         0.997980                  0.481315                  0.398445                  0.120240                 1.374363 0.839015                  1.0               0.839015                            1
         2 client_1     ds1         1        3.0    race_robust (g=1.08, a=0.22, b=4.50, t=2.80, k=5, eg=0.11, mix=0.76)         1.08                   0.22                      4.5       2.8                 5.0       0.11       0.76    0.360671       0.299282          0.355640   0.836000        18     0.383593 0.944444             0.952381          0.969697      0.958486                0.952381             0.944444         0.945258      0.357814         0.283127                    NaN         1.000000              NaN         1.000000         0.986111                  0.507100                  0.217490                  0.275410                 1.474821 0.954976                  2.0               0.954976                            2
         2 client_3     ds2         1        3.0 race_smoothmix (g=0.90, a=0.20, b=4.00, t=2.10, k=7, eg=0.07, mix=0.70)         0.90                   0.20                      4.0       2.1                 7.0       0.07       0.70    0.287683       0.217582          0.260098   0.876789        86     0.233827 0.953488             0.727500          0.722222      0.724359                0.942674             0.953488         0.947227      0.240133         1.022773               0.923235         0.752941         0.963591         0.976407         1.000000                  0.660102                  0.295946                  0.043952                 1.102561 0.781641                  1.0               0.781641                            1
         2 client_4     ds2         1        2.0    race_robust (g=1.08, a=0.22, b=4.50, t=2.80, k=5, eg=0.11, mix=0.76)         1.08                   0.22                      4.5       2.8                 5.0       0.11       0.76    0.139278       0.085259          0.222120   0.953510        72     0.068399 0.972222             0.743750          0.691071      0.714512                0.986458             0.972222         0.978357      0.075313         0.842679                    NaN         1.000000              NaN         1.000000         1.000000                  0.421090                  0.365745                  0.213165                 1.525768 0.778940                  2.0               0.778940                            2
         2 client_2     ds1         0        NaN            NaN                                                      NaN          NaN                    NaN                      NaN       NaN                 NaN        NaN        NaN         NaN            NaN               NaN        NaN         0          NaN      NaN                  NaN               NaN           NaN                     NaN                  NaN              NaN           NaN              NaN                    NaN              NaN              NaN              NaN              NaN                       NaN                       NaN                       NaN                      NaN      NaN                  NaN                    NaN                            1
         2 client_5     ds2         0        NaN            NaN                                                      NaN          NaN                    NaN                      NaN       NaN                 NaN        NaN        NaN         NaN            NaN               NaN        NaN         0          NaN      NaN                  NaN               NaN           NaN                     NaN                  NaN              NaN           NaN              NaN                    NaN              NaN              NaN              NaN              NaN                       NaN                       NaN                       NaN                      NaN      NaN                  NaN                    NaN                            1
         3 client_0     ds1         1        4.0 race_edge_plus (g=1.02, a=0.32, b=6.00, t=2.60, k=3, eg=0.15, mix=0.84)         1.02                   0.32                      6.0       2.6                 3.0       0.15       0.84    0.370314       0.269937          0.284026   0.844898        67     0.229935 0.895522             0.840030          0.833333      0.830598                0.899437             0.895522         0.892267      0.262063         0.790818               0.987510         0.975641         0.975410         1.000000         0.998990                  0.430508                  0.407919                  0.161573                 1.408706 0.846829                  2.0               0.846829                            2
         3 client_2     ds1         1        3.0    race_robust (g=1.08, a=0.22, b=4.50, t=2.80, k=5, eg=0.11, mix=0.76)         1.08                   0.22                      4.5       2.8                 5.0       0.11       0.76    0.400135       0.336936          0.269276   0.808081        27     0.467797 0.814815             0.425000          0.475000      0.447222                0.785185             0.814815         0.798354      0.486569         0.387114               0.948929         0.860000         0.935714         1.000000         1.000000                  0.640278                  0.207242                  0.152480                 1.283870 0.539120                  2.0               0.539120                            2
         3 client_5     ds2         1        2.0    race_robust (g=1.08, a=0.22, b=4.50, t=2.80, k=5, eg=0.11, mix=0.76)         1.08                   0.22                      4.5       2.8                 5.0       0.11       0.76    0.362299       0.231117          0.246715   0.861478       362     0.185717 0.950276             0.955228          0.948234      0.950925                0.951509             0.950276         0.950169      0.187864         3.508637               0.991898         0.994242         0.989640         0.986849         0.996861                  0.673221                  0.280182                  0.046597                 1.091173 0.950762                  2.0               0.950762                            2
         3 client_3     ds2         1        1.0  race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)         1.00                   0.24                      4.6       2.4                 5.0       0.10       0.78    0.246168       0.156062          0.227800   0.890302        86     0.161002 0.953488             0.722861          0.727096      0.724960                0.942261             0.953488         0.947826      0.175206         0.890932               0.975097         0.929412         0.983679         0.987296         1.000000                  0.791605                  0.172412                  0.035983                 0.858688 0.782092                  2.0               0.782092                            2
         3 client_1     ds1         0        NaN            NaN                                                      NaN          NaN                    NaN                      NaN       NaN                 NaN        NaN        NaN         NaN            NaN               NaN        NaN         0          NaN      NaN                  NaN               NaN           NaN                     NaN                  NaN              NaN           NaN              NaN                    NaN              NaN              NaN              NaN              NaN                       NaN                       NaN                       NaN                      NaN      NaN                  NaN                    NaN                            2
         3 client_4     ds2         0        NaN            NaN                                                      NaN          NaN                    NaN                      NaN       NaN                 NaN        NaN        NaN         NaN            NaN               NaN        NaN         0          NaN      NaN                  NaN               NaN           NaN                     NaN                  NaN              NaN           NaN              NaN                    NaN              NaN              NaN              NaN              NaN                       NaN                       NaN                       NaN                      NaN      NaN                  NaN                    NaN                            2
         4 client_1     ds1         1        4.0 race_edge_plus (g=1.02, a=0.32, b=6.00, t=2.60, k=3, eg=0.15, mix=0.84)         1.02                   0.32                      6.0       2.6                 3.0       0.15       0.84    0.391508       0.333613          0.273924   0.848000        18     0.186964 1.000000             1.000000          1.000000      1.000000                1.000000             1.000000         1.000000      0.124339         0.296150                    NaN         1.000000              NaN         1.000000         1.000000                  0.523488                  0.232343                  0.244169                 1.460563 1.000000                  3.0               1.000000                            3
         4 client_0     ds1         1        2.0   race_texture (g=0.92, a=0.34, b=5.80, t=2.30, k=7, eg=0.10, mix=0.80)         0.92                   0.34                      5.8       2.3                 7.0       0.10       0.80    0.198424       0.137725          0.185095   0.937755        67     0.146212 0.940299             0.908750          0.888636      0.896709                0.939776             0.940299         0.938964      0.169243         0.800111               0.992895         0.987179         0.986339         0.998062         1.000000                  0.797175                  0.160767                  0.042058                 0.825322 0.907606                  3.0               0.907606                            3
    ... showing first 20 of 90 rows
    
    ----------------------------------------------------------------------------------------------------------------------
    Participation UCB history
    ----------------------------------------------------------------------------------------------------------------------
     gid  arm_index  ucb_score  count    value  selected  round dataset
       0          0        inf      0 0.000000         0      1     ds1
       1          1        inf      0 0.000000         1      1     ds1
       2          2        inf      0 0.000000         1      1     ds1
       3          0        inf      0 0.000000         0      1     ds2
       4          1        inf      0 0.000000         1      1     ds2
       5          2        inf      0 0.000000         1      1     ds2
       0          0        inf      0 0.000000         1      2     ds1
       1          1   2.305872      1 0.890873         1      2     ds1
       2          2   2.079838      1 0.664839         0      2     ds1
       3          0        inf      0 0.000000         1      2     ds2
       4          1   2.350421      1 0.935423         1      2     ds2
       5          2   2.268518      1 0.853520         0      2     ds2
       0          0   2.551674      1 0.839015         1      3     ds1
       1          1   2.133957      2 0.922924         0      3     ds1
       2          2   2.377498      1 0.664839         1      3     ds1
       3          0   2.494300      1 0.781641         1      3     ds2
       4          1   2.068214      2 0.857181         0      3     ds2
       5          2   2.566179      1 0.853520         1      3     ds2
       0          0   2.174542      2 0.842922         1      4     ds1
       1          1   2.254544      2 0.922924         1      4     ds1
    ... showing first 20 of 90 rows
    
    ======================================================================================================================
    STEP 11: FINAL EVALUATION
    ======================================================================================================================
    
    ----------------------------------------------------------------------------------------------------------------------
    Client summary with RL-based participation and preset selection
    ----------------------------------------------------------------------------------------------------------------------
      client dataset  participation_rounds  best_theta_arm best_theta_name                                           best_theta_str  estimated_theta_value  theta_pulls
    client_0     ds1                    10               3     race_robust (g=1.08, a=0.22, b=4.50, t=2.80, k=5, eg=0.11, mix=0.76)               0.970306            7
    client_1     ds1                    12               0   race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)               1.000000            8
    client_2     ds1                     8               1      race_sharp (g=1.04, a=0.30, b=5.20, t=2.50, k=5, eg=0.12, mix=0.82)               1.000000            5
    client_3     ds2                     8               2     race_robust (g=1.08, a=0.22, b=4.50, t=2.80, k=5, eg=0.11, mix=0.76)               0.807073            5
    client_4     ds2                    10               0       race_soft (g=0.95, a=0.18, b=3.80, t=2.20, k=3, eg=0.08, mix=0.72)               0.984628            7
    client_5     ds2                    12               3  race_smoothmix (g=0.90, a=0.20, b=4.00, t=2.10, k=7, eg=0.07, mix=0.70)               0.968407            8
    
    ----------------------------------------------------------------------------------------------------------------------
    Validation ranking of RL-UCB candidates — DS1
    ----------------------------------------------------------------------------------------------------------------------
       theta_name    score  val_acc   val_f1  estimated_bandit_value  from_client_gid
      race_robust 0.954388 0.955752 0.953934                0.970306                0
    race_balanced 0.954388 0.955752 0.953934                1.000000                1
       race_sharp 0.954388 0.955752 0.953934                1.000000                2
    
    ----------------------------------------------------------------------------------------------------------------------
    Validation ranking of RL-UCB candidates — DS2
    ----------------------------------------------------------------------------------------------------------------------
        theta_name    score  val_acc   val_f1  estimated_bandit_value  from_client_gid
       race_robust 0.960513 0.962085 0.959989                0.807073                3
         race_soft 0.960513 0.962085 0.959989                0.984628                4
    race_smoothmix 0.960513 0.962085 0.959989                0.968407                5
    
    ----------------------------------------------------------------------------------------------------------------------
    Final strategy after RL-based participation training
    ----------------------------------------------------------------------------------------------------------------------
    dataset                                                  strategy     theta_names    score  val_acc   val_f1
        ds1 rl_based_client_participation_plus_rlucb_preset_selection ['race_robust'] 0.954388 0.955752 0.953934
        ds2 rl_based_client_participation_plus_rlucb_preset_selection ['race_robust'] 0.960513 0.962085 0.959989
    
    ======================================================================================================================
    STEP 12: EXTENDED METRICS + CALIBRATION + ERROR ANALYSIS
    ======================================================================================================================
    
    ----------------------------------------------------------------------------------------------------------------------
    Extended TEST metrics (DS1 vs DS2)
    ----------------------------------------------------------------------------------------------------------------------
     dataset      acc  balanced_acc  precision_macro  recall_macro  f1_macro  precision_weighted  recall_weighted  f1_weighted  log_loss      mcc    kappa  jaccard_macro  ppv_macro  npv_macro  specificity_macro  fpr_macro  fnr_macro      ece      mce  brier_multi  auc_roc_macro_ovr  auc_class_0  auc_class_1  auc_class_2  auc_class_3
    ds1_test 0.951327      0.950633         0.953433      0.950633  0.951043            0.953491         0.951327     0.951437  0.152901 0.935710 0.935074       0.907583   0.953433   0.984017           0.983779   0.016221   0.049367 0.015006 0.694734     0.070424           0.996582     0.991492     0.999256     0.997158     0.998424
    ds2_test 0.974408      0.973278         0.974200      0.973278  0.973108            0.975418         0.974408     0.974321  0.101240 0.966161 0.965797       0.948044   0.974200   0.991706           0.991598   0.008402   0.026722 0.022309 0.279857     0.046708           0.998682     0.999252     0.997129     0.999974     0.998372
    
    ----------------------------------------------------------------------------------------------------------------------
    Classwise metrics — DS1 TEST
    ----------------------------------------------------------------------------------------------------------------------
     class_id class_name  support  tp  fp  fn  tn  prevalence      ppv      npv   recall  specificity      fpr      fnr  jaccard  balanced_acc
            0     glioma       56  54   7   2 163    0.247788 0.885246 0.987879 0.964286     0.958824 0.041176 0.035714 0.857143      0.961555
            1 meningioma       55  49   1   6 170    0.243363 0.980000 0.965909 0.890909     0.994152 0.005848 0.109091 0.875000      0.942531
            2    notumor       59  58   2   1 165    0.261062 0.966667 0.993976 0.983051     0.988024 0.011976 0.016949 0.950820      0.985537
            3  pituitary       56  54   1   2 169    0.247788 0.981818 0.988304 0.964286     0.994118 0.005882 0.035714 0.947368      0.979202
    
    ----------------------------------------------------------------------------------------------------------------------
    Classwise metrics — DS2 TEST
    ----------------------------------------------------------------------------------------------------------------------
     class_id class_name  support  tp  fp  fn  tn  prevalence      ppv      npv   recall  specificity      fpr      fnr  jaccard  balanced_acc
            0     glioma      244 243  16   1 795    0.231280 0.938224 0.998744 0.995902     0.980271 0.019729 0.004098 0.934615      0.988086
            1 meningioma      247 227   1  20 807    0.234123 0.995614 0.975816 0.919028     0.998762 0.001238 0.080972 0.915323      0.958895
            2    notumor      300 298   0   2 755    0.284360 1.000000 0.997358 0.993333     1.000000 0.000000 0.006667 0.993333      0.996667
            3  pituitary      264 260  10   4 781    0.250237 0.962963 0.994904 0.984848     0.987358 0.012642 0.015152 0.948905      0.986103
    
    ----------------------------------------------------------------------------------------------------------------------
    Top confusion pairs — DS1 TEST
    ----------------------------------------------------------------------------------------------------------------------
    true_class pred_class  count
    meningioma     glioma      4
     pituitary     glioma      2
        glioma    notumor      1
        glioma meningioma      1
       notumor     glioma      1
    meningioma    notumor      1
    meningioma  pituitary      1
        glioma  pituitary      0
       notumor meningioma      0
       notumor  pituitary      0
    
    ----------------------------------------------------------------------------------------------------------------------
    Top confusion pairs — DS2 TEST
    ----------------------------------------------------------------------------------------------------------------------
    true_class pred_class  count
    meningioma     glioma     12
    meningioma  pituitary      8
     pituitary     glioma      3
        glioma  pituitary      1
     pituitary meningioma      1
       notumor     glioma      1
       notumor  pituitary      1
        glioma    notumor      0
        glioma meningioma      0
    meningioma    notumor      0
    
    ----------------------------------------------------------------------------------------------------------------------
    Calibration bins — DS1 TEST
    ----------------------------------------------------------------------------------------------------------------------
     bin_id  bin_left  bin_right  bin_confidence  bin_accuracy  bin_gap  bin_count
          0  0.000000   0.083333             NaN           NaN      NaN          0
          1  0.083333   0.166667             NaN           NaN      NaN          0
          2  0.166667   0.250000             NaN           NaN      NaN          0
          3  0.250000   0.333333             NaN           NaN      NaN          0
          4  0.333333   0.416667        0.415636      0.000000 0.415636          1
          5  0.416667   0.500000        0.444138      0.000000 0.444138          1
          6  0.500000   0.583333        0.538481      0.600000 0.061519          5
          7  0.583333   0.666667        0.608540      1.000000 0.391460          1
          8  0.666667   0.750000        0.694734      0.000000 0.694734          1
          9  0.750000   0.833333        0.776309      0.500000 0.276309          2
         10  0.833333   0.916667        0.879023      0.833333 0.045690          6
         11  0.916667   1.000000        0.979373      0.980861 0.001488        209
    
    ----------------------------------------------------------------------------------------------------------------------
    Calibration bins — DS2 TEST
    ----------------------------------------------------------------------------------------------------------------------
     bin_id  bin_left  bin_right  bin_confidence  bin_accuracy  bin_gap  bin_count
          0  0.000000   0.083333             NaN           NaN      NaN          0
          1  0.083333   0.166667             NaN           NaN      NaN          0
          2  0.166667   0.250000             NaN           NaN      NaN          0
          3  0.250000   0.333333             NaN           NaN      NaN          0
          4  0.333333   0.416667             NaN           NaN      NaN          0
          5  0.416667   0.500000        0.464331      0.666667 0.202335          3
          6  0.500000   0.583333        0.520143      0.800000 0.279857         15
          7  0.583333   0.666667        0.622455      0.600000 0.022455         10
          8  0.666667   0.750000        0.695917      0.916667 0.220750         12
          9  0.750000   0.833333        0.787118      0.764706 0.022412         17
         10  0.833333   0.916667        0.885755      0.750000 0.135755         24
         11  0.916667   1.000000        0.979242      0.991786 0.012545        974
    
    ----------------------------------------------------------------------------------------------------------------------
    VAL + TEST tables
    ----------------------------------------------------------------------------------------------------------------------
                            setting split      dataset      acc  precision_macro  recall_macro  f1_macro  precision_weighted  recall_weighted  f1_weighted  log_loss  auc_roc_macro_ovr  loss_ce  eval_time_s  balanced_acc      mcc    kappa  ppv_macro  npv_macro  specificity_macro      ece      mce  brier_multi
    ARCF-Net RL-Based Participation   VAL          ds1 0.955752         0.958525      0.954167  0.953934            0.958035         0.955752     0.954565  0.150838           0.998263 0.143086     4.634098           NaN      NaN      NaN        NaN        NaN                NaN      NaN      NaN          NaN
    ARCF-Net RL-Based Participation   VAL          ds2 0.962085         0.962231      0.960488  0.959989            0.963944         0.962085     0.961736  0.115786           0.998869 0.115734    20.484736           NaN      NaN      NaN        NaN        NaN                NaN      NaN      NaN          NaN
    ARCF-Net RL-Based Participation   VAL global_equal 0.958919         0.960378      0.957327  0.956961            0.960990         0.958919     0.958151  0.133312           0.998566 0.129410    12.559417           NaN      NaN      NaN        NaN        NaN                NaN      NaN      NaN          NaN
    ARCF-Net RL-Based Participation  TEST          ds1 0.951327         0.953433      0.950633  0.951043            0.953491         0.951327     0.951437  0.152901           0.996582 0.144896    10.166860      0.950633 0.935710 0.935074   0.953433   0.984017           0.983779 0.015006 0.694734     0.070424
    ARCF-Net RL-Based Participation  TEST          ds2 0.974408         0.974200      0.973278  0.973108            0.975418         0.974408     0.974321  0.101240           0.998682 0.101174    20.805106      0.973278 0.966161 0.965797   0.974200   0.991706           0.991598 0.022309 0.279857     0.046708
    ARCF-Net RL-Based Participation  TEST global_equal 0.962868         0.963816      0.961955  0.962075            0.964454         0.962868     0.962879  0.127070           0.997632 0.123035    15.485983           NaN      NaN      NaN        NaN        NaN                NaN      NaN      NaN          NaN
    
    Selection summary:
    - Best round: 10 | best_reward=1.1288
    - DS1 final strategy: rl_based_client_participation_plus_rlucb_preset_selection | names=['race_robust']
    - DS2 final strategy: rl_based_client_participation_plus_rlucb_preset_selection | names=['race_robust']
    
    ======================================================================================================================
    STEP 13: PREPROCESSING VALIDATION
    ======================================================================================================================
    
    ----------------------------------------------------------------------------------------------------------------------
    Preprocessing validation summary (DS1 VAL sample)
    ----------------------------------------------------------------------------------------------------------------------
                metric     mean      std       min      max
    edge_energy_before 0.041596 0.021720  0.014947 0.165471
     edge_energy_after 0.094055 0.024375  0.057067 0.214411
        entropy_before 5.820408 0.628255  3.547314 7.239670
         entropy_after 6.492028 0.566759  3.888517 7.470318
       contrast_before 0.187640 0.052597  0.101468 0.363759
        contrast_after 0.226862 0.026633  0.190911 0.342708
       edge_gain_ratio 2.445604 0.518596  1.176741 4.282521
         entropy_delta 0.671621 0.235525  0.230648 1.413039
        contrast_delta 0.039222 0.029151 -0.022064 0.100209
    
    ----------------------------------------------------------------------------------------------------------------------
    Preprocessing validation summary (DS2 VAL sample)
    ----------------------------------------------------------------------------------------------------------------------
                metric     mean      std       min      max
    edge_energy_before 0.069022 0.036905  0.016726 0.375572
     edge_energy_after 0.120571 0.034194  0.071821 0.402352
        entropy_before 6.866192 0.472092  5.135309 7.798752
         entropy_after 7.288234 0.279061  5.830363 7.890225
       contrast_before 0.233743 0.033322  0.144569 0.352019
        contrast_after 0.239958 0.022307  0.180111 0.335932
       edge_gain_ratio 1.891872 0.415752  1.071303 4.545427
         entropy_delta 0.422042 0.217386  0.043292 1.127475
        contrast_delta 0.006215 0.015060 -0.022230 0.064666
    
    ======================================================================================================================
    STEP 14: PARTICIPATION SUMMARY
    ======================================================================================================================
    
    ----------------------------------------------------------------------------------------------------------------------
    Participation summary per client
    ----------------------------------------------------------------------------------------------------------------------
      client dataset  participation_rounds  participation_bandit_count  participation_bandit_value
    client_0     ds1                    10                           7                    0.918901
    client_1     ds1                    12                           8                    0.980731
    client_2     ds1                     8                           5                    0.775559
    client_3     ds2                     8                           5                    0.758405
    client_4     ds2                    10                           7                    0.899265
    client_5     ds2                    12                           8                    0.951520
    
    ======================================================================================================================
    STEP 15: SAVING CHECKPOINT + CSV
    ======================================================================================================================
    Saved checkpoint: /kaggle/working/outputs/ARCFNet_Ablation_RLParticipation_checkpoint.pth
    Saved CSV: /kaggle/working/outputs/ARCFNet_Ablation_RLParticipation_outputs.csv
    
    DONE
    Method: ARCF-Net = Adaptive RACE-FELCM with CRAF Fusion Network
    Ablation: RL-Based Participation with RL-UCB Preset Selection
    Backbone: Residual Network-50
    Best round: 10
    Client pool => DS1=3, DS2=3, TOTAL=6
    RL-selected active per round => DS1=2, DS2=2
    Rounds completed: 15
    Global TEST acc: 0.9629
    Global TEST f1_macro: 0.9621
    DS1 TEST acc: 0.9513
    DS2 TEST acc: 0.9744
    DS1 final strategy: rl_based_client_participation_plus_rlucb_preset_selection | names=['race_robust']
    DS2 final strategy: rl_based_client_participation_plus_rlucb_preset_selection | names=['race_robust']

