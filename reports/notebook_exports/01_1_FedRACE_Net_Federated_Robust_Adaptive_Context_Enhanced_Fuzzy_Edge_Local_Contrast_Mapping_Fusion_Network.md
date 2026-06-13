```python
# ============================================================
# KAGGLE FULL SCRIPT
# TRUE FL + RL-UCB + RACE-FELCM + CRAF + ResNet-50
# METHOD ACRONYM: ARCF-Net
# FULL FORM: Adaptive RACE-FELCM with CRAF Fusion Network
# ------------------------------------------------------------
# KAGGLE-READY + SYMMETRIC DS1/DS2 IMPORTANCE
# - Uses BOTH datasets
# - Reads datasets from /kaggle/input automatically
# - Keeps stronger asymmetric train/tune/val per client regime
# - Exact 15 FL rounds
# - Proper FL with FedAvg + FedProx + prototype sharing
# - RL-UCB for SHARED client-count planning + per-client preprocessing preset selection
# - Tune-aware theta probing before local training
# - Equal DS1 / DS2 importance in best-round selection and merged reporting
# - Professional white-background plots
# - Clear bold labels on before/after images
# - Confusion matrix artifact fixed (no grid / no square-inside-square)
# - Adds MCC, Kappa, PPV, NPV, specificity, FPR, FNR, balanced acc, Jaccard
# - Adds ECE, MCE, Brier score, calibration tables and comparison plots
# - Adds error-analysis plots for BOTH datasets
# - Saves checkpoint WITH full-process info for later XAI / replay
# ============================================================

import os
import sys
import time
import math
import copy
import json
import random
import hashlib
import subprocess
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

import matplotlib.pyplot as plt
import matplotlib as mpl

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    log_loss, confusion_matrix, roc_auc_score,
    roc_curve, precision_recall_curve, average_precision_score,
    matthews_corrcoef, cohen_kappa_score, balanced_accuracy_score,
    jaccard_score
)

# -------------------------
# Helpers
# -------------------------
def pip_install(pkg):
    subprocess.check_call([sys.executable, "-m", "pip", "-q", "install", pkg])

def running_on_kaggle():
    return os.path.exists("/kaggle/input")

IS_KAGGLE = running_on_kaggle()

# Optional fallback only if datasets were not added as Kaggle inputs
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

try:
    from IPython.display import display
except Exception:
    display = print

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

# -------------------------
# Plotting style
# -------------------------
plt.style.use("seaborn-v0_8-white")
mpl.rcParams.update({
    "figure.dpi": 145,
    "savefig.dpi": 220,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#D0D7DE",
    "axes.linewidth": 1.0,
    "axes.titleweight": "bold",
    "axes.titlesize": 16,
    "axes.labelsize": 12,
    "axes.labelweight": "bold",
    "font.size": 11,
    "legend.frameon": True,
    "legend.facecolor": "white",
    "legend.edgecolor": "#D0D7DE",
    "grid.alpha": 0.18,
    "grid.linewidth": 0.8,
    "lines.linewidth": 2.2,
})

print("=" * 118)
print("TRUE FL + RL-UCB + RACE-FELCM + CRAF + ResNet-50")
print("METHOD: ARCF-Net (Adaptive RACE-FELCM with CRAF Fusion Network)")
print("=" * 118)
print(f"ENV: {'KAGGLE' if IS_KAGGLE else 'NON-KAGGLE'} | DEVICE: {DEVICE} | torch={torch.__version__}")
print("=" * 118)

# -------------------------
# Configuration
# -------------------------
CFG = {
    "rounds": 15,

    # shared RL for client-count planning
    "client_count_candidates": [3, 4, 5],
    "client_count_search_episodes": 10,

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

    # bandit
    "ucb_c": 1.35,
    "theta_probe_topk": 3,

    # final inference (same search space for both datasets)
    "final_use_tta": True,
    "final_try_topk_ds1": [1, 2, 3],
    "final_try_topk_ds2": [1, 2, 3],

    # reward
    "reward_f1_weight": 0.75,
    "reward_acc_weight": 0.25,

    # best-round selection: equal dataset importance
    "best_round_mass_ds1": 0.50,
    "best_round_mass_ds2": 0.50,
    "best_round_min_bonus": 0.15,

    # FedAvg tempering
    "fedavg_temper": 0.50,

    # misc / plots
    "quick_hash_subset_per_split": 300,
    "preproc_val_sample_n": 400,
    "before_after_n": 12,
    "make_plots": True,
    "calibration_bins": 12,
}

OUTDIR = "/kaggle/working/outputs" if IS_KAGGLE else "/content/outputs"
os.makedirs(OUTDIR, exist_ok=True)
MODEL_PATH = os.path.join(OUTDIR, "ARCFNet_RESNET50_KAGGLE_FULLINFO_checkpoint.pth")
CSV_PATH   = os.path.join(OUTDIR, "ALL_OUTPUTS_AND_METRICS_RESNET50_KAGGLE.csv")

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
}

# ============================================================
# CSV collector
# ============================================================
ALL_ROWS = []

def add_table_to_csv(df, table_name):
    df2 = df.copy()
    df2.insert(0, "table_name", table_name)
    for _, row in df2.iterrows():
        ALL_ROWS.append(row.to_dict())

def print_table(df, title):
    print("\n" + "-" * 118)
    print(title)
    print("-" * 118)
    display(df)

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
# STEP 0: ACCESS DATASETS (KAGGLE INPUT FIRST)
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

# Prefer Kaggle inputs
DS1_ROOT = search_kaggle_input_for_root(REQ1, prefer_raw=True)
DS2_ROOT = search_kaggle_input_for_root(REQ2, prefer_raw=False)

# Fallback to kagglehub if not found
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
        "Could not locate DS1 under /kaggle/input. In Kaggle, click Add Input and add "
        "'orvile/pmram-bangladeshi-brain-cancer-mri-dataset'."
    )
if DS2_ROOT is None:
    raise RuntimeError(
        "Could not locate DS2 under /kaggle/input. In Kaggle, click Add Input and add "
        "'yassinebazgour/preprocessed-brain-mri-scans-for-tumors-detection'."
    )

print(f"Dataset-1 RAW root detected:\n  {DS1_ROOT}")
print(f"Dataset-2 root detected:\n  {DS2_ROOT}")

# ============================================================
# STEP 1: BUILD MANIFESTS
# ============================================================
print("\n" + "=" * 118)
print("STEP 1: BUILD DATA MANIFESTS (NO MERGE)")
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

print("\n" + "-" * 118)
print("Building Dataset-1 (RAW only)")
df1 = build_df_from_root(DS1_ROOT, ["512Glioma", "512Meningioma", "512Normal", "512Pituitary"], "ds1_raw")
print("Building Dataset-2")
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
print(df1["label"].value_counts().reindex(labels, fill_value=0))
print("\nDataset-2 images:", len(df2))
print(df2["label"].value_counts().reindex(labels, fill_value=0))

# ============================================================
# STEP 2: TRAIN / VAL / TEST SPLIT
# ============================================================
print("\n" + "=" * 118)
print("STEP 2: TRAIN / VAL / TEST SPLIT (PER DATASET)")
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
    print_table(leak_df, f"Leakage / Sanity Summary — {name}")
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
# STEP 4: SHARED ADAPTIVE CLIENT COUNT BY RL-UCB
# ============================================================
print("\n" + "=" * 118)
print("STEP 4: SHARED ADAPTIVE CLIENT COUNT BY RL-UCB")
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

def partition_reward(train_df, client_indices, num_classes):
    sizes = np.array([len(v) for v in client_indices], dtype=np.float32)
    if len(sizes) == 0 or sizes.mean() <= 0:
        return 0.0

    entropies = []
    coverages = []
    for idxs in client_indices:
        if len(idxs) == 0:
            entropies.append(0.0)
            coverages.append(0.0)
            continue
        ys = train_df.loc[idxs, "y"].values
        counts = np.bincount(ys, minlength=num_classes).astype(np.float32)
        probs = counts / np.clip(counts.sum(), 1.0, None)
        probs_nz = probs[probs > 0]
        ent = float(-(probs_nz * np.log(probs_nz)).sum() / np.log(num_classes))
        cov = float((counts > 0).mean())
        entropies.append(ent)
        coverages.append(cov)

    size_balance = 1.0 - min(1.0, float(sizes.std() / np.clip(sizes.mean(), 1e-6, None)))
    min_size_ratio = min(1.0, float(sizes.min() / np.clip(sizes.mean(), 1e-6, None)))

    return float(
        0.45 * np.mean(entropies) +
        0.25 * np.mean(coverages) +
        0.20 * size_balance +
        0.10 * min_size_ratio
    )

def adaptive_client_count_rl_shared(train_df1, train_df2):
    bandit = UCBBandit(len(CFG["client_count_candidates"]), c=CFG["ucb_c"])
    rows = []

    for ep in range(CFG["client_count_search_episodes"]):
        arm = bandit.select()
        n_clients = CFG["client_count_candidates"][arm]

        idxs1 = make_clients_non_iid(
            train_df1,
            n_clients=n_clients,
            num_classes=NUM_CLASSES,
            min_per_class=CFG["min_per_class_per_client"],
            alpha=CFG["dirichlet_alpha"],
        )
        idxs2 = make_clients_non_iid(
            train_df2,
            n_clients=n_clients,
            num_classes=NUM_CLASSES,
            min_per_class=CFG["min_per_class_per_client"],
            alpha=CFG["dirichlet_alpha"],
        )

        reward1 = partition_reward(train_df1, idxs1, NUM_CLASSES)
        reward2 = partition_reward(train_df2, idxs2, NUM_CLASSES)
        reward = 0.5 * (reward1 + reward2)

        bandit.update(arm, reward)

        rows.append({
            "episode": ep + 1,
            "selected_n_clients": n_clients,
            "reward_ds1": reward1,
            "reward_ds2": reward2,
            "reward_mean": reward,
            "bandit_value": bandit.values[arm],
            "pulls_for_arm": bandit.counts[arm],
        })

    best_arm = bandit.best_arm()
    chosen = CFG["client_count_candidates"][best_arm]
    return chosen, pd.DataFrame(rows), bandit

shared_n_clients, plan_df_shared, shared_planner = adaptive_client_count_rl_shared(train1, train2)
n_clients_ds1 = shared_n_clients
n_clients_ds2 = shared_n_clients

print(f"Chosen shared adaptive clients for DS1: {n_clients_ds1}")
print(f"Chosen shared adaptive clients for DS2: {n_clients_ds2}")

print_table(plan_df_shared, "RL planning history — shared client count")
add_table_to_csv(plan_df_shared, "rl_client_count_planning_shared")

# ============================================================
# STEP 5: FINAL NON-IID CLIENT PARTITIONING
# ============================================================
print("\n" + "=" * 118)
print("STEP 5: FINAL NON-IID CLIENT PARTITIONING")
print("=" * 118)

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

# ============================================================
# STEP 6: DATA LOADERS
# ============================================================
print("\n" + "=" * 118)
print("STEP 6: DATA LOADERS (AUG ON)")
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
print_table(dist_df, "Adaptive client class distribution")
add_table_to_csv(dist_df, "adaptive_client_distribution")

val_loader_ds1 = make_loader(val1, list(range(len(val1))), CFG["batch_size"], EVAL_TFMS, shuffle=False, source_id=0)
val_loader_ds2 = make_loader(val2, list(range(len(val2))), CFG["batch_size"], EVAL_TFMS, shuffle=False, source_id=1)
test_loader_ds1 = make_loader(test1, list(range(len(test1))), CFG["batch_size"], EVAL_TFMS, shuffle=False, source_id=0)
test_loader_ds2 = make_loader(test2, list(range(len(test2))), CFG["batch_size"], EVAL_TFMS, shuffle=False, source_id=1)

print(f"Augmentation: {'ON ✅' if CFG['use_augmentation'] else 'OFF'}")
print(f"Preprocessing: {'ON ✅' if CFG['use_preprocessing'] else 'OFF'}")
print(f"Total adaptive clients: {CLIENTS_TOTAL}")

# ------------------------------------------------------------
# PROFESSIONAL BEFORE / AFTER GRID
# ------------------------------------------------------------
def plot_dual_image_grid(before_imgs, after_imgs, before_labs, title):
    B = len(before_imgs)
    fig, axes = plt.subplots(
        2, B,
        figsize=(max(16, 2.35 * B), 7.0),
        constrained_layout=True,
        facecolor="white"
    )
    if B == 1:
        axes = np.array(axes).reshape(2, 1)

    for i in range(B):
        ax1 = axes[0, i]
        ax2 = axes[1, i]

        ax1.imshow(before_imgs[i].permute(1, 2, 0).numpy())
        ax2.imshow(after_imgs[i].permute(1, 2, 0).numpy())

        for ax in [ax1, ax2]:
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
            ax.set_facecolor("white")
            for spine in ax.spines.values():
                spine.set_visible(False)

        ax1.set_title(
            f"Before Aug\n({before_labs[i]})",
            fontsize=11,
            fontweight="bold",
            color="#111827",
            pad=8,
        )
        ax2.set_title(
            f"After Aug\n({before_labs[i]})",
            fontsize=11,
            fontweight="bold",
            color="#111827",
            pad=8,
        )

    fig.suptitle(title, fontsize=18, fontweight="bold", color="#111827")
    plt.show()

@torch.no_grad()
def show_aug_before_after(frame, n=12, title_prefix="Augmentation"):
    if len(frame) == 0:
        return

    per_class = max(1, n // NUM_CLASSES)
    parts = []
    for lab in labels:
        sub = frame[frame["label"] == lab]
        if len(sub) > 0:
            parts.append(sub.sample(n=min(per_class, len(sub)), random_state=SEED))
    if parts:
        sample = pd.concat(parts, axis=0).drop_duplicates(subset=["path"])
    else:
        sample = frame.sample(min(n, len(frame)), random_state=SEED)

    if len(sample) < n:
        extra = frame.sample(min(n - len(sample), len(frame)), random_state=SEED + 7)
        sample = pd.concat([sample, extra], axis=0).drop_duplicates(subset=["path"])

    sample = sample.sample(min(n, len(sample)), random_state=SEED).reset_index(drop=True)
    idxs = list(range(len(sample)))

    raw_ds = MRIDataset(sample, indices=idxs, tfms=EVAL_TFMS, source_id=0)
    aug_ds = MRIDataset(sample, indices=idxs, tfms=TRAIN_TFMS, source_id=0)

    raws, augs, labs_out = [], [], []
    for i in range(len(sample)):
        x_raw, y, *_ = raw_ds[i]
        x_aug, _, *_ = aug_ds[i]
        raws.append(x_raw)
        augs.append(x_aug)
        labs_out.append(id2label[int(y)])

    plot_dual_image_grid(raws, augs, labs_out, f"{title_prefix}: Before vs After (TRAIN_TFMS)")

if CFG["make_plots"] and CFG["use_augmentation"]:
    print("\n" + "-" * 118)
    print("AUGMENTATION VISUAL CHECK (Before vs After) — BOTH DATASETS")
    print("-" * 118)
    show_aug_before_after(train1, n=CFG["before_after_n"], title_prefix="DS1 Augmentation")
    show_aug_before_after(train2, n=CFG["before_after_n"], title_prefix="DS2 Augmentation")

# ============================================================
# STEP 7: NOVEL PREPROCESSING — RACE-FELCM
# ============================================================
print("\n" + "=" * 118)
print("STEP 7: NOVEL PREPROCESSING — RACE-FELCM")
print("=" * 118)

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

for c in clients:
    c["preset_bank"] = PRESET_BANK_DS1 if c["dataset"] == "ds1" else PRESET_BANK_DS2
    c["theta_bandit"] = UCBBandit(len(c["preset_bank"]), c=CFG["ucb_c"])

# ============================================================
# STEP 8: MODEL — ResNet-50 + CRAF Fusion
# ============================================================
print("\n" + "=" * 118)
print("STEP 8: MODEL — ResNet-50 + CRAF Fusion")
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
# STEP 9: LOSSES + PROTOTYPE KNOWLEDGE SHARING
# ============================================================
print("\n" + "=" * 118)
print("STEP 9: LOSSES + PROTOTYPE KNOWLEDGE SHARING")
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
    t0 = time.time()

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
        "train_time_s": float(time.time() - t0),
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
# STEP 10: TUNE-AWARE RL-UCB PREPROCESSING SELECTION
# ============================================================
print("\n" + "=" * 118)
print("STEP 10: TUNE-AWARE RL-UCB PREPROCESSING SELECTION")
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
        return arm_candidates[0]

    best_arm = arm_candidates[0]
    best_score = -1.0

    for arm in arm_candidates:
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

    return best_arm

def select_theta_arm_with_probe(client, model):
    bandit = client["theta_bandit"]
    ucb_scores = [bandit.ucb(i) for i in range(bandit.n_arms)]
    topk = min(CFG["theta_probe_topk"], bandit.n_arms)
    candidates = list(np.argsort(ucb_scores)[::-1][:topk])
    return probe_theta_on_tune_loader(model, client["tune_loader"], client["preset_bank"], candidates, n_batches=2)

# ============================================================
# STEP 11: TRUE FEDERATED TRAINING
# ============================================================
print("\n" + "=" * 118)
print("STEP 11: TRUE FEDERATED TRAINING")
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

print(f"Adaptive clients => DS1={n_clients_ds1}, DS2={n_clients_ds2}, TOTAL={CLIENTS_TOTAL}")
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
    selected_ids = list(range(len(clients)))

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
            "train_time_s": float(np.sum([x["train_time_s"] for x in train_logs])),
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

    print("\n" + "-" * 118)
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
    print("-" * 118)

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
print(f"TRAINING COMPLETE ✅ | total_time={t_total:.1f}s | best_round={best_round_saved} | best_reward={best_reward:.4f}")
print("=" * 118)

glob_df = pd.DataFrame(history_global)
loc_df = pd.DataFrame(history_local)

print_table(glob_df, "GLOBAL per-round metrics")
print_table(loc_df, "LOCAL per-client per-round metrics")
add_table_to_csv(glob_df, "global_round_metrics_full")
add_table_to_csv(loc_df, "client_round_metrics_full")

# ============================================================
# STEP 12: FINAL EVALUATION
# ============================================================
print("\n" + "=" * 118)
print("STEP 12: FINAL EVALUATION")
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

@torch.no_grad()
def evaluate_with_multi_theta(model, loader, theta_list, use_tta=False):
    model.eval()
    freeze_backbone_bn_stats(model)

    members = []
    for theta in theta_list:
        pre = theta_to_module(theta).to(DEVICE).eval()
        members.append((pre, theta))

    all_y, all_p = [], []
    t0 = time.time()

    for x, y, _, source_id in loader:
        x = x.to(DEVICE, non_blocking=True)
        y = y.to(DEVICE, non_blocking=True)
        source_id = source_id.to(DEVICE, non_blocking=True)

        versions = [x]
        if use_tta:
            versions.append(torch.flip(x, dims=[3]))

        probs_total = None
        count_total = 0

        for xv in versions:
            for pre, theta in members:
                x_enh = pre(xv)
                x_res = torch.abs(x_enh - xv)
                x_raw_n = normalize_imagenet(xv)
                x_enh_n = normalize_imagenet(x_enh)
                x_res_n = normalize_imagenet(x_res)
                theta_vec = theta_to_vec(theta, xv.size(0))
                logits = model(x_raw_n, x_enh_n, x_res_n, theta_vec, source_id)
                probs = torch.softmax(logits, dim=1)
                probs_total = probs if probs_total is None else (probs_total + probs)
                count_total += 1

        probs = probs_total / max(1, count_total)
        all_y.append(y.detach().cpu().numpy())
        all_p.append(probs.detach().cpu().numpy())

    if len(all_y) == 0:
        return {"acc": np.nan}, np.array([]), np.array([])

    y_true = np.concatenate(all_y)
    p_pred = np.concatenate(all_p)
    y_hat = np.argmax(p_pred, axis=1)

    met = {
        "loss_ce": float(log_loss(y_true, p_pred, labels=list(range(NUM_CLASSES)))),
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
    return met, y_true, p_pred

@torch.no_grad()
def pick_best_dataset_strategy(model, val_loader, candidates, topk_list):
    ranked = []

    for name, theta, est, gid in candidates:
        met, _, _ = evaluate_with_single_theta(model, val_loader, theta, use_tta=CFG["final_use_tta"])
        ranked.append({
            "strategy": "single",
            "theta_names": [name],
            "theta_list": [theta],
            "score": score_metric(met),
            "val_acc": safe_float(met.get("acc")),
            "val_f1": safe_float(met.get("f1_macro")),
            "est_bandit_value": est,
            "from_client_gid": gid,
        })

    ranked = sorted(ranked, key=lambda x: x["score"], reverse=True)
    singles = ranked[:]

    for topk in topk_list:
        if topk <= 1 or len(singles) < topk:
            continue
        sel = singles[:topk]
        theta_list = [x["theta_list"][0] for x in sel]
        names = [x["theta_names"][0] for x in sel]
        metk, _, _ = evaluate_with_multi_theta(model, val_loader, theta_list, use_tta=CFG["final_use_tta"])
        ranked.append({
            "strategy": f"top{topk}_ensemble",
            "theta_names": names,
            "theta_list": theta_list,
            "score": score_metric(metk),
            "val_acc": safe_float(metk.get("acc")),
            "val_f1": safe_float(metk.get("f1_macro")),
            "est_bandit_value": np.nan,
            "from_client_gid": -1,
        })

    ranked = sorted(ranked, key=lambda x: x["score"], reverse=True)
    return ranked

ds1_clients = [c for c in clients if c["dataset"] == "ds1"]
ds2_clients = [c for c in clients if c["dataset"] == "ds2"]

client_theta_rows = []
for c in clients:
    arm, name, theta, val = best_theta_for_client(c)
    client_theta_rows.append({
        "client": f"client_{c['gid']}",
        "dataset": c["dataset"],
        "best_theta_arm": arm,
        "best_theta_name": name,
        "best_theta_str": theta_str(theta),
        "estimated_value": float(val),
        "pulls": int(c["theta_bandit"].counts.sum()),
    })
client_theta_df = pd.DataFrame(client_theta_rows)
print_table(client_theta_df, "Best RL-selected preprocessing preset per client")
add_table_to_csv(client_theta_df, "best_rl_selected_preprocessing_per_client")

cand_ds1 = unique_theta_candidates(ds1_clients)
cand_ds2 = unique_theta_candidates(ds2_clients)

ranked_ds1 = pick_best_dataset_strategy(global_model, val_loader_ds1, cand_ds1, CFG["final_try_topk_ds1"])
ranked_ds2 = pick_best_dataset_strategy(global_model, val_loader_ds2, cand_ds2, CFG["final_try_topk_ds2"])

choice_ds1 = ranked_ds1[0]
choice_ds2 = ranked_ds2[0]

choice_df = pd.DataFrame([
    {
        "dataset": "ds1",
        "strategy": choice_ds1["strategy"],
        "theta_names": str(choice_ds1["theta_names"]),
        "score": choice_ds1["score"],
        "val_acc": choice_ds1["val_acc"],
        "val_f1": choice_ds1["val_f1"],
    },
    {
        "dataset": "ds2",
        "strategy": choice_ds2["strategy"],
        "theta_names": str(choice_ds2["theta_names"]),
        "score": choice_ds2["score"],
        "val_acc": choice_ds2["val_acc"],
        "val_f1": choice_ds2["val_f1"],
    },
])
print_table(choice_df, "Chosen final validation-based preprocessing strategy")
add_table_to_csv(choice_df, "final_theta_strategy_choice")

if choice_ds1["strategy"] == "single":
    val_ds1, _, _ = evaluate_with_single_theta(global_model, val_loader_ds1, choice_ds1["theta_list"][0], use_tta=CFG["final_use_tta"])
    test_ds1, y_ds1, p_ds1 = evaluate_with_single_theta(global_model, test_loader_ds1, choice_ds1["theta_list"][0], use_tta=CFG["final_use_tta"])
else:
    val_ds1, _, _ = evaluate_with_multi_theta(global_model, val_loader_ds1, choice_ds1["theta_list"], use_tta=CFG["final_use_tta"])
    test_ds1, y_ds1, p_ds1 = evaluate_with_multi_theta(global_model, test_loader_ds1, choice_ds1["theta_list"], use_tta=CFG["final_use_tta"])

if choice_ds2["strategy"] == "single":
    val_ds2, _, _ = evaluate_with_single_theta(global_model, val_loader_ds2, choice_ds2["theta_list"][0], use_tta=CFG["final_use_tta"])
    test_ds2, y_ds2, p_ds2 = evaluate_with_single_theta(global_model, test_loader_ds2, choice_ds2["theta_list"][0], use_tta=CFG["final_use_tta"])
else:
    val_ds2, _, _ = evaluate_with_multi_theta(global_model, val_loader_ds2, choice_ds2["theta_list"], use_tta=CFG["final_use_tta"])
    test_ds2, y_ds2, p_ds2 = evaluate_with_multi_theta(global_model, test_loader_ds2, choice_ds2["theta_list"], use_tta=CFG["final_use_tta"])

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

val_global = equal_merge_metrics(val_ds1, val_ds2)
test_global = equal_merge_metrics(test_ds1, test_ds2)

# ============================================================
# STEP 12.5: EXTENDED METRICS + ERROR ANALYSIS + CALIBRATION
# ============================================================
print("\n" + "=" * 118)
print("STEP 12.5: EXTENDED METRICS + ERROR ANALYSIS + CALIBRATION")
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

print_table(ext_df, "Extended TEST metrics (DS1 vs DS2)")
print_table(class_df1, "Classwise metrics — DS1 TEST")
print_table(class_df2, "Classwise metrics — DS2 TEST")
print_table(conf_pairs1.head(10), "Top confusion pairs — DS1 TEST")
print_table(conf_pairs2.head(10), "Top confusion pairs — DS2 TEST")
print_table(cal_df1, "Calibration bins — DS1 TEST")
print_table(cal_df2, "Calibration bins — DS2 TEST")

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
    {"setting": "ARCF-Net (validation-selected)", "split": "VAL",  "dataset": "ds1",           **compact_metrics(val_ds1)},
    {"setting": "ARCF-Net (validation-selected)", "split": "VAL",  "dataset": "ds2",           **compact_metrics(val_ds2)},
    {"setting": "ARCF-Net (validation-selected)", "split": "VAL",  "dataset": "global_equal",  **compact_metrics(val_global)},
    {"setting": "ARCF-Net (validation-selected)", "split": "TEST", "dataset": "ds1",           **compact_metrics(test_ds1)},
    {"setting": "ARCF-Net (validation-selected)", "split": "TEST", "dataset": "ds2",           **compact_metrics(test_ds2)},
    {"setting": "ARCF-Net (validation-selected)", "split": "TEST", "dataset": "global_equal",  **compact_metrics(test_global)},
])

print_table(paper_df, "VAL + TEST tables (federated, per-dataset + global equal)")
add_table_to_csv(paper_df, "paper_ready_metrics")

print("\nPaper selection summary:")
print(f"- Best round: {best_round_saved} | best_reward={best_reward:.4f}")
print(f"- DS1 final strategy: {choice_ds1['strategy']} | names={choice_ds1['theta_names']}")
print(f"- DS2 final strategy: {choice_ds2['strategy']} | names={choice_ds2['theta_names']}")

rep_theta_ds1 = choice_ds1["theta_list"][0]
rep_theta_ds2 = choice_ds2["theta_list"][0]

# ============================================================
# STEP 13: PREPROCESSING VALIDATION
# ============================================================
print("\n" + "=" * 118)
print("STEP 13: PREPROCESSING VALIDATION (DS1 + DS2 VAL SAMPLE)")
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
        return pd.DataFrame(), pd.DataFrame(), None, None

    idx = np.random.choice(len(frame), size=n, replace=False)
    ds = MRIDataset(frame, indices=idx.tolist(), tfms=EVAL_TFMS, source_id=0)

    xs = []
    for i in range(len(ds)):
        x, _, *_ = ds[i]
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
    return dfm, summary, x, x_after

preproc_ds1 = theta_to_module(rep_theta_ds1).to(DEVICE)
preproc_ds2 = theta_to_module(rep_theta_ds2).to(DEVICE)

preproc_df1, preproc_summary_df1, _, _ = run_preproc_validation(val1, preproc_ds1, CFG["preproc_val_sample_n"])
preproc_df2, preproc_summary_df2, _, _ = run_preproc_validation(val2, preproc_ds2, CFG["preproc_val_sample_n"])

print_table(preproc_summary_df1, "Preprocessing validation summary (DS1 VAL sample)")
print_table(preproc_summary_df2, "Preprocessing validation summary (DS2 VAL sample)")
add_table_to_csv(preproc_summary_df1, "preprocessing_validation_summary_ds1")
add_table_to_csv(preproc_summary_df2, "preprocessing_validation_summary_ds2")

# ============================================================
# STEP 14: BEFORE vs AFTER PREPROCESSING
# ============================================================
print("\n" + "=" * 118)
print("STEP 14: BEFORE vs AFTER PREPROCESSING IMAGES — BOTH DATASETS")
print("=" * 118)

@torch.no_grad()
def show_before_after(preproc, frame, n=12, title="Preprocessing"):
    per_class = max(1, n // NUM_CLASSES)
    parts = []
    for lab in labels:
        sub = frame[frame["label"] == lab]
        if len(sub) > 0:
            parts.append(sub.sample(n=min(per_class, len(sub)), random_state=SEED))
    if parts:
        sample = pd.concat(parts, axis=0).drop_duplicates(subset=["path"])
    else:
        sample = frame.sample(min(n, len(frame)), random_state=SEED)

    if len(sample) < n:
        extra = frame.sample(min(n - len(sample), len(frame)), random_state=SEED + 3)
        sample = pd.concat([sample, extra], axis=0).drop_duplicates(subset=["path"])

    sample = sample.sample(min(n, len(sample)), random_state=SEED).reset_index(drop=True)
    ds = MRIDataset(sample, indices=list(range(len(sample))), tfms=EVAL_TFMS, source_id=0)

    xs, labs_out = [], []
    for i in range(len(ds)):
        x, y, *_ = ds[i]
        xs.append(x)
        labs_out.append(id2label[int(y)])

    x = torch.stack(xs).to(DEVICE)
    x_after = preproc(x).clamp(0, 1)

    plot_dual_image_grid(
        [z.cpu() for z in x],
        [z.cpu() for z in x_after],
        labs_out,
        title
    )

if CFG["make_plots"] and CFG["use_preprocessing"]:
    show_before_after(
        preproc_ds1,
        test1,
        n=CFG["before_after_n"],
        title=f"Before vs After RACE-FELCM | DS1 representative = {theta_str(rep_theta_ds1)}"
    )
    show_before_after(
        preproc_ds2,
        test2,
        n=CFG["before_after_n"],
        title=f"Before vs After RACE-FELCM | DS2 representative = {theta_str(rep_theta_ds2)}"
    )

# ============================================================
# STEP 15: FINAL REPORT PLOTS — BOTH DATASETS
# ============================================================
print("\n" + "=" * 118)
print("STEP 15: FINAL REPORT PLOTS — BOTH DATASETS")
print("=" * 118)

def draw_confusion_matrix(ax, cm, title, normalized=False):
    im = ax.imshow(cm, interpolation="nearest", aspect="equal", cmap="viridis")
    ax.set_title(title, fontweight="bold")
    ax.set_xticks(range(NUM_CLASSES))
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_yticks(range(NUM_CLASSES))
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted", fontweight="bold")
    ax.set_ylabel("True", fontweight="bold")
    ax.grid(False)
    ax.minorticks_off()
    ax.set_axisbelow(False)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    thresh = 0.5 if normalized else (cm.max() / 2.0 if cm.size > 0 else 0.0)
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            v = cm[i, j]
            txt = f"{v:.2f}" if normalized else str(int(v))
            ax.text(
                j, i, txt,
                ha="center", va="center",
                fontweight="bold",
                color="white" if v > thresh else "black"
            )
    return im

def plot_report_pack(y_true, p_pred, ds_title):
    if len(y_true) == 0:
        return

    y_hat = np.argmax(p_pred, axis=1)
    cm_counts = confusion_matrix(y_true, y_hat, labels=list(range(NUM_CLASSES)))
    cm_norm = cm_counts / np.clip(cm_counts.sum(axis=1, keepdims=True), 1, None)
    cal_df, ece, mce = top_label_calibration(y_true, p_pred, n_bins=CFG["calibration_bins"])
    conf_tbl = wrong_right_confidence_table(y_true, p_pred)

    fig = plt.figure(figsize=(18, 12), constrained_layout=True, facecolor="white")
    gs = fig.add_gridspec(2, 2)

    ax1 = fig.add_subplot(gs[0, 0])
    for c in range(NUM_CLASSES):
        yc = (y_true == c).astype(int)
        if yc.sum() == 0 or yc.sum() == len(yc):
            continue
        fpr, tpr, _ = roc_curve(yc, p_pred[:, c])
        auc_c = roc_auc_score(yc, p_pred[:, c])
        ax1.plot(fpr, tpr, label=f"{labels[c]} (AUC={auc_c:.3f})")
    ax1.plot([0, 1], [0, 1], linestyle="--", linewidth=2)
    ax1.set_title(f"ROC Curves (OvR) — {ds_title}", fontweight="bold")
    ax1.set_xlabel("False Positive Rate", fontweight="bold")
    ax1.set_ylabel("True Positive Rate", fontweight="bold")
    ax1.legend(loc="lower right")
    ax1.grid(alpha=0.20)

    ax2 = fig.add_subplot(gs[0, 1])
    for c in range(NUM_CLASSES):
        yc = (y_true == c).astype(int)
        if yc.sum() == 0:
            continue
        prec, rec, _ = precision_recall_curve(yc, p_pred[:, c])
        ap = average_precision_score(yc, p_pred[:, c])
        ax2.plot(rec, prec, label=f"{labels[c]} (AP={ap:.3f})")
    ax2.set_title(f"Precision–Recall Curves — {ds_title}", fontweight="bold")
    ax2.set_xlabel("Recall", fontweight="bold")
    ax2.set_ylabel("Precision", fontweight="bold")
    ax2.legend(loc="lower left")
    ax2.grid(alpha=0.20)

    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot([0, 1], [0, 1], linestyle="--", linewidth=2, label="Perfect")
    ax3.plot(cal_df["bin_confidence"], cal_df["bin_accuracy"], marker="o", linewidth=2.6, label="Model")
    ax3.set_title(f"Reliability Diagram — {ds_title}\nECE={ece:.4f} | MCE={mce:.4f}", fontweight="bold")
    ax3.set_xlabel("Confidence", fontweight="bold")
    ax3.set_ylabel("Accuracy", fontweight="bold")
    ax3.legend()
    ax3.grid(alpha=0.20)

    ax4 = fig.add_subplot(gs[1, 1])
    im = draw_confusion_matrix(ax4, cm_norm, f"Confusion Matrix (Row-normalized) — {ds_title}", normalized=True)
    cbar = fig.colorbar(im, ax=ax4, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel("Normalized value", rotation=270, labelpad=18, fontweight="bold")

    fig.suptitle(f"{ds_title} — Performance Dashboard", fontsize=20, fontweight="bold")
    plt.show()

    fig2, axc = plt.subplots(figsize=(7.6, 6.4), constrained_layout=True, facecolor="white")
    im2 = draw_confusion_matrix(axc, cm_counts, f"Confusion Matrix (Counts) — {ds_title}", normalized=False)
    fig2.colorbar(im2, ax=axc, fraction=0.046, pad=0.04)
    plt.show()

    fig3, ax = plt.subplots(figsize=(10, 5), constrained_layout=True, facecolor="white")
    wrong_conf = conf_tbl.loc[conf_tbl["status"] == "wrong", "confidence"].values
    right_conf = conf_tbl.loc[conf_tbl["status"] == "correct", "confidence"].values
    bins = np.linspace(0, 1, 16)
    ax.hist(right_conf, bins=bins, alpha=0.65, label="Correct")
    ax.hist(wrong_conf, bins=bins, alpha=0.65, label="Wrong")
    ax.set_title(f"Confidence Histogram — Correct vs Wrong — {ds_title}", fontweight="bold")
    ax.set_xlabel("Confidence", fontweight="bold")
    ax.set_ylabel("Count", fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.20)
    plt.show()

    return cal_df

if CFG["make_plots"]:
    plot_report_pack(y_ds1, p_ds1, "DS1 TEST")
    plot_report_pack(y_ds2, p_ds2, "DS2 TEST")

# ------------------------------------------------------------
# DS1 vs DS2 comparison plots
# ------------------------------------------------------------
def comparison_bar_plot(metric_names, values_a, values_b, label_a, label_b, title):
    x = np.arange(len(metric_names))
    w = 0.36
    fig, ax = plt.subplots(figsize=(max(10, 1.2 * len(metric_names)), 5.5), constrained_layout=True, facecolor="white")
    ax.bar(x - w/2, values_a, width=w, label=label_a)
    ax.bar(x + w/2, values_b, width=w, label=label_b)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names, rotation=25, ha="right")
    ax.set_title(title, fontweight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.20)
    plt.show()

def plot_dataset_comparison_panels():
    comparison_bar_plot(
        ["acc", "f1_macro", "mcc", "kappa", "auc_roc_macro_ovr", "balanced_acc"],
        [ext_ds1["acc"], ext_ds1["f1_macro"], ext_ds1["mcc"], ext_ds1["kappa"], ext_ds1["auc_roc_macro_ovr"], ext_ds1["balanced_acc"]],
        [ext_ds2["acc"], ext_ds2["f1_macro"], ext_ds2["mcc"], ext_ds2["kappa"], ext_ds2["auc_roc_macro_ovr"], ext_ds2["balanced_acc"]],
        "DS1", "DS2", "DS1 vs DS2 — Core Metrics"
    )

    comparison_bar_plot(
        ["ppv_macro", "npv_macro", "specificity_macro", "fpr_macro", "fnr_macro", "jaccard_macro"],
        [ext_ds1["ppv_macro"], ext_ds1["npv_macro"], ext_ds1["specificity_macro"], ext_ds1["fpr_macro"], ext_ds1["fnr_macro"], ext_ds1["jaccard_macro"]],
        [ext_ds2["ppv_macro"], ext_ds2["npv_macro"], ext_ds2["specificity_macro"], ext_ds2["fpr_macro"], ext_ds2["fnr_macro"], ext_ds2["jaccard_macro"]],
        "DS1", "DS2", "DS1 vs DS2 — Error Metrics"
    )

    comparison_bar_plot(
        ["ECE", "MCE", "Brier", "LogLoss"],
        [ext_ds1["ece"], ext_ds1["mce"], ext_ds1["brier_multi"], ext_ds1["log_loss"]],
        [ext_ds2["ece"], ext_ds2["mce"], ext_ds2["brier_multi"], ext_ds2["log_loss"]],
        "DS1", "DS2", "DS1 vs DS2 — Calibration Metrics"
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True, facecolor="white")
    axes[0].plot(cal_df1["bin_confidence"], cal_df1["bin_accuracy"], marker="o", label="DS1")
    axes[0].plot(cal_df2["bin_confidence"], cal_df2["bin_accuracy"], marker="o", label="DS2")
    axes[0].plot([0, 1], [0, 1], linestyle="--", linewidth=2)
    axes[0].set_title("Reliability Curves — DS1 vs DS2", fontweight="bold")
    axes[0].set_xlabel("Confidence", fontweight="bold")
    axes[0].set_ylabel("Accuracy", fontweight="bold")
    axes[0].legend()
    axes[0].grid(alpha=0.20)

    axes[1].bar(np.arange(len(cal_df1)), cal_df1["bin_count"], alpha=0.65, label="DS1")
    axes[1].bar(np.arange(len(cal_df2)), cal_df2["bin_count"], alpha=0.65, label="DS2")
    axes[1].set_title("Calibration Bin Counts — DS1 vs DS2", fontweight="bold")
    axes[1].set_xlabel("Bin ID", fontweight="bold")
    axes[1].set_ylabel("Count", fontweight="bold")
    axes[1].legend()
    axes[1].grid(axis="y", alpha=0.20)
    plt.show()

    conf1 = wrong_right_confidence_table(y_ds1, p_ds1)
    conf2 = wrong_right_confidence_table(y_ds2, p_ds2)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True, facecolor="white")
    bins = np.linspace(0, 1, 16)
    axes[0].hist(conf1.loc[conf1["status"] == "correct", "confidence"], bins=bins, alpha=0.65, label="Correct")
    axes[0].hist(conf1.loc[conf1["status"] == "wrong", "confidence"], bins=bins, alpha=0.65, label="Wrong")
    axes[0].set_title("Confidence Histogram — DS1", fontweight="bold")
    axes[0].set_xlabel("Confidence", fontweight="bold")
    axes[0].set_ylabel("Count", fontweight="bold")
    axes[0].legend()
    axes[0].grid(alpha=0.20)

    axes[1].hist(conf2.loc[conf2["status"] == "correct", "confidence"], bins=bins, alpha=0.65, label="Correct")
    axes[1].hist(conf2.loc[conf2["status"] == "wrong", "confidence"], bins=bins, alpha=0.65, label="Wrong")
    axes[1].set_title("Confidence Histogram — DS2", fontweight="bold")
    axes[1].set_xlabel("Confidence", fontweight="bold")
    axes[1].set_ylabel("Count", fontweight="bold")
    axes[1].legend()
    axes[1].grid(alpha=0.20)
    plt.show()

    fig, axes = plt.subplots(1, 2, figsize=(16, 5.5), constrained_layout=True, facecolor="white")
    top1 = conf_pairs1.head(8)
    top2 = conf_pairs2.head(8)
    y1 = np.arange(len(top1))
    y2 = np.arange(len(top2))
    axes[0].barh(y1, top1["count"])
    axes[0].set_yticks(y1)
    axes[0].set_yticklabels([f"{a}→{b}" for a, b in zip(top1["true_class"], top1["pred_class"])])
    axes[0].invert_yaxis()
    axes[0].set_title("Top Confusion Pairs — DS1", fontweight="bold")
    axes[0].grid(axis="x", alpha=0.20)

    axes[1].barh(y2, top2["count"])
    axes[1].set_yticks(y2)
    axes[1].set_yticklabels([f"{a}→{b}" for a, b in zip(top2["true_class"], top2["pred_class"])])
    axes[1].invert_yaxis()
    axes[1].set_title("Top Confusion Pairs — DS2", fontweight="bold")
    axes[1].grid(axis="x", alpha=0.20)
    plt.show()

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), constrained_layout=True, facecolor="white")
    x = np.arange(NUM_CLASSES)
    w = 0.36

    axes[0].bar(x - w/2, class_df1["ppv"], width=w, label="DS1")
    axes[0].bar(x + w/2, class_df2["ppv"], width=w, label="DS2")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=20, ha="right")
    axes[0].set_title("PPV by Class", fontweight="bold")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.20)

    axes[1].bar(x - w/2, class_df1["npv"], width=w, label="DS1")
    axes[1].bar(x + w/2, class_df2["npv"], width=w, label="DS2")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=20, ha="right")
    axes[1].set_title("NPV by Class", fontweight="bold")
    axes[1].legend()
    axes[1].grid(axis="y", alpha=0.20)

    axes[2].bar(x - w/2, class_df1["specificity"], width=w, label="DS1")
    axes[2].bar(x + w/2, class_df2["specificity"], width=w, label="DS2")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels, rotation=20, ha="right")
    axes[2].set_title("Specificity by Class", fontweight="bold")
    axes[2].legend()
    axes[2].grid(axis="y", alpha=0.20)
    plt.show()

if CFG["make_plots"]:
    plot_dataset_comparison_panels()

# ============================================================
# STEP 16: RADAR + EVOLUTION PLOTS
# ============================================================
print("\n" + "=" * 118)
print("STEP 16: RADAR + EVOLUTION PLOTS")
print("=" * 118)

def radar_plot(metrics_a, metrics_b, axes_keys, title):
    vals_a = [metrics_a.get(k, np.nan) for k in axes_keys]
    vals_b = [metrics_b.get(k, np.nan) for k in axes_keys]
    angles = np.linspace(0, 2 * np.pi, len(axes_keys), endpoint=False).tolist()
    vals_a += vals_a[:1]
    vals_b += vals_b[:1]
    angles += angles[:1]

    fig = plt.figure(figsize=(8, 7), constrained_layout=True, facecolor="white")
    ax = plt.subplot(111, polar=True)
    ax.plot(angles, vals_a, linewidth=2.8, label="DS1")
    ax.fill(angles, vals_a, alpha=0.15)
    ax.plot(angles, vals_b, linewidth=2.8, label="DS2")
    ax.fill(angles, vals_b, alpha=0.15)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(axes_keys, fontweight="bold")
    ax.set_title(title, y=1.08, fontweight="bold", fontsize=16)
    ax.legend(loc="upper right", bbox_to_anchor=(1.22, 1.15))
    plt.show()

def plot_global_training_dashboard(glob_df):
    fig = plt.figure(figsize=(16, 10), constrained_layout=True, facecolor="white")
    gs = fig.add_gridspec(2, 2)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(glob_df["round"], glob_df["global_acc"], marker="o", label="Global Acc")
    ax1.plot(glob_df["round"], glob_df["global_f1_macro"], marker="o", label="Global F1")
    ax1.plot(glob_df["round"], glob_df["ds1_acc"], marker="o", linestyle="--", label="DS1 Acc")
    ax1.plot(glob_df["round"], glob_df["ds2_acc"], marker="o", linestyle="--", label="DS2 Acc")
    ax1.set_title("Accuracy / F1 over Rounds", fontweight="bold")
    ax1.set_xlabel("Round", fontweight="bold")
    ax1.legend(ncol=2)
    ax1.grid(alpha=0.20)

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(glob_df["round"], glob_df["global_log_loss"], marker="o", label="Global LogLoss")
    ax2.plot(glob_df["round"], glob_df["ds1_log_loss"], marker="o", linestyle="--", label="DS1 LogLoss")
    ax2.plot(glob_df["round"], glob_df["ds2_log_loss"], marker="o", linestyle="--", label="DS2 LogLoss")
    ax2.set_title("LogLoss over Rounds", fontweight="bold")
    ax2.set_xlabel("Round", fontweight="bold")
    ax2.legend()
    ax2.grid(alpha=0.20)

    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(glob_df["round"], glob_df["global_reward"], marker="o", label="Best-Round Score")
    ax3.set_title("Balanced Selection Score over Rounds", fontweight="bold")
    ax3.set_xlabel("Round", fontweight="bold")
    ax3.legend()
    ax3.grid(alpha=0.20)

    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(glob_df["round"], glob_df["round_time_s"], marker="o", label="Round Time (s)")
    ax4.set_title("Round Time over Rounds", fontweight="bold")
    ax4.set_xlabel("Round", fontweight="bold")
    ax4.legend()
    ax4.grid(alpha=0.20)

    fig.suptitle("Federated Training Dashboard", fontsize=20, fontweight="bold")
    plt.show()

def plot_client_evolution(loc_df):
    for ds_name in ["ds1", "ds2"]:
        sub_all = loc_df[loc_df["dataset"] == ds_name].copy()
        if len(sub_all) == 0:
            continue

        fig1, ax1 = plt.subplots(figsize=(13, 5), constrained_layout=True, facecolor="white")
        for cname in sorted(sub_all["client"].unique()):
            sub = sub_all[sub_all["client"] == cname].sort_values("round")
            ax1.plot(sub["round"], sub["val_acc"], marker="o", label=f"{cname} val_acc")
        ax1.set_title(f"Client Validation Accuracy Evolution — {ds_name.upper()}", fontweight="bold")
        ax1.set_xlabel("Round", fontweight="bold")
        ax1.set_ylabel("val_acc", fontweight="bold")
        ax1.legend(ncol=2)
        ax1.grid(alpha=0.20)
        plt.show()

        fig2, ax2 = plt.subplots(figsize=(13, 5), constrained_layout=True, facecolor="white")
        for cname in sorted(sub_all["client"].unique()):
            sub = sub_all[sub_all["client"] == cname].sort_values("round")
            ax2.plot(sub["round"], sub["val_f1_macro"], marker="o", label=f"{cname} val_f1")
        ax2.set_title(f"Client Macro-F1 Evolution — {ds_name.upper()}", fontweight="bold")
        ax2.set_xlabel("Round", fontweight="bold")
        ax2.set_ylabel("val_f1_macro", fontweight="bold")
        ax2.legend(ncol=2)
        ax2.grid(alpha=0.20)
        plt.show()

def plot_theta_evolution(loc_df):
    theta_cols = [
        "gamma_power",
        "alpha_contrast_weight",
        "beta_contrast_sharpness",
        "tau_clip",
        "k_blur_kernel_size",
        "edge_gain",
        "blend_mix",
    ]
    loc_copy = loc_df.copy()
    for c in theta_cols:
        if c in loc_copy.columns:
            loc_copy[c] = pd.to_numeric(loc_copy[c], errors="coerce")

    theta_evo = loc_copy.groupby("round")[theta_cols].mean(numeric_only=True).reset_index()
    print_table(theta_evo, "Mean selected preprocessing parameters over rounds")
    add_table_to_csv(theta_evo, "theta_evolution_mean")

    fig, ax = plt.subplots(figsize=(15, 6), constrained_layout=True, facecolor="white")
    for col in theta_cols:
        ax.plot(theta_evo["round"], theta_evo[col], marker="o", label=col)
    ax.set_title("RACE-FELCM Parameter Evolution (Mean across clients)", fontweight="bold")
    ax.set_xlabel("Round", fontweight="bold")
    ax.legend(ncol=2)
    ax.grid(alpha=0.20)
    plt.show()

if CFG["make_plots"]:
    rad_keys = ["acc", "f1_macro", "precision_macro", "recall_macro", "auc_roc_macro_ovr", "mcc", "kappa", "log_loss"]
    radar_plot(test_ds1, test_ds2, rad_keys, "TEST Metrics Radar (DS1 vs DS2)")
    plot_global_training_dashboard(glob_df)
    plot_client_evolution(loc_df)
    plot_theta_evolution(loc_df)

# ============================================================
# STEP 17: SAVE CHECKPOINT + CSV
# ============================================================
print("\n" + "=" * 118)
print("STEP 17: SAVING ONLY TWO FILES (CHECKPOINT + ONE CSV)")
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

    "adaptive_clients_ds1": n_clients_ds1,
    "adaptive_clients_ds2": n_clients_ds2,
    "clients_total": CLIENTS_TOTAL,
    "shared_planner_state": shared_planner.state_dict(),
    "planner_history_shared": plan_df_shared.to_dict(orient="list"),

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
    "best_rl_selected_per_client": client_theta_df.to_dict(orient="list"),

    "final_choice_ds1": {
        "strategy": choice_ds1["strategy"],
        "theta_names": choice_ds1["theta_names"],
        "theta_list": choice_ds1["theta_list"],
        "score": choice_ds1["score"],
        "val_acc": choice_ds1["val_acc"],
        "val_f1": choice_ds1["val_f1"],
    },
    "final_choice_ds2": {
        "strategy": choice_ds2["strategy"],
        "theta_names": choice_ds2["theta_names"],
        "theta_list": choice_ds2["theta_list"],
        "score": choice_ds2["score"],
        "val_acc": choice_ds2["val_acc"],
        "val_f1": choice_ds2["val_f1"],
    },
    "representative_theta_ds1": rep_theta_ds1,
    "representative_theta_ds2": rep_theta_ds2,
    "representative_theta_str_ds1": theta_str(rep_theta_ds1),
    "representative_theta_str_ds2": theta_str(rep_theta_ds2),

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
    "adaptive_client_distribution_table": dist_df.to_dict(orient="list"),
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
print(f"✅ Saved checkpoint: {MODEL_PATH}")

all_df.to_csv(CSV_PATH, index=False)
print(f"✅ Saved CSV (ALL outputs): {CSV_PATH}")

print("\nDONE ✅")
print(f"Method: {METHOD_INFO['acronym']} = {METHOD_INFO['full_form']}")
print(f"Backbone: {METHOD_INFO['backbone_full_form']}")
print(f"Best round: {best_round_saved}")
print(f"Adaptive clients => DS1={n_clients_ds1}, DS2={n_clients_ds2}, TOTAL={CLIENTS_TOTAL}")
print(f"Rounds completed: {CFG['rounds']}")
print(f"Global TEST acc: {safe_float(test_global.get('acc')):.4f}")
print(f"Global TEST f1_macro: {safe_float(test_global.get('f1_macro')):.4f}")
print(f"DS1 TEST acc: {safe_float(test_ds1.get('acc')):.4f}")
print(f"DS2 TEST acc: {safe_float(test_ds2.get('acc')):.4f}")
print(f"DS1 final strategy: {choice_ds1['strategy']} | names={choice_ds1['theta_names']}")
print(f"DS2 final strategy: {choice_ds2['strategy']} | names={choice_ds2['theta_names']}")
```

    ======================================================================================================================
    TRUE FL + RL-UCB + RACE-FELCM + CRAF + ResNet-50
    METHOD: ARCF-Net (Adaptive RACE-FELCM with CRAF Fusion Network)
    ======================================================================================================================
    ENV: KAGGLE | DEVICE: cuda | torch=2.10.0+cu128
    ======================================================================================================================
    
    ======================================================================================================================
    STEP 0: ACCESS DATASETS
    ======================================================================================================================
    Warning: Looks like you're using an outdated `kagglehub` version (installed: 0.3.13), please consider upgrading to the latest version (1.0.0).
    Downloading from https://www.kaggle.com/api/v1/datasets/download/orvile/pmram-bangladeshi-brain-cancer-mri-dataset?dataset_version_number=2...


    100%|██████████| 161M/161M [00:01<00:00, 124MB/s]

    Extracting files...


    


    Warning: Looks like you're using an outdated `kagglehub` version (installed: 0.3.13), please consider upgrading to the latest version (1.0.0).
    Downloading from https://www.kaggle.com/api/v1/datasets/download/yassinebazgour/preprocessed-brain-mri-scans-for-tumors-detection?dataset_version_number=1...


    100%|██████████| 130M/130M [00:01<00:00, 127MB/s]

    Extracting files...


    


    Dataset-1 RAW root detected:
      /root/.cache/kagglehub/datasets/orvile/pmram-bangladeshi-brain-cancer-mri-dataset/versions/2/PMRAM Bangladeshi Brain Cancer - MRI Dataset/PMRAM Bangladeshi Brain Cancer - MRI Dataset/Raw Data/Raw
    Dataset-2 root detected:
      /root/.cache/kagglehub/datasets/yassinebazgour/preprocessed-brain-mri-scans-for-tumors-detection/versions/1/preprocessed_brain_mri_dataset
    
    ======================================================================================================================
    STEP 1: BUILD DATA MANIFESTS (NO MERGE)
    ======================================================================================================================
    
    ----------------------------------------------------------------------------------------------------------------------
    Building Dataset-1 (RAW only)
    ds1_raw: 512Glioma -> glioma | 373 images
    ds1_raw: 512Meningioma -> meningioma | 363 images
    ds1_raw: 512Normal -> notumor | 396 images
    ds1_raw: 512Pituitary -> pituitary | 373 images
    Building Dataset-2
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
    Name: count, dtype: int64
    
    Dataset-2 images: 7031
    label
    glioma        1621
    meningioma    1646
    notumor       2000
    pituitary     1764
    Name: count, dtype: int64
    
    ======================================================================================================================
    STEP 2: TRAIN / VAL / TEST SPLIT (PER DATASET)
    ======================================================================================================================
    DS1 TRAIN: 1053 | VAL: 226 | TEST: 226
    DS2 TRAIN: 4921 | VAL: 1055 | TEST: 1055
    
    ======================================================================================================================
    STEP 2.5: SANITY / LEAKAGE CHECKS
    ======================================================================================================================
    
    ----------------------------------------------------------------------------------------------------------------------
    Leakage / Sanity Summary — ds1
    ----------------------------------------------------------------------------------------------------------------------




  <div id="df-34c398e0-faf5-4a5a-876b-238f0c79afd0" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>path_overlap_train_val</th>
      <th>path_overlap_train_test</th>
      <th>path_overlap_val_test</th>
      <th>unique_paths_train</th>
      <th>unique_paths_val</th>
      <th>unique_paths_test</th>
      <th>filename_overlap_train_val</th>
      <th>filename_overlap_train_test</th>
      <th>filename_overlap_val_test</th>
      <th>subset_hash_train_val</th>
      <th>subset_hash_train_test</th>
      <th>subset_hash_val_test</th>
      <th>subset_hash_n_train</th>
      <th>subset_hash_n_val</th>
      <th>subset_hash_n_test</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1053</td>
      <td>226</td>
      <td>226</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>9</td>
      <td>4</td>
      <td>7</td>
      <td>297</td>
      <td>224</td>
      <td>225</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-34c398e0-faf5-4a5a-876b-238f0c79afd0')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-34c398e0-faf5-4a5a-876b-238f0c79afd0 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-34c398e0-faf5-4a5a-876b-238f0c79afd0');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


    </div>
  </div>



    
    ----------------------------------------------------------------------------------------------------------------------
    Leakage / Sanity Summary — ds2
    ----------------------------------------------------------------------------------------------------------------------




  <div id="df-1a919dfc-7965-4b2e-a77a-77e2c8481c8d" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>path_overlap_train_val</th>
      <th>path_overlap_train_test</th>
      <th>path_overlap_val_test</th>
      <th>unique_paths_train</th>
      <th>unique_paths_val</th>
      <th>unique_paths_test</th>
      <th>filename_overlap_train_val</th>
      <th>filename_overlap_train_test</th>
      <th>filename_overlap_val_test</th>
      <th>subset_hash_train_val</th>
      <th>subset_hash_train_test</th>
      <th>subset_hash_val_test</th>
      <th>subset_hash_n_train</th>
      <th>subset_hash_n_val</th>
      <th>subset_hash_n_test</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>4921</td>
      <td>1055</td>
      <td>1055</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>300</td>
      <td>297</td>
      <td>298</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-1a919dfc-7965-4b2e-a77a-77e2c8481c8d')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-1a919dfc-7965-4b2e-a77a-77e2c8481c8d button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-1a919dfc-7965-4b2e-a77a-77e2c8481c8d');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


    </div>
  </div>



    
    ======================================================================================================================
    STEP 3: RL-UCB BANDIT
    ======================================================================================================================
    
    ======================================================================================================================
    STEP 4: SHARED ADAPTIVE CLIENT COUNT BY RL-UCB
    ======================================================================================================================
    Chosen shared adaptive clients for DS1: 3
    Chosen shared adaptive clients for DS2: 3
    
    ----------------------------------------------------------------------------------------------------------------------
    RL planning history — shared client count
    ----------------------------------------------------------------------------------------------------------------------




  <div id="df-6981eadc-ee40-44ae-9459-a079a29c5f75" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>episode</th>
      <th>selected_n_clients</th>
      <th>reward_ds1</th>
      <th>reward_ds2</th>
      <th>reward_mean</th>
      <th>bandit_value</th>
      <th>pulls_for_arm</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>3</td>
      <td>0.709483</td>
      <td>0.663089</td>
      <td>0.686286</td>
      <td>0.686286</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>4</td>
      <td>0.721983</td>
      <td>0.738356</td>
      <td>0.730169</td>
      <td>0.730169</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>5</td>
      <td>0.675439</td>
      <td>0.698698</td>
      <td>0.687069</td>
      <td>0.687069</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>4</td>
      <td>0.753396</td>
      <td>0.829472</td>
      <td>0.791434</td>
      <td>0.760801</td>
      <td>2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>5</td>
      <td>0.702440</td>
      <td>0.692132</td>
      <td>0.697286</td>
      <td>0.692177</td>
      <td>2</td>
    </tr>
    <tr>
      <th>5</th>
      <td>6</td>
      <td>3</td>
      <td>0.870872</td>
      <td>0.708757</td>
      <td>0.789815</td>
      <td>0.738050</td>
      <td>2</td>
    </tr>
    <tr>
      <th>6</th>
      <td>7</td>
      <td>4</td>
      <td>0.705301</td>
      <td>0.608192</td>
      <td>0.656747</td>
      <td>0.726116</td>
      <td>3</td>
    </tr>
    <tr>
      <th>7</th>
      <td>8</td>
      <td>3</td>
      <td>0.766992</td>
      <td>0.753060</td>
      <td>0.760026</td>
      <td>0.745376</td>
      <td>3</td>
    </tr>
    <tr>
      <th>8</th>
      <td>9</td>
      <td>5</td>
      <td>0.664448</td>
      <td>0.652680</td>
      <td>0.658564</td>
      <td>0.680973</td>
      <td>3</td>
    </tr>
    <tr>
      <th>9</th>
      <td>10</td>
      <td>3</td>
      <td>0.753155</td>
      <td>0.660297</td>
      <td>0.706726</td>
      <td>0.735713</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-6981eadc-ee40-44ae-9459-a079a29c5f75')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-6981eadc-ee40-44ae-9459-a079a29c5f75 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-6981eadc-ee40-44ae-9459-a079a29c5f75');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


  <div id="id_f85ecfa0-748f-4895-b48e-d07bb8d35f25">
    <style>
      .colab-df-generate {
        background-color: #E8F0FE;
        border: none;
        border-radius: 50%;
        cursor: pointer;
        display: none;
        fill: #1967D2;
        height: 32px;
        padding: 0 0 0 0;
        width: 32px;
      }

      .colab-df-generate:hover {
        background-color: #E2EBFA;
        box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
        fill: #174EA6;
      }

      [theme=dark] .colab-df-generate {
        background-color: #3B4455;
        fill: #D2E3FC;
      }

      [theme=dark] .colab-df-generate:hover {
        background-color: #434B5C;
        box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
        filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
        fill: #FFFFFF;
      }
    </style>
    <button class="colab-df-generate" onclick="generateWithVariable('plan_df_shared')"
            title="Generate code using this dataframe."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M7,19H8.4L18.45,9,17,7.55,7,17.6ZM5,21V16.75L18.45,3.32a2,2,0,0,1,2.83,0l1.4,1.43a1.91,1.91,0,0,1,.58,1.4,1.91,1.91,0,0,1-.58,1.4L9.25,21ZM18.45,9,17,7.55Zm-12,3A5.31,5.31,0,0,0,4.9,8.1,5.31,5.31,0,0,0,1,6.5,5.31,5.31,0,0,0,4.9,4.9,5.31,5.31,0,0,0,6.5,1,5.31,5.31,0,0,0,8.1,4.9,5.31,5.31,0,0,0,12,6.5,5.46,5.46,0,0,0,6.5,12Z"/>
  </svg>
    </button>
    <script>
      (() => {
      const buttonEl =
        document.querySelector('#id_f85ecfa0-748f-4895-b48e-d07bb8d35f25 button.colab-df-generate');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      buttonEl.onclick = () => {
        google.colab.notebook.generateWithVariable('plan_df_shared');
      }
      })();
    </script>
  </div>

    </div>
  </div>



    
    ======================================================================================================================
    STEP 5: FINAL NON-IID CLIENT PARTITIONING
    ======================================================================================================================
    
    ======================================================================================================================
    STEP 6: DATA LOADERS (AUG ON)
    ======================================================================================================================
    ds1 | client_0 | train=271 | tune=42 | val=37
    ds1 | client_1 | train=167 | tune=27 | val=23
    ds1 | client_2 | train=375 | tune=59 | val=52
    ds2 | client_3 | train=1092 | tune=170 | val=150
    ds2 | client_4 | train=1690 | tune=262 | val=231
    ds2 | client_5 | train=1026 | tune=160 | val=140
    
    ----------------------------------------------------------------------------------------------------------------------
    Adaptive client class distribution
    ----------------------------------------------------------------------------------------------------------------------




  <div id="df-599ad912-e694-4cea-9e48-17bdf2d415c2" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>client</th>
      <th>dataset</th>
      <th>total_train</th>
      <th>total_tune</th>
      <th>total_val</th>
      <th>glioma</th>
      <th>meningioma</th>
      <th>notumor</th>
      <th>pituitary</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>client_0</td>
      <td>ds1</td>
      <td>271</td>
      <td>42</td>
      <td>37</td>
      <td>77</td>
      <td>34</td>
      <td>79</td>
      <td>81</td>
    </tr>
    <tr>
      <th>1</th>
      <td>client_1</td>
      <td>ds1</td>
      <td>167</td>
      <td>27</td>
      <td>23</td>
      <td>4</td>
      <td>45</td>
      <td>5</td>
      <td>113</td>
    </tr>
    <tr>
      <th>2</th>
      <td>client_2</td>
      <td>ds1</td>
      <td>375</td>
      <td>59</td>
      <td>52</td>
      <td>121</td>
      <td>117</td>
      <td>130</td>
      <td>7</td>
    </tr>
    <tr>
      <th>3</th>
      <td>client_3</td>
      <td>ds2</td>
      <td>1092</td>
      <td>170</td>
      <td>150</td>
      <td>39</td>
      <td>251</td>
      <td>223</td>
      <td>579</td>
    </tr>
    <tr>
      <th>4</th>
      <td>client_4</td>
      <td>ds2</td>
      <td>1690</td>
      <td>262</td>
      <td>231</td>
      <td>275</td>
      <td>209</td>
      <td>852</td>
      <td>354</td>
    </tr>
    <tr>
      <th>5</th>
      <td>client_5</td>
      <td>ds2</td>
      <td>1026</td>
      <td>160</td>
      <td>140</td>
      <td>563</td>
      <td>433</td>
      <td>8</td>
      <td>22</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-599ad912-e694-4cea-9e48-17bdf2d415c2')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-599ad912-e694-4cea-9e48-17bdf2d415c2 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-599ad912-e694-4cea-9e48-17bdf2d415c2');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


  <div id="id_f5a1c32e-f303-4b09-ae52-da962878d7af">
    <style>
      .colab-df-generate {
        background-color: #E8F0FE;
        border: none;
        border-radius: 50%;
        cursor: pointer;
        display: none;
        fill: #1967D2;
        height: 32px;
        padding: 0 0 0 0;
        width: 32px;
      }

      .colab-df-generate:hover {
        background-color: #E2EBFA;
        box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
        fill: #174EA6;
      }

      [theme=dark] .colab-df-generate {
        background-color: #3B4455;
        fill: #D2E3FC;
      }

      [theme=dark] .colab-df-generate:hover {
        background-color: #434B5C;
        box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
        filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
        fill: #FFFFFF;
      }
    </style>
    <button class="colab-df-generate" onclick="generateWithVariable('dist_df')"
            title="Generate code using this dataframe."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M7,19H8.4L18.45,9,17,7.55,7,17.6ZM5,21V16.75L18.45,3.32a2,2,0,0,1,2.83,0l1.4,1.43a1.91,1.91,0,0,1,.58,1.4,1.91,1.91,0,0,1-.58,1.4L9.25,21ZM18.45,9,17,7.55Zm-12,3A5.31,5.31,0,0,0,4.9,8.1,5.31,5.31,0,0,0,1,6.5,5.31,5.31,0,0,0,4.9,4.9,5.31,5.31,0,0,0,6.5,1,5.31,5.31,0,0,0,8.1,4.9,5.31,5.31,0,0,0,12,6.5,5.46,5.46,0,0,0,6.5,12Z"/>
  </svg>
    </button>
    <script>
      (() => {
      const buttonEl =
        document.querySelector('#id_f5a1c32e-f303-4b09-ae52-da962878d7af button.colab-df-generate');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      buttonEl.onclick = () => {
        google.colab.notebook.generateWithVariable('dist_df');
      }
      })();
    </script>
  </div>

    </div>
  </div>



    Augmentation: ON ✅
    Preprocessing: ON ✅
    Total adaptive clients: 6
    
    ----------------------------------------------------------------------------------------------------------------------
    AUGMENTATION VISUAL CHECK (Before vs After) — BOTH DATASETS
    ----------------------------------------------------------------------------------------------------------------------



    
![png](01_1_FedRACE_Net_Federated_Robust_Adaptive_Context_Enhanced_Fuzzy_Edge_Local_Contrast_Mapping_Fusion_Network_files/01_1_FedRACE_Net_Federated_Robust_Adaptive_Context_Enhanced_Fuzzy_Edge_Local_Contrast_Mapping_Fusion_Network_0_17.png)
    



    
![png](01_1_FedRACE_Net_Federated_Robust_Adaptive_Context_Enhanced_Fuzzy_Edge_Local_Contrast_Mapping_Fusion_Network_files/01_1_FedRACE_Net_Federated_Robust_Adaptive_Context_Enhanced_Fuzzy_Edge_Local_Contrast_Mapping_Fusion_Network_0_18.png)
    


    
    ======================================================================================================================
    STEP 7: NOVEL PREPROCESSING — RACE-FELCM
    ======================================================================================================================
    
    ======================================================================================================================
    STEP 8: MODEL — ResNet-50 + CRAF Fusion
    ======================================================================================================================
    Downloading: "https://download.pytorch.org/models/resnet50-11ad3fa6.pth" to /root/.cache/torch/hub/checkpoints/resnet50-11ad3fa6.pth


    100%|██████████| 97.8M/97.8M [00:00<00:00, 116MB/s]


    Backbone: ResNet-50 | pretrained_loaded=True
    Total params: 25,790,855
    Trainable params: 2,282,823 (8.85%)
    
    ======================================================================================================================
    STEP 9: LOSSES + PROTOTYPE KNOWLEDGE SHARING
    ======================================================================================================================
    
    ======================================================================================================================
    STEP 10: TUNE-AWARE RL-UCB PREPROCESSING SELECTION
    ======================================================================================================================
    
    ======================================================================================================================
    STEP 11: TRUE FEDERATED TRAINING
    ======================================================================================================================
    Adaptive clients => DS1=3, DS2=3, TOTAL=6
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
    Client 0 (ds1) | train_acc=0.6347 | val_acc=0.7568 | val_f1=0.7007 | val_auc=0.9813 | reward=0.7147 | theta=race_robust (g=1.08, a=0.22, b=4.50, t=2.80, k=5, eg=0.11, mix=0.76)
    Client 1 (ds1) | train_acc=0.6886 | val_acc=0.9565 | val_f1=0.6410 | val_auc=nan | reward=0.7199 | theta=race_edge_plus (g=1.02, a=0.32, b=6.00, t=2.60, k=3, eg=0.15, mix=0.84)
    Client 2 (ds1) | train_acc=0.6840 | val_acc=0.8654 | val_f1=0.6527 | val_auc=0.9748 | reward=0.7059 | theta=race_focus (g=1.10, a=0.36, b=6.40, t=2.70, k=3, eg=0.17, mix=0.86)
    Client 3 (ds2) | train_acc=0.7500 | val_acc=0.9000 | val_f1=0.8189 | val_auc=0.9681 | reward=0.8392 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    Client 4 (ds2) | train_acc=0.7698 | val_acc=0.9394 | val_f1=0.9148 | val_auc=0.9914 | reward=0.9209 | theta=race_smoothmix (g=0.90, a=0.20, b=4.00, t=2.10, k=7, eg=0.07, mix=0.70)
    Client 5 (ds2) | train_acc=0.7237 | val_acc=0.8286 | val_f1=0.5584 | val_auc=0.9419 | reward=0.6260 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    
    ----------------------------------------------------------------------------------------------------------------------
    GLOBAL VAL (Round 1) | global_acc=0.8894 | global_f1=0.7692 | ds1_acc=0.8482 | ds1_f1=0.6662 | ds2_acc=0.8983 | ds2_f1=0.7914 | reward=0.8717 | round_time=178.2s
    ----------------------------------------------------------------------------------------------------------------------
    
    ======================================================================================================================
    ROUND 2/15 | selected=[0, 1, 2, 3, 4, 5]
    ======================================================================================================================
    Client 0 (ds1) | train_acc=0.8432 | val_acc=0.8649 | val_f1=0.7670 | val_auc=0.9920 | reward=0.7915 | theta=race_focus (g=1.10, a=0.36, b=6.40, t=2.70, k=3, eg=0.17, mix=0.86)
    Client 1 (ds1) | train_acc=0.8683 | val_acc=0.9130 | val_f1=0.4583 | val_auc=nan | reward=0.5720 | theta=race_robust (g=1.08, a=0.22, b=4.50, t=2.80, k=5, eg=0.11, mix=0.76)
    Client 2 (ds1) | train_acc=0.8560 | val_acc=0.9038 | val_f1=0.6827 | val_auc=0.9761 | reward=0.7380 | theta=race_edge_plus (g=1.02, a=0.32, b=6.00, t=2.60, k=3, eg=0.15, mix=0.84)
    Client 3 (ds2) | train_acc=0.8640 | val_acc=0.8867 | val_f1=0.8397 | val_auc=0.9780 | reward=0.8514 | theta=race_smoothmix (g=0.90, a=0.20, b=4.00, t=2.10, k=7, eg=0.07, mix=0.70)
    Client 4 (ds2) | train_acc=0.8476 | val_acc=0.9437 | val_f1=0.9139 | val_auc=0.9923 | reward=0.9214 | theta=race_robust (g=1.08, a=0.22, b=4.50, t=2.80, k=5, eg=0.11, mix=0.76)
    Client 5 (ds2) | train_acc=0.8216 | val_acc=0.7857 | val_f1=0.5928 | val_auc=0.9512 | reward=0.6411 | theta=race_robust (g=1.08, a=0.22, b=4.50, t=2.80, k=5, eg=0.11, mix=0.76)
    
    ----------------------------------------------------------------------------------------------------------------------
    GLOBAL VAL (Round 2) | global_acc=0.8863 | global_f1=0.7812 | ds1_acc=0.8929 | ds1_f1=0.6645 | ds2_acc=0.8848 | ds2_f1=0.8063 | reward=0.8820 | round_time=160.4s
    ----------------------------------------------------------------------------------------------------------------------
    
    ======================================================================================================================
    ROUND 3/15 | selected=[0, 1, 2, 3, 4, 5]
    ======================================================================================================================
    Client 0 (ds1) | train_acc=0.8100 | val_acc=0.9459 | val_f1=0.9386 | val_auc=0.9851 | reward=0.9405 | theta=race_edge_plus (g=1.02, a=0.32, b=6.00, t=2.60, k=3, eg=0.15, mix=0.84)
    Client 1 (ds1) | train_acc=0.8323 | val_acc=0.9130 | val_f1=0.4583 | val_auc=nan | reward=0.5720 | theta=race_texture (g=0.92, a=0.34, b=5.80, t=2.30, k=7, eg=0.10, mix=0.80)
    Client 2 (ds1) | train_acc=0.8333 | val_acc=0.9038 | val_f1=0.6840 | val_auc=0.9928 | reward=0.7390 | theta=race_texture (g=0.92, a=0.34, b=5.80, t=2.30, k=7, eg=0.10, mix=0.80)
    Client 3 (ds2) | train_acc=0.8539 | val_acc=0.9600 | val_f1=0.9412 | val_auc=0.9871 | reward=0.9459 | theta=race_robust (g=1.08, a=0.22, b=4.50, t=2.80, k=5, eg=0.11, mix=0.76)
    Client 4 (ds2) | train_acc=0.8660 | val_acc=0.9654 | val_f1=0.9409 | val_auc=0.9987 | reward=0.9470 | theta=race_soft (g=0.95, a=0.18, b=3.80, t=2.20, k=3, eg=0.08, mix=0.72)
    Client 5 (ds2) | train_acc=0.8324 | val_acc=0.8929 | val_f1=0.8946 | val_auc=0.9820 | reward=0.8942 | theta=race_smoothmix (g=0.90, a=0.20, b=4.00, t=2.10, k=7, eg=0.07, mix=0.70)
    
    ----------------------------------------------------------------------------------------------------------------------
    GLOBAL VAL (Round 3) | global_acc=0.9400 | global_f1=0.8920 | ds1_acc=0.9196 | ds1_f1=0.7218 | ds2_acc=0.9443 | ds2_f1=0.9285 | reward=0.9676 | round_time=222.4s
    ----------------------------------------------------------------------------------------------------------------------
    
    ======================================================================================================================
    ROUND 4/15 | selected=[0, 1, 2, 3, 4, 5]
    ======================================================================================================================
    Client 0 (ds1) | train_acc=0.9077 | val_acc=0.8919 | val_f1=0.8375 | val_auc=0.9872 | reward=0.8511 | theta=race_sharp (g=1.04, a=0.30, b=5.20, t=2.50, k=5, eg=0.12, mix=0.82)
    Client 1 (ds1) | train_acc=0.9521 | val_acc=0.9565 | val_f1=0.6410 | val_auc=nan | reward=0.7199 | theta=race_focus (g=1.10, a=0.36, b=6.40, t=2.70, k=3, eg=0.17, mix=0.86)
    Client 2 (ds1) | train_acc=0.9187 | val_acc=0.9423 | val_f1=0.9543 | val_auc=0.9992 | reward=0.9513 | theta=race_sharp (g=1.04, a=0.30, b=5.20, t=2.50, k=5, eg=0.12, mix=0.82)
    Client 3 (ds2) | train_acc=0.9139 | val_acc=0.9733 | val_f1=0.9319 | val_auc=0.9974 | reward=0.9423 | theta=race_soft (g=0.95, a=0.18, b=3.80, t=2.20, k=3, eg=0.08, mix=0.72)
    Client 4 (ds2) | train_acc=0.9021 | val_acc=0.9784 | val_f1=0.9727 | val_auc=0.9983 | reward=0.9741 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    Client 5 (ds2) | train_acc=0.9172 | val_acc=0.9429 | val_f1=0.8913 | val_auc=0.9950 | reward=0.9042 | theta=race_soft (g=0.95, a=0.18, b=3.80, t=2.20, k=3, eg=0.08, mix=0.72)
    
    ----------------------------------------------------------------------------------------------------------------------
    GLOBAL VAL (Round 4) | global_acc=0.9605 | global_f1=0.9236 | ds1_acc=0.9286 | ds1_f1=0.8514 | ds2_acc=0.9674 | ds2_f1=0.9391 | reward=1.0390 | round_time=180.9s
    ----------------------------------------------------------------------------------------------------------------------
    
    ======================================================================================================================
    ROUND 5/15 | selected=[0, 1, 2, 3, 4, 5]
    ======================================================================================================================
    Client 0 (ds1) | train_acc=0.9004 | val_acc=0.8919 | val_f1=0.8773 | val_auc=0.9914 | reward=0.8809 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    Client 1 (ds1) | train_acc=0.9701 | val_acc=0.9565 | val_f1=0.6410 | val_auc=nan | reward=0.7199 | theta=race_sharp (g=1.04, a=0.30, b=5.20, t=2.50, k=5, eg=0.12, mix=0.82)
    Client 2 (ds1) | train_acc=0.9480 | val_acc=0.9615 | val_f1=0.8935 | val_auc=0.9961 | reward=0.9105 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    Client 3 (ds2) | train_acc=0.9368 | val_acc=0.9600 | val_f1=0.9402 | val_auc=0.9990 | reward=0.9452 | theta=race_robust (g=1.08, a=0.22, b=4.50, t=2.80, k=5, eg=0.11, mix=0.76)
    Client 4 (ds2) | train_acc=0.9349 | val_acc=0.9870 | val_f1=0.9843 | val_auc=0.9993 | reward=0.9850 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    Client 5 (ds2) | train_acc=0.9420 | val_acc=0.9714 | val_f1=0.9508 | val_auc=0.9988 | reward=0.9560 | theta=race_soft (g=0.95, a=0.18, b=3.80, t=2.20, k=3, eg=0.08, mix=0.72)
    
    ----------------------------------------------------------------------------------------------------------------------
    GLOBAL VAL (Round 5) | global_acc=0.9684 | global_f1=0.9403 | ds1_acc=0.9375 | ds1_f1=0.8363 | ds2_acc=0.9750 | ds2_f1=0.9626 | reward=1.0429 | round_time=178.5s
    ----------------------------------------------------------------------------------------------------------------------
    
    ======================================================================================================================
    ROUND 6/15 | selected=[0, 1, 2, 3, 4, 5]
    ======================================================================================================================
    Client 0 (ds1) | train_acc=0.9410 | val_acc=0.9459 | val_f1=0.9522 | val_auc=0.9913 | reward=0.9507 | theta=race_texture (g=0.92, a=0.34, b=5.80, t=2.30, k=7, eg=0.10, mix=0.80)
    Client 1 (ds1) | train_acc=0.9371 | val_acc=0.9565 | val_f1=0.7419 | val_auc=nan | reward=0.7956 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    Client 2 (ds1) | train_acc=0.9440 | val_acc=0.9423 | val_f1=0.7129 | val_auc=0.9987 | reward=0.7703 | theta=race_robust (g=1.08, a=0.22, b=4.50, t=2.80, k=5, eg=0.11, mix=0.76)
    Client 3 (ds2) | train_acc=0.9492 | val_acc=0.9600 | val_f1=0.9063 | val_auc=0.9981 | reward=0.9197 | theta=race_soft (g=0.95, a=0.18, b=3.80, t=2.20, k=3, eg=0.08, mix=0.72)
    Client 4 (ds2) | train_acc=0.9533 | val_acc=0.9784 | val_f1=0.9673 | val_auc=0.9990 | reward=0.9700 | theta=race_soft (g=0.95, a=0.18, b=3.80, t=2.20, k=3, eg=0.08, mix=0.72)
    Client 5 (ds2) | train_acc=0.9488 | val_acc=0.9429 | val_f1=0.9103 | val_auc=0.9956 | reward=0.9185 | theta=race_smoothmix (g=0.90, a=0.20, b=4.00, t=2.10, k=7, eg=0.07, mix=0.70)
    
    ----------------------------------------------------------------------------------------------------------------------
    GLOBAL VAL (Round 6) | global_acc=0.9605 | global_f1=0.9103 | ds1_acc=0.9464 | ds1_f1=0.7979 | ds2_acc=0.9635 | ds2_f1=0.9344 | reward=1.0136 | round_time=180.6s
    ----------------------------------------------------------------------------------------------------------------------
    
    ======================================================================================================================
    ROUND 7/15 | selected=[0, 1, 2, 3, 4, 5]
    ======================================================================================================================
    Client 0 (ds1) | train_acc=0.9594 | val_acc=0.9189 | val_f1=0.8930 | val_auc=0.9763 | reward=0.8995 | theta=race_texture (g=0.92, a=0.34, b=5.80, t=2.30, k=7, eg=0.10, mix=0.80)
    Client 1 (ds1) | train_acc=0.9731 | val_acc=0.9565 | val_f1=0.9636 | val_auc=nan | reward=0.9618 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    Client 2 (ds1) | train_acc=0.9707 | val_acc=0.9615 | val_f1=0.8935 | val_auc=0.9987 | reward=0.9105 | theta=race_sharp (g=1.04, a=0.30, b=5.20, t=2.50, k=5, eg=0.12, mix=0.82)
    Client 3 (ds2) | train_acc=0.9579 | val_acc=0.9733 | val_f1=0.9212 | val_auc=0.9988 | reward=0.9342 | theta=race_smoothmix (g=0.90, a=0.20, b=4.00, t=2.10, k=7, eg=0.07, mix=0.70)
    Client 4 (ds2) | train_acc=0.9464 | val_acc=0.9740 | val_f1=0.9635 | val_auc=0.9991 | reward=0.9661 | theta=race_robust (g=1.08, a=0.22, b=4.50, t=2.80, k=5, eg=0.11, mix=0.76)
    Client 5 (ds2) | train_acc=0.9644 | val_acc=0.9714 | val_f1=0.9851 | val_auc=0.9980 | reward=0.9817 | theta=race_robust (g=1.08, a=0.22, b=4.50, t=2.80, k=5, eg=0.11, mix=0.76)
    
    ----------------------------------------------------------------------------------------------------------------------
    GLOBAL VAL (Round 7) | global_acc=0.9684 | global_f1=0.9484 | ds1_acc=0.9464 | ds1_f1=0.9077 | ds2_acc=0.9731 | ds2_f1=0.9571 | reward=1.0769 | round_time=179.2s
    ----------------------------------------------------------------------------------------------------------------------
    
    ======================================================================================================================
    ROUND 8/15 | selected=[0, 1, 2, 3, 4, 5]
    ======================================================================================================================
    Client 0 (ds1) | train_acc=0.9465 | val_acc=0.9459 | val_f1=0.9148 | val_auc=1.0000 | reward=0.9226 | theta=race_edge_plus (g=1.02, a=0.32, b=6.00, t=2.60, k=3, eg=0.15, mix=0.84)
    Client 1 (ds1) | train_acc=0.9760 | val_acc=0.9130 | val_f1=0.4727 | val_auc=nan | reward=0.5828 | theta=race_focus (g=1.10, a=0.36, b=6.40, t=2.70, k=3, eg=0.17, mix=0.86)
    Client 2 (ds1) | train_acc=0.9600 | val_acc=0.9615 | val_f1=0.8935 | val_auc=0.9979 | reward=0.9105 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    Client 3 (ds2) | train_acc=0.9588 | val_acc=0.9533 | val_f1=0.9051 | val_auc=0.9969 | reward=0.9172 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    Client 4 (ds2) | train_acc=0.9559 | val_acc=0.9957 | val_f1=0.9922 | val_auc=0.9998 | reward=0.9931 | theta=race_smoothmix (g=0.90, a=0.20, b=4.00, t=2.10, k=7, eg=0.07, mix=0.70)
    Client 5 (ds2) | train_acc=0.9576 | val_acc=0.9786 | val_f1=0.9888 | val_auc=0.9997 | reward=0.9863 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    
    ----------------------------------------------------------------------------------------------------------------------
    GLOBAL VAL (Round 8) | global_acc=0.9731 | global_f1=0.9393 | ds1_acc=0.9464 | ds1_f1=0.8141 | ds2_acc=0.9789 | ds2_f1=0.9662 | reward=1.0354 | round_time=176.9s
    ----------------------------------------------------------------------------------------------------------------------
    
    ======================================================================================================================
    ROUND 9/15 | selected=[0, 1, 2, 3, 4, 5]
    ======================================================================================================================
    Client 0 (ds1) | train_acc=0.9594 | val_acc=0.9459 | val_f1=0.9376 | val_auc=0.9974 | reward=0.9397 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    Client 1 (ds1) | train_acc=0.9760 | val_acc=0.9565 | val_f1=0.9636 | val_auc=nan | reward=0.9618 | theta=race_edge_plus (g=1.02, a=0.32, b=6.00, t=2.60, k=3, eg=0.15, mix=0.84)
    Client 2 (ds1) | train_acc=0.9680 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | reward=1.0000 | theta=race_robust (g=1.08, a=0.22, b=4.50, t=2.80, k=5, eg=0.11, mix=0.76)
    Client 3 (ds2) | train_acc=0.9528 | val_acc=0.9667 | val_f1=0.9137 | val_auc=0.9981 | reward=0.9269 | theta=race_robust (g=1.08, a=0.22, b=4.50, t=2.80, k=5, eg=0.11, mix=0.76)
    Client 4 (ds2) | train_acc=0.9583 | val_acc=0.9913 | val_f1=0.9870 | val_auc=0.9998 | reward=0.9881 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    Client 5 (ds2) | train_acc=0.9591 | val_acc=0.9643 | val_f1=0.9812 | val_auc=0.9981 | reward=0.9769 | theta=race_soft (g=0.95, a=0.18, b=3.80, t=2.20, k=3, eg=0.08, mix=0.72)
    
    ----------------------------------------------------------------------------------------------------------------------
    GLOBAL VAL (Round 9) | global_acc=0.9763 | global_f1=0.9657 | ds1_acc=0.9732 | ds1_f1=0.9719 | ds2_acc=0.9770 | ds2_f1=0.9643 | reward=1.1150 | round_time=182.3s
    ----------------------------------------------------------------------------------------------------------------------
    
    ======================================================================================================================
    ROUND 10/15 | selected=[0, 1, 2, 3, 4, 5]
    ======================================================================================================================
    Client 0 (ds1) | train_acc=0.9594 | val_acc=0.9459 | val_f1=0.9373 | val_auc=1.0000 | reward=0.9395 | theta=race_sharp (g=1.04, a=0.30, b=5.20, t=2.50, k=5, eg=0.12, mix=0.82)
    Client 1 (ds1) | train_acc=0.9880 | val_acc=0.9130 | val_f1=0.6083 | val_auc=nan | reward=0.6845 | theta=race_sharp (g=1.04, a=0.30, b=5.20, t=2.50, k=5, eg=0.12, mix=0.82)
    Client 2 (ds1) | train_acc=0.9773 | val_acc=0.9423 | val_f1=0.7129 | val_auc=0.9987 | reward=0.7703 | theta=race_texture (g=0.92, a=0.34, b=5.80, t=2.30, k=7, eg=0.10, mix=0.80)
    Client 3 (ds2) | train_acc=0.9606 | val_acc=0.9667 | val_f1=0.9666 | val_auc=0.9994 | reward=0.9666 | theta=race_soft (g=0.95, a=0.18, b=3.80, t=2.20, k=3, eg=0.08, mix=0.72)
    Client 4 (ds2) | train_acc=0.9666 | val_acc=0.9740 | val_f1=0.9615 | val_auc=0.9984 | reward=0.9647 | theta=race_soft (g=0.95, a=0.18, b=3.80, t=2.20, k=3, eg=0.08, mix=0.72)
    Client 5 (ds2) | train_acc=0.9673 | val_acc=0.9786 | val_f1=0.9888 | val_auc=0.9995 | reward=0.9862 | theta=race_smoothmix (g=0.90, a=0.20, b=4.00, t=2.10, k=7, eg=0.07, mix=0.70)
    
    ----------------------------------------------------------------------------------------------------------------------
    GLOBAL VAL (Round 10) | global_acc=0.9668 | global_f1=0.9341 | ds1_acc=0.9375 | ds1_f1=0.7656 | ds2_acc=0.9731 | ds2_f1=0.9703 | reward=1.0111 | round_time=179.2s
    ----------------------------------------------------------------------------------------------------------------------
    
    ======================================================================================================================
    ROUND 11/15 | selected=[0, 1, 2, 3, 4, 5]
    ======================================================================================================================
    Client 0 (ds1) | train_acc=0.9649 | val_acc=0.9189 | val_f1=0.8920 | val_auc=0.9991 | reward=0.8988 | theta=race_focus (g=1.10, a=0.36, b=6.40, t=2.70, k=3, eg=0.17, mix=0.86)
    Client 1 (ds1) | train_acc=0.9671 | val_acc=0.9130 | val_f1=0.7141 | val_auc=nan | reward=0.7638 | theta=race_texture (g=0.92, a=0.34, b=5.80, t=2.30, k=7, eg=0.10, mix=0.80)
    Client 2 (ds1) | train_acc=0.9720 | val_acc=0.9808 | val_f1=0.9848 | val_auc=1.0000 | reward=0.9838 | theta=race_edge_plus (g=1.02, a=0.32, b=6.00, t=2.60, k=3, eg=0.15, mix=0.84)
    Client 3 (ds2) | train_acc=0.9712 | val_acc=0.9667 | val_f1=0.9472 | val_auc=0.9990 | reward=0.9521 | theta=race_smoothmix (g=0.90, a=0.20, b=4.00, t=2.10, k=7, eg=0.07, mix=0.70)
    Client 4 (ds2) | train_acc=0.9666 | val_acc=0.9870 | val_f1=0.9833 | val_auc=0.9998 | reward=0.9842 | theta=race_smoothmix (g=0.90, a=0.20, b=4.00, t=2.10, k=7, eg=0.07, mix=0.70)
    Client 5 (ds2) | train_acc=0.9615 | val_acc=0.9786 | val_f1=0.9888 | val_auc=0.9990 | reward=0.9862 | theta=race_robust (g=1.08, a=0.22, b=4.50, t=2.80, k=5, eg=0.11, mix=0.76)
    
    ----------------------------------------------------------------------------------------------------------------------
    GLOBAL VAL (Round 11) | global_acc=0.9731 | global_f1=0.9610 | ds1_acc=0.9464 | ds1_f1=0.8986 | ds2_acc=0.9789 | ds2_f1=0.9744 | reward=1.0796 | round_time=182.0s
    ----------------------------------------------------------------------------------------------------------------------
    
    ======================================================================================================================
    ROUND 12/15 | selected=[0, 1, 2, 3, 4, 5]
    ======================================================================================================================
    Client 0 (ds1) | train_acc=0.9576 | val_acc=0.9730 | val_f1=0.9760 | val_auc=0.9852 | reward=0.9752 | theta=race_robust (g=1.08, a=0.22, b=4.50, t=2.80, k=5, eg=0.11, mix=0.76)
    Client 1 (ds1) | train_acc=0.9910 | val_acc=0.9565 | val_f1=0.9636 | val_auc=nan | reward=0.9618 | theta=race_robust (g=1.08, a=0.22, b=4.50, t=2.80, k=5, eg=0.11, mix=0.76)
    Client 2 (ds1) | train_acc=0.9653 | val_acc=0.9808 | val_f1=0.9848 | val_auc=0.9996 | reward=0.9838 | theta=race_focus (g=1.10, a=0.36, b=6.40, t=2.70, k=3, eg=0.17, mix=0.86)
    Client 3 (ds2) | train_acc=0.9721 | val_acc=0.9667 | val_f1=0.9461 | val_auc=0.9853 | reward=0.9512 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    Client 4 (ds2) | train_acc=0.9683 | val_acc=0.9870 | val_f1=0.9849 | val_auc=0.9991 | reward=0.9854 | theta=race_robust (g=1.08, a=0.22, b=4.50, t=2.80, k=5, eg=0.11, mix=0.76)
    Client 5 (ds2) | train_acc=0.9722 | val_acc=0.9714 | val_f1=0.9368 | val_auc=0.9987 | reward=0.9454 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    
    ----------------------------------------------------------------------------------------------------------------------
    GLOBAL VAL (Round 12) | global_acc=0.9763 | global_f1=0.9638 | ds1_acc=0.9732 | ds1_f1=0.9776 | ds2_acc=0.9770 | ds2_f1=0.9608 | reward=1.1154 | round_time=180.0s
    ----------------------------------------------------------------------------------------------------------------------
    
    ======================================================================================================================
    ROUND 13/15 | selected=[0, 1, 2, 3, 4, 5]
    ======================================================================================================================
    Client 0 (ds1) | train_acc=0.9686 | val_acc=0.9459 | val_f1=0.9376 | val_auc=0.9965 | reward=0.9397 | theta=race_edge_plus (g=1.02, a=0.32, b=6.00, t=2.60, k=3, eg=0.15, mix=0.84)
    Client 1 (ds1) | train_acc=0.9940 | val_acc=0.9565 | val_f1=0.8586 | val_auc=nan | reward=0.8831 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    Client 2 (ds1) | train_acc=0.9813 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | reward=1.0000 | theta=race_sharp (g=1.04, a=0.30, b=5.20, t=2.50, k=5, eg=0.12, mix=0.82)
    Client 3 (ds2) | train_acc=0.9808 | val_acc=0.9733 | val_f1=0.9189 | val_auc=0.9993 | reward=0.9325 | theta=race_soft (g=0.95, a=0.18, b=3.80, t=2.20, k=3, eg=0.08, mix=0.72)
    Client 4 (ds2) | train_acc=0.9719 | val_acc=0.9957 | val_f1=0.9945 | val_auc=1.0000 | reward=0.9948 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    Client 5 (ds2) | train_acc=0.9766 | val_acc=0.9500 | val_f1=0.8578 | val_auc=0.9975 | reward=0.8809 | theta=race_soft (g=0.95, a=0.18, b=3.80, t=2.20, k=3, eg=0.08, mix=0.72)
    
    ----------------------------------------------------------------------------------------------------------------------
    GLOBAL VAL (Round 13) | global_acc=0.9763 | global_f1=0.9386 | ds1_acc=0.9732 | ds1_f1=0.9503 | ds2_acc=0.9770 | ds2_f1=0.9360 | reward=1.0931 | round_time=179.2s
    ----------------------------------------------------------------------------------------------------------------------
    
    ======================================================================================================================
    ROUND 14/15 | selected=[0, 1, 2, 3, 4, 5]
    ======================================================================================================================
    Client 0 (ds1) | train_acc=0.9649 | val_acc=0.9459 | val_f1=0.9373 | val_auc=1.0000 | reward=0.9395 | theta=race_texture (g=0.92, a=0.34, b=5.80, t=2.30, k=7, eg=0.10, mix=0.80)
    Client 1 (ds1) | train_acc=0.9940 | val_acc=1.0000 | val_f1=1.0000 | val_auc=nan | reward=1.0000 | theta=race_edge_plus (g=1.02, a=0.32, b=6.00, t=2.60, k=3, eg=0.15, mix=0.84)
    Client 2 (ds1) | train_acc=0.9827 | val_acc=0.9808 | val_f1=0.9848 | val_auc=1.0000 | reward=0.9838 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    Client 3 (ds2) | train_acc=0.9675 | val_acc=0.9733 | val_f1=0.9189 | val_auc=0.9985 | reward=0.9325 | theta=race_robust (g=1.08, a=0.22, b=4.50, t=2.80, k=5, eg=0.11, mix=0.76)
    Client 4 (ds2) | train_acc=0.9751 | val_acc=0.9870 | val_f1=0.9860 | val_auc=0.9975 | reward=0.9862 | theta=race_smoothmix (g=0.90, a=0.20, b=4.00, t=2.10, k=7, eg=0.07, mix=0.70)
    Client 5 (ds2) | train_acc=0.9771 | val_acc=0.9643 | val_f1=0.9813 | val_auc=0.9967 | reward=0.9771 | theta=race_smoothmix (g=0.90, a=0.20, b=4.00, t=2.10, k=7, eg=0.07, mix=0.70)
    
    ----------------------------------------------------------------------------------------------------------------------
    GLOBAL VAL (Round 14) | global_acc=0.9763 | global_f1=0.9666 | ds1_acc=0.9732 | ds1_f1=0.9723 | ds2_acc=0.9770 | ds2_f1=0.9654 | reward=1.1156 | round_time=181.7s
    ----------------------------------------------------------------------------------------------------------------------
    
    ======================================================================================================================
    ROUND 15/15 | selected=[0, 1, 2, 3, 4, 5]
    ======================================================================================================================
    Client 0 (ds1) | train_acc=0.9594 | val_acc=0.9459 | val_f1=0.9523 | val_auc=0.9911 | reward=0.9507 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    Client 1 (ds1) | train_acc=0.9850 | val_acc=0.8261 | val_f1=0.5743 | val_auc=nan | reward=0.6372 | theta=race_robust (g=1.08, a=0.22, b=4.50, t=2.80, k=5, eg=0.11, mix=0.76)
    Client 2 (ds1) | train_acc=0.9720 | val_acc=0.9808 | val_f1=0.9848 | val_auc=1.0000 | reward=0.9838 | theta=race_robust (g=1.08, a=0.22, b=4.50, t=2.80, k=5, eg=0.11, mix=0.76)
    Client 3 (ds2) | train_acc=0.9753 | val_acc=0.9800 | val_f1=0.9577 | val_auc=0.9995 | reward=0.9633 | theta=race_smoothmix (g=0.90, a=0.20, b=4.00, t=2.10, k=7, eg=0.07, mix=0.70)
    Client 4 (ds2) | train_acc=0.9663 | val_acc=0.9957 | val_f1=0.9945 | val_auc=0.9999 | reward=0.9948 | theta=race_soft (g=0.95, a=0.18, b=3.80, t=2.20, k=3, eg=0.08, mix=0.72)
    Client 5 (ds2) | train_acc=0.9868 | val_acc=0.9786 | val_f1=0.9888 | val_auc=0.9984 | reward=0.9862 | theta=race_robust (g=1.08, a=0.22, b=4.50, t=2.80, k=5, eg=0.11, mix=0.76)
    
    ----------------------------------------------------------------------------------------------------------------------
    GLOBAL VAL (Round 15) | global_acc=0.9779 | global_f1=0.9660 | ds1_acc=0.9375 | ds1_f1=0.8898 | ds2_acc=0.9866 | ds2_f1=0.9824 | reward=1.0778 | round_time=178.8s
    ----------------------------------------------------------------------------------------------------------------------
    
    ======================================================================================================================
    TRAINING COMPLETE ✅ | total_time=2721.0s | best_round=14 | best_reward=1.1156
    ======================================================================================================================
    
    ----------------------------------------------------------------------------------------------------------------------
    GLOBAL per-round metrics
    ----------------------------------------------------------------------------------------------------------------------




  <div id="df-c45cf19b-447c-4a71-9a6c-10ac20121c72" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>round</th>
      <th>round_time_s</th>
      <th>n_selected_clients</th>
      <th>active_fraction</th>
      <th>global_reward</th>
      <th>global_acc</th>
      <th>global_f1_macro</th>
      <th>global_precision_macro</th>
      <th>global_recall_macro</th>
      <th>global_log_loss</th>
      <th>global_loss_ce</th>
      <th>global_eval_time_s</th>
      <th>ds1_acc</th>
      <th>ds1_f1_macro</th>
      <th>ds1_log_loss</th>
      <th>ds2_acc</th>
      <th>ds2_f1_macro</th>
      <th>ds2_log_loss</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>178.234736</td>
      <td>6</td>
      <td>1.0</td>
      <td>0.871654</td>
      <td>0.889415</td>
      <td>0.769248</td>
      <td>0.768875</td>
      <td>0.781393</td>
      <td>0.377056</td>
      <td>0.382209</td>
      <td>2.037771</td>
      <td>0.848214</td>
      <td>0.666170</td>
      <td>0.415033</td>
      <td>0.898273</td>
      <td>0.791406</td>
      <td>0.368892</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>160.426105</td>
      <td>6</td>
      <td>1.0</td>
      <td>0.881985</td>
      <td>0.886256</td>
      <td>0.781183</td>
      <td>0.792927</td>
      <td>0.803639</td>
      <td>0.295947</td>
      <td>0.297450</td>
      <td>1.611734</td>
      <td>0.892857</td>
      <td>0.664493</td>
      <td>0.285355</td>
      <td>0.884837</td>
      <td>0.806268</td>
      <td>0.298224</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>222.433167</td>
      <td>6</td>
      <td>1.0</td>
      <td>0.967558</td>
      <td>0.939968</td>
      <td>0.891955</td>
      <td>0.915386</td>
      <td>0.877997</td>
      <td>0.211735</td>
      <td>0.217159</td>
      <td>1.626427</td>
      <td>0.919643</td>
      <td>0.721791</td>
      <td>0.309404</td>
      <td>0.944338</td>
      <td>0.928536</td>
      <td>0.190739</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>180.887834</td>
      <td>6</td>
      <td>1.0</td>
      <td>1.039026</td>
      <td>0.960506</td>
      <td>0.923557</td>
      <td>0.918162</td>
      <td>0.943587</td>
      <td>0.162861</td>
      <td>0.162764</td>
      <td>1.613990</td>
      <td>0.928571</td>
      <td>0.851408</td>
      <td>0.200091</td>
      <td>0.967370</td>
      <td>0.939067</td>
      <td>0.154858</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>178.518931</td>
      <td>6</td>
      <td>1.0</td>
      <td>1.042890</td>
      <td>0.968404</td>
      <td>0.940262</td>
      <td>0.935005</td>
      <td>0.953351</td>
      <td>0.107869</td>
      <td>0.105616</td>
      <td>1.627584</td>
      <td>0.937500</td>
      <td>0.836276</td>
      <td>0.202730</td>
      <td>0.975048</td>
      <td>0.962616</td>
      <td>0.087477</td>
    </tr>
    <tr>
      <th>5</th>
      <td>6</td>
      <td>180.605487</td>
      <td>6</td>
      <td>1.0</td>
      <td>1.013637</td>
      <td>0.960506</td>
      <td>0.910260</td>
      <td>0.916844</td>
      <td>0.916004</td>
      <td>0.132326</td>
      <td>0.127204</td>
      <td>1.676004</td>
      <td>0.946429</td>
      <td>0.797947</td>
      <td>0.200583</td>
      <td>0.963532</td>
      <td>0.934404</td>
      <td>0.117652</td>
    </tr>
    <tr>
      <th>6</th>
      <td>7</td>
      <td>179.206124</td>
      <td>6</td>
      <td>1.0</td>
      <td>1.076861</td>
      <td>0.968404</td>
      <td>0.948368</td>
      <td>0.959602</td>
      <td>0.948881</td>
      <td>0.117818</td>
      <td>0.120540</td>
      <td>1.620991</td>
      <td>0.946429</td>
      <td>0.907711</td>
      <td>0.192674</td>
      <td>0.973129</td>
      <td>0.957108</td>
      <td>0.101726</td>
    </tr>
    <tr>
      <th>7</th>
      <td>8</td>
      <td>176.912205</td>
      <td>6</td>
      <td>1.0</td>
      <td>1.035366</td>
      <td>0.973144</td>
      <td>0.939315</td>
      <td>0.955564</td>
      <td>0.935014</td>
      <td>0.092814</td>
      <td>0.097925</td>
      <td>1.644010</td>
      <td>0.946429</td>
      <td>0.814098</td>
      <td>0.141306</td>
      <td>0.978887</td>
      <td>0.966232</td>
      <td>0.082389</td>
    </tr>
    <tr>
      <th>8</th>
      <td>9</td>
      <td>182.281394</td>
      <td>6</td>
      <td>1.0</td>
      <td>1.114979</td>
      <td>0.976303</td>
      <td>0.965660</td>
      <td>0.980474</td>
      <td>0.958084</td>
      <td>0.104310</td>
      <td>0.102585</td>
      <td>1.632444</td>
      <td>0.973214</td>
      <td>0.971909</td>
      <td>0.121260</td>
      <td>0.976967</td>
      <td>0.964316</td>
      <td>0.100666</td>
    </tr>
    <tr>
      <th>9</th>
      <td>10</td>
      <td>179.173116</td>
      <td>6</td>
      <td>1.0</td>
      <td>1.011064</td>
      <td>0.966825</td>
      <td>0.934070</td>
      <td>0.935829</td>
      <td>0.934554</td>
      <td>0.113735</td>
      <td>0.114168</td>
      <td>1.624622</td>
      <td>0.937500</td>
      <td>0.765581</td>
      <td>0.173331</td>
      <td>0.973129</td>
      <td>0.970290</td>
      <td>0.100924</td>
    </tr>
    <tr>
      <th>10</th>
      <td>11</td>
      <td>182.025209</td>
      <td>6</td>
      <td>1.0</td>
      <td>1.079613</td>
      <td>0.973144</td>
      <td>0.960975</td>
      <td>0.969562</td>
      <td>0.955960</td>
      <td>0.086931</td>
      <td>0.084285</td>
      <td>1.666303</td>
      <td>0.946429</td>
      <td>0.898591</td>
      <td>0.165295</td>
      <td>0.978887</td>
      <td>0.974386</td>
      <td>0.070085</td>
    </tr>
    <tr>
      <th>11</th>
      <td>12</td>
      <td>180.043326</td>
      <td>6</td>
      <td>1.0</td>
      <td>1.115376</td>
      <td>0.976303</td>
      <td>0.963755</td>
      <td>0.981569</td>
      <td>0.951865</td>
      <td>0.114570</td>
      <td>0.110504</td>
      <td>1.630687</td>
      <td>0.973214</td>
      <td>0.977554</td>
      <td>0.168289</td>
      <td>0.976967</td>
      <td>0.960788</td>
      <td>0.103022</td>
    </tr>
    <tr>
      <th>12</th>
      <td>13</td>
      <td>179.215326</td>
      <td>6</td>
      <td>1.0</td>
      <td>1.093105</td>
      <td>0.976303</td>
      <td>0.938568</td>
      <td>0.937842</td>
      <td>0.959529</td>
      <td>0.094339</td>
      <td>0.094665</td>
      <td>1.655095</td>
      <td>0.973214</td>
      <td>0.950342</td>
      <td>0.099059</td>
      <td>0.976967</td>
      <td>0.936037</td>
      <td>0.093324</td>
    </tr>
    <tr>
      <th>13</th>
      <td>14</td>
      <td>181.704935</td>
      <td>6</td>
      <td>1.0</td>
      <td>1.115647</td>
      <td>0.976303</td>
      <td>0.966628</td>
      <td>0.979693</td>
      <td>0.960521</td>
      <td>0.083702</td>
      <td>0.084031</td>
      <td>1.623149</td>
      <td>0.973214</td>
      <td>0.972260</td>
      <td>0.064196</td>
      <td>0.976967</td>
      <td>0.965417</td>
      <td>0.087895</td>
    </tr>
    <tr>
      <th>14</th>
      <td>15</td>
      <td>178.838392</td>
      <td>6</td>
      <td>1.0</td>
      <td>1.077825</td>
      <td>0.977883</td>
      <td>0.966000</td>
      <td>0.971405</td>
      <td>0.964233</td>
      <td>0.093605</td>
      <td>0.089861</td>
      <td>1.638730</td>
      <td>0.937500</td>
      <td>0.889776</td>
      <td>0.214507</td>
      <td>0.986564</td>
      <td>0.982386</td>
      <td>0.067614</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-c45cf19b-447c-4a71-9a6c-10ac20121c72')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-c45cf19b-447c-4a71-9a6c-10ac20121c72 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-c45cf19b-447c-4a71-9a6c-10ac20121c72');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


  <div id="id_d685a71b-90e2-409e-b5ea-59a598e8be96">
    <style>
      .colab-df-generate {
        background-color: #E8F0FE;
        border: none;
        border-radius: 50%;
        cursor: pointer;
        display: none;
        fill: #1967D2;
        height: 32px;
        padding: 0 0 0 0;
        width: 32px;
      }

      .colab-df-generate:hover {
        background-color: #E2EBFA;
        box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
        fill: #174EA6;
      }

      [theme=dark] .colab-df-generate {
        background-color: #3B4455;
        fill: #D2E3FC;
      }

      [theme=dark] .colab-df-generate:hover {
        background-color: #434B5C;
        box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
        filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
        fill: #FFFFFF;
      }
    </style>
    <button class="colab-df-generate" onclick="generateWithVariable('glob_df')"
            title="Generate code using this dataframe."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M7,19H8.4L18.45,9,17,7.55,7,17.6ZM5,21V16.75L18.45,3.32a2,2,0,0,1,2.83,0l1.4,1.43a1.91,1.91,0,0,1,.58,1.4,1.91,1.91,0,0,1-.58,1.4L9.25,21ZM18.45,9,17,7.55Zm-12,3A5.31,5.31,0,0,0,4.9,8.1,5.31,5.31,0,0,0,1,6.5,5.31,5.31,0,0,0,4.9,4.9,5.31,5.31,0,0,0,6.5,1,5.31,5.31,0,0,0,8.1,4.9,5.31,5.31,0,0,0,12,6.5,5.46,5.46,0,0,0,6.5,12Z"/>
  </svg>
    </button>
    <script>
      (() => {
      const buttonEl =
        document.querySelector('#id_d685a71b-90e2-409e-b5ea-59a598e8be96 button.colab-df-generate');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      buttonEl.onclick = () => {
        google.colab.notebook.generateWithVariable('glob_df');
      }
      })();
    </script>
  </div>

    </div>
  </div>



    
    ----------------------------------------------------------------------------------------------------------------------
    LOCAL per-client per-round metrics
    ----------------------------------------------------------------------------------------------------------------------




  <div id="df-326f29db-43b4-4723-bdac-3a83cd92ef91" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>round</th>
      <th>client</th>
      <th>dataset</th>
      <th>selected</th>
      <th>theta_arm</th>
      <th>theta_name</th>
      <th>theta_str</th>
      <th>gamma_power</th>
      <th>alpha_contrast_weight</th>
      <th>beta_contrast_sharpness</th>
      <th>...</th>
      <th>val_auc_roc_macro_ovr</th>
      <th>val_auc_class_0</th>
      <th>val_auc_class_1</th>
      <th>val_auc_class_2</th>
      <th>val_auc_class_3</th>
      <th>val_fusion_gate_mean_raw</th>
      <th>val_fusion_gate_mean_enh</th>
      <th>val_fusion_gate_mean_res</th>
      <th>val_fusion_gate_entropy</th>
      <th>reward</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>client_0</td>
      <td>ds1</td>
      <td>1</td>
      <td>3</td>
      <td>race_robust</td>
      <td>(g=1.08, a=0.22, b=4.50, t=2.80, k=5, eg=0.11,...</td>
      <td>1.08</td>
      <td>0.22</td>
      <td>4.5</td>
      <td>...</td>
      <td>0.981290</td>
      <td>0.948148</td>
      <td>0.987500</td>
      <td>0.996503</td>
      <td>0.993007</td>
      <td>0.460025</td>
      <td>0.093792</td>
      <td>0.446183</td>
      <td>1.184199</td>
      <td>0.714735</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>client_1</td>
      <td>ds1</td>
      <td>1</td>
      <td>4</td>
      <td>race_edge_plus</td>
      <td>(g=1.02, a=0.32, b=6.00, t=2.60, k=3, eg=0.15,...</td>
      <td>1.02</td>
      <td>0.32</td>
      <td>6.0</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.000000</td>
      <td>0.863636</td>
      <td>1.000000</td>
      <td>0.000144</td>
      <td>0.000018</td>
      <td>0.999838</td>
      <td>0.002312</td>
      <td>0.719900</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>client_2</td>
      <td>ds1</td>
      <td>1</td>
      <td>5</td>
      <td>race_focus</td>
      <td>(g=1.10, a=0.36, b=6.40, t=2.70, k=3, eg=0.17,...</td>
      <td>1.10</td>
      <td>0.36</td>
      <td>6.4</td>
      <td>...</td>
      <td>0.974836</td>
      <td>0.946218</td>
      <td>0.953125</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.422747</td>
      <td>0.212920</td>
      <td>0.364333</td>
      <td>1.516152</td>
      <td>0.705873</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>client_3</td>
      <td>ds2</td>
      <td>1</td>
      <td>1</td>
      <td>race_balanced</td>
      <td>(g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10,...</td>
      <td>1.00</td>
      <td>0.24</td>
      <td>4.6</td>
      <td>...</td>
      <td>0.968139</td>
      <td>0.972414</td>
      <td>0.932302</td>
      <td>0.975874</td>
      <td>0.991964</td>
      <td>0.441560</td>
      <td>0.218194</td>
      <td>0.340246</td>
      <td>1.525581</td>
      <td>0.839164</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>client_4</td>
      <td>ds2</td>
      <td>1</td>
      <td>3</td>
      <td>race_smoothmix</td>
      <td>(g=0.90, a=0.20, b=4.00, t=2.10, k=7, eg=0.07,...</td>
      <td>0.90</td>
      <td>0.20</td>
      <td>4.0</td>
      <td>...</td>
      <td>0.991404</td>
      <td>0.987728</td>
      <td>0.980120</td>
      <td>0.999475</td>
      <td>0.998292</td>
      <td>0.712599</td>
      <td>0.257432</td>
      <td>0.029969</td>
      <td>0.816705</td>
      <td>0.920911</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>85</th>
      <td>15</td>
      <td>client_1</td>
      <td>ds1</td>
      <td>1</td>
      <td>3</td>
      <td>race_robust</td>
      <td>(g=1.08, a=0.22, b=4.50, t=2.80, k=5, eg=0.11,...</td>
      <td>1.08</td>
      <td>0.22</td>
      <td>4.5</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.980392</td>
      <td>1.000000</td>
      <td>0.964286</td>
      <td>0.632422</td>
      <td>0.293341</td>
      <td>0.074237</td>
      <td>1.026694</td>
      <td>0.637240</td>
    </tr>
    <tr>
      <th>86</th>
      <td>15</td>
      <td>client_2</td>
      <td>ds1</td>
      <td>1</td>
      <td>3</td>
      <td>race_robust</td>
      <td>(g=1.08, a=0.22, b=4.50, t=2.80, k=5, eg=0.11,...</td>
      <td>1.08</td>
      <td>0.22</td>
      <td>4.5</td>
      <td>...</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.967014</td>
      <td>0.012461</td>
      <td>0.020525</td>
      <td>0.194254</td>
      <td>0.983829</td>
    </tr>
    <tr>
      <th>87</th>
      <td>15</td>
      <td>client_3</td>
      <td>ds2</td>
      <td>1</td>
      <td>3</td>
      <td>race_smoothmix</td>
      <td>(g=0.90, a=0.20, b=4.00, t=2.10, k=7, eg=0.07,...</td>
      <td>0.90</td>
      <td>0.20</td>
      <td>4.0</td>
      <td>...</td>
      <td>0.999471</td>
      <td>1.000000</td>
      <td>0.999239</td>
      <td>0.998645</td>
      <td>1.000000</td>
      <td>0.957939</td>
      <td>0.023462</td>
      <td>0.018599</td>
      <td>0.257492</td>
      <td>0.963263</td>
    </tr>
    <tr>
      <th>88</th>
      <td>15</td>
      <td>client_4</td>
      <td>ds2</td>
      <td>1</td>
      <td>0</td>
      <td>race_soft</td>
      <td>(g=0.95, a=0.18, b=3.80, t=2.20, k=3, eg=0.08,...</td>
      <td>0.95</td>
      <td>0.18</td>
      <td>3.8</td>
      <td>...</td>
      <td>0.999937</td>
      <td>1.000000</td>
      <td>0.999824</td>
      <td>0.999925</td>
      <td>1.000000</td>
      <td>0.981563</td>
      <td>0.009391</td>
      <td>0.009046</td>
      <td>0.111722</td>
      <td>0.994824</td>
    </tr>
    <tr>
      <th>89</th>
      <td>15</td>
      <td>client_5</td>
      <td>ds2</td>
      <td>1</td>
      <td>2</td>
      <td>race_robust</td>
      <td>(g=1.08, a=0.22, b=4.50, t=2.80, k=5, eg=0.11,...</td>
      <td>1.08</td>
      <td>0.22</td>
      <td>4.5</td>
      <td>...</td>
      <td>0.998389</td>
      <td>0.997114</td>
      <td>0.996443</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.993837</td>
      <td>0.003599</td>
      <td>0.002565</td>
      <td>0.037165</td>
      <td>0.986239</td>
    </tr>
  </tbody>
</table>
<p>90 rows × 40 columns</p>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-326f29db-43b4-4723-bdac-3a83cd92ef91')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-326f29db-43b4-4723-bdac-3a83cd92ef91 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-326f29db-43b4-4723-bdac-3a83cd92ef91');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


  <div id="id_de38d25c-517c-4902-b021-98eb2330615a">
    <style>
      .colab-df-generate {
        background-color: #E8F0FE;
        border: none;
        border-radius: 50%;
        cursor: pointer;
        display: none;
        fill: #1967D2;
        height: 32px;
        padding: 0 0 0 0;
        width: 32px;
      }

      .colab-df-generate:hover {
        background-color: #E2EBFA;
        box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
        fill: #174EA6;
      }

      [theme=dark] .colab-df-generate {
        background-color: #3B4455;
        fill: #D2E3FC;
      }

      [theme=dark] .colab-df-generate:hover {
        background-color: #434B5C;
        box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
        filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
        fill: #FFFFFF;
      }
    </style>
    <button class="colab-df-generate" onclick="generateWithVariable('loc_df')"
            title="Generate code using this dataframe."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M7,19H8.4L18.45,9,17,7.55,7,17.6ZM5,21V16.75L18.45,3.32a2,2,0,0,1,2.83,0l1.4,1.43a1.91,1.91,0,0,1,.58,1.4,1.91,1.91,0,0,1-.58,1.4L9.25,21ZM18.45,9,17,7.55Zm-12,3A5.31,5.31,0,0,0,4.9,8.1,5.31,5.31,0,0,0,1,6.5,5.31,5.31,0,0,0,4.9,4.9,5.31,5.31,0,0,0,6.5,1,5.31,5.31,0,0,0,8.1,4.9,5.31,5.31,0,0,0,12,6.5,5.46,5.46,0,0,0,6.5,12Z"/>
  </svg>
    </button>
    <script>
      (() => {
      const buttonEl =
        document.querySelector('#id_de38d25c-517c-4902-b021-98eb2330615a button.colab-df-generate');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      buttonEl.onclick = () => {
        google.colab.notebook.generateWithVariable('loc_df');
      }
      })();
    </script>
  </div>

    </div>
  </div>



    
    ======================================================================================================================
    STEP 12: FINAL EVALUATION
    ======================================================================================================================
    
    ----------------------------------------------------------------------------------------------------------------------
    Best RL-selected preprocessing preset per client
    ----------------------------------------------------------------------------------------------------------------------




  <div id="df-a305d38d-6d14-4192-8477-276ffcb1bf1b" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>client</th>
      <th>dataset</th>
      <th>best_theta_arm</th>
      <th>best_theta_name</th>
      <th>best_theta_str</th>
      <th>estimated_value</th>
      <th>pulls</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>client_0</td>
      <td>ds1</td>
      <td>4</td>
      <td>race_edge_plus</td>
      <td>(g=1.02, a=0.32, b=6.00, t=2.60, k=3, eg=0.15,...</td>
      <td>0.934234</td>
      <td>14</td>
    </tr>
    <tr>
      <th>1</th>
      <td>client_1</td>
      <td>ds1</td>
      <td>4</td>
      <td>race_edge_plus</td>
      <td>(g=1.02, a=0.32, b=6.00, t=2.60, k=3, eg=0.15,...</td>
      <td>0.893912</td>
      <td>14</td>
    </tr>
    <tr>
      <th>2</th>
      <td>client_2</td>
      <td>ds1</td>
      <td>1</td>
      <td>race_sharp</td>
      <td>(g=1.04, a=0.30, b=5.20, t=2.50, k=5, eg=0.12,...</td>
      <td>0.953940</td>
      <td>14</td>
    </tr>
    <tr>
      <th>3</th>
      <td>client_3</td>
      <td>ds2</td>
      <td>0</td>
      <td>race_soft</td>
      <td>(g=0.95, a=0.18, b=3.80, t=2.20, k=3, eg=0.08,...</td>
      <td>0.940270</td>
      <td>14</td>
    </tr>
    <tr>
      <th>4</th>
      <td>client_4</td>
      <td>ds2</td>
      <td>1</td>
      <td>race_balanced</td>
      <td>(g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10,...</td>
      <td>0.985494</td>
      <td>14</td>
    </tr>
    <tr>
      <th>5</th>
      <td>client_5</td>
      <td>ds2</td>
      <td>3</td>
      <td>race_smoothmix</td>
      <td>(g=0.90, a=0.20, b=4.00, t=2.10, k=7, eg=0.07,...</td>
      <td>0.943979</td>
      <td>14</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-a305d38d-6d14-4192-8477-276ffcb1bf1b')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-a305d38d-6d14-4192-8477-276ffcb1bf1b button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-a305d38d-6d14-4192-8477-276ffcb1bf1b');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


  <div id="id_64e7e7aa-1d7e-48aa-9c78-515954fd4f98">
    <style>
      .colab-df-generate {
        background-color: #E8F0FE;
        border: none;
        border-radius: 50%;
        cursor: pointer;
        display: none;
        fill: #1967D2;
        height: 32px;
        padding: 0 0 0 0;
        width: 32px;
      }

      .colab-df-generate:hover {
        background-color: #E2EBFA;
        box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
        fill: #174EA6;
      }

      [theme=dark] .colab-df-generate {
        background-color: #3B4455;
        fill: #D2E3FC;
      }

      [theme=dark] .colab-df-generate:hover {
        background-color: #434B5C;
        box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
        filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
        fill: #FFFFFF;
      }
    </style>
    <button class="colab-df-generate" onclick="generateWithVariable('client_theta_df')"
            title="Generate code using this dataframe."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M7,19H8.4L18.45,9,17,7.55,7,17.6ZM5,21V16.75L18.45,3.32a2,2,0,0,1,2.83,0l1.4,1.43a1.91,1.91,0,0,1,.58,1.4,1.91,1.91,0,0,1-.58,1.4L9.25,21ZM18.45,9,17,7.55Zm-12,3A5.31,5.31,0,0,0,4.9,8.1,5.31,5.31,0,0,0,1,6.5,5.31,5.31,0,0,0,4.9,4.9,5.31,5.31,0,0,0,6.5,1,5.31,5.31,0,0,0,8.1,4.9,5.31,5.31,0,0,0,12,6.5,5.46,5.46,0,0,0,6.5,12Z"/>
  </svg>
    </button>
    <script>
      (() => {
      const buttonEl =
        document.querySelector('#id_64e7e7aa-1d7e-48aa-9c78-515954fd4f98 button.colab-df-generate');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      buttonEl.onclick = () => {
        google.colab.notebook.generateWithVariable('client_theta_df');
      }
      })();
    </script>
  </div>

    </div>
  </div>



    
    ----------------------------------------------------------------------------------------------------------------------
    Chosen final validation-based preprocessing strategy
    ----------------------------------------------------------------------------------------------------------------------




  <div id="df-f70b0c38-029a-4690-be89-2d74b597a35f" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>dataset</th>
      <th>strategy</th>
      <th>theta_names</th>
      <th>score</th>
      <th>val_acc</th>
      <th>val_f1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>ds1</td>
      <td>single</td>
      <td>['race_edge_plus']</td>
      <td>0.977603</td>
      <td>0.977876</td>
      <td>0.977512</td>
    </tr>
    <tr>
      <th>1</th>
      <td>ds2</td>
      <td>single</td>
      <td>['race_soft']</td>
      <td>0.974687</td>
      <td>0.975355</td>
      <td>0.974464</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-f70b0c38-029a-4690-be89-2d74b597a35f')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-f70b0c38-029a-4690-be89-2d74b597a35f button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-f70b0c38-029a-4690-be89-2d74b597a35f');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


  <div id="id_50843100-34a8-4566-8020-00edae1b03fd">
    <style>
      .colab-df-generate {
        background-color: #E8F0FE;
        border: none;
        border-radius: 50%;
        cursor: pointer;
        display: none;
        fill: #1967D2;
        height: 32px;
        padding: 0 0 0 0;
        width: 32px;
      }

      .colab-df-generate:hover {
        background-color: #E2EBFA;
        box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
        fill: #174EA6;
      }

      [theme=dark] .colab-df-generate {
        background-color: #3B4455;
        fill: #D2E3FC;
      }

      [theme=dark] .colab-df-generate:hover {
        background-color: #434B5C;
        box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
        filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
        fill: #FFFFFF;
      }
    </style>
    <button class="colab-df-generate" onclick="generateWithVariable('choice_df')"
            title="Generate code using this dataframe."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M7,19H8.4L18.45,9,17,7.55,7,17.6ZM5,21V16.75L18.45,3.32a2,2,0,0,1,2.83,0l1.4,1.43a1.91,1.91,0,0,1,.58,1.4,1.91,1.91,0,0,1-.58,1.4L9.25,21ZM18.45,9,17,7.55Zm-12,3A5.31,5.31,0,0,0,4.9,8.1,5.31,5.31,0,0,0,1,6.5,5.31,5.31,0,0,0,4.9,4.9,5.31,5.31,0,0,0,6.5,1,5.31,5.31,0,0,0,8.1,4.9,5.31,5.31,0,0,0,12,6.5,5.46,5.46,0,0,0,6.5,12Z"/>
  </svg>
    </button>
    <script>
      (() => {
      const buttonEl =
        document.querySelector('#id_50843100-34a8-4566-8020-00edae1b03fd button.colab-df-generate');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      buttonEl.onclick = () => {
        google.colab.notebook.generateWithVariable('choice_df');
      }
      })();
    </script>
  </div>

    </div>
  </div>



    
    ======================================================================================================================
    STEP 12.5: EXTENDED METRICS + ERROR ANALYSIS + CALIBRATION
    ======================================================================================================================
    
    ----------------------------------------------------------------------------------------------------------------------
    Extended TEST metrics (DS1 vs DS2)
    ----------------------------------------------------------------------------------------------------------------------




  <div id="df-8780418b-27c2-4a78-a856-6d9def76d87a" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>dataset</th>
      <th>acc</th>
      <th>balanced_acc</th>
      <th>precision_macro</th>
      <th>recall_macro</th>
      <th>f1_macro</th>
      <th>precision_weighted</th>
      <th>recall_weighted</th>
      <th>f1_weighted</th>
      <th>log_loss</th>
      <th>...</th>
      <th>fpr_macro</th>
      <th>fnr_macro</th>
      <th>ece</th>
      <th>mce</th>
      <th>brier_multi</th>
      <th>auc_roc_macro_ovr</th>
      <th>auc_class_0</th>
      <th>auc_class_1</th>
      <th>auc_class_2</th>
      <th>auc_class_3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>ds1_test</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.00000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.028428</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.026232</td>
      <td>0.465132</td>
      <td>0.005543</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>ds2_test</td>
      <td>0.980095</td>
      <td>0.979503</td>
      <td>0.979695</td>
      <td>0.979503</td>
      <td>0.979525</td>
      <td>0.98027</td>
      <td>0.980095</td>
      <td>0.980109</td>
      <td>0.074051</td>
      <td>...</td>
      <td>0.006581</td>
      <td>0.020497</td>
      <td>0.022557</td>
      <td>0.503564</td>
      <td>0.031057</td>
      <td>0.999399</td>
      <td>0.999111</td>
      <td>0.998983</td>
      <td>0.999956</td>
      <td>0.999545</td>
    </tr>
  </tbody>
</table>
<p>2 rows × 26 columns</p>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-8780418b-27c2-4a78-a856-6d9def76d87a')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-8780418b-27c2-4a78-a856-6d9def76d87a button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-8780418b-27c2-4a78-a856-6d9def76d87a');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


  <div id="id_dbcbba13-62b8-4902-b6d5-4902249504a0">
    <style>
      .colab-df-generate {
        background-color: #E8F0FE;
        border: none;
        border-radius: 50%;
        cursor: pointer;
        display: none;
        fill: #1967D2;
        height: 32px;
        padding: 0 0 0 0;
        width: 32px;
      }

      .colab-df-generate:hover {
        background-color: #E2EBFA;
        box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
        fill: #174EA6;
      }

      [theme=dark] .colab-df-generate {
        background-color: #3B4455;
        fill: #D2E3FC;
      }

      [theme=dark] .colab-df-generate:hover {
        background-color: #434B5C;
        box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
        filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
        fill: #FFFFFF;
      }
    </style>
    <button class="colab-df-generate" onclick="generateWithVariable('ext_df')"
            title="Generate code using this dataframe."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M7,19H8.4L18.45,9,17,7.55,7,17.6ZM5,21V16.75L18.45,3.32a2,2,0,0,1,2.83,0l1.4,1.43a1.91,1.91,0,0,1,.58,1.4,1.91,1.91,0,0,1-.58,1.4L9.25,21ZM18.45,9,17,7.55Zm-12,3A5.31,5.31,0,0,0,4.9,8.1,5.31,5.31,0,0,0,1,6.5,5.31,5.31,0,0,0,4.9,4.9,5.31,5.31,0,0,0,6.5,1,5.31,5.31,0,0,0,8.1,4.9,5.31,5.31,0,0,0,12,6.5,5.46,5.46,0,0,0,6.5,12Z"/>
  </svg>
    </button>
    <script>
      (() => {
      const buttonEl =
        document.querySelector('#id_dbcbba13-62b8-4902-b6d5-4902249504a0 button.colab-df-generate');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      buttonEl.onclick = () => {
        google.colab.notebook.generateWithVariable('ext_df');
      }
      })();
    </script>
  </div>

    </div>
  </div>



    
    ----------------------------------------------------------------------------------------------------------------------
    Classwise metrics — DS1 TEST
    ----------------------------------------------------------------------------------------------------------------------




  <div id="df-72084337-ef22-49e4-9263-0f296ff99273" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>class_id</th>
      <th>class_name</th>
      <th>support</th>
      <th>tp</th>
      <th>fp</th>
      <th>fn</th>
      <th>tn</th>
      <th>prevalence</th>
      <th>ppv</th>
      <th>npv</th>
      <th>recall</th>
      <th>specificity</th>
      <th>fpr</th>
      <th>fnr</th>
      <th>jaccard</th>
      <th>balanced_acc</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>glioma</td>
      <td>56</td>
      <td>56</td>
      <td>0</td>
      <td>0</td>
      <td>170</td>
      <td>0.247788</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>meningioma</td>
      <td>55</td>
      <td>55</td>
      <td>0</td>
      <td>0</td>
      <td>171</td>
      <td>0.243363</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>notumor</td>
      <td>59</td>
      <td>59</td>
      <td>0</td>
      <td>0</td>
      <td>167</td>
      <td>0.261062</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>pituitary</td>
      <td>56</td>
      <td>56</td>
      <td>0</td>
      <td>0</td>
      <td>170</td>
      <td>0.247788</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-72084337-ef22-49e4-9263-0f296ff99273')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-72084337-ef22-49e4-9263-0f296ff99273 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-72084337-ef22-49e4-9263-0f296ff99273');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


  <div id="id_e8587164-3ae8-4f36-aa98-6b569cc09543">
    <style>
      .colab-df-generate {
        background-color: #E8F0FE;
        border: none;
        border-radius: 50%;
        cursor: pointer;
        display: none;
        fill: #1967D2;
        height: 32px;
        padding: 0 0 0 0;
        width: 32px;
      }

      .colab-df-generate:hover {
        background-color: #E2EBFA;
        box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
        fill: #174EA6;
      }

      [theme=dark] .colab-df-generate {
        background-color: #3B4455;
        fill: #D2E3FC;
      }

      [theme=dark] .colab-df-generate:hover {
        background-color: #434B5C;
        box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
        filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
        fill: #FFFFFF;
      }
    </style>
    <button class="colab-df-generate" onclick="generateWithVariable('class_df1')"
            title="Generate code using this dataframe."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M7,19H8.4L18.45,9,17,7.55,7,17.6ZM5,21V16.75L18.45,3.32a2,2,0,0,1,2.83,0l1.4,1.43a1.91,1.91,0,0,1,.58,1.4,1.91,1.91,0,0,1-.58,1.4L9.25,21ZM18.45,9,17,7.55Zm-12,3A5.31,5.31,0,0,0,4.9,8.1,5.31,5.31,0,0,0,1,6.5,5.31,5.31,0,0,0,4.9,4.9,5.31,5.31,0,0,0,6.5,1,5.31,5.31,0,0,0,8.1,4.9,5.31,5.31,0,0,0,12,6.5,5.46,5.46,0,0,0,6.5,12Z"/>
  </svg>
    </button>
    <script>
      (() => {
      const buttonEl =
        document.querySelector('#id_e8587164-3ae8-4f36-aa98-6b569cc09543 button.colab-df-generate');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      buttonEl.onclick = () => {
        google.colab.notebook.generateWithVariable('class_df1');
      }
      })();
    </script>
  </div>

    </div>
  </div>



    
    ----------------------------------------------------------------------------------------------------------------------
    Classwise metrics — DS2 TEST
    ----------------------------------------------------------------------------------------------------------------------




  <div id="df-9c06e87f-a87e-4b29-8599-eb4f6775afba" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>class_id</th>
      <th>class_name</th>
      <th>support</th>
      <th>tp</th>
      <th>fp</th>
      <th>fn</th>
      <th>tn</th>
      <th>prevalence</th>
      <th>ppv</th>
      <th>npv</th>
      <th>recall</th>
      <th>specificity</th>
      <th>fpr</th>
      <th>fnr</th>
      <th>jaccard</th>
      <th>balanced_acc</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>glioma</td>
      <td>244</td>
      <td>237</td>
      <td>4</td>
      <td>7</td>
      <td>807</td>
      <td>0.231280</td>
      <td>0.983402</td>
      <td>0.991400</td>
      <td>0.971311</td>
      <td>0.995068</td>
      <td>0.004932</td>
      <td>0.028689</td>
      <td>0.955645</td>
      <td>0.983190</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>meningioma</td>
      <td>247</td>
      <td>239</td>
      <td>6</td>
      <td>8</td>
      <td>802</td>
      <td>0.234123</td>
      <td>0.975510</td>
      <td>0.990123</td>
      <td>0.967611</td>
      <td>0.992574</td>
      <td>0.007426</td>
      <td>0.032389</td>
      <td>0.944664</td>
      <td>0.980093</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>notumor</td>
      <td>300</td>
      <td>296</td>
      <td>1</td>
      <td>4</td>
      <td>754</td>
      <td>0.284360</td>
      <td>0.996633</td>
      <td>0.994723</td>
      <td>0.986667</td>
      <td>0.998675</td>
      <td>0.001325</td>
      <td>0.013333</td>
      <td>0.983389</td>
      <td>0.992671</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>pituitary</td>
      <td>264</td>
      <td>262</td>
      <td>10</td>
      <td>2</td>
      <td>781</td>
      <td>0.250237</td>
      <td>0.963235</td>
      <td>0.997446</td>
      <td>0.992424</td>
      <td>0.987358</td>
      <td>0.012642</td>
      <td>0.007576</td>
      <td>0.956204</td>
      <td>0.989891</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-9c06e87f-a87e-4b29-8599-eb4f6775afba')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-9c06e87f-a87e-4b29-8599-eb4f6775afba button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-9c06e87f-a87e-4b29-8599-eb4f6775afba');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


  <div id="id_400007ab-a195-4fa4-8560-60c2c819b4e0">
    <style>
      .colab-df-generate {
        background-color: #E8F0FE;
        border: none;
        border-radius: 50%;
        cursor: pointer;
        display: none;
        fill: #1967D2;
        height: 32px;
        padding: 0 0 0 0;
        width: 32px;
      }

      .colab-df-generate:hover {
        background-color: #E2EBFA;
        box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
        fill: #174EA6;
      }

      [theme=dark] .colab-df-generate {
        background-color: #3B4455;
        fill: #D2E3FC;
      }

      [theme=dark] .colab-df-generate:hover {
        background-color: #434B5C;
        box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
        filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
        fill: #FFFFFF;
      }
    </style>
    <button class="colab-df-generate" onclick="generateWithVariable('class_df2')"
            title="Generate code using this dataframe."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M7,19H8.4L18.45,9,17,7.55,7,17.6ZM5,21V16.75L18.45,3.32a2,2,0,0,1,2.83,0l1.4,1.43a1.91,1.91,0,0,1,.58,1.4,1.91,1.91,0,0,1-.58,1.4L9.25,21ZM18.45,9,17,7.55Zm-12,3A5.31,5.31,0,0,0,4.9,8.1,5.31,5.31,0,0,0,1,6.5,5.31,5.31,0,0,0,4.9,4.9,5.31,5.31,0,0,0,6.5,1,5.31,5.31,0,0,0,8.1,4.9,5.31,5.31,0,0,0,12,6.5,5.46,5.46,0,0,0,6.5,12Z"/>
  </svg>
    </button>
    <script>
      (() => {
      const buttonEl =
        document.querySelector('#id_400007ab-a195-4fa4-8560-60c2c819b4e0 button.colab-df-generate');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      buttonEl.onclick = () => {
        google.colab.notebook.generateWithVariable('class_df2');
      }
      })();
    </script>
  </div>

    </div>
  </div>



    
    ----------------------------------------------------------------------------------------------------------------------
    Top confusion pairs — DS1 TEST
    ----------------------------------------------------------------------------------------------------------------------




  <div id="df-9068af8b-4a7a-4e8b-8e92-111ddefc7592" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>true_class</th>
      <th>pred_class</th>
      <th>count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>glioma</td>
      <td>meningioma</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>glioma</td>
      <td>notumor</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>glioma</td>
      <td>pituitary</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>meningioma</td>
      <td>glioma</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>meningioma</td>
      <td>notumor</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>meningioma</td>
      <td>pituitary</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>notumor</td>
      <td>glioma</td>
      <td>0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>notumor</td>
      <td>meningioma</td>
      <td>0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>notumor</td>
      <td>pituitary</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>pituitary</td>
      <td>glioma</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-9068af8b-4a7a-4e8b-8e92-111ddefc7592')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-9068af8b-4a7a-4e8b-8e92-111ddefc7592 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-9068af8b-4a7a-4e8b-8e92-111ddefc7592');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


    </div>
  </div>



    
    ----------------------------------------------------------------------------------------------------------------------
    Top confusion pairs — DS2 TEST
    ----------------------------------------------------------------------------------------------------------------------




  <div id="df-85cc00d1-a503-4463-9a72-218ebdce641c" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>true_class</th>
      <th>pred_class</th>
      <th>count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>glioma</td>
      <td>meningioma</td>
      <td>6</td>
    </tr>
    <tr>
      <th>1</th>
      <td>meningioma</td>
      <td>pituitary</td>
      <td>5</td>
    </tr>
    <tr>
      <th>2</th>
      <td>notumor</td>
      <td>pituitary</td>
      <td>4</td>
    </tr>
    <tr>
      <th>3</th>
      <td>meningioma</td>
      <td>glioma</td>
      <td>2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>pituitary</td>
      <td>glioma</td>
      <td>2</td>
    </tr>
    <tr>
      <th>5</th>
      <td>glioma</td>
      <td>pituitary</td>
      <td>1</td>
    </tr>
    <tr>
      <th>6</th>
      <td>meningioma</td>
      <td>notumor</td>
      <td>1</td>
    </tr>
    <tr>
      <th>7</th>
      <td>glioma</td>
      <td>notumor</td>
      <td>0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>notumor</td>
      <td>meningioma</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>notumor</td>
      <td>glioma</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-85cc00d1-a503-4463-9a72-218ebdce641c')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-85cc00d1-a503-4463-9a72-218ebdce641c button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-85cc00d1-a503-4463-9a72-218ebdce641c');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


    </div>
  </div>



    
    ----------------------------------------------------------------------------------------------------------------------
    Calibration bins — DS1 TEST
    ----------------------------------------------------------------------------------------------------------------------




  <div id="df-95d6fb1c-6246-4cb9-b3fe-3099a44c6e4e" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>bin_id</th>
      <th>bin_left</th>
      <th>bin_right</th>
      <th>bin_confidence</th>
      <th>bin_accuracy</th>
      <th>bin_gap</th>
      <th>bin_count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0.000000</td>
      <td>0.083333</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>0.083333</td>
      <td>0.166667</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>0.166667</td>
      <td>0.250000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>0.250000</td>
      <td>0.333333</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>0.333333</td>
      <td>0.416667</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>5</td>
      <td>0.416667</td>
      <td>0.500000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>6</td>
      <td>0.500000</td>
      <td>0.583333</td>
      <td>0.534868</td>
      <td>1.0</td>
      <td>0.465132</td>
      <td>1</td>
    </tr>
    <tr>
      <th>7</th>
      <td>7</td>
      <td>0.583333</td>
      <td>0.666667</td>
      <td>0.597847</td>
      <td>1.0</td>
      <td>0.402153</td>
      <td>2</td>
    </tr>
    <tr>
      <th>8</th>
      <td>8</td>
      <td>0.666667</td>
      <td>0.750000</td>
      <td>0.735844</td>
      <td>1.0</td>
      <td>0.264156</td>
      <td>1</td>
    </tr>
    <tr>
      <th>9</th>
      <td>9</td>
      <td>0.750000</td>
      <td>0.833333</td>
      <td>0.827728</td>
      <td>1.0</td>
      <td>0.172272</td>
      <td>1</td>
    </tr>
    <tr>
      <th>10</th>
      <td>10</td>
      <td>0.833333</td>
      <td>0.916667</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
    </tr>
    <tr>
      <th>11</th>
      <td>11</td>
      <td>0.916667</td>
      <td>1.000000</td>
      <td>0.980894</td>
      <td>1.0</td>
      <td>0.019106</td>
      <td>221</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-95d6fb1c-6246-4cb9-b3fe-3099a44c6e4e')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-95d6fb1c-6246-4cb9-b3fe-3099a44c6e4e button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-95d6fb1c-6246-4cb9-b3fe-3099a44c6e4e');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


  <div id="id_0404baa9-373c-4c0c-93e2-f677bf2458db">
    <style>
      .colab-df-generate {
        background-color: #E8F0FE;
        border: none;
        border-radius: 50%;
        cursor: pointer;
        display: none;
        fill: #1967D2;
        height: 32px;
        padding: 0 0 0 0;
        width: 32px;
      }

      .colab-df-generate:hover {
        background-color: #E2EBFA;
        box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
        fill: #174EA6;
      }

      [theme=dark] .colab-df-generate {
        background-color: #3B4455;
        fill: #D2E3FC;
      }

      [theme=dark] .colab-df-generate:hover {
        background-color: #434B5C;
        box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
        filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
        fill: #FFFFFF;
      }
    </style>
    <button class="colab-df-generate" onclick="generateWithVariable('cal_df1')"
            title="Generate code using this dataframe."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M7,19H8.4L18.45,9,17,7.55,7,17.6ZM5,21V16.75L18.45,3.32a2,2,0,0,1,2.83,0l1.4,1.43a1.91,1.91,0,0,1,.58,1.4,1.91,1.91,0,0,1-.58,1.4L9.25,21ZM18.45,9,17,7.55Zm-12,3A5.31,5.31,0,0,0,4.9,8.1,5.31,5.31,0,0,0,1,6.5,5.31,5.31,0,0,0,4.9,4.9,5.31,5.31,0,0,0,6.5,1,5.31,5.31,0,0,0,8.1,4.9,5.31,5.31,0,0,0,12,6.5,5.46,5.46,0,0,0,6.5,12Z"/>
  </svg>
    </button>
    <script>
      (() => {
      const buttonEl =
        document.querySelector('#id_0404baa9-373c-4c0c-93e2-f677bf2458db button.colab-df-generate');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      buttonEl.onclick = () => {
        google.colab.notebook.generateWithVariable('cal_df1');
      }
      })();
    </script>
  </div>

    </div>
  </div>



    
    ----------------------------------------------------------------------------------------------------------------------
    Calibration bins — DS2 TEST
    ----------------------------------------------------------------------------------------------------------------------




  <div id="df-e7e86334-5984-462a-bc60-acc15e527efa" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>bin_id</th>
      <th>bin_left</th>
      <th>bin_right</th>
      <th>bin_confidence</th>
      <th>bin_accuracy</th>
      <th>bin_gap</th>
      <th>bin_count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0.000000</td>
      <td>0.083333</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>0.083333</td>
      <td>0.166667</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>0.166667</td>
      <td>0.250000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>0.250000</td>
      <td>0.333333</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>0.333333</td>
      <td>0.416667</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>5</td>
      <td>0.416667</td>
      <td>0.500000</td>
      <td>0.496436</td>
      <td>1.000000</td>
      <td>0.503564</td>
      <td>1</td>
    </tr>
    <tr>
      <th>6</th>
      <td>6</td>
      <td>0.500000</td>
      <td>0.583333</td>
      <td>0.545855</td>
      <td>0.571429</td>
      <td>0.025573</td>
      <td>7</td>
    </tr>
    <tr>
      <th>7</th>
      <td>7</td>
      <td>0.583333</td>
      <td>0.666667</td>
      <td>0.610660</td>
      <td>0.500000</td>
      <td>0.110660</td>
      <td>4</td>
    </tr>
    <tr>
      <th>8</th>
      <td>8</td>
      <td>0.666667</td>
      <td>0.750000</td>
      <td>0.701991</td>
      <td>0.400000</td>
      <td>0.301991</td>
      <td>5</td>
    </tr>
    <tr>
      <th>9</th>
      <td>9</td>
      <td>0.750000</td>
      <td>0.833333</td>
      <td>0.796871</td>
      <td>0.500000</td>
      <td>0.296871</td>
      <td>8</td>
    </tr>
    <tr>
      <th>10</th>
      <td>10</td>
      <td>0.833333</td>
      <td>0.916667</td>
      <td>0.888537</td>
      <td>0.807692</td>
      <td>0.080845</td>
      <td>26</td>
    </tr>
    <tr>
      <th>11</th>
      <td>11</td>
      <td>0.916667</td>
      <td>1.000000</td>
      <td>0.979397</td>
      <td>0.996016</td>
      <td>0.016619</td>
      <td>1004</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-e7e86334-5984-462a-bc60-acc15e527efa')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-e7e86334-5984-462a-bc60-acc15e527efa button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-e7e86334-5984-462a-bc60-acc15e527efa');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


  <div id="id_2605a5d9-1ff3-454b-a327-9364bbba4343">
    <style>
      .colab-df-generate {
        background-color: #E8F0FE;
        border: none;
        border-radius: 50%;
        cursor: pointer;
        display: none;
        fill: #1967D2;
        height: 32px;
        padding: 0 0 0 0;
        width: 32px;
      }

      .colab-df-generate:hover {
        background-color: #E2EBFA;
        box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
        fill: #174EA6;
      }

      [theme=dark] .colab-df-generate {
        background-color: #3B4455;
        fill: #D2E3FC;
      }

      [theme=dark] .colab-df-generate:hover {
        background-color: #434B5C;
        box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
        filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
        fill: #FFFFFF;
      }
    </style>
    <button class="colab-df-generate" onclick="generateWithVariable('cal_df2')"
            title="Generate code using this dataframe."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M7,19H8.4L18.45,9,17,7.55,7,17.6ZM5,21V16.75L18.45,3.32a2,2,0,0,1,2.83,0l1.4,1.43a1.91,1.91,0,0,1,.58,1.4,1.91,1.91,0,0,1-.58,1.4L9.25,21ZM18.45,9,17,7.55Zm-12,3A5.31,5.31,0,0,0,4.9,8.1,5.31,5.31,0,0,0,1,6.5,5.31,5.31,0,0,0,4.9,4.9,5.31,5.31,0,0,0,6.5,1,5.31,5.31,0,0,0,8.1,4.9,5.31,5.31,0,0,0,12,6.5,5.46,5.46,0,0,0,6.5,12Z"/>
  </svg>
    </button>
    <script>
      (() => {
      const buttonEl =
        document.querySelector('#id_2605a5d9-1ff3-454b-a327-9364bbba4343 button.colab-df-generate');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      buttonEl.onclick = () => {
        google.colab.notebook.generateWithVariable('cal_df2');
      }
      })();
    </script>
  </div>

    </div>
  </div>



    
    ----------------------------------------------------------------------------------------------------------------------
    VAL + TEST tables (federated, per-dataset + global equal)
    ----------------------------------------------------------------------------------------------------------------------




  <div id="df-73c23326-cf40-48b8-b8ec-40959f4c3b75" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>setting</th>
      <th>split</th>
      <th>dataset</th>
      <th>acc</th>
      <th>precision_macro</th>
      <th>recall_macro</th>
      <th>f1_macro</th>
      <th>precision_weighted</th>
      <th>recall_weighted</th>
      <th>f1_weighted</th>
      <th>...</th>
      <th>eval_time_s</th>
      <th>balanced_acc</th>
      <th>mcc</th>
      <th>kappa</th>
      <th>ppv_macro</th>
      <th>npv_macro</th>
      <th>specificity_macro</th>
      <th>ece</th>
      <th>mce</th>
      <th>brier_multi</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>ARCF-Net (validation-selected)</td>
      <td>VAL</td>
      <td>ds1</td>
      <td>0.977876</td>
      <td>0.977914</td>
      <td>0.977679</td>
      <td>0.977512</td>
      <td>0.978425</td>
      <td>0.977876</td>
      <td>0.977872</td>
      <td>...</td>
      <td>4.738907</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>ARCF-Net (validation-selected)</td>
      <td>VAL</td>
      <td>ds2</td>
      <td>0.975355</td>
      <td>0.974492</td>
      <td>0.974440</td>
      <td>0.974464</td>
      <td>0.975342</td>
      <td>0.975355</td>
      <td>0.975347</td>
      <td>...</td>
      <td>21.030175</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>ARCF-Net (validation-selected)</td>
      <td>VAL</td>
      <td>global_equal</td>
      <td>0.976616</td>
      <td>0.976203</td>
      <td>0.976059</td>
      <td>0.975988</td>
      <td>0.976883</td>
      <td>0.976616</td>
      <td>0.976609</td>
      <td>...</td>
      <td>12.884541</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>ARCF-Net (validation-selected)</td>
      <td>TEST</td>
      <td>ds1</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>...</td>
      <td>4.815784</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.026232</td>
      <td>0.465132</td>
      <td>0.005543</td>
    </tr>
    <tr>
      <th>4</th>
      <td>ARCF-Net (validation-selected)</td>
      <td>TEST</td>
      <td>ds2</td>
      <td>0.980095</td>
      <td>0.979695</td>
      <td>0.979503</td>
      <td>0.979525</td>
      <td>0.980270</td>
      <td>0.980095</td>
      <td>0.980109</td>
      <td>...</td>
      <td>21.053625</td>
      <td>0.979503</td>
      <td>0.973447</td>
      <td>0.973397</td>
      <td>0.979695</td>
      <td>0.993423</td>
      <td>0.993419</td>
      <td>0.022557</td>
      <td>0.503564</td>
      <td>0.031057</td>
    </tr>
    <tr>
      <th>5</th>
      <td>ARCF-Net (validation-selected)</td>
      <td>TEST</td>
      <td>global_equal</td>
      <td>0.990047</td>
      <td>0.989848</td>
      <td>0.989752</td>
      <td>0.989763</td>
      <td>0.990135</td>
      <td>0.990047</td>
      <td>0.990054</td>
      <td>...</td>
      <td>12.934705</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>6 rows × 23 columns</p>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-73c23326-cf40-48b8-b8ec-40959f4c3b75')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-73c23326-cf40-48b8-b8ec-40959f4c3b75 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-73c23326-cf40-48b8-b8ec-40959f4c3b75');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


  <div id="id_0c822806-d9c3-45a4-b5d9-d395a543d110">
    <style>
      .colab-df-generate {
        background-color: #E8F0FE;
        border: none;
        border-radius: 50%;
        cursor: pointer;
        display: none;
        fill: #1967D2;
        height: 32px;
        padding: 0 0 0 0;
        width: 32px;
      }

      .colab-df-generate:hover {
        background-color: #E2EBFA;
        box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
        fill: #174EA6;
      }

      [theme=dark] .colab-df-generate {
        background-color: #3B4455;
        fill: #D2E3FC;
      }

      [theme=dark] .colab-df-generate:hover {
        background-color: #434B5C;
        box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
        filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
        fill: #FFFFFF;
      }
    </style>
    <button class="colab-df-generate" onclick="generateWithVariable('paper_df')"
            title="Generate code using this dataframe."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M7,19H8.4L18.45,9,17,7.55,7,17.6ZM5,21V16.75L18.45,3.32a2,2,0,0,1,2.83,0l1.4,1.43a1.91,1.91,0,0,1,.58,1.4,1.91,1.91,0,0,1-.58,1.4L9.25,21ZM18.45,9,17,7.55Zm-12,3A5.31,5.31,0,0,0,4.9,8.1,5.31,5.31,0,0,0,1,6.5,5.31,5.31,0,0,0,4.9,4.9,5.31,5.31,0,0,0,6.5,1,5.31,5.31,0,0,0,8.1,4.9,5.31,5.31,0,0,0,12,6.5,5.46,5.46,0,0,0,6.5,12Z"/>
  </svg>
    </button>
    <script>
      (() => {
      const buttonEl =
        document.querySelector('#id_0c822806-d9c3-45a4-b5d9-d395a543d110 button.colab-df-generate');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      buttonEl.onclick = () => {
        google.colab.notebook.generateWithVariable('paper_df');
      }
      })();
    </script>
  </div>

    </div>
  </div>



    
    Paper selection summary:
    - Best round: 14 | best_reward=1.1156
    - DS1 final strategy: single | names=['race_edge_plus']
    - DS2 final strategy: single | names=['race_soft']
    
    ======================================================================================================================
    STEP 13: PREPROCESSING VALIDATION (DS1 + DS2 VAL SAMPLE)
    ======================================================================================================================
    
    ----------------------------------------------------------------------------------------------------------------------
    Preprocessing validation summary (DS1 VAL sample)
    ----------------------------------------------------------------------------------------------------------------------




  <div id="df-7348461c-618e-492d-93af-c07a3597886d" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>metric</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>max</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>edge_energy_before</td>
      <td>0.045663</td>
      <td>0.032394</td>
      <td>0.017934</td>
      <td>0.259006</td>
    </tr>
    <tr>
      <th>1</th>
      <td>edge_energy_after</td>
      <td>0.163169</td>
      <td>0.043031</td>
      <td>0.093722</td>
      <td>0.371093</td>
    </tr>
    <tr>
      <th>2</th>
      <td>entropy_before</td>
      <td>5.858106</td>
      <td>0.690578</td>
      <td>3.533567</td>
      <td>7.745560</td>
    </tr>
    <tr>
      <th>3</th>
      <td>entropy_after</td>
      <td>6.576083</td>
      <td>0.663432</td>
      <td>3.856666</td>
      <td>7.824759</td>
    </tr>
    <tr>
      <th>4</th>
      <td>contrast_before</td>
      <td>0.192887</td>
      <td>0.054095</td>
      <td>0.098088</td>
      <td>0.355714</td>
    </tr>
    <tr>
      <th>5</th>
      <td>contrast_after</td>
      <td>0.239274</td>
      <td>0.021384</td>
      <td>0.198772</td>
      <td>0.324934</td>
    </tr>
    <tr>
      <th>6</th>
      <td>edge_gain_ratio</td>
      <td>4.118444</td>
      <td>1.118872</td>
      <td>1.432758</td>
      <td>7.125557</td>
    </tr>
    <tr>
      <th>7</th>
      <td>entropy_delta</td>
      <td>0.717977</td>
      <td>0.267424</td>
      <td>0.079199</td>
      <td>1.509510</td>
    </tr>
    <tr>
      <th>8</th>
      <td>contrast_delta</td>
      <td>0.046387</td>
      <td>0.035665</td>
      <td>-0.030780</td>
      <td>0.122088</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-7348461c-618e-492d-93af-c07a3597886d')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-7348461c-618e-492d-93af-c07a3597886d button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-7348461c-618e-492d-93af-c07a3597886d');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


  <div id="id_198f2e53-9baf-458b-9962-d137a7faa829">
    <style>
      .colab-df-generate {
        background-color: #E8F0FE;
        border: none;
        border-radius: 50%;
        cursor: pointer;
        display: none;
        fill: #1967D2;
        height: 32px;
        padding: 0 0 0 0;
        width: 32px;
      }

      .colab-df-generate:hover {
        background-color: #E2EBFA;
        box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
        fill: #174EA6;
      }

      [theme=dark] .colab-df-generate {
        background-color: #3B4455;
        fill: #D2E3FC;
      }

      [theme=dark] .colab-df-generate:hover {
        background-color: #434B5C;
        box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
        filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
        fill: #FFFFFF;
      }
    </style>
    <button class="colab-df-generate" onclick="generateWithVariable('preproc_summary_df1')"
            title="Generate code using this dataframe."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M7,19H8.4L18.45,9,17,7.55,7,17.6ZM5,21V16.75L18.45,3.32a2,2,0,0,1,2.83,0l1.4,1.43a1.91,1.91,0,0,1,.58,1.4,1.91,1.91,0,0,1-.58,1.4L9.25,21ZM18.45,9,17,7.55Zm-12,3A5.31,5.31,0,0,0,4.9,8.1,5.31,5.31,0,0,0,1,6.5,5.31,5.31,0,0,0,4.9,4.9,5.31,5.31,0,0,0,6.5,1,5.31,5.31,0,0,0,8.1,4.9,5.31,5.31,0,0,0,12,6.5,5.46,5.46,0,0,0,6.5,12Z"/>
  </svg>
    </button>
    <script>
      (() => {
      const buttonEl =
        document.querySelector('#id_198f2e53-9baf-458b-9962-d137a7faa829 button.colab-df-generate');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      buttonEl.onclick = () => {
        google.colab.notebook.generateWithVariable('preproc_summary_df1');
      }
      })();
    </script>
  </div>

    </div>
  </div>



    
    ----------------------------------------------------------------------------------------------------------------------
    Preprocessing validation summary (DS2 VAL sample)
    ----------------------------------------------------------------------------------------------------------------------




  <div id="df-659b8da7-f6bc-4ae5-a527-cbaff00ad8dd" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>metric</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>max</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>edge_energy_before</td>
      <td>0.070510</td>
      <td>0.034451</td>
      <td>0.018920</td>
      <td>0.299886</td>
    </tr>
    <tr>
      <th>1</th>
      <td>edge_energy_after</td>
      <td>0.146118</td>
      <td>0.034845</td>
      <td>0.067523</td>
      <td>0.379317</td>
    </tr>
    <tr>
      <th>2</th>
      <td>entropy_before</td>
      <td>6.869180</td>
      <td>0.514067</td>
      <td>2.822053</td>
      <td>7.770264</td>
    </tr>
    <tr>
      <th>3</th>
      <td>entropy_after</td>
      <td>7.336706</td>
      <td>0.370034</td>
      <td>3.419416</td>
      <td>7.855065</td>
    </tr>
    <tr>
      <th>4</th>
      <td>contrast_before</td>
      <td>0.237638</td>
      <td>0.034813</td>
      <td>0.139120</td>
      <td>0.351084</td>
    </tr>
    <tr>
      <th>5</th>
      <td>contrast_after</td>
      <td>0.268464</td>
      <td>0.019321</td>
      <td>0.217977</td>
      <td>0.338963</td>
    </tr>
    <tr>
      <th>6</th>
      <td>edge_gain_ratio</td>
      <td>2.251028</td>
      <td>0.506660</td>
      <td>1.249483</td>
      <td>4.519500</td>
    </tr>
    <tr>
      <th>7</th>
      <td>entropy_delta</td>
      <td>0.467527</td>
      <td>0.205978</td>
      <td>0.036529</td>
      <td>1.183419</td>
    </tr>
    <tr>
      <th>8</th>
      <td>contrast_delta</td>
      <td>0.030827</td>
      <td>0.018911</td>
      <td>-0.015210</td>
      <td>0.095027</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-659b8da7-f6bc-4ae5-a527-cbaff00ad8dd')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-659b8da7-f6bc-4ae5-a527-cbaff00ad8dd button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-659b8da7-f6bc-4ae5-a527-cbaff00ad8dd');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


  <div id="id_fe2d8c31-ff3d-4195-8574-860ab4fcc0ca">
    <style>
      .colab-df-generate {
        background-color: #E8F0FE;
        border: none;
        border-radius: 50%;
        cursor: pointer;
        display: none;
        fill: #1967D2;
        height: 32px;
        padding: 0 0 0 0;
        width: 32px;
      }

      .colab-df-generate:hover {
        background-color: #E2EBFA;
        box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
        fill: #174EA6;
      }

      [theme=dark] .colab-df-generate {
        background-color: #3B4455;
        fill: #D2E3FC;
      }

      [theme=dark] .colab-df-generate:hover {
        background-color: #434B5C;
        box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
        filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
        fill: #FFFFFF;
      }
    </style>
    <button class="colab-df-generate" onclick="generateWithVariable('preproc_summary_df2')"
            title="Generate code using this dataframe."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M7,19H8.4L18.45,9,17,7.55,7,17.6ZM5,21V16.75L18.45,3.32a2,2,0,0,1,2.83,0l1.4,1.43a1.91,1.91,0,0,1,.58,1.4,1.91,1.91,0,0,1-.58,1.4L9.25,21ZM18.45,9,17,7.55Zm-12,3A5.31,5.31,0,0,0,4.9,8.1,5.31,5.31,0,0,0,1,6.5,5.31,5.31,0,0,0,4.9,4.9,5.31,5.31,0,0,0,6.5,1,5.31,5.31,0,0,0,8.1,4.9,5.31,5.31,0,0,0,12,6.5,5.46,5.46,0,0,0,6.5,12Z"/>
  </svg>
    </button>
    <script>
      (() => {
      const buttonEl =
        document.querySelector('#id_fe2d8c31-ff3d-4195-8574-860ab4fcc0ca button.colab-df-generate');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      buttonEl.onclick = () => {
        google.colab.notebook.generateWithVariable('preproc_summary_df2');
      }
      })();
    </script>
  </div>

    </div>
  </div>



    
    ======================================================================================================================
    STEP 14: BEFORE vs AFTER PREPROCESSING IMAGES — BOTH DATASETS
    ======================================================================================================================



    
![png](01_1_FedRACE_Net_Federated_Robust_Adaptive_Context_Enhanced_Fuzzy_Edge_Local_Contrast_Mapping_Fusion_Network_files/01_1_FedRACE_Net_Federated_Robust_Adaptive_Context_Enhanced_Fuzzy_Edge_Local_Contrast_Mapping_Fusion_Network_0_50.png)
    



    
![png](01_1_FedRACE_Net_Federated_Robust_Adaptive_Context_Enhanced_Fuzzy_Edge_Local_Contrast_Mapping_Fusion_Network_files/01_1_FedRACE_Net_Federated_Robust_Adaptive_Context_Enhanced_Fuzzy_Edge_Local_Contrast_Mapping_Fusion_Network_0_51.png)
    


    
    ======================================================================================================================
    STEP 15: FINAL REPORT PLOTS — BOTH DATASETS
    ======================================================================================================================



    
![png](01_1_FedRACE_Net_Federated_Robust_Adaptive_Context_Enhanced_Fuzzy_Edge_Local_Contrast_Mapping_Fusion_Network_files/01_1_FedRACE_Net_Federated_Robust_Adaptive_Context_Enhanced_Fuzzy_Edge_Local_Contrast_Mapping_Fusion_Network_0_53.png)
    



    
![png](01_1_FedRACE_Net_Federated_Robust_Adaptive_Context_Enhanced_Fuzzy_Edge_Local_Contrast_Mapping_Fusion_Network_files/01_1_FedRACE_Net_Federated_Robust_Adaptive_Context_Enhanced_Fuzzy_Edge_Local_Contrast_Mapping_Fusion_Network_0_54.png)
    



    
![png](01_1_FedRACE_Net_Federated_Robust_Adaptive_Context_Enhanced_Fuzzy_Edge_Local_Contrast_Mapping_Fusion_Network_files/01_1_FedRACE_Net_Federated_Robust_Adaptive_Context_Enhanced_Fuzzy_Edge_Local_Contrast_Mapping_Fusion_Network_0_55.png)
    



    
![png](01_1_FedRACE_Net_Federated_Robust_Adaptive_Context_Enhanced_Fuzzy_Edge_Local_Contrast_Mapping_Fusion_Network_files/01_1_FedRACE_Net_Federated_Robust_Adaptive_Context_Enhanced_Fuzzy_Edge_Local_Contrast_Mapping_Fusion_Network_0_56.png)
    



    
![png](01_1_FedRACE_Net_Federated_Robust_Adaptive_Context_Enhanced_Fuzzy_Edge_Local_Contrast_Mapping_Fusion_Network_files/01_1_FedRACE_Net_Federated_Robust_Adaptive_Context_Enhanced_Fuzzy_Edge_Local_Contrast_Mapping_Fusion_Network_0_57.png)
    



    
![png](01_1_FedRACE_Net_Federated_Robust_Adaptive_Context_Enhanced_Fuzzy_Edge_Local_Contrast_Mapping_Fusion_Network_files/01_1_FedRACE_Net_Federated_Robust_Adaptive_Context_Enhanced_Fuzzy_Edge_Local_Contrast_Mapping_Fusion_Network_0_58.png)
    



    
![png](01_1_FedRACE_Net_Federated_Robust_Adaptive_Context_Enhanced_Fuzzy_Edge_Local_Contrast_Mapping_Fusion_Network_files/01_1_FedRACE_Net_Federated_Robust_Adaptive_Context_Enhanced_Fuzzy_Edge_Local_Contrast_Mapping_Fusion_Network_0_59.png)
    



    
![png](01_1_FedRACE_Net_Federated_Robust_Adaptive_Context_Enhanced_Fuzzy_Edge_Local_Contrast_Mapping_Fusion_Network_files/01_1_FedRACE_Net_Federated_Robust_Adaptive_Context_Enhanced_Fuzzy_Edge_Local_Contrast_Mapping_Fusion_Network_0_60.png)
    



    
![png](01_1_FedRACE_Net_Federated_Robust_Adaptive_Context_Enhanced_Fuzzy_Edge_Local_Contrast_Mapping_Fusion_Network_files/01_1_FedRACE_Net_Federated_Robust_Adaptive_Context_Enhanced_Fuzzy_Edge_Local_Contrast_Mapping_Fusion_Network_0_61.png)
    



    
![png](01_1_FedRACE_Net_Federated_Robust_Adaptive_Context_Enhanced_Fuzzy_Edge_Local_Contrast_Mapping_Fusion_Network_files/01_1_FedRACE_Net_Federated_Robust_Adaptive_Context_Enhanced_Fuzzy_Edge_Local_Contrast_Mapping_Fusion_Network_0_62.png)
    



    
![png](01_1_FedRACE_Net_Federated_Robust_Adaptive_Context_Enhanced_Fuzzy_Edge_Local_Contrast_Mapping_Fusion_Network_files/01_1_FedRACE_Net_Federated_Robust_Adaptive_Context_Enhanced_Fuzzy_Edge_Local_Contrast_Mapping_Fusion_Network_0_63.png)
    



    
![png](01_1_FedRACE_Net_Federated_Robust_Adaptive_Context_Enhanced_Fuzzy_Edge_Local_Contrast_Mapping_Fusion_Network_files/01_1_FedRACE_Net_Federated_Robust_Adaptive_Context_Enhanced_Fuzzy_Edge_Local_Contrast_Mapping_Fusion_Network_0_64.png)
    



    
![png](01_1_FedRACE_Net_Federated_Robust_Adaptive_Context_Enhanced_Fuzzy_Edge_Local_Contrast_Mapping_Fusion_Network_files/01_1_FedRACE_Net_Federated_Robust_Adaptive_Context_Enhanced_Fuzzy_Edge_Local_Contrast_Mapping_Fusion_Network_0_65.png)
    


    
    ======================================================================================================================
    STEP 16: RADAR + EVOLUTION PLOTS
    ======================================================================================================================



    
![png](01_1_FedRACE_Net_Federated_Robust_Adaptive_Context_Enhanced_Fuzzy_Edge_Local_Contrast_Mapping_Fusion_Network_files/01_1_FedRACE_Net_Federated_Robust_Adaptive_Context_Enhanced_Fuzzy_Edge_Local_Contrast_Mapping_Fusion_Network_0_67.png)
    



    
![png](01_1_FedRACE_Net_Federated_Robust_Adaptive_Context_Enhanced_Fuzzy_Edge_Local_Contrast_Mapping_Fusion_Network_files/01_1_FedRACE_Net_Federated_Robust_Adaptive_Context_Enhanced_Fuzzy_Edge_Local_Contrast_Mapping_Fusion_Network_0_68.png)
    



    
![png](01_1_FedRACE_Net_Federated_Robust_Adaptive_Context_Enhanced_Fuzzy_Edge_Local_Contrast_Mapping_Fusion_Network_files/01_1_FedRACE_Net_Federated_Robust_Adaptive_Context_Enhanced_Fuzzy_Edge_Local_Contrast_Mapping_Fusion_Network_0_69.png)
    



    
![png](01_1_FedRACE_Net_Federated_Robust_Adaptive_Context_Enhanced_Fuzzy_Edge_Local_Contrast_Mapping_Fusion_Network_files/01_1_FedRACE_Net_Federated_Robust_Adaptive_Context_Enhanced_Fuzzy_Edge_Local_Contrast_Mapping_Fusion_Network_0_70.png)
    



    
![png](01_1_FedRACE_Net_Federated_Robust_Adaptive_Context_Enhanced_Fuzzy_Edge_Local_Contrast_Mapping_Fusion_Network_files/01_1_FedRACE_Net_Federated_Robust_Adaptive_Context_Enhanced_Fuzzy_Edge_Local_Contrast_Mapping_Fusion_Network_0_71.png)
    



    
![png](01_1_FedRACE_Net_Federated_Robust_Adaptive_Context_Enhanced_Fuzzy_Edge_Local_Contrast_Mapping_Fusion_Network_files/01_1_FedRACE_Net_Federated_Robust_Adaptive_Context_Enhanced_Fuzzy_Edge_Local_Contrast_Mapping_Fusion_Network_0_72.png)
    


    
    ----------------------------------------------------------------------------------------------------------------------
    Mean selected preprocessing parameters over rounds
    ----------------------------------------------------------------------------------------------------------------------




  <div id="df-5b087d12-fa39-4ad8-8ecc-ec3b14ea7c38" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>round</th>
      <th>gamma_power</th>
      <th>alpha_contrast_weight</th>
      <th>beta_contrast_sharpness</th>
      <th>tau_clip</th>
      <th>k_blur_kernel_size</th>
      <th>edge_gain</th>
      <th>blend_mix</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>1.016667</td>
      <td>0.263333</td>
      <td>5.016667</td>
      <td>2.500000</td>
      <td>4.666667</td>
      <td>0.116667</td>
      <td>0.786667</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1.043333</td>
      <td>0.256667</td>
      <td>4.983333</td>
      <td>2.633333</td>
      <td>4.666667</td>
      <td>0.120000</td>
      <td>0.780000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>0.965000</td>
      <td>0.266667</td>
      <td>4.983333</td>
      <td>2.383333</td>
      <td>5.333333</td>
      <td>0.101667</td>
      <td>0.770000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>1.013333</td>
      <td>0.260000</td>
      <td>4.833333</td>
      <td>2.416667</td>
      <td>4.000000</td>
      <td>0.111667</td>
      <td>0.786667</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>1.011667</td>
      <td>0.236667</td>
      <td>4.550000</td>
      <td>2.450000</td>
      <td>4.666667</td>
      <td>0.101667</td>
      <td>0.773333</td>
    </tr>
    <tr>
      <th>5</th>
      <td>6</td>
      <td>0.966667</td>
      <td>0.226667</td>
      <td>4.416667</td>
      <td>2.333333</td>
      <td>5.000000</td>
      <td>0.090000</td>
      <td>0.746667</td>
    </tr>
    <tr>
      <th>6</th>
      <td>7</td>
      <td>1.003333</td>
      <td>0.253333</td>
      <td>4.766667</td>
      <td>2.483333</td>
      <td>5.666667</td>
      <td>0.101667</td>
      <td>0.770000</td>
    </tr>
    <tr>
      <th>7</th>
      <td>8</td>
      <td>1.003333</td>
      <td>0.266667</td>
      <td>5.033333</td>
      <td>2.433333</td>
      <td>4.666667</td>
      <td>0.115000</td>
      <td>0.790000</td>
    </tr>
    <tr>
      <th>8</th>
      <td>9</td>
      <td>1.021667</td>
      <td>0.236667</td>
      <td>4.666667</td>
      <td>2.533333</td>
      <td>4.333333</td>
      <td>0.108333</td>
      <td>0.773333</td>
    </tr>
    <tr>
      <th>9</th>
      <td>10</td>
      <td>0.966667</td>
      <td>0.250000</td>
      <td>4.633333</td>
      <td>2.300000</td>
      <td>5.000000</td>
      <td>0.095000</td>
      <td>0.763333</td>
    </tr>
    <tr>
      <th>10</th>
      <td>11</td>
      <td>0.986667</td>
      <td>0.273333</td>
      <td>5.116667</td>
      <td>2.433333</td>
      <td>5.333333</td>
      <td>0.111667</td>
      <td>0.776667</td>
    </tr>
    <tr>
      <th>11</th>
      <td>12</td>
      <td>1.056667</td>
      <td>0.250000</td>
      <td>4.850000</td>
      <td>2.650000</td>
      <td>4.666667</td>
      <td>0.116667</td>
      <td>0.783333</td>
    </tr>
    <tr>
      <th>12</th>
      <td>13</td>
      <td>0.993333</td>
      <td>0.243333</td>
      <td>4.666667</td>
      <td>2.383333</td>
      <td>4.000000</td>
      <td>0.105000</td>
      <td>0.776667</td>
    </tr>
    <tr>
      <th>13</th>
      <td>14</td>
      <td>0.970000</td>
      <td>0.253333</td>
      <td>4.816667</td>
      <td>2.383333</td>
      <td>5.666667</td>
      <td>0.100000</td>
      <td>0.763333</td>
    </tr>
    <tr>
      <th>14</th>
      <td>15</td>
      <td>1.015000</td>
      <td>0.213333</td>
      <td>4.316667</td>
      <td>2.516667</td>
      <td>5.000000</td>
      <td>0.096667</td>
      <td>0.746667</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-5b087d12-fa39-4ad8-8ecc-ec3b14ea7c38')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-5b087d12-fa39-4ad8-8ecc-ec3b14ea7c38 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-5b087d12-fa39-4ad8-8ecc-ec3b14ea7c38');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


    </div>
  </div>




    
![png](01_1_FedRACE_Net_Federated_Robust_Adaptive_Context_Enhanced_Fuzzy_Edge_Local_Contrast_Mapping_Fusion_Network_files/01_1_FedRACE_Net_Federated_Robust_Adaptive_Context_Enhanced_Fuzzy_Edge_Local_Contrast_Mapping_Fusion_Network_0_75.png)
    


    
    ======================================================================================================================
    STEP 17: SAVING ONLY TWO FILES (CHECKPOINT + ONE CSV)
    ======================================================================================================================
    ✅ Saved checkpoint: /kaggle/working/outputs/ARCFNet_RESNET50_KAGGLE_FULLINFO_checkpoint.pth
    ✅ Saved CSV (ALL outputs): /kaggle/working/outputs/ALL_OUTPUTS_AND_METRICS_RESNET50_KAGGLE.csv
    
    DONE ✅
    Method: ARCF-Net = Adaptive RACE-FELCM with CRAF Fusion Network
    Backbone: Residual Network-50
    Best round: 14
    Adaptive clients => DS1=3, DS2=3, TOTAL=6
    Rounds completed: 15
    Global TEST acc: 0.9900
    Global TEST f1_macro: 0.9898
    DS1 TEST acc: 1.0000
    DS2 TEST acc: 0.9801
    DS1 final strategy: single | names=['race_edge_plus']
    DS2 final strategy: single | names=['race_soft']


# **Dataset Info**


```python
# ============================================================
# SECTION 3 VERIFIED DATASET AUDIT
# Based on:
# 1) Main notebook code/config
# 2) Main saved output values
# 3) New dataset-only audit from Kaggle files
#
# This code:
# - DOES download/read Kaggle datasets
# - DOES NOT train the model
# - DOES NOT run federated learning
# - DOES NOT run RACE-FELCM training/evaluation
# - Prints everything in the output cell
# - Separates source_type clearly:
#   main_code_config
#   main_saved_output
#   derived_from_main_saved_output
#   new_dataset_audit_from_kaggle_files
# ============================================================

import os
import sys
import json
import math
import random
import hashlib
import subprocess
import numpy as np
import pandas as pd
from PIL import Image

# -----------------------------
# Install required lightweight packages if missing
# -----------------------------
try:
    import kagglehub
except Exception:
    subprocess.check_call([sys.executable, "-m", "pip", "-q", "install", "kagglehub"])
    import kagglehub

try:
    from sklearn.model_selection import train_test_split
except Exception:
    subprocess.check_call([sys.executable, "-m", "pip", "-q", "install", "scikit-learn"])
    from sklearn.model_selection import train_test_split

pd.set_option("display.max_rows", 5000)
pd.set_option("display.max_columns", 500)
pd.set_option("display.width", 320)
pd.set_option("display.max_colwidth", 260)

# ============================================================
# Controls
# ============================================================

STRICT_MATCH_MAIN_OUTPUT = True
PRINT_PER_IMAGE_SIZE_TABLE = False
PRINT_ALL_HASH_ROWS = False
MAX_EXAMPLES_TO_PRINT = 80

# ============================================================
# Main-code constants verified from uploaded notebook
# ============================================================

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

LABELS = ["glioma", "meningioma", "notumor", "pituitary"]
label2id = {l: i for i, l in enumerate(LABELS)}
id2label = {i: l for l, i in label2id.items()}
NUM_CLASSES = len(LABELS)

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")

CFG = {
    "rounds": 15,

    # shared RL for client-count planning
    "client_count_candidates": [3, 4, 5],
    "client_count_search_episodes": 10,

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
    # Main saved run was KAGGLE + cuda, so model input size in that run was 224.
    "img_size_saved_run": 224,
    "batch_size_saved_run": 16,

    # global split
    "global_val_frac": 0.15,
    "test_frac": 0.15,

    # client split
    "client_val_frac": 0.12,
    "client_tune_frac": 0.12,
    "min_per_class_per_client": 5,

    # non-IID
    "dirichlet_alpha": 0.35,

    # preprocessing / augmentation
    "use_preprocessing": True,
    "use_augmentation": True,

    # transfer learning
    "freeze_backbone_rounds": 2,
    "unfreeze_last_blocks": 2,

    # bandit
    "ucb_c": 1.35,
    "theta_probe_topk": 3,

    # reward
    "reward_f1_weight": 0.75,
    "reward_acc_weight": 0.25,

    # best-round selection
    "best_round_mass_ds1": 0.50,
    "best_round_mass_ds2": 0.50,
    "best_round_min_bonus": 0.15,

    # FedAvg tempering
    "fedavg_temper": 0.50,

    # audit/plot config
    "quick_hash_subset_per_split": 300,
    "preproc_val_sample_n": 400,
    "before_after_n": 12,
    "calibration_bins": 12,
}

DS1_SLUG = "orvile/pmram-bangladeshi-brain-cancer-mri-dataset"
DS2_SLUG = "yassinebazgour/preprocessed-brain-mri-scans-for-tumors-detection"

REQ1 = {"512Glioma", "512Meningioma", "512Normal", "512Pituitary"}
REQ2 = {"glioma", "meningioma", "notumor", "pituitary"}

DS1_CLASS_DIRS = ["512Glioma", "512Meningioma", "512Normal", "512Pituitary"]
DS2_CLASS_DIRS = ["glioma", "meningioma", "notumor", "pituitary"]

# ============================================================
# Exact values from main saved output
# ============================================================

SAVED_FOLDER_COUNTS = pd.DataFrame([
    {
        "dataset": "DS1",
        "source_folder": "512Glioma",
        "mapped_label": "glioma",
        "n_images": 373,
        "source_type": "main_saved_output",
    },
    {
        "dataset": "DS1",
        "source_folder": "512Meningioma",
        "mapped_label": "meningioma",
        "n_images": 363,
        "source_type": "main_saved_output",
    },
    {
        "dataset": "DS1",
        "source_folder": "512Normal",
        "mapped_label": "notumor",
        "n_images": 396,
        "source_type": "main_saved_output",
    },
    {
        "dataset": "DS1",
        "source_folder": "512Pituitary",
        "mapped_label": "pituitary",
        "n_images": 373,
        "source_type": "main_saved_output",
    },
    {
        "dataset": "DS2",
        "source_folder": "glioma",
        "mapped_label": "glioma",
        "n_images": 1621,
        "source_type": "main_saved_output",
    },
    {
        "dataset": "DS2",
        "source_folder": "meningioma",
        "mapped_label": "meningioma",
        "n_images": 1646,
        "source_type": "main_saved_output",
    },
    {
        "dataset": "DS2",
        "source_folder": "notumor",
        "mapped_label": "notumor",
        "n_images": 2000,
        "source_type": "main_saved_output",
    },
    {
        "dataset": "DS2",
        "source_folder": "pituitary",
        "mapped_label": "pituitary",
        "n_images": 1764,
        "source_type": "main_saved_output",
    },
])

SAVED_DATASET_TOTALS = pd.DataFrame([
    {
        "dataset": "DS1",
        "total_images": 1505,
        "glioma": 373,
        "meningioma": 363,
        "notumor": 396,
        "pituitary": 373,
        "source_type": "main_saved_output",
    },
    {
        "dataset": "DS2",
        "total_images": 7031,
        "glioma": 1621,
        "meningioma": 1646,
        "notumor": 2000,
        "pituitary": 1764,
        "source_type": "main_saved_output",
    },
])

SAVED_SPLIT_SUMMARY = pd.DataFrame([
    {
        "dataset": "DS1",
        "train": 1053,
        "val": 226,
        "test": 226,
        "total": 1505,
        "source_type": "main_saved_output",
    },
    {
        "dataset": "DS2",
        "train": 4921,
        "val": 1055,
        "test": 1055,
        "total": 7031,
        "source_type": "main_saved_output",
    },
])

SAVED_LEAKAGE_SUMMARY = pd.DataFrame([
    {
        "dataset": "DS1",
        "path_overlap_train_val": 0,
        "path_overlap_train_test": 0,
        "path_overlap_val_test": 0,
        "unique_paths_train": 1053,
        "unique_paths_val": 226,
        "unique_paths_test": 226,
        "filename_overlap_train_val": 0,
        "filename_overlap_train_test": 0,
        "filename_overlap_val_test": 0,
        "subset_hash_train_val": 9,
        "subset_hash_train_test": 4,
        "subset_hash_val_test": 7,
        "subset_hash_n_train": 297,
        "subset_hash_n_val": 224,
        "subset_hash_n_test": 225,
        "source_type": "main_saved_output",
    },
    {
        "dataset": "DS2",
        "path_overlap_train_val": 0,
        "path_overlap_train_test": 0,
        "path_overlap_val_test": 0,
        "unique_paths_train": 4921,
        "unique_paths_val": 1055,
        "unique_paths_test": 1055,
        "filename_overlap_train_val": 0,
        "filename_overlap_train_test": 0,
        "filename_overlap_val_test": 0,
        "subset_hash_train_val": 2,
        "subset_hash_train_test": 2,
        "subset_hash_val_test": 2,
        "subset_hash_n_train": 300,
        "subset_hash_n_val": 297,
        "subset_hash_n_test": 298,
        "source_type": "main_saved_output",
    },
])

SAVED_RL_PLANNING_HISTORY = pd.DataFrame([
    {"episode": 1,  "selected_n_clients": 3, "reward_ds1": 0.709483, "reward_ds2": 0.663089, "reward_mean": 0.686286, "bandit_value": 0.686286, "pulls_for_arm": 1},
    {"episode": 2,  "selected_n_clients": 4, "reward_ds1": 0.721983, "reward_ds2": 0.738356, "reward_mean": 0.730169, "bandit_value": 0.730169, "pulls_for_arm": 1},
    {"episode": 3,  "selected_n_clients": 5, "reward_ds1": 0.675439, "reward_ds2": 0.698698, "reward_mean": 0.687069, "bandit_value": 0.687069, "pulls_for_arm": 1},
    {"episode": 4,  "selected_n_clients": 4, "reward_ds1": 0.753396, "reward_ds2": 0.829472, "reward_mean": 0.791434, "bandit_value": 0.760801, "pulls_for_arm": 2},
    {"episode": 5,  "selected_n_clients": 5, "reward_ds1": 0.702440, "reward_ds2": 0.692132, "reward_mean": 0.697286, "bandit_value": 0.692177, "pulls_for_arm": 2},
    {"episode": 6,  "selected_n_clients": 3, "reward_ds1": 0.870872, "reward_ds2": 0.708757, "reward_mean": 0.789815, "bandit_value": 0.738050, "pulls_for_arm": 2},
    {"episode": 7,  "selected_n_clients": 4, "reward_ds1": 0.705301, "reward_ds2": 0.608192, "reward_mean": 0.656747, "bandit_value": 0.726116, "pulls_for_arm": 3},
    {"episode": 8,  "selected_n_clients": 3, "reward_ds1": 0.766992, "reward_ds2": 0.753060, "reward_mean": 0.760026, "bandit_value": 0.745376, "pulls_for_arm": 3},
    {"episode": 9,  "selected_n_clients": 5, "reward_ds1": 0.664448, "reward_ds2": 0.652680, "reward_mean": 0.658564, "bandit_value": 0.680973, "pulls_for_arm": 3},
    {"episode": 10, "selected_n_clients": 3, "reward_ds1": 0.753155, "reward_ds2": 0.660297, "reward_mean": 0.706726, "bandit_value": 0.735713, "pulls_for_arm": 4},
])
SAVED_RL_PLANNING_HISTORY["source_type"] = "main_saved_output"

SAVED_CLIENT_DISTRIBUTION = pd.DataFrame([
    {
        "client": "client_0",
        "dataset": "ds1",
        "total_train": 271,
        "total_tune": 42,
        "total_val": 37,
        "glioma": 77,
        "meningioma": 34,
        "notumor": 79,
        "pituitary": 81,
        "source_type": "main_saved_output",
    },
    {
        "client": "client_1",
        "dataset": "ds1",
        "total_train": 167,
        "total_tune": 27,
        "total_val": 23,
        "glioma": 4,
        "meningioma": 45,
        "notumor": 5,
        "pituitary": 113,
        "source_type": "main_saved_output",
    },
    {
        "client": "client_2",
        "dataset": "ds1",
        "total_train": 375,
        "total_tune": 59,
        "total_val": 52,
        "glioma": 121,
        "meningioma": 117,
        "notumor": 130,
        "pituitary": 7,
        "source_type": "main_saved_output",
    },
    {
        "client": "client_3",
        "dataset": "ds2",
        "total_train": 1092,
        "total_tune": 170,
        "total_val": 150,
        "glioma": 39,
        "meningioma": 251,
        "notumor": 223,
        "pituitary": 579,
        "source_type": "main_saved_output",
    },
    {
        "client": "client_4",
        "dataset": "ds2",
        "total_train": 1690,
        "total_tune": 262,
        "total_val": 231,
        "glioma": 275,
        "meningioma": 209,
        "notumor": 852,
        "pituitary": 354,
        "source_type": "main_saved_output",
    },
    {
        "client": "client_5",
        "dataset": "ds2",
        "total_train": 1026,
        "total_tune": 160,
        "total_val": 140,
        "glioma": 563,
        "meningioma": 433,
        "notumor": 8,
        "pituitary": 22,
        "source_type": "main_saved_output",
    },
])

SAVED_TRAINING_SUMMARY = pd.DataFrame([
    {"item": "environment", "value": "KAGGLE", "source_type": "main_saved_output"},
    {"item": "device", "value": "cuda", "source_type": "main_saved_output"},
    {"item": "torch_version", "value": "2.10.0+cu128", "source_type": "main_saved_output"},
    {"item": "method_printed", "value": "ARCF-Net (Adaptive RACE-FELCM with CRAF Fusion Network)", "source_type": "main_saved_output"},
    {"item": "backbone", "value": "ResNet-50", "source_type": "main_saved_output"},
    {"item": "pretrained_loaded", "value": "True", "source_type": "main_saved_output"},
    {"item": "total_params", "value": "25,790,855", "source_type": "main_saved_output"},
    {"item": "trainable_params", "value": "2,282,823", "source_type": "main_saved_output"},
    {"item": "trainable_percent", "value": "8.85%", "source_type": "main_saved_output"},
    {"item": "adaptive_clients", "value": "DS1=3, DS2=3, TOTAL=6", "source_type": "main_saved_output"},
    {"item": "rounds", "value": "15", "source_type": "main_saved_output"},
    {"item": "local_epochs", "value": "2", "source_type": "main_saved_output"},
    {"item": "fedprox_mu", "value": "0.01", "source_type": "main_saved_output"},
    {"item": "proto_lambda", "value": "0.12", "source_type": "main_saved_output"},
    {"item": "fedavg_temper", "value": "0.50", "source_type": "main_saved_output"},
    {"item": "best_round", "value": "14", "source_type": "main_saved_output"},
    {"item": "best_reward", "value": "1.1156", "source_type": "main_saved_output"},
    {"item": "total_time_seconds", "value": "2721.0", "source_type": "main_saved_output"},
    {"item": "global_test_acc", "value": "0.9900", "source_type": "main_saved_output"},
    {"item": "global_test_f1_macro", "value": "0.9898", "source_type": "main_saved_output"},
    {"item": "ds1_test_acc", "value": "1.0000", "source_type": "main_saved_output"},
    {"item": "ds2_test_acc", "value": "0.9801", "source_type": "main_saved_output"},
    {"item": "ds1_final_strategy", "value": "single | names=['race_edge_plus']", "source_type": "main_saved_output"},
    {"item": "ds2_final_strategy", "value": "single | names=['race_soft']", "source_type": "main_saved_output"},
])

# ============================================================
# Helper functions
# ============================================================

def section(title):
    print("\n" + "=" * 150)
    print(title)
    print("=" * 150)

def sub(title):
    print("\n" + "-" * 150)
    print(title)
    print("-" * 150)

def show_df(df, title):
    sub(title)
    if df is None or len(df) == 0:
        print("EMPTY")
    else:
        print(df.to_string(index=False))

def pct(x, denom):
    return 100.0 * x / denom if denom else 0.0

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
    folder_rows = []

    for c in class_dirs:
        lab = norm_label(c)
        imgs = list_images_under_class_root(ds_root, c)

        folder_rows.append({
            "dataset": "DS1" if source_name == "ds1_raw" else "DS2",
            "source_folder": c,
            "mapped_label": lab,
            "n_images": len(imgs),
            "source_type": "new_dataset_audit_from_kaggle_files",
        })

        for p in imgs:
            rows.append({
                "path": p,
                "label": lab,
                "source": source_name,
            })

    dfm = pd.DataFrame(rows).dropna().reset_index(drop=True)

    if len(dfm) > 0:
        dfm["path"] = dfm["path"].astype(str)
        dfm["label"] = dfm["label"].astype(str)
        dfm["source"] = dfm["source"].astype(str)
        dfm = dfm.drop_duplicates(subset=["path"]).reset_index(drop=True)
        dfm["filename"] = dfm["path"].apply(os.path.basename)

    return dfm, pd.DataFrame(folder_rows)

def enforce_labels(df_):
    df_ = df_.copy()
    df_["label"] = df_["label"].astype(str).str.strip().str.lower()
    df_ = df_[df_["label"].isin(set(LABELS))].reset_index(drop=True)
    df_["y"] = df_["label"].map(label2id).astype(int)
    return df_

def class_count_table(df, dataset_name, source_type):
    rows = []
    total = len(df)

    for lab in LABELS:
        n = int((df["label"] == lab).sum())
        rows.append({
            "dataset": dataset_name,
            "class_label": lab,
            "n_images": n,
            "pct_within_dataset": round(pct(n, total), 6),
            "source_type": source_type,
        })

    return pd.DataFrame(rows)

def full_md5(path, chunk=1024 * 1024):
    h = hashlib.md5()
    try:
        with open(path, "rb") as f:
            while True:
                b = f.read(chunk)
                if not b:
                    break
                h.update(b)
        return h.hexdigest()
    except Exception:
        return None

def compare_folder_counts(computed_counts):
    ref = SAVED_FOLDER_COUNTS[["dataset", "source_folder", "mapped_label", "n_images"]].sort_values(
        ["dataset", "source_folder"]
    ).reset_index(drop=True)

    comp = computed_counts[["dataset", "source_folder", "mapped_label", "n_images"]].sort_values(
        ["dataset", "source_folder"]
    ).reset_index(drop=True)

    ok = ref.equals(comp)

    comparison = ref.copy()
    comparison = comparison.rename(columns={"n_images": "main_saved_output_n_images"})
    comparison["computed_kaggle_audit_n_images"] = comp["n_images"]
    comparison["match"] = comparison["main_saved_output_n_images"] == comparison["computed_kaggle_audit_n_images"]

    return ok, comparison

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

def entropy_from_counts(counts):
    arr = np.array(counts, dtype=float)
    total = arr.sum()
    if total <= 0:
        return 0.0
    p = arr / total
    nz = p[p > 0]
    return float(-(nz * np.log(nz)).sum() / np.log(len(arr)))

# ============================================================
# 0. Print exact main-code and main-output reference values
# ============================================================

section("0. EXACT REFERENCE VALUES FROM MAIN CODE AND SAVED MAIN OUTPUT")

cfg_rows = []
for k, v in CFG.items():
    cfg_rows.append({
        "parameter": k,
        "value": json.dumps(v),
        "source_type": "main_code_config",
    })

cfg_rows += [
    {
        "parameter": "IMG_EXTS",
        "value": ", ".join(IMG_EXTS),
        "source_type": "main_code_config",
    },
    {
        "parameter": "labels",
        "value": ", ".join(LABELS),
        "source_type": "main_code_config",
    },
    {
        "parameter": "DS1_Kaggle_handle",
        "value": DS1_SLUG,
        "source_type": "main_code_config",
    },
    {
        "parameter": "DS2_Kaggle_handle",
        "value": DS2_SLUG,
        "source_type": "main_code_config",
    },
]

show_df(pd.DataFrame(cfg_rows), "Main code/config values")

show_df(SAVED_TRAINING_SUMMARY, "Main saved-output training summary")
show_df(SAVED_FOLDER_COUNTS, "Main saved-output folder counts")
show_df(SAVED_DATASET_TOTALS, "Main saved-output dataset totals")
show_df(SAVED_SPLIT_SUMMARY, "Main saved-output train/validation/test split summary")
show_df(SAVED_LEAKAGE_SUMMARY, "Main saved-output leakage/sanity summary")
show_df(SAVED_RL_PLANNING_HISTORY, "Main saved-output RL-UCB client-count planning history")
show_df(SAVED_CLIENT_DISTRIBUTION, "Main saved-output adaptive client class distribution")

# ============================================================
# 1. Download/read Kaggle datasets
# ============================================================

section("1. DOWNLOAD DATASETS FROM KAGGLEHUB FOR DATASET-ONLY AUDIT")

print("This step downloads/reads the datasets only.")
print("It does NOT run model training.")
print("It does NOT run federated learning.")
print("It does NOT run RACE-FELCM training/evaluation.")
print()
print("Downloading/checking DS1:", DS1_SLUG)
DS1_BASE = kagglehub.dataset_download(DS1_SLUG)
print("DS1 downloaded/cached at:", DS1_BASE)

print()
print("Downloading/checking DS2:", DS2_SLUG)
DS2_BASE = kagglehub.dataset_download(DS2_SLUG)
print("DS2 downloaded/cached at:", DS2_BASE)

DS1_ROOT = find_root_with_required_class_dirs(DS1_BASE, REQ1, prefer_raw=True)
DS2_ROOT = find_root_with_required_class_dirs(DS2_BASE, REQ2, prefer_raw=False)

if DS1_ROOT is None:
    raise RuntimeError("Could not find DS1 root containing: " + ", ".join(sorted(REQ1)))
if DS2_ROOT is None:
    raise RuntimeError("Could not find DS2 root containing: " + ", ".join(sorted(REQ2)))

detected_paths = pd.DataFrame([
    {
        "dataset": "DS1",
        "kaggle_handle": DS1_SLUG,
        "download_base": DS1_BASE,
        "detected_root": DS1_ROOT,
        "required_folders": ", ".join(DS1_CLASS_DIRS),
        "source_type": "new_dataset_audit_from_kaggle_files",
    },
    {
        "dataset": "DS2",
        "kaggle_handle": DS2_SLUG,
        "download_base": DS2_BASE,
        "detected_root": DS2_ROOT,
        "required_folders": ", ".join(DS2_CLASS_DIRS),
        "source_type": "new_dataset_audit_from_kaggle_files",
    },
])

show_df(detected_paths, "Detected Kaggle dataset paths")

# ============================================================
# 2. Build manifests using main-code logic
# ============================================================

section("2. BUILD MANIFESTS USING SAME MAIN-CODE LOGIC")

df1_raw, folder_counts1 = build_df_from_root(DS1_ROOT, DS1_CLASS_DIRS, "ds1_raw")
df2_raw, folder_counts2 = build_df_from_root(DS2_ROOT, DS2_CLASS_DIRS, "ds2")

df1 = enforce_labels(df1_raw)
df2 = enforce_labels(df2_raw)

computed_folder_counts = pd.concat([folder_counts1, folder_counts2], ignore_index=True)
show_df(computed_folder_counts, "Computed folder counts from downloaded Kaggle files")

match_ok, comparison = compare_folder_counts(computed_folder_counts)
show_df(comparison, "Reference-vs-computed folder-count verification")

if STRICT_MATCH_MAIN_OUTPUT and not match_ok:
    raise RuntimeError(
        "Downloaded Kaggle data do NOT match the saved main-output folder counts. "
        "Stop here and do not use computed audit values as if they are from the same dataset version."
    )

computed_dataset_totals = pd.DataFrame([
    {
        "dataset": "DS1",
        "total_images": len(df1),
        "unique_paths": df1["path"].nunique(),
        "unique_filenames": df1["filename"].nunique(),
        "glioma": int((df1["label"] == "glioma").sum()),
        "meningioma": int((df1["label"] == "meningioma").sum()),
        "notumor": int((df1["label"] == "notumor").sum()),
        "pituitary": int((df1["label"] == "pituitary").sum()),
        "source_type": "new_dataset_audit_from_kaggle_files",
    },
    {
        "dataset": "DS2",
        "total_images": len(df2),
        "unique_paths": df2["path"].nunique(),
        "unique_filenames": df2["filename"].nunique(),
        "glioma": int((df2["label"] == "glioma").sum()),
        "meningioma": int((df2["label"] == "meningioma").sum()),
        "notumor": int((df2["label"] == "notumor").sum()),
        "pituitary": int((df2["label"] == "pituitary").sum()),
        "source_type": "new_dataset_audit_from_kaggle_files",
    },
])

show_df(computed_dataset_totals, "Computed dataset totals from downloaded Kaggle files")
show_df(class_count_table(df1, "DS1", "new_dataset_audit_from_kaggle_files"), "Computed DS1 class counts")
show_df(class_count_table(df2, "DS2", "new_dataset_audit_from_kaggle_files"), "Computed DS2 class counts")

# ============================================================
# 3.1 Dataset collection/source information
# ============================================================

section("3.1 DEVELOPMENT DATASET SOURCES — PAPER-READY AUDIT")

source_info = pd.DataFrame([
    {
        "dataset": "DS1",
        "dataset_name_to_write": "PMRAM Bangladeshi Brain Cancer MRI Dataset",
        "kaggle_handle_from_main_code": DS1_SLUG,
        "folder_names_used": "512Glioma, 512Meningioma, 512Normal, 512Pituitary",
        "dataset_state": "Raw MRI dataset",
        "kept_separate_before_splitting": True,
        "total_images_main_saved_output": 1505,
        "source_type": "main_code_config + main_saved_output",
    },
    {
        "dataset": "DS2",
        "dataset_name_to_write": "Preprocessed Brain MRI Scans for Tumors Detection",
        "kaggle_handle_from_main_code": DS2_SLUG,
        "folder_names_used": "glioma, meningioma, notumor, pituitary",
        "dataset_state": "Preprocessed MRI dataset",
        "kept_separate_before_splitting": True,
        "total_images_main_saved_output": 7031,
        "source_type": "main_code_config + main_saved_output",
    },
])

show_df(source_info, "Dataset source information for Section 3.1")

# ============================================================
# 3.2 Class labels and mapping
# ============================================================

section("3.2 CLASS LABELS AND SOURCE-TO-TARGET LABEL MAPPING")

label_mapping = pd.DataFrame([
    {"label_id": 0, "final_label": "glioma",     "DS1_source_folder": "512Glioma",     "DS2_source_folder": "glioma",     "source_type": "main_code_config"},
    {"label_id": 1, "final_label": "meningioma", "DS1_source_folder": "512Meningioma", "DS2_source_folder": "meningioma", "source_type": "main_code_config"},
    {"label_id": 2, "final_label": "notumor",    "DS1_source_folder": "512Normal",     "DS2_source_folder": "notumor",    "source_type": "main_code_config"},
    {"label_id": 3, "final_label": "pituitary",  "DS1_source_folder": "512Pituitary",  "DS2_source_folder": "pituitary",  "source_type": "main_code_config"},
])

show_df(label_mapping, "Label mapping table")

label_integrity = pd.DataFrame([
    {
        "dataset": "DS1",
        "unique_labels": df1["label"].nunique(),
        "labels_present": ", ".join(sorted(df1["label"].unique())),
        "integer_y_values": ", ".join(map(str, sorted(df1["y"].unique()))),
        "all_labels_match_target_set": set(df1["label"].unique()).issubset(set(LABELS)),
        "source_type": "new_dataset_audit_from_kaggle_files",
    },
    {
        "dataset": "DS2",
        "unique_labels": df2["label"].nunique(),
        "labels_present": ", ".join(sorted(df2["label"].unique())),
        "integer_y_values": ", ".join(map(str, sorted(df2["y"].unique()))),
        "all_labels_match_target_set": set(df2["label"].unique()).issubset(set(LABELS)),
        "source_type": "new_dataset_audit_from_kaggle_files",
    },
])

show_df(label_integrity, "Label integrity check")

# ============================================================
# 3.3 Inclusion, extensions, image size, unreadable audit
# ============================================================

section("3.3 IMAGE INCLUSION, DUPLICATE HANDLING, LABEL HARMONIZATION, AND IMAGE SIZE AUDIT")

inclusion_summary = pd.DataFrame([
    {
        "dataset": "DS1",
        "supported_extensions_from_main_code": ", ".join(IMG_EXTS),
        "rows_after_label_harmonization": len(df1),
        "unique_paths_after_duplicate_path_removal": df1["path"].nunique(),
        "duplicate_path_rows_remaining": len(df1) - df1["path"].nunique(),
        "unique_filenames": df1["filename"].nunique(),
        "image_loading_rule_from_main_code": "Image.open(path).convert('RGB'); on exception returns gray RGB placeholder of model input size",
        "model_input_resize_saved_run": "224x224",
        "source_type": "main_code_config + new_dataset_audit_from_kaggle_files",
    },
    {
        "dataset": "DS2",
        "supported_extensions_from_main_code": ", ".join(IMG_EXTS),
        "rows_after_label_harmonization": len(df2),
        "unique_paths_after_duplicate_path_removal": df2["path"].nunique(),
        "duplicate_path_rows_remaining": len(df2) - df2["path"].nunique(),
        "unique_filenames": df2["filename"].nunique(),
        "image_loading_rule_from_main_code": "Image.open(path).convert('RGB'); on exception returns gray RGB placeholder of model input size",
        "model_input_resize_saved_run": "224x224",
        "source_type": "main_code_config + new_dataset_audit_from_kaggle_files",
    },
])

show_df(inclusion_summary, "Inclusion and harmonization summary")

# Extension counts
ext_rows = []
for dataset_name, df in [("DS1", df1), ("DS2", df2)]:
    tmp = df.copy()
    tmp["extension"] = tmp["path"].map(lambda p: os.path.splitext(str(p))[1].lower().replace(".", ""))

    for ext, g in tmp.groupby("extension"):
        ext_rows.append({
            "dataset": dataset_name,
            "extension": ext,
            "n_images": len(g),
            "pct_within_dataset": round(pct(len(g), len(df)), 6),
            "source_type": "new_dataset_audit_from_kaggle_files",
        })

show_df(pd.DataFrame(ext_rows), "Included image extension counts")

# Image-size audit
size_rows = []
bad_size_rows = []

for dataset_name, df in [("DS1", df1), ("DS2", df2)]:
    for _, r in df.iterrows():
        p = r["path"]

        try:
            with Image.open(p) as im:
                w, h = im.size
                mode = im.mode
                fmt = im.format

            size_rows.append({
                "dataset": dataset_name,
                "label": r["label"],
                "filename": r["filename"],
                "width": int(w),
                "height": int(h),
                "size": f"{w}x{h}",
                "aspect_ratio": round(float(w) / float(h), 6) if h else np.nan,
                "mode": str(mode),
                "format": str(fmt),
                "path": p,
                "source_type": "new_dataset_audit_from_kaggle_files",
            })

        except Exception as e:
            bad_size_rows.append({
                "dataset": dataset_name,
                "label": r.get("label", "UNKNOWN"),
                "filename": r.get("filename", os.path.basename(str(p))),
                "path": p,
                "error": str(e)[:250],
                "source_type": "new_dataset_audit_from_kaggle_files",
            })

size_df = pd.DataFrame(size_rows)

show_df(
    size_df.groupby(["dataset", "size"]).size().reset_index(name="n_images").sort_values(
        ["dataset", "n_images"], ascending=[True, False]
    ).assign(source_type="new_dataset_audit_from_kaggle_files"),
    "Original/native image size distribution by dataset"
)

show_df(
    size_df.groupby(["dataset", "label", "size"]).size().reset_index(name="n_images").sort_values(
        ["dataset", "label", "n_images"], ascending=[True, True, False]
    ).assign(source_type="new_dataset_audit_from_kaggle_files"),
    "Original/native image size distribution by dataset and class"
)

size_summary = size_df.groupby("dataset").agg(
    n_images=("size", "count"),
    unique_sizes=("size", "nunique"),
    min_width=("width", "min"),
    max_width=("width", "max"),
    min_height=("height", "min"),
    max_height=("height", "max"),
    mean_width=("width", "mean"),
    mean_height=("height", "mean"),
    median_width=("width", "median"),
    median_height=("height", "median"),
    most_common_size=("size", lambda x: x.value_counts().index[0]),
    most_common_size_count=("size", lambda x: int(x.value_counts().iloc[0])),
).reset_index()

size_summary["mean_width"] = size_summary["mean_width"].round(6)
size_summary["mean_height"] = size_summary["mean_height"].round(6)
size_summary["source_type"] = "new_dataset_audit_from_kaggle_files"

show_df(size_summary, "Original/native image size summary")

show_df(
    size_df.groupby(["dataset", "mode"]).size().reset_index(name="n_images").sort_values(
        ["dataset", "n_images"], ascending=[True, False]
    ).assign(source_type="new_dataset_audit_from_kaggle_files"),
    "PIL image mode distribution"
)

show_df(
    size_df.groupby(["dataset", "format"]).size().reset_index(name="n_images").sort_values(
        ["dataset", "n_images"], ascending=[True, False]
    ).assign(source_type="new_dataset_audit_from_kaggle_files"),
    "PIL image format distribution"
)

if bad_size_rows:
    show_df(pd.DataFrame(bad_size_rows), "Unreadable image / size-read failures")
else:
    show_df(
        pd.DataFrame([{
            "dataset": "ALL",
            "unreadable_images_detected": 0,
            "note": "All images could be opened by PIL during this audit.",
            "source_type": "new_dataset_audit_from_kaggle_files",
        }]),
        "Unreadable image / size-read failures"
    )

if PRINT_PER_IMAGE_SIZE_TABLE:
    show_df(
        size_df[["dataset", "label", "filename", "width", "height", "size", "mode", "format", "path", "source_type"]],
        "Per-image original/native size table"
    )

# Duplicate filename summary
filename_dup_rows = []
for dataset_name, df in [("DS1", df1), ("DS2", df2)]:
    vc = df["filename"].value_counts()
    dup = vc[vc > 1]

    filename_dup_rows.append({
        "dataset": dataset_name,
        "duplicate_filename_values": int(len(dup)),
        "images_with_duplicate_filenames": int(dup.sum()) if len(dup) else 0,
        "source_type": "new_dataset_audit_from_kaggle_files",
    })

show_df(pd.DataFrame(filename_dup_rows), "Duplicate filename summary")

# ============================================================
# Full MD5 duplicate audit across datasets
# ============================================================

section("FULL MD5 DUPLICATE AUDIT FROM DOWNLOADED KAGGLE FILES")

print("Computing full MD5 hashes. This reads image bytes only; it does not train anything.")

hash_rows = []
for dataset_name, df in [("DS1", df1), ("DS2", df2)]:
    for _, r in df.iterrows():
        hash_rows.append({
            "dataset": dataset_name,
            "label": r["label"],
            "filename": r["filename"],
            "path": r["path"],
            "md5": full_md5(r["path"]),
            "source_type": "new_dataset_audit_from_kaggle_files",
        })

hash_df = pd.DataFrame(hash_rows)

show_df(
    hash_df.groupby("dataset")["md5"].agg(
        total_images="count",
        unique_md5="nunique"
    ).reset_index().assign(source_type="new_dataset_audit_from_kaggle_files"),
    "Full MD5 uniqueness summary"
)

dup_md5_rows = []
for dataset_name in ["DS1", "DS2"]:
    hds = hash_df[hash_df["dataset"] == dataset_name].copy()
    vc = hds["md5"].value_counts()
    dup = vc[vc > 1]

    dup_md5_rows.append({
        "dataset": dataset_name,
        "duplicate_md5_hashes_within_dataset": int(len(dup)),
        "images_in_duplicate_md5_groups": int(dup.sum()) if len(dup) else 0,
        "source_type": "new_dataset_audit_from_kaggle_files",
    })

cross = set(hash_df[hash_df["dataset"] == "DS1"]["md5"].dropna()) & set(hash_df[hash_df["dataset"] == "DS2"]["md5"].dropna())

dup_md5_rows.append({
    "dataset": "DS1_vs_DS2",
    "duplicate_md5_hashes_between_datasets": int(len(cross)),
    "images_in_cross_dataset_duplicate_groups": int(hash_df["md5"].isin(cross).sum()) if len(cross) else 0,
    "source_type": "new_dataset_audit_from_kaggle_files",
})

show_df(pd.DataFrame(dup_md5_rows), "Full MD5 duplicate summary")

duplicate_hashes = hash_df["md5"].value_counts()
duplicate_hashes = duplicate_hashes[duplicate_hashes > 1].index.tolist()

if duplicate_hashes:
    examples = hash_df[hash_df["md5"].isin(duplicate_hashes[:MAX_EXAMPLES_TO_PRINT])].sort_values(
        ["md5", "dataset", "label", "filename"]
    )

    show_df(
        examples[["dataset", "label", "filename", "md5", "path", "source_type"]],
        "Full MD5 duplicate examples"
    )
else:
    show_df(
        pd.DataFrame([{
            "result": "No duplicate full-MD5 hashes found across all audited images.",
            "source_type": "new_dataset_audit_from_kaggle_files",
        }]),
        "Full MD5 duplicate examples"
    )

if PRINT_ALL_HASH_ROWS:
    show_df(hash_df, "All image full-MD5 rows")

# ============================================================
# 3.4 Split reconstruction using main-code logic + full MD5 leakage
# ============================================================

section("3.4 TRAIN/VALIDATION/TEST SPLITTING AND LEAKAGE CONTROL")

train1, val1, test1 = split_dataset(df1)
train2, val2, test2 = split_dataset(df2)

computed_split_summary = pd.DataFrame([
    {
        "dataset": "DS1",
        "train": len(train1),
        "val": len(val1),
        "test": len(test1),
        "total": len(train1) + len(val1) + len(test1),
        "source_type": "computed_with_main_split_code_on_downloaded_kaggle_files",
    },
    {
        "dataset": "DS2",
        "train": len(train2),
        "val": len(val2),
        "test": len(test2),
        "total": len(train2) + len(val2) + len(test2),
        "source_type": "computed_with_main_split_code_on_downloaded_kaggle_files",
    },
])

show_df(SAVED_SPLIT_SUMMARY, "Main saved-output split summary")
show_df(computed_split_summary, "Computed split summary using main-code split logic")

split_match = (
    list(computed_split_summary["train"]) == list(SAVED_SPLIT_SUMMARY["train"]) and
    list(computed_split_summary["val"]) == list(SAVED_SPLIT_SUMMARY["val"]) and
    list(computed_split_summary["test"]) == list(SAVED_SPLIT_SUMMARY["test"])
)

if STRICT_MATCH_MAIN_OUTPUT and not split_match:
    raise RuntimeError(
        "Computed split sizes do not match main saved-output split sizes. "
        "Do not use computed split audit as if it matches the original run."
    )

split_rows = []
split_sets = {
    "DS1": {"train": train1, "val": val1, "test": test1},
    "DS2": {"train": train2, "val": val2, "test": test2},
}

for dataset_name, parts in split_sets.items():
    total = sum(len(x) for x in parts.values())

    for split_name, frame in parts.items():
        row = {
            "dataset": dataset_name,
            "split": split_name,
            "n_images": len(frame),
            "pct_of_dataset": round(pct(len(frame), total), 6),
            "source_type": "computed_with_main_split_code_on_downloaded_kaggle_files",
        }

        for lab in LABELS:
            n = int((frame["label"] == lab).sum())
            row[f"n_{lab}"] = n
            row[f"pct_{lab}_within_split"] = round(pct(n, len(frame)), 6)

        split_rows.append(row)

show_df(pd.DataFrame(split_rows), "Computed split-wise class counts and percentages")

split_config = pd.DataFrame([
    {
        "parameter": "split_scope",
        "value": "DS1 and DS2 split separately",
        "source_type": "main_code_config",
    },
    {
        "parameter": "split_method",
        "value": "stratified train/temp split, then stratified validation/test split",
        "source_type": "main_code_config",
    },
    {
        "parameter": "global_val_frac",
        "value": CFG["global_val_frac"],
        "source_type": "main_code_config",
    },
    {
        "parameter": "test_frac",
        "value": CFG["test_frac"],
        "source_type": "main_code_config",
    },
    {
        "parameter": "random_state",
        "value": SEED,
        "source_type": "main_code_config",
    },
])

show_df(split_config, "Main-code split configuration")

# Full MD5 leakage across computed splits
hash_map = dict(zip(hash_df["path"], hash_df["md5"]))

split_hash_rows = []
for dataset_name, parts in split_sets.items():
    for split_name, frame in parts.items():
        for _, r in frame.iterrows():
            split_hash_rows.append({
                "dataset": dataset_name,
                "split": split_name,
                "label": r["label"],
                "filename": r["filename"],
                "path": r["path"],
                "md5": hash_map.get(r["path"]),
                "source_type": "computed_with_main_split_code_on_downloaded_kaggle_files",
            })

split_hash_df = pd.DataFrame(split_hash_rows)

full_leak_rows = []
for dataset_name in ["DS1", "DS2"]:
    hds = split_hash_df[split_hash_df["dataset"] == dataset_name].copy()

    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        A = hds[hds["split"] == a]
        B = hds[hds["split"] == b]

        path_overlap = set(A["path"]) & set(B["path"])
        filename_overlap = set(A["filename"]) & set(B["filename"])
        md5_overlap = set(A["md5"].dropna()) & set(B["md5"].dropna())

        conflict_labels = 0
        for hv in md5_overlap:
            labs = set(hds[hds["md5"] == hv]["label"])
            if len(labs) > 1:
                conflict_labels += 1

        full_leak_rows.append({
            "dataset": dataset_name,
            "split_pair": f"{a}_vs_{b}",
            "path_overlap": int(len(path_overlap)),
            "filename_overlap": int(len(filename_overlap)),
            "full_md5_overlap_unique_hashes": int(len(md5_overlap)),
            "full_md5_overlap_images_in_pair": int(A["md5"].isin(md5_overlap).sum() + B["md5"].isin(md5_overlap).sum()),
            "overlap_hashes_with_conflicting_labels": int(conflict_labels),
            "source_type": "computed_with_main_split_code_on_downloaded_kaggle_files",
        })

full_leak_df = pd.DataFrame(full_leak_rows)

show_df(SAVED_LEAKAGE_SUMMARY, "Main saved-output quick/subset leakage summary")
show_df(full_leak_df, "Computed full-MD5 leakage summary across train/validation/test")

if full_leak_df["full_md5_overlap_unique_hashes"].sum() > 0:
    example_hashes = []

    for dataset_name in ["DS1", "DS2"]:
        hds = split_hash_df[split_hash_df["dataset"] == dataset_name].copy()

        for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
            A = hds[hds["split"] == a]
            B = hds[hds["split"] == b]
            example_hashes.extend(list(set(A["md5"].dropna()) & set(B["md5"].dropna())))

    example_hashes = list(dict.fromkeys(example_hashes))[:MAX_EXAMPLES_TO_PRINT]

    leakage_examples = split_hash_df[split_hash_df["md5"].isin(example_hashes)].sort_values(
        ["dataset", "md5", "split", "label", "filename"]
    )

    show_df(
        leakage_examples[["dataset", "split", "label", "filename", "md5", "path", "source_type"]],
        "Computed full-MD5 train/validation/test leakage examples"
    )
else:
    show_df(
        pd.DataFrame([{
            "result": "No full-MD5 train/validation/test leakage detected in computed split audit.",
            "source_type": "computed_with_main_split_code_on_downloaded_kaggle_files",
        }]),
        "Computed full-MD5 train/validation/test leakage examples"
    )

# ============================================================
# 3.5 Non-IID federated cohort simulation from saved output
# ============================================================

section("3.5 NON-IID FEDERATED COHORT SIMULATION")

federated_config = pd.DataFrame([
    {"parameter": "candidate_client_counts", "value": "[3, 4, 5]", "source_type": "main_code_config"},
    {"parameter": "client_count_search_episodes", "value": "10", "source_type": "main_code_config"},
    {"parameter": "selected_DS1_clients", "value": "3", "source_type": "main_saved_output"},
    {"parameter": "selected_DS2_clients", "value": "3", "source_type": "main_saved_output"},
    {"parameter": "total_adaptive_clients", "value": "6", "source_type": "main_saved_output"},
    {"parameter": "dirichlet_alpha", "value": "0.35", "source_type": "main_code_config"},
    {"parameter": "min_per_class_per_client", "value": "5", "source_type": "main_code_config"},
    {"parameter": "client_tune_frac", "value": "0.12", "source_type": "main_code_config"},
    {"parameter": "client_val_frac", "value": "0.12", "source_type": "main_code_config"},
    {"parameter": "weighted_sampler", "value": "1/sqrt(class_count) local class weighting", "source_type": "main_code_config"},
    {"parameter": "client_participation_in_saved_training", "value": "all six clients selected every round", "source_type": "main_saved_output"},
])

show_df(federated_config, "Federated simulation configuration")

show_df(SAVED_RL_PLANNING_HISTORY, "Exact RL-UCB planning history from main saved output")
show_df(SAVED_CLIENT_DISTRIBUTION, "Exact adaptive client distribution from main saved output")

# Derive non-IID mathematical statistics from saved client distribution
derived_client_rows = []
for _, r in SAVED_CLIENT_DISTRIBUTION.iterrows():
    counts = [int(r[lab]) for lab in LABELS]
    total = sum(counts)
    arr = np.array(counts, dtype=float)
    probs = arr / arr.sum() if arr.sum() else np.zeros_like(arr)
    nonzero = arr[arr > 0]

    derived_client_rows.append({
        "client": r["client"],
        "dataset": r["dataset"],
        "total_train": int(r["total_train"]),
        "glioma": int(r["glioma"]),
        "meningioma": int(r["meningioma"]),
        "notumor": int(r["notumor"]),
        "pituitary": int(r["pituitary"]),
        "train_entropy_normalized": round(entropy_from_counts(counts), 6),
        "dominant_class": LABELS[int(np.argmax(arr))],
        "dominant_class_fraction": round(float(probs.max()) if len(probs) else 0.0, 6),
        "class_coverage_fraction": round(float((arr > 0).mean()), 6),
        "imbalance_ratio_max_min_nonzero": round(float(nonzero.max() / nonzero.min()) if len(nonzero) else 0.0, 6),
        "source_type": "derived_from_main_saved_output_client_distribution",
    })

derived_client_df = pd.DataFrame(derived_client_rows)

show_df(derived_client_df, "Derived non-IID client statistics from saved client distribution")

derived_dataset_summary = derived_client_df.groupby("dataset").agg(
    n_clients=("client", "count"),
    total_train=("total_train", "sum"),
    min_client_train=("total_train", "min"),
    max_client_train=("total_train", "max"),
    mean_client_train=("total_train", "mean"),
    mean_entropy=("train_entropy_normalized", "mean"),
    mean_dominant_class_fraction=("dominant_class_fraction", "mean"),
    mean_imbalance_ratio=("imbalance_ratio_max_min_nonzero", "mean"),
).reset_index()

for col in ["mean_client_train", "mean_entropy", "mean_dominant_class_fraction", "mean_imbalance_ratio"]:
    derived_dataset_summary[col] = derived_dataset_summary[col].round(6)

derived_dataset_summary["source_type"] = "derived_from_main_saved_output_client_distribution"

show_df(derived_dataset_summary, "Derived dataset-level non-IID summary from saved client distribution")

# ============================================================
# Final Section 3 checklist
# ============================================================

section("FINAL SECTION 3 CHECKLIST — WHAT EACH VALUE IS BASED ON")

checklist = pd.DataFrame([
    {
        "section": "3.1 Development Dataset Sources",
        "available_now": "Yes",
        "basis": "main code Kaggle handles + main saved-output counts + downloaded-file verification",
    },
    {
        "section": "3.2 Class Labels and Source-to-Target Label Mapping",
        "available_now": "Yes",
        "basis": "main code label mapping and label IDs",
    },
    {
        "section": "3.3 Image Inclusion, Duplicate Handling, and Label Harmonization",
        "available_now": "Yes",
        "basis": "main code extension/filter/load logic + new Kaggle file audit for image sizes, formats, modes, duplicates, unreadable files",
    },
    {
        "section": "3.4 Train/Validation/Test Splitting and Leakage Control",
        "available_now": "Yes",
        "basis": "main saved-output split/leakage table + recomputed full-MD5 audit using same split code",
    },
    {
        "section": "3.5 Non-IID Federated Cohort Simulation",
        "available_now": "Yes",
        "basis": "main code config + exact saved RL-UCB history + exact saved adaptive client distribution + derived non-IID statistics",
    },
])

show_df(checklist, "Section 3 evidence map")

print("\nDONE.")
print("No model training was run.")
print("No federated training was run.")
print("No RACE-FELCM training/evaluation was run.")
print("All source types are labeled in the printed tables.")
print("If STRICT_MATCH_MAIN_OUTPUT=True and counts mismatch, this code stops instead of giving unsafe values.")
```

    
    ======================================================================================================================================================
    0. EXACT REFERENCE VALUES FROM MAIN CODE AND SAVED MAIN OUTPUT
    ======================================================================================================================================================
    
    ------------------------------------------------------------------------------------------------------------------------------------------------------
    Main code/config values
    ------------------------------------------------------------------------------------------------------------------------------------------------------
                       parameter                                                            value      source_type
                          rounds                                                               15 main_code_config
         client_count_candidates                                                        [3, 4, 5] main_code_config
    client_count_search_episodes                                                               10 main_code_config
                    local_epochs                                                                2 main_code_config
                         lr_head                                                            0.001 main_code_config
                     lr_backbone                                                           0.0002 main_code_config
                    weight_decay                                                           0.0005 main_code_config
                   warmup_epochs                                                                1 main_code_config
                 label_smoothing                                                             0.02 main_code_config
                     focal_gamma                                                             1.35 main_code_config
                       grad_clip                                                              1.0 main_code_config
                      fedprox_mu                                                             0.01 main_code_config
                    proto_lambda                                                             0.12 main_code_config
              img_size_saved_run                                                              224 main_code_config
            batch_size_saved_run                                                               16 main_code_config
                 global_val_frac                                                             0.15 main_code_config
                       test_frac                                                             0.15 main_code_config
                 client_val_frac                                                             0.12 main_code_config
                client_tune_frac                                                             0.12 main_code_config
        min_per_class_per_client                                                                5 main_code_config
                 dirichlet_alpha                                                             0.35 main_code_config
               use_preprocessing                                                             true main_code_config
                use_augmentation                                                             true main_code_config
          freeze_backbone_rounds                                                                2 main_code_config
            unfreeze_last_blocks                                                                2 main_code_config
                           ucb_c                                                             1.35 main_code_config
                theta_probe_topk                                                                3 main_code_config
                reward_f1_weight                                                             0.75 main_code_config
               reward_acc_weight                                                             0.25 main_code_config
             best_round_mass_ds1                                                              0.5 main_code_config
             best_round_mass_ds2                                                              0.5 main_code_config
            best_round_min_bonus                                                             0.15 main_code_config
                   fedavg_temper                                                              0.5 main_code_config
     quick_hash_subset_per_split                                                              300 main_code_config
            preproc_val_sample_n                                                              400 main_code_config
                  before_after_n                                                               12 main_code_config
                calibration_bins                                                               12 main_code_config
                        IMG_EXTS                      .jpg, .jpeg, .png, .bmp, .tif, .tiff, .webp main_code_config
                          labels                           glioma, meningioma, notumor, pituitary main_code_config
               DS1_Kaggle_handle                orvile/pmram-bangladeshi-brain-cancer-mri-dataset main_code_config
               DS2_Kaggle_handle yassinebazgour/preprocessed-brain-mri-scans-for-tumors-detection main_code_config
    
    ------------------------------------------------------------------------------------------------------------------------------------------------------
    Main saved-output training summary
    ------------------------------------------------------------------------------------------------------------------------------------------------------
                    item                                                   value       source_type
             environment                                                  KAGGLE main_saved_output
                  device                                                    cuda main_saved_output
           torch_version                                            2.10.0+cu128 main_saved_output
          method_printed ARCF-Net (Adaptive RACE-FELCM with CRAF Fusion Network) main_saved_output
                backbone                                               ResNet-50 main_saved_output
       pretrained_loaded                                                    True main_saved_output
            total_params                                              25,790,855 main_saved_output
        trainable_params                                               2,282,823 main_saved_output
       trainable_percent                                                   8.85% main_saved_output
        adaptive_clients                                   DS1=3, DS2=3, TOTAL=6 main_saved_output
                  rounds                                                      15 main_saved_output
            local_epochs                                                       2 main_saved_output
              fedprox_mu                                                    0.01 main_saved_output
            proto_lambda                                                    0.12 main_saved_output
           fedavg_temper                                                    0.50 main_saved_output
              best_round                                                      14 main_saved_output
             best_reward                                                  1.1156 main_saved_output
      total_time_seconds                                                  2721.0 main_saved_output
         global_test_acc                                                  0.9900 main_saved_output
    global_test_f1_macro                                                  0.9898 main_saved_output
            ds1_test_acc                                                  1.0000 main_saved_output
            ds2_test_acc                                                  0.9801 main_saved_output
      ds1_final_strategy                       single | names=['race_edge_plus'] main_saved_output
      ds2_final_strategy                            single | names=['race_soft'] main_saved_output
    
    ------------------------------------------------------------------------------------------------------------------------------------------------------
    Main saved-output folder counts
    ------------------------------------------------------------------------------------------------------------------------------------------------------
    dataset source_folder mapped_label  n_images       source_type
        DS1     512Glioma       glioma       373 main_saved_output
        DS1 512Meningioma   meningioma       363 main_saved_output
        DS1     512Normal      notumor       396 main_saved_output
        DS1  512Pituitary    pituitary       373 main_saved_output
        DS2        glioma       glioma      1621 main_saved_output
        DS2    meningioma   meningioma      1646 main_saved_output
        DS2       notumor      notumor      2000 main_saved_output
        DS2     pituitary    pituitary      1764 main_saved_output
    
    ------------------------------------------------------------------------------------------------------------------------------------------------------
    Main saved-output dataset totals
    ------------------------------------------------------------------------------------------------------------------------------------------------------
    dataset  total_images  glioma  meningioma  notumor  pituitary       source_type
        DS1          1505     373         363      396        373 main_saved_output
        DS2          7031    1621        1646     2000       1764 main_saved_output
    
    ------------------------------------------------------------------------------------------------------------------------------------------------------
    Main saved-output train/validation/test split summary
    ------------------------------------------------------------------------------------------------------------------------------------------------------
    dataset  train  val  test  total       source_type
        DS1   1053  226   226   1505 main_saved_output
        DS2   4921 1055  1055   7031 main_saved_output
    
    ------------------------------------------------------------------------------------------------------------------------------------------------------
    Main saved-output leakage/sanity summary
    ------------------------------------------------------------------------------------------------------------------------------------------------------
    dataset  path_overlap_train_val  path_overlap_train_test  path_overlap_val_test  unique_paths_train  unique_paths_val  unique_paths_test  filename_overlap_train_val  filename_overlap_train_test  filename_overlap_val_test  subset_hash_train_val  subset_hash_train_test  subset_hash_val_test  subset_hash_n_train  subset_hash_n_val  subset_hash_n_test       source_type
        DS1                       0                        0                      0                1053               226                226                           0                            0                          0                      9                       4                     7                  297                224                 225 main_saved_output
        DS2                       0                        0                      0                4921              1055               1055                           0                            0                          0                      2                       2                     2                  300                297                 298 main_saved_output
    
    ------------------------------------------------------------------------------------------------------------------------------------------------------
    Main saved-output RL-UCB client-count planning history
    ------------------------------------------------------------------------------------------------------------------------------------------------------
     episode  selected_n_clients  reward_ds1  reward_ds2  reward_mean  bandit_value  pulls_for_arm       source_type
           1                   3    0.709483    0.663089     0.686286      0.686286              1 main_saved_output
           2                   4    0.721983    0.738356     0.730169      0.730169              1 main_saved_output
           3                   5    0.675439    0.698698     0.687069      0.687069              1 main_saved_output
           4                   4    0.753396    0.829472     0.791434      0.760801              2 main_saved_output
           5                   5    0.702440    0.692132     0.697286      0.692177              2 main_saved_output
           6                   3    0.870872    0.708757     0.789815      0.738050              2 main_saved_output
           7                   4    0.705301    0.608192     0.656747      0.726116              3 main_saved_output
           8                   3    0.766992    0.753060     0.760026      0.745376              3 main_saved_output
           9                   5    0.664448    0.652680     0.658564      0.680973              3 main_saved_output
          10                   3    0.753155    0.660297     0.706726      0.735713              4 main_saved_output
    
    ------------------------------------------------------------------------------------------------------------------------------------------------------
    Main saved-output adaptive client class distribution
    ------------------------------------------------------------------------------------------------------------------------------------------------------
      client dataset  total_train  total_tune  total_val  glioma  meningioma  notumor  pituitary       source_type
    client_0     ds1          271          42         37      77          34       79         81 main_saved_output
    client_1     ds1          167          27         23       4          45        5        113 main_saved_output
    client_2     ds1          375          59         52     121         117      130          7 main_saved_output
    client_3     ds2         1092         170        150      39         251      223        579 main_saved_output
    client_4     ds2         1690         262        231     275         209      852        354 main_saved_output
    client_5     ds2         1026         160        140     563         433        8         22 main_saved_output
    
    ======================================================================================================================================================
    1. DOWNLOAD DATASETS FROM KAGGLEHUB FOR DATASET-ONLY AUDIT
    ======================================================================================================================================================
    This step downloads/reads the datasets only.
    It does NOT run model training.
    It does NOT run federated learning.
    It does NOT run RACE-FELCM training/evaluation.
    
    Downloading/checking DS1: orvile/pmram-bangladeshi-brain-cancer-mri-dataset
    DS1 downloaded/cached at: /root/.cache/kagglehub/datasets/orvile/pmram-bangladeshi-brain-cancer-mri-dataset/versions/2
    
    Downloading/checking DS2: yassinebazgour/preprocessed-brain-mri-scans-for-tumors-detection
    Using Colab cache for faster access to the 'preprocessed-brain-mri-scans-for-tumors-detection' dataset.
    DS2 downloaded/cached at: /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection
    
    ------------------------------------------------------------------------------------------------------------------------------------------------------
    Detected Kaggle dataset paths
    ------------------------------------------------------------------------------------------------------------------------------------------------------
    dataset                                                    kaggle_handle                                                                                download_base                                                                                                                                                                                       detected_root                                  required_folders                         source_type
        DS1                orvile/pmram-bangladeshi-brain-cancer-mri-dataset /root/.cache/kagglehub/datasets/orvile/pmram-bangladeshi-brain-cancer-mri-dataset/versions/2 /root/.cache/kagglehub/datasets/orvile/pmram-bangladeshi-brain-cancer-mri-dataset/versions/2/PMRAM Bangladeshi Brain Cancer - MRI Dataset/PMRAM Bangladeshi Brain Cancer - MRI Dataset/Raw Data/Raw 512Glioma, 512Meningioma, 512Normal, 512Pituitary new_dataset_audit_from_kaggle_files
        DS2 yassinebazgour/preprocessed-brain-mri-scans-for-tumors-detection                              /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection                                                                                                      /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset            glioma, meningioma, notumor, pituitary new_dataset_audit_from_kaggle_files
    
    ======================================================================================================================================================
    2. BUILD MANIFESTS USING SAME MAIN-CODE LOGIC
    ======================================================================================================================================================
    
    ------------------------------------------------------------------------------------------------------------------------------------------------------
    Computed folder counts from downloaded Kaggle files
    ------------------------------------------------------------------------------------------------------------------------------------------------------
    dataset source_folder mapped_label  n_images                         source_type
        DS1     512Glioma       glioma       373 new_dataset_audit_from_kaggle_files
        DS1 512Meningioma   meningioma       363 new_dataset_audit_from_kaggle_files
        DS1     512Normal      notumor       396 new_dataset_audit_from_kaggle_files
        DS1  512Pituitary    pituitary       373 new_dataset_audit_from_kaggle_files
        DS2        glioma       glioma      1621 new_dataset_audit_from_kaggle_files
        DS2    meningioma   meningioma      1646 new_dataset_audit_from_kaggle_files
        DS2       notumor      notumor      2000 new_dataset_audit_from_kaggle_files
        DS2     pituitary    pituitary      1764 new_dataset_audit_from_kaggle_files
    
    ------------------------------------------------------------------------------------------------------------------------------------------------------
    Reference-vs-computed folder-count verification
    ------------------------------------------------------------------------------------------------------------------------------------------------------
    dataset source_folder mapped_label  main_saved_output_n_images  computed_kaggle_audit_n_images  match
        DS1     512Glioma       glioma                         373                             373   True
        DS1 512Meningioma   meningioma                         363                             363   True
        DS1     512Normal      notumor                         396                             396   True
        DS1  512Pituitary    pituitary                         373                             373   True
        DS2        glioma       glioma                        1621                            1621   True
        DS2    meningioma   meningioma                        1646                            1646   True
        DS2       notumor      notumor                        2000                            2000   True
        DS2     pituitary    pituitary                        1764                            1764   True
    
    ------------------------------------------------------------------------------------------------------------------------------------------------------
    Computed dataset totals from downloaded Kaggle files
    ------------------------------------------------------------------------------------------------------------------------------------------------------
    dataset  total_images  unique_paths  unique_filenames  glioma  meningioma  notumor  pituitary                         source_type
        DS1          1505          1505              1505     373         363      396        373 new_dataset_audit_from_kaggle_files
        DS2          7031          7031              7031    1621        1646     2000       1764 new_dataset_audit_from_kaggle_files
    
    ------------------------------------------------------------------------------------------------------------------------------------------------------
    Computed DS1 class counts
    ------------------------------------------------------------------------------------------------------------------------------------------------------
    dataset class_label  n_images  pct_within_dataset                         source_type
        DS1      glioma       373           24.784053 new_dataset_audit_from_kaggle_files
        DS1  meningioma       363           24.119601 new_dataset_audit_from_kaggle_files
        DS1     notumor       396           26.312292 new_dataset_audit_from_kaggle_files
        DS1   pituitary       373           24.784053 new_dataset_audit_from_kaggle_files
    
    ------------------------------------------------------------------------------------------------------------------------------------------------------
    Computed DS2 class counts
    ------------------------------------------------------------------------------------------------------------------------------------------------------
    dataset class_label  n_images  pct_within_dataset                         source_type
        DS2      glioma      1621           23.055042 new_dataset_audit_from_kaggle_files
        DS2  meningioma      1646           23.410610 new_dataset_audit_from_kaggle_files
        DS2     notumor      2000           28.445456 new_dataset_audit_from_kaggle_files
        DS2   pituitary      1764           25.088892 new_dataset_audit_from_kaggle_files
    
    ======================================================================================================================================================
    3.1 DEVELOPMENT DATASET SOURCES — PAPER-READY AUDIT
    ======================================================================================================================================================
    
    ------------------------------------------------------------------------------------------------------------------------------------------------------
    Dataset source information for Section 3.1
    ------------------------------------------------------------------------------------------------------------------------------------------------------
    dataset                             dataset_name_to_write                                     kaggle_handle_from_main_code                                 folder_names_used            dataset_state  kept_separate_before_splitting  total_images_main_saved_output                          source_type
        DS1        PMRAM Bangladeshi Brain Cancer MRI Dataset                orvile/pmram-bangladeshi-brain-cancer-mri-dataset 512Glioma, 512Meningioma, 512Normal, 512Pituitary          Raw MRI dataset                            True                            1505 main_code_config + main_saved_output
        DS2 Preprocessed Brain MRI Scans for Tumors Detection yassinebazgour/preprocessed-brain-mri-scans-for-tumors-detection            glioma, meningioma, notumor, pituitary Preprocessed MRI dataset                            True                            7031 main_code_config + main_saved_output
    
    ======================================================================================================================================================
    3.2 CLASS LABELS AND SOURCE-TO-TARGET LABEL MAPPING
    ======================================================================================================================================================
    
    ------------------------------------------------------------------------------------------------------------------------------------------------------
    Label mapping table
    ------------------------------------------------------------------------------------------------------------------------------------------------------
     label_id final_label DS1_source_folder DS2_source_folder      source_type
            0      glioma         512Glioma            glioma main_code_config
            1  meningioma     512Meningioma        meningioma main_code_config
            2     notumor         512Normal           notumor main_code_config
            3   pituitary      512Pituitary         pituitary main_code_config
    
    ------------------------------------------------------------------------------------------------------------------------------------------------------
    Label integrity check
    ------------------------------------------------------------------------------------------------------------------------------------------------------
    dataset  unique_labels                         labels_present integer_y_values  all_labels_match_target_set                         source_type
        DS1              4 glioma, meningioma, notumor, pituitary       0, 1, 2, 3                         True new_dataset_audit_from_kaggle_files
        DS2              4 glioma, meningioma, notumor, pituitary       0, 1, 2, 3                         True new_dataset_audit_from_kaggle_files
    
    ======================================================================================================================================================
    3.3 IMAGE INCLUSION, DUPLICATE HANDLING, LABEL HARMONIZATION, AND IMAGE SIZE AUDIT
    ======================================================================================================================================================
    
    ------------------------------------------------------------------------------------------------------------------------------------------------------
    Inclusion and harmonization summary
    ------------------------------------------------------------------------------------------------------------------------------------------------------
    dataset         supported_extensions_from_main_code  rows_after_label_harmonization  unique_paths_after_duplicate_path_removal  duplicate_path_rows_remaining  unique_filenames                                                              image_loading_rule_from_main_code model_input_resize_saved_run                                            source_type
        DS1 .jpg, .jpeg, .png, .bmp, .tif, .tiff, .webp                            1505                                       1505                              0              1505 Image.open(path).convert('RGB'); on exception returns gray RGB placeholder of model input size                      224x224 main_code_config + new_dataset_audit_from_kaggle_files
        DS2 .jpg, .jpeg, .png, .bmp, .tif, .tiff, .webp                            7031                                       7031                              0              7031 Image.open(path).convert('RGB'); on exception returns gray RGB placeholder of model input size                      224x224 main_code_config + new_dataset_audit_from_kaggle_files
    
    ------------------------------------------------------------------------------------------------------------------------------------------------------
    Included image extension counts
    ------------------------------------------------------------------------------------------------------------------------------------------------------
    dataset extension  n_images  pct_within_dataset                         source_type
        DS1       jpg      1505               100.0 new_dataset_audit_from_kaggle_files
        DS2       jpg      7031               100.0 new_dataset_audit_from_kaggle_files
    
    ------------------------------------------------------------------------------------------------------------------------------------------------------
    Original/native image size distribution by dataset
    ------------------------------------------------------------------------------------------------------------------------------------------------------
    dataset    size  n_images                         source_type
        DS1 512x512      1505 new_dataset_audit_from_kaggle_files
        DS2 224x224      7031 new_dataset_audit_from_kaggle_files
    
    ------------------------------------------------------------------------------------------------------------------------------------------------------
    Original/native image size distribution by dataset and class
    ------------------------------------------------------------------------------------------------------------------------------------------------------
    dataset      label    size  n_images                         source_type
        DS1     glioma 512x512       373 new_dataset_audit_from_kaggle_files
        DS1 meningioma 512x512       363 new_dataset_audit_from_kaggle_files
        DS1    notumor 512x512       396 new_dataset_audit_from_kaggle_files
        DS1  pituitary 512x512       373 new_dataset_audit_from_kaggle_files
        DS2     glioma 224x224      1621 new_dataset_audit_from_kaggle_files
        DS2 meningioma 224x224      1646 new_dataset_audit_from_kaggle_files
        DS2    notumor 224x224      2000 new_dataset_audit_from_kaggle_files
        DS2  pituitary 224x224      1764 new_dataset_audit_from_kaggle_files
    
    ------------------------------------------------------------------------------------------------------------------------------------------------------
    Original/native image size summary
    ------------------------------------------------------------------------------------------------------------------------------------------------------
    dataset  n_images  unique_sizes  min_width  max_width  min_height  max_height  mean_width  mean_height  median_width  median_height most_common_size  most_common_size_count                         source_type
        DS1      1505             1        512        512         512         512       512.0        512.0         512.0          512.0          512x512                    1505 new_dataset_audit_from_kaggle_files
        DS2      7031             1        224        224         224         224       224.0        224.0         224.0          224.0          224x224                    7031 new_dataset_audit_from_kaggle_files
    
    ------------------------------------------------------------------------------------------------------------------------------------------------------
    PIL image mode distribution
    ------------------------------------------------------------------------------------------------------------------------------------------------------
    dataset mode  n_images                         source_type
        DS1  RGB      1505 new_dataset_audit_from_kaggle_files
        DS2    L      7031 new_dataset_audit_from_kaggle_files
    
    ------------------------------------------------------------------------------------------------------------------------------------------------------
    PIL image format distribution
    ------------------------------------------------------------------------------------------------------------------------------------------------------
    dataset format  n_images                         source_type
        DS1   JPEG      1505 new_dataset_audit_from_kaggle_files
        DS2   JPEG      7031 new_dataset_audit_from_kaggle_files
    
    ------------------------------------------------------------------------------------------------------------------------------------------------------
    Unreadable image / size-read failures
    ------------------------------------------------------------------------------------------------------------------------------------------------------
    dataset  unreadable_images_detected                                                 note                         source_type
        ALL                           0 All images could be opened by PIL during this audit. new_dataset_audit_from_kaggle_files
    
    ------------------------------------------------------------------------------------------------------------------------------------------------------
    Duplicate filename summary
    ------------------------------------------------------------------------------------------------------------------------------------------------------
    dataset  duplicate_filename_values  images_with_duplicate_filenames                         source_type
        DS1                          0                                0 new_dataset_audit_from_kaggle_files
        DS2                          0                                0 new_dataset_audit_from_kaggle_files
    
    ======================================================================================================================================================
    FULL MD5 DUPLICATE AUDIT FROM DOWNLOADED KAGGLE FILES
    ======================================================================================================================================================
    Computing full MD5 hashes. This reads image bytes only; it does not train anything.
    
    ------------------------------------------------------------------------------------------------------------------------------------------------------
    Full MD5 uniqueness summary
    ------------------------------------------------------------------------------------------------------------------------------------------------------
    dataset  total_images  unique_md5                         source_type
        DS1          1505        1410 new_dataset_audit_from_kaggle_files
        DS2          7031        6596 new_dataset_audit_from_kaggle_files
    
    ------------------------------------------------------------------------------------------------------------------------------------------------------
    Full MD5 duplicate summary
    ------------------------------------------------------------------------------------------------------------------------------------------------------
       dataset  duplicate_md5_hashes_within_dataset  images_in_duplicate_md5_groups                         source_type  duplicate_md5_hashes_between_datasets  images_in_cross_dataset_duplicate_groups
           DS1                                 89.0                           184.0 new_dataset_audit_from_kaggle_files                                    NaN                                       NaN
           DS2                                308.0                           743.0 new_dataset_audit_from_kaggle_files                                    NaN                                       NaN
    DS1_vs_DS2                                  NaN                             NaN new_dataset_audit_from_kaggle_files                                    0.0                                       0.0
    
    ------------------------------------------------------------------------------------------------------------------------------------------------------
    Full MD5 duplicate examples
    ------------------------------------------------------------------------------------------------------------------------------------------------------
    dataset      label           filename                              md5                                                                                                                                                                                                                           path                         source_type
        DS2    notumor     Te-no_0037.jpg 00cffda2923013b46bc363cc3d66cd98                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Te-no_0037.jpg new_dataset_audit_from_kaggle_files
        DS2    notumor     Tr-no_0113.jpg 00cffda2923013b46bc363cc3d66cd98                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Tr-no_0113.jpg new_dataset_audit_from_kaggle_files
        DS2    notumor     Tr-no_1093.jpg 00cffda2923013b46bc363cc3d66cd98                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Tr-no_1093.jpg new_dataset_audit_from_kaggle_files
        DS2    notumor     Te-no_0047.jpg 027c073e335cf85dac60708a08000d35                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Te-no_0047.jpg new_dataset_audit_from_kaggle_files
        DS2    notumor     Te-no_0077.jpg 027c073e335cf85dac60708a08000d35                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Te-no_0077.jpg new_dataset_audit_from_kaggle_files
        DS2    notumor     Tr-no_0213.jpg 027c073e335cf85dac60708a08000d35                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Tr-no_0213.jpg new_dataset_audit_from_kaggle_files
        DS2    notumor     Tr-no_0341.jpg 027c073e335cf85dac60708a08000d35                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Tr-no_0341.jpg new_dataset_audit_from_kaggle_files
        DS2    notumor     Tr-no_1181.jpg 027c073e335cf85dac60708a08000d35                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Tr-no_1181.jpg new_dataset_audit_from_kaggle_files
        DS2    notumor     Te-no_0103.jpg 0328f727dbc09a0c7db6ef85b3208dfe                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Te-no_0103.jpg new_dataset_audit_from_kaggle_files
        DS2    notumor     Tr-no_0319.jpg 0328f727dbc09a0c7db6ef85b3208dfe                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Tr-no_0319.jpg new_dataset_audit_from_kaggle_files
        DS2    notumor     Tr-no_1171.jpg 0328f727dbc09a0c7db6ef85b3208dfe                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Tr-no_1171.jpg new_dataset_audit_from_kaggle_files
        DS2    notumor     Tr-no_0238.jpg 0566d92ecb85ee8a289482d48f946e4d                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Tr-no_0238.jpg new_dataset_audit_from_kaggle_files
        DS2    notumor     Tr-no_0267.jpg 0566d92ecb85ee8a289482d48f946e4d                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Tr-no_0267.jpg new_dataset_audit_from_kaggle_files
        DS2    notumor     Tr-no_1146.jpg 0566d92ecb85ee8a289482d48f946e4d                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Tr-no_1146.jpg new_dataset_audit_from_kaggle_files
        DS2    notumor     Te-no_0063.jpg 0576fd2efac8259a32b618c77d643470                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Te-no_0063.jpg new_dataset_audit_from_kaggle_files
        DS2    notumor     Tr-no_0326.jpg 0576fd2efac8259a32b618c77d643470                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Tr-no_0326.jpg new_dataset_audit_from_kaggle_files
        DS2    notumor     Tr-no_1174.jpg 0576fd2efac8259a32b618c77d643470                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Tr-no_1174.jpg new_dataset_audit_from_kaggle_files
        DS2    notumor     Tr-no_0118.jpg 0b6f4812df9ed1e21067424c3e7bb73b                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Tr-no_0118.jpg new_dataset_audit_from_kaggle_files
        DS2    notumor     Tr-no_0123.jpg 0b6f4812df9ed1e21067424c3e7bb73b                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Tr-no_0123.jpg new_dataset_audit_from_kaggle_files
        DS2    notumor     Tr-no_0136.jpg 0b6f4812df9ed1e21067424c3e7bb73b                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Tr-no_0136.jpg new_dataset_audit_from_kaggle_files
        DS2    notumor     Te-no_0012.jpg 11241b33d38b730c2ae2d3a558b0076f                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Te-no_0012.jpg new_dataset_audit_from_kaggle_files
        DS2    notumor     Te-no_0044.jpg 11241b33d38b730c2ae2d3a558b0076f                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Te-no_0044.jpg new_dataset_audit_from_kaggle_files
        DS2    notumor     Te-no_0055.jpg 11241b33d38b730c2ae2d3a558b0076f                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Te-no_0055.jpg new_dataset_audit_from_kaggle_files
        DS2    notumor     Te-no_0061.jpg 11241b33d38b730c2ae2d3a558b0076f                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Te-no_0061.jpg new_dataset_audit_from_kaggle_files
        DS2    notumor     Tr-no_0057.jpg 11241b33d38b730c2ae2d3a558b0076f                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Tr-no_0057.jpg new_dataset_audit_from_kaggle_files
        DS2    notumor     Tr-no_0180.jpg 11241b33d38b730c2ae2d3a558b0076f                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Tr-no_0180.jpg new_dataset_audit_from_kaggle_files
        DS2    notumor     Tr-no_0291.jpg 11241b33d38b730c2ae2d3a558b0076f                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Tr-no_0291.jpg new_dataset_audit_from_kaggle_files
        DS2    notumor     Tr-no_0324.jpg 11241b33d38b730c2ae2d3a558b0076f                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Tr-no_0324.jpg new_dataset_audit_from_kaggle_files
        DS2    notumor     Tr-no_0984.jpg 11241b33d38b730c2ae2d3a558b0076f                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Tr-no_0984.jpg new_dataset_audit_from_kaggle_files
        DS2    notumor     Te-no_0045.jpg 1618568d42bc8954dc6313cb9d9882b3                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Te-no_0045.jpg new_dataset_audit_from_kaggle_files
        DS2    notumor     Te-no_0050.jpg 1618568d42bc8954dc6313cb9d9882b3                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Te-no_0050.jpg new_dataset_audit_from_kaggle_files
        DS2    notumor     Te-no_0054.jpg 1618568d42bc8954dc6313cb9d9882b3                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Te-no_0054.jpg new_dataset_audit_from_kaggle_files
        DS2    notumor     Tr-no_0191.jpg 1618568d42bc8954dc6313cb9d9882b3                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Tr-no_0191.jpg new_dataset_audit_from_kaggle_files
        DS2    notumor     Tr-no_0235.jpg 1618568d42bc8954dc6313cb9d9882b3                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Tr-no_0235.jpg new_dataset_audit_from_kaggle_files
        DS2    notumor     Tr-no_0279.jpg 1618568d42bc8954dc6313cb9d9882b3                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Tr-no_0279.jpg new_dataset_audit_from_kaggle_files
        DS2    notumor     Tr-no_0234.jpg 17461c88b4612f658fb2f5ba8446bdb0                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Tr-no_0234.jpg new_dataset_audit_from_kaggle_files
        DS2    notumor     Tr-no_0239.jpg 17461c88b4612f658fb2f5ba8446bdb0                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Tr-no_0239.jpg new_dataset_audit_from_kaggle_files
        DS2    notumor     Tr-no_1135.jpg 17461c88b4612f658fb2f5ba8446bdb0                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Tr-no_1135.jpg new_dataset_audit_from_kaggle_files
        DS2    notumor     Te-no_0100.jpg 1916be25961a88a5885d555a15882ed6                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Te-no_0100.jpg new_dataset_audit_from_kaggle_files
        DS2    notumor     Tr-no_0364.jpg 1916be25961a88a5885d555a15882ed6                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Tr-no_0364.jpg new_dataset_audit_from_kaggle_files
        DS2    notumor     Tr-no_0979.jpg 1916be25961a88a5885d555a15882ed6                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Tr-no_0979.jpg new_dataset_audit_from_kaggle_files
        DS2    notumor     Te-no_0089.jpg 194ccd5eb602a26d249e2f619468b292                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Te-no_0089.jpg new_dataset_audit_from_kaggle_files
        DS2    notumor     Tr-no_0353.jpg 194ccd5eb602a26d249e2f619468b292                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Tr-no_0353.jpg new_dataset_audit_from_kaggle_files
        DS2    notumor     Tr-no_0969.jpg 194ccd5eb602a26d249e2f619468b292                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Tr-no_0969.jpg new_dataset_audit_from_kaggle_files
        DS1    notumor     normal (1).jpg 1e2fd8da45a9cc434c617ae227ac9e27   /root/.cache/kagglehub/datasets/orvile/pmram-bangladeshi-brain-cancer-mri-dataset/versions/2/PMRAM Bangladeshi Brain Cancer - MRI Dataset/PMRAM Bangladeshi Brain Cancer - MRI Dataset/Raw Data/Raw/512Normal/normal (1).jpg new_dataset_audit_from_kaggle_files
        DS1    notumor   normal (197).jpg 1e2fd8da45a9cc434c617ae227ac9e27 /root/.cache/kagglehub/datasets/orvile/pmram-bangladeshi-brain-cancer-mri-dataset/versions/2/PMRAM Bangladeshi Brain Cancer - MRI Dataset/PMRAM Bangladeshi Brain Cancer - MRI Dataset/Raw Data/Raw/512Normal/normal (197).jpg new_dataset_audit_from_kaggle_files
        DS1    notumor    normal (22).jpg 1e2fd8da45a9cc434c617ae227ac9e27  /root/.cache/kagglehub/datasets/orvile/pmram-bangladeshi-brain-cancer-mri-dataset/versions/2/PMRAM Bangladeshi Brain Cancer - MRI Dataset/PMRAM Bangladeshi Brain Cancer - MRI Dataset/Raw Data/Raw/512Normal/normal (22).jpg new_dataset_audit_from_kaggle_files
        DS2    notumor     Te-no_0011.jpg 20102a69e19927570da93f7e67117f13                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Te-no_0011.jpg new_dataset_audit_from_kaggle_files
        DS2    notumor     Te-no_0048.jpg 20102a69e19927570da93f7e67117f13                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Te-no_0048.jpg new_dataset_audit_from_kaggle_files
        DS2    notumor     Te-no_0053.jpg 20102a69e19927570da93f7e67117f13                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Te-no_0053.jpg new_dataset_audit_from_kaggle_files
        DS2    notumor     Te-no_0068.jpg 20102a69e19927570da93f7e67117f13                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Te-no_0068.jpg new_dataset_audit_from_kaggle_files
        DS2    notumor     Te-no_0087.jpg 20102a69e19927570da93f7e67117f13                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Te-no_0087.jpg new_dataset_audit_from_kaggle_files
        DS2    notumor     Tr-no_0056.jpg 20102a69e19927570da93f7e67117f13                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Tr-no_0056.jpg new_dataset_audit_from_kaggle_files
        DS2    notumor     Tr-no_0224.jpg 20102a69e19927570da93f7e67117f13                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Tr-no_0224.jpg new_dataset_audit_from_kaggle_files
        DS2    notumor     Tr-no_0268.jpg 20102a69e19927570da93f7e67117f13                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Tr-no_0268.jpg new_dataset_audit_from_kaggle_files
        DS2    notumor     Tr-no_0332.jpg 20102a69e19927570da93f7e67117f13                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Tr-no_0332.jpg new_dataset_audit_from_kaggle_files
        DS2    notumor     Tr-no_0350.jpg 20102a69e19927570da93f7e67117f13                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Tr-no_0350.jpg new_dataset_audit_from_kaggle_files
        DS2    notumor     Tr-no_0982.jpg 20102a69e19927570da93f7e67117f13                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Tr-no_0982.jpg new_dataset_audit_from_kaggle_files
        DS2  pituitary Tr-pi_0046 (1).jpg 217580974402e32239d8f9d6d0a39234                                                                                                    /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/pituitary/Tr-pi_0046 (1).jpg new_dataset_audit_from_kaggle_files
        DS2  pituitary     Tr-pi_0046.jpg 217580974402e32239d8f9d6d0a39234                                                                                                        /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/pituitary/Tr-pi_0046.jpg new_dataset_audit_from_kaggle_files
        DS2  pituitary     Tr-pi_0157.jpg 217580974402e32239d8f9d6d0a39234                                                                                                        /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/pituitary/Tr-pi_0157.jpg new_dataset_audit_from_kaggle_files
        DS2    notumor     Tr-no_0140.jpg 2dfa6e329951bec1ac0c7e91f97feb6c                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Tr-no_0140.jpg new_dataset_audit_from_kaggle_files
        DS2    notumor     Tr-no_0223.jpg 2dfa6e329951bec1ac0c7e91f97feb6c                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Tr-no_0223.jpg new_dataset_audit_from_kaggle_files
        DS2    notumor     Tr-no_1101.jpg 2dfa6e329951bec1ac0c7e91f97feb6c                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Tr-no_1101.jpg new_dataset_audit_from_kaggle_files
        DS2    notumor   Te-noTr_0007.jpg 3249b0c9246ab9b0f7dd898097615281                                                                                                        /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Te-noTr_0007.jpg new_dataset_audit_from_kaggle_files
        DS2    notumor     Te-no_0088.jpg 3249b0c9246ab9b0f7dd898097615281                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Te-no_0088.jpg new_dataset_audit_from_kaggle_files
        DS2    notumor     Tr-no_0052.jpg 3249b0c9246ab9b0f7dd898097615281                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Tr-no_0052.jpg new_dataset_audit_from_kaggle_files
        DS2    notumor     Tr-no_0352.jpg 3249b0c9246ab9b0f7dd898097615281                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Tr-no_0352.jpg new_dataset_audit_from_kaggle_files
        DS2    notumor     Tr-no_1089.jpg 3249b0c9246ab9b0f7dd898097615281                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Tr-no_1089.jpg new_dataset_audit_from_kaggle_files
        DS2 meningioma     Te-me_0052.jpg 325d2be8d11d1ddc9bbb7573eb559668                                                                                                       /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/meningioma/Te-me_0052.jpg new_dataset_audit_from_kaggle_files
        DS2 meningioma     Tr-me_0231.jpg 325d2be8d11d1ddc9bbb7573eb559668                                                                                                       /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/meningioma/Tr-me_0231.jpg new_dataset_audit_from_kaggle_files
        DS1    notumor    normal (14).jpg 332baa8d1fb21c1ceec02e65ad46e3e3  /root/.cache/kagglehub/datasets/orvile/pmram-bangladeshi-brain-cancer-mri-dataset/versions/2/PMRAM Bangladeshi Brain Cancer - MRI Dataset/PMRAM Bangladeshi Brain Cancer - MRI Dataset/Raw Data/Raw/512Normal/normal (14).jpg new_dataset_audit_from_kaggle_files
        DS1    notumor   normal (217).jpg 332baa8d1fb21c1ceec02e65ad46e3e3 /root/.cache/kagglehub/datasets/orvile/pmram-bangladeshi-brain-cancer-mri-dataset/versions/2/PMRAM Bangladeshi Brain Cancer - MRI Dataset/PMRAM Bangladeshi Brain Cancer - MRI Dataset/Raw Data/Raw/512Normal/normal (217).jpg new_dataset_audit_from_kaggle_files
        DS2    notumor     Te-no_0018.jpg 35524e36a91606f5e476efc87ac3b785                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Te-no_0018.jpg new_dataset_audit_from_kaggle_files
        DS2    notumor     Te-no_0070.jpg 35524e36a91606f5e476efc87ac3b785                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Te-no_0070.jpg new_dataset_audit_from_kaggle_files
        DS2    notumor     Tr-no_0063.jpg 35524e36a91606f5e476efc87ac3b785                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Tr-no_0063.jpg new_dataset_audit_from_kaggle_files
        DS2    notumor     Tr-no_0334.jpg 35524e36a91606f5e476efc87ac3b785                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Tr-no_0334.jpg new_dataset_audit_from_kaggle_files
        DS2    notumor     Tr-no_0974.jpg 35524e36a91606f5e476efc87ac3b785                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Tr-no_0974.jpg new_dataset_audit_from_kaggle_files
        DS2    notumor     Te-no_0039.jpg 35af0dced93b6e1c83a7b5f480e027cf                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Te-no_0039.jpg new_dataset_audit_from_kaggle_files
        DS2    notumor     Tr-no_0124.jpg 35af0dced93b6e1c83a7b5f480e027cf                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Tr-no_0124.jpg new_dataset_audit_from_kaggle_files
        DS2    notumor     Tr-no_1094.jpg 35af0dced93b6e1c83a7b5f480e027cf                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Tr-no_1094.jpg new_dataset_audit_from_kaggle_files
        DS2 meningioma   Te-meTr_0009.jpg 36011557cc4888d46881b6fb8829a1b7                                                                                                     /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/meningioma/Te-meTr_0009.jpg new_dataset_audit_from_kaggle_files
        DS2 meningioma     Tr-me_0364.jpg 36011557cc4888d46881b6fb8829a1b7                                                                                                       /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/meningioma/Tr-me_0364.jpg new_dataset_audit_from_kaggle_files
        DS2    notumor     Te-no_0023.jpg 3601a775241f2f3c927d41b176d82490                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Te-no_0023.jpg new_dataset_audit_from_kaggle_files
        DS2    notumor     Tr-no_0179.jpg 3601a775241f2f3c927d41b176d82490                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Tr-no_0179.jpg new_dataset_audit_from_kaggle_files
        DS2    notumor     Tr-no_1141.jpg 3601a775241f2f3c927d41b176d82490                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Tr-no_1141.jpg new_dataset_audit_from_kaggle_files
        DS2    notumor     Te-no_0042.jpg 36c29079ece87eba87b5079e2c1f1030                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Te-no_0042.jpg new_dataset_audit_from_kaggle_files
        DS2    notumor     Tr-no_0157.jpg 36c29079ece87eba87b5079e2c1f1030                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Tr-no_0157.jpg new_dataset_audit_from_kaggle_files
        DS2    notumor     Tr-no_1110.jpg 36c29079ece87eba87b5079e2c1f1030                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Tr-no_1110.jpg new_dataset_audit_from_kaggle_files
        DS1    notumor   normal (286).jpg 3af353cf97090c4bef8e4434a6af77cd /root/.cache/kagglehub/datasets/orvile/pmram-bangladeshi-brain-cancer-mri-dataset/versions/2/PMRAM Bangladeshi Brain Cancer - MRI Dataset/PMRAM Bangladeshi Brain Cancer - MRI Dataset/Raw Data/Raw/512Normal/normal (286).jpg new_dataset_audit_from_kaggle_files
        DS1    notumor   normal (295).jpg 3af353cf97090c4bef8e4434a6af77cd /root/.cache/kagglehub/datasets/orvile/pmram-bangladeshi-brain-cancer-mri-dataset/versions/2/PMRAM Bangladeshi Brain Cancer - MRI Dataset/PMRAM Bangladeshi Brain Cancer - MRI Dataset/Raw Data/Raw/512Normal/normal (295).jpg new_dataset_audit_from_kaggle_files
        DS2  pituitary   Te-piTr_0001.jpg 3ed0ad1dc8a941348d79b9c0d2502cd2                                                                                                      /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/pituitary/Te-piTr_0001.jpg new_dataset_audit_from_kaggle_files
        DS2  pituitary     Tr-pi_0516.jpg 3ed0ad1dc8a941348d79b9c0d2502cd2                                                                                                        /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/pituitary/Tr-pi_0516.jpg new_dataset_audit_from_kaggle_files
        DS2  pituitary     Tr-pi_0517.jpg 3ed0ad1dc8a941348d79b9c0d2502cd2                                                                                                        /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/pituitary/Tr-pi_0517.jpg new_dataset_audit_from_kaggle_files
        DS2  pituitary     Tr-pi_0518.jpg 3ed0ad1dc8a941348d79b9c0d2502cd2                                                                                                        /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/pituitary/Tr-pi_0518.jpg new_dataset_audit_from_kaggle_files
        DS2    notumor     Te-no_0066.jpg 4200b4dba065c27da813a0d468e92504                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Te-no_0066.jpg new_dataset_audit_from_kaggle_files
        DS2    notumor     Tr-no_0330.jpg 4200b4dba065c27da813a0d468e92504                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Tr-no_0330.jpg new_dataset_audit_from_kaggle_files
        DS2    notumor     Tr-no_0971.jpg 4200b4dba065c27da813a0d468e92504                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Tr-no_0971.jpg new_dataset_audit_from_kaggle_files
        DS2    notumor     Te-no_0030.jpg 4550856b1cb1668dcfccf8515deabd2e                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Te-no_0030.jpg new_dataset_audit_from_kaggle_files
        DS2    notumor     Te-no_0065.jpg 4550856b1cb1668dcfccf8515deabd2e                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Te-no_0065.jpg new_dataset_audit_from_kaggle_files
        DS2    notumor     Tr-no_0328.jpg 4550856b1cb1668dcfccf8515deabd2e                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Tr-no_0328.jpg new_dataset_audit_from_kaggle_files
        DS2    notumor     Tr-no_0362.jpg 4550856b1cb1668dcfccf8515deabd2e                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Tr-no_0362.jpg new_dataset_audit_from_kaggle_files
        DS2    notumor     Tr-no_0977.jpg 4550856b1cb1668dcfccf8515deabd2e                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Tr-no_0977.jpg new_dataset_audit_from_kaggle_files
        DS2    notumor     Tr-no_0195.jpg 47a77b247dc0306c9a9c4c4491f9e127                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Tr-no_0195.jpg new_dataset_audit_from_kaggle_files
        DS2    notumor     Tr-no_0229.jpg 47a77b247dc0306c9a9c4c4491f9e127                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Tr-no_0229.jpg new_dataset_audit_from_kaggle_files
        DS2    notumor     Tr-no_1130.jpg 47a77b247dc0306c9a9c4c4491f9e127                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Tr-no_1130.jpg new_dataset_audit_from_kaggle_files
        DS2 meningioma     Te-me_0106.jpg 4e8aac352b46fa2921dcb14e3ead3d57                                                                                                       /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/meningioma/Te-me_0106.jpg new_dataset_audit_from_kaggle_files
        DS2 meningioma     Te-me_0143.jpg 4e8aac352b46fa2921dcb14e3ead3d57                                                                                                       /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/meningioma/Te-me_0143.jpg new_dataset_audit_from_kaggle_files
        DS2 meningioma     Tr-me_0283.jpg 4e8aac352b46fa2921dcb14e3ead3d57                                                                                                       /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/meningioma/Tr-me_0283.jpg new_dataset_audit_from_kaggle_files
        DS2    notumor     Te-no_0016.jpg 551eba92bdb3951cd186ed0ad8dbfcad                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Te-no_0016.jpg new_dataset_audit_from_kaggle_files
        DS2    notumor     Tr-no_0042.jpg 551eba92bdb3951cd186ed0ad8dbfcad                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Tr-no_0042.jpg new_dataset_audit_from_kaggle_files
        DS2    notumor     Tr-no_1147.jpg 551eba92bdb3951cd186ed0ad8dbfcad                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Tr-no_1147.jpg new_dataset_audit_from_kaggle_files
        DS2    notumor     Te-no_0036.jpg 57f26bc27df568af94c3f183f596180a                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Te-no_0036.jpg new_dataset_audit_from_kaggle_files
        DS2    notumor     Tr-no_0102.jpg 57f26bc27df568af94c3f183f596180a                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Tr-no_0102.jpg new_dataset_audit_from_kaggle_files
        DS2    notumor     Tr-no_0968.jpg 57f26bc27df568af94c3f183f596180a                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Tr-no_0968.jpg new_dataset_audit_from_kaggle_files
        DS2    notumor     Te-no_0060.jpg 6066da064cf0ce60cd44b632a8869fd0                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Te-no_0060.jpg new_dataset_audit_from_kaggle_files
        DS2    notumor     Tr-no_0046.jpg 6066da064cf0ce60cd44b632a8869fd0                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Tr-no_0046.jpg new_dataset_audit_from_kaggle_files
        DS2    notumor     Tr-no_0967.jpg 6066da064cf0ce60cd44b632a8869fd0                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Tr-no_0967.jpg new_dataset_audit_from_kaggle_files
        DS2 meningioma     Te-me_0086.jpg 6507ff46332f19db4edfb07b3678b123                                                                                                       /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/meningioma/Te-me_0086.jpg new_dataset_audit_from_kaggle_files
        DS2 meningioma     Tr-me_0263.jpg 6507ff46332f19db4edfb07b3678b123                                                                                                       /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/meningioma/Tr-me_0263.jpg new_dataset_audit_from_kaggle_files
        DS2    notumor   Te-noTr_0008.jpg 6c28f9df021a7d691150ad1218f7c9e4                                                                                                        /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Te-noTr_0008.jpg new_dataset_audit_from_kaggle_files
        DS2    notumor     Tr-no_0053.jpg 6c28f9df021a7d691150ad1218f7c9e4                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Tr-no_0053.jpg new_dataset_audit_from_kaggle_files
        DS2    notumor     Tr-no_1090.jpg 6c28f9df021a7d691150ad1218f7c9e4                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Tr-no_1090.jpg new_dataset_audit_from_kaggle_files
        DS2    notumor     Tr-no_0197.jpg 6c33e7c78e263e0fc14fc96867aa1f49                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Tr-no_0197.jpg new_dataset_audit_from_kaggle_files
        DS2    notumor     Tr-no_0276.jpg 6c33e7c78e263e0fc14fc96867aa1f49                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Tr-no_0276.jpg new_dataset_audit_from_kaggle_files
        DS2    notumor     Tr-no_1151.jpg 6c33e7c78e263e0fc14fc96867aa1f49                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Tr-no_1151.jpg new_dataset_audit_from_kaggle_files
        DS2    notumor     Te-no_0035.jpg 6f06a50a935cd27ba58f618729cf261c                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Te-no_0035.jpg new_dataset_audit_from_kaggle_files
        DS2    notumor     Te-no_0081.jpg 6f06a50a935cd27ba58f618729cf261c                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Te-no_0081.jpg new_dataset_audit_from_kaggle_files
        DS2    notumor     Tr-no_0091.jpg 6f06a50a935cd27ba58f618729cf261c                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Tr-no_0091.jpg new_dataset_audit_from_kaggle_files
        DS2    notumor     Tr-no_0345.jpg 6f06a50a935cd27ba58f618729cf261c                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Tr-no_0345.jpg new_dataset_audit_from_kaggle_files
        DS2    notumor     Tr-no_0985.jpg 6f06a50a935cd27ba58f618729cf261c                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Tr-no_0985.jpg new_dataset_audit_from_kaggle_files
        DS2    notumor     Tr-no_0120.jpg 788dfc769b6f8390eacb88277de5cdb5                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Tr-no_0120.jpg new_dataset_audit_from_kaggle_files
        DS2    notumor     Tr-no_0125.jpg 788dfc769b6f8390eacb88277de5cdb5                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Tr-no_0125.jpg new_dataset_audit_from_kaggle_files
        DS2    notumor     Tr-no_0365.jpg 788dfc769b6f8390eacb88277de5cdb5                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Tr-no_0365.jpg new_dataset_audit_from_kaggle_files
        DS2    notumor     Tr-no_0368.jpg 788dfc769b6f8390eacb88277de5cdb5                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Tr-no_0368.jpg new_dataset_audit_from_kaggle_files
        DS1 meningioma          M_146.jpg 78b8cb0306c8c7c51d84cb8215388425    /root/.cache/kagglehub/datasets/orvile/pmram-bangladeshi-brain-cancer-mri-dataset/versions/2/PMRAM Bangladeshi Brain Cancer - MRI Dataset/PMRAM Bangladeshi Brain Cancer - MRI Dataset/Raw Data/Raw/512Meningioma/M_146.jpg new_dataset_audit_from_kaggle_files
        DS1 meningioma          M_162.jpg 78b8cb0306c8c7c51d84cb8215388425    /root/.cache/kagglehub/datasets/orvile/pmram-bangladeshi-brain-cancer-mri-dataset/versions/2/PMRAM Bangladeshi Brain Cancer - MRI Dataset/PMRAM Bangladeshi Brain Cancer - MRI Dataset/Raw Data/Raw/512Meningioma/M_162.jpg new_dataset_audit_from_kaggle_files
        DS2    notumor     Te-no_0021.jpg 7d37eea80e3bbd565e968eb6fef55450                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Te-no_0021.jpg new_dataset_audit_from_kaggle_files
        DS2    notumor     Te-no_0072.jpg 7d37eea80e3bbd565e968eb6fef55450                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Te-no_0072.jpg new_dataset_audit_from_kaggle_files
        DS2    notumor     Tr-no_0335.jpg 7d37eea80e3bbd565e968eb6fef55450                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Tr-no_0335.jpg new_dataset_audit_from_kaggle_files
        DS2    notumor     Tr-no_0394.jpg 7d37eea80e3bbd565e968eb6fef55450                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Tr-no_0394.jpg new_dataset_audit_from_kaggle_files
        DS2    notumor     Tr-no_0978.jpg 7d37eea80e3bbd565e968eb6fef55450                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Tr-no_0978.jpg new_dataset_audit_from_kaggle_files
        DS2    notumor     Tr-no_0150.jpg 7e3c5058f9bab77e3e28d145161e3144                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Tr-no_0150.jpg new_dataset_audit_from_kaggle_files
        DS2    notumor     Tr-no_0164.jpg 7e3c5058f9bab77e3e28d145161e3144                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Tr-no_0164.jpg new_dataset_audit_from_kaggle_files
        DS2    notumor     Tr-no_0242.jpg 7e3c5058f9bab77e3e28d145161e3144                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Tr-no_0242.jpg new_dataset_audit_from_kaggle_files
        DS2    notumor     Tr-no_1117.jpg 7e3c5058f9bab77e3e28d145161e3144                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Tr-no_1117.jpg new_dataset_audit_from_kaggle_files
        DS2    notumor     Tr-no_0142.jpg 81372ac4d397c006a9c3438ce1cdf5fa                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Tr-no_0142.jpg new_dataset_audit_from_kaggle_files
        DS2    notumor     Tr-no_0143.jpg 81372ac4d397c006a9c3438ce1cdf5fa                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Tr-no_0143.jpg new_dataset_audit_from_kaggle_files
        DS2    notumor     Tr-no_0149.jpg 81372ac4d397c006a9c3438ce1cdf5fa                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Tr-no_0149.jpg new_dataset_audit_from_kaggle_files
        DS2    notumor     Te-no_0032.jpg 81e47d9416bf2e78f2245479f9d22642                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Te-no_0032.jpg new_dataset_audit_from_kaggle_files
        DS2    notumor     Tr-no_0384.jpg 81e47d9416bf2e78f2245479f9d22642                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Tr-no_0384.jpg new_dataset_audit_from_kaggle_files
        DS2    notumor     Tr-no_0989.jpg 81e47d9416bf2e78f2245479f9d22642                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Tr-no_0989.jpg new_dataset_audit_from_kaggle_files
        DS2    notumor   Te-noTr_0001.jpg 8db27552e259ba3274490ee4a8a7f583                                                                                                        /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Te-noTr_0001.jpg new_dataset_audit_from_kaggle_files
        DS2    notumor     Te-no_0031.jpg 8db27552e259ba3274490ee4a8a7f583                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Te-no_0031.jpg new_dataset_audit_from_kaggle_files
        DS2    notumor     Te-no_0067.jpg 8db27552e259ba3274490ee4a8a7f583                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Te-no_0067.jpg new_dataset_audit_from_kaggle_files
        DS2    notumor     Te-no_0084.jpg 8db27552e259ba3274490ee4a8a7f583                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Te-no_0084.jpg new_dataset_audit_from_kaggle_files
        DS2    notumor     Tr-no_0051.jpg 8db27552e259ba3274490ee4a8a7f583                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Tr-no_0051.jpg new_dataset_audit_from_kaggle_files
        DS2    notumor     Tr-no_0331.jpg 8db27552e259ba3274490ee4a8a7f583                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Tr-no_0331.jpg new_dataset_audit_from_kaggle_files
        DS2    notumor     Tr-no_0347.jpg 8db27552e259ba3274490ee4a8a7f583                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Tr-no_0347.jpg new_dataset_audit_from_kaggle_files
        DS2    notumor     Tr-no_0373.jpg 8db27552e259ba3274490ee4a8a7f583                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Tr-no_0373.jpg new_dataset_audit_from_kaggle_files
        DS2    notumor     Tr-no_1088.jpg 8db27552e259ba3274490ee4a8a7f583                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Tr-no_1088.jpg new_dataset_audit_from_kaggle_files
        DS2  pituitary     Tr-pi_0526.jpg 9024bacc218fc0e4d4d8cbe5935ea21c                                                                                                        /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/pituitary/Tr-pi_0526.jpg new_dataset_audit_from_kaggle_files
        DS2  pituitary     Tr-pi_0527.jpg 9024bacc218fc0e4d4d8cbe5935ea21c                                                                                                        /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/pituitary/Tr-pi_0527.jpg new_dataset_audit_from_kaggle_files
        DS2  pituitary     Tr-pi_0529.jpg 9024bacc218fc0e4d4d8cbe5935ea21c                                                                                                        /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/pituitary/Tr-pi_0529.jpg new_dataset_audit_from_kaggle_files
        DS2 meningioma     Te-me_0104.jpg 95e067d47203c24f76b2f1c947522b11                                                                                                       /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/meningioma/Te-me_0104.jpg new_dataset_audit_from_kaggle_files
        DS2 meningioma     Tr-me_0175.jpg 95e067d47203c24f76b2f1c947522b11                                                                                                       /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/meningioma/Tr-me_0175.jpg new_dataset_audit_from_kaggle_files
        DS2 meningioma     Tr-me_0281.jpg 95e067d47203c24f76b2f1c947522b11                                                                                                       /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/meningioma/Tr-me_0281.jpg new_dataset_audit_from_kaggle_files
        DS2    notumor     Te-no_0014.jpg 98c9be6cc4e9ab7ec0e714095723b40d                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Te-no_0014.jpg new_dataset_audit_from_kaggle_files
        DS2    notumor     Tr-no_0059.jpg 98c9be6cc4e9ab7ec0e714095723b40d                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Tr-no_0059.jpg new_dataset_audit_from_kaggle_files
        DS2    notumor     Tr-no_1173.jpg 98c9be6cc4e9ab7ec0e714095723b40d                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Tr-no_1173.jpg new_dataset_audit_from_kaggle_files
        DS2    notumor     Te-no_0064.jpg 99df150eed69030bbc53dba523d748c1                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Te-no_0064.jpg new_dataset_audit_from_kaggle_files
        DS2    notumor     Tr-no_0327.jpg 99df150eed69030bbc53dba523d748c1                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Tr-no_0327.jpg new_dataset_audit_from_kaggle_files
        DS2    notumor     Tr-no_0970.jpg 99df150eed69030bbc53dba523d748c1                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Tr-no_0970.jpg new_dataset_audit_from_kaggle_files
        DS2    notumor   Te-noTr_0009.jpg 9d7c5b6de5e2f1f643513174e18ba6b5                                                                                                        /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Te-noTr_0009.jpg new_dataset_audit_from_kaggle_files
        DS2    notumor     Tr-no_0054.jpg 9d7c5b6de5e2f1f643513174e18ba6b5                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Tr-no_0054.jpg new_dataset_audit_from_kaggle_files
        DS2    notumor     Tr-no_1091.jpg 9d7c5b6de5e2f1f643513174e18ba6b5                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Tr-no_1091.jpg new_dataset_audit_from_kaggle_files
        DS2    notumor     Te-no_0083.jpg 9e07da527e38405b6815918097a3d7d5                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Te-no_0083.jpg new_dataset_audit_from_kaggle_files
        DS2    notumor     Te-no_0093.jpg 9e07da527e38405b6815918097a3d7d5                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Te-no_0093.jpg new_dataset_audit_from_kaggle_files
        DS2    notumor     Tr-no_0050.jpg 9e07da527e38405b6815918097a3d7d5                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Tr-no_0050.jpg new_dataset_audit_from_kaggle_files
        DS2    notumor     Tr-no_0346.jpg 9e07da527e38405b6815918097a3d7d5                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Tr-no_0346.jpg new_dataset_audit_from_kaggle_files
        DS2    notumor     Tr-no_1103.jpg 9e07da527e38405b6815918097a3d7d5                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Tr-no_1103.jpg new_dataset_audit_from_kaggle_files
        DS1    notumor   normal (276).jpg a350100047d5befbda844c9055abbe66 /root/.cache/kagglehub/datasets/orvile/pmram-bangladeshi-brain-cancer-mri-dataset/versions/2/PMRAM Bangladeshi Brain Cancer - MRI Dataset/PMRAM Bangladeshi Brain Cancer - MRI Dataset/Raw Data/Raw/512Normal/normal (276).jpg new_dataset_audit_from_kaggle_files
        DS1    notumor   normal (281).jpg a350100047d5befbda844c9055abbe66 /root/.cache/kagglehub/datasets/orvile/pmram-bangladeshi-brain-cancer-mri-dataset/versions/2/PMRAM Bangladeshi Brain Cancer - MRI Dataset/PMRAM Bangladeshi Brain Cancer - MRI Dataset/Raw Data/Raw/512Normal/normal (281).jpg new_dataset_audit_from_kaggle_files
        DS1    notumor   normal (294).jpg a350100047d5befbda844c9055abbe66 /root/.cache/kagglehub/datasets/orvile/pmram-bangladeshi-brain-cancer-mri-dataset/versions/2/PMRAM Bangladeshi Brain Cancer - MRI Dataset/PMRAM Bangladeshi Brain Cancer - MRI Dataset/Raw Data/Raw/512Normal/normal (294).jpg new_dataset_audit_from_kaggle_files
        DS2    notumor     Te-no_0026.jpg a3f433913a6ec7ec0ad6fbe124565467                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Te-no_0026.jpg new_dataset_audit_from_kaggle_files
        DS2    notumor     Tr-no_0340.jpg a3f433913a6ec7ec0ad6fbe124565467                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Tr-no_0340.jpg new_dataset_audit_from_kaggle_files
        DS2    notumor     Tr-no_1153.jpg a3f433913a6ec7ec0ad6fbe124565467                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Tr-no_1153.jpg new_dataset_audit_from_kaggle_files
        DS2    notumor     Te-no_0057.jpg a5cd99b2acf87717ad5a4f0760d1bc8f                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Te-no_0057.jpg new_dataset_audit_from_kaggle_files
        DS2    notumor     Tr-no_0313.jpg a5cd99b2acf87717ad5a4f0760d1bc8f                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Tr-no_0313.jpg new_dataset_audit_from_kaggle_files
        DS2    notumor     Tr-no_1166.jpg a5cd99b2acf87717ad5a4f0760d1bc8f                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Tr-no_1166.jpg new_dataset_audit_from_kaggle_files
        DS2    notumor     Te-no_0022.jpg a697f892e58613c39d315bece33fb3c7                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Te-no_0022.jpg new_dataset_audit_from_kaggle_files
        DS2    notumor     Te-no_0075.jpg a697f892e58613c39d315bece33fb3c7                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Te-no_0075.jpg new_dataset_audit_from_kaggle_files
        DS2    notumor     Tr-no_0069.jpg a697f892e58613c39d315bece33fb3c7                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Tr-no_0069.jpg new_dataset_audit_from_kaggle_files
        DS2    notumor     Tr-no_0338.jpg a697f892e58613c39d315bece33fb3c7                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Tr-no_0338.jpg new_dataset_audit_from_kaggle_files
        DS2    notumor     Tr-no_1084.jpg a697f892e58613c39d315bece33fb3c7                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Tr-no_1084.jpg new_dataset_audit_from_kaggle_files
        DS2    notumor     Te-no_0080.jpg a7638d37f715c0b57c2c8bbc30baf4cd                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Te-no_0080.jpg new_dataset_audit_from_kaggle_files
        DS2    notumor     Tr-no_0148.jpg a7638d37f715c0b57c2c8bbc30baf4cd                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Tr-no_0148.jpg new_dataset_audit_from_kaggle_files
        DS2    notumor     Tr-no_0222.jpg a7638d37f715c0b57c2c8bbc30baf4cd                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Tr-no_0222.jpg new_dataset_audit_from_kaggle_files
        DS2    notumor     Tr-no_0282.jpg a7638d37f715c0b57c2c8bbc30baf4cd                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Tr-no_0282.jpg new_dataset_audit_from_kaggle_files
        DS2    notumor     Tr-no_0344.jpg a7638d37f715c0b57c2c8bbc30baf4cd                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Tr-no_0344.jpg new_dataset_audit_from_kaggle_files
        DS2    notumor     Tr-no_1104.jpg a7638d37f715c0b57c2c8bbc30baf4cd                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Tr-no_1104.jpg new_dataset_audit_from_kaggle_files
        DS2    notumor     Te-no_0046.jpg ad338f8854d5c467aa337cc332cf36b2                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Te-no_0046.jpg new_dataset_audit_from_kaggle_files
        DS2    notumor     Tr-no_0202.jpg ad338f8854d5c467aa337cc332cf36b2                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Tr-no_0202.jpg new_dataset_audit_from_kaggle_files
        DS2    notumor     Tr-no_1178.jpg ad338f8854d5c467aa337cc332cf36b2                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Tr-no_1178.jpg new_dataset_audit_from_kaggle_files
        DS2 meningioma     Te-me_0071.jpg b6c9a71babd2557fb1d620640e264b17                                                                                                       /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/meningioma/Te-me_0071.jpg new_dataset_audit_from_kaggle_files
        DS2 meningioma     Te-me_0073.jpg b6c9a71babd2557fb1d620640e264b17                                                                                                       /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/meningioma/Te-me_0073.jpg new_dataset_audit_from_kaggle_files
        DS2 meningioma     Tr-me_0249.jpg b6c9a71babd2557fb1d620640e264b17                                                                                                       /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/meningioma/Tr-me_0249.jpg new_dataset_audit_from_kaggle_files
        DS2 meningioma     Tr-me_0251.jpg b6c9a71babd2557fb1d620640e264b17                                                                                                       /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/meningioma/Tr-me_0251.jpg new_dataset_audit_from_kaggle_files
        DS2    notumor     Te-no_0040.jpg b81d4a86910cb4386c7a0ed88cb9c928                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Te-no_0040.jpg new_dataset_audit_from_kaggle_files
        DS2    notumor     Te-no_0052.jpg b81d4a86910cb4386c7a0ed88cb9c928                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Te-no_0052.jpg new_dataset_audit_from_kaggle_files
        DS2    notumor     Tr-no_0135.jpg b81d4a86910cb4386c7a0ed88cb9c928                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Tr-no_0135.jpg new_dataset_audit_from_kaggle_files
        DS2    notumor     Tr-no_0257.jpg b81d4a86910cb4386c7a0ed88cb9c928                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Tr-no_0257.jpg new_dataset_audit_from_kaggle_files
        DS2    notumor     Tr-no_1096.jpg b81d4a86910cb4386c7a0ed88cb9c928                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Tr-no_1096.jpg new_dataset_audit_from_kaggle_files
        DS2    notumor     Te-no_0051.jpg bbcdc80eafd2ffb5a59468191960b9cf                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Te-no_0051.jpg new_dataset_audit_from_kaggle_files
        DS2    notumor     Tr-no_0246.jpg bbcdc80eafd2ffb5a59468191960b9cf                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Tr-no_0246.jpg new_dataset_audit_from_kaggle_files
        DS2    notumor     Tr-no_1182.jpg bbcdc80eafd2ffb5a59468191960b9cf                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Tr-no_1182.jpg new_dataset_audit_from_kaggle_files
        DS2    notumor     Te-no_0043.jpg bf12a15125a897a2b6e8edaa9bc9c54b                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Te-no_0043.jpg new_dataset_audit_from_kaggle_files
        DS2    notumor     Tr-no_0168.jpg bf12a15125a897a2b6e8edaa9bc9c54b                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Tr-no_0168.jpg new_dataset_audit_from_kaggle_files
        DS2    notumor     Tr-no_1122.jpg bf12a15125a897a2b6e8edaa9bc9c54b                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Tr-no_1122.jpg new_dataset_audit_from_kaggle_files
        DS2    notumor     Te-no_0059.jpg c945ddcc82615720c95807e20f9a661d                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Te-no_0059.jpg new_dataset_audit_from_kaggle_files
        DS2    notumor     Tr-no_0323.jpg c945ddcc82615720c95807e20f9a661d                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Tr-no_0323.jpg new_dataset_audit_from_kaggle_files
        DS2    notumor     Tr-no_0966.jpg c945ddcc82615720c95807e20f9a661d                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Tr-no_0966.jpg new_dataset_audit_from_kaggle_files
        DS2    notumor     Tr-no_0011.jpg cae61c6483c14eb68652a280f2571da1                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Tr-no_0011.jpg new_dataset_audit_from_kaggle_files
        DS2    notumor     Tr-no_0012.jpg cae61c6483c14eb68652a280f2571da1                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Tr-no_0012.jpg new_dataset_audit_from_kaggle_files
        DS2    notumor     Tr-no_0024.jpg cae61c6483c14eb68652a280f2571da1                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Tr-no_0024.jpg new_dataset_audit_from_kaggle_files
        DS2    notumor     Tr-no_0231.jpg cae61c6483c14eb68652a280f2571da1                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Tr-no_0231.jpg new_dataset_audit_from_kaggle_files
        DS2    notumor     Tr-no_0298.jpg cae61c6483c14eb68652a280f2571da1                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Tr-no_0298.jpg new_dataset_audit_from_kaggle_files
        DS2    notumor     Tr-no_1131.jpg cae61c6483c14eb68652a280f2571da1                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Tr-no_1131.jpg new_dataset_audit_from_kaggle_files
        DS2    notumor     Te-no_0027.jpg ce3d34156bcaa998da961e563c71eff3                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Te-no_0027.jpg new_dataset_audit_from_kaggle_files
        DS2    notumor     Tr-no_0043.jpg ce3d34156bcaa998da961e563c71eff3                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Tr-no_0043.jpg new_dataset_audit_from_kaggle_files
        DS2    notumor     Tr-no_1162.jpg ce3d34156bcaa998da961e563c71eff3                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Tr-no_1162.jpg new_dataset_audit_from_kaggle_files
        DS2    notumor     Tr-no_0264.jpg cee5eadaadf77769554e1e3756832c33                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Tr-no_0264.jpg new_dataset_audit_from_kaggle_files
        DS2    notumor     Tr-no_0270.jpg cee5eadaadf77769554e1e3756832c33                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Tr-no_0270.jpg new_dataset_audit_from_kaggle_files
        DS2    notumor     Tr-no_1149.jpg cee5eadaadf77769554e1e3756832c33                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Tr-no_1149.jpg new_dataset_audit_from_kaggle_files
        DS2    notumor     Te-no_0076.jpg cefc8fe39853df7445490b3af0f726ee                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Te-no_0076.jpg new_dataset_audit_from_kaggle_files
        DS2    notumor     Tr-no_0339.jpg cefc8fe39853df7445490b3af0f726ee                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Tr-no_0339.jpg new_dataset_audit_from_kaggle_files
        DS2    notumor     Tr-no_0981.jpg cefc8fe39853df7445490b3af0f726ee                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Tr-no_0981.jpg new_dataset_audit_from_kaggle_files
        DS2    notumor     Tr-no_0013.jpg d14c2ec45a8c43a0c1183a49fad6e957                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Tr-no_0013.jpg new_dataset_audit_from_kaggle_files
        DS2    notumor     Tr-no_0014.jpg d14c2ec45a8c43a0c1183a49fad6e957                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Tr-no_0014.jpg new_dataset_audit_from_kaggle_files
        DS2    notumor     Tr-no_1055.jpg d14c2ec45a8c43a0c1183a49fad6e957                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Tr-no_1055.jpg new_dataset_audit_from_kaggle_files
        DS1    notumor   normal (308).jpg d4272507eb73bd6127bac8ad2c3d1057 /root/.cache/kagglehub/datasets/orvile/pmram-bangladeshi-brain-cancer-mri-dataset/versions/2/PMRAM Bangladeshi Brain Cancer - MRI Dataset/PMRAM Bangladeshi Brain Cancer - MRI Dataset/Raw Data/Raw/512Normal/normal (308).jpg new_dataset_audit_from_kaggle_files
        DS1    notumor   normal (322).jpg d4272507eb73bd6127bac8ad2c3d1057 /root/.cache/kagglehub/datasets/orvile/pmram-bangladeshi-brain-cancer-mri-dataset/versions/2/PMRAM Bangladeshi Brain Cancer - MRI Dataset/PMRAM Bangladeshi Brain Cancer - MRI Dataset/Raw Data/Raw/512Normal/normal (322).jpg new_dataset_audit_from_kaggle_files
        DS1    notumor   normal (386).jpg d4272507eb73bd6127bac8ad2c3d1057 /root/.cache/kagglehub/datasets/orvile/pmram-bangladeshi-brain-cancer-mri-dataset/versions/2/PMRAM Bangladeshi Brain Cancer - MRI Dataset/PMRAM Bangladeshi Brain Cancer - MRI Dataset/Raw Data/Raw/512Normal/normal (386).jpg new_dataset_audit_from_kaggle_files
        DS2    notumor     Te-no_0086.jpg d869dcd28bfa2c4fe24d1bc139e47b43                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Te-no_0086.jpg new_dataset_audit_from_kaggle_files
        DS2    notumor     Tr-no_0349.jpg d869dcd28bfa2c4fe24d1bc139e47b43                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Tr-no_0349.jpg new_dataset_audit_from_kaggle_files
        DS2    notumor     Tr-no_0964.jpg d869dcd28bfa2c4fe24d1bc139e47b43                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Tr-no_0964.jpg new_dataset_audit_from_kaggle_files
        DS2    notumor     Tr-no_0116.jpg dabf0bf1fa1833aceb8379634d9e8d96                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Tr-no_0116.jpg new_dataset_audit_from_kaggle_files
        DS2    notumor     Tr-no_0130.jpg dabf0bf1fa1833aceb8379634d9e8d96                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Tr-no_0130.jpg new_dataset_audit_from_kaggle_files
        DS2    notumor     Tr-no_0132.jpg dabf0bf1fa1833aceb8379634d9e8d96                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Tr-no_0132.jpg new_dataset_audit_from_kaggle_files
        DS1    notumor   normal (300).jpg db17176dade907bdf5c4e179d4543d56 /root/.cache/kagglehub/datasets/orvile/pmram-bangladeshi-brain-cancer-mri-dataset/versions/2/PMRAM Bangladeshi Brain Cancer - MRI Dataset/PMRAM Bangladeshi Brain Cancer - MRI Dataset/Raw Data/Raw/512Normal/normal (300).jpg new_dataset_audit_from_kaggle_files
        DS1    notumor   normal (301).jpg db17176dade907bdf5c4e179d4543d56 /root/.cache/kagglehub/datasets/orvile/pmram-bangladeshi-brain-cancer-mri-dataset/versions/2/PMRAM Bangladeshi Brain Cancer - MRI Dataset/PMRAM Bangladeshi Brain Cancer - MRI Dataset/Raw Data/Raw/512Normal/normal (301).jpg new_dataset_audit_from_kaggle_files
        DS1    notumor   normal (307).jpg db17176dade907bdf5c4e179d4543d56 /root/.cache/kagglehub/datasets/orvile/pmram-bangladeshi-brain-cancer-mri-dataset/versions/2/PMRAM Bangladeshi Brain Cancer - MRI Dataset/PMRAM Bangladeshi Brain Cancer - MRI Dataset/Raw Data/Raw/512Normal/normal (307).jpg new_dataset_audit_from_kaggle_files
        DS2    notumor     Tr-no_0185.jpg e2a9705d5e78b764288711292825d3b2                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Tr-no_0185.jpg new_dataset_audit_from_kaggle_files
        DS2    notumor     Tr-no_0187.jpg e2a9705d5e78b764288711292825d3b2                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Tr-no_0187.jpg new_dataset_audit_from_kaggle_files
        DS2    notumor     Tr-no_1179.jpg e2a9705d5e78b764288711292825d3b2                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Tr-no_1179.jpg new_dataset_audit_from_kaggle_files
        DS2 meningioma     Te-me_0010.jpg e2ce9c2850231613de6b2952dd9a4ace                                                                                                       /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/meningioma/Te-me_0010.jpg new_dataset_audit_from_kaggle_files
        DS2 meningioma     Te-me_0031.jpg e2ce9c2850231613de6b2952dd9a4ace                                                                                                       /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/meningioma/Te-me_0031.jpg new_dataset_audit_from_kaggle_files
        DS2 meningioma     Tr-me_0176.jpg e2ce9c2850231613de6b2952dd9a4ace                                                                                                       /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/meningioma/Tr-me_0176.jpg new_dataset_audit_from_kaggle_files
        DS2 meningioma     Tr-me_0211.jpg e2ce9c2850231613de6b2952dd9a4ace                                                                                                       /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/meningioma/Tr-me_0211.jpg new_dataset_audit_from_kaggle_files
        DS2 meningioma     Tr-me_0366.jpg e2ce9c2850231613de6b2952dd9a4ace                                                                                                       /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/meningioma/Tr-me_0366.jpg new_dataset_audit_from_kaggle_files
        DS2    notumor     Tr-no_0145.jpg e2eaa67f16d59ca7ab63c803251adf88                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Tr-no_0145.jpg new_dataset_audit_from_kaggle_files
        DS2    notumor     Tr-no_0182.jpg e2eaa67f16d59ca7ab63c803251adf88                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Tr-no_0182.jpg new_dataset_audit_from_kaggle_files
        DS2    notumor     Tr-no_1102.jpg e2eaa67f16d59ca7ab63c803251adf88                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Tr-no_1102.jpg new_dataset_audit_from_kaggle_files
        DS1    notumor   normal (169).jpg e3c3ccdc157450a7649c35437153fd22 /root/.cache/kagglehub/datasets/orvile/pmram-bangladeshi-brain-cancer-mri-dataset/versions/2/PMRAM Bangladeshi Brain Cancer - MRI Dataset/PMRAM Bangladeshi Brain Cancer - MRI Dataset/Raw Data/Raw/512Normal/normal (169).jpg new_dataset_audit_from_kaggle_files
        DS1    notumor     normal (5).jpg e3c3ccdc157450a7649c35437153fd22   /root/.cache/kagglehub/datasets/orvile/pmram-bangladeshi-brain-cancer-mri-dataset/versions/2/PMRAM Bangladeshi Brain Cancer - MRI Dataset/PMRAM Bangladeshi Brain Cancer - MRI Dataset/Raw Data/Raw/512Normal/normal (5).jpg new_dataset_audit_from_kaggle_files
        DS1    notumor    normal (55).jpg e3c3ccdc157450a7649c35437153fd22  /root/.cache/kagglehub/datasets/orvile/pmram-bangladeshi-brain-cancer-mri-dataset/versions/2/PMRAM Bangladeshi Brain Cancer - MRI Dataset/PMRAM Bangladeshi Brain Cancer - MRI Dataset/Raw Data/Raw/512Normal/normal (55).jpg new_dataset_audit_from_kaggle_files
        DS2    notumor     Te-no_0029.jpg ead6beeabe5a77f2de90e8d546f9a615                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Te-no_0029.jpg new_dataset_audit_from_kaggle_files
        DS2    notumor     Te-no_0069.jpg ead6beeabe5a77f2de90e8d546f9a615                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Te-no_0069.jpg new_dataset_audit_from_kaggle_files
        DS2    notumor     Tr-no_0333.jpg ead6beeabe5a77f2de90e8d546f9a615                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Tr-no_0333.jpg new_dataset_audit_from_kaggle_files
        DS2    notumor     Te-no_0010.jpg eedad04ca8d1fdbddaf181f887dd8063                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Te-no_0010.jpg new_dataset_audit_from_kaggle_files
        DS2    notumor     Tr-no_0055.jpg eedad04ca8d1fdbddaf181f887dd8063                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Tr-no_0055.jpg new_dataset_audit_from_kaggle_files
        DS2    notumor     Tr-no_1039.jpg eedad04ca8d1fdbddaf181f887dd8063                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Tr-no_1039.jpg new_dataset_audit_from_kaggle_files
        DS1    notumor   normal (274).jpg f3400a6369d7f615744dc64f6d86c763 /root/.cache/kagglehub/datasets/orvile/pmram-bangladeshi-brain-cancer-mri-dataset/versions/2/PMRAM Bangladeshi Brain Cancer - MRI Dataset/PMRAM Bangladeshi Brain Cancer - MRI Dataset/Raw Data/Raw/512Normal/normal (274).jpg new_dataset_audit_from_kaggle_files
        DS1    notumor   normal (288).jpg f3400a6369d7f615744dc64f6d86c763 /root/.cache/kagglehub/datasets/orvile/pmram-bangladeshi-brain-cancer-mri-dataset/versions/2/PMRAM Bangladeshi Brain Cancer - MRI Dataset/PMRAM Bangladeshi Brain Cancer - MRI Dataset/Raw Data/Raw/512Normal/normal (288).jpg new_dataset_audit_from_kaggle_files
        DS1    notumor   normal (290).jpg f3400a6369d7f615744dc64f6d86c763 /root/.cache/kagglehub/datasets/orvile/pmram-bangladeshi-brain-cancer-mri-dataset/versions/2/PMRAM Bangladeshi Brain Cancer - MRI Dataset/PMRAM Bangladeshi Brain Cancer - MRI Dataset/Raw Data/Raw/512Normal/normal (290).jpg new_dataset_audit_from_kaggle_files
        DS2    notumor     Tr-no_0111.jpg f565b1b202ecf70e5833f979db4f1ea5                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Tr-no_0111.jpg new_dataset_audit_from_kaggle_files
        DS2    notumor     Tr-no_0159.jpg f565b1b202ecf70e5833f979db4f1ea5                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Tr-no_0159.jpg new_dataset_audit_from_kaggle_files
        DS2    notumor     Tr-no_1114.jpg f565b1b202ecf70e5833f979db4f1ea5                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Tr-no_1114.jpg new_dataset_audit_from_kaggle_files
        DS2    notumor     Te-no_0017.jpg f835546804600895d80483418aac4caf                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Te-no_0017.jpg new_dataset_audit_from_kaggle_files
        DS2    notumor     Tr-no_0062.jpg f835546804600895d80483418aac4caf                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Tr-no_0062.jpg new_dataset_audit_from_kaggle_files
        DS2    notumor     Tr-no_1159.jpg f835546804600895d80483418aac4caf                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Tr-no_1159.jpg new_dataset_audit_from_kaggle_files
        DS2    notumor     Tr-no_0117.jpg f9d65447a891873c222028e6e4e830f8                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Tr-no_0117.jpg new_dataset_audit_from_kaggle_files
        DS2    notumor     Tr-no_0131.jpg f9d65447a891873c222028e6e4e830f8                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Tr-no_0131.jpg new_dataset_audit_from_kaggle_files
        DS2    notumor     Tr-no_0366.jpg f9d65447a891873c222028e6e4e830f8                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Tr-no_0366.jpg new_dataset_audit_from_kaggle_files
        DS2    notumor     Tr-no_0369.jpg f9d65447a891873c222028e6e4e830f8                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Tr-no_0369.jpg new_dataset_audit_from_kaggle_files
        DS2    notumor     Tr-no_1095.jpg f9d65447a891873c222028e6e4e830f8                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Tr-no_1095.jpg new_dataset_audit_from_kaggle_files
        DS2    notumor     Te-no_0019.jpg fadd9cb36cd4a8db664687cf3f9a3b1b                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Te-no_0019.jpg new_dataset_audit_from_kaggle_files
        DS2    notumor     Te-no_0062.jpg fadd9cb36cd4a8db664687cf3f9a3b1b                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Te-no_0062.jpg new_dataset_audit_from_kaggle_files
        DS2    notumor     Te-no_0073.jpg fadd9cb36cd4a8db664687cf3f9a3b1b                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Te-no_0073.jpg new_dataset_audit_from_kaggle_files
        DS2    notumor     Tr-no_0064.jpg fadd9cb36cd4a8db664687cf3f9a3b1b                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Tr-no_0064.jpg new_dataset_audit_from_kaggle_files
        DS2    notumor     Tr-no_0325.jpg fadd9cb36cd4a8db664687cf3f9a3b1b                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Tr-no_0325.jpg new_dataset_audit_from_kaggle_files
        DS2    notumor     Tr-no_0336.jpg fadd9cb36cd4a8db664687cf3f9a3b1b                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Tr-no_0336.jpg new_dataset_audit_from_kaggle_files
        DS2    notumor     Tr-no_1183.jpg fadd9cb36cd4a8db664687cf3f9a3b1b                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Tr-no_1183.jpg new_dataset_audit_from_kaggle_files
    
    ======================================================================================================================================================
    3.4 TRAIN/VALIDATION/TEST SPLITTING AND LEAKAGE CONTROL
    ======================================================================================================================================================
    
    ------------------------------------------------------------------------------------------------------------------------------------------------------
    Main saved-output split summary
    ------------------------------------------------------------------------------------------------------------------------------------------------------
    dataset  train  val  test  total       source_type
        DS1   1053  226   226   1505 main_saved_output
        DS2   4921 1055  1055   7031 main_saved_output
    
    ------------------------------------------------------------------------------------------------------------------------------------------------------
    Computed split summary using main-code split logic
    ------------------------------------------------------------------------------------------------------------------------------------------------------
    dataset  train  val  test  total                                              source_type
        DS1   1053  226   226   1505 computed_with_main_split_code_on_downloaded_kaggle_files
        DS2   4921 1055  1055   7031 computed_with_main_split_code_on_downloaded_kaggle_files
    
    ------------------------------------------------------------------------------------------------------------------------------------------------------
    Computed split-wise class counts and percentages
    ------------------------------------------------------------------------------------------------------------------------------------------------------
    dataset split  n_images  pct_of_dataset                                              source_type  n_glioma  pct_glioma_within_split  n_meningioma  pct_meningioma_within_split  n_notumor  pct_notumor_within_split  n_pituitary  pct_pituitary_within_split
        DS1 train      1053       69.966777 computed_with_main_split_code_on_downloaded_kaggle_files       261                24.786325           254                    24.121557        277                 26.305793          261                   24.786325
        DS1   val       226       15.016611 computed_with_main_split_code_on_downloaded_kaggle_files        56                24.778761            54                    23.893805         60                 26.548673           56                   24.778761
        DS1  test       226       15.016611 computed_with_main_split_code_on_downloaded_kaggle_files        56                24.778761            55                    24.336283         59                 26.106195           56                   24.778761
        DS2 train      4921       69.990044 computed_with_main_split_code_on_downloaded_kaggle_files      1134                23.044097          1152                    23.409876       1400                 28.449502         1235                   25.096525
        DS2   val      1055       15.004978 computed_with_main_split_code_on_downloaded_kaggle_files       243                23.033175           247                    23.412322        300                 28.436019          265                   25.118483
        DS2  test      1055       15.004978 computed_with_main_split_code_on_downloaded_kaggle_files       244                23.127962           247                    23.412322        300                 28.436019          264                   25.023697
    
    ------------------------------------------------------------------------------------------------------------------------------------------------------
    Main-code split configuration
    ------------------------------------------------------------------------------------------------------------------------------------------------------
          parameter                                                              value      source_type
        split_scope                                       DS1 and DS2 split separately main_code_config
       split_method stratified train/temp split, then stratified validation/test split main_code_config
    global_val_frac                                                               0.15 main_code_config
          test_frac                                                               0.15 main_code_config
       random_state                                                                 42 main_code_config
    
    ------------------------------------------------------------------------------------------------------------------------------------------------------
    Main saved-output quick/subset leakage summary
    ------------------------------------------------------------------------------------------------------------------------------------------------------
    dataset  path_overlap_train_val  path_overlap_train_test  path_overlap_val_test  unique_paths_train  unique_paths_val  unique_paths_test  filename_overlap_train_val  filename_overlap_train_test  filename_overlap_val_test  subset_hash_train_val  subset_hash_train_test  subset_hash_val_test  subset_hash_n_train  subset_hash_n_val  subset_hash_n_test       source_type
        DS1                       0                        0                      0                1053               226                226                           0                            0                          0                      9                       4                     7                  297                224                 225 main_saved_output
        DS2                       0                        0                      0                4921              1055               1055                           0                            0                          0                      2                       2                     2                  300                297                 298 main_saved_output
    
    ------------------------------------------------------------------------------------------------------------------------------------------------------
    Computed full-MD5 leakage summary across train/validation/test
    ------------------------------------------------------------------------------------------------------------------------------------------------------
    dataset    split_pair  path_overlap  filename_overlap  full_md5_overlap_unique_hashes  full_md5_overlap_images_in_pair  overlap_hashes_with_conflicting_labels                                              source_type
        DS1  train_vs_val             0                 0                              18                               37                                       0 computed_with_main_split_code_on_downloaded_kaggle_files
        DS1 train_vs_test             0                 0                              21                               45                                       0 computed_with_main_split_code_on_downloaded_kaggle_files
        DS1   val_vs_test             0                 0                               3                                6                                       0 computed_with_main_split_code_on_downloaded_kaggle_files
        DS2  train_vs_val             0                 0                              61                              172                                       0 computed_with_main_split_code_on_downloaded_kaggle_files
        DS2 train_vs_test             0                 0                              75                              198                                       0 computed_with_main_split_code_on_downloaded_kaggle_files
        DS2   val_vs_test             0                 0                              41                               92                                       0 computed_with_main_split_code_on_downloaded_kaggle_files
    
    ------------------------------------------------------------------------------------------------------------------------------------------------------
    Computed full-MD5 train/validation/test leakage examples
    ------------------------------------------------------------------------------------------------------------------------------------------------------
    dataset split      label           filename                              md5                                                                                                                                                                                                                           path                                              source_type
        DS1  test    notumor    normal (78).jpg 130893ec3b6f15fd489fc2ece20323b8  /root/.cache/kagglehub/datasets/orvile/pmram-bangladeshi-brain-cancer-mri-dataset/versions/2/PMRAM Bangladeshi Brain Cancer - MRI Dataset/PMRAM Bangladeshi Brain Cancer - MRI Dataset/Raw Data/Raw/512Normal/normal (78).jpg computed_with_main_split_code_on_downloaded_kaggle_files
        DS1   val    notumor   normal (104).jpg 130893ec3b6f15fd489fc2ece20323b8 /root/.cache/kagglehub/datasets/orvile/pmram-bangladeshi-brain-cancer-mri-dataset/versions/2/PMRAM Bangladeshi Brain Cancer - MRI Dataset/PMRAM Bangladeshi Brain Cancer - MRI Dataset/Raw Data/Raw/512Normal/normal (104).jpg computed_with_main_split_code_on_downloaded_kaggle_files
        DS1  test meningioma          M_161.jpg 15ab418722ae8ce42490285b7c5de7d3    /root/.cache/kagglehub/datasets/orvile/pmram-bangladeshi-brain-cancer-mri-dataset/versions/2/PMRAM Bangladeshi Brain Cancer - MRI Dataset/PMRAM Bangladeshi Brain Cancer - MRI Dataset/Raw Data/Raw/512Meningioma/M_161.jpg computed_with_main_split_code_on_downloaded_kaggle_files
        DS1 train meningioma           M_13.jpg 15ab418722ae8ce42490285b7c5de7d3     /root/.cache/kagglehub/datasets/orvile/pmram-bangladeshi-brain-cancer-mri-dataset/versions/2/PMRAM Bangladeshi Brain Cancer - MRI Dataset/PMRAM Bangladeshi Brain Cancer - MRI Dataset/Raw Data/Raw/512Meningioma/M_13.jpg computed_with_main_split_code_on_downloaded_kaggle_files
        DS1  test    notumor    normal (16).jpg 1868801e49e6cd18fb2e4189250921e4  /root/.cache/kagglehub/datasets/orvile/pmram-bangladeshi-brain-cancer-mri-dataset/versions/2/PMRAM Bangladeshi Brain Cancer - MRI Dataset/PMRAM Bangladeshi Brain Cancer - MRI Dataset/Raw Data/Raw/512Normal/normal (16).jpg computed_with_main_split_code_on_downloaded_kaggle_files
        DS1 train    notumor   normal (232).jpg 1868801e49e6cd18fb2e4189250921e4 /root/.cache/kagglehub/datasets/orvile/pmram-bangladeshi-brain-cancer-mri-dataset/versions/2/PMRAM Bangladeshi Brain Cancer - MRI Dataset/PMRAM Bangladeshi Brain Cancer - MRI Dataset/Raw Data/Raw/512Normal/normal (232).jpg computed_with_main_split_code_on_downloaded_kaggle_files
        DS1  test    notumor     normal (1).jpg 1e2fd8da45a9cc434c617ae227ac9e27   /root/.cache/kagglehub/datasets/orvile/pmram-bangladeshi-brain-cancer-mri-dataset/versions/2/PMRAM Bangladeshi Brain Cancer - MRI Dataset/PMRAM Bangladeshi Brain Cancer - MRI Dataset/Raw Data/Raw/512Normal/normal (1).jpg computed_with_main_split_code_on_downloaded_kaggle_files
        DS1 train    notumor   normal (197).jpg 1e2fd8da45a9cc434c617ae227ac9e27 /root/.cache/kagglehub/datasets/orvile/pmram-bangladeshi-brain-cancer-mri-dataset/versions/2/PMRAM Bangladeshi Brain Cancer - MRI Dataset/PMRAM Bangladeshi Brain Cancer - MRI Dataset/Raw Data/Raw/512Normal/normal (197).jpg computed_with_main_split_code_on_downloaded_kaggle_files
        DS1 train    notumor    normal (22).jpg 1e2fd8da45a9cc434c617ae227ac9e27  /root/.cache/kagglehub/datasets/orvile/pmram-bangladeshi-brain-cancer-mri-dataset/versions/2/PMRAM Bangladeshi Brain Cancer - MRI Dataset/PMRAM Bangladeshi Brain Cancer - MRI Dataset/Raw Data/Raw/512Normal/normal (22).jpg computed_with_main_split_code_on_downloaded_kaggle_files
        DS1 train    notumor   normal (184).jpg 234bef5f52771afb417f07b260c261f9 /root/.cache/kagglehub/datasets/orvile/pmram-bangladeshi-brain-cancer-mri-dataset/versions/2/PMRAM Bangladeshi Brain Cancer - MRI Dataset/PMRAM Bangladeshi Brain Cancer - MRI Dataset/Raw Data/Raw/512Normal/normal (184).jpg computed_with_main_split_code_on_downloaded_kaggle_files
        DS1   val    notumor     normal (7).jpg 234bef5f52771afb417f07b260c261f9   /root/.cache/kagglehub/datasets/orvile/pmram-bangladeshi-brain-cancer-mri-dataset/versions/2/PMRAM Bangladeshi Brain Cancer - MRI Dataset/PMRAM Bangladeshi Brain Cancer - MRI Dataset/Raw Data/Raw/512Normal/normal (7).jpg computed_with_main_split_code_on_downloaded_kaggle_files
        DS1  test meningioma          M_352.jpg 246a25a4d05b67b63c41b0c27f9204bf    /root/.cache/kagglehub/datasets/orvile/pmram-bangladeshi-brain-cancer-mri-dataset/versions/2/PMRAM Bangladeshi Brain Cancer - MRI Dataset/PMRAM Bangladeshi Brain Cancer - MRI Dataset/Raw Data/Raw/512Meningioma/M_352.jpg computed_with_main_split_code_on_downloaded_kaggle_files
        DS1 train meningioma           M_57.jpg 246a25a4d05b67b63c41b0c27f9204bf     /root/.cache/kagglehub/datasets/orvile/pmram-bangladeshi-brain-cancer-mri-dataset/versions/2/PMRAM Bangladeshi Brain Cancer - MRI Dataset/PMRAM Bangladeshi Brain Cancer - MRI Dataset/Raw Data/Raw/512Meningioma/M_57.jpg computed_with_main_split_code_on_downloaded_kaggle_files
        DS1 train    notumor   normal (263).jpg 2edab9d1eb87e811edc6ac3dc06e0826 /root/.cache/kagglehub/datasets/orvile/pmram-bangladeshi-brain-cancer-mri-dataset/versions/2/PMRAM Bangladeshi Brain Cancer - MRI Dataset/PMRAM Bangladeshi Brain Cancer - MRI Dataset/Raw Data/Raw/512Normal/normal (263).jpg computed_with_main_split_code_on_downloaded_kaggle_files
        DS1   val    notumor   normal (262).jpg 2edab9d1eb87e811edc6ac3dc06e0826 /root/.cache/kagglehub/datasets/orvile/pmram-bangladeshi-brain-cancer-mri-dataset/versions/2/PMRAM Bangladeshi Brain Cancer - MRI Dataset/PMRAM Bangladeshi Brain Cancer - MRI Dataset/Raw Data/Raw/512Normal/normal (262).jpg computed_with_main_split_code_on_downloaded_kaggle_files
        DS1  test meningioma            M_6.jpg 39b7e8e4b4ad8f609c0b5509b7945f22      /root/.cache/kagglehub/datasets/orvile/pmram-bangladeshi-brain-cancer-mri-dataset/versions/2/PMRAM Bangladeshi Brain Cancer - MRI Dataset/PMRAM Bangladeshi Brain Cancer - MRI Dataset/Raw Data/Raw/512Meningioma/M_6.jpg computed_with_main_split_code_on_downloaded_kaggle_files
        DS1 train meningioma          M_249.jpg 39b7e8e4b4ad8f609c0b5509b7945f22    /root/.cache/kagglehub/datasets/orvile/pmram-bangladeshi-brain-cancer-mri-dataset/versions/2/PMRAM Bangladeshi Brain Cancer - MRI Dataset/PMRAM Bangladeshi Brain Cancer - MRI Dataset/Raw Data/Raw/512Meningioma/M_249.jpg computed_with_main_split_code_on_downloaded_kaggle_files
        DS1 train    notumor   normal (295).jpg 3af353cf97090c4bef8e4434a6af77cd /root/.cache/kagglehub/datasets/orvile/pmram-bangladeshi-brain-cancer-mri-dataset/versions/2/PMRAM Bangladeshi Brain Cancer - MRI Dataset/PMRAM Bangladeshi Brain Cancer - MRI Dataset/Raw Data/Raw/512Normal/normal (295).jpg computed_with_main_split_code_on_downloaded_kaggle_files
        DS1   val    notumor   normal (286).jpg 3af353cf97090c4bef8e4434a6af77cd /root/.cache/kagglehub/datasets/orvile/pmram-bangladeshi-brain-cancer-mri-dataset/versions/2/PMRAM Bangladeshi Brain Cancer - MRI Dataset/PMRAM Bangladeshi Brain Cancer - MRI Dataset/Raw Data/Raw/512Normal/normal (286).jpg computed_with_main_split_code_on_downloaded_kaggle_files
        DS1  test meningioma          M_179.jpg 3c0fd673b2aad68f3c115d045a0c7043    /root/.cache/kagglehub/datasets/orvile/pmram-bangladeshi-brain-cancer-mri-dataset/versions/2/PMRAM Bangladeshi Brain Cancer - MRI Dataset/PMRAM Bangladeshi Brain Cancer - MRI Dataset/Raw Data/Raw/512Meningioma/M_179.jpg computed_with_main_split_code_on_downloaded_kaggle_files
        DS1 train meningioma          M_164.jpg 3c0fd673b2aad68f3c115d045a0c7043    /root/.cache/kagglehub/datasets/orvile/pmram-bangladeshi-brain-cancer-mri-dataset/versions/2/PMRAM Bangladeshi Brain Cancer - MRI Dataset/PMRAM Bangladeshi Brain Cancer - MRI Dataset/Raw Data/Raw/512Meningioma/M_164.jpg computed_with_main_split_code_on_downloaded_kaggle_files
        DS1  test    notumor   normal (204).jpg 3d7311be68f2708b8ccbe6fbce5e35a9 /root/.cache/kagglehub/datasets/orvile/pmram-bangladeshi-brain-cancer-mri-dataset/versions/2/PMRAM Bangladeshi Brain Cancer - MRI Dataset/PMRAM Bangladeshi Brain Cancer - MRI Dataset/Raw Data/Raw/512Normal/normal (204).jpg computed_with_main_split_code_on_downloaded_kaggle_files
        DS1 train    notumor    normal (58).jpg 3d7311be68f2708b8ccbe6fbce5e35a9  /root/.cache/kagglehub/datasets/orvile/pmram-bangladeshi-brain-cancer-mri-dataset/versions/2/PMRAM Bangladeshi Brain Cancer - MRI Dataset/PMRAM Bangladeshi Brain Cancer - MRI Dataset/Raw Data/Raw/512Normal/normal (58).jpg computed_with_main_split_code_on_downloaded_kaggle_files
        DS1 train meningioma            M_9.jpg 47af150d8969efeed9ab471015346c91      /root/.cache/kagglehub/datasets/orvile/pmram-bangladeshi-brain-cancer-mri-dataset/versions/2/PMRAM Bangladeshi Brain Cancer - MRI Dataset/PMRAM Bangladeshi Brain Cancer - MRI Dataset/Raw Data/Raw/512Meningioma/M_9.jpg computed_with_main_split_code_on_downloaded_kaggle_files
        DS1   val meningioma          M_252.jpg 47af150d8969efeed9ab471015346c91    /root/.cache/kagglehub/datasets/orvile/pmram-bangladeshi-brain-cancer-mri-dataset/versions/2/PMRAM Bangladeshi Brain Cancer - MRI Dataset/PMRAM Bangladeshi Brain Cancer - MRI Dataset/Raw Data/Raw/512Meningioma/M_252.jpg computed_with_main_split_code_on_downloaded_kaggle_files
        DS1 train    notumor   normal (275).jpg 4d74676fc0690c8257b13fa6f1e2c37b /root/.cache/kagglehub/datasets/orvile/pmram-bangladeshi-brain-cancer-mri-dataset/versions/2/PMRAM Bangladeshi Brain Cancer - MRI Dataset/PMRAM Bangladeshi Brain Cancer - MRI Dataset/Raw Data/Raw/512Normal/normal (275).jpg computed_with_main_split_code_on_downloaded_kaggle_files
        DS1   val    notumor   normal (289).jpg 4d74676fc0690c8257b13fa6f1e2c37b /root/.cache/kagglehub/datasets/orvile/pmram-bangladeshi-brain-cancer-mri-dataset/versions/2/PMRAM Bangladeshi Brain Cancer - MRI Dataset/PMRAM Bangladeshi Brain Cancer - MRI Dataset/Raw Data/Raw/512Normal/normal (289).jpg computed_with_main_split_code_on_downloaded_kaggle_files
        DS1 train    notumor   normal (380).jpg 5522aa3bd241b57ed05c1896e7267a53 /root/.cache/kagglehub/datasets/orvile/pmram-bangladeshi-brain-cancer-mri-dataset/versions/2/PMRAM Bangladeshi Brain Cancer - MRI Dataset/PMRAM Bangladeshi Brain Cancer - MRI Dataset/Raw Data/Raw/512Normal/normal (380).jpg computed_with_main_split_code_on_downloaded_kaggle_files
        DS1   val    notumor   normal (306).jpg 5522aa3bd241b57ed05c1896e7267a53 /root/.cache/kagglehub/datasets/orvile/pmram-bangladeshi-brain-cancer-mri-dataset/versions/2/PMRAM Bangladeshi Brain Cancer - MRI Dataset/PMRAM Bangladeshi Brain Cancer - MRI Dataset/Raw Data/Raw/512Normal/normal (306).jpg computed_with_main_split_code_on_downloaded_kaggle_files
        DS1  test    notumor    normal (68).jpg 573cdc63783a4e66074bb48b7753c71a  /root/.cache/kagglehub/datasets/orvile/pmram-bangladeshi-brain-cancer-mri-dataset/versions/2/PMRAM Bangladeshi Brain Cancer - MRI Dataset/PMRAM Bangladeshi Brain Cancer - MRI Dataset/Raw Data/Raw/512Normal/normal (68).jpg computed_with_main_split_code_on_downloaded_kaggle_files
        DS1 train    notumor   normal (172).jpg 573cdc63783a4e66074bb48b7753c71a /root/.cache/kagglehub/datasets/orvile/pmram-bangladeshi-brain-cancer-mri-dataset/versions/2/PMRAM Bangladeshi Brain Cancer - MRI Dataset/PMRAM Bangladeshi Brain Cancer - MRI Dataset/Raw Data/Raw/512Normal/normal (172).jpg computed_with_main_split_code_on_downloaded_kaggle_files
        DS1 train     glioma    glioma (82).jpg 5943ec4dd3ffa9e503ac0a6bd79f84e4  /root/.cache/kagglehub/datasets/orvile/pmram-bangladeshi-brain-cancer-mri-dataset/versions/2/PMRAM Bangladeshi Brain Cancer - MRI Dataset/PMRAM Bangladeshi Brain Cancer - MRI Dataset/Raw Data/Raw/512Glioma/glioma (82).jpg computed_with_main_split_code_on_downloaded_kaggle_files
        DS1   val     glioma    glioma (60).jpg 5943ec4dd3ffa9e503ac0a6bd79f84e4  /root/.cache/kagglehub/datasets/orvile/pmram-bangladeshi-brain-cancer-mri-dataset/versions/2/PMRAM Bangladeshi Brain Cancer - MRI Dataset/PMRAM Bangladeshi Brain Cancer - MRI Dataset/Raw Data/Raw/512Glioma/glioma (60).jpg computed_with_main_split_code_on_downloaded_kaggle_files
        DS1 train meningioma          M_172.jpg 635457122fc3ea9136fd04a2fe32c4ab    /root/.cache/kagglehub/datasets/orvile/pmram-bangladeshi-brain-cancer-mri-dataset/versions/2/PMRAM Bangladeshi Brain Cancer - MRI Dataset/PMRAM Bangladeshi Brain Cancer - MRI Dataset/Raw Data/Raw/512Meningioma/M_172.jpg computed_with_main_split_code_on_downloaded_kaggle_files
        DS1   val meningioma          M_246.jpg 635457122fc3ea9136fd04a2fe32c4ab    /root/.cache/kagglehub/datasets/orvile/pmram-bangladeshi-brain-cancer-mri-dataset/versions/2/PMRAM Bangladeshi Brain Cancer - MRI Dataset/PMRAM Bangladeshi Brain Cancer - MRI Dataset/Raw Data/Raw/512Meningioma/M_246.jpg computed_with_main_split_code_on_downloaded_kaggle_files
        DS1  test    notumor   normal (377).jpg 64622674a568513acad504018a822282 /root/.cache/kagglehub/datasets/orvile/pmram-bangladeshi-brain-cancer-mri-dataset/versions/2/PMRAM Bangladeshi Brain Cancer - MRI Dataset/PMRAM Bangladeshi Brain Cancer - MRI Dataset/Raw Data/Raw/512Normal/normal (377).jpg computed_with_main_split_code_on_downloaded_kaggle_files
        DS1 train    notumor   normal (351).jpg 64622674a568513acad504018a822282 /root/.cache/kagglehub/datasets/orvile/pmram-bangladeshi-brain-cancer-mri-dataset/versions/2/PMRAM Bangladeshi Brain Cancer - MRI Dataset/PMRAM Bangladeshi Brain Cancer - MRI Dataset/Raw Data/Raw/512Normal/normal (351).jpg computed_with_main_split_code_on_downloaded_kaggle_files
        DS1  test    notumor   normal (137).jpg 690810abdacc4dba5dc6fb3f672ae52e /root/.cache/kagglehub/datasets/orvile/pmram-bangladeshi-brain-cancer-mri-dataset/versions/2/PMRAM Bangladeshi Brain Cancer - MRI Dataset/PMRAM Bangladeshi Brain Cancer - MRI Dataset/Raw Data/Raw/512Normal/normal (137).jpg computed_with_main_split_code_on_downloaded_kaggle_files
        DS1 train    notumor   normal (193).jpg 690810abdacc4dba5dc6fb3f672ae52e /root/.cache/kagglehub/datasets/orvile/pmram-bangladeshi-brain-cancer-mri-dataset/versions/2/PMRAM Bangladeshi Brain Cancer - MRI Dataset/PMRAM Bangladeshi Brain Cancer - MRI Dataset/Raw Data/Raw/512Normal/normal (193).jpg computed_with_main_split_code_on_downloaded_kaggle_files
        DS1  test meningioma            M_8.jpg 6b303edae8c3c4aadfb211a388fa33ab      /root/.cache/kagglehub/datasets/orvile/pmram-bangladeshi-brain-cancer-mri-dataset/versions/2/PMRAM Bangladeshi Brain Cancer - MRI Dataset/PMRAM Bangladeshi Brain Cancer - MRI Dataset/Raw Data/Raw/512Meningioma/M_8.jpg computed_with_main_split_code_on_downloaded_kaggle_files
        DS1 train meningioma          M_251.jpg 6b303edae8c3c4aadfb211a388fa33ab    /root/.cache/kagglehub/datasets/orvile/pmram-bangladeshi-brain-cancer-mri-dataset/versions/2/PMRAM Bangladeshi Brain Cancer - MRI Dataset/PMRAM Bangladeshi Brain Cancer - MRI Dataset/Raw Data/Raw/512Meningioma/M_251.jpg computed_with_main_split_code_on_downloaded_kaggle_files
        DS1  test    notumor   normal (170).jpg 6dec8aac2278da4e1b9ac1387edcb16c /root/.cache/kagglehub/datasets/orvile/pmram-bangladeshi-brain-cancer-mri-dataset/versions/2/PMRAM Bangladeshi Brain Cancer - MRI Dataset/PMRAM Bangladeshi Brain Cancer - MRI Dataset/Raw Data/Raw/512Normal/normal (170).jpg computed_with_main_split_code_on_downloaded_kaggle_files
        DS1 train    notumor     normal (6).jpg 6dec8aac2278da4e1b9ac1387edcb16c   /root/.cache/kagglehub/datasets/orvile/pmram-bangladeshi-brain-cancer-mri-dataset/versions/2/PMRAM Bangladeshi Brain Cancer - MRI Dataset/PMRAM Bangladeshi Brain Cancer - MRI Dataset/Raw Data/Raw/512Normal/normal (6).jpg computed_with_main_split_code_on_downloaded_kaggle_files
        DS1  test meningioma          M_124.jpg 8795d3c2c1a2aabbb559c75c846f4549    /root/.cache/kagglehub/datasets/orvile/pmram-bangladeshi-brain-cancer-mri-dataset/versions/2/PMRAM Bangladeshi Brain Cancer - MRI Dataset/PMRAM Bangladeshi Brain Cancer - MRI Dataset/Raw Data/Raw/512Meningioma/M_124.jpg computed_with_main_split_code_on_downloaded_kaggle_files
        DS1 train meningioma          M_144.jpg 8795d3c2c1a2aabbb559c75c846f4549    /root/.cache/kagglehub/datasets/orvile/pmram-bangladeshi-brain-cancer-mri-dataset/versions/2/PMRAM Bangladeshi Brain Cancer - MRI Dataset/PMRAM Bangladeshi Brain Cancer - MRI Dataset/Raw Data/Raw/512Meningioma/M_144.jpg computed_with_main_split_code_on_downloaded_kaggle_files
        DS1 train    notumor   normal (358).jpg 91f05f53f646130c5d209e4f5bd79bf4 /root/.cache/kagglehub/datasets/orvile/pmram-bangladeshi-brain-cancer-mri-dataset/versions/2/PMRAM Bangladeshi Brain Cancer - MRI Dataset/PMRAM Bangladeshi Brain Cancer - MRI Dataset/Raw Data/Raw/512Normal/normal (358).jpg computed_with_main_split_code_on_downloaded_kaggle_files
        DS1   val    notumor   normal (352).jpg 91f05f53f646130c5d209e4f5bd79bf4 /root/.cache/kagglehub/datasets/orvile/pmram-bangladeshi-brain-cancer-mri-dataset/versions/2/PMRAM Bangladeshi Brain Cancer - MRI Dataset/PMRAM Bangladeshi Brain Cancer - MRI Dataset/Raw Data/Raw/512Normal/normal (352).jpg computed_with_main_split_code_on_downloaded_kaggle_files
        DS1  test meningioma          M_159.jpg 95ccaf7407781c2238f1858d2653af5e    /root/.cache/kagglehub/datasets/orvile/pmram-bangladeshi-brain-cancer-mri-dataset/versions/2/PMRAM Bangladeshi Brain Cancer - MRI Dataset/PMRAM Bangladeshi Brain Cancer - MRI Dataset/Raw Data/Raw/512Meningioma/M_159.jpg computed_with_main_split_code_on_downloaded_kaggle_files
        DS1 train meningioma          M_112.jpg 95ccaf7407781c2238f1858d2653af5e    /root/.cache/kagglehub/datasets/orvile/pmram-bangladeshi-brain-cancer-mri-dataset/versions/2/PMRAM Bangladeshi Brain Cancer - MRI Dataset/PMRAM Bangladeshi Brain Cancer - MRI Dataset/Raw Data/Raw/512Meningioma/M_112.jpg computed_with_main_split_code_on_downloaded_kaggle_files
        DS1 train    notumor   normal (250).jpg 9b3e4c42a0adeef59a5cedcfba0b78cb /root/.cache/kagglehub/datasets/orvile/pmram-bangladeshi-brain-cancer-mri-dataset/versions/2/PMRAM Bangladeshi Brain Cancer - MRI Dataset/PMRAM Bangladeshi Brain Cancer - MRI Dataset/Raw Data/Raw/512Normal/normal (250).jpg computed_with_main_split_code_on_downloaded_kaggle_files
        DS1   val    notumor    normal (21).jpg 9b3e4c42a0adeef59a5cedcfba0b78cb  /root/.cache/kagglehub/datasets/orvile/pmram-bangladeshi-brain-cancer-mri-dataset/versions/2/PMRAM Bangladeshi Brain Cancer - MRI Dataset/PMRAM Bangladeshi Brain Cancer - MRI Dataset/Raw Data/Raw/512Normal/normal (21).jpg computed_with_main_split_code_on_downloaded_kaggle_files
        DS1  test    notumor   normal (332).jpg 9e80ac169ae0b8129cc41939fa0bafa3 /root/.cache/kagglehub/datasets/orvile/pmram-bangladeshi-brain-cancer-mri-dataset/versions/2/PMRAM Bangladeshi Brain Cancer - MRI Dataset/PMRAM Bangladeshi Brain Cancer - MRI Dataset/Raw Data/Raw/512Normal/normal (332).jpg computed_with_main_split_code_on_downloaded_kaggle_files
        DS1 train    notumor   normal (347).jpg 9e80ac169ae0b8129cc41939fa0bafa3 /root/.cache/kagglehub/datasets/orvile/pmram-bangladeshi-brain-cancer-mri-dataset/versions/2/PMRAM Bangladeshi Brain Cancer - MRI Dataset/PMRAM Bangladeshi Brain Cancer - MRI Dataset/Raw Data/Raw/512Normal/normal (347).jpg computed_with_main_split_code_on_downloaded_kaggle_files
        DS1  test    notumor   normal (276).jpg a350100047d5befbda844c9055abbe66 /root/.cache/kagglehub/datasets/orvile/pmram-bangladeshi-brain-cancer-mri-dataset/versions/2/PMRAM Bangladeshi Brain Cancer - MRI Dataset/PMRAM Bangladeshi Brain Cancer - MRI Dataset/Raw Data/Raw/512Normal/normal (276).jpg computed_with_main_split_code_on_downloaded_kaggle_files
        DS1 train    notumor   normal (281).jpg a350100047d5befbda844c9055abbe66 /root/.cache/kagglehub/datasets/orvile/pmram-bangladeshi-brain-cancer-mri-dataset/versions/2/PMRAM Bangladeshi Brain Cancer - MRI Dataset/PMRAM Bangladeshi Brain Cancer - MRI Dataset/Raw Data/Raw/512Normal/normal (281).jpg computed_with_main_split_code_on_downloaded_kaggle_files
        DS1 train    notumor   normal (294).jpg a350100047d5befbda844c9055abbe66 /root/.cache/kagglehub/datasets/orvile/pmram-bangladeshi-brain-cancer-mri-dataset/versions/2/PMRAM Bangladeshi Brain Cancer - MRI Dataset/PMRAM Bangladeshi Brain Cancer - MRI Dataset/Raw Data/Raw/512Normal/normal (294).jpg computed_with_main_split_code_on_downloaded_kaggle_files
        DS1  test meningioma          M_248.jpg aa47ac12377eb4f98469be783012fd15    /root/.cache/kagglehub/datasets/orvile/pmram-bangladeshi-brain-cancer-mri-dataset/versions/2/PMRAM Bangladeshi Brain Cancer - MRI Dataset/PMRAM Bangladeshi Brain Cancer - MRI Dataset/Raw Data/Raw/512Meningioma/M_248.jpg computed_with_main_split_code_on_downloaded_kaggle_files
        DS1   val meningioma            M_5.jpg aa47ac12377eb4f98469be783012fd15      /root/.cache/kagglehub/datasets/orvile/pmram-bangladeshi-brain-cancer-mri-dataset/versions/2/PMRAM Bangladeshi Brain Cancer - MRI Dataset/PMRAM Bangladeshi Brain Cancer - MRI Dataset/Raw Data/Raw/512Meningioma/M_5.jpg computed_with_main_split_code_on_downloaded_kaggle_files
        DS1  test    notumor   normal (202).jpg b333cabc7f301686fecc68e690c42189 /root/.cache/kagglehub/datasets/orvile/pmram-bangladeshi-brain-cancer-mri-dataset/versions/2/PMRAM Bangladeshi Brain Cancer - MRI Dataset/PMRAM Bangladeshi Brain Cancer - MRI Dataset/Raw Data/Raw/512Normal/normal (202).jpg computed_with_main_split_code_on_downloaded_kaggle_files
        DS1   val    notumor    normal (12).jpg b333cabc7f301686fecc68e690c42189  /root/.cache/kagglehub/datasets/orvile/pmram-bangladeshi-brain-cancer-mri-dataset/versions/2/PMRAM Bangladeshi Brain Cancer - MRI Dataset/PMRAM Bangladeshi Brain Cancer - MRI Dataset/Raw Data/Raw/512Normal/normal (12).jpg computed_with_main_split_code_on_downloaded_kaggle_files
        DS1 train    notumor   normal (340).jpg b985091648e4137b16027b096231de1d /root/.cache/kagglehub/datasets/orvile/pmram-bangladeshi-brain-cancer-mri-dataset/versions/2/PMRAM Bangladeshi Brain Cancer - MRI Dataset/PMRAM Bangladeshi Brain Cancer - MRI Dataset/Raw Data/Raw/512Normal/normal (340).jpg computed_with_main_split_code_on_downloaded_kaggle_files
        DS1   val    notumor   normal (303).jpg b985091648e4137b16027b096231de1d /root/.cache/kagglehub/datasets/orvile/pmram-bangladeshi-brain-cancer-mri-dataset/versions/2/PMRAM Bangladeshi Brain Cancer - MRI Dataset/PMRAM Bangladeshi Brain Cancer - MRI Dataset/Raw Data/Raw/512Normal/normal (303).jpg computed_with_main_split_code_on_downloaded_kaggle_files
        DS1 train meningioma          M_169.jpg ba3045f533bdb44e66ce67788f8903af    /root/.cache/kagglehub/datasets/orvile/pmram-bangladeshi-brain-cancer-mri-dataset/versions/2/PMRAM Bangladeshi Brain Cancer - MRI Dataset/PMRAM Bangladeshi Brain Cancer - MRI Dataset/Raw Data/Raw/512Meningioma/M_169.jpg computed_with_main_split_code_on_downloaded_kaggle_files
        DS1   val meningioma          M_212.jpg ba3045f533bdb44e66ce67788f8903af    /root/.cache/kagglehub/datasets/orvile/pmram-bangladeshi-brain-cancer-mri-dataset/versions/2/PMRAM Bangladeshi Brain Cancer - MRI Dataset/PMRAM Bangladeshi Brain Cancer - MRI Dataset/Raw Data/Raw/512Meningioma/M_212.jpg computed_with_main_split_code_on_downloaded_kaggle_files
        DS1  test    notumor   normal (265).jpg bcdaaf419af925b4df266a78c626e70e /root/.cache/kagglehub/datasets/orvile/pmram-bangladeshi-brain-cancer-mri-dataset/versions/2/PMRAM Bangladeshi Brain Cancer - MRI Dataset/PMRAM Bangladeshi Brain Cancer - MRI Dataset/Raw Data/Raw/512Normal/normal (265).jpg computed_with_main_split_code_on_downloaded_kaggle_files
        DS1 train    notumor   normal (264).jpg bcdaaf419af925b4df266a78c626e70e /root/.cache/kagglehub/datasets/orvile/pmram-bangladeshi-brain-cancer-mri-dataset/versions/2/PMRAM Bangladeshi Brain Cancer - MRI Dataset/PMRAM Bangladeshi Brain Cancer - MRI Dataset/Raw Data/Raw/512Normal/normal (264).jpg computed_with_main_split_code_on_downloaded_kaggle_files
        DS1 train meningioma            M_7.jpg bf80e340a7b4fcf553202847221bfe1f      /root/.cache/kagglehub/datasets/orvile/pmram-bangladeshi-brain-cancer-mri-dataset/versions/2/PMRAM Bangladeshi Brain Cancer - MRI Dataset/PMRAM Bangladeshi Brain Cancer - MRI Dataset/Raw Data/Raw/512Meningioma/M_7.jpg computed_with_main_split_code_on_downloaded_kaggle_files
        DS1   val meningioma          M_250.jpg bf80e340a7b4fcf553202847221bfe1f    /root/.cache/kagglehub/datasets/orvile/pmram-bangladeshi-brain-cancer-mri-dataset/versions/2/PMRAM Bangladeshi Brain Cancer - MRI Dataset/PMRAM Bangladeshi Brain Cancer - MRI Dataset/Raw Data/Raw/512Meningioma/M_250.jpg computed_with_main_split_code_on_downloaded_kaggle_files
        DS1  test    notumor    normal (15).jpg c33d9bfe54114ba89a9aefbd2f913e84  /root/.cache/kagglehub/datasets/orvile/pmram-bangladeshi-brain-cancer-mri-dataset/versions/2/PMRAM Bangladeshi Brain Cancer - MRI Dataset/PMRAM Bangladeshi Brain Cancer - MRI Dataset/Raw Data/Raw/512Normal/normal (15).jpg computed_with_main_split_code_on_downloaded_kaggle_files
        DS1 train    notumor   normal (218).jpg c33d9bfe54114ba89a9aefbd2f913e84 /root/.cache/kagglehub/datasets/orvile/pmram-bangladeshi-brain-cancer-mri-dataset/versions/2/PMRAM Bangladeshi Brain Cancer - MRI Dataset/PMRAM Bangladeshi Brain Cancer - MRI Dataset/Raw Data/Raw/512Normal/normal (218).jpg computed_with_main_split_code_on_downloaded_kaggle_files
        DS1  test    notumor   normal (277).jpg cc459212e1751891fafd5f5a8803952f /root/.cache/kagglehub/datasets/orvile/pmram-bangladeshi-brain-cancer-mri-dataset/versions/2/PMRAM Bangladeshi Brain Cancer - MRI Dataset/PMRAM Bangladeshi Brain Cancer - MRI Dataset/Raw Data/Raw/512Normal/normal (277).jpg computed_with_main_split_code_on_downloaded_kaggle_files
        DS1 train    notumor   normal (280).jpg cc459212e1751891fafd5f5a8803952f /root/.cache/kagglehub/datasets/orvile/pmram-bangladeshi-brain-cancer-mri-dataset/versions/2/PMRAM Bangladeshi Brain Cancer - MRI Dataset/PMRAM Bangladeshi Brain Cancer - MRI Dataset/Raw Data/Raw/512Normal/normal (280).jpg computed_with_main_split_code_on_downloaded_kaggle_files
        DS1  test meningioma          M_368.jpg d3af767935fc181fae0941a42a3dc4fb    /root/.cache/kagglehub/datasets/orvile/pmram-bangladeshi-brain-cancer-mri-dataset/versions/2/PMRAM Bangladeshi Brain Cancer - MRI Dataset/PMRAM Bangladeshi Brain Cancer - MRI Dataset/Raw Data/Raw/512Meningioma/M_368.jpg computed_with_main_split_code_on_downloaded_kaggle_files
        DS1 train meningioma          M_183.jpg d3af767935fc181fae0941a42a3dc4fb    /root/.cache/kagglehub/datasets/orvile/pmram-bangladeshi-brain-cancer-mri-dataset/versions/2/PMRAM Bangladeshi Brain Cancer - MRI Dataset/PMRAM Bangladeshi Brain Cancer - MRI Dataset/Raw Data/Raw/512Meningioma/M_183.jpg computed_with_main_split_code_on_downloaded_kaggle_files
        DS1  test    notumor   normal (308).jpg d4272507eb73bd6127bac8ad2c3d1057 /root/.cache/kagglehub/datasets/orvile/pmram-bangladeshi-brain-cancer-mri-dataset/versions/2/PMRAM Bangladeshi Brain Cancer - MRI Dataset/PMRAM Bangladeshi Brain Cancer - MRI Dataset/Raw Data/Raw/512Normal/normal (308).jpg computed_with_main_split_code_on_downloaded_kaggle_files
        DS1  test    notumor   normal (322).jpg d4272507eb73bd6127bac8ad2c3d1057 /root/.cache/kagglehub/datasets/orvile/pmram-bangladeshi-brain-cancer-mri-dataset/versions/2/PMRAM Bangladeshi Brain Cancer - MRI Dataset/PMRAM Bangladeshi Brain Cancer - MRI Dataset/Raw Data/Raw/512Normal/normal (322).jpg computed_with_main_split_code_on_downloaded_kaggle_files
        DS1 train    notumor   normal (386).jpg d4272507eb73bd6127bac8ad2c3d1057 /root/.cache/kagglehub/datasets/orvile/pmram-bangladeshi-brain-cancer-mri-dataset/versions/2/PMRAM Bangladeshi Brain Cancer - MRI Dataset/PMRAM Bangladeshi Brain Cancer - MRI Dataset/Raw Data/Raw/512Normal/normal (386).jpg computed_with_main_split_code_on_downloaded_kaggle_files
        DS1 train    notumor   normal (189).jpg d8e9b8048e0681b93face9b065e31037 /root/.cache/kagglehub/datasets/orvile/pmram-bangladeshi-brain-cancer-mri-dataset/versions/2/PMRAM Bangladeshi Brain Cancer - MRI Dataset/PMRAM Bangladeshi Brain Cancer - MRI Dataset/Raw Data/Raw/512Normal/normal (189).jpg computed_with_main_split_code_on_downloaded_kaggle_files
        DS1   val    notumor    normal (65).jpg d8e9b8048e0681b93face9b065e31037  /root/.cache/kagglehub/datasets/orvile/pmram-bangladeshi-brain-cancer-mri-dataset/versions/2/PMRAM Bangladeshi Brain Cancer - MRI Dataset/PMRAM Bangladeshi Brain Cancer - MRI Dataset/Raw Data/Raw/512Normal/normal (65).jpg computed_with_main_split_code_on_downloaded_kaggle_files
        DS1 train    notumor   normal (301).jpg db17176dade907bdf5c4e179d4543d56 /root/.cache/kagglehub/datasets/orvile/pmram-bangladeshi-brain-cancer-mri-dataset/versions/2/PMRAM Bangladeshi Brain Cancer - MRI Dataset/PMRAM Bangladeshi Brain Cancer - MRI Dataset/Raw Data/Raw/512Normal/normal (301).jpg computed_with_main_split_code_on_downloaded_kaggle_files
        DS1 train    notumor   normal (307).jpg db17176dade907bdf5c4e179d4543d56 /root/.cache/kagglehub/datasets/orvile/pmram-bangladeshi-brain-cancer-mri-dataset/versions/2/PMRAM Bangladeshi Brain Cancer - MRI Dataset/PMRAM Bangladeshi Brain Cancer - MRI Dataset/Raw Data/Raw/512Normal/normal (307).jpg computed_with_main_split_code_on_downloaded_kaggle_files
        DS1   val    notumor   normal (300).jpg db17176dade907bdf5c4e179d4543d56 /root/.cache/kagglehub/datasets/orvile/pmram-bangladeshi-brain-cancer-mri-dataset/versions/2/PMRAM Bangladeshi Brain Cancer - MRI Dataset/PMRAM Bangladeshi Brain Cancer - MRI Dataset/Raw Data/Raw/512Normal/normal (300).jpg computed_with_main_split_code_on_downloaded_kaggle_files
        DS1 train meningioma          M_171.jpg e612cf796dea4a29a23cf68178fd41a3    /root/.cache/kagglehub/datasets/orvile/pmram-bangladeshi-brain-cancer-mri-dataset/versions/2/PMRAM Bangladeshi Brain Cancer - MRI Dataset/PMRAM Bangladeshi Brain Cancer - MRI Dataset/Raw Data/Raw/512Meningioma/M_171.jpg computed_with_main_split_code_on_downloaded_kaggle_files
        DS1   val meningioma          M_235.jpg e612cf796dea4a29a23cf68178fd41a3    /root/.cache/kagglehub/datasets/orvile/pmram-bangladeshi-brain-cancer-mri-dataset/versions/2/PMRAM Bangladeshi Brain Cancer - MRI Dataset/PMRAM Bangladeshi Brain Cancer - MRI Dataset/Raw Data/Raw/512Meningioma/M_235.jpg computed_with_main_split_code_on_downloaded_kaggle_files
        DS1 train    notumor   normal (201).jpg eaeb102cd99661f157923b9eff689053 /root/.cache/kagglehub/datasets/orvile/pmram-bangladeshi-brain-cancer-mri-dataset/versions/2/PMRAM Bangladeshi Brain Cancer - MRI Dataset/PMRAM Bangladeshi Brain Cancer - MRI Dataset/Raw Data/Raw/512Normal/normal (201).jpg computed_with_main_split_code_on_downloaded_kaggle_files
        DS1   val    notumor    normal (11).jpg eaeb102cd99661f157923b9eff689053  /root/.cache/kagglehub/datasets/orvile/pmram-bangladeshi-brain-cancer-mri-dataset/versions/2/PMRAM Bangladeshi Brain Cancer - MRI Dataset/PMRAM Bangladeshi Brain Cancer - MRI Dataset/Raw Data/Raw/512Normal/normal (11).jpg computed_with_main_split_code_on_downloaded_kaggle_files
        DS1 train    notumor    normal (19).jpg f95511593d8fc69f9f0a7197cb1df0dd  /root/.cache/kagglehub/datasets/orvile/pmram-bangladeshi-brain-cancer-mri-dataset/versions/2/PMRAM Bangladeshi Brain Cancer - MRI Dataset/PMRAM Bangladeshi Brain Cancer - MRI Dataset/Raw Data/Raw/512Normal/normal (19).jpg computed_with_main_split_code_on_downloaded_kaggle_files
        DS1   val    notumor   normal (248).jpg f95511593d8fc69f9f0a7197cb1df0dd /root/.cache/kagglehub/datasets/orvile/pmram-bangladeshi-brain-cancer-mri-dataset/versions/2/PMRAM Bangladeshi Brain Cancer - MRI Dataset/PMRAM Bangladeshi Brain Cancer - MRI Dataset/Raw Data/Raw/512Normal/normal (248).jpg computed_with_main_split_code_on_downloaded_kaggle_files
        DS2 train  pituitary     Tr-pi_0038.jpg 0070782338a1865c8255e2a1547dc1b8                                                                                                        /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/pituitary/Tr-pi_0038.jpg computed_with_main_split_code_on_downloaded_kaggle_files
        DS2   val  pituitary Tr-pi_0038 (1).jpg 0070782338a1865c8255e2a1547dc1b8                                                                                                    /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/pituitary/Tr-pi_0038 (1).jpg computed_with_main_split_code_on_downloaded_kaggle_files
        DS2 train    notumor     Tr-no_1067.jpg 019132081b0bb280fb4f31a0a54e9800                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Tr-no_1067.jpg computed_with_main_split_code_on_downloaded_kaggle_files
        DS2   val    notumor     Tr-no_0025.jpg 019132081b0bb280fb4f31a0a54e9800                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Tr-no_0025.jpg computed_with_main_split_code_on_downloaded_kaggle_files
        DS2  test    notumor     Te-no_0044.jpg 11241b33d38b730c2ae2d3a558b0076f                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Te-no_0044.jpg computed_with_main_split_code_on_downloaded_kaggle_files
        DS2  test    notumor     Tr-no_0180.jpg 11241b33d38b730c2ae2d3a558b0076f                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Tr-no_0180.jpg computed_with_main_split_code_on_downloaded_kaggle_files
        DS2  test    notumor     Tr-no_0324.jpg 11241b33d38b730c2ae2d3a558b0076f                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Tr-no_0324.jpg computed_with_main_split_code_on_downloaded_kaggle_files
        DS2 train    notumor     Te-no_0055.jpg 11241b33d38b730c2ae2d3a558b0076f                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Te-no_0055.jpg computed_with_main_split_code_on_downloaded_kaggle_files
        DS2 train    notumor     Te-no_0061.jpg 11241b33d38b730c2ae2d3a558b0076f                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Te-no_0061.jpg computed_with_main_split_code_on_downloaded_kaggle_files
        DS2 train    notumor     Tr-no_0057.jpg 11241b33d38b730c2ae2d3a558b0076f                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Tr-no_0057.jpg computed_with_main_split_code_on_downloaded_kaggle_files
        DS2 train    notumor     Tr-no_0291.jpg 11241b33d38b730c2ae2d3a558b0076f                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Tr-no_0291.jpg computed_with_main_split_code_on_downloaded_kaggle_files
        DS2 train    notumor     Tr-no_0984.jpg 11241b33d38b730c2ae2d3a558b0076f                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Tr-no_0984.jpg computed_with_main_split_code_on_downloaded_kaggle_files
        DS2   val    notumor     Te-no_0012.jpg 11241b33d38b730c2ae2d3a558b0076f                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Te-no_0012.jpg computed_with_main_split_code_on_downloaded_kaggle_files
        DS2 train meningioma   Te-meTr_0007.jpg 135243b66c75170a0ccbcea602f05425                                                                                                     /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/meningioma/Te-meTr_0007.jpg computed_with_main_split_code_on_downloaded_kaggle_files
        DS2   val meningioma     Tr-me_0197.jpg 135243b66c75170a0ccbcea602f05425                                                                                                       /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/meningioma/Tr-me_0197.jpg computed_with_main_split_code_on_downloaded_kaggle_files
        DS2  test    notumor     Tr-no_0279.jpg 1618568d42bc8954dc6313cb9d9882b3                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Tr-no_0279.jpg computed_with_main_split_code_on_downloaded_kaggle_files
        DS2 train    notumor     Te-no_0054.jpg 1618568d42bc8954dc6313cb9d9882b3                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Te-no_0054.jpg computed_with_main_split_code_on_downloaded_kaggle_files
        DS2 train    notumor     Tr-no_0191.jpg 1618568d42bc8954dc6313cb9d9882b3                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Tr-no_0191.jpg computed_with_main_split_code_on_downloaded_kaggle_files
        DS2 train    notumor     Tr-no_0235.jpg 1618568d42bc8954dc6313cb9d9882b3                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Tr-no_0235.jpg computed_with_main_split_code_on_downloaded_kaggle_files
        DS2   val    notumor     Te-no_0045.jpg 1618568d42bc8954dc6313cb9d9882b3                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Te-no_0045.jpg computed_with_main_split_code_on_downloaded_kaggle_files
        DS2   val    notumor     Te-no_0050.jpg 1618568d42bc8954dc6313cb9d9882b3                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Te-no_0050.jpg computed_with_main_split_code_on_downloaded_kaggle_files
        DS2 train    notumor     Tr-no_0154.jpg 1899b75330af3e5ec56b3ed85610d23a                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Tr-no_0154.jpg computed_with_main_split_code_on_downloaded_kaggle_files
        DS2   val    notumor     Tr-no_0237.jpg 1899b75330af3e5ec56b3ed85610d23a                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Tr-no_0237.jpg computed_with_main_split_code_on_downloaded_kaggle_files
        DS2 train    notumor     Tr-no_0160.jpg 232b1749e4d955824b8628b8a1818c3c                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Tr-no_0160.jpg computed_with_main_split_code_on_downloaded_kaggle_files
        DS2   val    notumor     Tr-no_1115.jpg 232b1749e4d955824b8628b8a1818c3c                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Tr-no_1115.jpg computed_with_main_split_code_on_downloaded_kaggle_files
        DS2 train meningioma     Te-me_0019.jpg 243c325a3404d60b75fde90f2ad22a41                                                                                                       /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/meningioma/Te-me_0019.jpg computed_with_main_split_code_on_downloaded_kaggle_files
        DS2   val meningioma     Tr-me_0199.jpg 243c325a3404d60b75fde90f2ad22a41                                                                                                       /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/meningioma/Tr-me_0199.jpg computed_with_main_split_code_on_downloaded_kaggle_files
        DS2 train    notumor     Tr-no_0140.jpg 2dfa6e329951bec1ac0c7e91f97feb6c                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Tr-no_0140.jpg computed_with_main_split_code_on_downloaded_kaggle_files
        DS2 train    notumor     Tr-no_0223.jpg 2dfa6e329951bec1ac0c7e91f97feb6c                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Tr-no_0223.jpg computed_with_main_split_code_on_downloaded_kaggle_files
        DS2   val    notumor     Tr-no_1101.jpg 2dfa6e329951bec1ac0c7e91f97feb6c                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Tr-no_1101.jpg computed_with_main_split_code_on_downloaded_kaggle_files
        DS2  test    notumor     Tr-no_0052.jpg 3249b0c9246ab9b0f7dd898097615281                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Tr-no_0052.jpg computed_with_main_split_code_on_downloaded_kaggle_files
        DS2 train    notumor   Te-noTr_0007.jpg 3249b0c9246ab9b0f7dd898097615281                                                                                                        /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Te-noTr_0007.jpg computed_with_main_split_code_on_downloaded_kaggle_files
        DS2 train    notumor     Tr-no_1089.jpg 3249b0c9246ab9b0f7dd898097615281                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Tr-no_1089.jpg computed_with_main_split_code_on_downloaded_kaggle_files
        DS2   val    notumor     Te-no_0088.jpg 3249b0c9246ab9b0f7dd898097615281                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Te-no_0088.jpg computed_with_main_split_code_on_downloaded_kaggle_files
        DS2   val    notumor     Tr-no_0352.jpg 3249b0c9246ab9b0f7dd898097615281                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Tr-no_0352.jpg computed_with_main_split_code_on_downloaded_kaggle_files
        DS2  test    notumor     Tr-no_0974.jpg 35524e36a91606f5e476efc87ac3b785                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Tr-no_0974.jpg computed_with_main_split_code_on_downloaded_kaggle_files
        DS2 train    notumor     Te-no_0070.jpg 35524e36a91606f5e476efc87ac3b785                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Te-no_0070.jpg computed_with_main_split_code_on_downloaded_kaggle_files
        DS2 train    notumor     Tr-no_0063.jpg 35524e36a91606f5e476efc87ac3b785                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Tr-no_0063.jpg computed_with_main_split_code_on_downloaded_kaggle_files
        DS2 train    notumor     Tr-no_0334.jpg 35524e36a91606f5e476efc87ac3b785                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Tr-no_0334.jpg computed_with_main_split_code_on_downloaded_kaggle_files
        DS2   val    notumor     Te-no_0018.jpg 35524e36a91606f5e476efc87ac3b785                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Te-no_0018.jpg computed_with_main_split_code_on_downloaded_kaggle_files
        DS2 train  pituitary Tr-pi_0220 (1).jpg 497fb246ddc02264212f2fca8e36179d                                                                                                    /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/pituitary/Tr-pi_0220 (1).jpg computed_with_main_split_code_on_downloaded_kaggle_files
        DS2   val  pituitary     Tr-pi_0220.jpg 497fb246ddc02264212f2fca8e36179d                                                                                                        /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/pituitary/Tr-pi_0220.jpg computed_with_main_split_code_on_downloaded_kaggle_files
        DS2 train meningioma     Te-me_0143.jpg 4e8aac352b46fa2921dcb14e3ead3d57                                                                                                       /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/meningioma/Te-me_0143.jpg computed_with_main_split_code_on_downloaded_kaggle_files
        DS2 train meningioma     Tr-me_0283.jpg 4e8aac352b46fa2921dcb14e3ead3d57                                                                                                       /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/meningioma/Tr-me_0283.jpg computed_with_main_split_code_on_downloaded_kaggle_files
        DS2   val meningioma     Te-me_0106.jpg 4e8aac352b46fa2921dcb14e3ead3d57                                                                                                       /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/meningioma/Te-me_0106.jpg computed_with_main_split_code_on_downloaded_kaggle_files
        DS2 train  pituitary     Tr-pi_0159.jpg 56134d19acce34b4368ac862d1e9a262                                                                                                        /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/pituitary/Tr-pi_0159.jpg computed_with_main_split_code_on_downloaded_kaggle_files
        DS2   val  pituitary     Tr-pi_0160.jpg 56134d19acce34b4368ac862d1e9a262                                                                                                        /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/pituitary/Tr-pi_0160.jpg computed_with_main_split_code_on_downloaded_kaggle_files
        DS2 train    notumor     Tr-no_1108.jpg 5fac8346394280e1d84c34e109674967                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Tr-no_1108.jpg computed_with_main_split_code_on_downloaded_kaggle_files
        DS2   val    notumor     Tr-no_0153.jpg 5fac8346394280e1d84c34e109674967                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Tr-no_0153.jpg computed_with_main_split_code_on_downloaded_kaggle_files
        DS2 train    notumor     Te-no_0060.jpg 6066da064cf0ce60cd44b632a8869fd0                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Te-no_0060.jpg computed_with_main_split_code_on_downloaded_kaggle_files
        DS2 train    notumor     Tr-no_0967.jpg 6066da064cf0ce60cd44b632a8869fd0                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Tr-no_0967.jpg computed_with_main_split_code_on_downloaded_kaggle_files
        DS2   val    notumor     Tr-no_0046.jpg 6066da064cf0ce60cd44b632a8869fd0                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Tr-no_0046.jpg computed_with_main_split_code_on_downloaded_kaggle_files
        DS2 train meningioma   Te-meTr_0004.jpg 6cf76c6f8ed017c4089a69f62b04c536                                                                                                     /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/meningioma/Te-meTr_0004.jpg computed_with_main_split_code_on_downloaded_kaggle_files
        DS2   val meningioma     Tr-me_0595.jpg 6cf76c6f8ed017c4089a69f62b04c536                                                                                                       /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/meningioma/Tr-me_0595.jpg computed_with_main_split_code_on_downloaded_kaggle_files
        DS2 train meningioma     Tr-me_0182.jpg 7105119db06153a5852bad7a1923fece                                                                                                       /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/meningioma/Tr-me_0182.jpg computed_with_main_split_code_on_downloaded_kaggle_files
        DS2   val meningioma     Te-me_0015.jpg 7105119db06153a5852bad7a1923fece                                                                                                       /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/meningioma/Te-me_0015.jpg computed_with_main_split_code_on_downloaded_kaggle_files
        DS2 train meningioma     Te-me_0083.jpg 7316754f4d41ed129f05b877a7b8840c                                                                                                       /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/meningioma/Te-me_0083.jpg computed_with_main_split_code_on_downloaded_kaggle_files
        DS2   val meningioma     Tr-me_0260.jpg 7316754f4d41ed129f05b877a7b8840c                                                                                                       /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/meningioma/Tr-me_0260.jpg computed_with_main_split_code_on_downloaded_kaggle_files
        DS2  test    notumor     Tr-no_0125.jpg 788dfc769b6f8390eacb88277de5cdb5                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Tr-no_0125.jpg computed_with_main_split_code_on_downloaded_kaggle_files
        DS2 train    notumor     Tr-no_0365.jpg 788dfc769b6f8390eacb88277de5cdb5                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Tr-no_0365.jpg computed_with_main_split_code_on_downloaded_kaggle_files
        DS2 train    notumor     Tr-no_0368.jpg 788dfc769b6f8390eacb88277de5cdb5                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Tr-no_0368.jpg computed_with_main_split_code_on_downloaded_kaggle_files
        DS2   val    notumor     Tr-no_0120.jpg 788dfc769b6f8390eacb88277de5cdb5                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Tr-no_0120.jpg computed_with_main_split_code_on_downloaded_kaggle_files
        DS2 train    notumor     Tr-no_0308.jpg 7b1ba98b6fc11f4e0cf936daf4ab9b25                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Tr-no_0308.jpg computed_with_main_split_code_on_downloaded_kaggle_files
        DS2   val    notumor     Tr-no_1164.jpg 7b1ba98b6fc11f4e0cf936daf4ab9b25                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Tr-no_1164.jpg computed_with_main_split_code_on_downloaded_kaggle_files
        DS2  test    notumor     Tr-no_0373.jpg 8db27552e259ba3274490ee4a8a7f583                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Tr-no_0373.jpg computed_with_main_split_code_on_downloaded_kaggle_files
        DS2 train    notumor   Te-noTr_0001.jpg 8db27552e259ba3274490ee4a8a7f583                                                                                                        /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Te-noTr_0001.jpg computed_with_main_split_code_on_downloaded_kaggle_files
        DS2 train    notumor     Te-no_0031.jpg 8db27552e259ba3274490ee4a8a7f583                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Te-no_0031.jpg computed_with_main_split_code_on_downloaded_kaggle_files
        DS2 train    notumor     Tr-no_0051.jpg 8db27552e259ba3274490ee4a8a7f583                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Tr-no_0051.jpg computed_with_main_split_code_on_downloaded_kaggle_files
        DS2 train    notumor     Tr-no_0331.jpg 8db27552e259ba3274490ee4a8a7f583                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Tr-no_0331.jpg computed_with_main_split_code_on_downloaded_kaggle_files
        DS2 train    notumor     Tr-no_0347.jpg 8db27552e259ba3274490ee4a8a7f583                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Tr-no_0347.jpg computed_with_main_split_code_on_downloaded_kaggle_files
        DS2 train    notumor     Tr-no_1088.jpg 8db27552e259ba3274490ee4a8a7f583                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Tr-no_1088.jpg computed_with_main_split_code_on_downloaded_kaggle_files
        DS2   val    notumor     Te-no_0067.jpg 8db27552e259ba3274490ee4a8a7f583                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Te-no_0067.jpg computed_with_main_split_code_on_downloaded_kaggle_files
        DS2   val    notumor     Te-no_0084.jpg 8db27552e259ba3274490ee4a8a7f583                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Te-no_0084.jpg computed_with_main_split_code_on_downloaded_kaggle_files
        DS2 train  pituitary     Tr-pi_0526.jpg 9024bacc218fc0e4d4d8cbe5935ea21c                                                                                                        /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/pituitary/Tr-pi_0526.jpg computed_with_main_split_code_on_downloaded_kaggle_files
        DS2   val  pituitary     Tr-pi_0527.jpg 9024bacc218fc0e4d4d8cbe5935ea21c                                                                                                        /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/pituitary/Tr-pi_0527.jpg computed_with_main_split_code_on_downloaded_kaggle_files
        DS2   val  pituitary     Tr-pi_0529.jpg 9024bacc218fc0e4d4d8cbe5935ea21c                                                                                                        /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/pituitary/Tr-pi_0529.jpg computed_with_main_split_code_on_downloaded_kaggle_files
        DS2 train meningioma     Tr-me_0257.jpg 9bb66fd981c609c149f13754f1cef20d                                                                                                       /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/meningioma/Tr-me_0257.jpg computed_with_main_split_code_on_downloaded_kaggle_files
        DS2   val meningioma     Te-me_0079.jpg 9bb66fd981c609c149f13754f1cef20d                                                                                                       /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/meningioma/Te-me_0079.jpg computed_with_main_split_code_on_downloaded_kaggle_files
        DS2  test    notumor     Tr-no_1153.jpg a3f433913a6ec7ec0ad6fbe124565467                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Tr-no_1153.jpg computed_with_main_split_code_on_downloaded_kaggle_files
        DS2 train    notumor     Tr-no_0340.jpg a3f433913a6ec7ec0ad6fbe124565467                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Tr-no_0340.jpg computed_with_main_split_code_on_downloaded_kaggle_files
        DS2   val    notumor     Te-no_0026.jpg a3f433913a6ec7ec0ad6fbe124565467                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Te-no_0026.jpg computed_with_main_split_code_on_downloaded_kaggle_files
        DS2 train    notumor     Tr-no_0360.jpg a643c6b387f97833286b2e9eef12373d                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Tr-no_0360.jpg computed_with_main_split_code_on_downloaded_kaggle_files
        DS2   val    notumor     Te-no_0097.jpg a643c6b387f97833286b2e9eef12373d                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Te-no_0097.jpg computed_with_main_split_code_on_downloaded_kaggle_files
        DS2 train    notumor     Tr-no_0161.jpg b1274d3776160d11906ac560d87618f9                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Tr-no_0161.jpg computed_with_main_split_code_on_downloaded_kaggle_files
        DS2   val    notumor     Tr-no_1116.jpg b1274d3776160d11906ac560d87618f9                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Tr-no_1116.jpg computed_with_main_split_code_on_downloaded_kaggle_files
        DS2  test meningioma     Te-me_0071.jpg b6c9a71babd2557fb1d620640e264b17                                                                                                       /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/meningioma/Te-me_0071.jpg computed_with_main_split_code_on_downloaded_kaggle_files
        DS2 train meningioma     Tr-me_0249.jpg b6c9a71babd2557fb1d620640e264b17                                                                                                       /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/meningioma/Tr-me_0249.jpg computed_with_main_split_code_on_downloaded_kaggle_files
        DS2 train meningioma     Tr-me_0251.jpg b6c9a71babd2557fb1d620640e264b17                                                                                                       /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/meningioma/Tr-me_0251.jpg computed_with_main_split_code_on_downloaded_kaggle_files
        DS2   val meningioma     Te-me_0073.jpg b6c9a71babd2557fb1d620640e264b17                                                                                                       /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/meningioma/Te-me_0073.jpg computed_with_main_split_code_on_downloaded_kaggle_files
        DS2  test    notumor     Tr-no_0257.jpg b81d4a86910cb4386c7a0ed88cb9c928                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Tr-no_0257.jpg computed_with_main_split_code_on_downloaded_kaggle_files
        DS2 train    notumor     Te-no_0040.jpg b81d4a86910cb4386c7a0ed88cb9c928                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Te-no_0040.jpg computed_with_main_split_code_on_downloaded_kaggle_files
        DS2 train    notumor     Te-no_0052.jpg b81d4a86910cb4386c7a0ed88cb9c928                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Te-no_0052.jpg computed_with_main_split_code_on_downloaded_kaggle_files
        DS2 train    notumor     Tr-no_0135.jpg b81d4a86910cb4386c7a0ed88cb9c928                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Tr-no_0135.jpg computed_with_main_split_code_on_downloaded_kaggle_files
        DS2   val    notumor     Tr-no_1096.jpg b81d4a86910cb4386c7a0ed88cb9c928                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Tr-no_1096.jpg computed_with_main_split_code_on_downloaded_kaggle_files
        DS2 train    notumor     Tr-no_0323.jpg c945ddcc82615720c95807e20f9a661d                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Tr-no_0323.jpg computed_with_main_split_code_on_downloaded_kaggle_files
        DS2 train    notumor     Tr-no_0966.jpg c945ddcc82615720c95807e20f9a661d                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Tr-no_0966.jpg computed_with_main_split_code_on_downloaded_kaggle_files
        DS2   val    notumor     Te-no_0059.jpg c945ddcc82615720c95807e20f9a661d                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Te-no_0059.jpg computed_with_main_split_code_on_downloaded_kaggle_files
        DS2  test    notumor     Tr-no_0231.jpg cae61c6483c14eb68652a280f2571da1                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Tr-no_0231.jpg computed_with_main_split_code_on_downloaded_kaggle_files
        DS2 train    notumor     Tr-no_0011.jpg cae61c6483c14eb68652a280f2571da1                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Tr-no_0011.jpg computed_with_main_split_code_on_downloaded_kaggle_files
        DS2 train    notumor     Tr-no_0012.jpg cae61c6483c14eb68652a280f2571da1                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Tr-no_0012.jpg computed_with_main_split_code_on_downloaded_kaggle_files
        DS2 train    notumor     Tr-no_0024.jpg cae61c6483c14eb68652a280f2571da1                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Tr-no_0024.jpg computed_with_main_split_code_on_downloaded_kaggle_files
        DS2 train    notumor     Tr-no_0298.jpg cae61c6483c14eb68652a280f2571da1                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Tr-no_0298.jpg computed_with_main_split_code_on_downloaded_kaggle_files
        DS2   val    notumor     Tr-no_1131.jpg cae61c6483c14eb68652a280f2571da1                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Tr-no_1131.jpg computed_with_main_split_code_on_downloaded_kaggle_files
        DS2  test    notumor     Tr-no_0339.jpg cefc8fe39853df7445490b3af0f726ee                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Tr-no_0339.jpg computed_with_main_split_code_on_downloaded_kaggle_files
        DS2 train    notumor     Tr-no_0981.jpg cefc8fe39853df7445490b3af0f726ee                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Tr-no_0981.jpg computed_with_main_split_code_on_downloaded_kaggle_files
        DS2   val    notumor     Te-no_0076.jpg cefc8fe39853df7445490b3af0f726ee                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Te-no_0076.jpg computed_with_main_split_code_on_downloaded_kaggle_files
        DS2 train    notumor     Tr-no_0249.jpg d83d7f6ac1f75dd6e5eedeff32ffe44e                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Tr-no_0249.jpg computed_with_main_split_code_on_downloaded_kaggle_files
        DS2   val    notumor     Te-no_0101.jpg d83d7f6ac1f75dd6e5eedeff32ffe44e                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Te-no_0101.jpg computed_with_main_split_code_on_downloaded_kaggle_files
        DS2 train meningioma     Te-me_0097.jpg dd2743e1038f630cfaf57033a22ec9e5                                                                                                       /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/meningioma/Te-me_0097.jpg computed_with_main_split_code_on_downloaded_kaggle_files
        DS2   val meningioma     Tr-me_0274.jpg dd2743e1038f630cfaf57033a22ec9e5                                                                                                       /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/meningioma/Tr-me_0274.jpg computed_with_main_split_code_on_downloaded_kaggle_files
        DS2  test meningioma     Te-me_0031.jpg e2ce9c2850231613de6b2952dd9a4ace                                                                                                       /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/meningioma/Te-me_0031.jpg computed_with_main_split_code_on_downloaded_kaggle_files
        DS2 train meningioma     Te-me_0010.jpg e2ce9c2850231613de6b2952dd9a4ace                                                                                                       /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/meningioma/Te-me_0010.jpg computed_with_main_split_code_on_downloaded_kaggle_files
        DS2 train meningioma     Tr-me_0211.jpg e2ce9c2850231613de6b2952dd9a4ace                                                                                                       /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/meningioma/Tr-me_0211.jpg computed_with_main_split_code_on_downloaded_kaggle_files
        DS2 train meningioma     Tr-me_0366.jpg e2ce9c2850231613de6b2952dd9a4ace                                                                                                       /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/meningioma/Tr-me_0366.jpg computed_with_main_split_code_on_downloaded_kaggle_files
        DS2   val meningioma     Tr-me_0176.jpg e2ce9c2850231613de6b2952dd9a4ace                                                                                                       /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/meningioma/Tr-me_0176.jpg computed_with_main_split_code_on_downloaded_kaggle_files
        DS2  test    notumor     Te-no_0069.jpg ead6beeabe5a77f2de90e8d546f9a615                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Te-no_0069.jpg computed_with_main_split_code_on_downloaded_kaggle_files
        DS2 train    notumor     Te-no_0029.jpg ead6beeabe5a77f2de90e8d546f9a615                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Te-no_0029.jpg computed_with_main_split_code_on_downloaded_kaggle_files
        DS2   val    notumor     Tr-no_0333.jpg ead6beeabe5a77f2de90e8d546f9a615                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Tr-no_0333.jpg computed_with_main_split_code_on_downloaded_kaggle_files
        DS2 train    notumor     Tr-no_0241.jpg f39e6559e6493487cada6e266a953bb6                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Tr-no_0241.jpg computed_with_main_split_code_on_downloaded_kaggle_files
        DS2   val    notumor     Tr-no_1137.jpg f39e6559e6493487cada6e266a953bb6                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Tr-no_1137.jpg computed_with_main_split_code_on_downloaded_kaggle_files
        DS2 train    notumor     Tr-no_0159.jpg f565b1b202ecf70e5833f979db4f1ea5                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Tr-no_0159.jpg computed_with_main_split_code_on_downloaded_kaggle_files
        DS2 train    notumor     Tr-no_1114.jpg f565b1b202ecf70e5833f979db4f1ea5                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Tr-no_1114.jpg computed_with_main_split_code_on_downloaded_kaggle_files
        DS2   val    notumor     Tr-no_0111.jpg f565b1b202ecf70e5833f979db4f1ea5                                                                                                          /kaggle/input/preprocessed-brain-mri-scans-for-tumors-detection/preprocessed_brain_mri_dataset/notumor/Tr-no_0111.jpg computed_with_main_split_code_on_downloaded_kaggle_files
    
    ======================================================================================================================================================
    3.5 NON-IID FEDERATED COHORT SIMULATION
    ======================================================================================================================================================
    
    ------------------------------------------------------------------------------------------------------------------------------------------------------
    Federated simulation configuration
    ------------------------------------------------------------------------------------------------------------------------------------------------------
                                 parameter                                     value       source_type
                   candidate_client_counts                                 [3, 4, 5]  main_code_config
              client_count_search_episodes                                        10  main_code_config
                      selected_DS1_clients                                         3 main_saved_output
                      selected_DS2_clients                                         3 main_saved_output
                    total_adaptive_clients                                         6 main_saved_output
                           dirichlet_alpha                                      0.35  main_code_config
                  min_per_class_per_client                                         5  main_code_config
                          client_tune_frac                                      0.12  main_code_config
                           client_val_frac                                      0.12  main_code_config
                          weighted_sampler 1/sqrt(class_count) local class weighting  main_code_config
    client_participation_in_saved_training      all six clients selected every round main_saved_output
    
    ------------------------------------------------------------------------------------------------------------------------------------------------------
    Exact RL-UCB planning history from main saved output
    ------------------------------------------------------------------------------------------------------------------------------------------------------
     episode  selected_n_clients  reward_ds1  reward_ds2  reward_mean  bandit_value  pulls_for_arm       source_type
           1                   3    0.709483    0.663089     0.686286      0.686286              1 main_saved_output
           2                   4    0.721983    0.738356     0.730169      0.730169              1 main_saved_output
           3                   5    0.675439    0.698698     0.687069      0.687069              1 main_saved_output
           4                   4    0.753396    0.829472     0.791434      0.760801              2 main_saved_output
           5                   5    0.702440    0.692132     0.697286      0.692177              2 main_saved_output
           6                   3    0.870872    0.708757     0.789815      0.738050              2 main_saved_output
           7                   4    0.705301    0.608192     0.656747      0.726116              3 main_saved_output
           8                   3    0.766992    0.753060     0.760026      0.745376              3 main_saved_output
           9                   5    0.664448    0.652680     0.658564      0.680973              3 main_saved_output
          10                   3    0.753155    0.660297     0.706726      0.735713              4 main_saved_output
    
    ------------------------------------------------------------------------------------------------------------------------------------------------------
    Exact adaptive client distribution from main saved output
    ------------------------------------------------------------------------------------------------------------------------------------------------------
      client dataset  total_train  total_tune  total_val  glioma  meningioma  notumor  pituitary       source_type
    client_0     ds1          271          42         37      77          34       79         81 main_saved_output
    client_1     ds1          167          27         23       4          45        5        113 main_saved_output
    client_2     ds1          375          59         52     121         117      130          7 main_saved_output
    client_3     ds2         1092         170        150      39         251      223        579 main_saved_output
    client_4     ds2         1690         262        231     275         209      852        354 main_saved_output
    client_5     ds2         1026         160        140     563         433        8         22 main_saved_output
    
    ------------------------------------------------------------------------------------------------------------------------------------------------------
    Derived non-IID client statistics from saved client distribution
    ------------------------------------------------------------------------------------------------------------------------------------------------------
      client dataset  total_train  glioma  meningioma  notumor  pituitary  train_entropy_normalized dominant_class  dominant_class_fraction  class_coverage_fraction  imbalance_ratio_max_min_nonzero                                        source_type
    client_0     ds1          271      77          34       79         81                  0.965350      pituitary                 0.298893                      1.0                         2.382353 derived_from_main_saved_output_client_distribution
    client_1     ds1          167       4          45        5        113                  0.585795      pituitary                 0.676647                      1.0                        28.250000 derived_from_main_saved_output_client_distribution
    client_2     ds1          375     121         117      130          7                  0.843941        notumor                 0.346667                      1.0                        18.571429 derived_from_main_saved_output_client_distribution
    client_3     ds2         1092      39         251      223        579                  0.806308      pituitary                 0.530220                      1.0                        14.846154 derived_from_main_saved_output_client_distribution
    client_4     ds2         1690     275         209      852        354                  0.884851        notumor                 0.504142                      1.0                         4.076555 derived_from_main_saved_output_client_distribution
    client_5     ds2         1026     563         433        8         22                  0.586912         glioma                 0.548733                      1.0                        70.375000 derived_from_main_saved_output_client_distribution
    
    ------------------------------------------------------------------------------------------------------------------------------------------------------
    Derived dataset-level non-IID summary from saved client distribution
    ------------------------------------------------------------------------------------------------------------------------------------------------------
    dataset  n_clients  total_train  min_client_train  max_client_train  mean_client_train  mean_entropy  mean_dominant_class_fraction  mean_imbalance_ratio                                        source_type
        ds1          3          813               167               375         271.000000      0.798362                      0.440736             16.401261 derived_from_main_saved_output_client_distribution
        ds2          3         3808              1026              1690        1269.333333      0.759357                      0.527698             29.765903 derived_from_main_saved_output_client_distribution
    
    ======================================================================================================================================================
    FINAL SECTION 3 CHECKLIST — WHAT EACH VALUE IS BASED ON
    ======================================================================================================================================================
    
    ------------------------------------------------------------------------------------------------------------------------------------------------------
    Section 3 evidence map
    ------------------------------------------------------------------------------------------------------------------------------------------------------
                                                             section available_now                                                                                                                       basis
                                     3.1 Development Dataset Sources           Yes                                          main code Kaggle handles + main saved-output counts + downloaded-file verification
                 3.2 Class Labels and Source-to-Target Label Mapping           Yes                                                                                       main code label mapping and label IDs
    3.3 Image Inclusion, Duplicate Handling, and Label Harmonization           Yes main code extension/filter/load logic + new Kaggle file audit for image sizes, formats, modes, duplicates, unreadable files
             3.4 Train/Validation/Test Splitting and Leakage Control           Yes                                     main saved-output split/leakage table + recomputed full-MD5 audit using same split code
                             3.5 Non-IID Federated Cohort Simulation           Yes       main code config + exact saved RL-UCB history + exact saved adaptive client distribution + derived non-IID statistics
    
    DONE.
    No model training was run.
    No federated training was run.
    No RACE-FELCM training/evaluation was run.
    All source types are labeled in the printed tables.
    If STRICT_MATCH_MAIN_OUTPUT=True and counts mismatch, this code stops instead of giving unsafe values.


# **Error and Calibration**


```python
import numpy as np
import matplotlib.pyplot as plt

# =========================
# Global style settings
# =========================
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.weight": "bold",
    "axes.titleweight": "bold",
    "axes.labelweight": "bold",
    "axes.edgecolor": "#D0D7DE",
    "axes.linewidth": 1.0,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
})

# =========================
# Data
# =========================

# Plot 1: Calibration Metrics
calibration_metrics = ["ECE", "MCE", "Brier", "LogLoss"]
calibration_ds1 = [0.027, 0.465, 0.006, 0.029]
calibration_ds2 = [0.023, 0.504, 0.031, 0.074]

# Plot 2: Error Metrics
error_metrics = [
    "ppv_macro",
    "npv_macro",
    "specificity_macro",
    "fpr_macro",
    "fnr_macro",
    "jaccard_macro"
]

error_ds1 = [1.000, 1.000, 1.000, 0.000, 0.000, 1.000]
error_ds2 = [0.980, 0.994, 0.994, 0.006, 0.020, 0.960]


# =========================
# Plotting function
# =========================
def draw_grouped_bar_plot(
    metrics,
    ds1_values,
    ds2_values,
    title,
    ylim,
    yticks,
    figsize=(7.5, 3.6)
):
    x = np.arange(len(metrics))
    width = 0.34

    fig, ax = plt.subplots(figsize=figsize, dpi=160)

    bars1 = ax.bar(
        x - width / 2,
        ds1_values,
        width,
        label="DS1",
        color="#1f77b4",
        edgecolor="white",
        linewidth=1.2
    )

    bars2 = ax.bar(
        x + width / 2,
        ds2_values,
        width,
        label="DS2",
        color="#ff7f0e",
        edgecolor="white",
        linewidth=1.2
    )

    # Title
    ax.set_title(
        title,
        fontsize=15,
        fontweight="bold",
        pad=10
    )

    # X-axis
    ax.set_xticks(x)
    ax.set_xticklabels(
        metrics,
        rotation=25,
        ha="right",
        fontweight="bold"
    )

    # Y-axis
    ax.set_ylim(ylim)
    ax.set_yticks(yticks)

    # Grid
    ax.grid(
        axis="y",
        linestyle="-",
        linewidth=0.7,
        alpha=0.25
    )
    ax.set_axisbelow(True)

    # Legend moved outside to avoid overlap
    legend = ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        frameon=True,
        fancybox=True,
        borderpad=0.5
    )

    for text in legend.get_texts():
        text.set_fontweight("bold")

    # Clean spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.tick_params(axis="both", length=0)

    # Value labels
    offset = (ylim[1] - ylim[0]) * 0.018

    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()

            label_y = height + offset

            ax.text(
                bar.get_x() + bar.get_width() / 2,
                label_y,
                f"{height:.3f}",
                ha="center",
                va="bottom",
                fontsize=7.5,
                fontweight="bold",
                clip_on=False
            )

    plt.tight_layout()
    plt.show()


# =========================
# Draw both plots
# =========================

draw_grouped_bar_plot(
    metrics=calibration_metrics,
    ds1_values=calibration_ds1,
    ds2_values=calibration_ds2,
    title="DS1 vs DS2 — Calibration Metrics",
    ylim=(0, 0.56),
    yticks=np.arange(0, 0.57, 0.1),
    figsize=(7.4, 3.6)
)

draw_grouped_bar_plot(
    metrics=error_metrics,
    ds1_values=error_ds1,
    ds2_values=error_ds2,
    title="DS1 vs DS2 — Error Metrics",
    ylim=(0, 1.12),
    yticks=np.arange(0, 1.11, 0.2),
    figsize=(8.2, 3.8)
)
```


    
![png](01_1_FedRACE_Net_Federated_Robust_Adaptive_Context_Enhanced_Fuzzy_Edge_Local_Contrast_Mapping_Fusion_Network_files/01_1_FedRACE_Net_Federated_Robust_Adaptive_Context_Enhanced_Fuzzy_Edge_Local_Contrast_Mapping_Fusion_Network_4_0.png)
    



    
![png](01_1_FedRACE_Net_Federated_Robust_Adaptive_Context_Enhanced_Fuzzy_Edge_Local_Contrast_Mapping_Fusion_Network_files/01_1_FedRACE_Net_Federated_Robust_Adaptive_Context_Enhanced_Fuzzy_Edge_Local_Contrast_Mapping_Fusion_Network_4_1.png)
    

