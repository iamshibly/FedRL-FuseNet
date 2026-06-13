# Auto-exported from: 2-arcf-net-preprocessing-ablation.ipynb
# Public Python export for GitHub visibility.
# Notebook outputs are available in reports/notebook_exports/.

# %% [markdown] Cell 1
# **1. No Preprocessing**

# %% Cell 2
# ============================================================
# KAGGLE FULL SCRIPT
# ABLATION 1: NO PREPROCESSING
# TRUE FL + RL-UCB + IDENTITY + CRAF + ResNet-50
# METHOD: ARCF-Net (Ablation 1 - No Preprocessing)
# ------------------------------------------------------------
# KAGGLE-READY + SYMMETRIC DS1/DS2 IMPORTANCE
# - Uses BOTH datasets
# - Reads datasets from /kaggle/input automatically
# - Keeps stronger asymmetric train/tune/val per client regime
# - Exact 15 FL rounds
# - Proper FL with FedAvg + FedProx + prototype sharing
# - RL-UCB for SHARED client-count planning + per-client fixed identity preset
# - Tune-aware theta probing remains in place but only one fixed arm exists
# - Equal DS1 / DS2 importance in best-round selection and merged reporting
# - NO PLOTS
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
print("TRUE FL + RL-UCB + NO PREPROCESSING + CRAF + ResNet-50")
print("METHOD: ARCF-Net (Ablation 1 - No Preprocessing)")
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
    "use_preprocessing": False,
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
    "make_plots": False,
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
    "preprocessing_full_form": "No Preprocessing (Identity pass-through)",
    "fusion_full_form": "Cross-Residual Adaptive Fusion",
    "backbone_full_form": "Residual Network-50",
    "ablation_name": "Ablation 1 - No Preprocessing",
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
# STEP 7: ABLATION 1 — NO PREPROCESSING (IDENTITY)
# ============================================================
print("\n" + "=" * 118)
print("STEP 7: ABLATION 1 — NO PREPROCESSING (IDENTITY)")
print("=" * 118)

THETA_FULLFORMS = {
    "gamma": "Identity placeholder parameter",
    "alpha": "Identity placeholder parameter",
    "beta": "Identity placeholder parameter",
    "tau": "Identity placeholder parameter",
    "k": "Identity placeholder parameter",
    "edge_gain": "Identity placeholder parameter",
    "blend": "Identity placeholder parameter",
}

NO_PREPROC_THETA = (1.00, 0.00, 0.00, 1.00, 1, 0.00, 0.00)

PRESET_BANK_DS1 = [
    ("no_preprocessing", NO_PREPROC_THETA),
]

PRESET_BANK_DS2 = [
    ("no_preprocessing", NO_PREPROC_THETA),
]

class IdentityPreprocessing(nn.Module):
    def __init__(self):
        super().__init__()
        lap = torch.tensor([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=torch.float32)
        self.register_buffer("lap", lap.view(1, 1, 3, 3))

    def forward(self, x):
        return x

def theta_to_module(theta):
    return IdentityPreprocessing()

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
print("Preprocessing: No Preprocessing (Identity)")
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
    {"setting": "ARCF-Net Ablation 1 (No Preprocessing)", "split": "VAL",  "dataset": "ds1",           **compact_metrics(val_ds1)},
    {"setting": "ARCF-Net Ablation 1 (No Preprocessing)", "split": "VAL",  "dataset": "ds2",           **compact_metrics(val_ds2)},
    {"setting": "ARCF-Net Ablation 1 (No Preprocessing)", "split": "VAL",  "dataset": "global_equal",  **compact_metrics(val_global)},
    {"setting": "ARCF-Net Ablation 1 (No Preprocessing)", "split": "TEST", "dataset": "ds1",           **compact_metrics(test_ds1)},
    {"setting": "ARCF-Net Ablation 1 (No Preprocessing)", "split": "TEST", "dataset": "ds2",           **compact_metrics(test_ds2)},
    {"setting": "ARCF-Net Ablation 1 (No Preprocessing)", "split": "TEST", "dataset": "global_equal",  **compact_metrics(test_global)},
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
    lap_kernel = preproc.lap if hasattr(preproc, "lap") else IdentityPreprocessing().to(DEVICE).lap

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
        title=f"Before vs After Identity | DS1 representative = {theta_str(rep_theta_ds1)}"
    )
    show_before_after(
        preproc_ds2,
        test2,
        n=CFG["before_after_n"],
        title=f"Before vs After Identity | DS2 representative = {theta_str(rep_theta_ds2)}"
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
    ax.set_title("Identity Placeholder Parameter Evolution (Mean across clients)", fontweight="bold")
    ax.set_xlabel("Round", fontweight="bold")
    ax.legend(ncol=2)
    ax.grid(alpha=0.20)
    plt.show()

# still save theta evolution table even though no plots
plot_theta_evolution(loc_df)

if CFG["make_plots"]:
    rad_keys = ["acc", "f1_macro", "precision_macro", "recall_macro", "auc_roc_macro_ovr", "mcc", "kappa", "log_loss"]
    radar_plot(test_ds1, test_ds2, rad_keys, "TEST Metrics Radar (DS1 vs DS2)")
    plot_global_training_dashboard(glob_df)
    plot_client_evolution(loc_df)

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
    "preprocessing_name": "No Preprocessing (Identity)",

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
print(f"Ablation: {METHOD_INFO['ablation_name']}")
print(f"Best round: {best_round_saved}")
print(f"Adaptive clients => DS1={n_clients_ds1}, DS2={n_clients_ds2}, TOTAL={CLIENTS_TOTAL}")
print(f"Rounds completed: {CFG['rounds']}")
print(f"Global TEST acc: {safe_float(test_global.get('acc')):.4f}")
print(f"Global TEST f1_macro: {safe_float(test_global.get('f1_macro')):.4f}")
print(f"DS1 TEST acc: {safe_float(test_ds1.get('acc')):.4f}")
print(f"DS2 TEST acc: {safe_float(test_ds2.get('acc')):.4f}")
print(f"DS1 final strategy: {choice_ds1['strategy']} | names={choice_ds1['theta_names']}")
print(f"DS2 final strategy: {choice_ds2['strategy']} | names={choice_ds2['theta_names']}")

# %% [markdown] Cell 3
# **2. Fixed RACE-FELCM Preset**

# %% Cell 4
# ============================================================
# KAGGLE FULL SCRIPT
# ABLATION 2: FIXED RACE-FELCM PRESET
# TRUE FL + RL-UCB + FIXED RACE-FELCM + CRAF + ResNet-50
# METHOD: ARCF-Net (Ablation 2 - Fixed RACE-FELCM Preset)
# ------------------------------------------------------------
# KAGGLE-READY + SYMMETRIC DS1/DS2 IMPORTANCE
# - Uses BOTH datasets
# - Reads datasets from /kaggle/input automatically
# - Keeps stronger asymmetric train/tune/val per client regime
# - Exact 15 FL rounds
# - Proper FL with FedAvg + FedProx + prototype sharing
# - RL-UCB for SHARED client-count planning
# - FIXED RACE-FELCM preset for preprocessing (same fixed preset, no RL preset selection)
# - Tune-aware theta probing path remains structurally intact but only one fixed arm exists
# - Equal DS1 / DS2 importance in best-round selection and merged reporting
# - NO PLOTS
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
print("TRUE FL + RL-UCB + FIXED RACE-FELCM + CRAF + ResNet-50")
print("METHOD: ARCF-Net (Ablation 2 - Fixed RACE-FELCM Preset)")
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
    "make_plots": False,
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
    "preprocessing_full_form": "Robust Adaptive Context-Enhanced Fuzzy Edge Local Contrast Mapping (Fixed Preset)",
    "fusion_full_form": "Cross-Residual Adaptive Fusion",
    "backbone_full_form": "Residual Network-50",
    "ablation_name": "Ablation 2 - Fixed RACE-FELCM Preset",
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
# STEP 7: FIXED RACE-FELCM PRESET
# ============================================================
print("\n" + "=" * 118)
print("STEP 7: FIXED RACE-FELCM PRESET")
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

# Fixed preset for Ablation 2
# Using the same balanced preset across both datasets so only the
# preset-selection ablation changes while the rest of the pipeline stays same.
FIXED_RACE_THETA = (1.00, 0.24, 4.6, 2.4, 5, 0.10, 0.78)

PRESET_BANK_DS1 = [
    ("fixed_race_balanced", FIXED_RACE_THETA),
]

PRESET_BANK_DS2 = [
    ("fixed_race_balanced", FIXED_RACE_THETA),
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
# STEP 10: TUNE-AWARE FIXED PREPROCESSING SELECTION
# ============================================================
print("\n" + "=" * 118)
print("STEP 10: TUNE-AWARE FIXED PREPROCESSING SELECTION")
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
print("Preprocessing: Fixed RACE-FELCM preset")
print(f"Fixed theta: {theta_str(FIXED_RACE_THETA)}")
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
    {"setting": "ARCF-Net Ablation 2 (Fixed RACE-FELCM Preset)", "split": "VAL",  "dataset": "ds1",           **compact_metrics(val_ds1)},
    {"setting": "ARCF-Net Ablation 2 (Fixed RACE-FELCM Preset)", "split": "VAL",  "dataset": "ds2",           **compact_metrics(val_ds2)},
    {"setting": "ARCF-Net Ablation 2 (Fixed RACE-FELCM Preset)", "split": "VAL",  "dataset": "global_equal",  **compact_metrics(val_global)},
    {"setting": "ARCF-Net Ablation 2 (Fixed RACE-FELCM Preset)", "split": "TEST", "dataset": "ds1",           **compact_metrics(test_ds1)},
    {"setting": "ARCF-Net Ablation 2 (Fixed RACE-FELCM Preset)", "split": "TEST", "dataset": "ds2",           **compact_metrics(test_ds2)},
    {"setting": "ARCF-Net Ablation 2 (Fixed RACE-FELCM Preset)", "split": "TEST", "dataset": "global_equal",  **compact_metrics(test_global)},
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

def build_theta_evolution_table(loc_df):
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
    return theta_evo

def plot_theta_evolution(theta_evo):
    fig, ax = plt.subplots(figsize=(15, 6), constrained_layout=True, facecolor="white")
    for col in theta_evo.columns:
        if col != "round":
            ax.plot(theta_evo["round"], theta_evo[col], marker="o", label=col)
    ax.set_title("RACE-FELCM Parameter Evolution (Mean across clients)", fontweight="bold")
    ax.set_xlabel("Round", fontweight="bold")
    ax.legend(ncol=2)
    ax.grid(alpha=0.20)
    plt.show()

theta_evo_df = build_theta_evolution_table(loc_df)

if CFG["make_plots"]:
    rad_keys = ["acc", "f1_macro", "precision_macro", "recall_macro", "auc_roc_macro_ovr", "mcc", "kappa", "log_loss"]
    radar_plot(test_ds1, test_ds2, rad_keys, "TEST Metrics Radar (DS1 vs DS2)")
    plot_global_training_dashboard(glob_df)
    plot_client_evolution(loc_df)
    plot_theta_evolution(theta_evo_df)

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
    "preprocessing_name": "RACE-FELCM (Fixed Preset)",

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
    "fixed_race_theta": FIXED_RACE_THETA,

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
    "theta_evolution_mean": theta_evo_df.to_dict(orient="list"),
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
print(f"Ablation: {METHOD_INFO['ablation_name']}")
print(f"Fixed theta: {theta_str(FIXED_RACE_THETA)}")
print(f"Best round: {best_round_saved}")
print(f"Adaptive clients => DS1={n_clients_ds1}, DS2={n_clients_ds2}, TOTAL={CLIENTS_TOTAL}")
print(f"Rounds completed: {CFG['rounds']}")
print(f"Global TEST acc: {safe_float(test_global.get('acc')):.4f}")
print(f"Global TEST f1_macro: {safe_float(test_global.get('f1_macro')):.4f}")
print(f"DS1 TEST acc: {safe_float(test_ds1.get('acc')):.4f}")
print(f"DS2 TEST acc: {safe_float(test_ds2.get('acc')):.4f}")
print(f"DS1 final strategy: {choice_ds1['strategy']} | names={choice_ds1['theta_names']}")
print(f"DS2 final strategy: {choice_ds2['strategy']} | names={choice_ds2['theta_names']}")

# %% [markdown] Cell 5
# **3. RL-Selected RACE-FELCM**

# %% Cell 6
# ============================================================
# KAGGLE FULL SCRIPT
# ABLATION 3: RL-SELECTED RACE-FELCM
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
# - NO PLOTS
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
print("METHOD: ARCF-Net (Ablation 3 - RL-Selected RACE-FELCM)")
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
    "make_plots": False,
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
    "ablation_name": "Ablation 3 - RL-Selected RACE-FELCM",
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
print("Preprocessing: RACE-FELCM (RL-selected)")
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
    {"setting": "ARCF-Net (RL-selected RACE-FELCM)", "split": "VAL",  "dataset": "ds1",           **compact_metrics(val_ds1)},
    {"setting": "ARCF-Net (RL-selected RACE-FELCM)", "split": "VAL",  "dataset": "ds2",           **compact_metrics(val_ds2)},
    {"setting": "ARCF-Net (RL-selected RACE-FELCM)", "split": "VAL",  "dataset": "global_equal",  **compact_metrics(val_global)},
    {"setting": "ARCF-Net (RL-selected RACE-FELCM)", "split": "TEST", "dataset": "ds1",           **compact_metrics(test_ds1)},
    {"setting": "ARCF-Net (RL-selected RACE-FELCM)", "split": "TEST", "dataset": "ds2",           **compact_metrics(test_ds2)},
    {"setting": "ARCF-Net (RL-selected RACE-FELCM)", "split": "TEST", "dataset": "global_equal",  **compact_metrics(test_global)},
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

def build_theta_evolution_table(loc_df):
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
    return theta_evo

def plot_theta_evolution(theta_evo):
    fig, ax = plt.subplots(figsize=(15, 6), constrained_layout=True, facecolor="white")
    for col in theta_evo.columns:
        if col != "round":
            ax.plot(theta_evo["round"], theta_evo[col], marker="o", label=col)
    ax.set_title("RACE-FELCM Parameter Evolution (Mean across clients)", fontweight="bold")
    ax.set_xlabel("Round", fontweight="bold")
    ax.legend(ncol=2)
    ax.grid(alpha=0.20)
    plt.show()

theta_evo_df = build_theta_evolution_table(loc_df)

if CFG["make_plots"]:
    rad_keys = ["acc", "f1_macro", "precision_macro", "recall_macro", "auc_roc_macro_ovr", "mcc", "kappa", "log_loss"]
    radar_plot(test_ds1, test_ds2, rad_keys, "TEST Metrics Radar (DS1 vs DS2)")
    plot_global_training_dashboard(glob_df)
    plot_client_evolution(loc_df)
    plot_theta_evolution(theta_evo_df)

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
    "theta_evolution_mean": theta_evo_df.to_dict(orient="list"),
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
print(f"Ablation: {METHOD_INFO['ablation_name']}")
print(f"Best round: {best_round_saved}")
print(f"Adaptive clients => DS1={n_clients_ds1}, DS2={n_clients_ds2}, TOTAL={CLIENTS_TOTAL}")
print(f"Rounds completed: {CFG['rounds']}")
print(f"Global TEST acc: {safe_float(test_global.get('acc')):.4f}")
print(f"Global TEST f1_macro: {safe_float(test_global.get('f1_macro')):.4f}")
print(f"DS1 TEST acc: {safe_float(test_ds1.get('acc')):.4f}")
print(f"DS2 TEST acc: {safe_float(test_ds2.get('acc')):.4f}")
print(f"DS1 final strategy: {choice_ds1['strategy']} | names={choice_ds1['theta_names']}")
print(f"DS2 final strategy: {choice_ds2['strategy']} | names={choice_ds2['theta_names']}")

# %% [markdown] Cell 7
# **4. Raw Only**

# %% Cell 8
# ============================================================
# KAGGLE FULL SCRIPT
# ABLATION 4: RAW ONLY
# TRUE FL + RL-UCB + RACE-FELCM + RAW-ONLY + ResNet-50
# METHOD ACRONYM: ARCF-Net
# FULL FORM: Adaptive RACE-FELCM with Raw-Only Network
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
# - ABLATION 4: RAW ONLY INPUT BRANCH
# - NO PLOTS
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
print("TRUE FL + RL-UCB + RACE-FELCM + RAW-ONLY + ResNet-50")
print("METHOD: ARCF-Net (Ablation 4 - Raw Only)")
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

    # final inference
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
    "make_plots": False,
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
    "full_form": "Adaptive RACE-FELCM with Raw-Only Network",
    "preprocessing_full_form": "Robust Adaptive Context-Enhanced Fuzzy Edge Local Contrast Mapping",
    "fusion_full_form": "Raw-Only Head",
    "backbone_full_form": "Residual Network-50",
    "ablation_name": "Ablation 4 - Raw Only",
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
# STEP 8: MODEL — ResNet-50 RAW-ONLY HEAD
# ============================================================
print("\n" + "=" * 118)
print("STEP 8: MODEL — ResNet-50 RAW-ONLY HEAD")
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

class ResNet50RawOnly(nn.Module):
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
        self.cond_proj = nn.Linear(cond_dim, fuse_dim)

        self.raw_proj = CompactProjector(self.backbone_dim, fuse_dim)
        self.out_norm = nn.LayerNorm(fuse_dim)

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
        fr = self.raw_proj(f_raw)
        fused = self.out_norm(fr + self.cond_proj(cond))

        embed = self.neck(fused)
        logits = self.classifier(embed)

        if return_extra:
            gates = torch.zeros((x_raw_n.size(0), 3), device=x_raw_n.device, dtype=embed.dtype)
            gates[:, 0] = 1.0
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

global_model = ResNet50RawOnly(
    num_classes=NUM_CLASSES,
    cond_dim=64,
    fuse_dim=256,
    embed_dim=256,
    pretrained=True,
).to(DEVICE)

set_trainable_for_round(global_model, rnd=1)
total_params, trainable_params = count_params(global_model)

print("Backbone: ResNet-50 | pretrained_loaded=True")
print("Branch ablation: RAW ONLY")
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
print("Preprocessing: RACE-FELCM (RL-selected)")
print("Input branch ablation: RAW ONLY")
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

        local_model = ResNet50RawOnly(
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
    {"setting": "ARCF-Net (Raw Only)", "split": "VAL",  "dataset": "ds1",           **compact_metrics(val_ds1)},
    {"setting": "ARCF-Net (Raw Only)", "split": "VAL",  "dataset": "ds2",           **compact_metrics(val_ds2)},
    {"setting": "ARCF-Net (Raw Only)", "split": "VAL",  "dataset": "global_equal",  **compact_metrics(val_global)},
    {"setting": "ARCF-Net (Raw Only)", "split": "TEST", "dataset": "ds1",           **compact_metrics(test_ds1)},
    {"setting": "ARCF-Net (Raw Only)", "split": "TEST", "dataset": "ds2",           **compact_metrics(test_ds2)},
    {"setting": "ARCF-Net (Raw Only)", "split": "TEST", "dataset": "global_equal",  **compact_metrics(test_global)},
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

def build_theta_evolution_table(loc_df):
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
    return theta_evo

def plot_theta_evolution(theta_evo):
    fig, ax = plt.subplots(figsize=(15, 6), constrained_layout=True, facecolor="white")
    for col in theta_evo.columns:
        if col != "round":
            ax.plot(theta_evo["round"], theta_evo[col], marker="o", label=col)
    ax.set_title("RACE-FELCM Parameter Evolution (Mean across clients)", fontweight="bold")
    ax.set_xlabel("Round", fontweight="bold")
    ax.legend(ncol=2)
    ax.grid(alpha=0.20)
    plt.show()

theta_evo_df = build_theta_evolution_table(loc_df)

if CFG["make_plots"]:
    rad_keys = ["acc", "f1_macro", "precision_macro", "recall_macro", "auc_roc_macro_ovr", "mcc", "kappa", "log_loss"]
    radar_plot(test_ds1, test_ds2, rad_keys, "TEST Metrics Radar (DS1 vs DS2)")
    plot_global_training_dashboard(glob_df)
    plot_client_evolution(loc_df)
    plot_theta_evolution(theta_evo_df)

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
    "fusion_name": "raw_only_head",
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
    "theta_evolution_mean": theta_evo_df.to_dict(orient="list"),
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
print(f"Ablation: {METHOD_INFO['ablation_name']}")
print(f"Best round: {best_round_saved}")
print(f"Adaptive clients => DS1={n_clients_ds1}, DS2={n_clients_ds2}, TOTAL={CLIENTS_TOTAL}")
print(f"Rounds completed: {CFG['rounds']}")
print(f"Global TEST acc: {safe_float(test_global.get('acc')):.4f}")
print(f"Global TEST f1_macro: {safe_float(test_global.get('f1_macro')):.4f}")
print(f"DS1 TEST acc: {safe_float(test_ds1.get('acc')):.4f}")
print(f"DS2 TEST acc: {safe_float(test_ds2.get('acc')):.4f}")
print(f"DS1 final strategy: {choice_ds1['strategy']} | names={choice_ds1['theta_names']}")
print(f"DS2 final strategy: {choice_ds2['strategy']} | names={choice_ds2['theta_names']}")

# %% [markdown] Cell 9
# **5. Enhanced Only**

# %% Cell 10
# ============================================================
# KAGGLE FULL SCRIPT
# ABLATION 5: ENHANCED ONLY
# TRUE FL + RL-UCB + RACE-FELCM + ENHANCED-ONLY + ResNet-50
# METHOD ACRONYM: ARCF-Net
# FULL FORM: Adaptive RACE-FELCM with Enhanced-Only Network
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
# - ABLATION 5: ENHANCED ONLY INPUT BRANCH
# - NO PLOTS
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
print("TRUE FL + RL-UCB + RACE-FELCM + ENHANCED-ONLY + ResNet-50")
print("METHOD: ARCF-Net (Ablation 5 - Enhanced Only)")
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

    # final inference
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
    "make_plots": False,
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
    "full_form": "Adaptive RACE-FELCM with Enhanced-Only Network",
    "preprocessing_full_form": "Robust Adaptive Context-Enhanced Fuzzy Edge Local Contrast Mapping",
    "fusion_full_form": "Enhanced-Only Head",
    "backbone_full_form": "Residual Network-50",
    "ablation_name": "Ablation 5 - Enhanced Only",
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
# STEP 8: MODEL — ResNet-50 ENHANCED-ONLY HEAD
# ============================================================
print("\n" + "=" * 118)
print("STEP 8: MODEL — ResNet-50 ENHANCED-ONLY HEAD")
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

class ResNet50EnhancedOnly(nn.Module):
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
        self.cond_proj = nn.Linear(cond_dim, fuse_dim)

        self.enh_proj = CompactProjector(self.backbone_dim, fuse_dim)
        self.out_norm = nn.LayerNorm(fuse_dim)

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

        f_enh = self._encode(x_enh_n)
        fe = self.enh_proj(f_enh)
        fused = self.out_norm(fe + self.cond_proj(cond))

        embed = self.neck(fused)
        logits = self.classifier(embed)

        if return_extra:
            gates = torch.zeros((x_raw_n.size(0), 3), device=x_raw_n.device, dtype=embed.dtype)
            gates[:, 1] = 1.0
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

global_model = ResNet50EnhancedOnly(
    num_classes=NUM_CLASSES,
    cond_dim=64,
    fuse_dim=256,
    embed_dim=256,
    pretrained=True,
).to(DEVICE)

set_trainable_for_round(global_model, rnd=1)
total_params, trainable_params = count_params(global_model)

print("Backbone: ResNet-50 | pretrained_loaded=True")
print("Branch ablation: ENHANCED ONLY")
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
print("Preprocessing: RACE-FELCM (RL-selected)")
print("Input branch ablation: ENHANCED ONLY")
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

        local_model = ResNet50EnhancedOnly(
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
    {"setting": "ARCF-Net (Enhanced Only)", "split": "VAL",  "dataset": "ds1",           **compact_metrics(val_ds1)},
    {"setting": "ARCF-Net (Enhanced Only)", "split": "VAL",  "dataset": "ds2",           **compact_metrics(val_ds2)},
    {"setting": "ARCF-Net (Enhanced Only)", "split": "VAL",  "dataset": "global_equal",  **compact_metrics(val_global)},
    {"setting": "ARCF-Net (Enhanced Only)", "split": "TEST", "dataset": "ds1",           **compact_metrics(test_ds1)},
    {"setting": "ARCF-Net (Enhanced Only)", "split": "TEST", "dataset": "ds2",           **compact_metrics(test_ds2)},
    {"setting": "ARCF-Net (Enhanced Only)", "split": "TEST", "dataset": "global_equal",  **compact_metrics(test_global)},
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

def build_theta_evolution_table(loc_df):
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
    return theta_evo

def plot_theta_evolution(theta_evo):
    fig, ax = plt.subplots(figsize=(15, 6), constrained_layout=True, facecolor="white")
    for col in theta_evo.columns:
        if col != "round":
            ax.plot(theta_evo["round"], theta_evo[col], marker="o", label=col)
    ax.set_title("RACE-FELCM Parameter Evolution (Mean across clients)", fontweight="bold")
    ax.set_xlabel("Round", fontweight="bold")
    ax.legend(ncol=2)
    ax.grid(alpha=0.20)
    plt.show()

theta_evo_df = build_theta_evolution_table(loc_df)

if CFG["make_plots"]:
    rad_keys = ["acc", "f1_macro", "precision_macro", "recall_macro", "auc_roc_macro_ovr", "mcc", "kappa", "log_loss"]
    radar_plot(test_ds1, test_ds2, rad_keys, "TEST Metrics Radar (DS1 vs DS2)")
    plot_global_training_dashboard(glob_df)
    plot_client_evolution(loc_df)
    plot_theta_evolution(theta_evo_df)

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
    "fusion_name": "enhanced_only_head",
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
    "theta_evolution_mean": theta_evo_df.to_dict(orient="list"),
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
print(f"Ablation: {METHOD_INFO['ablation_name']}")
print(f"Best round: {best_round_saved}")
print(f"Adaptive clients => DS1={n_clients_ds1}, DS2={n_clients_ds2}, TOTAL={CLIENTS_TOTAL}")
print(f"Rounds completed: {CFG['rounds']}")
print(f"Global TEST acc: {safe_float(test_global.get('acc')):.4f}")
print(f"Global TEST f1_macro: {safe_float(test_global.get('f1_macro')):.4f}")
print(f"DS1 TEST acc: {safe_float(test_ds1.get('acc')):.4f}")
print(f"DS2 TEST acc: {safe_float(test_ds2.get('acc')):.4f}")
print(f"DS1 final strategy: {choice_ds1['strategy']} | names={choice_ds1['theta_names']}")
print(f"DS2 final strategy: {choice_ds2['strategy']} | names={choice_ds2['theta_names']}")

# %% [markdown] Cell 11
# **6. raw + enhanced**

# %% Cell 12
# ============================================================
# KAGGLE FULL SCRIPT
# ABLATION 6: RAW AND ENHANCED
# TRUE FL + RL-UCB + RACE-FELCM + RAW+ENHANCED + ResNet-50
# METHOD ACRONYM: ARCF-Net
# FULL FORM: Adaptive RACE-FELCM with Raw-Enhanced Fusion Network
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
# - ABLATION 6: RAW + ENHANCED INPUT BRANCHES ONLY
# - NO PLOTS
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
print("TRUE FL + RL-UCB + RACE-FELCM + RAW+ENHANCED + ResNet-50")
print("METHOD: ARCF-Net (Ablation 6 - Raw and Enhanced)")
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

    # final inference
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
    "make_plots": False,
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
    "full_form": "Adaptive RACE-FELCM with Raw-Enhanced Fusion Network",
    "preprocessing_full_form": "Robust Adaptive Context-Enhanced Fuzzy Edge Local Contrast Mapping",
    "fusion_full_form": "Raw-Enhanced Adaptive Fusion",
    "backbone_full_form": "Residual Network-50",
    "ablation_name": "Ablation 6 - Raw and Enhanced",
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
# STEP 8: MODEL — ResNet-50 RAW+ENHANCED FUSION
# ============================================================
print("\n" + "=" * 118)
print("STEP 8: MODEL — ResNet-50 RAW+ENHANCED FUSION")
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

class RawEnhancedFusion(nn.Module):
    def __init__(self, in_dim, fuse_dim=256, cond_dim=64):
        super().__init__()
        self.proj_raw = CompactProjector(in_dim, fuse_dim)
        self.proj_enh = CompactProjector(in_dim, fuse_dim)

        self.router = nn.Sequential(
            nn.Linear(fuse_dim * 3 + cond_dim, fuse_dim),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(fuse_dim, 2),
        )

        self.refine = nn.Sequential(
            nn.Linear(fuse_dim * 2 + cond_dim, fuse_dim),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(fuse_dim, fuse_dim),
        )

        self.out_norm = nn.LayerNorm(fuse_dim)

    def forward(self, f_raw, f_enh, cond):
        fr = self.proj_raw(f_raw)
        fe = self.proj_enh(f_enh)

        d_re = torch.abs(fr - fe)

        router_in = torch.cat([fr, fe, d_re, cond], dim=1)
        gates2 = torch.softmax(self.router(router_in), dim=1)

        routed = gates2[:, 0:1] * fr + gates2[:, 1:2] * fe
        refine = self.refine(torch.cat([routed, d_re, cond], dim=1))
        fused = self.out_norm(routed + refine)

        gates3 = torch.cat([
            gates2[:, 0:1],
            gates2[:, 1:2],
            torch.zeros_like(gates2[:, 0:1]),
        ], dim=1)

        return fused, gates3

class ResNet50RawEnhanced(nn.Module):
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

        self.fusion = RawEnhancedFusion(self.backbone_dim, fuse_dim=fuse_dim, cond_dim=cond_dim)

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

        fused, gates = self.fusion(f_raw, f_enh, cond)
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

global_model = ResNet50RawEnhanced(
    num_classes=NUM_CLASSES,
    cond_dim=64,
    fuse_dim=256,
    embed_dim=256,
    pretrained=True,
).to(DEVICE)

set_trainable_for_round(global_model, rnd=1)
total_params, trainable_params = count_params(global_model)

print("Backbone: ResNet-50 | pretrained_loaded=True")
print("Branch ablation: RAW AND ENHANCED")
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
print("Preprocessing: RACE-FELCM (RL-selected)")
print("Input branch ablation: RAW AND ENHANCED")
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

        local_model = ResNet50RawEnhanced(
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
    {"setting": "ARCF-Net (Raw + Enhanced)", "split": "VAL",  "dataset": "ds1",           **compact_metrics(val_ds1)},
    {"setting": "ARCF-Net (Raw + Enhanced)", "split": "VAL",  "dataset": "ds2",           **compact_metrics(val_ds2)},
    {"setting": "ARCF-Net (Raw + Enhanced)", "split": "VAL",  "dataset": "global_equal",  **compact_metrics(val_global)},
    {"setting": "ARCF-Net (Raw + Enhanced)", "split": "TEST", "dataset": "ds1",           **compact_metrics(test_ds1)},
    {"setting": "ARCF-Net (Raw + Enhanced)", "split": "TEST", "dataset": "ds2",           **compact_metrics(test_ds2)},
    {"setting": "ARCF-Net (Raw + Enhanced)", "split": "TEST", "dataset": "global_equal",  **compact_metrics(test_global)},
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

def build_theta_evolution_table(loc_df):
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
    return theta_evo

def plot_theta_evolution(theta_evo):
    fig, ax = plt.subplots(figsize=(15, 6), constrained_layout=True, facecolor="white")
    for col in theta_evo.columns:
        if col != "round":
            ax.plot(theta_evo["round"], theta_evo[col], marker="o", label=col)
    ax.set_title("RACE-FELCM Parameter Evolution (Mean across clients)", fontweight="bold")
    ax.set_xlabel("Round", fontweight="bold")
    ax.legend(ncol=2)
    ax.grid(alpha=0.20)
    plt.show()

theta_evo_df = build_theta_evolution_table(loc_df)

if CFG["make_plots"]:
    rad_keys = ["acc", "f1_macro", "precision_macro", "recall_macro", "auc_roc_macro_ovr", "mcc", "kappa", "log_loss"]
    radar_plot(test_ds1, test_ds2, rad_keys, "TEST Metrics Radar (DS1 vs DS2)")
    plot_global_training_dashboard(glob_df)
    plot_client_evolution(loc_df)
    plot_theta_evolution(theta_evo_df)

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
    "fusion_name": "raw_enhanced_adaptive_fusion",
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
    "theta_evolution_mean": theta_evo_df.to_dict(orient="list"),
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
print(f"Ablation: {METHOD_INFO['ablation_name']}")
print(f"Best round: {best_round_saved}")
print(f"Adaptive clients => DS1={n_clients_ds1}, DS2={n_clients_ds2}, TOTAL={CLIENTS_TOTAL}")
print(f"Rounds completed: {CFG['rounds']}")
print(f"Global TEST acc: {safe_float(test_global.get('acc')):.4f}")
print(f"Global TEST f1_macro: {safe_float(test_global.get('f1_macro')):.4f}")
print(f"DS1 TEST acc: {safe_float(test_ds1.get('acc')):.4f}")
print(f"DS2 TEST acc: {safe_float(test_ds2.get('acc')):.4f}")
print(f"DS1 final strategy: {choice_ds1['strategy']} | names={choice_ds1['theta_names']}")
print(f"DS2 final strategy: {choice_ds2['strategy']} | names={choice_ds2['theta_names']}")

# %% [markdown] Cell 13
# **7. raw + enhanced + residual**

# %% Cell 14
# ============================================================
# KAGGLE FULL SCRIPT
# ABLATION 7: RAW, ENHANCED, AND RESIDUAL
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
# - ABLATION 7: RAW + ENHANCED + RESIDUAL
# - NO PLOTS
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
print("METHOD: ARCF-Net (Ablation 7 - Raw, Enhanced, and Residual)")
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
    "make_plots": False,
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
    "ablation_name": "Ablation 7 - Raw, Enhanced, and Residual",
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
print("Branch ablation: RAW + ENHANCED + RESIDUAL")
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
print("Input branch ablation: RAW + ENHANCED + RESIDUAL")
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
    {"setting": "ARCF-Net (Raw + Enhanced + Residual)", "split": "VAL",  "dataset": "ds1",           **compact_metrics(val_ds1)},
    {"setting": "ARCF-Net (Raw + Enhanced + Residual)", "split": "VAL",  "dataset": "ds2",           **compact_metrics(val_ds2)},
    {"setting": "ARCF-Net (Raw + Enhanced + Residual)", "split": "VAL",  "dataset": "global_equal",  **compact_metrics(val_global)},
    {"setting": "ARCF-Net (Raw + Enhanced + Residual)", "split": "TEST", "dataset": "ds1",           **compact_metrics(test_ds1)},
    {"setting": "ARCF-Net (Raw + Enhanced + Residual)", "split": "TEST", "dataset": "ds2",           **compact_metrics(test_ds2)},
    {"setting": "ARCF-Net (Raw + Enhanced + Residual)", "split": "TEST", "dataset": "global_equal",  **compact_metrics(test_global)},
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

def build_theta_evolution_table(loc_df):
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
    return theta_evo

def plot_theta_evolution(theta_evo):
    fig, ax = plt.subplots(figsize=(15, 6), constrained_layout=True, facecolor="white")
    for col in theta_evo.columns:
        if col != "round":
            ax.plot(theta_evo["round"], theta_evo[col], marker="o", label=col)
    ax.set_title("RACE-FELCM Parameter Evolution (Mean across clients)", fontweight="bold")
    ax.set_xlabel("Round", fontweight="bold")
    ax.legend(ncol=2)
    ax.grid(alpha=0.20)
    plt.show()

theta_evo_df = build_theta_evolution_table(loc_df)

if CFG["make_plots"]:
    rad_keys = ["acc", "f1_macro", "precision_macro", "recall_macro", "auc_roc_macro_ovr", "mcc", "kappa", "log_loss"]
    radar_plot(test_ds1, test_ds2, rad_keys, "TEST Metrics Radar (DS1 vs DS2)")
    plot_global_training_dashboard(glob_df)
    plot_client_evolution(loc_df)
    plot_theta_evolution(theta_evo_df)

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
    "theta_evolution_mean": theta_evo_df.to_dict(orient="list"),
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
print(f"Ablation: {METHOD_INFO['ablation_name']}")
print(f"Best round: {best_round_saved}")
print(f"Adaptive clients => DS1={n_clients_ds1}, DS2={n_clients_ds2}, TOTAL={CLIENTS_TOTAL}")
print(f"Rounds completed: {CFG['rounds']}")
print(f"Global TEST acc: {safe_float(test_global.get('acc')):.4f}")
print(f"Global TEST f1_macro: {safe_float(test_global.get('f1_macro')):.4f}")
print(f"DS1 TEST acc: {safe_float(test_ds1.get('acc')):.4f}")
print(f"DS2 TEST acc: {safe_float(test_ds2.get('acc')):.4f}")
print(f"DS1 final strategy: {choice_ds1['strategy']} | names={choice_ds1['theta_names']}")
print(f"DS2 final strategy: {choice_ds2['strategy']} | names={choice_ds2['theta_names']}")

# %% Cell 15

