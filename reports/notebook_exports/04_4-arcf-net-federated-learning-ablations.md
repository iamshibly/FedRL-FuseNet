**1.FedAvg only**


```python
# ============================================================
# KAGGLE FULL SCRIPT
# TRUE FL + RL-UCB + RACE-FELCM + CRAF + ResNet-50
# ABLATION: FEDAVG ONLY
# METHOD ACRONYM: ARCF-Net
# FULL FORM: Adaptive RACE-FELCM with CRAF Fusion Network
# ------------------------------------------------------------
# KAGGLE-READY + SYMMETRIC DS1/DS2 IMPORTANCE
# - Uses BOTH datasets
# - Reads datasets from /kaggle/input automatically
# - Exact 15 FL rounds
# - Proper FL with FedAvg only
# - RL-UCB for SHARED client-count planning + per-client preprocessing preset selection
# - Tune-aware theta probing before local training
# - Equal DS1 / DS2 importance in best-round selection and merged reporting
# - No plots
# - Saves checkpoint WITH full-process info for later replay / XAI
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

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    log_loss, confusion_matrix, roc_auc_score,
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

print("=" * 118)
print("TRUE FL + RL-UCB + RACE-FELCM + CRAF + ResNet-50")
print("METHOD: ARCF-Net (Adaptive RACE-FELCM with CRAF Fusion Network)")
print("ABLATION: FedAvg only")
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

    # true FedAvg weighting
    "fedavg_temper": 1.0,

    # misc
    "quick_hash_subset_per_split": 300,
    "preproc_val_sample_n": 400,
    "calibration_bins": 12,
}

OUTDIR = "/kaggle/working/outputs" if IS_KAGGLE else "/content/outputs"
os.makedirs(OUTDIR, exist_ok=True)
MODEL_PATH = os.path.join(OUTDIR, "ARCFNet_RESNET50_FEDAVG_ONLY_checkpoint.pth")
CSV_PATH   = os.path.join(OUTDIR, "ALL_OUTPUTS_AND_METRICS_RESNET50_FEDAVG_ONLY.csv")

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
    "federated_variant": "FedAvg only",
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
print("STEP 6: DATA LOADERS")
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

        counts = df_src.loc[tr_idx, "label"].value_counts().reindex(labels, fill_value=0)

        clients.append({
            "gid": gid,
            "local_id": local_id,
            "dataset": ds_name,
            "source_id": source_id,
            "train_loader": train_loader,
            "tune_loader": tune_loader,
            "val_loader": val_loader,
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

        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 1], [-1, 0, 1]], dtype=torch.float32)
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
# STEP 9: LOSSES
# ============================================================
print("\n" + "=" * 118)
print("STEP 9: LOSSES")
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

def train_one_epoch(model, loader, optimizer, preproc, theta, scheduler=None):
    model.train()
    freeze_backbone_bn_stats(model)
    preproc.eval()

    losses, ce_losses, correct, total = [], [], 0, 0
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

            logits, _, _ = model(x_raw_n, x_enh_n, x_res_n, theta_vec, source_id, return_extra=True)
            ce = criterion(logits, y)
            loss = ce

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

        preds = logits.argmax(dim=1)
        correct += int((preds == y).sum().item())
        total += int(y.size(0))

    return {
        "loss": float(np.mean(losses)) if losses else np.nan,
        "ce_loss": float(np.mean(ce_losses)) if ce_losses else np.nan,
        "acc": float(correct / max(1, total)),
        "train_time_s": float(time.time() - t0),
    }

def build_fedavg_weights(clients_subset):
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
# STEP 11: TRUE FEDERATED TRAINING (FEDAVG ONLY)
# ============================================================
print("\n" + "=" * 118)
print("STEP 11: TRUE FEDERATED TRAINING (FEDAVG ONLY)")
print("=" * 118)

history_global = []
history_local = []

best_reward = -1.0
best_round_saved = None
best_model_state = None
best_theta_bandit_states = None

t_global_start = time.time()

print(f"Adaptive clients => DS1={n_clients_ds1}, DS2={n_clients_ds2}, TOTAL={CLIENTS_TOTAL}")
print(f"Rounds: {CFG['rounds']} | Local epochs: {CFG['local_epochs']}")
print(f"Augmentation ON: {CFG['use_augmentation']}")
print("Transfer backbone: ResNet-50")
print("Preprocessing: RACE-FELCM")
print("Fusion: CRAF")
print("Federated method: FedAvg only")
print(f"FedAvg exponent = {CFG['fedavg_temper']:.2f}")
print(f"Best-round masses => DS1={CFG['best_round_mass_ds1']:.2f}, DS2={CFG['best_round_mass_ds2']:.2f}, min-bonus={CFG['best_round_min_bonus']:.2f}")

for rnd in range(1, CFG["rounds"] + 1):
    round_t0 = time.time()
    selected_ids = list(range(len(clients)))

    print("\n" + "=" * 118)
    print(f"ROUND {rnd}/{CFG['rounds']} | selected={selected_ids}")
    print("=" * 118)

    local_models = []
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
                scheduler=scheduler,
            )
            train_logs.append(log_ep)

        met_loc, _, _ = evaluate_full(local_model, client["val_loader"], preproc, theta, return_gates=True, use_tta=False)

        reward = score_metric(met_loc)
        client["theta_bandit"].update(theta_arm, reward)

        local_models.append(local_model)
        selected_clients_meta.append(client)

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

    agg_weights = build_fedavg_weights(selected_clients_meta)
    fedavg_update(global_model, local_models, agg_weights)

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
        best_theta_bandit_states = [copy.deepcopy(c["theta_bandit"].state_dict()) for c in clients]

if best_model_state is not None:
    global_model.load_state_dict({k: v.to(DEVICE) for k, v in best_model_state.items()})

if best_theta_bandit_states is not None:
    for c, sd in zip(clients, best_theta_bandit_states):
        c["theta_bandit"].load_state_dict(sd)

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
    {"setting": "ARCF-Net + FedAvg only", "split": "VAL",  "dataset": "ds1",          **compact_metrics(val_ds1)},
    {"setting": "ARCF-Net + FedAvg only", "split": "VAL",  "dataset": "ds2",          **compact_metrics(val_ds2)},
    {"setting": "ARCF-Net + FedAvg only", "split": "VAL",  "dataset": "global_equal", **compact_metrics(val_global)},
    {"setting": "ARCF-Net + FedAvg only", "split": "TEST", "dataset": "ds1",          **compact_metrics(test_ds1)},
    {"setting": "ARCF-Net + FedAvg only", "split": "TEST", "dataset": "ds2",          **compact_metrics(test_ds2)},
    {"setting": "ARCF-Net + FedAvg only", "split": "TEST", "dataset": "global_equal", **compact_metrics(test_global)},
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
        return pd.DataFrame(), pd.DataFrame()

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
    return dfm, summary

preproc_ds1 = theta_to_module(rep_theta_ds1).to(DEVICE)
preproc_ds2 = theta_to_module(rep_theta_ds2).to(DEVICE)

preproc_df1, preproc_summary_df1 = run_preproc_validation(val1, preproc_ds1, CFG["preproc_val_sample_n"])
preproc_df2, preproc_summary_df2 = run_preproc_validation(val2, preproc_ds2, CFG["preproc_val_sample_n"])

print_table(preproc_summary_df1, "Preprocessing validation summary (DS1 VAL sample)")
print_table(preproc_summary_df2, "Preprocessing validation summary (DS2 VAL sample)")
add_table_to_csv(preproc_summary_df1, "preprocessing_validation_summary_ds1")
add_table_to_csv(preproc_summary_df2, "preprocessing_validation_summary_ds2")

# ============================================================
# STEP 14: PARAMETER EVOLUTION TABLE
# ============================================================
print("\n" + "=" * 118)
print("STEP 14: PARAMETER EVOLUTION TABLE")
print("=" * 118)

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
for ccol in theta_cols:
    if ccol in loc_copy.columns:
        loc_copy[ccol] = pd.to_numeric(loc_copy[ccol], errors="coerce")

theta_evo = loc_copy.groupby("round")[theta_cols].mean(numeric_only=True).reset_index()
print_table(theta_evo, "Mean selected preprocessing parameters over rounds")
add_table_to_csv(theta_evo, "theta_evolution_mean")

# ============================================================
# STEP 15: SAVE CHECKPOINT + CSV
# ============================================================
print("\n" + "=" * 118)
print("STEP 15: SAVING ONLY TWO FILES (CHECKPOINT + ONE CSV)")
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
    "federated_variant": "FedAvg only",

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
    "theta_evolution_mean": theta_evo.to_dict(orient="list"),
}

torch.save(checkpoint, MODEL_PATH)
print(f"✅ Saved checkpoint: {MODEL_PATH}")

all_df.to_csv(CSV_PATH, index=False)
print(f"✅ Saved CSV (ALL outputs): {CSV_PATH}")

print("\nDONE ✅")
print(f"Method: {METHOD_INFO['acronym']} = {METHOD_INFO['full_form']}")
print(f"Backbone: {METHOD_INFO['backbone_full_form']}")
print("Federated variant: FedAvg only")
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
    ABLATION: FedAvg only
    ======================================================================================================================
    ENV: KAGGLE | DEVICE: cuda | torch=2.9.0+cu126
    ======================================================================================================================
    
    ======================================================================================================================
    STEP 0: ACCESS DATASETS
    ======================================================================================================================
    Dataset-1 RAW root detected:
      /kaggle/input/datasets/orvile/pmram-bangladeshi-brain-cancer-mri-dataset/PMRAM Bangladeshi Brain Cancer - MRI Dataset/PMRAM Bangladeshi Brain Cancer - MRI Dataset/Raw Data/Raw
    Dataset-2 root detected:
      /kaggle/input/datasets/chubskuy/brain-tumor-image/Testing
    
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
    ds2: glioma -> glioma | 300 images
    ds2: meningioma -> meningioma | 306 images
    ds2: notumor -> notumor | 405 images
    ds2: pituitary -> pituitary | 300 images
    
    Dataset-1 images: 1505
    label
    glioma        373
    meningioma    363
    notumor       396
    pituitary     373
    Name: count, dtype: int64
    
    Dataset-2 images: 1311
    label
    glioma        300
    meningioma    306
    notumor       405
    pituitary     300
    Name: count, dtype: int64
    
    ======================================================================================================================
    STEP 2: TRAIN / VAL / TEST SPLIT (PER DATASET)
    ======================================================================================================================
    DS1 TRAIN: 1053 | VAL: 226 | TEST: 226
    DS2 TRAIN: 917 | VAL: 197 | TEST: 197
    
    ======================================================================================================================
    STEP 2.5: SANITY / LEAKAGE CHECKS
    ======================================================================================================================
    
    ----------------------------------------------------------------------------------------------------------------------
    Leakage / Sanity Summary — ds1
    ----------------------------------------------------------------------------------------------------------------------



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
      <td>5</td>
      <td>5</td>
      <td>6</td>
      <td>298</td>
      <td>222</td>
      <td>224</td>
    </tr>
  </tbody>
</table>
</div>


    
    ----------------------------------------------------------------------------------------------------------------------
    Leakage / Sanity Summary — ds2
    ----------------------------------------------------------------------------------------------------------------------



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
      <td>917</td>
      <td>197</td>
      <td>197</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>300</td>
      <td>194</td>
      <td>197</td>
    </tr>
  </tbody>
</table>
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
      <td>0.755612</td>
      <td>0.767340</td>
      <td>0.761476</td>
      <td>0.761476</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>4</td>
      <td>0.679705</td>
      <td>0.635628</td>
      <td>0.657667</td>
      <td>0.657667</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>5</td>
      <td>0.721049</td>
      <td>0.775174</td>
      <td>0.748111</td>
      <td>0.748111</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>3</td>
      <td>0.841039</td>
      <td>0.726403</td>
      <td>0.783721</td>
      <td>0.772599</td>
      <td>2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>5</td>
      <td>0.688321</td>
      <td>0.701388</td>
      <td>0.694855</td>
      <td>0.721483</td>
      <td>2</td>
    </tr>
    <tr>
      <th>5</th>
      <td>6</td>
      <td>4</td>
      <td>0.665267</td>
      <td>0.691320</td>
      <td>0.678293</td>
      <td>0.667980</td>
      <td>2</td>
    </tr>
    <tr>
      <th>6</th>
      <td>7</td>
      <td>3</td>
      <td>0.766538</td>
      <td>0.693916</td>
      <td>0.730227</td>
      <td>0.758475</td>
      <td>3</td>
    </tr>
    <tr>
      <th>7</th>
      <td>8</td>
      <td>5</td>
      <td>0.677792</td>
      <td>0.707763</td>
      <td>0.692778</td>
      <td>0.711915</td>
      <td>3</td>
    </tr>
    <tr>
      <th>8</th>
      <td>9</td>
      <td>4</td>
      <td>0.715564</td>
      <td>0.681741</td>
      <td>0.698653</td>
      <td>0.678204</td>
      <td>3</td>
    </tr>
    <tr>
      <th>9</th>
      <td>10</td>
      <td>3</td>
      <td>0.734619</td>
      <td>0.707902</td>
      <td>0.721261</td>
      <td>0.749171</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>


    
    ======================================================================================================================
    STEP 5: FINAL NON-IID CLIENT PARTITIONING
    ======================================================================================================================
    
    ======================================================================================================================
    STEP 6: DATA LOADERS
    ======================================================================================================================
    ds1 | client_0 | train=286 | tune=45 | val=40
    ds1 | client_1 | train=257 | tune=41 | val=36
    ds1 | client_2 | train=269 | tune=42 | val=37
    ds2 | client_3 | train=100 | tune=16 | val=14
    ds2 | client_4 | train=342 | tune=54 | val=47
    ds2 | client_5 | train=265 | tune=42 | val=37
    
    ----------------------------------------------------------------------------------------------------------------------
    Adaptive client class distribution
    ----------------------------------------------------------------------------------------------------------------------



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
      <td>286</td>
      <td>45</td>
      <td>40</td>
      <td>40</td>
      <td>33</td>
      <td>166</td>
      <td>47</td>
    </tr>
    <tr>
      <th>1</th>
      <td>client_1</td>
      <td>ds1</td>
      <td>257</td>
      <td>41</td>
      <td>36</td>
      <td>135</td>
      <td>4</td>
      <td>31</td>
      <td>87</td>
    </tr>
    <tr>
      <th>2</th>
      <td>client_2</td>
      <td>ds1</td>
      <td>269</td>
      <td>42</td>
      <td>37</td>
      <td>25</td>
      <td>159</td>
      <td>17</td>
      <td>68</td>
    </tr>
    <tr>
      <th>3</th>
      <td>client_3</td>
      <td>ds2</td>
      <td>100</td>
      <td>16</td>
      <td>14</td>
      <td>74</td>
      <td>18</td>
      <td>4</td>
      <td>4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>client_4</td>
      <td>ds2</td>
      <td>342</td>
      <td>54</td>
      <td>47</td>
      <td>5</td>
      <td>121</td>
      <td>154</td>
      <td>62</td>
    </tr>
    <tr>
      <th>5</th>
      <td>client_5</td>
      <td>ds2</td>
      <td>265</td>
      <td>42</td>
      <td>37</td>
      <td>82</td>
      <td>25</td>
      <td>61</td>
      <td>97</td>
    </tr>
  </tbody>
</table>
</div>


    Augmentation: ON ✅
    Preprocessing: ON ✅
    Total adaptive clients: 6
    
    ======================================================================================================================
    STEP 7: NOVEL PREPROCESSING — RACE-FELCM
    ======================================================================================================================
    
    ======================================================================================================================
    STEP 8: MODEL — ResNet-50 + CRAF Fusion
    ======================================================================================================================
    Backbone: ResNet-50 | pretrained_loaded=True
    Total params: 25,790,855
    Trainable params: 2,282,823 (8.85%)
    
    ======================================================================================================================
    STEP 9: LOSSES
    ======================================================================================================================
    
    ======================================================================================================================
    STEP 10: TUNE-AWARE RL-UCB PREPROCESSING SELECTION
    ======================================================================================================================
    
    ======================================================================================================================
    STEP 11: TRUE FEDERATED TRAINING (FEDAVG ONLY)
    ======================================================================================================================
    Adaptive clients => DS1=3, DS2=3, TOTAL=6
    Rounds: 15 | Local epochs: 2
    Augmentation ON: True
    Transfer backbone: ResNet-50
    Preprocessing: RACE-FELCM
    Fusion: CRAF
    Federated method: FedAvg only
    FedAvg exponent = 1.00
    Best-round masses => DS1=0.50, DS2=0.50, min-bonus=0.15
    
    ======================================================================================================================
    ROUND 1/15 | selected=[0, 1, 2, 3, 4, 5]
    ======================================================================================================================
    Client 0 (ds1) | train_acc=0.6783 | val_acc=0.8000 | val_f1=0.6846 | val_auc=0.9155 | reward=0.7134 | theta=race_focus (g=1.10, a=0.36, b=6.40, t=2.70, k=3, eg=0.17, mix=0.86)
    Client 1 (ds1) | train_acc=0.7296 | val_acc=0.9167 | val_f1=0.7099 | val_auc=0.9893 | reward=0.7616 | theta=race_focus (g=1.10, a=0.36, b=6.40, t=2.70, k=3, eg=0.17, mix=0.86)
    Client 2 (ds1) | train_acc=0.6580 | val_acc=0.5405 | val_f1=0.5729 | val_auc=0.8447 | reward=0.5648 | theta=race_focus (g=1.10, a=0.36, b=6.40, t=2.70, k=3, eg=0.17, mix=0.86)
    Client 3 (ds2) | train_acc=0.5600 | val_acc=0.7857 | val_f1=0.5185 | val_auc=nan | reward=0.5853 | theta=race_robust (g=1.08, a=0.22, b=4.50, t=2.80, k=5, eg=0.11, mix=0.76)
    Client 4 (ds2) | train_acc=0.6930 | val_acc=0.8936 | val_f1=0.6803 | val_auc=0.9325 | reward=0.7337 | theta=race_smoothmix (g=0.90, a=0.20, b=4.00, t=2.10, k=7, eg=0.07, mix=0.70)
    Client 5 (ds2) | train_acc=0.6358 | val_acc=0.7568 | val_f1=0.6264 | val_auc=0.9566 | reward=0.6590 | theta=race_robust (g=1.08, a=0.22, b=4.50, t=2.80, k=5, eg=0.11, mix=0.76)
    
    ----------------------------------------------------------------------------------------------------------------------
    GLOBAL VAL (Round 1) | global_acc=0.7867 | global_f1=0.6472 | ds1_acc=0.7522 | ds1_f1=0.6561 | ds2_acc=0.8265 | ds2_f1=0.6369 | reward=0.7842 | round_time=43.4s
    ----------------------------------------------------------------------------------------------------------------------
    
    ======================================================================================================================
    ROUND 2/15 | selected=[0, 1, 2, 3, 4, 5]
    ======================================================================================================================
    Client 0 (ds1) | train_acc=0.8479 | val_acc=0.7250 | val_f1=0.5425 | val_auc=0.9584 | reward=0.5881 | theta=race_edge_plus (g=1.02, a=0.32, b=6.00, t=2.60, k=3, eg=0.15, mix=0.84)
    Client 1 (ds1) | train_acc=0.9047 | val_acc=0.8611 | val_f1=0.6345 | val_auc=0.9569 | reward=0.6911 | theta=race_edge_plus (g=1.02, a=0.32, b=6.00, t=2.60, k=3, eg=0.15, mix=0.84)
    Client 2 (ds1) | train_acc=0.8327 | val_acc=0.7838 | val_f1=0.6667 | val_auc=0.9445 | reward=0.6959 | theta=race_edge_plus (g=1.02, a=0.32, b=6.00, t=2.60, k=3, eg=0.15, mix=0.84)
    Client 3 (ds2) | train_acc=0.6800 | val_acc=0.8571 | val_f1=0.7063 | val_auc=nan | reward=0.7440 | theta=race_soft (g=0.95, a=0.18, b=3.80, t=2.20, k=3, eg=0.08, mix=0.72)
    Client 4 (ds2) | train_acc=0.7807 | val_acc=0.8936 | val_f1=0.6788 | val_auc=0.9717 | reward=0.7325 | theta=race_robust (g=1.08, a=0.22, b=4.50, t=2.80, k=5, eg=0.11, mix=0.76)
    Client 5 (ds2) | train_acc=0.8038 | val_acc=0.8108 | val_f1=0.6423 | val_auc=0.9540 | reward=0.6844 | theta=race_soft (g=0.95, a=0.18, b=3.80, t=2.20, k=3, eg=0.08, mix=0.72)
    
    ----------------------------------------------------------------------------------------------------------------------
    GLOBAL VAL (Round 2) | global_acc=0.8199 | global_f1=0.6387 | ds1_acc=0.7876 | ds1_f1=0.6125 | ds2_acc=0.8571 | ds2_f1=0.6690 | reward=0.7846 | round_time=36.0s
    ----------------------------------------------------------------------------------------------------------------------
    
    ======================================================================================================================
    ROUND 3/15 | selected=[0, 1, 2, 3, 4, 5]
    ======================================================================================================================
    Client 0 (ds1) | train_acc=0.8671 | val_acc=0.8000 | val_f1=0.6713 | val_auc=0.9183 | reward=0.7034 | theta=race_robust (g=1.08, a=0.22, b=4.50, t=2.80, k=5, eg=0.11, mix=0.76)
    Client 1 (ds1) | train_acc=0.8521 | val_acc=0.9444 | val_f1=0.7268 | val_auc=0.9811 | reward=0.7812 | theta=race_texture (g=0.92, a=0.34, b=5.80, t=2.30, k=7, eg=0.10, mix=0.80)
    Client 2 (ds1) | train_acc=0.7881 | val_acc=0.9730 | val_f1=0.9111 | val_auc=1.0000 | reward=0.9266 | theta=race_robust (g=1.08, a=0.22, b=4.50, t=2.80, k=5, eg=0.11, mix=0.76)
    Client 3 (ds2) | train_acc=0.8250 | val_acc=0.8571 | val_f1=0.8030 | val_auc=nan | reward=0.8166 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    Client 4 (ds2) | train_acc=0.8260 | val_acc=0.9149 | val_f1=0.7021 | val_auc=0.9259 | reward=0.7553 | theta=race_soft (g=0.95, a=0.18, b=3.80, t=2.20, k=3, eg=0.08, mix=0.72)
    Client 5 (ds2) | train_acc=0.8208 | val_acc=0.8378 | val_f1=0.6778 | val_auc=0.9611 | reward=0.7178 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    
    ----------------------------------------------------------------------------------------------------------------------
    GLOBAL VAL (Round 3) | global_acc=0.8910 | global_f1=0.7396 | ds1_acc=0.9027 | ds1_f1=0.7675 | ds2_acc=0.8776 | ds2_f1=0.7073 | reward=0.8881 | round_time=55.4s
    ----------------------------------------------------------------------------------------------------------------------
    
    ======================================================================================================================
    ROUND 4/15 | selected=[0, 1, 2, 3, 4, 5]
    ======================================================================================================================
    Client 0 (ds1) | train_acc=0.8147 | val_acc=0.8500 | val_f1=0.7513 | val_auc=0.9737 | reward=0.7760 | theta=race_texture (g=0.92, a=0.34, b=5.80, t=2.30, k=7, eg=0.10, mix=0.80)
    Client 1 (ds1) | train_acc=0.9066 | val_acc=0.9722 | val_f1=0.7436 | val_auc=0.9913 | reward=0.8007 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    Client 2 (ds1) | train_acc=0.8476 | val_acc=0.9459 | val_f1=0.9381 | val_auc=0.9981 | reward=0.9401 | theta=race_sharp (g=1.04, a=0.30, b=5.20, t=2.50, k=5, eg=0.12, mix=0.82)
    Client 3 (ds2) | train_acc=0.8200 | val_acc=0.8571 | val_f1=0.5697 | val_auc=nan | reward=0.6416 | theta=race_soft (g=0.95, a=0.18, b=3.80, t=2.20, k=3, eg=0.08, mix=0.72)
    Client 4 (ds2) | train_acc=0.8713 | val_acc=0.9149 | val_f1=0.6835 | val_auc=0.9821 | reward=0.7414 | theta=race_smoothmix (g=0.90, a=0.20, b=4.00, t=2.10, k=7, eg=0.07, mix=0.70)
    Client 5 (ds2) | train_acc=0.7887 | val_acc=0.8649 | val_f1=0.8414 | val_auc=0.9859 | reward=0.8473 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    
    ----------------------------------------------------------------------------------------------------------------------
    GLOBAL VAL (Round 4) | global_acc=0.9052 | global_f1=0.7714 | ds1_acc=0.9204 | ds1_f1=0.8100 | ds2_acc=0.8878 | ds2_f1=0.7269 | reward=0.9174 | round_time=41.2s
    ----------------------------------------------------------------------------------------------------------------------
    
    ======================================================================================================================
    ROUND 5/15 | selected=[0, 1, 2, 3, 4, 5]
    ======================================================================================================================
    Client 0 (ds1) | train_acc=0.8951 | val_acc=0.7750 | val_f1=0.6447 | val_auc=0.9696 | reward=0.6773 | theta=race_sharp (g=1.04, a=0.30, b=5.20, t=2.50, k=5, eg=0.12, mix=0.82)
    Client 1 (ds1) | train_acc=0.9280 | val_acc=0.9722 | val_f1=0.7436 | val_auc=0.9992 | reward=0.8007 | theta=race_sharp (g=1.04, a=0.30, b=5.20, t=2.50, k=5, eg=0.12, mix=0.82)
    Client 2 (ds1) | train_acc=0.9238 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | reward=1.0000 | theta=race_texture (g=0.92, a=0.34, b=5.80, t=2.30, k=7, eg=0.10, mix=0.80)
    Client 3 (ds2) | train_acc=0.8900 | val_acc=0.8571 | val_f1=0.5397 | val_auc=nan | reward=0.6190 | theta=race_smoothmix (g=0.90, a=0.20, b=4.00, t=2.10, k=7, eg=0.07, mix=0.70)
    Client 4 (ds2) | train_acc=0.8962 | val_acc=0.9787 | val_f1=0.9762 | val_auc=0.9979 | reward=0.9768 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    Client 5 (ds2) | train_acc=0.8981 | val_acc=0.8108 | val_f1=0.6515 | val_auc=0.9550 | reward=0.6913 | theta=race_smoothmix (g=0.90, a=0.20, b=4.00, t=2.10, k=7, eg=0.07, mix=0.70)
    
    ----------------------------------------------------------------------------------------------------------------------
    GLOBAL VAL (Round 5) | global_acc=0.9052 | global_f1=0.7919 | ds1_acc=0.9115 | ds1_f1=0.7925 | ds2_acc=0.8980 | ds2_f1=0.7912 | reward=0.9428 | round_time=40.7s
    ----------------------------------------------------------------------------------------------------------------------
    
    ======================================================================================================================
    ROUND 6/15 | selected=[0, 1, 2, 3, 4, 5]
    ======================================================================================================================
    Client 0 (ds1) | train_acc=0.9353 | val_acc=0.8250 | val_f1=0.6578 | val_auc=0.9804 | reward=0.6996 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    Client 1 (ds1) | train_acc=0.9553 | val_acc=0.9167 | val_f1=0.6851 | val_auc=1.0000 | reward=0.7430 | theta=race_robust (g=1.08, a=0.22, b=4.50, t=2.80, k=5, eg=0.11, mix=0.76)
    Client 2 (ds1) | train_acc=0.9684 | val_acc=0.9459 | val_f1=0.9381 | val_auc=0.9860 | reward=0.9401 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    Client 3 (ds2) | train_acc=0.9700 | val_acc=1.0000 | val_f1=1.0000 | val_auc=nan | reward=1.0000 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    Client 4 (ds2) | train_acc=0.9444 | val_acc=0.9149 | val_f1=0.6773 | val_auc=0.9961 | reward=0.7367 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    Client 5 (ds2) | train_acc=0.9038 | val_acc=0.8378 | val_f1=0.7796 | val_auc=0.9460 | reward=0.7942 | theta=race_smoothmix (g=0.90, a=0.20, b=4.00, t=2.10, k=7, eg=0.07, mix=0.70)
    
    ----------------------------------------------------------------------------------------------------------------------
    GLOBAL VAL (Round 6) | global_acc=0.8957 | global_f1=0.7600 | ds1_acc=0.8938 | ds1_f1=0.7583 | ds2_acc=0.8980 | ds2_f1=0.7620 | reward=0.9129 | round_time=41.1s
    ----------------------------------------------------------------------------------------------------------------------
    
    ======================================================================================================================
    ROUND 7/15 | selected=[0, 1, 2, 3, 4, 5]
    ======================================================================================================================
    Client 0 (ds1) | train_acc=0.9633 | val_acc=0.8000 | val_f1=0.6700 | val_auc=0.9793 | reward=0.7025 | theta=race_texture (g=0.92, a=0.34, b=5.80, t=2.30, k=7, eg=0.10, mix=0.80)
    Client 1 (ds1) | train_acc=0.9805 | val_acc=0.9722 | val_f1=0.7436 | val_auc=0.9992 | reward=0.8007 | theta=race_sharp (g=1.04, a=0.30, b=5.20, t=2.50, k=5, eg=0.12, mix=0.82)
    Client 2 (ds1) | train_acc=0.9442 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | reward=1.0000 | theta=race_texture (g=0.92, a=0.34, b=5.80, t=2.30, k=7, eg=0.10, mix=0.80)
    Client 3 (ds2) | train_acc=0.9400 | val_acc=0.9286 | val_f1=0.9175 | val_auc=nan | reward=0.9202 | theta=race_smoothmix (g=0.90, a=0.20, b=4.00, t=2.10, k=7, eg=0.07, mix=0.70)
    Client 4 (ds2) | train_acc=0.9284 | val_acc=0.9149 | val_f1=0.9261 | val_auc=0.9753 | reward=0.9233 | theta=race_soft (g=0.95, a=0.18, b=3.80, t=2.20, k=3, eg=0.08, mix=0.72)
    Client 5 (ds2) | train_acc=0.9170 | val_acc=0.9189 | val_f1=0.8907 | val_auc=0.9787 | reward=0.8978 | theta=race_soft (g=0.95, a=0.18, b=3.80, t=2.20, k=3, eg=0.08, mix=0.72)
    
    ----------------------------------------------------------------------------------------------------------------------
    GLOBAL VAL (Round 7) | global_acc=0.9194 | global_f1=0.8526 | ds1_acc=0.9204 | ds1_f1=0.8015 | ds2_acc=0.9184 | ds2_f1=0.9115 | reward=0.9969 | round_time=40.8s
    ----------------------------------------------------------------------------------------------------------------------
    
    ======================================================================================================================
    ROUND 8/15 | selected=[0, 1, 2, 3, 4, 5]
    ======================================================================================================================
    Client 0 (ds1) | train_acc=0.9441 | val_acc=0.8750 | val_f1=0.8196 | val_auc=0.9814 | reward=0.8334 | theta=race_focus (g=1.10, a=0.36, b=6.40, t=2.70, k=3, eg=0.17, mix=0.86)
    Client 1 (ds1) | train_acc=0.9436 | val_acc=0.8889 | val_f1=0.9071 | val_auc=0.9980 | reward=0.9026 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    Client 2 (ds1) | train_acc=0.9387 | val_acc=0.9459 | val_f1=0.9381 | val_auc=0.9992 | reward=0.9401 | theta=race_sharp (g=1.04, a=0.30, b=5.20, t=2.50, k=5, eg=0.12, mix=0.82)
    Client 3 (ds2) | train_acc=0.9400 | val_acc=0.9286 | val_f1=0.9175 | val_auc=nan | reward=0.9202 | theta=race_robust (g=1.08, a=0.22, b=4.50, t=2.80, k=5, eg=0.11, mix=0.76)
    Client 4 (ds2) | train_acc=0.9006 | val_acc=0.9362 | val_f1=0.9402 | val_auc=0.9972 | reward=0.9392 | theta=race_robust (g=1.08, a=0.22, b=4.50, t=2.80, k=5, eg=0.11, mix=0.76)
    Client 5 (ds2) | train_acc=0.9396 | val_acc=0.8919 | val_f1=0.8625 | val_auc=0.9866 | reward=0.8699 | theta=race_robust (g=1.08, a=0.22, b=4.50, t=2.80, k=5, eg=0.11, mix=0.76)
    
    ----------------------------------------------------------------------------------------------------------------------
    GLOBAL VAL (Round 8) | global_acc=0.9100 | global_f1=0.8962 | ds1_acc=0.9027 | ds1_f1=0.8863 | ds2_acc=0.9184 | ds2_f1=0.9076 | reward=1.0339 | round_time=41.0s
    ----------------------------------------------------------------------------------------------------------------------
    
    ======================================================================================================================
    ROUND 9/15 | selected=[0, 1, 2, 3, 4, 5]
    ======================================================================================================================
    Client 0 (ds1) | train_acc=0.9615 | val_acc=0.9000 | val_f1=0.8549 | val_auc=0.9916 | reward=0.8662 | theta=race_robust (g=1.08, a=0.22, b=4.50, t=2.80, k=5, eg=0.11, mix=0.76)
    Client 1 (ds1) | train_acc=0.9475 | val_acc=0.9444 | val_f1=0.7268 | val_auc=0.9985 | reward=0.7812 | theta=race_texture (g=0.92, a=0.34, b=5.80, t=2.30, k=7, eg=0.10, mix=0.80)
    Client 2 (ds1) | train_acc=0.9461 | val_acc=0.9459 | val_f1=0.9103 | val_auc=0.9973 | reward=0.9192 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    Client 3 (ds2) | train_acc=0.9400 | val_acc=1.0000 | val_f1=1.0000 | val_auc=nan | reward=1.0000 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    Client 4 (ds2) | train_acc=0.9488 | val_acc=0.9362 | val_f1=0.9402 | val_auc=0.9888 | reward=0.9392 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    Client 5 (ds2) | train_acc=0.9264 | val_acc=0.8108 | val_f1=0.7128 | val_auc=0.9205 | reward=0.7373 | theta=race_soft (g=0.95, a=0.18, b=3.80, t=2.20, k=3, eg=0.08, mix=0.72)
    
    ----------------------------------------------------------------------------------------------------------------------
    GLOBAL VAL (Round 9) | global_acc=0.9147 | global_f1=0.8465 | ds1_acc=0.9292 | ds1_f1=0.8322 | ds2_acc=0.8980 | ds2_f1=0.8629 | reward=0.9925 | round_time=41.2s
    ----------------------------------------------------------------------------------------------------------------------
    
    ======================================================================================================================
    ROUND 10/15 | selected=[0, 1, 2, 3, 4, 5]
    ======================================================================================================================
    Client 0 (ds1) | train_acc=0.9615 | val_acc=0.8500 | val_f1=0.7846 | val_auc=0.9743 | reward=0.8010 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    Client 1 (ds1) | train_acc=0.9591 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | reward=1.0000 | theta=race_focus (g=1.10, a=0.36, b=6.40, t=2.70, k=3, eg=0.17, mix=0.86)
    Client 2 (ds1) | train_acc=0.9647 | val_acc=0.9189 | val_f1=0.9135 | val_auc=1.0000 | reward=0.9149 | theta=race_robust (g=1.08, a=0.22, b=4.50, t=2.80, k=5, eg=0.11, mix=0.76)
    Client 3 (ds2) | train_acc=0.9350 | val_acc=0.9286 | val_f1=0.9175 | val_auc=nan | reward=0.9202 | theta=race_smoothmix (g=0.90, a=0.20, b=4.00, t=2.10, k=7, eg=0.07, mix=0.70)
    Client 4 (ds2) | train_acc=0.9327 | val_acc=0.9149 | val_f1=0.6835 | val_auc=0.9936 | reward=0.7414 | theta=race_soft (g=0.95, a=0.18, b=3.80, t=2.20, k=3, eg=0.08, mix=0.72)
    Client 5 (ds2) | train_acc=0.9623 | val_acc=0.9459 | val_f1=0.9342 | val_auc=0.9776 | reward=0.9371 | theta=race_smoothmix (g=0.90, a=0.20, b=4.00, t=2.10, k=7, eg=0.07, mix=0.70)
    
    ----------------------------------------------------------------------------------------------------------------------
    GLOBAL VAL (Round 10) | global_acc=0.9242 | global_f1=0.8565 | ds1_acc=0.9204 | ds1_f1=0.8955 | ds2_acc=0.9286 | ds2_f1=0.8116 | reward=0.9974 | round_time=40.9s
    ----------------------------------------------------------------------------------------------------------------------
    
    ======================================================================================================================
    ROUND 11/15 | selected=[0, 1, 2, 3, 4, 5]
    ======================================================================================================================
    Client 0 (ds1) | train_acc=0.9685 | val_acc=0.9000 | val_f1=0.8271 | val_auc=0.9945 | reward=0.8453 | theta=race_sharp (g=1.04, a=0.30, b=5.20, t=2.50, k=5, eg=0.12, mix=0.82)
    Client 1 (ds1) | train_acc=0.9805 | val_acc=0.9722 | val_f1=0.7436 | val_auc=1.0000 | reward=0.8007 | theta=race_robust (g=1.08, a=0.22, b=4.50, t=2.80, k=5, eg=0.11, mix=0.76)
    Client 2 (ds1) | train_acc=0.9647 | val_acc=0.9730 | val_f1=0.9664 | val_auc=1.0000 | reward=0.9680 | theta=race_edge_plus (g=1.02, a=0.32, b=6.00, t=2.60, k=3, eg=0.15, mix=0.84)
    Client 3 (ds2) | train_acc=0.9200 | val_acc=0.9286 | val_f1=0.9175 | val_auc=nan | reward=0.9202 | theta=race_robust (g=1.08, a=0.22, b=4.50, t=2.80, k=5, eg=0.11, mix=0.76)
    Client 4 (ds2) | train_acc=0.9635 | val_acc=0.9149 | val_f1=0.9261 | val_auc=0.9967 | reward=0.9233 | theta=race_robust (g=1.08, a=0.22, b=4.50, t=2.80, k=5, eg=0.11, mix=0.76)
    Client 5 (ds2) | train_acc=0.9585 | val_acc=0.8649 | val_f1=0.7845 | val_auc=0.9934 | reward=0.8046 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    
    ----------------------------------------------------------------------------------------------------------------------
    GLOBAL VAL (Round 11) | global_acc=0.9242 | global_f1=0.8579 | ds1_acc=0.9469 | ds1_f1=0.8461 | ds2_acc=0.8980 | ds2_f1=0.8714 | reward=1.0054 | round_time=40.8s
    ----------------------------------------------------------------------------------------------------------------------
    
    ======================================================================================================================
    ROUND 12/15 | selected=[0, 1, 2, 3, 4, 5]
    ======================================================================================================================
    Client 0 (ds1) | train_acc=0.9493 | val_acc=0.8750 | val_f1=0.8328 | val_auc=0.9907 | reward=0.8434 | theta=race_edge_plus (g=1.02, a=0.32, b=6.00, t=2.60, k=3, eg=0.15, mix=0.84)
    Client 1 (ds1) | train_acc=0.9844 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | reward=1.0000 | theta=race_edge_plus (g=1.02, a=0.32, b=6.00, t=2.60, k=3, eg=0.15, mix=0.84)
    Client 2 (ds1) | train_acc=0.9554 | val_acc=0.9730 | val_f1=0.9664 | val_auc=1.0000 | reward=0.9680 | theta=race_focus (g=1.10, a=0.36, b=6.40, t=2.70, k=3, eg=0.17, mix=0.86)
    Client 3 (ds2) | train_acc=0.9300 | val_acc=0.9286 | val_f1=0.7368 | val_auc=nan | reward=0.7848 | theta=race_soft (g=0.95, a=0.18, b=3.80, t=2.20, k=3, eg=0.08, mix=0.72)
    Client 4 (ds2) | train_acc=0.9401 | val_acc=0.9362 | val_f1=0.6940 | val_auc=0.9917 | reward=0.7546 | theta=race_smoothmix (g=0.90, a=0.20, b=4.00, t=2.10, k=7, eg=0.07, mix=0.70)
    Client 5 (ds2) | train_acc=0.9340 | val_acc=0.9189 | val_f1=0.8792 | val_auc=0.9658 | reward=0.8891 | theta=race_robust (g=1.08, a=0.22, b=4.50, t=2.80, k=5, eg=0.11, mix=0.76)
    
    ----------------------------------------------------------------------------------------------------------------------
    GLOBAL VAL (Round 12) | global_acc=0.9384 | global_f1=0.8556 | ds1_acc=0.9469 | ds1_f1=0.9298 | ds2_acc=0.9286 | ds2_f1=0.7700 | reward=0.9933 | round_time=40.8s
    ----------------------------------------------------------------------------------------------------------------------
    
    ======================================================================================================================
    ROUND 13/15 | selected=[0, 1, 2, 3, 4, 5]
    ======================================================================================================================
    Client 0 (ds1) | train_acc=0.9720 | val_acc=0.8500 | val_f1=0.7061 | val_auc=0.9836 | reward=0.7421 | theta=race_robust (g=1.08, a=0.22, b=4.50, t=2.80, k=5, eg=0.11, mix=0.76)
    Client 1 (ds1) | train_acc=0.9825 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | reward=1.0000 | theta=race_focus (g=1.10, a=0.36, b=6.40, t=2.70, k=3, eg=0.17, mix=0.86)
    Client 2 (ds1) | train_acc=0.9796 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | reward=1.0000 | theta=race_texture (g=0.92, a=0.34, b=5.80, t=2.30, k=7, eg=0.10, mix=0.80)
    Client 3 (ds2) | train_acc=0.9550 | val_acc=1.0000 | val_f1=1.0000 | val_auc=nan | reward=1.0000 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    Client 4 (ds2) | train_acc=0.9737 | val_acc=0.9362 | val_f1=0.7073 | val_auc=0.9892 | reward=0.7645 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    Client 5 (ds2) | train_acc=0.9491 | val_acc=0.9189 | val_f1=0.9084 | val_auc=0.9856 | reward=0.9110 | theta=race_smoothmix (g=0.90, a=0.20, b=4.00, t=2.10, k=7, eg=0.07, mix=0.70)
    
    ----------------------------------------------------------------------------------------------------------------------
    GLOBAL VAL (Round 13) | global_acc=0.9431 | global_f1=0.8630 | ds1_acc=0.9469 | ds1_f1=0.8960 | ds2_acc=0.9388 | ds2_f1=0.8250 | reward=1.0091 | round_time=41.0s
    ----------------------------------------------------------------------------------------------------------------------
    
    ======================================================================================================================
    ROUND 14/15 | selected=[0, 1, 2, 3, 4, 5]
    ======================================================================================================================
    Client 0 (ds1) | train_acc=0.9510 | val_acc=0.9500 | val_f1=0.9095 | val_auc=0.9924 | reward=0.9196 | theta=race_focus (g=1.10, a=0.36, b=6.40, t=2.70, k=3, eg=0.17, mix=0.86)
    Client 1 (ds1) | train_acc=0.9805 | val_acc=0.9722 | val_f1=0.7436 | val_auc=0.9977 | reward=0.8007 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    Client 2 (ds1) | train_acc=0.9833 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | reward=1.0000 | theta=race_sharp (g=1.04, a=0.30, b=5.20, t=2.50, k=5, eg=0.12, mix=0.82)
    Client 3 (ds2) | train_acc=0.9600 | val_acc=0.9286 | val_f1=0.9175 | val_auc=nan | reward=0.9202 | theta=race_smoothmix (g=0.90, a=0.20, b=4.00, t=2.10, k=7, eg=0.07, mix=0.70)
    Client 4 (ds2) | train_acc=0.9795 | val_acc=0.9574 | val_f1=0.7204 | val_auc=0.9982 | reward=0.7796 | theta=race_soft (g=0.95, a=0.18, b=3.80, t=2.20, k=3, eg=0.08, mix=0.72)
    Client 5 (ds2) | train_acc=0.9132 | val_acc=0.8919 | val_f1=0.8507 | val_auc=0.9902 | reward=0.8610 | theta=race_robust (g=1.08, a=0.22, b=4.50, t=2.80, k=5, eg=0.11, mix=0.76)
    
    ----------------------------------------------------------------------------------------------------------------------
    GLOBAL VAL (Round 14) | global_acc=0.9526 | global_f1=0.8451 | ds1_acc=0.9735 | ds1_f1=0.8863 | ds2_acc=0.9286 | ds2_f1=0.7977 | reward=0.9938 | round_time=40.8s
    ----------------------------------------------------------------------------------------------------------------------
    
    ======================================================================================================================
    ROUND 15/15 | selected=[0, 1, 2, 3, 4, 5]
    ======================================================================================================================
    Client 0 (ds1) | train_acc=0.9703 | val_acc=0.8750 | val_f1=0.7737 | val_auc=0.9843 | reward=0.7990 | theta=race_sharp (g=1.04, a=0.30, b=5.20, t=2.50, k=5, eg=0.12, mix=0.82)
    Client 1 (ds1) | train_acc=0.9864 | val_acc=0.9444 | val_f1=0.6979 | val_auc=0.9980 | reward=0.7595 | theta=race_edge_plus (g=1.02, a=0.32, b=6.00, t=2.60, k=3, eg=0.15, mix=0.84)
    Client 2 (ds1) | train_acc=0.9758 | val_acc=0.9459 | val_f1=0.9472 | val_auc=0.9955 | reward=0.9469 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    Client 3 (ds2) | train_acc=0.9600 | val_acc=1.0000 | val_f1=1.0000 | val_auc=nan | reward=1.0000 | theta=race_robust (g=1.08, a=0.22, b=4.50, t=2.80, k=5, eg=0.11, mix=0.76)
    Client 4 (ds2) | train_acc=0.9766 | val_acc=0.9574 | val_f1=0.9628 | val_auc=0.9990 | reward=0.9615 | theta=race_robust (g=1.08, a=0.22, b=4.50, t=2.80, k=5, eg=0.11, mix=0.76)
    Client 5 (ds2) | train_acc=0.9623 | val_acc=0.8919 | val_f1=0.8507 | val_auc=0.9747 | reward=0.8610 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    
    ----------------------------------------------------------------------------------------------------------------------
    GLOBAL VAL (Round 15) | global_acc=0.9289 | global_f1=0.8618 | ds1_acc=0.9204 | ds1_f1=0.8063 | ds2_acc=0.9388 | ds2_f1=0.9258 | reward=1.0072 | round_time=41.1s
    ----------------------------------------------------------------------------------------------------------------------
    
    ======================================================================================================================
    TRAINING COMPLETE ✅ | total_time=626.7s | best_round=8 | best_reward=1.0339
    ======================================================================================================================
    
    ----------------------------------------------------------------------------------------------------------------------
    GLOBAL per-round metrics
    ----------------------------------------------------------------------------------------------------------------------



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
      <td>43.443416</td>
      <td>6</td>
      <td>1.0</td>
      <td>0.784222</td>
      <td>0.786730</td>
      <td>0.647165</td>
      <td>0.688479</td>
      <td>0.662944</td>
      <td>0.561535</td>
      <td>0.559193</td>
      <td>0.776473</td>
      <td>0.752212</td>
      <td>0.656093</td>
      <td>0.615757</td>
      <td>0.826531</td>
      <td>0.636870</td>
      <td>0.499015</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>35.958439</td>
      <td>6</td>
      <td>1.0</td>
      <td>0.784566</td>
      <td>0.819905</td>
      <td>0.638704</td>
      <td>0.676046</td>
      <td>0.668386</td>
      <td>0.468554</td>
      <td>0.458387</td>
      <td>0.506553</td>
      <td>0.787611</td>
      <td>0.612471</td>
      <td>0.498786</td>
      <td>0.857143</td>
      <td>0.668952</td>
      <td>0.433694</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>55.394881</td>
      <td>6</td>
      <td>1.0</td>
      <td>0.888068</td>
      <td>0.890995</td>
      <td>0.739552</td>
      <td>0.781035</td>
      <td>0.754274</td>
      <td>0.422311</td>
      <td>0.421431</td>
      <td>0.498642</td>
      <td>0.902655</td>
      <td>0.767499</td>
      <td>0.477257</td>
      <td>0.877551</td>
      <td>0.707328</td>
      <td>0.358954</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>41.216906</td>
      <td>6</td>
      <td>1.0</td>
      <td>0.917409</td>
      <td>0.905213</td>
      <td>0.771398</td>
      <td>0.782234</td>
      <td>0.797036</td>
      <td>0.318393</td>
      <td>0.309526</td>
      <td>0.511531</td>
      <td>0.920354</td>
      <td>0.810020</td>
      <td>0.277182</td>
      <td>0.887755</td>
      <td>0.726866</td>
      <td>0.365912</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>40.739300</td>
      <td>6</td>
      <td>1.0</td>
      <td>0.942784</td>
      <td>0.905213</td>
      <td>0.791932</td>
      <td>0.796106</td>
      <td>0.797021</td>
      <td>0.275301</td>
      <td>0.274613</td>
      <td>0.496480</td>
      <td>0.911504</td>
      <td>0.792543</td>
      <td>0.229222</td>
      <td>0.897959</td>
      <td>0.791228</td>
      <td>0.328433</td>
    </tr>
    <tr>
      <th>5</th>
      <td>6</td>
      <td>41.095936</td>
      <td>6</td>
      <td>1.0</td>
      <td>0.912924</td>
      <td>0.895735</td>
      <td>0.760037</td>
      <td>0.773884</td>
      <td>0.769818</td>
      <td>0.293205</td>
      <td>0.281193</td>
      <td>0.501978</td>
      <td>0.893805</td>
      <td>0.758294</td>
      <td>0.305501</td>
      <td>0.897959</td>
      <td>0.762047</td>
      <td>0.279028</td>
    </tr>
    <tr>
      <th>6</th>
      <td>7</td>
      <td>40.817750</td>
      <td>6</td>
      <td>1.0</td>
      <td>0.996911</td>
      <td>0.919431</td>
      <td>0.852607</td>
      <td>0.879884</td>
      <td>0.867417</td>
      <td>0.282423</td>
      <td>0.280813</td>
      <td>0.504464</td>
      <td>0.920354</td>
      <td>0.801503</td>
      <td>0.303424</td>
      <td>0.918367</td>
      <td>0.911533</td>
      <td>0.258208</td>
    </tr>
    <tr>
      <th>7</th>
      <td>8</td>
      <td>41.043477</td>
      <td>6</td>
      <td>1.0</td>
      <td>1.033903</td>
      <td>0.909953</td>
      <td>0.896199</td>
      <td>0.900539</td>
      <td>0.905470</td>
      <td>0.222821</td>
      <td>0.192111</td>
      <td>0.502530</td>
      <td>0.902655</td>
      <td>0.886285</td>
      <td>0.246705</td>
      <td>0.918367</td>
      <td>0.907630</td>
      <td>0.195282</td>
    </tr>
    <tr>
      <th>8</th>
      <td>9</td>
      <td>41.177123</td>
      <td>6</td>
      <td>1.0</td>
      <td>0.992543</td>
      <td>0.914692</td>
      <td>0.846478</td>
      <td>0.854994</td>
      <td>0.854925</td>
      <td>0.253433</td>
      <td>0.269209</td>
      <td>0.506248</td>
      <td>0.929204</td>
      <td>0.832240</td>
      <td>0.207883</td>
      <td>0.897959</td>
      <td>0.862896</td>
      <td>0.305954</td>
    </tr>
    <tr>
      <th>9</th>
      <td>10</td>
      <td>40.871363</td>
      <td>6</td>
      <td>1.0</td>
      <td>0.997379</td>
      <td>0.924171</td>
      <td>0.856499</td>
      <td>0.855399</td>
      <td>0.869009</td>
      <td>0.227412</td>
      <td>0.227353</td>
      <td>0.505901</td>
      <td>0.920354</td>
      <td>0.895452</td>
      <td>0.225381</td>
      <td>0.928571</td>
      <td>0.811584</td>
      <td>0.229755</td>
    </tr>
    <tr>
      <th>10</th>
      <td>11</td>
      <td>40.820076</td>
      <td>6</td>
      <td>1.0</td>
      <td>1.005371</td>
      <td>0.924171</td>
      <td>0.857858</td>
      <td>0.892293</td>
      <td>0.855743</td>
      <td>0.193010</td>
      <td>0.191806</td>
      <td>0.500599</td>
      <td>0.946903</td>
      <td>0.846103</td>
      <td>0.139573</td>
      <td>0.897959</td>
      <td>0.871412</td>
      <td>0.254625</td>
    </tr>
    <tr>
      <th>11</th>
      <td>12</td>
      <td>40.844383</td>
      <td>6</td>
      <td>1.0</td>
      <td>0.993333</td>
      <td>0.938389</td>
      <td>0.855610</td>
      <td>0.876200</td>
      <td>0.856403</td>
      <td>0.218608</td>
      <td>0.199195</td>
      <td>0.496209</td>
      <td>0.946903</td>
      <td>0.929823</td>
      <td>0.154430</td>
      <td>0.928571</td>
      <td>0.770038</td>
      <td>0.292609</td>
    </tr>
    <tr>
      <th>12</th>
      <td>13</td>
      <td>40.950130</td>
      <td>6</td>
      <td>1.0</td>
      <td>1.009095</td>
      <td>0.943128</td>
      <td>0.863012</td>
      <td>0.895969</td>
      <td>0.862641</td>
      <td>0.244259</td>
      <td>0.241824</td>
      <td>0.502016</td>
      <td>0.946903</td>
      <td>0.895973</td>
      <td>0.200987</td>
      <td>0.938776</td>
      <td>0.825007</td>
      <td>0.294154</td>
    </tr>
    <tr>
      <th>13</th>
      <td>14</td>
      <td>40.844285</td>
      <td>6</td>
      <td>1.0</td>
      <td>0.993822</td>
      <td>0.952607</td>
      <td>0.845148</td>
      <td>0.852163</td>
      <td>0.843832</td>
      <td>0.167637</td>
      <td>0.160653</td>
      <td>0.498205</td>
      <td>0.973451</td>
      <td>0.886260</td>
      <td>0.126867</td>
      <td>0.928571</td>
      <td>0.797744</td>
      <td>0.214646</td>
    </tr>
    <tr>
      <th>14</th>
      <td>15</td>
      <td>41.101463</td>
      <td>6</td>
      <td>1.0</td>
      <td>1.007164</td>
      <td>0.928910</td>
      <td>0.861818</td>
      <td>0.890569</td>
      <td>0.869220</td>
      <td>0.274940</td>
      <td>0.259734</td>
      <td>0.494808</td>
      <td>0.920354</td>
      <td>0.806323</td>
      <td>0.301059</td>
      <td>0.938776</td>
      <td>0.925807</td>
      <td>0.244823</td>
    </tr>
  </tbody>
</table>
</div>


    
    ----------------------------------------------------------------------------------------------------------------------
    LOCAL per-client per-round metrics
    ----------------------------------------------------------------------------------------------------------------------



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
      <td>5</td>
      <td>race_focus</td>
      <td>(g=1.10, a=0.36, b=6.40, t=2.70, k=3, eg=0.17,...</td>
      <td>1.10</td>
      <td>0.36</td>
      <td>6.4</td>
      <td>...</td>
      <td>0.915480</td>
      <td>0.784314</td>
      <td>0.951389</td>
      <td>0.956522</td>
      <td>0.969697</td>
      <td>0.383171</td>
      <td>0.461368</td>
      <td>0.155461</td>
      <td>1.313499</td>
      <td>0.713413</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>client_1</td>
      <td>ds1</td>
      <td>1</td>
      <td>5</td>
      <td>race_focus</td>
      <td>(g=1.10, a=0.36, b=6.40, t=2.70, k=3, eg=0.17,...</td>
      <td>1.10</td>
      <td>0.36</td>
      <td>6.4</td>
      <td>...</td>
      <td>0.989345</td>
      <td>0.972136</td>
      <td>1.000000</td>
      <td>0.992188</td>
      <td>0.993056</td>
      <td>0.358040</td>
      <td>0.264466</td>
      <td>0.377494</td>
      <td>1.562467</td>
      <td>0.761619</td>
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
      <td>0.844733</td>
      <td>0.439394</td>
      <td>0.969697</td>
      <td>0.985714</td>
      <td>0.984127</td>
      <td>0.099812</td>
      <td>0.284976</td>
      <td>0.615213</td>
      <td>1.162766</td>
      <td>0.564839</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>client_3</td>
      <td>ds2</td>
      <td>1</td>
      <td>2</td>
      <td>race_robust</td>
      <td>(g=1.08, a=0.22, b=4.50, t=2.80, k=5, eg=0.11,...</td>
      <td>1.08</td>
      <td>0.22</td>
      <td>4.5</td>
      <td>...</td>
      <td>NaN</td>
      <td>0.950000</td>
      <td>0.909091</td>
      <td>NaN</td>
      <td>1.000000</td>
      <td>0.377787</td>
      <td>0.386142</td>
      <td>0.236071</td>
      <td>1.534665</td>
      <td>0.585317</td>
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
      <td>0.932536</td>
      <td>0.760870</td>
      <td>0.978431</td>
      <td>0.990842</td>
      <td>1.000000</td>
      <td>0.460398</td>
      <td>0.278822</td>
      <td>0.260781</td>
      <td>1.469759</td>
      <td>0.733662</td>
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
      <td>4</td>
      <td>race_edge_plus</td>
      <td>(g=1.02, a=0.32, b=6.00, t=2.60, k=3, eg=0.15,...</td>
      <td>1.02</td>
      <td>0.32</td>
      <td>6.0</td>
      <td>...</td>
      <td>0.998047</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.992188</td>
      <td>1.000000</td>
      <td>0.600612</td>
      <td>0.239215</td>
      <td>0.160173</td>
      <td>1.155721</td>
      <td>0.759518</td>
    </tr>
    <tr>
      <th>86</th>
      <td>15</td>
      <td>client_2</td>
      <td>ds1</td>
      <td>1</td>
      <td>0</td>
      <td>race_balanced</td>
      <td>(g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10,...</td>
      <td>1.00</td>
      <td>0.24</td>
      <td>4.6</td>
      <td>...</td>
      <td>0.995455</td>
      <td>0.984848</td>
      <td>0.996970</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.465011</td>
      <td>0.331409</td>
      <td>0.203580</td>
      <td>1.463308</td>
      <td>0.946856</td>
    </tr>
    <tr>
      <th>87</th>
      <td>15</td>
      <td>client_3</td>
      <td>ds2</td>
      <td>1</td>
      <td>2</td>
      <td>race_robust</td>
      <td>(g=1.08, a=0.22, b=4.50, t=2.80, k=5, eg=0.11,...</td>
      <td>1.08</td>
      <td>0.22</td>
      <td>4.5</td>
      <td>...</td>
      <td>NaN</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>NaN</td>
      <td>1.000000</td>
      <td>0.550353</td>
      <td>0.303016</td>
      <td>0.146631</td>
      <td>1.301831</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>88</th>
      <td>15</td>
      <td>client_4</td>
      <td>ds2</td>
      <td>1</td>
      <td>2</td>
      <td>race_robust</td>
      <td>(g=1.08, a=0.22, b=4.50, t=2.80, k=5, eg=0.11,...</td>
      <td>1.08</td>
      <td>0.22</td>
      <td>4.5</td>
      <td>...</td>
      <td>0.999020</td>
      <td>1.000000</td>
      <td>0.996078</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.440638</td>
      <td>0.473930</td>
      <td>0.085432</td>
      <td>1.268208</td>
      <td>0.961472</td>
    </tr>
    <tr>
      <th>89</th>
      <td>15</td>
      <td>client_5</td>
      <td>ds2</td>
      <td>1</td>
      <td>1</td>
      <td>race_balanced</td>
      <td>(g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10,...</td>
      <td>1.00</td>
      <td>0.24</td>
      <td>4.6</td>
      <td>...</td>
      <td>0.974680</td>
      <td>0.933333</td>
      <td>0.969697</td>
      <td>0.995690</td>
      <td>1.000000</td>
      <td>0.598651</td>
      <td>0.259245</td>
      <td>0.142104</td>
      <td>1.244755</td>
      <td>0.861016</td>
    </tr>
  </tbody>
</table>
<p>90 rows × 39 columns</p>
</div>


    
    ======================================================================================================================
    STEP 12: FINAL EVALUATION
    ======================================================================================================================
    
    ----------------------------------------------------------------------------------------------------------------------
    Best RL-selected preprocessing preset per client
    ----------------------------------------------------------------------------------------------------------------------



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
      <td>5</td>
      <td>race_focus</td>
      <td>(g=1.10, a=0.36, b=6.40, t=2.70, k=3, eg=0.17,...</td>
      <td>0.773427</td>
      <td>8</td>
    </tr>
    <tr>
      <th>1</th>
      <td>client_1</td>
      <td>ds1</td>
      <td>0</td>
      <td>race_balanced</td>
      <td>(g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10,...</td>
      <td>0.851664</td>
      <td>8</td>
    </tr>
    <tr>
      <th>2</th>
      <td>client_2</td>
      <td>ds1</td>
      <td>2</td>
      <td>race_texture</td>
      <td>(g=0.92, a=0.34, b=5.80, t=2.30, k=7, eg=0.10,...</td>
      <td>1.000000</td>
      <td>8</td>
    </tr>
    <tr>
      <th>3</th>
      <td>client_3</td>
      <td>ds2</td>
      <td>1</td>
      <td>race_balanced</td>
      <td>(g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10,...</td>
      <td>0.908279</td>
      <td>8</td>
    </tr>
    <tr>
      <th>4</th>
      <td>client_4</td>
      <td>ds2</td>
      <td>1</td>
      <td>race_balanced</td>
      <td>(g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10,...</td>
      <td>0.856776</td>
      <td>8</td>
    </tr>
    <tr>
      <th>5</th>
      <td>client_5</td>
      <td>ds2</td>
      <td>0</td>
      <td>race_soft</td>
      <td>(g=0.95, a=0.18, b=3.80, t=2.20, k=3, eg=0.08,...</td>
      <td>0.791109</td>
      <td>8</td>
    </tr>
  </tbody>
</table>
</div>


    
    ----------------------------------------------------------------------------------------------------------------------
    Chosen final validation-based preprocessing strategy
    ----------------------------------------------------------------------------------------------------------------------



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
      <td>['race_texture']</td>
      <td>0.968800</td>
      <td>0.969027</td>
      <td>0.968725</td>
    </tr>
    <tr>
      <th>1</th>
      <td>ds2</td>
      <td>single</td>
      <td>['race_balanced']</td>
      <td>0.958039</td>
      <td>0.959391</td>
      <td>0.957588</td>
    </tr>
  </tbody>
</table>
</div>


    
    ======================================================================================================================
    STEP 12.5: EXTENDED METRICS + ERROR ANALYSIS + CALIBRATION
    ======================================================================================================================
    
    ----------------------------------------------------------------------------------------------------------------------
    Extended TEST metrics (DS1 vs DS2)
    ----------------------------------------------------------------------------------------------------------------------



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
      <td>0.969027</td>
      <td>0.969204</td>
      <td>0.970402</td>
      <td>0.969204</td>
      <td>0.968750</td>
      <td>0.970893</td>
      <td>0.969027</td>
      <td>0.968915</td>
      <td>0.119941</td>
      <td>...</td>
      <td>0.010268</td>
      <td>0.030796</td>
      <td>0.026375</td>
      <td>0.576835</td>
      <td>0.054097</td>
      <td>0.999056</td>
      <td>0.997374</td>
      <td>0.999787</td>
      <td>0.999797</td>
      <td>0.999265</td>
    </tr>
    <tr>
      <th>1</th>
      <td>ds2_test</td>
      <td>0.934010</td>
      <td>0.930563</td>
      <td>0.936121</td>
      <td>0.930563</td>
      <td>0.928136</td>
      <td>0.941492</td>
      <td>0.934010</td>
      <td>0.932923</td>
      <td>0.189087</td>
      <td>...</td>
      <td>0.021392</td>
      <td>0.069437</td>
      <td>0.028713</td>
      <td>0.365202</td>
      <td>0.103181</td>
      <td>0.993345</td>
      <td>0.998977</td>
      <td>0.985315</td>
      <td>0.999759</td>
      <td>0.989327</td>
    </tr>
  </tbody>
</table>
<p>2 rows × 26 columns</p>
</div>


    
    ----------------------------------------------------------------------------------------------------------------------
    Classwise metrics — DS1 TEST
    ----------------------------------------------------------------------------------------------------------------------



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
      <td>51</td>
      <td>0</td>
      <td>5</td>
      <td>170</td>
      <td>0.247788</td>
      <td>1.000000</td>
      <td>0.971429</td>
      <td>0.910714</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.089286</td>
      <td>0.910714</td>
      <td>0.955357</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>meningioma</td>
      <td>55</td>
      <td>55</td>
      <td>3</td>
      <td>0</td>
      <td>168</td>
      <td>0.243363</td>
      <td>0.948276</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.982456</td>
      <td>0.017544</td>
      <td>0.000000</td>
      <td>0.948276</td>
      <td>0.991228</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>notumor</td>
      <td>59</td>
      <td>57</td>
      <td>0</td>
      <td>2</td>
      <td>167</td>
      <td>0.261062</td>
      <td>1.000000</td>
      <td>0.988166</td>
      <td>0.966102</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.033898</td>
      <td>0.966102</td>
      <td>0.983051</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>pituitary</td>
      <td>56</td>
      <td>56</td>
      <td>4</td>
      <td>0</td>
      <td>166</td>
      <td>0.247788</td>
      <td>0.933333</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.976471</td>
      <td>0.023529</td>
      <td>0.000000</td>
      <td>0.933333</td>
      <td>0.988235</td>
    </tr>
  </tbody>
</table>
</div>


    
    ----------------------------------------------------------------------------------------------------------------------
    Classwise metrics — DS2 TEST
    ----------------------------------------------------------------------------------------------------------------------



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
      <td>45</td>
      <td>45</td>
      <td>2</td>
      <td>0</td>
      <td>150</td>
      <td>0.228426</td>
      <td>0.957447</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.986842</td>
      <td>0.013158</td>
      <td>0.000000</td>
      <td>0.957447</td>
      <td>0.993421</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>meningioma</td>
      <td>46</td>
      <td>35</td>
      <td>1</td>
      <td>11</td>
      <td>150</td>
      <td>0.233503</td>
      <td>0.972222</td>
      <td>0.931677</td>
      <td>0.760870</td>
      <td>0.993377</td>
      <td>0.006623</td>
      <td>0.239130</td>
      <td>0.744681</td>
      <td>0.877124</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>notumor</td>
      <td>61</td>
      <td>60</td>
      <td>0</td>
      <td>1</td>
      <td>136</td>
      <td>0.309645</td>
      <td>1.000000</td>
      <td>0.992701</td>
      <td>0.983607</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.016393</td>
      <td>0.983607</td>
      <td>0.991803</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>pituitary</td>
      <td>45</td>
      <td>44</td>
      <td>10</td>
      <td>1</td>
      <td>142</td>
      <td>0.228426</td>
      <td>0.814815</td>
      <td>0.993007</td>
      <td>0.977778</td>
      <td>0.934211</td>
      <td>0.065789</td>
      <td>0.022222</td>
      <td>0.800000</td>
      <td>0.955994</td>
    </tr>
  </tbody>
</table>
</div>


    
    ----------------------------------------------------------------------------------------------------------------------
    Top confusion pairs — DS1 TEST
    ----------------------------------------------------------------------------------------------------------------------



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
      <td>3</td>
    </tr>
    <tr>
      <th>1</th>
      <td>glioma</td>
      <td>pituitary</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>notumor</td>
      <td>pituitary</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>glioma</td>
      <td>notumor</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>meningioma</td>
      <td>glioma</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>meningioma</td>
      <td>notumor</td>
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
      <td>meningioma</td>
      <td>pituitary</td>
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
      <td>pituitary</td>
      <td>glioma</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>


    
    ----------------------------------------------------------------------------------------------------------------------
    Top confusion pairs — DS2 TEST
    ----------------------------------------------------------------------------------------------------------------------



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
      <td>meningioma</td>
      <td>pituitary</td>
      <td>9</td>
    </tr>
    <tr>
      <th>1</th>
      <td>meningioma</td>
      <td>glioma</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>notumor</td>
      <td>pituitary</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>pituitary</td>
      <td>meningioma</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>glioma</td>
      <td>pituitary</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>glioma</td>
      <td>notumor</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>glioma</td>
      <td>meningioma</td>
      <td>0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>meningioma</td>
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


    
    ----------------------------------------------------------------------------------------------------------------------
    Calibration bins — DS1 TEST
    ----------------------------------------------------------------------------------------------------------------------



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
      <td>0.406937</td>
      <td>0.000000</td>
      <td>0.406937</td>
      <td>1</td>
    </tr>
    <tr>
      <th>5</th>
      <td>5</td>
      <td>0.416667</td>
      <td>0.500000</td>
      <td>0.423165</td>
      <td>1.000000</td>
      <td>0.576835</td>
      <td>1</td>
    </tr>
    <tr>
      <th>6</th>
      <td>6</td>
      <td>0.500000</td>
      <td>0.583333</td>
      <td>0.525676</td>
      <td>0.666667</td>
      <td>0.140991</td>
      <td>3</td>
    </tr>
    <tr>
      <th>7</th>
      <td>7</td>
      <td>0.583333</td>
      <td>0.666667</td>
      <td>0.634891</td>
      <td>0.750000</td>
      <td>0.115109</td>
      <td>4</td>
    </tr>
    <tr>
      <th>8</th>
      <td>8</td>
      <td>0.666667</td>
      <td>0.750000</td>
      <td>0.709615</td>
      <td>0.714286</td>
      <td>0.004671</td>
      <td>7</td>
    </tr>
    <tr>
      <th>9</th>
      <td>9</td>
      <td>0.750000</td>
      <td>0.833333</td>
      <td>0.804467</td>
      <td>1.000000</td>
      <td>0.195533</td>
      <td>2</td>
    </tr>
    <tr>
      <th>10</th>
      <td>10</td>
      <td>0.833333</td>
      <td>0.916667</td>
      <td>0.884597</td>
      <td>0.833333</td>
      <td>0.051263</td>
      <td>6</td>
    </tr>
    <tr>
      <th>11</th>
      <td>11</td>
      <td>0.916667</td>
      <td>1.000000</td>
      <td>0.978405</td>
      <td>0.995049</td>
      <td>0.016644</td>
      <td>202</td>
    </tr>
  </tbody>
</table>
</div>


    
    ----------------------------------------------------------------------------------------------------------------------
    Calibration bins — DS2 TEST
    ----------------------------------------------------------------------------------------------------------------------



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
      <td>0.539432</td>
      <td>0.714286</td>
      <td>0.174853</td>
      <td>7</td>
    </tr>
    <tr>
      <th>7</th>
      <td>7</td>
      <td>0.583333</td>
      <td>0.666667</td>
      <td>0.634856</td>
      <td>0.500000</td>
      <td>0.134856</td>
      <td>4</td>
    </tr>
    <tr>
      <th>8</th>
      <td>8</td>
      <td>0.666667</td>
      <td>0.750000</td>
      <td>0.712148</td>
      <td>0.666667</td>
      <td>0.045482</td>
      <td>3</td>
    </tr>
    <tr>
      <th>9</th>
      <td>9</td>
      <td>0.750000</td>
      <td>0.833333</td>
      <td>0.803213</td>
      <td>0.750000</td>
      <td>0.053213</td>
      <td>4</td>
    </tr>
    <tr>
      <th>10</th>
      <td>10</td>
      <td>0.833333</td>
      <td>0.916667</td>
      <td>0.865202</td>
      <td>0.500000</td>
      <td>0.365202</td>
      <td>8</td>
    </tr>
    <tr>
      <th>11</th>
      <td>11</td>
      <td>0.916667</td>
      <td>1.000000</td>
      <td>0.978818</td>
      <td>0.982456</td>
      <td>0.003639</td>
      <td>171</td>
    </tr>
  </tbody>
</table>
</div>


    
    ----------------------------------------------------------------------------------------------------------------------
    VAL + TEST tables (federated, per-dataset + global equal)
    ----------------------------------------------------------------------------------------------------------------------



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
      <td>ARCF-Net + FedAvg only</td>
      <td>VAL</td>
      <td>ds1</td>
      <td>0.969027</td>
      <td>0.969828</td>
      <td>0.969180</td>
      <td>0.968725</td>
      <td>0.970857</td>
      <td>0.969027</td>
      <td>0.969176</td>
      <td>...</td>
      <td>4.694563</td>
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
      <td>ARCF-Net + FedAvg only</td>
      <td>VAL</td>
      <td>ds2</td>
      <td>0.959391</td>
      <td>0.957284</td>
      <td>0.958832</td>
      <td>0.957588</td>
      <td>0.960539</td>
      <td>0.959391</td>
      <td>0.959512</td>
      <td>...</td>
      <td>4.060049</td>
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
      <td>ARCF-Net + FedAvg only</td>
      <td>VAL</td>
      <td>global_equal</td>
      <td>0.964209</td>
      <td>0.963556</td>
      <td>0.964006</td>
      <td>0.963156</td>
      <td>0.965698</td>
      <td>0.964209</td>
      <td>0.964344</td>
      <td>...</td>
      <td>4.377306</td>
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
      <td>ARCF-Net + FedAvg only</td>
      <td>TEST</td>
      <td>ds1</td>
      <td>0.969027</td>
      <td>0.970402</td>
      <td>0.969204</td>
      <td>0.968750</td>
      <td>0.970893</td>
      <td>0.969027</td>
      <td>0.968915</td>
      <td>...</td>
      <td>4.815831</td>
      <td>0.969204</td>
      <td>0.959378</td>
      <td>0.958702</td>
      <td>0.970402</td>
      <td>0.989899</td>
      <td>0.989732</td>
      <td>0.026375</td>
      <td>0.576835</td>
      <td>0.054097</td>
    </tr>
    <tr>
      <th>4</th>
      <td>ARCF-Net + FedAvg only</td>
      <td>TEST</td>
      <td>ds2</td>
      <td>0.934010</td>
      <td>0.936121</td>
      <td>0.930563</td>
      <td>0.928136</td>
      <td>0.941492</td>
      <td>0.934010</td>
      <td>0.932923</td>
      <td>...</td>
      <td>4.212896</td>
      <td>0.930563</td>
      <td>0.914471</td>
      <td>0.911531</td>
      <td>0.936121</td>
      <td>0.979346</td>
      <td>0.978608</td>
      <td>0.028713</td>
      <td>0.365202</td>
      <td>0.103181</td>
    </tr>
    <tr>
      <th>5</th>
      <td>ARCF-Net + FedAvg only</td>
      <td>TEST</td>
      <td>global_equal</td>
      <td>0.951518</td>
      <td>0.953262</td>
      <td>0.949884</td>
      <td>0.948443</td>
      <td>0.956193</td>
      <td>0.951518</td>
      <td>0.950919</td>
      <td>...</td>
      <td>4.514363</td>
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


    
    Paper selection summary:
    - Best round: 8 | best_reward=1.0339
    - DS1 final strategy: single | names=['race_texture']
    - DS2 final strategy: single | names=['race_balanced']
    
    ======================================================================================================================
    STEP 13: PREPROCESSING VALIDATION (DS1 + DS2 VAL SAMPLE)
    ======================================================================================================================
    
    ----------------------------------------------------------------------------------------------------------------------
    Preprocessing validation summary (DS1 VAL sample)
    ----------------------------------------------------------------------------------------------------------------------



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
      <td>0.041596</td>
      <td>0.021720</td>
      <td>0.014947</td>
      <td>0.165471</td>
    </tr>
    <tr>
      <th>1</th>
      <td>edge_energy_after</td>
      <td>0.122313</td>
      <td>0.030703</td>
      <td>0.069450</td>
      <td>0.262682</td>
    </tr>
    <tr>
      <th>2</th>
      <td>entropy_before</td>
      <td>5.820408</td>
      <td>0.628255</td>
      <td>3.547314</td>
      <td>7.239670</td>
    </tr>
    <tr>
      <th>3</th>
      <td>entropy_after</td>
      <td>6.726817</td>
      <td>0.568965</td>
      <td>4.021207</td>
      <td>7.666635</td>
    </tr>
    <tr>
      <th>4</th>
      <td>contrast_before</td>
      <td>0.187640</td>
      <td>0.052597</td>
      <td>0.101468</td>
      <td>0.363759</td>
    </tr>
    <tr>
      <th>5</th>
      <td>contrast_after</td>
      <td>0.258639</td>
      <td>0.019839</td>
      <td>0.224627</td>
      <td>0.346018</td>
    </tr>
    <tr>
      <th>6</th>
      <td>edge_gain_ratio</td>
      <td>3.211548</td>
      <td>0.786428</td>
      <td>1.320958</td>
      <td>6.118102</td>
    </tr>
    <tr>
      <th>7</th>
      <td>entropy_delta</td>
      <td>0.906409</td>
      <td>0.262735</td>
      <td>0.362860</td>
      <td>1.703715</td>
    </tr>
    <tr>
      <th>8</th>
      <td>contrast_delta</td>
      <td>0.070999</td>
      <td>0.035756</td>
      <td>-0.018766</td>
      <td>0.137007</td>
    </tr>
  </tbody>
</table>
</div>


    
    ----------------------------------------------------------------------------------------------------------------------
    Preprocessing validation summary (DS2 VAL sample)
    ----------------------------------------------------------------------------------------------------------------------



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
      <td>0.041718</td>
      <td>0.015947</td>
      <td>0.012571</td>
      <td>0.121283</td>
    </tr>
    <tr>
      <th>1</th>
      <td>edge_energy_after</td>
      <td>0.116413</td>
      <td>0.023715</td>
      <td>0.043626</td>
      <td>0.167539</td>
    </tr>
    <tr>
      <th>2</th>
      <td>entropy_before</td>
      <td>5.811255</td>
      <td>0.822915</td>
      <td>3.222212</td>
      <td>7.570240</td>
    </tr>
    <tr>
      <th>3</th>
      <td>entropy_after</td>
      <td>6.663698</td>
      <td>0.784854</td>
      <td>3.615744</td>
      <td>7.739758</td>
    </tr>
    <tr>
      <th>4</th>
      <td>contrast_before</td>
      <td>0.190753</td>
      <td>0.055869</td>
      <td>0.098556</td>
      <td>0.341468</td>
    </tr>
    <tr>
      <th>5</th>
      <td>contrast_after</td>
      <td>0.250023</td>
      <td>0.022308</td>
      <td>0.198615</td>
      <td>0.320288</td>
    </tr>
    <tr>
      <th>6</th>
      <td>edge_gain_ratio</td>
      <td>3.015589</td>
      <td>0.781781</td>
      <td>1.308301</td>
      <td>5.730748</td>
    </tr>
    <tr>
      <th>7</th>
      <td>entropy_delta</td>
      <td>0.852443</td>
      <td>0.296184</td>
      <td>0.169518</td>
      <td>1.613087</td>
    </tr>
    <tr>
      <th>8</th>
      <td>contrast_delta</td>
      <td>0.059269</td>
      <td>0.037111</td>
      <td>-0.022614</td>
      <td>0.130598</td>
    </tr>
  </tbody>
</table>
</div>


    
    ======================================================================================================================
    STEP 14: PARAMETER EVOLUTION TABLE
    ======================================================================================================================
    
    ----------------------------------------------------------------------------------------------------------------------
    Mean selected preprocessing parameters over rounds
    ----------------------------------------------------------------------------------------------------------------------



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
      <td>1.060000</td>
      <td>0.286667</td>
      <td>5.366667</td>
      <td>2.633333</td>
      <td>4.333333</td>
      <td>0.133333</td>
      <td>0.800000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1.006667</td>
      <td>0.256667</td>
      <td>5.016667</td>
      <td>2.500000</td>
      <td>3.333333</td>
      <td>0.120000</td>
      <td>0.786667</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1.005000</td>
      <td>0.240000</td>
      <td>4.633333</td>
      <td>2.483333</td>
      <td>5.000000</td>
      <td>0.100000</td>
      <td>0.766667</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>0.968333</td>
      <td>0.250000</td>
      <td>4.666667</td>
      <td>2.316667</td>
      <td>5.333333</td>
      <td>0.095000</td>
      <td>0.766667</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0.966667</td>
      <td>0.263333</td>
      <td>4.800000</td>
      <td>2.316667</td>
      <td>6.000000</td>
      <td>0.096667</td>
      <td>0.770000</td>
    </tr>
    <tr>
      <th>5</th>
      <td>6</td>
      <td>0.996667</td>
      <td>0.230000</td>
      <td>4.483333</td>
      <td>2.416667</td>
      <td>5.333333</td>
      <td>0.096667</td>
      <td>0.763333</td>
    </tr>
    <tr>
      <th>6</th>
      <td>7</td>
      <td>0.946667</td>
      <td>0.256667</td>
      <td>4.733333</td>
      <td>2.266667</td>
      <td>5.333333</td>
      <td>0.091667</td>
      <td>0.760000</td>
    </tr>
    <tr>
      <th>7</th>
      <td>8</td>
      <td>1.063333</td>
      <td>0.260000</td>
      <td>4.950000</td>
      <td>2.666667</td>
      <td>4.666667</td>
      <td>0.120000</td>
      <td>0.790000</td>
    </tr>
    <tr>
      <th>8</th>
      <td>9</td>
      <td>0.991667</td>
      <td>0.243333</td>
      <td>4.650000</td>
      <td>2.416667</td>
      <td>5.000000</td>
      <td>0.098333</td>
      <td>0.770000</td>
    </tr>
    <tr>
      <th>9</th>
      <td>10</td>
      <td>0.988333</td>
      <td>0.233333</td>
      <td>4.550000</td>
      <td>2.383333</td>
      <td>5.000000</td>
      <td>0.100000</td>
      <td>0.753333</td>
    </tr>
    <tr>
      <th>10</th>
      <td>11</td>
      <td>1.050000</td>
      <td>0.253333</td>
      <td>4.883333</td>
      <td>2.650000</td>
      <td>4.666667</td>
      <td>0.116667</td>
      <td>0.786667</td>
    </tr>
    <tr>
      <th>11</th>
      <td>12</td>
      <td>1.011667</td>
      <td>0.266667</td>
      <td>5.116667</td>
      <td>2.500000</td>
      <td>4.000000</td>
      <td>0.121667</td>
      <td>0.786667</td>
    </tr>
    <tr>
      <th>12</th>
      <td>13</td>
      <td>1.000000</td>
      <td>0.266667</td>
      <td>4.983333</td>
      <td>2.450000</td>
      <td>5.333333</td>
      <td>0.108333</td>
      <td>0.780000</td>
    </tr>
    <tr>
      <th>13</th>
      <td>14</td>
      <td>1.011667</td>
      <td>0.250000</td>
      <td>4.750000</td>
      <td>2.450000</td>
      <td>4.666667</td>
      <td>0.108333</td>
      <td>0.773333</td>
    </tr>
    <tr>
      <th>14</th>
      <td>15</td>
      <td>1.036667</td>
      <td>0.256667</td>
      <td>4.900000</td>
      <td>2.583333</td>
      <td>4.666667</td>
      <td>0.115000</td>
      <td>0.790000</td>
    </tr>
  </tbody>
</table>
</div>


    
    ======================================================================================================================
    STEP 15: SAVING ONLY TWO FILES (CHECKPOINT + ONE CSV)
    ======================================================================================================================
    ✅ Saved checkpoint: /kaggle/working/outputs/ARCFNet_RESNET50_FEDAVG_ONLY_checkpoint.pth
    ✅ Saved CSV (ALL outputs): /kaggle/working/outputs/ALL_OUTPUTS_AND_METRICS_RESNET50_FEDAVG_ONLY.csv
    
    DONE ✅
    Method: ARCF-Net = Adaptive RACE-FELCM with CRAF Fusion Network
    Backbone: Residual Network-50
    Federated variant: FedAvg only
    Best round: 8
    Adaptive clients => DS1=3, DS2=3, TOTAL=6
    Rounds completed: 15
    Global TEST acc: 0.9515
    Global TEST f1_macro: 0.9484
    DS1 TEST acc: 0.9690
    DS2 TEST acc: 0.9340
    DS1 final strategy: single | names=['race_texture']
    DS2 final strategy: single | names=['race_balanced']


**2. FedAvg + FedProx**


```python
# ============================================================
# KAGGLE FULL SCRIPT
# TRUE FL + RL-UCB + RACE-FELCM + CRAF + ResNet-50
# ABLATION: FEDAVG + FEDPROX
# METHOD ACRONYM: ARCF-Net
# FULL FORM: Adaptive RACE-FELCM with CRAF Fusion Network
# ------------------------------------------------------------
# KAGGLE-READY + SYMMETRIC DS1/DS2 IMPORTANCE
# - Uses BOTH datasets
# - Reads datasets from /kaggle/input automatically
# - Exact 15 FL rounds
# - Proper FL with FedAvg + FedProx
# - NO prototype sharing
# - RL-UCB for SHARED client-count planning + per-client preprocessing preset selection
# - Tune-aware theta probing before local training
# - Equal DS1 / DS2 importance in best-round selection and merged reporting
# - No plots
# - Saves checkpoint WITH full-process info for later replay / XAI
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

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    log_loss, confusion_matrix, roc_auc_score,
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

print("=" * 118)
print("TRUE FL + RL-UCB + RACE-FELCM + CRAF + ResNet-50")
print("METHOD: ARCF-Net (Adaptive RACE-FELCM with CRAF Fusion Network)")
print("ABLATION: FedAvg + FedProx")
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

    # FedAvg tempering (same as canonical full script)
    "fedavg_temper": 0.50,

    # misc / tables
    "quick_hash_subset_per_split": 300,
    "preproc_val_sample_n": 400,
    "calibration_bins": 12,
}

OUTDIR = "/kaggle/working/outputs" if IS_KAGGLE else "/content/outputs"
os.makedirs(OUTDIR, exist_ok=True)
MODEL_PATH = os.path.join(OUTDIR, "ARCFNet_RESNET50_FEDAVG_FEDPROX_checkpoint.pth")
CSV_PATH   = os.path.join(OUTDIR, "ALL_OUTPUTS_AND_METRICS_RESNET50_FEDAVG_FEDPROX.csv")

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
    "federated_variant": "FedAvg + FedProx",
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
print("STEP 6: DATA LOADERS")
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

        counts = df_src.loc[tr_idx, "label"].value_counts().reindex(labels, fill_value=0)

        clients.append({
            "gid": gid,
            "local_id": local_id,
            "dataset": ds_name,
            "source_id": source_id,
            "train_loader": train_loader,
            "tune_loader": tune_loader,
            "val_loader": val_loader,
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
# STEP 9: LOSSES + FEDPROX
# ============================================================
print("\n" + "=" * 118)
print("STEP 9: LOSSES + FEDPROX")
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
    loss = torch.tensor(0.0, device=DEVICE)
    for p_local, p_global in zip(local_model.parameters(), global_model.parameters()):
        loss = loss + ((p_local - p_global.detach()) ** 2).sum()
    return loss

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

def train_one_epoch(model, loader, optimizer, preproc, theta, global_model=None, scheduler=None):
    model.train()
    freeze_backbone_bn_stats(model)
    preproc.eval()

    losses, ce_losses, prox_losses, correct, total = [], [], [], 0, 0
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

            logits, _, _ = model(x_raw_n, x_enh_n, x_res_n, theta_vec, source_id, return_extra=True)

            ce = criterion(logits, y)
            prox = torch.tensor(0.0, device=DEVICE)
            if global_model is not None and CFG["fedprox_mu"] > 0:
                prox = 0.5 * CFG["fedprox_mu"] * fedprox_term(model, global_model)

            loss = ce + prox

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
        prox_losses.append(float(prox.item()))

        preds = logits.argmax(dim=1)
        correct += int((preds == y).sum().item())
        total += int(y.size(0))

    return {
        "loss": float(np.mean(losses)) if losses else np.nan,
        "ce_loss": float(np.mean(ce_losses)) if ce_losses else np.nan,
        "fedprox_loss": float(np.mean(prox_losses)) if prox_losses else np.nan,
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
best_theta_bandit_states = None

t_global_start = time.time()

print(f"Adaptive clients => DS1={n_clients_ds1}, DS2={n_clients_ds2}, TOTAL={CLIENTS_TOTAL}")
print(f"Rounds: {CFG['rounds']} | Local epochs: {CFG['local_epochs']}")
print(f"Augmentation ON: {CFG['use_augmentation']}")
print("Transfer backbone: ResNet-50")
print("Preprocessing: RACE-FELCM")
print("Fusion: CRAF")
print(f"FedProx μ={CFG['fedprox_mu']}")
print("NO prototype sharing")
print(f"Tempered FedAvg exponent = {CFG['fedavg_temper']:.2f}")
print(f"Best-round masses => DS1={CFG['best_round_mass_ds1']:.2f}, DS2={CFG['best_round_mass_ds2']:.2f}, min-bonus={CFG['best_round_min_bonus']:.2f}")

for rnd in range(1, CFG["rounds"] + 1):
    round_t0 = time.time()
    selected_ids = list(range(len(clients)))

    print("\n" + "=" * 118)
    print(f"ROUND {rnd}/{CFG['rounds']} | selected={selected_ids}")
    print("=" * 118)

    local_models = []
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
                scheduler=scheduler,
            )
            train_logs.append(log_ep)

        met_loc, _, _ = evaluate_full(local_model, client["val_loader"], preproc, theta, return_gates=True, use_tta=False)

        reward = score_metric(met_loc)
        client["theta_bandit"].update(theta_arm, reward)

        local_models.append(local_model)
        selected_clients_meta.append(client)

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
            "train_fedprox_loss": float(np.mean([x["fedprox_loss"] for x in train_logs])),
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
        best_theta_bandit_states = [copy.deepcopy(c["theta_bandit"].state_dict()) for c in clients]

if best_model_state is not None:
    global_model.load_state_dict({k: v.to(DEVICE) for k, v in best_model_state.items()})

if best_theta_bandit_states is not None:
    for c, sd in zip(clients, best_theta_bandit_states):
        c["theta_bandit"].load_state_dict(sd)

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
    {"setting": "ARCF-Net + FedAvg + FedProx", "split": "VAL",  "dataset": "ds1",          **compact_metrics(val_ds1)},
    {"setting": "ARCF-Net + FedAvg + FedProx", "split": "VAL",  "dataset": "ds2",          **compact_metrics(val_ds2)},
    {"setting": "ARCF-Net + FedAvg + FedProx", "split": "VAL",  "dataset": "global_equal", **compact_metrics(val_global)},
    {"setting": "ARCF-Net + FedAvg + FedProx", "split": "TEST", "dataset": "ds1",          **compact_metrics(test_ds1)},
    {"setting": "ARCF-Net + FedAvg + FedProx", "split": "TEST", "dataset": "ds2",          **compact_metrics(test_ds2)},
    {"setting": "ARCF-Net + FedAvg + FedProx", "split": "TEST", "dataset": "global_equal", **compact_metrics(test_global)},
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
# STEP 14: PARAMETER EVOLUTION TABLE
# ============================================================
print("\n" + "=" * 118)
print("STEP 14: PARAMETER EVOLUTION TABLE")
print("=" * 118)

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
for col in theta_cols:
    if col in loc_copy.columns:
        loc_copy[col] = pd.to_numeric(loc_copy[col], errors="coerce")

theta_evo = loc_copy.groupby("round")[theta_cols].mean(numeric_only=True).reset_index()
print_table(theta_evo, "Mean selected preprocessing parameters over rounds")
add_table_to_csv(theta_evo, "theta_evolution_mean")

# ============================================================
# STEP 15: SAVE CHECKPOINT + CSV
# ============================================================
print("\n" + "=" * 118)
print("STEP 15: SAVING ONLY TWO FILES (CHECKPOINT + ONE CSV)")
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
    "federated_variant": "FedAvg + FedProx",

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
    "theta_evolution_mean": theta_evo.to_dict(orient="list"),
}

torch.save(checkpoint, MODEL_PATH)
print(f"✅ Saved checkpoint: {MODEL_PATH}")

all_df.to_csv(CSV_PATH, index=False)
print(f"✅ Saved CSV (ALL outputs): {CSV_PATH}")

print("\nDONE ✅")
print(f"Method: {METHOD_INFO['acronym']} = {METHOD_INFO['full_form']}")
print(f"Backbone: {METHOD_INFO['backbone_full_form']}")
print("Federated variant: FedAvg + FedProx")
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
    ABLATION: FedAvg + FedProx
    ======================================================================================================================
    ENV: KAGGLE | DEVICE: cuda | torch=2.9.0+cu126
    ======================================================================================================================
    
    ======================================================================================================================
    STEP 0: ACCESS DATASETS
    ======================================================================================================================
    Dataset-1 RAW root detected:
      /kaggle/input/datasets/orvile/pmram-bangladeshi-brain-cancer-mri-dataset/PMRAM Bangladeshi Brain Cancer - MRI Dataset/PMRAM Bangladeshi Brain Cancer - MRI Dataset/Raw Data/Raw
    Dataset-2 root detected:
      /kaggle/input/datasets/chubskuy/brain-tumor-image/Testing
    
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
    ds2: glioma -> glioma | 300 images
    ds2: meningioma -> meningioma | 306 images
    ds2: notumor -> notumor | 405 images
    ds2: pituitary -> pituitary | 300 images
    
    Dataset-1 images: 1505
    label
    glioma        373
    meningioma    363
    notumor       396
    pituitary     373
    Name: count, dtype: int64
    
    Dataset-2 images: 1311
    label
    glioma        300
    meningioma    306
    notumor       405
    pituitary     300
    Name: count, dtype: int64
    
    ======================================================================================================================
    STEP 2: TRAIN / VAL / TEST SPLIT (PER DATASET)
    ======================================================================================================================
    DS1 TRAIN: 1053 | VAL: 226 | TEST: 226
    DS2 TRAIN: 917 | VAL: 197 | TEST: 197
    
    ======================================================================================================================
    STEP 2.5: SANITY / LEAKAGE CHECKS
    ======================================================================================================================
    
    ----------------------------------------------------------------------------------------------------------------------
    Leakage / Sanity Summary — ds1
    ----------------------------------------------------------------------------------------------------------------------



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
      <td>5</td>
      <td>5</td>
      <td>6</td>
      <td>298</td>
      <td>222</td>
      <td>224</td>
    </tr>
  </tbody>
</table>
</div>


    
    ----------------------------------------------------------------------------------------------------------------------
    Leakage / Sanity Summary — ds2
    ----------------------------------------------------------------------------------------------------------------------



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
      <td>917</td>
      <td>197</td>
      <td>197</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>300</td>
      <td>194</td>
      <td>197</td>
    </tr>
  </tbody>
</table>
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
      <td>0.755612</td>
      <td>0.767340</td>
      <td>0.761476</td>
      <td>0.761476</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>4</td>
      <td>0.679705</td>
      <td>0.635628</td>
      <td>0.657667</td>
      <td>0.657667</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>5</td>
      <td>0.721049</td>
      <td>0.775174</td>
      <td>0.748111</td>
      <td>0.748111</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>3</td>
      <td>0.841039</td>
      <td>0.726403</td>
      <td>0.783721</td>
      <td>0.772599</td>
      <td>2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>5</td>
      <td>0.688321</td>
      <td>0.701388</td>
      <td>0.694855</td>
      <td>0.721483</td>
      <td>2</td>
    </tr>
    <tr>
      <th>5</th>
      <td>6</td>
      <td>4</td>
      <td>0.665267</td>
      <td>0.691320</td>
      <td>0.678293</td>
      <td>0.667980</td>
      <td>2</td>
    </tr>
    <tr>
      <th>6</th>
      <td>7</td>
      <td>3</td>
      <td>0.766538</td>
      <td>0.693916</td>
      <td>0.730227</td>
      <td>0.758475</td>
      <td>3</td>
    </tr>
    <tr>
      <th>7</th>
      <td>8</td>
      <td>5</td>
      <td>0.677792</td>
      <td>0.707763</td>
      <td>0.692778</td>
      <td>0.711915</td>
      <td>3</td>
    </tr>
    <tr>
      <th>8</th>
      <td>9</td>
      <td>4</td>
      <td>0.715564</td>
      <td>0.681741</td>
      <td>0.698653</td>
      <td>0.678204</td>
      <td>3</td>
    </tr>
    <tr>
      <th>9</th>
      <td>10</td>
      <td>3</td>
      <td>0.734619</td>
      <td>0.707902</td>
      <td>0.721261</td>
      <td>0.749171</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>


    
    ======================================================================================================================
    STEP 5: FINAL NON-IID CLIENT PARTITIONING
    ======================================================================================================================
    
    ======================================================================================================================
    STEP 6: DATA LOADERS
    ======================================================================================================================
    ds1 | client_0 | train=286 | tune=45 | val=40
    ds1 | client_1 | train=257 | tune=41 | val=36
    ds1 | client_2 | train=269 | tune=42 | val=37
    ds2 | client_3 | train=100 | tune=16 | val=14
    ds2 | client_4 | train=342 | tune=54 | val=47
    ds2 | client_5 | train=265 | tune=42 | val=37
    
    ----------------------------------------------------------------------------------------------------------------------
    Adaptive client class distribution
    ----------------------------------------------------------------------------------------------------------------------



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
      <td>286</td>
      <td>45</td>
      <td>40</td>
      <td>40</td>
      <td>33</td>
      <td>166</td>
      <td>47</td>
    </tr>
    <tr>
      <th>1</th>
      <td>client_1</td>
      <td>ds1</td>
      <td>257</td>
      <td>41</td>
      <td>36</td>
      <td>135</td>
      <td>4</td>
      <td>31</td>
      <td>87</td>
    </tr>
    <tr>
      <th>2</th>
      <td>client_2</td>
      <td>ds1</td>
      <td>269</td>
      <td>42</td>
      <td>37</td>
      <td>25</td>
      <td>159</td>
      <td>17</td>
      <td>68</td>
    </tr>
    <tr>
      <th>3</th>
      <td>client_3</td>
      <td>ds2</td>
      <td>100</td>
      <td>16</td>
      <td>14</td>
      <td>74</td>
      <td>18</td>
      <td>4</td>
      <td>4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>client_4</td>
      <td>ds2</td>
      <td>342</td>
      <td>54</td>
      <td>47</td>
      <td>5</td>
      <td>121</td>
      <td>154</td>
      <td>62</td>
    </tr>
    <tr>
      <th>5</th>
      <td>client_5</td>
      <td>ds2</td>
      <td>265</td>
      <td>42</td>
      <td>37</td>
      <td>82</td>
      <td>25</td>
      <td>61</td>
      <td>97</td>
    </tr>
  </tbody>
</table>
</div>


    Augmentation: ON ✅
    Preprocessing: ON ✅
    Total adaptive clients: 6
    
    ======================================================================================================================
    STEP 7: NOVEL PREPROCESSING — RACE-FELCM
    ======================================================================================================================
    
    ======================================================================================================================
    STEP 8: MODEL — ResNet-50 + CRAF Fusion
    ======================================================================================================================
    Backbone: ResNet-50 | pretrained_loaded=True
    Total params: 25,790,855
    Trainable params: 2,282,823 (8.85%)
    
    ======================================================================================================================
    STEP 9: LOSSES + FEDPROX
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
    FedProx μ=0.01
    NO prototype sharing
    Tempered FedAvg exponent = 0.50
    Best-round masses => DS1=0.50, DS2=0.50, min-bonus=0.15
    
    ======================================================================================================================
    ROUND 1/15 | selected=[0, 1, 2, 3, 4, 5]
    ======================================================================================================================
    Client 0 (ds1) | train_acc=0.6923 | val_acc=0.8250 | val_f1=0.7109 | val_auc=0.9318 | reward=0.7394 | theta=race_edge_plus (g=1.02, a=0.32, b=6.00, t=2.60, k=3, eg=0.15, mix=0.84)
    Client 1 (ds1) | train_acc=0.7315 | val_acc=0.9167 | val_f1=0.6920 | val_auc=0.9896 | reward=0.7481 | theta=race_focus (g=1.10, a=0.36, b=6.40, t=2.70, k=3, eg=0.17, mix=0.86)
    Client 2 (ds1) | train_acc=0.6580 | val_acc=0.5405 | val_f1=0.5729 | val_auc=0.8409 | reward=0.5648 | theta=race_focus (g=1.10, a=0.36, b=6.40, t=2.70, k=3, eg=0.17, mix=0.86)
    Client 3 (ds2) | train_acc=0.5700 | val_acc=0.7857 | val_f1=0.5185 | val_auc=nan | reward=0.5853 | theta=race_robust (g=1.08, a=0.22, b=4.50, t=2.80, k=5, eg=0.11, mix=0.76)
    Client 4 (ds2) | train_acc=0.6930 | val_acc=0.8936 | val_f1=0.6803 | val_auc=0.9316 | reward=0.7337 | theta=race_smoothmix (g=0.90, a=0.20, b=4.00, t=2.10, k=7, eg=0.07, mix=0.70)
    Client 5 (ds2) | train_acc=0.6226 | val_acc=0.7568 | val_f1=0.6264 | val_auc=0.9531 | reward=0.6590 | theta=race_robust (g=1.08, a=0.22, b=4.50, t=2.80, k=5, eg=0.11, mix=0.76)
    
    ----------------------------------------------------------------------------------------------------------------------
    GLOBAL VAL (Round 1) | global_acc=0.7915 | global_f1=0.6491 | ds1_acc=0.7611 | ds1_f1=0.6597 | ds2_acc=0.8265 | ds2_f1=0.6369 | reward=0.7873 | round_time=40.1s
    ----------------------------------------------------------------------------------------------------------------------
    
    ======================================================================================================================
    ROUND 2/15 | selected=[0, 1, 2, 3, 4, 5]
    ======================================================================================================================
    Client 0 (ds1) | train_acc=0.8654 | val_acc=0.8000 | val_f1=0.6484 | val_auc=0.9699 | reward=0.6863 | theta=race_robust (g=1.08, a=0.22, b=4.50, t=2.80, k=5, eg=0.11, mix=0.76)
    Client 1 (ds1) | train_acc=0.9086 | val_acc=0.8611 | val_f1=0.6345 | val_auc=0.9807 | reward=0.6911 | theta=race_edge_plus (g=1.02, a=0.32, b=6.00, t=2.60, k=3, eg=0.15, mix=0.84)
    Client 2 (ds1) | train_acc=0.8439 | val_acc=0.7568 | val_f1=0.6702 | val_auc=0.9579 | reward=0.6918 | theta=race_edge_plus (g=1.02, a=0.32, b=6.00, t=2.60, k=3, eg=0.15, mix=0.84)
    Client 3 (ds2) | train_acc=0.6950 | val_acc=0.8571 | val_f1=0.8030 | val_auc=nan | reward=0.8166 | theta=race_soft (g=0.95, a=0.18, b=3.80, t=2.20, k=3, eg=0.08, mix=0.72)
    Client 4 (ds2) | train_acc=0.7895 | val_acc=0.8936 | val_f1=0.6783 | val_auc=0.9539 | reward=0.7321 | theta=race_robust (g=1.08, a=0.22, b=4.50, t=2.80, k=5, eg=0.11, mix=0.76)
    Client 5 (ds2) | train_acc=0.8019 | val_acc=0.8108 | val_f1=0.6515 | val_auc=0.9671 | reward=0.6913 | theta=race_smoothmix (g=0.90, a=0.20, b=4.00, t=2.10, k=7, eg=0.07, mix=0.70)
    
    ----------------------------------------------------------------------------------------------------------------------
    GLOBAL VAL (Round 2) | global_acc=0.8294 | global_f1=0.6673 | ds1_acc=0.8053 | ds1_f1=0.6511 | ds2_acc=0.8571 | ds2_f1=0.6860 | reward=0.8127 | round_time=37.4s
    ----------------------------------------------------------------------------------------------------------------------
    
    ======================================================================================================================
    ROUND 3/15 | selected=[0, 1, 2, 3, 4, 5]
    ======================================================================================================================
    Client 0 (ds1) | train_acc=0.8671 | val_acc=0.8000 | val_f1=0.6674 | val_auc=0.9723 | reward=0.7005 | theta=race_focus (g=1.10, a=0.36, b=6.40, t=2.70, k=3, eg=0.17, mix=0.86)
    Client 1 (ds1) | train_acc=0.8580 | val_acc=0.9167 | val_f1=0.6851 | val_auc=0.9780 | reward=0.7430 | theta=race_texture (g=0.92, a=0.34, b=5.80, t=2.30, k=7, eg=0.10, mix=0.80)
    Client 2 (ds1) | train_acc=0.8457 | val_acc=0.9189 | val_f1=0.8683 | val_auc=0.9947 | reward=0.8810 | theta=race_texture (g=0.92, a=0.34, b=5.80, t=2.30, k=7, eg=0.10, mix=0.80)
    Client 3 (ds2) | train_acc=0.7700 | val_acc=0.8571 | val_f1=0.8030 | val_auc=nan | reward=0.8166 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    Client 4 (ds2) | train_acc=0.8099 | val_acc=0.8723 | val_f1=0.6664 | val_auc=0.9182 | reward=0.7179 | theta=race_soft (g=0.95, a=0.18, b=3.80, t=2.20, k=3, eg=0.08, mix=0.72)
    Client 5 (ds2) | train_acc=0.7811 | val_acc=0.8919 | val_f1=0.8526 | val_auc=0.9840 | reward=0.8624 | theta=race_soft (g=0.95, a=0.18, b=3.80, t=2.20, k=3, eg=0.08, mix=0.72)
    
    ----------------------------------------------------------------------------------------------------------------------
    GLOBAL VAL (Round 3) | global_acc=0.8768 | global_f1=0.7469 | ds1_acc=0.8761 | ds1_f1=0.7388 | ds2_acc=0.8776 | ds2_f1=0.7562 | reward=0.8958 | round_time=45.3s
    ----------------------------------------------------------------------------------------------------------------------
    
    ======================================================================================================================
    ROUND 4/15 | selected=[0, 1, 2, 3, 4, 5]
    ======================================================================================================================
    Client 0 (ds1) | train_acc=0.7815 | val_acc=0.7500 | val_f1=0.6303 | val_auc=0.9464 | reward=0.6602 | theta=race_sharp (g=1.04, a=0.30, b=5.20, t=2.50, k=5, eg=0.12, mix=0.82)
    Client 1 (ds1) | train_acc=0.8969 | val_acc=0.8889 | val_f1=0.6516 | val_auc=0.9922 | reward=0.7109 | theta=race_robust (g=1.08, a=0.22, b=4.50, t=2.80, k=5, eg=0.11, mix=0.76)
    Client 2 (ds1) | train_acc=0.8848 | val_acc=0.9459 | val_f1=0.8953 | val_auc=0.9943 | reward=0.9080 | theta=race_sharp (g=1.04, a=0.30, b=5.20, t=2.50, k=5, eg=0.12, mix=0.82)
    Client 3 (ds2) | train_acc=0.7800 | val_acc=0.8571 | val_f1=0.6131 | val_auc=nan | reward=0.6741 | theta=race_soft (g=0.95, a=0.18, b=3.80, t=2.20, k=3, eg=0.08, mix=0.72)
    Client 4 (ds2) | train_acc=0.8480 | val_acc=0.9149 | val_f1=0.6807 | val_auc=0.9877 | reward=0.7392 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    Client 5 (ds2) | train_acc=0.8075 | val_acc=0.8378 | val_f1=0.6712 | val_auc=0.9441 | reward=0.7128 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    
    ----------------------------------------------------------------------------------------------------------------------
    GLOBAL VAL (Round 4) | global_acc=0.8673 | global_f1=0.6977 | ds1_acc=0.8584 | ds1_f1=0.7239 | ds2_acc=0.8776 | ds2_f1=0.6674 | reward=0.8467 | round_time=45.4s
    ----------------------------------------------------------------------------------------------------------------------
    
    ======================================================================================================================
    ROUND 5/15 | selected=[0, 1, 2, 3, 4, 5]
    ======================================================================================================================
    Client 0 (ds1) | train_acc=0.9161 | val_acc=0.8500 | val_f1=0.7031 | val_auc=0.9737 | reward=0.7398 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    Client 1 (ds1) | train_acc=0.9300 | val_acc=0.9444 | val_f1=0.6932 | val_auc=0.9953 | reward=0.7560 | theta=race_sharp (g=1.04, a=0.30, b=5.20, t=2.50, k=5, eg=0.12, mix=0.82)
    Client 2 (ds1) | train_acc=0.9517 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | reward=1.0000 | theta=race_robust (g=1.08, a=0.22, b=4.50, t=2.80, k=5, eg=0.11, mix=0.76)
    Client 3 (ds2) | train_acc=0.7700 | val_acc=0.9286 | val_f1=0.6190 | val_auc=nan | reward=0.6964 | theta=race_smoothmix (g=0.90, a=0.20, b=4.00, t=2.10, k=7, eg=0.07, mix=0.70)
    Client 4 (ds2) | train_acc=0.8743 | val_acc=0.9362 | val_f1=0.6982 | val_auc=0.9964 | reward=0.7577 | theta=race_smoothmix (g=0.90, a=0.20, b=4.00, t=2.10, k=7, eg=0.07, mix=0.70)
    Client 5 (ds2) | train_acc=0.8472 | val_acc=0.7838 | val_f1=0.6347 | val_auc=0.8987 | reward=0.6720 | theta=race_soft (g=0.95, a=0.18, b=3.80, t=2.20, k=3, eg=0.08, mix=0.72)
    
    ----------------------------------------------------------------------------------------------------------------------
    GLOBAL VAL (Round 5) | global_acc=0.9052 | global_f1=0.7348 | ds1_acc=0.9292 | ds1_f1=0.7972 | ds2_acc=0.8776 | ds2_f1=0.6629 | reward=0.8809 | round_time=44.7s
    ----------------------------------------------------------------------------------------------------------------------
    
    ======================================================================================================================
    ROUND 6/15 | selected=[0, 1, 2, 3, 4, 5]
    ======================================================================================================================
    Client 0 (ds1) | train_acc=0.9371 | val_acc=0.8000 | val_f1=0.6725 | val_auc=0.9651 | reward=0.7044 | theta=race_texture (g=0.92, a=0.34, b=5.80, t=2.30, k=7, eg=0.10, mix=0.80)
    Client 1 (ds1) | train_acc=0.9650 | val_acc=0.9444 | val_f1=0.6811 | val_auc=0.9909 | reward=0.7469 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    Client 2 (ds1) | train_acc=0.9461 | val_acc=0.9730 | val_f1=0.9664 | val_auc=1.0000 | reward=0.9680 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    Client 3 (ds2) | train_acc=0.9350 | val_acc=0.9286 | val_f1=0.9175 | val_auc=nan | reward=0.9202 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    Client 4 (ds2) | train_acc=0.9167 | val_acc=0.9362 | val_f1=0.9402 | val_auc=0.9945 | reward=0.9392 | theta=race_robust (g=1.08, a=0.22, b=4.50, t=2.80, k=5, eg=0.11, mix=0.76)
    Client 5 (ds2) | train_acc=0.9264 | val_acc=0.8378 | val_f1=0.7796 | val_auc=0.9717 | reward=0.7942 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    
    ----------------------------------------------------------------------------------------------------------------------
    GLOBAL VAL (Round 6) | global_acc=0.9005 | global_f1=0.8202 | ds1_acc=0.9027 | ds1_f1=0.7715 | ds2_acc=0.8980 | ds2_f1=0.8763 | reward=0.9636 | round_time=44.9s
    ----------------------------------------------------------------------------------------------------------------------
    
    ======================================================================================================================
    ROUND 7/15 | selected=[0, 1, 2, 3, 4, 5]
    ======================================================================================================================
    Client 0 (ds1) | train_acc=0.9580 | val_acc=0.8750 | val_f1=0.7793 | val_auc=0.9880 | reward=0.8032 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    Client 1 (ds1) | train_acc=0.9767 | val_acc=0.9444 | val_f1=0.7091 | val_auc=0.9992 | reward=0.7679 | theta=race_sharp (g=1.04, a=0.30, b=5.20, t=2.50, k=5, eg=0.12, mix=0.82)
    Client 2 (ds1) | train_acc=0.9554 | val_acc=0.9730 | val_f1=0.9664 | val_auc=1.0000 | reward=0.9680 | theta=race_robust (g=1.08, a=0.22, b=4.50, t=2.80, k=5, eg=0.11, mix=0.76)
    Client 3 (ds2) | train_acc=0.9100 | val_acc=0.9286 | val_f1=0.7368 | val_auc=nan | reward=0.7848 | theta=race_smoothmix (g=0.90, a=0.20, b=4.00, t=2.10, k=7, eg=0.07, mix=0.70)
    Client 4 (ds2) | train_acc=0.9123 | val_acc=0.9149 | val_f1=0.6892 | val_auc=0.9953 | reward=0.7456 | theta=race_robust (g=1.08, a=0.22, b=4.50, t=2.80, k=5, eg=0.11, mix=0.76)
    Client 5 (ds2) | train_acc=0.9283 | val_acc=0.8919 | val_f1=0.8625 | val_auc=0.9608 | reward=0.8699 | theta=race_smoothmix (g=0.90, a=0.20, b=4.00, t=2.10, k=7, eg=0.07, mix=0.70)
    
    ----------------------------------------------------------------------------------------------------------------------
    GLOBAL VAL (Round 7) | global_acc=0.9194 | global_f1=0.7918 | ds1_acc=0.9292 | ds1_f1=0.8182 | ds2_acc=0.9082 | ds2_f1=0.7614 | reward=0.9417 | round_time=45.1s
    ----------------------------------------------------------------------------------------------------------------------
    
    ======================================================================================================================
    ROUND 8/15 | selected=[0, 1, 2, 3, 4, 5]
    ======================================================================================================================
    Client 0 (ds1) | train_acc=0.9178 | val_acc=0.8750 | val_f1=0.8518 | val_auc=0.9819 | reward=0.8576 | theta=race_edge_plus (g=1.02, a=0.32, b=6.00, t=2.60, k=3, eg=0.15, mix=0.84)
    Client 1 (ds1) | train_acc=0.9358 | val_acc=0.9722 | val_f1=0.7436 | val_auc=1.0000 | reward=0.8007 | theta=race_focus (g=1.10, a=0.36, b=6.40, t=2.70, k=3, eg=0.17, mix=0.86)
    Client 2 (ds1) | train_acc=0.9498 | val_acc=0.9730 | val_f1=0.9797 | val_auc=1.0000 | reward=0.9780 | theta=race_texture (g=0.92, a=0.34, b=5.80, t=2.30, k=7, eg=0.10, mix=0.80)
    Client 3 (ds2) | train_acc=0.9100 | val_acc=0.9286 | val_f1=0.9175 | val_auc=nan | reward=0.9202 | theta=race_robust (g=1.08, a=0.22, b=4.50, t=2.80, k=5, eg=0.11, mix=0.76)
    Client 4 (ds2) | train_acc=0.9474 | val_acc=0.9574 | val_f1=0.9639 | val_auc=0.9968 | reward=0.9623 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    Client 5 (ds2) | train_acc=0.9208 | val_acc=0.8649 | val_f1=0.8021 | val_auc=0.9823 | reward=0.8178 | theta=race_robust (g=1.08, a=0.22, b=4.50, t=2.80, k=5, eg=0.11, mix=0.76)
    
    ----------------------------------------------------------------------------------------------------------------------
    GLOBAL VAL (Round 8) | global_acc=0.9289 | global_f1=0.8764 | ds1_acc=0.9381 | ds1_f1=0.8592 | ds2_acc=0.9184 | ds2_f1=0.8962 | reward=1.0222 | round_time=45.0s
    ----------------------------------------------------------------------------------------------------------------------
    
    ======================================================================================================================
    ROUND 9/15 | selected=[0, 1, 2, 3, 4, 5]
    ======================================================================================================================
    Client 0 (ds1) | train_acc=0.9668 | val_acc=0.9250 | val_f1=0.8872 | val_auc=0.9925 | reward=0.8967 | theta=race_focus (g=1.10, a=0.36, b=6.40, t=2.70, k=3, eg=0.17, mix=0.86)
    Client 1 (ds1) | train_acc=0.9689 | val_acc=0.9444 | val_f1=0.6775 | val_auc=0.9980 | reward=0.7442 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    Client 2 (ds1) | train_acc=0.9257 | val_acc=0.9730 | val_f1=0.9442 | val_auc=0.9992 | reward=0.9514 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    Client 3 (ds2) | train_acc=0.9600 | val_acc=1.0000 | val_f1=1.0000 | val_auc=nan | reward=1.0000 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    Client 4 (ds2) | train_acc=0.9430 | val_acc=0.9787 | val_f1=0.9762 | val_auc=0.9995 | reward=0.9768 | theta=race_soft (g=0.95, a=0.18, b=3.80, t=2.20, k=3, eg=0.08, mix=0.72)
    Client 5 (ds2) | train_acc=0.9264 | val_acc=0.8378 | val_f1=0.7511 | val_auc=0.9624 | reward=0.7728 | theta=race_smoothmix (g=0.90, a=0.20, b=4.00, t=2.10, k=7, eg=0.07, mix=0.70)
    
    ----------------------------------------------------------------------------------------------------------------------
    GLOBAL VAL (Round 9) | global_acc=0.9384 | global_f1=0.8649 | ds1_acc=0.9469 | ds1_f1=0.8391 | ds2_acc=0.9286 | ds2_f1=0.8946 | reward=1.0145 | round_time=45.4s
    ----------------------------------------------------------------------------------------------------------------------
    
    ======================================================================================================================
    ROUND 10/15 | selected=[0, 1, 2, 3, 4, 5]
    ======================================================================================================================
    Client 0 (ds1) | train_acc=0.9703 | val_acc=0.9000 | val_f1=0.8113 | val_auc=0.9850 | reward=0.8335 | theta=race_texture (g=0.92, a=0.34, b=5.80, t=2.30, k=7, eg=0.10, mix=0.80)
    Client 1 (ds1) | train_acc=0.9689 | val_acc=0.9722 | val_f1=0.7436 | val_auc=1.0000 | reward=0.8007 | theta=race_robust (g=1.08, a=0.22, b=4.50, t=2.80, k=5, eg=0.11, mix=0.76)
    Client 2 (ds1) | train_acc=0.9721 | val_acc=0.9189 | val_f1=0.8701 | val_auc=0.9958 | reward=0.8823 | theta=race_sharp (g=1.04, a=0.30, b=5.20, t=2.50, k=5, eg=0.12, mix=0.82)
    Client 3 (ds2) | train_acc=0.9750 | val_acc=0.9286 | val_f1=0.9175 | val_auc=nan | reward=0.9202 | theta=race_robust (g=1.08, a=0.22, b=4.50, t=2.80, k=5, eg=0.11, mix=0.76)
    Client 4 (ds2) | train_acc=0.9547 | val_acc=0.8723 | val_f1=0.8892 | val_auc=0.9878 | reward=0.8850 | theta=race_smoothmix (g=0.90, a=0.20, b=4.00, t=2.10, k=7, eg=0.07, mix=0.70)
    Client 5 (ds2) | train_acc=0.9264 | val_acc=0.8919 | val_f1=0.8625 | val_auc=0.9447 | reward=0.8699 | theta=race_soft (g=0.95, a=0.18, b=3.80, t=2.20, k=3, eg=0.08, mix=0.72)
    
    ----------------------------------------------------------------------------------------------------------------------
    GLOBAL VAL (Round 10) | global_acc=0.9100 | global_f1=0.8434 | ds1_acc=0.9292 | ds1_f1=0.8090 | ds2_acc=0.8878 | ds2_f1=0.8832 | reward=0.9875 | round_time=45.1s
    ----------------------------------------------------------------------------------------------------------------------
    
    ======================================================================================================================
    ROUND 11/15 | selected=[0, 1, 2, 3, 4, 5]
    ======================================================================================================================
    Client 0 (ds1) | train_acc=0.9843 | val_acc=0.9250 | val_f1=0.8947 | val_auc=0.9926 | reward=0.9023 | theta=race_focus (g=1.10, a=0.36, b=6.40, t=2.70, k=3, eg=0.17, mix=0.86)
    Client 1 (ds1) | train_acc=0.9825 | val_acc=0.9444 | val_f1=0.7268 | val_auc=0.9985 | reward=0.7812 | theta=race_texture (g=0.92, a=0.34, b=5.80, t=2.30, k=7, eg=0.10, mix=0.80)
    Client 2 (ds1) | train_acc=0.9647 | val_acc=0.9459 | val_f1=0.9381 | val_auc=1.0000 | reward=0.9401 | theta=race_focus (g=1.10, a=0.36, b=6.40, t=2.70, k=3, eg=0.17, mix=0.86)
    Client 3 (ds2) | train_acc=0.9550 | val_acc=0.9286 | val_f1=0.9175 | val_auc=nan | reward=0.9202 | theta=race_soft (g=0.95, a=0.18, b=3.80, t=2.20, k=3, eg=0.08, mix=0.72)
    Client 4 (ds2) | train_acc=0.9591 | val_acc=0.9149 | val_f1=0.6835 | val_auc=0.9918 | reward=0.7414 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    Client 5 (ds2) | train_acc=0.9509 | val_acc=0.9189 | val_f1=0.8722 | val_auc=0.9954 | reward=0.8839 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    
    ----------------------------------------------------------------------------------------------------------------------
    GLOBAL VAL (Round 11) | global_acc=0.9289 | global_f1=0.8242 | ds1_acc=0.9381 | ds1_f1=0.8554 | ds2_acc=0.9184 | ds2_f1=0.7882 | reward=0.9715 | round_time=44.9s
    ----------------------------------------------------------------------------------------------------------------------
    
    ======================================================================================================================
    ROUND 12/15 | selected=[0, 1, 2, 3, 4, 5]
    ======================================================================================================================
    Client 0 (ds1) | train_acc=0.9650 | val_acc=0.9000 | val_f1=0.8299 | val_auc=0.9816 | reward=0.8474 | theta=race_robust (g=1.08, a=0.22, b=4.50, t=2.80, k=5, eg=0.11, mix=0.76)
    Client 1 (ds1) | train_acc=0.9611 | val_acc=0.9167 | val_f1=0.9104 | val_auc=0.9872 | reward=0.9120 | theta=race_edge_plus (g=1.02, a=0.32, b=6.00, t=2.60, k=3, eg=0.15, mix=0.84)
    Client 2 (ds1) | train_acc=0.9480 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | reward=1.0000 | theta=race_edge_plus (g=1.02, a=0.32, b=6.00, t=2.60, k=3, eg=0.15, mix=0.84)
    Client 3 (ds2) | train_acc=0.9700 | val_acc=0.9286 | val_f1=0.9175 | val_auc=nan | reward=0.9202 | theta=race_smoothmix (g=0.90, a=0.20, b=4.00, t=2.10, k=7, eg=0.07, mix=0.70)
    Client 4 (ds2) | train_acc=0.9488 | val_acc=0.8936 | val_f1=0.6345 | val_auc=0.9780 | reward=0.6993 | theta=race_soft (g=0.95, a=0.18, b=3.80, t=2.20, k=3, eg=0.08, mix=0.72)
    Client 5 (ds2) | train_acc=0.9472 | val_acc=0.8649 | val_f1=0.8372 | val_auc=0.9802 | reward=0.8441 | theta=race_robust (g=1.08, a=0.22, b=4.50, t=2.80, k=5, eg=0.11, mix=0.76)
    
    ----------------------------------------------------------------------------------------------------------------------
    GLOBAL VAL (Round 12) | global_acc=0.9147 | global_f1=0.8370 | ds1_acc=0.9381 | ds1_f1=0.9112 | ds2_acc=0.8878 | ds2_f1=0.7515 | reward=0.9696 | round_time=45.1s
    ----------------------------------------------------------------------------------------------------------------------
    
    ======================================================================================================================
    ROUND 13/15 | selected=[0, 1, 2, 3, 4, 5]
    ======================================================================================================================
    Client 0 (ds1) | train_acc=0.9825 | val_acc=0.9500 | val_f1=0.9169 | val_auc=0.9964 | reward=0.9252 | theta=race_edge_plus (g=1.02, a=0.32, b=6.00, t=2.60, k=3, eg=0.15, mix=0.84)
    Client 1 (ds1) | train_acc=0.9844 | val_acc=0.9722 | val_f1=0.9655 | val_auc=0.9965 | reward=0.9672 | theta=race_edge_plus (g=1.02, a=0.32, b=6.00, t=2.60, k=3, eg=0.15, mix=0.84)
    Client 2 (ds1) | train_acc=0.9777 | val_acc=0.8378 | val_f1=0.7852 | val_auc=1.0000 | reward=0.7984 | theta=race_robust (g=1.08, a=0.22, b=4.50, t=2.80, k=5, eg=0.11, mix=0.76)
    Client 3 (ds2) | train_acc=0.9650 | val_acc=0.9286 | val_f1=0.9175 | val_auc=nan | reward=0.9202 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    Client 4 (ds2) | train_acc=0.9854 | val_acc=0.9574 | val_f1=0.7311 | val_auc=1.0000 | reward=0.7877 | theta=race_soft (g=0.95, a=0.18, b=3.80, t=2.20, k=3, eg=0.08, mix=0.72)
    Client 5 (ds2) | train_acc=0.9623 | val_acc=0.9189 | val_f1=0.8792 | val_auc=0.9215 | reward=0.8891 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    
    ----------------------------------------------------------------------------------------------------------------------
    GLOBAL VAL (Round 13) | global_acc=0.9289 | global_f1=0.8541 | ds1_acc=0.9204 | ds1_f1=0.8893 | ds2_acc=0.9388 | ds2_f1=0.8136 | reward=0.9977 | round_time=45.1s
    ----------------------------------------------------------------------------------------------------------------------
    
    ======================================================================================================================
    ROUND 14/15 | selected=[0, 1, 2, 3, 4, 5]
    ======================================================================================================================
    Client 0 (ds1) | train_acc=0.9773 | val_acc=0.9500 | val_f1=0.9095 | val_auc=0.9917 | reward=0.9196 | theta=race_sharp (g=1.04, a=0.30, b=5.20, t=2.50, k=5, eg=0.12, mix=0.82)
    Client 1 (ds1) | train_acc=0.9767 | val_acc=0.9444 | val_f1=0.9243 | val_auc=0.9965 | reward=0.9294 | theta=race_focus (g=1.10, a=0.36, b=6.40, t=2.70, k=3, eg=0.17, mix=0.86)
    Client 2 (ds1) | train_acc=0.9740 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | reward=1.0000 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    Client 3 (ds2) | train_acc=0.9900 | val_acc=1.0000 | val_f1=1.0000 | val_auc=nan | reward=1.0000 | theta=race_robust (g=1.08, a=0.22, b=4.50, t=2.80, k=5, eg=0.11, mix=0.76)
    Client 4 (ds2) | train_acc=0.9693 | val_acc=0.9149 | val_f1=0.9261 | val_auc=0.9977 | reward=0.9233 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    Client 5 (ds2) | train_acc=0.9396 | val_acc=0.9459 | val_f1=0.9342 | val_auc=0.9847 | reward=0.9371 | theta=race_soft (g=0.95, a=0.18, b=3.80, t=2.20, k=3, eg=0.08, mix=0.72)
    
    ----------------------------------------------------------------------------------------------------------------------
    GLOBAL VAL (Round 14) | global_acc=0.9526 | global_f1=0.9419 | ds1_acc=0.9646 | ds1_f1=0.9438 | ds2_acc=0.9388 | ds2_f1=0.9397 | reward=1.0852 | round_time=44.7s
    ----------------------------------------------------------------------------------------------------------------------
    
    ======================================================================================================================
    ROUND 15/15 | selected=[0, 1, 2, 3, 4, 5]
    ======================================================================================================================
    Client 0 (ds1) | train_acc=0.9808 | val_acc=0.8750 | val_f1=0.7829 | val_auc=0.9806 | reward=0.8059 | theta=race_sharp (g=1.04, a=0.30, b=5.20, t=2.50, k=5, eg=0.12, mix=0.82)
    Client 1 (ds1) | train_acc=0.9708 | val_acc=0.9722 | val_f1=0.9579 | val_auc=1.0000 | reward=0.9615 | theta=race_texture (g=0.92, a=0.34, b=5.80, t=2.30, k=7, eg=0.10, mix=0.80)
    Client 2 (ds1) | train_acc=0.9888 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | reward=1.0000 | theta=race_texture (g=0.92, a=0.34, b=5.80, t=2.30, k=7, eg=0.10, mix=0.80)
    Client 3 (ds2) | train_acc=0.9800 | val_acc=1.0000 | val_f1=1.0000 | val_auc=nan | reward=1.0000 | theta=race_soft (g=0.95, a=0.18, b=3.80, t=2.20, k=3, eg=0.08, mix=0.72)
    Client 4 (ds2) | train_acc=0.9708 | val_acc=0.9787 | val_f1=0.9777 | val_auc=1.0000 | reward=0.9780 | theta=race_robust (g=1.08, a=0.22, b=4.50, t=2.80, k=5, eg=0.11, mix=0.76)
    Client 5 (ds2) | train_acc=0.9792 | val_acc=0.9189 | val_f1=0.8629 | val_auc=0.9855 | reward=0.8769 | theta=race_smoothmix (g=0.90, a=0.20, b=4.00, t=2.10, k=7, eg=0.07, mix=0.70)
    
    ----------------------------------------------------------------------------------------------------------------------
    GLOBAL VAL (Round 15) | global_acc=0.9526 | global_f1=0.9226 | ds1_acc=0.9469 | ds1_f1=0.9097 | ds2_acc=0.9592 | ds2_f1=0.9375 | reward=1.0688 | round_time=45.2s
    ----------------------------------------------------------------------------------------------------------------------
    
    ======================================================================================================================
    TRAINING COMPLETE ✅ | total_time=663.6s | best_round=14 | best_reward=1.0852
    ======================================================================================================================
    
    ----------------------------------------------------------------------------------------------------------------------
    GLOBAL per-round metrics
    ----------------------------------------------------------------------------------------------------------------------



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
      <td>40.111688</td>
      <td>6</td>
      <td>1.0</td>
      <td>0.787296</td>
      <td>0.791469</td>
      <td>0.649082</td>
      <td>0.689576</td>
      <td>0.666314</td>
      <td>0.558016</td>
      <td>0.559919</td>
      <td>0.614699</td>
      <td>0.761062</td>
      <td>0.659673</td>
      <td>0.609661</td>
      <td>0.826531</td>
      <td>0.636870</td>
      <td>0.498466</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>37.399964</td>
      <td>6</td>
      <td>1.0</td>
      <td>0.812657</td>
      <td>0.829384</td>
      <td>0.667298</td>
      <td>0.702199</td>
      <td>0.689253</td>
      <td>0.444867</td>
      <td>0.430144</td>
      <td>0.514053</td>
      <td>0.805310</td>
      <td>0.651090</td>
      <td>0.472424</td>
      <td>0.857143</td>
      <td>0.685986</td>
      <td>0.413094</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>45.289140</td>
      <td>6</td>
      <td>1.0</td>
      <td>0.895812</td>
      <td>0.876777</td>
      <td>0.746894</td>
      <td>0.788860</td>
      <td>0.752927</td>
      <td>0.347306</td>
      <td>0.336165</td>
      <td>0.505693</td>
      <td>0.876106</td>
      <td>0.738811</td>
      <td>0.301701</td>
      <td>0.877551</td>
      <td>0.756213</td>
      <td>0.399890</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>45.356140</td>
      <td>6</td>
      <td>1.0</td>
      <td>0.846734</td>
      <td>0.867299</td>
      <td>0.697664</td>
      <td>0.708945</td>
      <td>0.709435</td>
      <td>0.344887</td>
      <td>0.340087</td>
      <td>0.506835</td>
      <td>0.858407</td>
      <td>0.723871</td>
      <td>0.343757</td>
      <td>0.877551</td>
      <td>0.667446</td>
      <td>0.346189</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>44.731879</td>
      <td>6</td>
      <td>1.0</td>
      <td>0.880868</td>
      <td>0.905213</td>
      <td>0.734818</td>
      <td>0.752089</td>
      <td>0.763681</td>
      <td>0.288239</td>
      <td>0.290344</td>
      <td>0.498482</td>
      <td>0.929204</td>
      <td>0.797158</td>
      <td>0.223824</td>
      <td>0.877551</td>
      <td>0.662936</td>
      <td>0.362513</td>
    </tr>
    <tr>
      <th>5</th>
      <td>6</td>
      <td>44.887458</td>
      <td>6</td>
      <td>1.0</td>
      <td>0.963646</td>
      <td>0.900474</td>
      <td>0.820177</td>
      <td>0.870677</td>
      <td>0.816541</td>
      <td>0.287403</td>
      <td>0.278682</td>
      <td>0.494486</td>
      <td>0.902655</td>
      <td>0.771483</td>
      <td>0.291686</td>
      <td>0.897959</td>
      <td>0.876324</td>
      <td>0.282464</td>
    </tr>
    <tr>
      <th>6</th>
      <td>7</td>
      <td>45.146214</td>
      <td>6</td>
      <td>1.0</td>
      <td>0.941740</td>
      <td>0.919431</td>
      <td>0.791821</td>
      <td>0.800196</td>
      <td>0.810743</td>
      <td>0.252791</td>
      <td>0.243223</td>
      <td>0.502876</td>
      <td>0.929204</td>
      <td>0.818183</td>
      <td>0.206616</td>
      <td>0.908163</td>
      <td>0.761424</td>
      <td>0.306035</td>
    </tr>
    <tr>
      <th>7</th>
      <td>8</td>
      <td>45.035618</td>
      <td>6</td>
      <td>1.0</td>
      <td>1.022160</td>
      <td>0.928910</td>
      <td>0.876377</td>
      <td>0.872609</td>
      <td>0.887704</td>
      <td>0.227023</td>
      <td>0.216457</td>
      <td>0.505564</td>
      <td>0.938053</td>
      <td>0.859228</td>
      <td>0.200429</td>
      <td>0.918367</td>
      <td>0.896150</td>
      <td>0.257686</td>
    </tr>
    <tr>
      <th>8</th>
      <td>9</td>
      <td>45.353781</td>
      <td>6</td>
      <td>1.0</td>
      <td>1.014465</td>
      <td>0.938389</td>
      <td>0.864863</td>
      <td>0.878997</td>
      <td>0.864698</td>
      <td>0.205294</td>
      <td>0.216622</td>
      <td>0.508482</td>
      <td>0.946903</td>
      <td>0.839053</td>
      <td>0.180362</td>
      <td>0.928571</td>
      <td>0.894624</td>
      <td>0.234042</td>
    </tr>
    <tr>
      <th>9</th>
      <td>10</td>
      <td>45.063540</td>
      <td>6</td>
      <td>1.0</td>
      <td>0.987535</td>
      <td>0.909953</td>
      <td>0.843442</td>
      <td>0.854231</td>
      <td>0.850571</td>
      <td>0.243313</td>
      <td>0.225279</td>
      <td>0.506886</td>
      <td>0.929204</td>
      <td>0.808985</td>
      <td>0.177554</td>
      <td>0.887755</td>
      <td>0.883174</td>
      <td>0.319138</td>
    </tr>
    <tr>
      <th>10</th>
      <td>11</td>
      <td>44.879538</td>
      <td>6</td>
      <td>1.0</td>
      <td>0.971518</td>
      <td>0.928910</td>
      <td>0.824197</td>
      <td>0.837234</td>
      <td>0.829286</td>
      <td>0.210110</td>
      <td>0.191793</td>
      <td>0.496572</td>
      <td>0.938053</td>
      <td>0.855427</td>
      <td>0.205643</td>
      <td>0.918367</td>
      <td>0.788187</td>
      <td>0.215262</td>
    </tr>
    <tr>
      <th>11</th>
      <td>12</td>
      <td>45.103329</td>
      <td>6</td>
      <td>1.0</td>
      <td>0.969564</td>
      <td>0.914692</td>
      <td>0.837023</td>
      <td>0.860199</td>
      <td>0.856512</td>
      <td>0.294976</td>
      <td>0.272827</td>
      <td>0.495643</td>
      <td>0.938053</td>
      <td>0.911225</td>
      <td>0.232057</td>
      <td>0.887755</td>
      <td>0.751462</td>
      <td>0.367526</td>
    </tr>
    <tr>
      <th>12</th>
      <td>13</td>
      <td>45.058043</td>
      <td>6</td>
      <td>1.0</td>
      <td>0.997698</td>
      <td>0.928910</td>
      <td>0.854118</td>
      <td>0.851195</td>
      <td>0.883573</td>
      <td>0.219780</td>
      <td>0.205241</td>
      <td>0.501607</td>
      <td>0.920354</td>
      <td>0.889251</td>
      <td>0.223234</td>
      <td>0.938776</td>
      <td>0.813607</td>
      <td>0.215799</td>
    </tr>
    <tr>
      <th>13</th>
      <td>14</td>
      <td>44.690950</td>
      <td>6</td>
      <td>1.0</td>
      <td>1.085186</td>
      <td>0.952607</td>
      <td>0.941934</td>
      <td>0.950968</td>
      <td>0.935943</td>
      <td>0.157226</td>
      <td>0.140854</td>
      <td>0.499195</td>
      <td>0.964602</td>
      <td>0.943845</td>
      <td>0.127433</td>
      <td>0.938776</td>
      <td>0.939730</td>
      <td>0.191578</td>
    </tr>
    <tr>
      <th>14</th>
      <td>15</td>
      <td>45.181026</td>
      <td>6</td>
      <td>1.0</td>
      <td>1.068836</td>
      <td>0.952607</td>
      <td>0.922641</td>
      <td>0.941932</td>
      <td>0.926832</td>
      <td>0.153723</td>
      <td>0.153535</td>
      <td>0.500164</td>
      <td>0.946903</td>
      <td>0.909723</td>
      <td>0.164583</td>
      <td>0.959184</td>
      <td>0.937535</td>
      <td>0.141201</td>
    </tr>
  </tbody>
</table>
</div>


    
    ----------------------------------------------------------------------------------------------------------------------
    LOCAL per-client per-round metrics
    ----------------------------------------------------------------------------------------------------------------------



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
      <td>4</td>
      <td>race_edge_plus</td>
      <td>(g=1.02, a=0.32, b=6.00, t=2.60, k=3, eg=0.15,...</td>
      <td>1.02</td>
      <td>0.32</td>
      <td>6.0</td>
      <td>...</td>
      <td>0.931842</td>
      <td>0.828431</td>
      <td>0.937500</td>
      <td>0.974425</td>
      <td>0.987013</td>
      <td>0.533478</td>
      <td>0.275084</td>
      <td>0.191437</td>
      <td>1.403111</td>
      <td>0.739389</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>client_1</td>
      <td>ds1</td>
      <td>1</td>
      <td>5</td>
      <td>race_focus</td>
      <td>(g=1.10, a=0.36, b=6.40, t=2.70, k=3, eg=0.17,...</td>
      <td>1.10</td>
      <td>0.36</td>
      <td>6.4</td>
      <td>...</td>
      <td>0.989562</td>
      <td>0.972136</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.986111</td>
      <td>0.358194</td>
      <td>0.318540</td>
      <td>0.323267</td>
      <td>1.576462</td>
      <td>0.748131</td>
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
      <td>0.840945</td>
      <td>0.424242</td>
      <td>0.969697</td>
      <td>0.985714</td>
      <td>0.984127</td>
      <td>0.112677</td>
      <td>0.274428</td>
      <td>0.612895</td>
      <td>1.199979</td>
      <td>0.564839</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>client_3</td>
      <td>ds2</td>
      <td>1</td>
      <td>2</td>
      <td>race_robust</td>
      <td>(g=1.08, a=0.22, b=4.50, t=2.80, k=5, eg=0.11,...</td>
      <td>1.08</td>
      <td>0.22</td>
      <td>4.5</td>
      <td>...</td>
      <td>NaN</td>
      <td>0.950000</td>
      <td>0.939394</td>
      <td>NaN</td>
      <td>1.000000</td>
      <td>0.338466</td>
      <td>0.321586</td>
      <td>0.339949</td>
      <td>1.566741</td>
      <td>0.585317</td>
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
      <td>0.931555</td>
      <td>0.760870</td>
      <td>0.974510</td>
      <td>0.990842</td>
      <td>1.000000</td>
      <td>0.437954</td>
      <td>0.294646</td>
      <td>0.267400</td>
      <td>1.490459</td>
      <td>0.733662</td>
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
      <td>2</td>
      <td>race_texture</td>
      <td>(g=0.92, a=0.34, b=5.80, t=2.30, k=7, eg=0.10,...</td>
      <td>0.92</td>
      <td>0.34</td>
      <td>5.8</td>
      <td>...</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.623667</td>
      <td>0.217715</td>
      <td>0.158618</td>
      <td>1.252712</td>
      <td>0.961462</td>
    </tr>
    <tr>
      <th>86</th>
      <td>15</td>
      <td>client_2</td>
      <td>ds1</td>
      <td>1</td>
      <td>2</td>
      <td>race_texture</td>
      <td>(g=0.92, a=0.34, b=5.80, t=2.30, k=7, eg=0.10,...</td>
      <td>0.92</td>
      <td>0.34</td>
      <td>5.8</td>
      <td>...</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.719572</td>
      <td>0.136924</td>
      <td>0.143504</td>
      <td>1.012658</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>87</th>
      <td>15</td>
      <td>client_3</td>
      <td>ds2</td>
      <td>1</td>
      <td>0</td>
      <td>race_soft</td>
      <td>(g=0.95, a=0.18, b=3.80, t=2.20, k=3, eg=0.08,...</td>
      <td>0.95</td>
      <td>0.18</td>
      <td>3.8</td>
      <td>...</td>
      <td>NaN</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>NaN</td>
      <td>1.000000</td>
      <td>0.845531</td>
      <td>0.076776</td>
      <td>0.077693</td>
      <td>0.694521</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>88</th>
      <td>15</td>
      <td>client_4</td>
      <td>ds2</td>
      <td>1</td>
      <td>2</td>
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
      <td>0.732877</td>
      <td>0.196703</td>
      <td>0.070420</td>
      <td>0.962552</td>
      <td>0.977970</td>
    </tr>
    <tr>
      <th>89</th>
      <td>15</td>
      <td>client_5</td>
      <td>ds2</td>
      <td>1</td>
      <td>3</td>
      <td>race_smoothmix</td>
      <td>(g=0.90, a=0.20, b=4.00, t=2.10, k=7, eg=0.07,...</td>
      <td>0.90</td>
      <td>0.20</td>
      <td>4.0</td>
      <td>...</td>
      <td>0.985530</td>
      <td>0.980000</td>
      <td>0.962121</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.605211</td>
      <td>0.176666</td>
      <td>0.218123</td>
      <td>1.192485</td>
      <td>0.876873</td>
    </tr>
  </tbody>
</table>
<p>90 rows × 40 columns</p>
</div>


    
    ======================================================================================================================
    STEP 12: FINAL EVALUATION
    ======================================================================================================================
    
    ----------------------------------------------------------------------------------------------------------------------
    Best RL-selected preprocessing preset per client
    ----------------------------------------------------------------------------------------------------------------------



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
      <td>0.840730</td>
      <td>14</td>
    </tr>
    <tr>
      <th>1</th>
      <td>client_1</td>
      <td>ds1</td>
      <td>4</td>
      <td>race_edge_plus</td>
      <td>(g=1.02, a=0.32, b=6.00, t=2.60, k=3, eg=0.15,...</td>
      <td>0.856752</td>
      <td>14</td>
    </tr>
    <tr>
      <th>2</th>
      <td>client_2</td>
      <td>ds1</td>
      <td>0</td>
      <td>race_balanced</td>
      <td>(g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10,...</td>
      <td>0.973144</td>
      <td>14</td>
    </tr>
    <tr>
      <th>3</th>
      <td>client_3</td>
      <td>ds2</td>
      <td>1</td>
      <td>race_balanced</td>
      <td>(g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10,...</td>
      <td>0.914259</td>
      <td>14</td>
    </tr>
    <tr>
      <th>4</th>
      <td>client_4</td>
      <td>ds2</td>
      <td>1</td>
      <td>race_balanced</td>
      <td>(g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10,...</td>
      <td>0.841547</td>
      <td>14</td>
    </tr>
    <tr>
      <th>5</th>
      <td>client_5</td>
      <td>ds2</td>
      <td>0</td>
      <td>race_soft</td>
      <td>(g=0.95, a=0.18, b=3.80, t=2.20, k=3, eg=0.08,...</td>
      <td>0.835344</td>
      <td>14</td>
    </tr>
  </tbody>
</table>
</div>


    
    ----------------------------------------------------------------------------------------------------------------------
    Chosen final validation-based preprocessing strategy
    ----------------------------------------------------------------------------------------------------------------------



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
      <td>0.964061</td>
      <td>0.964602</td>
      <td>0.963881</td>
    </tr>
    <tr>
      <th>1</th>
      <td>ds2</td>
      <td>single</td>
      <td>['race_balanced']</td>
      <td>0.951583</td>
      <td>0.954315</td>
      <td>0.950672</td>
    </tr>
  </tbody>
</table>
</div>


    
    ======================================================================================================================
    STEP 12.5: EXTENDED METRICS + ERROR ANALYSIS + CALIBRATION
    ======================================================================================================================
    
    ----------------------------------------------------------------------------------------------------------------------
    Extended TEST metrics (DS1 vs DS2)
    ----------------------------------------------------------------------------------------------------------------------



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
      <td>0.973451</td>
      <td>0.973279</td>
      <td>0.974242</td>
      <td>0.973279</td>
      <td>0.973271</td>
      <td>0.974631</td>
      <td>0.973451</td>
      <td>0.973555</td>
      <td>0.080652</td>
      <td>...</td>
      <td>0.008806</td>
      <td>0.026721</td>
      <td>0.023841</td>
      <td>0.691786</td>
      <td>0.037425</td>
      <td>0.999447</td>
      <td>0.999055</td>
      <td>0.998937</td>
      <td>0.999797</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>ds2_test</td>
      <td>0.939086</td>
      <td>0.935998</td>
      <td>0.940686</td>
      <td>0.935998</td>
      <td>0.933455</td>
      <td>0.945805</td>
      <td>0.939086</td>
      <td>0.937907</td>
      <td>0.196728</td>
      <td>...</td>
      <td>0.019737</td>
      <td>0.064002</td>
      <td>0.039196</td>
      <td>0.824118</td>
      <td>0.096627</td>
      <td>0.994063</td>
      <td>0.999561</td>
      <td>0.985315</td>
      <td>1.000000</td>
      <td>0.991374</td>
    </tr>
  </tbody>
</table>
<p>2 rows × 26 columns</p>
</div>


    
    ----------------------------------------------------------------------------------------------------------------------
    Classwise metrics — DS1 TEST
    ----------------------------------------------------------------------------------------------------------------------



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
      <td>53</td>
      <td>0</td>
      <td>3</td>
      <td>170</td>
      <td>0.247788</td>
      <td>1.000000</td>
      <td>0.982659</td>
      <td>0.946429</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.053571</td>
      <td>0.946429</td>
      <td>0.973214</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>meningioma</td>
      <td>55</td>
      <td>53</td>
      <td>2</td>
      <td>2</td>
      <td>169</td>
      <td>0.243363</td>
      <td>0.963636</td>
      <td>0.988304</td>
      <td>0.963636</td>
      <td>0.988304</td>
      <td>0.011696</td>
      <td>0.036364</td>
      <td>0.929825</td>
      <td>0.975970</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>notumor</td>
      <td>59</td>
      <td>58</td>
      <td>0</td>
      <td>1</td>
      <td>167</td>
      <td>0.261062</td>
      <td>1.000000</td>
      <td>0.994048</td>
      <td>0.983051</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.016949</td>
      <td>0.983051</td>
      <td>0.991525</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>pituitary</td>
      <td>56</td>
      <td>56</td>
      <td>4</td>
      <td>0</td>
      <td>166</td>
      <td>0.247788</td>
      <td>0.933333</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.976471</td>
      <td>0.023529</td>
      <td>0.000000</td>
      <td>0.933333</td>
      <td>0.988235</td>
    </tr>
  </tbody>
</table>
</div>


    
    ----------------------------------------------------------------------------------------------------------------------
    Classwise metrics — DS2 TEST
    ----------------------------------------------------------------------------------------------------------------------



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
      <td>45</td>
      <td>45</td>
      <td>5</td>
      <td>0</td>
      <td>147</td>
      <td>0.228426</td>
      <td>0.900000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.967105</td>
      <td>0.032895</td>
      <td>0.000000</td>
      <td>0.900000</td>
      <td>0.983553</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>meningioma</td>
      <td>46</td>
      <td>36</td>
      <td>0</td>
      <td>10</td>
      <td>151</td>
      <td>0.233503</td>
      <td>1.000000</td>
      <td>0.937888</td>
      <td>0.782609</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.217391</td>
      <td>0.782609</td>
      <td>0.891304</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>notumor</td>
      <td>61</td>
      <td>60</td>
      <td>0</td>
      <td>1</td>
      <td>136</td>
      <td>0.309645</td>
      <td>1.000000</td>
      <td>0.992701</td>
      <td>0.983607</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.016393</td>
      <td>0.983607</td>
      <td>0.991803</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>pituitary</td>
      <td>45</td>
      <td>44</td>
      <td>7</td>
      <td>1</td>
      <td>145</td>
      <td>0.228426</td>
      <td>0.862745</td>
      <td>0.993151</td>
      <td>0.977778</td>
      <td>0.953947</td>
      <td>0.046053</td>
      <td>0.022222</td>
      <td>0.846154</td>
      <td>0.965863</td>
    </tr>
  </tbody>
</table>
</div>


    
    ----------------------------------------------------------------------------------------------------------------------
    Top confusion pairs — DS1 TEST
    ----------------------------------------------------------------------------------------------------------------------



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
      <td>2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>meningioma</td>
      <td>pituitary</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>notumor</td>
      <td>pituitary</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>glioma</td>
      <td>pituitary</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>meningioma</td>
      <td>glioma</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>glioma</td>
      <td>notumor</td>
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
      <td>meningioma</td>
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
      <td>pituitary</td>
      <td>glioma</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>


    
    ----------------------------------------------------------------------------------------------------------------------
    Top confusion pairs — DS2 TEST
    ----------------------------------------------------------------------------------------------------------------------



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
      <td>meningioma</td>
      <td>pituitary</td>
      <td>6</td>
    </tr>
    <tr>
      <th>1</th>
      <td>meningioma</td>
      <td>glioma</td>
      <td>4</td>
    </tr>
    <tr>
      <th>2</th>
      <td>notumor</td>
      <td>pituitary</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>pituitary</td>
      <td>glioma</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>glioma</td>
      <td>pituitary</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>glioma</td>
      <td>notumor</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>glioma</td>
      <td>meningioma</td>
      <td>0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>meningioma</td>
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


    
    ----------------------------------------------------------------------------------------------------------------------
    Calibration bins — DS1 TEST
    ----------------------------------------------------------------------------------------------------------------------



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
      <td>0.308214</td>
      <td>1.000000</td>
      <td>0.691786</td>
      <td>1</td>
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
      <td>0.450341</td>
      <td>0.000000</td>
      <td>0.450341</td>
      <td>1</td>
    </tr>
    <tr>
      <th>6</th>
      <td>6</td>
      <td>0.500000</td>
      <td>0.583333</td>
      <td>0.551286</td>
      <td>0.500000</td>
      <td>0.051286</td>
      <td>2</td>
    </tr>
    <tr>
      <th>7</th>
      <td>7</td>
      <td>0.583333</td>
      <td>0.666667</td>
      <td>0.624376</td>
      <td>0.000000</td>
      <td>0.624376</td>
      <td>1</td>
    </tr>
    <tr>
      <th>8</th>
      <td>8</td>
      <td>0.666667</td>
      <td>0.750000</td>
      <td>0.670888</td>
      <td>0.500000</td>
      <td>0.170888</td>
      <td>2</td>
    </tr>
    <tr>
      <th>9</th>
      <td>9</td>
      <td>0.750000</td>
      <td>0.833333</td>
      <td>0.792681</td>
      <td>0.500000</td>
      <td>0.292681</td>
      <td>2</td>
    </tr>
    <tr>
      <th>10</th>
      <td>10</td>
      <td>0.833333</td>
      <td>0.916667</td>
      <td>0.875870</td>
      <td>1.000000</td>
      <td>0.124130</td>
      <td>4</td>
    </tr>
    <tr>
      <th>11</th>
      <td>11</td>
      <td>0.916667</td>
      <td>1.000000</td>
      <td>0.985467</td>
      <td>0.995305</td>
      <td>0.009838</td>
      <td>213</td>
    </tr>
  </tbody>
</table>
</div>


    
    ----------------------------------------------------------------------------------------------------------------------
    Calibration bins — DS2 TEST
    ----------------------------------------------------------------------------------------------------------------------



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
      <td>0.345284</td>
      <td>0.000000</td>
      <td>0.345284</td>
      <td>1</td>
    </tr>
    <tr>
      <th>5</th>
      <td>5</td>
      <td>0.416667</td>
      <td>0.500000</td>
      <td>0.462842</td>
      <td>0.500000</td>
      <td>0.037158</td>
      <td>2</td>
    </tr>
    <tr>
      <th>6</th>
      <td>6</td>
      <td>0.500000</td>
      <td>0.583333</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>7</td>
      <td>0.583333</td>
      <td>0.666667</td>
      <td>0.651897</td>
      <td>0.000000</td>
      <td>0.651897</td>
      <td>2</td>
    </tr>
    <tr>
      <th>8</th>
      <td>8</td>
      <td>0.666667</td>
      <td>0.750000</td>
      <td>0.717467</td>
      <td>1.000000</td>
      <td>0.282533</td>
      <td>5</td>
    </tr>
    <tr>
      <th>9</th>
      <td>9</td>
      <td>0.750000</td>
      <td>0.833333</td>
      <td>0.824118</td>
      <td>0.000000</td>
      <td>0.824118</td>
      <td>1</td>
    </tr>
    <tr>
      <th>10</th>
      <td>10</td>
      <td>0.833333</td>
      <td>0.916667</td>
      <td>0.869442</td>
      <td>0.428571</td>
      <td>0.440871</td>
      <td>7</td>
    </tr>
    <tr>
      <th>11</th>
      <td>11</td>
      <td>0.916667</td>
      <td>1.000000</td>
      <td>0.987013</td>
      <td>0.983240</td>
      <td>0.003772</td>
      <td>179</td>
    </tr>
  </tbody>
</table>
</div>


    
    ----------------------------------------------------------------------------------------------------------------------
    VAL + TEST tables (federated, per-dataset + global equal)
    ----------------------------------------------------------------------------------------------------------------------



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
      <td>ARCF-Net + FedAvg + FedProx</td>
      <td>VAL</td>
      <td>ds1</td>
      <td>0.964602</td>
      <td>0.967310</td>
      <td>0.963294</td>
      <td>0.963881</td>
      <td>0.967599</td>
      <td>0.964602</td>
      <td>0.964721</td>
      <td>...</td>
      <td>4.718545</td>
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
      <td>ARCF-Net + FedAvg + FedProx</td>
      <td>VAL</td>
      <td>ds2</td>
      <td>0.954315</td>
      <td>0.951935</td>
      <td>0.952303</td>
      <td>0.950672</td>
      <td>0.955835</td>
      <td>0.954315</td>
      <td>0.953733</td>
      <td>...</td>
      <td>4.103737</td>
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
      <td>ARCF-Net + FedAvg + FedProx</td>
      <td>VAL</td>
      <td>global_equal</td>
      <td>0.959458</td>
      <td>0.959623</td>
      <td>0.957798</td>
      <td>0.957276</td>
      <td>0.961717</td>
      <td>0.959458</td>
      <td>0.959227</td>
      <td>...</td>
      <td>4.411141</td>
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
      <td>ARCF-Net + FedAvg + FedProx</td>
      <td>TEST</td>
      <td>ds1</td>
      <td>0.973451</td>
      <td>0.974242</td>
      <td>0.973279</td>
      <td>0.973271</td>
      <td>0.974631</td>
      <td>0.973451</td>
      <td>0.973555</td>
      <td>...</td>
      <td>4.835295</td>
      <td>0.973279</td>
      <td>0.964924</td>
      <td>0.964596</td>
      <td>0.974242</td>
      <td>0.991253</td>
      <td>0.991194</td>
      <td>0.023841</td>
      <td>0.691786</td>
      <td>0.037425</td>
    </tr>
    <tr>
      <th>4</th>
      <td>ARCF-Net + FedAvg + FedProx</td>
      <td>TEST</td>
      <td>ds2</td>
      <td>0.939086</td>
      <td>0.940686</td>
      <td>0.935998</td>
      <td>0.933455</td>
      <td>0.945805</td>
      <td>0.939086</td>
      <td>0.937907</td>
      <td>...</td>
      <td>4.188238</td>
      <td>0.935998</td>
      <td>0.920915</td>
      <td>0.918336</td>
      <td>0.940686</td>
      <td>0.980935</td>
      <td>0.980263</td>
      <td>0.039196</td>
      <td>0.824118</td>
      <td>0.096627</td>
    </tr>
    <tr>
      <th>5</th>
      <td>ARCF-Net + FedAvg + FedProx</td>
      <td>TEST</td>
      <td>global_equal</td>
      <td>0.956269</td>
      <td>0.957464</td>
      <td>0.954639</td>
      <td>0.953363</td>
      <td>0.960218</td>
      <td>0.956269</td>
      <td>0.955731</td>
      <td>...</td>
      <td>4.511766</td>
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


    
    Paper selection summary:
    - Best round: 14 | best_reward=1.0852
    - DS1 final strategy: single | names=['race_edge_plus']
    - DS2 final strategy: single | names=['race_balanced']
    
    ======================================================================================================================
    STEP 13: PREPROCESSING VALIDATION (DS1 + DS2 VAL SAMPLE)
    ======================================================================================================================
    
    ----------------------------------------------------------------------------------------------------------------------
    Preprocessing validation summary (DS1 VAL sample)
    ----------------------------------------------------------------------------------------------------------------------



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
      <td>0.041596</td>
      <td>0.021720</td>
      <td>0.014947</td>
      <td>0.165471</td>
    </tr>
    <tr>
      <th>1</th>
      <td>edge_energy_after</td>
      <td>0.159103</td>
      <td>0.033185</td>
      <td>0.093724</td>
      <td>0.285806</td>
    </tr>
    <tr>
      <th>2</th>
      <td>entropy_before</td>
      <td>5.820408</td>
      <td>0.628255</td>
      <td>3.547314</td>
      <td>7.239670</td>
    </tr>
    <tr>
      <th>3</th>
      <td>entropy_after</td>
      <td>6.559343</td>
      <td>0.597442</td>
      <td>3.875062</td>
      <td>7.564398</td>
    </tr>
    <tr>
      <th>4</th>
      <td>contrast_before</td>
      <td>0.187640</td>
      <td>0.052597</td>
      <td>0.101468</td>
      <td>0.363759</td>
    </tr>
    <tr>
      <th>5</th>
      <td>contrast_after</td>
      <td>0.238172</td>
      <td>0.021438</td>
      <td>0.204715</td>
      <td>0.334330</td>
    </tr>
    <tr>
      <th>6</th>
      <td>edge_gain_ratio</td>
      <td>4.217192</td>
      <td>1.039354</td>
      <td>1.458519</td>
      <td>7.844800</td>
    </tr>
    <tr>
      <th>7</th>
      <td>entropy_delta</td>
      <td>0.738936</td>
      <td>0.257089</td>
      <td>0.257075</td>
      <td>1.537737</td>
    </tr>
    <tr>
      <th>8</th>
      <td>contrast_delta</td>
      <td>0.050532</td>
      <td>0.034326</td>
      <td>-0.032019</td>
      <td>0.117838</td>
    </tr>
  </tbody>
</table>
</div>


    
    ----------------------------------------------------------------------------------------------------------------------
    Preprocessing validation summary (DS2 VAL sample)
    ----------------------------------------------------------------------------------------------------------------------



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
      <td>0.041718</td>
      <td>0.015947</td>
      <td>0.012571</td>
      <td>0.121283</td>
    </tr>
    <tr>
      <th>1</th>
      <td>edge_energy_after</td>
      <td>0.116392</td>
      <td>0.023684</td>
      <td>0.043676</td>
      <td>0.167522</td>
    </tr>
    <tr>
      <th>2</th>
      <td>entropy_before</td>
      <td>5.811255</td>
      <td>0.822915</td>
      <td>3.222212</td>
      <td>7.570240</td>
    </tr>
    <tr>
      <th>3</th>
      <td>entropy_after</td>
      <td>6.664449</td>
      <td>0.785474</td>
      <td>3.612907</td>
      <td>7.741302</td>
    </tr>
    <tr>
      <th>4</th>
      <td>contrast_before</td>
      <td>0.190753</td>
      <td>0.055869</td>
      <td>0.098556</td>
      <td>0.341468</td>
    </tr>
    <tr>
      <th>5</th>
      <td>contrast_after</td>
      <td>0.250066</td>
      <td>0.022336</td>
      <td>0.198854</td>
      <td>0.318548</td>
    </tr>
    <tr>
      <th>6</th>
      <td>edge_gain_ratio</td>
      <td>3.015276</td>
      <td>0.781821</td>
      <td>1.308311</td>
      <td>5.732603</td>
    </tr>
    <tr>
      <th>7</th>
      <td>entropy_delta</td>
      <td>0.853194</td>
      <td>0.296688</td>
      <td>0.171062</td>
      <td>1.614185</td>
    </tr>
    <tr>
      <th>8</th>
      <td>contrast_delta</td>
      <td>0.059313</td>
      <td>0.037080</td>
      <td>-0.022945</td>
      <td>0.130606</td>
    </tr>
  </tbody>
</table>
</div>


    
    ======================================================================================================================
    STEP 14: PARAMETER EVOLUTION TABLE
    ======================================================================================================================
    
    ----------------------------------------------------------------------------------------------------------------------
    Mean selected preprocessing parameters over rounds
    ----------------------------------------------------------------------------------------------------------------------



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
      <td>1.046667</td>
      <td>0.280000</td>
      <td>5.300000</td>
      <td>2.616667</td>
      <td>4.333333</td>
      <td>0.130000</td>
      <td>0.796667</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1.008333</td>
      <td>0.243333</td>
      <td>4.800000</td>
      <td>2.516667</td>
      <td>4.333333</td>
      <td>0.111667</td>
      <td>0.770000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>0.973333</td>
      <td>0.273333</td>
      <td>5.033333</td>
      <td>2.350000</td>
      <td>4.666667</td>
      <td>0.105000</td>
      <td>0.780000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>1.018333</td>
      <td>0.246667</td>
      <td>4.650000</td>
      <td>2.466667</td>
      <td>4.666667</td>
      <td>0.105000</td>
      <td>0.780000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0.978333</td>
      <td>0.223333</td>
      <td>4.350000</td>
      <td>2.350000</td>
      <td>5.333333</td>
      <td>0.091667</td>
      <td>0.746667</td>
    </tr>
    <tr>
      <th>5</th>
      <td>6</td>
      <td>1.000000</td>
      <td>0.253333</td>
      <td>4.783333</td>
      <td>2.450000</td>
      <td>5.333333</td>
      <td>0.101667</td>
      <td>0.780000</td>
    </tr>
    <tr>
      <th>6</th>
      <td>7</td>
      <td>1.000000</td>
      <td>0.230000</td>
      <td>4.466667</td>
      <td>2.450000</td>
      <td>5.666667</td>
      <td>0.096667</td>
      <td>0.753333</td>
    </tr>
    <tr>
      <th>7</th>
      <td>8</td>
      <td>1.033333</td>
      <td>0.283333</td>
      <td>5.300000</td>
      <td>2.600000</td>
      <td>4.666667</td>
      <td>0.123333</td>
      <td>0.800000</td>
    </tr>
    <tr>
      <th>8</th>
      <td>9</td>
      <td>0.991667</td>
      <td>0.243333</td>
      <td>4.666667</td>
      <td>2.366667</td>
      <td>4.666667</td>
      <td>0.103333</td>
      <td>0.770000</td>
    </tr>
    <tr>
      <th>9</th>
      <td>10</td>
      <td>0.995000</td>
      <td>0.243333</td>
      <td>4.633333</td>
      <td>2.450000</td>
      <td>5.333333</td>
      <td>0.098333</td>
      <td>0.760000</td>
    </tr>
    <tr>
      <th>10</th>
      <td>11</td>
      <td>1.011667</td>
      <td>0.286667</td>
      <td>5.266667</td>
      <td>2.450000</td>
      <td>4.333333</td>
      <td>0.120000</td>
      <td>0.800000</td>
    </tr>
    <tr>
      <th>11</th>
      <td>12</td>
      <td>1.008333</td>
      <td>0.243333</td>
      <td>4.800000</td>
      <td>2.516667</td>
      <td>4.333333</td>
      <td>0.111667</td>
      <td>0.770000</td>
    </tr>
    <tr>
      <th>12</th>
      <td>13</td>
      <td>1.011667</td>
      <td>0.253333</td>
      <td>4.916667</td>
      <td>2.500000</td>
      <td>4.000000</td>
      <td>0.115000</td>
      <td>0.786667</td>
    </tr>
    <tr>
      <th>13</th>
      <td>14</td>
      <td>1.028333</td>
      <td>0.256667</td>
      <td>4.850000</td>
      <td>2.500000</td>
      <td>4.333333</td>
      <td>0.113333</td>
      <td>0.786667</td>
    </tr>
    <tr>
      <th>14</th>
      <td>15</td>
      <td>0.968333</td>
      <td>0.263333</td>
      <td>4.850000</td>
      <td>2.366667</td>
      <td>5.666667</td>
      <td>0.096667</td>
      <td>0.766667</td>
    </tr>
  </tbody>
</table>
</div>


    
    ======================================================================================================================
    STEP 15: SAVING ONLY TWO FILES (CHECKPOINT + ONE CSV)
    ======================================================================================================================
    ✅ Saved checkpoint: /kaggle/working/outputs/ARCFNet_RESNET50_FEDAVG_FEDPROX_checkpoint.pth
    ✅ Saved CSV (ALL outputs): /kaggle/working/outputs/ALL_OUTPUTS_AND_METRICS_RESNET50_FEDAVG_FEDPROX.csv
    
    DONE ✅
    Method: ARCF-Net = Adaptive RACE-FELCM with CRAF Fusion Network
    Backbone: Residual Network-50
    Federated variant: FedAvg + FedProx
    Best round: 14
    Adaptive clients => DS1=3, DS2=3, TOTAL=6
    Rounds completed: 15
    Global TEST acc: 0.9563
    Global TEST f1_macro: 0.9534
    DS1 TEST acc: 0.9735
    DS2 TEST acc: 0.9391
    DS1 final strategy: single | names=['race_edge_plus']
    DS2 final strategy: single | names=['race_balanced']


**3. FedAvg + prototype sharing**


```python
# ============================================================
# KAGGLE FULL SCRIPT
# TRUE FL + RL-UCB + RACE-FELCM + CRAF + ResNet-50
# ABLATION: FEDAVG + PROTOTYPE SHARING
# METHOD ACRONYM: ARCF-Net
# FULL FORM: Adaptive RACE-FELCM with CRAF Fusion Network
# ------------------------------------------------------------
# KAGGLE-READY + SYMMETRIC DS1/DS2 IMPORTANCE
# - Uses BOTH datasets
# - Reads datasets from /kaggle/input automatically
# - Exact 15 FL rounds
# - Proper FL with FedAvg + prototype sharing
# - NO FedProx
# - RL-UCB for SHARED client-count planning + per-client preprocessing preset selection
# - Tune-aware theta probing before local training
# - Equal DS1 / DS2 importance in best-round selection and merged reporting
# - No plots
# - Saves checkpoint WITH full-process info for later replay / XAI
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

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    log_loss, confusion_matrix, roc_auc_score,
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

print("=" * 118)
print("TRUE FL + RL-UCB + RACE-FELCM + CRAF + ResNet-50")
print("METHOD: ARCF-Net (Adaptive RACE-FELCM with CRAF Fusion Network)")
print("ABLATION: FedAvg + prototype sharing")
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

    # misc / tables
    "quick_hash_subset_per_split": 300,
    "preproc_val_sample_n": 400,
    "calibration_bins": 12,
}

OUTDIR = "/kaggle/working/outputs" if IS_KAGGLE else "/content/outputs"
os.makedirs(OUTDIR, exist_ok=True)
MODEL_PATH = os.path.join(OUTDIR, "ARCFNet_RESNET50_FEDAVG_PROTOTYPE_checkpoint.pth")
CSV_PATH   = os.path.join(OUTDIR, "ALL_OUTPUTS_AND_METRICS_RESNET50_FEDAVG_PROTOTYPE.csv")

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
    "federated_variant": "FedAvg + prototype sharing",
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
print("STEP 6: DATA LOADERS")
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

def train_one_epoch(model, loader, optimizer, preproc, theta, global_prototypes=None, scheduler=None):
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
print(f"Prototype λ={CFG['proto_lambda']}")
print("NO FedProx")
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
    {"setting": "ARCF-Net + FedAvg + prototype sharing", "split": "VAL",  "dataset": "ds1",          **compact_metrics(val_ds1)},
    {"setting": "ARCF-Net + FedAvg + prototype sharing", "split": "VAL",  "dataset": "ds2",          **compact_metrics(val_ds2)},
    {"setting": "ARCF-Net + FedAvg + prototype sharing", "split": "VAL",  "dataset": "global_equal", **compact_metrics(val_global)},
    {"setting": "ARCF-Net + FedAvg + prototype sharing", "split": "TEST", "dataset": "ds1",          **compact_metrics(test_ds1)},
    {"setting": "ARCF-Net + FedAvg + prototype sharing", "split": "TEST", "dataset": "ds2",          **compact_metrics(test_ds2)},
    {"setting": "ARCF-Net + FedAvg + prototype sharing", "split": "TEST", "dataset": "global_equal", **compact_metrics(test_global)},
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
# STEP 14: PARAMETER EVOLUTION TABLE
# ============================================================
print("\n" + "=" * 118)
print("STEP 14: PARAMETER EVOLUTION TABLE")
print("=" * 118)

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
for ccol in theta_cols:
    if ccol in loc_copy.columns:
        loc_copy[ccol] = pd.to_numeric(loc_copy[ccol], errors="coerce")

theta_evo = loc_copy.groupby("round")[theta_cols].mean(numeric_only=True).reset_index()
print_table(theta_evo, "Mean selected preprocessing parameters over rounds")
add_table_to_csv(theta_evo, "theta_evolution_mean")

# ============================================================
# STEP 15: SAVE CHECKPOINT + CSV
# ============================================================
print("\n" + "=" * 118)
print("STEP 15: SAVING ONLY TWO FILES (CHECKPOINT + ONE CSV)")
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
    "federated_variant": "FedAvg + prototype sharing",

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
    "theta_evolution_mean": theta_evo.to_dict(orient="list"),
}

torch.save(checkpoint, MODEL_PATH)
print(f"✅ Saved checkpoint: {MODEL_PATH}")

all_df.to_csv(CSV_PATH, index=False)
print(f"✅ Saved CSV (ALL outputs): {CSV_PATH}")

print("\nDONE ✅")
print(f"Method: {METHOD_INFO['acronym']} = {METHOD_INFO['full_form']}")
print(f"Backbone: {METHOD_INFO['backbone_full_form']}")
print("Federated variant: FedAvg + prototype sharing")
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
    ABLATION: FedAvg + prototype sharing
    ======================================================================================================================
    ENV: KAGGLE | DEVICE: cuda | torch=2.9.0+cu126
    ======================================================================================================================
    
    ======================================================================================================================
    STEP 0: ACCESS DATASETS
    ======================================================================================================================
    Dataset-1 RAW root detected:
      /kaggle/input/datasets/orvile/pmram-bangladeshi-brain-cancer-mri-dataset/PMRAM Bangladeshi Brain Cancer - MRI Dataset/PMRAM Bangladeshi Brain Cancer - MRI Dataset/Raw Data/Raw
    Dataset-2 root detected:
      /kaggle/input/datasets/chubskuy/brain-tumor-image/Testing
    
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
    ds2: glioma -> glioma | 300 images
    ds2: meningioma -> meningioma | 306 images
    ds2: notumor -> notumor | 405 images
    ds2: pituitary -> pituitary | 300 images
    
    Dataset-1 images: 1505
    label
    glioma        373
    meningioma    363
    notumor       396
    pituitary     373
    Name: count, dtype: int64
    
    Dataset-2 images: 1311
    label
    glioma        300
    meningioma    306
    notumor       405
    pituitary     300
    Name: count, dtype: int64
    
    ======================================================================================================================
    STEP 2: TRAIN / VAL / TEST SPLIT (PER DATASET)
    ======================================================================================================================
    DS1 TRAIN: 1053 | VAL: 226 | TEST: 226
    DS2 TRAIN: 917 | VAL: 197 | TEST: 197
    
    ======================================================================================================================
    STEP 2.5: SANITY / LEAKAGE CHECKS
    ======================================================================================================================
    
    ----------------------------------------------------------------------------------------------------------------------
    Leakage / Sanity Summary — ds1
    ----------------------------------------------------------------------------------------------------------------------



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
      <td>5</td>
      <td>5</td>
      <td>6</td>
      <td>298</td>
      <td>222</td>
      <td>224</td>
    </tr>
  </tbody>
</table>
</div>


    
    ----------------------------------------------------------------------------------------------------------------------
    Leakage / Sanity Summary — ds2
    ----------------------------------------------------------------------------------------------------------------------



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
      <td>917</td>
      <td>197</td>
      <td>197</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>300</td>
      <td>194</td>
      <td>197</td>
    </tr>
  </tbody>
</table>
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
      <td>0.755612</td>
      <td>0.767340</td>
      <td>0.761476</td>
      <td>0.761476</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>4</td>
      <td>0.679705</td>
      <td>0.635628</td>
      <td>0.657667</td>
      <td>0.657667</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>5</td>
      <td>0.721049</td>
      <td>0.775174</td>
      <td>0.748111</td>
      <td>0.748111</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>3</td>
      <td>0.841039</td>
      <td>0.726403</td>
      <td>0.783721</td>
      <td>0.772599</td>
      <td>2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>5</td>
      <td>0.688321</td>
      <td>0.701388</td>
      <td>0.694855</td>
      <td>0.721483</td>
      <td>2</td>
    </tr>
    <tr>
      <th>5</th>
      <td>6</td>
      <td>4</td>
      <td>0.665267</td>
      <td>0.691320</td>
      <td>0.678293</td>
      <td>0.667980</td>
      <td>2</td>
    </tr>
    <tr>
      <th>6</th>
      <td>7</td>
      <td>3</td>
      <td>0.766538</td>
      <td>0.693916</td>
      <td>0.730227</td>
      <td>0.758475</td>
      <td>3</td>
    </tr>
    <tr>
      <th>7</th>
      <td>8</td>
      <td>5</td>
      <td>0.677792</td>
      <td>0.707763</td>
      <td>0.692778</td>
      <td>0.711915</td>
      <td>3</td>
    </tr>
    <tr>
      <th>8</th>
      <td>9</td>
      <td>4</td>
      <td>0.715564</td>
      <td>0.681741</td>
      <td>0.698653</td>
      <td>0.678204</td>
      <td>3</td>
    </tr>
    <tr>
      <th>9</th>
      <td>10</td>
      <td>3</td>
      <td>0.734619</td>
      <td>0.707902</td>
      <td>0.721261</td>
      <td>0.749171</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>


    
    ======================================================================================================================
    STEP 5: FINAL NON-IID CLIENT PARTITIONING
    ======================================================================================================================
    
    ======================================================================================================================
    STEP 6: DATA LOADERS
    ======================================================================================================================
    ds1 | client_0 | train=286 | tune=45 | val=40
    ds1 | client_1 | train=257 | tune=41 | val=36
    ds1 | client_2 | train=269 | tune=42 | val=37
    ds2 | client_3 | train=100 | tune=16 | val=14
    ds2 | client_4 | train=342 | tune=54 | val=47
    ds2 | client_5 | train=265 | tune=42 | val=37
    
    ----------------------------------------------------------------------------------------------------------------------
    Adaptive client class distribution
    ----------------------------------------------------------------------------------------------------------------------



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
      <td>286</td>
      <td>45</td>
      <td>40</td>
      <td>40</td>
      <td>33</td>
      <td>166</td>
      <td>47</td>
    </tr>
    <tr>
      <th>1</th>
      <td>client_1</td>
      <td>ds1</td>
      <td>257</td>
      <td>41</td>
      <td>36</td>
      <td>135</td>
      <td>4</td>
      <td>31</td>
      <td>87</td>
    </tr>
    <tr>
      <th>2</th>
      <td>client_2</td>
      <td>ds1</td>
      <td>269</td>
      <td>42</td>
      <td>37</td>
      <td>25</td>
      <td>159</td>
      <td>17</td>
      <td>68</td>
    </tr>
    <tr>
      <th>3</th>
      <td>client_3</td>
      <td>ds2</td>
      <td>100</td>
      <td>16</td>
      <td>14</td>
      <td>74</td>
      <td>18</td>
      <td>4</td>
      <td>4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>client_4</td>
      <td>ds2</td>
      <td>342</td>
      <td>54</td>
      <td>47</td>
      <td>5</td>
      <td>121</td>
      <td>154</td>
      <td>62</td>
    </tr>
    <tr>
      <th>5</th>
      <td>client_5</td>
      <td>ds2</td>
      <td>265</td>
      <td>42</td>
      <td>37</td>
      <td>82</td>
      <td>25</td>
      <td>61</td>
      <td>97</td>
    </tr>
  </tbody>
</table>
</div>


    Augmentation: ON ✅
    Preprocessing: ON ✅
    Total adaptive clients: 6
    
    ======================================================================================================================
    STEP 7: NOVEL PREPROCESSING — RACE-FELCM
    ======================================================================================================================
    
    ======================================================================================================================
    STEP 8: MODEL — ResNet-50 + CRAF Fusion
    ======================================================================================================================
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
    Prototype λ=0.12
    NO FedProx
    Tempered FedAvg exponent = 0.50
    Best-round masses => DS1=0.50, DS2=0.50, min-bonus=0.15
    
    ======================================================================================================================
    ROUND 1/15 | selected=[0, 1, 2, 3, 4, 5]
    ======================================================================================================================
    Client 0 (ds1) | train_acc=0.6941 | val_acc=0.8250 | val_f1=0.7109 | val_auc=0.9325 | reward=0.7394 | theta=race_edge_plus (g=1.02, a=0.32, b=6.00, t=2.60, k=3, eg=0.15, mix=0.84)
    Client 1 (ds1) | train_acc=0.7335 | val_acc=0.7778 | val_f1=0.6097 | val_auc=0.9577 | reward=0.6517 | theta=race_focus (g=1.10, a=0.36, b=6.40, t=2.70, k=3, eg=0.17, mix=0.86)
    Client 2 (ds1) | train_acc=0.6859 | val_acc=0.7568 | val_f1=0.7107 | val_auc=0.9157 | reward=0.7222 | theta=race_focus (g=1.10, a=0.36, b=6.40, t=2.70, k=3, eg=0.17, mix=0.86)
    Client 3 (ds2) | train_acc=0.5550 | val_acc=0.9286 | val_f1=0.6508 | val_auc=nan | reward=0.7202 | theta=race_robust (g=1.08, a=0.22, b=4.50, t=2.80, k=5, eg=0.11, mix=0.76)
    Client 4 (ds2) | train_acc=0.6871 | val_acc=0.8298 | val_f1=0.6354 | val_auc=0.9493 | reward=0.6840 | theta=race_smoothmix (g=0.90, a=0.20, b=4.00, t=2.10, k=7, eg=0.07, mix=0.70)
    Client 5 (ds2) | train_acc=0.6358 | val_acc=0.7568 | val_f1=0.6264 | val_auc=0.9462 | reward=0.6590 | theta=race_robust (g=1.08, a=0.22, b=4.50, t=2.80, k=5, eg=0.11, mix=0.76)
    
    ----------------------------------------------------------------------------------------------------------------------
    GLOBAL VAL (Round 1) | global_acc=0.8009 | global_f1=0.6580 | ds1_acc=0.7876 | ds1_f1=0.6786 | ds2_acc=0.8163 | ds2_f1=0.6342 | reward=0.7948 | round_time=56.1s
    ----------------------------------------------------------------------------------------------------------------------
    
    ======================================================================================================================
    ROUND 2/15 | selected=[0, 1, 2, 3, 4, 5]
    ======================================================================================================================
    Client 0 (ds1) | train_acc=0.8619 | val_acc=0.8500 | val_f1=0.7494 | val_auc=0.9744 | reward=0.7746 | theta=race_robust (g=1.08, a=0.22, b=4.50, t=2.80, k=5, eg=0.11, mix=0.76)
    Client 1 (ds1) | train_acc=0.8852 | val_acc=0.8611 | val_f1=0.6451 | val_auc=0.9685 | reward=0.6991 | theta=race_edge_plus (g=1.02, a=0.32, b=6.00, t=2.60, k=3, eg=0.15, mix=0.84)
    Client 2 (ds1) | train_acc=0.8587 | val_acc=0.8649 | val_f1=0.7880 | val_auc=0.9630 | reward=0.8072 | theta=race_edge_plus (g=1.02, a=0.32, b=6.00, t=2.60, k=3, eg=0.15, mix=0.84)
    Client 3 (ds2) | train_acc=0.7150 | val_acc=0.8571 | val_f1=0.5397 | val_auc=nan | reward=0.6190 | theta=race_soft (g=0.95, a=0.18, b=3.80, t=2.20, k=3, eg=0.08, mix=0.72)
    Client 4 (ds2) | train_acc=0.8056 | val_acc=0.8723 | val_f1=0.6656 | val_auc=0.9586 | reward=0.7173 | theta=race_robust (g=1.08, a=0.22, b=4.50, t=2.80, k=5, eg=0.11, mix=0.76)
    Client 5 (ds2) | train_acc=0.7906 | val_acc=0.7838 | val_f1=0.6227 | val_auc=0.9479 | reward=0.6630 | theta=race_smoothmix (g=0.90, a=0.20, b=4.00, t=2.10, k=7, eg=0.07, mix=0.70)
    
    ----------------------------------------------------------------------------------------------------------------------
    GLOBAL VAL (Round 2) | global_acc=0.8483 | global_f1=0.6836 | ds1_acc=0.8584 | ds1_f1=0.7288 | ds2_acc=0.8367 | ds2_f1=0.6314 | reward=0.8244 | round_time=53.1s
    ----------------------------------------------------------------------------------------------------------------------
    
    ======================================================================================================================
    ROUND 3/15 | selected=[0, 1, 2, 3, 4, 5]
    ======================================================================================================================
    Client 0 (ds1) | train_acc=0.8007 | val_acc=0.7750 | val_f1=0.5699 | val_auc=0.9399 | reward=0.6212 | theta=race_focus (g=1.10, a=0.36, b=6.40, t=2.70, k=3, eg=0.17, mix=0.86)
    Client 1 (ds1) | train_acc=0.8677 | val_acc=0.9444 | val_f1=0.7268 | val_auc=0.9479 | reward=0.7812 | theta=race_texture (g=0.92, a=0.34, b=5.80, t=2.30, k=7, eg=0.10, mix=0.80)
    Client 2 (ds1) | train_acc=0.8290 | val_acc=0.9459 | val_f1=0.8701 | val_auc=0.9949 | reward=0.8890 | theta=race_texture (g=0.92, a=0.34, b=5.80, t=2.30, k=7, eg=0.10, mix=0.80)
    Client 3 (ds2) | train_acc=0.7900 | val_acc=0.8571 | val_f1=0.8030 | val_auc=nan | reward=0.8166 | theta=race_robust (g=1.08, a=0.22, b=4.50, t=2.80, k=5, eg=0.11, mix=0.76)
    Client 4 (ds2) | train_acc=0.8392 | val_acc=0.9362 | val_f1=0.6912 | val_auc=0.9788 | reward=0.7524 | theta=race_robust (g=1.08, a=0.22, b=4.50, t=2.80, k=5, eg=0.11, mix=0.76)
    Client 5 (ds2) | train_acc=0.8509 | val_acc=0.8108 | val_f1=0.7504 | val_auc=0.9681 | reward=0.7655 | theta=race_soft (g=0.95, a=0.18, b=3.80, t=2.20, k=3, eg=0.08, mix=0.72)
    
    ----------------------------------------------------------------------------------------------------------------------
    GLOBAL VAL (Round 3) | global_acc=0.8815 | global_f1=0.7234 | ds1_acc=0.8850 | ds1_f1=0.7182 | ds2_acc=0.8776 | ds2_f1=0.7295 | reward=0.8772 | round_time=57.7s
    ----------------------------------------------------------------------------------------------------------------------
    
    ======================================================================================================================
    ROUND 4/15 | selected=[0, 1, 2, 3, 4, 5]
    ======================================================================================================================
    Client 0 (ds1) | train_acc=0.8636 | val_acc=0.8000 | val_f1=0.6339 | val_auc=0.9691 | reward=0.6754 | theta=race_sharp (g=1.04, a=0.30, b=5.20, t=2.50, k=5, eg=0.12, mix=0.82)
    Client 1 (ds1) | train_acc=0.8619 | val_acc=0.8889 | val_f1=0.6512 | val_auc=0.9745 | reward=0.7106 | theta=race_robust (g=1.08, a=0.22, b=4.50, t=2.80, k=5, eg=0.11, mix=0.76)
    Client 2 (ds1) | train_acc=0.8903 | val_acc=0.8919 | val_f1=0.8917 | val_auc=1.0000 | reward=0.8917 | theta=race_sharp (g=1.04, a=0.30, b=5.20, t=2.50, k=5, eg=0.12, mix=0.82)
    Client 3 (ds2) | train_acc=0.8150 | val_acc=0.8571 | val_f1=0.8556 | val_auc=nan | reward=0.8560 | theta=race_smoothmix (g=0.90, a=0.20, b=4.00, t=2.10, k=7, eg=0.07, mix=0.70)
    Client 4 (ds2) | train_acc=0.8699 | val_acc=0.9362 | val_f1=0.6973 | val_auc=0.9951 | reward=0.7570 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    Client 5 (ds2) | train_acc=0.8434 | val_acc=0.8919 | val_f1=0.8582 | val_auc=0.9791 | reward=0.8666 | theta=race_smoothmix (g=0.90, a=0.20, b=4.00, t=2.10, k=7, eg=0.07, mix=0.70)
    
    ----------------------------------------------------------------------------------------------------------------------
    GLOBAL VAL (Round 4) | global_acc=0.8815 | global_f1=0.7502 | ds1_acc=0.8584 | ds1_f1=0.7238 | ds2_acc=0.9082 | ds2_f1=0.7807 | reward=0.8986 | round_time=58.0s
    ----------------------------------------------------------------------------------------------------------------------
    
    ======================================================================================================================
    ROUND 5/15 | selected=[0, 1, 2, 3, 4, 5]
    ======================================================================================================================
    Client 0 (ds1) | train_acc=0.8951 | val_acc=0.8250 | val_f1=0.6839 | val_auc=0.9688 | reward=0.7192 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    Client 1 (ds1) | train_acc=0.9514 | val_acc=0.9167 | val_f1=0.6683 | val_auc=0.9945 | reward=0.7304 | theta=race_sharp (g=1.04, a=0.30, b=5.20, t=2.50, k=5, eg=0.12, mix=0.82)
    Client 2 (ds1) | train_acc=0.9424 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | reward=1.0000 | theta=race_robust (g=1.08, a=0.22, b=4.50, t=2.80, k=5, eg=0.11, mix=0.76)
    Client 3 (ds2) | train_acc=0.8600 | val_acc=0.9286 | val_f1=0.8222 | val_auc=nan | reward=0.8488 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    Client 4 (ds2) | train_acc=0.8684 | val_acc=0.9787 | val_f1=0.7442 | val_auc=1.0000 | reward=0.8028 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    Client 5 (ds2) | train_acc=0.8509 | val_acc=0.7297 | val_f1=0.6555 | val_auc=0.9245 | reward=0.6741 | theta=race_robust (g=1.08, a=0.22, b=4.50, t=2.80, k=5, eg=0.11, mix=0.76)
    
    ----------------------------------------------------------------------------------------------------------------------
    GLOBAL VAL (Round 5) | global_acc=0.8957 | global_f1=0.7543 | ds1_acc=0.9115 | ds1_f1=0.7824 | ds2_acc=0.8776 | ds2_f1=0.7219 | reward=0.9019 | round_time=57.9s
    ----------------------------------------------------------------------------------------------------------------------
    
    ======================================================================================================================
    ROUND 6/15 | selected=[0, 1, 2, 3, 4, 5]
    ======================================================================================================================
    Client 0 (ds1) | train_acc=0.9283 | val_acc=0.8750 | val_f1=0.8205 | val_auc=0.9668 | reward=0.8341 | theta=race_texture (g=0.92, a=0.34, b=5.80, t=2.30, k=7, eg=0.10, mix=0.80)
    Client 1 (ds1) | train_acc=0.9533 | val_acc=0.9444 | val_f1=0.7018 | val_auc=0.8913 | reward=0.7625 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    Client 2 (ds1) | train_acc=0.9480 | val_acc=0.9730 | val_f1=0.9587 | val_auc=0.9973 | reward=0.9623 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    Client 3 (ds2) | train_acc=0.9000 | val_acc=1.0000 | val_f1=1.0000 | val_auc=nan | reward=1.0000 | theta=race_smoothmix (g=0.90, a=0.20, b=4.00, t=2.10, k=7, eg=0.07, mix=0.70)
    Client 4 (ds2) | train_acc=0.9181 | val_acc=0.9149 | val_f1=0.6847 | val_auc=0.9142 | reward=0.7422 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    Client 5 (ds2) | train_acc=0.8811 | val_acc=0.8649 | val_f1=0.8197 | val_auc=0.9676 | reward=0.8310 | theta=race_smoothmix (g=0.90, a=0.20, b=4.00, t=2.10, k=7, eg=0.07, mix=0.70)
    
    ----------------------------------------------------------------------------------------------------------------------
    GLOBAL VAL (Round 6) | global_acc=0.9194 | global_f1=0.8060 | ds1_acc=0.9292 | ds1_f1=0.8279 | ds2_acc=0.9082 | ds2_f1=0.7807 | reward=0.9548 | round_time=58.1s
    ----------------------------------------------------------------------------------------------------------------------
    
    ======================================================================================================================
    ROUND 7/15 | selected=[0, 1, 2, 3, 4, 5]
    ======================================================================================================================
    Client 0 (ds1) | train_acc=0.9773 | val_acc=0.8500 | val_f1=0.7061 | val_auc=0.9892 | reward=0.7421 | theta=race_texture (g=0.92, a=0.34, b=5.80, t=2.30, k=7, eg=0.10, mix=0.80)
    Client 1 (ds1) | train_acc=0.9494 | val_acc=0.9444 | val_f1=0.7266 | val_auc=1.0000 | reward=0.7811 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    Client 2 (ds1) | train_acc=0.9554 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | reward=1.0000 | theta=race_robust (g=1.08, a=0.22, b=4.50, t=2.80, k=5, eg=0.11, mix=0.76)
    Client 3 (ds2) | train_acc=0.9450 | val_acc=0.8571 | val_f1=0.7889 | val_auc=nan | reward=0.8060 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    Client 4 (ds2) | train_acc=0.9371 | val_acc=0.9362 | val_f1=0.6912 | val_auc=0.9979 | reward=0.7524 | theta=race_soft (g=0.95, a=0.18, b=3.80, t=2.20, k=3, eg=0.08, mix=0.72)
    Client 5 (ds2) | train_acc=0.9302 | val_acc=0.9189 | val_f1=0.8792 | val_auc=0.9739 | reward=0.8891 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    
    ----------------------------------------------------------------------------------------------------------------------
    GLOBAL VAL (Round 7) | global_acc=0.9242 | global_f1=0.7937 | ds1_acc=0.9292 | ds1_f1=0.8089 | ds2_acc=0.9184 | ds2_f1=0.7761 | reward=0.9471 | round_time=58.0s
    ----------------------------------------------------------------------------------------------------------------------
    
    ======================================================================================================================
    ROUND 8/15 | selected=[0, 1, 2, 3, 4, 5]
    ======================================================================================================================
    Client 0 (ds1) | train_acc=0.9458 | val_acc=0.9500 | val_f1=0.9169 | val_auc=0.9904 | reward=0.9252 | theta=race_robust (g=1.08, a=0.22, b=4.50, t=2.80, k=5, eg=0.11, mix=0.76)
    Client 1 (ds1) | train_acc=0.9708 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | reward=1.0000 | theta=race_texture (g=0.92, a=0.34, b=5.80, t=2.30, k=7, eg=0.10, mix=0.80)
    Client 2 (ds1) | train_acc=0.8885 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | reward=1.0000 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    Client 3 (ds2) | train_acc=0.9750 | val_acc=0.9286 | val_f1=0.9175 | val_auc=nan | reward=0.9202 | theta=race_soft (g=0.95, a=0.18, b=3.80, t=2.20, k=3, eg=0.08, mix=0.72)
    Client 4 (ds2) | train_acc=0.9415 | val_acc=0.9362 | val_f1=0.7083 | val_auc=0.9953 | reward=0.7653 | theta=race_soft (g=0.95, a=0.18, b=3.80, t=2.20, k=3, eg=0.08, mix=0.72)
    Client 5 (ds2) | train_acc=0.8943 | val_acc=0.9189 | val_f1=0.8841 | val_auc=0.9929 | reward=0.8928 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    
    ----------------------------------------------------------------------------------------------------------------------
    GLOBAL VAL (Round 8) | global_acc=0.9573 | global_f1=0.8935 | ds1_acc=0.9823 | ds1_f1=0.9706 | ds2_acc=0.9286 | ds2_f1=0.8045 | reward=1.0299 | round_time=58.1s
    ----------------------------------------------------------------------------------------------------------------------
    
    ======================================================================================================================
    ROUND 9/15 | selected=[0, 1, 2, 3, 4, 5]
    ======================================================================================================================
    Client 0 (ds1) | train_acc=0.9685 | val_acc=0.9250 | val_f1=0.8854 | val_auc=0.9856 | reward=0.8953 | theta=race_edge_plus (g=1.02, a=0.32, b=6.00, t=2.60, k=3, eg=0.15, mix=0.84)
    Client 1 (ds1) | train_acc=0.9533 | val_acc=0.9444 | val_f1=0.7018 | val_auc=0.9977 | reward=0.7625 | theta=race_edge_plus (g=1.02, a=0.32, b=6.00, t=2.60, k=3, eg=0.15, mix=0.84)
    Client 2 (ds1) | train_acc=0.9572 | val_acc=0.9730 | val_f1=0.9810 | val_auc=0.9992 | reward=0.9790 | theta=race_sharp (g=1.04, a=0.30, b=5.20, t=2.50, k=5, eg=0.12, mix=0.82)
    Client 3 (ds2) | train_acc=0.9750 | val_acc=1.0000 | val_f1=1.0000 | val_auc=nan | reward=1.0000 | theta=race_smoothmix (g=0.90, a=0.20, b=4.00, t=2.10, k=7, eg=0.07, mix=0.70)
    Client 4 (ds2) | train_acc=0.9605 | val_acc=0.9574 | val_f1=0.9504 | val_auc=0.9736 | reward=0.9522 | theta=race_smoothmix (g=0.90, a=0.20, b=4.00, t=2.10, k=7, eg=0.07, mix=0.70)
    Client 5 (ds2) | train_acc=0.9245 | val_acc=0.8649 | val_f1=0.8008 | val_auc=0.9817 | reward=0.8168 | theta=race_soft (g=0.95, a=0.18, b=3.80, t=2.20, k=3, eg=0.08, mix=0.72)
    
    ----------------------------------------------------------------------------------------------------------------------
    GLOBAL VAL (Round 9) | global_acc=0.9384 | global_f1=0.8781 | ds1_acc=0.9469 | ds1_f1=0.8582 | ds2_acc=0.9286 | ds2_f1=0.9010 | reward=1.0262 | round_time=57.9s
    ----------------------------------------------------------------------------------------------------------------------
    
    ======================================================================================================================
    ROUND 10/15 | selected=[0, 1, 2, 3, 4, 5]
    ======================================================================================================================
    Client 0 (ds1) | train_acc=0.9738 | val_acc=0.9250 | val_f1=0.9069 | val_auc=0.9929 | reward=0.9114 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    Client 1 (ds1) | train_acc=0.9416 | val_acc=0.9167 | val_f1=0.8286 | val_auc=0.9898 | reward=0.8506 | theta=race_sharp (g=1.04, a=0.30, b=5.20, t=2.50, k=5, eg=0.12, mix=0.82)
    Client 2 (ds1) | train_acc=0.9498 | val_acc=0.9459 | val_f1=0.9249 | val_auc=1.0000 | reward=0.9302 | theta=race_texture (g=0.92, a=0.34, b=5.80, t=2.30, k=7, eg=0.10, mix=0.80)
    Client 3 (ds2) | train_acc=0.9800 | val_acc=0.9286 | val_f1=0.9175 | val_auc=nan | reward=0.9202 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    Client 4 (ds2) | train_acc=0.9678 | val_acc=0.9787 | val_f1=0.9762 | val_auc=1.0000 | reward=0.9768 | theta=race_smoothmix (g=0.90, a=0.20, b=4.00, t=2.10, k=7, eg=0.07, mix=0.70)
    Client 5 (ds2) | train_acc=0.9264 | val_acc=0.8378 | val_f1=0.7605 | val_auc=0.9476 | reward=0.7799 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    
    ----------------------------------------------------------------------------------------------------------------------
    GLOBAL VAL (Round 10) | global_acc=0.9242 | global_f1=0.8872 | ds1_acc=0.9292 | ds1_f1=0.8878 | ds2_acc=0.9184 | ds2_f1=0.8864 | reward=1.0304 | round_time=57.5s
    ----------------------------------------------------------------------------------------------------------------------
    
    ======================================================================================================================
    ROUND 11/15 | selected=[0, 1, 2, 3, 4, 5]
    ======================================================================================================================
    Client 0 (ds1) | train_acc=0.9755 | val_acc=0.9250 | val_f1=0.8778 | val_auc=0.9951 | reward=0.8896 | theta=race_sharp (g=1.04, a=0.30, b=5.20, t=2.50, k=5, eg=0.12, mix=0.82)
    Client 1 (ds1) | train_acc=0.9630 | val_acc=0.9722 | val_f1=0.7436 | val_auc=1.0000 | reward=0.8007 | theta=race_robust (g=1.08, a=0.22, b=4.50, t=2.80, k=5, eg=0.11, mix=0.76)
    Client 2 (ds1) | train_acc=0.9647 | val_acc=0.9459 | val_f1=0.9631 | val_auc=1.0000 | reward=0.9588 | theta=race_edge_plus (g=1.02, a=0.32, b=6.00, t=2.60, k=3, eg=0.15, mix=0.84)
    Client 3 (ds2) | train_acc=0.9800 | val_acc=0.9286 | val_f1=0.9175 | val_auc=nan | reward=0.9202 | theta=race_soft (g=0.95, a=0.18, b=3.80, t=2.20, k=3, eg=0.08, mix=0.72)
    Client 4 (ds2) | train_acc=0.9591 | val_acc=0.9574 | val_f1=0.9540 | val_auc=0.9820 | reward=0.9549 | theta=race_smoothmix (g=0.90, a=0.20, b=4.00, t=2.10, k=7, eg=0.07, mix=0.70)
    Client 5 (ds2) | train_acc=0.9660 | val_acc=0.9189 | val_f1=0.8792 | val_auc=0.9826 | reward=0.8891 | theta=race_soft (g=0.95, a=0.18, b=3.80, t=2.20, k=3, eg=0.08, mix=0.72)
    
    ----------------------------------------------------------------------------------------------------------------------
    GLOBAL VAL (Round 11) | global_acc=0.9431 | global_f1=0.8897 | ds1_acc=0.9469 | ds1_f1=0.8630 | ds2_acc=0.9388 | ds2_f1=0.9205 | reward=1.0371 | round_time=58.2s
    ----------------------------------------------------------------------------------------------------------------------
    
    ======================================================================================================================
    ROUND 12/15 | selected=[0, 1, 2, 3, 4, 5]
    ======================================================================================================================
    Client 0 (ds1) | train_acc=0.9563 | val_acc=0.9000 | val_f1=0.8671 | val_auc=0.9908 | reward=0.8753 | theta=race_focus (g=1.10, a=0.36, b=6.40, t=2.70, k=3, eg=0.17, mix=0.86)
    Client 1 (ds1) | train_acc=0.9747 | val_acc=0.9444 | val_f1=0.9361 | val_auc=1.0000 | reward=0.9382 | theta=race_focus (g=1.10, a=0.36, b=6.40, t=2.70, k=3, eg=0.17, mix=0.86)
    Client 2 (ds1) | train_acc=0.9665 | val_acc=0.9730 | val_f1=0.9442 | val_auc=1.0000 | reward=0.9514 | theta=race_focus (g=1.10, a=0.36, b=6.40, t=2.70, k=3, eg=0.17, mix=0.86)
    Client 3 (ds2) | train_acc=0.9850 | val_acc=1.0000 | val_f1=1.0000 | val_auc=nan | reward=1.0000 | theta=race_robust (g=1.08, a=0.22, b=4.50, t=2.80, k=5, eg=0.11, mix=0.76)
    Client 4 (ds2) | train_acc=0.9591 | val_acc=0.9574 | val_f1=0.9730 | val_auc=0.9985 | reward=0.9691 | theta=race_soft (g=0.95, a=0.18, b=3.80, t=2.20, k=3, eg=0.08, mix=0.72)
    Client 5 (ds2) | train_acc=0.9491 | val_acc=0.9459 | val_f1=0.9342 | val_auc=0.9958 | reward=0.9371 | theta=race_robust (g=1.08, a=0.22, b=4.50, t=2.80, k=5, eg=0.11, mix=0.76)
    
    ----------------------------------------------------------------------------------------------------------------------
    GLOBAL VAL (Round 12) | global_acc=0.9479 | global_f1=0.9366 | ds1_acc=0.9381 | ds1_f1=0.9143 | ds2_acc=0.9592 | ds2_f1=0.9622 | reward=1.0789 | round_time=58.1s
    ----------------------------------------------------------------------------------------------------------------------
    
    ======================================================================================================================
    ROUND 13/15 | selected=[0, 1, 2, 3, 4, 5]
    ======================================================================================================================
    Client 0 (ds1) | train_acc=0.9773 | val_acc=0.9250 | val_f1=0.8872 | val_auc=0.9877 | reward=0.8967 | theta=race_robust (g=1.08, a=0.22, b=4.50, t=2.80, k=5, eg=0.11, mix=0.76)
    Client 1 (ds1) | train_acc=0.9650 | val_acc=0.9722 | val_f1=0.7436 | val_auc=1.0000 | reward=0.8007 | theta=race_texture (g=0.92, a=0.34, b=5.80, t=2.30, k=7, eg=0.10, mix=0.80)
    Client 2 (ds1) | train_acc=0.9721 | val_acc=0.9730 | val_f1=0.9442 | val_auc=1.0000 | reward=0.9514 | theta=race_robust (g=1.08, a=0.22, b=4.50, t=2.80, k=5, eg=0.11, mix=0.76)
    Client 3 (ds2) | train_acc=0.9850 | val_acc=0.9286 | val_f1=0.9175 | val_auc=nan | reward=0.9202 | theta=race_smoothmix (g=0.90, a=0.20, b=4.00, t=2.10, k=7, eg=0.07, mix=0.70)
    Client 4 (ds2) | train_acc=0.9766 | val_acc=0.9787 | val_f1=0.9762 | val_auc=0.9780 | reward=0.9768 | theta=race_robust (g=1.08, a=0.22, b=4.50, t=2.80, k=5, eg=0.11, mix=0.76)
    Client 5 (ds2) | train_acc=0.9585 | val_acc=0.9189 | val_f1=0.8907 | val_auc=0.9891 | reward=0.8978 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    
    ----------------------------------------------------------------------------------------------------------------------
    GLOBAL VAL (Round 13) | global_acc=0.9526 | global_f1=0.8951 | ds1_acc=0.9558 | ds1_f1=0.8601 | ds2_acc=0.9490 | ds2_f1=0.9355 | reward=1.0441 | round_time=58.0s
    ----------------------------------------------------------------------------------------------------------------------
    
    ======================================================================================================================
    ROUND 14/15 | selected=[0, 1, 2, 3, 4, 5]
    ======================================================================================================================
    Client 0 (ds1) | train_acc=0.9685 | val_acc=0.8750 | val_f1=0.7925 | val_auc=0.9983 | reward=0.8131 | theta=race_edge_plus (g=1.02, a=0.32, b=6.00, t=2.60, k=3, eg=0.15, mix=0.84)
    Client 1 (ds1) | train_acc=0.9922 | val_acc=0.9444 | val_f1=0.9361 | val_auc=0.9892 | reward=0.9382 | theta=race_focus (g=1.10, a=0.36, b=6.40, t=2.70, k=3, eg=0.17, mix=0.86)
    Client 2 (ds1) | train_acc=0.9777 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | reward=1.0000 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    Client 3 (ds2) | train_acc=0.9650 | val_acc=1.0000 | val_f1=1.0000 | val_auc=nan | reward=1.0000 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    Client 4 (ds2) | train_acc=0.9825 | val_acc=0.9787 | val_f1=0.9762 | val_auc=1.0000 | reward=0.9768 | theta=race_soft (g=0.95, a=0.18, b=3.80, t=2.20, k=3, eg=0.08, mix=0.72)
    Client 5 (ds2) | train_acc=0.9491 | val_acc=0.8919 | val_f1=0.8620 | val_auc=0.9899 | reward=0.8694 | theta=race_smoothmix (g=0.90, a=0.20, b=4.00, t=2.10, k=7, eg=0.07, mix=0.70)
    
    ----------------------------------------------------------------------------------------------------------------------
    GLOBAL VAL (Round 14) | global_acc=0.9431 | global_f1=0.9203 | ds1_acc=0.9381 | ds1_f1=0.9062 | ds2_acc=0.9490 | ds2_f1=0.9365 | reward=1.0640 | round_time=58.1s
    ----------------------------------------------------------------------------------------------------------------------
    
    ======================================================================================================================
    ROUND 15/15 | selected=[0, 1, 2, 3, 4, 5]
    ======================================================================================================================
    Client 0 (ds1) | train_acc=0.9808 | val_acc=0.9250 | val_f1=0.9164 | val_auc=0.9901 | reward=0.9185 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    Client 1 (ds1) | train_acc=0.9630 | val_acc=0.9444 | val_f1=0.6951 | val_auc=0.9609 | reward=0.7574 | theta=race_sharp (g=1.04, a=0.30, b=5.20, t=2.50, k=5, eg=0.12, mix=0.82)
    Client 2 (ds1) | train_acc=0.9684 | val_acc=0.9730 | val_f1=0.9797 | val_auc=1.0000 | reward=0.9780 | theta=race_sharp (g=1.04, a=0.30, b=5.20, t=2.50, k=5, eg=0.12, mix=0.82)
    Client 3 (ds2) | train_acc=0.9600 | val_acc=0.9286 | val_f1=0.9348 | val_auc=nan | reward=0.9333 | theta=race_robust (g=1.08, a=0.22, b=4.50, t=2.80, k=5, eg=0.11, mix=0.76)
    Client 4 (ds2) | train_acc=0.9678 | val_acc=0.9362 | val_f1=0.9492 | val_auc=0.9955 | reward=0.9460 | theta=race_robust (g=1.08, a=0.22, b=4.50, t=2.80, k=5, eg=0.11, mix=0.76)
    Client 5 (ds2) | train_acc=0.9679 | val_acc=0.8649 | val_f1=0.8117 | val_auc=0.9784 | reward=0.8250 | theta=race_soft (g=0.95, a=0.18, b=3.80, t=2.20, k=3, eg=0.08, mix=0.72)
    
    ----------------------------------------------------------------------------------------------------------------------
    GLOBAL VAL (Round 15) | global_acc=0.9289 | global_f1=0.8799 | ds1_acc=0.9469 | ds1_f1=0.8666 | ds2_acc=0.9082 | ds2_f1=0.8953 | reward=1.0256 | round_time=58.4s
    ----------------------------------------------------------------------------------------------------------------------
    
    ======================================================================================================================
    TRAINING COMPLETE ✅ | total_time=863.7s | best_round=12 | best_reward=1.0789
    ======================================================================================================================
    
    ----------------------------------------------------------------------------------------------------------------------
    GLOBAL per-round metrics
    ----------------------------------------------------------------------------------------------------------------------



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
      <td>56.123893</td>
      <td>6</td>
      <td>1.0</td>
      <td>0.794754</td>
      <td>0.800948</td>
      <td>0.657976</td>
      <td>0.690791</td>
      <td>0.674594</td>
      <td>0.552755</td>
      <td>0.542756</td>
      <td>0.608487</td>
      <td>0.787611</td>
      <td>0.678578</td>
      <td>0.595338</td>
      <td>0.816327</td>
      <td>0.634220</td>
      <td>0.503655</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>53.141806</td>
      <td>6</td>
      <td>1.0</td>
      <td>0.824394</td>
      <td>0.848341</td>
      <td>0.683581</td>
      <td>0.690769</td>
      <td>0.705528</td>
      <td>0.423012</td>
      <td>0.410643</td>
      <td>0.524172</td>
      <td>0.858407</td>
      <td>0.728824</td>
      <td>0.419872</td>
      <td>0.836735</td>
      <td>0.631414</td>
      <td>0.426633</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>57.690765</td>
      <td>6</td>
      <td>1.0</td>
      <td>0.877174</td>
      <td>0.881517</td>
      <td>0.723437</td>
      <td>0.770809</td>
      <td>0.714838</td>
      <td>0.361908</td>
      <td>0.358838</td>
      <td>0.507315</td>
      <td>0.884956</td>
      <td>0.718179</td>
      <td>0.376674</td>
      <td>0.877551</td>
      <td>0.729499</td>
      <td>0.344881</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>57.990919</td>
      <td>6</td>
      <td>1.0</td>
      <td>0.898626</td>
      <td>0.881517</td>
      <td>0.750226</td>
      <td>0.773446</td>
      <td>0.775196</td>
      <td>0.308873</td>
      <td>0.308032</td>
      <td>0.507264</td>
      <td>0.858407</td>
      <td>0.723818</td>
      <td>0.348073</td>
      <td>0.908163</td>
      <td>0.780676</td>
      <td>0.263674</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>57.851919</td>
      <td>6</td>
      <td>1.0</td>
      <td>0.901860</td>
      <td>0.895735</td>
      <td>0.754300</td>
      <td>0.787905</td>
      <td>0.764886</td>
      <td>0.292966</td>
      <td>0.287888</td>
      <td>0.501380</td>
      <td>0.911504</td>
      <td>0.782432</td>
      <td>0.251314</td>
      <td>0.877551</td>
      <td>0.721863</td>
      <td>0.340992</td>
    </tr>
    <tr>
      <th>5</th>
      <td>6</td>
      <td>58.141534</td>
      <td>6</td>
      <td>1.0</td>
      <td>0.954789</td>
      <td>0.919431</td>
      <td>0.805990</td>
      <td>0.815871</td>
      <td>0.808781</td>
      <td>0.291856</td>
      <td>0.275813</td>
      <td>0.521756</td>
      <td>0.929204</td>
      <td>0.827925</td>
      <td>0.255463</td>
      <td>0.908163</td>
      <td>0.780698</td>
      <td>0.333820</td>
    </tr>
    <tr>
      <th>6</th>
      <td>7</td>
      <td>57.990996</td>
      <td>6</td>
      <td>1.0</td>
      <td>0.947070</td>
      <td>0.924171</td>
      <td>0.793661</td>
      <td>0.818322</td>
      <td>0.806848</td>
      <td>0.204166</td>
      <td>0.211962</td>
      <td>0.506483</td>
      <td>0.929204</td>
      <td>0.808881</td>
      <td>0.210004</td>
      <td>0.918367</td>
      <td>0.776111</td>
      <td>0.197434</td>
    </tr>
    <tr>
      <th>7</th>
      <td>8</td>
      <td>58.140305</td>
      <td>6</td>
      <td>1.0</td>
      <td>1.029863</td>
      <td>0.957346</td>
      <td>0.893465</td>
      <td>0.898298</td>
      <td>0.896325</td>
      <td>0.159323</td>
      <td>0.148292</td>
      <td>0.503628</td>
      <td>0.982301</td>
      <td>0.970585</td>
      <td>0.136739</td>
      <td>0.928571</td>
      <td>0.804540</td>
      <td>0.185364</td>
    </tr>
    <tr>
      <th>8</th>
      <td>9</td>
      <td>57.873515</td>
      <td>6</td>
      <td>1.0</td>
      <td>1.026205</td>
      <td>0.938389</td>
      <td>0.878095</td>
      <td>0.891714</td>
      <td>0.876377</td>
      <td>0.233955</td>
      <td>0.211619</td>
      <td>0.506081</td>
      <td>0.946903</td>
      <td>0.858221</td>
      <td>0.222589</td>
      <td>0.928571</td>
      <td>0.901012</td>
      <td>0.247059</td>
    </tr>
    <tr>
      <th>9</th>
      <td>10</td>
      <td>57.510485</td>
      <td>6</td>
      <td>1.0</td>
      <td>1.030436</td>
      <td>0.924171</td>
      <td>0.887164</td>
      <td>0.906826</td>
      <td>0.917542</td>
      <td>0.274237</td>
      <td>0.288089</td>
      <td>0.495392</td>
      <td>0.929204</td>
      <td>0.887845</td>
      <td>0.227386</td>
      <td>0.918367</td>
      <td>0.886378</td>
      <td>0.328258</td>
    </tr>
    <tr>
      <th>10</th>
      <td>11</td>
      <td>58.169411</td>
      <td>6</td>
      <td>1.0</td>
      <td>1.037119</td>
      <td>0.943128</td>
      <td>0.889709</td>
      <td>0.892085</td>
      <td>0.897801</td>
      <td>0.203531</td>
      <td>0.187400</td>
      <td>0.517786</td>
      <td>0.946903</td>
      <td>0.862963</td>
      <td>0.168675</td>
      <td>0.938776</td>
      <td>0.920548</td>
      <td>0.243722</td>
    </tr>
    <tr>
      <th>11</th>
      <td>12</td>
      <td>58.129834</td>
      <td>6</td>
      <td>1.0</td>
      <td>1.078891</td>
      <td>0.947867</td>
      <td>0.936563</td>
      <td>0.945061</td>
      <td>0.942646</td>
      <td>0.164421</td>
      <td>0.158763</td>
      <td>0.514189</td>
      <td>0.938053</td>
      <td>0.914319</td>
      <td>0.168331</td>
      <td>0.959184</td>
      <td>0.962211</td>
      <td>0.159911</td>
    </tr>
    <tr>
      <th>12</th>
      <td>13</td>
      <td>57.976779</td>
      <td>6</td>
      <td>1.0</td>
      <td>1.044068</td>
      <td>0.952607</td>
      <td>0.895149</td>
      <td>0.897919</td>
      <td>0.899119</td>
      <td>0.167858</td>
      <td>0.169349</td>
      <td>0.513845</td>
      <td>0.955752</td>
      <td>0.860120</td>
      <td>0.151007</td>
      <td>0.948980</td>
      <td>0.935539</td>
      <td>0.187288</td>
    </tr>
    <tr>
      <th>13</th>
      <td>14</td>
      <td>58.102709</td>
      <td>6</td>
      <td>1.0</td>
      <td>1.063999</td>
      <td>0.943128</td>
      <td>0.920252</td>
      <td>0.938768</td>
      <td>0.923186</td>
      <td>0.162396</td>
      <td>0.157814</td>
      <td>0.508999</td>
      <td>0.938053</td>
      <td>0.906195</td>
      <td>0.189228</td>
      <td>0.948980</td>
      <td>0.936461</td>
      <td>0.131457</td>
    </tr>
    <tr>
      <th>14</th>
      <td>15</td>
      <td>58.392950</td>
      <td>6</td>
      <td>1.0</td>
      <td>1.025584</td>
      <td>0.928910</td>
      <td>0.879913</td>
      <td>0.890660</td>
      <td>0.877239</td>
      <td>0.275398</td>
      <td>0.271733</td>
      <td>0.520446</td>
      <td>0.946903</td>
      <td>0.866611</td>
      <td>0.204313</td>
      <td>0.908163</td>
      <td>0.895251</td>
      <td>0.357362</td>
    </tr>
  </tbody>
</table>
</div>


    
    ----------------------------------------------------------------------------------------------------------------------
    LOCAL per-client per-round metrics
    ----------------------------------------------------------------------------------------------------------------------



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
      <td>4</td>
      <td>race_edge_plus</td>
      <td>(g=1.02, a=0.32, b=6.00, t=2.60, k=3, eg=0.15,...</td>
      <td>1.02</td>
      <td>0.32</td>
      <td>6.0</td>
      <td>...</td>
      <td>0.932496</td>
      <td>0.828431</td>
      <td>0.944444</td>
      <td>0.974425</td>
      <td>0.982684</td>
      <td>0.543503</td>
      <td>0.277718</td>
      <td>0.178779</td>
      <td>1.386002</td>
      <td>0.739389</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>client_1</td>
      <td>ds1</td>
      <td>1</td>
      <td>5</td>
      <td>race_focus</td>
      <td>(g=1.10, a=0.36, b=6.40, t=2.70, k=3, eg=0.17,...</td>
      <td>1.10</td>
      <td>0.36</td>
      <td>6.4</td>
      <td>...</td>
      <td>0.957738</td>
      <td>0.891641</td>
      <td>0.971429</td>
      <td>0.992188</td>
      <td>0.975694</td>
      <td>0.760059</td>
      <td>0.070815</td>
      <td>0.169126</td>
      <td>0.980002</td>
      <td>0.651736</td>
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
      <td>0.915657</td>
      <td>0.727273</td>
      <td>0.957576</td>
      <td>0.985714</td>
      <td>0.992063</td>
      <td>0.257256</td>
      <td>0.008313</td>
      <td>0.734432</td>
      <td>0.822764</td>
      <td>0.722202</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>client_3</td>
      <td>ds2</td>
      <td>1</td>
      <td>2</td>
      <td>race_robust</td>
      <td>(g=1.08, a=0.22, b=4.50, t=2.80, k=5, eg=0.11,...</td>
      <td>1.08</td>
      <td>0.22</td>
      <td>4.5</td>
      <td>...</td>
      <td>NaN</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>NaN</td>
      <td>1.000000</td>
      <td>0.012412</td>
      <td>0.377775</td>
      <td>0.609813</td>
      <td>0.999128</td>
      <td>0.720238</td>
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
      <td>0.949348</td>
      <td>0.869565</td>
      <td>0.947059</td>
      <td>0.987179</td>
      <td>0.993590</td>
      <td>0.013731</td>
      <td>0.559445</td>
      <td>0.426824</td>
      <td>0.769847</td>
      <td>0.684009</td>
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
      <td>1</td>
      <td>race_sharp</td>
      <td>(g=1.04, a=0.30, b=5.20, t=2.50, k=5, eg=0.12,...</td>
      <td>1.04</td>
      <td>0.30</td>
      <td>5.2</td>
      <td>...</td>
      <td>0.960938</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.843750</td>
      <td>1.000000</td>
      <td>0.406939</td>
      <td>0.287716</td>
      <td>0.305345</td>
      <td>1.429486</td>
      <td>0.757402</td>
    </tr>
    <tr>
      <th>86</th>
      <td>15</td>
      <td>client_2</td>
      <td>ds1</td>
      <td>1</td>
      <td>1</td>
      <td>race_sharp</td>
      <td>(g=1.04, a=0.30, b=5.20, t=2.50, k=5, eg=0.12,...</td>
      <td>1.04</td>
      <td>0.30</td>
      <td>5.2</td>
      <td>...</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.638809</td>
      <td>0.217777</td>
      <td>0.143414</td>
      <td>1.058447</td>
      <td>0.978047</td>
    </tr>
    <tr>
      <th>87</th>
      <td>15</td>
      <td>client_3</td>
      <td>ds2</td>
      <td>1</td>
      <td>2</td>
      <td>race_robust</td>
      <td>(g=1.08, a=0.22, b=4.50, t=2.80, k=5, eg=0.11,...</td>
      <td>1.08</td>
      <td>0.22</td>
      <td>4.5</td>
      <td>...</td>
      <td>NaN</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>NaN</td>
      <td>1.000000</td>
      <td>0.630329</td>
      <td>0.230625</td>
      <td>0.139046</td>
      <td>1.201045</td>
      <td>0.933271</td>
    </tr>
    <tr>
      <th>88</th>
      <td>15</td>
      <td>client_4</td>
      <td>ds2</td>
      <td>1</td>
      <td>2</td>
      <td>race_robust</td>
      <td>(g=1.08, a=0.22, b=4.50, t=2.80, k=5, eg=0.11,...</td>
      <td>1.08</td>
      <td>0.22</td>
      <td>4.5</td>
      <td>...</td>
      <td>0.995456</td>
      <td>1.000000</td>
      <td>0.988235</td>
      <td>1.000000</td>
      <td>0.993590</td>
      <td>0.464666</td>
      <td>0.434575</td>
      <td>0.100759</td>
      <td>1.243704</td>
      <td>0.945974</td>
    </tr>
    <tr>
      <th>89</th>
      <td>15</td>
      <td>client_5</td>
      <td>ds2</td>
      <td>1</td>
      <td>0</td>
      <td>race_soft</td>
      <td>(g=0.95, a=0.18, b=3.80, t=2.20, k=3, eg=0.08,...</td>
      <td>0.95</td>
      <td>0.18</td>
      <td>3.8</td>
      <td>...</td>
      <td>0.978441</td>
      <td>0.970000</td>
      <td>0.946970</td>
      <td>1.000000</td>
      <td>0.996795</td>
      <td>0.397701</td>
      <td>0.342201</td>
      <td>0.260098</td>
      <td>1.097437</td>
      <td>0.824982</td>
    </tr>
  </tbody>
</table>
<p>90 rows × 40 columns</p>
</div>


    
    ======================================================================================================================
    STEP 12: FINAL EVALUATION
    ======================================================================================================================
    
    ----------------------------------------------------------------------------------------------------------------------
    Best RL-selected preprocessing preset per client
    ----------------------------------------------------------------------------------------------------------------------



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
      <td>3</td>
      <td>race_robust</td>
      <td>(g=1.08, a=0.22, b=4.50, t=2.80, k=5, eg=0.11,...</td>
      <td>0.849882</td>
      <td>12</td>
    </tr>
    <tr>
      <th>1</th>
      <td>client_1</td>
      <td>ds1</td>
      <td>2</td>
      <td>race_texture</td>
      <td>(g=0.92, a=0.34, b=5.80, t=2.30, k=7, eg=0.10,...</td>
      <td>0.890621</td>
      <td>12</td>
    </tr>
    <tr>
      <th>2</th>
      <td>client_2</td>
      <td>ds1</td>
      <td>3</td>
      <td>race_robust</td>
      <td>(g=1.08, a=0.22, b=4.50, t=2.80, k=5, eg=0.11,...</td>
      <td>1.000000</td>
      <td>12</td>
    </tr>
    <tr>
      <th>3</th>
      <td>client_3</td>
      <td>ds2</td>
      <td>3</td>
      <td>race_smoothmix</td>
      <td>(g=0.90, a=0.20, b=4.00, t=2.10, k=7, eg=0.07,...</td>
      <td>0.951984</td>
      <td>12</td>
    </tr>
    <tr>
      <th>4</th>
      <td>client_4</td>
      <td>ds2</td>
      <td>3</td>
      <td>race_smoothmix</td>
      <td>(g=0.90, a=0.20, b=4.00, t=2.10, k=7, eg=0.07,...</td>
      <td>0.891972</td>
      <td>12</td>
    </tr>
    <tr>
      <th>5</th>
      <td>client_5</td>
      <td>ds2</td>
      <td>1</td>
      <td>race_balanced</td>
      <td>(g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10,...</td>
      <td>0.853912</td>
      <td>12</td>
    </tr>
  </tbody>
</table>
</div>


    
    ----------------------------------------------------------------------------------------------------------------------
    Chosen final validation-based preprocessing strategy
    ----------------------------------------------------------------------------------------------------------------------



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
      <td>['race_robust']</td>
      <td>0.986805</td>
      <td>0.986726</td>
      <td>0.986832</td>
    </tr>
    <tr>
      <th>1</th>
      <td>ds2</td>
      <td>single</td>
      <td>['race_balanced']</td>
      <td>0.957096</td>
      <td>0.959391</td>
      <td>0.956331</td>
    </tr>
  </tbody>
</table>
</div>


    
    ======================================================================================================================
    STEP 12.5: EXTENDED METRICS + ERROR ANALYSIS + CALIBRATION
    ======================================================================================================================
    
    ----------------------------------------------------------------------------------------------------------------------
    Extended TEST metrics (DS1 vs DS2)
    ----------------------------------------------------------------------------------------------------------------------



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
      <td>0.986726</td>
      <td>0.987061</td>
      <td>0.987069</td>
      <td>0.987061</td>
      <td>0.986952</td>
      <td>0.986955</td>
      <td>0.986726</td>
      <td>0.986727</td>
      <td>0.062062</td>
      <td>...</td>
      <td>0.004438</td>
      <td>0.012939</td>
      <td>0.029188</td>
      <td>0.610408</td>
      <td>0.021834</td>
      <td>0.999793</td>
      <td>0.999475</td>
      <td>1.000000</td>
      <td>0.999696</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>ds2_test</td>
      <td>0.949239</td>
      <td>0.946868</td>
      <td>0.948853</td>
      <td>0.946868</td>
      <td>0.945575</td>
      <td>0.951805</td>
      <td>0.949239</td>
      <td>0.948403</td>
      <td>0.170556</td>
      <td>...</td>
      <td>0.016652</td>
      <td>0.053132</td>
      <td>0.019856</td>
      <td>0.484156</td>
      <td>0.080219</td>
      <td>0.996466</td>
      <td>0.999854</td>
      <td>0.992658</td>
      <td>0.999638</td>
      <td>0.993713</td>
    </tr>
  </tbody>
</table>
<p>2 rows × 26 columns</p>
</div>


    
    ----------------------------------------------------------------------------------------------------------------------
    Classwise metrics — DS1 TEST
    ----------------------------------------------------------------------------------------------------------------------



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
      <td>55</td>
      <td>0</td>
      <td>1</td>
      <td>170</td>
      <td>0.247788</td>
      <td>1.000000</td>
      <td>0.994152</td>
      <td>0.982143</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.017857</td>
      <td>0.982143</td>
      <td>0.991071</td>
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
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>notumor</td>
      <td>59</td>
      <td>57</td>
      <td>1</td>
      <td>2</td>
      <td>166</td>
      <td>0.261062</td>
      <td>0.982759</td>
      <td>0.988095</td>
      <td>0.966102</td>
      <td>0.994012</td>
      <td>0.005988</td>
      <td>0.033898</td>
      <td>0.950000</td>
      <td>0.980057</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>pituitary</td>
      <td>56</td>
      <td>56</td>
      <td>2</td>
      <td>0</td>
      <td>168</td>
      <td>0.247788</td>
      <td>0.965517</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.988235</td>
      <td>0.011765</td>
      <td>0.000000</td>
      <td>0.965517</td>
      <td>0.994118</td>
    </tr>
  </tbody>
</table>
</div>


    
    ----------------------------------------------------------------------------------------------------------------------
    Classwise metrics — DS2 TEST
    ----------------------------------------------------------------------------------------------------------------------



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
      <td>45</td>
      <td>45</td>
      <td>2</td>
      <td>0</td>
      <td>150</td>
      <td>0.228426</td>
      <td>0.957447</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.986842</td>
      <td>0.013158</td>
      <td>0.000000</td>
      <td>0.957447</td>
      <td>0.993421</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>meningioma</td>
      <td>46</td>
      <td>38</td>
      <td>1</td>
      <td>8</td>
      <td>150</td>
      <td>0.233503</td>
      <td>0.974359</td>
      <td>0.949367</td>
      <td>0.826087</td>
      <td>0.993377</td>
      <td>0.006623</td>
      <td>0.173913</td>
      <td>0.808511</td>
      <td>0.909732</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>notumor</td>
      <td>61</td>
      <td>60</td>
      <td>1</td>
      <td>1</td>
      <td>135</td>
      <td>0.309645</td>
      <td>0.983607</td>
      <td>0.992647</td>
      <td>0.983607</td>
      <td>0.992647</td>
      <td>0.007353</td>
      <td>0.016393</td>
      <td>0.967742</td>
      <td>0.988127</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>pituitary</td>
      <td>45</td>
      <td>44</td>
      <td>6</td>
      <td>1</td>
      <td>146</td>
      <td>0.228426</td>
      <td>0.880000</td>
      <td>0.993197</td>
      <td>0.977778</td>
      <td>0.960526</td>
      <td>0.039474</td>
      <td>0.022222</td>
      <td>0.862745</td>
      <td>0.969152</td>
    </tr>
  </tbody>
</table>
</div>


    
    ----------------------------------------------------------------------------------------------------------------------
    Top confusion pairs — DS1 TEST
    ----------------------------------------------------------------------------------------------------------------------



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
      <td>notumor</td>
      <td>pituitary</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>glioma</td>
      <td>notumor</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>glioma</td>
      <td>pituitary</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>glioma</td>
      <td>meningioma</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>meningioma</td>
      <td>glioma</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>meningioma</td>
      <td>notumor</td>
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
      <td>meningioma</td>
      <td>pituitary</td>
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
      <td>pituitary</td>
      <td>glioma</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>


    
    ----------------------------------------------------------------------------------------------------------------------
    Top confusion pairs — DS2 TEST
    ----------------------------------------------------------------------------------------------------------------------



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
      <td>meningioma</td>
      <td>pituitary</td>
      <td>5</td>
    </tr>
    <tr>
      <th>1</th>
      <td>meningioma</td>
      <td>glioma</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>meningioma</td>
      <td>notumor</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>pituitary</td>
      <td>meningioma</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>notumor</td>
      <td>pituitary</td>
      <td>1</td>
    </tr>
    <tr>
      <th>5</th>
      <td>glioma</td>
      <td>pituitary</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>glioma</td>
      <td>meningioma</td>
      <td>0</td>
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


    
    ----------------------------------------------------------------------------------------------------------------------
    Calibration bins — DS1 TEST
    ----------------------------------------------------------------------------------------------------------------------



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
      <td>0.538782</td>
      <td>1.000000</td>
      <td>0.461218</td>
      <td>1</td>
    </tr>
    <tr>
      <th>7</th>
      <td>7</td>
      <td>0.583333</td>
      <td>0.666667</td>
      <td>0.610408</td>
      <td>0.000000</td>
      <td>0.610408</td>
      <td>1</td>
    </tr>
    <tr>
      <th>8</th>
      <td>8</td>
      <td>0.666667</td>
      <td>0.750000</td>
      <td>0.713090</td>
      <td>1.000000</td>
      <td>0.286910</td>
      <td>1</td>
    </tr>
    <tr>
      <th>9</th>
      <td>9</td>
      <td>0.750000</td>
      <td>0.833333</td>
      <td>0.789179</td>
      <td>0.666667</td>
      <td>0.122513</td>
      <td>3</td>
    </tr>
    <tr>
      <th>10</th>
      <td>10</td>
      <td>0.833333</td>
      <td>0.916667</td>
      <td>0.870478</td>
      <td>0.750000</td>
      <td>0.120478</td>
      <td>4</td>
    </tr>
    <tr>
      <th>11</th>
      <td>11</td>
      <td>0.916667</td>
      <td>1.000000</td>
      <td>0.979683</td>
      <td>1.000000</td>
      <td>0.020317</td>
      <td>216</td>
    </tr>
  </tbody>
</table>
</div>


    
    ----------------------------------------------------------------------------------------------------------------------
    Calibration bins — DS2 TEST
    ----------------------------------------------------------------------------------------------------------------------



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
      <td>0.462210</td>
      <td>0.000000</td>
      <td>0.462210</td>
      <td>3</td>
    </tr>
    <tr>
      <th>6</th>
      <td>6</td>
      <td>0.500000</td>
      <td>0.583333</td>
      <td>0.515844</td>
      <td>1.000000</td>
      <td>0.484156</td>
      <td>2</td>
    </tr>
    <tr>
      <th>7</th>
      <td>7</td>
      <td>0.583333</td>
      <td>0.666667</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>8</td>
      <td>0.666667</td>
      <td>0.750000</td>
      <td>0.691795</td>
      <td>0.500000</td>
      <td>0.191795</td>
      <td>4</td>
    </tr>
    <tr>
      <th>9</th>
      <td>9</td>
      <td>0.750000</td>
      <td>0.833333</td>
      <td>0.796476</td>
      <td>0.750000</td>
      <td>0.046476</td>
      <td>4</td>
    </tr>
    <tr>
      <th>10</th>
      <td>10</td>
      <td>0.833333</td>
      <td>0.916667</td>
      <td>0.883533</td>
      <td>0.857143</td>
      <td>0.026390</td>
      <td>7</td>
    </tr>
    <tr>
      <th>11</th>
      <td>11</td>
      <td>0.916667</td>
      <td>1.000000</td>
      <td>0.980685</td>
      <td>0.983051</td>
      <td>0.002366</td>
      <td>177</td>
    </tr>
  </tbody>
</table>
</div>


    
    ----------------------------------------------------------------------------------------------------------------------
    VAL + TEST tables (federated, per-dataset + global equal)
    ----------------------------------------------------------------------------------------------------------------------



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
      <td>ARCF-Net + FedAvg + prototype sharing</td>
      <td>VAL</td>
      <td>ds1</td>
      <td>0.986726</td>
      <td>0.987288</td>
      <td>0.986905</td>
      <td>0.986832</td>
      <td>0.987401</td>
      <td>0.986726</td>
      <td>0.986800</td>
      <td>...</td>
      <td>4.743692</td>
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
      <td>ARCF-Net + FedAvg + prototype sharing</td>
      <td>VAL</td>
      <td>ds2</td>
      <td>0.959391</td>
      <td>0.956624</td>
      <td>0.957737</td>
      <td>0.956331</td>
      <td>0.960125</td>
      <td>0.959391</td>
      <td>0.958967</td>
      <td>...</td>
      <td>4.092076</td>
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
      <td>ARCF-Net + FedAvg + prototype sharing</td>
      <td>VAL</td>
      <td>global_equal</td>
      <td>0.973058</td>
      <td>0.971956</td>
      <td>0.972321</td>
      <td>0.971581</td>
      <td>0.973763</td>
      <td>0.973058</td>
      <td>0.972883</td>
      <td>...</td>
      <td>4.417884</td>
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
      <td>ARCF-Net + FedAvg + prototype sharing</td>
      <td>TEST</td>
      <td>ds1</td>
      <td>0.986726</td>
      <td>0.987069</td>
      <td>0.987061</td>
      <td>0.986952</td>
      <td>0.986955</td>
      <td>0.986726</td>
      <td>0.986727</td>
      <td>...</td>
      <td>4.886230</td>
      <td>0.987061</td>
      <td>0.982375</td>
      <td>0.982298</td>
      <td>0.987069</td>
      <td>0.995562</td>
      <td>0.995562</td>
      <td>0.029188</td>
      <td>0.610408</td>
      <td>0.021834</td>
    </tr>
    <tr>
      <th>4</th>
      <td>ARCF-Net + FedAvg + prototype sharing</td>
      <td>TEST</td>
      <td>ds2</td>
      <td>0.949239</td>
      <td>0.948853</td>
      <td>0.946868</td>
      <td>0.945575</td>
      <td>0.951805</td>
      <td>0.949239</td>
      <td>0.948403</td>
      <td>...</td>
      <td>4.209553</td>
      <td>0.946868</td>
      <td>0.933161</td>
      <td>0.931902</td>
      <td>0.948853</td>
      <td>0.983803</td>
      <td>0.983348</td>
      <td>0.019856</td>
      <td>0.484156</td>
      <td>0.080219</td>
    </tr>
    <tr>
      <th>5</th>
      <td>ARCF-Net + FedAvg + prototype sharing</td>
      <td>TEST</td>
      <td>global_equal</td>
      <td>0.967982</td>
      <td>0.967961</td>
      <td>0.966964</td>
      <td>0.966263</td>
      <td>0.969380</td>
      <td>0.967982</td>
      <td>0.967565</td>
      <td>...</td>
      <td>4.547891</td>
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


    
    Paper selection summary:
    - Best round: 12 | best_reward=1.0789
    - DS1 final strategy: single | names=['race_robust']
    - DS2 final strategy: single | names=['race_balanced']
    
    ======================================================================================================================
    STEP 13: PREPROCESSING VALIDATION (DS1 + DS2 VAL SAMPLE)
    ======================================================================================================================
    
    ----------------------------------------------------------------------------------------------------------------------
    Preprocessing validation summary (DS1 VAL sample)
    ----------------------------------------------------------------------------------------------------------------------



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
      <td>0.041596</td>
      <td>0.021720</td>
      <td>0.014947</td>
      <td>0.165471</td>
    </tr>
    <tr>
      <th>1</th>
      <td>edge_energy_after</td>
      <td>0.094055</td>
      <td>0.024375</td>
      <td>0.057067</td>
      <td>0.214411</td>
    </tr>
    <tr>
      <th>2</th>
      <td>entropy_before</td>
      <td>5.820408</td>
      <td>0.628255</td>
      <td>3.547314</td>
      <td>7.239670</td>
    </tr>
    <tr>
      <th>3</th>
      <td>entropy_after</td>
      <td>6.492075</td>
      <td>0.566734</td>
      <td>3.888517</td>
      <td>7.470275</td>
    </tr>
    <tr>
      <th>4</th>
      <td>contrast_before</td>
      <td>0.187640</td>
      <td>0.052597</td>
      <td>0.101468</td>
      <td>0.363759</td>
    </tr>
    <tr>
      <th>5</th>
      <td>contrast_after</td>
      <td>0.226861</td>
      <td>0.026633</td>
      <td>0.190911</td>
      <td>0.342708</td>
    </tr>
    <tr>
      <th>6</th>
      <td>edge_gain_ratio</td>
      <td>2.445604</td>
      <td>0.518596</td>
      <td>1.176741</td>
      <td>4.282522</td>
    </tr>
    <tr>
      <th>7</th>
      <td>entropy_delta</td>
      <td>0.671668</td>
      <td>0.235584</td>
      <td>0.230605</td>
      <td>1.413039</td>
    </tr>
    <tr>
      <th>8</th>
      <td>contrast_delta</td>
      <td>0.039221</td>
      <td>0.029151</td>
      <td>-0.022064</td>
      <td>0.100209</td>
    </tr>
  </tbody>
</table>
</div>


    
    ----------------------------------------------------------------------------------------------------------------------
    Preprocessing validation summary (DS2 VAL sample)
    ----------------------------------------------------------------------------------------------------------------------



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
      <td>0.041718</td>
      <td>0.015947</td>
      <td>0.012571</td>
      <td>0.121283</td>
    </tr>
    <tr>
      <th>1</th>
      <td>edge_energy_after</td>
      <td>0.116392</td>
      <td>0.023684</td>
      <td>0.043676</td>
      <td>0.167522</td>
    </tr>
    <tr>
      <th>2</th>
      <td>entropy_before</td>
      <td>5.811255</td>
      <td>0.822915</td>
      <td>3.222212</td>
      <td>7.570240</td>
    </tr>
    <tr>
      <th>3</th>
      <td>entropy_after</td>
      <td>6.664449</td>
      <td>0.785474</td>
      <td>3.612907</td>
      <td>7.741302</td>
    </tr>
    <tr>
      <th>4</th>
      <td>contrast_before</td>
      <td>0.190753</td>
      <td>0.055869</td>
      <td>0.098556</td>
      <td>0.341468</td>
    </tr>
    <tr>
      <th>5</th>
      <td>contrast_after</td>
      <td>0.250066</td>
      <td>0.022336</td>
      <td>0.198854</td>
      <td>0.318548</td>
    </tr>
    <tr>
      <th>6</th>
      <td>edge_gain_ratio</td>
      <td>3.015276</td>
      <td>0.781821</td>
      <td>1.308311</td>
      <td>5.732603</td>
    </tr>
    <tr>
      <th>7</th>
      <td>entropy_delta</td>
      <td>0.853194</td>
      <td>0.296688</td>
      <td>0.171062</td>
      <td>1.614185</td>
    </tr>
    <tr>
      <th>8</th>
      <td>contrast_delta</td>
      <td>0.059313</td>
      <td>0.037080</td>
      <td>-0.022945</td>
      <td>0.130606</td>
    </tr>
  </tbody>
</table>
</div>


    
    ======================================================================================================================
    STEP 14: PARAMETER EVOLUTION TABLE
    ======================================================================================================================
    
    ----------------------------------------------------------------------------------------------------------------------
    Mean selected preprocessing parameters over rounds
    ----------------------------------------------------------------------------------------------------------------------



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
      <td>1.046667</td>
      <td>0.280000</td>
      <td>5.300000</td>
      <td>2.616667</td>
      <td>4.333333</td>
      <td>0.130000</td>
      <td>0.796667</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1.008333</td>
      <td>0.243333</td>
      <td>4.800000</td>
      <td>2.516667</td>
      <td>4.333333</td>
      <td>0.111667</td>
      <td>0.770000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1.008333</td>
      <td>0.276667</td>
      <td>5.133333</td>
      <td>2.516667</td>
      <td>5.000000</td>
      <td>0.111667</td>
      <td>0.783333</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>0.993333</td>
      <td>0.243333</td>
      <td>4.583333</td>
      <td>2.400000</td>
      <td>5.666667</td>
      <td>0.098333</td>
      <td>0.763333</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>1.033333</td>
      <td>0.243333</td>
      <td>4.666667</td>
      <td>2.550000</td>
      <td>5.000000</td>
      <td>0.106667</td>
      <td>0.780000</td>
    </tr>
    <tr>
      <th>5</th>
      <td>6</td>
      <td>0.953333</td>
      <td>0.243333</td>
      <td>4.600000</td>
      <td>2.283333</td>
      <td>6.000000</td>
      <td>0.090000</td>
      <td>0.756667</td>
    </tr>
    <tr>
      <th>6</th>
      <td>7</td>
      <td>0.991667</td>
      <td>0.243333</td>
      <td>4.650000</td>
      <td>2.416667</td>
      <td>5.000000</td>
      <td>0.098333</td>
      <td>0.770000</td>
    </tr>
    <tr>
      <th>7</th>
      <td>8</td>
      <td>0.983333</td>
      <td>0.233333</td>
      <td>4.516667</td>
      <td>2.383333</td>
      <td>4.666667</td>
      <td>0.095000</td>
      <td>0.760000</td>
    </tr>
    <tr>
      <th>8</th>
      <td>9</td>
      <td>0.971667</td>
      <td>0.253333</td>
      <td>4.833333</td>
      <td>2.350000</td>
      <td>4.666667</td>
      <td>0.106667</td>
      <td>0.770000</td>
    </tr>
    <tr>
      <th>9</th>
      <td>10</td>
      <td>0.976667</td>
      <td>0.260000</td>
      <td>4.800000</td>
      <td>2.350000</td>
      <td>5.666667</td>
      <td>0.098333</td>
      <td>0.776667</td>
    </tr>
    <tr>
      <th>10</th>
      <td>11</td>
      <td>0.990000</td>
      <td>0.233333</td>
      <td>4.550000</td>
      <td>2.400000</td>
      <td>4.333333</td>
      <td>0.101667</td>
      <td>0.760000</td>
    </tr>
    <tr>
      <th>11</th>
      <td>12</td>
      <td>1.068333</td>
      <td>0.283333</td>
      <td>5.333333</td>
      <td>2.650000</td>
      <td>3.666667</td>
      <td>0.135000</td>
      <td>0.803333</td>
    </tr>
    <tr>
      <th>12</th>
      <td>13</td>
      <td>1.010000</td>
      <td>0.240000</td>
      <td>4.650000</td>
      <td>2.533333</td>
      <td>5.666667</td>
      <td>0.100000</td>
      <td>0.760000</td>
    </tr>
    <tr>
      <th>13</th>
      <td>14</td>
      <td>0.995000</td>
      <td>0.256667</td>
      <td>4.900000</td>
      <td>2.400000</td>
      <td>4.333333</td>
      <td>0.111667</td>
      <td>0.780000</td>
    </tr>
    <tr>
      <th>14</th>
      <td>15</td>
      <td>1.031667</td>
      <td>0.243333</td>
      <td>4.633333</td>
      <td>2.533333</td>
      <td>4.666667</td>
      <td>0.106667</td>
      <td>0.776667</td>
    </tr>
  </tbody>
</table>
</div>


    
    ======================================================================================================================
    STEP 15: SAVING ONLY TWO FILES (CHECKPOINT + ONE CSV)
    ======================================================================================================================
    ✅ Saved checkpoint: /kaggle/working/outputs/ARCFNet_RESNET50_FEDAVG_PROTOTYPE_checkpoint.pth
    ✅ Saved CSV (ALL outputs): /kaggle/working/outputs/ALL_OUTPUTS_AND_METRICS_RESNET50_FEDAVG_PROTOTYPE.csv
    
    DONE ✅
    Method: ARCF-Net = Adaptive RACE-FELCM with CRAF Fusion Network
    Backbone: Residual Network-50
    Federated variant: FedAvg + prototype sharing
    Best round: 12
    Adaptive clients => DS1=3, DS2=3, TOTAL=6
    Rounds completed: 15
    Global TEST acc: 0.9680
    Global TEST f1_macro: 0.9663
    DS1 TEST acc: 0.9867
    DS2 TEST acc: 0.9492
    DS1 final strategy: single | names=['race_robust']
    DS2 final strategy: single | names=['race_balanced']


**4. FedAvg + FedProx + prototype sharing**


```python
# ============================================================
# KAGGLE FULL SCRIPT
# TRUE FL + RL-UCB + RACE-FELCM + CRAF + ResNet-50
# ABLATION: FEDAVG + FEDPROX + PROTOTYPE SHARING
# METHOD ACRONYM: ARCF-Net
# FULL FORM: Adaptive RACE-FELCM with CRAF Fusion Network
# ------------------------------------------------------------
# KAGGLE-READY + SYMMETRIC DS1/DS2 IMPORTANCE
# - Uses BOTH datasets
# - Reads datasets from /kaggle/input automatically
# - Exact 15 FL rounds
# - Proper FL with FedAvg + FedProx + prototype sharing
# - RL-UCB for SHARED client-count planning + per-client preprocessing preset selection
# - Tune-aware theta probing before local training
# - Equal DS1 / DS2 importance in best-round selection and merged reporting
# - No plots
# - Saves checkpoint WITH full-process info for later replay / XAI
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

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    log_loss, confusion_matrix, roc_auc_score,
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

print("=" * 118)
print("TRUE FL + RL-UCB + RACE-FELCM + CRAF + ResNet-50")
print("METHOD: ARCF-Net (Adaptive RACE-FELCM with CRAF Fusion Network)")
print("ABLATION: FedAvg + FedProx + prototype sharing")
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

    # misc
    "quick_hash_subset_per_split": 300,
    "preproc_val_sample_n": 400,
    "calibration_bins": 12,
}

OUTDIR = "/kaggle/working/outputs" if IS_KAGGLE else "/content/outputs"
os.makedirs(OUTDIR, exist_ok=True)
MODEL_PATH = os.path.join(OUTDIR, "ARCFNet_RESNET50_FEDAVG_FEDPROX_PROTO_checkpoint.pth")
CSV_PATH   = os.path.join(OUTDIR, "ALL_OUTPUTS_AND_METRICS_RESNET50_FEDAVG_FEDPROX_PROTO.csv")

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
    "federated_variant": "FedAvg + FedProx + prototype sharing",
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
print("STEP 6: DATA LOADERS")
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
# STEP 9: LOSSES + FEDPROX + PROTOTYPE KNOWLEDGE SHARING
# ============================================================
print("\n" + "=" * 118)
print("STEP 9: LOSSES + FEDPROX + PROTOTYPE KNOWLEDGE SHARING")
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
    loss = torch.tensor(0.0, device=DEVICE)
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

    losses, ce_losses, proto_losses, prox_losses, correct, total = [], [], [], [], 0, 0
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
            prox = torch.tensor(0.0, device=DEVICE)
            if global_model is not None and CFG["fedprox_mu"] > 0:
                prox = 0.5 * CFG["fedprox_mu"] * fedprox_term(model, global_model)

            loss = ce + CFG["proto_lambda"] * proto + prox

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
        prox_losses.append(float(prox.item()))

        preds = logits.argmax(dim=1)
        correct += int((preds == y).sum().item())
        total += int(y.size(0))

    return {
        "loss": float(np.mean(losses)) if losses else np.nan,
        "ce_loss": float(np.mean(ce_losses)) if ce_losses else np.nan,
        "proto_loss": float(np.mean(proto_losses)) if proto_losses else np.nan,
        "fedprox_loss": float(np.mean(prox_losses)) if prox_losses else np.nan,
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
            "train_fedprox_loss": float(np.mean([x["fedprox_loss"] for x in train_logs])),
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
    {"setting": "ARCF-Net + FedAvg + FedProx + prototype sharing", "split": "VAL",  "dataset": "ds1",          **compact_metrics(val_ds1)},
    {"setting": "ARCF-Net + FedAvg + FedProx + prototype sharing", "split": "VAL",  "dataset": "ds2",          **compact_metrics(val_ds2)},
    {"setting": "ARCF-Net + FedAvg + FedProx + prototype sharing", "split": "VAL",  "dataset": "global_equal", **compact_metrics(val_global)},
    {"setting": "ARCF-Net + FedAvg + FedProx + prototype sharing", "split": "TEST", "dataset": "ds1",          **compact_metrics(test_ds1)},
    {"setting": "ARCF-Net + FedAvg + FedProx + prototype sharing", "split": "TEST", "dataset": "ds2",          **compact_metrics(test_ds2)},
    {"setting": "ARCF-Net + FedAvg + FedProx + prototype sharing", "split": "TEST", "dataset": "global_equal", **compact_metrics(test_global)},
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
# STEP 14: PARAMETER EVOLUTION TABLE
# ============================================================
print("\n" + "=" * 118)
print("STEP 14: PARAMETER EVOLUTION TABLE")
print("=" * 118)

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
for ccol in theta_cols:
    if ccol in loc_copy.columns:
        loc_copy[ccol] = pd.to_numeric(loc_copy[ccol], errors="coerce")

theta_evo = loc_copy.groupby("round")[theta_cols].mean(numeric_only=True).reset_index()
print_table(theta_evo, "Mean selected preprocessing parameters over rounds")
add_table_to_csv(theta_evo, "theta_evolution_mean")

# ============================================================
# STEP 15: SAVE CHECKPOINT + CSV
# ============================================================
print("\n" + "=" * 118)
print("STEP 15: SAVING ONLY TWO FILES (CHECKPOINT + ONE CSV)")
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
    "federated_variant": "FedAvg + FedProx + prototype sharing",

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
    "theta_evolution_mean": theta_evo.to_dict(orient="list"),
}

torch.save(checkpoint, MODEL_PATH)
print(f"✅ Saved checkpoint: {MODEL_PATH}")

all_df.to_csv(CSV_PATH, index=False)
print(f"✅ Saved CSV (ALL outputs): {CSV_PATH}")

print("\nDONE ✅")
print(f"Method: {METHOD_INFO['acronym']} = {METHOD_INFO['full_form']}")
print(f"Backbone: {METHOD_INFO['backbone_full_form']}")
print("Federated variant: FedAvg + FedProx + prototype sharing")
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
    ABLATION: FedAvg + FedProx + prototype sharing
    ======================================================================================================================
    ENV: KAGGLE | DEVICE: cuda | torch=2.9.0+cu126
    ======================================================================================================================
    
    ======================================================================================================================
    STEP 0: ACCESS DATASETS
    ======================================================================================================================
    Dataset-1 RAW root detected:
      /kaggle/input/datasets/orvile/pmram-bangladeshi-brain-cancer-mri-dataset/PMRAM Bangladeshi Brain Cancer - MRI Dataset/PMRAM Bangladeshi Brain Cancer - MRI Dataset/Raw Data/Raw
    Dataset-2 root detected:
      /kaggle/input/datasets/chubskuy/brain-tumor-image/Testing
    
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
    ds2: glioma -> glioma | 300 images
    ds2: meningioma -> meningioma | 306 images
    ds2: notumor -> notumor | 405 images
    ds2: pituitary -> pituitary | 300 images
    
    Dataset-1 images: 1505
    label
    glioma        373
    meningioma    363
    notumor       396
    pituitary     373
    Name: count, dtype: int64
    
    Dataset-2 images: 1311
    label
    glioma        300
    meningioma    306
    notumor       405
    pituitary     300
    Name: count, dtype: int64
    
    ======================================================================================================================
    STEP 2: TRAIN / VAL / TEST SPLIT (PER DATASET)
    ======================================================================================================================
    DS1 TRAIN: 1053 | VAL: 226 | TEST: 226
    DS2 TRAIN: 917 | VAL: 197 | TEST: 197
    
    ======================================================================================================================
    STEP 2.5: SANITY / LEAKAGE CHECKS
    ======================================================================================================================
    
    ----------------------------------------------------------------------------------------------------------------------
    Leakage / Sanity Summary — ds1
    ----------------------------------------------------------------------------------------------------------------------



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
      <td>5</td>
      <td>5</td>
      <td>6</td>
      <td>298</td>
      <td>222</td>
      <td>224</td>
    </tr>
  </tbody>
</table>
</div>


    
    ----------------------------------------------------------------------------------------------------------------------
    Leakage / Sanity Summary — ds2
    ----------------------------------------------------------------------------------------------------------------------



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
      <td>917</td>
      <td>197</td>
      <td>197</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>300</td>
      <td>194</td>
      <td>197</td>
    </tr>
  </tbody>
</table>
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
      <td>0.755612</td>
      <td>0.767340</td>
      <td>0.761476</td>
      <td>0.761476</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>4</td>
      <td>0.679705</td>
      <td>0.635628</td>
      <td>0.657667</td>
      <td>0.657667</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>5</td>
      <td>0.721049</td>
      <td>0.775174</td>
      <td>0.748111</td>
      <td>0.748111</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>3</td>
      <td>0.841039</td>
      <td>0.726403</td>
      <td>0.783721</td>
      <td>0.772599</td>
      <td>2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>5</td>
      <td>0.688321</td>
      <td>0.701388</td>
      <td>0.694855</td>
      <td>0.721483</td>
      <td>2</td>
    </tr>
    <tr>
      <th>5</th>
      <td>6</td>
      <td>4</td>
      <td>0.665267</td>
      <td>0.691320</td>
      <td>0.678293</td>
      <td>0.667980</td>
      <td>2</td>
    </tr>
    <tr>
      <th>6</th>
      <td>7</td>
      <td>3</td>
      <td>0.766538</td>
      <td>0.693916</td>
      <td>0.730227</td>
      <td>0.758475</td>
      <td>3</td>
    </tr>
    <tr>
      <th>7</th>
      <td>8</td>
      <td>5</td>
      <td>0.677792</td>
      <td>0.707763</td>
      <td>0.692778</td>
      <td>0.711915</td>
      <td>3</td>
    </tr>
    <tr>
      <th>8</th>
      <td>9</td>
      <td>4</td>
      <td>0.715564</td>
      <td>0.681741</td>
      <td>0.698653</td>
      <td>0.678204</td>
      <td>3</td>
    </tr>
    <tr>
      <th>9</th>
      <td>10</td>
      <td>3</td>
      <td>0.734619</td>
      <td>0.707902</td>
      <td>0.721261</td>
      <td>0.749171</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>


    
    ======================================================================================================================
    STEP 5: FINAL NON-IID CLIENT PARTITIONING
    ======================================================================================================================
    
    ======================================================================================================================
    STEP 6: DATA LOADERS
    ======================================================================================================================
    ds1 | client_0 | train=286 | tune=45 | val=40
    ds1 | client_1 | train=257 | tune=41 | val=36
    ds1 | client_2 | train=269 | tune=42 | val=37
    ds2 | client_3 | train=100 | tune=16 | val=14
    ds2 | client_4 | train=342 | tune=54 | val=47
    ds2 | client_5 | train=265 | tune=42 | val=37
    
    ----------------------------------------------------------------------------------------------------------------------
    Adaptive client class distribution
    ----------------------------------------------------------------------------------------------------------------------



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
      <td>286</td>
      <td>45</td>
      <td>40</td>
      <td>40</td>
      <td>33</td>
      <td>166</td>
      <td>47</td>
    </tr>
    <tr>
      <th>1</th>
      <td>client_1</td>
      <td>ds1</td>
      <td>257</td>
      <td>41</td>
      <td>36</td>
      <td>135</td>
      <td>4</td>
      <td>31</td>
      <td>87</td>
    </tr>
    <tr>
      <th>2</th>
      <td>client_2</td>
      <td>ds1</td>
      <td>269</td>
      <td>42</td>
      <td>37</td>
      <td>25</td>
      <td>159</td>
      <td>17</td>
      <td>68</td>
    </tr>
    <tr>
      <th>3</th>
      <td>client_3</td>
      <td>ds2</td>
      <td>100</td>
      <td>16</td>
      <td>14</td>
      <td>74</td>
      <td>18</td>
      <td>4</td>
      <td>4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>client_4</td>
      <td>ds2</td>
      <td>342</td>
      <td>54</td>
      <td>47</td>
      <td>5</td>
      <td>121</td>
      <td>154</td>
      <td>62</td>
    </tr>
    <tr>
      <th>5</th>
      <td>client_5</td>
      <td>ds2</td>
      <td>265</td>
      <td>42</td>
      <td>37</td>
      <td>82</td>
      <td>25</td>
      <td>61</td>
      <td>97</td>
    </tr>
  </tbody>
</table>
</div>


    Augmentation: ON ✅
    Preprocessing: ON ✅
    Total adaptive clients: 6
    
    ======================================================================================================================
    STEP 7: NOVEL PREPROCESSING — RACE-FELCM
    ======================================================================================================================
    
    ======================================================================================================================
    STEP 8: MODEL — ResNet-50 + CRAF Fusion
    ======================================================================================================================
    Backbone: ResNet-50 | pretrained_loaded=True
    Total params: 25,790,855
    Trainable params: 2,282,823 (8.85%)
    
    ======================================================================================================================
    STEP 9: LOSSES + FEDPROX + PROTOTYPE KNOWLEDGE SHARING
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
    Client 0 (ds1) | train_acc=0.6923 | val_acc=0.8250 | val_f1=0.7109 | val_auc=0.9318 | reward=0.7394 | theta=race_edge_plus (g=1.02, a=0.32, b=6.00, t=2.60, k=3, eg=0.15, mix=0.84)
    Client 1 (ds1) | train_acc=0.7296 | val_acc=0.7778 | val_f1=0.6097 | val_auc=0.9586 | reward=0.6517 | theta=race_focus (g=1.10, a=0.36, b=6.40, t=2.70, k=3, eg=0.17, mix=0.86)
    Client 2 (ds1) | train_acc=0.6970 | val_acc=0.7297 | val_f1=0.6696 | val_auc=0.9348 | reward=0.6846 | theta=race_focus (g=1.10, a=0.36, b=6.40, t=2.70, k=3, eg=0.17, mix=0.86)
    Client 3 (ds2) | train_acc=0.5600 | val_acc=0.9286 | val_f1=0.6508 | val_auc=nan | reward=0.7202 | theta=race_robust (g=1.08, a=0.22, b=4.50, t=2.80, k=5, eg=0.11, mix=0.76)
    Client 4 (ds2) | train_acc=0.6798 | val_acc=0.8511 | val_f1=0.6503 | val_auc=0.9631 | reward=0.7005 | theta=race_smoothmix (g=0.90, a=0.20, b=4.00, t=2.10, k=7, eg=0.07, mix=0.70)
    Client 5 (ds2) | train_acc=0.6358 | val_acc=0.7568 | val_f1=0.6264 | val_auc=0.9426 | reward=0.6590 | theta=race_robust (g=1.08, a=0.22, b=4.50, t=2.80, k=5, eg=0.11, mix=0.76)
    
    ----------------------------------------------------------------------------------------------------------------------
    GLOBAL VAL (Round 1) | global_acc=0.8009 | global_f1=0.6541 | ds1_acc=0.7788 | ds1_f1=0.6651 | ds2_acc=0.8265 | ds2_f1=0.6414 | reward=0.7937 | round_time=57.8s
    ----------------------------------------------------------------------------------------------------------------------
    
    ======================================================================================================================
    ROUND 2/15 | selected=[0, 1, 2, 3, 4, 5]
    ======================================================================================================================
    Client 0 (ds1) | train_acc=0.8724 | val_acc=0.8000 | val_f1=0.6766 | val_auc=0.9748 | reward=0.7075 | theta=race_robust (g=1.08, a=0.22, b=4.50, t=2.80, k=5, eg=0.11, mix=0.76)
    Client 1 (ds1) | train_acc=0.9047 | val_acc=0.8611 | val_f1=0.6332 | val_auc=0.9784 | reward=0.6902 | theta=race_edge_plus (g=1.02, a=0.32, b=6.00, t=2.60, k=3, eg=0.15, mix=0.84)
    Client 2 (ds1) | train_acc=0.8532 | val_acc=0.8649 | val_f1=0.7880 | val_auc=0.9676 | reward=0.8072 | theta=race_edge_plus (g=1.02, a=0.32, b=6.00, t=2.60, k=3, eg=0.15, mix=0.84)
    Client 3 (ds2) | train_acc=0.7000 | val_acc=0.8571 | val_f1=0.5397 | val_auc=nan | reward=0.6190 | theta=race_soft (g=0.95, a=0.18, b=3.80, t=2.20, k=3, eg=0.08, mix=0.72)
    Client 4 (ds2) | train_acc=0.8041 | val_acc=0.8723 | val_f1=0.6567 | val_auc=0.9536 | reward=0.7106 | theta=race_robust (g=1.08, a=0.22, b=4.50, t=2.80, k=5, eg=0.11, mix=0.76)
    Client 5 (ds2) | train_acc=0.8151 | val_acc=0.8108 | val_f1=0.7082 | val_auc=0.9557 | reward=0.7338 | theta=race_soft (g=0.95, a=0.18, b=3.80, t=2.20, k=3, eg=0.08, mix=0.72)
    
    ----------------------------------------------------------------------------------------------------------------------
    GLOBAL VAL (Round 2) | global_acc=0.8436 | global_f1=0.6808 | ds1_acc=0.8407 | ds1_f1=0.6993 | ds2_acc=0.8469 | ds2_f1=0.6594 | reward=0.8264 | round_time=53.7s
    ----------------------------------------------------------------------------------------------------------------------
    
    ======================================================================================================================
    ROUND 3/15 | selected=[0, 1, 2, 3, 4, 5]
    ======================================================================================================================
    Client 0 (ds1) | train_acc=0.8077 | val_acc=0.8250 | val_f1=0.7212 | val_auc=0.9450 | reward=0.7471 | theta=race_focus (g=1.10, a=0.36, b=6.40, t=2.70, k=3, eg=0.17, mix=0.86)
    Client 1 (ds1) | train_acc=0.8774 | val_acc=0.9722 | val_f1=0.7436 | val_auc=0.9629 | reward=0.8007 | theta=race_texture (g=0.92, a=0.34, b=5.80, t=2.30, k=7, eg=0.10, mix=0.80)
    Client 2 (ds1) | train_acc=0.7881 | val_acc=0.9459 | val_f1=0.9103 | val_auc=0.9934 | reward=0.9192 | theta=race_texture (g=0.92, a=0.34, b=5.80, t=2.30, k=7, eg=0.10, mix=0.80)
    Client 3 (ds2) | train_acc=0.7550 | val_acc=0.6429 | val_f1=0.6742 | val_auc=nan | reward=0.6664 | theta=race_robust (g=1.08, a=0.22, b=4.50, t=2.80, k=5, eg=0.11, mix=0.76)
    Client 4 (ds2) | train_acc=0.7982 | val_acc=0.9149 | val_f1=0.6773 | val_auc=0.9159 | reward=0.7367 | theta=race_robust (g=1.08, a=0.22, b=4.50, t=2.80, k=5, eg=0.11, mix=0.76)
    Client 5 (ds2) | train_acc=0.8491 | val_acc=0.8649 | val_f1=0.8226 | val_auc=0.9756 | reward=0.8332 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    
    ----------------------------------------------------------------------------------------------------------------------
    GLOBAL VAL (Round 3) | global_acc=0.8863 | global_f1=0.7631 | ds1_acc=0.9115 | ds1_f1=0.7903 | ds2_acc=0.8571 | ds2_f1=0.7317 | reward=0.9063 | round_time=61.6s
    ----------------------------------------------------------------------------------------------------------------------
    
    ======================================================================================================================
    ROUND 4/15 | selected=[0, 1, 2, 3, 4, 5]
    ======================================================================================================================
    Client 0 (ds1) | train_acc=0.8601 | val_acc=0.8250 | val_f1=0.6675 | val_auc=0.9616 | reward=0.7069 | theta=race_sharp (g=1.04, a=0.30, b=5.20, t=2.50, k=5, eg=0.12, mix=0.82)
    Client 1 (ds1) | train_acc=0.8852 | val_acc=0.8889 | val_f1=0.6780 | val_auc=0.9772 | reward=0.7307 | theta=race_robust (g=1.08, a=0.22, b=4.50, t=2.80, k=5, eg=0.11, mix=0.76)
    Client 2 (ds1) | train_acc=0.8309 | val_acc=0.8378 | val_f1=0.8461 | val_auc=0.9844 | reward=0.8440 | theta=race_sharp (g=1.04, a=0.30, b=5.20, t=2.50, k=5, eg=0.12, mix=0.82)
    Client 3 (ds2) | train_acc=0.7700 | val_acc=0.7857 | val_f1=0.7524 | val_auc=nan | reward=0.7607 | theta=race_smoothmix (g=0.90, a=0.20, b=4.00, t=2.10, k=7, eg=0.07, mix=0.70)
    Client 4 (ds2) | train_acc=0.8670 | val_acc=0.9574 | val_f1=0.9540 | val_auc=0.9858 | reward=0.9549 | theta=race_smoothmix (g=0.90, a=0.20, b=4.00, t=2.10, k=7, eg=0.07, mix=0.70)
    Client 5 (ds2) | train_acc=0.8132 | val_acc=0.8919 | val_f1=0.8435 | val_auc=0.9775 | reward=0.8556 | theta=race_smoothmix (g=0.90, a=0.20, b=4.00, t=2.10, k=7, eg=0.07, mix=0.70)
    
    ----------------------------------------------------------------------------------------------------------------------
    GLOBAL VAL (Round 4) | global_acc=0.8768 | global_f1=0.8009 | ds1_acc=0.8496 | ds1_f1=0.7293 | ds2_acc=0.9082 | ds2_f1=0.8835 | reward=0.9384 | round_time=61.3s
    ----------------------------------------------------------------------------------------------------------------------
    
    ======================================================================================================================
    ROUND 5/15 | selected=[0, 1, 2, 3, 4, 5]
    ======================================================================================================================
    Client 0 (ds1) | train_acc=0.9108 | val_acc=0.8500 | val_f1=0.7725 | val_auc=0.9647 | reward=0.7919 | theta=race_texture (g=0.92, a=0.34, b=5.80, t=2.30, k=7, eg=0.10, mix=0.80)
    Client 1 (ds1) | train_acc=0.9455 | val_acc=0.9444 | val_f1=0.7091 | val_auc=0.9977 | reward=0.7679 | theta=race_sharp (g=1.04, a=0.30, b=5.20, t=2.50, k=5, eg=0.12, mix=0.82)
    Client 2 (ds1) | train_acc=0.9312 | val_acc=0.9730 | val_f1=0.9664 | val_auc=1.0000 | reward=0.9680 | theta=race_texture (g=0.92, a=0.34, b=5.80, t=2.30, k=7, eg=0.10, mix=0.80)
    Client 3 (ds2) | train_acc=0.6900 | val_acc=0.9286 | val_f1=0.6508 | val_auc=nan | reward=0.7202 | theta=race_smoothmix (g=0.90, a=0.20, b=4.00, t=2.10, k=7, eg=0.07, mix=0.70)
    Client 4 (ds2) | train_acc=0.8523 | val_acc=0.9574 | val_f1=0.7194 | val_auc=1.0000 | reward=0.7789 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    Client 5 (ds2) | train_acc=0.8566 | val_acc=0.7568 | val_f1=0.6156 | val_auc=0.9698 | reward=0.6509 | theta=race_smoothmix (g=0.90, a=0.20, b=4.00, t=2.10, k=7, eg=0.07, mix=0.70)
    
    ----------------------------------------------------------------------------------------------------------------------
    GLOBAL VAL (Round 5) | global_acc=0.9005 | global_f1=0.7483 | ds1_acc=0.9204 | ds1_f1=0.8158 | ds2_acc=0.8776 | ds2_f1=0.6704 | reward=0.8904 | round_time=61.4s
    ----------------------------------------------------------------------------------------------------------------------
    
    ======================================================================================================================
    ROUND 6/15 | selected=[0, 1, 2, 3, 4, 5]
    ======================================================================================================================
    Client 0 (ds1) | train_acc=0.9423 | val_acc=0.9000 | val_f1=0.8263 | val_auc=0.9879 | reward=0.8447 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    Client 1 (ds1) | train_acc=0.9611 | val_acc=0.9444 | val_f1=0.7018 | val_auc=0.9770 | reward=0.7625 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    Client 2 (ds1) | train_acc=0.9387 | val_acc=0.9730 | val_f1=0.9664 | val_auc=0.9992 | reward=0.9680 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    Client 3 (ds2) | train_acc=0.8750 | val_acc=1.0000 | val_f1=1.0000 | val_auc=nan | reward=1.0000 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    Client 4 (ds2) | train_acc=0.9254 | val_acc=0.9362 | val_f1=0.7083 | val_auc=0.9924 | reward=0.7653 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    Client 5 (ds2) | train_acc=0.8604 | val_acc=0.8378 | val_f1=0.7696 | val_auc=0.9649 | reward=0.7866 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    
    ----------------------------------------------------------------------------------------------------------------------
    GLOBAL VAL (Round 6) | global_acc=0.9242 | global_f1=0.8049 | ds1_acc=0.9381 | ds1_f1=0.8325 | ds2_acc=0.9082 | ds2_f1=0.7731 | reward=0.9539 | round_time=61.2s
    ----------------------------------------------------------------------------------------------------------------------
    
    ======================================================================================================================
    ROUND 7/15 | selected=[0, 1, 2, 3, 4, 5]
    ======================================================================================================================
    Client 0 (ds1) | train_acc=0.9580 | val_acc=0.8000 | val_f1=0.6470 | val_auc=0.9617 | reward=0.6853 | theta=race_texture (g=0.92, a=0.34, b=5.80, t=2.30, k=7, eg=0.10, mix=0.80)
    Client 1 (ds1) | train_acc=0.9397 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | reward=1.0000 | theta=race_texture (g=0.92, a=0.34, b=5.80, t=2.30, k=7, eg=0.10, mix=0.80)
    Client 2 (ds1) | train_acc=0.9684 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | reward=1.0000 | theta=race_robust (g=1.08, a=0.22, b=4.50, t=2.80, k=5, eg=0.11, mix=0.76)
    Client 3 (ds2) | train_acc=0.9000 | val_acc=0.9286 | val_f1=0.9175 | val_auc=nan | reward=0.9202 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    Client 4 (ds2) | train_acc=0.9284 | val_acc=0.9787 | val_f1=0.9866 | val_auc=1.0000 | reward=0.9846 | theta=race_soft (g=0.95, a=0.18, b=3.80, t=2.20, k=3, eg=0.08, mix=0.72)
    Client 5 (ds2) | train_acc=0.9302 | val_acc=0.9189 | val_f1=0.8792 | val_auc=0.9573 | reward=0.8891 | theta=race_soft (g=0.95, a=0.18, b=3.80, t=2.20, k=3, eg=0.08, mix=0.72)
    
    ----------------------------------------------------------------------------------------------------------------------
    GLOBAL VAL (Round 7) | global_acc=0.9384 | global_f1=0.9034 | ds1_acc=0.9292 | ds1_f1=0.8750 | ds2_acc=0.9490 | ds2_f1=0.9362 | reward=1.0473 | round_time=61.2s
    ----------------------------------------------------------------------------------------------------------------------
    
    ======================================================================================================================
    ROUND 8/15 | selected=[0, 1, 2, 3, 4, 5]
    ======================================================================================================================
    Client 0 (ds1) | train_acc=0.9545 | val_acc=0.9750 | val_f1=0.9495 | val_auc=1.0000 | reward=0.9559 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    Client 1 (ds1) | train_acc=0.9553 | val_acc=0.9444 | val_f1=0.7018 | val_auc=0.9985 | reward=0.7625 | theta=race_sharp (g=1.04, a=0.30, b=5.20, t=2.50, k=5, eg=0.12, mix=0.82)
    Client 2 (ds1) | train_acc=0.9480 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | reward=1.0000 | theta=race_robust (g=1.08, a=0.22, b=4.50, t=2.80, k=5, eg=0.11, mix=0.76)
    Client 3 (ds2) | train_acc=0.8100 | val_acc=0.8571 | val_f1=0.7889 | val_auc=nan | reward=0.8060 | theta=race_soft (g=0.95, a=0.18, b=3.80, t=2.20, k=3, eg=0.08, mix=0.72)
    Client 4 (ds2) | train_acc=0.9591 | val_acc=0.9362 | val_f1=0.9402 | val_auc=0.9955 | reward=0.9392 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    Client 5 (ds2) | train_acc=0.9038 | val_acc=0.9459 | val_f1=0.9167 | val_auc=0.9883 | reward=0.9240 | theta=race_robust (g=1.08, a=0.22, b=4.50, t=2.80, k=5, eg=0.11, mix=0.76)
    
    ----------------------------------------------------------------------------------------------------------------------
    GLOBAL VAL (Round 8) | global_acc=0.9526 | global_f1=0.8976 | ds1_acc=0.9735 | ds1_f1=0.8871 | ds2_acc=0.9286 | ds2_f1=0.9097 | reward=1.0479 | round_time=61.3s
    ----------------------------------------------------------------------------------------------------------------------
    
    ======================================================================================================================
    ROUND 9/15 | selected=[0, 1, 2, 3, 4, 5]
    ======================================================================================================================
    Client 0 (ds1) | train_acc=0.9685 | val_acc=0.9250 | val_f1=0.8500 | val_auc=0.9797 | reward=0.8688 | theta=race_focus (g=1.10, a=0.36, b=6.40, t=2.70, k=3, eg=0.17, mix=0.86)
    Client 1 (ds1) | train_acc=0.9981 | val_acc=0.9722 | val_f1=0.9655 | val_auc=0.9957 | reward=0.9672 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    Client 2 (ds1) | train_acc=0.9554 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | reward=1.0000 | theta=race_sharp (g=1.04, a=0.30, b=5.20, t=2.50, k=5, eg=0.12, mix=0.82)
    Client 3 (ds2) | train_acc=0.9750 | val_acc=0.9286 | val_f1=0.9175 | val_auc=nan | reward=0.9202 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    Client 4 (ds2) | train_acc=0.9430 | val_acc=0.9149 | val_f1=0.6728 | val_auc=0.9878 | reward=0.7333 | theta=race_soft (g=0.95, a=0.18, b=3.80, t=2.20, k=3, eg=0.08, mix=0.72)
    Client 5 (ds2) | train_acc=0.9396 | val_acc=0.8919 | val_f1=0.8369 | val_auc=0.9785 | reward=0.8507 | theta=race_soft (g=0.95, a=0.18, b=3.80, t=2.20, k=3, eg=0.08, mix=0.72)
    
    ----------------------------------------------------------------------------------------------------------------------
    GLOBAL VAL (Round 9) | global_acc=0.9384 | global_f1=0.8587 | ds1_acc=0.9646 | ds1_f1=0.9359 | ds2_acc=0.9082 | ds2_f1=0.7697 | reward=0.9943 | round_time=60.7s
    ----------------------------------------------------------------------------------------------------------------------
    
    ======================================================================================================================
    ROUND 10/15 | selected=[0, 1, 2, 3, 4, 5]
    ======================================================================================================================
    Client 0 (ds1) | train_acc=0.9458 | val_acc=0.9750 | val_f1=0.9495 | val_auc=0.9946 | reward=0.9559 | theta=race_edge_plus (g=1.02, a=0.32, b=6.00, t=2.60, k=3, eg=0.15, mix=0.84)
    Client 1 (ds1) | train_acc=0.9553 | val_acc=0.9722 | val_f1=0.7436 | val_auc=0.9929 | reward=0.8007 | theta=race_robust (g=1.08, a=0.22, b=4.50, t=2.80, k=5, eg=0.11, mix=0.76)
    Client 2 (ds1) | train_acc=0.9628 | val_acc=0.9730 | val_f1=0.9442 | val_auc=1.0000 | reward=0.9514 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    Client 3 (ds2) | train_acc=0.9850 | val_acc=0.9286 | val_f1=0.9175 | val_auc=nan | reward=0.9202 | theta=race_smoothmix (g=0.90, a=0.20, b=4.00, t=2.10, k=7, eg=0.07, mix=0.70)
    Client 4 (ds2) | train_acc=0.9708 | val_acc=0.9362 | val_f1=0.7177 | val_auc=0.9763 | reward=0.7723 | theta=race_soft (g=0.95, a=0.18, b=3.80, t=2.20, k=3, eg=0.08, mix=0.72)
    Client 5 (ds2) | train_acc=0.9472 | val_acc=0.9189 | val_f1=0.8722 | val_auc=0.9635 | reward=0.8839 | theta=race_robust (g=1.08, a=0.22, b=4.50, t=2.80, k=5, eg=0.11, mix=0.76)
    
    ----------------------------------------------------------------------------------------------------------------------
    GLOBAL VAL (Round 10) | global_acc=0.9526 | global_f1=0.8461 | ds1_acc=0.9735 | ds1_f1=0.8822 | ds2_acc=0.9286 | ds2_f1=0.8046 | reward=0.9956 | round_time=61.0s
    ----------------------------------------------------------------------------------------------------------------------
    
    ======================================================================================================================
    ROUND 11/15 | selected=[0, 1, 2, 3, 4, 5]
    ======================================================================================================================
    Client 0 (ds1) | train_acc=0.9808 | val_acc=0.9000 | val_f1=0.8886 | val_auc=0.9834 | reward=0.8914 | theta=race_robust (g=1.08, a=0.22, b=4.50, t=2.80, k=5, eg=0.11, mix=0.76)
    Client 1 (ds1) | train_acc=0.9825 | val_acc=0.9167 | val_f1=0.6819 | val_auc=0.9953 | reward=0.7406 | theta=race_edge_plus (g=1.02, a=0.32, b=6.00, t=2.60, k=3, eg=0.15, mix=0.84)
    Client 2 (ds1) | train_acc=0.9628 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | reward=1.0000 | theta=race_edge_plus (g=1.02, a=0.32, b=6.00, t=2.60, k=3, eg=0.15, mix=0.84)
    Client 3 (ds2) | train_acc=0.9900 | val_acc=0.9286 | val_f1=0.9175 | val_auc=nan | reward=0.9202 | theta=race_soft (g=0.95, a=0.18, b=3.80, t=2.20, k=3, eg=0.08, mix=0.72)
    Client 4 (ds2) | train_acc=0.9561 | val_acc=0.9149 | val_f1=0.6847 | val_auc=0.9977 | reward=0.7422 | theta=race_smoothmix (g=0.90, a=0.20, b=4.00, t=2.10, k=7, eg=0.07, mix=0.70)
    Client 5 (ds2) | train_acc=0.9547 | val_acc=0.8919 | val_f1=0.8600 | val_auc=0.9702 | reward=0.8680 | theta=race_smoothmix (g=0.90, a=0.20, b=4.00, t=2.10, k=7, eg=0.07, mix=0.70)
    
    ----------------------------------------------------------------------------------------------------------------------
    GLOBAL VAL (Round 11) | global_acc=0.9242 | global_f1=0.8243 | ds1_acc=0.9381 | ds1_f1=0.8592 | ds2_acc=0.9082 | ds2_f1=0.7841 | reward=0.9693 | round_time=61.3s
    ----------------------------------------------------------------------------------------------------------------------
    
    ======================================================================================================================
    ROUND 12/15 | selected=[0, 1, 2, 3, 4, 5]
    ======================================================================================================================
    Client 0 (ds1) | train_acc=0.9738 | val_acc=0.8750 | val_f1=0.8112 | val_auc=0.9890 | reward=0.8271 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    Client 1 (ds1) | train_acc=0.9961 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | reward=1.0000 | theta=race_focus (g=1.10, a=0.36, b=6.40, t=2.70, k=3, eg=0.17, mix=0.86)
    Client 2 (ds1) | train_acc=0.9517 | val_acc=0.9459 | val_f1=0.9103 | val_auc=1.0000 | reward=0.9192 | theta=race_focus (g=1.10, a=0.36, b=6.40, t=2.70, k=3, eg=0.17, mix=0.86)
    Client 3 (ds2) | train_acc=0.9750 | val_acc=0.9286 | val_f1=0.9175 | val_auc=nan | reward=0.9202 | theta=race_robust (g=1.08, a=0.22, b=4.50, t=2.80, k=5, eg=0.11, mix=0.76)
    Client 4 (ds2) | train_acc=0.9605 | val_acc=0.9787 | val_f1=0.9762 | val_auc=0.9892 | reward=0.9768 | theta=race_robust (g=1.08, a=0.22, b=4.50, t=2.80, k=5, eg=0.11, mix=0.76)
    Client 5 (ds2) | train_acc=0.9566 | val_acc=0.9189 | val_f1=0.8792 | val_auc=0.9714 | reward=0.8891 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    
    ----------------------------------------------------------------------------------------------------------------------
    GLOBAL VAL (Round 12) | global_acc=0.9431 | global_f1=0.9165 | ds1_acc=0.9381 | ds1_f1=0.9038 | ds2_acc=0.9490 | ds2_f1=0.9312 | reward=1.0608 | round_time=61.2s
    ----------------------------------------------------------------------------------------------------------------------
    
    ======================================================================================================================
    ROUND 13/15 | selected=[0, 1, 2, 3, 4, 5]
    ======================================================================================================================
    Client 0 (ds1) | train_acc=0.9895 | val_acc=0.9250 | val_f1=0.8500 | val_auc=0.9970 | reward=0.8688 | theta=race_sharp (g=1.04, a=0.30, b=5.20, t=2.50, k=5, eg=0.12, mix=0.82)
    Client 1 (ds1) | train_acc=0.9689 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | reward=1.0000 | theta=race_texture (g=0.92, a=0.34, b=5.80, t=2.30, k=7, eg=0.10, mix=0.80)
    Client 2 (ds1) | train_acc=0.9777 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | reward=1.0000 | theta=race_robust (g=1.08, a=0.22, b=4.50, t=2.80, k=5, eg=0.11, mix=0.76)
    Client 3 (ds2) | train_acc=0.9450 | val_acc=1.0000 | val_f1=1.0000 | val_auc=nan | reward=1.0000 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    Client 4 (ds2) | train_acc=0.9766 | val_acc=0.9787 | val_f1=0.9762 | val_auc=0.9964 | reward=0.9768 | theta=race_soft (g=0.95, a=0.18, b=3.80, t=2.20, k=3, eg=0.08, mix=0.72)
    Client 5 (ds2) | train_acc=0.9736 | val_acc=0.9459 | val_f1=0.9108 | val_auc=0.9884 | reward=0.9196 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    
    ----------------------------------------------------------------------------------------------------------------------
    GLOBAL VAL (Round 13) | global_acc=0.9716 | global_f1=0.9506 | ds1_acc=0.9735 | ds1_f1=0.9469 | ds2_acc=0.9694 | ds2_f1=0.9549 | reward=1.0991 | round_time=61.4s
    ----------------------------------------------------------------------------------------------------------------------
    
    ======================================================================================================================
    ROUND 14/15 | selected=[0, 1, 2, 3, 4, 5]
    ======================================================================================================================
    Client 0 (ds1) | train_acc=0.9773 | val_acc=0.9500 | val_f1=0.9169 | val_auc=0.9964 | reward=0.9252 | theta=race_edge_plus (g=1.02, a=0.32, b=6.00, t=2.60, k=3, eg=0.15, mix=0.84)
    Client 1 (ds1) | train_acc=0.9747 | val_acc=0.9722 | val_f1=0.9579 | val_auc=1.0000 | reward=0.9615 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    Client 2 (ds1) | train_acc=0.9833 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | reward=1.0000 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    Client 3 (ds2) | train_acc=0.9850 | val_acc=0.9286 | val_f1=0.9175 | val_auc=nan | reward=0.9202 | theta=race_smoothmix (g=0.90, a=0.20, b=4.00, t=2.10, k=7, eg=0.07, mix=0.70)
    Client 4 (ds2) | train_acc=0.9737 | val_acc=0.9787 | val_f1=0.9762 | val_auc=1.0000 | reward=0.9768 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    Client 5 (ds2) | train_acc=0.9642 | val_acc=0.8649 | val_f1=0.8185 | val_auc=0.9695 | reward=0.8301 | theta=race_smoothmix (g=0.90, a=0.20, b=4.00, t=2.10, k=7, eg=0.07, mix=0.70)
    
    ----------------------------------------------------------------------------------------------------------------------
    GLOBAL VAL (Round 14) | global_acc=0.9526 | global_f1=0.9345 | ds1_acc=0.9735 | ds1_f1=0.9572 | ds2_acc=0.9286 | ds2_f1=0.9083 | reward=1.0743 | round_time=61.3s
    ----------------------------------------------------------------------------------------------------------------------
    
    ======================================================================================================================
    ROUND 15/15 | selected=[0, 1, 2, 3, 4, 5]
    ======================================================================================================================
    Client 0 (ds1) | train_acc=0.9825 | val_acc=0.9000 | val_f1=0.8507 | val_auc=0.9962 | reward=0.8630 | theta=race_focus (g=1.10, a=0.36, b=6.40, t=2.70, k=3, eg=0.17, mix=0.86)
    Client 1 (ds1) | train_acc=0.9883 | val_acc=0.9722 | val_f1=0.9579 | val_auc=1.0000 | reward=0.9615 | theta=race_focus (g=1.10, a=0.36, b=6.40, t=2.70, k=3, eg=0.17, mix=0.86)
    Client 2 (ds1) | train_acc=0.9907 | val_acc=0.9730 | val_f1=0.9664 | val_auc=0.9992 | reward=0.9680 | theta=race_texture (g=0.92, a=0.34, b=5.80, t=2.30, k=7, eg=0.10, mix=0.80)
    Client 3 (ds2) | train_acc=1.0000 | val_acc=0.9286 | val_f1=0.9175 | val_auc=nan | reward=0.9202 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    Client 4 (ds2) | train_acc=0.9810 | val_acc=0.9787 | val_f1=0.9762 | val_auc=0.9982 | reward=0.9768 | theta=race_robust (g=1.08, a=0.22, b=4.50, t=2.80, k=5, eg=0.11, mix=0.76)
    Client 5 (ds2) | train_acc=0.9792 | val_acc=0.8919 | val_f1=0.8583 | val_auc=0.9883 | reward=0.8667 | theta=race_soft (g=0.95, a=0.18, b=3.80, t=2.20, k=3, eg=0.08, mix=0.72)
    
    ----------------------------------------------------------------------------------------------------------------------
    GLOBAL VAL (Round 15) | global_acc=0.9431 | global_f1=0.9230 | ds1_acc=0.9469 | ds1_f1=0.9227 | ds2_acc=0.9388 | ds2_f1=0.9233 | reward=1.0670 | round_time=61.1s
    ----------------------------------------------------------------------------------------------------------------------
    
    ======================================================================================================================
    TRAINING COMPLETE ✅ | total_time=907.9s | best_round=13 | best_reward=1.0991
    ======================================================================================================================
    
    ----------------------------------------------------------------------------------------------------------------------
    GLOBAL per-round metrics
    ----------------------------------------------------------------------------------------------------------------------



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
      <td>57.827329</td>
      <td>6</td>
      <td>1.0</td>
      <td>0.793736</td>
      <td>0.800948</td>
      <td>0.654082</td>
      <td>0.685601</td>
      <td>0.675877</td>
      <td>0.548019</td>
      <td>0.535673</td>
      <td>0.638788</td>
      <td>0.778761</td>
      <td>0.665123</td>
      <td>0.584533</td>
      <td>0.826531</td>
      <td>0.641351</td>
      <td>0.505916</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>53.655375</td>
      <td>6</td>
      <td>1.0</td>
      <td>0.826405</td>
      <td>0.843602</td>
      <td>0.680756</td>
      <td>0.683601</td>
      <td>0.704732</td>
      <td>0.397064</td>
      <td>0.396601</td>
      <td>0.507348</td>
      <td>0.840708</td>
      <td>0.699265</td>
      <td>0.395748</td>
      <td>0.846939</td>
      <td>0.659414</td>
      <td>0.398581</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>61.630185</td>
      <td>6</td>
      <td>1.0</td>
      <td>0.906292</td>
      <td>0.886256</td>
      <td>0.763076</td>
      <td>0.776532</td>
      <td>0.776203</td>
      <td>0.392914</td>
      <td>0.389987</td>
      <td>0.509979</td>
      <td>0.911504</td>
      <td>0.790253</td>
      <td>0.405739</td>
      <td>0.857143</td>
      <td>0.731740</td>
      <td>0.378127</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>61.264557</td>
      <td>6</td>
      <td>1.0</td>
      <td>0.938420</td>
      <td>0.876777</td>
      <td>0.800918</td>
      <td>0.829816</td>
      <td>0.807452</td>
      <td>0.390197</td>
      <td>0.390237</td>
      <td>0.512290</td>
      <td>0.849558</td>
      <td>0.729315</td>
      <td>0.489117</td>
      <td>0.908163</td>
      <td>0.883481</td>
      <td>0.276136</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>61.355637</td>
      <td>6</td>
      <td>1.0</td>
      <td>0.890414</td>
      <td>0.900474</td>
      <td>0.748290</td>
      <td>0.755909</td>
      <td>0.761373</td>
      <td>0.354420</td>
      <td>0.355648</td>
      <td>0.501421</td>
      <td>0.920354</td>
      <td>0.815800</td>
      <td>0.308877</td>
      <td>0.877551</td>
      <td>0.670446</td>
      <td>0.406934</td>
    </tr>
    <tr>
      <th>5</th>
      <td>6</td>
      <td>61.223200</td>
      <td>6</td>
      <td>1.0</td>
      <td>0.953911</td>
      <td>0.924171</td>
      <td>0.804915</td>
      <td>0.817118</td>
      <td>0.814682</td>
      <td>0.250936</td>
      <td>0.239908</td>
      <td>0.511123</td>
      <td>0.938053</td>
      <td>0.832495</td>
      <td>0.229564</td>
      <td>0.908163</td>
      <td>0.773113</td>
      <td>0.275578</td>
    </tr>
    <tr>
      <th>6</th>
      <td>7</td>
      <td>61.233854</td>
      <td>6</td>
      <td>1.0</td>
      <td>1.047266</td>
      <td>0.938389</td>
      <td>0.903434</td>
      <td>0.930815</td>
      <td>0.913795</td>
      <td>0.262384</td>
      <td>0.270300</td>
      <td>0.501296</td>
      <td>0.929204</td>
      <td>0.875047</td>
      <td>0.295492</td>
      <td>0.948980</td>
      <td>0.936166</td>
      <td>0.224208</td>
    </tr>
    <tr>
      <th>7</th>
      <td>8</td>
      <td>61.338977</td>
      <td>6</td>
      <td>1.0</td>
      <td>1.047865</td>
      <td>0.952607</td>
      <td>0.897607</td>
      <td>0.902949</td>
      <td>0.901074</td>
      <td>0.206127</td>
      <td>0.186221</td>
      <td>0.492901</td>
      <td>0.973451</td>
      <td>0.887116</td>
      <td>0.138742</td>
      <td>0.928571</td>
      <td>0.909703</td>
      <td>0.283827</td>
    </tr>
    <tr>
      <th>8</th>
      <td>9</td>
      <td>60.691149</td>
      <td>6</td>
      <td>1.0</td>
      <td>0.994343</td>
      <td>0.938389</td>
      <td>0.858707</td>
      <td>0.870887</td>
      <td>0.862749</td>
      <td>0.232533</td>
      <td>0.213632</td>
      <td>0.498379</td>
      <td>0.964602</td>
      <td>0.935901</td>
      <td>0.198336</td>
      <td>0.908163</td>
      <td>0.769699</td>
      <td>0.271964</td>
    </tr>
    <tr>
      <th>9</th>
      <td>10</td>
      <td>60.960427</td>
      <td>6</td>
      <td>1.0</td>
      <td>0.995617</td>
      <td>0.952607</td>
      <td>0.846127</td>
      <td>0.857695</td>
      <td>0.850444</td>
      <td>0.181061</td>
      <td>0.187422</td>
      <td>0.497648</td>
      <td>0.973451</td>
      <td>0.882159</td>
      <td>0.099600</td>
      <td>0.928571</td>
      <td>0.804581</td>
      <td>0.274990</td>
    </tr>
    <tr>
      <th>10</th>
      <td>11</td>
      <td>61.308589</td>
      <td>6</td>
      <td>1.0</td>
      <td>0.969295</td>
      <td>0.924171</td>
      <td>0.824334</td>
      <td>0.837093</td>
      <td>0.820532</td>
      <td>0.243417</td>
      <td>0.212969</td>
      <td>0.503145</td>
      <td>0.938053</td>
      <td>0.859213</td>
      <td>0.211809</td>
      <td>0.908163</td>
      <td>0.784117</td>
      <td>0.279864</td>
    </tr>
    <tr>
      <th>11</th>
      <td>12</td>
      <td>61.216216</td>
      <td>6</td>
      <td>1.0</td>
      <td>1.060846</td>
      <td>0.943128</td>
      <td>0.916511</td>
      <td>0.920310</td>
      <td>0.933771</td>
      <td>0.223020</td>
      <td>0.220086</td>
      <td>0.509842</td>
      <td>0.938053</td>
      <td>0.903798</td>
      <td>0.218159</td>
      <td>0.948980</td>
      <td>0.931169</td>
      <td>0.228626</td>
    </tr>
    <tr>
      <th>12</th>
      <td>13</td>
      <td>61.352746</td>
      <td>6</td>
      <td>1.0</td>
      <td>1.099068</td>
      <td>0.971564</td>
      <td>0.950625</td>
      <td>0.960852</td>
      <td>0.952903</td>
      <td>0.110676</td>
      <td>0.098129</td>
      <td>0.505856</td>
      <td>0.973451</td>
      <td>0.946903</td>
      <td>0.072676</td>
      <td>0.969388</td>
      <td>0.954916</td>
      <td>0.154492</td>
    </tr>
    <tr>
      <th>13</th>
      <td>14</td>
      <td>61.294860</td>
      <td>6</td>
      <td>1.0</td>
      <td>1.074290</td>
      <td>0.952607</td>
      <td>0.934452</td>
      <td>0.953930</td>
      <td>0.924781</td>
      <td>0.147300</td>
      <td>0.124079</td>
      <td>0.500833</td>
      <td>0.973451</td>
      <td>0.957165</td>
      <td>0.098529</td>
      <td>0.928571</td>
      <td>0.908264</td>
      <td>0.203537</td>
    </tr>
    <tr>
      <th>14</th>
      <td>15</td>
      <td>61.134783</td>
      <td>6</td>
      <td>1.0</td>
      <td>1.067050</td>
      <td>0.943128</td>
      <td>0.922999</td>
      <td>0.939406</td>
      <td>0.923507</td>
      <td>0.215476</td>
      <td>0.205524</td>
      <td>0.505650</td>
      <td>0.946903</td>
      <td>0.922734</td>
      <td>0.232672</td>
      <td>0.938776</td>
      <td>0.923303</td>
      <td>0.195649</td>
    </tr>
  </tbody>
</table>
</div>


    
    ----------------------------------------------------------------------------------------------------------------------
    LOCAL per-client per-round metrics
    ----------------------------------------------------------------------------------------------------------------------



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
      <td>4</td>
      <td>race_edge_plus</td>
      <td>(g=1.02, a=0.32, b=6.00, t=2.60, k=3, eg=0.15,...</td>
      <td>1.02</td>
      <td>0.32</td>
      <td>6.0</td>
      <td>...</td>
      <td>0.931842</td>
      <td>0.828431</td>
      <td>0.937500</td>
      <td>0.974425</td>
      <td>0.987013</td>
      <td>0.533478</td>
      <td>0.275084</td>
      <td>0.191437</td>
      <td>1.403111</td>
      <td>0.739389</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>client_1</td>
      <td>ds1</td>
      <td>1</td>
      <td>5</td>
      <td>race_focus</td>
      <td>(g=1.10, a=0.36, b=6.40, t=2.70, k=3, eg=0.17,...</td>
      <td>1.10</td>
      <td>0.36</td>
      <td>6.4</td>
      <td>...</td>
      <td>0.958606</td>
      <td>0.891641</td>
      <td>0.971429</td>
      <td>0.992188</td>
      <td>0.979167</td>
      <td>0.729433</td>
      <td>0.094636</td>
      <td>0.175931</td>
      <td>1.074689</td>
      <td>0.651736</td>
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
      <td>0.934758</td>
      <td>0.780303</td>
      <td>0.966667</td>
      <td>1.000000</td>
      <td>0.992063</td>
      <td>0.334138</td>
      <td>0.027760</td>
      <td>0.638102</td>
      <td>1.036380</td>
      <td>0.684625</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>client_3</td>
      <td>ds2</td>
      <td>1</td>
      <td>2</td>
      <td>race_robust</td>
      <td>(g=1.08, a=0.22, b=4.50, t=2.80, k=5, eg=0.11,...</td>
      <td>1.08</td>
      <td>0.22</td>
      <td>4.5</td>
      <td>...</td>
      <td>NaN</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>NaN</td>
      <td>1.000000</td>
      <td>0.016156</td>
      <td>0.387959</td>
      <td>0.595885</td>
      <td>1.029396</td>
      <td>0.720238</td>
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
      <td>0.963087</td>
      <td>0.934783</td>
      <td>0.937255</td>
      <td>0.983516</td>
      <td>0.996795</td>
      <td>0.023178</td>
      <td>0.480103</td>
      <td>0.496719</td>
      <td>0.927481</td>
      <td>0.700480</td>
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
      <td>5</td>
      <td>race_focus</td>
      <td>(g=1.10, a=0.36, b=6.40, t=2.70, k=3, eg=0.17,...</td>
      <td>1.10</td>
      <td>0.36</td>
      <td>6.4</td>
      <td>...</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.551742</td>
      <td>0.160402</td>
      <td>0.287856</td>
      <td>1.226403</td>
      <td>0.961462</td>
    </tr>
    <tr>
      <th>86</th>
      <td>15</td>
      <td>client_2</td>
      <td>ds1</td>
      <td>1</td>
      <td>2</td>
      <td>race_texture</td>
      <td>(g=0.92, a=0.34, b=5.80, t=2.30, k=7, eg=0.10,...</td>
      <td>0.92</td>
      <td>0.34</td>
      <td>5.8</td>
      <td>...</td>
      <td>0.999242</td>
      <td>1.000000</td>
      <td>0.996970</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.688924</td>
      <td>0.125594</td>
      <td>0.185481</td>
      <td>1.035521</td>
      <td>0.968049</td>
    </tr>
    <tr>
      <th>87</th>
      <td>15</td>
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
      <td>NaN</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>NaN</td>
      <td>1.000000</td>
      <td>0.658781</td>
      <td>0.204800</td>
      <td>0.136419</td>
      <td>1.199026</td>
      <td>0.920238</td>
    </tr>
    <tr>
      <th>88</th>
      <td>15</td>
      <td>client_4</td>
      <td>ds2</td>
      <td>1</td>
      <td>2</td>
      <td>race_robust</td>
      <td>(g=1.08, a=0.22, b=4.50, t=2.80, k=5, eg=0.11,...</td>
      <td>1.08</td>
      <td>0.22</td>
      <td>4.5</td>
      <td>...</td>
      <td>0.998218</td>
      <td>1.000000</td>
      <td>0.996078</td>
      <td>1.000000</td>
      <td>0.996795</td>
      <td>0.649126</td>
      <td>0.273923</td>
      <td>0.076951</td>
      <td>1.097432</td>
      <td>0.976824</td>
    </tr>
    <tr>
      <th>89</th>
      <td>15</td>
      <td>client_5</td>
      <td>ds2</td>
      <td>1</td>
      <td>0</td>
      <td>race_soft</td>
      <td>(g=0.95, a=0.18, b=3.80, t=2.20, k=3, eg=0.08,...</td>
      <td>0.95</td>
      <td>0.18</td>
      <td>3.8</td>
      <td>...</td>
      <td>0.988258</td>
      <td>0.983333</td>
      <td>0.969697</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.555580</td>
      <td>0.260654</td>
      <td>0.183766</td>
      <td>1.348036</td>
      <td>0.866723</td>
    </tr>
  </tbody>
</table>
<p>90 rows × 41 columns</p>
</div>


    
    ======================================================================================================================
    STEP 12: FINAL EVALUATION
    ======================================================================================================================
    
    ----------------------------------------------------------------------------------------------------------------------
    Best RL-selected preprocessing preset per client
    ----------------------------------------------------------------------------------------------------------------------



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
      <td>0</td>
      <td>race_balanced</td>
      <td>(g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10,...</td>
      <td>0.875902</td>
      <td>13</td>
    </tr>
    <tr>
      <th>1</th>
      <td>client_1</td>
      <td>ds1</td>
      <td>2</td>
      <td>race_texture</td>
      <td>(g=0.92, a=0.34, b=5.80, t=2.30, k=7, eg=0.10,...</td>
      <td>0.933583</td>
      <td>13</td>
    </tr>
    <tr>
      <th>2</th>
      <td>client_2</td>
      <td>ds1</td>
      <td>3</td>
      <td>race_robust</td>
      <td>(g=1.08, a=0.22, b=4.50, t=2.80, k=5, eg=0.11,...</td>
      <td>1.000000</td>
      <td>13</td>
    </tr>
    <tr>
      <th>3</th>
      <td>client_3</td>
      <td>ds2</td>
      <td>1</td>
      <td>race_balanced</td>
      <td>(g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10,...</td>
      <td>0.960119</td>
      <td>13</td>
    </tr>
    <tr>
      <th>4</th>
      <td>client_4</td>
      <td>ds2</td>
      <td>0</td>
      <td>race_soft</td>
      <td>(g=0.95, a=0.18, b=3.80, t=2.20, k=3, eg=0.08,...</td>
      <td>0.866769</td>
      <td>13</td>
    </tr>
    <tr>
      <th>5</th>
      <td>client_5</td>
      <td>ds2</td>
      <td>1</td>
      <td>race_balanced</td>
      <td>(g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10,...</td>
      <td>0.857137</td>
      <td>13</td>
    </tr>
  </tbody>
</table>
</div>


    
    ----------------------------------------------------------------------------------------------------------------------
    Chosen final validation-based preprocessing strategy
    ----------------------------------------------------------------------------------------------------------------------



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
      <td>['race_balanced']</td>
      <td>0.986780</td>
      <td>0.986726</td>
      <td>0.986798</td>
    </tr>
    <tr>
      <th>1</th>
      <td>ds2</td>
      <td>single</td>
      <td>['race_balanced']</td>
      <td>0.951502</td>
      <td>0.954315</td>
      <td>0.950564</td>
    </tr>
  </tbody>
</table>
</div>


    
    ======================================================================================================================
    STEP 12.5: EXTENDED METRICS + ERROR ANALYSIS + CALIBRATION
    ======================================================================================================================
    
    ----------------------------------------------------------------------------------------------------------------------
    Extended TEST metrics (DS1 vs DS2)
    ----------------------------------------------------------------------------------------------------------------------



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
      <td>0.986726</td>
      <td>0.987061</td>
      <td>0.986839</td>
      <td>0.987061</td>
      <td>0.986913</td>
      <td>0.986727</td>
      <td>0.986726</td>
      <td>0.986689</td>
      <td>0.060405</td>
      <td>...</td>
      <td>0.004438</td>
      <td>0.012939</td>
      <td>0.018152</td>
      <td>0.393173</td>
      <td>0.020410</td>
      <td>0.998191</td>
      <td>0.993172</td>
      <td>1.000000</td>
      <td>0.999696</td>
      <td>0.999895</td>
    </tr>
    <tr>
      <th>1</th>
      <td>ds2_test</td>
      <td>0.949239</td>
      <td>0.946868</td>
      <td>0.947965</td>
      <td>0.946868</td>
      <td>0.944978</td>
      <td>0.952325</td>
      <td>0.949239</td>
      <td>0.948517</td>
      <td>0.163820</td>
      <td>...</td>
      <td>0.016458</td>
      <td>0.053132</td>
      <td>0.020071</td>
      <td>0.679853</td>
      <td>0.073085</td>
      <td>0.994834</td>
      <td>0.999854</td>
      <td>0.987763</td>
      <td>0.999759</td>
      <td>0.991959</td>
    </tr>
  </tbody>
</table>
<p>2 rows × 26 columns</p>
</div>


    
    ----------------------------------------------------------------------------------------------------------------------
    Classwise metrics — DS1 TEST
    ----------------------------------------------------------------------------------------------------------------------



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
      <td>55</td>
      <td>1</td>
      <td>1</td>
      <td>169</td>
      <td>0.247788</td>
      <td>0.982143</td>
      <td>0.994118</td>
      <td>0.982143</td>
      <td>0.994118</td>
      <td>0.005882</td>
      <td>0.017857</td>
      <td>0.964912</td>
      <td>0.988130</td>
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
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>notumor</td>
      <td>59</td>
      <td>57</td>
      <td>1</td>
      <td>2</td>
      <td>166</td>
      <td>0.261062</td>
      <td>0.982759</td>
      <td>0.988095</td>
      <td>0.966102</td>
      <td>0.994012</td>
      <td>0.005988</td>
      <td>0.033898</td>
      <td>0.950000</td>
      <td>0.980057</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>pituitary</td>
      <td>56</td>
      <td>56</td>
      <td>1</td>
      <td>0</td>
      <td>169</td>
      <td>0.247788</td>
      <td>0.982456</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.994118</td>
      <td>0.005882</td>
      <td>0.000000</td>
      <td>0.982456</td>
      <td>0.997059</td>
    </tr>
  </tbody>
</table>
</div>


    
    ----------------------------------------------------------------------------------------------------------------------
    Classwise metrics — DS2 TEST
    ----------------------------------------------------------------------------------------------------------------------



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
      <td>45</td>
      <td>45</td>
      <td>3</td>
      <td>0</td>
      <td>149</td>
      <td>0.228426</td>
      <td>0.937500</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.980263</td>
      <td>0.019737</td>
      <td>0.000000</td>
      <td>0.937500</td>
      <td>0.990132</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>meningioma</td>
      <td>46</td>
      <td>38</td>
      <td>1</td>
      <td>8</td>
      <td>150</td>
      <td>0.233503</td>
      <td>0.974359</td>
      <td>0.949367</td>
      <td>0.826087</td>
      <td>0.993377</td>
      <td>0.006623</td>
      <td>0.173913</td>
      <td>0.808511</td>
      <td>0.909732</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>notumor</td>
      <td>61</td>
      <td>60</td>
      <td>0</td>
      <td>1</td>
      <td>136</td>
      <td>0.309645</td>
      <td>1.000000</td>
      <td>0.992701</td>
      <td>0.983607</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.016393</td>
      <td>0.983607</td>
      <td>0.991803</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>pituitary</td>
      <td>45</td>
      <td>44</td>
      <td>6</td>
      <td>1</td>
      <td>146</td>
      <td>0.228426</td>
      <td>0.880000</td>
      <td>0.993197</td>
      <td>0.977778</td>
      <td>0.960526</td>
      <td>0.039474</td>
      <td>0.022222</td>
      <td>0.862745</td>
      <td>0.969152</td>
    </tr>
  </tbody>
</table>
</div>


    
    ----------------------------------------------------------------------------------------------------------------------
    Top confusion pairs — DS1 TEST
    ----------------------------------------------------------------------------------------------------------------------



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
      <td>notumor</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>notumor</td>
      <td>glioma</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>notumor</td>
      <td>pituitary</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>glioma</td>
      <td>meningioma</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>meningioma</td>
      <td>glioma</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>glioma</td>
      <td>pituitary</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>meningioma</td>
      <td>pituitary</td>
      <td>0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>meningioma</td>
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
      <td>pituitary</td>
      <td>glioma</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>


    
    ----------------------------------------------------------------------------------------------------------------------
    Top confusion pairs — DS2 TEST
    ----------------------------------------------------------------------------------------------------------------------



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
      <td>meningioma</td>
      <td>pituitary</td>
      <td>5</td>
    </tr>
    <tr>
      <th>1</th>
      <td>meningioma</td>
      <td>glioma</td>
      <td>3</td>
    </tr>
    <tr>
      <th>2</th>
      <td>notumor</td>
      <td>pituitary</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>pituitary</td>
      <td>meningioma</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>glioma</td>
      <td>pituitary</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>glioma</td>
      <td>notumor</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>glioma</td>
      <td>meningioma</td>
      <td>0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>meningioma</td>
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


    
    ----------------------------------------------------------------------------------------------------------------------
    Calibration bins — DS1 TEST
    ----------------------------------------------------------------------------------------------------------------------



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
      <td>0.393173</td>
      <td>0.000000</td>
      <td>0.393173</td>
      <td>1</td>
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
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>7</td>
      <td>0.583333</td>
      <td>0.666667</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>8</td>
      <td>0.666667</td>
      <td>0.750000</td>
      <td>0.715174</td>
      <td>0.500000</td>
      <td>0.215174</td>
      <td>2</td>
    </tr>
    <tr>
      <th>9</th>
      <td>9</td>
      <td>0.750000</td>
      <td>0.833333</td>
      <td>0.812862</td>
      <td>1.000000</td>
      <td>0.187138</td>
      <td>2</td>
    </tr>
    <tr>
      <th>10</th>
      <td>10</td>
      <td>0.833333</td>
      <td>0.916667</td>
      <td>0.881485</td>
      <td>1.000000</td>
      <td>0.118515</td>
      <td>4</td>
    </tr>
    <tr>
      <th>11</th>
      <td>11</td>
      <td>0.916667</td>
      <td>1.000000</td>
      <td>0.984191</td>
      <td>0.995392</td>
      <td>0.011201</td>
      <td>217</td>
    </tr>
  </tbody>
</table>
</div>


    
    ----------------------------------------------------------------------------------------------------------------------
    Calibration bins — DS2 TEST
    ----------------------------------------------------------------------------------------------------------------------



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
      <td>0.444014</td>
      <td>0.333333</td>
      <td>0.110680</td>
      <td>3</td>
    </tr>
    <tr>
      <th>6</th>
      <td>6</td>
      <td>0.500000</td>
      <td>0.583333</td>
      <td>0.557965</td>
      <td>0.333333</td>
      <td>0.224632</td>
      <td>3</td>
    </tr>
    <tr>
      <th>7</th>
      <td>7</td>
      <td>0.583333</td>
      <td>0.666667</td>
      <td>0.588730</td>
      <td>0.000000</td>
      <td>0.588730</td>
      <td>1</td>
    </tr>
    <tr>
      <th>8</th>
      <td>8</td>
      <td>0.666667</td>
      <td>0.750000</td>
      <td>0.679853</td>
      <td>0.000000</td>
      <td>0.679853</td>
      <td>1</td>
    </tr>
    <tr>
      <th>9</th>
      <td>9</td>
      <td>0.750000</td>
      <td>0.833333</td>
      <td>0.802556</td>
      <td>1.000000</td>
      <td>0.197444</td>
      <td>2</td>
    </tr>
    <tr>
      <th>10</th>
      <td>10</td>
      <td>0.833333</td>
      <td>0.916667</td>
      <td>0.862888</td>
      <td>1.000000</td>
      <td>0.137112</td>
      <td>3</td>
    </tr>
    <tr>
      <th>11</th>
      <td>11</td>
      <td>0.916667</td>
      <td>1.000000</td>
      <td>0.983006</td>
      <td>0.978261</td>
      <td>0.004745</td>
      <td>184</td>
    </tr>
  </tbody>
</table>
</div>


    
    ----------------------------------------------------------------------------------------------------------------------
    VAL + TEST tables (federated, per-dataset + global equal)
    ----------------------------------------------------------------------------------------------------------------------



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
      <td>ARCF-Net + FedAvg + FedProx + prototype sharing</td>
      <td>VAL</td>
      <td>ds1</td>
      <td>0.986726</td>
      <td>0.986764</td>
      <td>0.986905</td>
      <td>0.986798</td>
      <td>0.986881</td>
      <td>0.986726</td>
      <td>0.986766</td>
      <td>...</td>
      <td>4.670397</td>
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
      <td>ARCF-Net + FedAvg + FedProx + prototype sharing</td>
      <td>VAL</td>
      <td>ds2</td>
      <td>0.954315</td>
      <td>0.951439</td>
      <td>0.952303</td>
      <td>0.950564</td>
      <td>0.955382</td>
      <td>0.954315</td>
      <td>0.953634</td>
      <td>...</td>
      <td>4.065940</td>
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
      <td>ARCF-Net + FedAvg + FedProx + prototype sharing</td>
      <td>VAL</td>
      <td>global_equal</td>
      <td>0.970520</td>
      <td>0.969102</td>
      <td>0.969604</td>
      <td>0.968681</td>
      <td>0.971132</td>
      <td>0.970520</td>
      <td>0.970200</td>
      <td>...</td>
      <td>4.368169</td>
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
      <td>ARCF-Net + FedAvg + FedProx + prototype sharing</td>
      <td>TEST</td>
      <td>ds1</td>
      <td>0.986726</td>
      <td>0.986839</td>
      <td>0.987061</td>
      <td>0.986913</td>
      <td>0.986727</td>
      <td>0.986726</td>
      <td>0.986689</td>
      <td>...</td>
      <td>4.782820</td>
      <td>0.987061</td>
      <td>0.982324</td>
      <td>0.982298</td>
      <td>0.986839</td>
      <td>0.995553</td>
      <td>0.995562</td>
      <td>0.018152</td>
      <td>0.393173</td>
      <td>0.020410</td>
    </tr>
    <tr>
      <th>4</th>
      <td>ARCF-Net + FedAvg + FedProx + prototype sharing</td>
      <td>TEST</td>
      <td>ds2</td>
      <td>0.949239</td>
      <td>0.947965</td>
      <td>0.946868</td>
      <td>0.944978</td>
      <td>0.952325</td>
      <td>0.949239</td>
      <td>0.948517</td>
      <td>...</td>
      <td>4.237615</td>
      <td>0.946868</td>
      <td>0.933294</td>
      <td>0.931940</td>
      <td>0.947965</td>
      <td>0.983816</td>
      <td>0.983542</td>
      <td>0.020071</td>
      <td>0.679853</td>
      <td>0.073085</td>
    </tr>
    <tr>
      <th>5</th>
      <td>ARCF-Net + FedAvg + FedProx + prototype sharing</td>
      <td>TEST</td>
      <td>global_equal</td>
      <td>0.967982</td>
      <td>0.967402</td>
      <td>0.966964</td>
      <td>0.965945</td>
      <td>0.969526</td>
      <td>0.967982</td>
      <td>0.967603</td>
      <td>...</td>
      <td>4.510217</td>
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


    
    Paper selection summary:
    - Best round: 13 | best_reward=1.0991
    - DS1 final strategy: single | names=['race_balanced']
    - DS2 final strategy: single | names=['race_balanced']
    
    ======================================================================================================================
    STEP 13: PREPROCESSING VALIDATION (DS1 + DS2 VAL SAMPLE)
    ======================================================================================================================
    
    ----------------------------------------------------------------------------------------------------------------------
    Preprocessing validation summary (DS1 VAL sample)
    ----------------------------------------------------------------------------------------------------------------------



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
      <td>0.041596</td>
      <td>0.021720</td>
      <td>0.014947</td>
      <td>0.165471</td>
    </tr>
    <tr>
      <th>1</th>
      <td>edge_energy_after</td>
      <td>0.110492</td>
      <td>0.027505</td>
      <td>0.066604</td>
      <td>0.242659</td>
    </tr>
    <tr>
      <th>2</th>
      <td>entropy_before</td>
      <td>5.820408</td>
      <td>0.628255</td>
      <td>3.547314</td>
      <td>7.239670</td>
    </tr>
    <tr>
      <th>3</th>
      <td>entropy_after</td>
      <td>6.604174</td>
      <td>0.584000</td>
      <td>3.940457</td>
      <td>7.583500</td>
    </tr>
    <tr>
      <th>4</th>
      <td>contrast_before</td>
      <td>0.187640</td>
      <td>0.052597</td>
      <td>0.101468</td>
      <td>0.363759</td>
    </tr>
    <tr>
      <th>5</th>
      <td>contrast_after</td>
      <td>0.247487</td>
      <td>0.022271</td>
      <td>0.209219</td>
      <td>0.344404</td>
    </tr>
    <tr>
      <th>6</th>
      <td>edge_gain_ratio</td>
      <td>2.891605</td>
      <td>0.663157</td>
      <td>1.247970</td>
      <td>5.213968</td>
    </tr>
    <tr>
      <th>7</th>
      <td>entropy_delta</td>
      <td>0.783767</td>
      <td>0.252603</td>
      <td>0.282053</td>
      <td>1.541243</td>
    </tr>
    <tr>
      <th>8</th>
      <td>contrast_delta</td>
      <td>0.059847</td>
      <td>0.033573</td>
      <td>-0.020711</td>
      <td>0.124809</td>
    </tr>
  </tbody>
</table>
</div>


    
    ----------------------------------------------------------------------------------------------------------------------
    Preprocessing validation summary (DS2 VAL sample)
    ----------------------------------------------------------------------------------------------------------------------



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
      <td>0.041718</td>
      <td>0.015947</td>
      <td>0.012571</td>
      <td>0.121283</td>
    </tr>
    <tr>
      <th>1</th>
      <td>edge_energy_after</td>
      <td>0.116392</td>
      <td>0.023684</td>
      <td>0.043676</td>
      <td>0.167522</td>
    </tr>
    <tr>
      <th>2</th>
      <td>entropy_before</td>
      <td>5.811255</td>
      <td>0.822915</td>
      <td>3.222212</td>
      <td>7.570240</td>
    </tr>
    <tr>
      <th>3</th>
      <td>entropy_after</td>
      <td>6.664449</td>
      <td>0.785474</td>
      <td>3.612907</td>
      <td>7.741302</td>
    </tr>
    <tr>
      <th>4</th>
      <td>contrast_before</td>
      <td>0.190753</td>
      <td>0.055869</td>
      <td>0.098556</td>
      <td>0.341468</td>
    </tr>
    <tr>
      <th>5</th>
      <td>contrast_after</td>
      <td>0.250066</td>
      <td>0.022336</td>
      <td>0.198854</td>
      <td>0.318548</td>
    </tr>
    <tr>
      <th>6</th>
      <td>edge_gain_ratio</td>
      <td>3.015276</td>
      <td>0.781821</td>
      <td>1.308311</td>
      <td>5.732603</td>
    </tr>
    <tr>
      <th>7</th>
      <td>entropy_delta</td>
      <td>0.853194</td>
      <td>0.296688</td>
      <td>0.171062</td>
      <td>1.614185</td>
    </tr>
    <tr>
      <th>8</th>
      <td>contrast_delta</td>
      <td>0.059313</td>
      <td>0.037080</td>
      <td>-0.022945</td>
      <td>0.130606</td>
    </tr>
  </tbody>
</table>
</div>


    
    ======================================================================================================================
    STEP 14: PARAMETER EVOLUTION TABLE
    ======================================================================================================================
    
    ----------------------------------------------------------------------------------------------------------------------
    Mean selected preprocessing parameters over rounds
    ----------------------------------------------------------------------------------------------------------------------



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
      <td>1.046667</td>
      <td>0.280000</td>
      <td>5.300000</td>
      <td>2.616667</td>
      <td>4.333333</td>
      <td>0.130000</td>
      <td>0.796667</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1.016667</td>
      <td>0.240000</td>
      <td>4.766667</td>
      <td>2.533333</td>
      <td>3.666667</td>
      <td>0.113333</td>
      <td>0.773333</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1.016667</td>
      <td>0.286667</td>
      <td>5.266667</td>
      <td>2.550000</td>
      <td>5.333333</td>
      <td>0.115000</td>
      <td>0.793333</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>0.976667</td>
      <td>0.236667</td>
      <td>4.483333</td>
      <td>2.350000</td>
      <td>6.000000</td>
      <td>0.093333</td>
      <td>0.750000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0.946667</td>
      <td>0.270000</td>
      <td>4.900000</td>
      <td>2.283333</td>
      <td>6.333333</td>
      <td>0.093333</td>
      <td>0.766667</td>
    </tr>
    <tr>
      <th>5</th>
      <td>6</td>
      <td>1.000000</td>
      <td>0.240000</td>
      <td>4.600000</td>
      <td>2.400000</td>
      <td>5.000000</td>
      <td>0.100000</td>
      <td>0.780000</td>
    </tr>
    <tr>
      <th>6</th>
      <td>7</td>
      <td>0.970000</td>
      <td>0.250000</td>
      <td>4.716667</td>
      <td>2.366667</td>
      <td>5.000000</td>
      <td>0.095000</td>
      <td>0.763333</td>
    </tr>
    <tr>
      <th>7</th>
      <td>8</td>
      <td>1.025000</td>
      <td>0.233333</td>
      <td>4.533333</td>
      <td>2.516667</td>
      <td>4.666667</td>
      <td>0.103333</td>
      <td>0.770000</td>
    </tr>
    <tr>
      <th>8</th>
      <td>9</td>
      <td>1.006667</td>
      <td>0.250000</td>
      <td>4.733333</td>
      <td>2.400000</td>
      <td>4.000000</td>
      <td>0.108333</td>
      <td>0.780000</td>
    </tr>
    <tr>
      <th>9</th>
      <td>10</td>
      <td>1.005000</td>
      <td>0.230000</td>
      <td>4.566667</td>
      <td>2.483333</td>
      <td>4.666667</td>
      <td>0.103333</td>
      <td>0.760000</td>
    </tr>
    <tr>
      <th>10</th>
      <td>11</td>
      <td>0.978333</td>
      <td>0.240000</td>
      <td>4.716667</td>
      <td>2.400000</td>
      <td>4.666667</td>
      <td>0.105000</td>
      <td>0.760000</td>
    </tr>
    <tr>
      <th>11</th>
      <td>12</td>
      <td>1.060000</td>
      <td>0.273333</td>
      <td>5.166667</td>
      <td>2.633333</td>
      <td>4.333333</td>
      <td>0.126667</td>
      <td>0.800000</td>
    </tr>
    <tr>
      <th>12</th>
      <td>13</td>
      <td>0.998333</td>
      <td>0.253333</td>
      <td>4.750000</td>
      <td>2.433333</td>
      <td>5.000000</td>
      <td>0.101667</td>
      <td>0.776667</td>
    </tr>
    <tr>
      <th>13</th>
      <td>14</td>
      <td>0.970000</td>
      <td>0.240000</td>
      <td>4.633333</td>
      <td>2.333333</td>
      <td>5.333333</td>
      <td>0.098333</td>
      <td>0.763333</td>
    </tr>
    <tr>
      <th>14</th>
      <td>15</td>
      <td>1.025000</td>
      <td>0.283333</td>
      <td>5.250000</td>
      <td>2.516667</td>
      <td>4.333333</td>
      <td>0.121667</td>
      <td>0.796667</td>
    </tr>
  </tbody>
</table>
</div>


    
    ======================================================================================================================
    STEP 15: SAVING ONLY TWO FILES (CHECKPOINT + ONE CSV)
    ======================================================================================================================
    ✅ Saved checkpoint: /kaggle/working/outputs/ARCFNet_RESNET50_FEDAVG_FEDPROX_PROTO_checkpoint.pth
    ✅ Saved CSV (ALL outputs): /kaggle/working/outputs/ALL_OUTPUTS_AND_METRICS_RESNET50_FEDAVG_FEDPROX_PROTO.csv
    
    DONE ✅
    Method: ARCF-Net = Adaptive RACE-FELCM with CRAF Fusion Network
    Backbone: Residual Network-50
    Federated variant: FedAvg + FedProx + prototype sharing
    Best round: 13
    Adaptive clients => DS1=3, DS2=3, TOTAL=6
    Rounds completed: 15
    Global TEST acc: 0.9680
    Global TEST f1_macro: 0.9659
    DS1 TEST acc: 0.9867
    DS2 TEST acc: 0.9492
    DS1 final strategy: single | names=['race_balanced']
    DS2 final strategy: single | names=['race_balanced']


**5. grouped FedAvg with private dataset heads vs all-shared model**


```python
# ============================================================
# KAGGLE FULL SCRIPT
# TRUE FL + RL-UCB + RACE-FELCM + CRAF + ResNet-50
# ABLATION: ARCF-Net with Grouped Federated Averaging and Dataset-Private Heads
# METHOD ACRONYM: ARCF-Net
# FULL FORM: Adaptive RACE-FELCM with CRAF Fusion Network
# ------------------------------------------------------------
# KAGGLE-READY + SYMMETRIC DS1/DS2 IMPORTANCE
# - Uses BOTH datasets
# - Reads datasets from /kaggle/input automatically
# - Exact 15 FL rounds
# - Proper FL with Grouped FedAvg:
#     * shared backbone / fusion / neck aggregated across all clients
#     * DS1 private classifier head aggregated only across DS1 clients
#     * DS2 private classifier head aggregated only across DS2 clients
# - NO FedProx
# - NO prototype sharing
# - RL-UCB for SHARED client-count planning + per-client preprocessing preset selection
# - Tune-aware theta probing before local training
# - Equal DS1 / DS2 importance in best-round selection and merged reporting
# - No plots
# - Saves checkpoint WITH full-process info
# ============================================================

import os
import sys
import time
import math
import copy
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

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    log_loss, confusion_matrix, roc_auc_score,
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

print("=" * 118)
print("TRUE FL + RL-UCB + RACE-FELCM + CRAF + ResNet-50")
print("METHOD: ARCF-Net (Adaptive RACE-FELCM with CRAF Fusion Network)")
print("ABLATION: ARCF-Net with Grouped Federated Averaging and Dataset-Private Heads")
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

    # grouped FedAvg tempering
    "fedavg_temper": 0.50,

    # misc
    "quick_hash_subset_per_split": 300,
    "preproc_val_sample_n": 400,
    "calibration_bins": 12,
}

OUTDIR = "/kaggle/working/outputs" if IS_KAGGLE else "/content/outputs"
os.makedirs(OUTDIR, exist_ok=True)
MODEL_PATH = os.path.join(OUTDIR, "ARCFNet_RESNET50_GROUPED_PRIVATE_HEADS_checkpoint.pth")
CSV_PATH   = os.path.join(OUTDIR, "ALL_OUTPUTS_AND_METRICS_RESNET50_GROUPED_PRIVATE_HEADS.csv")

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], device=DEVICE).view(1, 3, 1, 1)
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225], device=DEVICE).view(1, 3, 1, 1)

labels = ["glioma", "meningioma", "notumor", "pituitary"]
label2id = {l: i for i, l in enumerate(labels)}
id2label = {i: l for l, i in label2id.items()}
NUM_CLASSES = len(labels)

SETTING_NAME = "ARCF-Net with Grouped Federated Averaging and Dataset-Private Heads"

METHOD_INFO = {
    "acronym": "ARCF-Net",
    "full_form": "Adaptive RACE-FELCM with CRAF Fusion Network",
    "preprocessing_full_form": "Robust Adaptive Context-Enhanced Fuzzy Edge Local Contrast Mapping",
    "fusion_full_form": "Cross-Residual Adaptive Fusion",
    "backbone_full_form": "Residual Network-50",
    "federated_variant": SETTING_NAME,
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

DS1_ROOT = search_kaggle_input_for_root(REQ1, prefer_raw=True)
DS2_ROOT = search_kaggle_input_for_root(REQ2, prefer_raw=False)

if DS1_ROOT is None:
    _, DS1_ROOT = try_kagglehub_download(
        "orvile/pmram-bangladeshi-brain-cancer-mri-dataset",
        REQ1,
        prefer_raw=True,
    )

if DS2_ROOT is None:
    _, DS2_ROOT = try_kagglehub_download(
        "yassinebazgour/preprocessed-brain-mri-scans-for-tumors-detection",
        REQ2,
        prefer_raw=False,
    )

if DS1_ROOT is None:
    raise RuntimeError(
        "Could not locate DS1 under /kaggle/input. In Kaggle, add "
        "'orvile/pmram-bangladeshi-brain-cancer-mri-dataset'."
    )
if DS2_ROOT is None:
    raise RuntimeError(
        "Could not locate DS2 under /kaggle/input. In Kaggle, add "
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
print(df1["label"].value_counts().reindex(labels, fill_value=0))
print("\nDataset-2 images:", len(df2))
print(df2["label"].value_counts().reindex(labels, fill_value=0))

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
# STEP 4: SHARED ADAPTIVE CLIENT COUNT
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
print("STEP 6: DATA LOADERS")
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

        counts = df_src.loc[tr_idx, "label"].value_counts().reindex(labels, fill_value=0)

        clients.append({
            "gid": gid,
            "local_id": local_id,
            "dataset": ds_name,
            "source_id": source_id,
            "train_loader": train_loader,
            "tune_loader": tune_loader,
            "val_loader": val_loader,
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
    theta_t = torch.tensor(
        [gamma, alpha, beta / 8.0, tau / 4.0, blur_k / 7.0, edge_gain, blend],
        device=DEVICE,
        dtype=torch.float32
    )
    return theta_t.unsqueeze(0).repeat(batch_size, 1)

def theta_str(theta):
    if theta is None:
        return "None"
    g, a, b, t, k, eg, m = theta
    return f"(g={g:.2f}, a={a:.2f}, b={b:.2f}, t={t:.2f}, k={k}, eg={eg:.2f}, mix={m:.2f})"

for c in clients:
    c["preset_bank"] = PRESET_BANK_DS1 if c["dataset"] == "ds1" else PRESET_BANK_DS2
    c["theta_bandit"] = UCBBandit(len(c["preset_bank"]), c=CFG["ucb_c"])

# ============================================================
# STEP 8: MODEL — ResNet-50 + CRAF Fusion + DATASET-PRIVATE HEADS
# ============================================================
print("\n" + "=" * 118)
print("STEP 8: MODEL — ResNet-50 + CRAF Fusion + DATASET-PRIVATE HEADS")
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

class ResNet50CRAFPrivateHeads(nn.Module):
    def __init__(self, num_classes, cond_dim=64, fuse_dim=256, embed_dim=256, pretrained=True):
        super().__init__()
        self.num_classes = int(num_classes)

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
        self.embed_dim = embed_dim

        self.classifier_ds1 = nn.Linear(embed_dim, num_classes)
        self.classifier_ds2 = nn.Linear(embed_dim, num_classes)

    def _encode(self, x):
        f = self.backbone(x)
        f = self.pool(f).flatten(1)
        return f

    def _route_logits(self, embed, source_id):
        logits = torch.zeros(
            embed.size(0),
            self.num_classes,
            device=embed.device,
            dtype=embed.dtype,
        )
        m0 = (source_id == 0)
        m1 = (source_id == 1)

        if m0.any():
            logits[m0] = self.classifier_ds1(embed[m0])
        if m1.any():
            logits[m1] = self.classifier_ds2(embed[m1])

        return logits

    def forward(self, x_raw_n, x_enh_n, x_res_n, theta_vec, source_id, return_extra=False):
        cond = self.theta_mlp(theta_vec) + self.source_emb(source_id)
        cond = self.cond_norm(cond)

        f_raw = self._encode(x_raw_n)
        f_enh = self._encode(x_enh_n)
        f_res = self._encode(x_res_n)

        fused, gates = self.fusion(f_raw, f_enh, f_res, cond)
        embed = self.neck(fused)
        logits = self._route_logits(embed, source_id)

        if return_extra:
            return logits, embed, gates
        return logits

def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

def set_trainable_for_round(model, rnd, client_dataset=None):
    for p in model.backbone.parameters():
        p.requires_grad = False

    for n, p in model.named_parameters():
        if not n.startswith("backbone."):
            p.requires_grad = True

    ds = str(client_dataset).lower() if client_dataset is not None else None
    if ds is not None:
        for p in model.classifier_ds1.parameters():
            p.requires_grad = (ds == "ds1")
        for p in model.classifier_ds2.parameters():
            p.requires_grad = (ds == "ds2")

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

global_model = ResNet50CRAFPrivateHeads(
    num_classes=NUM_CLASSES,
    cond_dim=64,
    fuse_dim=256,
    embed_dim=256,
    pretrained=True,
).to(DEVICE)

set_trainable_for_round(global_model, rnd=1, client_dataset=None)
total_params, trainable_params = count_params(global_model)

print("Backbone: ResNet-50 | pretrained_loaded=True")
print(f"Total params: {total_params:,}")
print(f"Trainable params: {trainable_params:,} ({(100.0 * trainable_params / total_params):.2f}%)")

# ============================================================
# STEP 9: LOSSES
# ============================================================
print("\n" + "=" * 118)
print("STEP 9: LOSSES")
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

def train_one_epoch(model, loader, optimizer, preproc, theta, scheduler=None):
    model.train()
    freeze_backbone_bn_stats(model)
    preproc.eval()

    losses, ce_losses, correct, total = [], [], 0, 0
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

            logits, _, _ = model(x_raw_n, x_enh_n, x_res_n, theta_vec, source_id, return_extra=True)
            ce = criterion(logits, y)
            loss = ce

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

        preds = logits.argmax(dim=1)
        correct += int((preds == y).sum().item())
        total += int(y.size(0))

    return {
        "loss": float(np.mean(losses)) if losses else np.nan,
        "ce_loss": float(np.mean(ce_losses)) if ce_losses else np.nan,
        "acc": float(correct / max(1, total)),
        "train_time_s": float(time.time() - t0),
    }

def build_tempered_fedavg_weights(clients_subset):
    sizes = np.array([c["n_train"] for c in clients_subset], dtype=np.float64)
    sizes = np.power(np.clip(sizes, 1.0, None), CFG["fedavg_temper"])
    sizes = sizes / max(1e-12, sizes.sum())
    return sizes.tolist()

def _aggregate_key_from_subset(local_models, subset_indices, weights, key, ref_dtype):
    acc = None
    for i, w in zip(subset_indices, weights):
        t = local_models[i].state_dict()[key].detach().float().cpu()
        acc = t * w if acc is None else acc + t * w
    return acc.to(ref_dtype)

def grouped_fedavg_update(global_model, local_models, clients_meta):
    global_sd = global_model.state_dict()
    new_sd = {}

    all_idx = list(range(len(local_models)))
    ds1_idx = [i for i, c in enumerate(clients_meta) if c["dataset"] == "ds1"]
    ds2_idx = [i for i, c in enumerate(clients_meta) if c["dataset"] == "ds2"]

    all_w = build_tempered_fedavg_weights(clients_meta)
    ds1_meta = [clients_meta[i] for i in ds1_idx]
    ds2_meta = [clients_meta[i] for i in ds2_idx]
    ds1_w = build_tempered_fedavg_weights(ds1_meta) if len(ds1_meta) else []
    ds2_w = build_tempered_fedavg_weights(ds2_meta) if len(ds2_meta) else []

    for key in global_sd.keys():
        ref = global_sd[key]

        if key.startswith("classifier_ds1."):
            subset_idx = ds1_idx
            subset_w = ds1_w
        elif key.startswith("classifier_ds2."):
            subset_idx = ds2_idx
            subset_w = ds2_w
        else:
            subset_idx = all_idx
            subset_w = all_w

        if len(subset_idx) == 0:
            new_sd[key] = ref.detach().clone()
            continue

        if not torch.is_floating_point(ref):
            new_sd[key] = local_models[subset_idx[0]].state_dict()[key].detach().clone()
        else:
            new_sd[key] = _aggregate_key_from_subset(local_models, subset_idx, subset_w, key, ref.dtype)

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
# STEP 11: TRAINING
# ============================================================
print("\n" + "=" * 118)
print("STEP 11: TRUE FEDERATED TRAINING")
print("=" * 118)

history_global = []
history_local = []

best_reward = -1.0
best_round_saved = None
best_model_state = None
best_theta_bandit_states = None

t_global_start = time.time()

print(f"Adaptive clients => DS1={n_clients_ds1}, DS2={n_clients_ds2}, TOTAL={CLIENTS_TOTAL}")
print(f"Rounds: {CFG['rounds']} | Local epochs: {CFG['local_epochs']}")
print(f"Augmentation ON: {CFG['use_augmentation']}")
print("Transfer backbone: ResNet-50")
print("Preprocessing: RACE-FELCM")
print("Fusion: CRAF")
print("Federated mode: Grouped FedAvg with dataset-private heads")
print(f"Tempered FedAvg exponent = {CFG['fedavg_temper']:.2f}")
print(f"Best-round masses => DS1={CFG['best_round_mass_ds1']:.2f}, DS2={CFG['best_round_mass_ds2']:.2f}, min-bonus={CFG['best_round_min_bonus']:.2f}")

for rnd in range(1, CFG["rounds"] + 1):
    round_t0 = time.time()
    selected_ids = list(range(len(clients)))

    print("\n" + "=" * 118)
    print(f"ROUND {rnd}/{CFG['rounds']} | selected={selected_ids}")
    print("=" * 118)

    local_models = []
    round_local_rows = []
    selected_clients_meta = []

    for cid in selected_ids:
        client = clients[cid]

        theta_arm = select_theta_arm_with_probe(client, global_model)
        theta_name, theta = client["preset_bank"][theta_arm]
        preproc = theta_to_module(theta).to(DEVICE)

        local_model = ResNet50CRAFPrivateHeads(
            num_classes=NUM_CLASSES,
            cond_dim=64,
            fuse_dim=256,
            embed_dim=256,
            pretrained=False,
        ).to(DEVICE)
        local_model.load_state_dict(global_model.state_dict(), strict=True)

        set_trainable_for_round(local_model, rnd, client_dataset=client["dataset"])
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
                scheduler=scheduler,
            )
            train_logs.append(log_ep)

        met_loc, _, _ = evaluate_full(local_model, client["val_loader"], preproc, theta, return_gates=True, use_tta=False)

        reward = score_metric(met_loc)
        client["theta_bandit"].update(theta_arm, reward)

        local_models.append(local_model)
        selected_clients_meta.append(client)

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

    grouped_fedavg_update(global_model, local_models, selected_clients_meta)

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
        best_theta_bandit_states = [copy.deepcopy(c["theta_bandit"].state_dict()) for c in clients]

if best_model_state is not None:
    global_model.load_state_dict({k: v.to(DEVICE) for k, v in best_model_state.items()})

if best_theta_bandit_states is not None:
    for c, sd in zip(clients, best_theta_bandit_states):
        c["theta_bandit"].load_state_dict(sd)

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
# STEP 12.5: EXTENDED METRICS + CALIBRATION + ERROR TABLES
# ============================================================
print("\n" + "=" * 118)
print("STEP 12.5: EXTENDED METRICS + CALIBRATION + ERROR TABLES")
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
    {"setting": SETTING_NAME, "split": "VAL",  "dataset": "ds1",          **compact_metrics(val_ds1)},
    {"setting": SETTING_NAME, "split": "VAL",  "dataset": "ds2",          **compact_metrics(val_ds2)},
    {"setting": SETTING_NAME, "split": "VAL",  "dataset": "global_equal", **compact_metrics(val_global)},
    {"setting": SETTING_NAME, "split": "TEST", "dataset": "ds1",          **compact_metrics(test_ds1)},
    {"setting": SETTING_NAME, "split": "TEST", "dataset": "ds2",          **compact_metrics(test_ds2)},
    {"setting": SETTING_NAME, "split": "TEST", "dataset": "global_equal", **compact_metrics(test_global)},
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
# STEP 14: PARAMETER EVOLUTION TABLE
# ============================================================
print("\n" + "=" * 118)
print("STEP 14: PARAMETER EVOLUTION TABLE")
print("=" * 118)

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
for ccol in theta_cols:
    if ccol in loc_copy.columns:
        loc_copy[ccol] = pd.to_numeric(loc_copy[ccol], errors="coerce")

theta_evo = loc_copy.groupby("round")[theta_cols].mean(numeric_only=True).reset_index()
print_table(theta_evo, "Mean selected preprocessing parameters over rounds")
add_table_to_csv(theta_evo, "theta_evolution_mean")

# ============================================================
# STEP 15: SAVE CHECKPOINT + CSV
# ============================================================
print("\n" + "=" * 118)
print("STEP 15: SAVING ONLY TWO FILES (CHECKPOINT + ONE CSV)")
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
    "federated_variant": SETTING_NAME,

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
    "theta_evolution_mean": theta_evo.to_dict(orient="list"),
}

torch.save(checkpoint, MODEL_PATH)
print(f"✅ Saved checkpoint: {MODEL_PATH}")

all_df.to_csv(CSV_PATH, index=False)
print(f"✅ Saved CSV (ALL outputs): {CSV_PATH}")

print("\nDONE ✅")
print(f"Method: {METHOD_INFO['acronym']} = {METHOD_INFO['full_form']}")
print(f"Setting: {SETTING_NAME}")
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
    ABLATION: ARCF-Net with Grouped Federated Averaging and Dataset-Private Heads
    ======================================================================================================================
    ENV: KAGGLE | DEVICE: cuda | torch=2.9.0+cu126
    ======================================================================================================================
    
    ======================================================================================================================
    STEP 0: ACCESS DATASETS
    ======================================================================================================================
    Dataset-1 RAW root detected:
      /kaggle/input/datasets/orvile/pmram-bangladeshi-brain-cancer-mri-dataset/PMRAM Bangladeshi Brain Cancer - MRI Dataset/PMRAM Bangladeshi Brain Cancer - MRI Dataset/Raw Data/Raw
    Dataset-2 root detected:
      /kaggle/input/datasets/chubskuy/brain-tumor-image/Testing
    
    ======================================================================================================================
    STEP 1: BUILD DATA MANIFESTS
    ======================================================================================================================
    ds1_raw: 512Glioma -> glioma | 373 images
    ds1_raw: 512Meningioma -> meningioma | 363 images
    ds1_raw: 512Normal -> notumor | 396 images
    ds1_raw: 512Pituitary -> pituitary | 373 images
    ds2: glioma -> glioma | 300 images
    ds2: meningioma -> meningioma | 306 images
    ds2: notumor -> notumor | 405 images
    ds2: pituitary -> pituitary | 300 images
    
    Dataset-1 images: 1505
    label
    glioma        373
    meningioma    363
    notumor       396
    pituitary     373
    Name: count, dtype: int64
    
    Dataset-2 images: 1311
    label
    glioma        300
    meningioma    306
    notumor       405
    pituitary     300
    Name: count, dtype: int64
    
    ======================================================================================================================
    STEP 2: TRAIN / VAL / TEST SPLIT
    ======================================================================================================================
    DS1 TRAIN: 1053 | VAL: 226 | TEST: 226
    DS2 TRAIN: 917 | VAL: 197 | TEST: 197
    
    ======================================================================================================================
    STEP 2.5: SANITY / LEAKAGE CHECKS
    ======================================================================================================================
    
    ----------------------------------------------------------------------------------------------------------------------
    Leakage / Sanity Summary — ds1
    ----------------------------------------------------------------------------------------------------------------------



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
      <td>5</td>
      <td>5</td>
      <td>6</td>
      <td>298</td>
      <td>222</td>
      <td>224</td>
    </tr>
  </tbody>
</table>
</div>


    
    ----------------------------------------------------------------------------------------------------------------------
    Leakage / Sanity Summary — ds2
    ----------------------------------------------------------------------------------------------------------------------



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
      <td>917</td>
      <td>197</td>
      <td>197</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>300</td>
      <td>194</td>
      <td>197</td>
    </tr>
  </tbody>
</table>
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
      <td>0.755612</td>
      <td>0.767340</td>
      <td>0.761476</td>
      <td>0.761476</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>4</td>
      <td>0.679705</td>
      <td>0.635628</td>
      <td>0.657667</td>
      <td>0.657667</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>5</td>
      <td>0.721049</td>
      <td>0.775174</td>
      <td>0.748111</td>
      <td>0.748111</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>3</td>
      <td>0.841039</td>
      <td>0.726403</td>
      <td>0.783721</td>
      <td>0.772599</td>
      <td>2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>5</td>
      <td>0.688321</td>
      <td>0.701388</td>
      <td>0.694855</td>
      <td>0.721483</td>
      <td>2</td>
    </tr>
    <tr>
      <th>5</th>
      <td>6</td>
      <td>4</td>
      <td>0.665267</td>
      <td>0.691320</td>
      <td>0.678293</td>
      <td>0.667980</td>
      <td>2</td>
    </tr>
    <tr>
      <th>6</th>
      <td>7</td>
      <td>3</td>
      <td>0.766538</td>
      <td>0.693916</td>
      <td>0.730227</td>
      <td>0.758475</td>
      <td>3</td>
    </tr>
    <tr>
      <th>7</th>
      <td>8</td>
      <td>5</td>
      <td>0.677792</td>
      <td>0.707763</td>
      <td>0.692778</td>
      <td>0.711915</td>
      <td>3</td>
    </tr>
    <tr>
      <th>8</th>
      <td>9</td>
      <td>4</td>
      <td>0.715564</td>
      <td>0.681741</td>
      <td>0.698653</td>
      <td>0.678204</td>
      <td>3</td>
    </tr>
    <tr>
      <th>9</th>
      <td>10</td>
      <td>3</td>
      <td>0.734619</td>
      <td>0.707902</td>
      <td>0.721261</td>
      <td>0.749171</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>


    
    ======================================================================================================================
    STEP 5: FINAL NON-IID CLIENT PARTITIONING
    ======================================================================================================================
    
    ======================================================================================================================
    STEP 6: DATA LOADERS
    ======================================================================================================================
    ds1 | client_0 | train=286 | tune=45 | val=40
    ds1 | client_1 | train=257 | tune=41 | val=36
    ds1 | client_2 | train=269 | tune=42 | val=37
    ds2 | client_3 | train=100 | tune=16 | val=14
    ds2 | client_4 | train=342 | tune=54 | val=47
    ds2 | client_5 | train=265 | tune=42 | val=37
    
    ----------------------------------------------------------------------------------------------------------------------
    Adaptive client class distribution
    ----------------------------------------------------------------------------------------------------------------------



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
      <td>286</td>
      <td>45</td>
      <td>40</td>
      <td>40</td>
      <td>33</td>
      <td>166</td>
      <td>47</td>
    </tr>
    <tr>
      <th>1</th>
      <td>client_1</td>
      <td>ds1</td>
      <td>257</td>
      <td>41</td>
      <td>36</td>
      <td>135</td>
      <td>4</td>
      <td>31</td>
      <td>87</td>
    </tr>
    <tr>
      <th>2</th>
      <td>client_2</td>
      <td>ds1</td>
      <td>269</td>
      <td>42</td>
      <td>37</td>
      <td>25</td>
      <td>159</td>
      <td>17</td>
      <td>68</td>
    </tr>
    <tr>
      <th>3</th>
      <td>client_3</td>
      <td>ds2</td>
      <td>100</td>
      <td>16</td>
      <td>14</td>
      <td>74</td>
      <td>18</td>
      <td>4</td>
      <td>4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>client_4</td>
      <td>ds2</td>
      <td>342</td>
      <td>54</td>
      <td>47</td>
      <td>5</td>
      <td>121</td>
      <td>154</td>
      <td>62</td>
    </tr>
    <tr>
      <th>5</th>
      <td>client_5</td>
      <td>ds2</td>
      <td>265</td>
      <td>42</td>
      <td>37</td>
      <td>82</td>
      <td>25</td>
      <td>61</td>
      <td>97</td>
    </tr>
  </tbody>
</table>
</div>


    Augmentation: ON ✅
    Preprocessing: ON ✅
    Total adaptive clients: 6
    
    ======================================================================================================================
    STEP 7: NOVEL PREPROCESSING — RACE-FELCM
    ======================================================================================================================
    
    ======================================================================================================================
    STEP 8: MODEL — ResNet-50 + CRAF Fusion + DATASET-PRIVATE HEADS
    ======================================================================================================================
    Downloading: "https://download.pytorch.org/models/resnet50-11ad3fa6.pth" to /root/.cache/torch/hub/checkpoints/resnet50-11ad3fa6.pth


    100%|██████████| 97.8M/97.8M [00:00<00:00, 201MB/s]


    Backbone: ResNet-50 | pretrained_loaded=True
    Total params: 25,791,883
    Trainable params: 2,283,851 (8.85%)
    
    ======================================================================================================================
    STEP 9: LOSSES
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
    Federated mode: Grouped FedAvg with dataset-private heads
    Tempered FedAvg exponent = 0.50
    Best-round masses => DS1=0.50, DS2=0.50, min-bonus=0.15
    
    ======================================================================================================================
    ROUND 1/15 | selected=[0, 1, 2, 3, 4, 5]
    ======================================================================================================================
    Client 0 (ds1) | train_acc=0.6871 | val_acc=0.8250 | val_f1=0.7118 | val_auc=0.9642 | reward=0.7401 | theta=race_edge_plus (g=1.02, a=0.32, b=6.00, t=2.60, k=3, eg=0.15, mix=0.84)
    Client 1 (ds1) | train_acc=0.6926 | val_acc=0.8056 | val_f1=0.6003 | val_auc=0.9844 | reward=0.6516 | theta=race_focus (g=1.10, a=0.36, b=6.40, t=2.70, k=3, eg=0.17, mix=0.86)
    Client 2 (ds1) | train_acc=0.7100 | val_acc=0.7838 | val_f1=0.5690 | val_auc=0.9209 | reward=0.6227 | theta=race_focus (g=1.10, a=0.36, b=6.40, t=2.70, k=3, eg=0.17, mix=0.86)
    Client 3 (ds2) | train_acc=0.4950 | val_acc=0.7143 | val_f1=0.2778 | val_auc=nan | reward=0.3869 | theta=race_smoothmix (g=0.90, a=0.20, b=4.00, t=2.10, k=7, eg=0.07, mix=0.70)
    Client 4 (ds2) | train_acc=0.6681 | val_acc=0.8723 | val_f1=0.6664 | val_auc=0.8886 | reward=0.7179 | theta=race_smoothmix (g=0.90, a=0.20, b=4.00, t=2.10, k=7, eg=0.07, mix=0.70)
    Client 5 (ds2) | train_acc=0.6679 | val_acc=0.8919 | val_f1=0.8581 | val_auc=0.9565 | reward=0.8666 | theta=race_smoothmix (g=0.90, a=0.20, b=4.00, t=2.10, k=7, eg=0.07, mix=0.70)
    
    ----------------------------------------------------------------------------------------------------------------------
    GLOBAL VAL (Round 1) | global_acc=0.8294 | global_f1=0.6545 | ds1_acc=0.8053 | ds1_f1=0.6295 | ds2_acc=0.8571 | ds2_f1=0.6833 | reward=0.8011 | round_time=46.2s
    ----------------------------------------------------------------------------------------------------------------------
    
    ======================================================================================================================
    ROUND 2/15 | selected=[0, 1, 2, 3, 4, 5]
    ======================================================================================================================
    Client 0 (ds1) | train_acc=0.8671 | val_acc=0.8500 | val_f1=0.7491 | val_auc=0.9823 | reward=0.7743 | theta=race_robust (g=1.08, a=0.22, b=4.50, t=2.80, k=5, eg=0.11, mix=0.76)
    Client 1 (ds1) | train_acc=0.8346 | val_acc=0.8333 | val_f1=0.6181 | val_auc=0.9711 | reward=0.6719 | theta=race_edge_plus (g=1.02, a=0.32, b=6.00, t=2.60, k=3, eg=0.15, mix=0.84)
    Client 2 (ds1) | train_acc=0.8662 | val_acc=0.8919 | val_f1=0.8429 | val_auc=0.9756 | reward=0.8551 | theta=race_robust (g=1.08, a=0.22, b=4.50, t=2.80, k=5, eg=0.11, mix=0.76)
    Client 3 (ds2) | train_acc=0.7000 | val_acc=0.7857 | val_f1=0.6556 | val_auc=nan | reward=0.6881 | theta=race_robust (g=1.08, a=0.22, b=4.50, t=2.80, k=5, eg=0.11, mix=0.76)
    Client 4 (ds2) | train_acc=0.8012 | val_acc=0.8723 | val_f1=0.6552 | val_auc=0.9292 | reward=0.7095 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    Client 5 (ds2) | train_acc=0.8208 | val_acc=0.8378 | val_f1=0.7505 | val_auc=0.9539 | reward=0.7723 | theta=race_soft (g=0.95, a=0.18, b=3.80, t=2.20, k=3, eg=0.08, mix=0.72)
    
    ----------------------------------------------------------------------------------------------------------------------
    GLOBAL VAL (Round 2) | global_acc=0.8531 | global_f1=0.7163 | ds1_acc=0.8584 | ds1_f1=0.7380 | ds2_acc=0.8469 | ds2_f1=0.6912 | reward=0.8587 | round_time=32.6s
    ----------------------------------------------------------------------------------------------------------------------
    
    ======================================================================================================================
    ROUND 3/15 | selected=[0, 1, 2, 3, 4, 5]
    ======================================================================================================================
    Client 0 (ds1) | train_acc=0.8392 | val_acc=0.8000 | val_f1=0.6943 | val_auc=0.9631 | reward=0.7207 | theta=race_texture (g=0.92, a=0.34, b=5.80, t=2.30, k=7, eg=0.10, mix=0.80)
    Client 1 (ds1) | train_acc=0.8658 | val_acc=0.9444 | val_f1=0.6932 | val_auc=0.9803 | reward=0.7560 | theta=race_robust (g=1.08, a=0.22, b=4.50, t=2.80, k=5, eg=0.11, mix=0.76)
    Client 2 (ds1) | train_acc=0.7602 | val_acc=0.9189 | val_f1=0.8701 | val_auc=0.9873 | reward=0.8823 | theta=race_texture (g=0.92, a=0.34, b=5.80, t=2.30, k=7, eg=0.10, mix=0.80)
    Client 3 (ds2) | train_acc=0.6600 | val_acc=0.8571 | val_f1=0.5857 | val_auc=nan | reward=0.6536 | theta=race_robust (g=1.08, a=0.22, b=4.50, t=2.80, k=5, eg=0.11, mix=0.76)
    Client 4 (ds2) | train_acc=0.8173 | val_acc=0.8936 | val_f1=0.6672 | val_auc=0.9789 | reward=0.7238 | theta=race_robust (g=1.08, a=0.22, b=4.50, t=2.80, k=5, eg=0.11, mix=0.76)
    Client 5 (ds2) | train_acc=0.8019 | val_acc=0.8108 | val_f1=0.6587 | val_auc=0.9077 | reward=0.6968 | theta=race_robust (g=1.08, a=0.22, b=4.50, t=2.80, k=5, eg=0.11, mix=0.76)
    
    ----------------------------------------------------------------------------------------------------------------------
    GLOBAL VAL (Round 3) | global_acc=0.8720 | global_f1=0.7055 | ds1_acc=0.8850 | ds1_f1=0.7515 | ds2_acc=0.8571 | ds2_f1=0.6524 | reward=0.8497 | round_time=73.0s
    ----------------------------------------------------------------------------------------------------------------------
    
    ======================================================================================================================
    ROUND 4/15 | selected=[0, 1, 2, 3, 4, 5]
    ======================================================================================================================
    Client 0 (ds1) | train_acc=0.9108 | val_acc=0.8250 | val_f1=0.6889 | val_auc=0.9709 | reward=0.7229 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    Client 1 (ds1) | train_acc=0.9144 | val_acc=0.9167 | val_f1=0.6851 | val_auc=0.9969 | reward=0.7430 | theta=race_sharp (g=1.04, a=0.30, b=5.20, t=2.50, k=5, eg=0.12, mix=0.82)
    Client 2 (ds1) | train_acc=0.8662 | val_acc=0.9189 | val_f1=0.8992 | val_auc=0.9920 | reward=0.9041 | theta=race_sharp (g=1.04, a=0.30, b=5.20, t=2.50, k=5, eg=0.12, mix=0.82)
    Client 3 (ds2) | train_acc=0.7200 | val_acc=0.7857 | val_f1=0.5167 | val_auc=nan | reward=0.5839 | theta=race_smoothmix (g=0.90, a=0.20, b=4.00, t=2.10, k=7, eg=0.07, mix=0.70)
    Client 4 (ds2) | train_acc=0.9035 | val_acc=0.9149 | val_f1=0.6847 | val_auc=0.9972 | reward=0.7422 | theta=race_robust (g=1.08, a=0.22, b=4.50, t=2.80, k=5, eg=0.11, mix=0.76)
    Client 5 (ds2) | train_acc=0.8604 | val_acc=0.8378 | val_f1=0.7745 | val_auc=0.9725 | reward=0.7903 | theta=race_smoothmix (g=0.90, a=0.20, b=4.00, t=2.10, k=7, eg=0.07, mix=0.70)
    
    ----------------------------------------------------------------------------------------------------------------------
    GLOBAL VAL (Round 4) | global_acc=0.8768 | global_f1=0.7278 | ds1_acc=0.8850 | ds1_f1=0.7565 | ds2_acc=0.8673 | ds2_f1=0.6946 | reward=0.8739 | round_time=39.9s
    ----------------------------------------------------------------------------------------------------------------------
    
    ======================================================================================================================
    ROUND 5/15 | selected=[0, 1, 2, 3, 4, 5]
    ======================================================================================================================
    Client 0 (ds1) | train_acc=0.9091 | val_acc=0.9250 | val_f1=0.8613 | val_auc=0.9929 | reward=0.8773 | theta=race_focus (g=1.10, a=0.36, b=6.40, t=2.70, k=3, eg=0.17, mix=0.86)
    Client 1 (ds1) | train_acc=0.9436 | val_acc=0.8889 | val_f1=0.6480 | val_auc=0.8444 | reward=0.7082 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    Client 2 (ds1) | train_acc=0.9071 | val_acc=0.9730 | val_f1=0.9442 | val_auc=1.0000 | reward=0.9514 | theta=race_sharp (g=1.04, a=0.30, b=5.20, t=2.50, k=5, eg=0.12, mix=0.82)
    Client 3 (ds2) | train_acc=0.7550 | val_acc=0.9286 | val_f1=0.9175 | val_auc=nan | reward=0.9202 | theta=race_soft (g=0.95, a=0.18, b=3.80, t=2.20, k=3, eg=0.08, mix=0.72)
    Client 4 (ds2) | train_acc=0.8772 | val_acc=0.8936 | val_f1=0.6735 | val_auc=0.9655 | reward=0.7286 | theta=race_soft (g=0.95, a=0.18, b=3.80, t=2.20, k=3, eg=0.08, mix=0.72)
    Client 5 (ds2) | train_acc=0.8868 | val_acc=0.8649 | val_f1=0.8374 | val_auc=0.9820 | reward=0.8443 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    
    ----------------------------------------------------------------------------------------------------------------------
    GLOBAL VAL (Round 5) | global_acc=0.9100 | global_f1=0.7972 | ds1_acc=0.9292 | ds1_f1=0.8205 | ds2_acc=0.8878 | ds2_f1=0.7702 | reward=0.9436 | round_time=39.3s
    ----------------------------------------------------------------------------------------------------------------------
    
    ======================================================================================================================
    ROUND 6/15 | selected=[0, 1, 2, 3, 4, 5]
    ======================================================================================================================
    Client 0 (ds1) | train_acc=0.9301 | val_acc=0.8750 | val_f1=0.7560 | val_auc=0.9786 | reward=0.7858 | theta=race_sharp (g=1.04, a=0.30, b=5.20, t=2.50, k=5, eg=0.12, mix=0.82)
    Client 1 (ds1) | train_acc=0.9241 | val_acc=0.9444 | val_f1=0.7018 | val_auc=0.9980 | reward=0.7625 | theta=race_texture (g=0.92, a=0.34, b=5.80, t=2.30, k=7, eg=0.10, mix=0.80)
    Client 2 (ds1) | train_acc=0.9238 | val_acc=0.9730 | val_f1=0.9664 | val_auc=1.0000 | reward=0.9680 | theta=race_texture (g=0.92, a=0.34, b=5.80, t=2.30, k=7, eg=0.10, mix=0.80)
    Client 3 (ds2) | train_acc=0.8850 | val_acc=0.9286 | val_f1=0.9175 | val_auc=nan | reward=0.9202 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    Client 4 (ds2) | train_acc=0.9386 | val_acc=0.9787 | val_f1=0.9777 | val_auc=0.9995 | reward=0.9780 | theta=race_soft (g=0.95, a=0.18, b=3.80, t=2.20, k=3, eg=0.08, mix=0.72)
    Client 5 (ds2) | train_acc=0.9302 | val_acc=0.8649 | val_f1=0.8202 | val_auc=0.9523 | reward=0.8313 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    
    ----------------------------------------------------------------------------------------------------------------------
    GLOBAL VAL (Round 6) | global_acc=0.9289 | global_f1=0.8550 | ds1_acc=0.9292 | ds1_f1=0.8076 | ds2_acc=0.9286 | ds2_f1=0.9096 | reward=1.0019 | round_time=39.8s
    ----------------------------------------------------------------------------------------------------------------------
    
    ======================================================================================================================
    ROUND 7/15 | selected=[0, 1, 2, 3, 4, 5]
    ======================================================================================================================
    Client 0 (ds1) | train_acc=0.9493 | val_acc=0.9000 | val_f1=0.8549 | val_auc=0.9758 | reward=0.8662 | theta=race_focus (g=1.10, a=0.36, b=6.40, t=2.70, k=3, eg=0.17, mix=0.86)
    Client 1 (ds1) | train_acc=0.9553 | val_acc=0.9722 | val_f1=0.7222 | val_auc=1.0000 | reward=0.7847 | theta=race_texture (g=0.92, a=0.34, b=5.80, t=2.30, k=7, eg=0.10, mix=0.80)
    Client 2 (ds1) | train_acc=0.9480 | val_acc=0.9459 | val_f1=0.9381 | val_auc=0.9981 | reward=0.9401 | theta=race_edge_plus (g=1.02, a=0.32, b=6.00, t=2.60, k=3, eg=0.15, mix=0.84)
    Client 3 (ds2) | train_acc=0.8650 | val_acc=0.9286 | val_f1=0.9175 | val_auc=nan | reward=0.9202 | theta=race_soft (g=0.95, a=0.18, b=3.80, t=2.20, k=3, eg=0.08, mix=0.72)
    Client 4 (ds2) | train_acc=0.9225 | val_acc=0.9574 | val_f1=0.9639 | val_auc=0.9985 | reward=0.9623 | theta=race_smoothmix (g=0.90, a=0.20, b=4.00, t=2.10, k=7, eg=0.07, mix=0.70)
    Client 5 (ds2) | train_acc=0.8868 | val_acc=0.8919 | val_f1=0.8283 | val_auc=0.9752 | reward=0.8442 | theta=race_soft (g=0.95, a=0.18, b=3.80, t=2.20, k=3, eg=0.08, mix=0.72)
    
    ----------------------------------------------------------------------------------------------------------------------
    GLOBAL VAL (Round 7) | global_acc=0.9336 | global_f1=0.8706 | ds1_acc=0.9381 | ds1_f1=0.8399 | ds2_acc=0.9286 | ds2_f1=0.9061 | reward=1.0177 | round_time=39.4s
    ----------------------------------------------------------------------------------------------------------------------
    
    ======================================================================================================================
    ROUND 8/15 | selected=[0, 1, 2, 3, 4, 5]
    ======================================================================================================================
    Client 0 (ds1) | train_acc=0.9423 | val_acc=0.8750 | val_f1=0.8396 | val_auc=0.9945 | reward=0.8484 | theta=race_sharp (g=1.04, a=0.30, b=5.20, t=2.50, k=5, eg=0.12, mix=0.82)
    Client 1 (ds1) | train_acc=0.9689 | val_acc=0.9444 | val_f1=0.8750 | val_auc=0.9980 | reward=0.8924 | theta=race_robust (g=1.08, a=0.22, b=4.50, t=2.80, k=5, eg=0.11, mix=0.76)
    Client 2 (ds1) | train_acc=0.9387 | val_acc=0.9459 | val_f1=0.9472 | val_auc=1.0000 | reward=0.9469 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    Client 3 (ds2) | train_acc=0.9850 | val_acc=0.9286 | val_f1=0.9175 | val_auc=nan | reward=0.9202 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    Client 4 (ds2) | train_acc=0.9386 | val_acc=0.9149 | val_f1=0.6807 | val_auc=0.9967 | reward=0.7392 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    Client 5 (ds2) | train_acc=0.9208 | val_acc=0.8919 | val_f1=0.8650 | val_auc=0.9895 | reward=0.8717 | theta=race_robust (g=1.08, a=0.22, b=4.50, t=2.80, k=5, eg=0.11, mix=0.76)
    
    ----------------------------------------------------------------------------------------------------------------------
    GLOBAL VAL (Round 8) | global_acc=0.9147 | global_f1=0.8387 | ds1_acc=0.9204 | ds1_f1=0.8861 | ds2_acc=0.9082 | ds2_f1=0.7841 | reward=0.9772 | round_time=39.9s
    ----------------------------------------------------------------------------------------------------------------------
    
    ======================================================================================================================
    ROUND 9/15 | selected=[0, 1, 2, 3, 4, 5]
    ======================================================================================================================
    Client 0 (ds1) | train_acc=0.9650 | val_acc=0.9250 | val_f1=0.8872 | val_auc=0.9920 | reward=0.8967 | theta=race_robust (g=1.08, a=0.22, b=4.50, t=2.80, k=5, eg=0.11, mix=0.76)
    Client 1 (ds1) | train_acc=0.9300 | val_acc=0.9167 | val_f1=0.6819 | val_auc=0.9992 | reward=0.7406 | theta=race_edge_plus (g=1.02, a=0.32, b=6.00, t=2.60, k=3, eg=0.15, mix=0.84)
    Client 2 (ds1) | train_acc=0.9591 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | reward=1.0000 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    Client 3 (ds2) | train_acc=0.9600 | val_acc=0.9286 | val_f1=0.9175 | val_auc=nan | reward=0.9202 | theta=race_soft (g=0.95, a=0.18, b=3.80, t=2.20, k=3, eg=0.08, mix=0.72)
    Client 4 (ds2) | train_acc=0.9503 | val_acc=0.9362 | val_f1=0.7061 | val_auc=0.9872 | reward=0.7636 | theta=race_soft (g=0.95, a=0.18, b=3.80, t=2.20, k=3, eg=0.08, mix=0.72)
    Client 5 (ds2) | train_acc=0.9472 | val_acc=0.8919 | val_f1=0.8581 | val_auc=0.9448 | reward=0.8666 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    
    ----------------------------------------------------------------------------------------------------------------------
    GLOBAL VAL (Round 9) | global_acc=0.9336 | global_f1=0.8285 | ds1_acc=0.9469 | ds1_f1=0.8587 | ds2_acc=0.9184 | ds2_f1=0.7937 | reward=0.9765 | round_time=39.6s
    ----------------------------------------------------------------------------------------------------------------------
    
    ======================================================================================================================
    ROUND 10/15 | selected=[0, 1, 2, 3, 4, 5]
    ======================================================================================================================
    Client 0 (ds1) | train_acc=0.9668 | val_acc=0.9250 | val_f1=0.8613 | val_auc=0.9873 | reward=0.8773 | theta=race_edge_plus (g=1.02, a=0.32, b=6.00, t=2.60, k=3, eg=0.15, mix=0.84)
    Client 1 (ds1) | train_acc=0.9786 | val_acc=0.9444 | val_f1=0.6932 | val_auc=1.0000 | reward=0.7560 | theta=race_sharp (g=1.04, a=0.30, b=5.20, t=2.50, k=5, eg=0.12, mix=0.82)
    Client 2 (ds1) | train_acc=0.9424 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | reward=1.0000 | theta=race_edge_plus (g=1.02, a=0.32, b=6.00, t=2.60, k=3, eg=0.15, mix=0.84)
    Client 3 (ds2) | train_acc=0.9650 | val_acc=0.9286 | val_f1=0.9175 | val_auc=nan | reward=0.9202 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    Client 4 (ds2) | train_acc=0.9810 | val_acc=0.9787 | val_f1=0.9762 | val_auc=0.9644 | reward=0.9768 | theta=race_smoothmix (g=0.90, a=0.20, b=4.00, t=2.10, k=7, eg=0.07, mix=0.70)
    Client 5 (ds2) | train_acc=0.9472 | val_acc=0.9459 | val_f1=0.9342 | val_auc=0.9937 | reward=0.9371 | theta=race_smoothmix (g=0.90, a=0.20, b=4.00, t=2.10, k=7, eg=0.07, mix=0.70)
    
    ----------------------------------------------------------------------------------------------------------------------
    GLOBAL VAL (Round 10) | global_acc=0.9573 | global_f1=0.8991 | ds1_acc=0.9558 | ds1_f1=0.8532 | ds2_acc=0.9592 | ds2_f1=0.9519 | reward=1.0481 | round_time=39.7s
    ----------------------------------------------------------------------------------------------------------------------
    
    ======================================================================================================================
    ROUND 11/15 | selected=[0, 1, 2, 3, 4, 5]
    ======================================================================================================================
    Client 0 (ds1) | train_acc=0.9528 | val_acc=0.9000 | val_f1=0.8126 | val_auc=0.9885 | reward=0.8344 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    Client 1 (ds1) | train_acc=0.9533 | val_acc=0.9722 | val_f1=0.9655 | val_auc=0.9973 | reward=0.9672 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    Client 2 (ds1) | train_acc=0.9554 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | reward=1.0000 | theta=race_robust (g=1.08, a=0.22, b=4.50, t=2.80, k=5, eg=0.11, mix=0.76)
    Client 3 (ds2) | train_acc=0.9600 | val_acc=1.0000 | val_f1=1.0000 | val_auc=nan | reward=1.0000 | theta=race_robust (g=1.08, a=0.22, b=4.50, t=2.80, k=5, eg=0.11, mix=0.76)
    Client 4 (ds2) | train_acc=0.9561 | val_acc=0.8936 | val_f1=0.6803 | val_auc=0.9885 | reward=0.7337 | theta=race_robust (g=1.08, a=0.22, b=4.50, t=2.80, k=5, eg=0.11, mix=0.76)
    Client 5 (ds2) | train_acc=0.9245 | val_acc=0.8919 | val_f1=0.8283 | val_auc=0.9703 | reward=0.8442 | theta=race_soft (g=0.95, a=0.18, b=3.80, t=2.20, k=3, eg=0.08, mix=0.72)
    
    ----------------------------------------------------------------------------------------------------------------------
    GLOBAL VAL (Round 11) | global_acc=0.9336 | global_f1=0.8573 | ds1_acc=0.9558 | ds1_f1=0.9227 | ds2_acc=0.9082 | ds2_f1=0.7819 | reward=0.9942 | round_time=39.6s
    ----------------------------------------------------------------------------------------------------------------------
    
    ======================================================================================================================
    ROUND 12/15 | selected=[0, 1, 2, 3, 4, 5]
    ======================================================================================================================
    Client 0 (ds1) | train_acc=0.9580 | val_acc=0.9000 | val_f1=0.8030 | val_auc=0.9904 | reward=0.8273 | theta=race_texture (g=0.92, a=0.34, b=5.80, t=2.30, k=7, eg=0.10, mix=0.80)
    Client 1 (ds1) | train_acc=0.9611 | val_acc=0.8889 | val_f1=0.6452 | val_auc=0.9992 | reward=0.7062 | theta=race_focus (g=1.10, a=0.36, b=6.40, t=2.70, k=3, eg=0.17, mix=0.86)
    Client 2 (ds1) | train_acc=0.9572 | val_acc=0.9459 | val_f1=0.9472 | val_auc=1.0000 | reward=0.9469 | theta=race_focus (g=1.10, a=0.36, b=6.40, t=2.70, k=3, eg=0.17, mix=0.86)
    Client 3 (ds2) | train_acc=0.9500 | val_acc=1.0000 | val_f1=1.0000 | val_auc=nan | reward=1.0000 | theta=race_soft (g=0.95, a=0.18, b=3.80, t=2.20, k=3, eg=0.08, mix=0.72)
    Client 4 (ds2) | train_acc=0.9693 | val_acc=0.9787 | val_f1=0.9777 | val_auc=0.9982 | reward=0.9780 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    Client 5 (ds2) | train_acc=0.9491 | val_acc=0.9189 | val_f1=0.8907 | val_auc=0.9937 | reward=0.8978 | theta=race_robust (g=1.08, a=0.22, b=4.50, t=2.80, k=5, eg=0.11, mix=0.76)
    
    ----------------------------------------------------------------------------------------------------------------------
    GLOBAL VAL (Round 12) | global_acc=0.9336 | global_f1=0.8687 | ds1_acc=0.9115 | ds1_f1=0.7999 | ds2_acc=0.9592 | ds2_f1=0.9481 | reward=1.0135 | round_time=39.6s
    ----------------------------------------------------------------------------------------------------------------------
    
    ======================================================================================================================
    ROUND 13/15 | selected=[0, 1, 2, 3, 4, 5]
    ======================================================================================================================
    Client 0 (ds1) | train_acc=0.9685 | val_acc=0.9500 | val_f1=0.9228 | val_auc=0.9994 | reward=0.9296 | theta=race_focus (g=1.10, a=0.36, b=6.40, t=2.70, k=3, eg=0.17, mix=0.86)
    Client 1 (ds1) | train_acc=0.9533 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | reward=1.0000 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    Client 2 (ds1) | train_acc=0.9777 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | reward=1.0000 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    Client 3 (ds2) | train_acc=0.9550 | val_acc=0.9286 | val_f1=0.9175 | val_auc=nan | reward=0.9202 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    Client 4 (ds2) | train_acc=0.9693 | val_acc=0.9574 | val_f1=0.9639 | val_auc=0.9887 | reward=0.9623 | theta=race_smoothmix (g=0.90, a=0.20, b=4.00, t=2.10, k=7, eg=0.07, mix=0.70)
    Client 5 (ds2) | train_acc=0.9509 | val_acc=0.9459 | val_f1=0.9167 | val_auc=0.9910 | reward=0.9240 | theta=race_smoothmix (g=0.90, a=0.20, b=4.00, t=2.10, k=7, eg=0.07, mix=0.70)
    
    ----------------------------------------------------------------------------------------------------------------------
    GLOBAL VAL (Round 13) | global_acc=0.9668 | global_f1=0.9572 | ds1_acc=0.9823 | ds1_f1=0.9727 | ds2_acc=0.9490 | ds2_f1=0.9394 | reward=1.0997 | round_time=39.5s
    ----------------------------------------------------------------------------------------------------------------------
    
    ======================================================================================================================
    ROUND 14/15 | selected=[0, 1, 2, 3, 4, 5]
    ======================================================================================================================
    Client 0 (ds1) | train_acc=0.9825 | val_acc=0.9250 | val_f1=0.8485 | val_auc=0.9975 | reward=0.8676 | theta=race_robust (g=1.08, a=0.22, b=4.50, t=2.80, k=5, eg=0.11, mix=0.76)
    Client 1 (ds1) | train_acc=0.9825 | val_acc=1.0000 | val_f1=1.0000 | val_auc=1.0000 | reward=1.0000 | theta=race_robust (g=1.08, a=0.22, b=4.50, t=2.80, k=5, eg=0.11, mix=0.76)
    Client 2 (ds1) | train_acc=0.9796 | val_acc=0.9730 | val_f1=0.9664 | val_auc=1.0000 | reward=0.9680 | theta=race_edge_plus (g=1.02, a=0.32, b=6.00, t=2.60, k=3, eg=0.15, mix=0.84)
    Client 3 (ds2) | train_acc=0.9800 | val_acc=0.9286 | val_f1=0.9175 | val_auc=nan | reward=0.9202 | theta=race_robust (g=1.08, a=0.22, b=4.50, t=2.80, k=5, eg=0.11, mix=0.76)
    Client 4 (ds2) | train_acc=0.9781 | val_acc=0.9362 | val_f1=0.9402 | val_auc=0.9930 | reward=0.9392 | theta=race_soft (g=0.95, a=0.18, b=3.80, t=2.20, k=3, eg=0.08, mix=0.72)
    Client 5 (ds2) | train_acc=0.9623 | val_acc=0.8649 | val_f1=0.8317 | val_auc=0.9733 | reward=0.8400 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    
    ----------------------------------------------------------------------------------------------------------------------
    GLOBAL VAL (Round 14) | global_acc=0.9384 | global_f1=0.9171 | ds1_acc=0.9646 | ds1_f1=0.9354 | ds2_acc=0.9082 | ds2_f1=0.8960 | reward=1.0557 | round_time=39.8s
    ----------------------------------------------------------------------------------------------------------------------
    
    ======================================================================================================================
    ROUND 15/15 | selected=[0, 1, 2, 3, 4, 5]
    ======================================================================================================================
    Client 0 (ds1) | train_acc=0.9703 | val_acc=0.8500 | val_f1=0.7604 | val_auc=0.9910 | reward=0.7828 | theta=race_sharp (g=1.04, a=0.30, b=5.20, t=2.50, k=5, eg=0.12, mix=0.82)
    Client 1 (ds1) | train_acc=0.9689 | val_acc=0.9722 | val_f1=0.9832 | val_auc=0.9992 | reward=0.9805 | theta=race_texture (g=0.92, a=0.34, b=5.80, t=2.30, k=7, eg=0.10, mix=0.80)
    Client 2 (ds1) | train_acc=0.9740 | val_acc=0.9459 | val_f1=0.9103 | val_auc=0.9977 | reward=0.9192 | theta=race_sharp (g=1.04, a=0.30, b=5.20, t=2.50, k=5, eg=0.12, mix=0.82)
    Client 3 (ds2) | train_acc=0.9750 | val_acc=0.9286 | val_f1=0.9175 | val_auc=nan | reward=0.9202 | theta=race_smoothmix (g=0.90, a=0.20, b=4.00, t=2.10, k=7, eg=0.07, mix=0.70)
    Client 4 (ds2) | train_acc=0.9678 | val_acc=0.9574 | val_f1=0.7204 | val_auc=1.0000 | reward=0.7796 | theta=race_balanced (g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10, mix=0.78)
    Client 5 (ds2) | train_acc=0.9604 | val_acc=0.9459 | val_f1=0.9342 | val_auc=0.9538 | reward=0.9371 | theta=race_robust (g=1.08, a=0.22, b=4.50, t=2.80, k=5, eg=0.11, mix=0.76)
    
    ----------------------------------------------------------------------------------------------------------------------
    GLOBAL VAL (Round 15) | global_acc=0.9336 | global_f1=0.8567 | ds1_acc=0.9204 | ds1_f1=0.8805 | ds2_acc=0.9490 | ds2_f1=0.8293 | reward=1.0037 | round_time=39.7s
    ----------------------------------------------------------------------------------------------------------------------
    
    ======================================================================================================================
    TRAINING COMPLETE ✅ | total_time=628.1s | best_round=13 | best_reward=1.0997
    ======================================================================================================================
    
    ----------------------------------------------------------------------------------------------------------------------
    GLOBAL per-round metrics
    ----------------------------------------------------------------------------------------------------------------------



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
      <td>46.176245</td>
      <td>6</td>
      <td>1.0</td>
      <td>0.801129</td>
      <td>0.829384</td>
      <td>0.654492</td>
      <td>0.656613</td>
      <td>0.679686</td>
      <td>0.507187</td>
      <td>0.475219</td>
      <td>1.046054</td>
      <td>0.805310</td>
      <td>0.629526</td>
      <td>0.481677</td>
      <td>0.857143</td>
      <td>0.683278</td>
      <td>0.536601</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>32.601531</td>
      <td>6</td>
      <td>1.0</td>
      <td>0.858654</td>
      <td>0.853081</td>
      <td>0.716285</td>
      <td>0.744904</td>
      <td>0.745441</td>
      <td>0.406082</td>
      <td>0.384221</td>
      <td>0.449692</td>
      <td>0.858407</td>
      <td>0.738034</td>
      <td>0.399528</td>
      <td>0.846939</td>
      <td>0.691206</td>
      <td>0.413639</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>72.990236</td>
      <td>6</td>
      <td>1.0</td>
      <td>0.849746</td>
      <td>0.872038</td>
      <td>0.705457</td>
      <td>0.700379</td>
      <td>0.727640</td>
      <td>0.389847</td>
      <td>0.386818</td>
      <td>0.470125</td>
      <td>0.884956</td>
      <td>0.751500</td>
      <td>0.363169</td>
      <td>0.857143</td>
      <td>0.652365</td>
      <td>0.420608</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>39.883242</td>
      <td>6</td>
      <td>1.0</td>
      <td>0.873869</td>
      <td>0.876777</td>
      <td>0.727759</td>
      <td>0.753007</td>
      <td>0.733450</td>
      <td>0.294105</td>
      <td>0.276226</td>
      <td>0.480275</td>
      <td>0.884956</td>
      <td>0.756544</td>
      <td>0.285949</td>
      <td>0.867347</td>
      <td>0.694569</td>
      <td>0.303510</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>39.318045</td>
      <td>6</td>
      <td>1.0</td>
      <td>0.943593</td>
      <td>0.909953</td>
      <td>0.797159</td>
      <td>0.795506</td>
      <td>0.812005</td>
      <td>0.271582</td>
      <td>0.250217</td>
      <td>0.469900</td>
      <td>0.929204</td>
      <td>0.820502</td>
      <td>0.231583</td>
      <td>0.887755</td>
      <td>0.770243</td>
      <td>0.317702</td>
    </tr>
    <tr>
      <th>5</th>
      <td>6</td>
      <td>39.835706</td>
      <td>6</td>
      <td>1.0</td>
      <td>1.001901</td>
      <td>0.928910</td>
      <td>0.855008</td>
      <td>0.875600</td>
      <td>0.863199</td>
      <td>0.220376</td>
      <td>0.206641</td>
      <td>0.468578</td>
      <td>0.929204</td>
      <td>0.807639</td>
      <td>0.211708</td>
      <td>0.928571</td>
      <td>0.909627</td>
      <td>0.230372</td>
    </tr>
    <tr>
      <th>6</th>
      <td>7</td>
      <td>39.419170</td>
      <td>6</td>
      <td>1.0</td>
      <td>1.017711</td>
      <td>0.933649</td>
      <td>0.870609</td>
      <td>0.873087</td>
      <td>0.878678</td>
      <td>0.235390</td>
      <td>0.210590</td>
      <td>0.461903</td>
      <td>0.938053</td>
      <td>0.839863</td>
      <td>0.245069</td>
      <td>0.928571</td>
      <td>0.906061</td>
      <td>0.224230</td>
    </tr>
    <tr>
      <th>7</th>
      <td>8</td>
      <td>39.928579</td>
      <td>6</td>
      <td>1.0</td>
      <td>0.977151</td>
      <td>0.914692</td>
      <td>0.838717</td>
      <td>0.841190</td>
      <td>0.853152</td>
      <td>0.215437</td>
      <td>0.209185</td>
      <td>0.477264</td>
      <td>0.920354</td>
      <td>0.886081</td>
      <td>0.166443</td>
      <td>0.908163</td>
      <td>0.784103</td>
      <td>0.271928</td>
    </tr>
    <tr>
      <th>8</th>
      <td>9</td>
      <td>39.643975</td>
      <td>6</td>
      <td>1.0</td>
      <td>0.976545</td>
      <td>0.933649</td>
      <td>0.828521</td>
      <td>0.848223</td>
      <td>0.819749</td>
      <td>0.219279</td>
      <td>0.194634</td>
      <td>0.470844</td>
      <td>0.946903</td>
      <td>0.858729</td>
      <td>0.167369</td>
      <td>0.918367</td>
      <td>0.793690</td>
      <td>0.279135</td>
    </tr>
    <tr>
      <th>9</th>
      <td>10</td>
      <td>39.662265</td>
      <td>6</td>
      <td>1.0</td>
      <td>1.048118</td>
      <td>0.957346</td>
      <td>0.899058</td>
      <td>0.913467</td>
      <td>0.902458</td>
      <td>0.181208</td>
      <td>0.171453</td>
      <td>0.473189</td>
      <td>0.955752</td>
      <td>0.853192</td>
      <td>0.185194</td>
      <td>0.959184</td>
      <td>0.951944</td>
      <td>0.176612</td>
    </tr>
    <tr>
      <th>10</th>
      <td>11</td>
      <td>39.572632</td>
      <td>6</td>
      <td>1.0</td>
      <td>0.994210</td>
      <td>0.933649</td>
      <td>0.857274</td>
      <td>0.862336</td>
      <td>0.863760</td>
      <td>0.232793</td>
      <td>0.213709</td>
      <td>0.472966</td>
      <td>0.955752</td>
      <td>0.922657</td>
      <td>0.145243</td>
      <td>0.908163</td>
      <td>0.781883</td>
      <td>0.333744</td>
    </tr>
    <tr>
      <th>11</th>
      <td>12</td>
      <td>39.552953</td>
      <td>6</td>
      <td>1.0</td>
      <td>1.013515</td>
      <td>0.933649</td>
      <td>0.868740</td>
      <td>0.875239</td>
      <td>0.891662</td>
      <td>0.209199</td>
      <td>0.184620</td>
      <td>0.467979</td>
      <td>0.911504</td>
      <td>0.799947</td>
      <td>0.219725</td>
      <td>0.959184</td>
      <td>0.948063</td>
      <td>0.197062</td>
    </tr>
    <tr>
      <th>12</th>
      <td>13</td>
      <td>39.539547</td>
      <td>6</td>
      <td>1.0</td>
      <td>1.099716</td>
      <td>0.966825</td>
      <td>0.957230</td>
      <td>0.961963</td>
      <td>0.957554</td>
      <td>0.142549</td>
      <td>0.131432</td>
      <td>0.465227</td>
      <td>0.982301</td>
      <td>0.972684</td>
      <td>0.075632</td>
      <td>0.948980</td>
      <td>0.939411</td>
      <td>0.219708</td>
    </tr>
    <tr>
      <th>13</th>
      <td>14</td>
      <td>39.822155</td>
      <td>6</td>
      <td>1.0</td>
      <td>1.055704</td>
      <td>0.938389</td>
      <td>0.917073</td>
      <td>0.926712</td>
      <td>0.926308</td>
      <td>0.211887</td>
      <td>0.199653</td>
      <td>0.465868</td>
      <td>0.964602</td>
      <td>0.935367</td>
      <td>0.119414</td>
      <td>0.908163</td>
      <td>0.895979</td>
      <td>0.318515</td>
    </tr>
    <tr>
      <th>14</th>
      <td>15</td>
      <td>39.694266</td>
      <td>6</td>
      <td>1.0</td>
      <td>1.003700</td>
      <td>0.933649</td>
      <td>0.856694</td>
      <td>0.875163</td>
      <td>0.863632</td>
      <td>0.246994</td>
      <td>0.242846</td>
      <td>0.463480</td>
      <td>0.920354</td>
      <td>0.880488</td>
      <td>0.260339</td>
      <td>0.948980</td>
      <td>0.829258</td>
      <td>0.231607</td>
    </tr>
  </tbody>
</table>
</div>


    
    ----------------------------------------------------------------------------------------------------------------------
    LOCAL per-client per-round metrics
    ----------------------------------------------------------------------------------------------------------------------



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
      <td>4</td>
      <td>race_edge_plus</td>
      <td>(g=1.02, a=0.32, b=6.00, t=2.60, k=3, eg=0.15,...</td>
      <td>1.02</td>
      <td>0.32</td>
      <td>6.0</td>
      <td>...</td>
      <td>0.964160</td>
      <td>0.911765</td>
      <td>0.972222</td>
      <td>0.976982</td>
      <td>0.995671</td>
      <td>0.497006</td>
      <td>0.105720</td>
      <td>0.397274</td>
      <td>1.198737</td>
      <td>0.740094</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>client_1</td>
      <td>ds1</td>
      <td>1</td>
      <td>5</td>
      <td>race_focus</td>
      <td>(g=1.10, a=0.36, b=6.40, t=2.70, k=3, eg=0.17,...</td>
      <td>1.10</td>
      <td>0.36</td>
      <td>6.4</td>
      <td>...</td>
      <td>0.984390</td>
      <td>0.956656</td>
      <td>1.000000</td>
      <td>0.984375</td>
      <td>0.996528</td>
      <td>0.331394</td>
      <td>0.331605</td>
      <td>0.337000</td>
      <td>1.568851</td>
      <td>0.651598</td>
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
      <td>0.920924</td>
      <td>0.765152</td>
      <td>0.960606</td>
      <td>0.985714</td>
      <td>0.972222</td>
      <td>0.441696</td>
      <td>0.092870</td>
      <td>0.465433</td>
      <td>1.307301</td>
      <td>0.622732</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
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
      <td>NaN</td>
      <td>0.900000</td>
      <td>0.939394</td>
      <td>NaN</td>
      <td>1.000000</td>
      <td>0.391165</td>
      <td>0.608227</td>
      <td>0.000608</td>
      <td>0.965033</td>
      <td>0.386905</td>
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
      <td>0.888578</td>
      <td>0.652174</td>
      <td>0.925490</td>
      <td>0.979853</td>
      <td>0.996795</td>
      <td>0.423859</td>
      <td>0.354534</td>
      <td>0.221607</td>
      <td>1.473821</td>
      <td>0.717896</td>
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
      <td>2</td>
      <td>race_texture</td>
      <td>(g=0.92, a=0.34, b=5.80, t=2.30, k=7, eg=0.10,...</td>
      <td>0.92</td>
      <td>0.34</td>
      <td>5.8</td>
      <td>...</td>
      <td>0.999226</td>
      <td>0.996904</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.594365</td>
      <td>0.203677</td>
      <td>0.201959</td>
      <td>1.169526</td>
      <td>0.980488</td>
    </tr>
    <tr>
      <th>86</th>
      <td>15</td>
      <td>client_2</td>
      <td>ds1</td>
      <td>1</td>
      <td>1</td>
      <td>race_sharp</td>
      <td>(g=1.04, a=0.30, b=5.20, t=2.50, k=5, eg=0.12,...</td>
      <td>1.04</td>
      <td>0.30</td>
      <td>5.2</td>
      <td>...</td>
      <td>0.997727</td>
      <td>1.000000</td>
      <td>0.990909</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.539880</td>
      <td>0.111287</td>
      <td>0.348833</td>
      <td>1.149644</td>
      <td>0.919225</td>
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
      <td>NaN</td>
      <td>0.750000</td>
      <td>0.969697</td>
      <td>NaN</td>
      <td>1.000000</td>
      <td>0.796090</td>
      <td>0.119535</td>
      <td>0.084376</td>
      <td>0.816913</td>
      <td>0.920238</td>
    </tr>
    <tr>
      <th>88</th>
      <td>15</td>
      <td>client_4</td>
      <td>ds2</td>
      <td>1</td>
      <td>1</td>
      <td>race_balanced</td>
      <td>(g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10,...</td>
      <td>1.00</td>
      <td>0.24</td>
      <td>4.6</td>
      <td>...</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.497076</td>
      <td>0.268752</td>
      <td>0.234172</td>
      <td>1.451062</td>
      <td>0.779644</td>
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
      <td>0.953788</td>
      <td>0.966667</td>
      <td>0.848485</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.643867</td>
      <td>0.143069</td>
      <td>0.213063</td>
      <td>1.163668</td>
      <td>0.937131</td>
    </tr>
  </tbody>
</table>
<p>90 rows × 39 columns</p>
</div>


    
    ======================================================================================================================
    STEP 12: FINAL EVALUATION
    ======================================================================================================================
    
    ----------------------------------------------------------------------------------------------------------------------
    Best RL-selected preprocessing preset per client
    ----------------------------------------------------------------------------------------------------------------------



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
      <td>5</td>
      <td>race_focus</td>
      <td>(g=1.10, a=0.36, b=6.40, t=2.70, k=3, eg=0.17,...</td>
      <td>0.891014</td>
      <td>13</td>
    </tr>
    <tr>
      <th>1</th>
      <td>client_1</td>
      <td>ds1</td>
      <td>0</td>
      <td>race_balanced</td>
      <td>(g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10,...</td>
      <td>0.891792</td>
      <td>13</td>
    </tr>
    <tr>
      <th>2</th>
      <td>client_2</td>
      <td>ds1</td>
      <td>0</td>
      <td>race_balanced</td>
      <td>(g=1.00, a=0.24, b=4.60, t=2.40, k=5, eg=0.10,...</td>
      <td>0.982285</td>
      <td>13</td>
    </tr>
    <tr>
      <th>3</th>
      <td>client_3</td>
      <td>ds2</td>
      <td>0</td>
      <td>race_soft</td>
      <td>(g=0.95, a=0.18, b=3.80, t=2.20, k=3, eg=0.08,...</td>
      <td>0.940179</td>
      <td>13</td>
    </tr>
    <tr>
      <th>4</th>
      <td>client_4</td>
      <td>ds2</td>
      <td>3</td>
      <td>race_smoothmix</td>
      <td>(g=0.90, a=0.20, b=4.00, t=2.10, k=7, eg=0.07,...</td>
      <td>0.904807</td>
      <td>13</td>
    </tr>
    <tr>
      <th>5</th>
      <td>client_5</td>
      <td>ds2</td>
      <td>3</td>
      <td>race_smoothmix</td>
      <td>(g=0.90, a=0.20, b=4.00, t=2.10, k=7, eg=0.07,...</td>
      <td>0.879501</td>
      <td>13</td>
    </tr>
  </tbody>
</table>
</div>


    
    ----------------------------------------------------------------------------------------------------------------------
    Chosen final validation-based preprocessing strategy
    ----------------------------------------------------------------------------------------------------------------------



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
      <td>['race_focus']</td>
      <td>0.986716</td>
      <td>0.986726</td>
      <td>0.986712</td>
    </tr>
    <tr>
      <th>1</th>
      <td>ds2</td>
      <td>single</td>
      <td>['race_soft']</td>
      <td>0.951890</td>
      <td>0.954315</td>
      <td>0.951081</td>
    </tr>
  </tbody>
</table>
</div>


    
    ======================================================================================================================
    STEP 12.5: EXTENDED METRICS + CALIBRATION + ERROR TABLES
    ======================================================================================================================
    
    ----------------------------------------------------------------------------------------------------------------------
    Extended TEST metrics (DS1 vs DS2)
    ----------------------------------------------------------------------------------------------------------------------



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
      <td>0.977876</td>
      <td>0.977970</td>
      <td>0.978297</td>
      <td>0.977970</td>
      <td>0.977937</td>
      <td>0.978260</td>
      <td>0.977876</td>
      <td>0.977874</td>
      <td>0.068667</td>
      <td>...</td>
      <td>0.007379</td>
      <td>0.022030</td>
      <td>0.024286</td>
      <td>0.728600</td>
      <td>0.034018</td>
      <td>0.999659</td>
      <td>0.999475</td>
      <td>0.999575</td>
      <td>0.999797</td>
      <td>0.999790</td>
    </tr>
    <tr>
      <th>1</th>
      <td>ds2_test</td>
      <td>0.934010</td>
      <td>0.930563</td>
      <td>0.934035</td>
      <td>0.930563</td>
      <td>0.927734</td>
      <td>0.938255</td>
      <td>0.934010</td>
      <td>0.931896</td>
      <td>0.203729</td>
      <td>...</td>
      <td>0.021586</td>
      <td>0.069437</td>
      <td>0.035614</td>
      <td>0.601413</td>
      <td>0.104470</td>
      <td>0.994268</td>
      <td>0.999561</td>
      <td>0.986035</td>
      <td>0.999518</td>
      <td>0.991959</td>
    </tr>
  </tbody>
</table>
<p>2 rows × 26 columns</p>
</div>


    
    ----------------------------------------------------------------------------------------------------------------------
    Classwise metrics — DS1 TEST
    ----------------------------------------------------------------------------------------------------------------------



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
      <td>55</td>
      <td>2</td>
      <td>1</td>
      <td>168</td>
      <td>0.247788</td>
      <td>0.964912</td>
      <td>0.994083</td>
      <td>0.982143</td>
      <td>0.988235</td>
      <td>0.011765</td>
      <td>0.017857</td>
      <td>0.948276</td>
      <td>0.985189</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>meningioma</td>
      <td>55</td>
      <td>53</td>
      <td>0</td>
      <td>2</td>
      <td>171</td>
      <td>0.243363</td>
      <td>1.000000</td>
      <td>0.988439</td>
      <td>0.963636</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.036364</td>
      <td>0.963636</td>
      <td>0.981818</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>notumor</td>
      <td>59</td>
      <td>57</td>
      <td>1</td>
      <td>2</td>
      <td>166</td>
      <td>0.261062</td>
      <td>0.982759</td>
      <td>0.988095</td>
      <td>0.966102</td>
      <td>0.994012</td>
      <td>0.005988</td>
      <td>0.033898</td>
      <td>0.950000</td>
      <td>0.980057</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>pituitary</td>
      <td>56</td>
      <td>56</td>
      <td>2</td>
      <td>0</td>
      <td>168</td>
      <td>0.247788</td>
      <td>0.965517</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.988235</td>
      <td>0.011765</td>
      <td>0.000000</td>
      <td>0.965517</td>
      <td>0.994118</td>
    </tr>
  </tbody>
</table>
</div>


    
    ----------------------------------------------------------------------------------------------------------------------
    Classwise metrics — DS2 TEST
    ----------------------------------------------------------------------------------------------------------------------



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
      <td>45</td>
      <td>45</td>
      <td>6</td>
      <td>0</td>
      <td>146</td>
      <td>0.228426</td>
      <td>0.882353</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.960526</td>
      <td>0.039474</td>
      <td>0.000000</td>
      <td>0.882353</td>
      <td>0.980263</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>meningioma</td>
      <td>46</td>
      <td>35</td>
      <td>1</td>
      <td>11</td>
      <td>150</td>
      <td>0.233503</td>
      <td>0.972222</td>
      <td>0.931677</td>
      <td>0.760870</td>
      <td>0.993377</td>
      <td>0.006623</td>
      <td>0.239130</td>
      <td>0.744681</td>
      <td>0.877124</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>notumor</td>
      <td>61</td>
      <td>60</td>
      <td>1</td>
      <td>1</td>
      <td>135</td>
      <td>0.309645</td>
      <td>0.983607</td>
      <td>0.992647</td>
      <td>0.983607</td>
      <td>0.992647</td>
      <td>0.007353</td>
      <td>0.016393</td>
      <td>0.967742</td>
      <td>0.988127</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>pituitary</td>
      <td>45</td>
      <td>44</td>
      <td>5</td>
      <td>1</td>
      <td>147</td>
      <td>0.228426</td>
      <td>0.897959</td>
      <td>0.993243</td>
      <td>0.977778</td>
      <td>0.967105</td>
      <td>0.032895</td>
      <td>0.022222</td>
      <td>0.880000</td>
      <td>0.972442</td>
    </tr>
  </tbody>
</table>
</div>


    
    ----------------------------------------------------------------------------------------------------------------------
    Top confusion pairs — DS1 TEST
    ----------------------------------------------------------------------------------------------------------------------



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
      <td>meningioma</td>
      <td>glioma</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>notumor</td>
      <td>pituitary</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>glioma</td>
      <td>notumor</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>glioma</td>
      <td>meningioma</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>glioma</td>
      <td>pituitary</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>meningioma</td>
      <td>notumor</td>
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
      <td>meningioma</td>
      <td>pituitary</td>
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
      <td>pituitary</td>
      <td>glioma</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>


    
    ----------------------------------------------------------------------------------------------------------------------
    Top confusion pairs — DS2 TEST
    ----------------------------------------------------------------------------------------------------------------------



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
      <td>meningioma</td>
      <td>glioma</td>
      <td>6</td>
    </tr>
    <tr>
      <th>1</th>
      <td>meningioma</td>
      <td>pituitary</td>
      <td>4</td>
    </tr>
    <tr>
      <th>2</th>
      <td>meningioma</td>
      <td>notumor</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>pituitary</td>
      <td>meningioma</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>notumor</td>
      <td>pituitary</td>
      <td>1</td>
    </tr>
    <tr>
      <th>5</th>
      <td>glioma</td>
      <td>pituitary</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>glioma</td>
      <td>meningioma</td>
      <td>0</td>
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


    
    ----------------------------------------------------------------------------------------------------------------------
    Calibration bins — DS1 TEST
    ----------------------------------------------------------------------------------------------------------------------



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
      <td>0.466057</td>
      <td>0.500000</td>
      <td>0.033943</td>
      <td>2</td>
    </tr>
    <tr>
      <th>6</th>
      <td>6</td>
      <td>0.500000</td>
      <td>0.583333</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>7</td>
      <td>0.583333</td>
      <td>0.666667</td>
      <td>0.616696</td>
      <td>1.000000</td>
      <td>0.383304</td>
      <td>2</td>
    </tr>
    <tr>
      <th>8</th>
      <td>8</td>
      <td>0.666667</td>
      <td>0.750000</td>
      <td>0.728600</td>
      <td>0.000000</td>
      <td>0.728600</td>
      <td>1</td>
    </tr>
    <tr>
      <th>9</th>
      <td>9</td>
      <td>0.750000</td>
      <td>0.833333</td>
      <td>0.790621</td>
      <td>0.666667</td>
      <td>0.123954</td>
      <td>3</td>
    </tr>
    <tr>
      <th>10</th>
      <td>10</td>
      <td>0.833333</td>
      <td>0.916667</td>
      <td>0.886479</td>
      <td>0.500000</td>
      <td>0.386479</td>
      <td>2</td>
    </tr>
    <tr>
      <th>11</th>
      <td>11</td>
      <td>0.916667</td>
      <td>1.000000</td>
      <td>0.982496</td>
      <td>0.995370</td>
      <td>0.012874</td>
      <td>216</td>
    </tr>
  </tbody>
</table>
</div>


    
    ----------------------------------------------------------------------------------------------------------------------
    Calibration bins — DS2 TEST
    ----------------------------------------------------------------------------------------------------------------------



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
      <td>0.476844</td>
      <td>1.000000</td>
      <td>0.523156</td>
      <td>2</td>
    </tr>
    <tr>
      <th>6</th>
      <td>6</td>
      <td>0.500000</td>
      <td>0.583333</td>
      <td>0.544370</td>
      <td>0.333333</td>
      <td>0.211037</td>
      <td>3</td>
    </tr>
    <tr>
      <th>7</th>
      <td>7</td>
      <td>0.583333</td>
      <td>0.666667</td>
      <td>0.601413</td>
      <td>0.000000</td>
      <td>0.601413</td>
      <td>1</td>
    </tr>
    <tr>
      <th>8</th>
      <td>8</td>
      <td>0.666667</td>
      <td>0.750000</td>
      <td>0.695756</td>
      <td>0.500000</td>
      <td>0.195756</td>
      <td>2</td>
    </tr>
    <tr>
      <th>9</th>
      <td>9</td>
      <td>0.750000</td>
      <td>0.833333</td>
      <td>0.787986</td>
      <td>0.400000</td>
      <td>0.387986</td>
      <td>5</td>
    </tr>
    <tr>
      <th>10</th>
      <td>10</td>
      <td>0.833333</td>
      <td>0.916667</td>
      <td>0.881058</td>
      <td>0.750000</td>
      <td>0.131058</td>
      <td>8</td>
    </tr>
    <tr>
      <th>11</th>
      <td>11</td>
      <td>0.916667</td>
      <td>1.000000</td>
      <td>0.984973</td>
      <td>0.977273</td>
      <td>0.007700</td>
      <td>176</td>
    </tr>
  </tbody>
</table>
</div>


    
    ----------------------------------------------------------------------------------------------------------------------
    VAL + TEST tables (federated, per-dataset + global equal)
    ----------------------------------------------------------------------------------------------------------------------



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
      <td>ARCF-Net with Grouped Federated Averaging and ...</td>
      <td>VAL</td>
      <td>ds1</td>
      <td>0.986726</td>
      <td>0.986915</td>
      <td>0.986739</td>
      <td>0.986712</td>
      <td>0.987031</td>
      <td>0.986726</td>
      <td>0.986764</td>
      <td>...</td>
      <td>4.461397</td>
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
      <td>ARCF-Net with Grouped Federated Averaging and ...</td>
      <td>VAL</td>
      <td>ds2</td>
      <td>0.954315</td>
      <td>0.954541</td>
      <td>0.952303</td>
      <td>0.951081</td>
      <td>0.958337</td>
      <td>0.954315</td>
      <td>0.954159</td>
      <td>...</td>
      <td>3.881733</td>
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
      <td>ARCF-Net with Grouped Federated Averaging and ...</td>
      <td>VAL</td>
      <td>global_equal</td>
      <td>0.970520</td>
      <td>0.970728</td>
      <td>0.969521</td>
      <td>0.968897</td>
      <td>0.972684</td>
      <td>0.970520</td>
      <td>0.970462</td>
      <td>...</td>
      <td>4.171565</td>
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
      <td>ARCF-Net with Grouped Federated Averaging and ...</td>
      <td>TEST</td>
      <td>ds1</td>
      <td>0.977876</td>
      <td>0.978297</td>
      <td>0.977970</td>
      <td>0.977937</td>
      <td>0.978260</td>
      <td>0.977876</td>
      <td>0.977874</td>
      <td>...</td>
      <td>4.570381</td>
      <td>0.977970</td>
      <td>0.970622</td>
      <td>0.970495</td>
      <td>0.978297</td>
      <td>0.992654</td>
      <td>0.992621</td>
      <td>0.024286</td>
      <td>0.728600</td>
      <td>0.034018</td>
    </tr>
    <tr>
      <th>4</th>
      <td>ARCF-Net with Grouped Federated Averaging and ...</td>
      <td>TEST</td>
      <td>ds2</td>
      <td>0.934010</td>
      <td>0.934035</td>
      <td>0.930563</td>
      <td>0.927734</td>
      <td>0.938255</td>
      <td>0.934010</td>
      <td>0.931896</td>
      <td>...</td>
      <td>4.046084</td>
      <td>0.930563</td>
      <td>0.913885</td>
      <td>0.911482</td>
      <td>0.934035</td>
      <td>0.979392</td>
      <td>0.978414</td>
      <td>0.035614</td>
      <td>0.601413</td>
      <td>0.104470</td>
    </tr>
    <tr>
      <th>5</th>
      <td>ARCF-Net with Grouped Federated Averaging and ...</td>
      <td>TEST</td>
      <td>global_equal</td>
      <td>0.955943</td>
      <td>0.956166</td>
      <td>0.954267</td>
      <td>0.952835</td>
      <td>0.958258</td>
      <td>0.955943</td>
      <td>0.954885</td>
      <td>...</td>
      <td>4.308233</td>
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


    
    Paper selection summary:
    - Best round: 13 | best_reward=1.0997
    - DS1 final strategy: single | names=['race_focus']
    - DS2 final strategy: single | names=['race_soft']
    
    ======================================================================================================================
    STEP 13: PREPROCESSING VALIDATION (DS1 + DS2 VAL SAMPLE)
    ======================================================================================================================
    
    ----------------------------------------------------------------------------------------------------------------------
    Preprocessing validation summary (DS1 VAL sample)
    ----------------------------------------------------------------------------------------------------------------------



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
      <td>0.041596</td>
      <td>0.021720</td>
      <td>0.014947</td>
      <td>0.165471</td>
    </tr>
    <tr>
      <th>1</th>
      <td>edge_energy_after</td>
      <td>0.164575</td>
      <td>0.033616</td>
      <td>0.095645</td>
      <td>0.288118</td>
    </tr>
    <tr>
      <th>2</th>
      <td>entropy_before</td>
      <td>5.820408</td>
      <td>0.628255</td>
      <td>3.547314</td>
      <td>7.239670</td>
    </tr>
    <tr>
      <th>3</th>
      <td>entropy_after</td>
      <td>6.532319</td>
      <td>0.588674</td>
      <td>3.862098</td>
      <td>7.515989</td>
    </tr>
    <tr>
      <th>4</th>
      <td>contrast_before</td>
      <td>0.187640</td>
      <td>0.052597</td>
      <td>0.101468</td>
      <td>0.363759</td>
    </tr>
    <tr>
      <th>5</th>
      <td>contrast_after</td>
      <td>0.228820</td>
      <td>0.021807</td>
      <td>0.198223</td>
      <td>0.328528</td>
    </tr>
    <tr>
      <th>6</th>
      <td>edge_gain_ratio</td>
      <td>4.366270</td>
      <td>1.079111</td>
      <td>1.503590</td>
      <td>8.154921</td>
    </tr>
    <tr>
      <th>7</th>
      <td>entropy_delta</td>
      <td>0.711911</td>
      <td>0.258967</td>
      <td>0.243141</td>
      <td>1.510475</td>
    </tr>
    <tr>
      <th>8</th>
      <td>contrast_delta</td>
      <td>0.041180</td>
      <td>0.034104</td>
      <td>-0.037996</td>
      <td>0.108149</td>
    </tr>
  </tbody>
</table>
</div>


    
    ----------------------------------------------------------------------------------------------------------------------
    Preprocessing validation summary (DS2 VAL sample)
    ----------------------------------------------------------------------------------------------------------------------



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
      <td>0.041718</td>
      <td>0.015947</td>
      <td>0.012571</td>
      <td>0.121283</td>
    </tr>
    <tr>
      <th>1</th>
      <td>edge_energy_after</td>
      <td>0.119036</td>
      <td>0.024377</td>
      <td>0.045461</td>
      <td>0.172808</td>
    </tr>
    <tr>
      <th>2</th>
      <td>entropy_before</td>
      <td>5.811255</td>
      <td>0.822915</td>
      <td>3.222212</td>
      <td>7.570240</td>
    </tr>
    <tr>
      <th>3</th>
      <td>entropy_after</td>
      <td>6.585727</td>
      <td>0.820581</td>
      <td>3.547999</td>
      <td>7.732978</td>
    </tr>
    <tr>
      <th>4</th>
      <td>contrast_before</td>
      <td>0.190753</td>
      <td>0.055869</td>
      <td>0.098556</td>
      <td>0.341468</td>
    </tr>
    <tr>
      <th>5</th>
      <td>contrast_after</td>
      <td>0.255969</td>
      <td>0.023935</td>
      <td>0.199012</td>
      <td>0.326174</td>
    </tr>
    <tr>
      <th>6</th>
      <td>edge_gain_ratio</td>
      <td>3.080745</td>
      <td>0.786803</td>
      <td>1.382105</td>
      <td>5.615081</td>
    </tr>
    <tr>
      <th>7</th>
      <td>entropy_delta</td>
      <td>0.774471</td>
      <td>0.284244</td>
      <td>0.162738</td>
      <td>1.456336</td>
    </tr>
    <tr>
      <th>8</th>
      <td>contrast_delta</td>
      <td>0.065215</td>
      <td>0.036171</td>
      <td>-0.016532</td>
      <td>0.135223</td>
    </tr>
  </tbody>
</table>
</div>


    
    ======================================================================================================================
    STEP 14: PARAMETER EVOLUTION TABLE
    ======================================================================================================================
    
    ----------------------------------------------------------------------------------------------------------------------
    Mean selected preprocessing parameters over rounds
    ----------------------------------------------------------------------------------------------------------------------



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
      <td>0.986667</td>
      <td>0.273333</td>
      <td>5.133333</td>
      <td>2.383333</td>
      <td>5.000000</td>
      <td>0.116667</td>
      <td>0.776667</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1.035000</td>
      <td>0.233333</td>
      <td>4.650000</td>
      <td>2.600000</td>
      <td>4.333333</td>
      <td>0.110000</td>
      <td>0.770000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1.026667</td>
      <td>0.260000</td>
      <td>4.933333</td>
      <td>2.633333</td>
      <td>5.666667</td>
      <td>0.106667</td>
      <td>0.773333</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>0.993333</td>
      <td>0.243333</td>
      <td>4.583333</td>
      <td>2.400000</td>
      <td>5.666667</td>
      <td>0.098333</td>
      <td>0.763333</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>1.006667</td>
      <td>0.250000</td>
      <td>4.733333</td>
      <td>2.400000</td>
      <td>4.000000</td>
      <td>0.108333</td>
      <td>0.780000</td>
    </tr>
    <tr>
      <th>5</th>
      <td>6</td>
      <td>0.971667</td>
      <td>0.273333</td>
      <td>4.966667</td>
      <td>2.350000</td>
      <td>5.333333</td>
      <td>0.100000</td>
      <td>0.783333</td>
    </tr>
    <tr>
      <th>6</th>
      <td>7</td>
      <td>0.973333</td>
      <td>0.263333</td>
      <td>4.966667</td>
      <td>2.350000</td>
      <td>4.333333</td>
      <td>0.108333</td>
      <td>0.773333</td>
    </tr>
    <tr>
      <th>7</th>
      <td>8</td>
      <td>1.033333</td>
      <td>0.243333</td>
      <td>4.666667</td>
      <td>2.550000</td>
      <td>5.000000</td>
      <td>0.106667</td>
      <td>0.780000</td>
    </tr>
    <tr>
      <th>8</th>
      <td>9</td>
      <td>1.000000</td>
      <td>0.230000</td>
      <td>4.550000</td>
      <td>2.433333</td>
      <td>4.000000</td>
      <td>0.103333</td>
      <td>0.766667</td>
    </tr>
    <tr>
      <th>9</th>
      <td>10</td>
      <td>0.980000</td>
      <td>0.263333</td>
      <td>4.966667</td>
      <td>2.383333</td>
      <td>5.000000</td>
      <td>0.110000</td>
      <td>0.780000</td>
    </tr>
    <tr>
      <th>10</th>
      <td>11</td>
      <td>1.031667</td>
      <td>0.220000</td>
      <td>4.416667</td>
      <td>2.566667</td>
      <td>4.666667</td>
      <td>0.101667</td>
      <td>0.760000</td>
    </tr>
    <tr>
      <th>11</th>
      <td>12</td>
      <td>1.025000</td>
      <td>0.283333</td>
      <td>5.250000</td>
      <td>2.516667</td>
      <td>4.333333</td>
      <td>0.121667</td>
      <td>0.796667</td>
    </tr>
    <tr>
      <th>12</th>
      <td>13</td>
      <td>0.983333</td>
      <td>0.246667</td>
      <td>4.700000</td>
      <td>2.350000</td>
      <td>5.333333</td>
      <td>0.101667</td>
      <td>0.766667</td>
    </tr>
    <tr>
      <th>13</th>
      <td>14</td>
      <td>1.035000</td>
      <td>0.233333</td>
      <td>4.650000</td>
      <td>2.600000</td>
      <td>4.333333</td>
      <td>0.110000</td>
      <td>0.770000</td>
    </tr>
    <tr>
      <th>14</th>
      <td>15</td>
      <td>0.996667</td>
      <td>0.266667</td>
      <td>4.883333</td>
      <td>2.433333</td>
      <td>5.666667</td>
      <td>0.103333</td>
      <td>0.780000</td>
    </tr>
  </tbody>
</table>
</div>


    
    ======================================================================================================================
    STEP 15: SAVING ONLY TWO FILES (CHECKPOINT + ONE CSV)
    ======================================================================================================================
    ✅ Saved checkpoint: /kaggle/working/outputs/ARCFNet_RESNET50_GROUPED_PRIVATE_HEADS_checkpoint.pth
    ✅ Saved CSV (ALL outputs): /kaggle/working/outputs/ALL_OUTPUTS_AND_METRICS_RESNET50_GROUPED_PRIVATE_HEADS.csv
    
    DONE ✅
    Method: ARCF-Net = Adaptive RACE-FELCM with CRAF Fusion Network
    Setting: ARCF-Net with Grouped Federated Averaging and Dataset-Private Heads
    Backbone: Residual Network-50
    Best round: 13
    Adaptive clients => DS1=3, DS2=3, TOTAL=6
    Rounds completed: 15
    Global TEST acc: 0.9559
    Global TEST f1_macro: 0.9528
    DS1 TEST acc: 0.9779
    DS2 TEST acc: 0.9340
    DS1 final strategy: single | names=['race_focus']
    DS2 final strategy: single | names=['race_soft']



```python

```
