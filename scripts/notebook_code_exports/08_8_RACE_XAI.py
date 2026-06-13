# Auto-exported from: 8_RACE_XAI.ipynb
# Public Python export for GitHub visibility.
# Notebook outputs are available in reports/notebook_exports/.

# %% Cell 1
# ============================================================
# FULL SINGLE-CELL CODE
# CHECKPOINT-FREE
# RACE-FELCM PREPROCESSING-XAI ONLY
# ------------------------------------------------------------
# Uses ONLY the two datasets and your RACE-FELCM preprocessing.
# No checkpoint needed.
# No training.
# No metrics.
# Only plot generation.
#
# Preprocessing name used in all figures:
#   RACE-FELCM
#   = Robust Adaptive Context-Enhanced Fuzzy Edge Local
#     Contrast Mapping
#
# Figure families generated:
#   1) Overview Panel
#   2) Texture / Denoising Panel
#   3) Local Contrast Panel
#   4) Structure Enhancement Panel
#   5) Intensity Remapping Panel
#   6) Summary XAI Panel
# ============================================================

# ============================================================
# OPTIONAL: MOUNT GOOGLE DRIVE
# ============================================================
# from google.colab import drive
# drive.mount("/content/drive")

# ============================================================
# IMPORTS
# ============================================================
import os
import random
import warnings
import subprocess
import sys

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
import matplotlib as mpl

warnings.filterwarnings("ignore")

# Only install kagglehub if missing
try:
    import kagglehub
except Exception:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "kagglehub"])
    import kagglehub

# ============================================================
# USER CONFIG
# ============================================================
SEED = 42
IMG_SIZE = 224
OUTPUT_DIR = "/content/RACE_FELCM_PREPROCESSING_XAI_OUTPUTS"

# Number of samples chosen PER CLASS
# 1 => 4 rows per dataset
SAMPLES_PER_CLASS = 1

SAVE_FIGURES = True
SHOW_FIGURES = True

# Try local paths first, then KaggleHub
DATASET_SEARCH_ROOTS = ["/content", "/content/drive/MyDrive"]

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# REPRODUCIBILITY + DEVICE
# ============================================================
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

plt.style.use("seaborn-v0_8-white")
mpl.rcParams.update({
    "figure.dpi": 145,
    "savefig.dpi": 220,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#D0D7DE",
    "axes.linewidth": 0.9,
    "axes.titleweight": "bold",
    "axes.titlesize": 15,
    "axes.labelsize": 11,
    "font.size": 10.5,
    "legend.frameon": True,
    "legend.facecolor": "white",
    "legend.edgecolor": "#D0D7DE",
    "grid.alpha": 0.18,
    "grid.linewidth": 0.8,
    "lines.linewidth": 2.0,
})

print("=" * 118)
print("RACE-FELCM PREPROCESSING-XAI PLOTS ONLY")
print(f"DEVICE: {DEVICE}")
print("=" * 118)

# ============================================================
# LABELS / HELPERS
# ============================================================
labels = ["glioma", "meningioma", "notumor", "pituitary"]
label2id = {l: i for i, l in enumerate(labels)}
id2label = {i: l for l, i in label2id.items()}
NUM_CLASSES = len(labels)
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")

def theta_str(theta):
    g, a, b, t, k, eg, m = theta
    return f"(g={g:.2f}, a={a:.2f}, b={b:.2f}, t={t:.2f}, k={int(k)}, eg={eg:.2f}, mix={m:.2f})"

def load_rgb(path):
    try:
        return Image.open(path).convert("RGB")
    except Exception:
        return Image.new("RGB", (IMG_SIZE, IMG_SIZE), (128, 128, 128))

def pil_to_tensor(img, size=IMG_SIZE):
    img = img.resize((size, size), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    arr = np.transpose(arr, (2, 0, 1))
    return torch.from_numpy(arr)

def clean_ax(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(False)

def title_box(ax, text, loc="tl"):
    if loc == "tl":
        x, y, va, ha = 0.02, 0.98, "top", "left"
    elif loc == "bl":
        x, y, va, ha = 0.02, 0.02, "bottom", "left"
    else:
        x, y, va, ha = 0.98, 0.98, "top", "right"

    ax.text(
        x, y, text,
        transform=ax.transAxes,
        ha=ha, va=va,
        fontsize=9.0,
        color="white",
        bbox=dict(
            boxstyle="round,pad=0.22",
            facecolor=(0, 0, 0, 0.56),
            edgecolor=(1, 1, 1, 0.08),
        ),
    )

def row_label(ax, txt):
    ax.text(
        -0.06, 0.50, txt,
        transform=ax.transAxes,
        ha="right", va="center",
        fontsize=12,
        fontweight="bold",
        color="#111827",
    )

def normalize_map(arr):
    arr = np.asarray(arr, dtype=np.float32)
    arr = arr - arr.min()
    arr = arr / (arr.max() + 1e-8)
    return arr

def signed_map(arr):
    arr = np.asarray(arr, dtype=np.float32)
    lim = np.max(np.abs(arr)) + 1e-8
    return np.clip(arr / lim, -1.0, 1.0)

def tensor_to_gray01(x):
    # x shape [1,C,H,W] or [1,1,H,W]
    arr = x[0].detach().cpu().float().numpy()
    if arr.ndim == 3:
        arr = arr.mean(axis=0)
    arr = arr - arr.min()
    arr = arr / (arr.max() + 1e-8)
    return arr

def tensor_to_gray_signed(x):
    arr = x[0].detach().cpu().float().numpy()
    if arr.ndim == 3:
        arr = arr.mean(axis=0)
    return signed_map(arr)

def overlay_heatmap(gray01, heat01, cmap="jet", alpha=0.48):
    gray_rgb = np.stack([gray01, gray01, gray01], axis=-1)
    cm = plt.get_cmap(cmap)(np.clip(heat01, 0, 1))[..., :3]
    out = (1 - alpha) * gray_rgb + alpha * cm
    return np.clip(out, 0, 1)

# ============================================================
# DATASET DISCOVERY
# ============================================================
REQ1 = {"512Glioma", "512Meningioma", "512Normal", "512Pituitary"}
REQ2 = {"glioma", "meningioma", "notumor", "pituitary"}

def norm_label(name):
    s = str(name).strip().lower()
    if "glioma" in s:
        return "glioma"
    if "meningioma" in s:
        return "meningioma"
    if "pituitary" in s:
        return "pituitary"
    if "normal" in s or "notumor" in s or "no_tumor" in s or "no tumor" in s:
        return "notumor"
    return None

def find_root_with_required_class_dirs(base_dir, required_set, prefer_raw=True):
    candidates = []
    for root, dirs, _ in os.walk(base_dir):
        if required_set.issubset(set(dirs)):
            candidates.append(root)
    if not candidates:
        return None

    def score(path):
        pl = path.lower()
        sc = 0
        if prefer_raw:
            if "raw data" in pl:
                sc += 7
            if os.path.basename(path).lower() == "raw":
                sc += 7
            if "/raw/" in pl or "\\raw\\" in pl:
                sc += 3
            if "augmented" in pl:
                sc -= 20
        sc -= 0.0001 * len(path)
        return sc

    return max(candidates, key=score)

def try_kagglehub_download(slug, required_set, prefer_raw=True):
    try:
        base = kagglehub.dataset_download(slug)
        root = find_root_with_required_class_dirs(base, required_set, prefer_raw=prefer_raw)
        return base, root
    except Exception:
        return None, None

def find_dataset_roots():
    ds1_root = None
    ds2_root = None

    for base in DATASET_SEARCH_ROOTS:
        if os.path.exists(base):
            if ds1_root is None:
                ds1_root = find_root_with_required_class_dirs(base, REQ1, prefer_raw=True)
            if ds2_root is None:
                ds2_root = find_root_with_required_class_dirs(base, REQ2, prefer_raw=False)

    if ds1_root is None:
        _, ds1_root = try_kagglehub_download(
            "orvile/pmram-bangladeshi-brain-cancer-mri-dataset",
            REQ1,
            prefer_raw=True,
        )
    if ds2_root is None:
        _, ds2_root = try_kagglehub_download(
            "yassinebazgour/preprocessed-brain-mri-scans-for-tumors-detection",
            REQ2,
            prefer_raw=False,
        )

    if ds1_root is None:
        raise RuntimeError("Could not find DS1.")
    if ds2_root is None:
        raise RuntimeError("Could not find DS2.")

    return ds1_root, ds2_root

# ============================================================
# RECORD BUILDING
# ============================================================
def list_images_under_class_root(class_root, class_dir_name):
    class_dir = os.path.join(class_root, class_dir_name)
    out = []
    for r, _, files in os.walk(class_dir):
        for fn in files:
            if fn.lower().endswith(IMG_EXTS):
                out.append(os.path.join(r, fn))
    return out

def build_records_from_root(ds_root, class_dirs, source_name):
    records = []
    for c in class_dirs:
        lab = norm_label(c)
        imgs = list_images_under_class_root(ds_root, c)
        for p in imgs:
            records.append({
                "path": str(p),
                "label": str(lab),
                "source": str(source_name),
                "filename": os.path.basename(p),
                "y": int(label2id[lab]),
            })

    seen = set()
    uniq = []
    for r in records:
        if r["path"] not in seen:
            seen.add(r["path"])
            uniq.append(r)
    return uniq

def pick_samples_per_class(records, n_per_class=1, seed=SEED):
    rng = random.Random(seed)
    out = []
    for lab in labels:
        sub = [r for r in records if r["label"] == lab]
        if len(sub) == 0:
            continue
        k = min(n_per_class, len(sub))
        out.extend(rng.sample(sub, k))
    return out

# ============================================================
# RACE-FELCM PREPROCESSING WITH INTERMEDIATE MAPS
# ============================================================
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

    def forward(self, x, return_dict=False):
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
        z_clip = ((x_dn - mu) / sd).clamp(-self.tau, self.tau)
        z = torch.sign(z_clip) * torch.pow(z_clip.abs().clamp_min(eps), self.gamma)

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

        structure_gain = self.edge_gain * edge * coherence
        enhanced_pre_norm = z + self.alpha * contrast_field + structure_gain

        mn = enhanced_pre_norm.amin(dim=(2, 3), keepdim=True)
        mx = enhanced_pre_norm.amax(dim=(2, 3), keepdim=True)
        enhanced_norm = (enhanced_pre_norm - mn) / (mx - mn + eps)

        out = self.blend * enhanced_norm + (1.0 - self.blend) * x
        out = out.clamp(0, 1)

        abs_residual = torch.abs(out - x)
        signed_delta = out - x

        composite = normalize_map(
            0.45 * tensor_to_gray01(contrast_field) +
            0.35 * tensor_to_gray01(edge) +
            0.20 * tensor_to_gray01(coherence)
        )

        bundle = {
            "raw": x,
            "raw_gray": gray0,
            "mu0": mu0,
            "var0": var0,
            "tex": tex,
            "tex_norm": tex_norm,
            "flat_gate": flat_gate,
            "x_smooth": x_smooth,
            "x_dn": x_dn,
            "z_clip": z_clip,
            "z": z,
            "gray_z": gray,
            "local_mean": local_mean,
            "local_std": local_std,
            "local_norm": local_norm,
            "contrast_field": contrast_field,
            "gx": gx,
            "gy": gy,
            "edge": edge,
            "lap": lap,
            "coherence": coherence,
            "structure_gain": structure_gain,
            "enhanced_pre_norm": enhanced_pre_norm,
            "enhanced_norm": enhanced_norm,
            "out": out,
            "abs_residual": abs_residual,
            "signed_delta": signed_delta,
            "composite": composite,
        }

        if return_dict:
            return bundle
        return out

# ============================================================
# PLOT PREP HELPERS
# ============================================================
def prepare_sample_bundle(preproc, row):
    img = load_rgb(row["path"])
    x = pil_to_tensor(img, IMG_SIZE).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        B = preproc(x, return_dict=True)

    raw_gray = tensor_to_gray01(B["raw"])
    out_gray = tensor_to_gray01(B["out"])

    structure_overlay = overlay_heatmap(raw_gray, tensor_to_gray01(B["structure_gain"]), cmap="jet", alpha=0.50)
    contrast_overlay = overlay_heatmap(raw_gray, normalize_map(np.abs(tensor_to_gray_signed(B["contrast_field"]))), cmap="jet", alpha=0.50)
    edge_overlay = overlay_heatmap(raw_gray, tensor_to_gray01(B["edge"]), cmap="jet", alpha=0.50)
    composite_overlay = overlay_heatmap(raw_gray, B["composite"], cmap="jet", alpha=0.52)

    payload = {
        "gt": row["label"],
        "theta_str": theta_str(THETA),
        "raw_gray": raw_gray,
        "out_gray": out_gray,
        "abs_residual": tensor_to_gray01(B["abs_residual"]),
        "signed_delta": tensor_to_gray_signed(B["signed_delta"]),
        "tex_norm": tensor_to_gray01(B["tex_norm"]),
        "flat_gate": tensor_to_gray01(B["flat_gate"]),
        "x_smooth": tensor_to_gray01(B["x_smooth"]),
        "x_dn": tensor_to_gray01(B["x_dn"]),
        "local_mean": tensor_to_gray01(B["local_mean"]),
        "local_std": tensor_to_gray01(B["local_std"]),
        "local_norm": tensor_to_gray_signed(B["local_norm"]),
        "contrast_field": tensor_to_gray_signed(B["contrast_field"]),
        "edge": tensor_to_gray01(B["edge"]),
        "lap": tensor_to_gray01(B["lap"]),
        "coherence": tensor_to_gray01(B["coherence"]),
        "structure_gain": tensor_to_gray01(B["structure_gain"]),
        "z_clip": tensor_to_gray_signed(B["z_clip"]),
        "z": tensor_to_gray_signed(B["z"]),
        "enhanced_pre_norm": tensor_to_gray_signed(B["enhanced_pre_norm"]),
        "enhanced_norm": tensor_to_gray01(B["enhanced_norm"]),
        "composite": B["composite"],
        "structure_overlay": structure_overlay,
        "contrast_overlay": contrast_overlay,
        "edge_overlay": edge_overlay,
        "composite_overlay": composite_overlay,
    }
    return payload

def render_cell(ax, payload, key, mode):
    if mode == "gray":
        ax.imshow(payload[key], cmap="gray", vmin=0, vmax=1)
    elif mode == "heat":
        ax.imshow(payload[key], cmap="jet", vmin=0, vmax=1)
    elif mode == "div":
        ax.imshow(payload[key], cmap="seismic", vmin=-1, vmax=1)
    elif mode == "rgb":
        ax.imshow(payload[key], vmin=0, vmax=1)
    clean_ax(ax)

def plot_family(dataset_title, family_title, rows_payload, columns, save_path=None):
    n = len(rows_payload)
    fig, axes = plt.subplots(
        n, len(columns),
        figsize=(3.4 * len(columns), 3.9 * n),
        constrained_layout=True,
        facecolor="white",
    )
    if n == 1:
        axes = np.array(axes).reshape(1, len(columns))

    for j, (col_title, _, _) in enumerate(columns):
        axes[0, j].set_title(col_title, fontsize=14, fontweight="bold", pad=10)

    for i, payload in enumerate(rows_payload):
        for j, (_, key, mode) in enumerate(columns):
            render_cell(axes[i, j], payload, key, mode)

        row_label(axes[i, 0], payload["gt"])
        title_box(axes[i, 0], f"GT={payload['gt']}", loc="bl")
        title_box(axes[i, 1], payload["theta_str"], loc="bl")

    fig.suptitle(
        f"{dataset_title} — {family_title} (RACE-FELCM)",
        fontsize=20,
        fontweight="bold",
        color="#111827",
    )
    fig.text(
        0.5, 0.005,
        "RACE-FELCM = Robust Adaptive Context-Enhanced Fuzzy Edge Local Contrast Mapping.",
        ha="center", va="bottom", fontsize=10.5, color="#374151"
    )

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight", facecolor="white")
        print(f"Saved: {save_path}")

    if SHOW_FIGURES:
        plt.show()
    else:
        plt.close(fig)

# ============================================================
# MAIN
# ============================================================
# Use your exact DS1 representative theta from your canonical script
THETA = (1.00, 0.24, 4.6, 2.4, 5, 0.10, 0.78)

preproc = RACEFELCM(*THETA).to(DEVICE).eval()

ds1_root, ds2_root = find_dataset_roots()
print(f"DS1 root: {ds1_root}")
print(f"DS2 root: {ds2_root}")

records_ds1 = build_records_from_root(
    ds1_root,
    ["512Glioma", "512Meningioma", "512Normal", "512Pituitary"],
    "ds1_raw",
)
records_ds2 = build_records_from_root(
    ds2_root,
    ["glioma", "meningioma", "notumor", "pituitary"],
    "ds2",
)

samples_ds1 = pick_samples_per_class(records_ds1, n_per_class=SAMPLES_PER_CLASS, seed=SEED)
samples_ds2 = pick_samples_per_class(records_ds2, n_per_class=SAMPLES_PER_CLASS, seed=SEED)

print(f"DS1 selected samples: {len(samples_ds1)}")
print(f"DS2 selected samples: {len(samples_ds2)}")

rows_payload_ds1 = [prepare_sample_bundle(preproc, row) for row in samples_ds1]
rows_payload_ds2 = [prepare_sample_bundle(preproc, row) for row in samples_ds2]

families = [
    (
        "Overview Panel",
        [
            ("Raw", "raw_gray", "gray"),
            ("RACE-FELCM", "out_gray", "gray"),
            ("|Delta|", "abs_residual", "heat"),
            ("Signed Delta", "signed_delta", "div"),
            ("Composite XAI", "composite_overlay", "rgb"),
        ],
        "overview",
    ),
    (
        "Texture / Denoising Panel",
        [
            ("Raw Gray", "raw_gray", "gray"),
            ("Texture Norm", "tex_norm", "heat"),
            ("Flat Gate", "flat_gate", "heat"),
            ("Smoothed", "x_smooth", "gray"),
            ("Denoised Branch", "x_dn", "gray"),
        ],
        "texture_denoising",
    ),
    (
        "Local Contrast Panel",
        [
            ("Denoised Gray", "x_dn", "gray"),
            ("Local Mean", "local_mean", "gray"),
            ("Local Std", "local_std", "heat"),
            ("Local Norm", "local_norm", "div"),
            ("Contrast Field", "contrast_field", "div"),
        ],
        "local_contrast",
    ),
    (
        "Structure Enhancement Panel",
        [
            ("Edge", "edge", "heat"),
            ("Laplacian", "lap", "heat"),
            ("Coherence", "coherence", "heat"),
            ("Structure Gain", "structure_gain", "heat"),
            ("Structure Overlay", "structure_overlay", "rgb"),
        ],
        "structure",
    ),
    (
        "Intensity Remapping Panel",
        [
            ("Z-Clip", "z_clip", "div"),
            ("Power Response", "z", "div"),
            ("Enhanced Pre-Norm", "enhanced_pre_norm", "div"),
            ("Enhanced Norm", "enhanced_norm", "gray"),
            ("Final Blend", "out_gray", "gray"),
        ],
        "intensity_remap",
    ),
    (
        "Summary XAI Panel",
        [
            ("Raw", "raw_gray", "gray"),
            ("RACE-FELCM", "out_gray", "gray"),
            ("Contrast Overlay", "contrast_overlay", "rgb"),
            ("Edge Overlay", "edge_overlay", "rgb"),
            ("Composite Overlay", "composite_overlay", "rgb"),
        ],
        "summary_xai",
    ),
]

for family_title, cols, slug in families:
    save1 = os.path.join(OUTPUT_DIR, f"DS1_{slug}_RACE_FELCM.png") if SAVE_FIGURES else None
    plot_family("Dataset-1", family_title, rows_payload_ds1, cols, save1)

    save2 = os.path.join(OUTPUT_DIR, f"DS2_{slug}_RACE_FELCM.png") if SAVE_FIGURES else None
    plot_family("Dataset-2", family_title, rows_payload_ds2, cols, save2)

# ============================================================
# FINAL COUNT
# ============================================================
family_count = len(families)
figure_file_count = 2 * family_count
rows_total = len(rows_payload_ds1) + len(rows_payload_ds2)
subplot_total = figure_file_count * 5 * len(rows_payload_ds1)  # same row count per dataset by default

print("\n" + "=" * 118)
print("DONE: RACE-FELCM PREPROCESSING-XAI FIGURES GENERATED")
print("=" * 118)
print(f"Figure families            : {family_count}")
print(f"Datasets                   : 2")
print(f"Saved figure files total   : {figure_file_count}")
print(f"Rows per dataset           : DS1={len(rows_payload_ds1)}, DS2={len(rows_payload_ds2)}")
print(f"Columns per figure         : 5")
print(f"Total subplot images       : {figure_file_count * 5 * len(rows_payload_ds1)}")
print(f"Output directory           : {OUTPUT_DIR}")
print("=" * 118)

# %% Cell 2
# ============================================================
# FULL SINGLE-CELL CODE
# CHECKPOINT-FREE
# RACE-FELCM PREPROCESSING-XAI ONLY
# ------------------------------------------------------------
# Uses ONLY the two datasets and your RACE-FELCM preprocessing.
# No checkpoint needed.
# No training.
# No metrics.
# Only plot generation.
#
# THETA VALUES TAKEN FROM YOUR SCREENSHOTS:
#   DS1 = (g=1.02, a=0.32, b=6.00, t=2.60, k=3, eg=0.15, mix=0.84)
#   DS2 = (g=0.95, a=0.18, b=3.80, t=2.20, k=3, eg=0.08, mix=0.72)
#
# IMPORTANT FIX:
#   - No caption/footer text is written inside the image anymore.
#   - Captions are printed below each figure as normal notebook text.
# ============================================================

# ============================================================
# OPTIONAL: MOUNT GOOGLE DRIVE
# ============================================================
# from google.colab import drive
# drive.mount("/content/drive")

# ============================================================
# IMPORTS
# ============================================================
import os
import random
import warnings
import subprocess
import sys

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
import matplotlib as mpl

warnings.filterwarnings("ignore")

# Only install kagglehub if missing
try:
    import kagglehub
except Exception:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "kagglehub"])
    import kagglehub

# ============================================================
# USER CONFIG
# ============================================================
SEED = 42
IMG_SIZE = 224

if os.path.exists("/kaggle/working"):
    OUTPUT_DIR = "/kaggle/working/RACE_FELCM_PREPROCESSING_XAI_OUTPUTS"
else:
    OUTPUT_DIR = "/content/RACE_FELCM_PREPROCESSING_XAI_OUTPUTS"

# Number of samples per class
# 1 => 4 rows per dataset
SAMPLES_PER_CLASS = 1

SAVE_FIGURES = True
SHOW_FIGURES = True

DATASET_SEARCH_ROOTS = ["/content", "/content/drive/MyDrive", "/kaggle/input"]

# THETA VALUES FROM YOUR SCREENSHOTS
THETA_DS1 = (1.02, 0.32, 6.00, 2.60, 3, 0.15, 0.84)
THETA_DS2 = (0.95, 0.18, 3.80, 2.20, 3, 0.08, 0.72)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# REPRODUCIBILITY + DEVICE
# ============================================================
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

plt.style.use("seaborn-v0_8-white")
mpl.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 240,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#D0D7DE",
    "axes.linewidth": 1.0,
    "axes.titleweight": "bold",
    "axes.titlesize": 18,
    "axes.labelsize": 14,
    "font.size": 13,
    "font.weight": "bold",
    "legend.frameon": True,
    "legend.facecolor": "white",
    "legend.edgecolor": "#D0D7DE",
    "grid.alpha": 0.18,
    "grid.linewidth": 0.8,
    "lines.linewidth": 2.0,
})

print("=" * 118)
print("RACE-FELCM PREPROCESSING-XAI PLOTS ONLY")
print(f"DEVICE: {DEVICE}")
print("=" * 118)

# ============================================================
# LABELS / HELPERS
# ============================================================
labels = ["glioma", "meningioma", "notumor", "pituitary"]
label2id = {l: i for i, l in enumerate(labels)}
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")

def theta_str(theta):
    g, a, b, t, k, eg, m = theta
    return f"(g={g:.2f}, a={a:.2f}, b={b:.2f}, t={t:.2f}, k={int(k)}, eg={eg:.2f}, mix={m:.2f})"

def theta_multiline(theta, tag):
    g, a, b, t, k, eg, m = theta
    return (
        f"{tag}\n\n"
        f"g = {g:.2f}\n"
        f"a = {a:.2f}\n"
        f"b = {b:.2f}\n"
        f"t = {t:.2f}\n"
        f"k = {int(k)}\n"
        f"eg = {eg:.2f}\n"
        f"mix = {m:.2f}"
    )

def load_rgb(path):
    try:
        return Image.open(path).convert("RGB")
    except Exception:
        return Image.new("RGB", (IMG_SIZE, IMG_SIZE), (128, 128, 128))

def pil_to_tensor(img, size=IMG_SIZE):
    img = img.resize((size, size), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    arr = np.transpose(arr, (2, 0, 1))
    return torch.from_numpy(arr)

def clean_ax(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(False)

def title_box(ax, text, loc="tl", fontsize=12):
    if loc == "tl":
        x, y, va, ha = 0.02, 0.98, "top", "left"
    elif loc == "bl":
        x, y, va, ha = 0.02, 0.02, "bottom", "left"
    else:
        x, y, va, ha = 0.98, 0.98, "top", "right"

    ax.text(
        x, y, text,
        transform=ax.transAxes,
        ha=ha, va=va,
        fontsize=fontsize,
        fontweight="bold",
        color="white",
        bbox=dict(
            boxstyle="round,pad=0.26",
            facecolor=(0, 0, 0, 0.62),
            edgecolor=(1, 1, 1, 0.10),
        ),
    )

def row_label(ax, txt):
    ax.text(
        -0.07, 0.50, txt,
        transform=ax.transAxes,
        ha="right", va="center",
        fontsize=15,
        fontweight="bold",
        color="#111827",
    )

def normalize_map(arr):
    arr = np.asarray(arr, dtype=np.float32)
    arr = arr - arr.min()
    arr = arr / (arr.max() + 1e-8)
    return arr

def signed_map(arr):
    arr = np.asarray(arr, dtype=np.float32)
    lim = np.max(np.abs(arr)) + 1e-8
    return np.clip(arr / lim, -1.0, 1.0)

def tensor_to_gray01(x):
    arr = x[0].detach().cpu().float().numpy()
    if arr.ndim == 3:
        arr = arr.mean(axis=0)
    arr = arr - arr.min()
    arr = arr / (arr.max() + 1e-8)
    return arr

def tensor_to_gray_signed(x):
    arr = x[0].detach().cpu().float().numpy()
    if arr.ndim == 3:
        arr = arr.mean(axis=0)
    return signed_map(arr)

def overlay_heatmap(gray01, heat01, cmap="jet", alpha=0.50):
    gray_rgb = np.stack([gray01, gray01, gray01], axis=-1)
    cm = plt.get_cmap(cmap)(np.clip(heat01, 0, 1))[..., :3]
    out = (1 - alpha) * gray_rgb + alpha * cm
    return np.clip(out, 0, 1)

# ============================================================
# DATASET DISCOVERY
# ============================================================
REQ1 = {"512Glioma", "512Meningioma", "512Normal", "512Pituitary"}
REQ2 = {"glioma", "meningioma", "notumor", "pituitary"}

def norm_label(name):
    s = str(name).strip().lower()
    if "glioma" in s:
        return "glioma"
    if "meningioma" in s:
        return "meningioma"
    if "pituitary" in s:
        return "pituitary"
    if "normal" in s or "notumor" in s or "no_tumor" in s or "no tumor" in s:
        return "notumor"
    return None

def find_root_with_required_class_dirs(base_dir, required_set, prefer_raw=True):
    candidates = []
    for root, dirs, _ in os.walk(base_dir):
        if required_set.issubset(set(dirs)):
            candidates.append(root)
    if not candidates:
        return None

    def score(path):
        pl = path.lower()
        sc = 0
        if prefer_raw:
            if "raw data" in pl:
                sc += 7
            if os.path.basename(path).lower() == "raw":
                sc += 7
            if "/raw/" in pl or "\\raw\\" in pl:
                sc += 3
            if "augmented" in pl:
                sc -= 20
        sc -= 0.0001 * len(path)
        return sc

    return max(candidates, key=score)

def try_kagglehub_download(slug, required_set, prefer_raw=True):
    try:
        base = kagglehub.dataset_download(slug)
        root = find_root_with_required_class_dirs(base, required_set, prefer_raw=prefer_raw)
        return base, root
    except Exception:
        return None, None

def find_dataset_roots():
    ds1_root = None
    ds2_root = None

    for base in DATASET_SEARCH_ROOTS:
        if os.path.exists(base):
            if ds1_root is None:
                ds1_root = find_root_with_required_class_dirs(base, REQ1, prefer_raw=True)
            if ds2_root is None:
                ds2_root = find_root_with_required_class_dirs(base, REQ2, prefer_raw=False)

    if ds1_root is None:
        _, ds1_root = try_kagglehub_download(
            "orvile/pmram-bangladeshi-brain-cancer-mri-dataset",
            REQ1,
            prefer_raw=True,
        )
    if ds2_root is None:
        _, ds2_root = try_kagglehub_download(
            "yassinebazgour/preprocessed-brain-mri-scans-for-tumors-detection",
            REQ2,
            prefer_raw=False,
        )

    if ds1_root is None:
        raise RuntimeError("Could not find DS1.")
    if ds2_root is None:
        raise RuntimeError("Could not find DS2.")

    return ds1_root, ds2_root

# ============================================================
# RECORD BUILDING
# ============================================================
def list_images_under_class_root(class_root, class_dir_name):
    class_dir = os.path.join(class_root, class_dir_name)
    out = []
    for r, _, files in os.walk(class_dir):
        for fn in files:
            if fn.lower().endswith(IMG_EXTS):
                out.append(os.path.join(r, fn))
    return out

def build_records_from_root(ds_root, class_dirs, source_name):
    records = []
    for c in class_dirs:
        lab = norm_label(c)
        imgs = list_images_under_class_root(ds_root, c)
        for p in imgs:
            records.append({
                "path": str(p),
                "label": str(lab),
                "source": str(source_name),
                "filename": os.path.basename(p),
                "y": int(label2id[lab]),
            })

    seen = set()
    uniq = []
    for r in records:
        if r["path"] not in seen:
            seen.add(r["path"])
            uniq.append(r)
    return uniq

def pick_samples_per_class(records, n_per_class=1, seed=SEED):
    rng = random.Random(seed)
    out = []
    for lab in labels:
        sub = [r for r in records if r["label"] == lab]
        if len(sub) == 0:
            continue
        k = min(n_per_class, len(sub))
        out.extend(rng.sample(sub, k))
    return out

# ============================================================
# RACE-FELCM PREPROCESSING WITH INTERMEDIATE MAPS
# ============================================================
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

    def forward(self, x, return_dict=False):
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
        z_clip = ((x_dn - mu) / sd).clamp(-self.tau, self.tau)
        z = torch.sign(z_clip) * torch.pow(z_clip.abs().clamp_min(eps), self.gamma)

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

        structure_gain = self.edge_gain * edge * coherence
        enhanced_pre_norm = z + self.alpha * contrast_field + structure_gain

        mn = enhanced_pre_norm.amin(dim=(2, 3), keepdim=True)
        mx = enhanced_pre_norm.amax(dim=(2, 3), keepdim=True)
        enhanced_norm = (enhanced_pre_norm - mn) / (mx - mn + eps)

        out = self.blend * enhanced_norm + (1.0 - self.blend) * x
        out = out.clamp(0, 1)

        abs_residual = torch.abs(out - x)
        signed_delta = out - x

        composite = normalize_map(
            0.45 * tensor_to_gray01(contrast_field) +
            0.35 * tensor_to_gray01(edge) +
            0.20 * tensor_to_gray01(coherence)
        )

        bundle = {
            "raw": x,
            "raw_gray": gray0,
            "tex_norm": tex_norm,
            "flat_gate": flat_gate,
            "x_smooth": x_smooth,
            "x_dn": x_dn,
            "z_clip": z_clip,
            "z": z,
            "local_mean": local_mean,
            "local_std": local_std,
            "local_norm": local_norm,
            "contrast_field": contrast_field,
            "edge": edge,
            "lap": lap,
            "coherence": coherence,
            "structure_gain": structure_gain,
            "enhanced_pre_norm": enhanced_pre_norm,
            "enhanced_norm": enhanced_norm,
            "out": out,
            "abs_residual": abs_residual,
            "signed_delta": signed_delta,
            "composite": composite,
        }

        if return_dict:
            return bundle
        return out

# ============================================================
# PREPARE PAYLOAD
# ============================================================
def prepare_sample_bundle(preproc, row, theta_tag, theta_val):
    img = load_rgb(row["path"])
    x = pil_to_tensor(img, IMG_SIZE).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        B = preproc(x, return_dict=True)

    raw_gray = tensor_to_gray01(B["raw"])
    out_gray = tensor_to_gray01(B["out"])

    structure_overlay = overlay_heatmap(raw_gray, tensor_to_gray01(B["structure_gain"]), cmap="jet", alpha=0.52)
    contrast_overlay = overlay_heatmap(raw_gray, normalize_map(np.abs(tensor_to_gray_signed(B["contrast_field"]))), cmap="jet", alpha=0.52)
    edge_overlay = overlay_heatmap(raw_gray, tensor_to_gray01(B["edge"]), cmap="jet", alpha=0.52)
    composite_overlay = overlay_heatmap(raw_gray, B["composite"], cmap="jet", alpha=0.54)

    return {
        "gt": row["label"],
        "theta_tag": theta_tag,
        "theta_text": theta_multiline(theta_val, theta_tag),
        "raw_gray": raw_gray,
        "out_gray": out_gray,
        "abs_residual": tensor_to_gray01(B["abs_residual"]),
        "signed_delta": tensor_to_gray_signed(B["signed_delta"]),
        "tex_norm": tensor_to_gray01(B["tex_norm"]),
        "flat_gate": tensor_to_gray01(B["flat_gate"]),
        "x_smooth": tensor_to_gray01(B["x_smooth"]),
        "x_dn": tensor_to_gray01(B["x_dn"]),
        "local_mean": tensor_to_gray01(B["local_mean"]),
        "local_std": tensor_to_gray01(B["local_std"]),
        "local_norm": tensor_to_gray_signed(B["local_norm"]),
        "contrast_field": tensor_to_gray_signed(B["contrast_field"]),
        "edge": tensor_to_gray01(B["edge"]),
        "lap": tensor_to_gray01(B["lap"]),
        "coherence": tensor_to_gray01(B["coherence"]),
        "structure_gain": tensor_to_gray01(B["structure_gain"]),
        "z_clip": tensor_to_gray_signed(B["z_clip"]),
        "z": tensor_to_gray_signed(B["z"]),
        "enhanced_pre_norm": tensor_to_gray_signed(B["enhanced_pre_norm"]),
        "enhanced_norm": tensor_to_gray01(B["enhanced_norm"]),
        "composite": B["composite"],
        "structure_overlay": structure_overlay,
        "contrast_overlay": contrast_overlay,
        "edge_overlay": edge_overlay,
        "composite_overlay": composite_overlay,
    }

# ============================================================
# FIGURE CAPTIONS AS OUTPUT TEXT ONLY
# ============================================================
CAPTION_MAP = {
    "Overview Panel": (
        "Overview Panel: Raw MRI is compared with the final RACE-FELCM output, "
        "the absolute residual magnitude, signed intensity change, and the composite preprocessing-response map."
    ),
    "Texture / Denoising Panel": (
        "Texture / Denoising Panel: This figure visualizes how RACE-FELCM estimates texture, "
        "suppresses flat noisy regions, and forms the denoised branch used before later enhancement."
    ),
    "Local Contrast Panel": (
        "Local Contrast Panel: This figure shows the local mean, local standard deviation, "
        "normalized local deviation, and the contrast field that drives fuzzy local contrast enhancement."
    ),
    "Structure Enhancement Panel": (
        "Structure Enhancement Panel: This figure visualizes edge strength, Laplacian response, "
        "coherence weighting, and the final structure-gain effect used for edge-preserving enhancement."
    ),
    "Intensity Remapping Panel": (
        "Intensity Remapping Panel: This figure shows robust z-clipping, power-law remapping, "
        "pre-normalized enhancement response, and normalized enhanced intensity before final blending."
    ),
    "Summary XAI Panel": (
        "Summary XAI Panel: This figure overlays the strongest preprocessing responses, "
        "highlighting where RACE-FELCM emphasizes contrast, edges, and the combined enhancement behavior."
    ),
}

def print_caption_outside(dataset_title, family_title):
    print(f"{dataset_title} — {family_title}")
    print("Caption:", CAPTION_MAP.get(family_title, "Preprocessing visualization for RACE-FELCM."))
    print("-" * 118)

# ============================================================
# PLOTTING
# ============================================================
def render_visual_cell(ax, payload, key, mode):
    if mode == "gray":
        ax.imshow(payload[key], cmap="gray", vmin=0, vmax=1)
    elif mode == "heat":
        ax.imshow(payload[key], cmap="jet", vmin=0, vmax=1)
    elif mode == "div":
        ax.imshow(payload[key], cmap="seismic", vmin=-1, vmax=1)
    elif mode == "rgb":
        ax.imshow(payload[key], vmin=0, vmax=1)
    clean_ax(ax)

def render_theta_cell(ax, payload):
    ax.set_facecolor("#F5F7FA")
    clean_ax(ax)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor("#CBD5E1")
        spine.set_linewidth(1.2)

    ax.text(
        0.5, 0.5, payload["theta_text"],
        transform=ax.transAxes,
        ha="center", va="center",
        fontsize=15,
        fontweight="bold",
        color="#111827",
        linespacing=1.35,
    )

def plot_family(dataset_title, family_title, rows_payload, columns, save_path=None):
    # columns:
    # [("Raw", key, mode), ("RACE-FELCM", key, mode), ("Map1", key, mode), ("Map2", key, mode), ("Map3", key, mode)]
    n = len(rows_payload)
    total_cols = 6  # Raw | THETA | RACE-FELCM | map1 | map2 | map3

    fig, axes = plt.subplots(
        n, total_cols,
        figsize=(24, 4.2 * n),
        constrained_layout=True,
        facecolor="white",
        gridspec_kw={"width_ratios": [1.0, 0.82, 1.0, 1.0, 1.0, 1.0]},
    )
    if n == 1:
        axes = np.array(axes).reshape(1, total_cols)

    header_titles = [
        columns[0][0],
        "Theta",
        columns[1][0],
        columns[2][0],
        columns[3][0],
        columns[4][0],
    ]

    for j, tt in enumerate(header_titles):
        axes[0, j].set_title(tt, fontsize=18, fontweight="bold", pad=12)

    for i, payload in enumerate(rows_payload):
        render_visual_cell(axes[i, 0], payload, columns[0][1], columns[0][2])
        render_theta_cell(axes[i, 1], payload)
        render_visual_cell(axes[i, 2], payload, columns[1][1], columns[1][2])
        render_visual_cell(axes[i, 3], payload, columns[2][1], columns[2][2])
        render_visual_cell(axes[i, 4], payload, columns[3][1], columns[3][2])
        render_visual_cell(axes[i, 5], payload, columns[4][1], columns[4][2])

        row_label(axes[i, 0], payload["gt"])
        title_box(axes[i, 0], f"GT={payload['gt']}", loc="bl", fontsize=12)
        title_box(axes[i, 2], payload["theta_tag"], loc="bl", fontsize=12)

    fig.suptitle(
        f"{dataset_title} — {family_title} (RACE-FELCM)",
        fontsize=26,
        fontweight="bold",
        color="#111827",
    )

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight", facecolor="white")
        print(f"Saved: {save_path}")

    if SHOW_FIGURES:
        plt.show()
    else:
        plt.close(fig)

    # caption is printed outside the image, not inside it
    print_caption_outside(dataset_title, family_title)

# ============================================================
# MAIN
# ============================================================
preproc_ds1 = RACEFELCM(*THETA_DS1).to(DEVICE).eval()
preproc_ds2 = RACEFELCM(*THETA_DS2).to(DEVICE).eval()

ds1_root, ds2_root = find_dataset_roots()
print(f"DS1 root: {ds1_root}")
print(f"DS2 root: {ds2_root}")

records_ds1 = build_records_from_root(
    ds1_root,
    ["512Glioma", "512Meningioma", "512Normal", "512Pituitary"],
    "ds1_raw",
)
records_ds2 = build_records_from_root(
    ds2_root,
    ["glioma", "meningioma", "notumor", "pituitary"],
    "ds2",
)

samples_ds1 = pick_samples_per_class(records_ds1, n_per_class=SAMPLES_PER_CLASS, seed=SEED)
samples_ds2 = pick_samples_per_class(records_ds2, n_per_class=SAMPLES_PER_CLASS, seed=SEED)

print(f"DS1 selected samples: {len(samples_ds1)}")
print(f"DS2 selected samples: {len(samples_ds2)}")

rows_payload_ds1 = [prepare_sample_bundle(preproc_ds1, row, "DS1 theta", THETA_DS1) for row in samples_ds1]
rows_payload_ds2 = [prepare_sample_bundle(preproc_ds2, row, "DS2 theta", THETA_DS2) for row in samples_ds2]

families = [
    (
        "Overview Panel",
        [
            ("Raw", "raw_gray", "gray"),
            ("RACE-FELCM", "out_gray", "gray"),
            ("|Delta|", "abs_residual", "heat"),
            ("Signed Delta", "signed_delta", "div"),
            ("Composite XAI", "composite_overlay", "rgb"),
        ],
        "overview",
    ),
    (
        "Texture / Denoising Panel",
        [
            ("Raw", "raw_gray", "gray"),
            ("RACE-FELCM", "out_gray", "gray"),
            ("Texture Norm", "tex_norm", "heat"),
            ("Flat Gate", "flat_gate", "heat"),
            ("Denoised Branch", "x_dn", "gray"),
        ],
        "texture_denoising",
    ),
    (
        "Local Contrast Panel",
        [
            ("Raw", "raw_gray", "gray"),
            ("RACE-FELCM", "out_gray", "gray"),
            ("Local Mean", "local_mean", "gray"),
            ("Local Std", "local_std", "heat"),
            ("Contrast Field", "contrast_field", "div"),
        ],
        "local_contrast",
    ),
    (
        "Structure Enhancement Panel",
        [
            ("Raw", "raw_gray", "gray"),
            ("RACE-FELCM", "out_gray", "gray"),
            ("Edge", "edge", "heat"),
            ("Coherence", "coherence", "heat"),
            ("Structure Overlay", "structure_overlay", "rgb"),
        ],
        "structure",
    ),
    (
        "Intensity Remapping Panel",
        [
            ("Raw", "raw_gray", "gray"),
            ("RACE-FELCM", "out_gray", "gray"),
            ("Z-Clip", "z_clip", "div"),
            ("Enhanced Norm", "enhanced_norm", "gray"),
            ("Enhanced Pre-Norm", "enhanced_pre_norm", "div"),
        ],
        "intensity_remap",
    ),
    (
        "Summary XAI Panel",
        [
            ("Raw", "raw_gray", "gray"),
            ("RACE-FELCM", "out_gray", "gray"),
            ("Contrast Overlay", "contrast_overlay", "rgb"),
            ("Edge Overlay", "edge_overlay", "rgb"),
            ("Composite Overlay", "composite_overlay", "rgb"),
        ],
        "summary_xai",
    ),
]

for family_title, cols, slug in families:
    save1 = os.path.join(OUTPUT_DIR, f"DS1_{slug}_RACE_FELCM.png") if SAVE_FIGURES else None
    plot_family("Dataset-1", family_title, rows_payload_ds1, cols, save1)

    save2 = os.path.join(OUTPUT_DIR, f"DS2_{slug}_RACE_FELCM.png") if SAVE_FIGURES else None
    plot_family("Dataset-2", family_title, rows_payload_ds2, cols, save2)

# ============================================================
# FINAL COUNT
# ============================================================
family_count = len(families)
figure_file_count = 2 * family_count
rows_per_dataset = len(rows_payload_ds1)  # same as ds2 by default
subplot_per_figure = rows_per_dataset * 6
subplot_total = figure_file_count * subplot_per_figure

print("\n" + "=" * 118)
print("DONE: RACE-FELCM PREPROCESSING-XAI FIGURES GENERATED")
print("=" * 118)
print(f"Figure families          : {family_count}")
print(f"Datasets                 : 2")
print(f"Saved figure files total : {figure_file_count}")
print(f"Rows per dataset         : DS1={len(rows_payload_ds1)}, DS2={len(rows_payload_ds2)}")
print(f"Columns per figure       : 6")
print(f"Subplots per figure      : {subplot_per_figure}")
print(f"Total subplot images     : {subplot_total}")
print(f"Output directory         : {OUTPUT_DIR}")
print("=" * 118)
