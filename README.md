# FedRL-FuseNet

Paper Title: FedRL-FuseNet: An Explainable Federated Reinforcement Learning-Guided Contrast-Enhanced Fusion Network for Brain Tumor MRI Classification

FedRL-FuseNet is a privacy-aware and explainable federated learning framework for four-class brain tumor MRI classification. The project combines adaptive contrast-enhanced preprocessing, raw-enhanced-residual representation learning, cross-residual adaptive fusion, reinforcement-guided preset selection, federated optimization, reliability analysis, external validation, and interpretability-focused model inspection.

This repository contains the core experiment notebooks, GitHub-viewable outputs, and supplementary experimental results used to support the manuscript.

## REPOSITORY STRUCTURE

| Folder / File | Description |
|---|---|
| `notebooks/original_small/` | Main experiment notebooks and GitHub-viewable experiment reports. |
| `Supplementary_Document.docx` | Word version of the supplementary experimental results document. |
| `Supplementary_Document.md` | GitHub-readable Markdown version of the supplementary experimental results document. |

## AVAILABLE EXPERIMENT FILES

| No. | Experiment Area | File |
|---:|---|---|
| 1 | Main FedRL-FuseNet development, RACE-FELCM preprocessing, fusion workflow, federated optimization, and evaluation outputs | [Main development notebook](notebooks/original_small/1_FedRACE_Net_Federated_Robust_Adaptive_Context_Enhanced_Fuzzy_Edge_Local_Contrast_Mapping_Fusion_Network.ipynb) |
| 2 | RACE-FELCM preprocessing validation and ablation results | [Preprocessing validation notebook](notebooks/original_small/2-arcf-net-preprocessing-ablation.ipynb) |
| 3 | CRAF fusion strategy ablation results | [Fusion ablation notebook](notebooks/original_small/3_ARCF_Net_Fusion_Ablations.ipynb) |
| 4 | Federated learning ablation results, including FedAvg, FedProx, and prototype sharing variants | [Federated learning ablation notebook](notebooks/original_small/4-arcf-net-federated-learning-ablations.ipynb) |
| 5 | RL-UCB preset selection and client-participation analysis | [RL-UCB analysis notebook](notebooks/original_small/5_ARCF_Net_RL_UCB_ablations.ipynb) |
| 6 | Baseline and backbone comparison experiments | [Baseline/backbone comparison notebook](notebooks/original_small/6-arcf-net-baseline-backbone-compare.ipynb) |
| 7 | GitHub-readable baseline and backbone comparison report | [Baseline/backbone comparison Markdown report](notebooks/original_small/6-arcf-net-baseline-backbone-compare.md) |
| 8 | External validation experiments | [External validation notebook](notebooks/original_small/7-arcf-net-external-validation.ipynb) |

## SUPPLEMENTARY MATERIAL

The supplementary experimental results are provided in two formats.

| File | Purpose |
|---|---|
| [Supplementary_Document.docx](Supplementary_Document.docx) | Formatted Word document containing additional experimental result tables. |
| [Supplementary_Document.md](Supplementary_Document.md) | GitHub-readable Markdown version for direct online viewing. |

The supplementary document includes additional preprocessing validation, fusion ablation, federated learning ablation, RL-UCB selection analysis, baseline/backbone comparison, and external validation outputs that support the main manuscript.

## FULL EXPLAINABILITY NOTEBOOK

The full explainability notebook is available through Google Drive:

[Download 8_RACE_XAI.ipynb](https://drive.google.com/file/d/1l09XniYVSa6bCjXupKm7lrAELxxZWJiX/view?usp=sharing)

This notebook contains the full explainability workflow used to inspect FedRL-FuseNet model behavior.

## HOW TO USE

Open the notebooks inside `notebooks/original_small/` directly on GitHub or download them for local execution. For the baseline/backbone comparison, the Markdown report is also provided because the original notebook output can be too large for reliable GitHub rendering.

For supplementary results, open `Supplementary_Document.md` directly on GitHub or download `Supplementary_Document.docx` for the formatted Word version.

## CITATION

If this repository supports your research, please cite the related paper or this repository.
