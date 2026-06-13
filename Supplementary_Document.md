Supplementary Document

Additional Experimental Results for FedRL-FuseNet

## RACE-FELCM Preprocessing Validation and Ablation Results

Ablation: No Preprocessing (Identity) vs RACE-FELCM variants

### No Preprocessing

Quick Summary

| Metric | Value |
| --- | --- |
| DS1 TEST Accuracy | 96.90% |
| DS2 TEST Accuracy | 97.91% |

Extended TEST Metrics

| Dataset | Acc | Balanced Acc | Precision (Macro) | Recall (Macro) | F1 (Macro) |
| --- | --- | --- | --- | --- | --- |
| DS1 TEST | 96.90% | 96.94% | 96.96% | 96.94% | 96.92% |
| DS2 TEST | 97.91% | 97.89% | 97.84% | 97.89% | 97.86% |

Paper-Ready Metrics (Validation and Test)

| Setting | Split | Dataset | Accuracy |
| --- | --- | --- | --- |
| FedRL-FuseNet Ablation 1 (No Preprocessing) | VAL | DS1 | 98.23% |
| FedRL-FuseNet Ablation 1 (No Preprocessing) | VAL | DS2 | 97.63% |
| FedRL-FuseNet Ablation 1 (No Preprocessing) | VAL | global-equal | 97.93% |
| FedRL-FuseNet Ablation 1 (No Preprocessing) | TEST | DS1 | 96.90% |
| FedRL-FuseNet Ablation 1 (No Preprocessing) | TEST | DS2 | 97.91% |
| FedRL-FuseNet Ablation 1 (No Preprocessing) | TEST | global-equal | 97.41% |

Classwise Metrics — DS1 TEST

| Class ID | Class Name | Support | TP | FP | FN |
| --- | --- | --- | --- | --- | --- |
| 0 | glioma | 56 | 54 | 4 | 2 |
| 1 | meningioma | 55 | 55 | 2 | 0 |
| 2 | no tumor | 59 | 56 | 1 | 3 |
| 3 | pituitary | 56 | 54 | 0 | 2 |

Classwise Metrics — DS2 TEST

| Class ID | Class Name | Support | TP | FP | FN |
| --- | --- | --- | --- | --- | --- |
| 0 | glioma | 244 | 242 | 5 | 2 |
| 1 | meningioma | 247 | 238 | 8 | 9 |
| 2 | no tumor | 300 | 296 | 0 | 4 |
| 3 | pituitary | 264 | 257 | 9 | 7 |

### Fixed RACE-FELCM Preset

Quick Summary

| Metric | Value |
| --- | --- |
| DS1 TEST Accuracy | 96.46% |
| DS2 TEST Accuracy | 98.01% |

Extended TEST Metrics

| Dataset | Acc | Balanced Acc | Precision (Macro) | Recall (Macro) | F1 (Macro) |
| --- | --- | --- | --- | --- | --- |
| DS1 TEST | 96.46% | 96.49% | 96.49% | 96.49% | 96.46% |
| DS2 TEST | 98.01% | 97.97% | 97.93% | 97.97% | 97.95% |

Classwise Metrics — DS1 TEST

| Class ID | Class Name | Support | TP | FP | FN |
| --- | --- | --- | --- | --- | --- |
| 0 | glioma | 56 | 54 | 4 | 2 |
| 1 | meningioma | 55 | 54 | 2 | 1 |
| 2 | no tumor | 59 | 56 | 0 | 3 |
| 3 | pituitary | 56 | 54 | 2 | 2 |

Classwise Metrics — DS2 TEST

| Class ID | Class Name | Support | TP | FP | FN |
| --- | --- | --- | --- | --- | --- |
| 0 | glioma | 244 | 241 | 4 | 3 |
| 1 | meningioma | 247 | 238 | 9 | 9 |
| 2 | no tumor | 300 | 296 | 0 | 4 |
| 3 | pituitary | 264 | 259 | 8 | 5 |

### RL-Selected RACE-FELCM

Quick Summary

| Metric | Value |
| --- | --- |
| DS1 TEST Accuracy | 97.79% |
| DS2 TEST Accuracy | 98.01% |

Extended TEST Metrics

| Dataset | Acc | Balanced Acc | Precision (Macro) | Recall (Macro) | F1 (Macro) |
| --- | --- | --- | --- | --- | --- |
| DS1 TEST | 97.79% | 97.79% | 97.81% | 97.79% | 97.79% |
| DS2 TEST | 98.01% | 97.99% | 97.92% | 97.99% | 97.95% |

Paper-Ready Metrics (Validation and Test)

| Setting | Split | Dataset | Accuracy |
| --- | --- | --- | --- |
| FedRL-FuseNet (RL-selected RACE-FELCM) | VAL | DS1 | 99.11% |
| FedRL-FuseNet (RL-selected RACE-FELCM) | VAL | DS2 | 97.73% |
| FedRL-FuseNet (RL-selected RACE-FELCM) | VAL | global-equal | 98.42% |
| FedRL-FuseNet (RL-selected RACE-FELCM) | TEST | DS1 | 97.79% |
| FedRL-FuseNet (RL-selected RACE-FELCM) | TEST | DS2 | 98.01% |
| FedRL-FuseNet (RL-selected RACE-FELCM) | TEST | global-equal | 97.90% |

Classwise Metrics — DS1 TEST

| Class ID | Class Name | Support | TP | FP | FN |
| --- | --- | --- | --- | --- | --- |
| 0 | glioma | 56 | 54 | 3 | 2 |
| 1 | meningioma | 55 | 55 | 1 | 0 |
| 2 | no tumor | 59 | 58 | 1 | 1 |
| 3 | pituitary | 56 | 54 | 0 | 2 |

Classwise Metrics — DS2 TEST

| Class ID | Class Name | Support | TP | FP | FN |
| --- | --- | --- | --- | --- | --- |
| 0 | glioma | 244 | 241 | 6 | 3 |
| 1 | meningioma | 247 | 240 | 9 | 7 |
| 2 | no tumor | 300 | 296 | 0 | 4 |
| 3 | pituitary | 264 | 257 | 6 | 7 |

### Raw Only

Quick Summary

| Metric | Value |
| --- | --- |
| DS1 TEST Accuracy | 97.35% |
| DS2 TEST Accuracy | 97.91% |

Extended TEST Metrics

| Dataset | Acc | Balanced Acc | Precision (Macro) | Recall (Macro) | F1 (Macro) |
| --- | --- | --- | --- | --- | --- |
| DS1 TEST | 97.35% | 97.34% | 97.40% | 97.34% | 97.35% |
| DS2 TEST | 97.91% | 97.89% | 97.83% | 97.89% | 97.85% |

Paper-Ready Metrics (Validation and Test)

| Setting | Split | Dataset | Accuracy |
| --- | --- | --- | --- |
| FedRL-FuseNet (Raw Only) | VAL | DS1 | 98.67% |
| FedRL-FuseNet (Raw Only) | VAL | DS2 | 97.82% |
| FedRL-FuseNet (Raw Only) | VAL | global-equal | 98.25% |
| FedRL-FuseNet (Raw Only) | TEST | DS1 | 97.35% |
| FedRL-FuseNet (Raw Only) | TEST | DS2 | 97.91% |
| FedRL-FuseNet (Raw Only) | TEST | global-equal | 97.63% |

Classwise Metrics — DS1 TEST

| Class ID | Class Name | Support | TP | FP | FN |
| --- | --- | --- | --- | --- | --- |
| 0 | glioma | 56 | 54 | 3 | 2 |
| 1 | meningioma | 55 | 54 | 1 | 1 |
| 2 | no tumor | 59 | 58 | 2 | 1 |
| 3 | pituitary | 56 | 54 | 0 | 2 |

Classwise Metrics — DS2 TEST

| Class ID | Class Name | Support | TP | FP | FN |
| --- | --- | --- | --- | --- | --- |
| 0 | glioma | 244 | 238 | 5 | 6 |
| 1 | meningioma | 247 | 243 | 14 | 4 |
| 2 | no tumor | 300 | 296 | 0 | 4 |
| 3 | pituitary | 264 | 256 | 3 | 8 |

### Enhanced Only

Quick Summary

| Metric | Value |
| --- | --- |
| DS1 TEST Accuracy | 96.02% |
| DS2 TEST Accuracy | 97.54% |

Extended TEST Metrics

| Dataset | Acc | Balanced Acc | Precision (Macro) | Recall (Macro) | F1 (Macro) |
| --- | --- | --- | --- | --- | --- |
| DS1 TEST | 96.02% | 96.10% | 96.09% | 96.10% | 96.04% |
| DS2 TEST | 97.54% | 97.50% | 97.42% | 97.50% | 97.45% |

Paper-Ready Metrics (Validation and Test)

| Setting | Split | Dataset | Accuracy |
| --- | --- | --- | --- |
| FedRL-FuseNet (Enhanced Only) | VAL | DS1 | 97.79% |
| FedRL-FuseNet (Enhanced Only) | VAL | DS2 | 97.63% |
| FedRL-FuseNet (Enhanced Only) | VAL | global-equal | 97.71% |
| FedRL-FuseNet (Enhanced Only) | TEST | DS1 | 96.02% |
| FedRL-FuseNet (Enhanced Only) | TEST | DS2 | 97.54% |
| FedRL-FuseNet (Enhanced Only) | TEST | global-equal | 96.78% |

Classwise Metrics — DS1 TEST

| Class ID | Class Name | Support | TP | FP | FN |
| --- | --- | --- | --- | --- | --- |
| 0 | glioma | 56 | 54 | 5 | 2 |
| 1 | meningioma | 55 | 55 | 1 | 0 |
| 2 | no tumor | 59 | 54 | 1 | 5 |
| 3 | pituitary | 56 | 54 | 2 | 2 |

Classwise Metrics — DS2 TEST

| Class ID | Class Name | Support | TP | FP | FN |
| --- | --- | --- | --- | --- | --- |
| 0 | glioma | 244 | 242 | 11 | 2 |
| 1 | meningioma | 247 | 237 | 9 | 10 |
| 2 | no tumor | 300 | 296 | 0 | 4 |
| 3 | pituitary | 264 | 254 | 6 | 10 |

### Raw + Enhanced

Quick Summary

| Metric | Value |
| --- | --- |
| DS1 TEST Accuracy | 96.90% |
| DS2 TEST Accuracy | 97.63% |

Extended TEST Metrics

| Dataset | Acc | Balanced Acc | Precision (Macro) | Recall (Macro) | F1 (Macro) |
| --- | --- | --- | --- | --- | --- |
| DS1 TEST | 96.90% | 96.94% | 96.91% | 96.94% | 96.91% |
| DS2 TEST | 97.63% | 97.61% | 97.56% | 97.61% | 97.58% |

Paper-Ready Metrics (Validation and Test)

| Setting | Split | Dataset | Accuracy |
| --- | --- | --- | --- |
| FedRL-FuseNet (Raw + Enhanced) | VAL | DS1 | 97.79% |
| FedRL-FuseNet (Raw + Enhanced) | VAL | DS2 | 97.63% |
| FedRL-FuseNet (Raw + Enhanced) | VAL | global-equal | 97.71% |
| FedRL-FuseNet (Raw + Enhanced) | TEST | DS1 | 96.90% |
| FedRL-FuseNet (Raw + Enhanced) | TEST | DS2 | 97.63% |
| FedRL-FuseNet (Raw + Enhanced) | TEST | global-equal | 97.27% |

Classwise Metrics — DS1 TEST

| Class ID | Class Name | Support | TP | FP | FN |
| --- | --- | --- | --- | --- | --- |
| 0 | glioma | 56 | 54 | 3 | 2 |
| 1 | meningioma | 55 | 55 | 1 | 0 |
| 2 | no tumor | 59 | 56 | 1 | 3 |
| 3 | pituitary | 56 | 54 | 2 | 2 |

Classwise Metrics — DS2 TEST

| Class ID | Class Name | Support | TP | FP | FN |
| --- | --- | --- | --- | --- | --- |
| 0 | glioma | 244 | 240 | 4 | 4 |
| 1 | meningioma | 247 | 239 | 9 | 8 |
| 2 | no tumor | 300 | 295 | 0 | 5 |
| 3 | pituitary | 264 | 256 | 12 | 8 |

### Raw + Enhanced + Residual

Quick Summary

| Metric | Value |
| --- | --- |
| DS1 TEST Accuracy | 97.79% |
| DS2 TEST Accuracy | 98.01% |

Extended TEST Metrics

| Dataset | Acc | Balanced Acc | Precision (Macro) | Recall (Macro) | F1 (Macro) |
| --- | --- | --- | --- | --- | --- |
| DS1 TEST | 97.79% | 97.79% | 97.81% | 97.79% | 97.79% |
| DS2 TEST | 98.01% | 97.99% | 97.92% | 97.99% | 97.95% |

Paper-Ready Metrics (Validation and Test)

| Setting | Split | Dataset | Accuracy |
| --- | --- | --- | --- |
| FedRL-FuseNet (Raw + Enhanced + Residual) | VAL | DS1 | 99.11% |
| FedRL-FuseNet (Raw + Enhanced + Residual) | VAL | DS2 | 97.73% |
| FedRL-FuseNet (Raw + Enhanced + Residual) | VAL | global-equal | 98.42% |
| FedRL-FuseNet (Raw + Enhanced + Residual) | TEST | DS1 | 97.79% |
| FedRL-FuseNet (Raw + Enhanced + Residual) | TEST | DS2 | 98.01% |
| FedRL-FuseNet (Raw + Enhanced + Residual) | TEST | global-equal | 97.90% |

Classwise Metrics — DS1 TEST

| Class ID | Class Name | Support | TP | FP | FN |
| --- | --- | --- | --- | --- | --- |
| 0 | glioma | 56 | 54 | 3 | 2 |
| 1 | meningioma | 55 | 55 | 1 | 0 |
| 2 | no tumor | 59 | 58 | 1 | 1 |
| 3 | pituitary | 56 | 54 | 0 | 2 |

Classwise Metrics — DS2 TEST

| Class ID | Class Name | Support | TP | FP | FN |
| --- | --- | --- | --- | --- | --- |
| 0 | glioma | 244 | 241 | 6 | 3 |
| 1 | meningioma | 247 | 240 | 9 | 7 |
| 2 | no tumor | 300 | 296 | 0 | 4 |
| 3 | pituitary | 264 | 257 | 6 | 7 |

## CRAF Fusion Strategy Ablation Results

Ablation: CRAF vs Late Concatenation / Weighted Sum / Attention / Dual-Head fusion

### Late Concatenation Fusion

Quick Summary

| Metric | Value |
| --- | --- |
| DS1 TEST Accuracy | 98.23% |
| DS2 TEST Accuracy | 97.35% |

Extended TEST Metrics

| Dataset | Acc | Balanced Acc | Precision (Macro) | Recall (Macro) | F1 (Macro) |
| --- | --- | --- | --- | --- | --- |
| DS1 TEST | 98.23% | 98.23% | 98.26% | 98.23% | 98.23% |
| DS2 TEST | 97.35% | 97.26% | 97.28% | 97.26% | 97.26% |

Paper-Ready Metrics (Validation and Test)

| Setting | Split | Dataset | Accuracy |
| --- | --- | --- | --- |
| FedRL-FuseNet LateConcat | VAL | DS1 | 97.35% |
| FedRL-FuseNet LateConcat | VAL | DS2 | 98.01% |
| FedRL-FuseNet LateConcat | VAL | global-equal | 97.68% |
| FedRL-FuseNet LateConcat | TEST | DS1 | 98.23% |
| FedRL-FuseNet LateConcat | TEST | DS2 | 97.35% |
| FedRL-FuseNet LateConcat | TEST | global-equal | 97.79% |

Classwise Metrics — DS1 TEST

| Class ID | Class Name | Support | TP | FP | FN |
| --- | --- | --- | --- | --- | --- |
| 0 | glioma | 56 | 54 | 1 | 2 |
| 1 | meningioma | 55 | 54 | 0 | 1 |
| 2 | no tumor | 59 | 58 | 1 | 1 |
| 3 | pituitary | 56 | 56 | 2 | 0 |

Classwise Metrics — DS2 TEST

| Class ID | Class Name | Support | TP | FP | FN |
| --- | --- | --- | --- | --- | --- |
| 0 | glioma | 244 | 233 | 4 | 11 |
| 1 | meningioma | 247 | 238 | 15 | 9 |
| 2 | no tumor | 300 | 295 | 3 | 5 |
| 3 | pituitary | 264 | 261 | 6 | 3 |

### Weighted Sum Fusion

Quick Summary

| Metric | Value |
| --- | --- |
| DS1 TEST Accuracy | 97.35% |
| DS2 TEST Accuracy | 97.54% |

Extended TEST Metrics

| Dataset | Acc | Balanced Acc | Precision (Macro) | Recall (Macro) | F1 (Macro) |
| --- | --- | --- | --- | --- | --- |
| DS1 TEST | 97.35% | 97.39% | 97.40% | 97.39% | 97.36% |
| DS2 TEST | 97.54% | 97.52% | 97.43% | 97.52% | 97.47% |

Paper-Ready Metrics (Validation and Test)

| Setting | Split | Dataset | Accuracy |
| --- | --- | --- | --- |
| FedRL-FuseNet WeightedSum | VAL | DS1 | 96.90% |
| FedRL-FuseNet WeightedSum | VAL | DS2 | 97.63% |
| FedRL-FuseNet WeightedSum | VAL | global-equal | 97.27% |
| FedRL-FuseNet WeightedSum | TEST | DS1 | 97.35% |
| FedRL-FuseNet WeightedSum | TEST | DS2 | 97.54% |
| FedRL-FuseNet WeightedSum | TEST | global-equal | 97.44% |

Classwise Metrics — DS1 TEST

| Class ID | Class Name | Support | TP | FP | FN |
| --- | --- | --- | --- | --- | --- |
| 0 | glioma | 56 | 55 | 4 | 1 |
| 1 | meningioma | 55 | 55 | 1 | 0 |
| 2 | no tumor | 59 | 56 | 0 | 3 |
| 3 | pituitary | 56 | 54 | 1 | 2 |

Classwise Metrics — DS2 TEST

| Class ID | Class Name | Support | TP | FP | FN |
| --- | --- | --- | --- | --- | --- |
| 0 | glioma | 244 | 241 | 7 | 3 |
| 1 | meningioma | 247 | 239 | 13 | 8 |
| 2 | no tumor | 300 | 295 | 0 | 5 |
| 3 | pituitary | 264 | 254 | 6 | 10 |

### Attention Fusion

Quick Summary

| Metric | Value |
| --- | --- |
| DS1 TEST Accuracy | 96.90% |
| DS2 TEST Accuracy | 97.73% |

Extended TEST Metrics

| Dataset | Acc | Balanced Acc | Precision (Macro) | Recall (Macro) | F1 (Macro) |
| --- | --- | --- | --- | --- | --- |
| DS1 TEST | 96.90% | 96.94% | 96.91% | 96.94% | 96.90% |
| DS2 TEST | 97.73% | 97.66% | 97.65% | 97.66% | 97.65% |

Paper-Ready Metrics (Validation and Test)

| Setting | Split | Dataset | Accuracy |
| --- | --- | --- | --- |
| FedRL-FuseNet AttentionFusion | VAL | DS1 | 97.79% |
| FedRL-FuseNet AttentionFusion | VAL | DS2 | 97.82% |
| FedRL-FuseNet AttentionFusion | VAL | global-equal | 97.80% |
| FedRL-FuseNet AttentionFusion | TEST | DS1 | 96.90% |
| FedRL-FuseNet AttentionFusion | TEST | DS2 | 97.73% |
| FedRL-FuseNet AttentionFusion | TEST | global-equal | 97.31% |

Classwise Metrics — DS1 TEST

| Class ID | Class Name | Support | TP | FP | FN |
| --- | --- | --- | --- | --- | --- |
| 0 | glioma | 56 | 54 | 3 | 2 |
| 1 | meningioma | 55 | 55 | 2 | 0 |
| 2 | no tumor | 59 | 56 | 0 | 3 |
| 3 | pituitary | 56 | 54 | 2 | 2 |

Classwise Metrics — DS2 TEST

| Class ID | Class Name | Support | TP | FP | FN |
| --- | --- | --- | --- | --- | --- |
| 0 | glioma | 244 | 239 | 5 | 5 |
| 1 | meningioma | 247 | 236 | 8 | 11 |
| 2 | no tumor | 300 | 296 | 0 | 4 |
| 3 | pituitary | 264 | 260 | 11 | 4 |

### Dual-Head FedRL-FuseNet

Quick Summary

| Metric | Value |
| --- | --- |
| DS1 TEST Accuracy | 96.46% |
| DS2 TEST Accuracy | 97.54% |

Extended TEST Metrics

| Dataset | Acc | Balanced Acc | Precision (Macro) | Recall (Macro) | F1 (Macro) |
| --- | --- | --- | --- | --- | --- |
| DS1 TEST | 96.46% | 96.49% | 96.53% | 96.49% | 96.49% |
| DS2 TEST | 97.54% | 97.48% | 97.44% | 97.48% | 97.46% |

Paper-Ready Metrics (Validation and Test)

| Setting | Split | Dataset | Accuracy |
| --- | --- | --- | --- |
| FedRL-FuseNet DualHead | VAL | DS1 | 96.90% |
| FedRL-FuseNet DualHead | VAL | DS2 | 97.63% |
| FedRL-FuseNet DualHead | VAL | global-equal | 97.27% |
| FedRL-FuseNet DualHead | TEST | DS1 | 96.46% |
| FedRL-FuseNet DualHead | TEST | DS2 | 97.54% |
| FedRL-FuseNet DualHead | TEST | global-equal | 97.00% |

Classwise Metrics — DS1 TEST

| Class ID | Class Name | Support | TP | FP | FN |
| --- | --- | --- | --- | --- | --- |
| 0 | glioma | 56 | 54 | 5 | 2 |
| 1 | meningioma | 55 | 54 | 1 | 1 |
| 2 | no tumor | 59 | 56 | 1 | 3 |
| 3 | pituitary | 56 | 54 | 1 | 2 |

Classwise Metrics — DS2 TEST

| Class ID | Class Name | Support | TP | FP | FN |
| --- | --- | --- | --- | --- | --- |
| 0 | glioma | 244 | 242 | 7 | 2 |
| 1 | meningioma | 247 | 234 | 8 | 13 |
| 2 | no tumor | 300 | 296 | 0 | 4 |
| 3 | pituitary | 264 | 257 | 11 | 7 |

## Federated Learning Ablation Results

Ablation: FedAvg only vs FedAvg+FedProx vs prototype sharing variants

### FedAvg Only

Quick Summary

| Metric | Value |
| --- | --- |
| DS1 TEST Accuracy | 96.90% |
| DS2 TEST Accuracy | 93.40% |

Extended TEST Metrics

| Dataset | Acc | Balanced Acc | Precision (Macro) | Recall (Macro) | F1 (Macro) |
| --- | --- | --- | --- | --- | --- |
| DS1 TEST | 96.90% | 96.92% | 97.04% | 96.92% | 96.88% |
| DS2 TEST | 93.40% | 93.06% | 93.61% | 93.06% | 92.81% |

Paper-Ready Metrics (Validation and Test)

| Setting | Split | Dataset | Accuracy |
| --- | --- | --- | --- |
| FedRL-FuseNet + FedAvg only | VAL | DS1 | 96.90% |
| FedRL-FuseNet + FedAvg only | VAL | DS2 | 95.94% |
| FedRL-FuseNet + FedAvg only | VAL | global-equal | 96.42% |
| FedRL-FuseNet + FedAvg only | TEST | DS1 | 96.90% |
| FedRL-FuseNet + FedAvg only | TEST | DS2 | 93.40% |
| FedRL-FuseNet + FedAvg only | TEST | global-equal | 95.15% |

Classwise Metrics — DS1 TEST

| Class ID | Class Name | Support | TP | FP | FN |
| --- | --- | --- | --- | --- | --- |
| 0 | glioma | 56 | 51 | 0 | 5 |
| 1 | meningioma | 55 | 55 | 3 | 0 |
| 2 | no tumor | 59 | 57 | 0 | 2 |
| 3 | pituitary | 56 | 56 | 4 | 0 |

Classwise Metrics — DS2 TEST

| Class ID | Class Name | Support | TP | FP | FN |
| --- | --- | --- | --- | --- | --- |
| 0 | glioma | 45 | 45 | 2 | 0 |
| 1 | meningioma | 46 | 35 | 1 | 11 |
| 2 | no tumor | 61 | 60 | 0 | 1 |
| 3 | pituitary | 45 | 44 | 10 | 1 |

### FedAvg + FedProx

Quick Summary

| Metric | Value |
| --- | --- |
| DS1 TEST Accuracy | 97.35% |
| DS2 TEST Accuracy | 93.91% |

Extended TEST Metrics

| Dataset | Acc | Balanced Acc | Precision (Macro) | Recall (Macro) | F1 (Macro) |
| --- | --- | --- | --- | --- | --- |
| DS1 TEST | 97.35% | 97.33% | 97.42% | 97.33% | 97.33% |
| DS2 TEST | 93.91% | 93.60% | 94.07% | 93.60% | 93.35% |

Paper-Ready Metrics (Validation and Test)

| Setting | Split | Dataset | Accuracy |
| --- | --- | --- | --- |
| FedRL-FuseNet + FedAvg + FedProx | VAL | DS1 | 96.46% |
| FedRL-FuseNet + FedAvg + FedProx | VAL | DS2 | 95.43% |
| FedRL-FuseNet + FedAvg + FedProx | VAL | global-equal | 95.95% |
| FedRL-FuseNet + FedAvg + FedProx | TEST | DS1 | 97.35% |
| FedRL-FuseNet + FedAvg + FedProx | TEST | DS2 | 93.91% |
| FedRL-FuseNet + FedAvg + FedProx | TEST | global-equal | 95.63% |

Classwise Metrics — DS1 TEST

| Class ID | Class Name | Support | TP | FP | FN |
| --- | --- | --- | --- | --- | --- |
| 0 | glioma | 56 | 53 | 0 | 3 |
| 1 | meningioma | 55 | 53 | 2 | 2 |
| 2 | no tumor | 59 | 58 | 0 | 1 |
| 3 | pituitary | 56 | 56 | 4 | 0 |

Classwise Metrics — DS2 TEST

| Class ID | Class Name | Support | TP | FP | FN |
| --- | --- | --- | --- | --- | --- |
| 0 | glioma | 45 | 45 | 5 | 0 |
| 1 | meningioma | 46 | 36 | 0 | 10 |
| 2 | no tumor | 61 | 60 | 0 | 1 |
| 3 | pituitary | 45 | 44 | 7 | 1 |

### FedAvg + Prototype Sharing

Quick Summary

| Metric | Value |
| --- | --- |
| DS1 TEST Accuracy | 98.67% |
| DS2 TEST Accuracy | 94.92% |

Extended TEST Metrics

| Dataset | Acc | Balanced Acc | Precision (Macro) | Recall (Macro) | F1 (Macro) |
| --- | --- | --- | --- | --- | --- |
| DS1 TEST | 98.67% | 98.71% | 98.71% | 98.71% | 98.70% |
| DS2 TEST | 94.92% | 94.69% | 94.89% | 94.69% | 94.56% |

Paper-Ready Metrics (Validation and Test)

| Setting | Split | Dataset | Accuracy |
| --- | --- | --- | --- |
| FedRL-FuseNet + FedAvg + prototype sharing | VAL | DS1 | 98.67% |
| FedRL-FuseNet + FedAvg + prototype sharing | VAL | DS2 | 95.94% |
| FedRL-FuseNet + FedAvg + prototype sharing | VAL | global-equal | 97.31% |
| FedRL-FuseNet + FedAvg + prototype sharing | TEST | DS1 | 98.67% |
| FedRL-FuseNet + FedAvg + prototype sharing | TEST | DS2 | 94.92% |
| FedRL-FuseNet + FedAvg + prototype sharing | TEST | global-equal | 96.80% |

Classwise Metrics — DS1 TEST

| Class ID | Class Name | Support | TP | FP | FN |
| --- | --- | --- | --- | --- | --- |
| 0 | glioma | 56 | 55 | 0 | 1 |
| 1 | meningioma | 55 | 55 | 0 | 0 |
| 2 | no tumor | 59 | 57 | 1 | 2 |
| 3 | pituitary | 56 | 56 | 2 | 0 |

Classwise Metrics — DS2 TEST

| Class ID | Class Name | Support | TP | FP | FN |
| --- | --- | --- | --- | --- | --- |
| 0 | glioma | 45 | 45 | 2 | 0 |
| 1 | meningioma | 46 | 38 | 1 | 8 |
| 2 | no tumor | 61 | 60 | 1 | 1 |
| 3 | pituitary | 45 | 44 | 6 | 1 |

### FedAvg + FedProx + Prototype Sharing

Quick Summary

| Metric | Value |
| --- | --- |
| DS1 TEST Accuracy | 98.67% |
| DS2 TEST Accuracy | 94.92% |

Extended TEST Metrics

| Dataset | Acc | Balanced Acc | Precision (Macro) | Recall (Macro) | F1 (Macro) |
| --- | --- | --- | --- | --- | --- |
| DS1 TEST | 98.67% | 98.71% | 98.68% | 98.71% | 98.69% |
| DS2 TEST | 94.92% | 94.69% | 94.80% | 94.69% | 94.50% |

Classwise Metrics — DS1 TEST

| Class ID | Class Name | Support | TP | FP | FN |
| --- | --- | --- | --- | --- | --- |
| 0 | glioma | 56 | 55 | 1 | 1 |
| 1 | meningioma | 55 | 55 | 0 | 0 |
| 2 | no tumor | 59 | 57 | 1 | 2 |
| 3 | pituitary | 56 | 56 | 1 | 0 |

Classwise Metrics — DS2 TEST

| Class ID | Class Name | Support | TP | FP | FN |
| --- | --- | --- | --- | --- | --- |
| 0 | glioma | 45 | 45 | 3 | 0 |
| 1 | meningioma | 46 | 38 | 1 | 8 |
| 2 | no tumor | 61 | 60 | 0 | 1 |
| 3 | pituitary | 45 | 44 | 6 | 1 |

### Grouped FedAvg with Private Dataset Heads

Quick Summary

| Metric | Value |
| --- | --- |
| DS1 TEST Accuracy | 97.79% |
| DS2 TEST Accuracy | 93.40% |

Extended TEST Metrics

| Dataset | Acc | Balanced Acc | Precision (Macro) | Recall (Macro) | F1 (Macro) |
| --- | --- | --- | --- | --- | --- |
| DS1 TEST | 97.79% | 97.80% | 97.83% | 97.80% | 97.79% |
| DS2 TEST | 93.40% | 93.06% | 93.40% | 93.06% | 92.77% |

Classwise Metrics — DS1 TEST

| Class ID | Class Name | Support | TP | FP | FN |
| --- | --- | --- | --- | --- | --- |
| 0 | glioma | 56 | 55 | 2 | 1 |
| 1 | meningioma | 55 | 53 | 0 | 2 |
| 2 | no tumor | 59 | 57 | 1 | 2 |
| 3 | pituitary | 56 | 56 | 2 | 0 |

Classwise Metrics — DS2 TEST

| Class ID | Class Name | Support | TP | FP | FN |
| --- | --- | --- | --- | --- | --- |
| 0 | glioma | 45 | 45 | 6 | 0 |
| 1 | meningioma | 46 | 35 | 1 | 11 |
| 2 | no tumor | 61 | 60 | 1 | 1 |
| 3 | pituitary | 45 | 44 | 5 | 1 |

## RL-UCB Selection Analysis Results

Ablation: no RL/fixed preset, random preset selection, best static preset selection, RL-UCB preset selection, fixed client participation, and RL-based client participation variants

### RL-UCB experiment summary

| Experiment | Best round | DS1 TEST Accuracy | DS2 TEST Accuracy | Global-equal TEST Accuracy | Global-equal TEST F1 (Macro) | Selected preset / strategy |
| --- | --- | --- | --- | --- | --- | --- |
| No RL, Fixed Preset | 10 | 98.23% | 97.35% | 97.79% | 97.70% | DS1: race_balanced; DS2: race_balanced |
| Random Preset per Round | 15 | 97.79% | 98.58% | 98.18% | 98.16% | DS1: race_balanced; DS2: race_smoothmix |
| Best Static Preset | 15 | 97.79% | 98.01% | 97.90% | 97.87% | DS1: race_edge_plus; DS2: race_balanced |
| RL-UCB Preset Selection | 13 | 96.90% | 98.29% | 97.60% | 97.57% | DS1: race_sharp; DS2: race_robust |
| Fixed Client Participation with RL-UCB Preset Selection | 15 | 97.35% | 98.39% | 97.87% | 97.85% | DS1: race_robust; DS2: race_soft |
| RL-Based Client Participation with RL-UCB Preset Selection | 10 | 95.13% | 97.44% | 96.29% | 96.21% | DS1: race_robust; DS2: race_robust; active clients DS1=2, DS2=2 |

### Validation metrics

| Experiment | Dataset | Accuracy | Precision (Macro) | Recall (Macro) | F1 (Macro) | Weighted F1 | Log Loss | AUC-ROC (Macro OVR) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| No RL, Fixed Preset | DS1 | 96.46% | 96.49% | 96.38% | 96.41% | 96.45% | 0.1102 | 0.9984 |
| No RL, Fixed Preset | DS2 | 97.73% | 97.67% | 97.61% | 97.62% | 97.71% | 0.0872 | 0.9983 |
| No RL, Fixed Preset | global-equal | 97.09% | 97.08% | 97.00% | 97.02% | 97.08% | 0.0987 | 0.9984 |
| Random Preset per Round | DS1 | 97.79% | 97.87% | 97.73% | 97.75% | 97.77% | 0.0726 | 0.9997 |
| Random Preset per Round | DS2 | 98.01% | 97.94% | 97.96% | 97.95% | 98.01% | 0.0692 | 0.9994 |
| Random Preset per Round | global-equal | 97.90% | 97.91% | 97.84% | 97.85% | 97.89% | 0.0709 | 0.9996 |
| Best Static Preset | DS1 | 96.90% | 97.08% | 96.81% | 96.84% | 96.86% | 0.1266 | 0.9952 |
| Best Static Preset | DS2 | 97.91% | 97.86% | 97.87% | 97.86% | 97.91% | 0.0693 | 0.9982 |
| Best Static Preset | global-equal | 97.41% | 97.47% | 97.34% | 97.35% | 97.39% | 0.0980 | 0.9967 |
| RL-UCB Preset Selection | DS1 | 96.90% | 97.11% | 96.81% | 96.85% | 96.86% | 0.0978 | 0.9992 |
| RL-UCB Preset Selection | DS2 | 97.91% | 97.85% | 97.84% | 97.84% | 97.92% | 0.0818 | 0.9991 |
| RL-UCB Preset Selection | global-equal | 97.41% | 97.48% | 97.32% | 97.35% | 97.39% | 0.0898 | 0.9991 |
| Fixed Client Participation with RL-UCB Preset Selection | DS1 | 96.90% | 97.06% | 96.85% | 96.88% | 96.88% | 0.0982 | 0.9990 |
| Fixed Client Participation with RL-UCB Preset Selection | DS2 | 98.29% | 98.24% | 98.27% | 98.25% | 98.29% | 0.0743 | 0.9988 |
| Fixed Client Participation with RL-UCB Preset Selection | global-equal | 97.60% | 97.65% | 97.56% | 97.57% | 97.59% | 0.0863 | 0.9989 |
| RL-Based Client Participation with RL-UCB Preset Selection | DS1 | 95.58% | 95.85% | 95.42% | 95.39% | 95.46% | 0.1508 | 0.9983 |
| RL-Based Client Participation with RL-UCB Preset Selection | DS2 | 96.21% | 96.22% | 96.05% | 96.00% | 96.17% | 0.1158 | 0.9989 |
| RL-Based Client Participation with RL-UCB Preset Selection | global-equal | 95.89% | 96.04% | 95.73% | 95.70% | 95.82% | 0.1333 | 0.9986 |

### Test metrics

| Experiment | Dataset | Accuracy | Precision (Macro) | Recall (Macro) | F1 (Macro) | Weighted F1 | Log Loss | AUC-ROC (Macro OVR) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| No RL, Fixed Preset | DS1 | 98.23% | 98.27% | 98.19% | 98.20% | 98.22% | 0.0598 | 0.9998 |
| No RL, Fixed Preset | DS2 | 97.35% | 97.31% | 97.20% | 97.20% | 97.32% | 0.0785 | 0.9994 |
| No RL, Fixed Preset | global-equal | 97.79% | 97.79% | 97.69% | 97.70% | 97.77% | 0.0692 | 0.9996 |
| Random Preset per Round | DS1 | 97.79% | 97.81% | 97.78% | 97.79% | 97.80% | 0.0817 | 0.9994 |
| Random Preset per Round | DS2 | 98.58% | 98.52% | 98.54% | 98.53% | 98.58% | 0.0585 | 0.9995 |
| Random Preset per Round | global-equal | 98.18% | 98.16% | 98.16% | 98.16% | 98.19% | 0.0701 | 0.9994 |
| Best Static Preset | DS1 | 97.79% | 97.87% | 97.77% | 97.80% | 97.80% | 0.0996 | 0.9990 |
| Best Static Preset | DS2 | 98.01% | 97.91% | 97.97% | 97.93% | 98.01% | 0.0614 | 0.9992 |
| Best Static Preset | global-equal | 97.90% | 97.89% | 97.87% | 97.87% | 97.90% | 0.0805 | 0.9991 |
| RL-UCB Preset Selection | DS1 | 96.90% | 96.99% | 96.88% | 96.92% | 96.91% | 0.1109 | 0.9976 |
| RL-UCB Preset Selection | DS2 | 98.29% | 98.23% | 98.24% | 98.23% | 98.29% | 0.0611 | 0.9992 |
| RL-UCB Preset Selection | global-equal | 97.60% | 97.61% | 97.56% | 97.57% | 97.60% | 0.0860 | 0.9984 |
| Fixed Client Participation with RL-UCB Preset Selection | DS1 | 97.35% | 97.39% | 97.36% | 97.36% | 97.36% | 0.1038 | 0.9988 |
| Fixed Client Participation with RL-UCB Preset Selection | DS2 | 98.39% | 98.31% | 98.36% | 98.34% | 98.39% | 0.0626 | 0.9994 |
| Fixed Client Participation with RL-UCB Preset Selection | global-equal | 97.87% | 97.85% | 97.86% | 97.85% | 97.88% | 0.0832 | 0.9991 |
| RL-Based Client Participation with RL-UCB Preset Selection | DS1 | 95.13% | 95.34% | 95.06% | 95.10% | 95.14% | 0.1529 | 0.9966 |
| RL-Based Client Participation with RL-UCB Preset Selection | DS2 | 97.44% | 97.42% | 97.33% | 97.31% | 97.43% | 0.1012 | 0.9987 |
| RL-Based Client Participation with RL-UCB Preset Selection | global-equal | 96.29% | 96.38% | 96.20% | 96.21% | 96.29% | 0.1271 | 0.9976 |

## Baseline and Backbone Comparison Results

Backbone comparison: ResNet-50, VGG-16, DenseNet-121, EfficientNet, MobileNetV3, ConvNeXt, Swin-Tiny

Backbone Accuracy Summary

| Backbone | DS1 TEST Accuracy | DS2 TEST Accuracy |
| --- | --- | --- |
| VGG-16 | 96.02% | 94.42% |
| DenseNet-121 | 96.46% | 92.39% |
| ResNeXt-50 (32x4d) | 96.02% | 88.83% |
| EfficientNet-B0 | 93.36% | 88.83% |
| MobileNetV3-Large | 92.92% | 88.83% |
| ConvNeXt-Tiny | 96.02% | 92.39% |
| Swin-Tiny | 96.02% | 89.34% |
| ResNet-50 (FedRL-FuseNet full) | 100.00% | 98.01% |

### VGG-16

Quick Summary

| Metric | Value |
| --- | --- |
| DS1 TEST Accuracy | 96.02% |
| DS2 TEST Accuracy | 94.42% |

Extended TEST Metrics

| Dataset | Acc | Balanced Acc | Precision (Macro) | Recall (Macro) | F1 (Macro) |
| --- | --- | --- | --- | --- | --- |
| DS1 TEST | 96.02% | 96.03% | 96.18% | 96.03% | 96.05% |
| DS2 TEST | 94.42% | 94.14% | 94.18% | 94.14% | 93.91% |

Classwise Metrics — DS1 TEST

| Class ID | Class Name | Support | TP | FP | FN |
| --- | --- | --- | --- | --- | --- |
| 0 | glioma | 56 | 54 | 6 | 2 |
| 1 | meningioma | 55 | 52 | 0 | 3 |
| 2 | no tumor | 59 | 56 | 1 | 3 |
| 3 | pituitary | 56 | 55 | 2 | 1 |

Classwise Metrics — DS2 TEST

| Class ID | Class Name | Support | TP | FP | FN |
| --- | --- | --- | --- | --- | --- |
| 0 | glioma | 45 | 45 | 6 | 0 |
| 1 | meningioma | 46 | 37 | 2 | 9 |
| 2 | no tumor | 61 | 60 | 0 | 1 |
| 3 | pituitary | 45 | 44 | 3 | 1 |

### DenseNet-121

Quick Summary

| Metric | Value |
| --- | --- |
| DS1 TEST Accuracy | 96.46% |
| DS2 TEST Accuracy | 92.39% |

Extended TEST Metrics

| Dataset | Acc | Balanced Acc | Precision (Macro) | Recall (Macro) | F1 (Macro) |
| --- | --- | --- | --- | --- | --- |
| DS1 TEST | 96.46% | 96.52% | 96.50% | 96.52% | 96.47% |
| DS2 TEST | 92.39% | 92.09% | 92.63% | 92.09% | 91.83% |

Classwise Metrics — DS1 TEST

| Class ID | Class Name | Support | TP | FP | FN |
| --- | --- | --- | --- | --- | --- |
| 0 | glioma | 56 | 53 | 2 | 3 |
| 1 | meningioma | 55 | 55 | 1 | 0 |
| 2 | no tumor | 59 | 55 | 1 | 4 |
| 3 | pituitary | 56 | 55 | 4 | 1 |

Classwise Metrics — DS2 TEST

| Class ID | Class Name | Support | TP | FP | FN |
| --- | --- | --- | --- | --- | --- |
| 0 | glioma | 45 | 45 | 11 | 0 |
| 1 | meningioma | 46 | 35 | 2 | 11 |
| 2 | no tumor | 61 | 59 | 0 | 2 |
| 3 | pituitary | 45 | 43 | 2 | 2 |

### ResNeXt-50 (32x4d)

Quick Summary

| Metric | Value |
| --- | --- |
| DS1 TEST Accuracy | 96.02% |
| DS2 TEST Accuracy | 88.83% |

Extended TEST Metrics

| Dataset | Acc | Balanced Acc | Precision (Macro) | Recall (Macro) | F1 (Macro) |
| --- | --- | --- | --- | --- | --- |
| DS1 TEST | 96.02% | 96.00% | 96.31% | 96.00% | 96.03% |
| DS2 TEST | 88.83% | 88.31% | 89.91% | 88.31% | 87.30% |

Classwise Metrics — DS1 TEST

| Class ID | Class Name | Support | TP | FP | FN |
| --- | --- | --- | --- | --- | --- |
| 0 | glioma | 56 | 53 | 0 | 3 |
| 1 | meningioma | 55 | 51 | 1 | 4 |
| 2 | no tumor | 59 | 57 | 1 | 2 |
| 3 | pituitary | 56 | 56 | 7 | 0 |

Classwise Metrics — DS2 TEST

| Class ID | Class Name | Support | TP | FP | FN |
| --- | --- | --- | --- | --- | --- |
| 0 | glioma | 45 | 45 | 5 | 0 |
| 1 | meningioma | 46 | 26 | 1 | 20 |
| 2 | no tumor | 61 | 59 | 1 | 2 |
| 3 | pituitary | 45 | 45 | 15 | 0 |

### EfficientNet-B0

Quick Summary

| Metric | Value |
| --- | --- |
| DS1 TEST Accuracy | 93.36% |
| DS2 TEST Accuracy | 88.83% |

Extended TEST Metrics

| Dataset | Acc | Balanced Acc | Precision (Macro) | Recall (Macro) | F1 (Macro) |
| --- | --- | --- | --- | --- | --- |
| DS1 TEST | 93.36% | 93.31% | 93.50% | 93.31% | 93.28% |
| DS2 TEST | 88.83% | 88.29% | 89.06% | 88.29% | 87.54% |

Classwise Metrics — DS1 TEST

| Class ID | Class Name | Support | TP | FP | FN |
| --- | --- | --- | --- | --- | --- |
| 0 | glioma | 56 | 51 | 3 | 5 |
| 1 | meningioma | 55 | 48 | 3 | 7 |
| 2 | no tumor | 59 | 56 | 2 | 3 |
| 3 | pituitary | 56 | 56 | 7 | 0 |

Classwise Metrics — DS2 TEST

| Class ID | Class Name | Support | TP | FP | FN |
| --- | --- | --- | --- | --- | --- |
| 0 | glioma | 45 | 44 | 7 | 1 |
| 1 | meningioma | 46 | 28 | 2 | 18 |
| 2 | no tumor | 61 | 59 | 3 | 2 |
| 3 | pituitary | 45 | 44 | 10 | 1 |

### MobileNetV3-Large

Quick Summary

| Metric | Value |
| --- | --- |
| DS1 TEST Accuracy | 92.92% |
| DS2 TEST Accuracy | 88.83% |

Extended TEST Metrics

| Dataset | Acc | Balanced Acc | Precision (Macro) | Recall (Macro) | F1 (Macro) |
| --- | --- | --- | --- | --- | --- |
| DS1 TEST | 92.92% | 92.88% | 92.87% | 92.88% | 92.85% |
| DS2 TEST | 88.83% | 88.29% | 88.85% | 88.29% | 87.55% |

Classwise Metrics — DS1 TEST

| Class ID | Class Name | Support | TP | FP | FN |
| --- | --- | --- | --- | --- | --- |
| 0 | glioma | 56 | 50 | 4 | 6 |
| 1 | meningioma | 55 | 49 | 5 | 6 |
| 2 | no tumor | 59 | 56 | 3 | 3 |
| 3 | pituitary | 56 | 55 | 4 | 1 |

Classwise Metrics — DS2 TEST

| Class ID | Class Name | Support | TP | FP | FN |
| --- | --- | --- | --- | --- | --- |
| 0 | glioma | 45 | 44 | 12 | 1 |
| 1 | meningioma | 46 | 28 | 3 | 18 |
| 2 | no tumor | 61 | 59 | 2 | 2 |
| 3 | pituitary | 45 | 44 | 5 | 1 |

### ConvNeXt-Tiny

Quick Summary

| Metric | Value |
| --- | --- |
| DS1 TEST Accuracy | 96.02% |
| DS2 TEST Accuracy | 92.39% |

Extended TEST Metrics

| Dataset | Acc | Balanced Acc | Precision (Macro) | Recall (Macro) | F1 (Macro) |
| --- | --- | --- | --- | --- | --- |
| DS1 TEST | 96.02% | 96.01% | 96.17% | 96.01% | 96.01% |
| DS2 TEST | 92.39% | 92.10% | 92.43% | 92.10% | 91.75% |

Classwise Metrics — DS1 TEST

| Class ID | Class Name | Support | TP | FP | FN |
| --- | --- | --- | --- | --- | --- |
| 0 | glioma | 56 | 51 | 0 | 5 |
| 1 | meningioma | 55 | 53 | 2 | 2 |
| 2 | no tumor | 59 | 57 | 3 | 2 |
| 3 | pituitary | 56 | 56 | 4 | 0 |

Classwise Metrics — DS2 TEST

| Class ID | Class Name | Support | TP | FP | FN |
| --- | --- | --- | --- | --- | --- |
| 0 | glioma | 45 | 45 | 9 | 0 |
| 1 | meningioma | 46 | 34 | 2 | 12 |
| 2 | no tumor | 61 | 59 | 1 | 2 |
| 3 | pituitary | 45 | 44 | 3 | 1 |

### Swin-Tiny

Quick Summary

| Metric | Value |
| --- | --- |
| DS1 TEST Accuracy | 96.02% |
| DS2 TEST Accuracy | 89.34% |

Extended TEST Metrics

| Dataset | Acc | Balanced Acc | Precision (Macro) | Recall (Macro) | F1 (Macro) |
| --- | --- | --- | --- | --- | --- |
| DS1 TEST | 96.02% | 95.99% | 95.97% | 95.99% | 95.97% |
| DS2 TEST | 89.34% | 88.98% | 89.32% | 88.98% | 88.19% |

Classwise Metrics — DS1 TEST

| Class ID | Class Name | Support | TP | FP | FN |
| --- | --- | --- | --- | --- | --- |
| 0 | glioma | 56 | 51 | 3 | 5 |
| 1 | meningioma | 55 | 53 | 3 | 2 |
| 2 | no tumor | 59 | 58 | 1 | 1 |
| 3 | pituitary | 56 | 55 | 2 | 1 |

Classwise Metrics — DS2 TEST

| Class ID | Class Name | Support | TP | FP | FN |
| --- | --- | --- | --- | --- | --- |
| 0 | glioma | 45 | 45 | 11 | 0 |
| 1 | meningioma | 46 | 29 | 3 | 17 |
| 2 | no tumor | 61 | 58 | 1 | 3 |
| 3 | pituitary | 45 | 44 | 6 | 1 |

## External Validation Results

Cross-dataset / external test generalization

Quick Summary

| Metric | Value |
| --- | --- |
| DS1 TEST Accuracy | 99.34% |
| DS2 TEST Accuracy | 99.15% |

Extended TEST Metrics (DS1 vs DS2)

| Dataset | Acc | Balanced Acc | Precision (Macro) | Recall (Macro) | F1 (Macro) |
| --- | --- | --- | --- | --- | --- |
| DS1 TEST | 99.34% | 99.29% | 99.31% | 99.29% | 99.30% |
| DS2 TEST | 99.15% | 99.14% | 99.14% | 99.14% | 99.14% |

Paper-Ready Metrics (Validation and Test)

| Setting | Split | Dataset | Accuracy |
| --- | --- | --- | --- |
| FedRL-FuseNet (validation-selected) | VAL | DS1 | 99.05% |
| FedRL-FuseNet (validation-selected) | VAL | DS2 | 99.10% |
| FedRL-FuseNet (validation-selected) | VAL | global-equal | 99.08% |
| FedRL-FuseNet (validation-selected) | TEST | DS1 | 99.34% |
| FedRL-FuseNet (validation-selected) | TEST | DS2 | 99.15% |
| FedRL-FuseNet (validation-selected) | TEST | global-equal | 99.24% |

Classwise Metrics — DS1 TEST

| Class ID | Class Name | Support | TP | FP | FN |
| --- | --- | --- | --- | --- | --- |
| 0 | glioma | 243 | 240 | 2 | 3 |
| 1 | meningioma | 247 | 243 | 3 | 4 |
| 2 | no tumor | 300 | 300 | 1 | 0 |
| 3 | pituitary | 264 | 264 | 1 | 0 |

Classwise Metrics — DS2 TEST

| Class ID | Class Name | Support | TP | FP | FN |
| --- | --- | --- | --- | --- | --- |
| 0 | glioma | 566 | 562 | 8 | 4 |
| 1 | meningioma | 571 | 561 | 7 | 10 |
| 2 | no tumor | 598 | 596 | 0 | 2 |
| 3 | pituitary | 606 | 602 | 5 | 4 |
