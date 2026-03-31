# Transformer Model Comparisons

This repository contains experiments comparing different deep learning model architectures, training configurations, and hyperparameters. I did this as part of a university course project, CMPE 401. Each experiment is self-contained in its own notebook with detailed analysis and benchmarking.

| Experiment | Task | Dataset | Focus |
|---|---|---|---|
| Timeseries Classification | Sequence classification | FordA | Transformer architecture tuning |
| Object Detection | Multi-class detection | VisDrone2019 | YOLOv11 configuration optimization |

## Experiment 1: Timeseries Classification with Transformer

**Notebook:** [timeseries_classification_transformer.ipynb](timeseries_classification_transformer.ipynb)

This note book is built on top of the [Keras example](https://keras.io/examples/timeseries/timeseries_classification_transformer/) by Theodoros Ntakouris. It  benchmarks a Keras Transformer model across multiple configurations on the FordA binary timeseries dataset.

### Baseline Hyperparameters

| Parameter | Value |
|---|---|
| Transformer blocks | 4 |
| Attention head size | 256 |
| Number of heads | 4 |
| Feed-forward dimension | 4 |
| MLP units | [128] |
| Dropout | 0.25 |
| MLP dropout | 0.40 |
| Learning rate | 1e-4 |
| Batch size | 64 |
| Epochs | 150 (early stopping, patience=10) |

### Experiments & Results

I tested three modifications to the baseline:

| # | Modification | Changed Parameter | New Value | Rationale |
|---|---|---|---|---|
| 1 | Reduced Capacity | `num_transformer_blocks` | 2 | Fewer blocks, faster convergence |
| 2 | More Parallel Attention | `head_size` / `num_heads` | 128 / 5 | Increased attention paths, lower projection |
| 3 | Lower Regularization | `dropout` / `mlp_dropout` | 0.10 / 0.20 | Less regularization, faster early learning |

All four configurations (baseline + 3 variants) basically achieved the same test accuracy (~51.6%) and best validation accuracy (~52%). This convergence suggests that the model's performance on this dataset is limited by factors other than just these architectural choices.

## Experiment 2: Object Detection with YOLOv11

**Notebook:** [yolo_comparative_study.ipynb](yolo_comparative_study.ipynb)

A comprehensive comparative study of YOLOv11 variants and hyperparameter configurations on the VisDrone2019 aerial drone footage dataset. The analysis is structured in five parts: baseline training, loss analysis, systematic experimentation, iterative improvement, and cross-version comparison.

### Part I: Baseline Model Training

Establishes baseline with YOLOv11n (nano):

- Parameters: 2.6M
- Epochs: 20
- Batch size: 16
- Image size: 640px
- Training time: ~737s on RTX 4070 Super

### Part II: Loss Analysis & Convergence

Analysis of training dynamics:

- Convergence: Model reaches stable state by around epoch 10
- Overfitting: Slight overfitting detected but acceptable (generalization gap 0.0476)
- Root Cause: Dataset size (3,140 images) is too small relative to model capacity (2.6M parameters)

### Part III: Hyperparameter Experimentation

Three systematic experiments conducted on VisDrone2019 data:

#### Experiment 3A: Model Size Comparison

Testing YOLOv11 variants (nano, small, medium):

| Model | Parameters | mAP₅₀ | mAP₅₀₋₉₅ | Precision | Recall | Train Loss | Time (s) |
|---|---|---|---|---|---|---|---|
| **YOLOv11n** (baseline) | 2.6M | 0.2891 | 0.1580 | 0.3476 | 0.2433 | 3.2030 | 737 |
| **YOLOv11s** | 9.2M | 0.3342 | 0.1902 | 0.3713 | 0.2822 | 2.8795 | 1140 |
| **YOLOv11m** | 20.1M | 0.3650 | 0.2136 | 0.4466 | 0.3042 | 2.7100 | 2348 |

#### Experiment 3B: Image Resolution Impact

Testing resolution on baseline YOLOv11n (smaller batch size of 8 due to memory):

| Resolution | mAP₅₀ | mAP₅₀₋₉₅ | Precision | Recall | Train Loss | Time (s) |
|---|---|---|---|---|---|---|
| 480px | 0.2439 | 0.1254 | 0.3217 | 0.2059 | 3.4243 | 626 |
| 640px | 0.2887 | 0.1569 | 0.3462 | 0.2409 | 3.2423 | 774 |

#### Experiment 3C: Batch Size Effect

Testing batch size on baseline YOLOv11n:

| Batch Size | mAP₅₀ | mAP₅₀₋₉₅ | Precision | Recall | Train Loss | Time (s) |
|---|---|---|---|---|---|---|
| 8 | 0.2887 | 0.1569 | 0.3462 | 0.2409 | 3.2423 | 766 |
| 12 | 0.2899 | 0.1576 | 0.5146 | 0.2433 | 3.2296 | 735 |
| 16 | 0.2891 | 0.1580 | 0.3476 | 0.2433 | 3.2030 | 724 |

#### Part III Summary

| Ranking | Factor | Impact | Recommendation |
|---|---|---|---|
| 1 | Model Size | 76% mAP₅₀ gain | Scale to largest feasible |
| 2 | Resolution | 45% mAP₅₀ gain | Use 640px minimum for small objects |
| 3 | Batch Size | <1% variation | Use 12-16 for stability |

### Part IV: Iterative Improvement with Regularization

Testing dropout regularization on the optimal small-scale configuration (YOLOv11m, 640px, batch 12):

| Metric | Baseline | With Dropout | Difference |
|---|---|---|---|
| **mAP₅₀** | 0.3650 | 0.3647 | -0.03% |
| **mAP₅₀₋₉₅** | 0.2136 | 0.2135 | -0.05% |
| **Precision** | 0.4466 | 0.4588 | +1.22% |
| **Recall** | 0.3042 | 0.3045 | +0.03% |
| **Train Loss** | 2.5009 | 2.4990 | -0.19% |
| **Val Loss** | 2.5485 | 2.5101 | -3.84% |
| **Generalization Gap** | 0.0476 | 0.0111 | -76.5% |

Adding dropout (0.2) produces identical results, confirming that the baseline configuration was already pretty optimal. The model shows no meaningful overfitting, which is probably due to the hyperparameters selected in Part III representing the optimal operating point for this dataset.

### Part V: YOLO Version Comparison

Cross-version benchmark comparing nano models across YOLO generations (all 640px, batch 16, 20 epochs):

| Version | mAP₅₀ | mAP₅₀₋₉₅ | Precision | Recall |
|---|---|---|---|---|
| **YOLOv11n** | 0.2891 | 0.1580 | 0.3476 | 0.2433 |
| **YOLOv10n** | — | — | — | — |
| **YOLOv8n** | — | — | — | — |
| **YOLOv5n** | — | — | — | — |

Unsuprisingly, YOLOv11 is the best choice as the latest generation with improved accuracy-efficiency trade-offs.

## Technical Notes

- **GPU Used**: NVIDIA RTX 4070 Super
- **Training Duration**: around 6 hours for full YOLO comparative study
- **Python Version**: 3.x with TensorFlow/Keras and PyTorch

For local execution, consider using a GPU with at least 8GB VRAM. Training larger models (YOLOv11m) is really slower on CPU-only systems.
