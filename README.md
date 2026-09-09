# VIGIL — Visual Inspection & Guided Intelligence Layer

An autonomous industrial visual-inspection system for anomaly detection on the **MVTec AD 2 Vial** dataset.

## Project Structure

```
VIGIL/
├── configs/                 # YAML configuration files
│   └── default.yaml
├── data/                    # Symlinks or references (do NOT store large data here)
├── dataset/                 # MVTec AD 2 Vial dataset (gitignored)
├── experiments/             # Experiment outputs, checkpoints, logs
├── notebooks/               # Jupyter notebooks for exploration
├── reports/
│   └── dataset_inspection/  # Phase 1 reports and visualizations
├── src/
│   ├── data/                # Dataset loading and discovery
│   ├── preprocessing/       # Image preprocessing pipelines
│   ├── models/              # Anomaly detection model implementations
│   ├── evaluation/          # Metrics and evaluation utilities
│   └── visualization/       # Plotting and display utilities
├── tests/                   # Unit and integration tests
├── requirements.txt
└── README.md
```

## Dataset

The MVTec AD 2 Vial dataset contains:
- **1400x1900** grayscale PNG images of pharmaceutical vials
- **Training**: 291 normal (defect-free) images
- **Validation**: 41 normal images
- **Test (public)**: 35 normal + 105 anomalous images with pixel-level ground-truth masks
- **Domain-shift variants**: regular, overexposed, underexposed, shift_1-4
- **Anomaly masks**: Binary (0/255), same resolution as images

## Setup

```bash
pip install -r requirements.txt
```

Set the dataset path (if not in default location):
```bash
export VIGIL_DATA_ROOT=/path/to/dataset
```

## Phase 1 — Dataset Investigation

Run the dataset inspection:
```bash
python reports/dataset_inspection/inspect_dataset.py
```

## Problem Formulation

This project treats the task as **unsupervised anomaly detection with pixel-level localization**:
- Training uses only normal/defect-free samples
- Previously unseen defect types may appear at test time
- Both image-level detection (is this vial defective?) and pixel-level segmentation (where is the defect?) are required

## License

Dataset: MVTec AD 2 — see [MVTec website](https://www.mvtec.com/company/research/datasets/mvtec-ad-2) for terms.
# Pair Extraordinaire Test
