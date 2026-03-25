# OASIS Brain MRI Analysis — ML Pipeline for Alzheimer's Detection

Machine learning pipeline for Alzheimer's detection using the OASIS-1 (Open Access Series of Imaging Studies) dataset. Combines tabular clinical data with MRI-derived structural imaging biomarkers to achieve clinically robust classification.

## Key Results

| Scenario | Best Model | Accuracy | AUC | MMSE Dependency |
|----------|-----------|----------|-----|-----------------|
| **Phase 1** (tabular only) | LogReg / AdaBoost | 88.1% | 0.938 | 37.8% |
| **Phase 2** (full enhanced) | XGBoost | **90.5%** | 0.938 | **9.1%** |
| **Phase 2** (no MMSE) | XGBoost | 85.7% | 0.922 | 0% |
| **Phase 2** (imaging only) | XGBoost | 84.5% | 0.920 | 0% |

MMSE dependency reduced from **37.8% → 9.1%** while best accuracy improved to **90.5%**.

## Project Structure

```
MLProject/
├── data/
│   ├── raw/                        # Original OASIS Excel files
│   ├── processed/
│   │   └── oasis1/                 # Phase 1 preprocessed splits
│   └── enhanced_features/          # Phase 2 imaging feature CSVs
│       ├── oasis1_full_enhanced_features.csv   # Final 416×116 dataset
│       ├── full_imaging_manifest.csv
│       ├── full_tissue_features.csv
│       └── full_regional_features.csv
├── models/
│   ├── phase1_oasis1/              # Phase 1 trained models
│   ├── phase2_full/                # Phase 2 models (all features)
│   ├── phase2_no_mmse/             # Phase 2 models (MMSE removed)
│   └── phase2_imaging_only/        # Phase 2 models (imaging only)
├── results/
│   ├── phase1_oasis1/              # Phase 1 plots & evaluation reports
│   └── phase2/                     # Phase 2 comparison CSVs
├── scripts/
│   ├── run_full_oasis1_pipeline.py # Full 12-disc feature extraction
│   ├── train_phase2_enhanced.py    # Phase 2 training + ablation
│   ├── pre_training_audit.py       # Data integrity verification
│   ├── preprocess_oasis1.py        # Phase 1 preprocessing
│   ├── train_all_models.py         # Phase 1 model training
│   ├── evaluate.py                 # Single model evaluation
│   └── evaluate_all_models.py      # Batch evaluation
├── src/
│   ├── models.py                   # 8 ML model definitions
│   ├── preprocessor.py             # OASISPreprocessor (CDR→binary)
│   ├── data_loader.py              # Data loading utilities
│   ├── utils.py                    # Plotting & JSON helpers
│   └── imaging/                    # MRI feature extraction modules
│       ├── tissue_features.py      # GM/WM/CSF volume extraction
│       ├── regional_features.py    # ROI-based regional extraction
│       ├── atlas_utils.py          # Talairach atlas ROI mapping
│       ├── io_utils.py             # Analyze→NIfTI conversion
│       ├── merge_utils.py          # Session-safe merge validation
│       └── qc.py                   # Quality control utilities
├── docs/
│   └── OASIS1_IMAGING_FEATURE_DICTIONARY.md
├── PHASE2_CLINICAL_ACCURACY_REPORT.md  # Full clinical accuracy report
├── oasis1-disc1/ ... oasis1-disc12/    # Raw OASIS-1 imaging data
└── requirements.txt
```

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

### Phase 1: Tabular-Only Baseline
```bash
python scripts/preprocess_oasis1.py
python scripts/train_all_models.py
python scripts/evaluate_all_models.py
```

### Phase 2: Enhanced MRI Features (Full Pipeline)
```bash
# Step 1: Extract imaging features from all 12 OASIS-1 discs
python scripts/run_full_oasis1_pipeline.py

# Step 2: Verify data integrity
python scripts/pre_training_audit.py

# Step 3: Train all models + ablation studies
python scripts/train_phase2_enhanced.py
```

## Models (8 classifiers)

Random Forest, Logistic Regression, SVM, XGBoost, Gradient Boosting, KNN, Naive Bayes, AdaBoost

## Features (116 total)

- **Original clinical (12):** Age, Gender, Education, SES, MMSE, CDR, eTIV, nWBV, ASF
- **Tissue features (57):** GM/WM/CSF volumes, fractions, ratios, voxel counts
- **Regional features (47):** Hippocampus, ventricles, entorhinal, temporal lobe volumes

**Target:** Binary CDR classification (CDR=0 → healthy, CDR>0 → dementia)

## Report

See [PHASE2_CLINICAL_ACCURACY_REPORT.md](PHASE2_CLINICAL_ACCURACY_REPORT.md) for the full clinical accuracy analysis.
