#!/bin/bash

# OASIS-1 Complete Pipeline
# This script runs the full Phase 1 + Phase 2 pipeline:
# 1. Phase 1: Preprocess tabular data, train baseline models, evaluate
# 2. Phase 2: Extract MRI features, train enhanced models + ablation

set -e  # Exit on error

echo "========================================================================"
echo "OASIS-1 Complete Pipeline (Phase 1 + Phase 2)"
echo "========================================================================"

# Activate virtual environment
source venv/bin/activate

echo ""
echo "========================================================================"
echo "PHASE 1: TABULAR BASELINE"
echo "========================================================================"

echo ""
echo "[1/3] Preprocessing OASIS-1..."
python scripts/preprocess_oasis1.py

echo ""
echo "[2/3] Training all 8 models..."
python scripts/train_all_models.py

echo ""
echo "[3/3] Evaluating all models..."
python scripts/evaluate_all_models.py

echo ""
echo "========================================================================"
echo "PHASE 2: MRI-ENHANCED FEATURES"
echo "========================================================================"

echo ""
echo "[1/3] Extracting imaging features from all 12 discs..."
python scripts/run_full_oasis1_pipeline.py

echo ""
echo "[2/3] Verifying data integrity..."
python scripts/pre_training_audit.py

echo ""
echo "[3/3] Training enhanced models + ablation studies..."
python scripts/train_phase2_enhanced.py

echo ""
echo "========================================================================"
echo "✓ ALL PIPELINES COMPLETED SUCCESSFULLY!"
echo "========================================================================"
echo ""
echo "Results locations:"
echo "  - Phase 1 models:  models/phase1_oasis1/"
echo "  - Phase 1 results: results/phase1_oasis1/"
echo "  - Phase 2 models:  models/phase2_full/, phase2_no_mmse/, phase2_imaging_only/"
echo "  - Phase 2 results: results/phase2/"
echo ""
