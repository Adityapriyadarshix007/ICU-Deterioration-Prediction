#!/bin/bash
# Complete reproducibility script
# Runs all experiments from raw data to final results

echo "========================================="
echo "ICU Deterioration Prediction - Reproducibility"
echo "========================================="

# Set random seeds
export PYTHONHASHSEED=42
export TF_DETERMINISTIC_OPS=1

echo ""
echo "[1/7] Running Phase 1: Data Extraction"
python3 phase1.py

echo ""
echo "[2/7] Running Phase 2: Organ Trajectories"
python3 phase2.py

echo ""
echo "[3/7] Running Phase 3: SOFA Onset Detection"
python3 phase3.py

echo ""
echo "[4/7] Running Phase 4: Feature Engineering"
python3 phase4.py

echo ""
echo "[5/7] Running Phase 5: XGBoost Baseline"
python3 phase5.py

echo ""
echo "[6/7] Running Phase 6: CNN-LSTM + Attention"
python3 phase6.py

echo ""
echo "[7/7] Running Phase 7: Final Evaluation"
python3 phase7.py

echo ""
echo "========================================="
echo "✅ All experiments complete!"
echo "Results saved to outputs/"
echo "========================================="
