#!/bin/bash
# Full Evaluation Pipeline
# WARNING: This runs all 5 tests sequentially with 60s delays
# Total time: ~30-40 minutes

set -e

echo "=========================================="
echo "RUNNING FULL EVALUATION PIPELINE"
echo "=========================================="
echo ""
echo "This will run 5 tests sequentially:"
echo "  1. FEVER full system (25 samples)"
echo "  2. FEVER no GraphVerify (25 samples)"
echo "  3. HotpotQA vector-only (25 samples)"
echo "  4. HotpotQA graph-only (25 samples)"
echo "  5. HotpotQA hybrid (25 samples)"
echo ""
echo "Total: ~125K tokens on Groq API"
echo "Estimated time: 30-40 minutes"
echo ""
read -p "Continue? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    exit 1
fi

# Step 1: Run evaluations
echo ""
echo "=========================================="
echo "STEP 1: Running Evaluations"
echo "=========================================="
python3 run_evaluations.py --all

# Step 2: Calculate metrics
echo ""
echo "=========================================="
echo "STEP 2: Calculating Metrics"
echo "=========================================="
python3 calculate_metrics.py

# Step 3: Generate plots
echo ""
echo "=========================================="
echo "STEP 3: Generating Plots"
echo "=========================================="
python3 generate_plots.py

echo ""
echo "=========================================="
echo "PIPELINE COMPLETE!"
echo "=========================================="
echo ""
echo "Results:"
echo "  - Raw results: tests/final_benchmark/results/*.json"
echo "  - Metrics: tests/final_benchmark/results/metrics_summary.json"
echo "  - Plots: tests/final_benchmark/plots/*.png"
echo ""
echo "Next steps:"
echo "  1. Review metrics_summary.json"
echo "  2. Check plots/ directory for figures"
echo "  3. Use figures in your paper"
echo "  4. Write results section with actual numbers"
echo ""
echo "Good luck with your submission! 🎉"
echo "=========================================="
