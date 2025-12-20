#!/bin/bash
# Quick Start Script for Final Benchmark Evaluation

set -e

echo "=========================================="
echo "FINAL BENCHMARK EVALUATION - QUICK START"
echo "=========================================="
echo ""

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "✓ Python version: $python_version"

# Check dependencies
echo ""
echo "Checking dependencies..."

dependencies=("matplotlib" "seaborn" "numpy")
missing=()

for dep in "${dependencies[@]}"; do
    if python3 -c "import $dep" 2>/dev/null; then
        echo "  ✓ $dep installed"
    else
        echo "  ✗ $dep missing"
        missing+=("$dep")
    fi
done

if [ ${#missing[@]} -gt 0 ]; then
    echo ""
    echo "Installing missing dependencies..."
    pip3 install "${missing[@]}"
fi

echo ""
echo "=========================================="
echo "SETUP COMPLETE"
echo "=========================================="
echo ""
echo "Available commands:"
echo ""
echo "1. Run single test:"
echo "   python run_evaluations.py --test fever_full_system"
echo ""
echo "2. Run all 5 tests (5 days, one per day):"
echo "   python run_evaluations.py --all"
echo ""
echo "3. Calculate metrics (after tests complete):"
echo "   python calculate_metrics.py"
echo ""
echo "4. Generate plots (after metrics):"
echo "   python generate_plots.py"
echo ""
echo "5. Full pipeline (run all):"
echo "   ./run_full_pipeline.sh"
echo ""
echo "=========================================="
echo "TIMELINE:"
echo "=========================================="
echo "Dec 19: Run fever_full_system"
echo "Dec 20: Run fever_no_graphverify"
echo "Dec 21: Run hotpotqa_vector_only"
echo "Dec 22: Run hotpotqa_graph_only"
echo "Dec 23: Run hotpotqa_hybrid"
echo "Dec 24: Calculate metrics + generate plots"
echo "Dec 25-27: Write paper"
echo "Dec 28: SUBMIT!"
echo "=========================================="
