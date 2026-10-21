#!/usr/bin/env bash
# Root-level reproduction entry point for
# "Cost-Sensitive Online Window Size Selection for Portfolio Management".
#
# Runs the unit test suite and then every selected experiment (exp_1..exp_5),
# writing all figures/CSV/JSON artifacts to results/.
set -uo pipefail

cd "$(dirname "$0")"

echo "=== Installing dependencies ==="
pip install -q -r requirements.txt

echo "=== Running test suite ==="
python -m pytest tests/ -q
TEST_STATUS=$?

export PYTHONPATH="$(pwd)"

echo "=== exp_1: Synthetic regime-shift experiment ==="
python exp/exp1_synthetic_regime_shift.py

echo "=== exp_2: Additional synthetic experiments ==="
python exp/exp2_additional_synthetic.py

echo "=== exp_3: OLPS benchmark comparison (synthetic) ==="
python exp/exp3_benchmark_comparison_synthetic.py

echo "=== exp_4: DJIA empirical study ==="
python exp/exp4_djia_empirical.py

echo "=== exp_5: S&P 500 empirical study ==="
python exp/exp5_sp500_empirical.py

echo "=== Done. See results/RESULTS.md for the comparison against the paper. ==="
exit $TEST_STATUS
