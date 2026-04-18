#!/usr/bin/env bash
set -e

if [ "${BMO_ENABLE_RUNTIME_PATCHES:-0}" = "1" ]; then
  echo "[ERROR] BMO_ENABLE_RUNTIME_PATCHES=1 is not allowed in run_regression_suite.sh"
  exit 1
fi

unset BMO_ENABLE_RUNTIME_PATCHES

echo "=== REGRESSION SUITE ==="
python test_commutativity_regression.py
python test_identity_regression.py
echo "--- Running offline audio regression ---"
python test_offline_regression.py
echo "=== ALL REGRESSION TESTS PASSED ==="

