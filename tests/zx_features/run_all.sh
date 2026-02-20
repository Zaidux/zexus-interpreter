#!/usr/bin/env bash
# ============================================================================
#  Run all 9 Zexus feature tests
# ============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$(dirname "$SCRIPT_DIR")/.."

TOTAL_PASS=0
TOTAL_FAIL=0
TOTAL_TESTS=0
FAILED_FEATURES=()

echo "================================================================"
echo "  ZEXUS NINE FEATURES — INTEGRATION TEST SUITE"
echo "================================================================"
echo ""

for i in 1 2 3 4 5 6 7 8 9; do
    FILE="tests/zx_features/test_${i}_*.zx"
    FILE=$(ls $FILE 2>/dev/null | head -1)
    if [ -z "$FILE" ]; then
        echo "  [SKIP] Feature $i — file not found"
        continue
    fi

    echo "Running $FILE ..."
    OUTPUT=$(python -m zexus run "$FILE" 2>&1)

    PASS=$(echo "$OUTPUT" | grep -c "\[PASS\]" || true)
    FAIL=$(echo "$OUTPUT" | grep -c "\[FAIL\]" || true)
    TESTS=$((PASS + FAIL))

    TOTAL_PASS=$((TOTAL_PASS + PASS))
    TOTAL_FAIL=$((TOTAL_FAIL + FAIL))
    TOTAL_TESTS=$((TOTAL_TESTS + TESTS))

    if [ "$FAIL" -gt 0 ]; then
        FAILED_FEATURES+=("Feature $i")
        echo "$OUTPUT" | grep "\[FAIL\]"
    fi
    echo ""
done

echo "================================================================"
echo "  GRAND TOTAL"
echo "================================================================"
echo "  Features: 9"
echo "  Tests:    $TOTAL_TESTS"
echo "  Passed:   $TOTAL_PASS"
echo "  Failed:   $TOTAL_FAIL"
echo ""
if [ "$TOTAL_FAIL" -eq 0 ]; then
    echo "  *** ALL FEATURES PASSED! ***"
else
    echo "  Failed features: ${FAILED_FEATURES[*]}"
fi
echo "================================================================"
