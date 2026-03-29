#!/usr/bin/env bash
# Resume the interrupted benchmark for mistral-large
# Appends results to results/results_0329_1823/

set -euo pipefail

cd "$(dirname "$0")/.."

RESULTS_DIR="results/results_0329_1823"
MODEL="mistral-large"

REMAINING=(
    "output_validator"
    "guardrails"
    "human_in_loop"
    "multi_agent"
)

COMPLETED=0
FAILED=0
TOTAL=$(( ${#REMAINING[@]} + 1 ))  # +1 for all_combined

echo "============================================================"
echo "  Resuming benchmark for $MODEL"
echo "  Results dir: $RESULTS_DIR"
echo "  Remaining:   ${REMAINING[*]} + all_combined"
echo "============================================================"
echo ""

for DEFENSE in "${REMAINING[@]}"; do
    RUN_NUM=$((COMPLETED + FAILED + 1))
    OUTPUT="$RESULTS_DIR/mistral_large_${DEFENSE}.json"
    echo "[$RUN_NUM/$TOTAL] $MODEL | $DEFENSE"

    if python scripts/run_attacks.py --model "$MODEL" --defense "$DEFENSE" --output "$OUTPUT" \
        > "${OUTPUT%.json}.log" 2>&1; then
        COMPLETED=$((COMPLETED + 1))
        echo "         OK -> $OUTPUT"
    else
        FAILED=$((FAILED + 1))
        echo "         FAILED (see ${OUTPUT%.json}.log)"
    fi
    echo ""
done

# All defenses combined
RUN_NUM=$((COMPLETED + FAILED + 1))
OUTPUT="$RESULTS_DIR/mistral_large_all_combined.json"
echo "[$RUN_NUM/$TOTAL] $MODEL | all_combined"

if python scripts/run_attacks.py --model "$MODEL" \
    --defense input_filter output_validator guardrails human_in_loop multi_agent \
    --output "$OUTPUT" \
    > "${OUTPUT%.json}.log" 2>&1; then
    COMPLETED=$((COMPLETED + 1))
    echo "         OK -> $OUTPUT"
else
    FAILED=$((FAILED + 1))
    echo "         FAILED (see ${OUTPUT%.json}.log)"
fi

echo ""
echo "============================================================"
echo "  Resume Complete"
echo "  Completed: $COMPLETED / $TOTAL"
echo "  Failed:    $FAILED / $TOTAL"
echo "============================================================"
