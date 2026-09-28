#!/bin/bash
# run_all.sh — Run all server checks in order.
# Must be run from the repo root: bash scratch/server_checks/run_all.sh
#
# IMPORTANT: this file MUST use LF line endings (not CRLF).

set -o pipefail

export WANDB_MODE=disabled
export PYTHONPATH="./lib:$PYTHONPATH"

CFG="config/train_init_mesh.yaml"
OUT_DIR="${1:-logs/server_checks}"
mkdir -p "$OUT_DIR"

SUMMARY="$OUT_DIR/summary.md"
echo "# Server Checks Summary" > "$SUMMARY"
echo "" >> "$SUMMARY"
echo "| # | Script | Status | Time (s) | Log |" >> "$SUMMARY"
echo "|---|--------|--------|----------|-----|" >> "$SUMMARY"

# Ordered: pure-CPU first, then dataset-dependent, then model-dependent
SCRIPTS=(
    "check_roundtrip_6d.py"
    "check_noise_kp2d.py"
    "check_mask_3dpw.py"
    "check_overlay.py"
    "check_forward_shapes.py"
    "check_grad_paths.py"
    "check_overfit_ddim.py"
)

IDX=0
PASS_COUNT=0
FAIL_COUNT=0

for script in "${SCRIPTS[@]}"; do
    IDX=$((IDX + 1))
    echo ""
    echo "========================================"
    echo "[$IDX/${#SCRIPTS[@]}] Running $script ..."
    echo "========================================"

    EXTRA_ARGS=""
    # Scripts that need --real_batch
    if [[ "$script" == "check_grad_paths.py" || "$script" == "check_forward_shapes.py" ]]; then
        EXTRA_ARGS="--real_batch"
    fi
    # Scripts that can run on CPU
    if [[ "$script" == "check_roundtrip_6d.py" || "$script" == "check_noise_kp2d.py" ]]; then
        EXTRA_ARGS="$EXTRA_ARGS --device cpu"
    fi

    START=$(date +%s)
    python "scratch/server_checks/$script" --cfg "$CFG" --out_dir "$OUT_DIR" $EXTRA_ARGS
    EXIT_CODE=$?
    END=$(date +%s)
    ELAPSED=$((END - START))

    if [ $EXIT_CODE -eq 0 ]; then
        STATUS="✅ PASS"
        PASS_COUNT=$((PASS_COUNT + 1))
    else
        STATUS="❌ FAIL"
        FAIL_COUNT=$((FAIL_COUNT + 1))
    fi

    LOG_NAME="${script%.*}.log"
    echo "| $IDX | $script | $STATUS | ${ELAPSED}s | $OUT_DIR/$LOG_NAME |" >> "$SUMMARY"
done

echo "" >> "$SUMMARY"
echo "**Total: $PASS_COUNT PASS, $FAIL_COUNT FAIL out of ${#SCRIPTS[@]}**" >> "$SUMMARY"
echo "" >> "$SUMMARY"
echo "Generated at: $(date -Iseconds)" >> "$SUMMARY"

echo ""
echo "========================================"
echo "SUMMARY"
echo "========================================"
cat "$SUMMARY"
