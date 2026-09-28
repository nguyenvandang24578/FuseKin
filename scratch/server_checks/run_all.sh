#!/bin/bash

OUT_DIR="logs/server_checks"
mkdir -p $OUT_DIR
SUMMARY_FILE="$OUT_DIR/summary.md"

echo "# Server Checks Summary" > $SUMMARY_FILE
echo "" >> $SUMMARY_FILE
echo "| Script | Status | Log File |" >> $SUMMARY_FILE
echo "|---|---|---|" >> $SUMMARY_FILE

SCRIPTS=(
    "check_mask_3dpw.py"
    "check_overlay.py"
    "check_roundtrip_6d.py"
    "check_grad_paths.py"
    "check_overfit_ddim.py"
    "check_forward_shapes.py"
    "check_noise_kp2d.py"
)

for script in "${SCRIPTS[@]}"; do
    echo "Running $script..."
    
    if [[ "$script" == "check_grad_paths.py" || "$script" == "check_forward_shapes.py" ]]; then
        python "scratch/server_checks/$script" --cfg config/train_init_mesh.yaml --real_batch
    else
        python "scratch/server_checks/$script" --cfg config/train_init_mesh.yaml
    fi
    
    if [ $? -eq 0 ]; then
        STATUS="✅ PASS"
    else
        STATUS="❌ FAIL"
    fi
    LOG_NAME="${script%.*}.log"
    echo "| $script | $STATUS | $OUT_DIR/$LOG_NAME |" >> $SUMMARY_FILE
done

echo "" >> $SUMMARY_FILE
echo "Done! Check individual logs for details." >> $SUMMARY_FILE
cat $SUMMARY_FILE
