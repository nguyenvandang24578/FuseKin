@echo off
setlocal enabledelayedexpansion

set WANDB_MODE=disabled
set CFG=config\train_init_mesh.yaml
set OUT_DIR=logs\server_checks
if not exist %OUT_DIR% mkdir %OUT_DIR%
set SUMMARY_FILE=%OUT_DIR%\summary.md

echo # Server Checks Summary > %SUMMARY_FILE%
echo. >> %SUMMARY_FILE%
echo ^| # ^| Script ^| Status ^| Log File ^| >> %SUMMARY_FILE%
echo ^|---^|--------^|--------^|---------^| >> %SUMMARY_FILE%

set IDX=0
set PASS_COUNT=0
set FAIL_COUNT=0

REM Order: CPU-only first, then dataset-dependent, then model-dependent
set SCRIPTS=check_roundtrip_6d.py check_noise_kp2d.py check_mask_3dpw.py check_overlay.py check_forward_shapes.py check_grad_paths.py check_overfit_ddim.py

for %%s in (%SCRIPTS%) do (
    set /a IDX+=1
    echo.
    echo ========================================
    echo [!IDX!] Running %%s ...
    echo ========================================

    set EXTRA_ARGS=

    if "%%s"=="check_grad_paths.py" set EXTRA_ARGS=--real_batch
    if "%%s"=="check_forward_shapes.py" set EXTRA_ARGS=--real_batch
    if "%%s"=="check_roundtrip_6d.py" set EXTRA_ARGS=--device cpu
    if "%%s"=="check_noise_kp2d.py" set EXTRA_ARGS=--device cpu

    python scratch\server_checks\%%s --cfg %CFG% --out_dir %OUT_DIR% !EXTRA_ARGS!

    if !ERRORLEVEL! EQU 0 (
        set STATUS=PASS
        set /a PASS_COUNT+=1
    ) else (
        set STATUS=FAIL
        set /a FAIL_COUNT+=1
    )
    echo ^| !IDX! ^| %%s ^| !STATUS! ^| %OUT_DIR%\%%~ns.log ^| >> %SUMMARY_FILE%
)

echo. >> %SUMMARY_FILE%
echo **Total: !PASS_COUNT! PASS, !FAIL_COUNT! FAIL** >> %SUMMARY_FILE%
echo. >> %SUMMARY_FILE%
echo Done! Check individual logs for details. >> %SUMMARY_FILE%

echo.
echo ========================================
echo SUMMARY
echo ========================================
type %SUMMARY_FILE%
