@echo off
setlocal enabledelayedexpansion

set OUT_DIR=logs\server_checks
if not exist %OUT_DIR% mkdir %OUT_DIR%
set SUMMARY_FILE=%OUT_DIR%\summary.md

echo # Server Checks Summary > %SUMMARY_FILE%
echo. >> %SUMMARY_FILE%
echo ^| Script ^| Status ^| Log File ^| >> %SUMMARY_FILE%
echo ^|---^|---^|---^| >> %SUMMARY_FILE%

set SCRIPTS=check_mask_3dpw.py check_overlay.py check_roundtrip_6d.py check_grad_paths.py check_overfit_ddim.py check_forward_shapes.py check_noise_kp2d.py

for %%s in (%SCRIPTS%) do (
    echo Running %%s...
    if "%%s"=="check_grad_paths.py" (
        python scratch\server_checks\%%s --cfg config\train_init_mesh.yaml --real_batch
    ) else if "%%s"=="check_forward_shapes.py" (
        python scratch\server_checks\%%s --cfg config\train_init_mesh.yaml --real_batch
    ) else (
        python scratch\server_checks\%%s --cfg config\train_init_mesh.yaml
    )
    if !ERRORLEVEL! EQU 0 (
        set STATUS=PASS
    ) else (
        set STATUS=FAIL
    )
    echo ^| %%s ^| !STATUS! ^| %OUT_DIR%\%%~ns.log ^| >> %SUMMARY_FILE%
)

echo. >> %SUMMARY_FILE%
echo Done! Check individual logs for details. >> %SUMMARY_FILE%
type %SUMMARY_FILE%
