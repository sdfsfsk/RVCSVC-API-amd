@echo off
chcp 65001 >nul
setlocal
title RVC API - AMD (Native ROCm)
cd /d "%~dp0"

set "PYTHONIOENCODING=utf-8"
set "PYTHONUTF8=1"
set "NATIVE_PYTHON=%~dp0runtime-rocm\python.exe"
set "LEGACY_PYTHON=%~dp0Python310\python.exe"
set "GATEWAY_PORT=3333"
set "PATH=%~dp0ffmpeg\bin;%PATH%"

echo [INFO] Checking for an old RVC gateway listener on port %GATEWAY_PORT%...
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -Command "$port = %GATEWAY_PORT%; $listeners = @(Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue); foreach ($listener in $listeners) { $ownerId = [int]$listener.OwningProcess; $proc = Get-CimInstance Win32_Process -Filter ('ProcessId=' + $ownerId) -ErrorAction SilentlyContinue; if ($null -eq $proc) { continue }; if (($proc.Name -ine 'python.exe') -and ($proc.Name -ine 'pythonw.exe')) { Write-Host ('[ERROR] Port ' + $port + ' is occupied by unrelated process PID ' + $ownerId + ' (' + $proc.Name + ').'); exit 21 }; if ($proc.CommandLine -notlike '*app_rvc.py*') { Write-Host ('[ERROR] Port ' + $port + ' is occupied by an unrelated Python process PID ' + $ownerId + '.'); exit 21 }; Write-Host ('[INFO] Stopping old RVC gateway PID ' + $ownerId + '...'); Stop-Process -Id $ownerId -Force -ErrorAction Stop }; $deadline = (Get-Date).AddSeconds(10); while ((Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue) -and ((Get-Date) -lt $deadline)) { Start-Sleep -Milliseconds 250 }; if (Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue) { Write-Host ('[ERROR] Port ' + $port + ' was not released within 10 seconds.'); exit 22 }"
if errorlevel 1 (
    echo [ERROR] Unable to clear port %GATEWAY_PORT%. RVC gateway will not be started.
    pause
    exit /b 1
)

if exist "%NATIVE_PYTHON%" (
    set "MIOPEN_LOG_LEVEL=3"
    echo [RVC] Checking native AMD ROCm runtime...
    "%NATIVE_PYTHON%" -c "import torch; assert torch.cuda.is_available(), 'AMD ROCm GPU is unavailable'; print('[ROCm] PyTorch', torch.__version__, '- HIP', torch.version.hip, '- GPU', torch.cuda.get_device_name(0))"
    if errorlevel 1 (
        echo [ERROR] Native ROCm GPU self-check failed. Startup cancelled.
        pause
        exit /b 4
    )
    echo [RVC] Starting gateway with native AMD ROCm 7.2.1 + PyTorch 2.9.1...
    "%NATIVE_PYTHON%" "%~dp0app_rvc.py" --is_nohalf
) else if exist "%LEGACY_PYTHON%" (
    echo [WARN] Native ROCm runtime not found; falling back to legacy DirectML...
    set "RVC_USE_DML=1"
    "%LEGACY_PYTHON%" "%~dp0app_rvc.py" --dml --is_nohalf
) else (
    echo [ERROR] No Python runtime found.
    echo [ERROR] Run install_rocm_env.bat first.
    pause
    exit /b 2
)

set "RVC_EXIT_CODE=%ERRORLEVEL%"
echo.
echo [INFO] RVC gateway exited with code: %RVC_EXIT_CODE%
pause
exit /b %RVC_EXIT_CODE%
