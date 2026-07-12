@echo off
chcp 65001 >nul
echo ========================================
echo   RVC API - AMD DirectML 版本
echo ========================================
echo.

set RVC_USE_DML=1

set "PYTHON_EXE=python"
if exist "%~dp0Python310\python.exe" (
    set "PYTHON_EXE=%~dp0Python310\python.exe"
)

echo 使用的 Python: %PYTHON_EXE%
"%PYTHON_EXE%" app_rvc.py --dml --is_nohalf
pause