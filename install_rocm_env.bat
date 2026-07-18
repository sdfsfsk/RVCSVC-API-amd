@echo off
chcp 65001 >nul
setlocal
title 自动配置原生 AMD ROCm 运行环境
echo ========================================
echo   RVCSVC-API-amd ROCm 一键环境配置脚本
echo ========================================
echo.

cd /d "%~dp0"

if exist "runtime-rocm\python.exe" (
    echo [提示] 检测到 runtime-rocm 已存在 Python，无需重复安装！
    pause
    exit /b 0
)

echo [1/4] 正在下载 Python 3.12.10 Portable 版本...
powershell -Command "Invoke-WebRequest -Uri 'https://www.python.org/ftp/python/3.12.10/python-3.12.10-embed-amd64.zip' -OutFile 'python-embed.zip'"
if not exist "python-embed.zip" (
    echo [错误] 下载失败，请检查网络！
    pause
    exit /b 1
)

echo [2/4] 正在解压 Python 运行环境...
powershell -Command "Expand-Archive -Path 'python-embed.zip' -DestinationPath 'runtime-rocm' -Force"
del python-embed.zip

echo [3/4] 正在配置 Pip 包管理器...
:: 修改 _pth 文件，允许加载 site-packages
powershell -Command "(Get-Content 'runtime-rocm\python312._pth') -replace '#import site', 'import site' | Set-Content 'runtime-rocm\python312._pth'"

powershell -Command "Invoke-WebRequest -Uri 'https://bootstrap.pypa.io/get-pip.py' -OutFile 'runtime-rocm\get-pip.py'"
"runtime-rocm\python.exe" "runtime-rocm\get-pip.py"
del "runtime-rocm\get-pip.py"

echo [4/4] 正在安装 AMD ROCm 7.2.1 与应用依赖（约 4GB，请耐心等待）...
"runtime-rocm\python.exe" -m pip install -r requirements-rocm.txt
if errorlevel 1 (
    echo [错误] 依赖安装失败，请检查网络后重试！
    pause
    exit /b 1
)

echo.
echo [自检] 验证 AMD ROCm GPU...
set "MIOPEN_LOG_LEVEL=3"
"runtime-rocm\python.exe" -c "import torch; assert torch.cuda.is_available(), 'AMD ROCm GPU is unavailable'; print('[ROCm]', torch.__version__, '- HIP', torch.version.hip, '-', torch.cuda.get_device_name(0))"
if errorlevel 1 (
    echo [错误] ROCm GPU 自检失败！
    pause
    exit /b 1
)

echo.
echo ========================================
echo   环境配置完成！现在你可以直接双击启动脚本了。
echo ========================================
pause
