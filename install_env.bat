@echo off
chcp 65001 >nul
title 自动配置 Python 运行环境
echo ========================================
echo      RVCSVC-API 一键环境配置脚本
echo ========================================
echo.

cd /d "%~dp0"

if exist "Python310\python.exe" (
    echo [提示] 检测到 Python310 文件夹已存在 Python，无需重复安装！
    pause
    exit /b 0
)

echo [1/4] 正在下载 Python 3.10.11 Portable 版本...
powershell -Command "Invoke-WebRequest -Uri 'https://www.python.org/ftp/python/3.10.11/python-3.10.11-embed-amd64.zip' -OutFile 'python-embed.zip'"
if not exist "python-embed.zip" (
    echo [错误] 下载失败，请检查网络！
    pause
    exit /b 1
)

echo [2/4] 正在解压 Python 运行环境...
powershell -Command "Expand-Archive -Path 'python-embed.zip' -DestinationPath 'Python310' -Force"
del python-embed.zip

echo [3/4] 正在配置 Pip 包管理器...
:: 修改 _pth 文件，允许加载 site-packages
powershell -Command "(Get-Content 'Python310\python310._pth') -replace '#import site', 'import site' | Set-Content 'Python310\python310._pth'"

powershell -Command "Invoke-WebRequest -Uri 'https://bootstrap.pypa.io/get-pip.py' -OutFile 'Python310\get-pip.py'"
"Python310\python.exe" "Python310\get-pip.py"
del "Python310\get-pip.py"

echo [4/4] 正在安装 requirements.txt 依赖...
"Python310\python.exe" -m pip install -r requirements.txt

echo.
echo ========================================
echo   环境配置完成！现在你可以直接双击启动脚本了。
echo ========================================
pause
