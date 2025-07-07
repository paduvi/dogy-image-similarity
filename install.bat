@echo off

:: Parse arguments (handle key=value or key value)
:parse_args
if "%~1"=="" goto check_python

:: Handle split format: cuda 12.8
if /i "%~1"=="cuda" (
    if not "%~2"=="" (
        set "CUDA_VERSION=%~2"
        shift
    )
)

shift
goto parse_args

:check_python
echo Detected CUDA version: [%CUDA_VERSION%]
echo Checking for Python installation...

where python >nul 2>nul
if errorlevel 1 (
    echo ❌ Python is not installed or not in PATH!
    echo Please install Python 3.12 from https://www.python.org/downloads/
    pause
    exit /b 1
)

for /f "tokens=2 delims= " %%v in ('python --version 2^>nul') do set PY_VERSION=%%v
for /f "tokens=1,2 delims=." %%a in ("%PY_VERSION%") do (
    set MAJOR=%%a
    set MINOR=%%b
)

echo ✅ Python version detected: %PY_VERSION%

if %MAJOR% LSS 3 (
    echo ❌ Python version too old. Please install Python 3.12 or higher.
    pause
    exit /b 1
)

if %MAJOR%==3 if %MINOR% LSS 12 (
    echo ⚠️ You are using Python %PY_VERSION%. It is recommended to use Python 3.12.
)

cd /d "%~dp0"

:setup_venv
echo.
echo Setting up virtual environment...

if exist ".venv" (
    echo Removing existing .venv...
    rmdir /s /q ".venv"
)

python\python.exe -m venv .venv
if not exist ".venv" (
    echo Failed to create virtual environment!
    pause
    exit /b 1
)

call .venv\Scripts\activate.bat

echo Upgrading pip...
python -m pip install --upgrade pip

:: Create requirements.txt
if exist "requirements.txt.template" (
    if "%CUDA_VERSION%"=="11.8" (
        echo --extra-index-url https://download.pytorch.org/whl/cu118 > requirements.txt
        echo. >> requirements.txt
        type requirements.txt.template >> requirements.txt
    ) else if "%CUDA_VERSION%"=="12.6" (
        echo --extra-index-url https://download.pytorch.org/whl/cu126 > requirements.txt
        echo. >> requirements.txt
        type requirements.txt.template >> requirements.txt
    ) else if "%CUDA_VERSION%"=="12.8" (
        echo --extra-index-url https://download.pytorch.org/whl/cu128 > requirements.txt
        echo. >> requirements.txt
        type requirements.txt.template >> requirements.txt
    ) else (
        copy requirements.txt.template requirements.txt
    )
    echo requirements.txt created successfully.
) else (
    echo requirements.txt.template not found!
    pause
    exit /b 1
)

echo Installing requirements...
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo Failed to install requirements!
    pause
    exit /b 1
)

echo.
echo Setup completed successfully!
if not "%CUDA_VERSION%"=="" (
    echo CUDA version %CUDA_VERSION% used.
)
echo Run `.venv\Scripts\activate.bat` to activate the environment.
pause
