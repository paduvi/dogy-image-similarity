@echo off
cd /d %~dp0
if not exist ".venv" (
    echo Failed to activate virtual environment!
    pause
    exit /b 1
)

:: Activate virtual environment
echo Activating virtual environment...
call .venv\Scripts\activate.bat

:: Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip

:: Install requirements
if exist "requirements.txt" (
    echo Installing requirements from requirements.txt...
    pip install -r requirements.txt
    if %errorlevel% neq 0 (
        echo Failed to install some requirements!
        pause
        exit /b 1
    )
) else (
    echo requirements.txt not found in current directory!
    pause
    exit /b 1
)

echo.
echo Setup completed successfully!