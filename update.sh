#!/bin/bash

# Change to the directory where the script is located
cd "$(dirname "$0")"

# Optional: confirm where we are
echo "Current working dir: $(pwd)"

if [ ! -d ".venv" ]; then
    echo "Failed to activate virtual environment!"
    exit 1
fi

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
python -m pip install --upgrade pip

# Install requirements
if [ -f "requirements.txt" ]; then
    echo "Installing requirements from requirements.txt..."
    pip install -r requirements.txt
    if [ $? -ne 0 ]; then
        echo "Failed to install some requirements!"
        exit 1
    fi
else
    echo "requirements.txt not found in current directory!"
    exit 1
fi

echo
echo "Setup completed successfully!"