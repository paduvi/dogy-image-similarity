#!/bin/bash

# Change to the directory where the script is located
cd "$(dirname "$0")"

# Optional: confirm where we are
echo "Current working dir: $(pwd)"
echo "Starting Python virtual environment setup..."
echo

# Detect operating system
OS=$(uname -s)
if [[ "$OS" == "Darwin" ]]; then
    IS_MACOS=true
    echo "Detected macOS"
else
    IS_MACOS=false
    echo "Detected Linux"
fi

# Parse command line arguments for CUDA version
CUDA_VERSION=""
for arg in "$@"; do
    case $arg in
        cuda=11.8)
            if [[ "$IS_MACOS" == true ]]; then
                echo "Warning: CUDA is not supported on macOS. Ignoring CUDA argument."
            else
                CUDA_VERSION="11.8"
            fi
            ;;
        cuda=12.6)
            if [[ "$IS_MACOS" == true ]]; then
                echo "Warning: CUDA is not supported on macOS. Ignoring CUDA argument."
            else
                CUDA_VERSION="12.6"
            fi
            ;;
        cuda=12.8)
            if [[ "$IS_MACOS" == true ]]; then
                echo "Warning: CUDA is not supported on macOS. Ignoring CUDA argument."
            else
                CUDA_VERSION="12.8"
            fi
            ;;
    esac
done

# Check if Python 3.12 is available
if ! command -v python3 &> /dev/null; then
    echo "Python 3 is not installed or not in PATH!"
    if [[ "$IS_MACOS" == true ]]; then
        echo "Please install Python 3.12 using Homebrew, official installer, or your preferred method."
    else
        echo "Please install Python 3.12 using your distribution's package manager or from the official Python website."
    fi
    exit 1
fi

# Check Python version
PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
if [[ "$PYTHON_VERSION" != "3.12" ]]; then
    echo "Warning: Python version is $PYTHON_VERSION, but 3.12 is recommended."
fi

echo "Setting up virtual environment..."

# Remove existing .venv if it exists
if [ -d ".venv" ]; then
    echo "Removing existing .venv directory..."
    rm -rf ".venv"
fi

# Create new virtual environment
python3 -m venv .venv

if [ ! -d ".venv" ]; then
    echo "Failed to create virtual environment!"
    exit 1
fi

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
python -m pip install --upgrade pip

# Create requirements.txt from template
echo "Creating requirements.txt from template..."
if [ -f "requirements.txt.template" ]; then
    if [[ "$IS_MACOS" == true ]]; then
        # macOS: always use CPU version (supports MPS for Apple Silicon)
        cp requirements.txt.template requirements.txt
        echo "requirements.txt created successfully."
    else
        # Linux: check for CUDA version
        if [ "$CUDA_VERSION" = "11.8" ]; then
            echo "--extra-index-url https://download.pytorch.org/whl/cu118" > requirements.txt
            echo "" >> requirements.txt
            cat requirements.txt.template >> requirements.txt
        elif [ "$CUDA_VERSION" = "12.6" ]; then
            echo "--extra-index-url https://download.pytorch.org/whl/cu126" > requirements.txt
            echo "" >> requirements.txt
            cat requirements.txt.template >> requirements.txt
        elif [ "$CUDA_VERSION" = "12.8" ]; then
            echo "--extra-index-url https://download.pytorch.org/whl/cu128" > requirements.txt
            echo "" >> requirements.txt
            cat requirements.txt.template >> requirements.txt
        else
            cp requirements.txt.template requirements.txt
        fi
        echo "requirements.txt created successfully."
    fi
else
    echo "requirements.txt.template not found in current directory!"
    exit 1
fi

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
echo "Virtual environment created and activated."
echo "All requirements have been installed."
if [[ "$IS_MACOS" == true ]]; then
    echo "Note: macOS users will get the CPU version of PyTorch."
elif [ -n "$CUDA_VERSION" ]; then
    echo "CUDA version $CUDA_VERSION PyTorch index URL was added to requirements.txt"
fi
echo
echo "To activate the virtual environment in the future, run:"
echo "  source .venv/bin/activate"
echo
echo "To deactivate the virtual environment, run:"
echo "  deactivate"
echo