# dogy-image-similarity

A Python project for image similarity analysis using PyTorch and computer vision libraries.

## Environment Setup

This project requires Python 3.12 and several dependencies including PyTorch, OpenCV, and other computer vision libraries.

### Windows

For Windows users, we provide an automated setup script:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/paduvi/dogy-image-similarity.git
   cd dogy-image-similarity
   ```

2. **Run the installation script:**
   ```bash
   install.bat
   ```

   **For CUDA support (if you have an NVIDIA GPU):**
   ```bash
   install.bat cuda=11.8
   # or
   install.bat cuda=12.6
   # or  
   install.bat cuda=12.8
   ```

3. **Activate the virtual environment:**
   ```bash
   .venv\Scripts\activate.bat
   ```

The script will:
- Check for Python 3.12 installation
- Create a virtual environment
- Install all required dependencies
- Configure PyTorch with CUDA support if specified

### Linux and macOS

For Linux and macOS users, we provide an automated setup script:

1. **Prerequisites:**
   - **Linux:** Install Python 3.12 using your distribution's package manager or from the official Python website
   - **macOS:** Install Python 3.12 using Homebrew, the official Python installer, or your preferred method

2. **Clone the repository:**
   ```bash
   git clone https://github.com/paduvi/dogy-image-similarity.git
   cd dogy-image-similarity
   ```

3. **Make the installation script executable:**
   ```bash
   chmod +x install.sh
   ```

4. **Run the installation script:**
   
   **For CPU-only installation:**
   ```bash
   ./install.sh
   ```

   **For CUDA support (Linux only, if you have an NVIDIA GPU):**
   ```bash
   ./install.sh cuda=11.8
   # or
   ./install.sh cuda=12.6
   # or  
   ./install.sh cuda=12.8
   ```

   Note: CUDA support is only available on Linux. macOS users will automatically get the CPU version of PyTorch with Metal Performance Shaders (MPS) support for Apple Silicon Macs.

5. **The script will automatically activate the virtual environment. For future use:**
   ```bash
   source .venv/bin/activate
   ```

The script will:
- Detect your operating system (Linux or macOS)
- Check for Python 3.12 installation
- Create a virtual environment
- Install all required dependencies
- Configure PyTorch with CUDA support if specified (Linux only)

## Usage

After setting up the environment, activate your virtual environment and run the project:

**Windows:**
```bash
.venv\Scripts\activate.bat
python your_script.py
```

**Linux and macOS:**
```bash
source .venv/bin/activate
python3 your_script.py
```
