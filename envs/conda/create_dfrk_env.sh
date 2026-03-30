#!/bin/bash
#
# Create and setup Conda environment for dfrk
#
# This script creates a conda environment named 'dfrk' and installs the dfrk package.
#
# Usage:
#   ./create_dfrk_env.sh
#

set -e

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
ENV_FILE="${SCRIPT_DIR}/environment_dfrk.yml"

echo "Creating conda environment 'dfrk'..."

# Initialize conda
# Try multiple methods to find conda
CONDA_BASE=""

# Method 1: Try conda info --base (works if conda is already initialized)
if command -v conda &> /dev/null || type conda &> /dev/null 2>&1; then
    CONDA_BASE=$(conda info --base 2>/dev/null || echo "")
fi

# Method 2: Try common conda installation paths
if [ -z "$CONDA_BASE" ] || [ ! -d "$CONDA_BASE" ]; then
    for path in "${HOME}/miniconda3" "${HOME}/anaconda3" "${HOME}/.conda" "/opt/conda" "/opt/miniconda3" "/opt/anaconda3"; do
        if [ -d "$path" ] && [ -f "${path}/etc/profile.d/conda.sh" ]; then
            CONDA_BASE="$path"
            break
        fi
    done
fi

# Method 3: Try to find conda from CONDA_PREFIX (when environment is already activated)
if [ -z "$CONDA_BASE" ] || [ ! -d "$CONDA_BASE" ]; then
    if [ -n "$CONDA_PREFIX" ]; then
        # CONDA_PREFIX points to the environment (e.g., /path/to/conda/envs/dfrk)
        # Base is typically two levels up: /path/to/conda
        POSSIBLE_BASE=$(dirname "$(dirname "$CONDA_PREFIX" 2>/dev/null)" 2>/dev/null || echo "")
        if [ -n "$POSSIBLE_BASE" ] && [ -f "${POSSIBLE_BASE}/etc/profile.d/conda.sh" ]; then
            CONDA_BASE="$POSSIBLE_BASE"
        fi
    fi
fi

if [ -z "$CONDA_BASE" ] || [ ! -d "$CONDA_BASE" ]; then
    echo "Error: Conda not found. Please install conda/miniconda first."
    echo "Or ensure conda is in your PATH and properly initialized."
    exit 1
fi

# Source conda.sh to initialize conda in this script
if [ -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]; then
    source "${CONDA_BASE}/etc/profile.d/conda.sh"
else
    echo "Error: Conda initialization script not found at ${CONDA_BASE}/etc/profile.d/conda.sh"
    exit 1
fi

# Remove existing environment if it exists
if conda env list | grep -q "^dfrk "; then
    echo "Environment 'dfrk' already exists. Removing it first..."
    conda env remove -n dfrk -y
fi

# Create environment from yml file
echo "Creating environment from ${ENV_FILE}..."
conda env create -f "${ENV_FILE}"

# Activate environment
conda activate dfrk

# Install dfrk package in editable mode
echo "Installing dfrk package in editable mode..."
cd "${PROJECT_ROOT}"

# Check for pip lock files that might cause hanging
PIP_CACHE_DIR="${HOME}/.cache/pip"
if [ -f "${PIP_CACHE_DIR}/.lock" ] || [ -f "/tmp/pip-*.lock" ] 2>/dev/null; then
    echo "Warning: Found pip lock files. Cleaning up..."
    rm -f "${PIP_CACHE_DIR}/.lock" /tmp/pip-*.lock 2>/dev/null || true
fi

# Check if another pip process is running
if pgrep -f "pip install" > /dev/null; then
    echo "Warning: Another pip process appears to be running. Waiting 5 seconds..."
    sleep 5
fi

# Use pip with optimizations to avoid hanging
# Most dependencies are already installed via conda, so this should be quick
echo "Running: pip install -e ."
echo "This creates an editable link to the package (no compilation needed)..."

# Set pip to use shorter timeouts and disable cache to avoid hanging
export PIP_DEFAULT_TIMEOUT=60
export PIP_NO_CACHE_DIR=yes

# Try installation - if it hangs, user can Ctrl+C and we'll provide alternative
echo "Starting installation..."
echo "Note: If this hangs for more than 2 minutes, press Ctrl+C and install manually"
echo ""

# Run pip install with no cache and explicit timeout
if pip install --no-cache-dir --default-timeout=60 -e .; then
    echo "✅ Package installed successfully!"
else
    EXIT_CODE=$?
    echo ""
    echo "⚠️  Installation failed or was interrupted."
    echo ""
    echo "You can try installing manually:"
    echo "  conda activate dfrk"
    echo "  cd ${PROJECT_ROOT}"
    echo "  pip install -e ."
    echo ""
    exit $EXIT_CODE
fi

# Install Jupyter kernel
echo "Registering Jupyter kernel..."
python -m ipykernel install --user --name=dfrk --display-name="Python (dfrk)"

# Install optional dependencies if needed. Set JAX_GPU=1 for CUDA 12.
INSTALL_JAX=${INSTALL_JAX:-yes}
JAX_GPU=${JAX_GPU:-0}
if [[ "$INSTALL_JAX" != "no" ]]; then
    if [[ "$JAX_GPU" == "1" ]] || [[ "$JAX_GPU" == "yes" ]]; then
        echo "Installing JAX with CUDA 12 for MRTS (GPU)..."
        pip install "jax[cuda12]"
    else
        echo "Installing JAX for MRTS support (CPU)..."
        pip install "jax>=0.4.0" "jaxlib>=0.4.0"
    fi
else
    echo "Skipping JAX installation (set INSTALL_JAX=yes to install)"
fi

# Build C++ extensions
echo "Building C++ extensions..."
if [ -f "${PROJECT_ROOT}/scripts/build_cpp_extensions.sh" ]; then
    bash "${PROJECT_ROOT}/scripts/build_cpp_extensions.sh"
else
    echo "Warning: build_cpp_extensions.sh not found. Skipping C++ build."
    echo "You can build manually with: make build-cpp"
fi

echo ""
echo "✅ Conda environment 'dfrk' created successfully!"
echo ""
echo "To activate the environment, run:"
echo "  conda activate dfrk"
echo ""
echo "To start Jupyter Lab, run:"
echo "  conda activate dfrk"
echo "  jupyter lab"
echo ""
echo "Or use the helper script:"
echo "  ./envs/jupyter/start_jupyter_lab.sh -k dfrk"
