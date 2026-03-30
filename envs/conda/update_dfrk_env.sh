#!/bin/bash
#
# Update Conda environment 'dfrk' with latest dependencies
#
# This script updates the dfrk package installation in the conda environment
# by reinstalling it in editable mode with updated dependencies.
#
# Usage:
#   ./update_dfrk_env.sh

set -e

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo "Updating conda environment 'dfrk'..."

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

# Check if environment exists
if ! conda env list | grep -q "^dfrk "; then
    echo "Error: Conda environment 'dfrk' does not exist."
    echo "Please create it first with: make conda-env"
    exit 1
fi

# Activate environment
echo "Activating conda environment 'dfrk'..."
conda activate dfrk

# Change to project root
cd "${PROJECT_ROOT}"

# Upgrade pip first
echo "Upgrading pip..."
pip install --upgrade pip

# Reinstall package in editable mode with jupyter extras (includes matplotlib)
# This will pick up any updated dependencies from pyproject.toml
echo "Reinstalling dfrk package in editable mode with jupyter extras..."
pip install --force-reinstall -e ".[jupyter]"

# Re-register Jupyter kernel (will overwrite if exists)
echo "Re-registering Jupyter kernel..."
python -m ipykernel install --user --name=dfrk --display-name="Python (dfrk)"

# Optionally install JAX if requested (default: yes). Use JAX_GPU=1 for CUDA.
INSTALL_JAX=${INSTALL_JAX:-yes}
JAX_GPU=${JAX_GPU:-0}
if [[ "$INSTALL_JAX" != "no" ]]; then
    if [[ "$JAX_GPU" == "1" ]] || [[ "$JAX_GPU" == "yes" ]]; then
        echo "Installing/updating JAX with CUDA 12 for MRTS (GPU)..."
        pip install --upgrade "jax[cuda12]"
    else
        echo "Installing/updating JAX for MRTS support (CPU)..."
        pip install --upgrade "jax>=0.4.0,<0.5.0" "jaxlib>=0.4.0,<0.5.0"
    fi
else
    echo "Skipping JAX installation (set INSTALL_JAX=no to skip)"
fi

# Rebuild C++ extensions
echo "Rebuilding C++ extensions..."
if [ -f "${PROJECT_ROOT}/scripts/build_cpp_extensions.sh" ]; then
    bash "${PROJECT_ROOT}/scripts/build_cpp_extensions.sh"
else
    echo "Warning: build_cpp_extensions.sh not found. Skipping C++ build."
    echo "You can build manually with: make build-cpp"
fi

echo ""
echo "✅ Conda environment 'dfrk' updated successfully!"
echo ""
echo "The package has been reinstalled with updated dependencies."
echo "To activate the environment, run:"
echo "  conda activate dfrk"
echo ""
