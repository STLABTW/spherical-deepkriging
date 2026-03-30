#!/bin/bash
#
# Install Jupyter dependencies and register a kernel for a conda environment.
#
# Parameters:
#   -k|--kernel_env: Conda environment / kernel name (default: spherical-deepkriging)
#

set -e

KERNEL_ENV="spherical-deepkriging"

while [[ $# -gt 0 ]]; do
    case $1 in
        -k|--kernel_env)
            KERNEL_ENV="$2"
            shift 2
            ;;
        *)
            echo "Invalid option: $1"
            exit 1
            ;;
    esac
done

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${DIR}/../.bin/color_map.sh"

CONDA_BASE=$(conda info --base 2>/dev/null || echo "/opt/conda")
source "${CONDA_BASE}/etc/profile.d/conda.sh"

echo -e "${FG_YELLOW}Activating Conda env '${KERNEL_ENV}' for Jupyter setup...${FG_RESET}"
conda activate "${KERNEL_ENV}"

echo -e "${FG_YELLOW}Installing notebook dependencies (jupyterlab, ipykernel)...${FG_RESET}"
python -m pip install --no-cache-dir jupyterlab ipykernel

echo -e "${FG_YELLOW}Registering Jupyter kernel '${KERNEL_ENV}'...${FG_RESET}"
python -m ipykernel install --user --name "${KERNEL_ENV}" --display-name "Python (${KERNEL_ENV})"

conda deactivate
echo -e "${FG_GREEN}Jupyter kernel setup complete for '${KERNEL_ENV}'.${FG_RESET}"
