#!/bin/bash
#
# Helper functions for building a conda environment
#
CONDA_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${CONDA_DIR}/conda_env_info.sh"
source "${COLOR_MAP_PATH}"
source "${EXIT_CODE_PATH}"


initialize_conda() {
  local CONDA_BASE=$(conda info --base)
  local CONDA_DIRS=(
    "$CONDA_BASE"
    "/opt/conda"
    "/opt/miniconda"
    "/opt/miniconda2"
    "/opt/miniconda3"
  )

  local IS_CONDA_FOUND=false
  for dir in "${CONDA_DIRS[@]}"; do
    if [ -f "$dir/etc/profile.d/conda.sh" ]; then
      CONDA_BASE="$dir"
      IS_CONDA_FOUND=true
      break
    fi
  done

  if ! $IS_CONDA_FOUND; then
    echo -e "${FG_RED}No Conda environment found matching${FG_RESET}"  >&2
    return ${ERROR_EXITCODE}
  fi

  echo -e "${FG_YELLOW}Intializing conda${FG_RESET}"
  source "$CONDA_BASE/etc/profile.d/conda.sh"
}


find_conda_env_path() {
  # Will return `CONDA_ENV_DIR`
  local ENV_NAME=$1

  initialize_conda
  IFS=' ' read -r -a CONDA_INFO <<<"$(conda env list | grep "${ENV_NAME}")"

  if [ ${#CONDA_INFO[@]} -eq 0 ]; then
    echo -e "${FG_RED}No Conda environment found matching '${ENV_NAME}'${FG_RESET}"
    return "${ERROR_EXITCODE}"
  fi

  AVAILABLE_ENV_NAME="${CONDA_INFO[0]}"

  if [[ "x${AVAILABLE_ENV_NAME}x" != "x${ENV_NAME}x" ]]; then
    echo -e "${FG_RED}Conda Env '${ENV_NAME}' is not available${FG_RESET}"
    return "${ERROR_EXITCODE}"
  fi

  if [ "x${CONDA_INFO[1]}x" == "x*x" ]; then
    CONDA_ENV_DIR="${CONDA_INFO[2]}"
  else
    CONDA_ENV_DIR="${CONDA_INFO[1]}"
  fi
  # DO NOT REMOVE
  echo "${CONDA_ENV_DIR}"
}


initialize_conda_env() {
  local CONDA_ENV=$1
  local PYTHON_VERSION=$2

  conda create -c conda-forge -n "${CONDA_ENV}" "python=${PYTHON_VERSION}" -y
  source activate "${CONDA_ENV}"
  pip install --no-cache-dir "poetry==${POETRY_VERSION}"
  conda deactivate
}

retry_to_find_conda_env_path() {
  local CONDA_ENV=$1
  local PYTHON_VERSION=$2

  if [ "x${PYTHON_VERSION}x" == "xx" ]; then
    PYTHON_VERSION="${DEFAULT_PYTHON_VERSION}"
  fi

  initialize_conda_env "${CONDA_ENV}" "${PYTHON_VERSION}"

  find_conda_env_path "${CONDA_ENV}"
  if [ "$?" == "${ERROR_EXITCODE}" ]; then
    echo -e "${FG_RED}Unknown exception occurs from the side of Conda infra${FG_RESET}"
  fi

}

install_python_package() {
  local TARGET_PROJECT_DIR=$1

  pushd "${TARGET_PROJECT_DIR}" || exit

  if [ -d "${PWD}"/dist/ ]; then
    FILE_COUNT=$(ls "${PWD}/dist/*" 2>/dev/null | wc -l)
    if [ "x${FILE_COUNT//[[:space:]]/}x" != "x0x" ]; then
      echo -e "${FG_YELLOW}Removing ${PWD}/dist/* files${FG_RESET}"
      rm "${PWD}/dist/*"
    fi
  fi

  echo -e "${FG_YELLOW}Installing python package${FG_RESET}"
  # Check if README.md exists (pyproject.toml references it via `readme = "README.md"`)
  if [ ! -f README.md ]; then
    echo "Temporary README.md" > README.md
    TEMP_README=true
  else
    TEMP_README=false
  fi

  # Install using pip/setuptools (avoid Poetry because pyproject.toml has no [tool.poetry])
  python -m pip install --no-cache-dir .

  # Remove the temporary README.md if it was created
  if [ "$TEMP_README" = true ]; then
    rm README.md
  fi

  popd || exit
}


activate_conda_environment() {
  local CONDA_ENV=$1
  initialize_conda
  if [ "$(command -v conda)" ]; then
    echo -e "${FG_YELLOW}Activating Conda Env: ${CONDA_ENV}${FG_RESET}"
    conda activate ${CONDA_ENV}
  else
    echo -e "${FG_RED}Activation Failed. Conda is not installed.${FG_RESET}"
  fi
}

update_conda_environment() {
  local PACKAGE_BASE_PATH=$1
  local CONDA_ENV=$2

  if [ "$(command -v conda)" ]; then
    echo -e "${FG_YELLOW}Updating Conda environment - ${CONDA_ENV}${FG_RESET}"
    bash "${CONDA_DIR}/build_conda_env.sh" --conda_env ${CONDA_ENV}
  else
    echo -e "${FG_RED}Update Failed. Conda is not installed.${FG_RESET}"
  fi
}

install_mac_packages() {
    echo "Installing system packages for macOS..."
    brew update
    brew install armadillo pybind11 cmake || {
        echo -e "${FG_RED}Failed to install system packages via Homebrew.${FG_RESET}"
        exit 1
    }
}

install_linux_packages() {
    echo -e "${FG_YELLOW}Installing system packages for Linux...${FG_RESET}"

    if [[ -f /etc/debian_version ]]; then
        sudo apt update
        sudo apt install -y libarmadillo-dev pybind11-dev cmake build-essential \
                            libxrender1 libsm6 libxext6 || { \
            echo -e "${FG_RED}Failed to install system packages. Ensure you have sudo privileges.${FG_RESET}"
            exit 1
        }
    elif [[ -f /etc/redhat-release ]]; then
        sudo yum install -y armadillo-devel pybind11-devel cmake gcc gcc-c++ \
                            libXrender libSM libXext || { \
            echo -e "${FG_RED}Failed to install system packages. Ensure you have sudo privileges.${FG_RESET}"
            exit 1
        }
    else
        echo -e "${FG_RED}Unsupported Linux distribution. Please install Armadillo, Pybind11, CMake, OpenBLAS, LAPACK, and X11 libraries manually.${FG_RESET}"
        exit 1
    fi

    echo -e "${FG_GREEN}✅ System packages installed successfully.${FG_RESET}"
}

# Function to install dependencies using Conda
install_conda_packages() {
    initialize_conda
    echo -e "${FG_YELLOW}🚀 Installing dependencies via Conda (with acceleration)...${FG_RESET}"

    if ! command -v conda &> /dev/null; then
        echo -e "${FG_RED}❌ Conda is not installed. Please install Miniconda or Anaconda first.${FG_RESET}"
        exit 1
    fi

    # Ensure conda-forge has strict priority
    conda config --add channels conda-forge
    conda config --set channel_priority strict

    # Install mamba if available
    if ! command -v mamba &> /dev/null; then
        echo -e "${FG_YELLOW}Installing mamba for faster dependency resolution...${FG_RESET}"
        conda install -n base -c conda-forge mamba -y || {
            echo -e "${FG_RED}⚠️ Failed to install mamba. Falling back to conda...${FG_RESET}"
        }
    fi

    INSTALL_CMD="conda install"
    if command -v mamba &> /dev/null; then
        INSTALL_CMD="mamba install"
    fi

    # Packages to install
    DEPS=(
        python=${DEFAULT_PYTHON_VERSION}
        cmake
        pybind11
        boost-cpp
        armadillo
        openblas
        gfortran_linux-64
        libgfortran5
        eigen
        "libblas=*=*openblas"
        "liblapack=*=*openblas"
    )

    echo -e "${FG_YELLOW}🔧 Installing: ${DEPS[*]}${FG_RESET}"
    $INSTALL_CMD -y -c conda-forge "${DEPS[@]}" || {
        echo -e "${FG_RED}❌ Failed to install packages with ${INSTALL_CMD}.${FG_RESET}"
        exit 1
    }

    # Environment variable setup
    export PKG_CONFIG_PATH="$CONDA_PREFIX/lib/pkgconfig:$PKG_CONFIG_PATH"
    export OpenBLAS_ROOT="$CONDA_PREFIX"
    export CMAKE_PREFIX_PATH="$CONDA_PREFIX"
    export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

    echo -e "${FG_GREEN}✅ Conda environment and dependencies installed successfully using ${INSTALL_CMD}.${FG_RESET}"
}


build_cpp() {
    echo -e "${FG_YELLOW}🔹 Building C++ extensions in the evn $CONDA_PREFIX ...${FG_RESET}"

    # Ensure OpenBLAS is correctly exposed to CMake and linker
    export PKG_CONFIG_PATH="$CONDA_PREFIX/lib/pkgconfig:$PKG_CONFIG_PATH"
    export OpenBLAS_ROOT="$CONDA_PREFIX"
    export CMAKE_PREFIX_PATH="$CONDA_PREFIX"
    export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

    # Navigate to the spherical basis C++ extension directory
    cd ${CONDA_DIR}/../../spherical_deepkriging/basis_functions/mrts_sphere/cpp_extensions || {
        echo -e "${FG_RED}❌ Error: C++ extensions directory not found.${FG_RESET}"
        exit 1
    }

    # Remove old build files
    echo "🧹 Cleaning old build files..."
    rm -rf build CMakeCache.txt CMakeFiles spherical_basis*.so cmake_install.cmake Makefile

    # Create a new build directory
    mkdir -p build && cd build

    # Run CMake (source is one directory up; build dir is current)
    cmake \
      -DCMAKE_BUILD_TYPE=Release \
      -DPython3_EXECUTABLE="$CONDA_PREFIX/bin/python" \
      ..

    # Compile using all available cores
    echo "⚙️  Compiling C++ extensions..."
    make -j$(nproc)

    # Verify shared library exists
    echo "🔍 Verifying compiled spherical_basis shared library..."
    if ! ls spherical_basis*.so >/dev/null 2>&1; then
        echo -e "${FG_RED}❌ Error: spherical_basis.so not built successfully.${FG_RESET}"
        exit 1
    fi

    # Return to the original working directory
    cd - || exit

    echo -e "${FG_GREEN}🎉 C++ extension build complete.${FG_RESET}"
}

fix_lapack_blas_linking() {
    echo -e "${FG_YELLOW}Verifying BLAS/LAPACK linking...${FG_RESET}"

    if ! ldconfig -p | grep -q "liblapack.so"; then
        echo -e "${FG_RED}❌ LAPACK is missing! Installing...${FG_RESET}"
        sudo apt install -y liblapack-dev libopenblas-dev || {
            echo -e "${FG_RED}Failed to install LAPACK. Please check manually.${FG_RESET}"
            exit 1
        }
    fi

    if ! ldconfig -p | grep -q "libgfortran.so.3"; then
        if [ ! -e /lib/x86_64-linux-gnu/libgfortran.so.3 ]; then
            echo -e "${FG_YELLOW}Creating a symlink for libgfortran.so.3 -> libgfortran.so.5...${FG_RESET}"
            sudo ln -s /lib/x86_64-linux-gnu/libgfortran.so.5 /lib/x86_64-linux-gnu/libgfortran.so.3
        else
            echo -e "${FG_GREEN}✔ libgfortran.so.3 already exists.${FG_RESET}"
        fi
    else
        echo -e "${FG_GREEN}✔ libgfortran.so.3 is already registered by ldconfig.${FG_RESET}"
    fi

    echo -e "${FG_GREEN}✅ BLAS, LAPACK, and Fortran libraries are properly linked.${FG_RESET}"
}
