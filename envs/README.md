# Environment Setup for Spherical DeepKriging

This directory contains scripts and configuration files for setting up development environments.

## Conda Environment

### Create Conda Environment

To create a conda environment named `spherical-deepkriging` with all dependencies:

```bash
make install-dev
```

This will:
1. Create a conda environment named `spherical-deepkriging`
2. Install dependencies and your package via `poetry install`
3. Build available C++ extensions (spherical basis)

### Start Jupyter Lab with Conda

To start Jupyter Lab using the conda environment:

```bash
make run-local-jupyter

# Or script directly
./envs/jupyter/start_jupyter_lab.sh -k spherical-deepkriging -p 8888
```

### Manual Activation

To manually activate the conda environment:

```bash
conda activate spherical-deepkriging
```

## Virtual Environment (Alternative)

If you prefer using a Python virtual environment instead of conda:

```bash
# Create venv
python -m venv .venv
source .venv/bin/activate

# Install package + Jupyter extras
pip install -e ".[jupyter]"

# Start Jupyter
python -m jupyter lab
```

## Environment Files

- `envs/conda/build_conda_env.sh` - Script to build the `spherical-deepkriging` conda environment
- `envs/jupyter/start_jupyter_lab.sh` - Script to start Jupyter Lab with a conda environment
