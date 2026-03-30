SHELL := /bin/bash
CONDA_BIN ?= /home/egpivo/miniconda3/bin/conda
TEST_ENV ?= spherical-deepkriging

.PHONY: clean install-dev install-test-deps setup-jupyter-kernel build-spherical-cpp test run-local-jupyter help

## Clean up temporary files
clean:
	@echo "Cleaning up..."
	@find . -type f -name '*.py[co]' -delete
	@find . -type d -name '__pycache__' -delete
	@rm -rf build/ dist/ .eggs/
	@find . -name '*.egg-info' -exec rm -rf {} +
	@rm -f .coverage
	@rm -rf .pytest_cache
	@find . -type d -name '.ipynb_checkpoints' -exec rm -rf {} +

## Install development dependencies
install-dev:
	@echo "Installing development dependencies..."
	@$(SHELL) envs/conda/build_conda_env.sh -c $(TEST_ENV) && \
	$(MAKE) install-test-deps && \
	$(MAKE) setup-jupyter-kernel

## Install test dependencies in the Conda environment
install-test-deps:
	@echo "Installing test dependencies into $(TEST_ENV)..."
	@$(CONDA_BIN) run -n $(TEST_ENV) python -m pip install -e ".[dev]"

## Install Jupyter dependencies and register kernel
setup-jupyter-kernel:
	@echo "Setting up Jupyter kernel..."
	@$(SHELL) envs/jupyter/setup_jupyter_kernel.sh -k $(TEST_ENV)

## Build spherical basis C++ extension (mrts_sphere)
build-spherical-cpp:
	@echo "Building spherical basis C++ extension..."
	@cd spherical_deepkriging/basis_functions/mrts_sphere/cpp_extensions && \
	rm -rf build && \
	mkdir -p build && \
	cmake -S . -B build && \
	cmake --build build -j$$(nproc)

## Run tests
test:
	@echo "Running tests..."
	@$(CONDA_BIN) run -n $(TEST_ENV) python -c "import pytest, pytest_cov" >/dev/null 2>&1 || (echo "Missing test dependencies in Conda env $(TEST_ENV). Run: make install-test-deps" && exit 1)
	@PYTHONPATH=. $(CONDA_BIN) run -n $(TEST_ENV) python -m pytest --cov=spherical_deepkriging --cov-config=.coveragerc

## Start Jupyter server locally
run-local-jupyter:
	@echo "Starting local Jupyter server..."
	@$(SHELL) envs/jupyter/start_jupyter_lab.sh --port 8501

## Display help information
help:
	@echo "Available targets:"
	@echo "  clean                : Clean up temporary files"
	@echo "  install-dev          : Install dependencies, build package/C++ extension, and setup Jupyter kernel"
	@echo "  install-test-deps    : Install pytest/coverage tooling into the Conda env"
	@echo "  setup-jupyter-kernel : Install Jupyter deps and register kernel for spherical-deepkriging"
	@echo "  build-spherical-cpp  : Build spherical basis C++ extension"
	@echo "  test                 : Run tests"
	@echo "  run-local-jupyter    : Start Jupyter server locally"
	@echo "  help                 : Display this help message"
