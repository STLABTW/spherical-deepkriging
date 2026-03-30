SHELL := /bin/bash
EXECUTABLE := poetry run

.PHONY: clean install-dev setup-jupyter-kernel build-spherical-cpp test run-local-jupyter help

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
	@$(SHELL) envs/conda/build_conda_env.sh -c spherical-deepkriging && \
	$(MAKE) setup-jupyter-kernel

## Install Jupyter dependencies and register kernel
setup-jupyter-kernel:
	@echo "Setting up Jupyter kernel..."
	@$(SHELL) envs/jupyter/setup_jupyter_kernel.sh -k spherical-deepkriging

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
	@$(EXECUTABLE) pytest --cov=spherical_deepkriging

## Start Jupyter server locally
run-local-jupyter:
	@echo "Starting local Jupyter server..."
	@$(SHELL) envs/jupyter/start_jupyter_lab.sh --port 8501

## Display help information
help:
	@echo "Available targets:"
	@echo "  clean                : Clean up temporary files"
	@echo "  install-dev          : Install dependencies, build package/C++ extension, and setup Jupyter kernel"
	@echo "  setup-jupyter-kernel : Install Jupyter deps and register kernel for spherical-deepkriging"
	@echo "  build-spherical-cpp : Build spherical basis C++ extension"
	@echo "  test                 : Run tests"
	@echo "  run-local-jupyter    : Start Jupyter server locally"
	@echo "  help                 : Display this help message"
