# Examples

This directory contains runnable examples and paper-oriented experiment artifacts.

## Structure

- `toy/`: Minimal smoke-test example for training sphere DeepKriging end-to-end.
- `simulation/`: Paper simulation scripts and pre-executed output notebooks.
- `real_data/`: Real-data experiment notebooks.

## Toy Example (Quick Start)

Use the toy notebook when you only want to verify that the module works:

- `toy/toy_sphere_deepkriging.ipynb`

The notebook is intentionally minimal and split into:

1. import
2. config
3. data generation
4. model training
5. evaluation

## Simulation Experiments (Paper)

Run from this directory:

```bash
conda activate spherical-deepkriging
cd examples/simulation
```

Main scripts:

- `run_stdscaler_nonoise_50reps.py`
- `run_stdscaler_outliers_50reps.py`

These scripts checkpoint progress and can be resumed safely.

Pre-executed output notebooks are also included in `examples/simulation/` for result inspection.

