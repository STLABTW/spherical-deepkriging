# Examples

This directory contains runnable examples and paper-oriented experiment artifacts.

## Structure

- `toy/`: Minimal smoke-test example for training sphere DeepKriging end-to-end.
- `simulation/`: Paper simulation scripts and pre-executed output notebooks.
- `real_data/`: Real-data experiment notebooks.

## Toy Example (Quick Start)

Use the toy notebook when you only want to verify that the module works:

- `toy/toy_sphere_deepkriging.ipynb` (local / your own Jupyter)
- `toy/toy_sphere_deepkriging_colab.ipynb` — **Google Colab** variant: installs [`spherical-deepkriging` from PyPI](https://pypi.org/project/spherical-deepkriging/) and installs `cmake` + `g++` on Colab only if needed (sdist build). Open in Colab: [colab link](https://colab.research.google.com/github/STLABTW/spherical-deepkriging/blob/main/examples/toy/toy_sphere_deepkriging_colab.ipynb) (requires this repo on GitHub at `main`).

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


### Reproducing Simulation Experiment

The `examples/simulation/` folder contains self-contained Python scripts and output notebooks for reproducing the 50-repeat simulation study used in the paper. Each script runs a full experiment, checkpoints after every repeat, and can be safely interrupted and resumed.

#### Experiment scripts

| Script | Scenario | Output checkpoint |
|--------|----------|-------------------|
| `run_stdscaler_nonoise_50reps.py` | Local plus nonstationary signal, no noise | `results_sphere_stdscaler_nonoise_50reps_checkpoint.csv` |
| `run_stdscaler_outliers_50reps.py` | Local plus nonstationary signal with outliers | `results_sphere_stdscaler_outliers_50reps_checkpoint.csv` |

Each script benchmarks the following models over 50 random seeds:

- `OLS_wendland` — Ordinary Least Squares with Wendland basis
- `OLS_sphere` — Ordinary Least Squares with spherical harmonic (MRTS-sphere) basis
- `DeepKriging_wendland` — DeepKriging with Wendland basis
- `DeepKriging_mrts` — DeepKriging with MRTS basis
- `DeepKriging_sphere` — DeepKriging with MRTS-sphere basis (MSE loss)
- `DeepKriging_sphere_Huber` — DeepKriging with MRTS-sphere basis (Huber loss, robust to outliers)
- `UniversalKriging` — Universal Kriging with MRTS-sphere basis

#### How to run

Make sure the Conda environment is activated, then navigate to the Rerun folder and run:

```bash
conda activate deepkriging
cd notebook/simulation/Rerun

# No-noise scenario
python run_stdscaler_nonoise_50reps.py

# Outliers scenario
python run_stdscaler_outliers_50reps.py
```

Each script:
- Runs 50 independent repeats with different random seeds.
- Saves a checkpoint CSV after every completed repeat — safe to kill (`Ctrl+C`) and re-run; already-completed repeats are skipped automatically.
- Prints a per-repeat results table (MSPE, RMSE, MAE, R², Time) to stdout.
- Prints a final mean±std summary table across all 50 repeats at the end.

#### Key hyperparameters (configured inside each script)

| Parameter | Value |
|-----------|-------|
| Training samples | 2500 |
| Repeats | 50 |
| Huber delta | 1.345 |
| Knot / order budget | 1400 |
| Batch size | see script |

#### Output notebooks

Pre-executed output notebooks are included in `examples/simulation/Rerun/` for reference:

| Notebook | Description |
|----------|-------------|
| `simulation_spherical_nn_sim_RSHC_c15_GP_pure_50reps_out.ipynb` | GP signal, no noise |
| `simulation_spherical_nn_sim_RSHC_c15_GP_noise_50reps_out.ipynb` | GP signal with noise |
| `simulation_spherical_nn_sim_RSHC_c15_GP_outliers_50reps_out.ipynb` | GP signal with outliers |
| `simulation_spherical_nn_sim_RSHC_c15_noGP_pure_50reps_out.ipynb` | Local extreme signal, no noise |
| `simulation_spherical_nn_sim_RSHC_c15_noise_noGP_50reps_out.ipynb` | Local extreme signal with noise |s