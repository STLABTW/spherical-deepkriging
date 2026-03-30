# Spherical DeepKriging
[![Tests](https://github.com/STLABTW/spherical-deepkriging/workflows/Test/badge.svg)](https://github.com/STLABTW/spherical-deepkriging/actions)
[![codecov](https://codecov.io/github/STLABTW/spherical-deepkriging/graph/badge.svg?token=OF0LKVDII6)](https://codecov.io/github/STLABTW/spherical-deepkriging)

A deep learning framework for spatial prediction on the sphere, combining DeepKriging with spherical harmonic basis functions (MRTS-sphere), Wendland basis, and Universal Kriging.

---

### Available Basis Functions

The package currently provides the following basis families:

| Basis family | Module path | Role in this project |
|---|---|---|
| **MRTS-sphere** | `spherical_deepkriging.basis_functions.mrts_sphere` | **Primary basis (first choice)** for spherical-coordinate modeling in this project |
| **MRTS (Euclidean)** | `spherical_deepkriging.basis_functions.mrts` | Secondary/auxiliary basis for Euclidean reference experiments |
| **Wendland** | `spherical_deepkriging.basis_functions.wendland` | Secondary/auxiliary compact-support basis for baseline and comparison setups |

---

### Feedforward Backbone

The default feedforward hidden-block design is illustrated below:

<img src="artifacts/hidden_block.png" alt="DeepKriging hidden block" width="60%" />

---

### Prerequisites

Before installation, make sure Miniconda is available.

- **Miniconda**: Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html).
- **Windows users**: Use WSL for development and builds.

---

### Installation

In most cases, make install-dev sets up everything you need for local development.
```bash
    git clone https://github.com/STLABTW/spherical-deepkriging.git
    cd spherical-deepkriging
    make install-dev
```


Notes:
- `make install-dev` creates the spherical-deepkriging conda environment and installs project dependencies.
- `make build-spherical-cpp` builds the MRTS-sphere C++ extension.

---

### Examples

Examples are organized under `examples/`:

- Quick module smoke test: `examples/toy/toy_sphere_deepkriging.ipynb`
- Paper simulations: `examples/simulation/`
- Real-data notebooks: `examples/real_data/`

For run instructions and detailed notes, see:

- `examples/README.md`

---

### Paper

The corresponding paper reference and citation will be added here in a later update.
