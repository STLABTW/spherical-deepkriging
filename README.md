# Spherical DeepKriging

[![Tests](https://github.com/STLABTW/spherical-deepkriging/workflows/Test/badge.svg)](https://github.com/STLABTW/spherical-deepkriging/actions)
[![codecov](https://codecov.io/github/STLABTW/spherical-deepkriging/graph/badge.svg?token=OF0LKVDII6)](https://codecov.io/github/STLABTW/spherical-deepkriging)

Code for **DeepKriging on the Global Data** ([arXiv:2604.01689](https://arxiv.org/abs/2604.01689)): spherical spatial prediction with DeepKriging, MRTS-sphere / Wendland bases, and universal kriging. Implementation lives under `spherical_deepkriging/`.

## Install

**From PyPI** ([project page](https://pypi.org/project/spherical-deepkriging/)) — typical use (Python ≥ 3.10):

```bash
pip install spherical-deepkriging
```

Optional extras, e.g. Jupyter / plotting / everything bundled in `[all]`:

```bash
pip install "spherical-deepkriging[jupyter,viz]"
pip install "spherical-deepkriging[all]"
```

**From source (development)** — conda-based env and repo scripts:

Needs [Miniconda](https://docs.conda.io/en/latest/miniconda.html). On Windows, use WSL.

```bash
git clone https://github.com/STLABTW/spherical-deepkriging.git
cd spherical-deepkriging
make install-dev
```

`make install-dev` creates the conda environment and installs dependencies; `make build-spherical-cpp` builds the MRTS-sphere C++ extension. For tests and packaging tools in a plain venv: `pip install -e ".[dev]"`.

## Examples

- Smoke test: `examples/toy/toy_sphere_deepkriging.ipynb`
- Same toy on **Google Colab** (pip install + run): [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/STLABTW/spherical-deepkriging/blob/main/examples/toy/toy_sphere_deepkriging_colab.ipynb) — notebook: `examples/toy/toy_sphere_deepkriging_colab.ipynb`
- Simulations: `examples/simulation/`
- Real data: `examples/real_data/`

See `examples/README.md` for run notes.

## Citation

```bibtex
@misc{huang2026deepkrigingglobaldata,
      title={DeepKriging on the Global Data},
      author={Hao-Yun Huang and Wen-Ting Wang and Ping-Hsun Chiang and Wei-Ying Wu},
      year={2026},
      eprint={2604.01689},
      archivePrefix={arXiv},
      primaryClass={stat.ME},
      url={https://arxiv.org/abs/2604.01689},
}
```
