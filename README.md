# Spherical DeepKriging

[![Tests](https://github.com/STLABTW/spherical-deepkriging/workflows/Test/badge.svg)](https://github.com/STLABTW/spherical-deepkriging/actions)
[![codecov](https://codecov.io/github/STLABTW/spherical-deepkriging/graph/badge.svg?token=OF0LKVDII6)](https://codecov.io/github/STLABTW/spherical-deepkriging)

Code for **DeepKriging on the Global Data** ([arXiv:2604.01689](https://arxiv.org/abs/2604.01689)): spherical spatial prediction with DeepKriging, MRTS-sphere / Wendland bases, and universal kriging. Implementation lives under `spherical_deepkriging/`.

## Setup

Needs [Miniconda](https://docs.conda.io/en/latest/miniconda.html). On Windows, use WSL.

```bash
git clone https://github.com/STLABTW/spherical-deepkriging.git
cd spherical-deepkriging
make install-dev
```

`make install-dev` creates the conda env and installs dependencies; `make build-spherical-cpp` builds the MRTS-sphere C++ extension.

## Examples

- Smoke test: `examples/toy/toy_sphere_deepkriging.ipynb`
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
