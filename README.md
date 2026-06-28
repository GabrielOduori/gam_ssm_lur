# Hybrid Generalized Additive Model–State Space Model for Land Use Regression

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A Python framework for spatio-temporal air pollution modelling that integrates Generalized Additive Models (GAMs) with State Space Models (SSMs) for improved prediction accuracy and uncertainty quantification.

## Overview

Traditional Land Use Regression (LUR) models capture spatial heterogeneity in air pollution but treat temporal variation through model structure rather than as an explicit dynamical process. This package implements a hybrid framework that:

1. **GAM Component**: Captures persistent spatial patterns through smooth functions of land use, road network, and traffic covariates
2. **SSM Component**: Models temporal dynamics via Kalman filtering with Expectation-Maximisation parameter estimation. The GAM-regression coefficient (β) and external forcing coefficients (e.g. traffic anomaly, wind; B̃) are jointly estimated alongside the latent dynamics by an augmented-state EM, rather than pre-estimated by OLS and subtracted — avoiding silently discarding their estimation error into the dynamics (Harvey, 1989, Ch. 3.3).
3. **Uncertainty Quantification**: Provides prediction intervals through posterior state distributions

## Model Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Observations y(s,t)                          │
│                           │                                     │
│                           ▼                                     │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              GAM Spatial Component                       │   │
│  │   μ(s) = β₀ + Σⱼ fⱼ(xⱼ(s))                              │   │
│  │   • Land use features                                    │   │
│  │   • Road network density                                 │   │
│  │   • Traffic proximity                                    │   │
│  └─────────────────────────────────────────────────────────┘   │
│                           │                                     │
│                           ▼                                     │
│                    Residuals r(t) = y(t) - μ̂                    │
│                           │                                     │
│                           ▼                                     │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              SSM Temporal Component                      │   │
│  │   Measurement: yₜ = Zαₜ + β·g̃ + B̃uₜ + εₜ,  εₜ ~ N(0, H) │   │
│  │   Transition:  αₜ₊₁ = Tαₜ + Rηₜ,  ηₜ ~ N(0, Q)          │   │
│  │   • β, B̃ are jointly-estimated fixed-effect states       │   │
│  │     (identity transition, zero process noise)            │   │
│  │   • Kalman filtering (forward pass)                      │   │
│  │   • RTS smoothing (backward pass)                        │   │
│  │   • Augmented-state EM parameter estimation               │   │
│  └─────────────────────────────────────────────────────────┘   │
│                           │                                     │
│                           ▼                                     │
│              Smoothed estimates α̂ₜ|ₜ with P̂ₜ|ₜ                  │
│              → Predictions with calibrated uncertainty          │
└─────────────────────────────────────────────────────────────────┘
```

## Installation

```bash
git clone https://github.com/GabrielOduori/gam_ssm_lur.git
cd gam_ssm_lur
pip install -e ".[dev]"
```

Or, using the `Makefile` shortcut (also registers both pre-commit hook stages — see Development below):

```bash
make install-dev
```

## Development

The `Makefile` covers two different jobs — installing/checking the code, and running the experiment pipeline. It is not just a dev-quality tool:

```bash
make help          # list all available targets

# Setup
make install       # pip install -e .
make install-dev   # pip install -e ".[dev]" + register both pre-commit hook stages

# Code quality
make test          # run tests with coverage
make lint          # ruff check
make format        # ruff format + ruff check --fix
make check         # format + lint + test, in one go

# Run the experiment (equivalent to the manual command in
# "Reproducing Paper Results" below, using its default output path)
make reproduce
```

Linting and formatting are enforced via [ruff](https://docs.astral.sh/ruff/) on every commit through `pre-commit`, and commit messages are checked for AI-attribution trailers (e.g. `Co-Authored-By: Claude...`) via a local commit-msg hook:

```bash
pre-commit install                          # one-time, after pip install -e ".[dev]"
pre-commit install --hook-type commit-msg   # also enforce the commit-msg check
```

(`make install-dev` runs both of the above automatically.) After that, `git commit` runs `ruff check --fix` and `ruff format` against staged files, plus the commit-msg check, automatically.

**Note:** the commit-msg hook script (`.tools/check_attribution.sh`) lives in a gitignored directory and is not part of the repository — it exists only on machines where it's been set up already. On a fresh clone, `pre-commit install --hook-type commit-msg` will fail until that script is recreated locally (its content is in this repo's git history if needed).

To run the lint/format steps manually outside of a commit:

```bash
ruff check src/ tests/        # lint
ruff format src/ tests/       # format
```

Rule selection (`E`, `W`, `F`, `I`, `C`, `B` — pycodestyle, Pyflakes, isort, flake8-comprehensions, flake8-bugbear) and per-file exceptions (e.g. `E402` for the standalone scripts under `experiments/`/`.tools/` that bootstrap `sys.path` before importing the package) live in `pyproject.toml` under `[tool.ruff]`.

## Data Directory Structure

```
data/
├── features.csv              # Static spatial features per grid cell
├── target.csv                # Static NO₂ target values per grid cell
├── grid/
│   └── grid.geojson          # Grid cell geometries
└── time_series/
    ├── satellite_retreavals.csv   # Daily TROPOMI satellite observations
    ├── epa_timeseries.csv         # EPA ground station measurements
    ├── traffic_timeseries.csv     # Daily traffic activity
    └── wind_sector_2023-06_daily.csv  # ERA5 wind sector data
```

### Data Access

The datasets above are archived on Zenodo: **[10.5281/zenodo.16534137](https://doi.org/10.5281/zenodo.16534137)** (concept DOI — always resolves to the latest version).

You don't need to download this manually. If `data/` is missing any required file, `experiments/reproduce_paper.py` automatically fetches and extracts the archive on first run via `ensure_data_available()` in `src/gam_ssm_lur/fetch_data.py`. To trigger this fetch standalone:

```python
from pathlib import Path
from gam_ssm_lur.fetch_data import ensure_data_available

ensure_data_available(Path("data"))
```

The results reported in the paper were generated from version v3 of the dataset (DOI: [10.5281/zenodo.20793214](https://doi.org/10.5281/zenodo.20793214)); the concept DOI may point to a newer version by the time you read this.

## Reproducing Paper Results

```bash
python experiments/reproduce_paper.py \
  --data-dir data/ \
  --output-dir experiments/results/my_run
```

Or, without a custom output directory (uses the script's own auto-timestamped default):

```bash
make reproduce
```

### Regenerating figures only (no retraining)

```bash
# Most recent run
python experiments/regenerate_figures.py --latest

# Specific run
python experiments/regenerate_figures.py \
  --run-dir experiments/results/run_20260405_213102

# Write to a different figures directory
python experiments/regenerate_figures.py --latest \
  --fig-dir experiments/results/run_20260405_213102/figures_v2
```

Outputs are saved to `experiments/results/run_YYYYMMDD_HHMMSS/` by default.

### Key CLI Options

| Option | Default | Description |
|--------|---------|-------------|
| `--data-dir` | `data/` | Root data directory |
| `--output-dir` | `experiments/results/run_<ts>` | Output directory |
| `--target-col` | `atmos_no2` | Target column in target.csv |
| `--importance-threshold` | `0.95` | Cumulative RF importance threshold for feature selection |
| `--skip-feature-selection` | off | Reuse features from a previous run |
| `--n-splines` | `10` | GAM spline basis functions per feature |
| `--em-max-iter` | `50` | EM algorithm maximum iterations |
| `--scalability-mode` | `auto` | Matrix mode: `auto`, `dense`, `diagonal`, `block` |

### Output Files

All outputs are written to `experiments/results/run_YYYYMMDD_HHMMSS/`:

**Model:**
- `model/` — saved model weights and parameters
- `selected_features.txt` — features selected by the pipeline
- `feature_importances.csv` — Random Forest importances
- `model_summary.txt` — performance metrics and calibration report
- `model_comparison.csv` — GAM-LUR vs GAM-SSM vs LOOCV side-by-side with improvement row
- `model_comparison.tex` — LaTeX table ready for manuscript inclusion

**Figures (`figures/`):**
- `static_lur_prior.png` — GAM spatial prediction surface
- `spatial_residuals.png` — signed and absolute GAM residuals
- `residual_diagnostics.png` — 2×3 residual diagnostic panel
- `em_convergence.png` — EM algorithm convergence trace
- `ssm_selected_days.png` — SSM temporal snapshots (min / lower-tercile / upper-tercile / max pollution days, excluding the first/last calendar day — RTS smoother boundary estimates are least reliable there)
- `ssm_daily_mean_barchart.png` — daily area-mean NO₂ bar chart with map days highlighted
- `station_timeseries.png` — per-station time series (GAM prior, SSM-corrected, observed)
- `loocv_scatter.png` — leave-one-out cross-validation scatter
- `wind_sector_map.png` — GAM NO₂ map per dominant wind sector
- `svd_scree.png` — SVD factor-selection diagnostic
- `factor_heatmap.png` — latent SSM factor × day heatmap
- `shap_summary.png` — SHAP beeswarm feature importance (exact closed-form additive Shapley values)
- `moran_scatterplot.png` — Moran's I spatial autocorrelation of GAM residuals (if computed)
- `tropomi_epa_calibration_scatter.png` — TROPOMI-EPA satellite-to-surface OLS calibration scatter
- `reliability_diagram.png` — probabilistic calibration (reliability + sharpness + ISS)
- `epa_vs_predicted_timeseries.png` — per-station observed-vs-predicted time series
- `epa_daily_mean_timeseries.png` — daily-mean observed-vs-predicted time series

### Scalability Modes

| Mode | Network Size | Strategy |
|------|--------------|----------|
| `dense` | n < 1,000 | Full matrix operations |
| `diagonal` | 1,000 ≤ n < 5,000 | Diagonal covariance approximation |
| `block` | n ≥ 5,000 | Block-diagonal decomposition |

## Citation

```bibtex
@article{oduori2025hybrid,
  title={Hybrid Generalized Additive-State Space Modelling for Urban {NO}$_2$ 
         Prediction: Integrating Spatial and Temporal Dynamics},
  author={Oduori, Gabriel and Cocco, Chiara and Sajadi, Payam and Pilla, Francesco},
  journal={Environmental Modelling \& Software},
  year={2025},
  note={Submitted}
}
```

## Acknowledgments

This work was supported by:
- CitiObs (European Union, grant agreement 101086421)
- University College Dublin Spatial Dynamics Lab

## Contact

- Gabriel Oduori — gabriel.oduori@ucdconnect.ie
- Project: https://github.com/GabrielOduori/gam_ssm_lur
