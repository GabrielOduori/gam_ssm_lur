# GAM-SSM-LUR: Hybrid Generalized Additive Model–State Space Model for Land Use Regression

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A Python framework for spatio-temporal air pollution modelling that integrates Generalized Additive Models (GAMs) with State Space Models (SSMs) for improved prediction accuracy and uncertainty quantification.

## Overview

Traditional Land Use Regression (LUR) models capture spatial heterogeneity in air pollution but treat temporal variation through model structure rather than as an explicit dynamical process. This package implements a hybrid framework that:

1. **GAM Component**: Captures persistent spatial patterns through smooth functions of land use, road network, and traffic covariates
2. **SSM Component**: Models temporal dynamics of residuals via Kalman filtering with Expectation-Maximisation parameter estimation
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
│  │   Measurement: rₜ = Zαₜ + εₜ,  εₜ ~ N(0, H)             │   │
│  │   Transition:  αₜ₊₁ = Tαₜ + Rηₜ,  ηₜ ~ N(0, Q)          │   │
│  │   • Kalman filtering (forward pass)                      │   │
│  │   • RTS smoothing (backward pass)                        │   │
│  │   • EM parameter estimation                              │   │
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

## Reproducing Paper Results

```bash
python experiments/reproduce_paper.py \
  --data-dir data/ \
  --output-dir experiments/results/my_run
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
- `ssm_selected_days.png` — SSM temporal snapshots (min / lower-tercile / upper-tercile / max pollution days)
- `ssm_daily_mean_barchart.png` — daily area-mean NO₂ bar chart with map days highlighted
- `ssm_factors.png` — latent state factor time series
- `loocv_scatter.png` — leave-one-out cross-validation scatter
- `gam_partial_response.png` — GAM partial dependence plots
- `shap_summary.png` — SHAP beeswarm feature importance
- `feature_importance.png` — Random Forest feature importances
- `reliability_diagram.png` — probabilistic calibration (reliability + sharpness + ISS)
- `epa_vs_predicted_timeseries.png` — per-station and daily-mean time series

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
