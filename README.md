# GAM-SSM-LUR: Hybrid Generalized Additive Model–State Space Model for Land Use Regression

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A Python framework for spatiotemporal air pollution modeling that integrates Generalized Additive Models (GAMs) with State Space Models (SSMs) for improved prediction accuracy and uncertainty quantification.

## Overview

Traditional Land Use Regression (LUR) models capture spatial heterogeneity in air pollution but treat temporal variation through model structure rather than as an explicit dynamical process. This package implements a hybrid framework that:

1. **GAM Component**: Captures persistent spatial patterns through smooth functions of land use, road network, and traffic covariates
2. **SSM Component**: Models temporal dynamics of residuals via Kalman filtering with Expectation-Maximisation parameter estimation
3. **Uncertainty Quantification**: Provides calibrated prediction intervals through posterior state distributions

## Key Features

- 🎯 **Hybrid spatiotemporal modeling** combining interpretable GAMs with dynamic state space models
- 📊 **Principled uncertainty quantification** via Kalman filtering and smoothing
- ⚡ **Scalable inference** through adaptive matrix representations (dense, sparse, block-diagonal)
- 🔧 **Modular design** allowing component-wise customization
- 📈 **Comprehensive diagnostics** for model validation and assessment

## Installation

### From PyPI (recommended)

```bash
pip install gam-ssm-lur
```

### From source

```bash
git clone https://github.com/GabrielOduori/lur_space_state_model.git
cd lur_space_state_model
pip install -e ".[dev]"
```

### Optional dependencies

```bash
# For OpenStreetMap feature extraction
pip install gam-ssm-lur[osm]

# For Google Earth Engine satellite data
pip install gam-ssm-lur[satellite]

# For development
pip install gam-ssm-lur[dev]
```

## Quick Start

```python
import numpy as np
import pandas as pd
from gam_ssm_lur import HybridGAMSSM
from gam_ssm_lur.features import FeatureSelector
from gam_ssm_lur.evaluation import ModelEvaluator

# Load your data
data = pd.read_csv("no2_observations.csv")
features = pd.read_csv("spatial_features.csv")

# Feature selection pipeline
selector = FeatureSelector(
    correlation_threshold=0.8,
    vif_threshold=10.0,
    n_top_features=30
)
selected_features = selector.fit_transform(features, data["no2"])

# Fit hybrid model
model = HybridGAMSSM(
    n_splines=10,
    state_dim=None,  # Auto-determined from data
    em_max_iter=50,
    em_tol=1e-6,
    scalability_mode="auto"
)
model.fit(selected_features, data["no2"], time_index=data["timestamp"])

# Predict with uncertainty
predictions = model.predict(selected_features, return_std=True)
print(f"RMSE: {predictions['rmse']:.3f}")
print(f"Coverage: {predictions['coverage_95']:.1%}")

# Evaluate model
evaluator = ModelEvaluator(model)
evaluator.diagnostic_plots(save_path="diagnostics/")
evaluator.summary_report()
```

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

## Documentation

### Core Classes

| Class | Description |
|-------|-------------|
| `HybridGAMSSM` | Main model class integrating GAM and SSM components |
| `SpatialGAM` | Generalized Additive Model for spatial covariates |
| `StateSpaceModel` | Linear Gaussian SSM with Kalman filter/smoother |
| `EMEstimator` | Expectation-Maximisation parameter estimation |
| `FeatureSelector` | Multi-stage feature selection pipeline |
| `ModelEvaluator` | Comprehensive model diagnostics and evaluation |

### Scalability Modes

The framework automatically selects computational strategy based on problem size:

| Mode | Network Size | Strategy | Complexity |
|------|--------------|----------|------------|
| `dense` | n < 1,000 | Full matrix operations | O(n³) |
| `diagonal` | 1,000 ≤ n < 5,000 | Diagonal covariance approximation | O(n) |
| `block` | n ≥ 5,000 | Block-diagonal decomposition | O(n³/b²) |

## Examples

See the `experiments/` directory for detailed tutorials:

- `01_basic_usage.py`: Simple end-to-end workflow
- `02_feature_engineering.py`: OSM and traffic data extraction
- `03_model_comparison.py`: Benchmarking against static LUR
- `04_uncertainty_analysis.py`: Prediction interval calibration
- `05_scalability_demo.py`: Large network inference

## Reproducing Paper Results

To reproduce the results from the paper:

```bash
# Option 1: Run with demo data (no file needed)
python experiments/reproduce_paper.py

# Option 2: Run with your own data (auto-detects columns interactively)
python experiments/reproduce_paper.py --data-file /path/to/your_data.csv

# Option 3: Quick test with limited records (faster for testing)
python experiments/reproduce_paper.py \
  --data-file /path/to/your_data.csv \
  --max-records 100 \
  --yes

# Option 4: Run with specific column names (non-interactive)
python experiments/reproduce_paper.py \
  --data-file /path/to/your_data.csv \
  --timestamp-col timestamp \
  --target-col epa_no2 \
  --location-col grid_id \
  --lat-col latitude \
  --lon-col longitude \
  --yes

# Option 5: Customize output location
python experiments/reproduce_paper.py \
  --data-file /path/to/your_data.csv \
  --output-dir my_results \
  --run-name experiment_2024
```

**Your CSV file should contain:**
- A timestamp column (e.g., `timestamp`, `date`, `datetime`)
- A target NO₂ column (e.g., `epa_no2`, `no2`)
- A location identifier (e.g., `grid_id`, `location_id`, `station_id`)
- Latitude and longitude columns (e.g., `latitude`, `longitude`)
- Spatial feature columns (e.g., `traffic_volume`, `motorway_50m`, etc.)

The script will auto-detect these columns and ask for confirmation before running.

## Citation

If you use this software in your research, please cite:

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

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

This work was supported by:
- CitiObs (European Union, grant agreement 101086421)
- University College Dublin Spatial Dynamics Lab

## Contact

- Gabriel Oduori - gabriel.oduori@ucd.ie
- Project Link: https://github.com/GabrielOduori/lur_space_state_model
