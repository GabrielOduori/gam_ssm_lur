# Regenerate Figures Script

## Overview

The `regenerate_figures.py` script allows you to regenerate figures from previously trained models **without retraining**. This is useful when you want to:

- Update figure styling or aesthetics
- Fix plot issues
- Generate figures with different parameters
- Save time by not re-running the entire training pipeline

## Usage

### Basic Usage

Regenerate figures from the most recent experiment:

```bash
python experiments/regenerate_figures.py --latest
```

Regenerate from a specific experiment directory:

```bash
python experiments/regenerate_figures.py --experiment-dir results/experiment_20251209_173407
```

### Options

- `--latest`: Use the most recent experiment in the results/ directory
- `--experiment-dir PATH`: Specify a particular experiment directory
- `--results-dir PATH`: Base results directory (default: results/)
- `--output-dir PATH`: Custom output directory for figures (default: same as experiment)
- `--skip-gridded`: Skip gridded plots (Figures 15-18) to save time
- `--skip-shap`: Skip SHAP plot (Figure 11)

### Examples

```bash
# Regenerate all figures from latest experiment
python experiments/regenerate_figures.py --latest

# Regenerate without gridded plots (faster)
python experiments/regenerate_figures.py --latest --skip-gridded

# Regenerate to a custom output directory
python experiments/regenerate_figures.py --experiment-dir results/experiment_20251209_173407 --output-dir custom_figures/
```

## What Gets Regenerated

The script regenerates the following figures (when possible):

✅ **Always regenerated:**
- Figure 8: Residual diagnostics
- Figure 9: Uncertainty timeseries
- Figure 9b: Temporal evolution
- Figure 9c: Spatial patterns
- Figure 10: Spatial comparison
- Figure 13: Moran's I (if libraries available)

⚠️ **Conditionally regenerated:**
- Figure 6: Convergence (requires EM history - not saved with model)
- Figure 7: Observed vs Predicted (requires training indices - not saved with model)
- Figure 11: SHAP importance (requires original feature matrix X)
- Figure 12: Residual hotspots (may fail if indices can't be reconstructed)
- Figures 15-18: Gridded plots (skip with `--skip-gridded`)

## How It Works

1. **Loads the saved model** from `models/hybrid_gam_ssm/`
2. **Loads predictions** from `tables/predictions_with_intervals.csv`
3. **Reconstructs data structures** needed for plotting
4. **Regenerates figures** using the current plotting code

## Limitations

Some figures require data that isn't saved with the model:
- EM convergence history
- Training/test split indices
- Original feature matrix (for SHAP plots)

These figures will be skipped with a warning message.

## Time Comparison

| Operation | Time |
|-----------|------|
| Full reproduce_paper.py | 30-60+ minutes |
| regenerate_figures.py --latest | ~10-30 seconds |
| regenerate_figures.py --latest --skip-gridded | ~5-10 seconds |

## Typical Workflow

1. **Initial run** (once):
   ```bash
   python experiments/reproduce_paper.py --data-dir data/ --output-dir results/
   ```

2. **Update figure styling** in the code

3. **Regenerate figures** (fast):
   ```bash
   python experiments/regenerate_figures.py --latest --skip-gridded
   ```

4. **Iterate** on steps 2-3 until figures look perfect

This workflow saves significant time when refining figure aesthetics!
