"""
Model Evaluation and Diagnostics for GAM-SSM-LUR.

This module provides comprehensive tools for:
1. Predictive accuracy assessment (RMSE, MAE, R², correlation)
2. Uncertainty calibration (coverage, interval width)
3. Residual diagnostics (normality, autocorrelation, heteroscedasticity)
4. Visualization of results

References
----------
.. [1] Gneiting, T., & Raftery, A. E. (2007). Strictly proper scoring rules,
       prediction, and estimation. Journal of the American Statistical Association.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class AccuracyMetrics:
    """Container for accuracy metrics.
    
    Attributes
    ----------
    rmse : float
        Root mean square error
    mae : float
        Mean absolute error
    mbe : float
        Mean bias error
    r2 : float
        Coefficient of determination
    correlation : float
        Pearson correlation coefficient
    """
    rmse: float
    mae: float
    mbe: float
    r2: float
    correlation: float
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            'rmse': self.rmse,
            'mae': self.mae,
            'mbe': self.mbe,
            'r2': self.r2,
            'correlation': self.correlation,
        }


@dataclass
class CalibrationMetrics:
    """Container for uncertainty calibration metrics.
    
    Attributes
    ----------
    coverage : Dict[str, float]
        Coverage probability at different levels (50%, 80%, 90%, 95%)
    mean_interval_width : float
        Average width of 95% prediction interval
    interval_skill_score : float
        Interval skill score (Gneiting & Raftery, 2007)
    crps : float
        Continuous Ranked Probability Score
    """
    coverage: Dict[str, float]
    mean_interval_width: float
    interval_skill_score: float
    crps: float


@dataclass
class ResidualDiagnostics:
    """Container for residual diagnostic results.
    
    Attributes
    ----------
    mean : float
        Mean of residuals (should be ~0)
    std : float
        Standard deviation of residuals
    skewness : float
        Skewness of residuals
    kurtosis : float
        Excess kurtosis of residuals
    shapiro_statistic : float
        Shapiro-Wilk test statistic
    shapiro_pvalue : float
        Shapiro-Wilk test p-value
    ljung_box_statistic : float
        Ljung-Box test statistic for autocorrelation
    ljung_box_pvalue : float
        Ljung-Box test p-value
    acf : NDArray
        Autocorrelation function values
    """
    mean: float
    std: float
    skewness: float
    kurtosis: float
    shapiro_statistic: float
    shapiro_pvalue: float
    ljung_box_statistic: float
    ljung_box_pvalue: float
    acf: NDArray


class ModelEvaluator:
    """Comprehensive model evaluation and diagnostics.
    
    Provides tools for assessing model performance including:
    - Accuracy metrics (RMSE, MAE, R², etc.)
    - Uncertainty calibration (coverage, interval width)
    - Residual diagnostics (normality, autocorrelation)
    - Visualization tools
    
    Parameters
    ----------
    model : HybridGAMSSM, optional
        Fitted model to evaluate. Can also pass predictions directly.
        
    Examples
    --------
    >>> evaluator = ModelEvaluator(model)
    >>> metrics = evaluator.compute_accuracy(y_true, y_pred)
    >>> evaluator.diagnostic_plots(save_path="figures/")
    >>> evaluator.summary_report()
    """
    
    def __init__(self, model=None):
        self.model = model
        self._y_true: Optional[NDArray] = None
        self._y_pred: Optional[NDArray] = None
        self._y_std: Optional[NDArray] = None
        self._residuals: Optional[NDArray] = None
        
    def compute_accuracy(
        self,
        y_true: NDArray,
        y_pred: NDArray,
    ) -> AccuracyMetrics:
        """Compute accuracy metrics.
        
        Parameters
        ----------
        y_true : NDArray
            True values
        y_pred : NDArray
            Predicted values
            
        Returns
        -------
        AccuracyMetrics
            Container with accuracy metrics
        """
        y_true = np.asarray(y_true).flatten()
        y_pred = np.asarray(y_pred).flatten()
        
        # Store for later use
        self._y_true = y_true
        self._y_pred = y_pred
        self._residuals = y_true - y_pred
        
        # Compute metrics
        rmse = np.sqrt(np.mean(self._residuals**2))
        mae = np.mean(np.abs(self._residuals))
        mbe = np.mean(self._residuals)
        
        ss_res = np.sum(self._residuals**2)
        ss_tot = np.sum((y_true - np.mean(y_true))**2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        
        # Handle constant predictions
        if np.std(y_pred) > 0 and np.std(y_true) > 0:
            correlation = np.corrcoef(y_true, y_pred)[0, 1]
        else:
            correlation = 0.0
            
        return AccuracyMetrics(
            rmse=rmse,
            mae=mae,
            mbe=mbe,
            r2=r2,
            correlation=correlation,
        )
        
    def compute_calibration(
        self,
        y_true: NDArray,
        y_pred: NDArray,
        y_std: NDArray,
    ) -> CalibrationMetrics:
        """Compute uncertainty calibration metrics.
        
        Parameters
        ----------
        y_true : NDArray
            True values
        y_pred : NDArray
            Predicted means
        y_std : NDArray
            Predicted standard deviations
            
        Returns
        -------
        CalibrationMetrics
            Container with calibration metrics
        """
        from scipy import stats
        
        y_true = np.asarray(y_true).flatten()
        y_pred = np.asarray(y_pred).flatten()
        y_std = np.asarray(y_std).flatten()
        
        self._y_std = y_std
        
        # Coverage at different levels
        coverage = {}
        for level in [0.50, 0.80, 0.90, 0.95]:
            z = stats.norm.ppf((1 + level) / 2)
            lower = y_pred - z * y_std
            upper = y_pred + z * y_std
            coverage[f"{int(level*100)}%"] = np.mean((y_true >= lower) & (y_true <= upper))
            
        # Mean interval width (95%)
        z_95 = stats.norm.ppf(0.975)
        interval_width = np.mean(2 * z_95 * y_std)
        
        # Interval skill score
        alpha = 0.05
        lower_95 = y_pred - z_95 * y_std
        upper_95 = y_pred + z_95 * y_std
        width = upper_95 - lower_95
        penalty_lower = (2 / alpha) * (lower_95 - y_true) * (y_true < lower_95)
        penalty_upper = (2 / alpha) * (y_true - upper_95) * (y_true > upper_95)
        iss = np.mean(width + penalty_lower + penalty_upper)
        
        # CRPS (Continuous Ranked Probability Score)
        # For Gaussian predictive distributions
        z = (y_true - y_pred) / y_std
        crps = np.mean(
            y_std * (z * (2 * stats.norm.cdf(z) - 1) + 
                     2 * stats.norm.pdf(z) - 1 / np.sqrt(np.pi))
        )
        
        return CalibrationMetrics(
            coverage=coverage,
            mean_interval_width=interval_width,
            interval_skill_score=iss,
            crps=crps,
        )
        
    def compute_residual_diagnostics(
        self,
        residuals: Optional[NDArray] = None,
        max_lag: int = 20,
    ) -> ResidualDiagnostics:
        """Compute residual diagnostics.
        
        Parameters
        ----------
        residuals : NDArray, optional
            Residuals to analyze. Uses stored residuals if not provided.
        max_lag : int
            Maximum lag for autocorrelation
            
        Returns
        -------
        ResidualDiagnostics
            Container with diagnostic results
        """
        from scipy import stats
        
        if residuals is None:
            if self._residuals is None:
                raise ValueError("No residuals available. Call compute_accuracy first.")
            residuals = self._residuals
        else:
            residuals = np.asarray(residuals).flatten()
            
        # Basic statistics
        mean = np.mean(residuals)
        std = np.std(residuals)
        skewness = stats.skew(residuals)
        kurtosis = stats.kurtosis(residuals)
        
        # Normality test (use subset if too many points)
        if len(residuals) > 5000:
            sample_idx = np.random.choice(len(residuals), 5000, replace=False)
            shapiro_stat, shapiro_p = stats.shapiro(residuals[sample_idx])
        else:
            shapiro_stat, shapiro_p = stats.shapiro(residuals)
            
        # Autocorrelation
        acf = self._compute_acf(residuals, max_lag)
        
        # Ljung-Box test for autocorrelation
        n = len(residuals)
        lb_stat = n * (n + 2) * np.sum(acf[1:]**2 / (n - np.arange(1, max_lag + 1)))
        lb_p = 1 - stats.chi2.cdf(lb_stat, max_lag)
        
        return ResidualDiagnostics(
            mean=mean,
            std=std,
            skewness=skewness,
            kurtosis=kurtosis,
            shapiro_statistic=shapiro_stat,
            shapiro_pvalue=shapiro_p,
            ljung_box_statistic=lb_stat,
            ljung_box_pvalue=lb_p,
            acf=acf,
        )
        
    def _compute_acf(self, x: NDArray, max_lag: int) -> NDArray:
        """Compute autocorrelation function."""
        n = len(x)
        mean = np.mean(x)
        var = np.var(x)
        
        if var == 0:
            return np.zeros(max_lag + 1)
            
        acf = np.zeros(max_lag + 1)
        for lag in range(max_lag + 1):
            if lag == 0:
                acf[lag] = 1.0
            else:
                acf[lag] = np.mean((x[:n-lag] - mean) * (x[lag:] - mean)) / var
                
        return acf
        
    def diagnostic_plots(
        self,
        y_true: Optional[NDArray] = None,
        y_pred: Optional[NDArray] = None,
        y_std: Optional[NDArray] = None,
        save_path: Optional[Union[str, Path]] = None,
        figsize: Tuple[int, int] = (12, 10),
    ) -> None:
        """Generate diagnostic plots.
        
        Creates a 2x3 grid of diagnostic plots:
        1. Observed vs Predicted scatter
        2. Residuals vs Fitted
        3. Q-Q plot
        4. Residual histogram
        5. Autocorrelation function
        6. Temporal residual pattern
        
        Parameters
        ----------
        y_true : NDArray, optional
            True values
        y_pred : NDArray, optional
            Predicted values
        y_std : NDArray, optional
            Prediction standard deviations
        save_path : str or Path, optional
            Path to save figure
        figsize : tuple
            Figure size
        """
        import matplotlib.pyplot as plt
        from scipy import stats
        
        # Use stored values if not provided
        y_true = y_true if y_true is not None else self._y_true
        y_pred = y_pred if y_pred is not None else self._y_pred
        
        if y_true is None or y_pred is None:
            raise ValueError("No predictions available. Call compute_accuracy first or provide data.")
            
        y_true = np.asarray(y_true).flatten()
        y_pred = np.asarray(y_pred).flatten()
        residuals = y_true - y_pred
        
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        
        # 1. Observed vs Predicted
        ax = axes[0, 0]
        ax.scatter(y_pred, y_true, alpha=0.5, s=10, c='steelblue')
        min_val = min(y_pred.min(), y_true.min())
        max_val = max(y_pred.max(), y_true.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='1:1 line')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Observed')
        ax.set_title(f'Observed vs Predicted\n(r = {np.corrcoef(y_true, y_pred)[0,1]:.3f})')
        ax.legend()
        
        # 2. Residuals vs Fitted
        ax = axes[0, 1]
        ax.scatter(y_pred, residuals, alpha=0.5, s=10, c='steelblue')
        ax.axhline(y=0, color='r', linestyle='--', lw=2)
        ax.set_xlabel('Fitted Values')
        ax.set_ylabel('Residuals')
        ax.set_title('Residuals vs Fitted')
        
        # Add loess smoothing line
        try:
            from scipy.ndimage import uniform_filter1d
            sorted_idx = np.argsort(y_pred)
            smoothed = uniform_filter1d(residuals[sorted_idx], size=max(len(residuals)//20, 10))
            ax.plot(y_pred[sorted_idx], smoothed, 'orange', lw=2, label='Smoothed')
            ax.legend()
        except Exception:
            pass
            
        # 3. Q-Q Plot
        ax = axes[0, 2]
        stats.probplot(residuals, dist="norm", plot=ax)
        ax.set_title('Normal Q-Q Plot')
        ax.get_lines()[0].set_markersize(3)
        ax.get_lines()[0].set_alpha(0.5)
        
        # 4. Residual Histogram
        ax = axes[1, 0]
        ax.hist(residuals, bins=50, density=True, alpha=0.7, color='steelblue', edgecolor='white')
        
        # Overlay normal distribution
        x_range = np.linspace(residuals.min(), residuals.max(), 100)
        ax.plot(x_range, stats.norm.pdf(x_range, np.mean(residuals), np.std(residuals)), 
                'r-', lw=2, label='Normal')
        ax.set_xlabel('Residuals')
        ax.set_ylabel('Density')
        ax.set_title(f'Residual Distribution\n(skew={stats.skew(residuals):.2f}, kurt={stats.kurtosis(residuals):.2f})')
        ax.legend()
        
        # 5. Autocorrelation Function
        ax = axes[1, 1]
        max_lag = min(50, len(residuals) // 4)
        acf = self._compute_acf(residuals, max_lag)
        
        ax.bar(range(len(acf)), acf, color='steelblue', edgecolor='white')
        # Confidence bounds (approximate 95% CI)
        ci = 1.96 / np.sqrt(len(residuals))
        ax.axhline(y=ci, color='r', linestyle='--', lw=1)
        ax.axhline(y=-ci, color='r', linestyle='--', lw=1)
        ax.axhline(y=0, color='k', lw=0.5)
        ax.set_xlabel('Lag')
        ax.set_ylabel('ACF')
        ax.set_title('Residual Autocorrelation')
        
        # 6. Temporal Pattern (if data has temporal structure)
        ax = axes[1, 2]
        if len(residuals) > 100:
            # Rolling mean and std
            window = max(len(residuals) // 50, 10)
            rolling_mean = pd.Series(residuals).rolling(window=window, center=True).mean()
            rolling_std = pd.Series(residuals).rolling(window=window, center=True).std()
            
            time_idx = np.arange(len(residuals))
            ax.fill_between(time_idx, rolling_mean - rolling_std, rolling_mean + rolling_std,
                           alpha=0.3, color='steelblue', label='±1 SD')
            ax.plot(time_idx, rolling_mean, 'b-', lw=1.5, label='Rolling mean')
            ax.axhline(y=0, color='r', linestyle='--', lw=1)
            ax.set_xlabel('Time Index')
            ax.set_ylabel('Residual')
            ax.set_title('Temporal Residual Pattern')
            ax.legend()
        else:
            ax.plot(residuals, 'o-', markersize=3, alpha=0.7)
            ax.axhline(y=0, color='r', linestyle='--')
            ax.set_xlabel('Index')
            ax.set_ylabel('Residual')
            ax.set_title('Residuals Over Time')
            
        plt.tight_layout()
        
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Diagnostic plots saved to {save_path}")
            
        plt.show()
        
    def convergence_plot(
        self,
        log_likelihoods: Optional[List[float]] = None,
        save_path: Optional[Union[str, Path]] = None,
        figsize: Tuple[int, int] = (12, 4),
    ) -> None:
        """Plot EM algorithm convergence diagnostics.
        
        Parameters
        ----------
        log_likelihoods : list of float, optional
            Log-likelihood values at each iteration. Uses model if not provided.
        save_path : str or Path, optional
            Path to save figure
        figsize : tuple
            Figure size
        """
        import matplotlib.pyplot as plt
        
        if log_likelihoods is None:
            if self.model is None:
                raise ValueError("No model or log-likelihoods provided")
            log_likelihoods = self.model.ssm_.em_result_.log_likelihoods
            
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        iterations = range(len(log_likelihoods))
        
        # 1. Log-likelihood
        ax = axes[0]
        ax.plot(iterations, log_likelihoods, 'b-o', markersize=4)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Log-Likelihood')
        ax.set_title('EM Convergence: Log-Likelihood')
        ax.grid(True, alpha=0.3)
        
        # 2. Log-likelihood increment
        ax = axes[1]
        if len(log_likelihoods) > 1:
            ll_diff = np.diff(log_likelihoods)
            ax.semilogy(range(1, len(log_likelihoods)), np.abs(ll_diff), 'g-o', markersize=4)
            ax.axhline(y=1e-6, color='r', linestyle='--', label='Tolerance')
            ax.set_xlabel('Iteration')
            ax.set_ylabel('|ΔLog-Likelihood|')
            ax.set_title('EM Convergence: Increment')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
        # 3. Parameter traces (if available)
        ax = axes[2]
        if self.model is not None and hasattr(self.model, 'ssm_'):
            ssm = self.model.ssm_
            ax.text(0.5, 0.7, f"Final tr(T) = {np.trace(ssm.T_):.4f}", 
                   transform=ax.transAxes, ha='center', fontsize=10)
            ax.text(0.5, 0.5, f"Final tr(Q) = {np.trace(ssm.Q_):.4f}",
                   transform=ax.transAxes, ha='center', fontsize=10)
            ax.text(0.5, 0.3, f"Final tr(H) = {np.trace(ssm.H_):.4f}",
                   transform=ax.transAxes, ha='center', fontsize=10)
            ax.set_title('Final Parameter Values')
        ax.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Convergence plot saved to {save_path}")
            
        plt.show()
        
    def spatial_residual_map(
        self,
        residuals: NDArray,
        coordinates: NDArray,
        save_path: Optional[Union[str, Path]] = None,
        figsize: Tuple[int, int] = (10, 8),
    ) -> None:
        """Plot spatial distribution of residuals.
        
        Parameters
        ----------
        residuals : NDArray
            Residual values for each location
        coordinates : NDArray
            Location coordinates, shape (n_locations, 2)
        save_path : str or Path, optional
            Path to save figure
        figsize : tuple
            Figure size
        """
        import matplotlib.pyplot as plt
        
        residuals = np.asarray(residuals).flatten()
        coordinates = np.asarray(coordinates)
        
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # 1. Mean residual by location
        ax = axes[0]
        scatter = ax.scatter(
            coordinates[:, 0], coordinates[:, 1],
            c=residuals, cmap='RdBu_r', s=20, alpha=0.7,
            vmin=-np.percentile(np.abs(residuals), 95),
            vmax=np.percentile(np.abs(residuals), 95),
        )
        plt.colorbar(scatter, ax=ax, label='Residual')
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_title('Spatial Distribution of Residuals')
        
        # 2. Absolute residual (error magnitude)
        ax = axes[1]
        scatter = ax.scatter(
            coordinates[:, 0], coordinates[:, 1],
            c=np.abs(residuals), cmap='YlOrRd', s=20, alpha=0.7,
        )
        plt.colorbar(scatter, ax=ax, label='|Residual|')
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_title('Spatial Distribution of Absolute Errors')
        
        plt.tight_layout()
        
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Spatial residual map saved to {save_path}")
            
        plt.show()
        
    def summary_report(
        self,
        y_true: Optional[NDArray] = None,
        y_pred: Optional[NDArray] = None,
        y_std: Optional[NDArray] = None,
    ) -> str:
        """Generate comprehensive summary report.
        
        Parameters
        ----------
        y_true : NDArray, optional
            True values
        y_pred : NDArray, optional
            Predicted values
        y_std : NDArray, optional
            Prediction standard deviations
            
        Returns
        -------
        str
            Formatted summary report
        """
        # Use stored values if not provided
        y_true = y_true if y_true is not None else self._y_true
        y_pred = y_pred if y_pred is not None else self._y_pred
        y_std = y_std if y_std is not None else self._y_std
        
        lines = [
            "=" * 60,
            "MODEL EVALUATION SUMMARY",
            "=" * 60,
            "",
        ]
        
        if y_true is not None and y_pred is not None:
            # Accuracy metrics
            accuracy = self.compute_accuracy(y_true, y_pred)
            lines.extend([
                "ACCURACY METRICS",
                "-" * 40,
                f"  RMSE:        {accuracy.rmse:.4f}",
                f"  MAE:         {accuracy.mae:.4f}",
                f"  MBE:         {accuracy.mbe:.4f}",
                f"  R²:          {accuracy.r2:.4f}",
                f"  Correlation: {accuracy.correlation:.4f}",
                "",
            ])
            
            # Calibration metrics (if std available)
            if y_std is not None:
                calibration = self.compute_calibration(y_true, y_pred, y_std)
                lines.extend([
                    "UNCERTAINTY CALIBRATION",
                    "-" * 40,
                    "  Coverage probabilities:",
                ])
                for level, cov in calibration.coverage.items():
                    lines.append(f"    {level}: {cov:.1%}")
                lines.extend([
                    f"  Mean interval width: {calibration.mean_interval_width:.4f}",
                    f"  Interval skill score: {calibration.interval_skill_score:.4f}",
                    f"  CRPS: {calibration.crps:.4f}",
                    "",
                ])
                
            # Residual diagnostics
            diagnostics = self.compute_residual_diagnostics()
            lines.extend([
                "RESIDUAL DIAGNOSTICS",
                "-" * 40,
                f"  Mean:     {diagnostics.mean:.4e}",
                f"  Std:      {diagnostics.std:.4f}",
                f"  Skewness: {diagnostics.skewness:.4f}",
                f"  Kurtosis: {diagnostics.kurtosis:.4f}",
                "",
                "  Normality test (Shapiro-Wilk):",
                f"    Statistic: {diagnostics.shapiro_statistic:.4f}",
                f"    P-value:   {diagnostics.shapiro_pvalue:.4e}",
                "",
                "  Autocorrelation test (Ljung-Box):",
                f"    Statistic: {diagnostics.ljung_box_statistic:.4f}",
                f"    P-value:   {diagnostics.ljung_box_pvalue:.4e}",
                "",
            ])
            
        # Model info (if available)
        if self.model is not None:
            lines.extend([
                "MODEL INFORMATION",
                "-" * 40,
            ])
            
            if hasattr(self.model, 'gam_') and self.model.gam_ is not None:
                gam_summary = self.model.gam_.summary()
                lines.extend([
                    "  GAM Component:",
                    f"    R²: {gam_summary.r_squared:.4f}",
                    f"    Features: {gam_summary.n_features}",
                ])
                
            if hasattr(self.model, 'ssm_') and self.model.ssm_ is not None:
                ssm_diag = self.model.ssm_.get_diagnostics()
                lines.extend([
                    "  SSM Component:",
                    f"    Log-likelihood: {ssm_diag.log_likelihood:.4e}",
                    f"    AIC: {ssm_diag.aic:.4e}",
                    f"    BIC: {ssm_diag.bic:.4e}",
                    f"    EM converged: {ssm_diag.em_converged}",
                    f"    EM iterations: {ssm_diag.em_iterations}",
                ])
                
        lines.append("=" * 60)
        
        report = "\n".join(lines)
        print(report)
        return report
        
    def compare_models(
        self,
        models: Dict[str, Tuple[NDArray, NDArray]],
        y_true: NDArray,
    ) -> pd.DataFrame:
        """Compare multiple models.
        
        Parameters
        ----------
        models : dict
            Dictionary mapping model names to (y_pred, y_std) tuples
        y_true : NDArray
            True values
            
        Returns
        -------
        pd.DataFrame
            Comparison table with metrics for each model
        """
        results = []
        
        for name, (y_pred, y_std) in models.items():
            accuracy = self.compute_accuracy(y_true, y_pred)
            row = {'Model': name}
            row.update(accuracy.to_dict())
            
            if y_std is not None:
                calibration = self.compute_calibration(y_true, y_pred, y_std)
                row['coverage_95'] = calibration.coverage['95%']
                row['interval_width'] = calibration.mean_interval_width
                row['crps'] = calibration.crps
                
            results.append(row)
            
        return pd.DataFrame(results).set_index('Model')
