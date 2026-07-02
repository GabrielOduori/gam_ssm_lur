"""Model evaluation, calibration, and residual diagnostics."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


@dataclass
class AccuracyMetrics:
    """Accuracy metrics container."""

    rmse: float
    mae: float
    mbe: float
    r2: float
    correlation: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "rmse": self.rmse,
            "mae": self.mae,
            "mbe": self.mbe,
            "r2": self.r2,
            "correlation": self.correlation,
        }


@dataclass
class CalibrationMetrics:
    """Uncertainty calibration metrics container."""

    coverage: Dict[str, float]
    mean_interval_width: float
    interval_skill_score: float
    crps: float


@dataclass
class ResidualDiagnostics:
    """Residual diagnostic statistics."""

    mean: float
    std: float
    kurtosis: float
    acf: NDArray


class ModelEvaluator:
    """Model evaluation and diagnostics."""

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
        """Compute RMSE/MAE/MBE/R²/correlation."""
        y_true = np.asarray(y_true).flatten()
        y_pred = np.asarray(y_pred).flatten()

        self._y_true = y_true
        self._y_pred = y_pred
        self._residuals = y_true - y_pred

        rmse = np.sqrt(np.mean(self._residuals**2))
        mae = np.mean(np.abs(self._residuals))
        mbe = np.mean(self._residuals)

        ss_res = np.sum(self._residuals**2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

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
        """Compute coverage, ISS, CRPS at multiple quantile levels."""
        from scipy import stats

        y_true = np.asarray(y_true).flatten()
        y_pred = np.asarray(y_pred).flatten()
        y_std = np.asarray(y_std).flatten()

        self._y_std = y_std

        coverage = {}
        for level in [0.50, 0.80, 0.90, 0.95]:
            z = stats.norm.ppf((1 + level) / 2)
            lower = y_pred - z * y_std
            upper = y_pred + z * y_std
            coverage[f"{int(level * 100)}%"] = np.mean(
                (y_true >= lower) & (y_true <= upper)
            )

        z_95 = stats.norm.ppf(0.975)
        interval_width = np.mean(2 * z_95 * y_std)

        alpha = 0.05
        lower_95 = y_pred - z_95 * y_std
        upper_95 = y_pred + z_95 * y_std
        width = upper_95 - lower_95
        penalty_lower = (2 / alpha) * (lower_95 - y_true) * (y_true < lower_95)
        penalty_upper = (2 / alpha) * (y_true - upper_95) * (y_true > upper_95)
        iss = np.mean(width + penalty_lower + penalty_upper)

        z = (y_true - y_pred) / y_std
        crps = np.mean(
            y_std
            * (
                z * (2 * stats.norm.cdf(z) - 1)
                + 2 * stats.norm.pdf(z)
                - 1 / np.sqrt(np.pi)
            )
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
        """Compute mean/std/kurtosis/ACF of residuals."""
        from scipy import stats

        if residuals is None:
            if self._residuals is None:
                raise ValueError("No residuals available. Call compute_accuracy first.")
            residuals = self._residuals
        else:
            residuals = np.asarray(residuals).flatten()

        return ResidualDiagnostics(
            mean=np.mean(residuals),
            std=np.std(residuals),
            kurtosis=stats.kurtosis(residuals),
            acf=self._compute_acf(residuals, max_lag),
        )

    def _compute_acf(self, x: NDArray, max_lag: int) -> NDArray:
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
                acf[lag] = np.mean((x[: n - lag] - mean) * (x[lag:] - mean)) / var

        return acf

    def diagnostic_plots(
        self,
        y_true: Optional[NDArray] = None,
        y_pred: Optional[NDArray] = None,
        y_std: Optional[NDArray] = None,
        save_path: Optional[Union[str, Path]] = None,
        figsize: Tuple[int, int] = (12, 10),
    ) -> None:
        """2×3 diagnostic panel: obs/pred, residuals, Q-Q, hist, ACF, temporal."""
        import matplotlib.pyplot as plt
        from scipy import stats

        y_true = y_true if y_true is not None else self._y_true
        y_pred = y_pred if y_pred is not None else self._y_pred

        if y_true is None or y_pred is None:
            raise ValueError(
                "No predictions available. Call compute_accuracy first or provide data."
            )

        y_true = np.asarray(y_true).flatten()
        y_pred = np.asarray(y_pred).flatten()
        residuals = y_true - y_pred

        fig, axes = plt.subplots(2, 3, figsize=figsize)

        ax = axes[0, 0]
        ax.scatter(y_pred, y_true, alpha=0.5, s=10, c="steelblue")
        min_val = min(y_pred.min(), y_true.min())
        max_val = max(y_pred.max(), y_true.max())
        ax.plot([min_val, max_val], [min_val, max_val], "r--", lw=2, label="1:1 line")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Observed")
        ax.set_title(
            f"Observed vs Predicted\n(r = {np.corrcoef(y_true, y_pred)[0, 1]:.3f})"
        )
        ax.legend()

        ax = axes[0, 1]
        ax.scatter(y_pred, residuals, alpha=0.5, s=10, c="steelblue")
        ax.axhline(y=0, color="r", linestyle="--", lw=2)
        ax.set_xlabel("Fitted Values")
        ax.set_ylabel("Residuals")
        ax.set_title("Residuals vs Fitted")
        try:
            from scipy.ndimage import uniform_filter1d

            sorted_idx = np.argsort(y_pred)
            smoothed = uniform_filter1d(
                residuals[sorted_idx], size=max(len(residuals) // 20, 10)
            )
            ax.plot(y_pred[sorted_idx], smoothed, "orange", lw=2, label="Smoothed")
            ax.legend()
        except Exception:
            pass

        ax = axes[0, 2]
        stats.probplot(residuals, dist="norm", plot=ax)
        ax.set_title("Normal Q-Q Plot")
        ax.get_lines()[0].set_markersize(3)
        ax.get_lines()[0].set_alpha(0.5)

        ax = axes[1, 0]
        ax.hist(
            residuals,
            bins=50,
            density=True,
            alpha=0.7,
            color="steelblue",
            edgecolor="white",
        )
        x_range = np.linspace(residuals.min(), residuals.max(), 100)
        ax.plot(
            x_range,
            stats.norm.pdf(x_range, np.mean(residuals), np.std(residuals)),
            "r-",
            lw=2,
            label="Normal",
        )
        ax.set_xlabel("Residuals")
        ax.set_ylabel("Density")
        ax.set_title(f"Residual Distribution\n(kurt={stats.kurtosis(residuals):.2f})")
        ax.legend()

        ax = axes[1, 1]
        max_lag = min(50, len(residuals) // 4)
        acf = self._compute_acf(residuals, max_lag)
        ax.bar(range(len(acf)), acf, color="steelblue", edgecolor="white")
        ci = 1.96 / np.sqrt(len(residuals))
        ax.axhline(y=ci, color="r", linestyle="--", lw=1)
        ax.axhline(y=-ci, color="r", linestyle="--", lw=1)
        ax.axhline(y=0, color="k", lw=0.5)
        ax.set_xlabel("Lag")
        ax.set_ylabel("ACF")
        ax.set_title("Residual Autocorrelation")

        ax = axes[1, 2]
        if len(residuals) > 100:
            window = max(len(residuals) // 50, 10)
            rolling_mean = (
                pd.Series(residuals).rolling(window=window, center=True).mean()
            )
            rolling_std = pd.Series(residuals).rolling(window=window, center=True).std()
            time_idx = np.arange(len(residuals))
            ax.fill_between(
                time_idx,
                rolling_mean - rolling_std,
                rolling_mean + rolling_std,
                alpha=0.3,
                color="steelblue",
                label="±1 SD",
            )
            ax.plot(time_idx, rolling_mean, "b-", lw=1.5, label="Rolling mean")
            ax.axhline(y=0, color="r", linestyle="--", lw=1)
            ax.set_xlabel("Time Index")
            ax.set_ylabel("Residual")
            ax.set_title("Temporal Residual Pattern")
            ax.legend()
        else:
            ax.plot(residuals, "o-", markersize=3, alpha=0.7)
            ax.axhline(y=0, color="r", linestyle="--")
            ax.set_xlabel("Index")
            ax.set_ylabel("Residual")
            ax.set_title("Residuals Over Time")

        plt.tight_layout()

        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"Diagnostic plots saved to {save_path}")

        plt.show()

    def convergence_plot(
        self,
        log_likelihoods: Optional[List[float]] = None,
        save_path: Optional[Union[str, Path]] = None,
        figsize: Tuple[int, int] = (12, 4),
    ) -> None:
        """EM convergence: log-lik trace, increment, final parameters."""
        import matplotlib.pyplot as plt

        if log_likelihoods is None:
            if self.model is None:
                raise ValueError("No model or log-likelihoods provided")
            log_likelihoods = self.model.ssm_.em_result_.log_likelihoods

        fig, axes = plt.subplots(1, 3, figsize=figsize)
        iterations = range(len(log_likelihoods))

        ax = axes[0]
        ax.plot(iterations, log_likelihoods, "b-o", markersize=4)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Log-Likelihood")
        ax.set_title("EM Convergence: Log-Likelihood")
        ax.grid(True, alpha=0.3)

        ax = axes[1]
        if len(log_likelihoods) > 1:
            ll_diff = np.diff(log_likelihoods)
            ax.semilogy(
                range(1, len(log_likelihoods)), np.abs(ll_diff), "g-o", markersize=4
            )
            ax.axhline(y=1e-6, color="r", linestyle="--", label="Tolerance")
            ax.set_xlabel("Iteration")
            ax.set_ylabel("|ΔLog-Likelihood|")
            ax.set_title("EM Convergence: Increment")
            ax.legend()
            ax.grid(True, alpha=0.3)

        ax = axes[2]
        if self.model is not None and hasattr(self.model, "ssm_"):
            ssm = self.model.ssm_
            ax.text(
                0.5,
                0.7,
                f"Final tr(T) = {np.trace(ssm.T_):.4f}",
                transform=ax.transAxes,
                ha="center",
                fontsize=10,
            )
            ax.text(
                0.5,
                0.5,
                f"Final tr(Q) = {np.trace(ssm.Q_):.4f}",
                transform=ax.transAxes,
                ha="center",
                fontsize=10,
            )
            ax.text(
                0.5,
                0.3,
                f"Final tr(H) = {np.trace(ssm.H_):.4f}",
                transform=ax.transAxes,
                ha="center",
                fontsize=10,
            )
            ax.set_title("Final Parameter Values")
        ax.axis("off")

        plt.tight_layout()

        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"Convergence plot saved to {save_path}")

        plt.show()

    def spatial_residual_map(
        self,
        residuals: NDArray,
        coordinates: NDArray,
        save_path: Optional[Union[str, Path]] = None,
        figsize: Tuple[int, int] = (10, 8),
    ) -> None:
        """Scatter map of residuals by (lon, lat)."""
        import matplotlib.pyplot as plt

        residuals = np.asarray(residuals).flatten()
        coordinates = np.asarray(coordinates)

        fig, axes = plt.subplots(1, 2, figsize=figsize)

        ax = axes[0]
        scatter = ax.scatter(
            coordinates[:, 0],
            coordinates[:, 1],
            c=residuals,
            cmap="RdBu_r",
            s=20,
            alpha=0.7,
            vmin=-np.percentile(np.abs(residuals), 95),
            vmax=np.percentile(np.abs(residuals), 95),
        )
        plt.colorbar(scatter, ax=ax, label="Residual")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_title("Spatial Distribution of Residuals")

        ax = axes[1]
        scatter = ax.scatter(
            coordinates[:, 0],
            coordinates[:, 1],
            c=np.abs(residuals),
            cmap="YlOrRd",
            s=20,
            alpha=0.7,
        )
        plt.colorbar(scatter, ax=ax, label="|Residual|")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_title("Spatial Distribution of Absolute Errors")

        plt.tight_layout()

        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"Spatial residual map saved to {save_path}")

        plt.show()

    def summary_report(
        self,
        y_true: Optional[NDArray] = None,
        y_pred: Optional[NDArray] = None,
        y_std: Optional[NDArray] = None,
    ) -> str:
        """Print and return a text summary of model accuracy/calibration/diagnostics."""
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
            accuracy = self.compute_accuracy(y_true, y_pred)
            lines.extend(
                [
                    "ACCURACY METRICS",
                    "-" * 40,
                    f"  RMSE:        {accuracy.rmse:.4f}",
                    f"  MAE:         {accuracy.mae:.4f}",
                    f"  MBE:         {accuracy.mbe:.4f}",
                    f"  R²:          {accuracy.r2:.4f}",
                    f"  Correlation: {accuracy.correlation:.4f}",
                    "",
                ]
            )

            if y_std is not None:
                calibration = self.compute_calibration(y_true, y_pred, y_std)
                lines.extend(
                    [
                        "UNCERTAINTY CALIBRATION",
                        "-" * 40,
                        "  Coverage probabilities:",
                    ]
                )
                for level, cov in calibration.coverage.items():
                    lines.append(f"    {level}: {cov:.1%}")
                lines.extend(
                    [
                        f"  Mean interval width: {calibration.mean_interval_width:.4f}",
                        f"  Interval skill score: {calibration.interval_skill_score:.4f}",  # noqa: E501
                        f"  CRPS: {calibration.crps:.4f}",
                        "",
                    ]
                )

            diagnostics = self.compute_residual_diagnostics()
            lines.extend(
                [
                    "RESIDUAL DIAGNOSTICS",
                    "-" * 40,
                    f"  Mean:     {diagnostics.mean:.4e}",
                    f"  Std:      {diagnostics.std:.4f}",
                    f"  Kurtosis: {diagnostics.kurtosis:.4f}",
                    "",
                ]
            )

        if self.model is not None:
            lines.extend(["MODEL INFORMATION", "-" * 40])

            if hasattr(self.model, "gam_") and self.model.gam_ is not None:
                gam_summary = self.model.gam_.summary()
                lines.extend(
                    [
                        "  GAM Component:",
                        f"    R²: {gam_summary.r_squared:.4f}",
                        f"    Features: {gam_summary.n_features}",
                    ]
                )

            if hasattr(self.model, "ssm_") and self.model.ssm_ is not None:
                ssm_diag = self.model.ssm_.get_diagnostics()
                lines.extend(
                    [
                        "  SSM Component:",
                        f"    Log-likelihood: {ssm_diag.log_likelihood:.4e}",
                        f"    AIC: {ssm_diag.aic:.4e}",
                        f"    BIC: {ssm_diag.bic:.4e}",
                        f"    EM converged: {ssm_diag.em_converged}",
                        f"    EM iterations: {ssm_diag.em_iterations}",
                    ]
                )

        lines.append("=" * 60)

        report = "\n".join(lines)
        print(report)
        return report

    def loocv_stations(
        self,
        station_obs: pd.DataFrame,
        station_col: str = "station_id",
        obs_col: str = "epa_no2",
        grid_id_col: str = "grid_id",
    ) -> pd.DataFrame:
        """Leave-one-station-out CV of the GAM spatial component."""
        if (
            self.model is None
            or not hasattr(self.model, "gam_")
            or self.model.gam_ is None
        ):
            raise RuntimeError(
                "Model has not been fitted.  Call model.fit() or "
                "model.fit_from_dataset() first."
            )

        X = self.model._X_train
        y = self.model._y_train
        loc_ids = list(self.model.location_ids_)

        station_summary = station_obs.groupby(
            [station_col, grid_id_col], as_index=False
        )[obs_col].mean()

        records = []
        for _, row in station_summary.iterrows():
            sid = row[station_col]
            gid = row[grid_id_col]
            y_obs = float(row[obs_col])

            if gid not in loc_ids:
                logger.warning(
                    "Station %s grid_id=%s not found in model locations — skipping",
                    sid,
                    gid,
                )
                continue

            idx = loc_ids.index(gid)

            mask = np.ones(len(loc_ids), dtype=bool)
            mask[idx] = False
            X_loo = X[mask]
            y_loo = y[mask]

            from gam_ssm_lur.models.spatial_gam import SpatialGAM

            gam_loo = SpatialGAM(
                n_splines=self.model.gam_.n_splines,
                lam=self.model.gam_.lam,
            )
            gam_loo.fit(X_loo, y_loo)
            y_pred = float(gam_loo.predict(X[[idx]])[0])

            records.append(
                {
                    station_col: sid,
                    grid_id_col: gid,
                    obs_col: y_obs,
                    "no2": y_pred,
                }
            )

        result = pd.DataFrame(records).rename(
            columns={obs_col: "obs_no2", station_col: "station_id"}
        )
        logger.info(
            "LOOCV complete: %d / %d stations evaluated",
            len(result),
            len(station_summary),
        )
        return result

    def compare_models(
        self,
        models: Dict[str, Tuple[NDArray, NDArray]],
        y_true: NDArray,
    ) -> pd.DataFrame:
        """Accuracy table comparing multiple (name → (pred, std)) models."""
        results = []
        for name, (y_pred, y_std) in models.items():
            accuracy = self.compute_accuracy(y_true, y_pred)
            row = {"Model": name}
            row.update(accuracy.to_dict())

            if y_std is not None:
                calibration = self.compute_calibration(y_true, y_pred, y_std)
                row["coverage_95"] = calibration.coverage["95%"]
                row["interval_width"] = calibration.mean_interval_width
                row["crps"] = calibration.crps

            results.append(row)

        return pd.DataFrame(results).set_index("Model")
