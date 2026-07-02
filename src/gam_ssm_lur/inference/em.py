"""
EM algorithm for linear Gaussian state-space model parameter estimation,
following Shumway & Stoffer (1982). E-step runs Kalman filter + RTS smoother;
M-step updates {T, Q, H, alpha_0, P_0} in closed form.

Supports a "dynamic_dim < state_dim" split: the leading sub-block evolves
under T/Q as usual, the trailing block is held at identity transition with
exactly zero process noise -- i.e. fixed regression-effect states (Harvey,
1989, Ch. 3.3) rather than latent dynamics. None means everything is dynamic,
matching the original (pre-augmented-state) behaviour exactly.

Key references
----------
.. [1] Dempster, A. P., Laird, N. M., & Rubin, D. B. (1977). Maximum likelihood
       from incomplete data via the EM algorithm. JRSS B, 39(1), 1-22.
.. [2] Shumway, R. H., & Stoffer, D. S. (1982). An approach to time series
       smoothing and forecasting using the EM algorithm. JTSA, 3(4), 253-264.
.. [3] Ghahramani, Z., & Hinton, G. E. (1996). Parameter estimation for linear
       dynamical systems. CRG-TR-96-2, University of Toronto.
.. [4] Harvey, A. C. (1989). Forecasting, structural time series models and
       the Kalman filter. Cambridge University Press.
.. [5] Durbin, J., & Koopman, S. J. (2012). Time series analysis by state
       space methods. Oxford University Press.
.. [6] Hamilton, J. D. (1994). Time series analysis. Princeton University Press.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from gam_ssm_lur.inference.kalman import FilterResult, KalmanFilter
from gam_ssm_lur.inference.smoother import RTSSmoother, SmootherResult

logger = logging.getLogger(__name__)


@dataclass
class EMResult:
    """
    Final EM output: estimated {T, Q, H, alpha_0, P_0}, LL trace,
    and final smoothed states.
    """

    T: NDArray
    Q: NDArray
    H: NDArray
    initial_mean: NDArray
    initial_covariance: NDArray
    log_likelihoods: List[float]
    n_iterations: int
    converged: bool
    smoothed_states: NDArray
    smoothed_covariances: NDArray


@dataclass
class EMDiagnostics:
    """Per-iteration convergence record (LL, delta-LL, Frobenius param changes)."""

    iteration: int
    log_likelihood: float
    ll_change: float
    param_changes: Dict[str, float] = field(default_factory=dict)


class EMEstimator:
    """EM estimator for linear Gaussian SSM parameters {T, Q, H, alpha_0, P_0}."""

    def __init__(
        self,
        max_iterations: int = 100,
        tolerance: float = 1e-6,
        min_iterations: int = 5,
        regularization: float = 1e-6,
        estimate_T: bool = True,
        estimate_Q: bool = True,
        estimate_H: bool = True,
        estimate_initial: bool = True,
        diagonal_Q: bool = False,
        diagonal_H: bool = False,
        verbose: bool = True,
        dynamic_dim: Optional[int] = None,
        max_eigenvalue: float = 0.98,
    ):
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.min_iterations = min_iterations
        self.regularization = regularization
        self.estimate_T = estimate_T
        self.estimate_Q = estimate_Q
        self.estimate_H = estimate_H
        self.estimate_initial = estimate_initial
        self.diagonal_Q = diagonal_Q
        self.diagonal_H = diagonal_H
        self.verbose = verbose
        self.dynamic_dim = dynamic_dim
        # Unconstrained M-step T = S10 @ S00^-1 has no stability guarantee and
        # can return an explosive matrix on a short series. Stationarity of
        # alpha_{t+1} = T alpha_t + noise requires all eigenvalues of T inside
        # the unit circle (Hamilton, 1994, Ch. 1).
        self.max_eigenvalue = max_eigenvalue

        self.diagnostics_history: List[EMDiagnostics] = []

    def _stabilize_transition(self, T: NDArray) -> NDArray:
        """Uniformly shrink T toward 0 if its spectral radius exceeds max_eigenvalue.

        Scalar rescaling (vs. clipping individual eigenvalues) preserves T's
        eigenvector structure -- only the overall growth/decay rate changes.
        """
        eigvals = np.linalg.eigvals(T)
        rho = float(np.max(np.abs(eigvals)))
        if rho > self.max_eigenvalue and rho > 0:
            shrink = self.max_eigenvalue / rho
            logger.warning(
                "Transition matrix spectral radius %.4f exceeded stability "
                "threshold %.2f; shrinking uniformly by factor %.4f.",
                rho,
                self.max_eigenvalue,
                shrink,
            )
            T = T * shrink
        return T

    def _initialize_parameters(
        self,
        observations: NDArray,
        state_dim: int,
        obs_dim: int,
        T_init: Optional[NDArray] = None,
        Q_init: Optional[NDArray] = None,
        H_init: Optional[NDArray] = None,
    ) -> Tuple[NDArray, NDArray, NDArray, NDArray, NDArray, NDArray]:
        """Defaults: T = 0.95*I, Z = I, Q/H from obs variance, alpha_0 = 0, P_0 = I."""
        obs_var = np.var(observations, axis=0).mean()
        d = self.dynamic_dim if self.dynamic_dim is not None else state_dim

        if T_init is not None:
            T = T_init.copy()
        else:
            # static block (if any) stays exactly identity -- no dynamics
            T = np.eye(state_dim)
            T[:d, :d] = 0.95 * np.eye(d)

        Z = np.eye(obs_dim, state_dim)

        if Q_init is not None:
            Q = Q_init.copy()
        else:
            # static block gets exactly zero process noise
            Q = np.zeros((state_dim, state_dim))
            Q[:d, :d] = 0.1 * obs_var * np.eye(d)

        if H_init is not None:
            H = H_init.copy()
        else:
            H = 0.5 * obs_var * np.eye(obs_dim)

        initial_mean = np.zeros(state_dim)
        initial_cov = np.eye(state_dim)

        return T, Z, Q, H, initial_mean, initial_cov

    def _e_step(
        self, observations: NDArray, kf: KalmanFilter
    ) -> Tuple[FilterResult, SmootherResult]:
        """Kalman forward pass + RTS backward pass -> posterior over latent states."""
        filter_result = kf.filter(observations)
        smoother_result = RTSSmoother(kf).smooth(filter_result)
        return filter_result, smoother_result

    def _compute_sufficient_statistics(
        self,
        observations: NDArray,
        smoother_result: SmootherResult,
        Z: Optional[NDArray] = None,
    ) -> Dict[str, NDArray]:
        """S11/S11_trans/S10/S00/Sy1/Syy/S_zaz/S_zay sums feeding the M-step.

        S11_trans (t=1..T_len-1, excluding t=0) is kept separate from the full
        S11 (all t) -- Q is driven by transitions only, H by every observation.

        If Z is time-varying (T_len, obs_dim, state_dim), it must be applied
        per-timestep here, not to pooled S11/Sy1 afterwards -- pooling across
        Z_t would mix contributions from different observation matrices
        (Durbin & Koopman, 2012, Sec. 3.1). S_zaz/S_zay give the H update a
        single formula that's correct for fixed and time-varying Z alike.
        """
        T_len = observations.shape[0]
        state_dim = smoother_result.smoothed_means.shape[1]
        obs_dim = observations.shape[1]

        Z_is_time_varying = Z is not None and np.asarray(Z).ndim == 3
        if Z is None:
            Z = np.eye(obs_dim, state_dim)

        S11 = np.zeros((state_dim, state_dim))
        S11_trans = np.zeros((state_dim, state_dim))
        S_zaz = np.zeros((obs_dim, obs_dim))
        S_zay = np.zeros((obs_dim, obs_dim))
        S10 = np.zeros((state_dim, state_dim))
        S00 = np.zeros((state_dim, state_dim))
        Sy1 = np.zeros((obs_dim, state_dim))
        Syy = np.zeros((obs_dim, obs_dim))

        for t in range(T_len):
            alpha_t = smoother_result.smoothed_means[t]
            P_t = smoother_result.smoothed_covariances[t]
            y_t = observations[t]

            P_t_full = np.diag(P_t) if P_t.ndim == 1 else P_t

            E_alpha_alpha = P_t_full + np.outer(alpha_t, alpha_t)
            S11 += E_alpha_alpha
            Sy1 += np.outer(y_t, alpha_t)
            Syy += np.outer(y_t, y_t)

            Z_t = Z[t] if Z_is_time_varying else Z
            S_zaz += Z_t @ E_alpha_alpha @ Z_t.T
            S_zay += np.outer(Z_t @ alpha_t, y_t)

            if t > 0:
                # belongs to transition (t-1 -> t); first state has none (Shumway
                # & Stoffer, 1982, Sec. 3 -- process-noise sum excludes the first state)
                S11_trans += E_alpha_alpha

                alpha_tm1 = smoother_result.smoothed_means[t - 1]
                P_tm1 = smoother_result.smoothed_covariances[t - 1]
                cross_cov = smoother_result.cross_covariances[t]

                P_tm1_full = np.diag(P_tm1) if P_tm1.ndim == 1 else P_tm1
                cross_cov_full = (
                    np.diag(cross_cov) if cross_cov.ndim == 1 else cross_cov
                )

                S10 += cross_cov_full + np.outer(alpha_t, alpha_tm1)
                S00 += P_tm1_full + np.outer(alpha_tm1, alpha_tm1)

        return {
            "S11": S11,
            "S11_trans": S11_trans,
            "S10": S10,
            "S00": S00,
            "Sy1": Sy1,
            "Syy": Syy,
            "S_zaz": S_zaz,
            "S_zay": S_zay,
            "T_len": T_len,
        }

    def _m_step(
        self,
        observations: NDArray,
        smoother_result: SmootherResult,
        current_T: NDArray,
        current_Q: NDArray,
        current_H: NDArray,
        current_Z: NDArray,
    ) -> Tuple[NDArray, NDArray, NDArray, NDArray, NDArray]:
        """T = S10 S00^-1;  Q = (S11_trans - T S10')/(T_len-1);
        H = (Syy - 2 S_zay + S_zaz)/T_len.

        (Shumway & Stoffer, 1982, Sec. 3.) T/Q updates are restricted to the
        leading dynamic_dim x dynamic_dim block; the trailing "fixed-effect"
        block (Harvey, 1989, Ch. 3.3; Durbin & Koopman, 2012, Sec. 3.2.2)
        stays identity-transition / zero-noise and is never touched here.
        """
        T_len = observations.shape[0]
        state_dim = current_T.shape[0]
        d = self.dynamic_dim if self.dynamic_dim is not None else state_dim

        stats = self._compute_sufficient_statistics(
            observations, smoother_result, Z=current_Z
        )

        if self.estimate_T:
            S00_dd = stats["S00"][:d, :d] + self.regularization * np.eye(d)
            try:
                T_dd = stats["S10"][:d, :d] @ np.linalg.inv(S00_dd)
            except np.linalg.LinAlgError:
                T_dd = stats["S10"][:d, :d] @ np.linalg.pinv(S00_dd)
            T_dd = self._stabilize_transition(T_dd)
            new_T = np.eye(state_dim)
            new_T[:d, :d] = T_dd
        else:
            new_T = current_T

        if self.estimate_Q:
            S11_trans_dd = stats["S11_trans"][:d, :d]
            S10_dd = stats["S10"][:d, :d]
            Q_dd = (S11_trans_dd - new_T[:d, :d] @ S10_dd.T) / (T_len - 1)
            Q_dd = 0.5 * (Q_dd + Q_dd.T)
            eigvals, eigvecs = np.linalg.eigh(Q_dd)
            eigvals = np.maximum(eigvals, self.regularization)
            Q_dd = eigvecs @ np.diag(eigvals) @ eigvecs.T

            if self.diagonal_Q:
                Q_dd = np.diag(np.diag(Q_dd))

            new_Q = np.zeros((state_dim, state_dim))
            new_Q[:d, :d] = Q_dd
        else:
            new_Q = current_Q

        if self.estimate_H:
            new_H = (stats["Syy"] - 2 * stats["S_zay"] + stats["S_zaz"]) / T_len
            new_H = 0.5 * (new_H + new_H.T)
            eigvals, eigvecs = np.linalg.eigh(new_H)
            eigvals = np.maximum(eigvals, self.regularization)
            new_H = eigvecs @ np.diag(eigvals) @ eigvecs.T

            if self.diagonal_H:
                new_H = np.diag(np.diag(new_H))
        else:
            new_H = current_H

        if self.estimate_initial:
            new_initial_mean = smoother_result.smoothed_means[0]
            P0 = smoother_result.smoothed_covariances[0]
            new_initial_cov = np.diag(P0) if P0.ndim == 1 else P0
        else:
            new_initial_mean = np.zeros(state_dim)
            new_initial_cov = np.eye(state_dim)

        return new_T, new_Q, new_H, new_initial_mean, new_initial_cov

    def _check_convergence(self, log_likelihoods: List[float], iteration: int) -> bool:
        """Relative-LL stopping rule: |dLL| / max(1, |LL|) < tolerance."""
        if iteration < self.min_iterations:
            return False
        if len(log_likelihoods) < 2:
            return False

        ll_change = log_likelihoods[-1] - log_likelihoods[-2]
        if ll_change < -1e-4 * max(1.0, abs(log_likelihoods[-2])):
            logger.warning(
                "Log-likelihood decreased at iteration %d: %.6e", iteration, ll_change
            )

        relative_change = abs(ll_change) / max(1.0, abs(log_likelihoods[-1]))
        return relative_change < self.tolerance

    def _compute_param_changes(
        self,
        T_old: NDArray,
        T_new: NDArray,
        Q_old: NDArray,
        Q_new: NDArray,
        H_old: NDArray,
        H_new: NDArray,
    ) -> Dict[str, float]:
        return {
            "T": np.linalg.norm(T_new - T_old, "fro"),
            "Q": np.linalg.norm(Q_new - Q_old, "fro"),
            "H": np.linalg.norm(H_new - H_old, "fro"),
        }

    def fit(
        self,
        observations: NDArray,
        state_dim: Optional[int] = None,
        obs_dim: Optional[int] = None,
        T_init: Optional[NDArray] = None,
        Q_init: Optional[NDArray] = None,
        H_init: Optional[NDArray] = None,
        Z: Optional[NDArray] = None,
        scalability_mode: Literal["auto", "dense", "diagonal", "block"] = "auto",
    ) -> EMResult:
        """Run EM to convergence. observations: (T_len, obs_dim). Z fixed
        (obs_dim, state_dim) or time-varying (T_len, obs_dim, state_dim) --
        the latter requires scalability_mode='dense' (enforced by KalmanFilter).
        """
        T_len, obs_dim_data = observations.shape
        obs_dim = obs_dim or obs_dim_data
        state_dim = state_dim or obs_dim

        logger.info(
            f"Starting EM with T={T_len}, state_dim={state_dim}, obs_dim={obs_dim}"
        )

        T, Z_default, Q, H, initial_mean, initial_cov = self._initialize_parameters(
            observations, state_dim, obs_dim, T_init, Q_init, H_init
        )
        Z = Z_default if Z is None else np.asarray(Z, dtype=float)

        kf = KalmanFilter(
            state_dim=state_dim,
            obs_dim=obs_dim,
            mode=scalability_mode,
            regularization=self.regularization,
        )

        log_likelihoods: List[float] = []
        converged = False

        for iteration in range(self.max_iterations):
            kf.initialize(
                T=T,
                Z=Z,
                Q=Q,
                H=H,
                initial_mean=initial_mean,
                initial_covariance=initial_cov,
            )

            filter_result, smoother_result = self._e_step(observations, kf)
            log_likelihoods.append(filter_result.log_likelihood)

            T_old, Q_old, H_old = T.copy(), Q.copy(), H.copy()
            T, Q, H, initial_mean, initial_cov = self._m_step(
                observations, smoother_result, T, Q, H, Z
            )

            param_changes = self._compute_param_changes(T_old, T, Q_old, Q, H_old, H)
            ll_change = (
                log_likelihoods[-1] - log_likelihoods[-2]
                if len(log_likelihoods) > 1
                else float("inf")
            )

            diagnostics = EMDiagnostics(
                iteration=iteration,
                log_likelihood=log_likelihoods[-1],
                ll_change=ll_change,
                param_changes=param_changes,
            )
            self.diagnostics_history.append(diagnostics)

            if self.verbose:
                logger.info(
                    f"Iteration {iteration + 1}: "
                    f"LL={log_likelihoods[-1]:.4e}, "
                    f"ΔLL={ll_change:.4e}, "
                    f"||ΔT||={param_changes['T']:.4e}"
                )

            if self._check_convergence(log_likelihoods, iteration):
                converged = True
                if self.verbose:
                    logger.info(f"Converged after {iteration + 1} iterations")
                break

        if not converged and self.verbose:
            logger.warning(
                f"EM did not converge after {self.max_iterations} iterations"
            )

        return EMResult(
            T=T,
            Q=Q,
            H=H,
            initial_mean=initial_mean,
            initial_covariance=initial_cov,
            log_likelihoods=log_likelihoods,
            n_iterations=len(log_likelihoods),
            converged=converged,
            smoothed_states=smoother_result.smoothed_means,
            smoothed_covariances=smoother_result.smoothed_covariances,
        )

    def get_diagnostics_dataframe(self):
        import pandas as pd

        records = []
        for d in self.diagnostics_history:
            record = {
                "iteration": d.iteration,
                "log_likelihood": d.log_likelihood,
                "ll_change": d.ll_change,
            }
            record.update({f"delta_{k}": v for k, v in d.param_changes.items()})
            records.append(record)

        return pd.DataFrame(records)
