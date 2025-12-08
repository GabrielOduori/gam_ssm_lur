"""
Lightweight profiling script to surface bottlenecks in the hybrid pipeline.

Usage
-----
PYTHONPATH=src python experiments/profile_performance.py \
    --n-locs 40 --n-times 80 --n-features 8 --mode dense

Flags let you scale the workload; defaults keep runs short. Set
PROFILE_OUT=prof.out to write a cProfile dump for deeper analysis
with snakeviz or pstats.
"""

from __future__ import annotations

import argparse
import cProfile
import pstats
import time

import numpy as np

# from gam_ssm_lur.hybrid_model import HybridGAMSSM

from gam_ssm_lur import HybridGAMSSM


def generate_synthetic(
    n_locations: int, n_times: int, n_features: int, random_state: int = 42
):
    rng = np.random.default_rng(random_state)
    X = rng.standard_normal((n_locations * n_times, n_features))
    true_coef = rng.standard_normal(n_features) * 0.5
    spatial = X @ true_coef
    temporal = np.zeros(n_times)
    for t in range(1, n_times):
        temporal[t] = 0.85 * temporal[t - 1] + rng.normal(scale=0.25)
    temporal_expanded = np.tile(temporal, n_locations)
    y = spatial + temporal_expanded + rng.normal(scale=0.4, size=X.shape[0])
    time_idx = np.repeat(np.arange(n_times), n_locations)
    loc_idx = np.tile(np.arange(n_locations), n_times)
    return X, y, time_idx, loc_idx


def main(args: argparse.Namespace) -> None:
    X, y, t_idx, l_idx = generate_synthetic(
        args.n_locs, args.n_times, args.n_features, args.seed
    )

    model = HybridGAMSSM(
        n_splines=args.n_splines,
        em_max_iter=args.em_max_iter,
        em_tol=args.em_tol,
        scalability_mode=args.mode,
        random_state=args.seed,
    )

    def timed(step, fn):
        start = time.perf_counter()
        out = fn()
        return out, time.perf_counter() - start

    # Fit with profiling if requested
    prof = cProfile.Profile()
    if args.profile_out:
        prof.enable()

    (_, fit_seconds) = timed(
        "fit", lambda: model.fit(X, y, time_index=t_idx, location_index=l_idx)
    )

    if args.profile_out:
        prof.disable()
        prof.dump_stats(args.profile_out)

    (_, predict_seconds) = timed("predict", model.predict)

    metrics = model.evaluate(model._y_matrix.flatten(), model.predict().total.flatten())

    print("=== Timing (s) ===")
    print(f"fit:      {fit_seconds:8.3f}")
    print(f"predict:  {predict_seconds:8.3f}")
    print("=== Metrics ===")
    print(f"RMSE: {metrics['rmse']:.3f}  R2: {metrics['r2']:.3f}")

    if args.profile_out:
        print(f"Profile written to {args.profile_out}. Top 20 calls:")
        stats = pstats.Stats(args.profile_out)
        stats.strip_dirs().sort_stats(pstats.SortKey.CUMULATIVE).print_stats(20)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Profile HybridGAMSSM performance.")
    parser.add_argument("--n-locs", type=int, default=30, dest="n_locs")
    parser.add_argument("--n-times", type=int, default=60, dest="n_times")
    parser.add_argument("--n-features", type=int, default=6, dest="n_features")
    parser.add_argument("--n-splines", type=int, default=8)
    parser.add_argument("--em-max-iter", type=int, default=15, dest="em_max_iter")
    parser.add_argument("--em-tol", type=float, default=1e-5)
    parser.add_argument(
        "--mode",
        choices=["auto", "dense", "diagonal", "block"],
        default="dense",
        help="Kalman scalability mode",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--profile-out",
        dest="profile_out",
        default=None,
        help="Path to write cProfile stats (optional)",
    )
    main(parser.parse_args())
