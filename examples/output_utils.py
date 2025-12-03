"""
Utility helpers for organizing example outputs in timestamped experiment folders.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, Tuple


def make_experiment_dirs(
    base: Path | str = "outputs",
    groups: Iterable[str] = (),
    run_name: str | None = None,
) -> Tuple[Path, Dict[str, Path]]:
    """
    Create a unique experiment directory with optional subdirectories.

    Parameters
    ----------
    base : Path | str
        Root directory where experiments are stored (e.g., "outputs" or "results").
    groups : Iterable[str]
        Named subfolders to create under the experiment root (e.g., ["figures", "models"]).
    run_name : str, optional
        Custom experiment name. If not provided, uses experiment_YYYYmmdd_HHMMSS.

    Returns
    -------
    experiment_root : Path
        The root directory for this run (base / experiment_<timestamp>).
    group_dirs : Dict[str, Path]
        Mapping of group name to its Path under the experiment root.
    """
    base_path = Path(base)
    name = run_name or f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    experiment_root = base_path / name
    experiment_root.mkdir(parents=True, exist_ok=True)

    group_dirs: Dict[str, Path] = {}
    for group in groups:
        group_path = experiment_root / group
        group_path.mkdir(parents=True, exist_ok=True)
        group_dirs[group] = group_path

    return experiment_root, group_dirs
