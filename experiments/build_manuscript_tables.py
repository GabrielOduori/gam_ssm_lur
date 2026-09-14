#!/usr/bin/env python
"""Assemble manuscript Tables 5-8 as LaTeX from experiment output CSVs.

Sources: buffer_ablation_v2/ (T5), run_*/station_temporal_r2.csv (T6),
run_*/model_comparison.csv + nested/forecast/ml/baseline CSVs (T7),
component_ablations/ (T8).

Usage:
    python experiments/build_manuscript_tables.py
"""

from __future__ import annotations

from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import pandas as pd

EXPERIMENTS = Path(__file__).resolve().parent
RESULTS = EXPERIMENTS / "results"
OUT = RESULTS / "manuscript_tables"


def f2(v):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "---"
    try:
        d = Decimal(str(float(v))).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
        return str(d)
    except (TypeError, ValueError, ArithmeticError):
        return "---"


def latest_run(fname):
    runs = sorted(RESULTS.glob(f"run_*/{fname}"))
    if not runs:
        raise FileNotFoundError(fname)
    return runs[-1]


def table5():
    df = pd.read_csv(RESULTS / "buffer_ablation" / "buffer_ablation.csv")
    df = df[df["feature_set"] != "wind_sector_paper31"]
    atmos = pd.read_csv(
        RESULTS / "buffer_ablation" / "loocv_vs_atmosplan.csv"
    ).set_index("arm")
    rows = []
    for _, r in df.iterrows():
        a = atmos.loc[r["feature_set"]]
        label = "Wind-sector" if r["feature_set"] == "wind_sector" else "Circular"
        rows.append(
            f"\t\t{label} & {int(r['n_candidate']):,} & {int(r['n_selected'])} & "
            f"{f2(r['fit_r2'])} & {f2(r['fit_rmse'])} & {f2(a['r2'])} & {f2(a['loocv_vs_atmos_rmse'])} \\\\"
        )
    body = "\n".join(rows)
    return (
        "\\begin{tabular}{l r r r r r r}\n\t\\toprule\n"
        "\tFeature set & Candidates & Selected & Fit $R^2$ & Fit RMSE & LOOCV $R^2$ & LOOCV RMSE \\\\\n"
        "\t\\midrule\n" + body + "\n\t\\bottomrule\n\\end{tabular}\n"
    )


def table6():
    df = pd.read_csv(latest_run("station_temporal_r2.csv"))
    rows = "\n".join(
        f"\t\t{r['station_id']} & {r['temporal_r2']:.2f} \\\\" for _, r in df.iterrows()
    )
    return (
        "\\begin{tabular}{l r}\n\t\\toprule\n\tStation ID & Temporal $R^2$ \\\\\n"
        "\t\\midrule\n" + rows + "\n\t\\bottomrule\n\\end{tabular}\n"
    )


COLMAP = {
    "model": "model",
    "rmse": "rmse",
    "mae": "mae",
    "mbe": "mbe",
    "r\u00b2": "r2",
    "r2": "r2",
    "r": "r",
    "coverage 50%": "coverage_50",
    "coverage_50": "coverage_50",
    "coverage 95%": "coverage_95",
    "coverage_95": "coverage_95",
    "interval width": "interval_width",
    "interval_width": "interval_width",
    "crps": "crps",
    "n": "n",
    "variant": "model",
}


def norm(df):
    return df.rename(columns={c: COLMAP.get(c.strip().lower(), c) for c in df.columns})


def table7():
    mc = norm(pd.read_csv(latest_run("model_comparison.csv")))
    nested = norm(
        pd.read_csv(RESULTS / "nested_loocv_run1" / "nested_vs_global_comparison.csv")
    )
    fc = norm(pd.read_csv(RESULTS / "forecast_and_k" / "forecast_eval.csv"))
    ml = norm(pd.read_csv(RESULTS / "ml_geo_benchmarks" / "ml_geo_benchmarks.csv"))
    bl = norm(pd.read_csv(RESULTS / "baseline_metrics.csv"))

    def row(label, d, cov50=None, cov95=None, width=None, crps=None):
        cells = [
            f2(d.get("rmse")),
            f2(d.get("mae")),
            f2(d.get("mbe")),
            f2(d.get("r2")),
            f2(d.get("r")),
        ]
        try:
            cov95s = f"{float(cov95):.3f}"
        except (TypeError, ValueError):
            cov95s = "---"
        cells += [
            f2(cov50) if cov50 is not None else "---",
            cov95s,
            f2(width) if width is not None else "---",
            f2(crps) if crps is not None else "---",
        ]
        return f"\t\t{label} & " + " & ".join(cells) + " \\\\"

    lines = [
        "\t\t\\multicolumn{10}{l}{\\textit{Model components and validation modes}} \\\\"
    ]
    for _, r in mc.iterrows():
        d = r.to_dict()
        lines.append(
            row(
                str(r.get("model", r.iloc[0])),
                d,
                d.get("coverage_50"),
                d.get("coverage_95"),
                d.get("interval_width"),
                d.get("crps"),
            )
        )
    lines.append("\t\t\\midrule")
    lines.append(
        "\t\t\\multicolumn{10}{l}{\\textit{Leakage-free, forecast-mode, and competitor evaluations}} \\\\"
    )
    nrow = nested[nested["model"].str.contains("nested")].iloc[0].to_dict()
    lines.append(
        row(
            "GAM-SSM nested LOOCV (leakage-free)",
            nrow,
            nrow.get("coverage_50"),
            nrow.get("coverage_95"),
            nrow.get("interval_width"),
            nrow.get("crps"),
        )
    )
    oda = fc[fc["model"].str.contains("t-1")].iloc[0].to_dict()
    lines.append(row("GAM-SSM one-day-ahead (filter-only)", oda))
    for _, r in ml.iterrows():
        lines.append(row(str(r["model"]), r.to_dict()))
    lines.append("\t\t\\midrule")
    lines.append("\t\t\\multicolumn{10}{l}{\\textit{Naive temporal references}} \\\\")
    for _, r in bl.iterrows():
        d = r.to_dict()
        lines.append(row(str(r["model"]), d, crps=d.get("crps")))
    body = "\n".join(lines)
    return (
        "\\begin{tabular}{l r r r r r r r r r}\n\t\\toprule\n"
        "\tModel & RMSE & MAE & MBE & $R^2$ & $r$ & Cov.50\\% & Cov.95\\% & Width & CRPS \\\\\n"
        "\t\\midrule\n" + body + "\n\t\\bottomrule\n\\end{tabular}\n"
    )


def table8():
    df = pd.read_csv(RESULTS / "component_ablations" / "component_ablations.csv")
    rows = "\n".join(
        f"\t\t{r['model']} & {f2(r['rmse'])} & {f2(r['mae'])} & {f2(r['mbe'])} & "
        f"{f2(r['r2'])} & {f2(r['r'])} & {int(r['n'])} \\\\"
        for _, r in df.iterrows()
    )
    return (
        "\\begin{tabular}{l r r r r r r}\n\t\\toprule\n"
        "\tVariant & RMSE & MAE & MBE & $R^2$ & $r$ & $n$ \\\\\n"
        "\t\\midrule\n" + rows + "\n\t\\bottomrule\n\\end{tabular}\n"
    )


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for name, builder in [
        ("table5", table5),
        ("table6", table6),
        ("table7", table7),
        ("table8", table8),
    ]:
        tex = builder()
        (OUT / f"{name}.tex").write_text(tex)
        print(f"{name}.tex written ({len(tex)} bytes)")


if __name__ == "__main__":
    main()
