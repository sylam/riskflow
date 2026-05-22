"""Marimo notebook: inspect HP-screen results and individual training runs.

Run with:  marimo edit check_data.py
"""
import marimo

__generated_with = "0.23.5"
app = marimo.App(width="medium")


@app.cell
def __():
    from pathlib import Path
    import json as jsonlib
    import pandas as pd
    import numpy as np
    import plotly.express as px
    import plotly.graph_objects as go
    import marimo as mo
    return Path, go, jsonlib, mo, np, pd, px


@app.cell
def __(Path):
    # Default relative paths to diagnostic outputs (from project root)
    ROOT          = Path(".")
    SCREEN_8_CSV  = ROOT / "artifacts" / "hp_postbounds.csv"
    SCREEN_27_CSV = ROOT / "artifacts" / "hp_postbounds_27.csv"
    RUNS_DIR      = ROOT / "artifacts" / "daily_runs"
    return RUNS_DIR, SCREEN_27_CSV, SCREEN_8_CSV


@app.cell
def __(SCREEN_27_CSV, SCREEN_8_CSV, mo, pd):
    # Live screen results — re-run this cell to refresh as screens write rows.
    s8  = pd.read_csv(SCREEN_8_CSV)  if SCREEN_8_CSV.exists()  else pd.DataFrame()
    s27 = pd.read_csv(SCREEN_27_CSV) if SCREEN_27_CSV.exists() else pd.DataFrame()
    mo.md(f"**Screens:** 8-cell `{len(s8)}` rows, 27-cell `{len(s27)}` rows")
    return s27, s8


@app.cell
def __(mo, pd, s27, s8):
    # Cell-level aggregation by asymmetric_score (mean across seeds).
    GROUP_KEYS = ["position_bounds_penalty", "reward_scale", "entropy_coef"]
    def aggregate(df):
        if df.empty or "asymmetric_score" not in df.columns:
            return pd.DataFrame()
        return (df.groupby(GROUP_KEYS)
                  .agg(asym=("asymmetric_score", "mean"),
                       asym_std=("asymmetric_score", "std"),
                       mean_pnl=("final_net_pnl", "mean"),
                       std_pnl=("final_std_net_pnl", "mean"),
                       p5=("final_p5_net_pnl", "mean"),
                       worst=("final_worst_net_pnl", "mean"),
                       trade=("final_action_abs", "mean"))
                  .sort_values("asym", ascending=False)
                  .reset_index())
    cells_8  = aggregate(s8)
    cells_27 = aggregate(s27)
    mo.vstack([
        mo.md("### 8-cell screen — ranked by mean asymmetric_score across seeds"),
        cells_8,
        mo.md("### 27-cell screen — ranked by mean asymmetric_score across seeds"),
        cells_27,
    ])
    return GROUP_KEYS, cells_27, cells_8


@app.cell
def __(GROUP_KEYS, cells_27, cells_8, mo, px):
    # Per-knob marginal effect on asymmetric_score (uses 27-cell screen if populated, else 8-cell).
    src = cells_27 if not cells_27.empty else cells_8
    if src.empty:
        plots = mo.md("_No data yet_")
    else:
        figs = []
        for k in GROUP_KEYS:
            agg = src.groupby(k)["asym"].agg(["mean", "min", "max"]).reset_index()
            scatter_fig = px.scatter(agg, x=k, y="mean", error_y=agg["max"] - agg["mean"],
                                     error_y_minus=agg["mean"] - agg["min"], log_x=True,
                                     title=f"{k} → asymmetric_score")
            figs.append(scatter_fig)
        plots = mo.hstack(figs)
    plots
    return


@app.cell
def __(RUNS_DIR, mo):
    # Browse individual daily-run dirs and pick one.
    if RUNS_DIR.exists():
        run_dirs = sorted([p.name for p in RUNS_DIR.iterdir() if p.is_dir()], reverse=True)
    else:
        run_dirs = []
    selected_run = mo.ui.dropdown(options=run_dirs, value=run_dirs[0] if run_dirs else None,
                                   label="Run dir")
    selected_run
    return selected_run,


@app.cell
def __(RUNS_DIR, jsonlib, mo, pd, selected_run):
    # Load the chosen run's summary.csv + config.json.
    run_summary = pd.DataFrame()
    run_config = {}
    run_path = None
    if selected_run.value:
        run_path = RUNS_DIR / selected_run.value
        sp = run_path / "summary.csv"
        cp = run_path / "config.json"
        if sp.exists():
            run_summary = pd.read_csv(sp)
        if cp.exists():
            run_config = jsonlib.loads(cp.read_text())
    mo.vstack([
        mo.md(f"### `{selected_run.value or 'no run selected'}`"),
        mo.md("**Config:**"),
        run_config,
        mo.md("**Summary (full distribution):**"),
        run_summary,
    ])
    return run_path,


@app.cell
def __(go, mo, pd, run_path):
    # Plot per-day total |position| trajectories for best/worst/p5/p95/mean cases.
    # ML = solid, textbook = dashed; color encodes case.
    fig = go.Figure()
    if run_path is not None:
        for label, fname, dash in (("ml", "ml_paths.csv", "solid"),
                                    ("textbook", "textbook_paths.csv", "dash")):
            csv = run_path / fname
            if not csv.exists():
                continue
            df = pd.read_csv(csv)
            df["day"] = pd.to_datetime(df["day"])
            for case in df["case"].unique():
                sub = df[df["case"] == case]
                total_abs = sub.filter(like="_position").abs().sum(axis=1)
                fig.add_scatter(x=sub["day"], y=total_abs, mode="lines",
                                name=f"{label}:{case}",
                                line=dict(dash=dash),
                                legendgroup=case)
        fig.update_layout(title="Total |position| over time — ML (solid) vs textbook (dash)",
                          xaxis_title="date", yaxis_title="Σ|position_i| (contracts)")
    if not fig.data:
        fig.add_annotation(text="no paths CSV", showarrow=False)
    mo.ui.plotly(fig)
    return


if __name__ == "__main__":
    app.run()
