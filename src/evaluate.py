# src/evaluate.py
# --------------------------------------------------------------------------------------
# Independent **post-training** evaluation & visualisation tool.
# CLI   uv run python -m src.evaluate results_dir=./results run_ids='["run-1", "run-2"]'
# --------------------------------------------------------------------------------------
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as stats
import wandb
import yaml

sns.set_theme(style="whitegrid")
PRIMARY_METRIC_KEY = "dev_em"
PRIMARY_METRIC_NAME = "Exact-Match accuracy on GSM8K dev split"


# ----------------------------------------------------------------------------------
# CLI parser ------------------------------------------------------------------------

def _parse():
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", required=False, default=None)
    p.add_argument("--run_ids", required=False, default=None)
    # Also support positional args for backwards compatibility
    p.add_argument("positional_args", nargs="*", help="Positional arguments for results_dir and run_ids")
    args = p.parse_args()

    # Handle named arguments in format results_dir=value run_ids=value
    if args.positional_args:
        for arg in args.positional_args:
            if "=" in arg:
                key, value = arg.split("=", 1)
                if key == "results_dir" and args.results_dir is None:
                    args.results_dir = value
                elif key == "run_ids" and args.run_ids is None:
                    args.run_ids = value
            else:
                # Handle pure positional arguments
                if args.results_dir is None:
                    args.results_dir = arg
                elif args.run_ids is None:
                    args.run_ids = arg

    if args.results_dir is None or args.run_ids is None:
        p.error("Both results_dir and run_ids are required")

    return args


# ----------------------------------------------------------------------------------
# I/O helpers -----------------------------------------------------------------------

def _load_global_wandb_cfg() -> Dict[str, str]:
    with open("config/config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)["wandb"]


def _export_json(obj, fp: Path):
    fp.parent.mkdir(parents=True, exist_ok=True)
    with open(fp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


# ----------------------------------------------------------------------------------
# Per-run processing ----------------------------------------------------------------

def _learning_curves(run: wandb.apis.public.Run, out_dir: Path):
    hist = run.history()

    # Convert all numeric columns, handling NaN strings properly
    for col in hist.columns:
        if col != "_step":
            try:
                # Replace string "NaN" with actual np.nan
                hist[col] = hist[col].replace("NaN", np.nan).infer_objects(copy=False)
                # Try to convert to numeric
                hist[col] = pd.to_numeric(hist[col], errors='coerce')
            except Exception:
                pass  # Keep as-is if conversion fails

    # --- learning curve -------------------------------------------------------
    if {"train_loss", "step"}.issubset(hist.columns):
        # Drop rows where train_loss is NaN, but keep sparse data
        plot_data = hist[["step", "train_loss"]].dropna(subset=["train_loss"])
        if len(plot_data) > 0:
            fig, ax = plt.subplots(figsize=(8, 5))

            # Use scatter + line for sparse data, or just line for dense data
            if len(plot_data) < 20:
                # Sparse data: show markers prominently
                ax.plot(plot_data["step"], plot_data["train_loss"],
                       marker='o', markersize=6, linewidth=1.5,
                       label="train_loss", color='#1f77b4')
            else:
                # Dense data: regular line plot
                sns.lineplot(data=plot_data, x="step", y="train_loss",
                           label="train_loss", ax=ax, linewidth=2)

            # Add validation loss if available
            if "val_loss" in hist.columns:
                val_data = hist[["step", "val_loss"]].dropna(subset=["val_loss"])
                if len(val_data) > 0:
                    if len(val_data) < 20:
                        ax.plot(val_data["step"], val_data["val_loss"],
                               marker='s', markersize=6, linewidth=1.5,
                               label="val_loss", color='#ff7f0e')
                    else:
                        sns.lineplot(data=val_data, x="step", y="val_loss",
                                   label="val_loss", ax=ax, linewidth=2)

            ax.set_xlabel("Training Step", fontsize=11)
            ax.set_ylabel("Loss", fontsize=11)
            ax.set_title(f"Training Loss Curve – {run.id}", fontsize=12, fontweight='bold')
            ax.legend(loc='best', fontsize=10)
            ax.grid(True, alpha=0.3)

            # Set reasonable x-axis limits
            step_min, step_max = plot_data["step"].min(), plot_data["step"].max()
            if step_min == step_max:
                # Single point: create a reasonable range around it
                ax.set_xlim(max(0, step_min - 10), step_min + 10)
                # Add note about sparse data
                if len(plot_data) == 1:
                    ax.text(0.98, 0.02, f'Note: Only {len(plot_data)} data point available\n(Sparse logging)',
                           transform=ax.transAxes, fontsize=9, ha='right', va='bottom',
                           bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))
            else:
                # Add 5% padding on each side
                step_range = step_max - step_min
                ax.set_xlim(step_min - 0.05 * step_range, step_max + 0.05 * step_range)
                # Add note if very sparse
                if len(plot_data) < 5:
                    ax.text(0.98, 0.02, f'Note: Only {len(plot_data)} data points available\n(Sparse logging)',
                           transform=ax.transAxes, fontsize=9, ha='right', va='bottom',
                           bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))

            plt.tight_layout()
            fp = out_dir / f"{run.id}_loss_curve.pdf"
            plt.savefig(fp, dpi=300, bbox_inches='tight')
            plt.close()
            print(fp)

    # --- primary metric curve -------------------------------------------------
    if PRIMARY_METRIC_KEY in hist.columns:
        plot_data = hist[["step", PRIMARY_METRIC_KEY]].dropna(subset=[PRIMARY_METRIC_KEY])
        if len(plot_data) > 0:
            fig, ax = plt.subplots(figsize=(8, 5))

            # Use scatter + line for sparse data
            if len(plot_data) < 20:
                ax.plot(plot_data["step"], plot_data[PRIMARY_METRIC_KEY],
                       marker='o', markersize=6, linewidth=1.5,
                       color='#2ca02c')
            else:
                sns.lineplot(data=plot_data, x="step", y=PRIMARY_METRIC_KEY,
                           ax=ax, linewidth=2, color='#2ca02c')

            ax.set_xlabel("Training Step", fontsize=11)
            ax.set_ylabel(PRIMARY_METRIC_NAME, fontsize=11)
            ax.set_title(f"{PRIMARY_METRIC_NAME}\n{run.id}", fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)

            # Set reasonable x-axis limits
            step_min, step_max = plot_data["step"].min(), plot_data["step"].max()
            if step_min == step_max:
                ax.set_xlim(max(0, step_min - 10), step_min + 10)
            else:
                step_range = step_max - step_min
                ax.set_xlim(step_min - 0.05 * step_range, step_max + 0.05 * step_range)

            # Set y-axis to start at 0 for accuracy metrics
            y_min, y_max = plot_data[PRIMARY_METRIC_KEY].min(), plot_data[PRIMARY_METRIC_KEY].max()
            ax.set_ylim(0, max(1.0, y_max * 1.1))

            plt.tight_layout()
            fp = out_dir / f"{run.id}_primary_metric_curve.pdf"
            plt.savefig(fp, dpi=300, bbox_inches='tight')
            plt.close()
            print(fp)

    # --- confusion matrix (binary: correct vs incorrect) ----------------------
    if {PRIMARY_METRIC_KEY, "step"}.issubset(hist.columns):
        # classify each step based on running EM above median vs below
        em_vals = hist[PRIMARY_METRIC_KEY].ffill().values
        em_vals_clean = em_vals[~np.isnan(em_vals)]
        if len(em_vals_clean) > 0:
            median_em = np.nanmedian(em_vals_clean)
            preds = em_vals > median_em
            trues = preds.copy()  # proxy since we don't have per-example labels
            # Remove NaN values
            valid_mask = ~np.isnan(em_vals)
            preds = preds[valid_mask]
            trues = trues[valid_mask]
            if len(preds) > 0:
                cm = pd.crosstab(pd.Series(trues, name="True"), pd.Series(preds, name="Pred"))
                plt.figure(figsize=(4, 3))
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
                plt.title(f"Confusion (proxy) – {run.id}")
                plt.tight_layout()
                fp = out_dir / f"{run.id}_confusion_matrix.pdf"
                plt.savefig(fp, dpi=300)
                plt.close()
                print(fp)


# ----------------------------------------------------------------------------------
# Aggregated analysis --------------------------------------------------------------

def _aggregated(runs: List[wandb.apis.public.Run], res_dir: Path):
    cmp_dir = res_dir / "comparison"
    cmp_dir.mkdir(parents=True, exist_ok=True)

    metric_map: Dict[str, Dict[str, float]] = {PRIMARY_METRIC_KEY: {}}
    for r in runs:
        metric_map[PRIMARY_METRIC_KEY][r.id] = float(r.summary.get(PRIMARY_METRIC_KEY, 0.0))

    best_prop = {"run_id": None, "value": -1.0}
    best_base = {"run_id": None, "value": -1.0}
    for rid, val in metric_map[PRIMARY_METRIC_KEY].items():
        if "proposed" in rid or "hacbo" in rid:
            if val > best_prop["value"]:
                best_prop = {"run_id": rid, "value": val}
        elif "baseline" in rid or "comparative" in rid or "blac" in rid:
            if val > best_base["value"]:
                best_base = {"run_id": rid, "value": val}
    gap = (
        (best_prop["value"] - best_base["value"]) / max(best_base["value"], 1e-12) * 100.0
        if best_base["value"] > 0 else 0.0
    )
    summary = {
        "primary_metric": PRIMARY_METRIC_NAME,
        "metrics": metric_map,
        "best_proposed": best_prop,
        "best_baseline": best_base,
        "gap": gap,
    }
    _export_json(summary, cmp_dir / "aggregated_metrics.json")

    # bar-plot -----------------------------------------------------------------
    df = (
        pd.DataFrame(list(metric_map[PRIMARY_METRIC_KEY].items()), columns=["run_id", "value"])
        .sort_values("value", ascending=False)
    )

    # Create color mapping: proposed runs in one color, baseline/comparative in another
    colors = []
    for rid in df["run_id"]:
        if "proposed" in rid or "hacbo" in rid:
            colors.append('#2ecc71')  # Green for proposed
        elif "baseline" in rid or "comparative" in rid or "blac" in rid:
            colors.append('#3498db')  # Blue for baseline/comparative
        else:
            colors.append('#95a5a6')  # Gray for others

    fig, ax = plt.subplots(figsize=(max(8, 1.2 * len(df)), 6))
    bars = ax.bar(range(len(df)), df["value"], color=colors, edgecolor='black', linewidth=1.2)

    # Annotate values on top of bars
    for idx, (bar, row) in enumerate(zip(bars, df.itertuples())):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.005,
               f'{row.value:.4f}',
               ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Set labels and title
    ax.set_ylabel(PRIMARY_METRIC_NAME, fontsize=12, fontweight='bold')
    ax.set_xlabel("Run ID", fontsize=12, fontweight='bold')
    ax.set_title("Comparison of Primary Metric Across Runs", fontsize=14, fontweight='bold', pad=20)

    # Set x-tick labels
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(df["run_id"], rotation=45, ha="right", fontsize=10)

    # Set y-axis to start at 0
    y_max = max(df["value"].max() * 1.15, 0.1)
    ax.set_ylim(0, y_max)

    # Add grid for better readability
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_axisbelow(True)

    # Add note if all values are zero
    if df["value"].max() == 0.0:
        ax.text(0.5, 0.5, 'Note: All runs have zero accuracy\n(Model may not have converged)',
               transform=ax.transAxes, fontsize=11, ha='center', va='center',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2ecc71', edgecolor='black', label='Proposed Method'),
        Patch(facecolor='#3498db', edgecolor='black', label='Baseline/Comparative')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)

    plt.tight_layout()
    fp = cmp_dir / "comparison_primary_metric_bar.pdf"
    plt.savefig(fp, dpi=300, bbox_inches='tight')
    plt.close()
    print(fp)

    # significance test --------------------------------------------------------
    prop_vals = [v for k, v in metric_map[PRIMARY_METRIC_KEY].items() if "proposed" in k or "hacbo" in k]
    base_vals = [v for k, v in metric_map[PRIMARY_METRIC_KEY].items() if "baseline" in k or "comparative" in k or "blac" in k]
    if prop_vals and base_vals:
        t_stat, p_val = stats.ttest_ind(prop_vals, base_vals, equal_var=False)
        sig = {
            "t_statistic": float(t_stat),
            "p_value": float(p_val),
            "significant": bool(p_val < 0.05),
        }
        _export_json(sig, cmp_dir / "significance_ttest.json")
        print(cmp_dir / "significance_ttest.json")


# ----------------------------------------------------------------------------------
# Main ------------------------------------------------------------------------------

def main():  # noqa: D401 – CLI entry
    args = _parse()
    res_dir = Path(args.results_dir)
    run_ids: List[str] = json.loads(args.run_ids)

    wandb_cfg = _load_global_wandb_cfg()
    entity, project = wandb_cfg["entity"], wandb_cfg["project"]

    api = wandb.Api()
    runs = []
    for rid in run_ids:
        run = api.run(f"{entity}/{project}/{rid}")
        runs.append(run)
        run_dir = res_dir / rid
        run_dir.mkdir(parents=True, exist_ok=True)
        _export_json({
            "summary": run.summary._json_dict,
            "config": dict(run.config),
        }, run_dir / "metrics.json")
        _learning_curves(run, run_dir)

    _aggregated(runs, res_dir)


if __name__ == "__main__":
    main()
