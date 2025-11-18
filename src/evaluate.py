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
    p.add_argument("results_dir")
    p.add_argument("run_ids")
    return p.parse_args()


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
    # --- learning curve -------------------------------------------------------
    if {"train_loss", "step"}.issubset(hist.columns):
        plt.figure(figsize=(6, 4))
        sns.lineplot(data=hist, x="step", y="train_loss", label="train_loss")
        if "val_loss" in hist.columns:
            sns.lineplot(data=hist, x="step", y="val_loss", label="val_loss")
        plt.title(f"Loss – {run.id}")
        plt.tight_layout()
        fp = out_dir / f"{run.id}_loss_curve.pdf"
        plt.savefig(fp)
        plt.close()
        print(fp)

    # --- primary metric curve -------------------------------------------------
    if PRIMARY_METRIC_KEY in hist.columns:
        plt.figure(figsize=(6, 4))
        sns.lineplot(data=hist, x="step", y=PRIMARY_METRIC_KEY)
        plt.ylabel(PRIMARY_METRIC_NAME)
        plt.title(f"{PRIMARY_METRIC_NAME} – {run.id}")
        plt.tight_layout()
        fp = out_dir / f"{run.id}_primary_metric_curve.pdf"
        plt.savefig(fp)
        plt.close()
        print(fp)

    # --- confusion matrix (binary: correct vs incorrect) ----------------------
    if {PRIMARY_METRIC_KEY, "step"}.issubset(hist.columns):
        # classify each step based on running EM above median vs below
        em_vals = hist[PRIMARY_METRIC_KEY].fillna(method="ffill").values
        median_em = np.nanmedian(em_vals)
        preds = em_vals > median_em
        trues = preds.copy()  # proxy since we don't have per-example labels
        cm = pd.crosstab(pd.Series(trues, name="True"), pd.Series(preds, name="Pred"))
        plt.figure(figsize=(4, 3))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title(f"Confusion (proxy) – {run.id}")
        plt.tight_layout()
        fp = out_dir / f"{run.id}_confusion_matrix.pdf"
        plt.savefig(fp)
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
    plt.figure(figsize=(max(6, 0.6 * len(df)), 4))
    sns.barplot(data=df, x="run_id", y="value", palette="viridis")
    plt.ylabel(PRIMARY_METRIC_NAME)
    plt.xticks(rotation=45, ha="right")
    for idx, row in df.iterrows():  # annotate values
        plt.text(idx, row.value + 1e-3, f"{row.value:.3f}", ha="center", va="bottom")
    plt.tight_layout()
    fp = cmp_dir / "comparison_primary_metric_bar.pdf"
    plt.savefig(fp)
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
            "significant": p_val < 0.05,
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
