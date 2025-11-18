# src/main.py
# --------------------------------------------------------------------------------------
# Hydra-driven orchestrator.  Spawns *exactly one* training subprocess.
# --------------------------------------------------------------------------------------
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import List

import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(config_path="../config", config_name="config")
def main(cfg: DictConfig):
    # Resolve run-specific YAML -------------------------------------------------
    run_yaml = Path("config") / "runs" / f"{cfg.run}.yaml"
    if not run_yaml.exists():
        raise FileNotFoundError(f"Run config {run_yaml} not found.  Ensure run={cfg.run} is valid.")
    run_cfg = OmegaConf.load(run_yaml)
    merged_cfg = OmegaConf.merge(cfg, run_cfg)

    # Mode overrides -----------------------------------------------------------
    if merged_cfg.mode == "trial":
        merged_cfg.wandb.mode = "disabled"
        merged_cfg.optuna.n_trials = 0
        merged_cfg.training.epochs = 1
        merged_cfg.training.total_steps = 10
    elif merged_cfg.mode == "full":
        merged_cfg.wandb.mode = "online"
    else:
        raise ValueError("mode must be trial|full")

    # Launch training subprocess ----------------------------------------------
    cmd: List[str] = [
        sys.executable,
        "-u",
        "-m",
        "src.train",
        f"run={merged_cfg.run}",
        f"mode={merged_cfg.mode}",
        f"results_dir={merged_cfg.results_dir}",
    ]
    os.environ["HYDRA_PARENT_CONFIG_JSON"] = OmegaConf.to_json(merged_cfg)
    print("[main] Running:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
