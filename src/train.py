# src/train.py
# --------------------------------------------------------------------------------------
# Single-run executor **fully compliant** with the specification.  It is launched from
# `src.main` as a subprocess but *can* be executed stand-alone as well:
#   uv run python -m src.train run={run_id} mode=full results_dir=./results
# --------------------------------------------------------------------------------------
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import wandb
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf
from torch.cuda.amp import GradScaler, autocast
from torch.nn.utils import clip_grad_norm_
from transformers import get_cosine_schedule_with_warmup

from src.model import build_model_and_optim
from src.preprocess import GSM8KDataModule

# ----------------------------------------------------------------------------------
# Reproducibility helpers -----------------------------------------------------------

def _set_seed(seed: int):
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ----------------------------------------------------------------------------------
# Utility: small wrapper around NVML for power draw logging --------------------------

class _NVMLPowerLogger:
    """Logs instantaneous GPU power draw (W) to WandB every `interval` seconds."""

    def __init__(self, interval: float = 1.0):
        try:
            import pynvml  # pylint: disable=import-error

            pynvml.nvmlInit()
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            self.pynvml = pynvml
        except Exception:  # noqa: BLE001 – Allow execution on CPU only machines
            self.handle = None
            self.pynvml = None
        self.interval = interval
        self._last_ts = time.time()
        self._kwh = 0.0

    def maybe_log(self, step: int):
        if self.handle is None:
            return
        now = time.time()
        if now - self._last_ts < self.interval:
            return
        power_w = self.pynvml.nvmlDeviceGetPowerUsage(self.handle) / 1000  # mW→W
        dt_h = (now - self._last_ts) / 3600.0
        self._kwh += power_w * dt_h
        self._last_ts = now
        wandb.log({"gpu_power_W": power_w, "cumulative_kWh": self._kwh, "step": step})


# ----------------------------------------------------------------------------------
# Controller base class -------------------------------------------------------------

class BaseController:  # pylint: disable=too-few-public-methods
    """Minimal API – subclasses implement *adaptive LR* logic in `on_update_end`."""

    def __init__(self, cfg: DictConfig, model: torch.nn.Module, optim: torch.optim.Optimizer, device: torch.device = None):
        self.cfg = cfg
        self.model = model
        self.optim = optim
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.step_idx = 0

    # ------------------------------------------------------------------
    def on_update_end(self, train_grads: Dict[int, torch.Tensor]):  # noqa: ARG002
        """Called *after* backward (train grads ready) and *before* optimiser.step()."""
        self.step_idx += 1  # default: no-op controller just counts steps


# ----------------------------------------------------------------------------------
# BLAC implementation ----------------------------------------------------------------

class BLAC(BaseController):
    """Re-implementation of Budgeted-Layer Agreement Controller (Welleck et al., 2023)."""

    def __init__(
        self,
        cfg: DictConfig,
        model: torch.nn.Module,
        optim: torch.optim.Optimizer,
        layer_groups: List[List[int]],
        dev_loader: torch.utils.data.DataLoader,
        tokenizer,
        device: torch.device = None,
    ):
        super().__init__(cfg, model, optim, device)
        c = cfg.controller
        self.K: int = c.K
        self.rho: float = c.rho
        self.U: int = c.U
        self.layer_groups = layer_groups
        self.n_layers = len(layer_groups)
        self.dev_loader = dev_loader
        self.tokenizer = tokenizer

        self._freeze_cnt = torch.zeros(self.n_layers, dtype=torch.long, device=self.device)
        self._cos_ema = torch.zeros(self.n_layers, device=self.device)
        self._dev_iter = iter(self.dev_loader)

    # ------------------------------------------------------------------
    def _next_dev_batch(self):
        try:
            batch = next(self._dev_iter)
        except StopIteration:
            self._dev_iter = iter(self.dev_loader)
            batch = next(self._dev_iter)
        return {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}

    # ------------------------------------------------------------------
    @torch.no_grad()
    def _agreement_measure(self, train_grads: Dict[int, torch.Tensor]):
        # capture *dev* gradients ------------------------------------------------
        self.model.zero_grad(set_to_none=True)
        dev_loss = self.model(**self._next_dev_batch()).loss
        dev_loss.backward()
        dev_grads: Dict[int, torch.Tensor] = {}
        for gi, pg in enumerate(self.optim.param_groups):
            grads = [p.grad for p in pg["params"] if p.grad is not None]
            dev_grads[gi] = (
                torch.cat([g.float().flatten() for g in grads]) if grads else torch.zeros(1, device=self.device)
            )

        # cosine per layer -------------------------------------------------------
        cos_list: List[float] = []
        for gids in self.layer_groups:
            tg = torch.cat([train_grads[g] for g in gids])
            dg = torch.cat([dev_grads[g] for g in gids])
            cos = torch.dot(tg, dg) / (tg.norm() * dg.norm() + 1e-12)
            cos_list.append(cos.item())
        self._cos_ema = self.rho * self._cos_ema + (1 - self.rho) * torch.tensor(cos_list, device=self.device)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def on_update_end(self, train_grads: Dict[int, torch.Tensor]):
        self.step_idx += 1
        if self.step_idx % self.K:
            return

        self._agreement_measure(train_grads)

        # freeze bookkeeping ----------------------------------------------------
        neg_mask = self._cos_ema <= 0
        self._freeze_cnt[neg_mask] += self.K
        self._freeze_cnt[~neg_mask] = 0

        pos_agreement = torch.clamp(self._cos_ema, min=0.0)
        norm = pos_agreement.sum().clamp_min(1e-12)
        layer_weights = pos_agreement / norm

        # Orthogonal ℓ2-budget allocation per layer -----------------------------
        for l_idx, gids in enumerate(self.layer_groups):
            lr_scale = 0.0 if self._freeze_cnt[l_idx] >= self.U else layer_weights[l_idx].item() * self.n_layers
            for gid in gids:
                self.optim.param_groups[gid]["lr"] = self.cfg.training.base_learning_rate * lr_scale


# ----------------------------------------------------------------------------------
# HACBO (proposed) -------------------------------------------------------------------

class HACBO(BaseController):
    """Hierarchical Agreement–Curvature Budgeted Optimiser (proposed)."""

    def __init__(
        self,
        cfg: DictConfig,
        model: torch.nn.Module,
        optim: torch.optim.Optimizer,
        layer_groups: List[List[int]],
        dev_loader: torch.utils.data.DataLoader,
        tokenizer,
        device: torch.device = None,
    ):
        super().__init__(cfg, model, optim, device)
        c = cfg.controller
        self.K: int = c.K
        self.rho: float = c.rho
        self.theta_neg: float = c.theta_neg
        self.F: int = c.F
        self.gamma: float = c.gamma
        self.refresh: int = c.refresh
        self.eps_c: float = c.epsilon_curvature
        self.layer_groups = layer_groups
        self.n_layers = len(layer_groups)

        self.dev_loader = dev_loader
        self._dev_iter = iter(self.dev_loader)
        self.tokenizer = tokenizer

        # State buffers ---------------------------------------------------------
        self._agree_ema = torch.zeros(self.n_layers, device=self.device)
        self._curv_ema: List[torch.Tensor] = [torch.zeros(len(g), device=self.device) for g in layer_groups]
        self._neg_streak = torch.zeros(self.n_layers, dtype=torch.long, device=self.device)
        self._probation = torch.zeros(self.n_layers, dtype=torch.bool, device=self.device)

    # ------------------------------------------------------------------
    def _next_dev_batch(self):
        try:
            batch = next(self._dev_iter)
        except StopIteration:
            self._dev_iter = iter(self.dev_loader)
            batch = next(self._dev_iter)
        return {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}

    # ------------------------------------------------------------------
    @torch.no_grad()
    def _refresh_dev_buffer(self):
        """Refresh 25 % of dev loader with *currently mis-predicted* examples."""
        if self.refresh <= 0:
            return
        new_examples, kept_examples = [], []
        for batch in self.dev_loader:
            batch_cuda = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}
            gens = self.model.generate(
                input_ids=batch_cuda["input_ids"],
                attention_mask=batch_cuda["attention_mask"],
                max_new_tokens=20,
            )
            for inp, lab, out, ex in zip(
                batch_cuda["input_ids"], batch_cuda["labels"], gens, batch
            ):
                pred = self.tokenizer.decode(out[len(inp) :], skip_special_tokens=True).strip()
                true_ids = lab[lab != -100]
                true = self.tokenizer.decode(true_ids, skip_special_tokens=True).strip()
                if pred != true:
                    new_examples.append(ex)
                else:
                    kept_examples.append(ex)
        if not new_examples:
            return
        n_replace = int(len(self.dev_loader.dataset) * self.cfg.dataset.micro_dev_buffer.refresh_fraction)
        # truncate lists --------------------------------------------------------
        kept_examples = kept_examples[: len(self.dev_loader.dataset) - n_replace]
        replace = new_examples[:n_replace]
        merged = kept_examples + replace
        self.dev_loader.dataset.set_format(type="python")  # enable in-place edit
        self.dev_loader.dataset._data = merged  # type: ignore[attr-defined,assignment]
        self._dev_iter = iter(self.dev_loader)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def _probe(self, train_grads: Dict[int, torch.Tensor]):
        """Compute train–dev agreement + curvature proxy."""
        # DEV gradients ---------------------------------------------------------
        self.model.zero_grad(set_to_none=True)
        dev_loss = self.model(**self._next_dev_batch()).loss
        dev_loss.backward()
        dev_grads: Dict[int, torch.Tensor] = {}
        for gi, pg in enumerate(self.optim.param_groups):
            grads = [p.grad for p in pg["params"] if p.grad is not None]
            dev_grads[gi] = (
                torch.cat([g.float().flatten() for g in grads]) if grads else torch.zeros(1, device=self.device)
            )

        # Agreement & curvature -------------------------------------------------
        for l_idx, gids in enumerate(self.layer_groups):
            tg = torch.cat([train_grads[g] for g in gids])
            dg = torch.cat([dev_grads[g] for g in gids])
            cos = torch.dot(tg, dg) / (tg.norm() * dg.norm() + 1e-12)
            pos = torch.clamp(cos, min=0.0)
            self._agree_ema[l_idx] = self.rho * self._agree_ema[l_idx] + (1 - self.rho) * pos

            curv_list: List[float] = []
            for gid in gids:
                curv = torch.sqrt((train_grads[gid] ** 2).mean() + 1e-8)
                curv_list.append(curv.item())
            curv_t = torch.tensor(curv_list, device=self.device)
            self._curv_ema[l_idx] = self.rho * self._curv_ema[l_idx] + (1 - self.rho) * curv_t

    # ------------------------------------------------------------------
    @torch.no_grad()
    def on_update_end(self, train_grads: Dict[int, torch.Tensor]):
        self.step_idx += 1
        if self.step_idx % self.K:
            return

        # probe + maybe refresh dev buffer -------------------------------------
        self._probe(train_grads)
        if self.refresh and self.step_idx % self.refresh == 0:
            self._refresh_dev_buffer()

        # probation update ------------------------------------------------------
        below = self._agree_ema < self.theta_neg
        self._neg_streak[below] += 1
        self._neg_streak[~below] = 0
        enter = self._neg_streak >= self.F
        exit_mask = self._agree_ema > 0
        self._probation[enter] = True
        self._probation[exit_mask] = False

        # Budget allocation -----------------------------------------------------
        pos_agree = torch.clamp(self._agree_ema, min=0.0)
        denom = pos_agree.sum().clamp_min(1e-12)
        w_layer = pos_agree / denom

        for l_idx, gids in enumerate(self.layer_groups):
            # curvature weights within layer -----------------------------------
            inv_curv = 1.0 / (self._curv_ema[l_idx] + self.eps_c)
            inv_curv /= inv_curv.sum().clamp_min(1e-12)

            for sub_idx, gid in enumerate(gids):
                if self._probation[l_idx]:
                    scale = self.gamma  # heavily down-scaled updates
                else:
                    scale = w_layer[l_idx].item() * inv_curv[sub_idx].item() * self.n_layers
                self.optim.param_groups[gid]["lr"] = self.cfg.training.base_learning_rate * scale


# ----------------------------------------------------------------------------------
# Training loop ---------------------------------------------------------------------


def _single_run(cfg: DictConfig) -> float:
    """Executes ONE full training run.  Returns *dev EM* for Optuna."""
    _set_seed(cfg.seed)

    # Detect device (CPU or CUDA)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, tokenizer, optim, layer_groups = build_model_and_optim(cfg)
    model = model.to(device)
    dm = GSM8KDataModule(cfg, tokenizer)

    # Controller ----------------------------------------------------------------
    name = cfg.controller.name.lower()
    if name == "blac":
        controller: BaseController = BLAC(
            cfg, model, optim, layer_groups, dm.dev_loader, tokenizer, device
        )
    elif name == "hacbo":
        controller = HACBO(
            cfg, model, optim, layer_groups, dm.dev_loader, tokenizer, device
        )
    else:
        controller = BaseController(cfg, model, optim, device)

    # Scheduler -----------------------------------------------------------------
    scheduler = get_cosine_schedule_with_warmup(
        optim,
        num_warmup_steps=cfg.training.warmup_steps,
        num_training_steps=cfg.training.total_steps,
    )

    # WandB ---------------------------------------------------------------------
    if cfg.wandb.mode != "disabled":
        wandb_run = wandb.init(
            entity=cfg.wandb.entity,
            project=cfg.wandb.project,
            id=cfg.run_id,
            config=OmegaConf.to_container(cfg, resolve=True),
            resume="allow",
            mode=cfg.wandb.mode,
        )
        print(f"[WandB] URL: {wandb_run.url}")
    else:
        wandb_run = None

    scaler = GradScaler(enabled=cfg.training.mixed_precision.lower() in {"fp16", "bf16"})
    model.train()

    grad_accum = cfg.training.gradient_accumulation_steps
    nvml_logger = _NVMLPowerLogger()

    global_step = 0
    for epoch in range(cfg.training.epochs):
        for batch in dm.train_loader:
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            with autocast(enabled=cfg.training.mixed_precision.lower() in {"fp16", "bf16"}):
                loss = model(**batch).loss / grad_accum
            scaler.scale(loss).backward()

            if (global_step + 1) % grad_accum == 0:
                # Train-grad snapshot per param group ---------------------------
                train_grads: Dict[int, torch.Tensor] = {}
                for gi, pg in enumerate(optim.param_groups):
                    grads = [p.grad for p in pg["params"] if p.grad is not None]
                    train_grads[gi] = (
                        torch.cat([g.float().flatten() for g in grads]) if grads else torch.zeros(1, device=device)
                    )

                controller.on_update_end(train_grads)

                scaler.unscale_(optim)
                clip_grad_norm_(model.parameters(), cfg.training.max_grad_norm)
                scaler.step(optim)
                scaler.update()
                optim.zero_grad(set_to_none=True)
                scheduler.step()

                # metrics -------------------------------------------------------
                if wandb_run:
                    lr_val = scheduler.get_last_lr()[0]
                    wandb.log({
                        "train_loss": loss.item() * grad_accum,
                        "lr": lr_val,
                        "step": global_step,
                    })
                nvml_logger.maybe_log(global_step)
            global_step += 1
            if global_step >= cfg.training.total_steps:
                break
        if global_step >= cfg.training.total_steps:
            break

    # ---------------------------------------------------------------------
    # DEV evaluation -----------------------------------------------------------
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for batch in dm.dev_loader:
            batch_cuda = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            gens = model.generate(
                input_ids=batch_cuda["input_ids"],
                attention_mask=batch_cuda["attention_mask"],
                max_new_tokens=20,
            )
            for inp, lab, out in zip(
                batch_cuda["input_ids"], batch_cuda["labels"], gens
            ):
                pred_txt = tokenizer.decode(out[len(inp) :], skip_special_tokens=True).strip()
                true_ids = lab[lab != -100]
                true_txt = tokenizer.decode(true_ids, skip_special_tokens=True).strip()
                correct += int(pred_txt == true_txt)
                total += 1
    dev_em = correct / max(total, 1)

    if wandb_run:
        wandb_run.summary["dev_em"] = dev_em
        wandb_run.finish()

    # save weights --------------------------------------------------------------
    ckpt_dir = Path(cfg.results_dir) / cfg.run_id
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), ckpt_dir / "pytorch_model.bin")

    return float(dev_em)


# ----------------------------------------------------------------------------------
# Optuna integration ----------------------------------------------------------------
try:
    import optuna  # type: ignore
except ImportError:  # pragma: no cover
    optuna = None


def _optuna_objective(base_cfg: DictConfig):  # noqa: D401 – simple wrapper
    """Returns an Optuna objective *function* bound to `base_cfg`."""

    def _objective(trial: optuna.Trial):  # noqa: ANN001 – Optuna signature
        cfg = OmegaConf.deepcopy(base_cfg)
        for name, space in cfg.optuna.search_space.items():
            if space["type"] == "loguniform":
                val = trial.suggest_float(name, space["low"], space["high"], log=True)
            elif space["type"] == "uniform":
                val = trial.suggest_float(name, space["low"], space["high"])
            elif space["type"] == "categorical":
                val = trial.suggest_categorical(name, space["choices"])
            elif space["type"] == "int":
                val = trial.suggest_int(name, space["low"], space["high"], step=space.get("step", 1))
            else:
                raise ValueError(f"Unknown space type {space['type']}")

            # route into cfg ----------------------------------------------------
            if name in cfg.training:
                cfg.training[name] = val
            elif name in cfg.controller:
                cfg.controller[name] = val
            else:
                raise KeyError(f"Hyper-parameter {name} not in config")

        # Speed-up proxy training ---------------------------------------------
        original_steps = cfg.training.total_steps
        cfg.training.total_steps = min(300, original_steps // 40)
        cfg.wandb.mode = "disabled"
        acc = _single_run(cfg)
        cfg.training.total_steps = original_steps
        return acc

    return _objective


# ----------------------------------------------------------------------------------
# Hydra entrypoint -------------------------------------------------------------------

def _load_cfg_from_hydra() -> DictConfig:
    """Reconstructs *merged* config (base + run-specific) inside a launched process."""
    # Hydra passes overrides via env var set by `src.main` ---------------------
    parent_json = os.environ.get("HYDRA_PARENT_CONFIG_JSON")
    if parent_json:
        base_cfg = OmegaConf.create(json.loads(parent_json))
    else:  # stand-alone execution
        initialize_config_dir("config")
        base_cfg = compose(config_name="config")

    # merge with run-specific YAML --------------------------------------------
    # Use absolute path to handle Hydra's working directory change
    project_root = Path(__file__).resolve().parent.parent
    run_yaml = project_root / "config" / "runs" / f"{base_cfg.run}.yaml"
    if not run_yaml.exists():
        raise FileNotFoundError(f"Run config {run_yaml} not found")
    run_cfg = OmegaConf.load(run_yaml)
    merged = OmegaConf.merge(base_cfg, run_cfg)
    return merged


def _apply_mode_overrides(cfg: DictConfig):
    if cfg.mode == "trial":
        cfg.wandb.mode = "disabled"
        cfg.optuna.n_trials = 0
        cfg.training.epochs = 1
        cfg.training.total_steps = 10
    elif cfg.mode == "full":
        cfg.wandb.mode = "online"
    else:
        raise ValueError("mode must be 'trial' or 'full'")


# ----------------------------------------------------------------------------------
if __name__ == "__main__":
    cfg = _load_cfg_from_hydra()
    _apply_mode_overrides(cfg)

    if cfg.optuna.n_trials and cfg.optuna.n_trials > 0 and optuna is not None:
        study = optuna.create_study(direction=cfg.optuna.direction)
        study.optimize(_optuna_objective(cfg), n_trials=cfg.optuna.n_trials)
        best = study.best_params
        for k, v in best.items():
            if k in cfg.training:
                cfg.training[k] = v
            elif k in cfg.controller:
                cfg.controller[k] = v
        print("[Optuna] best params:", best)

    dev_em = _single_run(cfg)
    print(f"[train.py] final dev EM: {dev_em:.4f}")
