# src/model.py
# --------------------------------------------------------------------------------------
# Model + optimiser construction utils.  Uses **.cache/** for HF downloads.
# --------------------------------------------------------------------------------------
from __future__ import annotations

from typing import List, Tuple

import torch
from omegaconf import DictConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

__all__ = ["build_model_and_optim"]


# ----------------------------------------------------------------------------------
# Helper to locate *transformer blocks* irrespective of architecture --------------

def _find_transformer_layers(model):
    for attr in ["model", "transformer", "gpt_neox"]:
        sub = getattr(model, attr, None)
        if sub is None:
            continue
        for layers_name in ["layers", "h", "blocks"]:
            layers = getattr(sub, layers_name, None)
            if layers is not None and isinstance(layers, (list, torch.nn.ModuleList)):
                return layers
    raise RuntimeError("Unable to locate transformer layers in the provided model")


# ----------------------------------------------------------------------------------

def build_model_and_optim(cfg: DictConfig):
    """Returns `(model, tokenizer, optim, layer_groups)`.

    *layer_groups* is a `List[List[int]]` mapping **transformer block → param-group indices**.
    """
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.name, cache_dir=".cache", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model.name,
        cache_dir=".cache",
        torch_dtype=getattr(torch, cfg.model.dtype),
        revision=cfg.model.revision,
        trust_remote_code=True,
        device_map="auto",
    )
    if cfg.model.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # Parameter groups ---------------------------------------------------------
    param_groups = []  # list[dict]
    layer_groups: List[List[int]] = []

    layers = _find_transformer_layers(model)
    for block in layers:  # type: ignore[operator]
        current: List[int] = []
        # Separate logical sub-modules: attn + mlp + ln ------------------------
        for mod_name in ["attn", "attention", "self_attn"]:
            if hasattr(block, mod_name):
                params = list(getattr(block, mod_name).parameters())
                if params:
                    param_groups.append({"params": params, "lr": cfg.training.base_learning_rate})
                    current.append(len(param_groups) - 1)
        for mod_name in ["mlp", "ffn", "feed_forward"]:
            if hasattr(block, mod_name):
                params = list(getattr(block, mod_name).parameters())
                if params:
                    param_groups.append({"params": params, "lr": cfg.training.base_learning_rate})
                    current.append(len(param_groups) - 1)
        ln_params = [p for n, p in block.named_parameters() if "norm" in n or "ln" in n]
        if ln_params:
            param_groups.append({"params": ln_params, "lr": cfg.training.base_learning_rate})
            current.append(len(param_groups) - 1)
        layer_groups.append(current)

    # residual parameters (LM head etc.) --------------------------------------
    covered = {p for pg in param_groups for p in pg["params"]}
    other = [p for p in model.parameters() if p not in covered]
    if other:
        param_groups.append({"params": other, "lr": cfg.training.base_learning_rate})

    optim = torch.optim.AdamW(
        param_groups,
        lr=cfg.training.base_learning_rate,
        betas=tuple(cfg.training.optimizer.betas),
        weight_decay=cfg.training.weight_decay,
    )
    return model, tokenizer, optim, layer_groups
