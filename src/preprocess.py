# src/preprocess.py
# --------------------------------------------------------------------------------------
# GSM8K data module – *complete*, leak-proof, with masking of prompt tokens.
# --------------------------------------------------------------------------------------
from __future__ import annotations

import os
import random
import re
from typing import Any, Dict, List

from datasets import load_dataset
from torch.utils.data import DataLoader

# ----------------------------------------------------------------------------------

def _clean_answer(ans: str, remove_commas: bool, strip_ws: bool) -> str:
    if remove_commas:
        ans = ans.replace(",", "")
    if strip_ws:
        ans = ans.strip()
    match = re.search(r"####\s*([-+]?\d*\.?\d+)", ans)
    return match.group(1) if match else ans


class GSM8KDataModule:  # pylint: disable=too-few-public-methods
    """Loads *openai/gsm8k* and provides (train/dev) PyTorch DataLoaders."""

    def __init__(self, cfg, tokenizer):
        cache_dir = ".cache"
        ds = load_dataset("openai/gsm8k", cfg.dataset.hf_subset, cache_dir=cache_dir)
        self.raw_train = ds[cfg.dataset.train_split]
        self.raw_dev = ds[cfg.dataset.dev_split]
        self.tokenizer = tokenizer
        self.cfg = cfg

        # pre-tokenise ----------------------------------------------------------
        def _proc(ex: Dict[str, str]):
            prompt = f"Question: {ex['question']}\nAnswer:"
            ans = _clean_answer(
                ex["answer"],
                cfg.dataset.preprocessing.remove_commas,
                cfg.dataset.preprocessing.strip_whitespace,
            )
            prompt_enc = tokenizer(prompt, add_special_tokens=False)
            ans_enc = tokenizer(" " + ans, add_special_tokens=False)
            ids = prompt_enc["input_ids"] + ans_enc["input_ids"] + [tokenizer.eos_token_id]
            labels = [-100] * len(prompt_enc["input_ids"]) + ans_enc["input_ids"] + [tokenizer.eos_token_id]
            attn = [1] * len(ids)
            return {"input_ids": ids, "labels": labels, "attention_mask": attn}

        cols = self.raw_train.column_names
        self.train_tok = self.raw_train.map(
            _proc,
            remove_columns=cols,
            num_proc=min(8, os.cpu_count() or 1),
            desc="tokenise-train",
        )
        self.dev_tok = self.raw_dev.map(
            _proc,
            remove_columns=cols,
            num_proc=min(8, os.cpu_count() or 1),
            desc="tokenise-dev",
        )

        self.train_loader = DataLoader(
            self.train_tok,
            batch_size=cfg.training.train_batch_size,
            shuffle=True,
            collate_fn=self._collate,
            num_workers=2,
            pin_memory=True,
        )
        self.dev_loader = DataLoader(
            self.dev_tok,
            batch_size=cfg.training.eval_batch_size,
            shuffle=False,
            collate_fn=self._collate,
            num_workers=2,
            pin_memory=True,
        )

    # ------------------------------------------------------------------
    def _collate(self, batch: List[Dict[str, Any]]):
        return self.tokenizer.pad(batch, padding="longest", return_tensors="pt")
