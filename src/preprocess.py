# src/preprocess.py
# --------------------------------------------------------------------------------------
# GSM8K data module – *complete*, leak-proof, with masking of prompt tokens.
# --------------------------------------------------------------------------------------
from __future__ import annotations

import os
import random
import re
from typing import Any, Dict, List

import torch
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
        # Find max length in batch
        max_len = max(len(item["input_ids"]) for item in batch)

        # Manually pad each field
        input_ids_list = []
        labels_list = []
        attention_mask_list = []

        for item in batch:
            curr_len = len(item["input_ids"])
            pad_len = max_len - curr_len

            # Pad input_ids with tokenizer.pad_token_id
            padded_input_ids = item["input_ids"] + [self.tokenizer.pad_token_id] * pad_len
            input_ids_list.append(padded_input_ids)

            # Pad labels with -100 (ignore index)
            padded_labels = item["labels"] + [-100] * pad_len
            labels_list.append(padded_labels)

            # Pad attention_mask with 0
            padded_attention_mask = item["attention_mask"] + [0] * pad_len
            attention_mask_list.append(padded_attention_mask)

        # Convert to tensors
        return {
            "input_ids": torch.tensor(input_ids_list, dtype=torch.long),
            "labels": torch.tensor(labels_list, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask_list, dtype=torch.long),
        }
