# utils_data.py
import json
from typing import List, Dict
from torch.utils.data import Dataset

class InstructionDataset(Dataset):
    def __init__(self, path: str, tokenizer, max_length: int = 512):
        self.samples = []
        self.tokenizer = tokenizer
        self.max_length = max_length

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                self.samples.append(obj)

    def _build_prompt(self, ins: str, inp: str, out: str) -> str:
        # 这里是一个非常简单的指令模板，可以按你选用的模型官方格式改
        if inp:
            return f"Instruction: {ins}\nInput: {inp}\nAnswer: {out}"
        else:
            return f"Instruction: {ins}\nAnswer: {out}"

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ex = self.samples[idx]
        ins = ex.get("instruction", "")
        inp = ex.get("input", "")
        out = ex.get("output", "")

        full_text = self._build_prompt(ins, inp, out)

        tokenized = self.tokenizer(
            full_text,
            max_length=self.max_length,
            truncation=True,
            padding=False,
            return_tensors="pt",
        )

        input_ids = tokenized.input_ids[0]
        attn_mask = tokenized.attention_mask[0]

        # 简单做法：自回归，label = input_ids
        labels = input_ids.clone()

        return {
            "input_ids": input_ids,
            "attention_mask": attn_mask,
            "labels": labels,
        }


def collate_fn(batch, pad_token_id: int):
    import torch

    input_ids = [b["input_ids"] for b in batch]
    attn_mask = [b["attention_mask"] for b in batch]
    labels = [b["labels"] for b in batch]

    max_len = max(x.size(0) for x in input_ids)

    def pad(seq, value):
        return torch.cat(
            [seq, torch.full((max_len - seq.size(0),), value, dtype=torch.long)]
        )

    input_ids = torch.stack([pad(x, pad_token_id) for x in input_ids])
    attn_mask = torch.stack([pad(x, 0) for x in attn_mask])
    labels = torch.stack([pad(x, -100) for x in labels])

    return {
        "input_ids": input_ids,
        "attention_mask": attn_mask,
        "labels": labels,
    }
