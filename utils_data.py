# utils_data.py
import json
from typing import List, Dict
from torch.utils.data import Dataset


class InstructionDataset(Dataset):
    """
    构建形式：
        <s> [INST] instruction ... input ... [/INST] output
    训练目标：只预测 output，其前面部分全部 mask = -100
    """
    def __init__(self, path: str, tokenizer, max_length: int = 2048):
        self.samples = []
        self.tokenizer = tokenizer
        self.max_length = max_length

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                self.samples.append(json.loads(line))

    def _build_prompt(self, ins: str, inp: str) -> str:
        """标准 LLaMA 风格指令模板（不包含 output）"""
        if inp:
            return f"<s>[INST] {ins}\n{inp} [/INST]"
        else:
            return f"<s>[INST] {ins} [/INST]"

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ex = self.samples[idx]
        ins = ex.get("instruction", "")
        inp = ex.get("input", "")
        out = ex.get("output", "")

        # ① 构造 prompt（不含 output）
        prompt = self._build_prompt(ins, inp)
        full_text = prompt + " " + out + "</s>"

        # ② tokenize（带 labels，用 -100 mask prompt）
        tokenized = self.tokenizer(
            full_text,
            max_length=self.max_length,
            padding=False,
            truncation=True,
            return_tensors="pt",
        )

        input_ids = tokenized.input_ids[0]
        attn_mask = tokenized.attention_mask[0]

        # ③ 构造 labels（只监督输出部分）
        labels = input_ids.clone()

        # 找到 output 在 full_text 里的起始位置
        # 我们通过重新 tokenize prompt 来得到 prompt token 长度
        prompt_ids = self.tokenizer(prompt, return_tensors="pt").input_ids[0]
        prompt_len = len(prompt_ids)

        # 在 prompt 的 token 范围内，不做训练 => mask = -100
        labels[:prompt_len] = -100

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
        return torch.cat([seq, torch.full((max_len - seq.size(0),), value, dtype=torch.long)])

    input_ids = torch.stack([pad(x, pad_token_id) for x in input_ids])
    attn_mask = torch.stack([pad(x, 0) for x in attn_mask])
    labels = torch.stack([pad(x, -100) for x in labels])

    return {
        "input_ids": input_ids,
        "attention_mask": attn_mask,
        "labels": labels,
    }
