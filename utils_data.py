# utils_data.py
import json
from typing import List, Dict
from torch.utils.data import Dataset
import torch


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

        # ② 分别 tokenize prompt 和 output（先不截断 full_text）
        #    注意：这里不用 tokenizer 自己加 special_tokens，因为我们手动控制了 <s> 和 </s>
        prompt_ids = self.tokenizer(
            prompt,
            add_special_tokens=False,
            return_tensors="pt",
        ).input_ids[0]

        out_text = " " + out + "</s>"
        out_ids = self.tokenizer(
            out_text,
            add_special_tokens=False,
            return_tensors="pt",
        ).input_ids[0]

        max_len = self.max_length

        # 如果 output 完全为空（极少），直接当成纯 prompt，labels 全 -100
        if out_ids.numel() == 0:
            tokens = prompt_ids[:max_len]
            input_ids = tokens
            attn_mask = torch.ones_like(tokens)
            labels = torch.full_like(tokens, -100)
            return {
                "input_ids": input_ids,
                "attention_mask": attn_mask,
                "labels": labels,
            }

        # ③ 手动截断：保证尽量保留 output
        total_len = prompt_ids.size(0) + out_ids.size(0)
        if total_len > max_len:
            # 情况 1：output 自己就 >= max_len，那只能保留 output 的最后 max_len 个 token，prompt 全砍
            if out_ids.size(0) >= max_len:
                out_ids = out_ids[-max_len:]
                prompt_ids = out_ids.new_zeros((0,), dtype=torch.long)
            else:
                # 情况 2：正常情况，截断 prompt 的长度，保证完整保留 output
                max_prompt_len = max_len - out_ids.size(0)
                prompt_ids = prompt_ids[:max_prompt_len]

        # ④ 拼接最终序列
        input_ids = torch.cat([prompt_ids, out_ids], dim=0)
        attn_mask = torch.ones_like(input_ids)

        # ⑤ 构造 labels：只监督 output 部分
        labels = input_ids.clone()
        prompt_len = prompt_ids.size(0)
        labels[:prompt_len] = -100  # prompt 部分不算 loss

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
