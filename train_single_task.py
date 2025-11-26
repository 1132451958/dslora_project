# train_single_task.py
import os
import json
import argparse
from dataclasses import asdict

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from transformers import AutoModelForCausalLM, AutoTokenizer

from configs.base_config import BaseConfig
from utils_data import InstructionDataset, collate_fn
from models.ds_lora import replace_with_ds_lora, get_ds_lora_param_groups


TASK_TO_FILE = {
    "T1_general": "data/T1_general.jsonl",
    "T2_math": "data/T2_math.jsonl",
    "T3_code": "data/T3_code.jsonl",
    "T4_tool": "data/T4_tool.jsonl",
    "T5_safety": "data/T5_safety.jsonl",
}


def evaluate_loss(model, tokenizer, data_file, cfg: BaseConfig, device, max_eval_samples: int = 200):
    """简单评估：在给定数据集上计算平均 loss（NLL）。"""
    model.eval()
    dataset = InstructionDataset(data_file, tokenizer, max_length=cfg.max_seq_len)

    # 只取前 max_eval_samples 条做 eval，避免太慢
    if len(dataset) > max_eval_samples:
        from torch.utils.data import Subset
        dataset = Subset(dataset, list(range(max_eval_samples)))

    loader = DataLoader(
        dataset,
        batch_size=cfg.per_device_batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, tokenizer.pad_token_id),
    )

    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc="Eval", leave=False):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )
            # loss 是平均到所有非 -100 label 的 token 上的
            loss = outputs.loss
            bs = batch["input_ids"].size(0)
            total_loss += loss.item() * bs
            total_tokens += bs

    model.train()
    if total_tokens == 0:
        return None
    return total_loss / total_tokens


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        type=str,
        choices=list(TASK_TO_FILE.keys()),
        default="T1_general",
        help="选择要训练的任务",
    )
    args = parser.parse_args()

    cfg = BaseConfig()
    # 根据任务选择对应的数据文件
    cfg.train_file = TASK_TO_FILE[args.task]

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    print("Train task:", args.task)
    print("Train file:", cfg.train_file)

    os.makedirs(cfg.save_dir, exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    # ===== 1. 加载 tokenizer & 模型 =====
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        torch_dtype=torch.float32,
    )

    model.resize_token_embeddings(len(tokenizer))

    # 替换上层 Linear 为 DS-LoRA
    model = replace_with_ds_lora(
        model,
        target_modules=cfg.lora_target_modules,
        r=cfg.lora_r,
        alpha=cfg.lora_alpha,
        dropout=cfg.lora_dropout,
        num_frozen_layers=cfg.num_frozen_layers,
    )

    model.to(device)
    model.train()

    # ===== 2. 准备数据 =====
    train_dataset = InstructionDataset(
        cfg.train_file, tokenizer, max_length=cfg.max_seq_len
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.per_device_batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, tokenizer.pad_token_id),
    )

    # ===== 3. 优化器（slow & fast 两组参数）=====
    optim_groups, _, _ = get_ds_lora_param_groups(
        model,
        lr_slow=cfg.lr_slow,
        lr_fast=cfg.lr_fast,
        weight_decay=cfg.weight_decay,
    )

    optimizer = torch.optim.AdamW(optim_groups)

    total_steps = (
        len(train_loader)
        * cfg.num_epochs
        // cfg.gradient_accumulation_steps
        + 1
    )

    print("Num training examples:", len(train_dataset))
    print("Total steps:", total_steps)
    print("Config:", asdict(cfg))

    global_step = 0
    model.zero_grad()

    # 日志文件
    run_name = f"single_{args.task}"
    train_log_path = os.path.join("logs", f"{run_name}_train_loss.jsonl")
    eval_log_path = os.path.join("logs", f"{run_name}_eval_loss.jsonl")

    # ===== 4. 训练循环 =====
    for epoch in range(cfg.num_epochs):
        pbar = tqdm(train_loader, desc=f"[{args.task}] Epoch {epoch+1}")
        running_loss = 0.0
        running_steps = 0

        for step, batch in enumerate(pbar):
            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )

            loss = outputs.loss
            loss = loss / cfg.gradient_accumulation_steps
            loss.backward()

            running_loss += loss.item()
            running_steps += 1

            if (step + 1) % cfg.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

                # 记录一次 loss（还原到未除 grad_acc 的尺度）
                full_loss = running_loss * cfg.gradient_accumulation_steps / running_steps
                pbar.set_postfix({"loss": full_loss})

                with open(train_log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps({
                        "step": global_step,
                        "epoch": epoch + 1,
                        "loss": float(full_loss),
                    }) + "\n")

                running_loss = 0.0
                running_steps = 0

        # ===== 每个 epoch 结束后做一次简单评估 =====
        eval_loss = evaluate_loss(model, tokenizer, cfg.train_file, cfg, device)
        print(f"[{args.task}] Epoch {epoch+1} eval loss: {eval_loss}")

        with open(eval_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "epoch": epoch + 1,
                "eval_loss": float(eval_loss),
            }) + "\n")

        # 保存 checkpoint
        save_path = os.path.join(cfg.save_dir, f"{run_name}_epoch{epoch+1}")
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
        print("Saved model to", save_path)


if __name__ == "__main__":
    main()
