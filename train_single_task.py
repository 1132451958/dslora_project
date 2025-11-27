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


# =========================================
# 自动选择空闲 GPU
# =========================================
def select_device():
    if not torch.cuda.is_available():
        print("[GPU SELECT] CUDA not available, using CPU.")
        return torch.device("cpu")

    num_devices = torch.cuda.device_count()
    infos = []
    best_idx = None
    best_free = -1

    for i in range(num_devices):
        name = torch.cuda.get_device_name(i)
        total_bytes = torch.cuda.get_device_properties(i).total_memory
        total_gb = total_bytes / 1024 ** 3

        try:
            free_bytes, _ = torch.cuda.mem_get_info(i)
            free_gb = free_bytes / 1024 ** 3
            used_gb = total_gb - free_gb
        except RuntimeError:
            free_bytes = 0
            free_gb = 0
            used_gb = total_gb

        infos.append((i, name, total_gb, used_gb, free_gb))

        if free_bytes > best_free:
            best_free = free_bytes
            best_idx = i

    print("[GPU SELECT] Available GPUs:")
    for i, name, total_gb, used_gb, free_gb in infos:
        print(
            f"  - cuda:{i} | {name} | "
            f"total={total_gb:.1f} GB, used={used_gb:.1f} GB, free={free_gb:.1f} GB"
        )

    device = torch.device(f"cuda:{best_idx}")
    print(f"[GPU SELECT] Using cuda:{best_idx}")
    return device


# =========================================
# 简单评估：平均 Token Loss
# =========================================
def evaluate_loss(model, tokenizer, data_file, cfg: BaseConfig, device, max_eval_samples: int = 200):
    model.eval()
    dataset = InstructionDataset(data_file, tokenizer, max_length=cfg.max_seq_len)

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
    total_batches = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc="Eval", leave=False):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )
            total_loss += outputs.loss.item()
            total_batches += 1

    model.train()
    return total_loss / max(1, total_batches)


# =========================================
# 主训练逻辑
# =========================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        type=str,
        choices=["T1_general", "T2_math", "T3_code", "T4_tool", "T5_safety"],
        default="T1_general",
        help="选择要训练的任务",
    )
    args = parser.parse_args()

    cfg = BaseConfig()

    # ---- 使用新的 split 配置 ----
    if cfg.use_toy_data:
        data_key = "toy"
    else:
        data_key = cfg.train_split  # train / tiny / full 由 BaseConfig 控制

    cfg.train_file = cfg.data_paths[args.task][data_key]
    print(f"[DATA] Using {data_key} Data → {cfg.train_file}")

    # ---- 自动选择 GPU ----
    device = select_device()

    os.makedirs(cfg.save_dir, exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    # =========================================
    # 1. 加载 tokenizer & LLaMA 模型
    # =========================================
    print(f"[CONFIG] model_name = {cfg.model_name}")
    print("[CHECK] Exists?", os.path.exists(cfg.model_name), "Is dir?", os.path.isdir(cfg.model_name))

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model_name,
        use_fast=False,
        local_files_only=True,  # 只用本地模型
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # dtype 选择：优先用 bfloat16（4090 支持），否则用 float16
    if torch.cuda.is_available() and hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
        torch_dtype = torch.bfloat16
        print("[DTYPE] Using bfloat16")
    else:
        torch_dtype = torch.float16
        print("[DTYPE] Using float16")

    print(f"[MODEL] Loading {cfg.model_name} ...")
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        torch_dtype=torch_dtype,
        local_files_only=True,
    )

    # LLaMA 常见操作：resize embedding 以适配 pad_token 等
    model.resize_token_embeddings(len(tokenizer))
    model.to(device)

    # ---- 注入 DS-LoRA ----
    model = replace_with_ds_lora(
        model,
        target_modules=cfg.lora_target_modules,
        r=cfg.lora_r,
        alpha=cfg.lora_alpha,
        dropout=cfg.lora_dropout,
        num_frozen_layers=cfg.num_frozen_layers,
    )

    # 只训练 LoRA 参数
    for name, param in model.named_parameters():
        if "lora_" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    model.train()

    # =========================================
    # 2. 数据加载
    # =========================================
    train_dataset = InstructionDataset(
        cfg.train_file,
        tokenizer,
        max_length=cfg.max_seq_len,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.per_device_batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, tokenizer.pad_token_id),
    )

    # =========================================
    # 3. 优化器 Slow/ Fast 参数组
    # =========================================
    optim_groups, _, _ = get_ds_lora_param_groups(
        model,
        lr_slow=cfg.lr_slow,
        lr_fast=cfg.lr_fast,
        weight_decay=cfg.weight_decay,
    )
    optimizer = torch.optim.AdamW(optim_groups)

    total_steps = (
        len(train_loader) * cfg.num_epochs // cfg.gradient_accumulation_steps + 1
    )

    print("Num training examples:", len(train_dataset))
    print("Total steps:", total_steps)
    print("Config:", asdict(cfg))

    # =========================================
    # 4. 训练循环
    # =========================================
    global_step = 0
    model.zero_grad()

    run_name = f"single_{args.task}"
    train_log_path = f"logs/{run_name}_train_loss.jsonl"
    eval_log_path = f"logs/{run_name}_eval_loss.jsonl"

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

            # ---- nan / inf 检查，直接跳过这个 batch ----
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"[WARN] Detected nan/inf loss at step {step}, skip this batch.")
                optimizer.zero_grad()
                continue

            loss = loss / cfg.gradient_accumulation_steps
            loss.backward()

            running_loss += loss.item()
            running_steps += 1

            if (step + 1) % cfg.gradient_accumulation_steps == 0:
                # --- 梯度裁剪，防止梯度爆炸 ---
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

                full_loss = running_loss * cfg.gradient_accumulation_steps / max(1, running_steps)
                pbar.set_postfix({"loss": full_loss})

                with open(train_log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps({
                        "step": global_step,
                        "epoch": epoch + 1,
                        "loss": float(full_loss),
                    }) + "\n")

                running_loss = 0.0
                running_steps = 0

        # ===== Eval =====
        eval_key = cfg.eval_split
        eval_file = cfg.data_paths[args.task].get(eval_key, cfg.train_file)

        eval_loss = evaluate_loss(model, tokenizer, eval_file, cfg, device)
        print(f"[{args.task}] Epoch {epoch+1} eval({eval_key}) loss: {eval_loss}")

        with open(eval_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "epoch": epoch + 1,
                "eval_loss": float(eval_loss),
            }) + "\n")

        # ===== Save checkpoint =====
        save_path = os.path.join(cfg.save_dir, f"{run_name}_epoch{epoch+1}")
        os.makedirs(save_path, exist_ok=True)

        # 修复 generation_config 中与 do_sample=False 冲突的参数，避免保存时报错
        if hasattr(model, "generation_config") and model.generation_config is not None:
            gc = model.generation_config
            if getattr(gc, "do_sample", False) is False:
                if hasattr(gc, "temperature"):
                    gc.temperature = None
                if hasattr(gc, "top_p"):
                    gc.top_p = None

        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
        print("Saved model to", save_path)


if __name__ == "__main__":
    main()
