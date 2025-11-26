# train_slsd_seq.py
"""
任务序列训练脚本：
- 结构：DS-LoRA（方法一）
- 训练策略：可选是否启用 SLSD（方法二）
- 数据：T1_general ~ T5_safety 五个 jsonl
- 日志：
  - logs/seq_<task>_train_loss.jsonl （可选扩展）
  - logs/seq_eval_loss.jsonl         （每阶段在多任务上的 eval loss）
"""

import os
import json
from copy import deepcopy

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from configs.base_config import BaseConfig
from utils_data import InstructionDataset, collate_fn
from models.ds_lora import replace_with_ds_lora, get_ds_lora_param_groups


TASK_FILES = {
    "T1_general": "data/T1_general.jsonl",
    "T2_math": "data/T2_math.jsonl",
    "T3_code": "data/T3_code.jsonl",
    "T4_tool": "data/T4_tool.jsonl",
    "T5_safety": "data/T5_safety.jsonl",
}


def evaluate_loss(model, tokenizer, data_file, cfg: BaseConfig, device, max_eval_samples: int = 200):
    """在指定数据集上计算平均 loss。"""
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
        for batch in tqdm(loader, desc=f"Eval on {os.path.basename(data_file)}", leave=False):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )
            loss = outputs.loss
            total_loss += loss.item()
            total_batches += 1

    model.train()
    if total_batches == 0:
        return None
    return total_loss / total_batches


def build_probe_buffer(
    teacher_model,
    teacher_tokenizer,
    data_file: str,
    cfg: BaseConfig,
    max_samples: int,
    entropy_threshold: float,
    device: torch.device,
):
    """
    用上一阶段模型 θ^(t-1) 构建 probe buffer:
    - 从当前任务数据采样若干条
    - 前向一次，得到 logits
    - 根据 teacher 输出熵筛选：熵低的保留
    """
    from math import log
    import torch.nn.functional as F

    dataset = InstructionDataset(
        data_file,
        teacher_tokenizer,
        max_length=cfg.max_seq_len,
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, teacher_tokenizer.pad_token_id),
    )

    buffer = []
    teacher_model.eval()

    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = teacher_model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            )
            logits = outputs.logits  # [1, L, V]
            # 这里只示意：用最后一个 token 的分布计算熵
            probs = torch.softmax(logits[:, -1, :], dim=-1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-12), dim=-1)  # [1]

            if entropy.item() < entropy_threshold:
                buffer.append(
                    {
                        "input_ids": batch["input_ids"].cpu(),
                        "attention_mask": batch["attention_mask"].cpu(),
                        "logits": logits.cpu(),
                    }
                )

            if len(buffer) >= max_samples:
                break

    teacher_model.train()
    print(f"[SLSD] Built probe buffer of size {len(buffer)} from {data_file}")
    return buffer


def kd_loss_from_buffer(
    student_model,
    buffer_batch,
    device: torch.device,
):
    """
    简单 KL 蒸馏损失：
    KL( p_teacher || p_student )
    """
    import torch.nn.functional as F

    input_ids = buffer_batch["input_ids"].to(device)
    attn_mask = buffer_batch["attention_mask"].to(device)
    teacher_logits = buffer_batch["logits"].to(device)

    with torch.no_grad():
        p_teacher = torch.log_softmax(teacher_logits, dim=-1)

    outputs = student_model(
        input_ids=input_ids,
        attention_mask=attn_mask,
    )
    p_student = torch.log_softmax(outputs.logits, dim=-1)

    kl = torch.sum(
        torch.exp(p_teacher) * (p_teacher - p_student),
        dim=-1,
    )  # [B, L]
    loss = kl.mean()
    return loss


def train_one_task_with_slsd(
    task_name: str,
    train_file: str,
    cfg: BaseConfig,
    device: torch.device,
    init_model=None,
    teacher_model=None,
    teacher_tokenizer=None,
):
    """
    用 DS-LoRA 结构训练单个任务：
    - init_model: 上一阶段学生
    - teacher_model: 上一阶段冻结 teacher（用于 SLSD）
    """
    os.makedirs(cfg.save_dir, exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    # 1. 加载 tokenizer & model
    if init_model is None:
        tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            cfg.model_name,
            torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
        )
        model.resize_token_embeddings(len(tokenizer))
    else:
        model = init_model
        tokenizer = teacher_tokenizer  # 沿用上一阶段 tokenizer

    # 替换为 DS-LoRA（如果还没替换），并冻结底层
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

    # 2. 构建 probe buffer（若启用 SLSD 且存在 teacher）
    probe_buffer = None
    if cfg.use_slsd and teacher_model is not None:
        probe_buffer = build_probe_buffer(
            teacher_model,
            teacher_tokenizer,
            data_file=train_file,
            cfg=cfg,
            max_samples=cfg.probe_size_per_task,
            entropy_threshold=cfg.entropy_threshold,
            device=device,
        )

    # 3. 数据 & 优化器
    dataset = InstructionDataset(
        train_file,
        tokenizer,
        max_length=cfg.max_seq_len,
    )
    loader = DataLoader(
        dataset,
        batch_size=cfg.per_device_batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, tokenizer.pad_token_id),
    )

    optim_groups, slow_params, fast_params = get_ds_lora_param_groups(
        model,
        lr_slow=cfg.lr_slow,
        lr_fast=cfg.lr_fast,
        weight_decay=cfg.weight_decay,
    )
    optimizer = torch.optim.AdamW(optim_groups)

    optimizer_slow = torch.optim.AdamW(
        [{"params": slow_params, "lr": cfg.lr_slow, "weight_decay": cfg.weight_decay}]
    )

    print(f"==== Train task {task_name} on {len(dataset)} examples ====")

    global_step = 0
    model.zero_grad()

    # 可选：训练日志（这里只记录每个 epoch 的 avg loss）
    train_log_path = os.path.join("logs", f"seq_{task_name}_train_loss.jsonl")

    for epoch in range(cfg.num_epochs):
        pbar = tqdm(loader, desc=f"{task_name} Epoch {epoch+1}")
        running_loss = 0.0
        running_steps = 0

        for step, batch in enumerate(pbar):
            batch = {k: v.to(device) for k, v in batch.items()}

            # -------- ① 监督损失（更新 slow + fast） --------
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )
            loss_task = outputs.loss / cfg.gradient_accumulation_steps
            loss_task.backward()

            running_loss += loss_task.item()
            running_steps += 1

            # -------- ② SLSD：只更新 slow 分支（可选）--------
            if cfg.use_slsd and probe_buffer and (step % 10 == 0):
                buf_ex = probe_buffer[step % len(probe_buffer)]
                kd_loss = kd_loss_from_buffer(model, buf_ex, device)
                kd_loss = cfg.kd_lambda * kd_loss
                kd_loss.backward()

                optimizer_slow.step()
                optimizer_slow.zero_grad()

            if (step + 1) % cfg.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

                avg_loss = running_loss / running_steps
                pbar.set_postfix({"loss": avg_loss})

        # 每个 epoch 记录一次平均训练 loss
        avg_loss_epoch = running_loss / max(running_steps, 1)
        with open(train_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "epoch": epoch + 1,
                "avg_train_loss": float(avg_loss_epoch),
            }) + "\n")

    # 返回当前阶段模型 & tokenizer
    return model, tokenizer


def main():
    cfg = BaseConfig()
    # 持续学习实验默认启用 SLSD，你可以改成 False 对比方法一
    cfg.use_slsd = True

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    print("Config:", cfg)

    os.makedirs(cfg.save_dir, exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    current_model = None
    current_tokenizer = None

    # 顺序任务列表（按论文：T1 -> T2 -> T3 -> T4 -> T5）
    task_order = list(TASK_FILES.keys())

    # eval 日志
    eval_log_path = os.path.join("logs", "seq_eval_loss.jsonl")

    for idx, t_name in enumerate(task_order):
        path = TASK_FILES[t_name]
        if not os.path.exists(path):
            print(f"[WARN] {path} not found, skip {t_name}")
            continue

        # teacher = 上一阶段模型（冻结）
        teacher_model = deepcopy(current_model).eval() if current_model is not None else None
        teacher_tokenizer = current_tokenizer

        current_model, current_tokenizer = train_one_task_with_slsd(
            task_name=t_name,
            train_file=path,
            cfg=cfg,
            device=device,
            init_model=current_model,
            teacher_model=teacher_model,
            teacher_tokenizer=teacher_tokenizer,
        )

        # 每阶段训练完，保存当前模型
        save_dir = os.path.join(cfg.save_dir, f"seq_{t_name}")
        os.makedirs(save_dir, exist_ok=True)
        current_model.save_pretrained(save_dir)
        current_tokenizer.save_pretrained(save_dir)
        print(f"Saved {t_name} model to {save_dir}")

        # ===== 在已见任务上做多任务评估（简单平均 loss）=====
        tasks_seen = task_order[: idx + 1]
        print(f"[Eval] After {t_name}, evaluate on tasks: {tasks_seen}")

        for eval_task in tasks_seen:
            eval_file = TASK_FILES[eval_task]
            eval_loss = evaluate_loss(
                current_model, current_tokenizer, eval_file, cfg, device
            )
            print(f"[Eval] Stage {t_name}, on {eval_task}: loss = {eval_loss}")

            with open(eval_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "stage": t_name,
                    "eval_task": eval_task,
                    "loss": float(eval_loss),
                }) + "\n")


if __name__ == "__main__":
    main()
