# train_slsd_seq.py
"""
任务序列训练脚本（DS-LoRA + SLSD）
- 支持 full/toy 数据自动切换（来自 BaseConfig）
- 使用本地 LLaMA-7B（cfg.model_name 指向 pretrained_models/llama2-7b-hf）
- 任务序列：T1 → T2 → T3 → T4 → T5
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


# ======================================================
# GPU 选择函数
# ======================================================
def select_device():
    if not torch.cuda.is_available():
        print("[GPU SELECT] CUDA not available, using CPU.")
        return torch.device("cpu")

    num_devices = torch.cuda.device_count()
    best_idx = None
    best_free = -1

    print("[GPU SELECT] Checking GPUs...")

    for i in range(num_devices):
        name = torch.cuda.get_device_name(i)
        total_bytes = torch.cuda.get_device_properties(i).total_memory
        total_gb = total_bytes / 1024 ** 3
        try:
            free_bytes, _ = torch.cuda.mem_get_info(i)
            free_gb = free_bytes / 1024 ** 3
        except RuntimeError:
            free_bytes = 0
            free_gb = 0

        print(f" - cuda:{i} | {name} | free={free_gb:.1f}GB / total={total_gb:.1f}GB")

        if free_bytes > best_free:
            best_free = free_bytes
            best_idx = i

    device = torch.device(f"cuda:{best_idx}")
    print(f"[GPU SELECT] Using cuda:{best_idx}")
    return device


# ======================================================
# Evaluate
# ======================================================
def evaluate_loss(model, tokenizer, data_file, cfg: BaseConfig, device, max_eval_samples=200):
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
        for batch in tqdm(loader, desc=f"Eval {os.path.basename(data_file)}", leave=False):
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


# ======================================================
# Probe Buffer for SLSD
# ======================================================
def build_probe_buffer(
    teacher_model,
    teacher_tokenizer,
    data_file,
    cfg: BaseConfig,
    max_samples,
    entropy_threshold,
    device,
):
    from math import log
    import torch.nn.functional as F

    dataset = InstructionDataset(data_file, teacher_tokenizer, max_length=cfg.max_seq_len)
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
            out = teacher_model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
            logits = out.logits  # [1, L, V]

            probs = torch.softmax(logits[:, -1, :], dim=-1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-12))

            if entropy.item() < entropy_threshold:
                buffer.append({
                    "input_ids": batch["input_ids"].cpu(),
                    "attention_mask": batch["attention_mask"].cpu(),
                    "logits": logits.cpu(),
                })

            if len(buffer) >= max_samples:
                break

    teacher_model.train()
    print(f"[SLSD] Built probe buffer = {len(buffer)}")
    return buffer


def kd_loss_from_buffer(student_model, buf, device):
    import torch.nn.functional as F

    ids = buf["input_ids"].to(device)
    att = buf["attention_mask"].to(device)
    teacher_logits = buf["logits"].to(device)

    with torch.no_grad():
        p_teacher = torch.log_softmax(teacher_logits, dim=-1)

    out = student_model(input_ids=ids, attention_mask=att)
    p_student = torch.log_softmax(out.logits, dim=-1)

    kl = torch.sum(torch.exp(p_teacher) * (p_teacher - p_student), dim=-1)
    return kl.mean()


# ======================================================
# 单任务训练（DS-LoRA + SLSD）
# ======================================================
def train_one_task_with_slsd(
    task_name,
    data_file,
    cfg: BaseConfig,
    device,
    init_model=None,
    teacher_model=None,
    tokenizer=None,
):
    os.makedirs(cfg.save_dir, exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    # --------------------------------------------------
    # 1. 加载模型 & tokenizer（第一次任务才从头加载）
    # --------------------------------------------------
    first_task = init_model is None

    if first_task:
        tokenizer = AutoTokenizer.from_pretrained(
            cfg.model_name,
            use_fast=False,
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            cfg.model_name,
            torch_dtype=torch.float16,
        )
        model.resize_token_embeddings(len(tokenizer))
        model.to(device)

        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False

        # 只在第一次插入一次 DS-LoRA
        model = replace_with_ds_lora(
            model,
            target_modules=cfg.lora_target_modules,
            r=cfg.lora_r,
            alpha=cfg.lora_alpha,
            dropout=cfg.lora_dropout,
            num_frozen_layers=cfg.num_frozen_layers,
        )

        # 冻结非 LoRA 参数，只训 LoRA
        for name, param in model.named_parameters():
            if "lora_" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
    else:
        model = init_model
        model.to(device)
        if tokenizer is None:
            raise ValueError("Tokenizer should be passed when reusing init_model.")

    model.train()

    # --------------------------------------------------
    # 2. SLSD probe buffer
    # --------------------------------------------------
    probe_buffer = None
    if cfg.use_slsd and teacher_model is not None:
        teacher_model.to(device)
        probe_buffer = build_probe_buffer(
            teacher_model,
            tokenizer,
            data_file,
            cfg,
            max_samples=cfg.probe_size_per_task,
            entropy_threshold=cfg.entropy_threshold,
            device=device,
        )

    # --------------------------------------------------
    # 3. 数据加载
    # --------------------------------------------------
    dataset = InstructionDataset(data_file, tokenizer, max_length=cfg.max_seq_len)
    loader = DataLoader(
        dataset,
        batch_size=cfg.per_device_batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, tokenizer.pad_token_id),
    )

    # --------------------------------------------------
    # 4. 优化器：slow & fast LoRA 分开更新
    # --------------------------------------------------
    # 我们只用 get_ds_lora_param_groups 返回 slow/fast 列表，然后自己构建两个 optimizer
    _, slow_params, fast_params = get_ds_lora_param_groups(
        model,
        lr_slow=cfg.lr_slow,
        lr_fast=cfg.lr_fast,
        weight_decay=cfg.weight_decay,
    )

    optimizer_fast = torch.optim.AdamW(
        [{"params": fast_params, "lr": cfg.lr_fast, "weight_decay": cfg.weight_decay}]
    )
    optimizer_slow = torch.optim.AdamW(
        [{"params": slow_params, "lr": cfg.lr_slow, "weight_decay": cfg.weight_decay}]
    )

    print(f"==== Train {task_name} on {len(dataset)} examples ====")

    # --------------------------------------------------
    # 5. 训练循环
    # --------------------------------------------------
    global_step = 0
    model.zero_grad()
    train_log = f"logs/seq_{task_name}_train_loss.jsonl"

    for epoch in range(cfg.num_epochs):
        pbar = tqdm(loader, desc=f"{task_name} Epoch {epoch+1}")
        running_loss = 0.0
        running_steps = 0

        for step, batch in enumerate(pbar):
            batch = {k: v.to(device) for k, v in batch.items()}

            # --- supervised loss（fast+slow 都会反向） ---
            out = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )
            sup_loss = out.loss / cfg.gradient_accumulation_steps
            sup_loss.backward()

            running_loss += sup_loss.item()
            running_steps += 1

            # --- KD loss（只让 slow 分支再学一点） ---
            if cfg.use_slsd and probe_buffer is not None and step % 10 == 0:
                buf_ex = probe_buffer[step % len(probe_buffer)]
                kd = kd_loss_from_buffer(model, buf_ex, device) * cfg.kd_lambda
                kd.backward()
                optimizer_slow.step()
                optimizer_slow.zero_grad()

            if (step + 1) % cfg.gradient_accumulation_steps == 0:
                optimizer_fast.step()
                optimizer_fast.zero_grad()
                global_step += 1
                pbar.set_postfix({"loss": running_loss / running_steps})

        avg = running_loss / max(running_steps, 1)
        with open(train_log, "a", encoding="utf-8") as f:
            f.write(json.dumps({"epoch": epoch + 1, "avg_train_loss": float(avg)}) + "\n")

    return model, tokenizer


# ======================================================
# 主程序（序列 T1→T2→T3→T4→T5）
# ======================================================
def main():
    cfg = BaseConfig()
    cfg.use_slsd = True   # 启用 SLSD

    if cfg.use_toy_data:
        data_key = "toy"
    else:
        data_key = cfg.train_split

    # 自动选择 GPU
    device = select_device()

    os.makedirs(cfg.save_dir, exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    current_model = None
    current_tokenizer = None

    # 任务顺序
    tasks = ["T1_general", "T2_math", "T3_code", "T4_tool", "T5_safety"]

    eval_log = "logs/seq_eval_loss.jsonl"

    for idx, task_name in enumerate(tasks):
        train_file = cfg.data_paths[task_name][data_key]
        print(f"\n===== Train {task_name} | file = {train_file} =====")

        teacher = deepcopy(current_model).eval() if current_model is not None else None

        current_model, current_tokenizer = train_one_task_with_slsd(
            task_name,
            train_file,
            cfg,
            device,
            init_model=current_model,
            teacher_model=teacher,
            tokenizer=current_tokenizer,
        )

        # 保存阶段模型
        save_dir = os.path.join(cfg.save_dir, f"seq_{task_name}")
        os.makedirs(save_dir, exist_ok=True)
        current_model.save_pretrained(save_dir)
        current_tokenizer.save_pretrained(save_dir)
        print(f"[SAVE] Model of {task_name} saved to {save_dir}")

        # =================================================
        # 在已见任务上做评估
        # =================================================
        seen = tasks[: idx + 1]
        print(f"[Eval] After {task_name}, evaluate on tasks: {seen}")

        for ev in seen:
            ev_key = cfg.eval_split
            ev_file = cfg.data_paths[ev].get(ev_key, train_file)
            ev_loss = evaluate_loss(current_model, current_tokenizer, ev_file, cfg, device)
            print(f"[Eval] stage={task_name} on {ev}: loss={ev_loss}")

            with open(eval_log, "a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "method": "ds_lora_slsd",   # 给自己的方法起个统一名字
                    "stage": task_name,
                    "eval_task": ev,
                    "loss": float(ev_loss),
                }) + "\n")


if __name__ == "__main__":
    main()
