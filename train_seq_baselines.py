"""
顺序多任务基线训练脚本：
- Seq-LoRA
- Replay
- EWC

使用单分支 LoRA（models/lora_simple.py），
与 DS-LoRA + SLSD（train_slsd_seq.py）区分开。
"""

import os
import json
import argparse
import random
from copy import deepcopy

import subprocess
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig  # NEW: import GenerationConfig

from configs.base_config import BaseConfig
from utils_data import InstructionDataset, collate_fn
from models.lora_simple import (
    replace_with_lora,
    mark_only_lora_as_trainable,
    get_lora_param_groups,
)


# ======================================================
# NaN / Inf 检查工具
# ======================================================
def check_nan(loss, model, task_name, method, global_step):
    """
    检查 loss 和梯度中是否存在 NaN。
    - 在 backward 之后调用（此时梯度已计算）
    - 如发现 NaN，返回 True；否则 False。
    """
    # loss 本身是 NaN
    if torch.isnan(loss):
        print(f"[NaN] Loss is NaN | task={task_name}, method={method}, step={global_step}")
        return True

    # 检查梯度是否有 NaN
    for name, p in model.named_parameters():
        if p.requires_grad and p.grad is not None and torch.isnan(p.grad).any():
            print(f"[NaN] Grad is NaN in {name} | task={task_name}, method={method}, step={global_step}")
            return True

    return False


# ======================================================
# GenerationConfig 修正工具（避免 save_pretrained 报错）
# ======================================================
def sanitize_generation_config(model):
    """
    修正模型的 generation_config，避免如下错误：
    - do_sample=False, 但 temperature/top_p 等被设置为非空
    """
    try:
        if hasattr(model, "generation_config") and model.generation_config is not None:
            gen_cfg = model.generation_config
        else:
            # 如果模型没有 generation_config，就从 config 构建一个
            gen_cfg = GenerationConfig.from_model_config(model.config)
    except Exception:
        # 极端情况下直接跳过，不影响训练
        return model

    # 我们在本项目中只需要 greedy / teacher-forcing loss，不需要采样
    gen_cfg.do_sample = False

    # 这些采样相关参数在 do_sample=False 时会触发新版本 transformers 的严格校验
    for attr in ["temperature", "top_p", "top_k", "typical_p"]:
        if hasattr(gen_cfg, attr):
            setattr(gen_cfg, attr, None)

    model.generation_config = gen_cfg
    return model


# ======================================================
# GPU 选择
# ======================================================
def select_device():
    """
    使用 nvidia-smi 查询每块 GPU 的空闲显存，只在最终选中的那张卡上创建 CUDA context。
    不再对每张卡调用 torch.cuda.xxx，从而避免一个进程在所有 GPU 上都占 300+MiB。
    """
    if not torch.cuda.is_available():
        print("[GPU SELECT] CUDA not available, using CPU.")
        return torch.device("cpu")

    try:
        # 用 nvidia-smi 查每块卡的 free / total 显存（单位 MiB）和名字
        result = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.free,memory.total,name",
                "--format=csv,noheader,nounits",
            ],
            encoding="utf-8",
        )
        lines = [l.strip() for l in result.strip().splitlines() if l.strip()]

        print("[GPU SELECT] Checking GPUs via nvidia-smi...")

        best_idx = None
        best_free_mb = -1

        for idx, line in enumerate(lines):
            # 每行格式类似： "20000, 24564, NVIDIA GeForce RTX 4090"
            parts = [p.strip() for p in line.split(",")]
            free_mb = int(parts[0])
            total_mb = int(parts[1])
            name = ",".join(parts[2:])  # 防止名字里还有逗号

            free_gb = free_mb / 1024.0
            total_gb = total_mb / 1024.0
            print(f" - cuda:{idx} | {name} | free={free_gb:.1f}GB / total={total_gb:.1f}GB")

            if free_mb > best_free_mb:
                best_free_mb = free_mb
                best_idx = idx

        if best_idx is None:
            print("[GPU SELECT] nvidia-smi returned no GPUs, fallback to cuda:0")
            return torch.device("cuda:0")

        device = torch.device(f"cuda:{best_idx}")
        print(f"[GPU SELECT] Using cuda:{best_idx}")
        return device

    except Exception as e:
        # 万一 nvidia-smi 出问题，就退回到简单策略
        print(f"[GPU SELECT] nvidia-smi query failed: {e}")
        print("[GPU SELECT] Fallback: using cuda:0")
        return torch.device("cuda:0")

# ======================================================
# Eval：平均 loss
# ======================================================
def evaluate_loss(model, tokenizer, data_file, cfg: BaseConfig, device, max_eval_samples=200):
    model.eval()
    dataset = InstructionDataset(data_file, tokenizer, max_length=cfg.max_seq_len)

    # 空数据返回 NaN，避免误认为是 0
    if len(dataset) == 0:
        print(f"[Eval WARNING] dataset is EMPTY: {data_file}, return NaN loss.")
        model.train()
        return float("nan")

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

            # 若 labels 全为 -100，则跳过这一 batch
            if (batch["labels"] != -100).sum() == 0:
                continue

            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )
            loss = outputs.loss

            # 如果 loss 非有限值，跳过
            if not torch.isfinite(loss):
                print(f"[Eval WARNING] Non-finite loss={loss.item()} in {data_file}, skip batch.")
                continue

            total_loss += loss.item()
            total_batches += 1

    model.train()
    if total_batches == 0:
        print(f"[Eval WARNING] No valid batches for {data_file}, return NaN.")
        return float("nan")

    return total_loss / total_batches


# ======================================================
# Replay buffer 构建：从某任务训练集采样若干 batch
# ======================================================
def build_replay_buffer_from_file(
    replay_buffer,
    data_file,
    tokenizer,
    cfg: BaseConfig,
    max_buffer_size: int,
    max_samples_per_task: int = 1024,
):
    """
    replay_buffer: List[batch_dict] （每个 batch_dict 都在 CPU 上）
    """
    dataset = InstructionDataset(data_file, tokenizer, max_length=cfg.max_seq_len)
    loader = DataLoader(
        dataset,
        batch_size=cfg.per_device_batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, tokenizer.pad_token_id),
    )

    added = 0
    for batch in loader:
        # 全部放在 CPU，后面训练时再 .to(device)
        cpu_batch = {k: v.clone().cpu() for k, v in batch.items()}
        replay_buffer.append(cpu_batch)
        added += cpu_batch["input_ids"].size(0)

        if added >= max_samples_per_task:
            break

        if len(replay_buffer) >= max_buffer_size:
            break

    # 如果超出总体上限，截断一下（保留最近的）
    if len(replay_buffer) > max_buffer_size:
        replay_buffer = replay_buffer[-max_buffer_size:]

    print(f"[Replay] Buffer size after task = {len(replay_buffer)} examples (approx.)")
    return replay_buffer


# ======================================================
# EWC Fisher 估计
# ======================================================
def compute_ewc_state(model, tokenizer, data_file, cfg: BaseConfig, device, max_samples=200):
    """
    只基于当前 task 的数据，估计 LoRA 参数上的 Fisher 对角阵。
    mark_only_lora_as_trainable 已保证只有 LoRA 参数 requires_grad=True。
    """
    model.eval()
    dataset = InstructionDataset(data_file, tokenizer, max_length=cfg.max_seq_len)

    if len(dataset) == 0:
        print(f"[EWC WARNING] Empty dataset for Fisher: {data_file}")
        model.train()
        return None

    if len(dataset) > max_samples:
        from torch.utils.data import Subset

        dataset = Subset(dataset, list(range(max_samples)))

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, tokenizer.pad_token_id),
    )

    fisher = {}
    prev_params = {}
    sample_count = 0

    for batch in tqdm(loader, desc=f"[EWC] Fisher on {os.path.basename(data_file)}", leave=False):
        # 若 labels 全为 -100，跳过
        if (batch["labels"] != -100).sum() == 0:
            continue

        sample_count += batch["input_ids"].size(0)
        model.zero_grad()
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        loss = outputs.loss
        loss.backward()

        for name, param in model.named_parameters():
            if not param.requires_grad or param.grad is None:
                continue
            g = param.grad.detach()  # shape like param
            if name not in fisher:
                fisher[name] = g.pow(2).clone()
                prev_params[name] = param.detach().clone().cpu()
            else:
                fisher[name] += g.pow(2)

        if sample_count >= max_samples:
            break

    if sample_count == 0:
        print(f"[EWC WARNING] No valid samples for Fisher in {data_file}")
        model.train()
        return None

    for name in fisher:
        fisher[name] /= sample_count

    model.train()
    print(f"[EWC] Collected fisher for {len(fisher)} tensors")
    return {"fisher": fisher, "params": prev_params}


def compute_ewc_loss(model, ewc_state, device, ewc_lambda: float):
    if ewc_state is None:
        return 0.0

    loss_ewc = 0.0
    for name, param in model.named_parameters():
        if name not in ewc_state["fisher"]:
            continue
        fisher = ewc_state["fisher"][name].to(device)
        prev = ewc_state["params"][name].to(device)
        loss_ewc = loss_ewc + (fisher * (param - prev) ** 2).sum()

    return 0.5 * ewc_lambda * loss_ewc


# ======================================================
# 单任务训练（支持：seq_lora / replay / ewc）
# ======================================================
def train_one_task(
    method: str,
    task_name: str,
    train_file: str,
    cfg: BaseConfig,
    device,
    init_model=None,
    tokenizer=None,
    replay_buffer=None,
    ewc_prev=None,
):
    """
    返回：
      model, tokenizer, replay_buffer, ewc_prev
    """
    os.makedirs(cfg.save_dir, exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    # 一些超参（如果 BaseConfig 没有，就给默认）
    replay_buffer_size = getattr(cfg, "replay_buffer_size", 2048)
    replay_lambda = getattr(cfg, "replay_lambda", 0.5)
    ewc_lambda = getattr(cfg, "ewc_lambda", 0.4)
    max_grad_norm = getattr(cfg, "max_grad_norm", 1.0)

    # --------------------------------------------------
    # 1. 模型 / tokenizer
    # --------------------------------------------------
    if init_model is None or tokenizer is None:
        print(f"[MODEL] Loading base model: {cfg.model_name}")
        tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            cfg.model_name,
            torch_dtype=torch.bfloat16,
        )
        model.to(device)
        model.config.use_cache = False  # 为梯度 checkpoint / 训练安全

        # NEW: 修正 generation_config，避免保存时报错
        model = sanitize_generation_config(model)

        # 注入单分支 LoRA
        model = replace_with_lora(
            model,
            target_modules=cfg.lora_target_modules,
            r=cfg.lora_r,
            alpha=cfg.lora_alpha,
            dropout=cfg.lora_dropout,
            num_frozen_layers=cfg.num_frozen_layers,
        )
        # 只训练 LoRA 参数
        mark_only_lora_as_trainable(model)
    else:
        model = init_model

    model.train()

    # --------------------------------------------------
    # 2. 数据加载
    # --------------------------------------------------
    dataset = InstructionDataset(train_file, tokenizer, max_length=cfg.max_seq_len)
    if len(dataset) == 0:
        print(f"[Train WARNING] Empty train dataset for {task_name}, skip training.")
        return model, tokenizer, replay_buffer, ewc_prev

    loader = DataLoader(
        dataset,
        batch_size=cfg.per_device_batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, tokenizer.pad_token_id),
    )

    total_steps = (
        len(loader) * cfg.num_epochs // cfg.gradient_accumulation_steps + 1
    )
    print(f"[Train] Task={task_name}, method={method}, examples={len(dataset)}, total_steps={total_steps}")

    # --------------------------------------------------
    # 3. 优化器（只更新 LoRA 参数）
    # --------------------------------------------------
    optim_groups = get_lora_param_groups(
        model,
        lr=cfg.lr_fast,
        weight_decay=cfg.weight_decay,
    )
    optimizer = torch.optim.AdamW(optim_groups)

    # --------------------------------------------------
    # 4. 训练循环
    # --------------------------------------------------
    global_step = 0
    train_log = os.path.join("logs", f"seq_{method}_{task_name}_train_loss.jsonl")

    for epoch in range(cfg.num_epochs):
        pbar = tqdm(loader, desc=f"{task_name} ({method}) Epoch {epoch+1}")
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

            # EWC penalty
            if method == "ewc" and ewc_prev is not None:
                loss_ewc = compute_ewc_loss(model, ewc_prev, device, ewc_lambda)
                loss = loss + loss_ewc

            # Replay loss
            if method == "replay" and replay_buffer:
                replay_batch = random.choice(replay_buffer)
                rb = {k: v.to(device) for k, v in replay_batch.items()}
                out_rep = model(
                    input_ids=rb["input_ids"],
                    attention_mask=rb["attention_mask"],
                    labels=rb["labels"],
                )
                loss_rep = out_rep.loss
                loss = loss + replay_lambda * loss_rep

            # ---------- 1) 跳过 labels 全为 -100 的 batch ----------
            if (batch["labels"] != -100).sum() == 0:
                print(f"[WARN] All labels=-100 | skip batch | task={task_name}, method={method}")
                continue

            # ---------- 2) loss 是否为有限值（非 NaN/inf） ----------
            if not torch.isfinite(loss):
                print(f"[WARN] Non-finite loss={loss.item()} | skip batch | task={task_name}, method={method}, step={global_step}")
                continue

            # 梯度累积缩放
            loss = loss / cfg.gradient_accumulation_steps
            loss.backward()

            # ---------- 3) NaN 检查（梯度已计算） ----------
            if check_nan(loss, model, task_name, method, global_step):
                print(f"[NaN] Skip batch | task={task_name}, method={method}, step={global_step}")
                optimizer.zero_grad()
                continue

            # 梯度裁剪（只对可训练参数）
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                max_grad_norm,
            )

            running_loss += loss.item()
            running_steps += 1

            if (step + 1) % cfg.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

                full_loss = running_loss * cfg.gradient_accumulation_steps / max(1, running_steps)
                pbar.set_postfix({"loss": full_loss})

                with open(train_log, "a", encoding="utf-8") as f:
                    f.write(
                        json.dumps(
                            {
                                "method": method,
                                "task": task_name,
                                "epoch": epoch + 1,
                                "step": global_step,
                                "loss": float(full_loss),
                            }
                        )
                        + "\n"
                    )

                running_loss = 0.0
                running_steps = 0

    # --------------------------------------------------
    # 5. Replay / EWC 状态更新
    # --------------------------------------------------
    if method == "replay":
        if replay_buffer is None:
            replay_buffer = []
        replay_buffer = build_replay_buffer_from_file(
            replay_buffer,
            train_file,
            tokenizer,
            cfg,
            max_buffer_size=replay_buffer_size,
            max_samples_per_task=min(len(dataset), 1024),
        )

    if method == "ewc":
        ewc_prev = compute_ewc_state(
            model,
            tokenizer,
            train_file,
            cfg,
            device,
            max_samples=200,
        )

    return model, tokenizer, replay_buffer, ewc_prev


# ======================================================
# 主程序：多任务顺序训练
# ======================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--method",
        type=str,
        required=True,
        choices=["seq_lora", "replay", "ewc"],
        help="选择顺序学习基线方法",
    )
    args = parser.parse_args()
    method = args.method

    cfg = BaseConfig()

    # 数据 key：toy / train_split（tiny/train/full 等）
    if getattr(cfg, "use_toy_data", False):
        data_key = "toy"
    else:
        data_key = getattr(cfg, "train_split", "train")

    eval_key = getattr(cfg, "eval_split", "val")

    # 设备
    device = select_device()

    os.makedirs(cfg.save_dir, exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    # 任务顺序
    tasks = ["T1_general", "T2_math", "T3_code", "T4_tool", "T5_safety"]

    current_model = None
    current_tokenizer = None
    replay_buffer = [] if method == "replay" else None
    ewc_prev = None

    eval_log_path = os.path.join("logs", "seq_eval_loss.jsonl")

    for idx, task_name in enumerate(tasks):
        # ------------------------------
        # 路径选择
        # ------------------------------
        if task_name not in cfg.data_paths:
            print(f"[WARN] Task {task_name} not in cfg.data_paths, skip.")
            continue

        if data_key not in cfg.data_paths[task_name]:
            print(f"[WARN] data_key={data_key} not in cfg.data_paths[{task_name}], fallback to 'train'")
            train_file = cfg.data_paths[task_name].get("train")
        else:
            train_file = cfg.data_paths[task_name][data_key]

        print(f"\n===== [{method}] Train {task_name} | file = {train_file} =====")

        # ------------------------------
        # 单任务训练
        # ------------------------------
        current_model, current_tokenizer, replay_buffer, ewc_prev = train_one_task(
            method=method,
            task_name=task_name,
            train_file=train_file,
            cfg=cfg,
            device=device,
            init_model=current_model,
            tokenizer=current_tokenizer,
            replay_buffer=replay_buffer,
            ewc_prev=ewc_prev,
        )

        if current_model is None:
            print(f"[WARN] Model is None after training {task_name}, skip to next.")
            continue

        # ------------------------------
        # 保存阶段模型
        # ------------------------------
        save_dir = os.path.join(cfg.save_dir, f"seq_{method}_{task_name}")
        os.makedirs(save_dir, exist_ok=True)

        # NEW: 再次保险地修正 generation_config，避免未来改动引入新不一致
        current_model = sanitize_generation_config(current_model)

        current_model.save_pretrained(save_dir)
        current_tokenizer.save_pretrained(save_dir)
        print(f"[SAVE] {method} model of {task_name} saved to {save_dir}")

        # ------------------------------
        # 在已见任务上做评估
        # ------------------------------
        seen_tasks = tasks[: idx + 1]
        print(f"[Eval] After {task_name}, evaluate on tasks: {seen_tasks}")

        for ev in seen_tasks:
            if ev not in cfg.data_paths:
                continue

            if eval_key in cfg.data_paths[ev]:
                ev_file = cfg.data_paths[ev][eval_key]
            elif "val" in cfg.data_paths[ev]:
                ev_file = cfg.data_paths[ev]["val"]
            else:
                ev_file = train_file

            ev_loss = evaluate_loss(current_model, current_tokenizer, ev_file, cfg, device)
            print(f"[Eval] method={method}, stage={task_name} on {ev}: loss={ev_loss:.4f}")

            with open(eval_log_path, "a", encoding="utf-8") as f:
                f.write(
                    json.dumps(
                        {
                            "method": method,
                            "stage": task_name,
                            "eval_task": ev,
                            "loss": float(ev_loss),
                        }
                    )
                    + "\n"
                )


if __name__ == "__main__":
    main()
