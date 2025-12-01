# train_slsd_seq.py
"""
任务序列训练脚本（DS-LoRA + SLSD）
- 支持 full/toy 数据自动切换（来自 BaseConfig）
- 使用本地 LLaMA-7B（cfg.model_name 指向 pretrained_models/llama2-7b-hf）
- 任务序列：T1 → T2 → T3 → T4 → T5
"""

import os
import json
import subprocess
from copy import deepcopy

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.nn.functional as F

from configs.base_config import BaseConfig
from utils_data import InstructionDataset, collate_fn
from models.ds_lora import replace_with_ds_lora, get_ds_lora_param_groups


# ======================================================
# NaN Debug 工具（可选，用于排查问题）
# ======================================================
def debug_batch_nan(batch, model, tokenizer, task_name, global_step):
    """
    定位产生 NaN 的 batch：
    - 检查 labels 的范围、unique 值
    - 检查 logits 是否已经包含 inf / nan
    """
    labels = batch["labels"]           # [B, L]
    input_ids = batch["input_ids"]     # [B, L]
    attn_mask = batch["attention_mask"]

    print("\n===== [NaN DEBUG] =====")
    print(f"Task = {task_name}, step = {global_step}")
    print("Batch size:", labels.size(0))
    print("Seq length:", labels.size(1))

    # 1) label 统计
    valid_before = (labels != -100).sum(dim=-1)
    shift_labels = labels[..., 1:].contiguous()
    valid_after = (shift_labels != -100).sum(dim=-1)

    print("valid_before_shift per sample:", valid_before.tolist())
    print("valid_after_shift  per sample:", valid_after.tolist())
    print("labels unique (first sample):", torch.unique(labels[0])[:20])
    print("labels min/max (first sample):", labels[0].min().item(), labels[0].max().item())

    # 2) 查看 logits 是否已经炸掉
    model.eval()
    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attn_mask)
        logits = out.logits    # [B, L, V]

        isfinite = torch.isfinite(logits)
        num_total = logits.numel()
        num_finite = isfinite.sum().item()
        num_nan = torch.isnan(logits).sum().item()
        num_inf = torch.isinf(logits).sum().item()

        print(f"logits finite ratio = {num_finite}/{num_total}")
        print(f"logits num_nan = {num_nan}, num_inf = {num_inf}")
        print("logits max/min (clipped to finite):")

        finite_logits = logits[isfinite]
        if finite_logits.numel() > 0:
            print("  max =", finite_logits.max().item())
            print("  min =", finite_logits.min().item())
        else:
            print("  all non-finite!")

        # 3) 手算一次 CE loss 看是不是同样 NaN
        try:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            loss_flat = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
                reduction="none",
            )
            print("manual CE loss stats (ignoring -100):")
            print("  has_nan =", torch.isnan(loss_flat).any().item())
            print("  has_inf =", torch.isinf(loss_flat).any().item())
            finite_loss_flat = loss_flat[torch.isfinite(loss_flat)]
            if finite_loss_flat.numel() > 0:
                print("  loss max =", finite_loss_flat.max().item())
                print("  loss min =", finite_loss_flat.min().item())
        except Exception as e:
            print("[NaN DEBUG] manual CE loss computation raised error:", repr(e))

    # 4) 打印一条样本的文本，帮助定位原始数据
    text = tokenizer.decode(input_ids[0], skip_special_tokens=False)
    print("Decoded sample[0] (truncated to 300 chars):")
    print(text[:300], " ...")

    print("===== END DEBUG =====\n")
    model.train()


# ======================================================
# NaN / Inf 检查工具
# ======================================================
def check_nan(loss, model, task_name, global_step):
    """
    检查 loss 和梯度中是否存在 NaN。
    - 在 backward 之后调用（此时梯度已计算）
    - 如发现 NaN，返回 True；否则 False。
    """
    if torch.isnan(loss):
        print(f"[NaN] Loss is NaN | task={task_name}, step={global_step}")
        return True

    for name, p in model.named_parameters():
        if p.requires_grad and p.grad is not None and torch.isnan(p.grad).any():
            print(f"[NaN] Grad is NaN in {name} | task={task_name}, step={global_step}")
            return True

    return False


# ======================================================
# GPU 选择函数（使用 nvidia-smi，不创建多卡 context）
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
            parts = [p.strip() for p in line.split(",")]
            free_mb = int(parts[0])
            total_mb = int(parts[1])
            name = ",".join(parts[2:])

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
        print(f"[GPU SELECT] nvidia-smi query failed: {e}")
        print("[GPU SELECT] Fallback: using cuda:0")
        return torch.device("cuda:0")


# ======================================================
# Evaluate（加了空数据、全 -100、非有限值处理）
# ======================================================
def evaluate_loss(model, tokenizer, data_file, cfg: BaseConfig, device, max_eval_samples=200):
    model.eval()
    dataset = InstructionDataset(data_file, tokenizer, max_length=cfg.max_seq_len)

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

            # 如果这一 batch 所有 labels 都是 -100（没监督 token），跳过
            if (batch["labels"] != -100).sum() == 0:
                continue

            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )
            loss = outputs.loss

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
    dataset = InstructionDataset(data_file, teacher_tokenizer, max_length=cfg.max_seq_len)
    if len(dataset) == 0:
        print(f"[SLSD WARNING] Empty dataset when building probe buffer: {data_file}")
        return []

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

    max_grad_norm = getattr(cfg, "max_grad_norm", 1.0)

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
            torch_dtype=torch.bfloat16,
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
    if len(dataset) == 0:
        print(f"[Train WARNING] Empty train dataset for {task_name}, skip training.")
        return model, tokenizer

    loader = DataLoader(
        dataset,
        batch_size=cfg.per_device_batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, tokenizer.pad_token_id),
    )

    # --------------------------------------------------
    # 4. 优化器：slow & fast LoRA 分开更新
    # --------------------------------------------------
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
    # 按 step 记录：与 baseline 对齐
    step_log = f"logs/seq_ds_lora_slsd_{task_name}_steps.jsonl"
    # 按 epoch 记录平均 loss
    epoch_log = f"logs/seq_{task_name}_train_loss.jsonl"

    for epoch in range(cfg.num_epochs):
        pbar = tqdm(loader, desc=f"{task_name} Epoch {epoch+1}")
        running_loss = 0.0          # 累积的是 scaled sup_loss（除过 grad_acc）
        running_steps = 0
        last_raw_loss = 0.0         # 最近一次原始 sup_loss（未缩放）

        epoch_loss_sum = 0.0        # 用于按 epoch 统计 full loss
        epoch_step_count = 0

        for step, batch in enumerate(pbar):
            batch = {k: v.to(device) for k, v in batch.items()}

            # 如果这一 batch 没有任何监督 token，跳过
            if (batch["labels"] != -100).sum() == 0:
                print(f"[WARN] All labels=-100 | skip batch | task={task_name}")
                continue

            # --- supervised loss ---
            out = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )
            sup_loss = out.loss

            # 记录原始 loss（未除 gradient_accumulation_steps），用于打印/画图
            last_raw_loss = sup_loss.detach().float().item()

            # 检查 loss 是否有限
            if not torch.isfinite(sup_loss):
                print(f"[WARN] Non-finite sup_loss={sup_loss.item()} | task={task_name}, step={global_step}")
                debug_batch_nan(batch, model, tokenizer, task_name, global_step)
                continue

            # 梯度累积缩放（训练用 scaled loss）
            sup_loss = sup_loss / cfg.gradient_accumulation_steps
            sup_loss.backward()

            # NaN 检查
            if check_nan(sup_loss, model, task_name, global_step):
                print(f"[NaN] Skip batch after sup_loss | task={task_name}, step={global_step}")
                optimizer_fast.zero_grad()
                optimizer_slow.zero_grad()
                continue

            running_loss += sup_loss.item()
            running_steps += 1

            # --- KD loss（SLSD，对 slow 分支做轻量蒸馏） ---
            if cfg.use_slsd and probe_buffer is not None and len(probe_buffer) > 0 and step % 10 == 0:
                buf_ex = probe_buffer[step % len(probe_buffer)]
                kd = kd_loss_from_buffer(model, buf_ex, device) * cfg.kd_lambda

                if not torch.isfinite(kd):
                    print(f"[SLSD WARN] Non-finite kd={kd.item()} | skip kd | task={task_name}, step={global_step}")
                else:
                    kd.backward()

                    if check_nan(kd, model, task_name, global_step):
                        print(f"[NaN] Skip batch after KD | task={task_name}, step={global_step}")
                        optimizer_fast.zero_grad()
                        optimizer_slow.zero_grad()
                        continue

                    # 只对 slow 分支进行一次更新
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in slow_params if p.requires_grad],
                        max_grad_norm,
                    )
                    optimizer_slow.step()
                    optimizer_slow.zero_grad()

            # --- fast 分支的梯度累积 & 更新 ---
            if (step + 1) % cfg.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in fast_params if p.requires_grad],
                    max_grad_norm,
                )
                optimizer_fast.step()
                optimizer_fast.zero_grad()
                global_step += 1

                # scaled_loss 是平均的 scaled sup_loss
                scaled_loss = running_loss / max(running_steps, 1)
                # full_loss 恢复到“原始 loss”量级，与 baseline 的 loss 可对齐
                full_loss = scaled_loss * cfg.gradient_accumulation_steps

                # tqdm 显示：原始 loss + full_loss
                pbar.set_postfix({
                    "raw_loss": f"{last_raw_loss:.4f}",
                    "loss": f"{full_loss:.4f}",
                })

                # 写 step 级别 log（与 baseline 风格一致，可直接画图对比）
                with open(step_log, "a", encoding="utf-8") as f:
                    f.write(json.dumps({
                        "method": "ds_lora_slsd",
                        "task": task_name,
                        "epoch": epoch + 1,
                        "step": global_step,
                        "loss": float(full_loss),          # 用于对比
                        "raw_loss": float(last_raw_loss),  # 额外保留一份原始 loss
                    }) + "\n")

                # 用于 epoch 级别统计
                epoch_loss_sum += full_loss
                epoch_step_count += 1

                # 重置累计
                running_loss = 0.0
                running_steps = 0

        # 一个 epoch 结束后，记录 epoch 平均 loss
        if epoch_step_count > 0:
            avg_epoch_loss = epoch_loss_sum / epoch_step_count
        else:
            avg_epoch_loss = 0.0

        with open(epoch_log, "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "epoch": epoch + 1,
                "avg_train_loss": float(avg_epoch_loss),
            }) + "\n")

        print(f"[Epoch {epoch+1}] {task_name} avg_train_loss = {avg_epoch_loss:.4f}")

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
                    "method": "ds_lora_slsd",
                    "stage": task_name,
                    "eval_task": ev,
                    "loss": float(ev_loss),
                }) + "\n")


if __name__ == "__main__":
    main()
