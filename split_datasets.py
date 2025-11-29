# split_datasets.py
"""
将每个任务的 full 数据拆分为 80/10/10 + tiny 四个子集：
  - train：约 80%
  - val  ：约 10%
  - test ：约 10%
  - tiny ：从 train 中再采样一小份（例如 1000 条），用于快速调参与 sanity check

并在拆分前做一次数据清洗，尽量保证样本“对训练是安全的”：
  - 必须包含 instruction / output 字段
  - instruction / output 转为字符串后去掉空白必须非空
  - 使用 LLaMA tokenizer 估算整条样本的 token 长度，超过 max_seq_len 的样本丢弃
  - 每个任务最多保留 10,000 条样本，控制训练成本、降低极端样本影响
"""

import os
import json
import random
from typing import List, Dict

from transformers import AutoTokenizer
from configs.base_config import BaseConfig

# ------------------ 全局配置 ------------------
RANDOM_SEED = 42  # 保证可复现实验

TASKS = [
    "T1_general",
    "T2_math",
    "T3_code",
    "T4_tool",
    "T5_safety",
]

# 每个任务最多保留的样本数（cap）
MAX_SAMPLES_PER_TASK = 10_000

# 使用 BaseConfig 中的模型名称和 max_seq_len
_cfg = BaseConfig()
_MODEL_NAME = _cfg.model_name
_MAX_SEQ_LEN = getattr(_cfg, "max_seq_len", 1024)

print(f"[CLEAN] Using model_name={_MODEL_NAME}, max_seq_len={_MAX_SEQ_LEN}")
_tokenizer = AutoTokenizer.from_pretrained(_MODEL_NAME)
if _tokenizer.pad_token_id is None:
    _tokenizer.pad_token = _tokenizer.eos_token


# ------------------ 工具函数 ------------------
def load_jsonl(path: str) -> List[Dict]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                # 解析失败的行直接跳过
                continue
            data.append(obj)
    return data


def save_jsonl(path: str, rows: List[Dict]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for ex in rows:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")


def clean_and_filter(data: List[Dict]) -> List[Dict]:
    """
    清洗逻辑：
      - 必须是 dict，且包含 instruction / output 字段
      - instruction / output 转为字符串后 strip() 必须非空
      - 用 tokenizer 计算 [INST] ... [/INST] output 这一整段的 token 数，
        若为 0 或 > _MAX_SEQ_LEN 则丢弃
    这样可以极大减少：
      - labels 全为 -100
      - 极端长样本导致 bf16 溢出 / NaN
    的情况。
    """
    cleaned = []
    dropped_not_dict = 0
    dropped_missing = 0
    dropped_empty_instr = 0
    dropped_empty_output = 0
    dropped_too_long = 0
    dropped_tokenize_error = 0

    for ex in data:
        if not isinstance(ex, dict):
            dropped_not_dict += 1
            continue

        if "instruction" not in ex or "output" not in ex:
            dropped_missing += 1
            continue

        instr = str(ex["instruction"]).strip()
        out = str(ex["output"]).strip()
        inp = str(ex.get("input", "")).strip()  # input 可以为空，但统一成字符串

        if instr == "":
            dropped_empty_instr += 1
            continue

        if out == "":
            dropped_empty_output += 1
            continue

        # 构造与 InstructionDataset 一致的 prompt 形式（近似即可）
        if inp:
            prompt = f"[INST] {instr}\n{inp} [/INST]"
        else:
            prompt = f"[INST] {instr} [/INST]"
        full_text = prompt + " " + out + "</s>"

        try:
            ids = _tokenizer(
                full_text,
                add_special_tokens=False,
            ).input_ids
        except Exception:
            dropped_tokenize_error += 1
            continue

        if len(ids) == 0:
            dropped_tokenize_error += 1
            continue

        if len(ids) > _MAX_SEQ_LEN:
            dropped_too_long += 1
            continue

        ex_clean = {
            "instruction": instr,
            "input": inp,
            "output": out,
        }
        cleaned.append(ex_clean)

    print(
        f"[CLEAN] kept={len(cleaned)}, "
        f"dropped_not_dict={dropped_not_dict}, "
        f"dropped_missing={dropped_missing}, "
        f"dropped_empty_instr={dropped_empty_instr}, "
        f"dropped_empty_output={dropped_empty_output}, "
        f"dropped_too_long={dropped_too_long}, "
        f"dropped_tokenize_error={dropped_tokenize_error}"
    )
    return cleaned


# ------------------ 拆分单个任务 ------------------
def split_one_task(task: str):
    full_path = os.path.join("data", "full", f"{task}_full.jsonl")

    print(f"\n===== [TASK] {task} =====")
    print(f"[PATH] full file = {full_path}")

    if not os.path.exists(full_path):
        print(f"[ERROR] Full file not found for {task}: {full_path}")
        return

    raw_data = load_jsonl(full_path)
    print(f"[INFO] Raw total examples: {len(raw_data)}")

    # ---------- 数据清洗 ----------
    data = clean_and_filter(raw_data)
    n = len(data)
    print(f"[INFO] After cleaning: {n} examples")

    if n == 0:
        print("[WARN] Empty file after cleaning, skip.")
        return

    rng = random.Random(RANDOM_SEED)

    # ---------- 每任务最多保留 MAX_SAMPLES_PER_TASK ----------
    if n > MAX_SAMPLES_PER_TASK:
        print(f"[INFO] {task}: {n} > {MAX_SAMPLES_PER_TASK}, subsample to {MAX_SAMPLES_PER_TASK}")
        data = rng.sample(data, MAX_SAMPLES_PER_TASK)
        n = len(data)

    rng.shuffle(data)

    # ---------- 80 / 10 / 10 拆分 ----------
    n_train = int(0.8 * n)
    n_val = int(0.1 * n)
    n_test = n - n_train - n_val  # 剩下的都给 test

    train_data = data[:n_train]
    val_data = data[n_train:n_train + n_val]
    test_data = data[n_train + n_val:]

    # tiny：从 train 中再采样一部分（这里取 min(1000, 0.1 * n_train)）
    tiny_size = min(1000, max(1, int(0.1 * len(train_data))))
    tiny_data = rng.sample(train_data, tiny_size)

    out_dir = os.path.join("data", "split")
    os.makedirs(out_dir, exist_ok=True)

    train_path = os.path.join(out_dir, f"{task}_train.jsonl")
    val_path = os.path.join(out_dir, f"{task}_val.jsonl")
    test_path = os.path.join(out_dir, f"{task}_test.jsonl")
    tiny_path = os.path.join(out_dir, f"{task}_tiny.jsonl")

    save_jsonl(train_path, train_data)
    save_jsonl(val_path, val_data)
    save_jsonl(test_path, test_data)
    save_jsonl(tiny_path, tiny_data)

    print(f"[DONE] {task}:")
    print(f"  train: {len(train_data)} -> {train_path}")
    print(f"  val  : {len(val_data)} -> {val_path}")
    print(f"  test : {len(test_data)} -> {test_path}")
    print(f"  tiny : {len(tiny_data)} -> {tiny_path}")


# ------------------ 主入口 ------------------
def main():
    print("=== Start splitting datasets (80/10/10 + tiny, with cleaning & max 10k/task) ===")
    print(f"cwd = {os.getcwd()}")
    print(f"data/full exists? {os.path.isdir('data/full')}")
    if os.path.isdir("data/full"):
        print("data/full files:", os.listdir("data/full"))
    else:
        print("[ERROR] data/full directory not found, please check your project structure.")

    for t in TASKS:
        split_one_task(t)

    print("\n=== All tasks processed ===")


if __name__ == "__main__":
    main()
