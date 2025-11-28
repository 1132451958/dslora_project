# split_datasets.py
"""
将每个任务的 full 数据拆分为 80/10/10 + tiny 四个子集：
  - train：约 80%
  - val  ：约 10%
  - test ：约 10%
  - tiny ：从 train 中再采样一小份（例如 1000 条），用于快速调参与 sanity check

<<<<<<< HEAD
并在拆分前做一次简单的数据清洗：
  - 丢弃 output 为空 / 只有空格 的样本
  - 丢弃缺少 instruction/output 字段的脏样本
=======
假设原始 full 文件为：
  data/full/T1_general_full.jsonl
  data/full/T2_math_full.jsonl
  data/full/T3_code_full.jsonl
  data/full/T4_tool_full.jsonl
  data/full/T5_safety_full.jsonl

输出文件放在：
  data/split/T1_general_train.jsonl
  data/split/T1_general_val.jsonl
  data/split/T1_general_test.jsonl
  data/split/T1_general_tiny.jsonl
  ... 其他任务同理
>>>>>>> 395136dc3b859244ef85234cd58ab3d60797141b
"""

import os
import json
import random
from typing import List, Dict

RANDOM_SEED = 42  # 保证可复现实验

TASKS = [
    "T1_general",
    "T2_math",
    "T3_code",
    "T4_tool",
    "T5_safety",
]


def load_jsonl(path: str) -> List[Dict]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data


def save_jsonl(path: str, rows: List[Dict]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for ex in rows:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")


<<<<<<< HEAD
def clean_and_filter(data: List[Dict]) -> List[Dict]:
    """
    简单清洗逻辑：
      - 必须包含 instruction / output 字段
      - output 去掉空白后不能为空字符串
    这样可以极大减少 labels 全为 -100 导致 eval 时出现 NaN 的情况。
    """
    cleaned = []
    dropped_missing = 0
    dropped_empty_output = 0

    for ex in data:
        if not isinstance(ex, dict):
            dropped_missing += 1
            continue

        if "instruction" not in ex or "output" not in ex:
            dropped_missing += 1
            continue

        instr = str(ex["instruction"]).strip()
        out = str(ex["output"]).strip()
        inp = str(ex.get("input", ""))  # input 可以为空，但统一转成字符串

        # 丢掉 output 为空的样本
        if out == "":
            dropped_empty_output += 1
            continue

        ex_clean = {
            "instruction": instr,
            "input": inp,
            "output": out,
        }
        cleaned.append(ex_clean)

    print(
        f"[CLEAN] kept={len(cleaned)}, "
        f"dropped_missing={dropped_missing}, "
        f"dropped_empty_output={dropped_empty_output}"
    )
    return cleaned


=======
>>>>>>> 395136dc3b859244ef85234cd58ab3d60797141b
def split_one_task(task: str):
    full_path = os.path.join("data", "full", f"{task}_full.jsonl")

    print(f"\n===== [TASK] {task} =====")
    print(f"[PATH] full file = {full_path}")

    if not os.path.exists(full_path):
        print(f"[ERROR] Full file not found for {task}: {full_path}")
        return

<<<<<<< HEAD
    raw_data = load_jsonl(full_path)
    print(f"[INFO] Raw total examples: {len(raw_data)}")

    # ---------- 数据清洗 ----------
    data = clean_and_filter(raw_data)
    n = len(data)
    print(f"[INFO] After cleaning: {n} examples")

    if n == 0:
        print("[WARN] Empty file after cleaning, skip.")
=======
    data = load_jsonl(full_path)
    n = len(data)
    print(f"[INFO] Total examples: {n}")

    if n == 0:
        print("[WARN] Empty file, skip.")
>>>>>>> 395136dc3b859244ef85234cd58ab3d60797141b
        return

    rng = random.Random(RANDOM_SEED)
    rng.shuffle(data)

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


def main():
<<<<<<< HEAD
    print("=== Start splitting datasets (80/10/10 + tiny, with cleaning) ===")
=======
    print("=== Start splitting datasets (80/10/10 + tiny) ===")
>>>>>>> 395136dc3b859244ef85234cd58ab3d60797141b
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
