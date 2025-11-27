# split_datasets.py
"""
将每个任务的 full 数据拆分为 80/10/10 + tiny 四个子集：
  - train：约 80%
  - val  ：约 10%
  - test ：约 10%
  - tiny ：从 train 中再采样一小份（例如 1000 条），用于快速调参与 sanity check

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


def split_one_task(task: str):
    full_path = os.path.join("data", "full", f"{task}_full.jsonl")

    print(f"\n===== [TASK] {task} =====")
    print(f"[PATH] full file = {full_path}")

    if not os.path.exists(full_path):
        print(f"[ERROR] Full file not found for {task}: {full_path}")
        return

    data = load_jsonl(full_path)
    n = len(data)
    print(f"[INFO] Total examples: {n}")

    if n == 0:
        print("[WARN] Empty file, skip.")
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
    print("=== Start splitting datasets (80/10/10 + tiny) ===")
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
