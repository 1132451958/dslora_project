# prepare_datasets.py
"""
构建五个任务的 FULL 数据集，写入：
  data/full/T1_general_full.jsonl
  data/full/T2_math_full.jsonl
  data/full/T3_code_full.jsonl
  data/full/T4_tool_full.jsonl
  data/full/T5_safety_full.jsonl

每行格式：
  {"instruction": "...", "input": "...", "output": "..."}

注：
  - 不覆盖 toy 数据（data/T*_xxx.jsonl）
  - full 数据全部写入 data/full/ 目录，已被 .gitignore 忽略
"""

import os
import json
from datasets import load_dataset

# ------------ 目录准备 ------------

os.makedirs("data", exist_ok=True)
os.makedirs("data/full", exist_ok=True)   # full 数据集固定放这里


def save_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[OK] Saved {len(rows)} rows → {path}")


# ======================================================
# T1: General 指令（Alpaca + Dolly）
# ======================================================

def build_T1_general_full():
    rows = []

    print("[T1] Loading tatsu-lab/alpaca ...")
    alpaca = load_dataset("tatsu-lab/alpaca", split="train")
    for ex in alpaca:
        rows.append({
            "instruction": ex["instruction"],
            "input": ex.get("input", "") or "",
            "output": ex["output"],
        })

    print("[T1] Loading databricks/databricks-dolly-15k ...")
    dolly = load_dataset("databricks/databricks-dolly-15k", split="train")
    for ex in dolly:
        rows.append({
            "instruction": ex["instruction"],
            "input": ex.get("context", "") or "",
            "output": ex["response"],
        })

    save_jsonl("data/full/T1_general_full.jsonl", rows)


# ======================================================
# T2: 数学（GSM8K）
# ======================================================

def build_T2_math_full():
    print("[T2] Loading openai/gsm8k (main) ...")
    gsm = load_dataset("openai/gsm8k", "main", split="train")

    rows = []
    for ex in gsm:
        q = ex["question"]
        a = ex["answer"]
        rows.append({
            "instruction": "Solve the math problem step by step and give the final answer.",
            "input": q,
            "output": a,
        })

    save_jsonl("data/full/T2_math_full.jsonl", rows)


# ======================================================
# T3: 代码（CodeAlpaca）
# ======================================================

def build_T3_code_full():
    print("[T3] Loading sahil2801/CodeAlpaca-20k ...")
    code = load_dataset("sahil2801/CodeAlpaca-20k", split="train")

    rows = []
    for ex in code:
        rows.append({
            "instruction": ex["instruction"],
            "input": ex.get("input", "") or "",
            "output": ex["output"],
        })

    save_jsonl("data/full/T3_code_full.jsonl", rows)


# ======================================================
# T4: 函数调用 / 工具使用（Mini-ToolBench + fallback）
# ======================================================

def _build_T4_tool_synthetic_fallback():
    """如果线上函数调用数据集下载失败，退回到原来那 5 条 synthetic 数据。"""
    print("[T4] HF dataset unavailable, fallback to small synthetic tool-calling dataset ...")
    rows = [
        {
            "instruction": "Call the weather API to get the weather in Beijing tomorrow.",
            "input": "",
            "output": '{"tool_name": "get_weather", "arguments": {"location": "Beijing", "date": "tomorrow"}}',
        },
        {
            "instruction": "Generate a function call to search flights from Shanghai to Tokyo on 2025-10-01.",
            "input": "",
            "output": '{"tool_name": "search_flights", "arguments": {"from": "Shanghai", "to": "Tokyo", "date": "2025-10-01"}}',
        },
        {
            "instruction": "Given a user asking for Italian restaurants in Nanjing tonight at 7pm, create a function call.",
            "input": "",
            "output": '{"tool_name": "search_restaurants", "arguments": {"city": "Nanjing", "cuisine": "Italian", "time": "19:00"}}',
        },
        {
            "instruction": "The user wants to convert 100 USD to CNY. Generate a function call to a currency converter API.",
            "input": "",
            "output": '{"tool_name": "convert_currency", "arguments": {"from": "USD", "to": "CNY", "amount": 100}}',
        },
        {
            "instruction": "Create a function call that adds an event 'Thesis defense' on 2025-06-30 at 15:00 in Beijing.",
            "input": "",
            "output": '{"tool_name": "create_calendar_event", "arguments": {"title": "Thesis defense", "date": "2025-06-30", "time": "15:00", "location": "Beijing"}}',
        },
    ]
    save_jsonl("data/full/T4_tool_full.jsonl", rows)
    print(f"[T4] Fallback synthetic dataset saved with {len(rows)} examples.")


def build_T4_tool_full(max_samples: int = 5000):
    """
    首选：使用 glaiveai/glaive-function-calling 构造 Mini-ToolBench 风格 T4。
    若 HF 下载失败或网络问题，则退回到 5 条 synthetic 数据。
    """
    from datasets import load_dataset

    try:
        print("[T4] Loading glaiveai/glaive-function-calling ...")
        ds = load_dataset("glaiveai/glaive-function-calling", split="train")
    except Exception as e:
        print(f"[T4] Failed to load glaiveai/glaive-function-calling: {e}")
        _build_T4_tool_synthetic_fallback()
        return

    rows = []
    n_total = len(ds)
    print(f"[T4] Total samples in dataset: {n_total}")
    max_samples = min(max_samples, n_total)
    print(f"[T4] Building Mini-ToolBench style T4 with up to {max_samples} samples ...")

    for ex in ds:
        # 尝试从常见字段拿文本
        text = None
        for key in ["text", "content", "json"]:
            if key in ex:
                text = ex[key]
                break
        if text is None:
            # 最后兜底：直接把整个字典转成字符串（大概率会被过滤掉）
            text = str(ex)

        # 简单解析 USER / ASSISTANT 片段
        user_idx = text.find("USER:")
        asst_idx = text.find("ASSISTANT:")
        if user_idx == -1 or asst_idx == -1 or asst_idx <= user_idx:
            continue

        user_text = text[user_idx + len("USER:"):asst_idx].strip()
        if not user_text:
            continue

        # 在 ASSISTANT 段里寻找 <functioncall>
        fc_idx = text.find("<functioncall>", asst_idx)
        if fc_idx == -1:
            # 没有函数调用的样本先跳过（保持任务风格统一）
            continue

        out_text = text[fc_idx:].strip()
        if not out_text:
            continue

        rows.append({
            "instruction": user_text,
            "input": "",
            "output": out_text,
        })

        if len(rows) >= max_samples:
            break

    if not rows:
        print("[T4] No valid samples parsed from HF dataset, fallback to synthetic.")
        _build_T4_tool_synthetic_fallback()
        return

    save_jsonl("data/full/T4_tool_full.jsonl", rows)



# ======================================================
# T5: 安全对齐（HH-RLHF）
# ======================================================

def build_T5_safety_full():
    print("[T5] Loading Anthropic/hh-rlhf ...")
    ds = load_dataset("Anthropic/hh-rlhf", split="train")

    rows = []
    for ex in ds:
        prompt = ex.get("prompt", "")
        chosen = ex.get("chosen", "")
        rejected = ex.get("rejected", "")

        if isinstance(chosen, dict):
            chosen = chosen.get("text", "") or chosen.get("completion", "") or ""

        if not prompt:
            prompt = "Please respond to the following request in a safe and helpful way."

        rows.append({
            "instruction": prompt,
            "input": "",
            "output": chosen or rejected,
        })

    save_jsonl("data/full/T5_safety_full.jsonl", rows)


# ======================================================
# Main
# ======================================================

if __name__ == "__main__":
    print("=== Building FULL datasets (for LLaMA-7B training) ===")
    build_T1_general_full()
    build_T2_math_full()
    build_T3_code_full()
    build_T4_tool_full()
    build_T5_safety_full()
    print("All FULL datasets generated → data/full/")
